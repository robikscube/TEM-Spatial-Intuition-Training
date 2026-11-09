#!/usr/bin/env python3
"""Capped, reproducible training run for an unmodified torch_tem checkout.

This script is an adapter, not a replacement implementation. It imports
upstream ``model.py``, ``world.py``, ``parameters.py`` and ``utils.py`` from
``research/reference/torch_tem`` and drives the same forward/backward loop as
upstream ``run.py`` with explicit seeds, a small iteration cap, metrics, and a
real checkpoint.

It does not claim convergence from a short pilot. Every number written here
comes from an actual PyTorch run.

Typical capped run (see docs/TRAINING.md):

    python scripts/train_tem.py --iterations 50 --batch-size 4 \
        --rollout 12 --seed 7 --out training/pilot

Exit codes:
  2  bad arguments or missing files
  3  torch/numpy/scipy missing
  4  upstream adapter incompatibility
  5  training completed but a written artifact failed verification
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tem_trained_adapter as adapter  # noqa: E402

SCHEMA_RUN = "tem.training_run.v1"
SCHEMA_SUMMARY = "tem.training_summary.v1"


class RunLogger:
    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = out_dir / "train.log"
        self.metrics_path = out_dir / "metrics.jsonl"
        self._text = self.text_path.open("w", encoding="utf-8")
        self._metrics = self.metrics_path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"{adapter.utc_now()} {message}"
        print(line, flush=True)
        self._text.write(line + "\n")
        self._text.flush()

    def event(self, payload: dict) -> None:
        self._metrics.write(json.dumps(payload) + "\n")
        self._metrics.flush()

    def close(self) -> None:
        self._text.close()
        self._metrics.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(adapter.DEFAULT_REPO))
    parser.add_argument("--env", default=str(adapter.DEFAULT_ENV))
    parser.add_argument("--out", default=str(adapter.ROOT / "training" / "pilot"))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rollout", type=int, default=12)
    parser.add_argument("--walk-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--schedule",
        choices=("pilot", "upstream"),
        default="pilot",
        help=(
            "pilot scales upstream's iteration ramps to the cap so a short run "
            "exercises every loss term; upstream keeps the published ramps and "
            "needs a long run. Both are recorded explicitly."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto defaults to CPU; cuda explicitly enables GPU training",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch CPU threads; 1 favours reproducibility (0 keeps torch default)",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="also keep a checkpoint every N iterations; 0 saves only the final one",
    )
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument(
        "--overrides",
        default="{}",
        help="JSON object of extra upstream parameter overrides (recorded)",
    )
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=101)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--eval-activations",
        action="store_true",
        help="include raw g_inf/p_inf in the post-training evaluation JSON",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path for warm start (project-generated tensor weights only).",
    )
    return parser


def _write_failure(out_dir: Path, exc: BaseException) -> None:
    payload = {
        "schema": "tem.training_failure.v1",
        "generated_utc": adapter.utc_now(),
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
    }
    adapter.write_json_atomic(out_dir / "failure.json", payload)


def _plain_hyper(value, numpy=None):
    """Remove NumPy scalar containers before saving hyperparameters.

    ``parameter_iteration`` returns NumPy float64 values for some schedules.
    Converting them to Python floats keeps future ``params.pt`` files loadable
    by the restricted checkpoint loader.
    """
    if numpy is None:
        numpy = adapter.require_numpy()
    if isinstance(value, dict):
        return {key: _plain_hyper(item, numpy) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_hyper(item, numpy) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_hyper(item, numpy) for item in value)
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    return value


def _save_checkpoint(model, out_dir: Path, iteration: int | None = None) -> tuple[Path, Path]:
    torch = adapter.require_torch()
    suffix = "" if iteration is None else f"_{iteration:06d}"
    checkpoint = out_dir / f"checkpoint{suffix}.pt"
    params_path = out_dir / f"params{suffix}.pt"
    torch.save(model.state_dict(), str(checkpoint))
    torch.save(_plain_hyper(model.hyper), str(params_path))
    return checkpoint, params_path


def run_training(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(out_dir)
    torch = adapter.require_torch()
    numpy = adapter.require_numpy()
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:  # pragma: no cover - dependency gate above
        scipy_version = None
    if scipy_version is None:
        raise adapter.AdapterError(
            "scipy is not installed; install the training dependencies first "
            "(python -m pip install torch scipy numpy)"
        )
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    upstream = adapter.load_upstream(Path(args.repo))
    environment = adapter.load_environment(Path(args.env))
    device = adapter.resolve_device(args.device, torch)
    try:
        overrides = json.loads(args.overrides)
    except json.JSONDecodeError as exc:
        raise adapter.AdapterError(f"--overrides is not JSON: {exc}") from exc
    if not isinstance(overrides, dict):
        raise adapter.AdapterError("--overrides must be a JSON object")
    if args.iterations < 1:
        raise adapter.AdapterError("--iterations must be at least 1")
    if args.eval_steps < 2:
        raise adapter.AdapterError("--eval-steps must be at least 2")
    if args.walk_length < args.rollout:
        raise adapter.AdapterError("--walk-length must be at least --rollout")

    params = adapter.default_params(upstream, iterations=args.iterations)
    schedule_applied = {}
    if args.schedule == "pilot":
        schedule_applied = adapter.apply_pilot_schedule(params, args.iterations)
    params.update(overrides)
    if params.get("n_x") != adapter.N_OBSERVATIONS:
        raise adapter.AdapterError(
            f"parameter overrides changed n_x to {params.get('n_x')!r}; "
            f"expected {adapter.N_OBSERVATIONS}"
        )
    if params.get("n_actions") != 4 or not params.get("has_static_action"):
        raise adapter.AdapterError(
            "parameter overrides changed the static-action encoding; "
            "expected n_actions=4 and has_static_action=True"
        )

    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = upstream.model.Model(params)
    initialization = {"mode": "scratch", "checkpoint_path": None, "checkpoint_sha256": None}
    if args.init_checkpoint is not None:
        try:
            checkpoint_state = adapter._unwrap_state_dict(
                adapter._load_torch_file(torch, args.init_checkpoint)
            )
            model.load_state_dict(checkpoint_state, strict=True)
        except Exception as exc:
            raise adapter.AdapterError(
                f"Strict warm-start load failed for {args.init_checkpoint}: {exc}"
            )
        initialization = {
            "mode": "warm_start",
            "checkpoint_path": str(args.init_checkpoint.resolve()),
            "checkpoint_sha256": adapter.sha256_file(args.init_checkpoint),
        }
    import hashlib
    state_hash = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        state_hash.update(key.encode("utf-8"))
        state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    initialization["state_sha256"] = state_hash.hexdigest()
    use_cuda = device["device"] == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        adapter.prepare_cuda_model(model, torch)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr_max"])

    heldout = adapter.heldout_assignments(environment)
    session = adapter.TrainingSession(
        upstream,
        environment,
        batch_size=args.batch_size,
        rollout=args.rollout,
        seed=args.seed,
        walk_length=args.walk_length,
        excluded_assignments=heldout.values(),
    )

    run_config = {
        "schema": SCHEMA_RUN,
        "generated_utc": adapter.utc_now(),
        "mode": "trained_tem",
        "status": "running",
        "command": [sys.executable, *sys.argv],
        "repo": str(Path(args.repo).resolve()),
        "environment_file": str(Path(args.env).resolve()),
        "upstream": {
            **upstream.commit,
            "file_sha256": upstream.file_hashes,
            "model_interface": adapter.model_interface(upstream),
        },
        "initialization": initialization,
        "environment_summary": adapter.validate_environment(environment),
        "vocabulary": {
            "size": adapter.N_OBSERVATIONS,
            "A_label_sensor_ids": adapter.evaluation_assignment("A", environment).as_json()["label_sensor_ids"],
            "B_label_sensor_ids": adapter.evaluation_assignment("B", environment).as_json()["label_sensor_ids"],
            "heldout_sensor_at_node": {
                world: list(values) for world, values in heldout.items()
            },
            "training_assignment": (
                "unique cues sampled from 45 active tokens per episode; void slots have no cues. "
                "Exact A/B tuples are excluded, plus 64 validation and 64 test assignments for House 3."
            ),
        },
        "seed": args.seed,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "rollout": args.rollout,
        "walk_length": args.walk_length,
        "schedule_mode": args.schedule,
        "schedule_applied": schedule_applied,
        "parameter_overrides": overrides,
        "model_params": {
            key: params.get(key)
            for key in (
                "n_x", "n_x_c", "n_f", "n_g", "n_g_subsampled", "n_p",
                "n_actions", "has_static_action", "do_sample", "use_p_inf",
                "i_attractor", "eta", "lambda", "kappa",
            )
        },
        "device": device,
        "torch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "scipy_version": scipy_version,
        "python_version": sys.version.split()[0],
        "cuda_available": device["cuda_available"],
        "metric_definitions": {
            "loss_components": [
                "L_p_g", "L_p_x", "L_x_gen", "L_x_g",
                "L_x_p", "L_g", "L_reg_g", "L_reg_p",
            ],
            "accuracy_keys": {
                "p": "generative x from inferred p",
                "g": "generative x from inferred g via memory",
                "gt": "generative x from transitioned g_gen (strict prefix)",
            },
            "scored_predictions": (
                "upstream run.py mask: only locations already visited in that "
                "environment contribute to the loss"
            ),
        },
        "pilot_note": (
            "Capped mechanical proof-of-life run, not a convergence result. "
            "Do not cite loss values as evidence of a trained model."
        ),
    }
    adapter.write_json_atomic(out_dir / "run_config.json", run_config)
    logger.log(
        "starting TEM pilot: iterations={} batch={} rollout={} seed={} schedule={}".format(
            args.iterations, args.batch_size, args.rollout, args.seed, args.schedule
        )
    )
    logger.log(f"upstream commit: {upstream.commit.get('commit')}")
    logger.log(f"device: {device['device']} ({device['reason']})")

    history = []
    stop_reason = None
    iterations_completed = 0
    try:
        for iteration in range(args.iterations):
            iteration_start = time.perf_counter()
            eta, lamb, p2g_scale_offset, lr, _walk_center, loss_weights = (
                upstream.parameters.parameter_iteration(iteration, params)
            )
            model.hyper["eta"] = eta
            model.hyper["lambda"] = lamb
            model.hyper["p2g_scale_offset"] = p2g_scale_offset
            for group in optimizer.param_groups:
                group["lr"] = lr

            chunk, regeneration = session.next_chunk()
            if use_cuda:
                observations = torch.stack([step[1] for step in chunk]).to("cuda")
                chunk = [
                    [locations, observation, actions]
                    for (locations, _, actions), observation in zip(chunk, observations)
                ]
                loss_weights = loss_weights.to("cuda")
                # Unmodified upstream constructs its temporary tensors here.
                with torch.device("cuda"):
                    forward = model(chunk, session.prev_iter)
            else:
                forward = model(chunk, session.prev_iter)
            loss_info = adapter.compute_loss(forward, session.visited, loss_weights)
            loss = loss_info["loss"]

            stepped = False
            if loss_info["requires_grad"]:
                optimizer.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)
                optimizer.step()
                stepped = True

            session.prev_iter = [forward[-1].detach()]
            accuracy = adapter.forward_accuracies(forward)
            max_hebb = float(session.prev_iter[0].M[0].abs().max())
            elapsed = time.perf_counter() - iteration_start
            event = {
                "t": adapter.utc_now(),
                "iteration": iteration,
                "elapsed_s": round(elapsed, 4),
                "loss": round(float(loss.detach()), 6),
                "loss_components": [round(value, 6) for value in loss_info["components"]],
                "scored_predictions": loss_info["scored_predictions"],
                "optimizer_step": stepped,
                "accuracy_pct": {key: round(value, 4) for key, value in accuracy.items()},
                "lr": float(lr),
                "eta": float(eta),
                "lambda": float(lamb),
                "p2g_scale_offset": float(p2g_scale_offset),
                "max_hebb": max_hebb,
                "regeneration": regeneration,
            }
            logger.event(event)
            history.append(event)
            iterations_completed = iteration + 1

            if args.log_every > 0 and (iteration % args.log_every == 0 or iteration == args.iterations - 1):
                logger.log(
                    "iter {:04d} loss={:.4f} acc_gt={:.2f}% lr={:.2e} eta={:.4f} "
                    "max_hebb={:.4f} scored={}".format(
                        iteration,
                        event["loss"],
                        event["accuracy_pct"]["gt"],
                        event["lr"],
                        event["eta"],
                        event["max_hebb"],
                        event["scored_predictions"],
                    )
                )
            if args.save_every > 0 and (iteration + 1) % args.save_every == 0:
                checkpoint, params_path = _save_checkpoint(model, out_dir, iteration + 1)
                logger.log(f"saved intermediate checkpoint {checkpoint.name}")

            if args.max_seconds > 0 and (time.perf_counter() - started) >= args.max_seconds:
                stop_reason = f"max_seconds={args.max_seconds} reached"
                logger.log(f"stopping early: {stop_reason}")
                break
    except Exception as exc:  # noqa: BLE001 - keep the real traceback
        run_config["status"] = "failed"
        run_config["iterations_completed"] = iterations_completed
        run_config["failure"] = f"{type(exc).__name__}: {exc}"
        adapter.write_json_atomic(out_dir / "run_config.json", run_config)
        _write_failure(out_dir, exc)
        logger.log(f"FAILED at iteration {iterations_completed}: {type(exc).__name__}: {exc}")
        logger.close()
        raise

    if use_cuda:
        device["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        device["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()

    checkpoint, params_path = _save_checkpoint(model, out_dir)
    run_config.update(
        {
            "status": "training_completed" if stop_reason is None else "stopped_early",
            "evaluation": "pending" if not args.no_eval else "skipped (--no-eval)",
            "iterations_completed": iterations_completed,
            "stop_reason": stop_reason,
            "training_seconds": round(time.perf_counter() - started, 3),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": adapter.sha256_file(checkpoint),
            },
            "params": {
                "path": str(params_path),
                "sha256": adapter.sha256_file(params_path),
            },
            "metrics_file": str(out_dir / "metrics.jsonl"),
        }
    )
    adapter.write_json_atomic(out_dir / "run_config.json", run_config)

    loss_values = [event["loss"] for event in history]
    last_ten = loss_values[-10:]
    summary = {
        "schema": SCHEMA_SUMMARY,
        "generated_utc": adapter.utc_now(),
        "mode": "trained_tem",
        "status": run_config["status"],
        "evaluation": run_config["evaluation"],
        "iterations_completed": iterations_completed,
        "training_seconds": run_config["training_seconds"],
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "loss_mean_last_10": (sum(last_ten) / len(last_ten)) if last_ten else None,
        "optimizer_steps": sum(1 for event in history if event["optimizer_step"]),
        "checkpoint": run_config["checkpoint"],
        "params": run_config["params"],
        "pilot_convergence_claim": False,
        "pilot_note": run_config["pilot_note"],
        "evaluations": {},
    }

    if not args.no_eval:
        try:
            loaded_model, _loaded_params, load_info = adapter.load_model_checkpoint(
                upstream, checkpoint, params_path, seed=args.eval_seed
            )
            provenance = adapter.provenance_snapshot(upstream, load_info)
            for world in ("A", "B"):
                evaluation = adapter.evaluate_model(
                    loaded_model,
                    world,
                    adapter.evaluation_assignment(world, environment),
                    seed=args.eval_seed,
                    steps=args.eval_steps,
                    include_activations=args.eval_activations,
                    environment=environment,
                )
                evaluation["provenance"] = provenance
                eval_path = out_dir / f"eval_world{world}.json"
                adapter.write_json_atomic(eval_path, evaluation)
                summary["evaluations"][world] = {
                    "path": str(eval_path),
                    **evaluation["summary"],
                }
                logger.log(
                    "held-out world {}: top1={} first_visit={} repeat={} (chance={})".format(
                        world,
                        evaluation["summary"]["top1_accuracy"],
                        evaluation["summary"]["first_visit_accuracy"],
                        evaluation["summary"]["repeat_visit_accuracy"],
                        evaluation["summary"]["chance_accuracy"],
                    )
                )
        except Exception as exc:  # noqa: BLE001 - preserve the real failure
            run_config["status"] = "failed_evaluation"
            run_config["evaluation"] = "failed"
            run_config["evaluation_error"] = f"{type(exc).__name__}: {exc}"
            adapter.write_json_atomic(out_dir / "run_config.json", run_config)
            _write_failure(out_dir, exc)
            summary["status"] = "failed_evaluation"
            summary["evaluation"] = "failed"
            summary["evaluation_error"] = run_config["evaluation_error"]
            adapter.write_json_atomic(out_dir / "summary.json", summary)
            logger.log(
                "FAILED during post-training evaluation: "
                f"{type(exc).__name__}: {exc}"
            )
            logger.close()
            raise
        run_config["status"] = "completed" if stop_reason is None else "stopped_early"
        run_config["evaluation"] = "completed"
        adapter.write_json_atomic(out_dir / "run_config.json", run_config)
        summary["status"] = run_config["status"]
        summary["evaluation"] = "completed"
    else:
        run_config["status"] = "completed" if stop_reason is None else "stopped_early"
        summary["status"] = run_config["status"]
        adapter.write_json_atomic(out_dir / "run_config.json", run_config)

    adapter.write_json_atomic(out_dir / "summary.json", summary)
    logger.log(
        "finished {} iterations in {:.2f}s; pilot result is not a convergence claim".format(
            iterations_completed, run_config["training_seconds"]
        )
    )
    logger.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_training(args)
    except adapter.AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if "not installed" in str(exc):
            return 3
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
