"""Prepare, check, fit, or score TEM on native, labelled passage graphs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import access_graph_worlds as worlds
import tem_trained_adapter as adapter
from train_tem import _save_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "tem.native_access.prepared.v1"
CODE_FILES = (
    "scripts/access_graph_tem.py", "scripts/access_graph_worlds.py",
    "scripts/tem_trained_adapter.py",
    "scripts/train_tem.py", "docker/upstream.lock.json", "docker/pyproject.toml", "docker/uv.lock",
)


def validate_settings(config):
    training, evaluation = config["training"], config["evaluation"]
    for section, settings, minimums, extra in (
        ("training", training, {
            "iterations": 1, "batch_size": 1, "rollout": 1, "walk_length": 2,
            "seed": 0, "threads": 1, "save_every": 0,
        }, {"device", "schedule", "max_seconds"}),
        ("evaluation", evaluation, {"episodes": 1, "steps": 2, "seed": 0}, set()),
    ):
        if settings.keys() != minimums.keys() | extra:
            raise ValueError(f"unexpected or missing {section} settings")
        for key, minimum in minimums.items():
            if type(settings[key]) is not int or settings[key] < minimum:
                raise ValueError(f"{section}.{key} must be an integer of at least {minimum}")
        if settings["seed"] >= 2**32:
            raise ValueError(f"{section}.seed must be less than 2**32")
    if training["device"] not in ("cpu", "cuda") or training["schedule"] not in ("pilot", "upstream"):
        raise ValueError("expected CPU or CUDA device and pilot or upstream schedule")
    seconds = training["max_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("training.max_seconds must be finite and nonnegative")


def provenance(repo, config):
    pin = json.loads((ROOT / "docker/upstream.lock.json").read_text(encoding="utf-8"))
    for relative, expected in pin["files"].items():
        if adapter.sha256_file(repo / relative) != expected:
            raise ValueError(f"pinned torch_tem file changed: {relative}")
    hashes = {name: adapter.sha256_file(ROOT / name) for name in CODE_FILES}
    if any(
        graph["split"] == "train" and graph["id"] == worlds.HOUSE3_GRAPH_ID
        for graph in config["graphs"]
    ):
        grid = worlds.require_house3_grid()
        hashes["local:architectural_grid.py"] = adapter.sha256_file(Path(grid.__file__))
    return {"upstream": pin, "code_sha256": hashes}


def manifest_for(config_bytes, config, repo):
    return {
        "schema": SCHEMA,
        "status": "prepared",
        "optimizer_steps": 0,
        "mode": config["mode"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "action_ids": {str(key): label for key, label in worlds.ACTION_LABELS.items()},
        "training": config["training"],
        "evaluation": config["evaluation"],
        "graphs": [graph_summary(graph) for graph in config["graphs"]],
        **provenance(repo, config),
    }


def graph_summary(graph):
    summary = {
        "id": graph["id"], "split": graph["split"], "spaces": len(graph["names"]),
        "passages": len(graph["edges"]), "layers": max(graph["depths"]) + 1,
    }
    if "fully_open" in graph:
        summary.update({
            "fully_open": graph["fully_open"],
            "closed_passages": [list(pair) for pair in graph["closed_passages"]],
            "reference_layers": max(graph["reference_depths"]) + 1,
        })
    return summary


def prepare(config_path, out, repo):
    config_bytes = config_path.read_bytes()
    config = worlds.load_config(config_path)
    validate_settings(config)
    TrainingBatches(config)
    manifest = manifest_for(config_bytes, config, repo)
    if config_path.read_bytes() != config_bytes:
        raise ValueError("configuration changed during preparation")
    out.mkdir(parents=True, exist_ok=False)
    (out / "experiment.toml").write_bytes(config_bytes)
    adapter.write_json_atomic(out / "manifest.json", manifest)
    return manifest


def load_prepared(prepared, repo):
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    config_path = prepared / "experiment.toml"
    if manifest["schema"] != SCHEMA or manifest["status"] != "prepared":
        raise ValueError("expected native access prepared manifest")
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != manifest["config_sha256"]:
        raise ValueError("prepared configuration digest mismatch")
    config = worlds.load_config(config_path)
    validate_settings(config)
    if manifest != manifest_for(config_bytes, config, repo):
        raise ValueError("prepared manifest, code or upstream provenance no longer matches")
    return config, manifest


def episode_assignment(graph, suite, episode, seed):
    return worlds.cue_assignment(graph, random.Random(f"cues:{seed}:{suite}:{graph['id']}:{episode}"))


def projected_reservations(graph):
    """Exclude the old 5×5 held-out room assignments from native House 3 fitting."""
    grid = worlds.require_house3_grid()
    environment = grid.build_environment(5)
    slots = {label: int(index) for index, label in environment["architecture"]["source_labels"].items()}
    return {
        tuple(assignment[slots[name]] for name in graph["names"])
        for assignment in grid.reserved_assignments(environment)
    }


class TrainingBatches:
    """Walk equal-length episodes in every lane and reset all lanes together."""

    def __init__(self, config):
        self.settings = config["training"]
        self.graphs = [graph for graph in config["graphs"] if graph["split"] == "train"]
        self.rng = random.Random(self.settings["seed"])
        self.rng.shuffle(self.graphs)
        self.schedule = self.graphs
        if config["mode"] == "closure-mixture":
            open_graphs = [graph for graph in self.graphs if graph["fully_open"]]
            closed_graphs = [graph for graph in self.graphs if not graph["fully_open"]]
            if len(open_graphs) != 1 or not closed_graphs:
                raise ValueError("closure mixture needs one fully open graph and at least one closed graph")
            self.schedule = []
            for graph in closed_graphs:
                self.schedule.extend((open_graphs[0], graph))
        self.visited = []
        self.graph_counts = Counter()
        self.rollout = self.settings["rollout"]
        self.walk_length = self.settings["walk_length"]
        if self.walk_length % self.rollout or self.walk_length <= max(
            len(graph["names"]) for graph in self.graphs
        ):
            raise ValueError("use a walk length divisible by rollout and longer than the largest graph")
        evaluation = config["evaluation"]
        self.excluded = {
            graph["id"]: {
                episode_assignment(graph, "remap", episode, evaluation["seed"])
                for episode in range(evaluation["episodes"])
            }
            for graph in self.graphs
        }
        if config["mode"] == "closure-mixture":
            reserved = {
                episode_assignment(
                    graph, "remap" if graph["split"] == "train" else graph["split"],
                    episode, evaluation["seed"],
                )
                for graph in config["graphs"]
                for episode in range(evaluation["episodes"])
            }
            self.excluded = dict.fromkeys(self.excluded, reserved)
        for graph in self.graphs:
            if graph["id"] == worlds.HOUSE3_GRAPH_ID:
                self.excluded[graph["id"]].update(projected_reservations(graph))
        self.cursor = 0
        self.position = self.walk_length
        self.episodes = []

    def _reset(self):
        self.visited = []
        self.episodes = []
        for _ in range(self.settings["batch_size"]):
            graph = self.schedule[self.cursor % len(self.schedule)]
            self.cursor += 1
            self.graph_counts[graph["id"]] += 1
            cues = tuple(worlds.cue_assignment(graph, self.rng))
            while cues in self.excluded[graph["id"]]:
                cues = tuple(worlds.cue_assignment(graph, self.rng))
            nodes, actions = worlds.walk(graph, self.rng, self.walk_length)
            self.visited.append([False] * len(cues))
            self.episodes.append((nodes, [cues[node] for node in nodes], actions))
        self.position = 0

    def next(self, torch):
        reset = self.position == self.walk_length
        if reset:
            self._reset()
        window = slice(self.position, self.position + self.rollout)
        self.position += self.rollout
        nodes, cues, actions = (
            list(zip(*(episode[field][window] for episode in self.episodes)))
            for field in range(3)
        )
        return model_walk(torch, cues, actions), nodes, reset


def model_walk(torch, cues, actions):
    observations = torch.nn.functional.one_hot(
        torch.tensor(cues, dtype=torch.long), num_classes=worlds.N_OBSERVATIONS,
    ).to(torch.float32)
    return [
        [[{"shiny": None} for _ in outbound], observation, list(outbound)]
        for observation, outbound in zip(observations, actions)
    ]


def initialize(config, repo):
    torch = adapter.require_torch()
    numpy = adapter.require_numpy()
    settings = config["training"]
    torch.set_num_threads(settings["threads"])
    numpy.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    upstream = adapter.load_upstream(repo)
    params = adapter.default_params(upstream, iterations=settings["iterations"])
    params["n_actions"] = len(worlds.ACTION_LABELS) - 1
    if settings["schedule"] == "pilot":
        adapter.apply_pilot_schedule(params, settings["iterations"])
    return torch, upstream, params, upstream.model.Model(params)


def scheduled_weights(upstream, model, params, update):
    eta, lamb, offset, lr, _, weights = upstream.parameters.parameter_iteration(update, params)
    model.hyper.update({"eta": eta, "lambda": lamb, "p2g_scale_offset": offset})
    return lr, weights


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def logits_for_walk(torch, model, cues, actions, *, device="cpu"):
    with torch.no_grad(), torch.device(device):
        output = model(model_walk(torch, [(cue,) for cue in cues], [(action,) for action in actions]))
        logits = torch.stack([step.x_logits[2][0] for step in output]).cpu()
    if tuple(logits.shape) != (len(cues), worlds.N_OBSERVATIONS) or not torch.isfinite(logits).all():
        raise ValueError("invalid native access logits")
    return logits


def predict(torch, model, nodes, actions, assignment, *, device="cpu"):
    """Frozen inference. Callers score from t=1 because t=0 has no prefix."""
    if len(nodes) != len(actions) or not nodes:
        raise ValueError("a prediction walk needs equally many node and outbound action entries")
    return logits_for_walk(torch, model, [assignment[node] for node in nodes], actions, device=device)


def score_walk(graph, nodes, actions, assignment, logits):
    log_probs = logits.double().log_softmax(dim=-1)
    predictions = log_probs.argmax(dim=-1).tolist()
    last_visit = {nodes[0]: 0}
    crossed = set()
    observed = {}
    events = []
    for t, (origin, node) in enumerate(zip(nodes, nodes[1:]), 1):
        action = actions[t - 1]
        if not action or graph["transitions"][origin].get(action) != node:
            raise ValueError(f"illegal recorded passage at t={t}")
        actual = assignment[node]
        key = (assignment[origin], action)
        edge = frozenset((origin, node))
        previous = last_visit.get(node)
        guess = observed.get(key)
        events.append({
            "t": t, "origin": graph["names"][origin], "arrival": graph["names"][node],
            "action": worlds.ACTION_LABELS[action], "actual": actual,
            "pre_reveal_top1": predictions[t], "hit": predictions[t] == actual,
            "pre_reveal_nll": -float(log_probs[t, actual]),
            "coverage_before": len(last_visit) / len(graph["names"]),
            "revisit_lag": t - previous if previous is not None else None,
            "strict_return": previous is not None and edge not in crossed,
            "observed_transition_hit": None if guess is None else guess == actual,
        })
        last_visit[node] = t
        crossed.add(edge)
        observed[key] = actual
    return {"start": graph["names"][nodes[0]], "initial_cue": assignment[nodes[0]],
            "visited": len(last_visit), "events": events}


def summarize(events):
    count = len(events)
    hits = sum(event["hit"] for event in events)
    observed = [event["observed_transition_hit"] for event in events
                if event["observed_transition_hit"] is not None]
    return {
        "eligible": count, "hits": hits, "top1": hits / count if count else None,
        "mean_nll": math.fsum(event["pre_reveal_nll"] for event in events) / count if count else None,
        "lookup_predictions": len(observed), "lookup_hits": sum(observed),
        "lookup_coverage": len(observed) / count if count else None,
    }


def check(prepared, repo, requested_device="cpu", *, backward=False):
    config, manifest = load_prepared(prepared, repo)
    batches = TrainingBatches(config)
    torch, upstream, params, model = initialize(config, repo)
    runtime = adapter.resolve_device(requested_device, torch)
    device = runtime["device"]
    if device == "cuda":
        adapter.prepare_cuda_model(model, torch)
    model.train(backward)
    before = state_digest(model)
    _, weights = scheduled_weights(upstream, model, params, 0)
    prev_iter = None
    visited_loss_examples = 0
    gradient_tensors = 0
    with torch.set_grad_enabled(backward):
        for _ in range(batches.walk_length // batches.rollout):
            with torch.device(device):
                chunk, nodes, reset = batches.next(torch)
                forward = model(chunk, None if reset else prev_iter)
                loss = adapter.compute_loss(forward, batches.visited, weights.to(device), node_ids=nodes)
            if not torch.isfinite(loss["loss"]):
                raise ValueError("non-finite visited-only loss in preflight")
            prev_iter = [forward[-1].detach()]
            visited_loss_examples += loss["scored_predictions"]
            if visited_loss_examples:
                if backward:
                    loss["loss"].backward()
                    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                        raise ValueError("preflight produced no gradients or non-finite gradients")
                    gradient_tensors = len(gradients)
                    model.zero_grad(set_to_none=True)
                break
    if not visited_loss_examples:
        raise ValueError("preflight produced no visited-location loss examples")
    model.eval()
    graph = next((g for g in config["graphs"] if g["split"] == "test"), batches.graphs[0])
    nodes, actions = worlds.walk(graph, random.Random(17), 32)
    assignment = episode_assignment(graph, "check", 0, config["evaluation"]["seed"])
    cues = [assignment[node] for node in nodes]
    logits = logits_for_walk(torch, model, cues, actions, device=device)
    t = len(nodes) - 1
    changed_cue = cues[:-1] + [(cues[-1] + 1) % worlds.N_OBSERVATIONS]
    changed_action = list(actions[:-1]) + [1]
    counterfactuals = {
        "arrival_cue": float((logits[t] - logits_for_walk(torch, model, changed_cue, actions, device=device)[t]).abs().max()),
        "current_outbound_action": float((logits[t] - logits_for_walk(torch, model, cues, changed_action, device=device)[t]).abs().max()),
    }
    if max(counterfactuals.values()) > 1e-6:
        raise ValueError(f"strict-prefix counterfactual failed: {counterfactuals}")
    trace = score_walk(graph, nodes, actions, assignment, logits)
    after = state_digest(model)
    if after != before:
        raise ValueError("preflight changed model weights")
    return {
        "status": "untrained_preflight", "optimizer_steps": 0, "device": device,
        "gpu": runtime.get("gpu"), "gradient_tensors": gradient_tensors,
        "config_sha256": manifest["config_sha256"], "torch": str(torch.__version__),
        "visited_loss_examples": visited_loss_examples, "training_graph_episodes": dict(batches.graph_counts),
        "strict_prefix_max_logit_delta": counterfactuals,
        "weights_unchanged": True, "state_sha256_before": before, "state_sha256_after": after,
        "scorer_check": {"graph": graph["id"], "events": len(trace["events"]),
                         "strict_returns": sum(event["strict_return"] for event in trace["events"])},
    }


def train(prepared, repo, approved_config_sha256):
    config, manifest = load_prepared(prepared, repo)
    if not approved_config_sha256 or approved_config_sha256 != manifest["config_sha256"]:
        raise ValueError("training blocked until the exact prepared configuration digest is approved")
    settings = config["training"]
    batches = TrainingBatches(config)
    out = prepared / "training"
    torch, upstream, params, model = initialize(config, repo)
    runtime = adapter.resolve_device(settings["device"], torch)
    device = runtime["device"]
    if device == "cuda":
        adapter.prepare_cuda_model(model, torch)
    model.train()
    out.mkdir(exist_ok=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr_max"])
    prev_iter = None
    optimizer_steps = 0
    chunk_count = 0
    started = time.perf_counter()
    with (out / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
        while optimizer_steps < settings["iterations"]:
            if settings["max_seconds"] and time.perf_counter() - started >= settings["max_seconds"]:
                break
            lr, weights = scheduled_weights(upstream, model, params, optimizer_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with torch.device(device):
                chunk, nodes, reset = batches.next(torch)
                forward = model(chunk, None if reset else prev_iter)
                loss = adapter.compute_loss(forward, batches.visited, weights.to(device), node_ids=nodes)
            if not torch.isfinite(loss["loss"]):
                raise ValueError("non-finite training loss")
            if loss["requires_grad"]:
                optimizer.zero_grad(set_to_none=True)
                loss["loss"].backward(retain_graph=True)
                optimizer.step()
                optimizer_steps += 1
                if settings["save_every"] and optimizer_steps % settings["save_every"] == 0:
                    _save_checkpoint(model, out, optimizer_steps)
                if optimizer_steps == 1 or optimizer_steps % 100 == 0:
                    print(f"{optimizer_steps}/{settings['iterations']} optimizer updates on {device} "
                          f"({time.perf_counter() - started:.1f} s)", file=sys.stderr, flush=True)
            prev_iter = [forward[-1].detach()]
            chunk_count += 1
            metrics.write(json.dumps({
                "chunk": chunk_count, "optimizer_steps": optimizer_steps,
                "visited_loss_examples": loss["scored_predictions"], "loss": float(loss["loss"].detach()),
            }) + "\n")
            metrics.flush()
    checkpoint, params_path = _save_checkpoint(model, out)
    summary = {
        "status": "fitted" if optimizer_steps == settings["iterations"] else "stopped_early",
        "config_sha256": manifest["config_sha256"], "device": device,
        "gpu": runtime.get("gpu"),
        "manifest_sha256": adapter.sha256_file(prepared / "manifest.json"),
        "optimizer_steps": optimizer_steps, "chunks": chunk_count,
        "graph_episodes": dict(batches.graph_counts),
        "checkpoint_sha256": adapter.sha256_file(checkpoint),
        "params_sha256": adapter.sha256_file(params_path),
        "torch": str(torch.__version__), "seconds": time.perf_counter() - started,
    }
    adapter.write_json_atomic(out / "summary.json", summary)
    return summary


def load_fitted(prepared: Path, repo: Path):
    """Return a frozen native model, prepared config, and verified checkpoint provenance."""
    config, manifest = load_prepared(prepared, repo)
    out = prepared / "training"
    summary_path = out / "summary.json"
    if not summary_path.is_file():
        raise ValueError("evaluation requires the completed checkpoint from this prepared run")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary["status"] != "fitted"
        or summary["optimizer_steps"] != config["training"]["iterations"]
        or summary["manifest_sha256"] != adapter.sha256_file(prepared / "manifest.json")
        or summary["config_sha256"] != manifest["config_sha256"]
    ):
        raise ValueError("evaluation requires the completed checkpoint from this prepared run")
    checkpoint, params_path = out / "checkpoint.pt", out / "params.pt"
    if adapter.sha256_file(checkpoint) != summary["checkpoint_sha256"] or (
        adapter.sha256_file(params_path) != summary["params_sha256"]
    ):
        raise ValueError("fitted checkpoint or parameters digest mismatch")
    torch = adapter.require_torch()
    torch.set_num_threads(config["training"]["threads"])
    upstream = adapter.load_upstream(repo)
    model, params, _ = adapter.load_model_checkpoint(
        upstream, checkpoint, params_path, seed=config["evaluation"]["seed"]
    )
    if params["n_actions"] != len(worlds.ACTION_LABELS) - 1 or not params["has_static_action"]:
        raise ValueError("checkpoint does not match the native action vocabulary")
    return model, config, summary


def evaluate(prepared, repo, suite, out, requested_device="cpu"):
    model, config, fitted = load_fitted(prepared, repo)
    torch = adapter.require_torch()
    device = adapter.resolve_device(requested_device, torch)["device"]
    if device == "cuda":
        adapter.prepare_cuda_model(model, torch)
    settings = config["evaluation"]
    split = "train" if suite == "remap" else suite
    graphs = [graph for graph in config["graphs"] if graph["split"] == split]
    if not graphs:
        raise ValueError(f"prepared configuration has no graphs for {suite}")
    before = state_digest(model)
    results = []
    for graph in graphs:
        episodes = []
        for index in range(settings["episodes"]):
            rng = random.Random(f"walk:{settings['seed']}:{suite}:{graph['id']}:{index}")
            nodes, actions = worlds.walk(graph, rng, settings["steps"])
            assignment = episode_assignment(graph, suite, index, settings["seed"])
            logits = predict(torch, model, nodes, actions, assignment, device=device)
            episodes.append(score_walk(graph, nodes, actions, assignment, logits))
        events = [event for episode in episodes for event in episode["events"]]
        results.append({
            "graph": graph["id"], "graph_summary": graph_summary(graph), "episodes": episodes,
            "ordinary_revisit": summarize([event for event in events if event["revisit_lag"] is not None]),
            "strict_return": summarize([event for event in events if event["strict_return"]]),
        })
    after = state_digest(model)
    if after != before:
        raise ValueError("frozen evaluation changed model weights")
    result = {
        "status": "frozen_evaluation", "suite": suite, "prediction_head": "x_logits[2]", "device": device,
        "state_sha256_before": before, "state_sha256_after": after,
        "checkpoint": fitted, "settings": settings, "graphs": results,
    }
    with out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    return {"status": result["status"], "suite": suite, "out": str(out), "graphs": [
        {"graph": row["graph"], "strict_return": row["strict_return"]} for row in results
    ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check", "train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--repo", type=Path, default=adapter.DEFAULT_REPO)
        if name == "prepare":
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--out", type=Path, required=True)
        else:
            command.add_argument("--prepared", type=Path, required=True)
        if name in ("check", "evaluate"):
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if name == "check":
            command.add_argument("--backward", action="store_true", help="verify finite gradients without an optimizer")
        if name == "train":
            command.add_argument("--approve-config-sha256")
        if name == "evaluate":
            command.add_argument("--suite", choices=("validation", "test", "remap"), required=True)
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.config, args.out, args.repo)
        elif args.command == "check":
            result = check(args.prepared, args.repo, args.device, backward=args.backward)
        elif args.command == "train":
            result = train(args.prepared, args.repo, args.approve_config_sha256)
        else:
            result = evaluate(args.prepared, args.repo, args.suite, args.out, args.device)
    except (OSError, KeyError, ValueError, adapter.AdapterError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
