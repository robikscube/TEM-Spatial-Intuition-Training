"""Contract and real-model tests for scripts/train_tem.py and its adapter.

Static tests (no torch) run everywhere. Tests marked with
``@unittest.skipUnless(adapter.torch_available(), ...)`` execute a real
upstream ``Model.forward``/``backward`` and a real checkpoint round-trip; they
are not mocked.

Run:
    python -m unittest discover -s tests/training -p "test_*.py" -v
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import tem_trained_adapter as adapter  # noqa: E402
import train_tem  # noqa: E402

UPSTREAM = ROOT / "research" / "reference" / "torch_tem"
ENV_PATH = UPSTREAM / "envs" / "4x4.json"

TRAINING_DEPS_AVAILABLE = adapter.training_dependencies_available()


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeIteration:
    def __init__(self, logits):
        self.x_logits = ([], [], [FakeTensor(logits)])
        self.g_inf = []
        self.p_inf = []


class ExactActionEncodingTest(unittest.TestCase):
    def test_4x4_json_uses_stay_north_east_south_west(self):
        environment = adapter.load_4x4(ENV_PATH)
        summary = adapter.validate_4x4(environment)
        self.assertEqual(summary["n_locations"], 16)
        self.assertEqual(summary["n_observations"], 45)
        self.assertEqual(summary["n_actions"], 5)
        self.assertEqual(
            summary["action_ids"],
            {"0": "stay", "1": "N", "2": "E", "3": "S", "4": "W"},
        )
        self.assertEqual(
            summary["move_degree"],
            [
                2, 3, 3, 2,
                3, 4, 4, 3,
                3, 4, 4, 3,
                2, 3, 3, 2,
            ],
        )

    def test_north_from_corner_is_blocked_in_the_json(self):
        environment = json.loads(ENV_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            environment["locations"][0]["actions"][1]["transition"],
            [0] * 16,
        )
        self.assertEqual(
            environment["locations"][0]["actions"][2]["transition"][1], 1
        )
        self.assertEqual(
            environment["locations"][0]["actions"][3]["transition"][4], 1
        )

    def test_tampered_action_encoding_is_rejected(self):
        environment = json.loads(ENV_PATH.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(environment)
        tampered["locations"][0]["actions"][1]["transition"][1] = 1
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_4x4(tampered)

    def test_blocked_action_with_nonzero_transition_is_rejected(self):
        environment = json.loads(ENV_PATH.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(environment)
        tampered["locations"][0]["actions"][1]["transition"][0] = 1
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_4x4(tampered)


class HeldOutVocabularyTest(unittest.TestCase):
    def test_world_assignments_are_stable_distinct_and_in_vocabulary(self):
        first_a = adapter.ui_world_assignment("A")
        second_a = adapter.ui_world_assignment("A")
        world_b = adapter.ui_world_assignment("B")
        self.assertEqual(first_a, second_a)
        self.assertEqual(len(set(first_a.node_to_sensor)), 16)
        self.assertEqual(len(set(world_b.node_to_sensor)), 16)
        self.assertEqual(set(first_a.node_to_sensor), set(range(16)))
        self.assertEqual(set(world_b.node_to_sensor), set(range(16, 32)))
        self.assertTrue(
            all(
                left != right
                for left, right in zip(
                    first_a.node_to_sensor, world_b.node_to_sensor
                )
            )
        )
        self.assertTrue(
            all(
                0 <= sensor < adapter.N_OBSERVATIONS
                for sensor in first_a.node_to_sensor + world_b.node_to_sensor
            )
        )

    def test_ui_order_is_a_permutation_of_the_label_list(self):
        for world, labels in (("A", adapter.ITEM_LABELS_A), ("B", adapter.ITEM_LABELS_B)):
            assignment = adapter.ui_world_assignment(world)
            self.assertEqual(sorted(assignment.node_to_label), sorted(labels))

    def test_training_assignments_exclude_both_held_out_worlds(self):
        heldout = adapter.heldout_assignments()
        excluded = set(heldout.values())
        for seed in range(25):
            assignment = adapter.random_training_assignment(seed, excluded)
            self.assertEqual(len(set(assignment)), 16)
            self.assertTrue(all(0 <= token < 45 for token in assignment))
            self.assertNotIn(assignment, excluded)
            self.assertNotIn(
                assignment,
                (heldout["A"], heldout["B"]),
            )


class StrictPrefixExtractionTest(unittest.TestCase):
    def test_step_zero_is_null_and_first_repeat_counts_are_correct(self):
        logits_1 = [0.0] * 45
        logits_1[6] = 5.0
        logits_2 = [0.0] * 45
        logits_2[7] = 5.0
        forward = [FakeIteration([0.0] * 45), FakeIteration(logits_1), FakeIteration(logits_2)]

        steps, summary = adapter.predictions_from_forward(
            forward,
            nodes=[0, 1, 1],
            sensors=[5, 6, 7],
            actions=[1, 1, 0],
        )

        self.assertIsNone(steps[0]["prediction"])
        self.assertIsNone(steps[0]["action_from_previous"])
        self.assertTrue(steps[1]["first_visit"])
        self.assertEqual(steps[1]["action_from_previous"], "N")
        self.assertTrue(steps[1]["correct_top1"])
        self.assertFalse(steps[2]["first_visit"])
        self.assertTrue(steps[2]["correct_top1"])
        self.assertEqual(summary["predictions_scored"], 2)
        self.assertEqual(summary["top1_accuracy"], 1.0)
        self.assertEqual(summary["first_visit_accuracy"], 1.0)
        self.assertEqual(summary["repeat_visit_accuracy"], 1.0)
        probabilities = steps[1]["prediction"]["probs"]
        self.assertAlmostEqual(sum(probabilities), 1.0, delta=1e-5)  # JSON rounds per-class probabilities to 6 decimals

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(adapter.AdapterError):
            adapter.predictions_from_forward(
                [FakeIteration([0.0] * 45)],
                nodes=[0],
                sensors=[1, 2],
                actions=[1],
            )


class EvaluationScaffoldTest(unittest.TestCase):
    def test_walk_is_legal_deterministic_and_labelled(self):
        walk = adapter.generate_evaluation_walk(seed=7, steps=32)
        repeated = adapter.generate_evaluation_walk(seed=7, steps=32)
        self.assertEqual(walk, repeated)
        self.assertEqual(len(walk["nodes"]), 32)
        self.assertEqual(len(walk["actions"]), 32)
        self.assertEqual(walk["nodes"][0], 0)
        self.assertEqual(walk["actions"][-1], 0)
        self.assertIn("not TEM planning", walk["controller"])
        for index in range(31):
            action = walk["actions"][index]
            self.assertIn(action, adapter.ACTION_LABELS)
            self.assertEqual(
                adapter.expected_target(walk["nodes"][index], action),
                walk["nodes"][index + 1],
            )

    def test_different_seeds_can_produce_different_walks(self):
        first = adapter.generate_evaluation_walk(seed=1, steps=16)
        second = adapter.generate_evaluation_walk(seed=2, steps=16)
        self.assertNotEqual(first["actions"], second["actions"])


class PreconditionTest(unittest.TestCase):
    def test_missing_upstream_clone_is_reported(self):
        with self.assertRaises(adapter.AdapterError):
            adapter.load_upstream(ROOT / "research" / "reference" / "absent")

    def test_missing_checkpoint_is_reported_before_importing_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.pt"
            with self.assertRaises(adapter.AdapterError) as context:
                adapter.load_model_checkpoint(object(), missing, missing)
            self.assertIn("checkpoint not found", str(context.exception))

    def test_model_walk_rejects_bad_values(self):
        if not TRAINING_DEPS_AVAILABLE:
            self.skipTest("torch/numpy/scipy not installed")
        with self.assertRaises(adapter.AdapterError):
            adapter.build_model_walk([0], [45], [1])
        with self.assertRaises(adapter.AdapterError):
            adapter.build_model_walk([0], [0], [5])
        with self.assertRaises(adapter.AdapterError):
            adapter.build_model_walk([16], [0], [1])


@unittest.skipUnless(shutil.which("git"), "git not available")
class UpstreamCommitTest(unittest.TestCase):
    def run_git(self, repo: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_untracked_bytecode_does_not_mark_tracked_source_dirty(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            self.assertEqual(self.run_git(repo, "init").returncode, 0)
            (repo / "tracked.py").write_text("value = 1\n", encoding="utf-8")
            self.assertEqual(self.run_git(repo, "add", "tracked.py").returncode, 0)
            commit = subprocess.run(
                [
                    "git", "-C", str(repo),
                    "-c", "user.name=training-test",
                    "-c", "user.email=training-test@example.invalid",
                    "commit", "-m", "init",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(commit.returncode, 0, commit.stderr)

            # Importing upstream creates __pycache__/; it must not look dirty.
            (repo / "__pycache__").mkdir()
            (repo / "__pycache__" / "tracked.cpython-313.pyc").write_bytes(b"bytecode")
            clean = adapter.read_upstream_commit(repo)
            self.assertIsNotNone(clean["commit"])
            self.assertIs(clean["dirty"], False, clean)

            (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
            modified = adapter.read_upstream_commit(repo)
            self.assertIs(modified["dirty"], True, modified)


@unittest.skipUnless(TRAINING_DEPS_AVAILABLE, "torch/numpy/scipy not installed")
class RealModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = adapter.load_upstream(UPSTREAM)
        cls.torch = adapter.require_torch()

    def test_interface_matches_the_verified_upstream_forward(self):
        interface = adapter.model_interface(self.upstream)
        self.assertEqual(interface["model_class"], "Model")
        self.assertEqual(
            interface["forward_parameters"], ["walk", "prev_iter", "prev_M"]
        )

    def test_real_forward_produces_45_way_logits_and_raw_activations(self):
        params = adapter.default_params(self.upstream, iterations=2)
        model = self.upstream.model.Model(params)
        model.eval()
        assignment = adapter.ui_world_assignment("A")
        walk = adapter.generate_evaluation_walk(seed=7, steps=6)
        sensors = [assignment.node_to_sensor[node] for node in walk["nodes"]]
        model_walk = adapter.build_model_walk(
            walk["nodes"], sensors, walk["actions"], torch=self.torch
        )
        self.assertIsInstance(model_walk[0][0], list)
        self.assertEqual(model_walk[0][0][0]["id"], walk["nodes"][0])
        self.assertEqual(tuple(model_walk[0][1].shape), (1, 45))
        self.assertEqual(model_walk[0][2], [walk["actions"][0]])
        with self.torch.no_grad():
            forward = model(model_walk, None)

        self.assertEqual(len(forward), 6)
        self.assertEqual(tuple(forward[-1].x_logits[2].shape), (1, 45))
        self.assertEqual(len(forward[-1].g_inf), params["n_f"])
        self.assertEqual(len(forward[-1].p_inf), params["n_f"])
        self.assertTrue(
            self.torch.isfinite(forward[-1].x_logits[2]).all().item()
        )
        for module in forward[-1].g_inf:
            self.assertTrue(self.torch.isfinite(module).all().item())

        steps, summary = adapter.predictions_from_forward(
            forward, walk["nodes"], sensors, walk["actions"]
        )
        self.assertIsNone(steps[0]["prediction"])
        self.assertEqual(summary["predictions_scored"], 5)
        self.assertIsNotNone(summary["top1_accuracy"])

    def test_training_session_is_deterministic_for_a_seed(self):
        environment = adapter.load_4x4(ENV_PATH)
        heldout = adapter.heldout_assignments()
        first = adapter.TrainingSession(
            self.upstream,
            environment,
            batch_size=2,
            rollout=2,
            seed=11,
            walk_length=4,
            excluded_assignments=heldout.values(),
        )
        second = adapter.TrainingSession(
            self.upstream,
            environment,
            batch_size=2,
            rollout=2,
            seed=11,
            walk_length=4,
            excluded_assignments=heldout.values(),
        )
        self.assertEqual(first.assignments, second.assignments)
        chunk_a, _ = first.next_chunk()
        chunk_b, _ = second.next_chunk()
        for step_a, step_b in zip(chunk_a, chunk_b):
            self.assertEqual(
                [location["id"] for location in step_a[0]],
                [location["id"] for location in step_b[0]],
            )
            self.assertEqual(step_a[2], step_b[2])
            self.assertTrue(self.torch.equal(step_a[1], step_b[1]))

    def test_compute_loss_masks_first_visits_and_backpropagates(self):
        from types import SimpleNamespace

        parameter = self.torch.nn.Parameter(self.torch.tensor(1.0))
        step = SimpleNamespace(
            g=[{"id": 0}],
            L=[parameter.unsqueeze(0) for _ in range(8)],  # upstream losses have a batch axis
        )
        visited = [[False]]
        first = adapter.compute_loss(
            [step], visited, self.torch.ones(8)
        )
        self.assertEqual(first["scored_predictions"], 0)
        self.assertFalse(first["requires_grad"])

        second = adapter.compute_loss(
            [step], visited, self.torch.ones(8)
        )
        self.assertEqual(second["scored_predictions"], 1)
        self.assertTrue(second["requires_grad"])
        second["loss"].backward()
        self.assertIsNotNone(parameter.grad)
        self.assertGreater(float(parameter.grad), 0.0)

    def test_project_checkpoint_loader_accepts_tensors_and_primitives(self):
        numpy = adapter.require_numpy()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "params.pt"
            payload = {
                "n_x": 45,
                "step": 3,
                "flag": True,
                "nested": [{"scale": 1.5}, None],
                "np_scalar": numpy.float64(0.25),
                "tensor": self.torch.arange(4),
            }
            self.torch.save(payload, str(path))
            loaded = adapter._load_torch_file(self.torch, path)
            self.assertEqual(loaded["n_x"], 45)
            self.assertEqual(loaded["step"], 3)
            self.assertIs(loaded["flag"], True)
            self.assertEqual(loaded["nested"], [{"scale": 1.5}, None])
            self.assertEqual(float(loaded["np_scalar"]), 0.25)
            self.assertTrue(self.torch.equal(loaded["tensor"], self.torch.arange(4)))

    def test_cuda_is_explicit_and_requires_an_available_device(self):
        self.assertEqual(adapter.resolve_device("auto", self.torch)["device"], "cpu")
        if self.torch.cuda.is_available():
            self.assertEqual(adapter.resolve_device("cuda", self.torch)["device"], "cuda")
        else:
            with self.assertRaisesRegex(adapter.AdapterError, "CUDA is unavailable"):
                adapter.resolve_device("cuda", self.torch)

    def test_pilot_training_writes_real_checkpoint_metrics_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            code = train_tem.main(
                [
                    "--repo", str(UPSTREAM),
                    "--env", str(ENV_PATH),
                    "--out", temp,
                    "--iterations", "2",
                    "--batch-size", "1",
                    "--rollout", "24",
                    "--walk-length", "200",
                    "--seed", "3",
                    "--schedule", "pilot",
                    "--threads", "1",
                    "--max-seconds", "120",
                    "--eval-steps", "6",
                    "--eval-seed", "5",
                    "--log-every", "1",
                ]
            )
            self.assertEqual(code, 0)
            out = Path(temp)
            for name in (
                "checkpoint.pt",
                "params.pt",
                "metrics.jsonl",
                "run_config.json",
                "summary.json",
                "train.log",
                "eval_worldA.json",
                "eval_worldB.json",
            ):
                self.assertTrue((out / name).is_file(), name)

            metrics = [
                json.loads(line)
                for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(metrics), 2)
            self.assertTrue(all("loss" in event for event in metrics))
            self.assertTrue(all(len(event["loss_components"]) == 8 for event in metrics))
            self.assertTrue(all(event["scored_predictions"] > 0 for event in metrics))

            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["schema"], "tem.training_summary.v1")
            self.assertEqual(summary["iterations_completed"], 2)
            self.assertFalse(summary["pilot_convergence_claim"])
            self.assertGreaterEqual(summary["optimizer_steps"], 1)
            self.assertIn("A", summary["evaluations"])
            self.assertIn("B", summary["evaluations"])

            run_config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["status"], "completed")
            self.assertEqual(run_config["schedule_mode"], "pilot")
            self.assertEqual(run_config["iterations_completed"], 2)
            self.assertEqual(
                run_config["environment_summary"]["action_ids"]["0"], "stay"
            )
            self.assertIn("eval_worldA.json", json.dumps(summary["evaluations"]["A"]))

            evaluation_a = json.loads(
                (out / "eval_worldA.json").read_text(encoding="utf-8")
            )
            self.assertEqual(evaluation_a["schema"], "tem.evaluation.v1")
            self.assertEqual(evaluation_a["mode"], "trained_tem")
            self.assertIsNone(evaluation_a["steps"][0]["prediction"])
            for step in evaluation_a["steps"][1:]:
                self.assertEqual(len(step["prediction"]["logits"]), 45)
                self.assertAlmostEqual(
                    sum(step["prediction"]["probs"]), 1.0, places=3
                )
                self.assertIn(step["action_from_previous"], ("stay", "N", "E", "S", "W"))
            self.assertFalse(evaluation_a["integrity"]["walk_is_tem_planning"])
            self.assertTrue(
                evaluation_a["integrity"]["predictions_use_strict_prefix"]
            )
            self.assertEqual(
                evaluation_a["provenance"]["upstream_commit"],
                self.upstream.commit.get("commit"),
            )

            state = adapter._load_torch_file(self.torch, out / "checkpoint.pt")
            self.assertIsInstance(state, dict)
            self.assertTrue(state)
            loaded_params = adapter._load_torch_file(self.torch, out / "params.pt")
            self.assertEqual(loaded_params["n_x"], 45)
            model = self.upstream.model.Model(loaded_params)
            result = model.load_state_dict(state, strict=True)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
