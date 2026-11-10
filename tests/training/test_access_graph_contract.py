"""Access-graph contract checks with synthetic passages and no fitting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import access_graph_tem as tem  # noqa: E402
import access_graph_worlds as worlds  # noqa: E402
import tem_trained_adapter as adapter  # noqa: E402

UPSTREAM = ROOT / "research" / "reference" / "torch_tem"
CLOSURE_CONFIG = """\
version = 1
mode = "closure-mixture"

[training]
iterations = 2
batch_size = 2
rollout = 4
walk_length = 16
seed = 11
threads = 1
save_every = 0
device = "cpu"
schedule = "pilot"
max_seconds = 0

[evaluation]
episodes = 1
steps = 32
seed = 19

[template]
id = "synthetic-ladder"
crossings = [["alpha", "beta"], ["alpha-mid", "beta-mid"], ["alpha-end", "beta-end"]]

[template.children]
exterior = ["alpha", "beta"]
alpha = ["alpha-mid"]
beta = ["beta-mid"]
alpha-mid = ["alpha-end"]
beta-mid = ["beta-end"]

[[graphs]]
id = "open"
split = "train"
closed = []

[[graphs]]
id = "entry-closure"
split = "train"
closed = [["exterior", "alpha"]]

[[graphs]]
id = "middle-closure"
split = "validation"
closed = [["alpha-mid", "beta-mid"]]

[[graphs]]
id = "end-closure"
split = "test"
closed = [["alpha-end", "beta-end"]]
"""


class AccessGraphContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config_path = self.directory / "synthetic.toml"
        self.config_path.write_text(CLOSURE_CONFIG, encoding="utf-8")

    def test_closing_an_entry_preserves_the_other_child_slot(self):
        config = worlds.load_config(self.config_path)
        graphs = {graph["id"]: graph for graph in config["graphs"]}
        opened = graphs["open"]
        closed = graphs["entry-closure"]
        root, alpha, beta = (closed["label_to_id"][name] for name in ("exterior", "alpha", "beta"))

        self.assertEqual(opened["transitions"][root], {2: alpha, 3: beta})
        self.assertEqual(closed["transitions"][root], {3: beta})
        self.assertNotIn(1, closed["transitions"][alpha])
        self.assertEqual(closed["reference_depths"][alpha], 1)
        self.assertEqual(closed["depths"][alpha], 2)

    def test_same_closure_mask_cannot_appear_in_multiple_splits(self):
        duplicate = CLOSURE_CONFIG.replace(
            'closed = [["alpha-end", "beta-end"]]',
            'closed = [["beta-mid", "alpha-mid"]]',
        )
        self.config_path.write_text(duplicate, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "closure masks must be distinct"):
            worlds.load_config(self.config_path)

    @unittest.skipUnless(adapter.torch_available(), "PyTorch is unavailable")
    def test_strict_return_requires_a_previously_uncrossed_undirected_passage(self):
        graph = worlds.load_config(self.config_path)["graphs"][0]
        root, alpha, beta = (graph["label_to_id"][name] for name in ("exterior", "alpha", "beta"))
        nodes = [root, alpha, beta, root, beta, root]
        actions = [2, 4, 1, 3, 1, 0]
        assignment = tuple(range(len(graph["names"])))
        torch = adapter.require_torch()
        logits = torch.zeros((len(nodes), worlds.N_OBSERVATIONS))
        for index, node in enumerate(nodes):
            logits[index, assignment[node]] = 1.0

        events = tem.score_walk(graph, nodes, actions, assignment, logits)["events"]
        self.assertEqual([event["t"] for event in events if event["strict_return"]], [3])
        self.assertEqual(events[2]["revisit_lag"], 3)
        self.assertIsNone(events[2]["observed_transition_hit"])
        self.assertEqual(events[3]["revisit_lag"], 2)
        self.assertTrue(events[4]["observed_transition_hit"])

    @unittest.skipUnless((UPSTREAM / "model.py").is_file(), "Pinned torch_tem checkout is unavailable")
    def test_prepare_keeps_zero_updates_and_binds_the_exact_config(self):
        prepared = self.directory / "prepared"
        manifest = tem.prepare(self.config_path, prepared, UPSTREAM)
        self.assertEqual(manifest["optimizer_steps"], 0)
        self.assertEqual(manifest["status"], "prepared")
        with self.assertRaisesRegex(ValueError, "completed checkpoint"):
            tem.load_fitted(prepared, UPSTREAM)

        config = prepared / "experiment.toml"
        config.write_bytes(config.read_bytes() + b"\n# changed after preparation\n")
        with self.assertRaisesRegex(ValueError, "prepared configuration digest mismatch"):
            tem.load_prepared(prepared, UPSTREAM)

    @unittest.skipUnless(
        (UPSTREAM / "model.py").is_file() and adapter.training_dependencies_available(),
        "Pinned torch_tem and its runtime dependencies are required",
    )
    def test_evaluation_scores_the_requested_split_with_frozen_weights(self):
        config = worlds.load_config(self.config_path)
        _, _, _, model = tem.initialize(config, UPSTREAM)
        model.eval()
        initial_digest = tem.state_digest(model)
        untrained_fixture = {"status": "untrained_fixture", "optimizer_steps": 0}
        output = self.directory / "test-result.json"

        # Exercise evaluation with genuine logits while leaving the fitted-run guard intact.
        with patch.object(tem, "load_fitted", return_value=(model, config, untrained_fixture)):
            summary = tem.evaluate(self.directory, UPSTREAM, "test", output)

        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "frozen_evaluation")
        self.assertEqual(result["suite"], "test")
        self.assertEqual(result["prediction_head"], "x_logits[2]")
        self.assertEqual(result["checkpoint"], untrained_fixture)
        self.assertEqual(result["state_sha256_before"], initial_digest)
        self.assertEqual(result["state_sha256_after"], initial_digest)
        self.assertEqual([graph["graph"] for graph in result["graphs"]], ["end-closure"])
        self.assertEqual(len(result["graphs"][0]["episodes"][0]["events"]), 31)


if __name__ == "__main__":
    unittest.main()
