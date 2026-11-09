"""Independent QA of the pinned upstream TEM clone and its data contracts.

These checks are static and require neither torch nor a checkpoint. They pin the
upstream snapshot so that later "trained TEM" claims can be tied to an exact
source revision, and they independently re-derive the 4x4 action semantics from
``envs/4x4.json`` rather than trusting prose in the task files.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "research" / "reference" / "torch_tem"
MODEL = UPSTREAM / "model.py"
WORLD = UPSTREAM / "world.py"
PARAMETERS = UPSTREAM / "parameters.py"
RUN = UPSTREAM / "run.py"
ENV_4X4 = UPSTREAM / "envs" / "4x4.json"

UPSTREAM_LOCK = json.loads((ROOT / "docker" / "upstream.lock.json").read_text(encoding="utf-8"))

ACTION_NAMES = {0: "stay", 1: "north", 2: "east", 3: "south", 4: "west"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo: Path) -> str:
    git_dir = repo / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[5:].strip()
        return (git_dir / ref).read_text(encoding="utf-8").strip()
    return head


def load_4x4() -> dict:
    return json.loads(ENV_4X4.read_text(encoding="utf-8"))


def find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found")


def literal_params_assignments() -> dict[str, object]:
    """Read literal ``params['key'] = value`` assignments from parameters.py."""
    tree = ast.parse(PARAMETERS.read_text(encoding="utf-8"))
    found: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "params"
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
        ):
            continue
        try:
            found[target.slice.value] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return found


@unittest.skipUnless(UPSTREAM.is_dir(), "pinned upstream clone not present")
class PinnedSnapshotTest(unittest.TestCase):
    def test_checkout_is_the_reviewed_commit(self):
        self.assertEqual(git_head(UPSTREAM), UPSTREAM_LOCK["commit"])

    def test_reviewed_files_are_unmodified(self):
        for relative, expected in UPSTREAM_LOCK["files"].items():
            with self.subTest(file=relative):
                path = UPSTREAM / relative
                self.assertTrue(path.is_file(), f"missing {relative}")
                self.assertEqual(
                    sha256(path),
                    expected,
                    f"{relative} differs from the reviewed snapshot; upstream "
                    "source must stay unmodified, or the QA record must be updated",
                )

    def test_no_pretrained_checkpoint_is_shipped(self):
        suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".pkl"}
        found = [
            str(path.relative_to(UPSTREAM))
            for path in UPSTREAM.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ]
        self.assertEqual(found, [])

    def test_clone_is_excluded_from_project_history(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", "research/reference/torch_tem/model.py"], cwd=ROOT
        )
        self.assertEqual(result.returncode, 0)


@unittest.skipUnless(ENV_4X4.is_file(), "pinned upstream clone not present")
class EnvironmentContractTest(unittest.TestCase):
    def test_4x4_counts_are_16_nodes_45_observations_5_actions(self):
        env = load_4x4()
        self.assertEqual(env["n_locations"], 16)
        self.assertEqual(env["n_observations"], 45)
        self.assertEqual(env["n_actions"], 5)
        self.assertEqual(len(env["locations"]), 16)
        self.assertEqual(len(env["adjacency"]), 16)

    def test_each_node_has_one_distinct_observation_in_the_45_class_vocabulary(self):
        env = load_4x4()
        observations = [location["observation"] for location in env["locations"]]
        self.assertEqual(len(set(observations)), 16)
        self.assertTrue(all(0 <= value < env["n_observations"] for value in observations))

    def test_adjacency_is_symmetric(self):
        env = load_4x4()
        matrix = env["adjacency"]
        self.assertTrue(
            all(matrix[row][col] == matrix[col][row] for row in range(16) for col in range(16))
        )

    def test_action_ids_are_stay_north_east_south_west(self):
        """Derive each action's direction from node coordinates, not from prose.

        Node 0 is (0.125, 0.125), so larger y is south and larger x is east.
        Every non-self action must move exactly one grid cell in its direction;
        out-of-bounds actions must have an all-zero transition vector.
        """
        env = load_4x4()
        for location in env["locations"]:
            node = location["id"]
            x, y = location["x"], location["y"]
            for action in location["actions"]:
                action_id = action["id"]
                targets = [i for i, value in enumerate(action["transition"]) if value]
                with self.subTest(node=node, action=action_id):
                    if action_id == 0:
                        self.assertEqual(targets, [node], "action 0 must be stay")
                        continue
                    if not targets:
                        self._assert_out_of_bounds(env, location, action_id)
                        continue
                    self.assertEqual(len(targets), 1, "movement action must have one target")
                    target = env["locations"][targets[0]]
                    dx = round(target["x"] - x, 6)
                    dy = round(target["y"] - y, 6)
                    expected = {
                        1: (0.0, -0.25),
                        2: (0.25, 0.0),
                        3: (0.0, 0.25),
                        4: (-0.25, 0.0),
                    }[action_id]
                    self.assertEqual((dx, dy), expected, ACTION_NAMES[action_id])

    @staticmethod
    def _assert_out_of_bounds(env: dict, location: dict, action_id: int) -> None:
        row = round((location["y"] - 0.125) / 0.25)
        col = round((location["x"] - 0.125) / 0.25)
        expected_edges = {
            1: row == 0,
            2: col == env["n_locations"] ** 0.5 - 1,
            3: row == env["n_locations"] ** 0.5 - 1,
            4: col == 0,
        }
        assert action_id in expected_edges, f"unexpected action id {action_id}"
        assert expected_edges[action_id], (
            f"node {location['id']} action {action_id} has no target but is not "
            f"on the expected boundary"
        )


@unittest.skipUnless(MODEL.is_file(), "pinned upstream clone not present")
class ModelContractTest(unittest.TestCase):
    def test_forward_takes_a_walk_not_observation_action_tensors(self):
        tree = ast.parse(MODEL.read_text(encoding="utf-8"))
        forward = next(
            item
            for item in find_class(tree, "Model").body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        )
        args = [arg.arg for arg in forward.args.args]
        self.assertEqual(args[:4], ["self", "walk", "prev_iter", "prev_M"])
        defaults = forward.args.defaults
        self.assertEqual([ast.literal_eval(value) for value in defaults], [None, None])

    def test_model_has_no_generic_observation_action_forward(self):
        source = MODEL.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Model":
                forwards = [
                    item for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name == "forward"
                ]
                self.assertEqual(len(forwards), 1)
        self.assertNotRegex(source, r"def\s+forward\s*\(\s*self\s*,\s*obs")

    def test_default_sensor_vocabulary_and_action_count_match_the_4x4_problem(self):
        params = literal_params_assignments()
        self.assertEqual(params["n_x"], 45)
        self.assertEqual(params["n_actions"], 4, "four movement actions, plus static")
        self.assertIs(params["has_static_action"], True)
        self.assertEqual(params["n_actions"] + 1, 5, "total action ids 0..4")


if __name__ == "__main__":
    unittest.main()
