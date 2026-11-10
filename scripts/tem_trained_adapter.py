#!/usr/bin/env python3
"""Reusable adapter for an unmodified torch_tem checkout.

This module is deliberately separate from the UI/exporter owned by other
agents. It knows the real upstream interface:

    Model(params).forward(walk, prev_iter=None, prev_M=None)
    walk = [[location_dict, one_hot_observation, action_id], ...]

The 4x4 environment uses five action ids: stay=0, north=1, east=2,
south=3, west=4. ``Model`` internally uses four one-hot transition inputs
because ``has_static_action=True`` encodes stay as a zero vector.

The adapter never edits upstream files. It also never invents model output:
every prediction returned here comes from a real ``Model.forward`` call.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import architectural_grid
except ModuleNotFoundError as error:
    if error.name != "architectural_grid":
        raise
    architectural_grid = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "research" / "reference" / "torch_tem"
DEFAULT_ENV = DEFAULT_REPO / "envs" / "4x4.json"
DEFAULT_WORLD_TS = ROOT / "src" / "sim" / "world.ts"

SCHEMA_EVALUATION = "tem.evaluation.v1"
GRID_SIZE = 4
N_NODES = GRID_SIZE * GRID_SIZE
N_OBSERVATIONS = 45
N_ACTIONS = 5

ACTION_LABELS = {0: "stay", 1: "N", 2: "E", 3: "S", 4: "W"}
ACTION_IDS = {label: action for action, label in ACTION_LABELS.items()}

# Stable sensor ids for the app's two held-out worlds. Ids are arbitrary
# identifiers, not meanings. A occupies 0..15, B occupies 16..31; training
# draws unique assignments from the full 45-token vocabulary and never uses
# either held-out tuple.
ITEM_LABELS_A: tuple[str, ...] = (
    "Window",
    "Sculpture",
    "Bench",
    "Door",
    "Potted Plant",
    "Column",
    "Arch",
    "Stair",
    "Lamp",
    "Fountain",
    "Tapestry",
    "Mirror",
    "Basin",
    "Shelf",
    "Rug",
    "Clock",
)
ITEM_LABELS_B: tuple[str, ...] = (
    "Reception Desk",
    "Patient Bed",
    "IV Stand",
    "Wheelchair",
    "ECG Monitor",
    "Exam Light",
    "Crash Cart",
    "Nurse Station",
    "X-Ray Viewer",
    "Medicine Trolley",
    "Surgical Sink",
    "Oxygen Tank",
    "Infusion Pump",
    "Privacy Curtain",
    "Sterile Cabinet",
    "Call Button",
)
LABEL_SENSOR_IDS: dict[str, int] = {
    label: index for index, label in enumerate(ITEM_LABELS_A + ITEM_LABELS_B)
}
ASSIGNMENT_SEEDS = {"A": 0x5EED_A11C, "B": 0x5EED_B0B0}


class AdapterError(RuntimeError):
    """Precondition or compatibility failure; CLI maps this to a non-zero exit."""


# --------------------------------------------------------------------- helpers


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AdapterError(
            "torch is not installed; install the training dependencies first "
            "(python -m pip install torch scipy numpy)"
        ) from exc
    return torch


def require_numpy():
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AdapterError(
            "numpy is not installed; install the training dependencies first "
            "(python -m pip install torch scipy numpy)"
        ) from exc
    return numpy


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def scipy_available() -> bool:
    try:
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


def training_dependencies_available() -> bool:
    return torch_available() and numpy_available() and scipy_available()


def read_upstream_commit(repo: Path) -> dict:
    """Record the upstream revision without depending on it for correctness."""
    info = {"commit": None, "source": None, "dirty": None, "error": None}
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()
            info["source"] = "git rev-parse"
        else:
            info["error"] = result.stderr.strip() or "git rev-parse failed"
    except (OSError, subprocess.SubprocessError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    if info["commit"] is None:
        try:
            head = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                ref_path = repo / ".git" / head[5:]
                info["commit"] = ref_path.read_text(encoding="utf-8").strip()
            else:
                info["commit"] = head
            info["source"] = "read .git/HEAD"
        except OSError as exc:
            info["error"] = (info["error"] or "") + f"; fallback: {exc}"

    try:
        status = subprocess.run(
            # Ignore untracked bytecode caches: importing upstream creates
            # __pycache__/, which must not make tracked source look modified.
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


# ------------------------------------------------------------------ world data


def _read_environment(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise AdapterError(f"environment not found at {path}")
    try:
        environment = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read environment {path}: {exc}") from exc
    if not isinstance(environment, dict):
        raise AdapterError("environment JSON must contain an object")
    return environment


def load_4x4(path: Path = DEFAULT_ENV) -> dict:
    environment = _read_environment(path)
    validate_4x4(environment)
    return environment


def load_environment(path: Path = DEFAULT_ENV) -> dict:
    environment = _read_environment(path)
    validate_environment(environment)
    return environment


def validate_environment(environment: dict) -> dict:
    if "architecture" not in environment:
        return validate_4x4(environment)
    if architectural_grid is None:
        raise AdapterError("architecture-specific layouts require the local study extension")
    metadata = environment["architecture"]
    if not isinstance(metadata, dict) or metadata.get("size") not in (4, 5, 6):
        raise AdapterError("expected a canonical House 3 grid at size 4, 5 or 6")
    if environment != architectural_grid.build_environment(metadata["size"]):
        raise AdapterError("environment differs from the canonical House 3 embedding")
    return {
        "n_locations": environment["n_locations"],
        "n_observations": environment["n_observations"],
        "n_actions": environment["n_actions"],
        "action_ids": {str(key): value for key, value in ACTION_LABELS.items()},
        "architecture": metadata,
    }


def expected_target(node: int, action: int) -> int | None:
    """Ground-truth target for one 4x4 action, or None if the action is blocked."""
    if not 0 <= node < N_NODES:
        raise AdapterError(f"node {node} is outside the 4x4 grid")
    if action not in ACTION_LABELS:
        raise AdapterError(f"action {action} is outside the five-action 4x4 space")
    row, col = divmod(node, GRID_SIZE)
    if action == 0:
        return node
    if action == 1:
        return node - GRID_SIZE if row > 0 else None
    if action == 2:
        return node + 1 if col < GRID_SIZE - 1 else None
    if action == 3:
        return node + GRID_SIZE if row < GRID_SIZE - 1 else None
    return node - 1 if col > 0 else None


def validate_4x4(environment: dict) -> dict:
    """Check the exact action encoding of envs/4x4.json, not just its size."""
    for key, expected in (
        ("n_locations", N_NODES),
        ("n_observations", N_OBSERVATIONS),
        ("n_actions", N_ACTIONS),
    ):
        if environment.get(key) != expected:
            raise AdapterError(
                f"4x4 environment {key}={environment.get(key)!r}, expected {expected}"
            )

    locations = environment.get("locations")
    if not isinstance(locations, list) or len(locations) != N_NODES:
        raise AdapterError("4x4 environment must contain 16 locations")

    adjacency = environment.get("adjacency")
    if not isinstance(adjacency, list) or len(adjacency) != N_NODES:
        raise AdapterError("4x4 environment must contain a 16x16 adjacency matrix")

    degrees = []
    for node, location in enumerate(locations):
        if location.get("id") != node:
            raise AdapterError(f"location {node} has id {location.get('id')!r}")
        actions = location.get("actions")
        if not isinstance(actions, list) or len(actions) != N_ACTIONS:
            raise AdapterError(f"location {node} must define all five actions")
        degree = 0
        for action, entry in enumerate(actions):
            if entry.get("id") != action:
                raise AdapterError(f"location {node} action {action} has mismatched id")
            transition = entry.get("transition")
            if not isinstance(transition, list) or len(transition) != N_NODES:
                raise AdapterError(f"location {node} action {action} has bad transition")
            target = expected_target(node, action)
            if target is None:
                if any(value != 0 for value in transition):
                    raise AdapterError(
                        f"blocked action {action} at node {node} has a transition"
                    )
                continue
            nonzero = [index for index, value in enumerate(transition) if value != 0]
            if nonzero != [target] or transition[target] != 1:
                raise AdapterError(
                    f"action {action} at node {node} should transition to {target}, "
                    f"got {nonzero}"
                )
            if action != 0:
                degree += 1

        for target in range(N_NODES):
            if adjacency[node][target] != adjacency[target][node]:
                raise AdapterError(f"adjacency is not symmetric at ({node}, {target})")
        row, col = divmod(node, GRID_SIZE)
        expected_degree = (
            2 if row in (0, 3) and col in (0, 3) else 3 if row in (0, 3) or col in (0, 3) else 4
        )
        if degree != expected_degree:
            raise AdapterError(
                f"node {node} has {degree} moves, expected {expected_degree}"
            )
        degrees.append(degree)

    return {
        "n_locations": N_NODES,
        "n_observations": N_OBSERVATIONS,
        "n_actions": N_ACTIONS,
        "action_ids": {str(key): value for key, value in ACTION_LABELS.items()},
        "move_degree": degrees,
    }


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _imul(left: int, right: int) -> int:
    # Low 32 bits of the signed product; exactly Math.imul for our purposes.
    return (_u32(left) * _u32(right)) & 0xFFFFFFFF


def mulberry32(seed: int):
    """Reference port of src/sim/world.ts mulberry32 (JS 32-bit semantics)."""
    state = _u32(seed)
    while True:
        state = _u32(state + 0x6D2B79F5)
        value = state
        value = _imul(value ^ (value >> 15), value | 1)
        value = _u32(
            value ^ _u32(value + _imul(value ^ (value >> 7), value | 61))
        )
        yield _u32(value ^ (value >> 14)) / 4294967296.0


def seeded_shuffle(values: Sequence[Any], seed: int) -> list[Any]:
    shuffled = list(values)
    generator = mulberry32(seed)
    for index in range(len(shuffled) - 1, 0, -1):
        swap = int(next(generator) * (index + 1))
        shuffled[index], shuffled[swap] = shuffled[swap], shuffled[index]
    return shuffled


@dataclass(frozen=True)
class WorldAssignment:
    world: str
    node_to_label: tuple[str, ...]
    node_to_sensor: tuple[int | None, ...]

    def as_json(self) -> dict:
        return {
            "world": self.world,
            "node_to_label": list(self.node_to_label),
            "sensor_at_node": {str(node): sensor for node, sensor in enumerate(self.node_to_sensor)},
            "label_sensor_ids": {
                label: sensor for label, sensor in zip(self.node_to_label, self.node_to_sensor)
                if sensor is not None
            },
        }


def ui_world_assignment(world: str) -> WorldAssignment:
    """Reproduce createWorld(world) from src/sim/world.ts without running TS."""
    if world not in ASSIGNMENT_SEEDS:
        raise AdapterError(f"unknown world {world!r}; expected 'A' or 'B'")
    labels = ITEM_LABELS_A if world == "A" else ITEM_LABELS_B
    order = tuple(seeded_shuffle(labels, ASSIGNMENT_SEEDS[world]))
    sensors = tuple(LABEL_SENSOR_IDS[label] for label in order)
    return WorldAssignment(world=world, node_to_label=order, node_to_sensor=sensors)


def evaluation_assignment(world: str, environment: dict | None = None) -> WorldAssignment:
    if environment is None or "architecture" not in environment:
        return ui_world_assignment(world)
    sensors = architectural_grid.heldout_assignment(environment, world)
    labels = tuple("void" if token is None else f"cue-{token:02d}" for token in sensors)
    return WorldAssignment(world, labels, sensors)


def heldout_assignments(environment: dict | None = None) -> dict[str, tuple[int | None, ...]]:
    return {
        world: evaluation_assignment(world, environment).node_to_sensor for world in ("A", "B")
    }


def parse_world_ts(path: Path = DEFAULT_WORLD_TS) -> dict:
    """Parse labels and seeds from the UI source so drift is detectable in tests."""
    path = Path(path)
    if not path.is_file():
        raise AdapterError(f"UI world source not found at {path}")
    text = path.read_text(encoding="utf-8")

    def labels(name: str) -> tuple[str, ...]:
        match = re.search(
            rf"const {name}: readonly string\[\] = \[(.*?)\];", text, re.DOTALL
        )
        if not match:
            raise AdapterError(f"could not find {name} in {path}")
        return tuple(re.findall(r"'([^']*)'", match.group(1)))

    seeds = {}
    for world in ("A", "B"):
        match = re.search(rf"{world}:\s*(0x[0-9a-fA-F_]+)", text)
        if not match:
            raise AdapterError(f"could not find assignment seed {world} in {path}")
        seeds[world] = int(match.group(1).replace("_", ""), 16)

    return {
        "A": labels("ITEM_LABELS_A"),
        "B": labels("ITEM_LABELS_B"),
        "seeds": seeds,
    }


def random_training_assignment(
    seed: int, excluded: Iterable[tuple[int | None, ...]] = (), *, environment: dict | None = None
) -> tuple[int | None, ...]:
    """Assign unique cues to the active nodes of one training episode."""
    excluded_set = {tuple(item) for item in excluded}
    rng = random.Random(seed)
    for _ in range(1000):
        tokens = (
            architectural_grid.cue_assignment(environment, rng)
            if environment is not None and "architecture" in environment
            else tuple(rng.sample(range(N_OBSERVATIONS), N_NODES))
        )
        if tokens not in excluded_set:
            return tokens
    raise AdapterError("could not draw a training assignment outside the held-out set")


# -------------------------------------------------------------------- upstream


@dataclass
class Upstream:
    repo: Path
    model: Any
    world: Any
    parameters: Any
    utils: Any
    commit: dict
    file_hashes: dict


_UPSTREAM_MODULES = ("model", "world", "parameters", "utils")


def load_upstream(repo: Path = DEFAULT_REPO) -> Upstream:
    repo = Path(repo).resolve()
    required = [repo / f"{name}.py" for name in _UPSTREAM_MODULES]
    required.append(repo / "envs" / "4x4.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AdapterError(
            "upstream torch_tem clone is incomplete; missing " + ", ".join(missing)
        )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    modules = {}
    for name in _UPSTREAM_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report the real import error
            raise AdapterError(
                f"cannot import upstream {name}.py: {type(exc).__name__}: {exc}"
            ) from exc
        module_file = Path(getattr(module, "__file__", "")).resolve()
        if module_file.parent != repo:
            raise AdapterError(
                f"imported {name} from {module_file}, expected the clone at {repo}"
            )
        modules[name] = module

    if not hasattr(modules["model"], "Model"):
        raise AdapterError("upstream model.py does not expose class Model")

    file_hashes = {
        path.name: sha256_file(path) for path in required if path.is_file()
    }
    return Upstream(
        repo=repo,
        model=modules["model"],
        world=modules["world"],
        parameters=modules["parameters"],
        utils=modules["utils"],
        commit=read_upstream_commit(repo),
        file_hashes=file_hashes,
    )


def model_interface(upstream: Upstream) -> dict:
    forward = inspect.signature(upstream.model.Model.forward)
    init = inspect.signature(upstream.model.Model.__init__)
    return {
        "model_class": "Model",
        "constructor": str(init),
        "forward": str(forward),
        "forward_parameters": [
            name for name in forward.parameters if name != "self"
        ],
    }


def default_params(
    upstream: Upstream,
    *,
    iterations: int,
    overrides: dict | None = None,
) -> dict:
    params = upstream.parameters.parameters()
    if params.get("n_x") != N_OBSERVATIONS:
        raise AdapterError(
            f"upstream n_x={params.get('n_x')!r}; this adapter expects {N_OBSERVATIONS}"
        )
    if params.get("n_actions") != 4 or not params.get("has_static_action"):
        raise AdapterError(
            "upstream parameter set changed: expected n_actions=4 and "
            "has_static_action=True for the five-id 4x4 action space"
        )
    params["train_it"] = int(iterations)
    if overrides:
        params.update(overrides)
    return params


def apply_pilot_schedule(params: dict, iterations: int) -> dict:
    """Scale upstream's iteration-based ramps to a small capped pilot.

    These are not upstream's published long-run settings. They exist so a
    short mechanical proof-of-life run can exercise every loss term and the
    Hebbian write ramp. Results must not be described as convergence.
    """
    iterations = max(1, int(iterations))
    applied = {
        "eta_it": max(1, iterations // 2),
        "lambda_it": max(1, iterations // 10),
        "loss_weights_p_g_it": max(1, iterations // 3),
        "p2g_sig_half_it": max(1, iterations // 5),
        "p2g_sig_scale_it": max(1, iterations // 10),
        "loss_weights_reg_p_it": max(1, iterations // 2),
        "lr_decay_steps": max(1, iterations),
    }
    params.update(applied)
    return applied


def resolve_device(requested: str, torch) -> dict:
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda":
        if not cuda_available:
            raise AdapterError("CUDA is unavailable in this Python environment. Use --device cpu.")
        return {
            "device": "cuda",
            "cuda_available": True,
            "reason": "explicit CUDA training with device-scoped upstream forward",
            "gpu": torch.cuda.get_device_name(),
        }
    if requested not in ("cpu", "auto"):
        raise AdapterError(f"unknown device {requested!r}; expected auto, cpu, or cuda")
    return {
        "device": "cpu",
        "cuda_available": cuda_available,
        "reason": (
            "CUDA is available. CPU is the default; opt in with --device cuda."
            if cuda_available
            else "CUDA not available in this Python environment; CPU selected"
        ),
    }


# ---------------------------------------------------------------- model inputs


def prepare_cuda_model(model, torch) -> None:
    """Move registered parameters and upstream's plain tensor hyperparameters."""
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to("cuda")
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    model.to("cuda")
    model.hyper = move(model.hyper)
    table = torch.stack(model.hyper["two_hot_table"])

    def compress(observations):
        # Same two-hot lookup as upstream, without a GPU-to-host index per item.
        return table[observations.argmax(dim=1)]

    model.f_c = compress


def make_training_world(
    upstream: Upstream,
    environment: dict,
    assignment: tuple[int | None, ...],
) -> Any:
    n = environment["n_locations"]
    active = set(environment.get("architecture", {}).get("active_nodes", range(n)))
    if len(assignment) != n:
        raise AdapterError(f"training assignment must contain {n} slots")
    tokens = {token for token in assignment if token is not None}
    if (
        len(tokens) != len(active)
        or any((token is not None) != (node in active) for node, token in enumerate(assignment))
    ):
        raise AdapterError("training assignment must have one unique cue per active node")
    if any(not 0 <= token < N_OBSERVATIONS for token in tokens):
        raise AdapterError("training assignment token outside the 45-token vocabulary")
    world = upstream.world.World(copy.deepcopy(environment), randomise_observations=False)
    for node, location in enumerate(world.locations):
        if location.get("id") != node:
            raise AdapterError(f"upstream world location {node} has id {location.get('id')!r}")
        location["observation"] = assignment[node]
    world.policy_random()
    return world


def build_model_walk(
    nodes: Sequence[int],
    sensors: Sequence[int],
    actions: Sequence[int],
    *,
    torch=None,
    n_nodes: int = N_NODES,
) -> list:
    """Build the exact batched walk upstream ``Model.forward`` consumes.

    Each step is ``[[location_dict], observation, [action_id]]``. Upstream
    reads ``x.shape[0]`` for the batch size and calls ``len(a)`` for batch
    actions, so the evaluation walk is emitted with a batch size of one:
    observations have shape ``(1, n_x)``, locations are a one-item list, and
    actions are a one-item list.

    ``actions[t]`` is the action selected at node ``t``; the model uses
    ``actions[t-1]`` to predict observation ``t``, which is the strict-prefix
    contract recorded in docs/TRAINING.md.
    """
    if not (len(nodes) == len(sensors) == len(actions)):
        raise AdapterError("nodes, sensors, and actions must have equal length")
    torch = require_torch() if torch is None else torch
    walk = []
    for node, sensor, action in zip(nodes, sensors, actions):
        if not 0 <= node < n_nodes:
            raise AdapterError(f"node {node} outside the configured grid")
        if sensor is None or not 0 <= sensor < N_OBSERVATIONS:
            raise AdapterError(f"sensor {sensor} outside 45-token vocabulary")
        if action not in ACTION_LABELS:
            raise AdapterError(f"action {action} outside the five-action space")
        observation = torch.zeros((1, N_OBSERVATIONS), dtype=torch.float32)
        observation[0, sensor] = 1.0
        walk.append([[{"id": int(node), "shiny": None}], observation, [int(action)]])
    return walk


def generate_evaluation_walk(
    seed: int, steps: int, allow_stay: bool = False, *, environment: dict | None = None
) -> dict:
    """Seeded evaluation scaffold; it is not TEM planning and never pretends to be."""
    if steps < 1:
        raise AdapterError("evaluation walk needs at least one step")
    rng = random.Random(seed)
    if environment is not None and "architecture" in environment:
        node = environment["architecture"]["root"]

        def transition(node, action):
            return architectural_grid.target(environment, node, action)
    else:
        node = 0
        transition = expected_target

    nodes = [node]
    actions: list[int] = []
    for _ in range(steps - 1):
        options = [
            (action, transition(node, action))
            for action in ((1, 2, 3, 4, 0) if allow_stay else (1, 2, 3, 4))
        ]
        action, node = rng.choice([(action, target) for action, target in options if target is not None])
        actions.append(action)
        nodes.append(node)
    actions.append(0)  # action at the final node; unused by the final prediction
    return {
        "controller": "seeded evaluation random walk (not TEM planning)",
        "seed": seed,
        "nodes": nodes,
        "actions": actions,
        "action_labels": [ACTION_LABELS[action] for action in actions],
    }


# ------------------------------------------------------------------- training


class TrainingSession:
    """Batched walk source matching upstream run.py's environment replacement."""

    def __init__(
        self,
        upstream: Upstream,
        environment: dict,
        *,
        batch_size: int,
        rollout: int,
        seed: int,
        walk_length: int,
        excluded_assignments: Iterable[tuple[int | None, ...]] = (),
    ) -> None:
        if batch_size < 1:
            raise AdapterError("batch_size must be positive")
        if rollout < 1 or walk_length < rollout:
            raise AdapterError("walk_length must be at least rollout")
        self.upstream = upstream
        self.environment = environment
        self.batch_size = int(batch_size)
        self.rollout = int(rollout)
        self.walk_length = int(walk_length)
        self.excluded = {tuple(item) for item in excluded_assignments}
        if "architecture" in environment:
            self.excluded.update(architectural_grid.reserved_assignments(environment))
        self.rng = random.Random(seed)
        self.numpy = require_numpy()
        self.torch = require_torch()
        self.environments: list[Any] = []
        self.walks: list[list] = []
        self.visited: list[list[bool]] = []
        self.assignments: list[tuple[int | None, ...]] = []
        self.prev_iter = None
        self.regen_count = 0
        for index in range(self.batch_size):
            self._new_environment(index)

    def _new_environment(self, index: int) -> dict:
        assignment = random_training_assignment(
            self.rng.randrange(0, 2**31 - 1), self.excluded, environment=self.environment
        )
        world = make_training_world(self.upstream, self.environment, assignment)
        if index < len(self.environments):
            self.environments[index] = world
            self.visited[index] = [False] * self.environment["n_locations"]
            self.assignments[index] = assignment
        else:
            self.environments.append(world)
            self.visited.append([False] * self.environment["n_locations"])
            self.assignments.append(assignment)
        self.walks.insert(index, self._walk(world))
        if index < len(self.walks) - 1:
            self.walks.pop(index + 1)
        self.regen_count += 1
        if self.prev_iter is not None:
            # Upstream run.py marks a new walk with a None previous action so
            # init_walks resets memory, g, and x for this batch entry.
            self.prev_iter[0].a[index] = None
        return {
            "batch_index": index,
            "assignment": list(assignment),
            "start_node": self.walks[index][0][0]["id"],
            "reason": "walk exhausted; new environment sampled",
        }

    def _walk(self, world: Any) -> list:
        # World.generate_walks uses numpy's global RNG; seed it per walk so a
        # session seed reproduces the whole training stream.
        self.numpy.random.seed(self.rng.randrange(0, 2**31 - 1))
        if "architecture" not in self.environment:
            return world.generate_walks(self.walk_length, 1)[0]
        exterior = world.locations[self.environment["architecture"]["root"]]
        first = [exterior, world.get_observation(exterior), world.get_action(exterior, [])]
        return [
            [{"id": location["id"], "shiny": location["shiny"]}, observation, action]
            for location, observation, action in world.walk_default([first], self.walk_length)
        ]

    def next_chunk(self) -> tuple[list, list[dict]]:
        events = []
        for index, walk in enumerate(self.walks):
            if len(walk) < self.rollout:
                events.append(self._new_environment(index))

        chunk: list = []
        for env_index in range(self.batch_size):
            for step_index in range(self.rollout):
                location, observation, action = self.walks[env_index].pop(0)
                if step_index == len(chunk):
                    chunk.append([[location], [observation], [action]])
                else:
                    chunk[step_index][0].append(location)
                    chunk[step_index][1].append(observation)
                    chunk[step_index][2].append(action)
        for step in chunk:
            step[1] = self.torch.stack(step[1], dim=0)
        return chunk, events


def compute_loss(
    forward: list, visited: list[list[bool]], loss_weights, *, node_ids=None
) -> dict:
    """Upstream run.py's visited-location loss mask, returned as plain data."""
    torch = require_torch()
    total = loss_weights.new_zeros(())
    component_rows = []
    scored = 0
    for index, step in enumerate(forward):
        step_losses = []
        for env_index, env_visited in enumerate(visited):
            node_id = step.g[env_index]["id"] if node_ids is None else node_ids[index][env_index]
            if env_visited[node_id]:
                step_losses.append(
                    loss_weights * torch.stack([loss[env_index] for loss in step.L])
                )
            else:
                env_visited[node_id] = True
        if step_losses:
            step_loss = torch.mean(torch.stack(step_losses, dim=0), dim=0)
            component_rows.append(step_loss.detach())
            total = total + torch.sum(step_loss)
            scored += len(step_losses)
    if component_rows:
        rows = torch.stack(component_rows).cpu().tolist()
        components = [sum(column) for column in zip(*rows)]
    else:
        components = [0.0] * 8
    return {
        "loss": total,
        "components": components,
        "scored_predictions": scored,
        "requires_grad": bool(total.requires_grad),
    }


def forward_accuracies(forward: list) -> dict:
    if forward[0].x.is_cuda:
        torch = require_torch()
        observed = torch.stack([step.x.detach().argmax(dim=-1) for step in forward])
        predicted = torch.stack([
            torch.stack([head.detach().argmax(dim=-1) for head in step.x_gen])
            for step in forward
        ])
        counts = (predicted == observed[:, None]).sum(dim=(0, 2)).cpu().tolist()
        values = [count * 100.0 / observed.numel() for count in counts]
    else:
        numpy = require_numpy()
        values = numpy.mean(
            [[numpy.mean(accuracy) for accuracy in step.correct()] for step in forward],
            axis=0,
        )
        values = [float(value) * 100.0 for value in values]
    return {"p": values[0], "g": values[1], "gt": values[2]}


# ----------------------------------------------------------------- evaluation


def _softmax(logits: Sequence[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _summarise_logits(logits: Sequence[float]) -> dict:
    probabilities = _softmax(logits)
    top = max(range(len(probabilities)), key=probabilities.__getitem__)
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return {
        "logits": [round(float(value), 6) for value in logits],
        "probs": [round(float(value), 6) for value in probabilities],
        "top1_sensor": top,
        "top1_prob": round(float(probabilities[top]), 6),
        "entropy": round(float(entropy), 6),
    }


def _activation_list(value: Sequence[Any]) -> list[list[float]]:
    return [
        [round(float(item), 6) for item in tensor.detach().cpu().reshape(-1).tolist()]
        for tensor in value
    ]


def predictions_from_forward(
    forward: list,
    nodes: Sequence[int],
    sensors: Sequence[int],
    actions: Sequence[int],
    *,
    include_activations: bool = False,
) -> tuple[list[dict], dict]:
    """Extract strict-prefix predictions from a real forward pass.

    ``x_logits[2]`` is the generative path ``g_gen -> gen_p -> gen_x``. It is
    the only one of the three observation heads that does not consume the
    current observation. Step 0 is therefore reported as unavailable.
    """
    if not (len(forward) == len(nodes) == len(sensors) == len(actions)):
        raise AdapterError("forward result and walk arrays must have equal length")
    seen: set[int] = set()
    steps = []
    overall_correct = 0
    overall_scored = 0
    first_correct = 0
    first_scored = 0
    repeat_correct = 0
    repeat_scored = 0

    for index, iteration in enumerate(forward):
        node = int(nodes[index])
        observed = int(sensors[index])
        first_visit = node not in seen
        seen.add(node)
        if index == 0:
            steps.append(
                {
                    "t": index,
                    "node": node,
                    "action_from_previous": None,
                    "action_from_previous_label": None,
                    "prediction": None,
                    "prediction_unavailable_reason": (
                        "no prior observation or action; step 0 has no strict prefix"
                    ),
                    "observation": {"sensor_id": observed},
                    "first_visit": first_visit,
                    "correct_top1": None,
                }
            )
            continue

        logits = iteration.x_logits[2][0].detach().cpu().tolist()
        prediction = _summarise_logits(logits)
        correct = prediction["top1_sensor"] == observed
        prediction["activations"] = (
            {
                "g_inf": _activation_list(iteration.g_inf),
                "p_inf": _activation_list(iteration.p_inf),
                "note": "raw model state after this step's real inference",
            }
            if include_activations
            else None
        )
        if not include_activations:
            prediction["activations_reason"] = "not requested; no placeholder state"
        steps.append(
            {
                "t": index,
                "node": node,
                "action_from_previous": ACTION_LABELS[int(actions[index - 1])],
                "action_from_previous_id": int(actions[index - 1]),
                "prediction": prediction,
                "prediction_unavailable_reason": None,
                "observation": {"sensor_id": observed},
                "first_visit": first_visit,
                "correct_top1": bool(correct),
            }
        )
        overall_scored += 1
        overall_correct += int(correct)
        if first_visit:
            first_scored += 1
            first_correct += int(correct)
        else:
            repeat_scored += 1
            repeat_correct += int(correct)

    summary = {
        "prediction_tensor": "x_logits[2] (g_gen -> gen_p -> gen_x)",
        "predictions_scored": overall_scored,
        "top1_accuracy": overall_correct / overall_scored if overall_scored else None,
        "first_visit_accuracy": first_correct / first_scored if first_scored else None,
        "repeat_visit_accuracy": repeat_correct / repeat_scored if repeat_scored else None,
        "first_visit_count": first_scored,
        "repeat_visit_count": repeat_scored,
        "chance_accuracy": round(1.0 / N_OBSERVATIONS, 6),
        "not_scored_note": "step 0 is excluded; it has no strict prefix",
    }
    return steps, summary


def evaluate_model(
    model: Any,
    world: str,
    assignment: WorldAssignment,
    *,
    seed: int,
    steps: int,
    include_activations: bool = False,
    environment: dict | None = None,
) -> dict:
    torch = require_torch()
    walk_info = generate_evaluation_walk(seed, steps, environment=environment)
    nodes = walk_info["nodes"]
    actions = walk_info["actions"]
    sensors = [assignment.node_to_sensor[node] for node in nodes]
    model.eval()
    n_nodes = environment["n_locations"] if environment is not None else N_NODES
    architecture = environment.get("architecture") if environment is not None else None
    walk = build_model_walk(nodes, sensors, actions, torch=torch, n_nodes=n_nodes)
    with torch.no_grad():
        forward = model(walk, None)
    step_records, summary = predictions_from_forward(
        forward,
        nodes,
        sensors,
        actions,
        include_activations=include_activations,
    )
    return {
        "schema": SCHEMA_EVALUATION,
        "mode": "trained_tem",
        "generated_utc": utc_now(),
        "world": world,
        "environment": {
            "name": f"house3-{architecture['size']}x{architecture['size']}" if architecture else "4x4",
            "num_nodes": n_nodes,
            **({"architecture": architecture} if architecture is not None else {}),
            "n_observations": N_OBSERVATIONS,
            "action_ids": {str(key): value for key, value in ACTION_LABELS.items()},
            "node_to_label": list(assignment.node_to_label),
            "sensor_at_node": {
                str(node): sensor for node, sensor in enumerate(assignment.node_to_sensor)
            },
        },
        "walk": {
            **walk_info,
            "sensor_at_node": {
                str(node): sensor for node, sensor in enumerate(assignment.node_to_sensor)
            },
        },
        "integrity": {
            "predictions_use_strict_prefix": True,
            "first_step_prediction_is_null": True,
            "walk_is_tem_planning": False,
            "activations_are_raw_model_state": True,
            "notes": (
                "Walk produced by the seeded evaluation scaffold, not by TEM. "
                "TEM predicts sensory items; it does not choose actions."
            ),
        },
        "summary": summary,
        "steps": step_records,
    }


def _unwrap_state_dict(state: Any) -> Any:
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                return candidate
    return state


def _numpy_load_globals() -> list[Any]:
    """Minimal NumPy allowlist for upstream parameter dicts.

    ``parameters.py`` leaves NumPy float64 values in the hyperparameter dict
    (for example ``p2g_scale_offset``), which appear in ``params.pt`` as
    ``numpy._core.multiarray.scalar`` + ``numpy.dtype``. These constructors are
    data-only; no arbitrary callables are allowed.
    """
    try:
        import numpy
    except ImportError:
        return []
    allowed: list[Any] = [numpy.dtype, type(numpy.dtype("float64"))]
    for module_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        scalar = getattr(module, "scalar", None)
        if scalar is not None:
            allowed.append(scalar)
        break
    return allowed


def _load_torch_file(torch, path: Path) -> Any:
    """Load a project-generated tensor container without arbitrary pickle.

    The loader uses ``weights_only=True`` with a narrow NumPy allowlist for
    float64 scalars in upstream parameter dictionaries. Use checkpoints from
    a trusted source.
    """
    try:
        allowed = _numpy_load_globals()
        safe_globals = getattr(getattr(torch, "serialization", None), "safe_globals", None)
        if safe_globals is not None and allowed:
            with safe_globals(allowed):
                return torch.load(str(path), map_location="cpu", weights_only=True)
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(
            f"cannot safely load {path} with weights_only=True: "
            f"{type(exc).__name__}: {exc}. Project-generated checkpoints use "
            "tensors, NumPy scalars, and primitives only; do not load untrusted "
            "pickle files."
        ) from exc


def load_model_checkpoint(
    upstream: Upstream,
    checkpoint: Path,
    params_path: Path,
    *,
    seed: int = 0,
) -> tuple[Any, dict, dict]:
    checkpoint = Path(checkpoint)
    params_path = Path(params_path)
    if not checkpoint.is_file():
        raise AdapterError(f"checkpoint not found at {checkpoint}")
    if not params_path.is_file():
        raise AdapterError(f"parameter file not found at {params_path}")
    torch = require_torch()
    numpy = require_numpy()

    params = _load_torch_file(torch, params_path)
    if not isinstance(params, dict):
        raise AdapterError("parameter file does not contain a dict")
    if params.get("n_x") != N_OBSERVATIONS:
        raise AdapterError(
            f"checkpoint n_x={params.get('n_x')!r}, expected {N_OBSERVATIONS}"
        )

    numpy.random.seed(seed)
    torch.manual_seed(seed)
    model = upstream.model.Model(params)
    state = _load_torch_file(torch, checkpoint)
    state = _unwrap_state_dict(state)
    try:
        load_result = model.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"checkpoint does not match upstream Model: {exc}") from exc
    model.eval()
    return model, params, {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "params_path": str(params_path.resolve()),
        "params_sha256": sha256_file(params_path),
        "checkpoint_load_result": str(load_result),
        "torch_version": torch.__version__,
        "numpy_version": numpy.__version__,
    }


def provenance_snapshot(upstream: Upstream, load_info: dict) -> dict:
    interface = model_interface(upstream)
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:  # pragma: no cover
        scipy_version = None
    return {
        "upstream_repo_path": str(upstream.repo),
        "upstream_commit": upstream.commit.get("commit"),
        "upstream_commit_source": upstream.commit.get("source"),
        "upstream_dirty": upstream.commit.get("dirty"),
        "upstream_file_sha256": upstream.file_hashes,
        "model_class": interface["model_class"],
        "constructor": interface["constructor"],
        "forward": interface["forward"],
        "forward_parameters": interface["forward_parameters"],
        "action_ids": {str(key): value for key, value in ACTION_LABELS.items()},
        "n_observations": N_OBSERVATIONS,
        "device": "cpu",
        "python_version": sys.version.split()[0],
        "scipy_version": scipy_version,
        **load_info,
    }


def evaluate_checkpoint(
    repo: Path,
    checkpoint: Path,
    params_path: Path,
    *,
    world: str = "A",
    seed: int = 7,
    steps: int = 24,
    include_activations: bool = False,
) -> dict:
    upstream = load_upstream(repo)
    model, _params, load_info = load_model_checkpoint(
        upstream, checkpoint, params_path, seed=seed
    )
    assignment = ui_world_assignment(world)
    evaluation = evaluate_model(
        model, world, assignment, seed=seed, steps=steps,
        include_activations=include_activations,
    )
    evaluation["provenance"] = provenance_snapshot(upstream, load_info)
    return evaluation


# ------------------------------------------------------------------------ CLI


def _cmd_inspect(args: argparse.Namespace) -> int:
    upstream = load_upstream(Path(args.repo))
    environment = load_environment(Path(args.env))
    report = {
        "schema": "tem.adapter_inspect.v1",
        "generated_utc": utc_now(),
        "upstream": {
            "repo": str(upstream.repo),
            **upstream.commit,
            "file_sha256": upstream.file_hashes,
        },
        "model_interface": model_interface(upstream),
        "environment": validate_environment(environment),
        "vocabulary": {
            "size": N_OBSERVATIONS,
            "A_label_sensor_ids": {
                label: LABEL_SENSOR_IDS[label] for label in ITEM_LABELS_A
            },
            "B_label_sensor_ids": {
                label: LABEL_SENSOR_IDS[label] for label in ITEM_LABELS_B
            },
            "unused_by_ui_worlds": list(range(32, N_OBSERVATIONS)),
        },
        "ui_worlds": {
            world: evaluation_assignment(world, environment).as_json() for world in ("A", "B")
        },
    }
    if args.checkpoint:
        if not args.params:
            raise AdapterError("--params is required with --checkpoint")
        _model, _params, load_info = load_model_checkpoint(
            upstream, Path(args.checkpoint), Path(args.params)
        )
        report["checkpoint"] = load_info
    print(json.dumps(report, indent=2))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    upstream = load_upstream(Path(args.repo))
    environment = load_environment(Path(args.env))
    model, _params, load_info = load_model_checkpoint(
        upstream, Path(args.checkpoint), Path(args.params), seed=args.seed
    )
    provenance = provenance_snapshot(upstream, load_info)
    worlds = ["A", "B"] if args.world == "both" else [args.world]
    outputs = []
    for world in worlds:
        assignment = evaluation_assignment(world, environment)
        evaluation = evaluate_model(
            model,
            world,
            assignment,
            seed=args.seed,
            steps=args.steps,
            include_activations=args.activations,
            environment=environment,
        )
        evaluation["provenance"] = provenance
        out_path = Path(args.out)
        if len(worlds) > 1 or out_path.is_dir() or not out_path.suffix:
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / f"eval_world{world}.json"
        write_json_atomic(out_path, evaluation)
        outputs.append(
            {
                "world": world,
                "out": str(out_path),
                "summary": evaluation["summary"],
            }
        )
    print(json.dumps({"evaluations": outputs}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="print verified interface details")
    inspect_parser.add_argument("--repo", default=str(DEFAULT_REPO))
    inspect_parser.add_argument("--env", default=str(DEFAULT_ENV))
    inspect_parser.add_argument("--checkpoint")
    inspect_parser.add_argument("--params")
    inspect_parser.set_defaults(func=_cmd_inspect)

    evaluate_parser = subparsers.add_parser("evaluate", help="run a real checkpoint")
    evaluate_parser.add_argument("--repo", default=str(DEFAULT_REPO))
    evaluate_parser.add_argument("--env", default=str(DEFAULT_ENV))
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--params", required=True)
    evaluate_parser.add_argument("--world", choices=("A", "B", "both"), default="both")
    evaluate_parser.add_argument("--seed", type=int, default=7)
    evaluate_parser.add_argument("--steps", type=int, default=24)
    evaluate_parser.add_argument("--out", required=True)
    evaluate_parser.add_argument(
        "--activations",
        action="store_true",
        help="include raw g_inf/p_inf tensors; never synthesized",
    )
    evaluate_parser.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4 if "torch is not installed" not in str(exc) else 3


if __name__ == "__main__":
    raise SystemExit(main())
