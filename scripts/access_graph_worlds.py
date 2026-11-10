"""Compile authored access graphs to labelled, deterministic TEM transitions."""

from __future__ import annotations

import random
import tomllib
from collections import Counter, deque
from importlib import import_module
from pathlib import Path

N_OBSERVATIONS = 45
ACTION_LABELS = {
    0: "unused",
    1: "parent",
    2: "child-A",
    3: "child-B",
    4: "cross",
    5: "child-C",
}
CHILD_ACTIONS = (2, 3, 5)
SPLITS = {"train", "validation", "test"}
HOUSE3_GRAPH_ID = "house3-native"


def require_house3_grid():
    """Load the private study extension only for the House 3 comparator."""
    try:
        return import_module("architectural_grid")
    except ModuleNotFoundError as exc:
        if exc.name != "architectural_grid":
            raise
        raise ValueError("House 3 mode requires the local architectural_grid module") from exc


def _graph_identity(spec: dict) -> tuple[str, str]:
    graph_id = spec.get("id")
    split = spec.get("split")
    if (
        not isinstance(graph_id, str)
        or not graph_id.strip()
        or not isinstance(split, str)
        or split not in SPLITS
    ):
        raise ValueError("each graph needs a nonempty ID and a valid split")
    return graph_id, split


def _shortest_depths(transitions: list[dict[int, int]], root: int) -> dict[int, int]:
    depths = {root: 0}
    queue = deque(depths)
    while queue:
        source = queue.popleft()
        for target in transitions[source].values():
            if target not in depths:
                depths[target] = depths[source] + 1
                queue.append(target)
    return depths


def _compile_graph(spec: dict, *, deepest_layer: int, child_limit: int) -> dict:
    graph_id, split = _graph_identity(spec)
    children = spec.get("children")
    crossings = spec.get("crossings", [])
    if not isinstance(children, dict) or not isinstance(crossings, list):
        raise ValueError(f"{graph_id}: expected children and crossings")

    names = ["exterior"]
    label_to_id = {"exterior": 0}
    depths = [0]
    transitions: list[dict[int, int]] = [{}]
    edges = []

    # Visit the declared tree in breadth-first order. Child slots come from
    # the authored order in each list, independent of label or drawing.
    for source, name in enumerate(names):
        branches = children.get(name, [])
        if not isinstance(branches, list) or len(branches) > child_limit:
            raise ValueError(f"{graph_id}: {name} has invalid child slots")
        for action, branch in zip(CHILD_ACTIONS, branches):
            if not isinstance(branch, str) or not branch.strip() or branch in label_to_id:
                raise ValueError(f"{graph_id}: invalid or repeated tree node {branch!r}")
            target = len(names)
            names.append(branch)
            label_to_id[branch] = target
            depths.append(depths[source] + 1)
            transitions[source][action] = target
            transitions.append({1: source})
            edges.append((source, target))
    if set(children) - label_to_id.keys():
        raise ValueError(f"{graph_id}: all tree nodes must be reachable from exterior")
    if max(depths) != deepest_layer or len(names) > N_OBSERVATIONS:
        raise ValueError(f"{graph_id}: expected layers 0–{deepest_layer} and at most 45 spaces")

    for crossing in crossings:
        if (
            not isinstance(crossing, list)
            or len(crossing) != 2
            or any(not isinstance(name, str) or name not in label_to_id for name in crossing)
        ):
            raise ValueError(f"{graph_id}: invalid cross passage {crossing!r}")
        source, target = (label_to_id[name] for name in crossing)
        if (
            source == target
            or target in transitions[source].values()
            or 4 in transitions[source]
            or 4 in transitions[target]
        ):
            raise ValueError(f"{graph_id}: duplicate passage or cross slot {crossing!r}")
        transitions[source][4] = target
        transitions[target][4] = source
        edges.append(tuple(sorted((source, target))))

    shortest = _shortest_depths(transitions, 0)
    if depths != [shortest[index] for index in range(len(names))]:
        raise ValueError(f"{graph_id}: parent tree must preserve full-graph shortest depths")

    return {
        "id": graph_id,
        "split": split,
        "root": 0,
        "names": names,
        "label_to_id": label_to_id,
        "depths": depths,
        "transitions": transitions,
        "edges": sorted(edges),
        "provenance": spec.get("provenance"),
    }


def _apply_closures(template: dict, spec: dict) -> dict:
    graph_id, split = _graph_identity(spec)
    closures = spec.get("closed")
    if not isinstance(closures, list):
        raise ValueError(f"{graph_id}: expected a list of closed passages")

    labels = template["label_to_id"]
    reference_edges = set(template["edges"])
    closed = set()
    for passage in closures:
        if (
            not isinstance(passage, list)
            or len(passage) != 2
            or any(not isinstance(label, str) or label not in labels for label in passage)
        ):
            raise ValueError(f"{graph_id}: invalid closed passage {passage!r}")
        edge = tuple(sorted(labels[label] for label in passage))
        if edge not in reference_edges or edge in closed:
            raise ValueError(f"{graph_id}: unknown or repeated closed passage {passage!r}")
        closed.add(edge)

    # Remove both directions from a copy. Every surviving action keeps its
    # original slot, including child B where child A has been closed.
    transitions = [
        {action: target for action, target in moves.items()
         if tuple(sorted((source, target))) not in closed}
        for source, moves in enumerate(template["transitions"])
    ]
    depths = _shortest_depths(transitions, template["root"])
    names = template["names"]
    if len(depths) != len(names):
        raise ValueError(f"{graph_id}: closures disconnect spaces from the exterior")
    edges = sorted(reference_edges - closed)
    if split != "train" and len(edges) < len(names):
        raise ValueError(f"{graph_id}: evaluation masks need a loop for unused-passage returns")

    return {
        **template,
        "id": graph_id,
        "split": split,
        "template_id": template["id"],
        "transitions": transitions,
        "edges": edges,
        "reference_depths": template["depths"],
        "depths": [depths[index] for index in range(len(names))],
        "fully_open": not closed,
        "closed_passages": [[names[a], names[b]] for a, b in sorted(closed)],
    }


def _compile_closure_mixture(config: dict, specs: list[dict]) -> list[dict]:
    template_spec = config.get("template")
    if not isinstance(template_spec, dict):
        raise ValueError("closure-mixture mode requires a four-layer template")
    template = _compile_graph({**template_spec, "split": "train"}, deepest_layer=3, child_limit=3)
    graphs = [_apply_closures(template, spec) for spec in specs]
    masks = {tuple(map(tuple, graph["closed_passages"])) for graph in graphs}
    if len(masks) != len(graphs):
        raise ValueError("closure masks must be distinct across training, validation, and test")
    splits = Counter(graph["split"] for graph in graphs)
    if not all(splits[split] for split in SPLITS) or splits["train"] < 2:
        raise ValueError("closure-mixture mode requires open and closed training masks and both evaluation splits")
    if not any(graph["split"] == "train" and graph["fully_open"] for graph in graphs):
        raise ValueError("the fully open graph must be in the training split")
    return graphs


def load_config(path: str | Path) -> dict:
    """Read a pilot config and compile its simulator-only graph metadata."""
    with Path(path).open("rb") as source:
        config = tomllib.load(source)
    if config.get("version") != 1:
        raise ValueError("access-graph config requires version 1")
    if not isinstance(config.get("training"), dict) or not isinstance(config.get("evaluation"), dict):
        raise ValueError("access-graph config requires training and evaluation settings")
    specs = config.get("graphs")
    if not isinstance(specs, list) or not all(isinstance(spec, dict) for spec in specs):
        raise ValueError("access-graph config requires graph tables")

    mode = config.get("mode")
    if mode == "four-layer":
        graphs = [_compile_graph(spec, deepest_layer=3, child_limit=3) for spec in specs]
        splits = Counter(graph["split"] for graph in graphs)
        if splits["train"] < 2 or not splits["validation"] or not splits["test"]:
            raise ValueError("four-layer mode requires multiple train graphs and validation and test graphs")
    elif mode == "closure-mixture":
        graphs = _compile_closure_mixture(config, specs)
    elif mode == "house3-comparator":
        if len(specs) != 1:
            raise ValueError("House 3 mode requires exactly one graph")
        graph = _compile_graph(specs[0], deepest_layer=2, child_limit=3)
        source_edges = {frozenset(edge) for edge in require_house3_grid().SOURCE_EDGES}
        edges = {frozenset(graph["names"][index] for index in edge) for edge in graph["edges"]}
        if graph["id"] != HOUSE3_GRAPH_ID or graph["split"] != "train" or edges != source_edges:
            raise ValueError("House 3 mode requires the complete house3-native graph in the train split")
        graphs = [graph]
    else:
        raise ValueError("access-graph mode must be four-layer, closure-mixture, or house3-comparator")

    if len({graph["id"] for graph in graphs}) != len(graphs):
        raise ValueError("graph IDs must be unique across splits")
    return {**config, "graphs": graphs}


def walk(
    graph: dict,
    rng: random.Random,
    length: int,
    *,
    start: int | None = None,
) -> tuple[list[int], list[int]]:
    """Sample legal door choices. The final action is an unused placeholder."""
    transitions = graph["transitions"]
    node = graph["root"] if start is None else start
    if length < 1 or not 0 <= node < len(transitions):
        raise ValueError("walk needs at least one observation and a valid start")
    nodes = [node]
    actions = []
    for _ in range(length - 1):
        action, node = rng.choice(sorted(transitions[node].items()))
        nodes.append(node)
        actions.append(action)
    return nodes, actions + [0]


def cue_assignment(graph: dict, rng: random.Random) -> tuple[int, ...]:
    """Draw a fresh symbolic cue for each space, distinct within a walk."""
    return tuple(rng.sample(range(N_OBSERVATIONS), len(graph["names"])))
