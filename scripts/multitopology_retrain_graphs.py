"""Fresh 17-node graph families and a fail-closed freshness ledger.

This module generates frozen synthetic graph cohorts for a multi-topology TEM
study. It never trains, scores or opens a model. It also never emits model
inputs: cue maps are recorded as a separate contract and the final test
families are sealed behind a SHA-256 commitment by default.

Structural contract per family:
  * 17 nodes on four backbone layers (depths 0..3);
  * 16 parent/child tree edges (ports 1 parent, 2/3/5 children, at most three
    children per node, child-action budgets 12/10/6);
  * 7 reciprocal cross edges on port 4 joining distinct same-depth nodes;
  * 23 edges total, one whole-graph root shortest-depth labelling.

Freshness contract:
  * every new family is non-isomorphic, as an undirected full graph *and* as an
    unrooted tree backbone, to every family in the supplied historic inventory;
  * the historic inventory is caller-provided.  The inventory is fail-closed: a
    missing required source or an unreconstructable 17-node record raises
    ``ExclusionAuditError`` rather than silently widening the cohort;
  * sample cue assignments avoid every assignment recovered from the supplied
    manifests.

With no supplied inventory the cohort is still generated, and the sealed
ledger records ``empty_inventory: true`` so a later reader can tell a disclosed
empty history apart from one that was silently dropped.

The public manifest never contains final test graph payloads.  ``unseal_test``
is an explicit, hash-checked operation documented for use only after checkpoint
freeze and independent QA.

Capacity.  The sampler admits one family per distinct unrooted backbone, so the
backbone shape space is a hard cohort ceiling.  ``backbone_space_index``
enumerates that space exhaustively and ``topology_capacity`` subtracts the
supplied historic inventory.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import random
import sys
import tomllib
import warnings

import networkx as nx

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import access_graph_worlds as worlds  # noqa: E402  (sibling module)
import multitopology_graphs as base  # noqa: E402  (existing validator)

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "tem.multitopology.retrain.graphs.v1"
SEALED_SCHEMA = "tem.multitopology.retrain.test-sealed.v1"
EXCLUSIONS_SCHEMA = "tem.multitopology.retrain.exclusions.v1"
REPORT_SCHEMA = "tem.multitopology.retrain.prepare-report.v1"

SPLITS = ("train", "validation", "test")
CHILD = (2, 3, 5)
CUE_VOCABULARY = 45
SAMPLE_CUE_MAPS_PER_FAMILY = 2
CHILD_ACTION_BUDGETS = (12, 10, 6)
GRAPH_SEED_NAMESPACE = "multitopology-retrain-graphs-v1"

# Files a single ``prepare`` invocation writes into its own output directory.
# They are excluded from *that cohort's* historic source inventory only; every
# other file under the scanned roots remains historic evidence.  Without this,
# the manifest/exclusions/sealed/report written by prepare are re-ingested by
# the very next ``audit --recheck-history`` and the cohort can never re-verify.
COHORT_OUTPUT_NAMES = (
    "graphs-manifest.json",
    "exclusions-private.json",
    "test-sealed.json",
    "prepare-report.json",
)

GENERATED_OUTPUT_POLICY = (
    "files this prepare invocation writes into its own output directory are "
    "excluded from its own historic source inventory; no other path is excluded "
    "and every pre-existing historic file stays in scope"
)

STRUCTURE = {
    "nodes": 17,
    "tree_edges": 16,
    "cross_edges": 7,
    "total_edges": 23,
    "layers": 4,
    "max_children": 3,
    "parent_action": 1,
    "child_actions": list(CHILD),
    "cross_action": 4,
    "child_action_budgets": dict(zip(("2", "3", "5"), CHILD_ACTION_BUDGETS)),
}

CAPACITY_SCHEMA = "tem.multitopology.retrain.topology-capacity.v1"
DEFAULT_COUNTS = {"train": 96, "validation": 24, "test": 24}
REQUESTED_VARIANTS = (
    ("gate_default", {"train": 96, "validation": 24, "test": 24}),
    ("critical_full", {"train": 96, "validation": 36, "test": 36}),
)

# Keys that would place evaluator-private or model-scored material into a graph
# record.  Graph payloads are public data; anything here is a contract breach.
FORBIDDEN_GRAPH_KEYS = frozenset({
    "cue", "cues", "assignment", "assignments", "cue_map", "remap",
    "target", "targets", "target_cue", "labels", "label", "hidden",
    "hidden_edges", "hidden_path", "predictions", "prediction", "logits",
    "scores", "score", "queries", "query", "remap_seed", "legal_mask",
    "private", "adjacency_private",
})

# The public package ships no historic inventory: a caller supplies its own
# source specifications.  An empty inventory is disclosed in the ledger rather
# than assumed to mean "no prior study exists".
DEFAULT_SOURCES = ()


# ---------------------------------------------------------------------------
# Protected-source registry
# ---------------------------------------------------------------------------
#
# A caller may register paths or byte digests that the historic scan must never
# deserialize (for example a sealed final cohort or an exclusion-ledger
# derivative).  The registry is empty by default so the public package embeds no
# private path or hash.  The guard runs over every path/glob match before any
# JSON/TOML parse.
#
# Limit, stated rather than papered over: path identity and byte SHA-256 catch
# explicit paths, nested glob matches and byte-identical renamed copies.  A
# semantic derivative whose bytes differ (reformatted JSON, a subset, a
# re-encoded ledger) cannot be recognised universally before parsing.  It must
# be registered by its own path and SHA-256.  This guard does not claim
# universal derivative detection.

PROTECTED_REGISTRY_SCHEMA = "tem.multitopology.protected-source-registry.v1"
PROTECTED_ROLES = frozenset({"sealed_final", "exclusion_ledger_derivative"})
_PROTECTED_ENTRY_KEYS = frozenset({"role", "path", "sha256"})
PROTECTED_REGISTRY_LIMITATION = (
    "path identity and byte SHA-256 refuse explicit paths, nested glob matches "
    "and byte-identical renamed copies; a byte-different semantic derivative "
    "must be registered by its own path and SHA-256 and cannot be detected "
    "universally before deserialization"
)


class ExclusionAuditError(RuntimeError):
    """Historic coverage cannot be established; generation must not proceed."""


class ProtectedSourceError(ExclusionAuditError):
    """A scan candidate is a registered protected source or derivative."""


class CohortAuditError(ValueError):
    """A frozen cohort violates its structural, split or cue contract."""


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _content_digest(value, field="content_sha256"):
    return canonical_hash({key: item for key, item in value.items() if key != field})


def _seal(value, field="content_sha256"):
    value[field] = _content_digest(value, field)
    return value


def _check_digest(value, field="content_sha256", kind="payload"):
    if value.get(field) != _content_digest(value, field):
        raise CohortAuditError(f"{kind} content digest mismatch")
    return value


def _is_int(value):
    return type(value) is int


def _valid_assignment(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 17
        and all(_is_int(cue) and 0 <= cue < CUE_VOCABULARY for cue in value)
        and len(set(value)) == 17
    )


def _tree_key(tree):
    """AHU canonical string for an unrooted tree; node IDs are ignored."""
    if tree is None or not isinstance(tree, nx.Graph) or not nx.is_tree(tree):
        raise ValueError("not a tree")
    if tree.number_of_nodes() == 0:
        return "()"

    def rooted(node, parent):
        return "(" + "".join(
            sorted(rooted(neighbour, node) for neighbour in tree[node] if neighbour != parent)
        ) + ")"

    return min(rooted(center, None) for center in nx.center(tree))


def _signature(graph):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            tuple(sorted(dict(graph.degree()).values())),
            nx.weisfeiler_lehman_graph_hash(graph),
        )


# ---------------------------------------------------------------------------
# Historic record reconstruction
# ---------------------------------------------------------------------------


def _record_node_count(record):
    for key in ("names", "parents", "children"):
        value = record.get(key)
        if isinstance(value, list) and value:
            return len(value)
    edges = record.get("edges")
    if isinstance(edges, list) and edges:
        try:
            return 1 + max(int(node) for edge in edges for node in edge)
        except (TypeError, ValueError):
            return None
    return None


def _record_full_graph(record):
    """Undirected full graph of a historic record, independent of port labels."""
    n = _record_node_count(record)
    if n is None:
        raise ExclusionAuditError("historic record has no recoverable node count")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    edges = record.get("edges")
    transitions = record.get("transitions")
    edge_set = None
    if isinstance(edges, list) and edges:
        edge_set = set()
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ExclusionAuditError("malformed historic edge")
            u, v = int(edge[0]), int(edge[1])
            if not (0 <= u < n and 0 <= v < n) or u == v:
                raise ExclusionAuditError("historic edge outside node range")
            graph.add_edge(u, v)
            edge_set.add(tuple(sorted((u, v))))
    if isinstance(transitions, list) and transitions:
        transition_edges = set()
        for u, moves in enumerate(transitions):
            for v in moves.values():
                graph.add_edge(u, int(v))
                transition_edges.add(tuple(sorted((u, int(v)))))
        if edge_set is not None and edge_set != transition_edges:
            raise ExclusionAuditError("historic edge list disagrees with transitions")
        return graph
    if edge_set is not None:
        return graph
    raise ExclusionAuditError("historic 17-node record has no edges or transitions")


def _record_backbone_tree(record):
    """Parent/child (CHILD-port) backbone, or ``None`` when not a tree."""
    n = _record_node_count(record)
    transitions = record.get("transitions")
    tree = nx.Graph()
    tree.add_nodes_from(range(n))
    if isinstance(transitions, list) and transitions:
        for u, moves in enumerate(transitions):
            for action, v in moves.items():
                if int(action) in CHILD:
                    tree.add_edge(u, int(v))
    else:
        parents = record.get("parents")
        children = record.get("children")
        if not isinstance(parents, list) or not isinstance(children, list):
            return None
        for u, branches in enumerate(children):
            for v in branches:
                tree.add_edge(u, int(v))
    if tree.number_of_nodes() and not nx.is_tree(tree):
        return None
    return tree


def _normalize_record(record):
    """Keep only structure needed to re-run the exclusion audit."""
    n = _record_node_count(record)
    result = {
        "names": [str(item) for item in record.get("names", [f"historic-{i}" for i in range(n)])],
        "edges": sorted({tuple(sorted((int(u), int(v)))) for u, v in record.get("edges", [])}),
    }
    for key in ("root", "depths"):
        if key in record:
            value = record[key]
            result[key] = int(value) if key == "root" else [int(item) for item in value]
    transitions = record.get("transitions")
    if isinstance(transitions, list) and transitions:
        result["transitions"] = [
            {int(action): int(target) for action, target in moves.items()}
            for moves in transitions
        ]
    parents = record.get("parents")
    children = record.get("children")
    if isinstance(parents, list) and isinstance(children, list):
        result["parents"] = [int(item) for item in parents]
        result["children"] = [[int(item) for item in branches] for branches in children]
    return result


def _is_graph_record(value):
    if not isinstance(value, dict):
        return False
    names = value.get("names")
    transitions = value.get("transitions")
    if (
        isinstance(names, list)
        and names
        and all(isinstance(item, str) for item in names)
        and isinstance(transitions, list)
        and len(transitions) == len(names)
        and all(isinstance(moves, dict) for moves in transitions)
    ):
        return True
    return False


def _is_family_record(value):
    if not isinstance(value, dict):
        return False
    parents = value.get("parents")
    children = value.get("children")
    if not (isinstance(parents, list) and isinstance(children, list)):
        return False
    if not parents or len(parents) != len(children):
        return False
    if not all(_is_int(item) for item in parents):
        return False
    if not all(
        isinstance(branches, list) and all(_is_int(item) for item in branches)
        for branches in children
    ):
        return False
    return True


def _extract_records(value, path, output, seen=None):
    """Recursively collect graph/family payloads from arbitrary manifests."""
    if seen is None:
        seen = set()
    if isinstance(value, dict):
        if id(value) in seen:
            return
        seen.add(id(value))
        if _is_family_record(value):
            output.append({"record": value, "path": path, "kind": "family"})
            for index, variant in enumerate(value.get("variants", [])):
                if isinstance(variant, dict) and isinstance(variant.get("edges"), list):
                    combined = {
                        "names": [f"variant-{i}" for i in range(len(value["parents"]))],
                        "parents": value["parents"],
                        "children": value["children"],
                        "edges": variant["edges"],
                    }
                    if "root" in value:
                        combined["root"] = value["root"]
                    if "depths" in value:
                        combined["depths"] = value["depths"]
                    output.append({
                        "record": combined,
                        "path": path,
                        "kind": f"variant[{index}]",
                    })
            return
        if _is_graph_record(value):
            output.append({"record": value, "path": path, "kind": "graph"})
            return
        for item in value.values():
            _extract_records(item, path, output, seen)
    elif isinstance(value, list):
        if id(value) in seen:
            return
        seen.add(id(value))
        for item in value:
            _extract_records(item, path, output, seen)


def _collect_assignments(value, output, seen=None):
    """Recover explicit cue assignments from historic manifests."""
    if seen is None:
        seen = set()
    if isinstance(value, dict):
        if id(value) in seen:
            return
        seen.add(id(value))
        for key, item in value.items():
            if key in ("reserved_assignments", "assignments") and isinstance(item, list):
                for assignment in item:
                    if _valid_assignment(assignment):
                        output.add(tuple(int(cue) for cue in assignment))
            elif key == "sample_cue_maps" and isinstance(item, dict):
                for maps in item.values():
                    if isinstance(maps, list):
                        for assignment in maps:
                            if _valid_assignment(assignment):
                                output.add(tuple(int(cue) for cue in assignment))
            elif key == "queries" and isinstance(item, list):
                for query in item:
                    assignment = query.get("assignment") if isinstance(query, dict) else None
                    if _valid_assignment(assignment):
                        output.add(tuple(int(cue) for cue in assignment))
            else:
                _collect_assignments(item, output, seen)
    elif isinstance(value, list):
        if id(value) in seen:
            return
        seen.add(id(value))
        for item in value:
            _collect_assignments(item, output, seen)


def _collect_remap_assignments(value, output):
    """Derive 17-cue assignments from ``episodes[*].remap_seed`` records."""
    if isinstance(value, dict):
        families = value.get("families")
        episodes = value.get("episodes")
        if isinstance(families, list) and isinstance(episodes, list):
            seventeen = {
                family.get("id") for family in families if _record_node_count(family) == 17
            }
            for episode in episodes:
                if not isinstance(episode, dict):
                    continue
                seed = episode.get("remap_seed")
                if episode.get("family_id") in seventeen and _is_int(seed):
                    output.add(tuple(random.Random(seed).sample(range(CUE_VOCABULARY), 17)))
        for item in value.values():
            _collect_remap_assignments(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_remap_assignments(item, output)


def _native_config_assignments(config):
    """Reproduce the native evaluation remaps that earlier training excluded."""
    evaluation = config.get("evaluation") or {}
    episodes = evaluation.get("episodes")
    seed = evaluation.get("seed")
    if not _is_int(episodes) or not _is_int(seed):
        return set()
    output = set()
    for graph in config.get("graphs", []):
        if len(graph.get("names", [])) != 17:
            continue
        suite = "remap" if graph.get("split") == "train" else graph.get("split")
        for index in range(episodes):
            rng = random.Random(f"cues:{seed}:{suite}:{graph['id']}:{index}")
            assignment = worlds.cue_assignment(graph, rng)
            if _valid_assignment(assignment):
                output.add(tuple(int(cue) for cue in assignment))
    return output


# ---------------------------------------------------------------------------
# Historic inventory (fail closed)
# ---------------------------------------------------------------------------


def _source_specs(sources):
    """JSON-safe, frozen copy of the source specification list.

    Recording the *specification* (explicit paths and glob patterns) rather
    than only its current expansion is what lets a later recheck detect a newly
    added historic file as well as a changed or removed one.
    """
    specs = []
    for spec in sources:
        if not isinstance(spec, dict):
            raise ExclusionAuditError("historic source spec must be an object")
        if "kind" not in spec:
            raise ExclusionAuditError("historic source spec is missing a kind")
        entry = {}
        for key in ("path", "glob"):
            if key in spec:
                entry[key] = str(spec[key])
        entry["kind"] = str(spec["kind"])
        entry["required"] = bool(spec.get("required", True))
        if "path" not in entry and "glob" not in entry:
            raise ExclusionAuditError("historic source spec needs a path or glob")
        specs.append(entry)
    return specs


def _is_lower_hex64(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_registry_entry(entry, origin):
    """Validate one registry entry; malformed or weakened metadata is refused."""
    if not isinstance(entry, dict):
        raise ExclusionAuditError(f"protected registry {origin}: entry must be an object")
    unknown = set(entry) - _PROTECTED_ENTRY_KEYS
    if unknown:
        raise ExclusionAuditError(
            f"protected registry {origin}: unknown metadata keys {sorted(unknown)}"
        )
    role = entry.get("role")
    if role not in PROTECTED_ROLES:
        raise ExclusionAuditError(
            f"protected registry {origin}: role must be one of {sorted(PROTECTED_ROLES)}"
        )
    path = entry.get("path")
    digest = entry.get("sha256")
    if path is not None:
        if not isinstance(path, str) or not path:
            raise ExclusionAuditError(
                f"protected registry {origin}: path must be a non-empty string"
            )
        if "\\" in path or (len(path) >= 2 and path[1] == ":" and path[0].isalpha()):
            raise ExclusionAuditError(
                f"protected registry {origin}: path must be root-relative POSIX"
            )
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path:
            raise ExclusionAuditError(
                f"protected registry {origin}: path must be normalized and root-relative"
            )
    if not _is_lower_hex64(digest):
        raise ExclusionAuditError(
            f"protected registry {origin}: sha256 must be 64 lowercase hex characters"
        )
    return {"role": role, "path": path, "sha256": digest}


def protected_registry(additional=None, default=()):
    """Validated protected-source registry: optional additions.

    The registry is empty by default so this package embeds no private path or
    hash.  Additions are validated; a path may not be remapped to a different
    hash and unknown metadata is refused.
    """
    if not isinstance(default, (tuple, list)):
        raise ExclusionAuditError("protected registry default must be a sequence")
    if additional is None:
        additional = ()
    if not isinstance(additional, (tuple, list)):
        raise ExclusionAuditError("protected registry additions must be a sequence")
    entries = [_validated_registry_entry(entry, "default") for entry in default]
    entries += [_validated_registry_entry(entry, "additional") for entry in additional]
    unique, seen, by_path = [], set(), {}
    for entry in entries:
        key = (entry["role"], entry["path"], entry["sha256"])
        if key in seen:
            continue
        path = entry["path"]
        if path is not None and path in by_path and by_path[path] != entry["sha256"]:
            raise ExclusionAuditError(
                f"protected registry remaps {path!r} to a conflicting hash"
            )
        if path is not None:
            by_path[path] = entry["sha256"]
        seen.add(key)
        unique.append(entry)
    return tuple(unique)


def _display_path(path, root):
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def assert_sources_not_protected(paths, root=ROOT, registry=None, withheld=frozenset()):
    """Refuse registered protected sources before any deserialization.

    Every explicit-path/glob candidate is checked by resolved repo-relative
    identity and by SHA-256 of the file bytes.  Byte-identical renamed copies
    are therefore refused by hash, not just by name.  ``withheld`` names exact
    resolved paths that this invocation is about to write itself; only those
    are skipped, and they are skipped before parsing rather than trusted.

    Returns the bytes digest map so the caller can reuse it for provenance
    without reading twice.
    """
    if registry is None:
        registry = protected_registry()
    withheld_keys = {
        os.path.normcase(str(Path(item).resolve())) for item in withheld
    }
    root = Path(root).resolve()
    path_map = {}
    for entry in registry:
        if entry.get("path") is not None:
            # normcase makes path identity robust to Windows case variants while
            # remaining identity on case-sensitive POSIX filesystems.
            path_map[os.path.normcase(str((root / entry["path"]).resolve()))] = entry
    hash_map = {}
    for entry in registry:
        hash_map.setdefault(entry["sha256"], entry)
    digests, checked = {}, 0
    for raw in paths:
        path = Path(raw)
        try:
            resolved = path.resolve()
            digest = sha256_file(resolved) if resolved.is_file() else None
        except OSError as exc:
            raise ExclusionAuditError(
                f"cannot inspect scan candidate {_display_path(path, root)}: {exc}"
            ) from exc
        if os.path.normcase(str(resolved)) in withheld_keys:
            # This exact path is one of the cohort's own outputs: it is
            # withheld from the scan and never parsed.
            digests[str(resolved)] = digest
            checked += 1
            continue
        entry = path_map.get(os.path.normcase(str(resolved)))
        if entry is not None:
            raise ProtectedSourceError(
                f"protected {entry['role']} source refused before deserialization: "
                f"{_display_path(resolved, root)}"
            )
        if digest is not None and digest in hash_map:
            entry = hash_map[digest]
            raise ProtectedSourceError(
                f"protected {entry['role']} bytes refused before deserialization "
                f"(renamed or copied source): {_display_path(resolved, root)}"
            )
        digests[str(resolved)] = digest
        checked += 1
    return {
        "schema": PROTECTED_REGISTRY_SCHEMA,
        "checked": checked,
        "withheld": sorted(
            _display_path(item, root) for item in withheld
        ),
        "digests": digests,
        "registry_sha256": canonical_hash([dict(entry) for entry in registry]),
        "limitation": PROTECTED_REGISTRY_LIMITATION,
    }


def _generated_output_record(root, exclude_generated):
    """Describe (relative to ``root``) the cohort's own upcoming outputs.

    Only the known cohort output filenames are accepted, and only paths inside
    the scanned root are recorded: a path outside the root cannot be re-ingested
    by the source inventory, so excluding it would be meaningless as well as a
    portability leak (absolute temporary paths must not enter the frozen ledger).
    """
    root = Path(root).resolve()
    paths = []
    for item in exclude_generated or ():
        path = Path(item)
        if path.name not in COHORT_OUTPUT_NAMES:
            raise ExclusionAuditError(
                f"refusing to exclude an unexpected generated output: {path.name!r}"
            )
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            continue  # outside every scanned root; cannot be re-ingested
        paths.append(relative.as_posix())
    return {"policy": GENERATED_OUTPUT_POLICY, "paths": sorted(set(paths))}


def generated_output_paths(payload, root=ROOT, cohort_dir=None):
    """Resolve and validate the cohort's recorded self-output exclusions.

    ``cohort_dir`` binds the record to the directory the exclusion ledger was
    loaded from.  A ledger that claims some other directory's files are its own
    output is rejected, so the exclusion can never be used to hide an older
    experiment.
    """
    record = payload.get("generated_outputs")
    if record is None:
        return []
    if not isinstance(record, dict):
        raise ExclusionAuditError("malformed generated-output exclusion record")
    paths = record.get("paths", [])
    if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
        raise ExclusionAuditError("malformed generated-output path list")
    root = Path(root).resolve()
    resolved, parents = [], set()
    for relative in paths:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ExclusionAuditError("generated-output exclusion must be root-relative")
        if candidate.name not in COHORT_OUTPUT_NAMES:
            raise ExclusionAuditError(
                f"generated-output exclusion names an unexpected file: {candidate.name!r}"
            )
        path = (root / candidate).resolve()
        parents.add(path.parent)
        resolved.append(path)
    if len(parents) > 1:
        raise ExclusionAuditError("generated outputs must share one cohort directory")
    if cohort_dir is not None and parents:
        expected = Path(cohort_dir).resolve()
        if parents.pop() != expected:
            raise ExclusionAuditError(
                "generated-output record does not belong to this cohort directory"
            )
    return sorted(resolved)


def cohort_output_paths(out_dir, sealed_test=True):
    """The exact files ``prepare`` will write into ``out_dir``."""
    names = ["graphs-manifest.json", "exclusions-private.json", "prepare-report.json"]
    if sealed_test:
        names.append("test-sealed.json")
    return [Path(out_dir) / name for name in names]


def _expand_sources(root, sources):
    """Expand every explicit path and glob without applying any exclusion.

    The protected-source guard consumes this unfiltered list, so a generated
    output exclusion can never hide a registered protected final from review.
    """
    expanded = []
    missing = []
    for spec in sources:
        if "path" in spec:
            path = Path(root) / spec["path"]
            matches = [path] if path.exists() else []
        else:
            matches = sorted(Path(root).glob(spec["glob"]))
        if not matches and spec.get("required", True):
            missing.append(spec.get("path") or spec.get("glob"))
            continue
        for match in matches:
            expanded.append({"path": match, "kind": spec["kind"]})
    if missing:
        raise ExclusionAuditError(
            "required historic sources are absent; freshness cannot be established: "
            + json.dumps(missing)
        )
    return expanded


def _dedupe_sources(expanded, exclude_paths=frozenset()):
    """Apply the exact generated-output exclusions and drop duplicate matches."""
    resolved = []
    for entry in expanded:
        if entry["path"].resolve() in exclude_paths:
            continue
        resolved.append(entry)
    # An explicit path and a glob can name the same file; keep the first kind
    # (explicit ``toml-access`` before generic ``toml-auto``).
    unique, seen_paths = [], set()
    for entry in resolved:
        key = entry["path"].resolve()
        if key in seen_paths:
            continue
        seen_paths.add(key)
        unique.append(entry)
    return unique


def _resolve_sources(root, sources, exclude_paths=frozenset()):
    return _dedupe_sources(_expand_sources(root, sources), exclude_paths)


def _load_toml_access(path):
    try:
        config = worlds.load_config(path)
    except Exception as exc:  # access-graph configs must compile or fail closed
        raise ExclusionAuditError(f"{path}: access graph config failed to compile: {exc}") from exc
    records = [graph for graph in config.get("graphs", [])]
    template = config.get("template")
    if isinstance(template, dict):
        records.append(template)
    assignments = _native_config_assignments(config)
    return records, assignments


def _load_toml_auto(path):
    with Path(path).open("rb") as source:
        config = tomllib.load(source)
    if config.get("mode") in ("four-layer", "closure-mixture"):
        return _load_toml_access(path)
    records = []
    _extract_records(config, str(path), records)
    assignments = set()
    _collect_assignments(config, assignments)
    return [item["record"] for item in records], assignments


def _load_cubigraph(path):
    """Seventeen-room full access graphs from a published CubiGraph corpus.

    Relations 1 (open) and 2 (door) are the primary access relations.  Only
    reciprocal primary passages are included.  These plans do not carry the
    authored port encoding, so only their full access graph is excluded; no
    backbone is inferred from a source plan that was never compiled into the
    single-cross grammar.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    records = []
    for source_id, item in data.items():
        raw = item.get("graph") if isinstance(item, dict) else None
        if not isinstance(raw, dict) or len(raw) != 17:
            continue
        names = sorted(raw)
        index = {name: position for position, name in enumerate(names)}
        edges = set()
        for source, moves in raw.items():
            if not isinstance(moves, dict):
                raise ExclusionAuditError(f"CubiGraph plan {source_id}: malformed adjacency")
            for target, relation in moves.items():
                if relation not in (1, 2) or target not in raw:
                    continue
                if raw[target].get(source) not in (1, 2):
                    raise ExclusionAuditError(
                        f"CubiGraph plan {source_id}: asymmetric primary access relation"
                    )
                edges.add(tuple(sorted((index[source], index[target]))))
        records.append({"names": names, "edges": [list(edge) for edge in sorted(edges)]})
    return records


def load_historic_exclusions(root=ROOT, sources=DEFAULT_SOURCES, exclude_generated=None,
                             protected_sources=None):
    """Return the frozen exclusion payload or raise ``ExclusionAuditError``.

    The payload contains every distinct 17-node historic graph (full graph and,
    where recoverable, unrooted backbone), the recovered historic cue
    assignments, and per-source SHA-256 provenance.  Missing required sources,
    unparseable files and unreconstructable 17-node records fail closed.  With
    no supplied sources the payload is empty and records
    ``empty_inventory: true``.

    ``exclude_generated`` names the files the *current* prepare invocation is
    about to write into its own output directory.  They are the only paths
    removed from the scan, and their relative names and shared parent are
    recorded so ``recheck_history`` can reproduce exactly the same inventory.

    ``protected_sources`` may register additional protected paths/hashes (for
    example an exclusion-ledger derivative).  The guard runs over every
    explicit-path and glob match *before* any JSON/TOML deserialization, so a
    protected source is refused rather than parsed.
    """
    root = Path(root)
    specs = _source_specs(sources)
    generated = _generated_output_record(root, exclude_generated)
    exclude_paths = frozenset(
        (root / relative).resolve() for relative in generated["paths"]
    )
    registry = protected_registry(additional=protected_sources)
    expanded = _expand_sources(root, specs)
    guard = assert_sources_not_protected(
        [entry["path"] for entry in expanded], root=root, registry=registry,
        withheld=exclude_paths,
    )
    resolved = _dedupe_sources(expanded, exclude_paths)
    raw_records, assignments, source_ledger = [], set(), []
    counts = Counter()
    for entry in resolved:
        path, kind = entry["path"], entry["kind"]
        relative = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        try:
            if kind == "json":
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                extracted = []
                _extract_records(data, relative, extracted)
                _collect_assignments(data, assignments)
                _collect_remap_assignments(data, assignments)
                records = [item["record"] for item in extracted]
            elif kind == "toml-access":
                records, found = _load_toml_access(path)
                assignments |= found
            elif kind == "toml-auto":
                records, found = _load_toml_auto(path)
                assignments |= found
            elif kind == "cubigraph":
                records = _load_cubigraph(path)
            else:
                raise ExclusionAuditError(f"unknown source kind {kind!r}")
        except ExclusionAuditError:
            raise
        except Exception as exc:
            raise ExclusionAuditError(f"{relative}: failed to parse historic source: {exc}") from exc

        seventeen = []
        for record in records:
            n = _record_node_count(record)
            if n == 17:
                # Validate reconstructability immediately; a 17-node record we
                # cannot audit must stop generation.
                _record_full_graph(record)
                seventeen.append({"record": record, "path": relative, "kind": kind})
        counts["records"] += len(records)
        counts["records_17"] += len(seventeen)
        raw_records.extend(seventeen)
        source_ledger.append({
            "path": relative,
            "sha256": guard["digests"].get(str(path.resolve())) or sha256_file(path),
            "kind": kind,
            "graph_records": len(records),
            "graph_records_17": len(seventeen),
        })

    if not raw_records and resolved:
        raise ExclusionAuditError("historic inventory yielded no 17-node graph records")

    # Deduplicate exact isomorphism classes.  WL/degree buckets keep the exact
    # VF2 comparisons local while preserving nonisomorphism guarantees.
    classes, bins = [], {}
    for item in raw_records:
        graph = _record_full_graph(item["record"])
        signature = _signature(graph)
        representative = None
        for existing in bins.get(signature, []):
            if nx.is_isomorphic(graph, existing["graph"]):
                representative = existing
                break
        if representative is None:
            representative = {"graph": graph, "record": item["record"]}
            bins.setdefault(signature, []).append(representative)
            classes.append(representative)
    # Attach provenance (source file + record index) to each class.
    provenance = {}
    for item in raw_records:
        graph = _record_full_graph(item["record"])
        signature = _signature(graph)
        for existing in bins.get(signature, []):
            if nx.is_isomorphic(graph, existing["graph"]):
                provenance.setdefault(id(existing), []).append(item["path"])
                break
    ledger_graphs = []
    backbones = set()
    for index, existing in enumerate(classes):
        record = _normalize_record(existing["record"])
        full = _record_full_graph(record)
        signature = _signature(full)
        tree = _record_backbone_tree(record)
        tree_key = _tree_key(tree) if tree is not None and nx.is_tree(tree) else None
        if tree_key is not None:
            backbones.add(tree_key)
        ledger_graphs.append({
            "class_id": f"historic-{index:03d}",
            "record": record,
            "full_graph_signature": {
                "degree_sequence": list(signature[0]),
                "weisfeiler_lehman": signature[1],
            },
            "backbone_key": tree_key,
            "source_paths": sorted(set(provenance.get(id(existing), []))),
        })

    payload = {
        "schema": EXCLUSIONS_SCHEMA,
        "purpose": "fail-closed historic graph and cue-assignment exclusion ledger",
        # Source paths are relative to the repository root; the root itself is
        # deliberately excluded so the frozen ledger hash is portable across
        # equivalent checkouts and container mounts.
        "sources": source_ledger,
        # The frozen *specification* is what recheck re-expands, so a file added
        # later under a glob is still detected as new historic exposure.
        "source_specs": specs,
        # Exactly which of this cohort's own outputs were withheld from its own
        # scan.  Pre-existing historic files are never listed here.
        "generated_outputs": generated,
        "empty_inventory": not resolved,
        "records_scanned": counts["records"],
        "records_17_scanned": counts["records_17"],
        "distinct_full_classes": len(ledger_graphs),
        "distinct_backbone_classes": len(backbones),
        "graphs": ledger_graphs,
        "assignments": [list(item) for item in sorted(assignments)],
        "assignment_count": len(assignments),
    }
    return _seal(payload)


def exclusion_state(payload):
    """Build fast lookup state from a sealed exclusion payload."""
    _check_digest(payload, kind="exclusion payload")
    if payload.get("schema") != EXCLUSIONS_SCHEMA:
        raise ExclusionAuditError("exclusion payload schema mismatch")
    trees, bins, graphs = set(), {}, []
    for entry in payload["graphs"]:
        record = entry["record"]
        graph = _record_full_graph(record)
        tree = _record_backbone_tree(record)
        tree_key = _tree_key(tree) if tree is not None and nx.is_tree(tree) else None
        if tree_key is not None:
            trees.add(tree_key)
        bins.setdefault(_signature(graph), []).append(graph)
        graphs.append({"record": record, "graph": graph, "tree_key": tree_key})
    assignments = {tuple(item) for item in payload.get("assignments", [])}
    return {"trees": trees, "bins": bins, "graphs": graphs, "assignments": assignments}


# ---------------------------------------------------------------------------
# Exhaustive topology capacity
# ---------------------------------------------------------------------------
#
# ``generate_cohort`` admits at most one family per distinct unrooted backbone,
# so the number of *fresh* backbone shapes is a hard ceiling for any cohort.
# The grammar admits a small number of unrooted tree shapes in total.  This
# section computes that space exhaustively instead of relying on a random
# candidate bound, so capacity failures can be reported as mathematics rather
# than as a sampler shortfall.  Nothing here fits a model, scores a query or
# opens the sealed test payload.


def _grammar_rootings(tree, layers, max_children, budgets, cross_edges):
    """Rooted orientations under which ``generate_candidate`` can emit ``tree``.

    ``generate_candidate`` builds a rooted tree with root depth 0, at most three
    children per node, exactly ``layers`` occupied depths, the disclosed child
    action budgets and at least seven same-depth pairs for the cross ports.  A
    tree shape is reachable exactly when at least one rooting satisfies all of
    those conditions.
    """
    degrees = dict(tree.degree())
    if not degrees or max(degrees.values()) > max_children + 1:
        return []
    reports = []
    for root in sorted(tree.nodes()):
        if degrees[root] > max_children:
            continue
        depths = nx.single_source_shortest_path_length(tree, root)
        if len(depths) != tree.number_of_nodes() or max(depths.values()) != layers - 1:
            continue
        children = [degrees[node] - (0 if node == root else 1) for node in tree.nodes()]
        if any(count > max_children for count in children):
            continue
        thresholds = [sum(count > slot for count in children) for slot in range(max_children)]
        if any(not count or count > budget for count, budget in zip(thresholds, budgets)):
            continue
        populations = tuple(
            sum(1 for depth in depths.values() if depth == layer) for layer in range(1, layers)
        )
        if sum(size // 2 for size in populations) < cross_edges:
            continue
        reports.append({
            "root": root,
            "depths": tuple(depths[node] for node in sorted(tree.nodes())),
            "populations": populations,
        })
    return reports


def backbone_space_index(layers=None, max_children=None, budgets=None, cross_edges=None,
                         nodes=None):
    """Exhaustively index every unrooted backbone shape the sampler can emit.

    ``networkx.nonisomorphic_trees`` yields every unlabelled tree on ``nodes``
    vertices exactly once, so the result is complete for the frozen grammar.  A
    candidate bound on random sampling is not needed to prove capacity, and no
    historic inventory is consulted here: this is the grammar's own ceiling.
    """
    layers = STRUCTURE["layers"] if layers is None else int(layers)
    max_children = STRUCTURE["max_children"] if max_children is None else int(max_children)
    budgets = tuple(CHILD_ACTION_BUDGETS if budgets is None else budgets)
    cross_edges = STRUCTURE["cross_edges"] if cross_edges is None else int(cross_edges)
    nodes = STRUCTURE["nodes"] if nodes is None else int(nodes)
    entries, considered, within_degree = [], 0, 0
    for tree in nx.nonisomorphic_trees(nodes):
        considered += 1
        if max(dict(tree.degree()).values()) > max_children + 1:
            continue
        within_degree += 1
        rootings = _grammar_rootings(tree, layers, max_children, budgets, cross_edges)
        if not rootings:
            continue
        entries.append({"key": _tree_key(tree), "tree": tree, "rootings": rootings})
    entries.sort(key=lambda entry: entry["key"])
    keys = [entry["key"] for entry in entries]
    if len(set(keys)) != len(keys):
        raise RuntimeError("backbone enumeration produced duplicate canonical keys")
    stats = {
        "method": "exhaustive unrooted-tree enumeration",
        "nodes": nodes,
        "layers": layers,
        "max_children": max_children,
        "child_action_budgets": list(budgets),
        "cross_edges": cross_edges,
        "unrooted_trees": considered,
        "within_degree_bound": within_degree,
        "distinct_backbone_shapes": len(entries),
    }
    return entries, stats


def enumerate_backbone_space(**kwargs):
    """Sorted canonical keys of every backbone shape the generator can emit."""
    entries, _ = backbone_space_index(**kwargs)
    return [entry["key"] for entry in entries]


def topology_capacity(exclusions, requested=None, space=None):
    """Exact capacity of the frozen single-cross grammar against a ledger.

    The upper bound is the number of fresh backbone shapes because the generator
    admits one family per distinct unrooted backbone.  This function performs no
    candidate sampling, model fit or query scoring, and never parses the sealed
    test payload.
    """
    entries, stats = backbone_space_index() if space is None else space
    state = exclusion_state(exclusions)
    keys = {entry["key"] for entry in entries}
    consumed = keys & state["trees"]
    fresh = keys - state["trees"]
    outside = state["trees"] - keys
    requested = dict(DEFAULT_COUNTS if requested is None else requested)
    if set(requested) != set(SPLITS):
        raise ValueError("requested counts must name exactly train, validation and test")
    requested = {split: int(requested[split]) for split in SPLITS}
    if any(value < 0 for value in requested.values()):
        raise ValueError("family counts must be non-negative integers")
    # The two named gate requests are always reported; a caller-specific
    # request is added only when it is not one of them.
    known = [
        (label, {split: int(counts[split]) for split in SPLITS})
        for label, counts in REQUESTED_VARIANTS
    ]
    known_keys = {tuple(counts[split] for split in SPLITS) for _, counts in known}
    variants = list(known)
    if tuple(requested[split] for split in SPLITS) not in known_keys:
        variants.insert(0, ("requested", requested))
    requests = []
    for label, counts in variants:
        total = sum(counts.values())
        requests.append({
            "label": label,
            "counts": counts,
            "total": total,
            "feasible": total <= len(fresh),
            "deficit": max(0, total - len(fresh)),
        })
    total = sum(requested.values())
    return {
        "schema": CAPACITY_SCHEMA,
        "status": "feasible" if total <= len(fresh) else "infeasible",
        "exhaustive": True,
        "candidate_bound_dependent": False,
        "enumeration": stats,
        "backbone_space": len(entries),
        "one_family_per_distinct_backbone": True,
        "history_backbones": len(state["trees"]),
        "history_backbones_inside_space": len(consumed),
        "history_backbones_outside_space": len(outside),
        "fresh_backbones": len(fresh),
        "max_families_under_current_rule": len(fresh),
        "training_families_available": len(fresh) >= requested["train"],
        "requested": requested,
        "requested_total": total,
        "deficit": max(0, total - len(fresh)),
        "requests": requests,
        "exclusion_content_sha256": exclusions.get("content_sha256"),
        "model_optimizer_updates": 0,
        "test_payload_opened": False,
        "final_test_scored": False,
    }


def _matchings(nodes, count):
    """All ``count``-edge matchings on ``nodes``, in deterministic order."""
    nodes = tuple(sorted(nodes))
    if count == 0:
        yield ()
        return
    if len(nodes) < 2 * count:
        return
    first, rest = nodes[0], nodes[1:]
    for matching in _matchings(rest, count):
        yield matching
    for index, partner in enumerate(rest):
        remaining = rest[:index] + rest[index + 1:]
        for matching in _matchings(remaining, count - 1):
            yield ((first, partner),) + matching


def _matching_solutions(populations, cross_edges):
    """Per-layer cross-pair counts that total exactly ``cross_edges``."""
    floors = [size // 2 for size in populations]
    solutions = []

    def walk(layer, remaining, chosen):
        if layer == len(floors):
            if remaining == 0:
                solutions.append(tuple(chosen))
            return
        for count in range(min(floors[layer], remaining) + 1):
            walk(layer + 1, remaining - count, chosen + [count])

    walk(0, cross_edges, [])
    return solutions


def _realisation_graph(tree, rooting, matching):
    """The full 23-link graph for one backbone rooting and cross matching."""
    nodes = tree.number_of_nodes()
    depths = rooting["depths"]
    children = [[] for _ in range(nodes)]
    for node in tree.nodes():
        for neighbour in tree[node]:
            if depths[neighbour] == depths[node] + 1:
                children[node].append(neighbour)
    transitions = [{} for _ in range(nodes)]
    edges = []
    for node, branches in enumerate(children):
        for action, child in zip(CHILD, sorted(branches)):
            transitions[node][action] = child
            transitions[child][1] = node
            edges.append([node, child])
    for u, v in matching:
        transitions[u][4] = v
        transitions[v][4] = u
        edges.append([u, v])
    graph = {
        "root": rooting["root"],
        "names": [f"capacity-space-{node:02d}" for node in range(nodes)],
        "depths": list(depths),
        "edges": sorted(edges),
        "transitions": transitions,
    }
    return base.graph_nx(base.validate_graph(graph))


def _realisations(entry, limit=None, layers=None, cross_edges=None):
    """Deterministically enumerate full-graph realisations of one shape."""
    layers = STRUCTURE["layers"] if layers is None else int(layers)
    cross_edges = STRUCTURE["cross_edges"] if cross_edges is None else int(cross_edges)
    produced = 0
    tree = entry["tree"]
    for rooting in entry["rootings"]:
        depths = rooting["depths"]
        by_layer = {
            layer: sorted(node for node in tree.nodes() if depths[node] == layer)
            for layer in range(1, layers)
        }
        for solution in _matching_solutions(rooting["populations"], cross_edges):
            pools = [
                list(_matchings(by_layer[layer], solution[layer - 1]))
                for layer in range(1, layers)
            ]
            for combination in itertools.product(*pools):
                matching = [pair for part in combination for pair in part]
                yield _realisation_graph(tree, rooting, matching)
                produced += 1
                if limit is not None and produced >= limit:
                    return


def _full_graph_fresh(graph, state, accepted):
    """Fail-closed full-graph freshness against history and accepted families."""
    signature = _signature(graph)
    for existing in state["bins"].get(signature, []):
        if nx.is_isomorphic(graph, existing):
            return False, signature
    for existing in accepted.get(signature, []):
        if nx.is_isomorphic(graph, existing):
            return False, signature
    return True, signature


def constructive_capacity(exclusions, space=None, max_matchings_per_shape=64):
    """Certify that every fresh backbone can supply one valid family.

    The strict generator admits one family per unrooted backbone, so its exact
    capacity equals the fresh-shape count only if every fresh shape has a
    full-graph realisation that is neither historic-isomorphic nor
    cohort-isomorphic.  This bounded deterministic search certifies that lower
    bound without emitting a cohort.
    """
    entries, stats = backbone_space_index() if space is None else space
    state = exclusion_state(exclusions)
    accepted, certified = {}, []
    rejections = Counter()
    fresh_entries = [entry for entry in entries if entry["key"] not in state["trees"]]
    for entry in fresh_entries:
        for graph in _realisations(
            entry, limit=max_matchings_per_shape,
            layers=stats["layers"], cross_edges=stats["cross_edges"],
        ):
            fresh, signature = _full_graph_fresh(graph, state, accepted)
            if not fresh:
                rejections["isomorphic_collision"] += 1
                continue
            accepted.setdefault(signature, []).append(graph)
            certified.append(entry["key"])
            break
        else:
            rejections["no_fresh_realisation"] += 1
    return {
        "schema": CAPACITY_SCHEMA,
        "status": "certified" if not rejections else "partial",
        "fresh_backbones": len(fresh_entries),
        "certified_families": len(certified),
        "max_matchings_per_shape": max_matchings_per_shape,
        "rejections": dict(rejections),
        "history_backbone_exclusion_retained": True,
        "history_full_graph_exclusion_retained": True,
        "intra_cohort_backbone_nonisomorphism": True,
        "intra_cohort_full_graph_nonisomorphism": True,
        "model_optimizer_updates": 0,
        "test_payload_opened": False,
    }


def relaxed_backbone_certificate(exclusions, requested=None, space=None,
                                 max_matchings_per_shape=512):
    """DIAGNOSTIC ONLY: capacity if a cohort may reuse fresh scaffolds.

    Historic backbone and full-graph exclusions stay in force, and families must
    still be pairwise full-graph nonisomorphic.  The one weakened condition is
    *intra-cohort* backbone nonisomorphism: two families may share the 16-edge
    tree scaffold and differ in the 7 reciprocal cross edges (and, where a
    scaffold has two valid rootings, the root).  That is a protocol amendment,
    not an authorized generator mode; this certificate only supplies an
    executable number for that decision.
    """
    entries, stats = backbone_space_index() if space is None else space
    state = exclusion_state(exclusions)
    requested = dict(DEFAULT_COUNTS if requested is None else requested)
    target = sum(int(requested[split]) for split in SPLITS)
    fresh_entries = [entry for entry in entries if entry["key"] not in state["trees"]]
    accepted, counts = {}, Counter()

    def attempt(entry, limit):
        for graph in _realisations(
            entry, limit=limit, layers=stats["layers"], cross_edges=stats["cross_edges"],
        ):
            fresh, signature = _full_graph_fresh(graph, state, accepted)
            if not fresh:
                continue
            accepted.setdefault(signature, []).append(graph)
            counts[entry["key"]] += 1
            return True
        return False

    # First give every fresh scaffold one family, so the relaxed cohort keeps as
    # much structural spread as the weakened rule allows.
    for entry in fresh_entries:
        attempt(entry, 1)
    while sum(counts.values()) < target:
        progress = 0
        for entry in fresh_entries:
            if sum(counts.values()) >= target:
                break
            if attempt(entry, max_matchings_per_shape):
                progress += 1
        if not progress:
            break
    total = sum(counts.values())
    return {
        "schema": CAPACITY_SCHEMA,
        "status": "certificate_reached" if total >= target else "certificate_short",
        "requires_explicit_approval": True,
        "authorized_to_generate": False,
        "scientific_authorization": (
            "NOT AUTHORIZED: dropping intra-cohort backbone nonisomorphism is a "
            "protocol amendment requiring independent review"
        ),
        "requested": {split: int(requested[split]) for split in SPLITS},
        "requested_total": target,
        "certified_families": total,
        "fresh_scaffolds": len(fresh_entries),
        "scaffolds_used": len(counts),
        "families_sharing_a_scaffold": total - len(counts),
        "max_families_on_one_scaffold": max(counts.values(), default=0),
        "history_backbone_exclusion_retained": True,
        "history_full_graph_exclusion_retained": True,
        "intra_cohort_backbone_nonisomorphism": False,
        "intra_cohort_full_graph_nonisomorphism": True,
        "model_optimizer_updates": 0,
        "test_payload_opened": False,
    }


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def generate_candidate(rng, budgets=CHILD_ACTION_BUDGETS):
    """One structurally valid candidate, or ``None`` when ineligible.

    Selection is structure-only: random parent choices, a shuffled disjoint
    same-depth matching for crosses, and the disclosed child-action budgets.
    No cue, score or graph identity enters the sampler.
    """
    children, depths = [[]], [0]
    for child in range(1, STRUCTURE["nodes"]):
        possible = [
            node
            for node in range(child)
            if depths[node] < STRUCTURE["layers"] - 1 and len(children[node]) < STRUCTURE["max_children"]
        ]
        if not possible:
            return None
        parent = rng.choice(possible)
        children[parent].append(child)
        children.append([])
        depths.append(depths[parent] + 1)
    if max(depths) != STRUCTURE["layers"] - 1:
        return None
    child_counts = [sum(len(branches) > slot for branches in children) for slot in range(len(CHILD))]
    if any(not count or count > budget for count, budget in zip(child_counts, budgets)):
        return None
    matching = []
    for depth in range(1, STRUCTURE["layers"]):
        nodes = [node for node, value in enumerate(depths) if value == depth]
        rng.shuffle(nodes)
        matching.extend(sorted(nodes[index:index + 2]) for index in range(0, len(nodes) - 1, 2))
    if len(matching) < STRUCTURE["cross_edges"]:
        return None
    rng.shuffle(matching)
    transitions = [{} for _ in children]
    edges = []
    for node, branches in enumerate(children):
        for action, child in zip(CHILD, branches):
            transitions[node][action] = child
            transitions[child][1] = node
            edges.append([node, child])
    for u, v in matching[:STRUCTURE["cross_edges"]]:
        transitions[u][4] = v
        transitions[v][4] = u
        edges.append([u, v])
    graph = {
        "root": 0,
        "names": [f"retrain-space-{node:02d}" for node in range(STRUCTURE["nodes"])],
        "depths": depths,
        "edges": sorted(edges),
        "transitions": transitions,
    }
    return base.validate_graph(graph)


def _is_fresh(graph, state, accepted_trees, accepted_bins):
    tree_key = base.backbone_key(graph)
    if tree_key in state["trees"] or tree_key in accepted_trees:
        return False, "backbone_isomorphic"
    full = base.graph_nx(graph)
    signature = _signature(full)
    for existing in state["bins"].get(signature, []):
        if nx.is_isomorphic(full, existing):
            return False, "full_isomorphic_history"
    for existing in accepted_bins.get(signature, []):
        if nx.is_isomorphic(full, existing):
            return False, "full_isomorphic_cohort"
    return True, tree_key


def generate_cohort(seed=0, counts=None, max_candidates=200000,
                    exclusions=None, sealed_test=True, root=ROOT):
    """Generate a frozen cohort manifest (and sealed test payload).

    ``counts`` defaults to the gate's 96/24/24 families.  ``exclusions`` may be
    supplied as an already-loaded sealed payload for tests or for a caller that
    owns its own historic ledger; when ``None`` the module loads the caller's
    configured sources, which are empty by default.  An empty inventory is
    disclosed in the manifest's exclusion ledger rather than assumed.
    """
    counts = dict(DEFAULT_COUNTS) if counts is None else dict(counts)
    if set(counts) != set(SPLITS):
        raise ValueError("counts must name exactly train, validation and test")
    if any(not _is_int(value) or value < 0 for value in counts.values()):
        raise ValueError("family counts must be non-negative integers")
    if sum(counts.values()) < 1:
        raise ValueError("at least one family is required")
    if not _is_int(max_candidates) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if exclusions is None:
        exclusions = load_historic_exclusions(root=root)
    state = exclusion_state(exclusions)
    reserved = sorted(state["assignments"])
    rng = random.Random(f"{GRAPH_SEED_NAMESPACE}:{seed}")
    requested = [(split, index) for split in SPLITS for index in range(counts[split])]
    graphs, accepted_trees, accepted_bins = [], set(), {}
    rejected = Counter()
    attempts = 0
    for attempts in range(1, max_candidates + 1):
        candidate = generate_candidate(rng)
        if candidate is None:
            rejected["structural_ineligible"] += 1
            continue
        fresh, tree_key = _is_fresh(candidate, state, accepted_trees, accepted_bins)
        if not fresh:
            rejected[tree_key] += 1
            continue
        split, index = requested[len(graphs)]
        candidate["id"] = f"retrain-{split}-{index:03d}"
        candidate["split"] = split
        candidate["provenance"] = {
            "kind": "authored synthetic retrain cohort",
            "not_a_real_building": True,
            "generator": "multitopology_retrain_graphs.py",
            "generator_schema": SCHEMA,
            "generator_seed": seed,
            "cross_grammar": "same backbone depth, reciprocal single cross port",
            "child_ports": list(CHILD),
            "selection_uses_model_scores": False,
            "freshness_basis": "full-graph and unrooted-backbone nonisomorphism against the supplied ledger",
        }
        graphs.append(candidate)
        accepted_trees.add(tree_key)
        accepted_bins.setdefault(_signature(base.graph_nx(candidate)), []).append(base.graph_nx(candidate))
        if len(graphs) == len(requested):
            break
    if len(graphs) != len(requested):
        # The sampler cannot exceed the grammar ceiling.  Report the exact,
        # exhaustive capacity so the coordinator can tell a bad seed or a small
        # candidate bound apart from a cohort that cannot exist at all.
        capacity = topology_capacity(exclusions, requested=counts)
        raise RuntimeError(json.dumps({
            "status": "infeasible_within_candidate_bound",
            "requested": counts,
            "eligible_distinct_families_found": len(graphs),
            "candidate_count": attempts,
            "rejections": dict(rejected),
            "exhaustive_impossibility_claimed": capacity["fresh_backbones"] < sum(counts.values()),
            "exhaustive_capacity": capacity,
        }))

    sample_maps = {
        graph["id"]: [
            list(base.cue_assignment(graph, graph["split"], episode, seed, reserved))
            for episode in range(SAMPLE_CUE_MAPS_PER_FAMILY)
        ]
        for graph in graphs
    }
    groups = {split: [graph for graph in graphs if graph["split"] == split] for split in SPLITS}
    test_graphs = groups["test"]
    public_graphs = [graph for graph in graphs if graph["split"] != "test"]

    sealed = None
    if sealed_test:
        sealed = _seal({
            "schema": SEALED_SCHEMA,
            "cohort_id": f"retrain-{seed}",
            "seed": seed,
            "purpose": "final test families; unseal only after checkpoint freeze and QA",
            "structure": copy.deepcopy(STRUCTURE),
            "graphs": test_graphs,
            "sample_cue_maps": {graph["id"]: sample_maps[graph["id"]] for graph in test_graphs},
        })
        test_commitment = {
            "status": "sealed",
            "sealed_file": "test-sealed.json",
            "payload_sha256": sealed["content_sha256"],
            "graph_count": len(test_graphs),
            "graph_ids": [graph["id"] for graph in test_graphs],
            "opened_for_scoring": False,
            "unseal_precondition": "checkpoint frozen, validation-only selection complete, independent QA passed",
        }
        public_maps = {graph["id"]: sample_maps[graph["id"]] for graph in public_graphs}
    else:
        public_graphs = graphs
        public_maps = sample_maps
        test_commitment = {
            "status": "unsealed",
            "graph_count": len(test_graphs),
            "graph_ids": [graph["id"] for graph in test_graphs],
            "opened_for_scoring": False,
            "warning": "structural payload left in the public manifest; sealed mode is the default",
        }

    exclusion_ledger = {
        "schema": EXCLUSIONS_SCHEMA,
        "file": "exclusions-private.json",
        "content_sha256": exclusions["content_sha256"],
        "empty_inventory": exclusions.get("empty_inventory", False),
        "records_scanned": exclusions["records_scanned"],
        "records_17_scanned": exclusions["records_17_scanned"],
        "distinct_full_classes": exclusions["distinct_full_classes"],
        "distinct_backbone_classes": exclusions["distinct_backbone_classes"],
        "assignment_count": exclusions["assignment_count"],
        "sources": [
            {"path": source["path"], "sha256": source["sha256"], "graph_records_17": source["graph_records_17"]}
            for source in exclusions["sources"]
        ],
        "fail_closed": True,
    }

    manifest = {
        "schema": SCHEMA,
        "cohort_id": f"retrain-{seed}",
        "seed": seed,
        "structure": copy.deepcopy(STRUCTURE),
        "graphs": public_graphs,
        "families": [
            {"family_id": graph["id"], "graph_id": graph["id"], "split": graph["split"]}
            for graph in public_graphs
        ],
        "splits": {split: [graph["id"] for graph in groups[split]] for split in SPLITS},
        "sample_cue_maps": public_maps,
        "cue_contract": {
            "vocabulary_size": CUE_VOCABULARY,
            "unique_cues_per_family": 17,
            "assignment_partition": "sha256(canonical ordered assignment) modulo 3: train=0 validation=1 test=2",
            "seed_namespace": "multitopology-cues-v1",
            "graph_independent_for_arm_parity": True,
            "sample_episodes_per_family": SAMPLE_CUE_MAPS_PER_FAMILY,
            "historic_reserved_assignments": len(reserved),
            "runtime_interface": "cue_assignment(graph, split, episode, seed, reserved)",
        },
        "exclusion_ledger": exclusion_ledger,
        "test_commitment": test_commitment,
        "generation": {
            "requested": counts,
            "candidate_count": attempts,
            "max_candidates": max_candidates,
            "rejections": dict(rejected),
            "full_and_unrooted_backbone_exclusion": True,
            "same_depth_cross_grammar": True,
            "selection_uses_model_scores": False,
            "model_optimizer_updates": 0,
        },
        "content_sha256": None,
    }
    _seal(manifest)
    return manifest, sealed


def cue_assignment(graph, split, episode, seed, reserved=()):
    """Documented runtime cue-map interface (historic reservations included).

    Delegates to the existing deterministic, graph-independent split-partitioned
    sampler so matched arms receive identical assignments.
    """
    return base.cue_assignment(graph, split, episode, seed, reserved)


# ---------------------------------------------------------------------------
# Auditing
# ---------------------------------------------------------------------------


def _validate_family_graph(graph):
    if not isinstance(graph, dict):
        raise CohortAuditError("graph payload must be an object")
    if not isinstance(graph.get("id"), str) or not graph["id"]:
        raise CohortAuditError("graph needs a non-empty ID")
    if graph.get("split") not in SPLITS:
        raise CohortAuditError(f"{graph['id']}: invalid split")
    forbidden = FORBIDDEN_GRAPH_KEYS & set(graph)
    if forbidden:
        raise CohortAuditError(f"{graph['id']}: private/scored fields present: {sorted(forbidden)}")
    provenance = graph.get("provenance")
    if not isinstance(provenance, dict):
        raise CohortAuditError(f"{graph['id']}: missing provenance")
    leaked = FORBIDDEN_GRAPH_KEYS & set(provenance)
    if leaked:
        raise CohortAuditError(f"{graph['id']}: provenance leaks private fields: {sorted(leaked)}")
    if provenance.get("selection_uses_model_scores") is not False:
        raise CohortAuditError(f"{graph['id']}: provenance must deny model-score selection")
    if provenance.get("not_a_real_building") is not True:
        raise CohortAuditError(f"{graph['id']}: synthetic provenance marker missing")
    try:
        validated = base.validate_graph(graph)
    except Exception as exc:
        raise CohortAuditError(f"{graph.get('id')}: invalid family graph: {exc}") from exc
    if len(validated["names"]) != STRUCTURE["nodes"] or len(validated["edges"]) != STRUCTURE["total_edges"]:
        raise CohortAuditError(f"{graph['id']}: wrong size")
    tree = base.graph_nx(validated, True)
    if tree.number_of_edges() != STRUCTURE["tree_edges"] or not nx.is_tree(tree):
        raise CohortAuditError(f"{graph['id']}: backbone is not a 16-edge tree")
    if max(validated["depths"]) != STRUCTURE["layers"] - 1:
        raise CohortAuditError(f"{graph['id']}: wrong layer count")
    if any(sum(action in CHILD for action in moves) > STRUCTURE["max_children"] for moves in validated["transitions"]):
        raise CohortAuditError(f"{graph['id']}: child port limit exceeded")
    return validated


def _assert_mutually_fresh(graphs, state, label):
    trees, bins = set(state["trees"]), {key: list(value) for key, value in state["bins"].items()}
    for graph in graphs:
        tree_key = base.backbone_key(graph)
        if tree_key in trees:
            raise CohortAuditError(f"{label}: backbone isomorphism with a rejected/historic family")
        trees.add(tree_key)
        full = base.graph_nx(graph)
        signature = _signature(full)
        for existing in bins.get(signature, []):
            if nx.is_isomorphic(full, existing):
                raise CohortAuditError(f"{label}: full-graph isomorphism with a rejected/historic family")
        bins.setdefault(signature, []).append(full)


def _audit_cue_maps(maps_by_split, reserved):
    """Validate cue maps under the documented graph-independent contract.

    The same assignment is intentionally shared by every family in a split at a
    given episode index; this is what pairs cue streams between matched arms.
    Uniqueness therefore applies *within* one assignment, while assignment
    sequences must be identical across families and disjoint across splits.
    """
    seen = {split: set() for split in SPLITS}
    for split, maps in maps_by_split.items():
        sequences = set()
        for family_id, assignments in maps.items():
            if not isinstance(assignments, list) or not assignments:
                raise CohortAuditError(f"{family_id}: missing cue assignments")
            local = set()
            for assignment in assignments:
                if not _valid_assignment(assignment):
                    raise CohortAuditError(f"{family_id}: invalid cue assignment")
                key = tuple(int(cue) for cue in assignment)
                if key in reserved:
                    raise CohortAuditError(f"{family_id}: historic cue assignment reused")
                if base.assignment_namespace(key) != SPLITS.index(split):
                    raise CohortAuditError(f"{family_id}: cue assignment belongs to another split")
                if key in local:
                    raise CohortAuditError(f"{family_id}: duplicate cue assignment for one family")
                local.add(key)
                seen[split].add(key)
            sequences.add(tuple(tuple(int(cue) for cue in assignment) for assignment in assignments))
        if len(sequences) > 1:
            raise CohortAuditError(f"{split}: cue maps are not graph-independent across families")
    for index, split in enumerate(SPLITS):
        for other in SPLITS[index + 1:]:
            if seen[split] & seen[other]:
                raise CohortAuditError("cue assignment reused across splits")
    return {split: len(values) for split, values in seen.items()}


def audit_cohort(manifest, exclusions=None, sealed=None, allow_test=False, root=ROOT):
    """Audit a frozen cohort without fitting or scoring any model.

    Without ``allow_test`` the sealed test payload is *not parsed*; only its
    SHA-256 commitment is checked.  ``allow_test=True`` explicitly opens the
    frozen test payload for structural validation and must be used only as part
    of the post-freeze independent QA gate.
    """
    _check_digest(manifest, kind="cohort manifest")
    if manifest.get("schema") != SCHEMA:
        raise CohortAuditError("cohort schema mismatch")
    if manifest.get("test_commitment", {}).get("opened_for_scoring") not in (False, None):
        raise CohortAuditError("test commitment claims scoring already occurred")
    if exclusions is None:
        raise CohortAuditError("exclusion ledger is required for an audit")
    state = exclusion_state(exclusions)
    reserved = set(state["assignments"])

    public_graphs = manifest.get("graphs")
    if not isinstance(public_graphs, list):
        raise CohortAuditError("manifest graphs must be a list")

    splits = manifest.get("splits", {})
    if set(splits) != set(SPLITS):
        raise CohortAuditError("splits must be exactly train, validation and test")
    for split in SPLITS:
        ids = splits[split]
        if not isinstance(ids, list) or len(set(ids)) != len(ids):
            raise CohortAuditError(f"{split}: invalid or duplicate split membership")
        if any(not isinstance(item, str) for item in ids):
            raise CohortAuditError(f"{split}: split IDs must be strings")
    all_ids = [item for split in SPLITS for item in splits[split]]
    if len(set(all_ids)) != len(all_ids):
        raise CohortAuditError("split leakage: a family ID appears in more than one split")

    commitment = manifest.get("test_commitment", {})
    status = commitment.get("status")
    if status not in ("sealed", "unsealed"):
        raise CohortAuditError("test commitment status must be sealed or unsealed")
    if status == "unsealed" and not allow_test:
        raise CohortAuditError("test payloads are present; pass allow_test=True to validate them")
    payload_splits = {"train", "validation"} if status == "sealed" else set(SPLITS)
    if status == "sealed" and any(graph.get("split") == "test" for graph in public_graphs):
        raise CohortAuditError("sealed manifest must not contain test graph payloads")
    public_by_id = {}
    for graph in public_graphs:
        validated = _validate_family_graph(graph)
        if validated["id"] in public_by_id:
            raise CohortAuditError("duplicate family ID")
        public_by_id[validated["id"]] = validated

    if set(public_by_id) != {family_id for split in payload_splits for family_id in splits[split]}:
        raise CohortAuditError("graph payloads must cover exactly the disclosed splits")
    for split in payload_splits:
        if [graph["id"] for graph in public_graphs if graph.get("split") == split] != splits[split]:
            raise CohortAuditError(f"{split}: payload order/split disagreement")
    if status == "unsealed" and set(public_by_id) != set(all_ids):
        raise CohortAuditError("unsealed manifest must carry every split payload")

    families = manifest.get("families")
    if not isinstance(families, list):
        raise CohortAuditError("family grouping table missing")
    family_ids = [entry.get("family_id") for entry in families]
    if len(set(family_ids)) != len(family_ids):
        raise CohortAuditError("family grouping repeats a family ID")
    expected_family_ids = (
        set(splits["train"]) | set(splits["validation"])
        if status == "sealed" else set(all_ids)
    )
    if set(family_ids) != expected_family_ids:
        raise CohortAuditError("family grouping must cover exactly the disclosed families")
    for entry in families:
        graph = public_by_id.get(entry["family_id"])
        if graph is None:
            raise CohortAuditError("family grouping refers to an uninspected family")
        if entry.get("graph_id") != entry["family_id"] or entry.get("split") != graph["split"]:
            raise CohortAuditError("family grouping disagrees with graph identity/split")

    # Test handling: hash-only unless explicitly opened.
    test_graphs = None
    if status == "sealed":
        if sealed is None:
            raise CohortAuditError("sealed test payload is missing")
        _check_digest(sealed, kind="sealed test payload")
        if sealed.get("schema") != SEALED_SCHEMA:
            raise CohortAuditError("sealed test schema mismatch")
        if sealed.get("content_sha256") != commitment.get("payload_sha256"):
            raise CohortAuditError("sealed test payload hash does not match the manifest commitment")
        if [graph.get("id") for graph in sealed.get("graphs", [])] != commitment.get("graph_ids"):
            raise CohortAuditError("sealed test family IDs do not match the commitment")
        if len(sealed.get("graphs", [])) != commitment.get("graph_count"):
            raise CohortAuditError("sealed test family count does not match the commitment")
        if allow_test:
            test_graphs = [_validate_family_graph(graph) for graph in sealed["graphs"]]
            if [graph["split"] for graph in test_graphs] != ["test"] * len(test_graphs):
                raise CohortAuditError("sealed families must all be in the test split")
            public_by_id.update({graph["id"]: graph for graph in test_graphs})
    else:
        test_graphs = [public_by_id[family_id] for family_id in splits["test"]]

    if len(splits["test"]) != len(commitment.get("graph_ids", [])):
        raise CohortAuditError("test split count disagrees with the commitment")
    if set(splits["test"]) != set(commitment.get("graph_ids", [])):
        raise CohortAuditError("test split IDs disagree with the commitment")

    # Mutual freshness of every inspected family, then against history.
    inspected_graphs = list(public_by_id.values())
    _assert_mutually_fresh(inspected_graphs, {"trees": set(), "bins": {}}, "inspected cohort")
    _assert_mutually_fresh(inspected_graphs, state, "inspected cohort vs history")

    maps = manifest.get("sample_cue_maps")
    if not isinstance(maps, dict):
        raise CohortAuditError("sample cue maps must be an object")
    public_family_ids = set(splits["train"]) | set(splits["validation"])
    expected_map_ids = public_family_ids if status == "sealed" else set(all_ids)
    if set(maps) != expected_map_ids:
        raise CohortAuditError("sample cue maps must cover exactly the disclosed families")
    maps_by_split = {
        split: {family_id: maps[family_id] for family_id in splits[split]}
        for split in ("train", "validation")
    }
    if test_graphs is not None:
        if status == "sealed":
            sealed_maps = sealed.get("sample_cue_maps", {})
            if set(sealed_maps) != set(splits["test"]):
                raise CohortAuditError("sealed cue maps must cover exactly the test families")
            maps_by_split["test"] = {family_id: sealed_maps[family_id] for family_id in splits["test"]}
        else:
            if set(maps) != set(all_ids):
                raise CohortAuditError("unsealed cue maps must cover every family")
            maps_by_split["test"] = {family_id: maps[family_id] for family_id in splits["test"]}
    counts = _audit_cue_maps(maps_by_split, reserved)

    expected = manifest.get("generation", {}).get("requested", {})
    for split in SPLITS:
        if expected.get(split) != len(splits[split]):
            raise CohortAuditError(f"{split}: family count disagrees with generation.requested")

    return {
        "status": "passed",
        "schema": SCHEMA,
        "cohort_id": manifest.get("cohort_id"),
        "seed": manifest.get("seed"),
        "families": {split: len(splits[split]) for split in SPLITS},
        "public_graphs": len(public_graphs),
        "test_payload_status": commitment.get("status"),
        "test_structurally_validated": bool(allow_test),
        "test_scored": False,
        "excluded_historic_classes": exclusions.get("distinct_full_classes"),
        "excluded_historic_backbones": exclusions.get("distinct_backbone_classes"),
        "reserved_historic_assignments": exclusions.get("assignment_count"),
        "empty_inventory": exclusions.get("empty_inventory", False),
        "cue_assignments_checked": counts,
        "model_optimizer_updates": 0,
        "full_and_backbone_nonisomorphism": True,
        "split_leakage": False,
        "private_fields": False,
    }


# ---------------------------------------------------------------------------
# Persistence and unsealing
# ---------------------------------------------------------------------------


def _write_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def write_cohort(manifest, exclusions, sealed=None, out_dir=ROOT / "cohort-runs" / "retrain-graphs"):
    """Write the manifest, private exclusion ledger and optional sealed test file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "graphs-manifest.json"
    exclusions_path = out_dir / "exclusions-private.json"
    sealed_path = out_dir / "test-sealed.json"
    for path in (manifest_path, exclusions_path):
        if path.exists():
            raise FileExistsError(f"frozen output already exists: {path}")
    if sealed is not None and sealed_path.exists():
        raise FileExistsError(f"frozen output already exists: {sealed_path}")
    _write_json_exclusive(manifest_path, manifest)
    _write_json_exclusive(exclusions_path, exclusions)
    if sealed is not None:
        _write_json_exclusive(sealed_path, sealed)
    return {"manifest": manifest_path, "exclusions": exclusions_path, "sealed_test": sealed_path if sealed else None}


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_exclusions(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_sealed(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unseal_test(manifest, sealed, acknowledge_test_open=False):
    """Return the frozen test families after explicit post-freeze acknowledgement."""
    if acknowledge_test_open is not True:
        raise CohortAuditError("unsealing the final test cohort requires acknowledge_test_open=True")
    commitment = manifest.get("test_commitment", {})
    if commitment.get("status") != "sealed":
        raise CohortAuditError("manifest does not carry a sealed test payload")
    _check_digest(sealed, kind="sealed test payload")
    if sealed.get("content_sha256") != commitment.get("payload_sha256"):
        raise CohortAuditError("sealed test payload hash does not match the manifest commitment")
    if [graph.get("id") for graph in sealed.get("graphs", [])] != commitment.get("graph_ids"):
        raise CohortAuditError("sealed test family IDs do not match the commitment")
    graphs = [_validate_family_graph(graph) for graph in sealed["graphs"]]
    return {
        "graphs": graphs,
        "sample_cue_maps": sealed.get("sample_cue_maps", {}),
        "commitment_sha256": commitment.get("payload_sha256"),
        "warning": "test families are now open; freeze the checkpoint and record this action",
    }


def _inventory_fingerprint(payload):
    """Canonical digest of the historic inventory's *evidence*, not its seal.

    Source-specification metadata may legitimately differ between a pinned
    recheck and the original glob expansion, but the resolved source ledger,
    record counts, graph classes and cue assignments may not.
    """
    return canonical_hash({
        "sources": payload.get("sources"),
        "empty_inventory": payload.get("empty_inventory", False),
        "records_scanned": payload.get("records_scanned"),
        "records_17_scanned": payload.get("records_17_scanned"),
        "distinct_full_classes": payload.get("distinct_full_classes"),
        "distinct_backbone_classes": payload.get("distinct_backbone_classes"),
        "graphs": payload.get("graphs"),
        "assignments": payload.get("assignments"),
    })


def recheck_history(manifest, exclusions, root=ROOT, sources=None, cohort_dir=None,
                    protected_sources=None):
    """Recompute the historic ledger and require the same inventory.

    The original source *specification* is frozen inside the sealed exclusion
    payload, so this re-expands the same explicit paths and globs.  That means:

    * a newly added historic file under a scanned glob changes the inventory and
      fails closed;
    * a modified or removed historic source changes provenance and fails closed;
    * only the cohort's own recorded output files (the manifest, private ledger,
      sealed test payload and prepare report written by its own ``prepare``) are
      withheld from the scan, which keeps prepare -> audit stable.

    ``sources`` may pin an explicit list for a caller that cannot tolerate a
    live worktree changing mid-test; the evidence fingerprint is still compared.
    ``cohort_dir`` binds any recorded self-output exclusion to the directory the
    ledger was loaded from, so it can never be repointed at another experiment.
    """
    try:
        _check_digest(exclusions, kind="exclusion payload")
    except CohortAuditError as exc:
        raise ExclusionAuditError(str(exc)) from exc
    if exclusions.get("schema") != EXCLUSIONS_SCHEMA:
        raise ExclusionAuditError("exclusion payload schema mismatch")
    if sources is None:
        specs = exclusions.get("source_specs")
        if not specs:
            # Legacy ledger written before the specification was frozen: fall
            # back to its resolved paths.  It cannot detect newly added glob
            # matches, which is why every new cohort records source_specs.
            specs = [
                {"path": source["path"], "kind": source["kind"], "required": True}
                for source in exclusions.get("sources", [])
            ]
    else:
        specs = sources
    outputs = generated_output_paths(exclusions, root=root, cohort_dir=cohort_dir)
    current = load_historic_exclusions(
        root=root, sources=specs, exclude_generated=outputs,
        protected_sources=protected_sources,
    )
    if _inventory_fingerprint(current) != _inventory_fingerprint(exclusions):
        raise ExclusionAuditError(
            "historic inventory changed since the cohort was frozen; regenerate the cohort"
        )
    ledger = manifest.get("exclusion_ledger", {})
    if ledger.get("content_sha256") != exclusions.get("content_sha256"):
        raise ExclusionAuditError("manifest exclusion ledger disagrees with rechecked history")
    return {
        "status": "passed",
        "content_sha256": current["content_sha256"],
        "inventory_fingerprint": _inventory_fingerprint(current),
        "generated_outputs_excluded": [str(path) for path in outputs],
        "distinct_full_classes": current["distinct_full_classes"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_sources(args):
    """Translate ``--source`` arguments into source specifications."""
    specs = []
    for item in args.source or ():
        path = Path(item)
        if path.suffix == ".toml":
            specs.append({"path": str(item), "kind": "toml-auto", "required": True})
        else:
            specs.append({"path": str(item), "kind": "json", "required": True})
    return tuple(specs)


def _prepare(args):
    out_dir = args.out
    # Freeze the source inventory at the start of preparation, withholding only
    # the files this invocation is about to write into its own output
    # directory.  Prepared cohorts therefore remain re-verifiable even when the
    # output directory lives under a scanned glob.
    self_outputs = cohort_output_paths(out_dir, sealed_test=not args.unsealed_test)
    exclusions = load_historic_exclusions(
        root=args.root, sources=_cli_sources(args), exclude_generated=self_outputs
    )
    manifest, sealed = generate_cohort(
        seed=args.seed,
        counts={"train": args.train, "validation": args.validation, "test": args.test},
        max_candidates=args.max_candidates,
        exclusions=exclusions,
        sealed_test=not args.unsealed_test,
        root=args.root,
    )
    if exclusions["content_sha256"] != manifest["exclusion_ledger"]["content_sha256"]:
        raise ExclusionAuditError("exclusion ledger changed between generation and write")
    audit = audit_cohort(manifest, exclusions=exclusions, sealed=sealed, allow_test=True, root=args.root)
    paths = write_cohort(manifest, exclusions, sealed=sealed, out_dir=out_dir)
    # Re-read the written bytes and audit the persisted artifacts.  The prepare
    # report records the structural test check but no model score.
    persisted_manifest = load_manifest(paths["manifest"])
    persisted_exclusions = load_exclusions(paths["exclusions"])
    persisted_sealed = load_sealed(paths["sealed_test"]) if paths["sealed_test"] else None
    persisted_audit = audit_cohort(
        persisted_manifest,
        exclusions=persisted_exclusions,
        sealed=persisted_sealed,
        allow_test=True,
        root=args.root,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "prepared_before_model_fitting",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_id": manifest["cohort_id"],
        "seed": args.seed,
        "counts": {"train": args.train, "validation": args.validation, "test": args.test},
        "manifest_sha256": sha256_file(paths["manifest"]),
        "exclusions_sha256": sha256_file(paths["exclusions"]),
        "sealed_test_sha256": sha256_file(paths["sealed_test"]) if paths["sealed_test"] else None,
        "audit_at_freeze": audit,
        "audit_persisted_artifacts": persisted_audit,
        "model_optimizer_updates": 0,
        "model_or_readout_fitting": False,
        "final_test_scored": False,
        "test_payload_status": manifest["test_commitment"]["status"],
        "command_argv": list(getattr(sys, "orig_argv", [sys.executable, *sys.argv])),
        "working_directory": str(Path.cwd()),
    }
    _write_json_exclusive(Path(out_dir) / "prepare-report.json", report)
    print(json.dumps({
        "status": "prepared",
        "cohort_id": report["cohort_id"],
        "families": persisted_audit["families"],
        "test_payload_status": report["test_payload_status"],
        "manifest": str(paths["manifest"]),
        "model_optimizer_updates": 0,
    }))
    return report


def _capacity(args):
    """Print the exact grammar ceiling against the supplied or frozen ledger."""
    if args.exclusions is not None:
        exclusions = load_exclusions(args.exclusions)
    else:
        exclusions = load_historic_exclusions(root=args.root, sources=_cli_sources(args))
    space = backbone_space_index()
    requested = {"train": args.train, "validation": args.validation, "test": args.test}
    report = topology_capacity(exclusions, requested=requested, space=space)
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    report["constructive"] = (
        constructive_capacity(exclusions, space=space) if args.constructive else None
    )
    report["relaxed_certificate"] = (
        relaxed_backbone_certificate(exclusions, requested=requested, space=space)
        if args.relaxed_certificate else None
    )
    if args.out is not None:
        _write_json_exclusive(args.out, report)
    print(json.dumps(report))
    return report


def _audit(args):
    manifest = load_manifest(args.manifest)
    directory = Path(args.manifest).resolve().parent
    exclusions = load_exclusions(args.exclusions or directory / "exclusions-private.json")
    sealed = None
    if manifest.get("test_commitment", {}).get("status") == "sealed":
        sealed_path = args.sealed_test or directory / "test-sealed.json"
        sealed = load_sealed(sealed_path)
    report = audit_cohort(manifest, exclusions=exclusions, sealed=sealed,
                          allow_test=args.allow_test, root=args.root)
    if args.recheck_history:
        report["history_recheck"] = recheck_history(
            manifest, exclusions, root=args.root, cohort_dir=directory
        )
    print(json.dumps(report))
    return report


def _unseal(args):
    manifest = load_manifest(args.manifest)
    sealed = load_sealed(args.sealed_test)
    result = unseal_test(manifest, sealed, acknowledge_test_open=args.acknowledge_test_open)
    if args.out:
        out = Path(args.out)
        _write_json_exclusive(out, _seal({
            "schema": "tem.multitopology.retrain.test-opened.v1",
            "cohort_id": manifest.get("cohort_id"),
            "commitment_sha256": result["commitment_sha256"],
            "graphs": result["graphs"],
            "sample_cue_maps": result["sample_cue_maps"],
            "warning": result["warning"],
        }))
    print(json.dumps({
        "status": "opened",
        "families": len(result["graphs"]),
        "commitment_sha256": result["commitment_sha256"],
        "warning": result["warning"],
    }))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root for historic sources")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="freeze a fresh cohort before any model fitting")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--train", type=int, default=96)
    prepare.add_argument("--validation", type=int, default=24)
    prepare.add_argument("--test", type=int, default=24)
    prepare.add_argument("--max-candidates", type=int, default=200000)
    prepare.add_argument(
        "--source", action="append", default=[],
        help="historic graph/cue source (JSON or TOML), relative to --root; repeatable",
    )
    prepare.add_argument("--unsealed-test", action="store_true",
                         help="write final test payloads into the public manifest (not recommended)")

    capacity = sub.add_parser(
        "capacity",
        help="exact exhaustive topology capacity; no fitting, scoring or test opening",
    )
    capacity.add_argument("--out", type=Path)
    capacity.add_argument("--exclusions", type=Path,
                          help="frozen exclusion ledger; recomputed from --root when absent")
    capacity.add_argument("--source", action="append", default=[],
                          help="historic source used when --exclusions is absent; repeatable")
    capacity.add_argument("--train", type=int, default=DEFAULT_COUNTS["train"])
    capacity.add_argument("--validation", type=int, default=DEFAULT_COUNTS["validation"])
    capacity.add_argument("--test", type=int, default=DEFAULT_COUNTS["test"])
    capacity.add_argument("--constructive", action="store_true",
                          help="certify that every fresh backbone supplies one family")
    capacity.add_argument("--relaxed-certificate", action="store_true",
                          help="diagnostic only: capacity if intra-cohort backbone reuse were approved")

    audit = sub.add_parser("audit", help="audit a frozen cohort without fitting or scoring")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--exclusions", type=Path)
    audit.add_argument("--sealed-test", type=Path)
    audit.add_argument("--allow-test", action="store_true",
                       help="explicitly open the frozen test payload for structural validation")
    audit.add_argument("--recheck-history", action="store_true",
                       help="recompute the historic inventory and require an exact ledger match")

    unseal = sub.add_parser("unseal", help="open the frozen test cohort after checkpoint freeze and QA")
    unseal.add_argument("--manifest", type=Path, required=True)
    unseal.add_argument("--sealed-test", type=Path, required=True)
    unseal.add_argument("--out", type=Path)
    unseal.add_argument("--acknowledge-test-open", action="store_true", required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "audit":
        _audit(args)
    elif args.command == "capacity":
        _capacity(args)
    else:
        _unseal(args)


if __name__ == "__main__":
    main()
