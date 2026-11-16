"""Synthetic multi-topology graph cohorts; no model inputs or model scores.

Full graphs and *unrooted unlabeled* backbones are excluded independently.
Cross links join equal-depth nodes, a privileged authored port grammar.
The shared 45-cue vocabulary is partitioned by whole assignment, not by cue.

This is the public, generic generator. It never reads historic experiment
files, private ledgers or sealed payloads. A caller that needs freshness
against a prior cohort supplies its own reference graph and reserved cue
assignments through ``generate_manifest(reference_graph=...,
reserved_assignments=...)``; without them the generated cohort is disclosed as
having an empty historic exclusion set.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "tem.multitopology.graphs.v1"
SPLITS = ("train", "validation", "test")
CHILD = (2, 3, 5)
CUE_VOCABULARY = 45


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize_graph(graph):
    result = copy.deepcopy(graph)
    result["transitions"] = [{int(a): v for a, v in row.items()} for row in result["transitions"]]
    return result


def graph_nx(graph, tree_only=False, action_labeled=False):
    graph = normalize_graph(graph)
    result = nx.DiGraph() if action_labeled else nx.Graph()
    result.add_nodes_from(range(len(graph["names"])))
    for u, moves in enumerate(graph["transitions"]):
        for action, v in moves.items():
            if not tree_only or action in CHILD:
                result.add_edge(u, v, action=action)
    return result


def validate_graph(graph):
    graph = normalize_graph(graph)
    if len(graph["names"]) != 17 or len(set(graph["names"])) != 17 or len(graph["transitions"]) != 17:
        raise ValueError("expected 17 unique nodes")
    root = graph["root"]
    if type(root) is not int or root not in range(17):
        raise ValueError("invalid root")
    expected, tree_edges, crosses = set(), set(), set()
    for u, moves in enumerate(graph["transitions"]):
        if len(moves) > 5:
            raise ValueError("port capacity exceeded")
        for action, v in moves.items():
            if action not in (1, 2, 3, 4, 5) or type(v) is not int or v not in range(17) or v == u:
                raise ValueError("invalid transition")
            reverse = graph["transitions"][v]
            edge = tuple(sorted((u, v)))
            expected.add(edge)
            if action in CHILD:
                if reverse.get(1) != u:
                    raise ValueError("nonreciprocal child")
                tree_edges.add(edge)
            elif action == 1:
                if sum(a in CHILD and target == u for a, target in reverse.items()) != 1:
                    raise ValueError("nonreciprocal parent")
            elif reverse.get(4) != u:
                raise ValueError("nonreciprocal cross")
            else:
                crosses.add(edge)
        if u == root and 1 in moves or u != root and 1 not in moves:
            raise ValueError("root/parent role mismatch")
        if len(set(moves.values())) != len(moves):
            raise ValueError("multiple ports to same neighbor")
    recorded = [tuple(sorted(edge)) for edge in graph["edges"]]
    if len(recorded) != 23 or len(set(recorded)) != 23 or set(recorded) != expected:
        raise ValueError("expected exactly 23 unique consistent edges")
    if len(tree_edges) != 16 or len(crosses) != 7 or tree_edges & crosses:
        raise ValueError("expected 16 backbone plus 7 disjoint cross edges")
    tree = graph_nx(graph, True)
    if not nx.is_tree(tree):
        raise ValueError("backbone is not a tree")
    depth = nx.single_source_shortest_path_length(tree, root)
    if max(depth.values()) != 3 or [depth[i] for i in range(17)] != graph["depths"]:
        raise ValueError("expected truthful four-layer backbone depths")
    if any(depth[u] != depth[v] for u, v in crosses):
        raise ValueError("cross links must obey disclosed same-depth grammar")
    if nx.single_source_shortest_path_length(graph_nx(graph), root) != depth:
        raise ValueError("backbone must preserve whole-graph root depth")
    return graph


def backbone_key(graph):
    """AHU canonical string for an unrooted tree; root/IDs/names ignored."""
    tree = graph_nx(graph, True)
    if not nx.is_tree(tree):
        raise ValueError("not a tree")

    def rooted(u, parent):
        return "(" + "".join(sorted(rooted(v, u) for v in tree[u] if v != parent)) + ")"
    return min(rooted(center, None) for center in nx.center(tree))


def _signature(graph):
    full = graph_nx(graph)
    return tuple(sorted(dict(full.degree()).values())), nx.weisfeiler_lehman_graph_hash(full)


def _candidate(rng):
    children, depths = [[]], [0]
    for child in range(1, 17):
        possible = [u for u in range(child) if depths[u] < 3 and len(children[u]) < 3]
        if not possible:
            return None
        parent = rng.choice(possible)
        children[parent].append(child)
        children.append([])
        depths.append(depths[parent] + 1)
    if max(depths) != 3:
        return None
    matching = []
    for depth in (1, 2, 3):
        nodes = [u for u, d in enumerate(depths) if d == depth]
        rng.shuffle(nodes)
        matching += [sorted(nodes[i:i + 2]) for i in range(0, len(nodes) - 1, 2)]
    if len(matching) < 7:
        return None
    rng.shuffle(matching)
    transitions = [{} for _ in children]
    edges = []
    for u, branches in enumerate(children):
        for action, v in zip(CHILD, branches):
            transitions[u][action], transitions[v][1] = v, u
            edges.append(sorted((u, v)))
    for u, v in matching[:7]:
        transitions[u][4], transitions[v][4] = v, u
        edges.append([u, v])
    return validate_graph({"root": 0, "names": [f"space-{u}" for u in range(17)],
                           "depths": depths, "edges": sorted(edges), "transitions": transitions,
                           "provenance": {"kind": "authored synthetic", "not_a_real_building": True,
                                          "cross_grammar": "same backbone depth, reciprocal single cross port"}})


def assignment_namespace(assignment):
    return int(canonical_hash(list(assignment)), 16) % len(SPLITS)


def cue_assignment(graph, split, episode, seed, reserved=()):
    """Independent remap per episode, deterministic and disjoint across splits.

    Ignoring graph identity pairs cue streams between matched arms. The
    SHA-partition rejection changes assignment support and is disclosed.
    """
    if split not in SPLITS or len(graph["names"]) != 17:
        raise ValueError("expected valid split and 17-node graph")
    reservations = {tuple(a) for a in reserved}
    rng = random.Random(f"multitopology-cues-v1:{seed}:{split}:{episode}")
    for _ in range(10000):
        assignment = tuple(rng.sample(range(45), 17))
        if assignment_namespace(assignment) == SPLITS.index(split) and assignment not in reservations:
            return assignment
    raise RuntimeError("cue rejection sampler exhausted 10000 attempts")


def _empty_exclusions():
    """Empty public exclusion set: no historic graph or cue is assumed.

    The manifest still records ``exclusion_sources == []`` so a downstream
    freshness audit can tell a disclosed empty history apart from one that
    silently dropped prior cohorts.
    """
    return [], [], []


def _content_digest(manifest):
    return canonical_hash({k: v for k, v in manifest.items() if k != "content_sha256"})


def generate_manifest(seed=0, train=96, validation=24, test=24,
                      max_candidates=200000, reference_graph=None, reserved_assignments=()):
    """Generate a frozen synthetic cohort manifest.

    ``reference_graph`` and ``reserved_assignments`` are the caller's only way
    to exclude prior historic families and cue maps; this public package never
    reads private experiment files. With ``reference_graph=None`` the exclusion
    set is empty and the manifest discloses that fact.
    """
    counts = dict(zip(SPLITS, (train, validation, test)))
    if any(type(n) is not int or n < 0 for n in counts.values()) or sum(counts.values()) < 1:
        raise ValueError("positive total integer family counts required")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("positive candidate bound required")
    if reference_graph is None:
        excluded, prior_reserved, sources = _empty_exclusions()
    else:
        excluded, prior_reserved, sources = [normalize_graph(reference_graph)], [], []
    reservations = sorted({tuple(a) for a in [*prior_reserved, *reserved_assignments]})
    seen_tree = {backbone_key(g) for g in excluded}
    bins = {}
    for g in excluded:
        bins.setdefault(_signature(g), []).append(graph_nx(g))
    graphs, rejected, rng = [], Counter(), random.Random(seed)
    requested = [(split, i) for split in SPLITS for i in range(counts[split])]
    attempts = 0
    for attempts in range(1, max_candidates + 1):
        graph = _candidate(rng)
        if graph is None:
            rejected["structural_ineligible"] += 1
            continue
        tree_key = backbone_key(graph)
        if tree_key in seen_tree:
            rejected["backbone_isomorphic"] += 1
            continue
        signature = _signature(graph)
        full = graph_nx(graph)
        if any(nx.is_isomorphic(full, old) for old in bins.get(signature, [])):
            rejected["full_isomorphic"] += 1
            continue
        split, i = requested[len(graphs)]
        graph.update(id=f"multitopology-{split}-{i:03d}", split=split)
        graphs.append(graph)
        seen_tree.add(tree_key)
        bins.setdefault(signature, []).append(full)
        if len(graphs) == len(requested):
            break
    if len(graphs) != len(requested):
        raise RuntimeError(json.dumps({"status": "infeasible_within_candidate_bound",
                                       "requested": counts, "eligible_distinct_families_found": len(graphs),
                                       "candidate_count": attempts, "rejections": dict(rejected),
                                       "exhaustive_impossibility_claimed": False}))
    manifest = {"schema": SCHEMA, "seed": seed, "graphs": graphs,
                "splits": {split: [g["id"] for g in graphs if g["split"] == split] for split in SPLITS},
                "excluded_graphs": excluded, "reserved_assignments": [list(a) for a in reservations],
                "exclusion_sources": sources,
                "cue_contract": {"vocabulary_size": 45, "unique_cues_per_episode": 17,
                                 "assignment_partition": "sha256(canonical JSON ordered tuple) modulo 3: train=0 validation=1 test=2",
                                 "seed_namespace": "multitopology-cues-v1", "graph_independent_for_arm_parity": True},
                "generation": {"requested": counts, "candidate_count": attempts, "max_candidates": max_candidates,
                               "rejections": dict(rejected), "full_and_unrooted_backbone_exclusion": True,
                               "same_depth_cross_grammar": True, "selection_uses_model_scores": False},
                "sample_cue_maps": {g["id"]: [list(cue_assignment(g, g["split"], i, seed, reservations))
                                              for i in range(2)] for g in graphs}}
    manifest["content_sha256"] = _content_digest(manifest)
    audit_manifest(manifest)
    return manifest


def audit_manifest(manifest):
    if manifest.get("schema") != SCHEMA or manifest.get("content_sha256") != _content_digest(manifest):
        raise ValueError("manifest schema/content digest mismatch")
    graphs = [validate_graph(g) for g in manifest["graphs"]]
    ids = [g["id"] for g in graphs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate graph IDs")
    for split in SPLITS:
        expected = [g["id"] for g in graphs if g["split"] == split]
        if expected != manifest["splits"][split] or len(expected) != manifest["generation"]["requested"][split]:
            raise ValueError("split membership/count mismatch")
    if any(g["split"] not in SPLITS for g in graphs):
        raise ValueError("invalid graph split")
    # Exact tree canonicalization and VF2 full-isomorphism verification, not WL alone.
    trees, bins, actions = set(), {}, []
    for g in manifest["excluded_graphs"]:
        trees.add(backbone_key(g))
        bins.setdefault(_signature(g), []).append(graph_nx(g))
        actions.append(graph_nx(g, action_labeled=True))
    comparisons = 0
    for graph in graphs:
        key = backbone_key(graph)
        if key in trees:
            raise ValueError("backbone isomorphism overlap")
        trees.add(key)
        sig, full = _signature(graph), graph_nx(graph)
        for old in bins.get(sig, []):
            comparisons += 1
            if nx.is_isomorphic(full, old):
                raise ValueError("full graph isomorphism overlap")
        bins.setdefault(sig, []).append(full)
        directed = graph_nx(graph, action_labeled=True)
        for old in actions:
            if sorted(dict(directed.degree()).values()) == sorted(dict(old.degree()).values()):
                if nx.is_isomorphic(directed, old, edge_match=nx.algorithms.isomorphism.categorical_edge_match("action", -1)):
                    raise ValueError("action-labelled equivalence overlap")
        actions.append(directed)
    reserved = {tuple(a) for a in manifest["reserved_assignments"]}
    sample_by_split = {split: set() for split in SPLITS}
    for graph in graphs:
        for a in manifest["sample_cue_maps"][graph["id"]]:
            if len(a) != 17 or len(set(a)) != 17 or any(type(x) is not int or x not in range(45) for x in a):
                raise ValueError("invalid cue assignment")
            if tuple(a) in reserved or assignment_namespace(a) != SPLITS.index(graph["split"]):
                raise ValueError("cue reservation or split overlap")
            sample_by_split[graph["split"]].add(tuple(a))
    for i, split in enumerate(SPLITS):
        if any(sample_by_split[split] & sample_by_split[other] for other in SPLITS[i + 1:]):
            raise ValueError("cue split collision")
    return {"status": "passed", "families": {s: len(manifest["splits"][s]) for s in SPLITS},
            "full_isomorphism_exact_comparisons": comparisons, "backbone_exact_canonical": True,
            "action_labelled_equivalence": "explicit directed VF2 audit passed",
            "excluded_graphs": len(manifest["excluded_graphs"]), "reserved_assignments": len(reserved),
            "content_sha256": manifest["content_sha256"], "model_optimizer_updates": 0}


def load_manifest(path, audit=True):
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if audit:
        audit_manifest(manifest)
    manifest["graphs"] = [normalize_graph(g) for g in manifest["graphs"]]
    manifest["excluded_graphs"] = [normalize_graph(g) for g in manifest["excluded_graphs"]]
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=0)
    for split, default in zip(SPLITS, (96, 24, 24)):
        prep.add_argument(f"--{split}", type=int, default=default)
    prep.add_argument("--max-candidates", type=int, default=200000)
    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.out.exists():
            raise FileExistsError("frozen output already exists")
        manifest = generate_manifest(args.seed, args.train, args.validation, args.test, args.max_candidates)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
        print(json.dumps(audit_manifest(manifest)))
    else:
        print(json.dumps(audit_manifest(load_manifest(args.manifest, audit=False))))


if __name__ == "__main__":
    main()
