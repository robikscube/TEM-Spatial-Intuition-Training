"""Score strict novel-edge returns in saved TEM evaluation traces."""

import argparse
import json
from pathlib import Path


ACTION_LABELS = {"0": "stay", "1": "N", "2": "E", "3": "S", "4": "W"}
OFFSETS = {0: (0, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}
METHODS = ("tem", "observed_transition", "supplied_compass")


def summarize(outcomes):
    eligible = len(outcomes)
    predicted = sum(value is not None for value in outcomes)
    hits = sum(value is True for value in outcomes)
    return {
        "eligible": eligible,
        "predicted": predicted,
        "hits": hits,
        "coverage": predicted / eligible if eligible else None,
        "top1_on_predictions": hits / predicted if predicted else None,
    }


def strict_returns(trace):
    if trace.get("schema") != "tem.evaluation.v1" or trace.get("mode") != "trained_tem":
        raise ValueError("expected a trained TEM evaluation trace")
    if trace.get("integrity", {}).get("predictions_use_strict_prefix") is not True:
        raise ValueError("trace does not declare strict-prefix predictions")
    if trace["environment"]["action_ids"] != ACTION_LABELS:
        raise ValueError("expected stay/N/E/S/W action encoding")
    first, *steps = trace["steps"]
    if first["t"] != 0 or first["prediction"] is not None:
        raise ValueError("initial observation must have no prediction")

    returns = {method: [] for method in METHODS}
    previous_node = first["node"]
    previous_cue = first["observation"]["sensor_id"]
    seen = {previous_node}
    crossed = set()
    transitions = {}
    position = (0, 0)
    cues_at_position = {position: previous_cue}

    for index, step in enumerate(steps, start=1):
        if step["t"] != index:
            raise ValueError("trace steps must be consecutive")
        node = step["node"]
        action = step["action_from_previous_id"]
        if (action == 0) != (previous_node == node):
            raise ValueError(f"step {index} is inconsistent with the stay action")
        dx, dy = OFFSETS[action]
        position = (position[0] + dx, position[1] + dy)
        edge = frozenset((previous_node, node))
        transition = (previous_cue, action)
        if action and node in seen and edge not in crossed:
            guesses = (
                step["prediction"]["top1_sensor"],
                transitions.get(transition),
                cues_at_position.get(position),
            )
            actual = step["observation"]["sensor_id"]
            for method, guess in zip(METHODS, guesses):
                returns[method].append(None if guess is None else guess == actual)

        # Reveal the arrival cue only after the baselines predict.
        previous_cue = step["observation"]["sensor_id"]
        transitions[transition] = previous_cue
        cues_at_position[position] = previous_cue
        seen.add(node)
        if action:
            crossed.add(edge)
        previous_node = node
    return returns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path, help="saved evaluation JSON files")
    args = parser.parse_args()
    per_trace = []
    pooled = {method: [] for method in METHODS}
    for path in args.traces:
        try:
            trace = json.loads(path.read_text(encoding="utf-8-sig"))
            outcomes = strict_returns(trace)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"{path}: {exc}")
        per_trace.append({
            "trace": str(path),
            "world": trace.get("world"),
            **{method: summarize(outcomes[method]) for method in METHODS},
        })
        for method in METHODS:
            pooled[method].extend(outcomes[method])
    print(json.dumps({
        "criterion": "seen destination, non-stay, passage never crossed in either direction",
        "traces": per_trace,
        "pooled": {method: summarize(pooled[method]) for method in METHODS},
    }, indent=2))


if __name__ == "__main__":
    main()
