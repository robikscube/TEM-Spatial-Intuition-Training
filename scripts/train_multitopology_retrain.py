"""Full-budget native TEM retraining for a multi-topology study arm.

This adapter leaves the upstream ``torch_tem`` checkout untouched. It is a
training path for native, labelled passage graphs and implements an explicit
scientific-gate contract:

* a minimum number of **real** optimizer updates per arm and independent seed,
  taken from the declared scientific stage (``standard`` or ``interim``);
* at least ``48`` observations of continuous within-episode ``g``, ``x`` and
  Hebbian ``M`` state, with an explicit reset only between episodes;
* full-episode backpropagation through time by default, so credit is assigned
  across the whole episode rather than truncated at each rollout;
* CUDA-on-Linux (Docker) for a scientific fit; native-Windows CUDA is refused;
* the upstream parameter schedule for a scientific fit, not a compressed pilot
  ramp;
* paired ``single`` versus ``diverse`` arms with identical model
  initialisation, cue maps, update count, batch size, episode length and
  observation budget per seed;
* declared checkpoints with atomic writes, optimizer state, recurrent carry and
  RNG state for interrupt resume.

The ``interim`` stage is a declared, smaller preliminary budget. It is a
distinct stage rather than ``scientific_mode = False`` so it cannot silently
bypass the gate: it is labelled ``scientific_interim``, is explicitly
preliminary, and can never set ``scientific_fit`` or any final capability claim.
The ``standard`` stage is the frozen confirmation contract and is the only
stage that can.

**Cohorts are caller-provided public manifests.** This module never reads a
private historic ledger, never opens a sealed test payload and never imports a
private cohort provider. A manifest that declares
``generation.cohort_kind`` is refused explicitly rather than audited with a
weaker public check.

Training routes are an explicit, declared policy. ``legal_walk`` is the frozen
historical default (uniform legal transitions) and is what an omitted
``route_policy`` means. ``reciprocal_return`` is a prospective
P1-compatible controller that builds, from public transitions only, a
long-delay action-4 reciprocal return with a distractor segment; see
``route_coverage`` for the measured distribution difference. Selecting it is a
prospective task declaration, not a post-hoc threshold change.

What this module does **not** do: open a sealed final holdout, score acquisition
or transfer, implement partial spatial inference, or claim
convergence/PASS. A completed fit is a prerequisite for an independent
reviewer, not evidence of one.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback
import tomllib

import access_graph_tem as native
import access_graph_worlds as worlds
import tem_trained_adapter as adapter
from train_tem import _plain_hyper

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "tem.multitopology.retrain.config.v1"
PREPARED_SCHEMA = "tem.multitopology.retrain.prepared.v1"
SUMMARY_SCHEMA = "tem.multitopology.retrain.summary.v1"
STATE_SCHEMA = "tem.multitopology.retrain.state.v1"
ARMS = ("single", "diverse")
BPTT_MODES = ("episode", "rollout")
DECLARED_CHECKPOINTS = (1000, 2500, 5000, 7500, 10000)
FULL_HISTORY_MIN = 48
FULL_UPDATES_MIN = 10000
# --- declared scientific stages ---------------------------------------------
# "standard" is the frozen confirmation contract above. "interim" is a declared
# preliminary budget: a smaller, exact update count and checkpoint list, with
# every other integrity condition intact.
STANDARD_STAGE = "standard"
INTERIM_STAGE = "interim"
SCIENTIFIC_STAGES = (STANDARD_STAGE, INTERIM_STAGE)
INTERIM_UPDATES = 5000
INTERIM_CHECKPOINTS = (1000, 2500, 5000)
STAGE_MIN_UPDATES = {STANDARD_STAGE: FULL_UPDATES_MIN, INTERIM_STAGE: INTERIM_UPDATES}
STAGE_CHECKPOINTS = {
    STANDARD_STAGE: DECLARED_CHECKPOINTS,
    INTERIM_STAGE: INTERIM_CHECKPOINTS,
}
SCIENTIFIC_STATUS = {
    STANDARD_STAGE: "scientific_standard",
    INTERIM_STAGE: "scientific_interim",
}
NON_SCIENTIFIC_STATUS = "non_scientific"
CONTINUATION_STATUS = "scientific_continuation"
MIN_PAIRED_SEEDS = 3
# Declared training-route policies. ``legal_walk`` stays the default so every
# frozen config keeps its historical data distribution; ``reciprocal_return`` is
# a prospective P1-compatible controller.
ROUTE_POLICIES = ("legal_walk", "reciprocal_return")
DEFAULT_ROUTE_POLICY = "legal_walk"
# P1 delay strata. The PRIMARY v2 acceptance stratum is >=24 observations since
# the last target encounter; 12-23 is a separate reported stratum.
P1_MIN_RETURN_DELAY = 24
P1_SHORT_STRATUM = (12, 23)
P1_MAX_RETURN_DELAY = 47  # search ceiling for an exact episode length
# Settings a continuation run may change; everything else must be identical.
CONTINUATION_MUTABLE_KEYS = frozenset({"iterations", "save_updates"})
# The only permitted stage transition: a completed interim endpoint extended
# into the standard confirmation budget. The reverse or any other pair is a
# different experiment, not a continuation.
CONTINUATION_STAGE_TRANSITION = (INTERIM_STAGE, STANDARD_STAGE)
CLAIMS = {
    "convergence": False,
    "acquisition_gate": "not evaluated by this trainer",
    "transfer_gate": "not evaluated by this trainer",
    "partial_inference": "not implemented",
    "completed_fit_is_scientific_pass": False,
    "final_capability_pass": False,
    "confirmation_program": "separate frozen program; a preliminary fit never satisfies it",
}
# The order is deterministic and is hashed into provenance.
CODE = (
    "scripts/train_multitopology_retrain.py",
    "scripts/multitopology_retrain_graphs.py",
    "scripts/multitopology_graphs.py",
    "scripts/access_graph_tem.py",
    "scripts/access_graph_worlds.py",
    "scripts/tem_trained_adapter.py",
    "scripts/train_tem.py",
    "docker/upstream.lock.json",
    "docker/pyproject.toml",
    "docker/uv.lock",
)
COHORT_SCHEMA = "tem.multitopology.retrain.graphs.v1"
EXCLUSIONS_SCHEMA = "tem.multitopology.retrain.exclusions.v1"
RESERVATIONS_SCHEMA = "tem.multitopology.retrain.reservations.v1"
TRAINING_KEYS = {
    "iterations", "batch_size", "episode_length", "rollout", "bptt",
    "schedule", "device", "scientific_mode", "seed", "cue_seed",
    "route_seed", "family_seed", "threads", "max_seconds", "save_updates",
    "log_every", "vram_budget_gib",
    # Optional and defaulted to "standard", so a frozen config that omits it
    # keeps its exact key set and its exact config digest.
    "scientific_stage",
    # Optional and defaulted to "legal_walk", i.e. the historical uniform
    # legal-transition route, so frozen configs keep their exact data stream.
    "route_policy",
}
REQUIRED_TRAINING_KEYS = TRAINING_KEYS - {"scientific_stage", "route_policy"}


class RefusalError(RuntimeError):
    """A scientific fit was requested under conditions the gate forbids."""


class ZeroScoredUpdate(RuntimeError):
    """The update had no eligible visited-location prediction and must not count."""


# --------------------------------------------------------------------- helpers


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: str | Path) -> str:
    return adapter.sha256_file(Path(path))


def write_json(path: str | Path, value) -> None:
    adapter.write_json_atomic(Path(path), value)


def graph_api():
    """The retrain cohort generator module."""
    import multitopology_retrain_graphs
    return multitopology_retrain_graphs


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _content_digest(value: dict) -> str:
    return _canonical_hash({key: item for key, item in value.items() if key != "content_sha256"})


PUBLIC_COHORT_KEYS = {
    "schema", "cohort_id", "seed", "structure", "graphs", "families", "splits",
    "sample_cue_maps", "cue_contract", "exclusion_ledger", "test_commitment",
    "generation", "content_sha256",
}


def _cohort_kind(manifest: dict) -> str | None:
    generation = manifest.get("generation")
    return generation.get("cohort_kind") if isinstance(generation, dict) else None


def _require_public_cohort(manifest: dict) -> None:
    """Refuse a cohort that claims a private, separately audited provenance.

    An unrecognised ``generation.cohort_kind`` marker means the cohort was
    produced by a provider with its own audit contract. This public package has
    no such provider, so it fails closed instead of substituting a weaker check.
    """
    kind = _cohort_kind(manifest)
    if kind is not None:
        raise ValueError(
            f"cohort declares generation.cohort_kind={kind!r}; cohorts with a "
            "provider-specific audit contract are not supported by this package. "
            "Supply an all-public ``tem.multitopology.retrain.graphs.v1`` cohort "
            "instead of relaxing the audit."
        )


def _validate_public_manifest(
    manifest: dict,
    *,
    expected_exclusions_sha256: str | None = None,
    required_train: int | None = None,
) -> dict:
    """Audit only the public manifest; never parse or hash a sealed payload."""
    if not isinstance(manifest, dict):
        raise ValueError("unexpected public cohort manifest schema")
    if manifest.get("schema") != COHORT_SCHEMA:
        raise ValueError(
            f"expected cohort schema {COHORT_SCHEMA}, got {manifest.get('schema')!r}"
        )
    if set(manifest) != PUBLIC_COHORT_KEYS:
        raise ValueError("unexpected public cohort manifest schema")
    if manifest.get("content_sha256") != _content_digest(manifest):
        raise ValueError("cohort manifest content digest mismatch")
    _require_public_cohort(manifest)
    commitment = manifest.get("test_commitment")
    if not isinstance(commitment, dict) or commitment.get("status") != "sealed":
        raise ValueError("retrain cohort must carry a sealed test commitment")
    if commitment.get("opened_for_scoring") is not False:
        raise ValueError("test commitment must not be open or already scored")
    graphs = manifest.get("graphs")
    if not isinstance(graphs, list):
        raise ValueError("cohort graphs must be a list")
    if any(graph.get("split") == "test" for graph in graphs):
        raise ValueError("sealed public manifest must not contain final test graph payloads")
    splits = manifest.get("splits", {})
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("cohort splits must be exactly train, validation and test")
    train = splits["train"]
    if not isinstance(train, list) or len(set(train)) != len(train) or not train:
        raise ValueError("retrain cohort needs a nonempty duplicate-free train split")
    if required_train is not None and len(train) != required_train:
        raise ValueError(
            f"retrain cohort requires exactly {required_train} train families, got {len(train)}"
        )
    if len(splits["test"]) != commitment.get("graph_count"):
        raise ValueError("test split count disagrees with the sealed commitment")
    if set(splits["test"]) != set(commitment.get("graph_ids", [])):
        raise ValueError("test split IDs disagree with the sealed commitment")
    payload_ids = {graph.get("id") for graph in graphs}
    if payload_ids != set(train) | set(splits["validation"]):
        raise ValueError("public payloads must cover exactly train and validation families")
    family_ids = {entry.get("family_id") for entry in manifest.get("families", [])}
    if family_ids != payload_ids:
        raise ValueError("family grouping must cover exactly the public families")
    cue_contract = manifest.get("cue_contract", {})
    if cue_contract.get("graph_independent_for_arm_parity") is not True:
        raise ValueError("cohort cue contract must guarantee arm-independent assignments")
    ledger = manifest.get("exclusion_ledger", {})
    if expected_exclusions_sha256 is not None and ledger.get("content_sha256") != expected_exclusions_sha256:
        raise ValueError("cohort exclusion ledger disagrees with the prepared exclusions")
    structure = manifest.get("structure", {})
    if structure.get("nodes") != 17 or structure.get("tree_edges") != 16 or structure.get("cross_edges") != 7:
        raise ValueError("cohort structure does not match the 17-node authored contract")
    return {
        "status": "passed",
        "schema": COHORT_SCHEMA,
        "cohort_kind": None,
        "cohort_id": manifest.get("cohort_id"),
        "seed": manifest.get("seed"),
        "families": {split: len(splits[split]) for split in ("train", "validation", "test")},
        "public_graphs": len(graphs),
        "test_payload_status": commitment["status"],
        "test_payload_sha256": commitment.get("payload_sha256"),
        "test_payload_opened": False,
        "test_payload_read": False,
        "audit_cohort": "public manifest only; sealed payload neither read nor hashed",
        "exclusion_ledger_sha256": ledger.get("content_sha256"),
        "empty_historic_inventory": bool(ledger.get("empty_inventory", False)),
        "model_optimizer_updates": 0,
    }


def _load_cohort_public(manifest_path: Path, exclusions_path: Path, *,
                        required_train: int | None = None):
    """Load the public manifest and its exclusion ledger, not the sealed test.

    The trainer never opens ``test-sealed.json``. The public prefix is audited
    with ``sealed=None`` and the documented "sealed test payload is missing"
    sentinel is the only accepted stop.
    """
    api = graph_api()
    manifest = api.load_manifest(Path(manifest_path))
    _require_public_cohort(manifest)
    exclusions = api.load_exclusions(Path(exclusions_path))
    try:
        state = api.exclusion_state(exclusions)
    except Exception as exc:  # noqa: BLE001 - generator digest/schema errors
        raise ValueError(f"exclusion ledger invalid: {type(exc).__name__}: {exc}") from exc
    audit = _validate_public_manifest(
        manifest,
        expected_exclusions_sha256=exclusions.get("content_sha256"),
        required_train=required_train,
    )
    try:
        api.audit_cohort(manifest, exclusions=exclusions, sealed=None, allow_test=False)
        audit["audit_cohort"] = "public manifest passed without a sealed payload"
    except api.CohortAuditError as exc:
        if "sealed test payload is missing" not in str(exc):
            raise ValueError(f"cohort public audit failed: {exc}") from exc
        audit["audit_cohort"] = (
            "public prefix passed; stopped at the sealed commitment and did not read the payload"
        )
    audit["sealed_payload_read"] = False
    # Digest validation and the public audit used the frozen JSON shape;
    # normalise transition keys only now, in memory, for the walker/native API.
    manifest = _normalize_cohort_graphs(manifest)
    reserved = [list(item) for item in sorted(state["assignments"])]
    if len(reserved) != exclusions.get("assignment_count"):
        raise ValueError("exclusion assignment count disagrees with the ledger")
    if manifest["exclusion_ledger"].get("content_sha256") != exclusions.get("content_sha256"):
        raise ValueError("manifest exclusion ledger disagrees with the exclusions file")
    return manifest, exclusions, reserved, audit


def _normalize_cohort_graphs(manifest: dict) -> dict:
    """Convert JSON object transition keys back to integers in memory.

    The manifest is JSON, so ``transitions`` keys arrive as strings. The digest
    is checked on the raw manifest first; this normalisation never touches the
    frozen file or its recorded content hash.
    """
    for graph in manifest.get("graphs", []):
        graph["transitions"] = [
            {int(action): int(target) for action, target in row.items()}
            for row in graph["transitions"]
        ]
    return manifest


def _reservations_payload(reserved: list[list[int]], exclusions: dict) -> dict:
    return {
        "schema": RESERVATIONS_SCHEMA,
        "count": len(reserved),
        "exclusions_sha256": exclusions["content_sha256"],
        "assignments": reserved,
    }


def _in_container() -> bool:
    """Docker marker, with an explicit equivalent-runtime override.

    The scientific gate names Docker/CUDA. Standard Docker creates
    ``/.dockerenv``; an equivalent Linux container runtime may set
    ``TEM_CONTAINER=1``. The value is recorded, never assumed.
    """
    return os.path.exists("/.dockerenv") or os.environ.get("TEM_CONTAINER") == "1"


def _git_commit() -> str | None:
    """Best-effort commit id; a slim Docker image may have no ``git`` binary.

    Absence is recorded honestly as ``None``. Source-file SHA-256 hashes remain
    mandatory and are the actual provenance authority.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def provenance(upstream_path: str | Path) -> dict:
    pin = json.loads((ROOT / "docker/upstream.lock.json").read_text(encoding="utf-8"))
    for name, digest in pin["files"].items():
        if sha(Path(upstream_path) / name) != digest:
            raise ValueError(f"upstream pin mismatch: {name}")
    return {
        "upstream": pin,
        "upstream_path": str(Path(upstream_path).resolve()),
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "git_commit": _git_commit(),
    }


# ------------------------------------------------------------- stage helpers


def scientific_stage(settings: dict) -> str:
    """The declared scientific stage; absent means the frozen standard gate."""
    value = settings.get("scientific_stage", STANDARD_STAGE)
    if value not in SCIENTIFIC_STAGES:
        raise ValueError(f"unknown scientific_stage {value!r}; expected one of {list(SCIENTIFIC_STAGES)}")
    return value


def stage_min_updates(settings: dict) -> int:
    return STAGE_MIN_UPDATES[scientific_stage(settings)]


def stage_checkpoints(settings: dict) -> tuple[int, ...]:
    return STAGE_CHECKPOINTS[scientific_stage(settings)]


def scientific_status(settings: dict, *, continued: bool = False) -> str:
    """A label no downstream reader can mistake for a full confirmation fit."""
    if not settings["scientific_mode"]:
        return NON_SCIENTIFIC_STATUS
    if continued:
        return CONTINUATION_STATUS
    return SCIENTIFIC_STATUS[scientific_stage(settings)]


# ------------------------------------------------------------ config validation


def _require_int(settings: dict, key: str, minimum: int) -> None:
    value = settings.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"training.{key} must be an integer >= {minimum}")


def _require_float(settings: dict, key: str) -> None:
    value = settings.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"training.{key} must be a finite nonnegative number")


def validate_config(config: dict, *, resolved: bool = False) -> None:
    """Reject any configuration that cannot satisfy the declared contract."""
    expected = {"schema", "training", "evaluation"}
    if resolved:
        expected |= {"arm", "training_graph_ids"}
    if set(config) != expected:
        raise ValueError(f"unexpected config keys: {sorted(set(config) ^ expected)}")
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected retraining config schema")
    settings, evaluation = config["training"], config["evaluation"]
    # ``scientific_stage`` and ``route_policy`` are independently optional, so
    # any subset of the declared keys that still contains every required key is
    # valid.
    if (
        not isinstance(settings, dict)
        or not REQUIRED_TRAINING_KEYS <= set(settings) <= TRAINING_KEYS
    ):
        raise ValueError("unexpected or missing training settings")
    if not isinstance(evaluation, dict) or set(evaluation) != {"seed"}:
        raise ValueError("evaluation.seed is required")

    for key in ("iterations", "batch_size", "episode_length", "rollout", "seed",
                "cue_seed", "route_seed", "family_seed", "threads", "log_every"):
        _require_int(settings, key, 0 if key.endswith("seed") else 1)
    if type(evaluation["seed"]) is not int or not 0 <= evaluation["seed"] < 2**32:
        raise ValueError("evaluation.seed must be an integer below 2**32")
    if any(settings[key] >= 2**32 for key in ("seed", "cue_seed", "route_seed", "family_seed")):
        raise ValueError("training seeds must be below 2**32")
    if settings["bptt"] not in BPTT_MODES:
        raise ValueError("training.bptt must be 'episode' or 'rollout'")
    if settings["schedule"] not in ("upstream", "pilot"):
        raise ValueError("training.schedule must be 'upstream' or 'pilot'")
    if settings["device"] not in ("cpu", "cuda"):
        raise ValueError("training.device must be explicit 'cpu' or 'cuda'")
    if settings.get("route_policy", DEFAULT_ROUTE_POLICY) not in ROUTE_POLICIES:
        raise ValueError(f"training.route_policy must be one of {list(ROUTE_POLICIES)}")
    if type(settings["scientific_mode"]) is not bool:
        raise ValueError("training.scientific_mode must be a boolean")
    _require_float(settings, "max_seconds")
    _require_float(settings, "vram_budget_gib")
    if settings["vram_budget_gib"] <= 0:
        raise ValueError("training.vram_budget_gib must be positive")
    if settings["episode_length"] < 2:
        raise ValueError("episode_length must contain an actual passage")
    if settings["rollout"] > settings["episode_length"] or settings["episode_length"] % settings["rollout"]:
        raise ValueError("episode_length must be divisible by rollout")
    if settings["iterations"] < 1:
        raise ValueError("iterations must be positive")

    saves = settings["save_updates"]
    if not isinstance(saves, list) or not saves or any(type(value) is not int for value in saves):
        raise ValueError("save_updates must be a nonempty integer list")
    if saves != sorted(set(saves)) or saves[-1] != settings["iterations"]:
        raise ValueError("save_updates must be strictly increasing and end at iterations")
    if any(not 1 <= value <= settings["iterations"] for value in saves):
        raise ValueError("save_updates entries must lie inside the fit budget")

    if resolved:
        if config["arm"] not in ARMS:
            raise ValueError(f"arm must be one of {list(ARMS)}")
        ids = config["training_graph_ids"]
        if not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("training_graph_ids must be a nonempty string list")
        if len(set(ids)) != len(ids):
            raise ValueError("training_graph_ids must not repeat a family")
        if config["arm"] == "single" and len(ids) != 1:
            raise ValueError("single arm requires exactly one training graph ID")
        if config["arm"] == "diverse" and len(ids) < 2:
            raise ValueError("diverse arm requires at least two training graph IDs")

    if settings["scientific_mode"]:
        stage = scientific_stage(settings)
        blockers = []
        if settings["iterations"] < stage_min_updates(settings):
            blockers.append(
                f"iterations {settings['iterations']} < {stage_min_updates(settings)} "
                f"for stage {stage}"
            )
        if settings["episode_length"] < FULL_HISTORY_MIN:
            blockers.append(f"episode_length {settings['episode_length']} < {FULL_HISTORY_MIN}")
        if settings["bptt"] != "episode":
            blockers.append("scientific mode requires full-episode BPTT; rollout truncates credit")
        if settings["schedule"] != "upstream":
            blockers.append("scientific mode requires the upstream parameter schedule")
        if settings["device"] != "cuda":
            blockers.append("scientific mode requires an explicit CUDA device")
        if settings["max_seconds"] != 0:
            blockers.append("scientific mode forbids a wall-clock cap")
        declared = stage_checkpoints(settings)
        if stage == INTERIM_STAGE:
            # The interim stage is a fixed preliminary budget, not a range. A
            # 4,999- or 5,001-update fit is not the declared amendment and a
            # 7,500-update interim checkpoint would not be a declared endpoint.
            if settings["iterations"] != INTERIM_UPDATES:
                blockers.append(
                    f"interim stage requires exactly {INTERIM_UPDATES} successful updates, "
                    f"not {settings['iterations']}"
                )
            if set(saves) != set(INTERIM_CHECKPOINTS):
                blockers.append(
                    f"interim stage save_updates must be exactly {list(INTERIM_CHECKPOINTS)}"
                )
        elif not set(declared) <= set(saves):
            blockers.append(f"save_updates must include {list(declared)}")
        if blockers:
            raise ValueError("scientific_mode blocked: " + "; ".join(blockers))
    elif scientific_stage(settings) == INTERIM_STAGE:
        # Disabling scientific_mode must not silently become a way to claim the
        # interim stage while bypassing its integrity conditions.
        raise ValueError(
            "the interim stage is a declared scientific stage and requires "
            "scientific_mode = true; use the standard stage for a non-scientific smoke"
        )


# ------------------------------------------------------------------- runtime


def runtime_attestation(torch, requested_device: str, *, allow_unavailable: bool = False) -> dict:
    """Resolve the device and record facts the scientific gate needs.

    ``allow_unavailable`` lets ``train`` record a clean refusal instead of
    raising when CUDA is missing. ``check`` keeps the strict behaviour.
    """
    try:
        runtime = adapter.resolve_device(requested_device, torch)
    except adapter.AdapterError as exc:
        if not allow_unavailable:
            raise
        runtime = {
            "device": "unavailable",
            "cuda_available": bool(torch.cuda.is_available()),
            "reason": str(exc),
        }
    info = {
        "requested_device": requested_device,
        "device": runtime["device"],
        "cuda_available": bool(torch.cuda.is_available()),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "in_container": _in_container(),
        "gpu": runtime.get("gpu"),
        "git_commit": _git_commit(),
    }
    if runtime["device"] == "cuda":
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        info.update({
            "gpu_name": props.name,
            "gpu_total_memory_bytes": int(total),
            "free_memory_bytes_at_start": int(free),
            "cuda_version": getattr(torch.version, "cuda", None),
        })
    return info


def scientific_blockers(config: dict, attestation: dict) -> list[str]:
    """Return every reason a scientific fit must not start.

    Kept separate from ``validate_config`` so unit tests can exercise the
    environment gate without a GPU. Empty list means the run may start.
    """
    settings = config["training"]
    if not settings["scientific_mode"]:
        return []
    reasons: list[str] = []
    stage = scientific_stage(settings)
    if attestation.get("device") != "cuda" or not attestation.get("cuda_available"):
        reasons.append("CUDA is unavailable or not explicitly selected")
    if not str(attestation.get("platform", "")).startswith("linux"):
        reasons.append("scientific full fit requires Linux/Docker CUDA; native Windows CUDA is not accepted")
    if not attestation.get("in_container"):
        reasons.append("scientific full fit requires the Docker runtime marker (/.dockerenv or TEM_CONTAINER=1)")
    total = attestation.get("gpu_total_memory_bytes")
    if type(total) is not int or total < (settings["vram_budget_gib"] + 1.0) * 2**30:
        reasons.append(
            f"GPU memory {total!r} cannot hold the declared {settings['vram_budget_gib']} GiB budget with 1 GiB headroom"
        )
    free = attestation.get("free_memory_bytes_at_start")
    if type(free) is not int or free < 2 * 2**30:
        reasons.append("less than 2 GiB free CUDA memory at start")
    if settings["iterations"] < stage_min_updates(settings):
        reasons.append(
            f"iterations {settings['iterations']} < {stage_min_updates(settings)} for stage {stage}"
        )
    if stage == INTERIM_STAGE:
        if settings["iterations"] != INTERIM_UPDATES:
            reasons.append(
                f"interim stage requires exactly {INTERIM_UPDATES} updates, not {settings['iterations']}"
            )
        if not set(INTERIM_CHECKPOINTS) <= set(settings["save_updates"]):
            reasons.append(f"interim stage requires checkpoints {list(INTERIM_CHECKPOINTS)}")
    if settings["episode_length"] < FULL_HISTORY_MIN:
        reasons.append(f"episode_length {settings['episode_length']} < {FULL_HISTORY_MIN}")
    if settings["bptt"] != "episode":
        reasons.append("truncated rollout BPTT is not eligible for the scientific gate")
    return reasons


# --------------------------------------------------------------- update ledger


@dataclass
class UpdateLedger:
    """Count only optimizer steps that actually completed with a real objective.

    The gate says empty-loss/skipped steps do not count, and a failed
    ``optimizer.step`` must not count either. Callers therefore use
    ``validate`` **before** the step and ``commit`` **only after** the step
    returns successfully. ``record`` keeps the combined behaviour for tests and
    non-training callers. The class is deliberately torch-free so the
    accounting can be exercised directly.
    """

    requested: int
    updates: int = 0
    skipped: int = 0

    def validate(self, *, loss_value: float, scored: int, requires_grad: bool,
                 gradients_finite: bool) -> None:
        if not math.isfinite(float(loss_value)):
            raise ValueError("nonfinite training loss")
        if scored <= 0:
            raise ZeroScoredUpdate("update has no scored visited-location prediction")
        if not requires_grad:
            raise ValueError("training loss does not require grad")
        if not gradients_finite:
            raise ValueError("training produced no gradients or nonfinite gradients")

    def commit(self) -> None:
        """Increment only after ``optimizer.step`` returned successfully."""
        self.updates += 1

    def record(self, *, loss_value: float, scored: int, requires_grad: bool,
               gradients_finite: bool) -> None:
        self.validate(loss_value=loss_value, scored=scored,
                      requires_grad=requires_grad, gradients_finite=gradients_finite)
        self.commit()

    def mark_skipped(self) -> None:
        self.skipped += 1

    def require_complete(self) -> None:
        if self.updates != self.requested:
            raise ValueError(f"incomplete fit: {self.updates}/{self.requested} real updates")


# ------------------------------------------------------------------- episodes


def _sequence_digest(*parts) -> bytes:
    return hashlib.blake2b(repr(parts).encode(), digest_size=16).digest()


# ------------------------------------------------------------------- routes
#
# The P1 objective asks the model to return over a reciprocal action-4 passage
# whose reverse direction it observed earlier, after a distractor. The
# historical trainer samples uniform legal transitions, which only *sometimes*
# contains that event. ``reciprocal_return`` builds the event deliberately from
# the public transition table: no cue, label, final payload, evaluator state or
# hidden adjacency enters it. Both arms and every seed share the controller and
# its seeding.


def action4_crossings(graph: dict) -> list[tuple[int, int]]:
    """Every directed action-4 crossing, verified reciprocal, in node order."""
    transitions = graph["transitions"]
    crossings = []
    for source in range(len(transitions)):
        target = transitions[source].get(4)
        if target is None:
            continue
        if transitions[target].get(4) != source:
            raise ValueError(
                f"{graph.get('id')}: action 4 from node {source} is not reciprocal"
            )
        crossings.append((source, target))
    return sorted(crossings)


def _shortest_path(transitions, start: int, goal: int, *, forbidden=None):
    """BFS over public transitions; ``forbidden`` is one directed edge."""
    queue = deque([start])
    previous: dict[int, int | None] = {start: None}
    previous_action: dict[int, int] = {}
    while queue:
        node = queue.popleft()
        if node == goal:
            break
        for action, target in sorted(transitions[node].items()):
            if forbidden is not None and (node, target) == forbidden:
                continue
            if target in previous:
                continue
            previous[target] = node
            previous_action[target] = action
            queue.append(target)
    if goal not in previous:
        return None
    nodes, actions = [goal], []
    node = goal
    while previous[node] is not None:
        actions.append(previous_action[node])
        node = previous[node]
        nodes.append(node)
    nodes.reverse()
    actions.reverse()
    return nodes, actions


def _append_walk(transitions, nodes, actions, rng, steps, *, forbid_nodes=(), forbidden=None):
    """Append ``steps`` legal transitions from ``nodes[-1]``.

    ``forbid_nodes`` keeps the distractor away from the target room and
    ``forbidden`` blocks one directed passage. The spanning tree of every
    authored graph guarantees an escape from any non-forbidden node.
    """
    for _ in range(steps):
        node = nodes[-1]
        choices = [
            (action, target)
            for action, target in sorted(transitions[node].items())
            if target not in forbid_nodes
            and not (forbidden is not None and (node, target) == forbidden)
        ]
        if not choices:
            return False
        action, target = choices[rng.randrange(len(choices))]
        actions.append(action)
        nodes.append(target)
    return True


def legal_walk(graph: dict, rng: random.Random, length: int) -> tuple[list[int], list[int]]:
    """Historical uniform legal-transition route (unchanged semantics)."""
    return worlds.walk(graph, rng, length)


def _reciprocal_attempt(graph, seed, length, *, leading, delay):
    """One deterministic attempt; returns ``(nodes, actions, lag)`` or ``None``."""
    transitions = graph["transitions"]
    rng = random.Random(f"p1:{seed}:{leading}:{delay}")
    crossings = action4_crossings(graph)
    target, source = crossings[rng.randrange(len(crossings))]
    focal = (source, target)
    nodes = [int(graph["root"])]
    actions: list[int] = []
    # Leading legal walk that never uses the queried direction S -> T.
    if not _append_walk(
        transitions, nodes, actions, rng, leading, forbidden=focal,
    ):
        return None
    # Shortest path to T, also avoiding an early S -> T crossing.
    path_nodes, path_actions = _shortest_path(
        transitions, nodes[-1], target, forbidden=focal,
    )
    if path_nodes is None:
        return None
    actions.extend(path_actions)
    nodes.extend(path_nodes[1:])
    cross_index = len(nodes) - 1  # T observed here
    actions.append(4)
    nodes.append(source)
    # Distractor: never re-enter T, never reuse the focal passage either way.
    if not _append_walk(
        transitions, nodes, actions, rng, delay,
        forbid_nodes={target}, forbidden=focal,
    ):
        return None
    back = _shortest_path(transitions, nodes[-1], source, forbidden=focal)
    if back is None or target in back[0]:
        return None
    back_nodes, back_actions = back
    actions.extend(back_actions)
    nodes.extend(back_nodes[1:])
    # Queried reverse crossing: queued at S, landing on T.
    query_index = len(nodes) - 1
    actions.append(4)
    nodes.append(target)
    if len(nodes) != length:
        return None
    for step, action in enumerate(actions):
        if transitions[nodes[step]].get(action) != nodes[step + 1]:
            raise ValueError(f"{graph.get('id')}: illegal reciprocal route step")
    return nodes, actions + [0], query_index - cross_index


def reciprocal_return_walk(graph: dict, rng: random.Random, length: int) -> tuple[list[int], list[int]]:
    """P1-compatible route: observe T -> S, distract, return, cross S -> T.

    The final transition is the reciprocal return S -> T (action 4) whose target
    cue was last observed 12-23 observations earlier at T. From start to finish
    the route is: an optional leading legal walk, a shortest path to T, the
    observed crossing T -> S, a distractor that never re-enters T or reuses the
    focal passage in either direction, a shortest path back to S, and the
    queried crossing S -> T.

    Exact length is resolved by a deterministic two-parameter search: the search
    only uses the public transition table, and the first feasible attempt in the
    declared order is frozen. No cue, label, score, final payload or evaluator
    state enters the choice.
    """
    if length < P1_MIN_RETURN_DELAY + 8:
        raise ValueError(
            f"reciprocal route needs at least {P1_MIN_RETURN_DELAY + 8} observations"
        )
    # The episode counter is the RNG seed's source; keep the declared seed in
    # the namespace through ``EpisodeBatches`` and pass its state as ``seed``.
    seed = rng.random()
    # Search the primary >=24 stratum first; the 12-23 stratum is not claimed.
    delays = list(range(P1_MIN_RETURN_DELAY, max(P1_MIN_RETURN_DELAY, length)))
    for leading in range(0, 9):
        for delay in delays:
            attempt = _reciprocal_attempt(graph, seed, length, leading=leading, delay=delay)
            if attempt is None:
                continue
            nodes, actions, lag = attempt
            if lag < P1_MIN_RETURN_DELAY:
                continue
            return nodes, actions
    raise ValueError(
        f"{graph.get('id')}: no exact-length P1 reciprocal return fits {length} observations"
    )


def route_for(policy: str):
    if policy not in ROUTE_POLICIES:
        raise ValueError(f"unknown route policy {policy!r}")
    return legal_walk if policy == "legal_walk" else reciprocal_return_walk


def plan_route_events(plan) -> dict:
    """Count P1-relevant route events from one episode plan.

    A P1 training event is: an action-4 crossing ``T -> S`` at step ``t``, then
    at least ``P1_MIN_RETURN_DELAY`` observations later the agent is at ``S``
    and the next action is action 4 back to ``T`` (``nodes[t]``). This is the
    training analogue of the pre-arrival reciprocal query, computed from the
    episode's public node/action sequence only.
    """
    nodes, actions = plan.nodes, plan.actions
    histogram = Counter(action for action in actions[:-1] if action != 0)
    seen: dict[int, int] = {}
    longest_lag = 0
    p1_events: list[int] = []
    for step, node in enumerate(nodes):
        if node in seen:
            longest_lag = max(longest_lag, step - seen[node])
        seen[node] = step
    for t in range(len(actions) - 1):
        if actions[t] != 4:
            continue
        target, source = nodes[t], nodes[t + 1]
        for u in range(t + P1_MIN_RETURN_DELAY, len(actions)):
            if nodes[u] == source and actions[u] == 4 and nodes[u + 1] == target:
                p1_events.append(u - t)
                break
    # The episode-final pre-arrival query: the last queued action is the
    # reciprocal return and its target was last observed ``final_lag`` earlier.
    final_lag = None
    if actions[-2] == 4:
        target = nodes[-1]
        prior = [index for index in range(len(nodes) - 1) if nodes[index] == target]
        if prior:
            final_lag = (len(nodes) - 1) - prior[-1]
    return {
        "action4_steps": int(histogram.get(4, 0)),
        "has_action4": bool(histogram.get(4, 0)),
        "p1_reciprocal_returns": len(p1_events),
        "p1_min_lag": min(p1_events) if p1_events else None,
        "p1_max_lag": max(p1_events) if p1_events else None,
        "final_return_lag": final_lag,
        "max_revisit_lag": longest_lag,
        "actions": dict(sorted(histogram.items())),
    }


def route_coverage(plans) -> dict:
    """Aggregate measured route-event frequencies for a set of episode plans."""
    plans = list(plans)
    if not plans:
        raise ValueError("route coverage needs at least one episode plan")
    events = [plan_route_events(plan) for plan in plans]
    total_actions = Counter()
    for item in events:
        total_actions.update(item["actions"])
    steps = sum(total_actions.values())
    long_lag = sum(1 for item in events if item["max_revisit_lag"] >= P1_MIN_RETURN_DELAY)
    p1 = [item for item in events if item["p1_reciprocal_returns"]]
    final = [item for item in events if item["final_return_lag"] is not None]
    strata = {"lt12": 0, "12_23": 0, "ge24": 0}
    for item in final:
        lag = item["final_return_lag"]
        if lag < P1_SHORT_STRATUM[0]:
            strata["lt12"] += 1
        elif lag <= P1_SHORT_STRATUM[1]:
            strata["12_23"] += 1
        else:
            strata["ge24"] += 1
    return {
        "episodes": len(events),
        "observations": sum(len(plan.nodes) for plan in plans),
        "action_histogram": {str(key): value for key, value in sorted(total_actions.items())},
        "action4_fraction": (total_actions.get(4, 0) / steps) if steps else 0.0,
        "episodes_with_action4": sum(1 for item in events if item["has_action4"]),
        "episodes_with_long_lag_revisit": long_lag,
        "episodes_with_p1_reciprocal_return": len(p1),
        "p1_return_fraction": len(p1) / len(events),
        "p1_lag_min": min((item["p1_min_lag"] for item in p1), default=None),
        "p1_lag_max": max((item["p1_max_lag"] for item in p1), default=None),
        "p1_lags": sorted(item["p1_min_lag"] for item in p1),
        "episodes_with_final_prearrival_return": len(final),
        "final_return_fraction": len(final) / len(events),
        "final_return_lag_min": min((item["final_return_lag"] for item in final), default=None),
        "final_return_lag_max": max((item["final_return_lag"] for item in final), default=None),
        "final_return_strata": strata,
        "primary_ge24_fraction": (
            strata["ge24"] / len(events) if events else 0.0
        ),
        "max_revisit_lag": max(item["max_revisit_lag"] for item in events),
        "distinct_graphs": len({plan.graph_id for plan in plans}),
    }


@dataclass(frozen=True)
class EpisodePlan:
    index: int
    graph_id: str
    graph_index: int
    nodes: tuple[int, ...]
    actions: tuple[int, ...]
    cues: tuple[int, ...]
    cue_assignment: tuple[int, ...]
    history_steps: int
    revisit_lag_mean: float | None
    revisit_lag_max: int | None
    action_counts: dict[int, int] = field(default_factory=dict)

    def chunk(self, torch, lo: int, hi: int, batch_size: int = 1):
        """One lane's chunk. Repeating a plan across lanes is forbidden.

        Independent lanes are assembled by ``BatchEpisodePlan``; a single plan
        may only be used at batch size one so a batch cannot silently become N
        clones of the same route, cue map and node sequence.
        """
        if batch_size != 1:
            raise ValueError("independent lanes require BatchEpisodePlan, not repeated EpisodePlan")
        cues_by_step = [(self.cues[t],) for t in range(lo, hi)]
        actions_by_step = [(self.actions[t],) for t in range(lo, hi)]
        walk = native.model_walk(torch, cues_by_step, actions_by_step)
        node_ids = [[self.nodes[t]] for t in range(lo, hi)]
        return walk, node_ids

    def full_walk(self, torch, batch_size: int = 1):
        return self.chunk(torch, 0, len(self.nodes), batch_size)


class BatchEpisodePlan:
    """One independent episode per batch lane.

    Each lane has its own graph, route, cue map and node-ID sequence. The
    per-step tensors are assembled in the ``[step][lane]`` layout that upstream
    ``Model.forward`` consumes, while the recurrent/fast/Hebbian state stays a
    single batched state with one independent lane per episode. The optimizer
    still takes exactly one step per batch update.
    """

    def __init__(self, plans: list[EpisodePlan] | tuple[EpisodePlan, ...]):
        if not plans:
            raise ValueError("a batch needs at least one episode plan")
        self.plans = tuple(plans)
        length = len(self.plans[0].nodes)
        if any(
            len(plan.nodes) != length or len(plan.cues) != length or len(plan.actions) != length
            for plan in self.plans
        ):
            raise ValueError("every lane must carry the same episode length")
        self.batch_size = len(self.plans)
        self.history_steps = length
        self.n_nodes = len(self.plans[0].cue_assignment)
        self.indices = tuple(plan.index for plan in self.plans)
        self.graph_ids = tuple(plan.graph_id for plan in self.plans)
        self.cue_assignments = tuple(plan.cue_assignment for plan in self.plans)
        self.route_sequences = tuple((plan.graph_id, plan.nodes) for plan in self.plans)
        self.cue_sequences = tuple(plan.cues for plan in self.plans)
        self.action_sequences = tuple(plan.actions for plan in self.plans)
        # Runtime independence guard: one lane per genuinely distinct episode.
        # Replicating a plan would count one episode many times, so a
        # duplicated route or cue sequence is a hard error rather than a
        # silently easier batch.
        if len(set(self.indices)) != self.batch_size:
            raise ValueError("batch lanes must be distinct episode indices")
        if len(set(self.route_sequences)) != self.batch_size:
            raise ValueError(
                "replicated batch lanes: every lane needs a distinct graph/route sequence"
            )
        if len(set(self.cue_sequences)) != self.batch_size:
            raise ValueError(
                "replicated batch lanes: every lane needs a distinct cue sequence"
            )

    def chunk(self, torch, lo: int, hi: int):
        cues_by_step = [tuple(plan.cues[t] for plan in self.plans) for t in range(lo, hi)]
        actions_by_step = [tuple(plan.actions[t] for plan in self.plans) for t in range(lo, hi)]
        node_ids = [[plan.nodes[t] for plan in self.plans] for t in range(lo, hi)]
        return native.model_walk(torch, cues_by_step, actions_by_step), node_ids

    def full_walk(self, torch):
        return self.chunk(torch, 0, self.history_steps)

    def diversity(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "distinct_graphs": len(set(self.graph_ids)),
            "distinct_routes": len(set(self.route_sequences)),
            "distinct_cue_sequences": len(set(self.cue_sequences)),
            "distinct_action_sequences": len(set(self.action_sequences)),
        }


def _as_batch(plan, batch_size: int) -> BatchEpisodePlan:
    """Compatibility for single-lane callers; never clones a lane."""
    if isinstance(plan, BatchEpisodePlan):
        if plan.batch_size != batch_size:
            raise ValueError("batch plan size disagrees with the update batch size")
        return plan
    if batch_size != 1:
        raise ValueError("independent lanes require BatchEpisodePlan, not repeated EpisodePlan")
    return BatchEpisodePlan([plan])


class EpisodeBatches:
    """Deterministic, counter-addressed episode stream shared by both arms.

    One update consumes ``batch_size`` independent episodes at global indices
    ``update * batch_size + lane``. Cue maps are generated from a canonical
    17-node graph and are therefore identical across the single-topology and
    diverse-topology arms for the same global episode index. Graph choice and
    route are keyed only by that index, so a resumed run reconstructs the same
    episode stream without depending on process RNG state; the model's own RNG
    streams are captured separately in ``training-state.pt``.
    """

    def __init__(self, config: dict, manifest: dict):
        settings = config["training"]
        self.settings = settings
        by_id = {graph["id"]: graph for graph in manifest["graphs"]}
        missing = [graph_id for graph_id in config["training_graph_ids"] if graph_id not in by_id]
        if missing:
            raise ValueError(f"manifest is missing training graphs: {missing}")
        self.graphs = [by_id[graph_id] for graph_id in config["training_graph_ids"]]
        self.reserved = manifest.get("reserved_assignments", ())
        self.cue_graph = self.graphs[0]
        self.used_assignments: set[tuple[int, ...]] = set()
        self.graph_counts: Counter[str] = Counter()
        self.action_counts: Counter[int] = Counter()
        self.route_digests: set[bytes] = set()
        self.cue_digests: set[bytes] = set()
        self.episodes = 0
        # Plans are cached only for the current rollout group, never for a
        # long episode-mode replay.
        self.plans: dict[int, EpisodePlan] = {}

    def assignment(self, episode: int) -> tuple[int, ...]:
        """Draw a unique complete cue assignment, deterministic in ``episode``.

        Only assignments from **committed** episodes are reserved. A failed
        episode is therefore retried with the same assignment after resume
        instead of silently drifting to another one.
        """
        attempt = 0
        while True:
            value = tuple(graph_api().cue_assignment(
                self.cue_graph, "train", episode + attempt * 10**12,
                self.settings["cue_seed"], self.reserved,
            ))
            if value not in self.used_assignments:
                return value
            attempt += 1

    def plan(self, episode: int, *, commit: bool = False, cache: bool = True) -> EpisodePlan:
        if cache and episode in self.plans:
            return self.plans[episode]
        family_rng = random.Random(f"family:{self.settings['family_seed']}:{episode}")
        graph_index = family_rng.randrange(len(self.graphs))
        graph = self.graphs[graph_index]
        assignment = self.assignment(episode)
        policy = self.settings.get("route_policy", DEFAULT_ROUTE_POLICY)
        route_rng = random.Random(f"route:{self.settings['route_seed']}:{episode}")
        nodes, actions = route_for(policy)(graph, route_rng, self.settings["episode_length"])
        cues = tuple(assignment[node] for node in nodes)
        last: dict[int, int] = {}
        lags: list[int] = []
        for step, node in enumerate(nodes):
            previous = last.get(node)
            if previous is not None:
                lags.append(step - previous)
            last[node] = step
        action_values = [action for action in actions[:-1] if action != 0]
        plan = EpisodePlan(
            index=episode,
            graph_id=graph["id"],
            graph_index=graph_index,
            nodes=tuple(nodes),
            actions=tuple(actions),
            cues=cues,
            cue_assignment=assignment,
            history_steps=len(nodes),
            revisit_lag_mean=(sum(lags) / len(lags)) if lags else None,
            revisit_lag_max=(max(lags) if lags else None),
            action_counts=dict(Counter(action_values)),
        )
        if cache:
            self.plans[episode] = plan
        if commit:
            self.commit(plan)
        return plan

    def batch(self, update_index: int, *, commit: bool = False, cache: bool = False) -> BatchEpisodePlan:
        """Build the independent lane episodes for one optimizer update."""
        if type(update_index) is not int or update_index < 0:
            raise ValueError("update index must be a nonnegative integer")
        base = update_index * self.settings["batch_size"]
        plans = [
            self.plan(base + lane, cache=cache)
            for lane in range(self.settings["batch_size"])
        ]
        value = BatchEpisodePlan(plans)
        if commit:
            self.commit_batch(value)
        return value

    def commit(self, plan: EpisodePlan) -> None:
        """Record a successfully consumed episode for exposure accounting."""
        if plan.cue_assignment in self.used_assignments:
            return
        self.used_assignments.add(plan.cue_assignment)
        self.graph_counts[plan.graph_id] += 1
        self.action_counts.update(
            action for action in plan.actions[:-1] if action != 0
        )
        self.route_digests.add(_sequence_digest(plan.graph_id, plan.nodes))
        self.cue_digests.add(_sequence_digest(plan.cues))
        self.episodes += 1

    def commit_batch(self, batch: BatchEpisodePlan) -> None:
        for plan in batch.plans:
            self.commit(plan)

    def clear_cache(self) -> None:
        self.plans.clear()

    def replay_to(self, episode: int) -> None:
        """Rebuild the exact stream state for episodes ``[0, episode)``.

        Only the deterministic plan generator runs; no model forward occurs.
        This is what makes episode reconstruction independent of process RNG
        state. The endpoint must be a whole batch boundary.
        """
        if type(episode) is not int or episode < 0 or episode % self.settings["batch_size"]:
            raise ValueError("replay endpoint must be a nonnegative multiple of batch_size")
        for index in range(episode):
            self.plan(index, commit=True, cache=False)


# ------------------------------------------------------------------- model IO


def initialize(config: dict, repo: str | Path = adapter.DEFAULT_REPO):
    """Construct a TEM model exactly as the native access adapter does."""
    torch = adapter.require_torch()
    numpy = adapter.require_numpy()
    settings = config["training"]
    torch.set_num_threads(settings["threads"])
    numpy.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings["seed"])
    upstream = adapter.load_upstream(Path(repo))
    params = adapter.default_params(upstream, iterations=settings["iterations"])
    params["n_actions"] = len(worlds.ACTION_LABELS) - 1
    if settings["schedule"] == "pilot":
        adapter.apply_pilot_schedule(params, settings["iterations"])
    return torch, upstream, params, upstream.model.Model(params)


def scheduled_weights(upstream, model, params: dict, update: int):
    return native.scheduled_weights(upstream, model, params, update)


def state_digest(model) -> str:
    return native.state_digest(model)


def _plain_hyper_cpu(value, numpy=None):
    """Make hyperparameters CPU/plain before ``params_*.pt`` is written."""
    numpy = adapter.require_numpy() if numpy is None else numpy
    value = _plain_hyper(value, numpy)
    torch = adapter.require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _plain_hyper_cpu(item, numpy) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_hyper_cpu(item, numpy) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_hyper_cpu(item, numpy) for item in value)
    return value


def _atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch = adapter.require_torch()
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _serialise_carry(prev) -> dict | None:
    if prev is None:
        return None
    iteration = prev[0]
    return {
        "a": list(iteration.a),
        "M": [tensor.detach().cpu() for tensor in iteration.M],
        "x_inf": [tensor.detach().cpu() for tensor in iteration.x_inf],
        "g_inf": [tensor.detach().cpu() for tensor in iteration.g_inf],
    }


def _restore_carry(value: dict | None, upstream, device: str):
    if value is None:
        return None
    iteration = upstream.model.Iteration(
        a=list(value["a"]),
        M=[tensor.to(device) for tensor in value["M"]],
        x_inf=[tensor.to(device) for tensor in value["x_inf"]],
        g_inf=[tensor.to(device) for tensor in value["g_inf"]],
    )
    return [iteration]


def _rng_snapshot(torch, device: str) -> dict:
    """Capture every RNG the fit could consult.

    ``do_sample`` is False upstream, so the current fit is deterministic, but a
    resume must not silently restart an RNG stream if that ever changed. The
    snapshot is stored in the training state and restored before the next
    update.
    """
    numpy = adapter.require_numpy()
    legacy = numpy.random.get_state()
    state = {
        "python": random.getstate(),
        # Stored as primitives so the adapter's weights_only=True loader (and
        # its NumPy safe-globals allowlist) can read the file without an
        # arbitrary unpickle path.
        "numpy": {
            "name": legacy[0],
            "keys": [int(value) for value in legacy[1]],
            "pos": int(legacy[2]),
            "has_gauss": int(legacy[3]),
            "cached_gaussian": float(legacy[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if device == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _rng_restore(state: dict | None, torch, device: str) -> None:
    if not isinstance(state, dict) or "python" not in state:
        raise ValueError("resume requested but the training state has no RNG snapshot")
    numpy = adapter.require_numpy()
    random.setstate(state["python"])
    legacy = state["numpy"]
    if isinstance(legacy, (list, tuple)):  # accept an older in-memory shape
        numpy.random.set_state(tuple(legacy))
    else:
        numpy.random.set_state((
            legacy["name"],
            numpy.array(legacy["keys"], dtype=numpy.uint32),
            legacy["pos"],
            legacy["has_gauss"],
            legacy["cached_gaussian"],
        ))
    torch.set_rng_state(state["torch_cpu"])
    if device == "cuda" and torch.cuda.is_available():
        if "torch_cuda" not in state:
            raise ValueError("CUDA resume requires the CUDA RNG snapshot")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rng_digest(state: dict) -> str:
    digest = hashlib.sha256()
    digest.update(repr(state.get("python")).encode())
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        digest.update(repr(numpy_state).encode())
    digest.update(state["torch_cpu"].detach().cpu().numpy().tobytes())
    for tensor in state.get("torch_cuda", ()):  # type: ignore[arg-type]
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_artifacts(model, optimizer, out: Path, update: int, *, record: dict,
                   state: dict, verify: dict) -> dict:
    """Atomically write the declared checkpoint, params and resume state."""
    checkpoint = out / f"checkpoint_{update:06d}.pt"
    params_path = out / f"params_{update:06d}.pt"
    state_path = out / f"training-state_{update:06d}.pt"
    _atomic_torch_save(model.state_dict(), checkpoint)
    _atomic_torch_save(_plain_hyper_cpu(model.hyper), params_path)
    payload = {
        "schema": STATE_SCHEMA,
        "update": update,
        "config_sha256": record["config_sha256"],
        "manifest_sha256": record["manifest_sha256"],
        "code_sha256": record["code_sha256"],
        "arm": record["arm"],
        "training_seed": record["training_seed"],
        "optimizer": optimizer.state_dict(),
        **state,
    }
    _atomic_torch_save(payload, state_path)
    entry = {
        "update": update,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha(checkpoint),
        "params": params_path.name,
        "params_sha256": sha(params_path),
        "training_state": state_path.name,
        "training_state_sha256": sha(state_path),
        **verify,
    }
    # The ledger is the commit point: an update is committed only once its
    # numbered files are complete and checkpoints.json records them. It is
    # written before the latest aliases so an interrupted save can never leave
    # ``training-state.pt`` pointing at an update the ledger does not know.
    manifest_path = out / "checkpoints.json"
    entries = []
    if manifest_path.is_file():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [item for item in entries if item["update"] != update] + [entry]
    entries.sort(key=lambda item: item["update"])
    write_json(manifest_path, entries)
    # Latest aliases are copies of the verified numbered files. They may lag
    # the ledger for an instant after an interrupted save; resume treats the
    # ledger as authoritative and recovers any stale alias.
    for source, target in (
        (checkpoint, out / "checkpoint.pt"),
        (params_path, out / "params.pt"),
        (state_path, out / "training-state.pt"),
    ):
        _atomic_copy(source, target)
    return entry


def archive_and_trim_metrics(out: Path, committed_update: int) -> dict | None:
    """Keep only committed metric rows canonical; archive discarded attempts.

    A resume from the last *saved* checkpoint may discard successful-but-unsaved
    updates. Their raw metric rows are preserved in a separate JSONL archive,
    and the canonical ``metrics.jsonl`` is trimmed to the committed lineage so
    it exactly matches ``checkpoints.json`` and the re-run update sequence.
    """
    path = Path(out) / "metrics.jsonl"
    if not path.is_file():
        return None
    kept, discarded = [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metrics.jsonl line {number} is not valid JSON") from exc
        if type(row.get("update")) is not int:
            raise ValueError(f"metrics.jsonl line {number} has no integer update")
        (kept if row["update"] <= committed_update else discarded).append(line)
    if not discarded:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = Path(out) / f"metrics-discarded-{stamp}.jsonl"
    if archive.exists():
        raise FileExistsError(f"discarded metrics archive already exists: {archive}")
    archive.write_text("\n".join(discarded) + "\n", encoding="utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    os.replace(temporary, path)
    return {
        "archive": archive.name,
        "archive_sha256": sha(archive),
        "discarded_rows": len(discarded),
        "discarded_updates": sorted({json.loads(line)["update"] for line in discarded}),
        "canonical_rows": len(kept),
    }


def save_failure_snapshot(model, optimizer, out: Path, update: int, *, record: dict,
                          state: dict) -> dict:
    """Persist a failure diagnostic that can never be mistaken for a checkpoint.

    A raised ``optimizer.step`` may have partially mutated weights. This file is
    explicitly non-resumable, is never aliased to ``training-state.pt``, and
    never enters ``checkpoints.json`` or a numbered checkpoint name.
    """
    path = out / "failed-snapshot.pt"
    temporary = path.with_name(path.name + ".tmp")
    torch = adapter.require_torch()
    payload = {
        "schema": STATE_SCHEMA,
        "non_resumable_failure_snapshot": True,
        "weights_committed": False,
        "update": update,
        "config_sha256": record["config_sha256"],
        "manifest_sha256": record["manifest_sha256"],
        "code_sha256": record["code_sha256"],
        "arm": record["arm"],
        "training_seed": record["training_seed"],
        "optimizer": optimizer.state_dict(),
        **state,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    document = {
        "status": "failed",
        "non_resumable": True,
        "weights_committed": False,
        "update_at_failure": update,
        "snapshot": path.name,
        "snapshot_sha256": sha(path),
        "resume_instruction": (
            "resume from the last committed training-state.pt; this failure "
            "snapshot has no authority"
        ),
    }
    write_json(out / "failed-snapshot.json", document)
    return document


# --------------------------------------------------------------- fit mechanics


def _gradients_finite(torch, model) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        return False
    return all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def _state_stream_delta(left_steps, right_steps, name: str) -> float:
    """Maximum elementwise delta of one recurrent field across two forwards."""
    worst = 0.0
    for left, right in zip(left_steps, right_steps):
        left_value, right_value = getattr(left, name), getattr(right, name)
        if left_value is None or right_value is None:
            continue
        if name == "x_logits":
            left_value, right_value = left_value[2], right_value[2]
        if isinstance(left_value, (list, tuple)):
            for left_item, right_item in zip(left_value, right_value):
                worst = max(worst, float((left_item - right_item).abs().max()))
            continue
        worst = max(worst, float((left_value - right_value).abs().max()))
    return worst


def _gradient_norm(model) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.norm())
    return total


def _vram_guard(torch, settings: dict) -> None:
    """Fail closed when the measured CUDA peak exceeds the declared budget."""
    if settings["device"] != "cuda":
        return
    peak = int(torch.cuda.max_memory_allocated())
    budget = int(settings["vram_budget_gib"] * 2**30)
    if peak > budget:
        raise RuntimeError(
            f"vram_budget_exceeded: peak {peak} bytes > declared {budget} bytes; "
            "lower the batch or amend the protocol explicitly"
        )


def _episode_update(torch, model, optimizer, plan, *, batch_size: int,
                    rollout: int, weights, device: str, ledger: UpdateLedger,
                    step_grad_norm: bool = True) -> dict:
    """One optimizer update over ``batch_size`` independent episodes.

    Each lane has its own graph, route and cue map (``BatchEpisodePlan``). The
    model keeps one batched recurrent/fast/Hebbian state with an independent
    lane per episode, reset once at the episode start. ``prev`` is intentionally
    **not** detached between chunks: the graph spans all observations per lane,
    so late observations assign credit to early episode state.
    """
    batch = _as_batch(plan, batch_size)
    visited = [[False] * batch.n_nodes for _ in range(batch.batch_size)]
    total = None
    scored = 0
    previous = None
    weights = weights.to(device)
    started = time.perf_counter()
    for lo in range(0, batch.history_steps, rollout):
        hi = min(lo + rollout, batch.history_steps)
        with torch.device(device):
            walk, node_ids = batch.chunk(torch, lo, hi)
            forward = model(walk, previous)
            loss = adapter.compute_loss(forward, visited, weights, node_ids=node_ids)
        # Keep the graph: this is what makes the credit assignment full-episode.
        previous = [forward[-1]]
        total = loss["loss"] if total is None else total + loss["loss"]
        scored += loss["scored_predictions"]
    if scored <= 0:
        raise ZeroScoredUpdate("full episode batch produced no scored visited-location prediction")
    if not bool(torch.isfinite(total)):
        raise ValueError("nonfinite full-episode loss")
    if not bool(total.requires_grad):
        raise ValueError("full-episode loss does not require grad")
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    finite = _gradients_finite(torch, model)
    # All pre-step validations happen here. Nothing is counted until the
    # optimizer step below actually returns.
    ledger.validate(loss_value=float(total.detach()), scored=scored,
                    requires_grad=True, gradients_finite=finite)
    norm = _gradient_norm(model) if step_grad_norm else None
    optimizer.step()
    ledger.commit()
    metrics = {
        "loss": float(total.detach()),
        "scored_predictions": scored,
        "history_steps": batch.history_steps,
        "grad_norm": norm,
        "seconds": time.perf_counter() - started,
        "max_hebb": float(previous[0].M[0].detach().abs().max()),
        **batch.diversity(),
    }
    del previous
    return metrics


def _rollout_update(torch, model, optimizer, plan, *, chunk_index: int,
                    batch_size: int, rollout: int, weights, device: str,
                    ledger: UpdateLedger, visited, previous, step_grad_norm: bool = True) -> dict:
    """One rollout chunk over independent lane episodes (truncated BPTT).

    This path exists for throughput/memory benchmarks and cannot be flagged as
    scientific. A chunk with no eligible prediction consumes observations but
    does **not** count as an optimizer update.
    """
    batch = _as_batch(plan, batch_size)
    lo = chunk_index * rollout
    hi = min(lo + rollout, batch.history_steps)
    weights = weights.to(device)
    started = time.perf_counter()
    with torch.device(device):
        walk, node_ids = batch.chunk(torch, lo, hi)
        forward = model(walk, previous)
        loss = adapter.compute_loss(forward, visited, weights, node_ids=node_ids)
    scored = loss["scored_predictions"]
    stepped = False
    norm = None
    if scored > 0:
        if not bool(torch.isfinite(loss["loss"])):
            raise ValueError("nonfinite rollout loss")
        if not bool(loss["loss"].requires_grad):
            raise ValueError("rollout loss does not require grad")
        optimizer.zero_grad(set_to_none=True)
        loss["loss"].backward()
        finite = _gradients_finite(torch, model)
        ledger.validate(loss_value=float(loss["loss"].detach()), scored=scored,
                        requires_grad=True, gradients_finite=finite)
        norm = _gradient_norm(model) if step_grad_norm else None
        optimizer.step()
        ledger.commit()
        stepped = True
    else:
        ledger.mark_skipped()
    # Detach after the backward so the next chunk cannot backprop through it.
    carry = [forward[-1].detach()]
    return {
        "loss": float(loss["loss"].detach()),
        "scored_predictions": scored,
        "history_steps": hi,
        "grad_norm": norm,
        "seconds": time.perf_counter() - started,
        "stepped": stepped,
        "max_hebb": float(carry[0].M[0].detach().abs().max()),
        "visited": visited,
        "previous": carry,
        "next_chunk": chunk_index + 1,
        **batch.diversity(),
    }


# --------------------------------------------------------------- preparation


def _load_raw_config(config_path: Path) -> tuple[bytes, dict]:
    raw = Path(config_path).read_bytes()
    return raw, tomllib.loads(raw.decode("utf-8-sig"))


def prepare(config_path: Path, manifest_path: Path, out: Path, arm: str,
            repo: Path, *, exclusions_path: Path | None = None,
            updates: int | None = None, seed: int | None = None,
            episode_length: int | None = None, batch_size: int | None = None,
            device: str | None = None, max_seconds: float | None = None) -> dict:
    raw, config = _load_raw_config(config_path)
    smoke = updates is not None
    smoke_override = None
    if smoke:
        if not 1 <= updates <= 20:
            raise ValueError("smoke updates must be between 1 and 20; a smoke is not a fit")
        settings = config["training"]
        settings["scientific_mode"] = False
        # A smoke exercises integration only; it cannot borrow the interim or
        # standard scientific label.
        settings["scientific_stage"] = STANDARD_STAGE
        settings["iterations"] = updates
        settings["save_updates"] = [updates]
        if episode_length is not None:
            if episode_length < 2:
                raise ValueError("smoke episode_length must be at least 2")
            settings["episode_length"] = episode_length
        if batch_size is not None:
            if batch_size < 1:
                raise ValueError("smoke batch_size must be positive")
            settings["batch_size"] = batch_size
        if device is not None:
            if device not in ("cpu", "cuda"):
                raise ValueError("smoke device must be cpu or cuda")
            settings["device"] = device
        if max_seconds is not None:
            if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds < 0:
                raise ValueError("smoke max_seconds must be a finite nonnegative number")
            settings["max_seconds"] = max_seconds
        smoke_override = {
            "updates": updates, "episode_length": episode_length,
            "batch_size": batch_size, "device": device, "max_seconds": max_seconds,
        }
    elif any(value is not None for value in (episode_length, batch_size, device, max_seconds)):
        raise ValueError("episode/batch/device/time overrides are smoke-only; edit and refreeze the config")
    if seed is not None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed override must be an integer below 2**32")
        config["training"]["seed"] = seed
    validate_config(config)

    exclusions_source = (
        Path(exclusions_path) if exclusions_path is not None
        else Path(manifest_path).with_name("exclusions-private.json")
    )
    manifest, exclusions, reserved, public_audit = _load_cohort_public(
        Path(manifest_path), exclusions_source,
    )
    train_ids = manifest["splits"]["train"]
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)}")
    if arm == "diverse" and len(train_ids) < 2:
        raise ValueError("diverse arm requires a cohort with at least two train families")
    resolved = {
        **config,
        "arm": arm,
        "training_graph_ids": train_ids[:1] if arm == "single" else train_ids,
    }
    validate_config(resolved, resolved=True)

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"prepared output already exists: {out}")
    config_bytes = json.dumps(resolved, sort_keys=True, indent=2).encode()
    source_manifest_bytes = Path(manifest_path).read_bytes()
    evidence = provenance(repo)
    out.mkdir(parents=True, exist_ok=False)
    (out / "source-config.toml").write_bytes(raw)
    (out / "config.json").write_bytes(config_bytes)
    (out / "graphs.json").write_bytes(source_manifest_bytes)
    # The exclusion ledger is copied so ``load_prepared`` can re-run the full
    # public audit without depending on the original source path. It is
    # metadata only; it is never the sealed final payload.
    (out / "cohort-exclusions.json").write_bytes(exclusions_source.read_bytes())
    reservations = _reservations_payload(reserved, exclusions)
    write_json(out / "reservations.json", reservations)
    stage = scientific_stage(resolved["training"])
    record = {
        "schema": PREPARED_SCHEMA,
        "status": "prepared",
        "created_utc": now(),
        "arm": arm,
        "optimizer_steps": 0,
        "scientific_mode": resolved["training"]["scientific_mode"],
        "scientific_stage": stage,
        "scientific_status": scientific_status(resolved["training"]),
        "scientific_contract": {
            "min_updates": stage_min_updates(resolved["training"]),
            "declared_checkpoints": list(stage_checkpoints(resolved["training"])),
            "episode_length_min": FULL_HISTORY_MIN,
            "interim_preliminary": stage == INTERIM_STAGE,
            "confirmation_required": stage == INTERIM_STAGE,
            "final_capability_pass": False,
        },
        "route_policy": resolved["training"].get("route_policy", DEFAULT_ROUTE_POLICY),
        "scientific_fixture": False,
        "cohort_kind": public_audit.get("cohort_kind"),
        "train_graph_count": len(resolved["training_graph_ids"]),
        "cohort_public_hashes": None,
        "sealed_final_assurance": "unverified: sealed payload neither read nor hashed",
        "credit_assignment": (
            "full_episode_bptt" if resolved["training"]["bptt"] == "episode"
            else "truncated_bptt_rollout"
        ),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_config_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "graphs_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "graph_content_sha256": manifest["content_sha256"],
        "manifest_content_sha256": manifest["content_sha256"],
        "cohort_schema": manifest["schema"],
        "cohort_id": manifest.get("cohort_id"),
        "cohort_seed": manifest.get("seed"),
        "structure": manifest.get("structure"),
        "cue_contract": manifest.get("cue_contract"),
        "exclusions_sha256": exclusions["content_sha256"],
        "cohort_exclusions_sha256": sha(out / "cohort-exclusions.json"),
        "reservations_sha256": sha(out / "reservations.json"),
        "reserved_assignments_count": len(reserved),
        "empty_historic_inventory": bool(exclusions.get("empty_inventory", False)),
        "test_payload_status": manifest["test_commitment"]["status"],
        "test_payload_verification": (
            "commitment metadata only; the trainer never reads or hashes "
            "test-sealed.json and makes no sealed-payload verification claim"
        ),
        "test_payload_sha256": manifest["test_commitment"].get("payload_sha256"),
        "test_graph_count": manifest["test_commitment"].get("graph_count"),
        "test_payload_opened": False,
        "test_payload_read": False,
        "public_audit": public_audit,
        "training_seed": resolved["training"]["seed"],
        "selection": "fixed budget final endpoint; no validation selection",
        "runtime_requirement": (
            "CUDA-on-Linux/Docker for a scientific fit; native Windows CUDA refused"
            if resolved["training"]["scientific_mode"] else "non-scientific pilot/benchmark"
        ),
        "stage_note": (
            "declared interim preliminary checkpoint; the standard confirmation "
            "program is separate and not superseded"
            if stage == INTERIM_STAGE else
            "declared standard scientific stage"
        ),
        "claims": CLAIMS,
        "smoke_override": smoke_override,
        **evidence,
    }
    write_json(out / "manifest.json", record)
    return record


def load_prepared(prepared: Path, repo: Path = adapter.DEFAULT_REPO):
    prepared = Path(prepared)
    record = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if record["schema"] != PREPARED_SCHEMA or record["status"] != "prepared":
        raise ValueError("invalid prepared retraining manifest")
    for name, key in (
        ("config.json", "config_sha256"),
        ("graphs.json", "manifest_sha256"),
        ("source-config.toml", "source_config_sha256"),
        ("cohort-exclusions.json", "cohort_exclusions_sha256"),
        ("reservations.json", "reservations_sha256"),
    ):
        if sha(prepared / name) != record[key]:
            raise ValueError(f"prepared input digest mismatch: {name}")
    current = provenance(repo)
    if current["code_sha256"] != record["code_sha256"] or current["upstream"] != record["upstream"]:
        raise ValueError("trainer/graph/native code or upstream changed after freeze")
    config = json.loads((prepared / "config.json").read_text(encoding="utf-8"))
    validate_config(config, resolved=True)
    manifest = graph_api().load_manifest(prepared / "graphs.json")
    _require_public_cohort(manifest)
    exclusions = graph_api().load_exclusions(prepared / "cohort-exclusions.json")
    if exclusions.get("content_sha256") != record["exclusions_sha256"]:
        raise ValueError("prepared exclusion ledger changed after freeze")
    if _cohort_kind(manifest) != record.get("cohort_kind"):
        raise ValueError("prepared cohort kind changed after freeze")
    _validate_public_manifest(
        manifest, expected_exclusions_sha256=record["exclusions_sha256"],
    )
    if manifest.get("content_sha256") != record["manifest_content_sha256"]:
        raise ValueError("prepared cohort content digest disagrees with the frozen manifest")
    manifest = _normalize_cohort_graphs(manifest)
    commitment = manifest["test_commitment"]
    if (
        record["test_payload_status"] != "sealed"
        or record["test_payload_sha256"] != commitment.get("payload_sha256")
        or record["test_graph_count"] != commitment.get("graph_count")
        or record["test_payload_opened"] is not False
        or record["test_payload_read"] is not False
    ):
        raise ValueError("prepared test commitment is not a sealed, unopened cohort")
    public_audit = record.get("public_audit", {})
    if public_audit.get("status") != "passed" or public_audit.get("sealed_payload_read") is not False:
        raise ValueError("prepared public cohort audit is missing or not sealed")
    reservations = json.loads((prepared / "reservations.json").read_text(encoding="utf-8"))
    if (
        reservations.get("schema") != RESERVATIONS_SCHEMA
        or reservations.get("count") != len(reservations.get("assignments", []))
        or reservations.get("exclusions_sha256") != record["exclusions_sha256"]
        or reservations.get("count") != record["reserved_assignments_count"]
    ):
        raise ValueError("prepared historic reservations are inconsistent")
    # Never read test-sealed.json; attach only the private historic reservations.
    manifest["reserved_assignments"] = reservations["assignments"]
    expected = manifest["splits"]["train"][:1] if config["arm"] == "single" else manifest["splits"]["train"]
    if config["training_graph_ids"] != expected:
        raise ValueError("training IDs disagree with the frozen comparison")
    if record["arm"] != config["arm"] or record["config_sha256"] != hashlib.sha256(
        (prepared / "config.json").read_bytes()
    ).hexdigest():
        raise ValueError("prepared arm/config provenance mismatch")
    return config, manifest, record


# ------------------------------------------------------------------ preflight


def check(prepared: Path, repo: Path = adapter.DEFAULT_REPO, requested_device: str = "cpu") -> dict:
    """No-optimizer preflight: prove state carry, gradients and memory cost."""
    config, graphs, record = load_prepared(Path(prepared), Path(repo))
    torch, upstream, params, model = initialize(config, Path(repo))
    runtime = runtime_attestation(torch, requested_device)
    device = runtime["device"]
    started = time.perf_counter()
    if device == "cuda":
        adapter.prepare_cuda_model(model, torch)
        torch.cuda.reset_peak_memory_stats()
    model.train()
    batches = EpisodeBatches(config, graphs)
    batch = batches.batch(0)
    settings = config["training"]
    _, weights = scheduled_weights(upstream, model, params, 0)
    before = state_digest(model)

    # 1) Exact state carry: chunked forward must equal one full forward.
    with torch.no_grad(), torch.device(device):
        # ``Model.forward`` iterates ``for g, x, a in walk``, so it needs the
        # time-major walk itself: step first, then lane. ``chunk`` returns
        # ``(walk, node_ids)``; node IDs are only needed for loss masking.
        full_walk, _ = batch.chunk(torch, 0, batch.history_steps)
        full = model(full_walk, None)
        chunked, previous = [], None
        for lo in range(0, batch.history_steps, settings["rollout"]):
            hi = min(lo + settings["rollout"], batch.history_steps)
            walk, _ = batch.chunk(torch, lo, hi)
            output = model(walk, previous)
            chunked.extend(output)
            previous = [output[-1]]
        carry_delta = max(
            float((left.x_logits[2] - right.x_logits[2]).abs().max())
            for left, right in zip(full, chunked)
        )
        # g_inf, g_gen and the Hebbian memory must carry too; a logits-only
        # check can miss a divergent recurrent state that happens to decode
        # the same next cue.
        carry_state_deltas = {
            name: _state_stream_delta(full, chunked, name)
            for name in ("g_inf", "g_gen", "M", "x_inf", "x_logits")
        }
        hebbian_max = max(float(step.M[0].abs().max()) for step in full)
        g_inf_max = max(
            float(value.abs().max()) for step in full for value in step.g_inf
        )
    if carry_delta > 1e-6:
        raise ValueError(f"chunked state carry diverged from a full forward: {carry_delta}")
    if any(value > 1e-6 for name, value in carry_state_deltas.items() if name != "x_logits"):
        raise ValueError(f"chunked state carry diverged in recurrent state: {carry_state_deltas}")

    # 2) Reset is explicit and changes the state.
    with torch.no_grad(), torch.device(device):
        first_walk, _ = batch.chunk(torch, 0, settings["rollout"])
        rest_walk, _ = batch.chunk(torch, settings["rollout"], 2 * settings["rollout"])
        first = model(first_walk, None)
        carried = model(rest_walk, [first[-1]])
        reset = model(rest_walk, None)
        reset_delta = float((carried[-1].x_logits[2] - reset[-1].x_logits[2]).abs().max())
    if reset_delta <= 0:
        raise ValueError("state reset did not change the forward result")

    # 3) A real backward with no optimizer step; weights must be unchanged.
    visited = [[False] * batch.n_nodes for _ in range(batch.batch_size)]
    total = None
    previous = None
    scored = 0
    with torch.device(device):
        for lo in range(0, batch.history_steps, settings["rollout"]):
            hi = min(lo + settings["rollout"], batch.history_steps)
            walk, node_ids = batch.chunk(torch, lo, hi)
            output = model(walk, previous)
            loss = adapter.compute_loss(output, visited, weights.to(device), node_ids=node_ids)
            previous = [output[-1]]
            total = loss["loss"] if total is None else total + loss["loss"]
            scored += loss["scored_predictions"]
    if scored <= 0 or not bool(torch.isfinite(total)):
        raise ValueError("preflight episode has no finite scored prediction")
    model.zero_grad(set_to_none=True)
    total.backward()
    gradients_finite = _gradients_finite(torch, model)
    if not gradients_finite:
        raise ValueError("preflight produced no gradients or nonfinite gradients")
    initial_state_gradient = max(
        (
            float(parameter.grad.abs().max())
            for name, parameter in model.named_parameters()
            if name.startswith("g_init.") and parameter.grad is not None
        ),
        default=0.0,
    )
    if initial_state_gradient <= 0:
        raise ValueError(
            "full-episode BPTT did not credit the episode-initial g state"
        )
    model.zero_grad(set_to_none=True)
    after = state_digest(model)
    if after != before:
        raise ValueError("preflight changed model weights")
    result = {
        "status": "checked_no_optimizer_updates",
        "optimizer_steps": 0,
        "arm": config["arm"],
        "config_sha256": record["config_sha256"],
        "device": device,
        "runtime": runtime,
        "scientific_mode": config["training"]["scientific_mode"],
        "scientific_stage": scientific_stage(config["training"]),
        "scientific_status": scientific_status(config["training"]),
        "scientific_contract": {
            "min_updates": stage_min_updates(config["training"]),
            "declared_checkpoints": list(stage_checkpoints(config["training"])),
            "final_capability_pass": False,
        },
        "scientific_blockers": scientific_blockers(config, runtime),
        "batch_size": settings["batch_size"],
        "episode_length": settings["episode_length"],
        "rollout": settings["rollout"],
        "bptt": config["training"]["bptt"],
        "route_policy": config["training"].get("route_policy", DEFAULT_ROUTE_POLICY),
        "state_carry_max_logit_delta": carry_delta,
        "state_carry_recurrent_deltas": carry_state_deltas,
        "state_carry_max_g_inf_delta": carry_state_deltas["g_inf"],
        "state_carry_max_g_gen_delta": carry_state_deltas["g_gen"],
        "state_carry_max_hebbian_delta": carry_state_deltas["M"],
        "state_carry_max_x_inf_delta": carry_state_deltas["x_inf"],
        "hebbian_memory_max_abs": hebbian_max,
        "g_inf_max_abs": g_inf_max,
        "reset_changes_state_delta": reset_delta,
        "scored_predictions": scored,
        "gradients_finite": gradients_finite,
        "full_episode_reaches_initial_state": True,
        "initial_state_gradient_max_abs": initial_state_gradient,
        "weights_unchanged": True,
        "state_sha256": before,
        "seconds_per_episode_estimate": time.perf_counter() - started,
        "planning_note": "single-episode timing is a planning estimate, not a success condition",
        **batch.diversity(),
    }
    if device == "cuda":
        peak = int(torch.cuda.max_memory_allocated())
        budget = int(settings["vram_budget_gib"] * 2**30)
        result["peak_allocated_bytes"] = peak
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        result["vram_budget_bytes"] = budget
        result["within_vram_budget"] = peak <= budget
        if peak > budget:
            raise ValueError(
                f"preflight peak {peak} bytes exceeds declared {budget} bytes; "
                "lower the batch or amend the protocol"
            )
    return result


# ---------------------------------------------------------------------- fit


def stage_contract(settings: dict, *, status: str, updates: int, skipped: int,
                   full_history_updates: int, continued: bool = False) -> dict:
    """The label contract for one finished (or interrupted) run.

    Kept pure so the interim/confirmation distinction is testable without
    running thousands of optimizer steps. An interim checkpoint can complete its
    own declared contract but can never set ``scientific_fit``, authorise final
    scoring or claim the confirmation budget.
    """
    stage = scientific_stage(settings)
    completed = status == "completed" and updates == settings["iterations"]
    met = bool(
        settings["scientific_mode"] and status == "completed" and completed
        and skipped == 0 and updates == settings["iterations"]
        and full_history_updates == updates
    )
    scientific_fit = bool(met and stage == STANDARD_STAGE and not continued)
    return {
        "stage": stage,
        "scientific_status": scientific_status(settings, continued=continued),
        "stage_contract_met": met,
        "scientific_fit": scientific_fit,
        "interim_contract_met": (bool(met) if stage == INTERIM_STAGE else None),
        "interim_preliminary": stage == INTERIM_STAGE,
        "capability_pass": False,
        "final_scoring_authorized": scientific_fit,
        "confirmation_required": (
            "standard stage confirmation budget; a preliminary interim fit does not satisfy it"
            if stage == INTERIM_STAGE else None
        ),
    }


def continuation_checks(prior_config: dict, prior_record: dict, prior_update: int,
                        new_config: dict, new_record: dict) -> dict:
    """Audit an interim -> standard continuation before any state is restored.

    A continuation keeps weights, optimizer moments, RNG streams and the
    counter-addressed episode position, so only the budget and the declared
    checkpoints may change. Anything else - topology IDs, seeds, arm, sequence
    length, batch, schedule, manifest, code - means a different experiment and
    is refused. The one permitted stage change is an explicit
    ``interim -> standard`` extension at exactly the standard budget, with every
    prior checkpoint retained and the upstream absolute schedule unchanged.

    It is nevertheless **not** the frozen from-scratch confirmation arm: its
    initialisation is the prior checkpoint rather than the seeded initial
    weights. Continuations are therefore labelled ``scientific_continuation``
    and never set ``scientific_fit``.
    """
    if not isinstance(prior_update, int) or prior_update < 0:
        raise ValueError("continuation source has no valid update counter")
    if prior_record["manifest_sha256"] != new_record["manifest_sha256"]:
        raise ValueError("continuation changed the frozen cohort manifest")
    if prior_record["code_sha256"] != new_record["code_sha256"]:
        raise ValueError("continuation changed trainer/graph/native code")
    if prior_record.get("upstream") != new_record.get("upstream"):
        raise ValueError("continuation changed the pinned upstream checkout")
    if prior_config["arm"] != new_config["arm"]:
        raise ValueError("continuation changed the study arm")
    if prior_config["training_graph_ids"] != new_config["training_graph_ids"]:
        raise ValueError("continuation changed the training graph IDs")
    prior_settings, new_settings = prior_config["training"], new_config["training"]
    prior_stage = scientific_stage(prior_settings)
    new_stage = scientific_stage(new_settings)
    stage_change = prior_stage != new_stage
    allowed = set(CONTINUATION_MUTABLE_KEYS)
    if stage_change:
        if (prior_stage, new_stage) != CONTINUATION_STAGE_TRANSITION:
            raise ValueError(
                "continuation may only change stage "
                f"{CONTINUATION_STAGE_TRANSITION[0]} -> {CONTINUATION_STAGE_TRANSITION[1]}; "
                f"got {prior_stage} -> {new_stage}"
            )
        allowed.add("scientific_stage")

    def effective(settings, key):
        """Compare declared behaviour, so an explicit default equals absence."""
        if key == "scientific_stage":
            return scientific_stage(settings)
        if key == "route_policy":
            return settings.get(key, DEFAULT_ROUTE_POLICY)
        return settings.get(key)

    differences = {
        key: (effective(prior_settings, key), effective(new_settings, key))
        for key in sorted(set(prior_settings) | set(new_settings))
        if effective(prior_settings, key) != effective(new_settings, key)
    }
    illegal = sorted(set(differences) - allowed)
    if illegal:
        raise ValueError(
            f"continuation may only change {sorted(allowed)}; changed {illegal}"
        )
    if new_settings["iterations"] <= prior_settings["iterations"]:
        raise ValueError("continuation must increase the update budget")
    if not set(prior_settings["save_updates"]) <= set(new_settings["save_updates"]):
        raise ValueError("continuation must retain every prior declared checkpoint")
    if prior_update not in set(prior_settings["save_updates"]):
        raise ValueError("continuation source is not a declared prior checkpoint")
    if prior_update > new_settings["iterations"]:
        raise ValueError("continuation checkpoint lies beyond the new update budget")
    if prior_config["evaluation"] != new_config["evaluation"]:
        raise ValueError("continuation changed the evaluation seed")
    if stage_change:
        # The interim stage is already pinned to exactly its declared update
        # count; the standard stage must be the full confirmation budget. The
        # upstream schedule is absolute in iteration number for a fixed episode
        # length, so extending the counter does not change the loss weights,
        # eta, lambda or learning rate at any shared update. This is checked
        # executably in the QA suite.
        if prior_settings["iterations"] != INTERIM_UPDATES:
            raise ValueError(
                f"interim continuation source must have exactly {INTERIM_UPDATES} updates"
            )
        if new_settings["iterations"] != FULL_UPDATES_MIN:
            raise ValueError(
                f"standard continuation must target exactly {FULL_UPDATES_MIN} updates"
            )
        if prior_settings["schedule"] != "upstream" or new_settings["schedule"] != "upstream":
            raise ValueError("a stage-extending continuation requires the upstream schedule")
    schedule_unchanged = (
        prior_settings["schedule"] == new_settings["schedule"]
        and not stage_change
        or (
            stage_change
            and prior_settings["schedule"] == new_settings["schedule"] == "upstream"
        )
    )
    if not schedule_unchanged:
        raise ValueError("continuation changed the parameter schedule")
    return {
        "status": "eligible",
        "prior_stage": prior_stage,
        "new_stage": new_stage,
        "stage_transition": f"{prior_stage}->{new_stage}" if stage_change else None,
        "prior_update": prior_update,
        "prior_iterations": prior_settings["iterations"],
        "new_iterations": new_settings["iterations"],
        "changed_settings": sorted(differences),
        "schedule_unchanged": schedule_unchanged,
        "reaches_standard_budget": bool(
            stage_change and new_settings["iterations"] == FULL_UPDATES_MIN
        ),
        "initialization": "continued_from_prior_checkpoint",
        "weights_and_optimizer": "restored_from_prior_training_state",
        "data_stream": "counter-addressed episodes replayed exactly to the prior stream position",
        "rng": (
            "Python, NumPy and torch CPU/CUDA RNG states are captured in "
            "training-state.pt and restored on resume, so any stochastic module "
            "operation continues the same stream; the counter-addressed episode "
            "stream is deterministic and needs no RNG state"
        ),
        "claim_boundary": (
            "a continuation is not the frozen from-scratch confirmation arm; "
            "it is labelled scientific_continuation and cannot set scientific_fit"
        ),
    }


def schedule_audit(prior_config: dict, new_config: dict,
                   repo: str | Path = adapter.DEFAULT_REPO,
                   updates: tuple[int, ...] = (0, 1, 100, 1000, 2500, 4999)) -> dict:
    """Executable proof that the used absolute schedule is budget-independent.

    Building ``params`` with a different ``iterations`` changes only
    ``train_it``; the upstream ``parameter_iteration`` values the trainer
    consumes (eta, lambda, p2g offset, learning rate, loss weights) do not
    change. ``walk_length_center`` is excluded because fixed-length episodes
    never use it. This is what makes a continuation an extension rather than a
    schedule change.
    """
    torch = adapter.require_torch()
    upstream = adapter.load_upstream(Path(repo))
    prior = adapter.default_params(upstream, iterations=prior_config["training"]["iterations"])
    new = adapter.default_params(upstream, iterations=new_config["training"]["iterations"])
    prior["n_actions"] = new["n_actions"] = len(worlds.ACTION_LABELS) - 1
    if prior_config["training"]["schedule"] == "pilot":
        adapter.apply_pilot_schedule(prior, prior_config["training"]["iterations"])
    if new_config["training"]["schedule"] == "pilot":
        adapter.apply_pilot_schedule(new, new_config["training"]["iterations"])
    checked = []
    for update in updates:
        if update >= min(prior_config["training"]["iterations"], new_config["training"]["iterations"]):
            continue
        left = upstream.parameters.parameter_iteration(update, prior)
        right = upstream.parameters.parameter_iteration(update, new)
        same = (
            all(abs(float(left[index]) - float(right[index])) < 1e-12 for index in range(4))
            and bool(torch.equal(left[5], right[5]))
        )
        checked.append({"update": int(update), "same": same})
    return {
        "status": "passed" if checked and all(item["same"] for item in checked) else "failed",
        "checked_updates": checked,
        "excluded": ["walk_length_center (unused by fixed-length episodes)"],
        "train_it": {
            "prior": prior["train_it"], "new": new["train_it"],
            "note": "iteration-derived value; only walk_length_center reads it",
        },
    }


def _load_resume_state(torch, prepared: Path, record: dict, config: dict | None = None,
                       *, continuation: tuple[dict, dict] | None = None,
                       repo: str | Path = adapter.DEFAULT_REPO):
    out = prepared / "training"
    ledger_path = out / "checkpoints.json"
    if not ledger_path.is_file():
        raise ValueError("resume ledger checkpoints.json is absent or incomplete")
    entries = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("resume ledger is empty or invalid")
    # The ledger is the authority: its last entry selects the resume target.
    # The training-state.pt alias is a convenience pointer and never selects an
    # update, so an alias that lagged or raced ahead of an interrupted commit
    # cannot strand or misdirect a resume.
    entries = sorted(entries, key=lambda item: int(item["update"]))
    if len({int(item["update"]) for item in entries}) != len(entries):
        raise ValueError("resume ledger contains duplicate update entries")
    entry = entries[-1]
    update = entry.get("update")
    if type(update) is not int or update < 0:
        raise ValueError("resume ledger entry has no valid update counter")
    numbered = {
        "training_state": out / entry["training_state"],
        "checkpoint": out / entry["checkpoint"],
        "params": out / entry["params"],
    }
    for key, path in numbered.items():
        if not path.is_file():
            raise ValueError(f"resume evidence is incomplete: missing {path.name}")
        expected = entry.get(f"{key}_sha256")
        if expected is None or sha(path) != expected:
            raise ValueError(f"resume evidence digest mismatch: {path.name}")
    # Aliases must be byte-identical copies of the committed numbered files.
    # A crash between the ledger commit and an alias copy can leave them stale
    # or missing, so recover them here from the hash-verified numbered files
    # instead of stranding automatic resume.
    for alias_name, key in (
        ("training-state.pt", "training_state"),
        ("checkpoint.pt", "checkpoint"),
        ("params.pt", "params"),
    ):
        alias_path = out / alias_name
        if not alias_path.is_file() or alias_path.read_bytes() != numbered[key].read_bytes():
            _atomic_copy(numbered[key], alias_path)
    state = adapter._load_torch_file(torch, numbered["training_state"])
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        raise ValueError("invalid numbered training-state schema")
    if state.get("update") != update or state.get("updates") != update:
        raise ValueError("numbered training state counters disagree")
    if state.get("weights_committed", True) is not True:
        raise ValueError("numbered training state is not a committed checkpoint")
    continuation_info = None
    if state.get("config_sha256") != record["config_sha256"]:
        if continuation is None:
            raise ValueError("resume config changed")
        prior_config, prior_record = continuation
        if state.get("config_sha256") != prior_record["config_sha256"]:
            raise ValueError("continuation source is not the run that produced this training state")
        if config is None:
            raise ValueError("a continuation resume needs the continuing configuration")
        continuation_info = continuation_checks(
            prior_config, prior_record, int(state.get("update", -1)), config, record,
        )
        continuation_info["schedule_audit"] = schedule_audit(prior_config, config, repo)
        if continuation_info["schedule_audit"]["status"] != "passed":
            raise ValueError("continuation schedule audit failed")
    if state.get("manifest_sha256") != record["manifest_sha256"]:
        raise ValueError("resume graph manifest changed")
    if state.get("code_sha256") != record["code_sha256"]:
        raise ValueError("resume code changed")
    if state.get("arm") != record["arm"] or state.get("training_seed") != record["training_seed"]:
        raise ValueError("resume arm/seed changed")
    if config is not None and "training" in config:
        settings = config["training"]
        stream = state.get("stream")
        if not isinstance(stream, dict) or type(stream.get("episode")) is not int:
            raise ValueError("resume state has no valid episode position")
        if stream["episode"] < 0 or stream["episode"] % settings["batch_size"]:
            raise ValueError("resume episode position is not a batch boundary")
        if settings["bptt"] == "episode" and stream["episode"] != update * settings["batch_size"]:
            raise ValueError(
                "resume episode position disagrees with the episode-mode update counter"
            )
    checkpoint = numbered["checkpoint"]
    return state, checkpoint, continuation_info


def train(prepared: Path, repo: Path = adapter.DEFAULT_REPO, approved_config_sha256: str | None = None,
          *, resume: bool = False, continuation_from: str | Path | None = None) -> dict:
    prepared = Path(prepared)
    config, graphs, record = load_prepared(prepared, repo)
    settings = config["training"]
    stage = scientific_stage(settings)
    if approved_config_sha256 != record["config_sha256"]:
        raise ValueError("training blocked until the exact prepared configuration digest is approved")
    continuation_source = None
    if continuation_from is not None:
        if not resume:
            raise ValueError("a continuation must also be a resume")
        prior_config, _prior_graphs, prior_record = load_prepared(Path(continuation_from), Path(repo))
        continuation_source = (prior_config, prior_record)
    out = prepared / "training"
    # Resolve and attest the device before allocating a model, so a forbidden
    # scientific environment produces a clean refusal artifact and zero updates.
    torch_probe = adapter.require_torch()
    runtime = runtime_attestation(torch_probe, settings["device"], allow_unavailable=True)
    blockers = scientific_blockers(config, runtime)
    if blockers:
        refusal = {
            "schema": SUMMARY_SCHEMA,
            "status": "refused",
            "arm": config["arm"],
            "config_sha256": record["config_sha256"],
            "optimizer_steps": 0,
            "scientific_mode": True,
            "scientific_stage": stage,
            "scientific_status": scientific_status(settings),
            "scientific_fit": False,
            "interim_contract_met": False if stage == INTERIM_STAGE else None,
            "capability_pass": False,
            "reasons": blockers,
            "runtime": runtime,
            "claims": CLAIMS,
            "created_utc": now(),
        }
        write_json(prepared / "refusal.json", refusal)
        raise RefusalError("scientific fit refused: " + "; ".join(blockers))
    if runtime["device"] == "unavailable":
        raise ValueError(f"requested device is unavailable: {runtime.get('reason')}")

    torch, upstream, params, model = initialize(config, repo)
    device = runtime["device"]
    if device == "cuda":
        adapter.prepare_cuda_model(model, torch)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr_max"])
    initial_state_sha256 = state_digest(model)
    initial_params_sha256 = hashlib.sha256(repr(sorted(params.items())).encode()).hexdigest()

    resumed_from = None
    metrics_archive = None
    prior_seconds = 0.0
    ledger = UpdateLedger(settings["iterations"])
    batches = EpisodeBatches(config, graphs)
    stream = {"episode": 0, "chunk": 0, "visited": None, "carry": None}
    previous = None
    visited = None
    chunks_consumed = 0
    full_history_updates = 0
    lineage: list[str] = []
    continuation_info = None
    if resume:
        state, checkpoint, continuation_info = _load_resume_state(
            torch, prepared, record, config, continuation=continuation_source, repo=repo,
        )
        model.load_state_dict(adapter._load_torch_file(torch, checkpoint), strict=True)
        optimizer.load_state_dict(state["optimizer"])
        _rng_restore(state.get("rng"), torch, device)
        ledger.updates = int(state.get("updates", state["update"]))
        ledger.skipped = int(state.get("skipped", 0))
        chunks_consumed = int(state.get("chunks_consumed", 0))
        full_history_updates = int(state.get("full_history_updates", ledger.updates))
        lineage = list(state.get("lineage", []))
        if continuation_info is not None:
            # Record the chain explicitly so a continuation can never be
            # mistaken for a from-scratch confirmation arm.
            lineage = lineage + [state["config_sha256"]]
        stream = state["stream"]
        if (type(stream.get("episode")) is not int
                or stream["episode"] < 0
                or stream["episode"] % settings["batch_size"]):
            raise ValueError("resume stream is not aligned to the batch boundary")
        batches.replay_to(int(stream["episode"]))
        if stream["carry"] is not None:
            # Mid-episode resume: replay prior groups, then materialise and
            # commit the current independent-lane group before consuming its
            # carried batched state from the checkpoint.
            batches.batch(
                int(stream["episode"]) // settings["batch_size"],
                commit=True, cache=True,
            )
            previous = _restore_carry(stream["carry"], upstream, device)
            visited = stream["visited"]
        prior_summary = out / "summary.json"
        if prior_summary.is_file():
            earlier = json.loads(prior_summary.read_text(encoding="utf-8"))
            # Keep the original initialization digest so pair audit still sees
            # the run's true starting weights after a resume.
            initial_state_sha256 = earlier.get("initial_state_sha256", initial_state_sha256)
            initial_params_sha256 = earlier.get("initial_params_sha256", initial_params_sha256)
            prior_seconds = float(earlier.get("seconds", 0.0))
        resumed_from = ledger.updates
        out.mkdir(parents=True, exist_ok=True)
        metrics_archive = archive_and_trim_metrics(out, ledger.updates)
        mode = "a"
    else:
        out.mkdir(parents=True, exist_ok=False)
        mode = "x"

    stop = {"signal": None}

    def _handle(signum, _frame):
        stop["signal"] = int(signum)

    for name in ("SIGINT", "SIGTERM"):
        value = getattr(signal, name, None)
        if value is not None:
            try:
                signal.signal(value, _handle)
            except (ValueError, OSError):
                pass

    started = time.perf_counter()
    failure = None
    status = "running"
    progress = {
        "status": "running",
        "arm": config["arm"],
        "optimizer_steps": ledger.updates,
        "requested_updates": settings["iterations"],
        "started_utc": now(),
        "resumed_from": resumed_from,
    }
    write_json(out / "progress.json", progress)
    updates_per_episode = settings["episode_length"] // settings["rollout"]
    last_metrics: dict | None = None

    def _snapshot(update: int, *, stopping: bool = False, committed: bool = True) -> None:
        nonlocal stream, previous, visited
        state = {
            "updates": ledger.updates,
            "skipped": ledger.skipped,
            # False after a raised optimizer.step: weights may be partially
            # mutated, so resume must fall back to the last committed state.
            "weights_committed": committed,
            "stream": {
                "episode": stream["episode"],
                "chunk": stream["chunk"],
                "visited": visited,
                "carry": _serialise_carry(previous),
            },
            "chunks_consumed": chunks_consumed,
            "full_history_updates": full_history_updates,
            "rng": _rng_snapshot(torch, device),
            "scientific_stage": stage,
            "lineage": list(lineage),
            "continuation": continuation_info is not None,
            "stopping": stopping,
        }
        if not committed:
            # A failed optimizer step is diagnostic only. It never becomes a
            # numbered checkpoint, never aliases training-state.pt and never
            # enters checkpoints.json.
            save_failure_snapshot(model, optimizer, out, update, record=record, state=state)
            return
        save_artifacts(
            model, optimizer, out, update, record=record, state=state,
            verify={"next_episode": stream["episode"], "next_chunk": stream["chunk"],
                    "batch_size": settings["batch_size"],
                    "loss": None if last_metrics is None else last_metrics["loss"],
                    "scored_predictions": None if last_metrics is None else last_metrics["scored_predictions"],
                    "history_steps": None if last_metrics is None else last_metrics["history_steps"]},
        )

    failure_snapshot_taken = False
    try:
        with (out / "metrics.jsonl").open(mode, encoding="utf-8") as metrics:
            while ledger.updates < settings["iterations"]:
                if stop["signal"] is not None:
                    status = "interrupted"
                    break
                if settings["max_seconds"] and time.perf_counter() - started >= settings["max_seconds"]:
                    status = "partial_time_limit"
                    break
                lr, weights = scheduled_weights(upstream, model, params, ledger.updates)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                if settings["bptt"] == "episode":
                    batch = batches.batch(
                        stream["episode"] // settings["batch_size"],
                        commit=False, cache=False,
                    )
                    result = _episode_update(
                        torch, model, optimizer, batch,
                        batch_size=settings["batch_size"], rollout=settings["rollout"],
                        weights=weights, device=device, ledger=ledger,
                        step_grad_norm=(ledger.updates % settings["log_every"] == 0),
                    )
                    # Commit only after a real scored optimizer step, so a
                    # failed batch is retried with the same assignments.
                    batches.commit_batch(batch)
                    chunks_consumed += updates_per_episode
                    full_history_updates += 1
                    stream = {
                        "episode": stream["episode"] + settings["batch_size"],
                        "chunk": 0, "visited": None, "carry": None,
                    }
                    previous, visited = None, None
                    last_metrics = result
                    lags = [plan.revisit_lag_mean for plan in batch.plans
                            if plan.revisit_lag_mean is not None]
                    metric = {
                        "update": ledger.updates,
                        "episode": batch.indices[0],
                        "episodes": list(batch.indices),
                        "graph_id": batch.graph_ids[0],
                        "graph_ids": list(batch.graph_ids),
                        "batch_size": batch.batch_size,
                        "distinct_graphs": result["distinct_graphs"],
                        "distinct_routes": result["distinct_routes"],
                        "distinct_cue_sequences": result["distinct_cue_sequences"],
                        "loss": result["loss"],
                        "scored_predictions": result["scored_predictions"],
                        "history_steps": result["history_steps"],
                        "grad_norm": result["grad_norm"],
                        "seconds": result["seconds"],
                        "lr": float(lr),
                        "eta": float(model.hyper["eta"]),
                        "lambda": float(model.hyper["lambda"]),
                        "revisit_lag_mean": (sum(lags) / len(lags)) if lags else None,
                        "revisit_lag_max": max(
                            (plan.revisit_lag_max for plan in batch.plans
                             if plan.revisit_lag_max is not None), default=None,
                        ),
                        "credit_assignment": "full_episode_bptt",
                    }
                else:
                    if previous is None or stream["chunk"] == 0:
                        batch = batches.batch(
                            stream["episode"] // settings["batch_size"],
                            commit=True, cache=True,
                        )
                        visited = [[False] * batch.n_nodes for _ in range(batch.batch_size)]
                        previous = None
                    else:
                        batch = batches.batch(
                            stream["episode"] // settings["batch_size"], cache=True,
                        )
                    result = _rollout_update(
                        torch, model, optimizer, batch,
                        chunk_index=stream["chunk"], batch_size=settings["batch_size"],
                        rollout=settings["rollout"], weights=weights, device=device,
                        ledger=ledger, visited=visited, previous=previous,
                        step_grad_norm=(ledger.updates % settings["log_every"] == 0),
                    )
                    chunks_consumed += 1
                    visited, previous = result["visited"], result["previous"]
                    if result["history_steps"] >= FULL_HISTORY_MIN:
                        full_history_updates += 1
                    next_chunk = result["next_chunk"]
                    finished = next_chunk * settings["rollout"] >= settings["episode_length"]
                    stream = {
                        "episode": stream["episode"] + settings["batch_size"] if finished
                                   else stream["episode"],
                        "chunk": 0 if finished else next_chunk,
                        "visited": None if finished else visited,
                        "carry": None if finished else _serialise_carry(previous),
                    }
                    if finished:
                        visited, previous = None, None
                        batches.clear_cache()
                    last_metrics = result
                    metric = {
                        "update": ledger.updates,
                        "episode": batch.indices[0],
                        "episodes": list(batch.indices),
                        "chunk": result["next_chunk"] - 1,
                        "graph_id": batch.graph_ids[0],
                        "graph_ids": list(batch.graph_ids),
                        "batch_size": batch.batch_size,
                        "distinct_graphs": result["distinct_graphs"],
                        "distinct_routes": result["distinct_routes"],
                        "distinct_cue_sequences": result["distinct_cue_sequences"],
                        "loss": result["loss"],
                        "scored_predictions": result["scored_predictions"],
                        "history_steps": result["history_steps"],
                        "grad_norm": result["grad_norm"],
                        "stepped": bool(result["stepped"]),
                        "seconds": result["seconds"],
                        "lr": float(lr),
                        "credit_assignment": "truncated_bptt_rollout",
                    }
                _vram_guard(torch, settings)
                metrics.write(json.dumps(metric) + "\n")
                metrics.flush()
                if ledger.updates % settings["log_every"] == 0 or ledger.updates == settings["iterations"]:
                    print(json.dumps({"update": ledger.updates, "loss": metric["loss"],
                                      "scored": metric["scored_predictions"],
                                      "history": metric["history_steps"],
                                      "skipped": ledger.skipped}), flush=True)
                progress.update({
                    "status": "running",
                    "optimizer_steps": ledger.updates,
                    "skipped_zero_scored_updates": ledger.skipped,
                    "last_loss": metric["loss"],
                    "last_history_steps": metric["history_steps"],
                    "seconds": time.perf_counter() - started,
                    "stream": {"episode": stream["episode"], "chunk": stream["chunk"]},
                })
                write_json(out / "progress.json", progress)
                if ledger.updates in settings["save_updates"]:
                    _snapshot(ledger.updates)
    except BaseException as exc:  # noqa: BLE001 - preserve the real failure
        failure = {"type": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc()}
        status = "failed"
        try:
            _snapshot(ledger.updates, stopping=True, committed=False)
            failure_snapshot_taken = True
        except Exception:  # pragma: no cover - best effort on a failing run
            pass
    finally:
        for name in ("SIGINT", "SIGTERM"):
            value = getattr(signal, name, None)
            if value is not None:
                try:
                    signal.signal(value, signal.SIG_DFL)
                except (ValueError, OSError):
                    pass
        # Only a clean interrupt/time-limit earns a committed snapshot. A
        # failure must never overwrite the last committed state or publish the
        # partially mutated weights as resumable.
        if failure is None and status in ("interrupted", "partial_time_limit"):
            _snapshot(ledger.updates, stopping=True, committed=True)

    completed = ledger.updates == settings["iterations"] and failure is None
    if failure is None and not completed and status == "running":
        status = "partial"
    elif failure is None and completed:
        status = "completed"
    continued = continuation_info is not None
    # A preliminary interim fit is preliminary by declaration, and a
    # continuation is not the frozen from-scratch arm. Neither can ever set the
    # standard scientific_fit flag, even when the run completes perfectly.
    contract = stage_contract(
        settings, status=status, updates=ledger.updates, skipped=ledger.skipped,
        full_history_updates=full_history_updates, continued=continued,
    )
    if device == "cuda":
        runtime["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        runtime["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": status,
        "failure": failure,
        "arm": config["arm"],
        "training_seed": settings["seed"],
        "completed_utc": now(),
        "config_sha256": record["config_sha256"],
        "manifest_sha256": record["manifest_sha256"],
        "graphs_sha256": record["graphs_sha256"],
        "graph_content_sha256": record["graph_content_sha256"],
        "optimizer_steps": ledger.updates,
        "requested_updates": settings["iterations"],
        "skipped_zero_scored_updates": ledger.skipped,
        "episodes": batches.episodes,
        "chunks_consumed": chunks_consumed,
        "batch_size": settings["batch_size"],
        "episode_length": settings["episode_length"],
        "rollout": settings["rollout"],
        "bptt": settings["bptt"],
        "credit_assignment": ("full_episode_bptt" if settings["bptt"] == "episode"
                              else "truncated_bptt_rollout"),
        "route_policy": settings.get("route_policy", DEFAULT_ROUTE_POLICY),
        "independent_lane_batches": (
            ledger.updates if settings["bptt"] == "episode" else None
        ),
        "full_history_updates": full_history_updates,
        "observed_transitions": (
            ledger.updates * settings["batch_size"] * settings["episode_length"]
            if settings["bptt"] == "episode"
            else chunks_consumed * settings["batch_size"] * settings["rollout"]
        ),
        "initial_state_sha256": initial_state_sha256,
        "initial_params_sha256": initial_params_sha256,
        "final_state_sha256": state_digest(model),
        "weights_committed": failure is None,
        "failure_snapshot_taken": failure_snapshot_taken,
        "resume_authority": (
            "training-state.pt from the last committed update"
            if failure is not None else
            "training-state.pt aliases the latest declared checkpoint"
        ),
        "graph_episodes": dict(batches.graph_counts),
        "action_counts": {str(key): value for key, value in sorted(batches.action_counts.items())},
        "unique_route_sequences": len(batches.route_digests),
        "unique_cue_sequences": len(batches.cue_digests),
        "runtime": runtime,
        "scientific_mode": settings["scientific_mode"],
        "scientific_stage": stage,
        "scientific_status": contract["scientific_status"],
        "stage_contract_met": contract["stage_contract_met"],
        "scientific_fit": contract["scientific_fit"],
        "interim_contract_met": contract["interim_contract_met"],
        "interim_preliminary": contract["interim_preliminary"],
        "capability_pass": False,
        "final_scoring_authorized": contract["final_scoring_authorized"],
        "confirmation_required": contract["confirmation_required"],
        "rng_state_digest": _rng_digest(_rng_snapshot(torch, device)),
        "continuation": continuation_info,
        "lineage": lineage,
        "claims": CLAIMS,
        "selection": record["selection"],
        "resumed_from": resumed_from,
        "metrics_discarded_on_resume": metrics_archive,
        "interrupt_signal": stop["signal"],
        "seconds": time.perf_counter() - started,
        "prior_seconds": prior_seconds,
        "seconds_total": prior_seconds + (time.perf_counter() - started),
        "python": sys.version,
        "torch": str(torch.__version__),
    }
    write_json(out / "summary.json", summary)
    write_json(out / "progress.json", summary)
    if failure is not None:
        raise RuntimeError(f"training failed; partial artifacts retained: {failure}")
    return summary


# ---------------------------------------------------------------- downstream


def load_fitted(prepared: Path, repo: Path = adapter.DEFAULT_REPO, *, update: int | None = None):
    """Return a frozen CPU model and verified evidence for an evaluator.

    The default is the declared final endpoint. A caller may request one of the
    declared intermediate checkpoints for validation-only curves; final-test
    selection remains outside this function.
    """
    config, _graphs, record = load_prepared(Path(prepared), Path(repo))
    out = Path(prepared) / "training"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] not in ("completed", "interrupted"):
        raise ValueError("evaluation requires a completed or explicitly interrupted fit")
    update = summary["optimizer_steps"] if update is None else int(update)
    entries = json.loads((out / "checkpoints.json").read_text(encoding="utf-8"))
    entry = next((item for item in entries if item["update"] == update), None)
    if entry is None:
        raise ValueError(f"checkpoint {update} is not declared in checkpoints.json")
    checkpoint = out / entry["checkpoint"]
    params_path = out / entry["params"]
    if sha(checkpoint) != entry["checkpoint_sha256"] or sha(params_path) != entry["params_sha256"]:
        raise ValueError("checkpoint or parameter digest mismatch")
    if summary["config_sha256"] != record["config_sha256"] or summary["manifest_sha256"] != record["manifest_sha256"]:
        raise ValueError("summary provenance mismatch")
    torch = adapter.require_torch()
    torch.set_num_threads(config["training"]["threads"])
    upstream = adapter.load_upstream(Path(repo))
    model, params, info = adapter.load_model_checkpoint(
        upstream, checkpoint, params_path, seed=config["evaluation"]["seed"],
    )
    if params["n_actions"] != len(worlds.ACTION_LABELS) - 1 or not params["has_static_action"]:
        raise ValueError("checkpoint native action contract mismatch")
    stage_used = summary.get("scientific_stage", STANDARD_STAGE)
    return model, params, {
        **summary, **info, "requested_checkpoint": update,
        "scientific_stage": stage_used,
        "scientific_status": summary.get("scientific_status", NON_SCIENTIFIC_STATUS),
        "interim_preliminary": stage_used == INTERIM_STAGE,
        # Downstream scoring may use these weights for validation/development
        # curves, but only a completed standard scientific fit authorises final
        # scoring. Interim and continuation runs never do.
        "final_scoring_authorized": bool(
            summary.get("final_scoring_authorized", False)
        ),
        "capability_pass": False,
    }


PAIR_FIELDS = (
    "iterations", "batch_size", "episode_length", "rollout", "bptt",
    "schedule", "seed", "cue_seed", "route_seed", "family_seed",
    "route_policy",
)


def lane_parity_audit(single_config: dict, single_manifest: dict,
                      diverse_config: dict, diverse_manifest: dict,
                      *, updates: int = 8) -> dict:
    """Executable lane independence and cross-arm parity audit.

    Every batch must contain ``batch_size`` genuinely distinct episodes in both
    arms, and the same global episode index must carry the same cue assignment
    and length in both arms. Only graph choice, route and topology may differ,
    because that is the declared treatment.
    """
    if updates < 1:
        raise ValueError("lane parity audit needs at least one update")
    left = EpisodeBatches(single_config, single_manifest)
    right = EpisodeBatches(diverse_config, diverse_manifest)
    batch_size = single_config["training"]["batch_size"]
    mismatches = 0
    min_routes = batch_size
    min_cues = batch_size
    assignments = 0
    lane_counts_match = True
    for update in range(updates):
        left_batch = left.batch(update)
        right_batch = right.batch(update)
        lane_counts_match = lane_counts_match and (
            left_batch.batch_size == batch_size == right_batch.batch_size
        )
        left_div, right_div = left_batch.diversity(), right_batch.diversity()
        min_routes = min(min_routes, left_div["distinct_routes"], right_div["distinct_routes"])
        min_cues = min(min_cues, left_div["distinct_cue_sequences"],
                       right_div["distinct_cue_sequences"])
        for left_plan, right_plan in zip(left_batch.plans, right_batch.plans):
            assignments += 1
            if (
                left_plan.cue_assignment != right_plan.cue_assignment
                or left_plan.history_steps != right_plan.history_steps
                or left_plan.index != right_plan.index
            ):
                mismatches += 1
    checks = {
        "same_route_policy": (
            single_config["training"].get("route_policy", DEFAULT_ROUTE_POLICY)
            == diverse_config["training"].get("route_policy", DEFAULT_ROUTE_POLICY)
        ),
        "same_data_seeds": all(
            single_config["training"][field] == diverse_config["training"][field]
            for field in ("cue_seed", "route_seed", "family_seed")
        ),
        "batch_lane_count": lane_counts_match,
        "distinct_routes_every_batch": min_routes == batch_size,
        "distinct_cue_sequences_every_batch": min_cues == batch_size,
        "cross_arm_cue_parity": mismatches == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "updates": updates,
        "lane_assignments_compared": assignments,
        "cross_arm_mismatches": mismatches,
        "minimum_distinct_routes": min_routes,
        "minimum_distinct_cue_sequences": min_cues,
        "route_policy": single_config["training"].get("route_policy", DEFAULT_ROUTE_POLICY),
        "note": "routes differ by topology; cue maps, indices, lengths and seeds do not",
    }


def pair_value(config, field):
    if field == "route_policy":
        return config["training"].get(field, DEFAULT_ROUTE_POLICY)
    return config["training"][field]


def pairing_checks(single_config: dict, diverse_config: dict,
                   single_record: dict, diverse_record: dict, *,
                   single_summary: dict | None = None,
                   diverse_summary: dict | None = None,
                   lanes_manifest: dict | None = None,
                   lane_updates: int = 8) -> dict:
    """Pure, executable single/diverse pairing contract.

    The file-loading wrappers below only gather inputs; every decision lives
    here so it can be exercised without a feasible frozen cohort.
    """
    if single_config["arm"] != "single" or diverse_config["arm"] != "diverse":
        raise ValueError("pair audit needs one 'single' and one 'diverse' prepared run")
    checks = {
        field: pair_value(single_config, field) == pair_value(diverse_config, field)
        for field in PAIR_FIELDS
    }
    checks["one_vs_many_graphs"] = (
        len(single_config["training_graph_ids"]) == 1
        and len(diverse_config["training_graph_ids"]) >= 2
        and len(set(diverse_config["training_graph_ids"])) == len(diverse_config["training_graph_ids"])
    )
    checks["graph_manifest"] = single_record["manifest_sha256"] == diverse_record["manifest_sha256"]
    lanes = None
    if lanes_manifest is not None:
        lanes = lane_parity_audit(
            single_config, lanes_manifest, diverse_config, lanes_manifest,
            updates=lane_updates,
        )
        checks["lane_parity"] = lanes["passed"]
    if single_summary is not None and diverse_summary is not None:
        checks["initial_weights"] = (
            single_summary["initial_state_sha256"] == diverse_summary["initial_state_sha256"]
        )
        checks["updates_completed"] = (
            single_summary["optimizer_steps"] == single_summary["requested_updates"]
            == diverse_summary["optimizer_steps"] == diverse_summary["requested_updates"]
        )
        checks["observed_transitions"] = (
            single_summary["observed_transitions"] == diverse_summary["observed_transitions"]
        )
    return {"passed": all(checks.values()), "checks": checks, "lane_parity": lanes,
            "pairing_note": "graph-driven routes differ by topology; cue maps, seeds and budgets do not"}


def pair_audit(single_prepared: Path, diverse_prepared: Path) -> dict:
    """Load two frozen prepared runs and apply the pure pairing contract."""
    single_config, single_manifest, single_record = load_prepared(Path(single_prepared))
    diverse_config, diverse_manifest, diverse_record = load_prepared(Path(diverse_prepared))
    single_summary_path = Path(single_prepared) / "training" / "summary.json"
    diverse_summary_path = Path(diverse_prepared) / "training" / "summary.json"
    single_summary = (
        json.loads(single_summary_path.read_text(encoding="utf-8"))
        if single_summary_path.is_file() else None
    )
    diverse_summary = (
        json.loads(diverse_summary_path.read_text(encoding="utf-8"))
        if diverse_summary_path.is_file() else None
    )
    return pairing_checks(
        single_config, diverse_config, single_record, diverse_record,
        single_summary=single_summary, diverse_summary=diverse_summary,
        lanes_manifest=single_manifest,
    )


PAIR_DESIGN_FIELDS = (
    "iterations", "batch_size", "episode_length", "rollout", "bptt", "schedule",
    "cue_seed", "route_seed", "family_seed", "route_policy",
)


def cohort_pair_audit(pairs: list[tuple[str | Path, str | Path]]) -> dict:
    """Audit an effective multi-seed x 2-arm pairing.

    Exactly ``MIN_PAIRED_SEEDS`` single/diverse pairs are required. Each pair
    must share one training seed, one data design and one initialization; the
    seeds must be distinct, and once fitted summaries exist the initialisations
    must also be distinct. Preparation alone is reported as ``partial`` rather
    than passed, so a missing fit cannot read as an audit pass.
    """
    if len(pairs) != MIN_PAIRED_SEEDS:
        raise ValueError(
            f"effective pairing requires exactly {MIN_PAIRED_SEEDS} single/diverse pairs, "
            f"got {len(pairs)}"
        )
    gathered = []
    for single, diverse in pairs:
        single_config, single_manifest, single_record = load_prepared(Path(single))
        diverse_config, _, diverse_record = load_prepared(Path(diverse))
        left = Path(single) / "training" / "summary.json"
        right = Path(diverse) / "training" / "summary.json"
        gathered.append({
            "single_config": single_config,
            "diverse_config": diverse_config,
            "single_record": single_record,
            "diverse_record": diverse_record,
            "lanes_manifest": single_manifest,
            "single_summary": (
                json.loads(left.read_text(encoding="utf-8")) if left.is_file() else None
            ),
            "diverse_summary": (
                json.loads(right.read_text(encoding="utf-8")) if right.is_file() else None
            ),
        })
    return cohort_pairing_checks(gathered)


def cohort_pairing_checks(pairs: list[dict]) -> dict:
    """Pure multi-seed x 2-arm pairing contract used by ``cohort_pair_audit``.

    Each input carries the two resolved configs, their records, the shared lanes
    manifest and (once fitted) the two summaries. Keeping this pure makes the
    parity contract executable even while no production cohort can legally be
    minted.
    """
    if len(pairs) != MIN_PAIRED_SEEDS:
        raise ValueError(
            f"effective pairing requires exactly {MIN_PAIRED_SEEDS} single/diverse pairs, "
            f"got {len(pairs)}"
        )
    seeds: list[int] = []
    initial: list[str] = []
    per_pair = []
    designs = set()
    for item in pairs:
        single_config = item["single_config"]
        diverse_config = item["diverse_config"]
        result = pairing_checks(
            single_config, diverse_config, item["single_record"], item["diverse_record"],
            single_summary=item.get("single_summary"),
            diverse_summary=item.get("diverse_summary"),
            lanes_manifest=item.get("lanes_manifest"),
        )
        seeds.append(single_config["training"]["seed"])
        per_pair.append(result)
        for config in (single_config, diverse_config):
            designs.add(_canonical_hash({
                field: config["training"].get(field, DEFAULT_ROUTE_POLICY)
                for field in PAIR_DESIGN_FIELDS
            }))
        if item.get("single_summary") is not None and item.get("diverse_summary") is not None:
            initial.append(item["single_summary"]["initial_state_sha256"])
            initial.append(item["diverse_summary"]["initial_state_sha256"])
    checks = {
        "pairs": len(per_pair),
        "each_pair_passed": all(result["passed"] for result in per_pair),
        "distinct_training_seeds": len(set(seeds)) == MIN_PAIRED_SEEDS,
        "matched_data_design": len(designs) == 1,
    }
    fitted = len(initial) == 2 * MIN_PAIRED_SEEDS
    if fitted:
        checks["initialisations_paired_within_seed"] = all(
            result["checks"].get("initial_weights", False) for result in per_pair
        )
        checks["initialisations_distinct_across_seeds"] = len(set(initial)) == MIN_PAIRED_SEEDS
    return {
        "passed": all(checks.values()),
        "partial": not fitted,
        "checks": checks,
        "seeds": seeds,
        "fit_evidence": "paired summaries audited" if fitted else "pending: fitted summaries absent",
        "audit_note": (
            "an interim programme audits pairing only; it cannot establish a capability pass"
        ),
    }


def route_coverage_from_prepared(prepared: Path, episodes: int = 64) -> dict:
    """Measured training-route distribution for one frozen prepared run."""
    if episodes < 1:
        raise ValueError("route coverage needs at least one episode")
    config, manifest, record = load_prepared(Path(prepared))
    batches = EpisodeBatches(config, manifest)
    plans = [batches.plan(index, commit=True) for index in range(episodes)]
    result = route_coverage(plans)
    result.update({
        "arm": config["arm"],
        "route_policy": config["training"].get("route_policy", DEFAULT_ROUTE_POLICY),
        "config_sha256": record["config_sha256"],
        "model_optimizer_updates": 0,
        "note": "route distribution only; no model forward, score or threshold change",
    })
    return result


# ---------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--exclusions", type=Path,
                                help="exclusion ledger; defaults to manifest sibling")
    prepare_parser.add_argument("--out", type=Path, required=True)
    prepare_parser.add_argument("--arm", choices=ARMS, required=True)
    prepare_parser.add_argument("--repo", type=Path, default=adapter.DEFAULT_REPO)
    prepare_parser.add_argument("--seed", type=int)
    prepare_parser.add_argument("--updates", type=int, help="smoke only; forces non-scientific")
    prepare_parser.add_argument("--episode-length", type=int, help="smoke only")
    prepare_parser.add_argument("--batch-size", type=int, help="smoke only")
    prepare_parser.add_argument("--device", choices=("cpu", "cuda"), help="smoke only")
    prepare_parser.add_argument("--max-seconds", type=float, help="smoke only")

    check_parser = sub.add_parser("check")
    check_parser.add_argument("--prepared", type=Path, required=True)
    check_parser.add_argument("--repo", type=Path, default=adapter.DEFAULT_REPO)
    check_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--prepared", type=Path, required=True)
    train_parser.add_argument("--repo", type=Path, default=adapter.DEFAULT_REPO)
    train_parser.add_argument("--approve-config-sha256")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument(
        "--continuation-from", type=Path,
        help="prior prepared run whose committed checkpoint is extended; labels the result "
             "scientific_continuation, never scientific_fit",
    )

    audit_parser = sub.add_parser("pair-audit")
    audit_parser.add_argument("--single", type=Path, required=True)
    audit_parser.add_argument("--diverse", type=Path, required=True)

    cohort_parser = sub.add_parser("cohort-audit")
    cohort_parser.add_argument(
        "--pair", action="append", nargs=2, metavar=("SINGLE", "DIVERSE"), required=True,
        help=f"one single/diverse prepared pair; pass exactly {MIN_PAIRED_SEEDS}",
    )

    route_parser = sub.add_parser("route-coverage")
    route_parser.add_argument("--prepared", type=Path, required=True)
    route_parser.add_argument("--episodes", type=int, default=64)
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(
                args.config, args.manifest, args.out, args.arm, args.repo,
                exclusions_path=args.exclusions,
                updates=args.updates, seed=args.seed, episode_length=args.episode_length,
                batch_size=args.batch_size, device=args.device, max_seconds=args.max_seconds,
            )
        elif args.command == "check":
            result = check(args.prepared, args.repo, args.device)
        elif args.command == "train":
            result = train(
                args.prepared, args.repo, args.approve_config_sha256,
                resume=args.resume, continuation_from=args.continuation_from,
            )
        elif args.command == "cohort-audit":
            result = cohort_pair_audit([tuple(pair) for pair in args.pair])
        elif args.command == "route-coverage":
            result = route_coverage_from_prepared(args.prepared, args.episodes)
        else:
            result = pair_audit(args.single, args.diverse)
    except RefusalError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    except (OSError, KeyError, ValueError, adapter.AdapterError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
