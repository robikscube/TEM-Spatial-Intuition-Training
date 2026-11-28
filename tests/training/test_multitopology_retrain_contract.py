"""Contract tests for the public multi-topology retraining package.

All fixtures are synthetic: a small generated cohort plus pure-Python configs.
Nothing here fits, evaluates or scores a model. Tests that need PyTorch only
exercise RNG snapshot/restore; tests that need the pinned upstream checkout are
skipped when it is absent.

Run:
    python -m unittest discover -s tests/training -p "test_*.py" -v
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import multitopology_retrain_graphs as graphs  # noqa: E402
import tem_trained_adapter as adapter  # noqa: E402
import train_multitopology_retrain as tem  # noqa: E402

UPSTREAM = ROOT / "research" / "reference" / "torch_tem"
TORCH_AVAILABLE = adapter.torch_available()
UPSTREAM_READY = (UPSTREAM / "model.py").is_file() and all(
    (ROOT / name).is_file() for name in tem.CODE
)

COUNTS = {"train": 2, "validation": 1, "test": 1}

CONFIG_TEMPLATE = """\
schema = "tem.multitopology.retrain.config.v1"

[training]
iterations = 3
batch_size = 2
episode_length = 48
rollout = 8
bptt = "episode"
schedule = "pilot"
device = "cpu"
scientific_mode = false
seed = 21
cue_seed = 5
route_seed = 6
family_seed = 7
threads = 1
max_seconds = 0
save_updates = [3]
log_every = 1
vram_budget_gib = 1.0

[evaluation]
seed = 13
"""


def training_settings(**overrides) -> dict:
    settings = {
        "iterations": 3,
        "batch_size": 2,
        "episode_length": 48,
        "rollout": 8,
        "bptt": "episode",
        "schedule": "pilot",
        "device": "cpu",
        "scientific_mode": False,
        "seed": 3,
        "cue_seed": 4,
        "route_seed": 5,
        "family_seed": 6,
        "threads": 1,
        "max_seconds": 0,
        "save_updates": [3],
        "log_every": 1,
        "vram_budget_gib": 1.0,
    }
    settings.update(overrides)
    if "iterations" in overrides and "save_updates" not in overrides:
        settings["save_updates"] = [settings["iterations"]]
    return settings


def config_dict(settings: dict | None = None, evaluation_seed: int = 9) -> dict:
    return {
        "schema": tem.CONFIG_SCHEMA,
        "training": settings if settings is not None else training_settings(),
        "evaluation": {"seed": evaluation_seed},
    }


def standard_settings(**overrides) -> dict:
    settings = training_settings(
        iterations=tem.FULL_UPDATES_MIN,
        schedule="upstream",
        device="cuda",
        scientific_mode=True,
        save_updates=list(tem.DECLARED_CHECKPOINTS),
    )
    settings.update(overrides)
    return settings


def interim_settings(**overrides) -> dict:
    settings = training_settings(
        iterations=tem.INTERIM_UPDATES,
        schedule="upstream",
        device="cuda",
        scientific_mode=True,
        scientific_stage=tem.INTERIM_STAGE,
        save_updates=list(tem.INTERIM_CHECKPOINTS),
    )
    settings.update(overrides)
    return settings


class SyntheticCohort:
    """One small generated cohort, shared read-only by the test classes."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.directory = self.root / "cohort"
        self.directory.mkdir()
        self.exclusions = graphs.load_historic_exclusions(root=ROOT)
        self.manifest, self.sealed = graphs.generate_cohort(
            seed=11, counts=COUNTS, exclusions=self.exclusions, root=ROOT,
        )
        self.paths = graphs.write_cohort(
            self.manifest, self.exclusions, sealed=self.sealed, out_dir=self.directory,
        )

    def close(self) -> None:
        self._temporary.cleanup()


class UpdateLedgerTest(unittest.TestCase):
    def test_validate_rejects_ineligible_updates_without_counting(self):
        ledger = tem.UpdateLedger(requested=2)
        with self.assertRaises(tem.ZeroScoredUpdate):
            ledger.validate(loss_value=1.0, scored=0, requires_grad=True, gradients_finite=True)
        with self.assertRaises(ValueError):
            ledger.validate(loss_value=float("nan"), scored=1, requires_grad=True, gradients_finite=True)
        with self.assertRaises(ValueError):
            ledger.validate(loss_value=1.0, scored=1, requires_grad=False, gradients_finite=True)
        with self.assertRaises(ValueError):
            ledger.validate(loss_value=1.0, scored=1, requires_grad=True, gradients_finite=False)
        self.assertEqual(ledger.updates, 0)

    def test_commit_is_the_only_increment_and_completion_is_exact(self):
        ledger = tem.UpdateLedger(requested=2)
        ledger.validate(loss_value=1.0, scored=4, requires_grad=True, gradients_finite=True)
        self.assertEqual(ledger.updates, 0, "validation alone must not count an update")
        ledger.commit()
        self.assertEqual(ledger.updates, 1)
        with self.assertRaisesRegex(ValueError, "incomplete fit"):
            ledger.require_complete()
        ledger.record(loss_value=0.5, scored=2, requires_grad=True, gradients_finite=True)
        ledger.require_complete()
        self.assertEqual(ledger.updates, 2)

    def test_skipped_updates_are_counted_separately(self):
        ledger = tem.UpdateLedger(requested=1)
        ledger.mark_skipped()
        self.assertEqual((ledger.updates, ledger.skipped), (0, 1))


class RoutePolicyDefaultTest(unittest.TestCase):
    def setUp(self):
        self.cohort = SyntheticCohort()
        self.addCleanup(self.cohort.close)
        self.manifest = dict(self.cohort.manifest)
        self.manifest["reserved_assignments"] = []

    def test_route_for_maps_declared_policies_and_rejects_unknown(self):
        self.assertIs(tem.route_for("legal_walk"), tem.legal_walk)
        self.assertIs(tem.route_for("reciprocal_return"), tem.reciprocal_return_walk)
        with self.assertRaisesRegex(ValueError, "unknown route policy"):
            tem.route_for("shortest_path")
        self.assertEqual(tem.DEFAULT_ROUTE_POLICY, "legal_walk")

    def test_omitted_route_policy_is_semantically_legal_walk(self):
        config = config_dict()
        config["arm"] = "single"
        config["training_graph_ids"] = [self.manifest["splits"]["train"][0]]
        tem.validate_config(config, resolved=True)
        self.assertEqual(tem.pair_value(config, "route_policy"), "legal_walk")

        explicit = json.loads(json.dumps(config))
        explicit["training"]["route_policy"] = "legal_walk"
        tem.validate_config(explicit, resolved=True)

        omitted_batches = tem.EpisodeBatches(config, self.manifest)
        explicit_batches = tem.EpisodeBatches(explicit, self.manifest)
        for episode in range(4):
            left = omitted_batches.plan(episode)
            right = explicit_batches.plan(episode)
            self.assertEqual(left.nodes, right.nodes)
            self.assertEqual(left.actions, right.actions)
            self.assertEqual(left.cues, right.cues)

        # Both spellings resolve to the same declared value for pair/continuation
        # comparisons, so an explicit default never reads as a changed design.
        self.assertEqual(
            tem.pair_value(config, "route_policy"),
            tem.pair_value(explicit, "route_policy"),
        )

    def test_explicit_legal_walk_and_omission_are_equivalent_for_continuation(self):
        prior = config_dict(interim_settings(route_policy="legal_walk"))
        new = config_dict(standard_settings())
        prior.update({"arm": "single", "training_graph_ids": ["g-0"]})
        new.update({"arm": "single", "training_graph_ids": ["g-0"]})
        result = tem.continuation_checks(
            prior, {"manifest_sha256": "m", "code_sha256": {"a": 1}, "upstream": {"commit": "c"}},
            tem.INTERIM_UPDATES,
            new, {"manifest_sha256": "m", "code_sha256": {"a": 1}, "upstream": {"commit": "c"}},
        )
        self.assertNotIn("route_policy", result["changed_settings"])
        self.assertTrue(result["schedule_unchanged"])

    def test_route_policy_does_not_change_the_cue_stream(self):
        legal = config_dict()
        legal.update({"arm": "single", "training_graph_ids": self.manifest["splits"]["train"][:1]})
        other = json.loads(json.dumps(legal))
        other["training"]["route_policy"] = "reciprocal_return"
        left = tem.EpisodeBatches(legal, self.manifest).plan(0)
        right = tem.EpisodeBatches(other, self.manifest).plan(0)
        self.assertEqual(left.cue_assignment, right.cue_assignment)
        self.assertEqual(left.index, right.index)


class ConfigValidationTest(unittest.TestCase):
    def test_non_scientific_config_is_accepted_and_has_no_blockers(self):
        config = config_dict()
        tem.validate_config(config)
        self.assertEqual(tem.scientific_blockers(config, {"device": "cpu"}), [])
        self.assertEqual(tem.scientific_status(config["training"]), tem.NON_SCIENTIFIC_STATUS)

    def test_standard_scientific_config_is_accepted(self):
        tem.validate_config(config_dict(standard_settings()))

    def test_scientific_mode_blocks_short_history_truncated_credit_and_cpu(self):
        cases = {
            "episode_length": 40,
            "bptt": "rollout",
            "schedule": "pilot",
            "device": "cpu",
            "max_seconds": 60,
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "scientific_mode blocked"):
                    tem.validate_config(config_dict(standard_settings(**{key: value})))
        with self.subTest(key="iterations"):
            short = standard_settings(
                iterations=9999, save_updates=[1000, 2500, 5000, 7500, 9999],
            )
            with self.assertRaisesRegex(ValueError, "iterations 9999"):
                tem.validate_config(config_dict(short))

    def test_interim_stage_is_pinned_to_its_declared_budget(self):
        tem.validate_config(config_dict(interim_settings()))
        with self.assertRaisesRegex(ValueError, "requires exactly 5000"):
            tem.validate_config(config_dict(
                interim_settings(iterations=4000, save_updates=[1000, 2500, 4000])
            ))
        wrong = interim_settings(save_updates=[1000, 2500, 5000, 7500])
        wrong["iterations"] = 7500
        with self.assertRaises(ValueError):
            tem.validate_config(config_dict(wrong))

    def test_interim_stage_cannot_bypass_the_gate_by_disabling_scientific_mode(self):
        with self.assertRaisesRegex(ValueError, "requires scientific_mode = true"):
            tem.validate_config(config_dict(interim_settings(scientific_mode=False)))

    def test_invalid_route_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "route_policy"):
            tem.validate_config(config_dict(training_settings(route_policy="greedy")))

    def test_resolved_arm_graph_counts(self):
        single = config_dict()
        single.update({"arm": "single", "training_graph_ids": ["a", "b"]})
        with self.assertRaisesRegex(ValueError, "exactly one training graph"):
            tem.validate_config(single, resolved=True)

        diverse = config_dict()
        diverse.update({"arm": "diverse", "training_graph_ids": ["a"]})
        with self.assertRaisesRegex(ValueError, "at least two"):
            tem.validate_config(diverse, resolved=True)

        duplicate = config_dict()
        duplicate.update({"arm": "diverse", "training_graph_ids": ["a", "a"]})
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            tem.validate_config(duplicate, resolved=True)

    def test_scientific_environment_blockers_are_reported_not_assumed(self):
        config = config_dict(standard_settings())
        attests = [
            {"device": "cpu", "cuda_available": False, "platform": "win32"},
            {"device": "cuda", "cuda_available": True, "platform": "win32", "in_container": False},
            {
                "device": "cuda", "cuda_available": True, "platform": "linux",
                "in_container": False, "gpu_total_memory_bytes": 16 * 2**30,
                "free_memory_bytes_at_start": 8 * 2**30,
            },
        ]
        for attestation in attests:
            reasons = tem.scientific_blockers(config, attestation)
            self.assertTrue(reasons)
        ready = {
            "device": "cuda", "cuda_available": True, "platform": "linux",
            "in_container": True, "gpu_total_memory_bytes": 16 * 2**30,
            "free_memory_bytes_at_start": 8 * 2**30,
        }
        self.assertEqual(tem.scientific_blockers(config, ready), [])


class StageLabelTest(unittest.TestCase):
    def test_standard_completed_run_is_the_only_scientific_fit(self):
        contract = tem.stage_contract(
            standard_settings(), status="completed", updates=tem.FULL_UPDATES_MIN,
            skipped=0, full_history_updates=tem.FULL_UPDATES_MIN,
        )
        self.assertTrue(contract["stage_contract_met"])
        self.assertTrue(contract["scientific_fit"])
        self.assertTrue(contract["final_scoring_authorized"])
        self.assertFalse(contract["capability_pass"])

    def test_interim_completion_is_preliminary_and_never_authorises_scoring(self):
        contract = tem.stage_contract(
            interim_settings(), status="completed", updates=tem.INTERIM_UPDATES,
            skipped=0, full_history_updates=tem.INTERIM_UPDATES,
        )
        self.assertTrue(contract["stage_contract_met"])
        self.assertTrue(contract["interim_contract_met"])
        self.assertTrue(contract["interim_preliminary"])
        self.assertFalse(contract["scientific_fit"])
        self.assertFalse(contract["final_scoring_authorized"])
        self.assertEqual(contract["scientific_status"], "scientific_interim")

    def test_continuation_is_labelled_and_never_a_scientific_fit(self):
        contract = tem.stage_contract(
            standard_settings(), status="completed", updates=tem.FULL_UPDATES_MIN,
            skipped=0, full_history_updates=tem.FULL_UPDATES_MIN, continued=True,
        )
        self.assertEqual(contract["scientific_status"], tem.CONTINUATION_STATUS)
        self.assertFalse(contract["scientific_fit"])

    def test_incomplete_or_skipped_runs_do_not_meet_the_contract(self):
        for status, updates, skipped in (
            ("interrupted", tem.FULL_UPDATES_MIN, 0),
            ("completed", tem.FULL_UPDATES_MIN - 1, 0),
            ("completed", tem.FULL_UPDATES_MIN, 1),
        ):
            contract = tem.stage_contract(
                standard_settings(), status=status, updates=updates,
                skipped=skipped, full_history_updates=updates,
            )
            self.assertFalse(contract["stage_contract_met"])

    def test_continuation_rejects_a_changed_design(self):
        prior = config_dict(interim_settings())
        new = config_dict(standard_settings(batch_size=4))
        prior.update({"arm": "single", "training_graph_ids": ["g-0"]})
        new.update({"arm": "single", "training_graph_ids": ["g-0"]})
        record = {"manifest_sha256": "m", "code_sha256": {"a": 1}, "upstream": {"commit": "c"}}
        with self.assertRaisesRegex(ValueError, "may only change"):
            tem.continuation_checks(prior, record, tem.INTERIM_UPDATES, new, record)

    def test_continuation_requires_the_declared_prior_checkpoint(self):
        prior = config_dict(interim_settings())
        new = config_dict(standard_settings())
        prior.update({"arm": "single", "training_graph_ids": ["g-0"]})
        new.update({"arm": "single", "training_graph_ids": ["g-0"]})
        record = {"manifest_sha256": "m", "code_sha256": {"a": 1}, "upstream": {"commit": "c"}}
        with self.assertRaisesRegex(ValueError, "not a declared prior checkpoint"):
            tem.continuation_checks(prior, record, 4321, new, record)


class IndependentLaneTest(unittest.TestCase):
    def setUp(self):
        self.cohort = SyntheticCohort()
        self.addCleanup(self.cohort.close)
        self.manifest = dict(self.cohort.manifest)
        self.manifest["reserved_assignments"] = []
        self.config = config_dict()
        self.config.update({
            "arm": "single",
            "training_graph_ids": self.manifest["splits"]["train"][:1],
        })

    def test_batch_lanes_are_distinct_episodes(self):
        batch = tem.EpisodeBatches(self.config, self.manifest).batch(0)
        diversity = batch.diversity()
        self.assertEqual(diversity["batch_size"], 2)
        self.assertEqual(diversity["distinct_routes"], 2)
        self.assertEqual(diversity["distinct_cue_sequences"], 2)

    def test_replicated_lanes_are_refused(self):
        plan = tem.EpisodeBatches(self.config, self.manifest).plan(0)
        with self.assertRaisesRegex(ValueError, "distinct episode indices"):
            tem.BatchEpisodePlan([plan, plan])
        twin = tem.EpisodePlan(
            index=1, graph_id=plan.graph_id, graph_index=plan.graph_index,
            nodes=plan.nodes, actions=plan.actions, cues=plan.cues,
            cue_assignment=plan.cue_assignment, history_steps=plan.history_steps,
            revisit_lag_mean=plan.revisit_lag_mean, revisit_lag_max=plan.revisit_lag_max,
        )
        with self.assertRaisesRegex(ValueError, "replicated batch lanes"):
            tem.BatchEpisodePlan([plan, twin])

    def test_single_plan_cannot_stand_in_for_a_batch(self):
        plan = tem.EpisodeBatches(self.config, self.manifest).plan(0)
        with self.assertRaisesRegex(ValueError, "independent lanes require BatchEpisodePlan"):
            tem._as_batch(plan, 2)


class ManifestAuditTest(unittest.TestCase):
    def setUp(self):
        self.cohort = SyntheticCohort()
        self.addCleanup(self.cohort.close)
        self.manifest = self.cohort.manifest
        self.exclusions = self.cohort.exclusions

    def resealed(self, manifest: dict) -> dict:
        manifest = json.loads(json.dumps(manifest))
        manifest["content_sha256"] = tem._content_digest(manifest)
        return manifest

    def test_generated_manifest_is_public_and_sealed(self):
        audit = tem._validate_public_manifest(self.manifest)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["test_payload_status"], "sealed")
        self.assertFalse(audit["test_payload_read"])
        self.assertTrue(audit["empty_historic_inventory"])
        self.assertFalse(any(graph["split"] == "test" for graph in self.manifest["graphs"]))

    def test_unexpected_keys_and_digest_changes_fail_closed(self):
        extra = json.loads(json.dumps(self.manifest))
        extra["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unexpected public cohort manifest schema"):
            tem._validate_public_manifest(extra)

        tampered = json.loads(json.dumps(self.manifest))
        tampered["seed"] += 1
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            tem._validate_public_manifest(tampered)

    def test_test_payloads_and_unsealed_commitments_are_refused(self):
        leaked = json.loads(json.dumps(self.manifest))
        test_graph = dict(leaked["graphs"][0])
        test_graph["split"] = "test"
        leaked["graphs"].append(test_graph)
        with self.assertRaisesRegex(ValueError, "must not contain final test graph payloads"):
            tem._validate_public_manifest(self.resealed(leaked))

        unsealed = json.loads(json.dumps(self.manifest))
        unsealed["test_commitment"]["status"] = "unsealed"
        with self.assertRaisesRegex(ValueError, "sealed test commitment"):
            tem._validate_public_manifest(self.resealed(unsealed))

    def test_private_cohort_kind_is_refused_not_weakened(self):
        tagged = json.loads(json.dumps(self.manifest))
        tagged["generation"]["cohort_kind"] = "legacy-mixed"
        with self.assertRaisesRegex(ValueError, "not supported by this package"):
            tem._validate_public_manifest(self.resealed(tagged))

    def test_public_audit_stops_at_the_sealed_commitment(self):
        with self.assertRaisesRegex(graphs.CohortAuditError, "sealed test payload is missing"):
            graphs.audit_cohort(
                self.manifest, exclusions=self.exclusions, sealed=None, allow_test=False,
            )

    def test_loader_records_that_the_sealed_payload_was_not_read(self):
        manifest, exclusions, reserved, audit = tem._load_cohort_public(
            self.cohort.paths["manifest"], self.cohort.paths["exclusions"],
        )
        self.assertFalse(audit["sealed_payload_read"])
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(reserved, [])
        self.assertEqual(
            exclusions["content_sha256"], manifest["exclusion_ledger"]["content_sha256"]
        )

    def test_loader_rejects_a_mismatched_exclusion_ledger(self):
        other = graphs.load_historic_exclusions(root=ROOT, sources=())
        other["content_sha256"] = "0" * 64
        path = Path(self.cohort.root) / "exclusions-mismatch.json"
        path.write_text(json.dumps(other), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exclusion ledger|exclusion payload"):
            tem._load_cohort_public(self.cohort.paths["manifest"], path)


class CohortGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.cohort = SyntheticCohort()
        self.addCleanup(self.cohort.close)

    def test_empty_inventory_is_disclosed_and_freshness_is_internal(self):
        ledger = self.cohort.exclusions
        self.assertTrue(ledger["empty_inventory"])
        self.assertEqual(ledger["distinct_full_classes"], 0)
        self.assertEqual(ledger["assignment_count"], 0)
        audit = graphs.audit_cohort(
            self.cohort.manifest, exclusions=ledger,
            sealed=self.cohort.sealed, allow_test=True, root=ROOT,
        )
        self.assertEqual(audit["status"], "passed")
        self.assertTrue(audit["empty_inventory"])
        self.assertFalse(audit["test_scored"])
        self.assertEqual(audit["model_optimizer_updates"], 0)

    def test_sealed_payload_is_not_in_the_public_manifest(self):
        payload = json.loads(self.cohort.paths["manifest"].read_text(encoding="utf-8"))
        ids = {graph["id"] for graph in payload["graphs"]}
        self.assertFalse(ids & set(payload["splits"]["test"]))
        self.assertEqual(
            payload["test_commitment"]["payload_sha256"],
            self.cohort.sealed["content_sha256"],
        )
        self.assertNotIn("test-sealed", json.dumps(payload["graphs"]))

    def test_historic_reference_graph_is_excluded(self):
        reference = self.cohort.manifest["graphs"][0]
        ledger = graphs.load_historic_exclusions(root=ROOT)
        state = graphs.exclusion_state(ledger)
        self.assertEqual(len(state["trees"]), 0)
        reference_input = {
            key: reference[key] for key in ("names", "edges", "transitions", "root", "depths")
        }
        fresh, _ = graphs._is_fresh(reference_input, state, set(), {})
        self.assertTrue(fresh, "an empty ledger excludes nothing")
        state["trees"].add(graphs.base.backbone_key(reference_input))
        fresh, reason = graphs._is_fresh(reference_input, state, set(), {})
        self.assertFalse(fresh)
        self.assertEqual(reason, "backbone_isomorphic")

    def test_capacity_enumeration_is_exhaustive_and_arithmetic(self):
        report = graphs.topology_capacity(self.cohort.exclusions)
        self.assertTrue(report["exhaustive"])
        self.assertFalse(report["candidate_bound_dependent"])
        self.assertEqual(report["backbone_space"], report["enumeration"]["distinct_backbone_shapes"])
        self.assertEqual(report["fresh_backbones"], report["backbone_space"])
        for request in report["requests"]:
            self.assertEqual(request["feasible"], request["total"] <= report["fresh_backbones"])
        self.assertEqual(report["model_optimizer_updates"], 0)
        self.assertFalse(report["test_payload_opened"])

    def test_unsealing_requires_explicit_acknowledgement(self):
        with self.assertRaisesRegex(graphs.CohortAuditError, "acknowledge_test_open"):
            graphs.unseal_test(self.cohort.manifest, self.cohort.sealed)
        opened = graphs.unseal_test(
            self.cohort.manifest, self.cohort.sealed, acknowledge_test_open=True,
        )
        self.assertEqual(len(opened["graphs"]), COUNTS["test"])
        self.assertEqual(
            opened["commitment_sha256"],
            self.cohort.manifest["test_commitment"]["payload_sha256"],
        )

    def test_tampered_sealed_payload_is_refused(self):
        sealed = json.loads(json.dumps(self.cohort.sealed))
        sealed["graphs"][0]["id"] = "retrain-test-999"
        sealed["content_sha256"] = graphs._content_digest(sealed)
        with self.assertRaisesRegex(graphs.CohortAuditError, "hash does not match"):
            graphs.audit_cohort(
                self.cohort.manifest, exclusions=self.cohort.exclusions,
                sealed=sealed, allow_test=False, root=ROOT,
            )

    def test_recheck_history_matches_the_frozen_empty_inventory(self):
        source_specs = self.cohort.exclusions["source_specs"]
        self.assertEqual(source_specs, [])
        result = graphs.recheck_history(
            self.cohort.manifest, self.cohort.exclusions, root=ROOT,
        )
        self.assertEqual(result["status"], "passed")


@unittest.skipUnless(
    UPSTREAM_READY and adapter.training_dependencies_available(),
    "pinned torch_tem checkout and its runtime dependencies are required",
)
class PrepareContractTest(unittest.TestCase):
    def setUp(self):
        self.cohort = SyntheticCohort()
        self.addCleanup(self.cohort.close)
        self.config_path = Path(self.cohort.root) / "experiment.toml"
        self.config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")

    def prepare(self, name: str, arm: str) -> tuple[Path, dict]:
        out = Path(self.cohort.root) / name
        record = tem.prepare(
            self.config_path, self.cohort.paths["manifest"], out, arm, UPSTREAM,
        )
        return out, record

    def test_prepare_writes_zero_updates_and_binds_every_digest(self):
        out, record = self.prepare("single", "single")
        self.assertEqual(record["status"], "prepared")
        self.assertEqual(record["optimizer_steps"], 0)
        self.assertEqual(record["scientific_status"], tem.NON_SCIENTIFIC_STATUS)
        self.assertEqual(record["test_payload_status"], "sealed")
        self.assertFalse(record["test_payload_opened"])
        self.assertFalse(record["test_payload_read"])
        self.assertEqual(record["route_policy"], "legal_walk")
        self.assertEqual(record["train_graph_count"], 1)
        config, manifest, loaded = tem.load_prepared(out, UPSTREAM)
        self.assertEqual(config["arm"], "single")
        self.assertEqual(config["training_graph_ids"], [manifest["splits"]["train"][0]])
        self.assertEqual(loaded["config_sha256"], record["config_sha256"])
        self.assertEqual(manifest["reserved_assignments"], [])

    def test_every_prepared_input_is_digest_checked(self):
        out, _ = self.prepare("single", "single")
        for name in ("config.json", "graphs.json", "source-config.toml",
                     "cohort-exclusions.json", "reservations.json"):
            with self.subTest(name=name):
                original = (out / name).read_bytes()
                (out / name).write_bytes(original + b"\n# tampered\n")
                with self.assertRaisesRegex(ValueError, "prepared input digest mismatch"):
                    tem.load_prepared(out, UPSTREAM)
                (out / name).write_bytes(original)

    def test_arm_graph_ids_must_match_the_frozen_split(self):
        out, _ = self.prepare("single", "single")
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        config["training_graph_ids"] = self.cohort.manifest["splits"]["train"]
        (out / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "prepared input digest mismatch"):
            tem.load_prepared(out, UPSTREAM)

    def test_pair_audit_passes_without_fitted_summaries(self):
        single, _ = self.prepare("single", "single")
        diverse, _ = self.prepare("diverse", "diverse")
        result = tem.pair_audit(single, diverse)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["checks"]["graph_manifest"])
        self.assertTrue(result["checks"]["one_vs_many_graphs"])
        self.assertTrue(result["checks"]["lane_parity"], result["lane_parity"])
        self.assertEqual(result["lane_parity"]["cross_arm_mismatches"], 0)

    def test_route_coverage_reports_no_optimizer_updates(self):
        single, _ = self.prepare("single", "single")
        result = tem.route_coverage_from_prepared(single, episodes=4)
        self.assertEqual(result["model_optimizer_updates"], 0)
        self.assertEqual(result["episodes"], 4)
        self.assertEqual(result["route_policy"], "legal_walk")
        self.assertEqual(result["observations"], 4 * 48)

    def test_cohort_audit_requires_the_declared_pair_count(self):
        single, _ = self.prepare("single", "single")
        diverse, _ = self.prepare("diverse", "diverse")
        with self.assertRaisesRegex(ValueError, f"exactly {tem.MIN_PAIRED_SEEDS}"):
            tem.cohort_pair_audit([(single, diverse)])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class RngResumeTest(unittest.TestCase):
    def test_rng_state_is_captured_and_restored(self):
        import random

        torch = adapter.require_torch()
        numpy = adapter.require_numpy()
        torch.manual_seed(1234)
        numpy.random.seed(1234)
        random.seed(1234)

        first = tem._rng_snapshot(torch, "cpu")
        digest = tem._rng_digest(first)
        torch.rand(3)
        numpy.random.rand(3)
        random.random()
        advanced = tem._rng_digest(tem._rng_snapshot(torch, "cpu"))
        self.assertNotEqual(advanced, digest, "advancing the RNG must change its digest")

        tem._rng_restore(first, torch, "cpu")
        restored = tem._rng_digest(tem._rng_snapshot(torch, "cpu"))
        self.assertEqual(restored, digest, "restore must reproduce the serialized RNG state")

        expected = (torch.rand(1).item(), float(numpy.random.rand()), random.random())
        tem._rng_restore(first, torch, "cpu")
        replayed = (torch.rand(1).item(), float(numpy.random.rand()), random.random())
        self.assertEqual(expected, replayed)

    def test_restore_requires_a_snapshot(self):
        torch = adapter.require_torch()
        with self.assertRaisesRegex(ValueError, "no RNG snapshot"):
            tem._rng_restore(None, torch, "cpu")


@unittest.skipUnless(
    TORCH_AVAILABLE and adapter.numpy_available(),
    "PyTorch and NumPy are required",
)
class CheckpointCommitTransactionTest(unittest.TestCase):
    """``checkpoints.json`` is the commit point and the resume authority.

    An interrupted ``save_artifacts`` must leave a directory that automatic
    resume can recover without manual repair: the ledger selects the resume
    update, its numbered files are hash-verified, and any stale or missing
    ``* .pt`` alias is rebuilt from them.
    """

    def setUp(self):
        self.torch = adapter.require_torch()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name) / "training"
        self.out.mkdir()
        self.model = self.torch.nn.Linear(2, 1)
        self.model.hyper = {"eta": 0.1, "lambda": 0.2, "n_actions": 4}
        self.optimizer = self.torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.record = {
            "config_sha256": "c" * 64,
            "manifest_sha256": "m" * 64,
            "code_sha256": {"scripts/train_multitopology_retrain.py": "a" * 64},
            "arm": "single",
            "training_seed": 21,
        }
        self.verify = {
            "next_episode": 0, "next_chunk": 0, "batch_size": 1,
            "loss": None, "scored_predictions": None, "history_steps": None,
        }

    def state(self, update: int) -> dict:
        return {
            "updates": update,
            "skipped": 0,
            "weights_committed": True,
            "stream": {"episode": 0, "chunk": 0, "visited": None, "carry": None},
            "chunks_consumed": 0,
            "full_history_updates": update,
            "rng": tem._rng_snapshot(self.torch, "cpu"),
            "scientific_stage": "standard",
            "lineage": [],
            "continuation": False,
            "stopping": False,
        }

    def save(self, update: int) -> dict:
        return tem.save_artifacts(
            self.model, self.optimizer, self.out, update,
            record=self.record, state=self.state(update), verify=self.verify,
        )

    def ledger_updates(self) -> list[int]:
        entries = json.loads((self.out / "checkpoints.json").read_text(encoding="utf-8"))
        return [entry["update"] for entry in entries]

    def resume(self, update: int | None = None):
        state, checkpoint, continuation = tem._load_resume_state(
            self.torch, self.out.parent, self.record, config=None,
        )
        if update is not None:
            self.assertEqual(state["update"], update)
        return state, checkpoint, continuation

    def test_commit_publishes_the_ledger_before_the_aliases(self):
        events: list[tuple[str, str]] = []
        real_write_json = tem.write_json
        real_copy = tem._atomic_copy

        def recording_write_json(path, value):
            events.append(("ledger", path.name))
            real_write_json(path, value)

        def recording_copy(source, target):
            events.append(("alias", target.name))
            real_copy(source, target)

        with mock.patch.object(tem, "write_json", side_effect=recording_write_json), \
             mock.patch.object(tem, "_atomic_copy", side_effect=recording_copy):
            self.save(1)

        self.assertLess(
            events.index(("ledger", "checkpoints.json")),
            events.index(("alias", "training-state.pt")),
            "the ledger must be committed before the aliases point at it",
        )

    def test_interrupt_during_alias_copy_resumes_from_the_committed_update(self):
        self.save(1)

        def boom(source, target):
            raise RuntimeError("interrupted during alias copy")

        with mock.patch.object(tem, "_atomic_copy", side_effect=boom):
            with self.assertRaises(RuntimeError):
                self.save(2)

        # The ledger committed update 2 before the alias copy started.
        self.assertEqual(self.ledger_updates(), [1, 2])
        self.resume(update=2)

        # The interrupted aliases were recovered from the hash-verified numbered
        # files, so the alias is consistent again after resume.
        entries = json.loads((self.out / "checkpoints.json").read_text(encoding="utf-8"))
        entry = next(item for item in entries if item["update"] == 2)
        for alias_name, key in (
            ("training-state.pt", "training_state"),
            ("checkpoint.pt", "checkpoint"),
            ("params.pt", "params"),
        ):
            self.assertEqual(
                (self.out / alias_name).read_bytes(),
                (self.out / entry[key]).read_bytes(),
            )

    def test_interrupt_before_ledger_write_rolls_back_to_the_prior_commit(self):
        self.save(1)

        def boom(path, value):
            raise RuntimeError("interrupted during ledger write")

        with mock.patch.object(tem, "write_json", side_effect=boom):
            with self.assertRaises(RuntimeError):
                self.save(2)

        # Numbered update-2 files may exist, but the ledger never recorded them,
        # so the authoritative resume target is still update 1.
        self.assertEqual(self.ledger_updates(), [1])
        self.resume(update=1)

    def test_interrupt_during_numbered_state_write_rolls_back_to_the_prior_commit(self):
        self.save(1)
        original_save = tem._atomic_torch_save

        def boom(value, path):
            if path.name.startswith("training-state_"):
                raise RuntimeError("interrupted during numbered state write")
            original_save(value, path)

        with mock.patch.object(tem, "_atomic_torch_save", side_effect=boom):
            with self.assertRaises(RuntimeError):
                self.save(2)

        self.assertEqual(self.ledger_updates(), [1])
        self.resume(update=1)

    def test_alias_ahead_of_the_ledger_is_ignored_and_recovered(self):
        # Reproduce the original failure on disk: numbered files and aliases for
        # update 2 exist, but the ledger was rewound to before its commit.
        self.save(1)
        self.save(2)
        entries = json.loads((self.out / "checkpoints.json").read_text(encoding="utf-8"))
        with open(self.out / "checkpoints.json", "w", encoding="utf-8") as handle:
            json.dump([item for item in entries if item["update"] == 1], handle)

        alias = adapter._load_torch_file(self.torch, self.out / "training-state.pt")
        self.assertEqual(alias["update"], 2, "the manufactured alias points ahead of the ledger")

        self.resume(update=1)
        alias = adapter._load_torch_file(self.torch, self.out / "training-state.pt")
        self.assertEqual(alias["update"], 1, "resume must rewind the alias to the committed ledger entry")


class GradClipAndNumericalDiagnosticsTest(unittest.TestCase):
    """Contract tests for tri-state numerical diagnostics and optional grad_clip_norm."""

    def test_grad_clip_config_validation_matrix(self):
        # Absent = disabled (valid)
        tem.validate_config(config_dict(training_settings()))
        # Positive finite int/float = valid
        for good in (1, 0.5, 500.0):
            with self.subTest(good=good):
                tem.validate_config(config_dict(training_settings(grad_clip_norm=good)))
        # Invalid values rejected
        for bad in (0, -1.0, True, False, None, float("nan"), float("inf"), float("-inf"), "500", [1.0], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    tem.validate_config(config_dict(training_settings(grad_clip_norm=bad)))

    def test_update_ledger_distinguishes_absent_and_nonfinite_gradients(self):
        ledger = tem.UpdateLedger(requested=2)
        with self.assertRaisesRegex(ValueError, "no gradients"):
            ledger.validate(loss_value=1.0, scored=1, requires_grad=True, gradient_state="absent")
        with self.assertRaisesRegex(ValueError, "nonfinite gradient"):
            ledger.validate(loss_value=1.0, scored=1, requires_grad=True, gradient_state="nonfinite")
        with self.assertRaisesRegex(ValueError, "invalid gradient_state"):
            ledger.validate(loss_value=1.0, scored=1, requires_grad=True, gradient_state="bogus")
        ledger.record(loss_value=1.0, scored=1, requires_grad=True, gradient_state="finite")
        self.assertEqual(ledger.updates, 1)

    def test_pairing_and_continuation_enforce_grad_clip_symmetry(self):
        single = {**config_dict(training_settings()), "arm": "single", "training_graph_ids": ["g0"]}
        diverse_same = {**config_dict(training_settings()), "arm": "diverse", "training_graph_ids": ["g0", "g1"]}
        diverse_diff = {**config_dict(training_settings(grad_clip_norm=50.0)), "arm": "diverse", "training_graph_ids": ["g0", "g1"]}
        rec = {"manifest_sha256": "a" * 64}
        self.assertTrue(tem.pairing_checks(single, diverse_same, rec, rec)["checks"]["grad_clip_norm"])
        self.assertFalse(tem.pairing_checks(single, diverse_diff, rec, rec)["checks"]["grad_clip_norm"])

        # Enabling grad_clip_norm across a continuation must fail closed.
        prior_cfg = {**config_dict(interim_settings()), "arm": "diverse", "training_graph_ids": ["g0", "g1"]}
        next_cfg = {**config_dict(standard_settings(grad_clip_norm=50.0)), "arm": "diverse", "training_graph_ids": ["g0", "g1"]}
        prior_rec = {"config_sha256": "1" * 64, "manifest_sha256": "a" * 64, "code_sha256": {"k": "v"}, "arm": "diverse", "training_seed": 3}
        next_rec = {"config_sha256": "2" * 64, "manifest_sha256": "a" * 64, "code_sha256": {"k": "v"}, "arm": "diverse", "training_seed": 3}
        with self.assertRaisesRegex(ValueError, "grad_clip_norm"):
            tem.continuation_checks(prior_cfg, prior_rec, tem.INTERIM_UPDATES, next_cfg, next_rec)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for tensor grad-clip tests")
    def test_apply_grad_clip_and_finiteness_diagnostics(self):
        torch = adapter.require_torch()
        model = torch.nn.Linear(2, 1, bias=False)
        self.assertEqual(tem._gradient_state(torch, model), "absent")

        model.weight.grad = torch.tensor([[3.0, 4.0]])
        self.assertEqual(tem._gradient_state(torch, model), "finite")
        # Disabled path leaves gradients untouched
        self.assertIsNone(tem._apply_grad_clip(torch, model, max_norm=None))
        self.assertAlmostEqual(float(model.weight.grad.norm()), 5.0, places=5)

        # Enabled path clips oversized finite gradient to threshold
        meta = tem._apply_grad_clip(torch, model, max_norm=2.5)
        self.assertTrue(meta["applied"])
        self.assertAlmostEqual(meta["raw_global_norm"], 5.0, places=5)
        self.assertLessEqual(meta["post_global_norm"], 2.5001)
        json.dumps(meta, allow_nan=False)

        # Nonfinite gradient classified cleanly and produces strict-JSON report
        model.weight.grad = torch.tensor([[float("inf"), 1.0]])
        self.assertEqual(tem._gradient_state(torch, model), "nonfinite")
        report = tem._finiteness_report(torch, model)
        self.assertFalse(report["all_grads_finite"])
        self.assertEqual(report["parameters"][0]["first_nonfinite_flat_index"], 0)
        json.dumps(report, allow_nan=False)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for step preparation tests")
    def test_prepare_step_gradients_preserves_legacy_preclip_and_fails_closed(self):
        torch = adapter.require_torch()
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.tensor([[10.0]])
        # Disabled path must never call clip_grad_norm_
        with mock.patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=AssertionError("must not clip")):
            norm, max_abs, clip = tem._prepare_step_gradients(
                torch, model, grad_clip_norm=None, gradient_state="finite",
                loss=1.0, loss_components=None, attempted_update=1,
            )
        self.assertIsNone(clip)
        self.assertAlmostEqual(norm, 10.0, places=5)
        self.assertAlmostEqual(max_abs, 10.0, places=5)

        # Enabled path preserves pre-clip legacy norm/max_abs while clipping gradient
        norm, max_abs, clip = tem._prepare_step_gradients(
            torch, model, grad_clip_norm=4.0, gradient_state="finite",
            loss=1.0, loss_components=None, attempted_update=1,
        )
        self.assertAlmostEqual(norm, 10.0, places=5)
        self.assertAlmostEqual(max_abs, 10.0, places=5)
        self.assertTrue(clip["applied"])
        self.assertAlmostEqual(float(model.weight.grad.abs().max()), 4.0, places=5)

        # Clip helper error wraps as NumericalTrainingError(phase="post_clip")
        model.weight.grad = torch.tensor([[10.0]])
        with mock.patch.object(tem, "_apply_grad_clip", side_effect=RuntimeError("reduction overflow")):
            with self.assertRaises(tem.NumericalTrainingError) as ctx:
                tem._prepare_step_gradients(
                    torch, model, grad_clip_norm=4.0, gradient_state="finite",
                    loss=1.0, loss_components=None, attempted_update=1,
                )
        self.assertEqual(ctx.exception.phase, "post_clip")
        json.dumps(ctx.exception.diagnostic, allow_nan=False)

    def test_inv_var_guard_config_pairing_and_continuation(self):
        cfg = config_dict(training_settings())
        tem.validate_config(cfg)
        ok = config_dict(training_settings(inv_var_guard="log_precision"))
        tem.validate_config(ok)
        for bad in (None, True, False, 1, "", "clamp", "invalid"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    tem.validate_config(config_dict(training_settings(inv_var_guard=bad)))

        single = {**cfg, "arm": "single", "training_graph_ids": ["g0"]}
        diverse_same = {**cfg, "arm": "diverse", "training_graph_ids": ["g0", "g1"]}
        diverse_diff = {**ok, "arm": "diverse", "training_graph_ids": ["g0", "g1"]}
        rec = {"manifest_sha256": "a" * 64}
        self.assertTrue(tem.pairing_checks(single, diverse_same, rec, rec)["checks"]["inv_var_guard"])
        self.assertFalse(tem.pairing_checks(single, diverse_diff, rec, rec)["checks"]["inv_var_guard"])

        prior_cfg = {**config_dict(interim_settings()), "arm": "diverse", "training_graph_ids": ["g0", "g1"]}
        next_cfg = {**config_dict(standard_settings(inv_var_guard="log_precision")), "arm": "diverse", "training_graph_ids": ["g0", "g1"]}
        prior_rec = {"config_sha256": "1" * 64, "manifest_sha256": "a" * 64, "code_sha256": {"k": "v"}, "arm": "diverse", "training_seed": 3}
        next_rec = {"config_sha256": "2" * 64, "manifest_sha256": "a" * 64, "code_sha256": {"k": "v"}, "arm": "diverse", "training_seed": 3}
        with self.assertRaisesRegex(ValueError, "inv_var_guard"):
            tem.continuation_checks(prior_cfg, prior_rec, tem.INTERIM_UPDATES, next_cfg, next_rec)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for inv_var_guard math tests")
    def test_inv_var_guard_log_precision_equivalence_and_extreme_sigma(self):
        import types
        torch = adapter.require_torch()

        def stock_inv_var_weight(mus, sigmas):
            sum_var = sum(1.0 / (s ** 2) for s in sigmas)
            inv_var_var = 1.0 / sum_var
            inv_var_avg = sum(m / (s ** 2) for m, s in zip(mus, sigmas)) * inv_var_var
            return inv_var_avg, torch.sqrt(inv_var_var)

        fake_utils = types.SimpleNamespace(inv_var_weight=stock_inv_var_weight)
        fake_upstream = types.SimpleNamespace(utils=fake_utils)

        m1, m2 = torch.randn(4, 30), torch.randn(4, 30)
        s1, s2 = torch.rand(4, 30).add(0.1), torch.rand(4, 30).add(0.1)
        a_ref, sg_ref = stock_inv_var_weight([m1, m2], [s1, s2])
        a_new, sg_new = tem._inv_var_weight_log_precision(torch, [m1, m2], [s1, s2])
        self.assertLess(float((a_ref - a_new).abs().max()), 1e-5)
        self.assertLess(float((sg_ref - sg_new).abs().max()), 1e-5)

        tem._install_inv_var_guard(torch, fake_upstream, "log_precision")
        try:
            x1 = torch.tensor([[1.0, -0.5]], requires_grad=True)
            x2 = torch.tensor([[0.2, 0.8]], requires_grad=True)
            z1 = torch.tensor([[-50.0, 0.0]], requires_grad=True)
            z2 = torch.tensor([[0.0, -50.0]], requires_grad=True)
            avg, sg = fake_utils.inv_var_weight([x1, x2], [torch.exp(z1), torch.exp(z2)])
            loss = (avg ** 2).sum() + (sg ** 2).sum()
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(bool(torch.isfinite(z1.grad).all()))
            self.assertTrue(bool(torch.isfinite(z2.grad).all()))
            self.assertTrue(bool(torch.isfinite(x1.grad).all()))
        finally:
            tem._install_inv_var_guard(torch, fake_upstream, None)
        self.assertIs(fake_utils.inv_var_weight, stock_inv_var_weight)


if __name__ == "__main__":
    unittest.main()
