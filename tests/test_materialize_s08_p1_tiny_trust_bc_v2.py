from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/materialize_s08_p1_tiny_trust_bc_v2.py"
SPEC = importlib.util.spec_from_file_location("s08_p1_tiny_materializer_v2", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def metric_record(
    *,
    set_correct: int = 80,
    hybrid_correct: int = 79,
    ordered_correct: int = 78,
    count_correct: int = 90,
    rows: int = 100,
) -> dict:
    return {
        "rows": rows,
        "set_exact_correct": set_correct,
        "set_exact_accuracy": set_correct / rows,
        "hybrid_order_exact_correct": hybrid_correct,
        "hybrid_order_exact_accuracy": hybrid_correct / rows,
        "ordered_exact_correct": ordered_correct,
        "ordered_exact_accuracy": ordered_correct / rows,
        "count_correct": count_correct,
    }


def behavior_record(
    semantics: str,
    *,
    candidate: bool,
    decrease: float = 1e-6,
) -> dict:
    losses = {}
    for mode, baseline in (("natural", 1.0), ("legacy_8_2", 1.2)):
        value = baseline - decrease if candidate else baseline
        losses[mode] = {
            "selection_loss": value,
            "weighted_numerator": value * 100.0,
            "selection_effective_weight_sum": 100.0,
            "batches": 1,
        }
    group = metric_record()
    return {
        "semantics": semantics,
        "device": "cpu" if semantics == "cpu_fp32" else "cuda:0",
        "batch_size": tool.base.BATCH_SIZE,
        "rows": 100,
        "row_key_sha256": "row",
        "metrics": metric_record(),
        "by_team": {"team": copy.deepcopy(group)},
        "by_seat": {"0": copy.deepcopy(group)},
        "losses": losses,
        "count_logits_float32_le_sha256": "count-logits",
        "action_count_prediction_sha256": "count-actions",
        "policy_action_sha256": "policy-actions",
    }


def behavior_bundle(specs: tuple, *, candidate: bool, decrease: float = 1e-6) -> dict:
    return {
        semantics: {
            spec.name: behavior_record(
                semantics, candidate=candidate, decrease=decrease
            )
            for spec in specs
        }
        for semantics in tool.SEMANTICS
    }


def reference_bundle(specs: tuple, prediction: float = 5e-6) -> dict:
    return {
        spec.name: {
            mode: {
                "reference_prediction": prediction,
                "formula": "-dot(g_frozen_cuda_bf16, actual_float32_delta)",
                "reference_gradient_semantics": "S8 CUDA/BF16 frozen geometry",
            }
            for mode in ("natural", "legacy_8_2")
        }
        for spec in specs
    }


class ClosureAndPlanTests(unittest.TestCase):
    def test_transitive_import_closure_is_preimport_and_plan_bound(self) -> None:
        self.assertEqual(
            tool.PREIMPORT_INPUTS["run_ppo_bc_repair"][1],
            tool.RUN_PPO_BC_REPAIR_SHA256,
        )
        self.assertEqual(tool.PREIMPORT_INPUTS["cg_sim_py"][1], tool.SIM_PY_SHA256)
        self.assertEqual(tool.PREIMPORT_INPUTS["cg_libcg"][1], tool.LIBCG_SHA256)
        plan = tool.build_plan()
        closure = plan["runtime_dependency_closure"]
        self.assertTrue(closure["shared_library_may_be_loaded_by_transitive_import"])
        self.assertFalse(closure["battle_API_function_invoked_by_materializer"])
        for key in ("run_ppo_bc_repair", "cg_sim_py", "cg_libcg"):
            self.assertIn(key, plan["inputs"])

    def test_plan_names_reference_ratio_honestly(self) -> None:
        protocol = tool.build_plan()["frozen_geometry_reference"]
        self.assertEqual(
            protocol["reference_prediction"],
            "-dot(g_frozen, actual_float32_candidate_delta)",
        )
        self.assertEqual(protocol["ratio_name"], "frozen_geometry_reference_ratio")
        self.assertFalse(protocol["cpu_fp32_Taylor_derivative_claimed"])
        self.assertFalse(protocol["cuda_bf16_Taylor_derivative_claimed"])
        self.assertTrue(protocol["final_gradients_recomputed_only_after_unique_radius_lock"])

    def test_plan_requires_second_independent_gate_and_result_first_transaction(self) -> None:
        plan = tool.build_plan()
        confirmation = plan["behavior_protocol"]["confirmation"]
        self.assertTrue(confirmation["second_evaluation_independently_compared_to_S8"])
        self.assertTrue(confirmation["second_complete_gate_must_pass"])
        transaction = plan["publication_transaction"]
        self.assertTrue(transaction["result_O_EXCL_written_before_checkpoint_payload"])
        self.assertTrue(
            transaction[
                "result_parent_directory_fsynced_before_checkpoint_payload"
            ]
        )
        self.assertTrue(transaction["checkpoint_valid_iff_file_sha256_and_bytes_match_result"])

    def test_plan_is_double_deterministic_and_zero_write(self) -> None:
        first = tool.canonical_json_bytes(tool.build_plan())
        second = tool.canonical_json_bytes(tool.build_plan())
        self.assertEqual(first, second)
        self.assertFalse(tool.PLAN_PATH.exists())
        self.assertFalse(tool.RESULT_PATH.exists())
        self.assertFalse(tool.CHECKPOINT_PATH.exists())


class ActualDeltaReferenceTests(unittest.TestCase):
    def _tiny_state(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.zeros(1, dtype=torch.float32) for name in tool.ACTOR6
        }

    def test_reference_is_negative_gradient_dot_actual_delta(self) -> None:
        parent = self._tiny_state()
        candidate = tool.base.clone_state(parent)
        for name in tool.ACTOR6:
            candidate[name].fill_(-2e-6)
        gradients = {
            mode: {
                "anti_cal": torch.arange(1, 7, dtype=torch.float64)
            }
            for mode in ("natural", "legacy_8_2")
        }
        old_elements = tool.P1_ELEMENTS
        tool.P1_ELEMENTS = 6
        try:
            records = tool.actual_delta_reference_predictions(
                parent, candidate, gradients, (tool.CAL_SPECS[0],)
            )
        finally:
            tool.P1_ELEMENTS = old_elements
        expected = 2e-6 * sum(range(1, 7))
        for mode in ("natural", "legacy_8_2"):
            record = records["anti_cal"][mode]
            self.assertAlmostEqual(record["reference_prediction"], expected, places=12)
            self.assertFalse(record["is_CPU_Taylor_prediction"])
            self.assertIn("actual_float32_delta", record["formula"])

    def test_radius_local_integrity_failure_can_be_recorded_for_fallback(self) -> None:
        parent = self._tiny_state()
        direction = torch.full((6,), 1.0 / math.sqrt(6), dtype=torch.float64)
        old_elements = tool.P1_ELEMENTS
        old_hash = tool.P1_VECTOR_SHA256
        old_integrity = tool.base.state_integrity
        tool.P1_ELEMENTS = 6
        tool.P1_VECTOR_SHA256 = tool.probe.vector_sha256(direction)
        tool.base.state_integrity = lambda *_args, **_kwargs: {
            "pass": False,
            "gates": {"quantized_integrity": False},
        }
        try:
            state, integrity = tool.materialize_candidate_state_v2(
                parent, direction, tool.RADIUS_PRIORITY[0]
            )
        finally:
            tool.P1_ELEMENTS = old_elements
            tool.P1_VECTOR_SHA256 = old_hash
            tool.base.state_integrity = old_integrity
        self.assertIsNone(state)
        self.assertFalse(integrity["pass"])
        self.assertTrue(integrity["radius_local_integrity_failure"])

    def test_global_direction_hash_failure_remains_fail_hard(self) -> None:
        parent = self._tiny_state()
        with self.assertRaisesRegex(RuntimeError, "Global frozen P1"):
            tool.materialize_candidate_state_v2(
                parent, torch.ones(6), tool.RADIUS_PRIORITY[0]
            )


class DualBehaviorGateTests(unittest.TestCase):
    def test_bundle_has_twelve_actual_reference_loss_items(self) -> None:
        baseline = behavior_bundle(tool.CAL_SPECS, candidate=False)
        candidate = behavior_bundle(tool.CAL_SPECS, candidate=True)
        references = reference_bundle(tool.CAL_SPECS)
        report = tool.compare_behavior_bundle_v2(
            baseline, candidate, tool.CAL_SPECS, references
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["loss_item_gate_count"], 12)
        sample = report["views"]["cpu_fp32"]["anti_cal"]["losses"]["natural"]
        self.assertAlmostEqual(
            sample["frozen_geometry_reference_ratio"], 0.2, places=10
        )
        self.assertFalse(sample["CPU_Taylor_derivative_claimed"])
        self.assertNotIn("predicted_first_order_decrease", sample)

    def test_absolute_decrease_stays_a_hard_dual_semantics_gate(self) -> None:
        baseline = behavior_bundle(tool.CAL_SPECS, candidate=False)
        candidate = behavior_bundle(tool.CAL_SPECS, candidate=True, decrease=1e-8)
        report = tool.compare_behavior_bundle_v2(
            baseline,
            candidate,
            tool.CAL_SPECS,
            reference_bundle(tool.CAL_SPECS, prediction=5e-8),
        )
        self.assertFalse(report["pass"])
        self.assertFalse(
            report["views"]["cuda_bf16"]["general_cal"]["gates"]
            ["natural_observed_decrease_at_least_2e_7"]
        )

    def test_second_evaluation_is_independently_regated(self) -> None:
        baseline = behavior_bundle(tool.CAL_SPECS, candidate=False)
        first = behavior_bundle(tool.CAL_SPECS, candidate=True)
        second = copy.deepcopy(first)
        passing = tool.complete_two_pass_gate(
            baseline,
            first,
            second,
            tool.CAL_SPECS,
            reference_bundle(tool.CAL_SPECS),
        )
        self.assertTrue(passing["pass"])
        for mode, base_loss in (("natural", 1.0), ("legacy_8_2", 1.2)):
            second["cpu_fp32"]["anti_cal"]["losses"][mode][
                "selection_loss"
            ] = base_loss
        failing = tool.complete_two_pass_gate(
            baseline,
            first,
            second,
            tool.CAL_SPECS,
            reference_bundle(tool.CAL_SPECS),
        )
        self.assertFalse(failing["pass"])
        self.assertFalse(failing["gates"]["second_complete_gate_pass"])
        self.assertFalse(
            failing["second_gate"]["views"]["cpu_fp32"]["anti_cal"]["pass"]
        )


class FrozenPlanEnvelopeTests(unittest.TestCase):
    def test_outer_sha_schema_mode_and_inner_sha_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan = {"schema_version": tool.SCHEMA_VERSION + "-plan", "x": 1}
            envelope = tool.plan_envelope(plan)
            path.write_bytes(tool.canonical_json_bytes(envelope))
            path.chmod(0o444)
            old_path = tool.PLAN_PATH
            tool.PLAN_PATH = path
            try:
                loaded = tool.load_frozen_plan(
                    path, tool.sha256_json(plan), tool.file_sha256(path)
                )
                self.assertEqual(loaded, plan)
                with self.assertRaisesRegex(RuntimeError, "outer plan file SHA"):
                    tool.load_frozen_plan(path, tool.sha256_json(plan), "0" * 64)
                path.chmod(0o644)
                with self.assertRaisesRegex(RuntimeError, "0444"):
                    tool.load_frozen_plan(
                        path, tool.sha256_json(plan), tool.file_sha256(path)
                    )
            finally:
                tool.PLAN_PATH = old_path


class PublicationTransactionTests(unittest.TestCase):
    def test_result_is_published_before_reserved_checkpoint_is_filled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            checkpoint_path = Path(directory) / "candidate.pt"
            old_result = tool.RESULT_PATH
            old_checkpoint = tool.CHECKPOINT_PATH
            old_snapshot = tool.snapshot_input_hashes
            tool.RESULT_PATH = result_path
            tool.CHECKPOINT_PATH = checkpoint_path
            tool.snapshot_input_hashes = lambda **_kwargs: {"frozen": "hash"}
            payload = b"valid-checkpoint-payload"
            audit = {
                "expected_payload_sha256": tool.hashlib.sha256(payload).hexdigest(),
                "expected_bytes": len(payload),
                "model_state_sha256": "state",
                "metadata_canonical_sha256": "metadata",
            }
            try:
                publication = tool.publish_passing_result_then_checkpoint(
                    {"schema_version": "fixture", "status": "pending"},
                    payload,
                    audit,
                    {"frozen": "hash"},
                )
            finally:
                tool.RESULT_PATH = old_result
                tool.CHECKPOINT_PATH = old_checkpoint
                tool.snapshot_input_hashes = old_snapshot
            result = json.loads(result_path.read_text())
            self.assertTrue(result["checkpoint_publication"]["result_written_before_checkpoint_payload"])
            self.assertTrue(
                result["checkpoint_publication"]
                ["result_parent_directory_fsync_required_before_checkpoint_payload"]
            )
            self.assertIn("valid_iff", result["checkpoint_publication"])
            self.assertEqual(checkpoint_path.read_bytes(), payload)
            self.assertTrue(publication["valid_by_result_contract"])
            self.assertEqual(checkpoint_path.stat().st_mode & 0o777, 0o444)

    def test_result_failure_leaves_only_empty_invalid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            checkpoint_path = Path(directory) / "candidate.pt"
            old_result = tool.RESULT_PATH
            old_checkpoint = tool.CHECKPOINT_PATH
            old_snapshot = tool.snapshot_input_hashes
            old_write = tool.base.write_exclusive
            tool.RESULT_PATH = result_path
            tool.CHECKPOINT_PATH = checkpoint_path
            tool.snapshot_input_hashes = lambda **_kwargs: {"frozen": "hash"}

            def fail_result(*_args, **_kwargs):
                raise FileExistsError("simulated result O_EXCL race")

            tool.base.write_exclusive = fail_result
            payload = b"must-not-be-published"
            audit = {
                "expected_payload_sha256": tool.hashlib.sha256(payload).hexdigest(),
                "expected_bytes": len(payload),
            }
            try:
                with self.assertRaises(FileExistsError):
                    tool.publish_passing_result_then_checkpoint(
                        {"schema_version": "fixture"},
                        payload,
                        audit,
                        {"frozen": "hash"},
                    )
            finally:
                tool.RESULT_PATH = old_result
                tool.CHECKPOINT_PATH = old_checkpoint
                tool.snapshot_input_hashes = old_snapshot
                tool.base.write_exclusive = old_write
            self.assertFalse(result_path.exists())
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(checkpoint_path.stat().st_size, 0)

    def test_directory_fsync_failure_leaves_checkpoint_empty_and_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            checkpoint_path = Path(directory) / "candidate.pt"
            old_result = tool.RESULT_PATH
            old_checkpoint = tool.CHECKPOINT_PATH
            old_snapshot = tool.snapshot_input_hashes
            old_directory_fsync = tool.fsync_parent_directory
            tool.RESULT_PATH = result_path
            tool.CHECKPOINT_PATH = checkpoint_path
            tool.snapshot_input_hashes = lambda **_kwargs: {"frozen": "hash"}

            def fail_directory_fsync(_path):
                raise OSError("simulated parent directory fsync failure")

            tool.fsync_parent_directory = fail_directory_fsync
            payload = b"must-not-be-published"
            audit = {
                "expected_payload_sha256": tool.hashlib.sha256(payload).hexdigest(),
                "expected_bytes": len(payload),
            }
            try:
                with self.assertRaisesRegex(OSError, "directory fsync failure"):
                    tool.publish_passing_result_then_checkpoint(
                        {"schema_version": "fixture"},
                        payload,
                        audit,
                        {"frozen": "hash"},
                    )
            finally:
                tool.RESULT_PATH = old_result
                tool.CHECKPOINT_PATH = old_checkpoint
                tool.snapshot_input_hashes = old_snapshot
                tool.fsync_parent_directory = old_directory_fsync
            self.assertTrue(result_path.exists())
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(checkpoint_path.stat().st_size, 0)

    def test_source_orders_result_and_directory_fsync_before_checkpoint_fill(self) -> None:
        source = inspect.getsource(tool.publish_passing_result_then_checkpoint)
        self.assertLess(
            source.index("base.write_exclusive(RESULT_PATH"),
            source.index("fsync_parent_directory(RESULT_PATH"),
        )
        self.assertLess(
            source.index("fsync_parent_directory(RESULT_PATH"),
            source.index("fill_reserved_checkpoint(descriptor"),
        )


class CheckpointAndStaticScopeTests(unittest.TestCase):
    def test_v2_checkpoint_is_eval_only_and_serializes_through_bound_v1(self) -> None:
        state = {name: torch.zeros(1) for name in tool.ACTOR6}
        state["count_head.0.weight"] = torch.ones(1)
        parent = {
            "feature_version": "fixture",
            "model_state_dict": state,
            "optimizer_state_dict": {"forbidden": True},
        }
        checkpoint = tool.build_checkpoint_v2(
            parent,
            state,
            frozen_plan_sha256="a" * 64,
            frozen_plan_file_sha256="b" * 64,
            radius_label="R50",
            radius=5e-5,
            integrity={"pass": True},
            calibration_record={"pass": True},
            final_record={"pass": True},
        )
        self.assertTrue(checkpoint["evaluation_only"])
        self.assertTrue(tool.base.FORBIDDEN_CHECKPOINT_KEYS.isdisjoint(checkpoint))
        payload = tool.base.serialize_checkpoint(checkpoint)
        audit = tool.verify_checkpoint_payload_v2(
            payload, state, "a" * 64, "b" * 64, "R50"
        )
        self.assertEqual(audit["expected_bytes"], len(payload))

    def test_static_scope_has_no_local_save_or_battle_call(self) -> None:
        raw = TOOL_PATH.read_bytes()
        audit = tool.static_scope_audit(raw)
        self.assertTrue(audit["no_local_torch_save"])
        self.assertTrue(audit["battle_import_closure_bound_but_not_called"])
        source = raw.decode()
        self.assertNotIn("BattleStart(", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("requests", source)
        tree = ast.parse(raw)
        leaves = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue({"backward", "step"}.isdisjoint(leaves))

    def test_final_gradient_and_behavior_calls_follow_radius_lock(self) -> None:
        source = inspect.getsource(tool.execute)
        lock = source.index("selected_label = base.select_radius_from_records")
        final_gradient = source.index("final_gradients, final_gradient_report")
        final_behavior = source.index("final_baseline = base.evaluate_bundle")
        self.assertLess(lock, final_gradient)
        self.assertLess(final_gradient, final_behavior)
        self.assertIn('"fallback_allowed": False', source)


if __name__ == "__main__":
    unittest.main()
