from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/materialize_s08_p1_tiny_trust_bc.py"
SPEC = importlib.util.spec_from_file_location("s08_p1_tiny_materializer", TOOL_PATH)
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
    target: str,
    radius: float,
    *,
    candidate: bool,
    loss_scale: float = 0.20,
) -> dict:
    losses = {}
    for mode in ("natural", "legacy_8_2"):
        base = 1.0 if mode == "natural" else 1.2
        decrease = max(
            3e-7,
            radius * tool.frozen_p1_dot(target, mode) * loss_scale,
        )
        value = base - decrease if candidate else base
        losses[mode] = {
            "selection_loss": value,
            "weighted_numerator": value * 100.0,
            "selection_effective_weight_sum": 100.0,
            "batches": 1,
        }
    metrics = metric_record()
    group = metric_record(rows=100)
    return {
        "semantics": semantics,
        "device": "cpu" if semantics == "cpu_fp32" else "cuda:0",
        "batch_size": tool.BATCH_SIZE,
        "rows": 100,
        "row_key_sha256": "row",
        "metrics": metrics,
        "by_team": {"team": copy.deepcopy(group)},
        "by_seat": {"0": copy.deepcopy(group)},
        "losses": losses,
        "count_logits_float32_le_sha256": "count-logits",
        "action_count_prediction_sha256": "count-actions",
        "policy_action_sha256": "policy-actions",
    }


def behavior_bundle(specs: tuple, radius: float, *, candidate: bool) -> dict:
    return {
        semantics: {
            spec.name: behavior_record(
                semantics,
                spec.name,
                radius,
                candidate=candidate,
            )
            for spec in specs
        }
        for semantics in tool.SEMANTICS
    }


class FrozenLineageTests(unittest.TestCase):
    def test_geometry_evidence_is_the_reviewed_P1(self) -> None:
        evidence = tool.load_geometry_evidence()
        self.assertEqual(evidence["plan_sha256"], tool.GEOMETRY_PLAN_SHA256)
        self.assertEqual(evidence["report_sha256"], tool.GEOMETRY_REPORT_SHA256)
        self.assertEqual(evidence["selected_candidate"], tool.P1_NAME)
        self.assertEqual(
            evidence["selected_direction"]["float64_le_sha256"],
            tool.P1_VECTOR_SHA256,
        )
        self.assertEqual(set(evidence["fit_gradients"]), set(tool.probe.FIT_TARGETS))

    def test_plan_binds_dual_semantics_and_sequential_radii(self) -> None:
        plan = tool.build_plan()
        self.assertEqual(
            plan["runtime"]["forward_semantics"]["cpu_fp32"]["dtype"],
            "float32",
        )
        self.assertEqual(
            plan["runtime"]["forward_semantics"]["cuda_bf16"]["dtype"],
            "bfloat16",
        )
        self.assertEqual(
            [item["label"] for item in plan["candidate_protocol"]["radius_priority"]],
            ["R50", "R25"],
        )
        self.assertTrue(
            plan["calibration_protocol"]
            ["R25_evaluated_only_if_R50_fails_complete_dual_semantics_gate"]
        )
        self.assertTrue(plan["final_protocol"]["opened_only_after_unique_radius_locked"])
        self.assertIn("effective_weight_sum", plan["behavior_measurement"]["global_selection_loss_aggregation"])

    def test_plan_is_deterministic_without_writes(self) -> None:
        frozen_before = None
        if tool.PLAN_PATH.exists():
            raw = tool.PLAN_PATH.read_bytes()
            frozen_before = {
                "bytes": raw,
                "sha256": tool.file_sha256(tool.PLAN_PATH),
                "mode": tool.PLAN_PATH.stat().st_mode & 0o777,
            }
            envelope = tool.strict_json_bytes(raw, "frozen test plan")
            self.assertEqual(raw, tool.canonical_json_bytes(envelope))
            self.assertEqual(
                envelope["plan_sha256"], tool.sha256_json(envelope["plan"])
            )
        first = tool.canonical_json_bytes(tool.build_plan())
        second = tool.canonical_json_bytes(tool.build_plan())
        self.assertEqual(first, second)
        if frozen_before is None:
            self.assertFalse(tool.PLAN_PATH.exists())
        else:
            frozen_after = {
                "bytes": tool.PLAN_PATH.read_bytes(),
                "sha256": tool.file_sha256(tool.PLAN_PATH),
                "mode": tool.PLAN_PATH.stat().st_mode & 0o777,
            }
            self.assertEqual(frozen_after, frozen_before)
            envelope = tool.strict_json_bytes(
                frozen_after["bytes"], "frozen test plan after build"
            )
            self.assertEqual(first, tool.canonical_json_bytes(envelope["plan"]))
            self.assertEqual(frozen_after["mode"], 0o444)
        self.assertFalse(tool.RESULT_PATH.exists())
        self.assertFalse(tool.CHECKPOINT_PATH.exists())


class GlobalAggregationTests(unittest.TestCase):
    def test_effective_weight_aggregation_not_row_reaverage(self) -> None:
        accumulator = tool.GlobalSelectionLoss()
        accumulator.add(2.0, 1.0)
        accumulator.add(4.0, 3.0)
        result = accumulator.finish()
        self.assertEqual(result["selection_loss"], 3.5)
        self.assertEqual(result["weighted_numerator"], 14.0)
        self.assertEqual(result["selection_effective_weight_sum"], 4.0)

    def test_invalid_weight_is_fail_closed(self) -> None:
        for weight in (0.0, -1.0, float("nan")):
            accumulator = tool.GlobalSelectionLoss()
            with self.assertRaises(RuntimeError):
                accumulator.add(1.0, weight)


class MaterializationIntegrityTests(unittest.TestCase):
    def test_integrity_accepts_actor_only_float32_step(self) -> None:
        parent = {
            name: torch.zeros(1, dtype=torch.float32) for name in tool.ACTOR6
        }
        parent["count_head.0.weight"] = torch.ones(2, dtype=torch.float32)
        direction = torch.full(
            (len(tool.ACTOR6),),
            1.0 / math.sqrt(len(tool.ACTOR6)),
            dtype=torch.float64,
        )
        radius = 5e-5
        candidate = tool.clone_state(parent)
        for index, name in enumerate(tool.ACTOR6):
            candidate[name] = (
                parent[name].double() - radius * direction[index]
            ).float()
        report = tool.state_integrity(parent, candidate, direction, radius)
        self.assertTrue(report["pass"])
        self.assertTrue(report["all_nonactor_tensors_bit_identical"])
        self.assertTrue(report["count_head_tensors_bit_identical"])
        self.assertEqual(set(report["changed_parameter_names"]), set(tool.ACTOR6))

    def test_nonactor_change_is_rejected(self) -> None:
        parent = {
            name: torch.zeros(1, dtype=torch.float32) for name in tool.ACTOR6
        }
        parent["count_head.0.weight"] = torch.ones(1)
        candidate = tool.clone_state(parent)
        candidate[tool.ACTOR6[0]].fill_(-1e-5)
        candidate["count_head.0.weight"].zero_()
        direction = torch.zeros(len(tool.ACTOR6), dtype=torch.float64)
        direction[0] = 1.0
        report = tool.state_integrity(parent, candidate, direction, 1e-5)
        self.assertFalse(report["pass"])
        self.assertFalse(report["all_nonactor_tensors_bit_identical"])
        self.assertFalse(report["count_head_tensors_bit_identical"])


class BehaviorGateTests(unittest.TestCase):
    def test_complete_dual_semantics_bundle_has_exactly_twelve_loss_items(self) -> None:
        radius = tool.RADIUS_PRIORITY[0]
        baseline = behavior_bundle(tool.CAL_SPECS, radius, candidate=False)
        candidate = behavior_bundle(tool.CAL_SPECS, radius, candidate=True)
        report = tool.compare_behavior_bundle(
            baseline, candidate, tool.CAL_SPECS, radius
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["loss_item_gate_count"], 12)
        self.assertTrue(all(report["loss_item_gates"].values()))

    def test_specialist_needs_no_plus_one_but_cannot_regress(self) -> None:
        radius = tool.RADIUS_PRIORITY[0]
        baseline = behavior_record(
            "cpu_fp32", "anti_cal", radius, candidate=False
        )
        candidate = behavior_record(
            "cpu_fp32", "anti_cal", radius, candidate=True
        )
        passing = tool.compare_behavior_view(
            baseline,
            candidate,
            target="anti_cal",
            source="anti",
            radius=radius,
        )
        self.assertTrue(passing["pass"])
        self.assertEqual(passing["correct_count_deltas"]["set_exact_correct"], 0)
        candidate["metrics"]["hybrid_order_exact_correct"] -= 1
        candidate["metrics"]["hybrid_order_exact_accuracy"] -= 0.01
        failing = tool.compare_behavior_view(
            baseline,
            candidate,
            target="anti_cal",
            source="anti",
            radius=radius,
        )
        self.assertFalse(failing["pass"])

    def test_loss_minimum_and_rho_are_both_required(self) -> None:
        radius = tool.RADIUS_PRIORITY[0]
        baseline = behavior_record(
            "cuda_bf16", "general_cal", radius, candidate=False
        )
        candidate = behavior_record(
            "cuda_bf16", "general_cal", radius, candidate=True
        )
        for mode in ("natural", "legacy_8_2"):
            candidate["losses"][mode]["selection_loss"] = (
                baseline["losses"][mode]["selection_loss"] - 1e-8
            )
        report = tool.compare_behavior_view(
            baseline,
            candidate,
            target="general_cal",
            source="general",
            radius=radius,
        )
        self.assertFalse(report["pass"])
        self.assertFalse(
            report["gates"]["natural_decrease_at_least_2e_7"]
        )

    def test_count_logits_and_action_counts_must_match_parent(self) -> None:
        radius = tool.RADIUS_PRIORITY[0]
        baseline = behavior_record(
            "cpu_fp32", "fros_cal", radius, candidate=False
        )
        candidate = behavior_record(
            "cpu_fp32", "fros_cal", radius, candidate=True
        )
        candidate["count_logits_float32_le_sha256"] = "changed"
        report = tool.compare_behavior_view(
            baseline,
            candidate,
            target="fros_cal",
            source="fros",
            radius=radius,
        )
        self.assertFalse(report["pass"])
        self.assertFalse(report["gates"]["count_logits_bit_identical"])

    def test_second_confirmation_is_digest_and_loss_strict(self) -> None:
        radius = tool.RADIUS_PRIORITY[0]
        first = behavior_bundle(tool.CAL_SPECS, radius, candidate=True)
        second = copy.deepcopy(first)
        passing = tool.confirm_behavior_bundle(first, second, tool.CAL_SPECS)
        self.assertTrue(passing["pass"])
        second["cuda_bf16"]["anti_cal"]["policy_action_sha256"] = "drift"
        failing = tool.confirm_behavior_bundle(first, second, tool.CAL_SPECS)
        self.assertFalse(failing["pass"])


class RadiusSelectionTests(unittest.TestCase):
    def test_R50_pass_forbids_R25_evaluation(self) -> None:
        self.assertEqual(
            tool.select_radius_from_records(
                {"R50": {"complete_calibration_pass": True}}
            ),
            "R50",
        )
        with self.assertRaisesRegex(RuntimeError, "cannot be evaluated"):
            tool.select_radius_from_records(
                {
                    "R50": {"complete_calibration_pass": True},
                    "R25": {"complete_calibration_pass": True},
                }
            )

    def test_R25_is_only_fallback_after_R50_failure(self) -> None:
        records = {
            "R50": {"complete_calibration_pass": False},
            "R25": {"complete_calibration_pass": True},
        }
        self.assertEqual(tool.select_radius_from_records(records), "R25")
        records["R25"]["complete_calibration_pass"] = False
        self.assertIsNone(tool.select_radius_from_records(records))


class CheckpointAndScopeTests(unittest.TestCase):
    def test_checkpoint_is_eval_only_and_has_no_optimizer_state(self) -> None:
        parent_state = {
            name: torch.zeros(1, dtype=torch.float32) for name in tool.ACTOR6
        }
        parent_state["count_head.0.weight"] = torch.ones(1)
        parent = {
            "feature_version": "fixture",
            "model_state_dict": parent_state,
            "optimizer_state_dict": {"forbidden": True},
        }
        checkpoint = tool.build_checkpoint(
            parent,
            parent_state,
            frozen_plan_sha256="a" * 64,
            radius_label="R50",
            radius=5e-5,
            integrity={"pass": True},
            calibration_record={"pass": True},
            final_record={"pass": True},
        )
        self.assertTrue(checkpoint["evaluation_only"])
        self.assertTrue(checkpoint["resume_forbidden"])
        self.assertTrue(
            tool.FORBIDDEN_CHECKPOINT_KEYS.isdisjoint(checkpoint)
        )
        payload = tool.serialize_checkpoint(checkpoint)
        audit = tool.verify_checkpoint_payload(
            payload, parent_state, "a" * 64, "R50"
        )
        self.assertEqual(audit["payload_sha256"], tool.sha256_bytes(payload))

    def test_exclusive_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            tool.write_exclusive(path, b"first\n", mode=0o600)
            with self.assertRaises(FileExistsError):
                tool.write_exclusive(path, b"second\n", mode=0o600)
            self.assertEqual(path.read_bytes(), b"first\n")

    def test_source_has_one_save_and_no_training_or_external_action(self) -> None:
        raw = TOOL_PATH.read_bytes()
        audit = tool.static_scope_audit(raw)
        self.assertTrue(audit["torch_save_call_sites_exactly_one"])
        source = raw.decode()
        self.assertNotIn("subprocess", source)
        self.assertNotIn("requests", source)
        tree = ast.parse(raw)
        call_leaves = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue({"backward", "step"}.isdisjoint(call_leaves))

    def test_final_views_cannot_enter_radius_selector(self) -> None:
        signature = inspect.signature(tool.select_radius_from_records)
        self.assertNotIn("final", signature.parameters)
        execute_source = inspect.getsource(tool.execute)
        self.assertLess(
            execute_source.index("selected_label = select_radius_from_records"),
            execute_source.index("final_baseline = evaluate_bundle"),
        )
        self.assertIn("fallback_allowed\": False", execute_source)


if __name__ == "__main__":
    unittest.main()
