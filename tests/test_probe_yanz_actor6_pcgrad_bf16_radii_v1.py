from __future__ import annotations

import inspect
import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import probe_yanz_actor6_pcgrad_bf16_radii_v1 as probe  # noqa: E402


class RadiusContractTests(unittest.TestCase):
    def test_radii_are_fixed_and_include_an_exact_source_baseline(self) -> None:
        self.assertEqual(
            probe.RADII,
            (
                0.0,
                2.5e-5,
                5.0e-5,
                1.0e-4,
                2.0e-4,
                4.0e-4,
                8.0e-4,
                1.6e-3,
                3.2e-3,
            ),
        )
        self.assertEqual(len(set(probe.RADII)), len(probe.RADII))
        self.assertEqual(tuple(sorted(probe.RADII)), probe.RADII)

    def test_scaled_descent_has_requested_float64_l2(self) -> None:
        direction = (
            torch.tensor([3.0, 4.0], dtype=torch.float32),
            torch.tensor([12.0], dtype=torch.float32),
        )
        scaled = probe.scaled_descent(direction, 2.5e-5)
        self.assertTrue(all(value.dtype == torch.float64 for value in scaled))
        self.assertAlmostEqual(probe.vector_norm(scaled), 2.5e-5, places=15)
        zero = probe.scaled_descent(direction, 0.0)
        self.assertEqual(probe.vector_norm(zero), 0.0)

    def test_delta_geometry_uses_actual_cast_delta_and_handles_zero(self) -> None:
        treatment = (torch.tensor([2.0, 0.0]),)
        retention = (torch.tensor([1.0, 1.0]),)
        actual = (torch.tensor([1.0, 0.0]),)
        report = probe.delta_geometry(actual, treatment, retention)
        self.assertEqual(report["actual_delta_l2"], 1.0)
        self.assertEqual(report["effects"]["treatment"]["dot"], 2.0)
        self.assertEqual(report["effects"]["treatment"]["cosine"], 1.0)
        self.assertAlmostEqual(
            report["effects"]["retention"]["cosine"],
            1.0 / math.sqrt(2.0),
        )
        zero = probe.delta_geometry(
            (torch.zeros(2),), treatment, retention
        )
        self.assertEqual(zero["actual_delta_l2"], 0.0)
        self.assertIsNone(zero["effects"]["treatment"]["cosine"])
        self.assertIsNone(zero["effects"]["retention"]["cosine"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_delta_geometry_accepts_cpu_delta_and_cuda_gradients(self) -> None:
        actual = (torch.tensor([1.0, 0.0]),)
        treatment = (torch.tensor([2.0, 0.0], device="cuda"),)
        retention = (torch.tensor([1.0, 1.0], device="cuda"),)
        report = probe.delta_geometry(actual, treatment, retention)
        self.assertEqual(report["effects"]["treatment"]["dot"], 2.0)
        self.assertAlmostEqual(
            report["effects"]["retention"]["cosine"],
            1.0 / math.sqrt(2.0),
        )


class BehaviorContractTests(unittest.TestCase):
    @staticmethod
    def behavior(
        *,
        actions: list[list[int]],
        correct: list[bool],
        nll: list[float],
        logits: torch.Tensor,
    ) -> dict[str, object]:
        ids = [f"{index:064x}" for index in range(len(actions))]
        return {
            "row_ids": ids,
            "contexts": [0] * len(ids),
            "actions": dict(zip(ids, actions)),
            "correct": dict(zip(ids, correct)),
            "per_row_nll": dict(zip(ids, nll)),
            "weighted_loss": sum(nll) / len(nll),
            "policy_logits": (logits,),
            "expanded_count_classes": 61,
            "raw_expanded_policy_bitwise": True,
        }

    def test_compare_behavior_separates_repairs_harm_and_action_quantization(self) -> None:
        source = self.behavior(
            actions=[[0], [1], [2]],
            correct=[False, False, True],
            nll=[2.0, 3.0, 0.5],
            logits=torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
        )
        moved = self.behavior(
            actions=[[1], [0], [0]],
            correct=[True, False, False],
            nll=[1.0, 3.5, 1.0],
            logits=torch.tensor([[0.0, 1.0], [2.0, 3.5], [4.0, 5.0]]),
        )
        report = probe.compare_behavior(source, moved)
        self.assertEqual(report["repairs"]["count"], 1)
        self.assertEqual(report["harm"]["source_correct_to_wrong_count"], 1)
        self.assertEqual(report["harm"]["nll_worsened_count"], 2)
        self.assertEqual(report["action_changed_count"], 3)
        self.assertEqual(report["policy_quantization"]["changed_elements"], 1)
        self.assertEqual(report["policy_quantization"]["max_abs_delta"], 0.5)


class SafetyContractTests(unittest.TestCase):
    def test_frozen_bindings_validate_without_opening_any_evaluation_dataset(self) -> None:
        bindings = probe.validate_frozen_bindings()
        self.assertEqual(bindings["profile"]["sha256"], probe.PROFILE_SHA256)
        self.assertEqual(bindings["runner"]["sha256"], probe.RUNNER_SHA256)
        self.assertEqual(bindings["protocol"]["sha256"], probe.PROTOCOL_SHA256)
        self.assertEqual(bindings["prior_train_only_audit"]["sha256"], probe.PRIOR_AUDIT_SHA256)

    def test_cli_only_accepts_the_new_json_output_path(self) -> None:
        args = probe.parse_args(["--output", "probe.json"])
        self.assertEqual(args.output, Path("probe.json"))
        with self.assertRaises(SystemExit):
            probe.parse_args(["--output", "probe.json", "--data", "x.zip"])
        with self.assertRaises(SystemExit):
            probe.parse_args(["--output", "probe.json", "--radius", "1e-4"])

    def test_source_contains_no_checkpoint_writer(self) -> None:
        source = inspect.getsource(probe)
        forbidden = "torch" + ".save"
        self.assertNotIn(forbidden, source)
        self.assertNotIn("atomic_torch_save", source)
        self.assertNotIn("candidate-output", source)


if __name__ == "__main__":
    unittest.main()
