from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_gold_push_marnie_ppo_v2 as v2  # noqa: E402


def command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def quota_values(command: list[str]) -> dict[str, int]:
    return {
        command[index + 1]: int(command[index + 2])
        for index, token in enumerate(command)
        if token == "--opponent-base-quota"
    }


class GoldPushMarniePpoV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase = v2.PHASES["full"]
        cls.command = v2.build_command(cls.phase)
        cls.preflight = v2.build_preflight(cls.phase)

    def test_full_is_two_update_conservative_fresh_run(self) -> None:
        self.assertEqual(command_value(self.command, "--updates"), "2")
        self.assertEqual(command_value(self.command, "--games-per-update"), "192")
        self.assertEqual(command_value(self.command, "--bc-checkpoint"), str(v2.base.LEARNER))
        self.assertNotIn("--resume", self.command)
        self.assertEqual(
            command_value(self.command, "--output-dir"),
            str(v2.OUTPUT_ROOT),
        )
        self.assertEqual(
            self.preflight.manifest["expected_terminal_checkpoint"],
            str(v2.OUTPUT_ROOT / "checkpoints/update-0002.pt"),
        )

    def test_only_actor_lr_and_bc_anchor_strength_are_light_tuned(self) -> None:
        self.assertEqual(command_value(self.command, "--learning-rate"), "0.000008")
        self.assertEqual(command_value(self.command, "--bc-kl-start"), "0.024")
        self.assertEqual(command_value(self.command, "--bc-kl-end"), "0.024")
        self.assertEqual(command_value(self.command, "--value-learning-rate"), "0.000025")
        self.assertEqual(command_value(self.command, "--target-kl"), "0.004")
        self.assertEqual(command_value(self.command, "--ppo-epochs"), "2")
        self.assertEqual(command_value(self.command, "--minibatch-size"), "512")

    def test_targeted_quotas_are_exact_and_seat_balanced(self) -> None:
        quotas = quota_values(self.command)
        expected = {
            "bc": 96,
            v2.base.opponent_name(
                v2.base.LUCARIO_BC, v2.base.LUCARIO_DECK
            ): 16,
            v2.base.opponent_name(
                v2.base.FROSLASS_BC, v2.base.FROSLASS_DECK
            ): 16,
            v2.base.opponent_name(
                v2.base.SOURCE_MARNIE_BC, v2.base.MARNIE_DECK
            ): 32,
            v2.base.opponent_name(v2.base.U472, v2.base.MARNIE_DECK): 32,
        }
        self.assertEqual(quotas, expected)
        self.assertEqual(sum(quotas.values()), 192)
        self.assertTrue(all(value % 2 == 0 for value in quotas.values()))
        self.assertIn("--opponent-quota-seat-balance", self.command)

    def test_existing_fixed_panel_gate_is_unchanged(self) -> None:
        gate = self.preflight.manifest["promotion_gate"]
        self.assertFalse(gate["changed"])
        self.assertEqual(gate["operator"], ">")
        self.assertEqual(gate["threshold"], 0.484375)
        self.assertEqual(gate["invalid_games_required"], 0)
        self.assertIn("--eval-all-permanent-opponents", self.command)
        self.assertEqual(command_value(self.command, "--selection-aggregation"), "min")
        self.assertEqual(command_value(self.command, "--eval-games"), "256")
        self.assertEqual(command_value(self.command, "--eval-interval"), "1")

    def test_failure_tail_and_bc_replay_contract_are_unchanged(self) -> None:
        self.assertIn("--failed-attempt-as-loss", self.command)
        self.assertEqual(
            command_value(self.command, "--failed-loss-tail-transitions"), "32"
        )
        self.assertNotIn("--truncation-as-loss", self.command)
        self.assertEqual(command_value(self.command, "--bc-replay-steps"), "2")
        self.assertEqual(command_value(self.command, "--bc-replay-lr-scale"), "0.05")
        self.assertEqual(command_value(self.command, "--bc-replay-loss"), "ordered")
        self.assertEqual(
            command_value(self.command, "--bc-replay-context34-rows-per-batch"),
            "4",
        )

    def test_v1_evidence_files_and_values_are_sha_locked(self) -> None:
        inputs = self.preflight.manifest["inputs"]
        self.assertEqual(inputs["v1_summary"]["sha256"], v2.V1_SUMMARY_SHA256)
        self.assertEqual(inputs["v1_metrics"]["sha256"], v2.V1_METRICS_SHA256)
        self.assertEqual(inputs["v1_launcher"]["sha256"], v2.BASE_LAUNCHER_SHA256)
        evidence = self.preflight.manifest["v1_evidence"]
        self.assertEqual(evidence["initial"]["selection_score"], 0.484375)
        self.assertEqual(evidence["update2"]["selection_score"], 0.46484375)
        self.assertEqual(evidence["update4"]["selection_score"], 0.46875)
        self.assertTrue(
            all(value == 0 for value in evidence["failed_started_games_by_update"].values())
        )

    def test_diagnosis_distinguishes_tradeoff_from_optimizer_explosion(self) -> None:
        diagnosis = self.preflight.manifest["v1_diagnosis"]
        self.assertTrue(diagnosis["failure_tail_was_inactive"])
        self.assertLess(diagnosis["observed_approx_kl_range"][1], 0.0001)
        self.assertLess(diagnosis["observed_clip_fraction_range"][1], 0.0013)
        deltas = diagnosis["deltas_by_opponent"]
        self.assertGreater(deltas["own_bc"]["update2_minus_update0_pp"], 0.0)
        self.assertLess(deltas["source_marnie"]["update2_minus_update0_pp"], -10.0)
        self.assertLess(deltas["u472"]["update4_minus_update0_pp"], -10.0)
        self.assertGreater(deltas["lucario"]["update4_minus_update0_pp"], 0.0)

    def test_smoke_uses_half_scale_even_quotas(self) -> None:
        command = v2.build_command(v2.PHASES["smoke"])
        quotas = quota_values(command)
        self.assertEqual(sorted(quotas.values()), [8, 8, 16, 16, 48])
        self.assertEqual(sum(quotas.values()), 96)
        self.assertEqual(command_value(command, "--updates"), "1")
        self.assertEqual(command_value(command, "--eval-games"), "32")
        self.assertEqual(command_value(command, "--output-dir"), str(v2.SMOKE_OUTPUT))

    def test_scope_excludes_pending_sweep_packaging_and_submission(self) -> None:
        scope = self.preflight.manifest["scope"]
        self.assertFalse(scope["pending_16_strategy_results_read"])
        self.assertFalse(scope["package"])
        self.assertFalse(scope["upload"])
        self.assertFalse(scope["submission"])
        lowered = " ".join(self.command).lower()
        for forbidden in ("kaggle", "submit", "submission", "upload", "package"):
            self.assertNotIn(forbidden, lowered)

    def test_default_dry_run_is_zero_write_and_zero_child(self) -> None:
        self.assertFalse(v2.OUTPUT_ROOT.exists())
        stdout = io.StringIO()
        with mock.patch.object(v2.subprocess, "run") as child:
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(v2.main(["--phase", "full", "--dry-run"]), 0)
        child.assert_not_called()
        self.assertFalse(v2.OUTPUT_ROOT.exists())
        self.assertIn(self.preflight.manifest_sha256, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
