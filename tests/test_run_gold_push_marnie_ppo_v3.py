from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_gold_push_marnie_ppo_v3 as v3  # noqa: E402


def command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def quota_values(command: list[str]) -> dict[str, int]:
    return {
        command[index + 1]: int(command[index + 2])
        for index, token in enumerate(command)
        if token == "--opponent-base-quota"
    }


def extra_opponents(command: list[str]) -> list[tuple[str, str]]:
    return [
        (command[index + 1], command[index + 2])
        for index, token in enumerate(command)
        if token == "--extra-opponent"
    ]


class GoldPushMarniePpoV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase = v3.PHASES["full"]
        cls.command = v3.build_command(cls.phase)
        cls.preflight = v3.build_preflight(cls.phase)

    def test_fresh_full_run_is_small_and_uses_a_new_output(self) -> None:
        self.assertEqual(command_value(self.command, "--updates"), "2")
        self.assertEqual(command_value(self.command, "--games-per-update"), "192")
        self.assertEqual(command_value(self.command, "--output-dir"), str(v3.OUTPUT_ROOT))
        self.assertNotIn("--resume", self.command)
        self.assertNotEqual(v3.OUTPUT_ROOT, v3.RAW_ROOT / "ppo_marnie_tail32_v2_lightanchor")
        self.assertEqual(
            self.preflight.manifest["expected_terminal_checkpoint"],
            str(v3.OUTPUT_ROOT / "checkpoints/update-0002.pt"),
        )

    def test_same_frozen_bc_parent_but_source_is_the_strong_kl_reference(self) -> None:
        self.assertEqual(command_value(self.command, "--bc-checkpoint"), str(v3.base.LEARNER))
        self.assertEqual(
            command_value(self.command, "--kl-reference-checkpoint"),
            str(v3.base.SOURCE_MARNIE_BC),
        )
        parent = self.preflight.manifest["parent_choice"]
        self.assertTrue(parent["fresh_run"])
        self.assertEqual(parent["initial_checkpoint_sha256"], v3.base.FILE_SHA256["learner_bc"])

    def test_tail_and_guard_opponents_are_all_permanent_and_unique(self) -> None:
        extras = extra_opponents(self.command)
        expected = [(str(item.checkpoint), str(item.deck)) for item in v3.OPPONENTS]
        self.assertEqual(extras, expected)
        self.assertEqual(len(extras), 5)
        names = [v3.opponent_name(item.checkpoint, item.deck) for item in v3.OPPONENTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn((str(v3.RAIHAN), str(v3.base.MARNIE_DECK)), extras)
        self.assertIn((str(v3.KDCYBERDUDE_TRAIN), str(v3.base.MARNIE_DECK)), extras)
        self.assertIn((str(v3.base.SOURCE_MARNIE_BC), str(v3.MARNIE_NAMED_DECK)), extras)
        self.assertIn((str(v3.base.U472), str(v3.base.MARNIE_DECK)), extras)
        self.assertIn((str(v3.base.FROSLASS_BC), str(v3.base.FROSLASS_DECK)), extras)

    def test_full_quotas_are_tail_weighted_exact_and_seat_balanced(self) -> None:
        expected = {
            "bc": 48,
            v3.opponent_name(v3.RAIHAN, v3.base.MARNIE_DECK): 40,
            v3.opponent_name(v3.KDCYBERDUDE_TRAIN, v3.base.MARNIE_DECK): 32,
            v3.opponent_name(v3.base.SOURCE_MARNIE_BC, v3.MARNIE_NAMED_DECK): 24,
            v3.opponent_name(v3.base.U472, v3.base.MARNIE_DECK): 24,
            v3.opponent_name(v3.base.FROSLASS_BC, v3.base.FROSLASS_DECK): 24,
        }
        self.assertEqual(quota_values(self.command), expected)
        self.assertEqual(sum(expected.values()), 192)
        self.assertTrue(all(value > 0 and value % 2 == 0 for value in expected.values()))
        self.assertEqual(self.preflight.manifest["rollout"]["tail_quota_games"], 72)
        self.assertEqual(self.preflight.manifest["rollout"]["tail_and_bottom4_quota_games"], 120)

    def test_ppo_is_low_lr_strong_kl_and_bc_replay_is_strengthened(self) -> None:
        self.assertEqual(command_value(self.command, "--learning-rate"), "0.000006")
        self.assertEqual(command_value(self.command, "--bc-kl-start"), "0.030")
        self.assertEqual(command_value(self.command, "--bc-kl-end"), "0.030")
        self.assertEqual(command_value(self.command, "--target-kl"), "0.004")
        self.assertEqual(command_value(self.command, "--ppo-epochs"), "2")
        self.assertEqual(command_value(self.command, "--bc-replay-steps"), "4")
        self.assertEqual(command_value(self.command, "--bc-replay-lr-scale"), "0.10")
        self.assertEqual(command_value(self.command, "--bc-replay-loss"), "ordered")
        self.assertFalse(self.preflight.manifest["bc_replay"]["context22_direct_stratification"])

    def test_fixed_min_is_internal_proxy_and_true_cvar_stays_external(self) -> None:
        self.assertIn("--eval-all-permanent-opponents", self.command)
        self.assertEqual(command_value(self.command, "--selection-aggregation"), "min")
        self.assertEqual(command_value(self.command, "--eval-games"), "256")
        self.assertEqual(command_value(self.command, "--eval-interval"), "1")
        selection = self.preflight.manifest["selection"]
        self.assertFalse(selection["internal"]["native_cvar_available"])
        self.assertEqual(selection["internal"]["total_evaluation_games"], 4608)
        self.assertIn("bottom-quartile CVaR", selection["external_required"])

    def test_raw_panel_evidence_records_only_cvar_failure(self) -> None:
        panel = self.preflight.manifest["frozen_raw_audit"]["raw_terminal_panel"]
        self.assertEqual(panel["macro"], 0.630615234375)
        self.assertEqual((panel["seat_0"], panel["seat_1"]), (0.638671875, 0.62255859375))
        self.assertEqual(panel["minimum_archetype"], 0.548828125)
        self.assertEqual(panel["bottom_quartile_cvar"], 0.4931640625)
        self.assertEqual(panel["only_failed_gate"], "bottom_quartile_cvar")
        self.assertEqual(panel["bottom_four"]["marnie_raihan_recent7_clone"], 0.453125)
        self.assertEqual(panel["bottom_four"]["marnie_kdcyberdude_recent7_clone"], 0.484375)

    def test_raw_context22_failure_is_explicit_not_hidden(self) -> None:
        behavior = self.preflight.manifest["frozen_raw_audit"]["raw_behavior"]
        self.assertEqual(behavior["only_failed_gate"], "major_context_drop_vs_source_at_most_0_01")
        context22 = behavior["context_22"]
        self.assertEqual((context22["rows"], context22["fixed_rows"], context22["flexible_rows"]), (657, 0, 657))
        self.assertEqual(context22["candidate_raw_ordered"], 0.2831050228310502)
        self.assertEqual(context22["source_ordered"], 0.4459665144596651)
        self.assertEqual(context22["raw_delta"], -0.1628614916286149)

    def test_hybrid_is_deployment_only_and_new_candidate_must_be_retested(self) -> None:
        deployment = self.preflight.manifest["deployment_order"]
        self.assertEqual(deployment["ppo_training_learner_order"], "existing raw Plackett-Luce order")
        self.assertFalse(deployment["candidate_canonical_order"])
        self.assertTrue(deployment["candidate_hybrid_order"])
        self.assertEqual(deployment["required_evaluator_flag"], "--candidate-hybrid-order")
        behavior_gate = self.preflight.manifest["promotion_gates"]["hybrid_behavior_for_new_checkpoint"]
        self.assertTrue(behavior_gate["old_candidate_metrics_are_not_reusable"])

    def test_all_evidence_and_new_inputs_are_sha_locked(self) -> None:
        inputs = self.preflight.manifest["inputs"]
        for label, digest in v3.EXTRA_SHA256.items():
            self.assertEqual(inputs[label]["sha256"], digest)
        self.assertEqual(inputs["trainer"]["sha256"], v3.base.TRAIN_PPO_SHA256)
        self.assertEqual(inputs["v1_launcher"]["sha256"], v3.BASE_LAUNCHER_SHA256)
        self.assertEqual(v3.EXTRA_SHA256["kdcyberdude_panel"], v3.EXTRA_SHA256["kdcyberdude_train"])
        self.assertTrue(self.preflight.manifest["gates"]["kd_last_byte_identical_to_panel_best"])

    def test_smoke_is_half_scale_and_does_not_reuse_full_output(self) -> None:
        phase = v3.PHASES["smoke"]
        command = v3.build_command(phase)
        self.assertEqual(command_value(command, "--updates"), "1")
        self.assertEqual(command_value(command, "--games-per-update"), "96")
        self.assertEqual(sorted(quota_values(command).values()), [12, 12, 12, 16, 20, 24])
        self.assertEqual(command_value(command, "--output-dir"), str(v3.SMOKE_OUTPUT))
        self.assertNotEqual(v3.SMOKE_OUTPUT, v3.OUTPUT_ROOT)

    def test_scope_has_no_packaging_upload_or_submission_path(self) -> None:
        scope = self.preflight.manifest["scope"]
        self.assertFalse(scope["package"])
        self.assertFalse(scope["upload"])
        self.assertFalse(scope["submission"])
        lowered = " ".join(self.command).lower()
        for forbidden in ("kaggle", "submit", "submission", "upload", "package"):
            self.assertNotIn(forbidden, lowered)

    def test_default_dry_run_is_zero_write_and_zero_child(self) -> None:
        self.assertFalse(v3.OUTPUT_ROOT.exists())
        stdout = io.StringIO()
        with mock.patch.object(v3.subprocess, "run") as child:
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(v3.main(["--phase", "full", "--dry-run"]), 0)
        child.assert_not_called()
        self.assertFalse(v3.OUTPUT_ROOT.exists())
        self.assertIn(self.preflight.manifest_sha256, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
