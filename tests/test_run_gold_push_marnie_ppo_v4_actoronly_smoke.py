from __future__ import annotations

import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_gold_push_marnie_ppo_v4_actoronly_smoke as v4  # noqa: E402
import train_ppo  # noqa: E402
from tests.test_train_ppo_unit import ActorReductionGradientAuditTests  # noqa: E402


def command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def quota_values(command: list[str]) -> dict[str, int]:
    return {
        command[index + 1]: int(command[index + 2])
        for index, token in enumerate(command)
        if token == "--opponent-base-quota"
    }


def evaluation_row(wins: int, *, invalid: int = 0) -> dict[str, object]:
    return {
        "valid_games": 128,
        "wins": wins,
        "losses": 128 - wins,
        "draws": 0,
        "invalid_games": invalid,
        "win_rate": wins / 128.0,
    }


class GoldPushMarnieV4ActorOnlySmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.command = v4.build_command()
        cls.preflight = v4.build_preflight()

    def test_single_fresh_192_game_smoke_has_exact_balanced_quotas(self) -> None:
        self.assertEqual(command_value(self.command, "--updates"), "1")
        self.assertEqual(command_value(self.command, "--games-per-update"), "192")
        self.assertEqual(command_value(self.command, "--bc-checkpoint"), str(v4.base.LEARNER))
        self.assertNotIn("--resume", self.command)
        self.assertEqual(quota_values(self.command), v4.FIXED_QUOTAS)
        self.assertEqual(sum(v4.FIXED_QUOTAS.values()), 192)
        self.assertTrue(all(value > 0 and value % 2 == 0 for value in v4.FIXED_QUOTAS.values()))

    def test_optimizer_contract_is_small_actor_only(self) -> None:
        expected = {
            "--ppo-epochs": "1",
            "--minibatch-size": "1024",
            "--learning-rate": "0.0000015",
            "--trainable-scope": "heads",
            "--value-coefficient": "0.0",
            "--value-trunk-gradient-scale": "0.0",
            "--value-learning-rate": "0.00000001",
            "--weight-decay": "0.0",
            "--max-grad-norm": "0.25",
            "--bc-kl-start": "0.04",
            "--bc-kl-end": "0.04",
            "--target-kl": "0.00005",
            "--bc-replay-steps": "2",
            "--bc-replay-lr-scale": "0.10",
        }
        for flag, value in expected.items():
            self.assertEqual(command_value(self.command, flag), value)
        self.assertEqual(
            command_value(self.command, "--kl-reference-checkpoint"),
            str(v4.base.LEARNER),
        )

    def test_actual_trainer_step_leaves_value_and_trunk_byte_exact(self) -> None:
        model, transitions, model_config, _ = ActorReductionGradientAuditTests.fixture()
        for index, transition in enumerate(transitions):
            transition["outcome_target"] = float(index % 2)
        reference = copy.deepcopy(model)
        actor, value, manifest = train_ppo.configure_trainable_scope(model, "heads")
        self.assertTrue(all(name.startswith(v4.ACTOR_PREFIXES) for name in manifest["actor_parameter_names"]))
        self.assertTrue(all(name.startswith(v4.VALUE_PREFIX) for name in manifest["value_parameter_names"]))
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        optimizer = torch.optim.AdamW(
            [
                {"params": actor, "lr": 1e-4},
                {"params": value, "lr": 1e-8},
            ],
            eps=1e-5,
            weight_decay=0.0,
        )
        config = SimpleNamespace(
            ppo_objective="standard",
            advantage_normalization="per_opponent",
            actor_reduction="episode_mean",
            opponent_base_quotas={"short": 1, "long": 1},
            games_per_update=2,
            learning_rate_schedule="constant",
            learning_rate=1e-4,
            value_learning_rate=1e-8,
            updates=1,
            schedule_start_update=1,
            ppo_epochs=1,
            minibatch_size=len(transitions),
            policy_temperature=0.8,
            clip_ratio=0.12,
            value_coefficient=0.0,
            value_trunk_gradient_scale=0.0,
            actor_value_gradient_mode="scalar",
            entropy_coefficient=0.0005,
            max_grad_norm=0.25,
            bc_kl_start=0.04,
            bc_kl_end=0.04,
            target_kl=1.0,
        )
        result = train_ppo.ppo_update(
            model,
            reference,
            optimizer,
            transitions,
            model_config,
            config,
            torch.device("cpu"),
            update=1,
        )
        after = model.state_dict()
        actor_changed = []
        for name, tensor in before.items():
            if name.startswith(v4.ACTOR_PREFIXES):
                actor_changed.append(not torch.equal(tensor, after[name]))
            else:
                self.assertTrue(torch.equal(tensor, after[name]), name)
        self.assertTrue(any(actor_changed))
        self.assertEqual(result["value_grad_norm"], 0.0)

    def test_tensor_audit_accepts_only_small_actor_delta(self) -> None:
        parent = {
            "actor_query.weight": torch.ones(4),
            "value_head.weight": torch.ones(3),
            "transformer.layers.0.weight": torch.ones(2),
        }
        candidate = {name: tensor.clone() for name, tensor in parent.items()}
        candidate["actor_query.weight"][0] += 1e-6
        audit = v4.tensor_state_audit(parent, candidate)
        self.assertTrue(audit["pass"])
        self.assertTrue(audit["actor_changed"])
        self.assertTrue(audit["value_tensors_byte_equal"])
        self.assertTrue(audit["trunk_tensors_byte_equal"])

        candidate["value_head.weight"][0] += 1e-6
        rejected = v4.tensor_state_audit(parent, candidate)
        self.assertFalse(rejected["pass"])
        self.assertFalse(rejected["value_tensors_byte_equal"])

    def test_tensor_audit_treats_signed_zero_as_a_byte_change(self) -> None:
        parent = {
            "actor_query.weight": torch.ones(1),
            "value_head.weight": torch.tensor([0.0]),
            "transformer.layers.0.weight": torch.ones(1),
        }
        candidate = {name: tensor.clone() for name, tensor in parent.items()}
        candidate["actor_query.weight"][0] += 1e-6
        candidate["value_head.weight"][0] = -0.0
        self.assertTrue(torch.equal(parent["value_head.weight"], candidate["value_head.weight"]))
        audit = v4.tensor_state_audit(parent, candidate)
        self.assertFalse(audit["pass"])
        self.assertFalse(audit["value_tensors_byte_equal"])

    def test_performance_gate_requires_all_nonregression_conditions(self) -> None:
        initial = {
            name: evaluation_row(64)
            for name in v4.PANEL_LABELS.values()
        }
        passing = {name: evaluation_row(64) for name in v4.PANEL_LABELS.values()}
        passing[v4.PANEL_LABELS["raihan"]] = evaluation_row(66)
        gate = v4.performance_gate(initial, passing)
        self.assertTrue(gate["pass"])
        self.assertTrue(all(gate["checks"].values()))

        guard_drop = dict(passing)
        guard_drop[v4.PANEL_LABELS["own_bc"]] = evaluation_row(61)
        rejected = v4.performance_gate(initial, guard_drop)
        self.assertFalse(rejected["pass"])
        self.assertFalse(rejected["checks"]["each_guard_within_drop_limit"])

        inconsistent = copy.deepcopy(passing)
        inconsistent[v4.PANEL_LABELS["u472"]]["wins"] = 65
        rejected = v4.performance_gate(initial, inconsistent)
        self.assertFalse(rejected["pass"])
        self.assertFalse(rejected["checks"]["outcomes_sum_to_valid_games"])
        self.assertFalse(rejected["checks"]["reported_win_rates_exact"])

    def test_posterior_manifest_freezes_every_hard_stop(self) -> None:
        gates = self.preflight.manifest["posterior_stop_gates"]
        self.assertEqual(gates["movement"]["approx_kl_max"], 2e-5)
        self.assertEqual(gates["movement"]["bc_anchor_kl_max"], 5e-4)
        self.assertEqual(gates["actor_only"]["actor_head_relative_l2_max"], 5e-5)
        self.assertTrue(gates["actor_only"]["all_value_tensors_byte_equal_update0"])
        self.assertTrue(gates["actor_only"]["all_trunk_tensors_byte_equal_update0"])
        self.assertEqual(gates["replay"]["steps_exact"], 2)
        self.assertEqual(gates["replay"]["rows_exact"], 512)
        self.assertEqual(gates["replay"]["loss_mode_exact"], "ordered")
        self.assertEqual(gates["performance"]["each_guard_drop_max"], 2 / 128)
        self.assertEqual(gates["performance"]["at_least_one_tail_improvement"], 2 / 128)
        self.assertEqual(gates["failure_action"], "reject learned checkpoint, stop route, never launch full")

    def test_rejected_r2_is_evidence_only_and_never_a_parent(self) -> None:
        diagnosis = self.preflight.manifest["rejected_r2_diagnosis"]
        self.assertEqual(diagnosis["status"], "experimentally_rejected_no_full")
        self.assertEqual((diagnosis["initial_min"], diagnosis["learned_min"]), (0.53125, 0.4375))
        self.assertFalse(diagnosis["learned_checkpoint_may_seed_v4"])
        parent = self.preflight.manifest["fresh_parent"]
        self.assertFalse(parent["learned_r2_checkpoint_used"])
        self.assertEqual(parent["bc_sha256"], v4.base.FILE_SHA256["learner_bc"])

    def test_scope_has_no_full_confirmation_package_or_submission(self) -> None:
        scope = self.preflight.manifest["scope"]
        self.assertFalse(scope["full_training"])
        self.assertFalse(scope["external_confirmation"])
        self.assertFalse(scope["package"])
        self.assertFalse(scope["upload"])
        self.assertFalse(scope["submission"])
        lowered = " ".join(self.command).lower()
        for forbidden in ("kaggle", "submit", "submission", "upload", "package"):
            self.assertNotIn(forbidden, lowered)

    def test_default_dry_run_is_zero_write_and_zero_child(self) -> None:
        self.assertFalse(v4.OUTPUT.exists())
        stdout = io.StringIO()
        with mock.patch.object(v4.subprocess, "run") as child:
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(v4.main(["--dry-run"]), 0)
        child.assert_not_called()
        self.assertFalse(v4.OUTPUT.exists())
        self.assertIn(self.preflight.manifest_sha256, stdout.getvalue())

    def test_execute_requires_the_reviewed_freeze_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            v4.main(["--execute"])
        with self.assertRaisesRegex(ValueError, "only valid"):
            v4.main(["--expected-manifest-sha256", "0" * 64])


if __name__ == "__main__":
    unittest.main()
