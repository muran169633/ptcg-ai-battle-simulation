from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_ppo_actor_reduction_audit as audit_runner  # noqa: E402
import run_ppo_gold_ab as gold_runner  # noqa: E402


class ActorReductionAuditRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "audit-output"
        self.base_output = self.root / "training-output"
        self.command = [
            sys.executable,
            str(gold_runner.TRAIN_SCRIPT),
            "--output-dir",
            str(self.base_output),
            "--actor-reduction",
            "quota_group_mean",
            "--skip-initial-eval",
        ]
        self.training_plan = {
            "schema_version": gold_runner.PLAN_SCHEMA,
            "inputs": {
                "train_script": str(gold_runner.TRAIN_SCRIPT),
                "train_script_sha256": gold_runner.file_sha256(
                    gold_runner.TRAIN_SCRIPT
                ),
            },
            "protocol": {"actor_reduction": "quota_group_mean"},
            "branches": {
                "B_gold_league": {
                    "runs": [
                        {
                            "seed": 20260730,
                            "output_dir": str(self.base_output),
                            "command": self.command,
                        }
                    ]
                }
            },
        }
        self.training_registration = {
            "schema_version": gold_runner.PREREGISTRATION_SCHEMA,
            "created_at": "frozen",
            "plan_sha256": gold_runner.canonical_json_sha256(
                self.training_plan
            ),
            "plan": self.training_plan,
            "status": "preregistered_not_started",
        }
        self.training_path = self.root / "training.preregister.json"
        self.training_path.write_text(
            json.dumps(self.training_registration),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_plan_selects_only_gold_and_only_adds_audit_semantics(self) -> None:
        plan = audit_runner.build_audit_plan(
            self.training_path,
            self.training_registration,
            output_dir=self.output,
            seed=20260730,
        )
        command = list(plan["command"])
        self.assertEqual(plan["branch"], "B_gold_league")
        self.assertEqual(
            audit_runner.command_value(command, "--output-dir"),
            str(self.output.resolve()),
        )
        self.assertEqual(
            audit_runner.command_value(command, "--actor-reduction"),
            "quota_group_mean",
        )
        self.assertIn("--skip-initial-eval", command)
        self.assertEqual(command.count("--actor-reduction-audit-only"), 1)
        normalized = list(command)
        audit_runner.replace_command_value(
            normalized,
            "--output-dir",
            str(self.base_output),
        )
        normalized.remove("--actor-reduction-audit-only")
        self.assertEqual(normalized, self.command)
        self.assertEqual(plan["protocol"]["optimizer_steps"], 0)
        self.assertEqual(plan["protocol"]["checkpoint_writes"], 0)

    def test_load_rejects_tampered_or_wrong_reduction(self) -> None:
        tampered = dict(self.training_registration)
        tampered["plan_sha256"] = "0" * 64
        self.training_path.write_text(
            json.dumps(tampered),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            audit_runner.load_training_preregistration(self.training_path)

        wrong_plan = {
            **self.training_plan,
            "protocol": {"actor_reduction": "transition_mean"},
        }
        wrong = {
            **self.training_registration,
            "plan": wrong_plan,
            "plan_sha256": gold_runner.canonical_json_sha256(wrong_plan),
        }
        self.training_path.write_text(
            json.dumps(wrong),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "quota_group_mean"):
            audit_runner.load_training_preregistration(self.training_path)

    def test_episode_mean_plan_reuses_zero_update_audit_path(self) -> None:
        episode_command = list(self.command)
        reduction_index = episode_command.index("--actor-reduction") + 1
        episode_command[reduction_index] = "episode_mean"
        episode_plan = {
            **self.training_plan,
            "protocol": {"actor_reduction": "episode_mean"},
            "branches": {
                "B_gold_league": {
                    "runs": [
                        {
                            "seed": 20260730,
                            "output_dir": str(self.base_output),
                            "command": episode_command,
                        }
                    ]
                }
            },
        }
        episode_registration = {
            **self.training_registration,
            "plan": episode_plan,
            "plan_sha256": gold_runner.canonical_json_sha256(episode_plan),
        }
        self.training_path.write_text(
            json.dumps(episode_registration),
            encoding="utf-8",
        )
        loaded = audit_runner.load_training_preregistration(
            self.training_path
        )
        plan = audit_runner.build_audit_plan(
            self.training_path,
            loaded,
            output_dir=self.output,
            seed=20260730,
        )
        self.assertEqual(
            plan["protocol"]["candidate_reduction"],
            "episode_mean",
        )
        self.assertEqual(
            audit_runner.command_value(
                plan["command"],
                "--actor-reduction",
            ),
            "episode_mean",
        )
        self.assertEqual(
            plan["command"].count("--actor-reduction-audit-only"),
            1,
        )

    def test_result_requires_zero_step_unchanged_model_and_no_checkpoint(
        self,
    ) -> None:
        self.output.mkdir()
        audit = {
            "schema_version": "ptcg-ppo-actor-reduction-audit-v1",
            "status": "completed_no_update",
            "optimizer_steps": 0,
            "model_state_sha256_before": "same",
            "model_state_sha256_after": "same",
            "candidate_reduction": "quota_group_mean",
            "row_weight_audit": {},
            "pre_registered_gate": {"training_authorized": True},
        }
        (self.output / "actor_reduction_audit.json").write_text(
            json.dumps(audit),
            encoding="utf-8",
        )
        (self.output / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "optimization": None,
                    "bc_replay": None,
                    "actor_reduction_audit": audit,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(
            audit_runner.validate_audit_result(self.output),
            audit,
        )
        (self.output / "unexpected.pt").write_bytes(b"checkpoint")
        with self.assertRaisesRegex(RuntimeError, "wrote checkpoints"):
            audit_runner.validate_audit_result(self.output)

    def test_result_rejects_mutation_or_optimizer_step(self) -> None:
        self.output.mkdir()
        path = self.output / "actor_reduction_audit.json"
        for audit in (
            {
                "schema_version": "ptcg-ppo-actor-reduction-audit-v1",
                "status": "completed_no_update",
                "optimizer_steps": 1,
                "model_state_sha256_before": "same",
                "model_state_sha256_after": "same",
                "candidate_reduction": "quota_group_mean",
                "row_weight_audit": {},
                "pre_registered_gate": {"training_authorized": True},
            },
            {
                "schema_version": "ptcg-ppo-actor-reduction-audit-v1",
                "status": "completed_no_update",
                "optimizer_steps": 0,
                "model_state_sha256_before": "before",
                "model_state_sha256_after": "after",
                "candidate_reduction": "quota_group_mean",
                "row_weight_audit": {},
                "pre_registered_gate": {"training_authorized": True},
            },
        ):
            with self.subTest(audit=audit):
                path.write_text(json.dumps(audit), encoding="utf-8")
                (self.output / "metrics.jsonl").write_text(
                    json.dumps(
                        {
                            "optimization": None,
                            "bc_replay": None,
                            "actor_reduction_audit": audit,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(RuntimeError):
                    audit_runner.validate_audit_result(self.output)

    def test_episode_result_requires_equal_game_coverage_invariants(
        self,
    ) -> None:
        self.output.mkdir()
        path = self.output / "actor_reduction_audit.json"
        valid_audit = {
            "schema_version": "ptcg-ppo-actor-reduction-audit-v1",
            "status": "completed_no_update",
            "optimizer_steps": 0,
            "model_state_sha256_before": "same",
            "model_state_sha256_after": "same",
            "candidate_reduction": "episode_mean",
            "row_weight_audit": {
                "mode": "episode_mean",
                "all_games_equal_weight": True,
                "all_transition_rows_accounted_for": True,
                "missing_game_uid_rows": 0,
                "mean_row_multiplier": 1.0,
            },
            "pre_registered_gate": {"training_authorized": True},
        }
        path.write_text(json.dumps(valid_audit), encoding="utf-8")
        (self.output / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "optimization": None,
                    "bc_replay": None,
                    "actor_reduction_audit": valid_audit,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(
            audit_runner.validate_audit_result(
                self.output,
                expected_candidate="episode_mean",
            ),
            valid_audit,
        )
        invalid_audit = {
            **valid_audit,
            "row_weight_audit": {
                **valid_audit["row_weight_audit"],
                "all_games_equal_weight": False,
            },
        }
        path.write_text(json.dumps(invalid_audit), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "coverage invariants"):
            audit_runner.validate_audit_result(
                self.output,
                expected_candidate="episode_mean",
            )


if __name__ == "__main__":
    unittest.main()
