from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_gold_push_marnie_ppo as launcher  # noqa: E402


def command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def repeated_pairs(command: list[str], flag: str) -> list[tuple[str, str]]:
    return [
        (command[index + 1], command[index + 2])
        for index, value in enumerate(command)
        if value == flag
    ]


class GoldPushMarniePpoLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full_phase = launcher.PHASES["full"]
        cls.smoke_phase = launcher.PHASES["smoke"]
        cls.full_command = launcher.build_command(cls.full_phase)
        cls.smoke_command = launcher.build_command(cls.smoke_phase)
        # The real v1 output may legitimately exist after training.  Preflight
        # behavior is tested against an isolated never-used target instead of
        # weakening the launcher's refusal-to-reuse contract.
        cls._preflight_temporary = tempfile.TemporaryDirectory()
        cls._preflight_target = Path(cls._preflight_temporary.name) / "full"
        with mock.patch.object(launcher, "OUTPUT_ROOT", cls._preflight_target):
            cls.full_preflight = launcher.build_preflight(cls.full_phase)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._preflight_temporary.cleanup()

    def test_full_budget_and_exact_even_quotas(self) -> None:
        command = self.full_command
        self.assertEqual(command_value(command, "--updates"), "4")
        self.assertEqual(command_value(command, "--games-per-update"), "192")
        self.assertEqual(self.full_phase.updates * self.full_phase.games_per_update, 768)

        quotas = {
            name: int(raw_quota)
            for name, raw_quota in repeated_pairs(command, "--opponent-base-quota")
        }
        self.assertEqual(quotas["bc"], 96)
        self.assertEqual(len(quotas), 5)
        self.assertEqual(sorted(value for name, value in quotas.items() if name != "bc"), [24] * 4)
        self.assertEqual(sum(quotas.values()), 192)
        self.assertTrue(all(value % 2 == 0 for value in quotas.values()))
        self.assertIn("--opponent-quota-seat-balance", command)

    def test_smoke_budget_preserves_full_protocol_ratio(self) -> None:
        command = self.smoke_command
        self.assertEqual(command_value(command, "--updates"), "1")
        self.assertEqual(command_value(command, "--games-per-update"), "96")
        quotas = {
            name: int(raw_quota)
            for name, raw_quota in repeated_pairs(command, "--opponent-base-quota")
        }
        self.assertEqual(quotas["bc"], 48)
        self.assertEqual(sorted(value for name, value in quotas.items() if name != "bc"), [12] * 4)
        self.assertEqual(sum(quotas.values()), 96)
        self.assertEqual(command_value(command, "--eval-games"), "32")

    def test_all_five_permanent_opponents_are_exact(self) -> None:
        command = self.full_command
        extras = repeated_pairs(command, "--extra-opponent")
        expected = [
            (str(value.checkpoint), str(value.deck)) for value in launcher.OPPONENTS
        ]
        self.assertEqual(extras, expected)
        self.assertEqual(command.count("--extra-opponent"), 4)
        self.assertEqual(command_value(command, "--bc-checkpoint"), str(launcher.LEARNER))
        self.assertEqual(
            command_value(command, "--kl-reference-checkpoint"),
            str(launcher.LEARNER),
        )

    def test_full_hyperparameters_are_locked(self) -> None:
        command = self.full_command
        expected = {
            "--ppo-epochs": "2",
            "--minibatch-size": "512",
            "--learning-rate": "0.000012",
            "--value-learning-rate": "0.000025",
            "--gamma": "1.0",
            "--gae-lambda": "1.0",
            "--advantage-normalization": "per_opponent",
            "--clip-ratio": "0.12",
            "--value-coefficient": "0.25",
            "--value-trunk-gradient-scale": "0.02",
            "--entropy-coefficient": "0.0005",
            "--max-grad-norm": "0.5",
            "--policy-temperature": "0.8",
            "--trainable-scope": "last_block_heads",
            "--actor-reduction": "episode_mean",
            "--actor-value-gradient-mode": "scalar",
            "--bc-kl-start": "0.020",
            "--bc-kl-end": "0.016",
            "--target-kl": "0.004",
        }
        for flag, value in expected.items():
            self.assertEqual(command_value(command, flag), value, flag)

    def test_ordered_bc_replay_and_context34_contract(self) -> None:
        command = self.full_command
        expected = {
            "--bc-replay-data": str(launcher.REPLAY),
            "--bc-replay-split": "train",
            "--bc-replay-batches": "72",
            "--bc-replay-batch-size": "256",
            "--bc-replay-workers": "8",
            "--bc-replay-steps": "2",
            "--bc-replay-lr-scale": "0.05",
            "--bc-replay-loss": "ordered",
            "--bc-replay-order-context-weight": "8.0",
            "--bc-replay-non-context34-fixed-multi-action-order-weight": "1.0",
            "--bc-replay-context34-rows-per-batch": "4",
        }
        for flag, value in expected.items():
            self.assertEqual(command_value(command, flag), value, flag)

    def test_failure_tail_is_opt_in_and_legacy_flag_is_absent(self) -> None:
        command = self.full_command
        self.assertEqual(command.count("--failed-attempt-as-loss"), 1)
        self.assertEqual(command_value(command, "--failed-loss-tail-transitions"), "32")
        self.assertNotIn("--truncation-as-loss", command)

    def test_robust_selection_uses_every_permanent_opponent(self) -> None:
        command = self.full_command
        self.assertIn("--eval-all-permanent-opponents", command)
        self.assertEqual(command_value(command, "--selection-aggregation"), "min")
        self.assertEqual(command_value(command, "--eval-games"), "256")
        self.assertEqual(command_value(command, "--eval-interval"), "2")
        self.assertEqual(command_value(command, "--checkpoint-interval"), "1")

    def test_versioned_outputs_do_not_overlap(self) -> None:
        self.assertEqual(launcher.output_dir(self.full_phase), launcher.OUTPUT_ROOT)
        self.assertEqual(launcher.output_dir(self.smoke_phase), launcher.SMOKE_OUTPUT)
        self.assertNotEqual(launcher.OUTPUT_ROOT, launcher.SMOKE_OUTPUT)
        self.assertEqual(launcher.OUTPUT_ROOT.name, "ppo_marnie_tail32_v1")
        self.assertEqual(launcher.SMOKE_OUTPUT.name, "ppo_marnie_tail32_v1_smoke")

    def test_live_preflight_records_all_full_sha256_bindings(self) -> None:
        manifest = self.full_preflight.manifest
        inputs = manifest["inputs"]
        for label, expected in launcher.FILE_SHA256.items():
            self.assertEqual(inputs[label]["sha256"], expected, label)
            self.assertEqual(inputs[label]["expected_sha256"], expected, label)
            self.assertEqual(len(expected), 64)
        self.assertEqual(inputs["trainer"]["sha256"], launcher.TRAIN_PPO_SHA256)
        self.assertEqual(inputs["python"]["sha256"], launcher.PYTHON_SHA256)

    def test_live_checkpoint_and_deck_metadata_are_compatible(self) -> None:
        manifest = self.full_preflight.manifest
        checkpoints = manifest["checkpoints"]
        self.assertEqual(checkpoints["learner_bc"]["feature_version"], launcher.BC_FEATURE)
        self.assertEqual(checkpoints["current_lucario_bc"]["deck_hash"], launcher.LUCARIO_DECK_HASH)
        self.assertEqual(checkpoints["current_froslass_bc"]["deck_hash"], launcher.FROSLASS_DECK_HASH)
        self.assertEqual(checkpoints["source_marnie_bc"]["deck_hash"], launcher.MARNIE_DECK_HASH)
        self.assertEqual(checkpoints["submitted_u472"]["feature_version"], launcher.PPO_FEATURE)
        self.assertEqual(checkpoints["submitted_u472"]["update"], 472)
        for record in checkpoints.values():
            self.assertEqual(record["model_config"], launcher.EXPECTED_MODEL_CONFIG)
            self.assertEqual(len(record["model_state_schema_sha256"]), 64)
        self.assertTrue(all(manifest["learner_lineage"]["checks"].values()))
        self.assertTrue(all(manifest["replay"]["checks"].values()))

    def test_manifest_is_self_consistent_and_complete(self) -> None:
        preflight = self.full_preflight
        manifest = preflight.manifest
        self.assertEqual(preflight.manifest_sha256, launcher.sha256_json(manifest))
        self.assertEqual(manifest["command_sha256"], launcher.sha256_json(preflight.command))
        self.assertEqual(manifest["rollout"]["total_valid_games"], 768)
        self.assertEqual(manifest["rollout"]["quota_sum"], 192)
        self.assertTrue(manifest["gates"]["all_input_sha256_exact"])
        self.assertTrue(manifest["gates"]["trainer_cli_contract"]["all_requested_flags_registered"])
        self.assertFalse(manifest["scope"]["package"])
        self.assertFalse(manifest["scope"]["upload"])
        self.assertFalse(manifest["scope"]["submission"])

    def test_command_contains_no_packaging_upload_or_submission(self) -> None:
        lowered = " ".join(self.full_command).lower()
        for forbidden in ("kaggle", "submit", "submission", "upload", "package"):
            self.assertNotIn(forbidden, lowered)

    def test_wrong_file_hash_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            launcher.validate_file_binding(
                "marnie_deck",
                launcher.MARNIE_DECK,
                "0" * 64,
            )

    def test_existing_file_directory_or_symlink_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing_file = root / "file"
            existing_file.touch()
            existing_directory = root / "directory"
            existing_directory.mkdir()
            symlink = root / "symlink"
            symlink.symlink_to(root / "missing")
            for target in (existing_file, existing_directory, symlink):
                with self.subTest(target=target):
                    with self.assertRaises(FileExistsError):
                        launcher.assert_target_absent(target)

    def test_default_and_explicit_dry_run_do_not_start_child_or_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "full"
            self.assertFalse(target.exists())
            for argv in (["--phase", "full"], ["--phase", "full", "--dry-run"]):
                with self.subTest(argv=argv):
                    output = io.StringIO()
                    with mock.patch.object(launcher, "OUTPUT_ROOT", target):
                        expected = launcher.build_preflight(self.full_phase)
                        with mock.patch.object(launcher.subprocess, "run") as child:
                            with contextlib.redirect_stdout(output):
                                self.assertEqual(launcher.main(argv), 0)
                    child.assert_not_called()
                    self.assertIn(expected.manifest_sha256, output.getvalue())
                    self.assertFalse(target.exists())

    def test_execute_requires_matching_dry_run_manifest_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --expected-manifest-sha256"):
            launcher.main(["--phase", "full", "--execute"])
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "full"
            with mock.patch.object(launcher, "OUTPUT_ROOT", target):
                with mock.patch.object(launcher.subprocess, "run") as child:
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(RuntimeError, "Manifest SHA-256 mismatch"):
                            launcher.main(
                                [
                                    "--phase",
                                    "full",
                                    "--execute",
                                    "--expected-manifest-sha256",
                                    "0" * 64,
                                ]
                            )
        child.assert_not_called()

    def test_execute_path_is_one_shot_and_writes_manifest_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "full"
            with mock.patch.object(launcher, "OUTPUT_ROOT", target):
                preflight = launcher.build_preflight(self.full_phase)
                completed = SimpleNamespace(returncode=0)
                with mock.patch.object(
                    launcher.subprocess, "run", return_value=completed
                ) as child:
                    self.assertEqual(launcher.execute(preflight), 0)
                child.assert_called_once()
                call = child.call_args
                self.assertEqual(call.args[0], preflight.command)
                self.assertEqual(call.kwargs["cwd"], launcher.ROOT)
                self.assertFalse(call.kwargs["shell"])
                self.assertFalse(call.kwargs["check"])
                manifest_path = target / "launcher_manifest.json"
                result_path = target / "launcher_result.json"
                self.assertTrue(manifest_path.is_file())
                self.assertTrue(result_path.is_file())
                stored = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(stored["manifest_sha256"], preflight.manifest_sha256)
                with self.assertRaises(FileExistsError):
                    launcher.assert_target_absent(target)


if __name__ == "__main__":
    unittest.main()
