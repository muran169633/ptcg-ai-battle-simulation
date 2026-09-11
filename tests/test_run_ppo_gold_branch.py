from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_ppo_gold_branch as branch_runner  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class GoldBranchRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output_root = self.root / "training-root"
        self.seed = 20260736
        self.b_output = (
            self.output_root
            / branch_runner.BRANCH_NAME
            / f"seed-{self.seed}"
        )
        self.a_output = (
            self.output_root / "A_marnie_control" / f"seed-{self.seed}"
        )
        self.b_command = [
            str(Path(sys.executable).resolve()),
            str(branch_runner.TRAIN_SCRIPT),
            "--output-dir",
            str(self.b_output),
            "--seed",
            str(self.seed),
        ]
        self.a_command = [
            str(Path(sys.executable).resolve()),
            str(branch_runner.TRAIN_SCRIPT),
            "--output-dir",
            str(self.a_output),
            "--seed",
            str(self.seed),
        ]
        self.source_path = self.root / "training.preregistration.json"
        self.plan = self.make_plan()
        self.write_source()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_plan(self) -> dict[str, object]:
        return {
            "schema_version": branch_runner.SOURCE_PLAN_SCHEMA,
            "output_root": str(self.output_root),
            "inputs": {
                "train_script": str(branch_runner.TRAIN_SCRIPT),
                "train_script_sha256": file_sha256(
                    branch_runner.TRAIN_SCRIPT
                ),
                "python_executable": str(Path(sys.executable).resolve()),
            },
            "branches": {
                "A_marnie_control": {
                    "runs": [
                        {
                            "seed": self.seed,
                            "output_dir": str(self.a_output),
                            "command": self.a_command,
                        }
                    ]
                },
                branch_runner.BRANCH_NAME: {
                    "runs": [
                        {
                            "seed": self.seed,
                            "output_dir": str(self.b_output),
                            "command": self.b_command,
                        }
                    ]
                },
            },
            "safety": {
                "local_training_only": True,
                "allowed_child_program": str(
                    branch_runner.TRAIN_SCRIPT
                ),
                "network_calls": False,
                "uploads": False,
                "submission": False,
                "packaging": False,
                "uses_open_submission_code": False,
            },
        }

    def write_source(self) -> None:
        registration = {
            "schema_version": (
                branch_runner.SOURCE_PREREGISTRATION_SCHEMA
            ),
            "created_at": "2026-07-31T00:00:00+00:00",
            "status": "preregistered_not_started",
            "plan_sha256": branch_runner.canonical_json_sha256(self.plan),
            "plan": self.plan,
        }
        self.source_path.write_text(
            json.dumps(registration, indent=2),
            encoding="utf-8",
        )

    def args(self, mode: str) -> object:
        return branch_runner.parse_args(
            [
                "--training-preregistration",
                str(self.source_path),
                "--seed",
                str(self.seed),
                mode,
            ]
        )

    def quiet_run(self, args: object) -> dict[str, object]:
        with redirect_stdout(io.StringIO()):
            return branch_runner.run(args)

    def test_preregister_only_writes_immutable_binding_without_child(self) -> None:
        with mock.patch.object(
            branch_runner.subprocess,
            "run",
        ) as child:
            result = self.quiet_run(self.args("--preregister-only"))

        child.assert_not_called()
        binding = result["binding"]
        preregistration = Path(
            binding["artifacts"]["preregistration"]
        )
        self.assertTrue(preregistration.is_file())
        self.assertFalse(self.output_root.exists())
        self.assertEqual(binding["branch"], branch_runner.BRANCH_NAME)
        self.assertEqual(binding["command"], self.b_command)
        self.assertEqual(
            binding["command_sha256"],
            branch_runner.canonical_json_sha256(self.b_command),
        )
        self.assertEqual(
            binding["source_training_preregistration_sha256"],
            file_sha256(self.source_path),
        )
        self.assertFalse(binding["safety"]["a_branch_execution"])

        with self.assertRaises(FileExistsError):
            self.quiet_run(self.args("--preregister-only"))

    def test_execute_runs_exactly_one_b_command_and_hashes_summary(self) -> None:
        self.quiet_run(self.args("--preregister-only"))

        def fake_run(command: list[str], **kwargs: object) -> object:
            self.assertEqual(command, self.b_command)
            self.assertNotIn("A_marnie_control", " ".join(command))
            self.assertFalse(kwargs["shell"])
            self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
            self.assertIs(kwargs["stderr"], subprocess.STDOUT)
            self.b_output.mkdir(parents=True)
            (self.b_output / "run_config.json").write_text(
                '{"branch":"B_gold_league"}\n',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(
            branch_runner.subprocess,
            "run",
            side_effect=fake_run,
        ) as child:
            result = self.quiet_run(self.args("--execute"))

        self.assertEqual(child.call_count, 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["branch"], branch_runner.BRANCH_NAME)
        self.assertEqual(
            result["command_sha256"],
            branch_runner.canonical_json_sha256(self.b_command),
        )
        binding = json.loads(
            Path(
                self.output_root.with_name(
                    f"{self.output_root.name}."
                    f"{branch_runner.BRANCH_NAME}.seed-{self.seed}."
                    "preregistration.json"
                )
            ).read_text(encoding="utf-8")
        )["binding"]
        run_summary = Path(binding["artifacts"]["run_summary"])
        summary_hash_file = Path(
            binding["artifacts"]["run_summary_sha256"]
        )
        self.assertTrue(run_summary.is_file())
        self.assertTrue(summary_hash_file.is_file())
        self.assertEqual(
            summary_hash_file.read_text(encoding="utf-8").split()[0],
            file_sha256(run_summary),
        )
        self.assertEqual(
            result["output_files"],
            [
                {
                    "path": "run_config.json",
                    "bytes": 27,
                    "sha256": file_sha256(
                        self.b_output / "run_config.json"
                    ),
                }
            ],
        )

    def test_execute_refuses_existing_output_root(self) -> None:
        self.output_root.mkdir()
        with mock.patch.object(
            branch_runner.subprocess,
            "run",
        ) as child:
            with self.assertRaises(FileExistsError):
                self.quiet_run(self.args("--execute"))
        child.assert_not_called()

    def test_rejects_tampered_source_plan_hash(self) -> None:
        source = json.loads(self.source_path.read_text(encoding="utf-8"))
        source["plan"]["output_root"] = str(self.root / "tampered")
        self.source_path.write_text(
            json.dumps(source),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "plan SHA-256 mismatch"):
            self.quiet_run(self.args("--dry-run"))

    def test_rejects_current_train_script_hash_mismatch(self) -> None:
        self.plan["inputs"]["train_script_sha256"] = "0" * 64
        self.write_source()
        with self.assertRaisesRegex(
            ValueError,
            "Current train_ppo.py SHA-256",
        ):
            self.quiet_run(self.args("--dry-run"))

    def test_rejects_duplicate_b_seed(self) -> None:
        duplicate = dict(
            self.plan["branches"][branch_runner.BRANCH_NAME]["runs"][0]
        )
        self.plan["branches"][branch_runner.BRANCH_NAME]["runs"].append(
            duplicate
        )
        self.write_source()
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            self.quiet_run(self.args("--dry-run"))

    def test_rejects_noncanonical_b_output(self) -> None:
        run = self.plan["branches"][branch_runner.BRANCH_NAME]["runs"][0]
        run["output_dir"] = str(self.a_output)
        run["command"] = self.a_command
        self.write_source()
        with self.assertRaisesRegex(
            ValueError,
            "A-branch path or argument|canonical B/seed path",
        ):
            self.quiet_run(self.args("--dry-run"))

    def test_rejects_command_seed_mismatch(self) -> None:
        run = self.plan["branches"][branch_runner.BRANCH_NAME]["runs"][0]
        command = list(run["command"])
        command[command.index("--seed") + 1] = str(self.seed + 1)
        run["command"] = command
        self.write_source()
        with self.assertRaisesRegex(
            ValueError,
            "command --seed differs",
        ):
            self.quiet_run(self.args("--dry-run"))

    def test_rejects_protected_equals_argument_form(self) -> None:
        run = self.plan["branches"][branch_runner.BRANCH_NAME]["runs"][0]
        run["command"] = [
            str(Path(sys.executable).resolve()),
            str(branch_runner.TRAIN_SCRIPT),
            f"--output-dir={self.b_output}",
            "--seed",
            str(self.seed),
        ]
        self.write_source()
        with self.assertRaisesRegex(
            ValueError,
            "uniquely auditable arguments",
        ):
            self.quiet_run(self.args("--dry-run"))

    def test_existing_branch_binding_detects_changed_source_file_hash(self) -> None:
        result = self.quiet_run(self.args("--preregister-only"))
        source = json.loads(self.source_path.read_text(encoding="utf-8"))
        source["created_at"] = "2026-07-31T00:00:01+00:00"
        self.source_path.write_text(
            json.dumps(source, indent=2),
            encoding="utf-8",
        )
        preregistration = Path(
            result["binding"]["artifacts"]["preregistration"]
        )
        self.assertTrue(preregistration.is_file())
        with mock.patch.object(
            branch_runner.subprocess,
            "run",
        ) as child:
            with self.assertRaisesRegex(
                ValueError,
                "differs from the current verified source",
            ):
                self.quiet_run(self.args("--execute"))
        child.assert_not_called()

    def test_default_mode_is_no_write_dry_run(self) -> None:
        args = branch_runner.parse_args(
            [
                "--training-preregistration",
                str(self.source_path),
                "--seed",
                str(self.seed),
            ]
        )
        with mock.patch.object(
            branch_runner.subprocess,
            "run",
        ) as child:
            result = self.quiet_run(args)
        child.assert_not_called()
        self.assertFalse(self.output_root.exists())
        self.assertFalse(
            Path(
                result["binding"]["artifacts"]["preregistration"]
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
