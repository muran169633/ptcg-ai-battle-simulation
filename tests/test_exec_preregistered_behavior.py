from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import exec_preregistered_behavior as launcher


class DirectBehaviorExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "tools").mkdir()
        self.evaluator = self.root / "tools" / "evaluate_policy_bc.py"
        self.evaluator.write_text("# evaluator fixture\n", encoding="utf-8")
        self.launcher_fixture = (
            self.root / "tools" / "exec_preregistered_behavior.py"
        )
        self.launcher_fixture.write_text(
            "# launcher fixture\n",
            encoding="utf-8",
        )
        self.checkpoint = self.root / "candidate.pt"
        self.checkpoint.write_bytes(b"checkpoint")
        self.data = self.root / "data.zip"
        self.data.write_bytes(b"data")
        self.output = self.root / "behavior.json"
        self.log = self.root / "behavior.log"
        self.receipt = self.root / "behavior.execution_result.json"
        self.preregistration = self.root / "behavior_preregistration.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_protocol(self, **panel_overrides: object) -> str:
        command = [
            sys.executable,
            "-u",
            str(self.evaluator),
            "--checkpoint",
            str(self.checkpoint),
            "--data",
            str(self.data),
            "--device",
            "cuda",
            "--prediction-order",
            "policy",
            "--json-output",
            str(self.output),
        ]
        panel = {
            "order": 1,
            "name": "old_retention",
            "attempts_authorized": 1,
            "expected_success_exit_code": 0,
            "command_abs": command,
            "command_sha256": launcher.canonical_json_sha256(command),
            "immutable_inputs": [
                {
                    "path": str(self.evaluator),
                    "sha256": self.sha256(self.evaluator),
                },
                {
                    "path": str(self.checkpoint),
                    "sha256": self.sha256(self.checkpoint),
                },
                {
                    "path": str(self.data),
                    "sha256": self.sha256(self.data),
                },
            ],
            "output": str(self.output),
            "stdout_stderr_log": str(self.log),
            "execution_result": str(self.receipt),
        }
        panel.update(panel_overrides)
        protocol = {
            "required_cwd": str(self.root),
            "execution_transport": dict(launcher.REQUIRED_TRANSPORT),
            "launcher": {
                "path": str(self.launcher_fixture),
                "sha256": self.sha256(self.launcher_fixture),
            },
            "ordered_evaluations": [panel],
        }
        self.preregistration.write_text(
            json.dumps(protocol, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.sha256(self.preregistration)

    def validate(self, digest: str) -> tuple[dict, dict, list[str]]:
        with (
            mock.patch.object(Path, "cwd", return_value=self.root),
            mock.patch.object(
                launcher,
                "__file__",
                str(self.launcher_fixture),
            ),
        ):
            return launcher.read_and_validate_panel(
                root=self.root,
                preregistration=self.preregistration,
                expected_preregistration_sha256=digest,
                panel_index=0,
            )

    def test_valid_protocol_binds_exact_command_and_inputs(self) -> None:
        digest = self.write_protocol()
        _, panel, command = self.validate(digest)
        self.assertEqual(panel["name"], "old_retention")
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[command.index("--device") + 1], "cuda")

    def test_protocol_hash_mismatch_is_rejected(self) -> None:
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, "preregistration SHA-256"):
            self.validate("0" * 64)

    def test_command_hash_mismatch_is_rejected(self) -> None:
        digest = self.write_protocol(command_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "command_abs SHA-256"):
            self.validate(digest)

    def test_transport_must_require_escalated_direct_exec(self) -> None:
        digest = self.write_protocol()
        protocol = json.loads(self.preregistration.read_text())
        protocol["execution_transport"]["sandbox_permissions"] = "use_default"
        self.preregistration.write_text(json.dumps(protocol) + "\n")
        digest = self.sha256(self.preregistration)
        with self.assertRaisesRegex(ValueError, "sandbox_permissions"):
            self.validate(digest)

    def test_existing_log_consumes_the_one_shot_claim(self) -> None:
        digest = self.write_protocol()
        self.log.write_text("already claimed\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "stdout/stderr log"):
            self.validate(digest)

    def test_output_log_and_receipt_must_be_distinct(self) -> None:
        digest = self.write_protocol(stdout_stderr_log=str(self.output))
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            self.validate(digest)

    def test_non_policy_decode_is_rejected(self) -> None:
        digest = self.write_protocol()
        protocol = json.loads(self.preregistration.read_text())
        command = protocol["ordered_evaluations"][0]["command_abs"]
        command[command.index("--prediction-order") + 1] = "auto"
        protocol["ordered_evaluations"][0][
            "command_sha256"
        ] = launcher.canonical_json_sha256(command)
        self.preregistration.write_text(json.dumps(protocol) + "\n")
        digest = self.sha256(self.preregistration)
        with self.assertRaisesRegex(ValueError, "policy prediction order"):
            self.validate(digest)

    def test_immutable_input_hash_mismatch_is_rejected(self) -> None:
        digest = self.write_protocol()
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Immutable input SHA-256"):
            self.validate(digest)

    def test_cuda_preflight_requires_visible_cuda(self) -> None:
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = False
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "no visible CUDA"):
                launcher.cuda_tensor_preflight()

    def test_claim_uses_exclusive_log_creation(self) -> None:
        self.log.write_text("claimed\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            launcher.claim_log(
                root=self.root,
                preregistration_sha256="0" * 64,
                panel={
                    "stdout_stderr_log": str(self.log),
                    "command_sha256": "1" * 64,
                    "name": "old_retention",
                },
            )

    def test_attempt_is_claimed_before_cuda_preflight(self) -> None:
        digest = self.write_protocol()
        _, panel, _ = self.validate(digest)
        fd = launcher.claim_log(
            root=self.root,
            preregistration_sha256=digest,
            panel=panel,
        )
        os.close(fd)
        event = json.loads(self.log.read_text().splitlines()[0])
        self.assertEqual(event["event"], "behavior_directexec_attempt_claimed")

    def test_second_panel_requires_first_terminal_receipt(self) -> None:
        digest = self.write_protocol()
        protocol = json.loads(self.preregistration.read_text())
        first = protocol["ordered_evaluations"][0]
        second = dict(first)
        second.update(
            {
                "order": 2,
                "name": "valid29",
                "output": str(self.root / "behavior_valid29.json"),
                "stdout_stderr_log": str(self.root / "behavior_valid29.log"),
                "execution_result": str(
                    self.root / "behavior_valid29.execution_result.json"
                ),
            }
        )
        command = list(second["command_abs"])
        command[command.index("--json-output") + 1] = second["output"]
        second["command_abs"] = command
        second["command_sha256"] = launcher.canonical_json_sha256(command)
        protocol["ordered_evaluations"].append(second)
        self.preregistration.write_text(json.dumps(protocol) + "\n")
        digest = self.sha256(self.preregistration)
        with (
            mock.patch.object(Path, "cwd", return_value=self.root),
            mock.patch.object(
                launcher,
                "__file__",
                str(self.launcher_fixture),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "consumed-attempt log"):
                launcher.read_and_validate_panel(
                    root=self.root,
                    preregistration=self.preregistration,
                    expected_preregistration_sha256=digest,
                    panel_index=1,
                )


if __name__ == "__main__":
    unittest.main()
