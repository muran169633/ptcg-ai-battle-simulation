from __future__ import annotations

import hashlib
import json
import os
import fcntl
import signal
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import exec_preregistered_ppo_training as launcher


class ExecCalled(RuntimeError):
    pass


class PreregisteredPpoTrainingExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "tools").mkdir()
        (self.root / "artifacts").mkdir()

        self.launcher_fixture = (
            self.root / "tools" / "exec_preregistered_ppo_training.py"
        )
        self.launcher_fixture.write_text("# launcher fixture\n", encoding="utf-8")
        self.trainer = self.root / "tools" / "train_ppo.py"
        self.trainer.write_text("# trainer fixture\n", encoding="utf-8")
        self.source_protocol = self.root / "artifacts" / "source_protocol.json"
        self.source_protocol.write_text('{"source":true}\n', encoding="utf-8")
        self.comprehensive = self.root / "artifacts" / "comprehensive.json"
        self.comprehensive.write_text(
            '{"comprehensive":true}\n',
            encoding="utf-8",
        )
        self.runner = self.root / "artifacts" / "runner.json"
        self.runner.write_text('{"runner":true}\n', encoding="utf-8")
        self.branch = self.root / "artifacts" / "branch.json"
        self.lock = self.root / "artifacts" / "transport_lock.json"
        self.output_dir = self.root / "artifacts" / "candidate"
        self.log = self.root / "artifacts" / "candidate.log"
        self.seed = 202607401
        self.start_marker = launcher.canonical_attempt_start_marker(
            root=self.root,
            log=self.log,
            seed=self.seed,
        )
        self.unpublished_witness = self.log.with_name(
            self.log.name + ".unpublished-witness"
        )
        self.receipt = self.root / "artifacts" / "candidate.receipt.json"
        self.terminal_checkpoint = (
            self.output_dir / "checkpoints" / "update-0456.pt"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def effective_command(
        self,
        command: list[str],
    ) -> tuple[list[str], list[int]]:
        effective = list(command)
        indices: list[int] = []
        for index, token in enumerate(command):
            if index == 1:
                continue
            path = Path(token)
            if not path.is_absolute():
                continue
            try:
                relative = path.relative_to(self.root)
            except ValueError:
                continue
            indices.append(index)
            effective[index] = str(
                Path(launcher.EFFECTIVE_ROOT_PATH) / relative
            )
        effective[1] = launcher.EFFECTIVE_TRAINER_PATH
        return effective, indices

    def write_lock(self) -> tuple[str, str, list[str]]:
        command = [
            sys.executable,
            str(self.trainer),
            "--output-dir",
            str(self.output_dir),
            "--updates",
            "456",
            "--seed",
            str(self.seed),
            "--device",
            "cuda",
        ]
        command_sha256 = launcher.canonical_json_sha256(command)
        effective_command, repo_path_indices = self.effective_command(command)
        binding = {
            "command": command,
            "command_sha256": command_sha256,
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "source_train_script": str(self.trainer),
            "source_train_script_sha256": self.sha256(self.trainer),
        }
        branch = {
            "binding": binding,
            "binding_sha256": launcher.canonical_json_sha256(binding),
            "created_at": "2026-07-31T00:00:00+00:00",
            "schema_version": "ptcg-ppo-gold-branch-preregistration-v1",
            "status": "preregistered_not_started",
        }
        self.branch.write_text(
            json.dumps(branch, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        protocol = {
            "schema_version": launcher.SCHEMA_VERSION,
            "status": launcher.LOCKED_STATUS,
            "bindings": {
                "launcher": {
                    "path": str(self.launcher_fixture),
                    "sha256": self.sha256(self.launcher_fixture),
                },
                "source_protocol": {
                    "path": str(self.source_protocol),
                    "sha256": self.sha256(self.source_protocol),
                },
                "comprehensive_preregistration": {
                    "path": str(self.comprehensive),
                    "sha256": self.sha256(self.comprehensive),
                },
                "runner_preregistration": {
                    "path": str(self.runner),
                    "sha256": self.sha256(self.runner),
                },
                "branch_preregistration": {
                    "path": str(self.branch),
                    "sha256": self.sha256(self.branch),
                },
                "trainer": {
                    "path": str(self.trainer),
                    "sha256": self.sha256(self.trainer),
                },
            },
            "child_command": {
                "source_binding": "branch_preregistration",
                "json_path": "binding.command",
                "token_count": len(command),
                "canonical_sha256": command_sha256,
                "effective_trainer_fd": launcher.EFFECTIVE_TRAINER_FD,
                "effective_trainer_path": launcher.EFFECTIVE_TRAINER_PATH,
                "sealed_trainer_payload_fd": (
                    launcher.SEALED_TRAINER_PAYLOAD_FD
                ),
                "sealed_trainer_payload_path": (
                    launcher.SEALED_TRAINER_PAYLOAD_PATH
                ),
                "sealed_bootstrap_sha256": hashlib.sha256(
                    launcher.build_trainer_bootstrap(
                        canonical_trainer_path=self.trainer,
                        trainer_sha256=self.sha256(self.trainer),
                    )
                ).hexdigest(),
                "effective_root_fd": launcher.EFFECTIVE_ROOT_FD,
                "effective_root_path": launcher.EFFECTIVE_ROOT_PATH,
                "effective_repo_path_token_indices": repo_path_indices,
                "effective_command_canonical_sha256": (
                    launcher.canonical_json_sha256(effective_command)
                ),
            },
            "attempt": {
                "seed": self.seed,
                "attempts_authorized": 1,
                "output_dir": str(self.output_dir),
                "log": str(self.log),
                "terminal_receipt": str(self.receipt),
                "terminal_checkpoint": str(self.terminal_checkpoint),
                "attempt_start_marker": str(self.start_marker),
                "unpublished_log_witness": str(
                    self.unpublished_witness
                ),
                "expected_terminal_update": 456,
                "absent_at_lock": {
                    "output_dir": True,
                    "log": True,
                    "terminal_receipt": True,
                    "terminal_checkpoint": True,
                    "attempt_start_marker": True,
                    "unpublished_log_witness": True,
                },
            },
            "expected_transport": {
                "tool": "functions.exec_command",
                "sandbox_permissions": "require_escalated",
                "login": False,
                "tty": False,
                "workdir": str(self.root),
                "shell": "/bin/bash",
                "topology": "supervised_fork_exec_v1",
            },
        }
        self.lock.write_text(
            json.dumps(protocol, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return (
            self.sha256(self.lock),
            self.sha256(self.launcher_fixture),
            command,
        )

    def validate(
        self,
        lock_sha256: str,
        *,
        cwd: Path | None = None,
    ) -> launcher.ValidatedLaunch:
        with (
            mock.patch.object(Path, "cwd", return_value=cwd or self.root),
            mock.patch.object(launcher, "__file__", str(self.launcher_fixture)),
        ):
            return launcher.read_and_validate_transport_lock(
                transport_lock=self.lock,
                expected_transport_lock_sha256=lock_sha256,
            )

    def rewrite_command(self, command: list[str]) -> str:
        branch = json.loads(self.branch.read_text(encoding="utf-8"))
        branch["binding"]["command"] = command
        branch["binding"]["command_sha256"] = launcher.canonical_json_sha256(
            command
        )
        branch["binding_sha256"] = launcher.canonical_json_sha256(
            branch["binding"]
        )
        self.branch.write_text(
            json.dumps(branch, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        protocol = json.loads(self.lock.read_text(encoding="utf-8"))
        protocol["bindings"]["branch_preregistration"][
            "sha256"
        ] = self.sha256(self.branch)
        protocol["child_command"]["token_count"] = len(command)
        protocol["child_command"][
            "canonical_sha256"
        ] = launcher.canonical_json_sha256(command)
        effective, repo_path_indices = self.effective_command(command)
        protocol["child_command"][
            "effective_repo_path_token_indices"
        ] = repo_path_indices
        protocol["child_command"][
            "effective_command_canonical_sha256"
        ] = launcher.canonical_json_sha256(effective)
        self.lock.write_text(
            json.dumps(protocol, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.sha256(self.lock)

    def test_valid_lock_binds_exact_command_and_all_files(self) -> None:
        lock_sha256, launcher_sha256, command = self.write_lock()
        validated = self.validate(lock_sha256)
        self.assertEqual(list(validated.command), command)
        self.assertEqual(validated.seed, self.seed)
        self.assertEqual(set(validated.binding_paths), launcher.REQUIRED_BINDINGS)

    def test_exported_schema_and_effective_command_builder_match_lock(
        self,
    ) -> None:
        _, _, command = self.write_lock()
        protocol = json.loads(self.lock.read_text(encoding="utf-8"))
        self.assertEqual(
            launcher.SCHEMA_VERSION,
            "ptcg-preregistered-ppo-training-transport-lock-v4",
        )
        self.assertEqual(self.start_marker.parent, self.root)
        self.assertEqual(
            self.start_marker,
            launcher.canonical_attempt_start_marker(
                root=self.root,
                log=self.log,
                seed=self.seed,
            ),
        )
        effective, indices = launcher.build_effective_command(
            command,
            root=self.root,
        )
        independently_built, independent_indices = self.effective_command(
            command
        )
        self.assertEqual(effective, tuple(independently_built))
        self.assertEqual(indices, tuple(independent_indices))
        self.assertEqual(
            set(protocol["child_command"]),
            launcher.CHILD_COMMAND_KEYS,
        )
        self.assertEqual(set(protocol["attempt"]), launcher.ATTEMPT_KEYS)
        self.assertEqual(
            set(protocol["expected_transport"]),
            launcher.EXPECTED_TRANSPORT_KEYS,
        )

    def test_wrong_transport_lock_hash_creates_no_artifacts(self) -> None:
        _, launcher_sha256, _ = self.write_lock()
        with self.assertRaisesRegex(ValueError, "Transport-lock SHA-256"):
            self.validate("0" * 64)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.output_dir.exists())

    def test_wrong_cwd_creates_no_artifacts(self) -> None:
        lock_sha256, launcher_sha256, _ = self.write_lock()
        wrong_cwd = self.root / "artifacts"
        with self.assertRaisesRegex(ValueError, "working directory"):
            self.validate(lock_sha256, cwd=wrong_cwd)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.output_dir.exists())

    def test_legacy_log_sibling_start_marker_is_rejected(self) -> None:
        self.write_lock()
        protocol = json.loads(self.lock.read_text(encoding="utf-8"))
        protocol["attempt"]["attempt_start_marker"] = str(
            self.log.with_name(self.log.name + ".attempt-start.json")
        )
        self.lock.write_text(
            json.dumps(protocol, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ValueError,
            "attempt_start_marker is not canonical",
        ):
            self.validate(self.sha256(self.lock))
        self.assertFalse(self.start_marker.exists())
        self.assertFalse(self.log.exists())
        self.assertFalse(self.output_dir.exists())

    def test_binding_drift_creates_no_artifacts(self) -> None:
        lock_sha256, launcher_sha256, _ = self.write_lock()
        self.trainer.write_text("# trainer drift\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Binding SHA-256 mismatch"):
            self.validate(lock_sha256)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.output_dir.exists())

    def test_command_binding_drift_is_rejected(self) -> None:
        lock_sha256, launcher_sha256, _ = self.write_lock()
        branch = json.loads(self.branch.read_text(encoding="utf-8"))
        branch["binding"]["command"][-1] = "cpu"
        branch["binding"]["command_sha256"] = launcher.canonical_json_sha256(
            branch["binding"]["command"]
        )
        branch["binding_sha256"] = launcher.canonical_json_sha256(
            branch["binding"]
        )
        self.branch.write_text(json.dumps(branch) + "\n", encoding="utf-8")
        protocol = json.loads(self.lock.read_text(encoding="utf-8"))
        protocol["bindings"]["branch_preregistration"][
            "sha256"
        ] = self.sha256(self.branch)
        self.lock.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
        lock_sha256 = self.sha256(self.lock)
        with self.assertRaisesRegex(ValueError, "canonical SHA-256"):
            self.validate(lock_sha256)
        self.assertFalse(self.log.exists())

    def test_schema_objects_and_integer_types_are_strict(self) -> None:
        cases = (
            ("top_level", "top_level", "transport_lock must contain exactly"),
            ("attempt_extra", "attempt_extra", "attempt must contain exactly"),
            (
                "child_extra",
                "child_extra",
                "child_command must contain exactly",
            ),
            (
                "transport_extra",
                "transport_extra",
                "expected_transport must contain exactly",
            ),
            (
                "binding_extra",
                "binding_extra",
                "must contain exactly path and sha256",
            ),
            (
                "bool_attempts",
                "bool_attempts",
                "attempts_authorized must equal one",
            ),
        )
        for label, mutation, error_pattern in cases:
            with self.subTest(label=label):
                self.write_lock()
                protocol = json.loads(self.lock.read_text(encoding="utf-8"))
                if mutation == "top_level":
                    protocol["note"] = "not schema-authorized"
                elif mutation == "attempt_extra":
                    protocol["attempt"]["note"] = "not schema-authorized"
                elif mutation == "child_extra":
                    protocol["child_command"]["note"] = "not schema-authorized"
                elif mutation == "transport_extra":
                    protocol["expected_transport"][
                        "nested_functions_exec_forbidden"
                    ] = True
                elif mutation == "binding_extra":
                    protocol["bindings"]["trainer"]["note"] = "forbidden"
                elif mutation == "bool_attempts":
                    protocol["attempt"]["attempts_authorized"] = True
                self.lock.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error_pattern):
                    self.validate(self.sha256(self.lock))
                self.assertFalse(self.log.exists())
                self.assertFalse(self.output_dir.exists())

    def test_branch_top_level_and_command_metadata_are_strict(self) -> None:
        for mutation, error_pattern in (
            ("top_extra", "branch_preregistration must contain exactly"),
            ("missing_output_dir", "missing command-related keys"),
            ("bool_seed", "branch binding.seed must be an integer"),
        ):
            with self.subTest(mutation=mutation):
                self.write_lock()
                branch = json.loads(self.branch.read_text(encoding="utf-8"))
                if mutation == "top_extra":
                    branch["note"] = "forbidden"
                elif mutation == "missing_output_dir":
                    del branch["binding"]["output_dir"]
                    branch["binding_sha256"] = launcher.canonical_json_sha256(
                        branch["binding"]
                    )
                elif mutation == "bool_seed":
                    branch["binding"]["seed"] = True
                    branch["binding_sha256"] = launcher.canonical_json_sha256(
                        branch["binding"]
                    )
                self.branch.write_text(
                    json.dumps(branch) + "\n",
                    encoding="utf-8",
                )
                protocol = json.loads(self.lock.read_text(encoding="utf-8"))
                protocol["bindings"]["branch_preregistration"][
                    "sha256"
                ] = self.sha256(self.branch)
                self.lock.write_text(
                    json.dumps(protocol) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, error_pattern):
                    self.validate(self.sha256(self.lock))
                self.assertFalse(self.log.exists())

    def test_effective_command_hash_is_lock_bound(self) -> None:
        self.write_lock()
        protocol = json.loads(self.lock.read_text(encoding="utf-8"))
        protocol["child_command"][
            "effective_command_canonical_sha256"
        ] = "0" * 64
        self.lock.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Effective binding.command"):
            self.validate(self.sha256(self.lock))
        self.assertFalse(self.log.exists())

    def test_sealed_memfd_transport_rejects_non_linux_platform(self) -> None:
        with (
            mock.patch.object(launcher.sys, "platform", "unsupported"),
            self.assertRaisesRegex(RuntimeError, "Linux memfd/seal ABI"),
        ):
            launcher.build_trainer_bootstrap(
                canonical_trainer_path=self.trainer,
                trainer_sha256=self.sha256(self.trainer),
            )

    def test_effective_root_schema_values_are_strict(self) -> None:
        for field, value, error_pattern in (
            ("effective_root_fd", True, "effective_root_fd must equal"),
            (
                "effective_root_path",
                "/proc/self/fd/999",
                "effective_root_path must equal",
            ),
            (
                "sealed_trainer_payload_fd",
                True,
                "sealed_trainer_payload_fd must equal",
            ),
            (
                "sealed_trainer_payload_path",
                "/proc/self/fd/998",
                "sealed_trainer_payload_path must equal",
            ),
            (
                "sealed_bootstrap_sha256",
                "0" * 64,
                "bootstrap SHA-256 mismatch",
            ),
            (
                "effective_repo_path_token_indices",
                [3, 1],
                "sorted unique integer list",
            ),
            (
                "effective_repo_path_token_indices",
                [3, True],
                "sorted unique integer list",
            ),
        ):
            with self.subTest(field=field, value=value):
                self.write_lock()
                protocol = json.loads(self.lock.read_text(encoding="utf-8"))
                protocol["child_command"][field] = value
                self.lock.write_text(
                    json.dumps(protocol) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, error_pattern):
                    self.validate(self.sha256(self.lock))
                self.assertFalse(self.log.exists())
                self.assertFalse(self.output_dir.exists())

    def test_additional_exact_path_sha_binding_is_allowed(self) -> None:
        self.write_lock()
        extra = self.root / "artifacts" / "independent_audit.json"
        extra.write_text('{"pass":true}\n', encoding="utf-8")
        protocol = json.loads(self.lock.read_text(encoding="utf-8"))
        protocol["bindings"]["independent_audit"] = {
            "path": str(extra),
            "sha256": self.sha256(extra),
        }
        self.lock.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
        validated = self.validate(self.sha256(self.lock))
        self.assertEqual(validated.binding_paths["independent_audit"], extra)

    def test_critical_flag_equals_abbreviations_and_duplicates_are_rejected(
        self,
    ) -> None:
        _, _, base = self.write_lock()
        variants: list[tuple[str, list[str]]] = []
        for flag in ("--seed", "--output-dir", "--updates"):
            index = base.index(flag)
            equals_command = list(base)
            equals_command[index : index + 2] = [
                f"{flag}={base[index + 1]}"
            ]
            variants.append((f"{flag}=value", equals_command))
        abbreviated = list(base)
        abbreviated[abbreviated.index("--seed")] = "--see"
        variants.append(("argparse abbreviation", abbreviated))
        duplicate = [*base, "--seed", str(self.seed)]
        variants.append(("duplicate seed", duplicate))

        for label, command in variants:
            with self.subTest(label=label):
                self.write_lock()
                lock_sha256 = self.rewrite_command(command)
                with self.assertRaisesRegex(
                    ValueError,
                    "forbidden|exactly once",
                ):
                    self.validate(lock_sha256)
                self.assertFalse(self.log.exists())
                self.assertFalse(self.output_dir.exists())

    def test_dangling_symlink_is_not_absent(self) -> None:
        lock_sha256, _, _ = self.write_lock()
        self.log.symlink_to(self.root / "artifacts" / "missing-log-target")
        self.assertTrue(os.path.lexists(self.log))
        self.assertFalse(self.log.exists())
        with self.assertRaisesRegex(FileExistsError, "stdout/stderr log"):
            self.validate(lock_sha256)
        self.assertTrue(self.log.is_symlink())
        self.assertFalse(self.output_dir.exists())

    def test_held_trainer_survives_source_path_replacement(self) -> None:
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        original = self.trainer.read_bytes()
        with mock.patch.object(Path, "cwd", return_value=self.root):
            held = launcher.open_held_trainer(validated)
        try:
            replacement = self.root / "tools" / "replacement.py"
            replacement.write_bytes(b"# malicious replacement\n")
            replacement.replace(self.trainer)
            self.assertNotEqual(self.trainer.read_bytes(), original)
            self.assertEqual(
                os.pread(held.payload_fd, len(original), 0),
                original,
            )
            self.assertTrue(os.get_inheritable(held.fd))
            self.assertTrue(os.get_inheritable(held.payload_fd))
            self.assertEqual(
                held.source_sha256,
                hashlib.sha256(original).hexdigest(),
            )
            self.assertEqual(held.payload_sha256, held.source_sha256)
            self.assertEqual(
                fcntl.fcntl(held.fd, launcher.F_GET_SEALS),
                launcher._required_memfd_seals(),
            )
            self.assertEqual(
                fcntl.fcntl(held.payload_fd, launcher.F_GET_SEALS),
                launcher._required_memfd_seals(),
            )
            launcher._revalidate_sealed_trainer(held)
            with self.assertRaises(OSError):
                os.pwrite(held.payload_fd, b"!", 0)
        finally:
            launcher._close_held_trainer(held)

    def test_sealed_trainer_survives_source_inode_truncate_before_exec(
        self,
    ) -> None:
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        original = self.trainer.read_bytes()
        truncated = False
        real_fork = os.fork

        def truncate_then_fork() -> int:
            nonlocal truncated
            descriptor = os.open(
                self.trainer,
                os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW,
            )
            os.close(descriptor)
            truncated = True
            return real_fork()

        with (
            mock.patch.object(Path, "cwd", return_value=self.root),
            mock.patch.object(
                launcher.os,
                "fork",
                side_effect=truncate_then_fork,
            ) as fork,
        ):
            return_code = launcher.claim_log_and_exec(validated)
        self.assertEqual(return_code, 0)
        fork.assert_called_once_with()
        self.assertTrue(truncated)
        self.assertEqual(self.trainer.read_bytes(), b"")
        self.assertTrue(self.log.is_file())
        self.assertTrue(self.start_marker.is_file())
        self.assertFalse(self.unpublished_witness.exists())
        records = [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            records[0]["source_trainer_sha256"],
            hashlib.sha256(original).hexdigest(),
        )
        self.assertEqual(records[-1]["mapped_return_code"], 0)
        with self.assertRaises(OSError):
            os.fstat(launcher.EFFECTIVE_TRAINER_FD)
        with self.assertRaises(OSError):
            os.fstat(launcher.SEALED_TRAINER_PAYLOAD_FD)

    def test_canonical_log_preclaim_causes_unpublished_witness(
        self,
    ) -> None:
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        replacement = b"attacker replacement\n"
        real_waitpid = os.waitpid
        preclaimed = False

        def wait_then_preclaim(
            child_pid: int,
            options: int,
        ) -> tuple[int, int]:
            nonlocal preclaimed
            result = real_waitpid(child_pid, options)
            if not preclaimed:
                self.log.write_bytes(replacement)
                preclaimed = True
            return result

        with (
            mock.patch.object(Path, "cwd", return_value=self.root),
            mock.patch.object(
                launcher.os,
                "waitpid",
                side_effect=wait_then_preclaim,
            ),
            self.assertRaises(FileExistsError),
        ):
            launcher.claim_log_and_exec(validated)
        self.assertTrue(preclaimed)
        self.assertEqual(self.log.read_bytes(), replacement)
        self.assertTrue(self.start_marker.is_file())
        witness_raw = self.unpublished_witness.read_bytes()
        self.assertTrue(witness_raw.endswith(b"\n"))
        witness_lines = witness_raw.decode("utf-8").splitlines()
        witness_header = json.loads(witness_lines[0])
        self.assertEqual(
            witness_header["event"],
            "preregistered_ppo_training_pre_exec_checks_passed",
        )
        self.assertEqual(
            json.loads(witness_lines[-1])["event"],
            "supervisor_publish_failed",
        )
        self.assertFalse(self.output_dir.exists())

    def test_root_replacement_after_revalidation_fails_before_log_claim(
        self,
    ) -> None:
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        replacement_temp = tempfile.TemporaryDirectory(dir=self.root.parent)
        replacement = Path(replacement_temp.name)
        (replacement / "artifacts").mkdir()
        moved = self.root.parent / f"{self.root.name}-held-original"
        real_revalidate = launcher._revalidate_before_claim
        swapped = False

        def revalidate_then_swap(
            launch: launcher.ValidatedLaunch,
            held_root: launcher.HeldRoot,
        ) -> None:
            nonlocal swapped
            real_revalidate(launch, held_root)
            self.root.rename(moved)
            replacement.rename(self.root)
            swapped = True

        try:
            with (
                mock.patch.object(Path, "cwd", return_value=self.root),
                mock.patch.object(
                    launcher,
                    "_revalidate_before_claim",
                    side_effect=revalidate_then_swap,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "workdir|working directory",
                ),
            ):
                launcher.claim_log_and_exec(validated)
            self.assertTrue(swapped)
            self.assertFalse((self.root / "artifacts" / self.log.name).exists())
            self.assertFalse((moved / "artifacts" / self.log.name).exists())
            self.assertFalse(
                (self.root / "artifacts" / self.output_dir.name).exists()
            )
        finally:
            if self.root.exists() and moved.exists():
                self.root.rename(replacement)
                moved.rename(self.root)
            replacement_temp.cleanup()

    def test_root_replacement_after_start_marker_publishes_witness(self) -> None:
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        replacement_temp = tempfile.TemporaryDirectory(dir=self.root.parent)
        replacement = Path(replacement_temp.name)
        (replacement / "artifacts").mkdir()
        moved = self.root.parent / f"{self.root.name}-claimed-original"
        real_fork = os.fork
        swapped = False

        def swap_then_fork() -> int:
            nonlocal swapped
            if not swapped:
                self.root.rename(moved)
                replacement.rename(self.root)
                swapped = True
            return real_fork()

        try:
            with (
                mock.patch.object(Path, "cwd", return_value=self.root),
                mock.patch.object(
                    launcher.os,
                    "fork",
                    side_effect=swap_then_fork,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "workdir|working directory",
                ),
            ):
                launcher.claim_log_and_exec(validated)
            self.assertTrue(swapped)
            self.assertFalse((self.root / "artifacts" / self.log.name).exists())
            retained_marker = moved / self.start_marker.name
            retained_witness = (
                moved / "artifacts" / self.unpublished_witness.name
            )
            self.assertTrue(retained_marker.is_file())
            self.assertTrue(retained_witness.is_file())
            self.assertGreater(retained_witness.stat().st_size, 0)
            self.assertFalse(
                (self.root / "artifacts" / self.output_dir.name).exists()
            )
            self.assertFalse(
                (moved / "artifacts" / self.output_dir.name).exists()
            )
        finally:
            if self.root.exists() and moved.exists():
                self.root.rename(replacement)
                moved.rename(self.root)
            replacement_temp.cleanup()

    def test_nested_log_parent_replacement_consumes_attempt_and_blocks_retry(
        self,
    ) -> None:
        nested_log_parent = self.root / "artifacts" / "logs"
        nested_log_parent.mkdir()
        displaced_log_parent = self.root / "artifacts" / "logs-displaced"
        self.log = nested_log_parent / "candidate.log"
        self.start_marker = launcher.canonical_attempt_start_marker(
            root=self.root,
            log=self.log,
            seed=self.seed,
        )
        self.unpublished_witness = self.log.with_name(
            self.log.name + ".unpublished-witness"
        )
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        real_open_anonymous_log = launcher._open_anonymous_log
        swapped = False

        def swap_then_open_anonymous(log_parent_fd: int) -> int:
            nonlocal swapped
            nested_log_parent.rename(displaced_log_parent)
            nested_log_parent.mkdir()
            swapped = True
            return real_open_anonymous_log(log_parent_fd)

        with (
            mock.patch.object(Path, "cwd", return_value=self.root),
            mock.patch.object(
                launcher,
                "_open_anonymous_log",
                side_effect=swap_then_open_anonymous,
            ),
            self.assertRaisesRegex(RuntimeError, "canonical directory"),
        ):
            launcher.claim_log_and_exec(validated)

        self.assertTrue(swapped)
        self.assertTrue(self.start_marker.is_file())
        self.assertFalse(self.log.exists())
        self.assertFalse(self.unpublished_witness.exists())
        self.assertFalse(
            (displaced_log_parent / self.log.name).exists()
        )
        self.assertFalse(
            (
                displaced_log_parent
                / self.unpublished_witness.name
            ).exists()
        )

        # The durable root marker must win over later mutable binding drift,
        # so the consumed one-shot attempt cannot be validated for retry.
        self.trainer.write_text("# binding drift after claim\n", encoding="utf-8")
        with self.assertRaisesRegex(
            FileExistsError,
            "attempt start marker",
        ):
            self.validate(lock_sha256)

    def test_every_repo_absolute_command_token_is_effectively_held(self) -> None:
        lock_sha256, _, command = self.write_lock()
        validated = self.validate(lock_sha256)
        effective, indices = self.effective_command(command)
        self.assertEqual(list(validated.effective_command), effective)
        self.assertEqual(
            list(validated.effective_repo_path_token_indices),
            indices,
        )
        for index in indices:
            self.assertTrue(
                effective[index].startswith(
                    launcher.EFFECTIVE_ROOT_PATH + "/"
                )
            )
        self.assertEqual(effective[1], launcher.EFFECTIVE_TRAINER_PATH)
        output_index = effective.index("--output-dir") + 1
        self.assertTrue(
            effective[output_index].startswith(
                launcher.EFFECTIVE_ROOT_PATH + "/"
            )
        )

    def test_existing_output_log_receipt_or_checkpoint_is_rejected(self) -> None:
        creators = (
            lambda: self.output_dir.mkdir(),
            lambda: self.log.write_text("claimed\n", encoding="utf-8"),
            lambda: self.receipt.write_text("{}\n", encoding="utf-8"),
            lambda: self.start_marker.write_text("{}\n", encoding="utf-8"),
            lambda: self.unpublished_witness.write_text(
                "claimed\n",
                encoding="utf-8",
            ),
            lambda: (
                self.terminal_checkpoint.parent.mkdir(parents=True),
                self.terminal_checkpoint.write_bytes(b"checkpoint"),
            ),
        )
        for creator in creators:
            with self.subTest(creator=creator):
                lock_sha256, launcher_sha256, _ = self.write_lock()
                creator()
                with self.assertRaises(FileExistsError):
                    self.validate(lock_sha256)
                self.assertFalse(self.log.exists() and self.log.stat().st_size == 0)
                if self.output_dir.exists():
                    if self.terminal_checkpoint.exists():
                        self.terminal_checkpoint.unlink()
                    checkpoints = self.output_dir / "checkpoints"
                    if checkpoints.exists():
                        checkpoints.rmdir()
                    self.output_dir.rmdir()
                if self.log.exists():
                    self.log.unlink()
                if self.receipt.exists():
                    self.receipt.unlink()
                if self.start_marker.exists():
                    self.start_marker.unlink()
                if self.unpublished_witness.exists():
                    self.unpublished_witness.unlink()

    def test_supervisor_header_sealed_exec_and_terminal_publish(self) -> None:
        lock_sha256, launcher_sha256, command = self.write_lock()
        validated = self.validate(lock_sha256)
        effective_command, repo_path_indices = self.effective_command(command)
        open_calls: list[tuple[object, int, int, int | None]] = []
        real_open = os.open

        def inspect_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            open_calls.append((path, flags, mode, dir_fd))
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with (
            mock.patch.object(Path, "cwd", return_value=self.root),
            mock.patch.object(os, "open", side_effect=inspect_open),
            mock.patch.object(
                launcher,
                "_sha256_open_fd",
                wraps=launcher._sha256_open_fd,
            ) as hashed_fds,
        ):
            return_code = launcher.claim_log_and_exec(validated)
        self.assertEqual(return_code, 0)
        self.assertTrue(self.log.is_file())
        self.assertTrue(self.start_marker.is_file())
        self.assertFalse(self.unpublished_witness.exists())
        self.assertFalse(self.output_dir.exists())
        self.assertFalse(self.receipt.exists())

        tmpfile_calls = [
            call
            for call in open_calls
            if call[0] == "." and call[1] & os.O_TMPFILE
        ]
        self.assertEqual(len(tmpfile_calls), 1)
        self.assertEqual(tmpfile_calls[0][1] & os.O_ACCMODE, os.O_RDWR)
        marker_claims = [
            call
            for call in open_calls
            if call[0] == self.start_marker.name
            and call[1] & os.O_CREAT
        ]
        self.assertEqual(len(marker_claims), 1)
        self.assertTrue(marker_claims[0][1] & os.O_EXCL)
        self.assertTrue(marker_claims[0][1] & os.O_NOFOLLOW)

        hashed_fixed_fds = [
            call.args[0]
            for call in hashed_fds.call_args_list
            if call.args[0]
            in {
                launcher.EFFECTIVE_TRAINER_FD,
                launcher.SEALED_TRAINER_PAYLOAD_FD,
            }
        ]
        self.assertGreaterEqual(
            hashed_fixed_fds.count(launcher.EFFECTIVE_TRAINER_FD),
            2,
        )
        self.assertGreaterEqual(
            hashed_fixed_fds.count(launcher.SEALED_TRAINER_PAYLOAD_FD),
            2,
        )
        with self.assertRaises(OSError):
            os.fstat(launcher.EFFECTIVE_TRAINER_FD)
        with self.assertRaises(OSError):
            os.fstat(launcher.SEALED_TRAINER_PAYLOAD_FD)

        lines = self.log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        header = json.loads(lines[0])
        terminal = json.loads(lines[1])
        self.assertEqual(
            launcher.canonical_json_bytes(header),
            lines[0].encode("utf-8"),
        )
        self.assertEqual(
            launcher.canonical_json_bytes(terminal),
            lines[1].encode("utf-8"),
        )
        self.assertEqual(
            header["event"],
            "preregistered_ppo_training_pre_exec_checks_passed",
        )
        self.assertEqual(header["topology"], "supervised_fork_exec_v1")
        self.assertEqual(header["transport_lock_sha256"], lock_sha256)
        self.assertEqual(header["launcher_sha256"], launcher_sha256)
        self.assertEqual(
            header["source_protocol_sha256"],
            self.sha256(self.source_protocol),
        )
        self.assertEqual(
            header["branch_preregistration_sha256"],
            self.sha256(self.branch),
        )
        self.assertEqual(
            header["child_command_sha256"],
            launcher.canonical_json_sha256(command),
        )
        self.assertEqual(
            header["effective_command_sha256"],
            launcher.canonical_json_sha256(effective_command),
        )
        self.assertEqual(
            header["effective_repo_path_token_indices"],
            repo_path_indices,
        )
        self.assertEqual(
            header["sealed_trainer_payload_fd"],
            launcher.SEALED_TRAINER_PAYLOAD_FD,
        )
        self.assertEqual(
            header["sealed_trainer_payload_sha256"],
            self.sha256(self.trainer),
        )
        self.assertEqual(
            header["sealed_trainer_payload_seals"],
            launcher._required_memfd_seals(),
        )
        self.assertEqual(
            header["sealed_bootstrap_sha256"],
            validated.sealed_bootstrap_sha256,
        )
        self.assertEqual(
            header["sealed_bootstrap_seals"],
            launcher._required_memfd_seals(),
        )
        self.assertEqual(
            header["attempt_start_marker"],
            str(self.start_marker),
        )
        self.assertEqual(
            header["unpublished_log_witness"],
            str(self.unpublished_witness),
        )
        self.assertEqual(
            header["expected_transport"]["topology"],
            "supervised_fork_exec_v1",
        )
        self.assertEqual(terminal["event"], "supervisor_child_terminal")
        self.assertEqual(terminal["child_exit_code"], 0)
        self.assertIsNone(terminal["child_signal"])
        self.assertEqual(terminal["mapped_return_code"], 0)

        marker = json.loads(self.start_marker.read_text(encoding="utf-8"))
        self.assertEqual(
            marker["event"],
            "preregistered_ppo_training_attempt_consumed",
        )
        self.assertEqual(marker["topology"], "supervised_fork_exec_v1")
        self.assertEqual(
            marker["anonymous_log_header_sha256"],
            hashlib.sha256(
                (lines[0] + "\n").encode("utf-8")
            ).hexdigest(),
        )

    def test_supervisor_child_nonzero_publishes_canonical_log(self) -> None:
        self.trainer.write_text(
            "raise SystemExit(7)\n",
            encoding="utf-8",
        )
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        with mock.patch.object(Path, "cwd", return_value=self.root):
            return_code = launcher.claim_log_and_exec(validated)
        self.assertEqual(return_code, 7)
        self.assertTrue(self.log.is_file())
        self.assertTrue(self.start_marker.is_file())
        self.assertFalse(self.unpublished_witness.exists())
        terminal = json.loads(
            self.log.read_text(encoding="utf-8").splitlines()[-1]
        )
        self.assertEqual(terminal["child_exit_code"], 7)
        self.assertIsNone(terminal["child_signal"])
        self.assertEqual(terminal["mapped_return_code"], 7)

    def test_supervisor_child_signal_publishes_canonical_log(self) -> None:
        self.trainer.write_text(
            "import os, signal\n"
            "os.kill(os.getpid(), signal.SIGTERM)\n",
            encoding="utf-8",
        )
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        with mock.patch.object(Path, "cwd", return_value=self.root):
            return_code = launcher.claim_log_and_exec(validated)
        self.assertEqual(return_code, 128 + signal.SIGTERM)
        self.assertTrue(self.log.is_file())
        self.assertTrue(self.start_marker.is_file())
        self.assertFalse(self.unpublished_witness.exists())
        terminal = json.loads(
            self.log.read_text(encoding="utf-8").splitlines()[-1]
        )
        self.assertIsNone(terminal["child_exit_code"])
        self.assertEqual(terminal["child_signal"], signal.SIGTERM)
        self.assertEqual(
            terminal["mapped_return_code"],
            128 + signal.SIGTERM,
        )

    def test_supervisor_fork_failure_consumes_attempt_and_keeps_witness(
        self,
    ) -> None:
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        with (
            mock.patch.object(Path, "cwd", return_value=self.root),
            mock.patch.object(
                launcher.os,
                "fork",
                side_effect=OSError("synthetic fork failure"),
            ),
            self.assertRaisesRegex(OSError, "synthetic fork failure"),
        ):
            launcher.claim_log_and_exec(validated)
        self.assertTrue(self.start_marker.is_file())
        self.assertFalse(self.log.exists())
        self.assertTrue(self.unpublished_witness.is_file())
        events = [
            json.loads(line)["event"]
            for line in self.unpublished_witness.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertIn("supervisor_fork_failed", events)
        self.assertIn("supervisor_publish_failed", events)
        with self.assertRaises(FileExistsError):
            self.validate(lock_sha256)

    def test_supervisor_parent_interrupt_reaps_child_and_keeps_witness(
        self,
    ) -> None:
        self.trainer.write_text(
            "import time\n"
            "time.sleep(30)\n",
            encoding="utf-8",
        )
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        real_waitpid = os.waitpid
        interrupted = False

        def interrupt_first_wait(
            child_pid: int,
            options: int,
        ) -> tuple[int, int]:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise launcher.SupervisorInterrupted(signal.SIGTERM)
            return real_waitpid(child_pid, options)

        with (
            mock.patch.object(Path, "cwd", return_value=self.root),
            mock.patch.object(
                launcher.os,
                "waitpid",
                side_effect=interrupt_first_wait,
            ),
            self.assertRaises(launcher.SupervisorInterrupted),
        ):
            launcher.claim_log_and_exec(validated)
        self.assertTrue(interrupted)
        self.assertTrue(self.start_marker.is_file())
        self.assertFalse(self.log.exists())
        self.assertTrue(self.unpublished_witness.is_file())
        events = [
            json.loads(line)["event"]
            for line in self.unpublished_witness.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertIn("supervisor_launcher_failed", events)

    def test_child_never_observes_canonical_log_before_terminal_publish(
        self,
    ) -> None:
        self.trainer.write_text(
            "from pathlib import Path\n"
            f"assert not Path({str(self.log)!r}).exists()\n"
            "print('CHILD_CANONICAL_LOG_ABSENT', flush=True)\n",
            encoding="utf-8",
        )
        lock_sha256, _, _ = self.write_lock()
        validated = self.validate(lock_sha256)
        with mock.patch.object(Path, "cwd", return_value=self.root):
            return_code = launcher.claim_log_and_exec(validated)
        self.assertEqual(return_code, 0)
        self.assertTrue(self.log.is_file())
        self.assertIn(
            "CHILD_CANONICAL_LOG_ABSENT",
            self.log.read_text(encoding="utf-8"),
        )

    def _legacy_header_and_os_execv_use_exact_locked_command(self) -> None:
        lock_sha256, launcher_sha256, command = self.write_lock()
        validated = self.validate(lock_sha256)
        effective_command, repo_path_indices = self.effective_command(command)
        observed: dict[str, object] = {}
        log_open_calls: list[tuple[int, int, int | None]] = []
        real_dup2 = os.dup2
        real_open = os.open

        def selective_dup2(
            source_fd: int,
            target_fd: int,
            inheritable: bool = True,
        ) -> int:
            if target_fd in {1, 2}:
                return target_fd
            return real_dup2(
                source_fd,
                target_fd,
                inheritable=inheritable,
            )

        def inspect_exec(path: str, arguments: list[str]) -> None:
            bootstrap_info = os.fstat(launcher.EFFECTIVE_TRAINER_FD)
            payload_info = os.fstat(launcher.SEALED_TRAINER_PAYLOAD_FD)
            observed["path"] = path
            observed["arguments"] = list(arguments)
            observed["inheritable"] = os.get_inheritable(
                launcher.EFFECTIVE_TRAINER_FD
            )
            observed["payload_inheritable"] = os.get_inheritable(
                launcher.SEALED_TRAINER_PAYLOAD_FD
            )
            observed["bootstrap"] = os.pread(
                launcher.EFFECTIVE_TRAINER_FD,
                bootstrap_info.st_size,
                0,
            )
            observed["payload"] = os.pread(
                launcher.SEALED_TRAINER_PAYLOAD_FD,
                payload_info.st_size,
                0,
            )
            observed["bootstrap_seals"] = fcntl.fcntl(
                launcher.EFFECTIVE_TRAINER_FD,
                launcher.F_GET_SEALS,
            )
            observed["payload_seals"] = fcntl.fcntl(
                launcher.SEALED_TRAINER_PAYLOAD_FD,
                launcher.F_GET_SEALS,
            )
            observed["root_inheritable"] = os.get_inheritable(
                launcher.EFFECTIVE_ROOT_FD
            )
            observed["root_samefile"] = os.path.samefile(
                launcher.EFFECTIVE_ROOT_PATH,
                self.root,
            )
            raise ExecCalled("exec")

        def inspect_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if path == self.log.name:
                log_open_calls.append((flags, mode, dir_fd))
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with (
            mock.patch.object(Path, "cwd", return_value=self.root),
            mock.patch.object(os, "dup2", side_effect=selective_dup2),
            mock.patch.object(os, "open", side_effect=inspect_open),
            mock.patch.object(
                launcher,
                "_sha256_open_fd",
                wraps=launcher._sha256_open_fd,
            ) as hashed_fds,
            mock.patch.object(
                os,
                "execv",
                side_effect=inspect_exec,
            ) as patched_execv,
        ):
            with self.assertRaises(ExecCalled):
                launcher.claim_log_and_exec(validated)
        patched_execv.assert_called_once_with(command[0], effective_command)
        self.assertEqual(observed["arguments"], effective_command)
        self.assertIs(observed["inheritable"], True)
        self.assertIs(observed["payload_inheritable"], True)
        self.assertIs(observed["root_inheritable"], True)
        self.assertIs(observed["root_samefile"], True)
        self.assertEqual(observed["payload"], self.trainer.read_bytes())
        self.assertEqual(
            hashlib.sha256(observed["bootstrap"]).hexdigest(),
            validated.sealed_bootstrap_sha256,
        )
        self.assertEqual(
            observed["bootstrap_seals"],
            launcher._required_memfd_seals(),
        )
        self.assertEqual(
            observed["payload_seals"],
            launcher._required_memfd_seals(),
        )
        hashed_fixed_fds = [
            call.args[0]
            for call in hashed_fds.call_args_list
            if call.args[0]
            in {
                launcher.EFFECTIVE_TRAINER_FD,
                launcher.SEALED_TRAINER_PAYLOAD_FD,
            }
        ]
        self.assertGreaterEqual(
            hashed_fixed_fds.count(launcher.EFFECTIVE_TRAINER_FD),
            2,
        )
        self.assertGreaterEqual(
            hashed_fixed_fds.count(launcher.SEALED_TRAINER_PAYLOAD_FD),
            2,
        )
        expected_log_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            expected_log_flags |= os.O_NOFOLLOW
        claim_calls = [
            call
            for call in log_open_calls
            if call[0] & os.O_CREAT
        ]
        reopen_calls = [
            call
            for call in log_open_calls
            if not call[0] & os.O_CREAT
        ]
        self.assertEqual(len(claim_calls), 1)
        self.assertEqual(claim_calls[0][0], expected_log_flags)
        self.assertEqual(claim_calls[0][1], 0o644)
        self.assertIs(type(claim_calls[0][2]), int)
        self.assertEqual(len(reopen_calls), 1)
        self.assertEqual(reopen_calls[0][0] & os.O_ACCMODE, os.O_RDONLY)
        self.assertIs(type(reopen_calls[0][2]), int)
        with self.assertRaises(OSError):
            os.fstat(launcher.EFFECTIVE_TRAINER_FD)
        with self.assertRaises(OSError):
            os.fstat(launcher.SEALED_TRAINER_PAYLOAD_FD)

        raw = self.log.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        header = json.loads(raw)
        self.assertEqual(
            raw,
            launcher.canonical_json_bytes(header) + b"\n",
        )
        self.assertEqual(header["transport_lock_sha256"], lock_sha256)
        self.assertEqual(header["launcher_sha256"], launcher_sha256)
        self.assertEqual(
            header["source_protocol_sha256"],
            self.sha256(self.source_protocol),
        )
        self.assertEqual(
            header["branch_preregistration_sha256"],
            self.sha256(self.branch),
        )
        self.assertEqual(
            header["child_command_sha256"],
            launcher.canonical_json_sha256(command),
        )
        self.assertEqual(
            header["source_command_sha256"],
            launcher.canonical_json_sha256(command),
        )
        self.assertEqual(
            header["effective_command_sha256"],
            launcher.canonical_json_sha256(effective_command),
        )
        self.assertEqual(
            header["effective_trainer_fd"],
            launcher.EFFECTIVE_TRAINER_FD,
        )
        self.assertEqual(
            header["effective_trainer_path"],
            launcher.EFFECTIVE_TRAINER_PATH,
        )
        self.assertEqual(
            header["effective_root_fd"],
            launcher.EFFECTIVE_ROOT_FD,
        )
        self.assertEqual(
            header["effective_root_path"],
            launcher.EFFECTIVE_ROOT_PATH,
        )
        self.assertEqual(
            header["effective_repo_path_token_indices"],
            repo_path_indices,
        )
        self.assertEqual(
            header["held_trainer_sha256"],
            self.sha256(self.trainer),
        )
        self.assertEqual(
            header["held_trainer_fstat"]["st_size"],
            self.trainer.stat().st_size,
        )
        self.assertEqual(
            header["sealed_trainer_payload_fd"],
            launcher.SEALED_TRAINER_PAYLOAD_FD,
        )
        self.assertEqual(
            header["sealed_trainer_payload_path"],
            launcher.SEALED_TRAINER_PAYLOAD_PATH,
        )
        self.assertEqual(
            header["sealed_trainer_payload_sha256"],
            self.sha256(self.trainer),
        )
        self.assertEqual(
            header["sealed_trainer_payload_seals"],
            launcher._required_memfd_seals(),
        )
        self.assertEqual(
            header["sealed_bootstrap_sha256"],
            validated.sealed_bootstrap_sha256,
        )
        self.assertEqual(
            header["sealed_bootstrap_seals"],
            launcher._required_memfd_seals(),
        )
        self.assertEqual(
            header["sealed_bootstrap_fstat"]["st_size"],
            len(observed["bootstrap"]),
        )
        self.assertEqual(
            header["source_trainer_sha256"],
            self.sha256(self.trainer),
        )
        self.assertEqual(
            header["source_trainer_fstat"]["st_size"],
            self.trainer.stat().st_size,
        )
        self.assertEqual(header["trainer_sha256"], self.sha256(self.trainer))
        self.assertEqual(header["seed"], self.seed)
        self.assertEqual(header["cwd"], str(self.root))
        self.assertEqual(
            header["expected_transport"],
            {
                "tool": "functions.exec_command",
                "sandbox_permissions": "require_escalated",
                "login": False,
                "tty": False,
                "workdir": str(self.root),
                "shell": "/bin/bash",
            },
        )
        self.assertFalse(self.output_dir.exists())
        self.assertFalse(self.receipt.exists())

    def test_launcher_sha_is_lock_bound(self) -> None:
        lock_sha256, launcher_sha256, _ = self.write_lock()
        protocol = json.loads(self.lock.read_text(encoding="utf-8"))
        self.assertEqual(
            set(protocol["bindings"]["launcher"]),
            {"path", "sha256"},
        )
        protocol["bindings"]["launcher"]["sha256"] = "0" * 64
        self.lock.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
        lock_sha256 = self.sha256(self.lock)
        with self.assertRaisesRegex(
            ValueError,
            "Binding SHA-256 mismatch: launcher",
        ):
            self.validate(lock_sha256)
        self.assertFalse(self.log.exists())
        self.assertNotEqual(launcher_sha256, "0" * 64)


if __name__ == "__main__":
    unittest.main()
