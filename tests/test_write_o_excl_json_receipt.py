from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import write_o_excl_json_receipt as writer


class WriteOExclJsonReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temporary.name).resolve()
        self.root = self.sandbox / "repo"
        self.root.mkdir()
        self.source = self.root / "source.json"
        self.target = self.root / "target.json"
        self.transport_lock = self.root / "transport-lock.json"
        self.transport_lock.write_bytes(b"locked transport protocol\n")
        self.log = self.root / "training.log"
        self.log.write_bytes(b"pre-exec header\nterminal output\n")
        self.checkpoint = self.root / "update-0456.pt"
        self.checkpoint.write_bytes(b"terminal checkpoint\n")
        self.root_patch = mock.patch.object(writer, "REPO_ROOT", self.root)
        self.root_patch.start()

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.temporary.cleanup()

    def payload_bytes(self, **overrides: object) -> bytes:
        payload: dict[str, object] = {
            "schema_version": writer.RECEIPT_SCHEMA_VERSION,
            "status": "completed",
            "terminal": True,
            "terminal_exit_code": 0,
            "attempt_consumed": True,
            "transport": {
                "tool": "functions.exec_command",
                "sandbox_permissions": "require_escalated",
                "login": False,
                "tty": False,
                "workdir": str(self.root),
                "shell": "/bin/bash",
            },
            "bindings": {
                "transport_lock": {
                    "path": str(self.transport_lock),
                    "sha256": self.file_sha256(self.transport_lock),
                },
                "effective_command_sha256": "1" * 64,
            },
            "artifacts": {
                "stdout_stderr_log": {
                    "path": str(self.log),
                    "sha256": self.file_sha256(self.log),
                },
                "terminal_checkpoint": {
                    "path": str(self.checkpoint),
                    "sha256": self.file_sha256(self.checkpoint),
                    "update": 456,
                },
            },
            "write_once_target": str(self.target),
            "original_receipt_must_not_be_overwritten": True,
        }
        payload.update(overrides)
        return (
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            + b"\n"
        )

    def write_source(self, raw: bytes | None = None) -> bytes:
        if raw is None:
            raw = self.payload_bytes()
        self.source.write_bytes(raw)
        return raw

    @staticmethod
    def sha256(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def call_writer(self, raw: bytes) -> str:
        return writer.write_receipt(
            source_value=str(self.source),
            expected_source_sha256=self.sha256(raw),
            target_value=str(self.target),
        )

    def test_duplicate_json_key_is_rejected_before_target_creation(self) -> None:
        raw = (
            b'{"write_once_target":'
            + json.dumps(str(self.target)).encode("utf-8")
            + b',"original_receipt_must_not_be_overwritten":true,'
            b'"terminal_exit_code":0,"terminal_exit_code":1}\n'
        )
        self.write_source(raw)

        with self.assertRaisesRegex(ValueError, "duplicate key"):
            self.call_writer(raw)
        self.assertFalse(os.path.lexists(self.target))

    def test_source_hash_mismatch_is_rejected(self) -> None:
        raw = self.write_source()

        with self.assertRaisesRegex(ValueError, "source SHA-256"):
            writer.write_receipt(
                source_value=str(self.source),
                expected_source_sha256="0" * 64,
                target_value=str(self.target),
            )
        self.assertFalse(os.path.lexists(self.target))

    def test_noncanonical_and_outside_paths_are_rejected(self) -> None:
        raw = self.write_source()
        noncanonical = str(self.root) + "/./source.json"
        with self.assertRaisesRegex(ValueError, "normalized"):
            writer.write_receipt(
                source_value=noncanonical,
                expected_source_sha256=self.sha256(raw),
                target_value=str(self.target),
            )

        outside = Path(self.temporary.name).parent / "outside-receipt.json"
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            writer.write_receipt(
                source_value=str(self.source),
                expected_source_sha256=self.sha256(raw),
                target_value=str(outside),
            )

    def test_symlinked_source_path_is_rejected(self) -> None:
        raw = self.write_source()
        link = self.root / "source-link.json"
        link.symlink_to(self.source)

        with self.assertRaisesRegex(ValueError, "symlink-free"):
            writer.write_receipt(
                source_value=str(link),
                expected_source_sha256=self.sha256(raw),
                target_value=str(self.target),
            )

    def test_existing_target_is_never_overwritten(self) -> None:
        raw = self.write_source()
        original = b"existing receipt\n"
        self.target.write_bytes(original)

        with self.assertRaises(FileExistsError):
            self.call_writer(raw)
        self.assertEqual(self.target.read_bytes(), original)

    def test_dangling_target_symlink_is_rejected(self) -> None:
        raw = self.write_source()
        self.target.symlink_to(self.root / "missing-referent.json")

        with self.assertRaises(FileExistsError):
            self.call_writer(raw)
        self.assertTrue(self.target.is_symlink())

    def test_success_preserves_exact_bytes_and_fsyncs_file_and_parent(self) -> None:
        self.target = self.root / "保留-UTF8-target.json"
        raw = self.write_source(self.payload_bytes())
        real_fsync = os.fsync
        real_open = os.open
        fsynced: list[tuple[int, int]] = []
        open_calls: list[tuple[str, int, int | None]] = []

        def recording_fsync(descriptor: int) -> None:
            fsynced.append((descriptor, os.fstat(descriptor).st_mode))
            real_fsync(descriptor)

        def recording_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            *args: object,
            **kwargs: object,
        ) -> int:
            dir_fd = kwargs.get("dir_fd")
            self.assertTrue(dir_fd is None or isinstance(dir_fd, int))
            open_calls.append(
                (
                    os.fsdecode(os.fspath(path)),
                    flags,
                    dir_fd if isinstance(dir_fd, int) else None,
                )
            )
            return real_open(path, flags, *args, **kwargs)

        with (
            mock.patch.object(writer.os, "fsync", side_effect=recording_fsync),
            mock.patch.object(writer.os, "open", side_effect=recording_open),
        ):
            digest = self.call_writer(raw)

        self.assertEqual(self.target.read_bytes(), raw)
        self.assertEqual(digest, self.sha256(raw))
        self.assertEqual(len(fsynced), 2)
        self.assertTrue(stat.S_ISREG(fsynced[0][1]))
        self.assertTrue(stat.S_ISDIR(fsynced[1][1]))
        target_calls = [
            (flags, dir_fd)
            for path, flags, dir_fd in open_calls
            if path == self.target.name
        ]
        self.assertEqual(
            len(target_calls),
            1,
        )
        target_flags, target_dir_fd = target_calls[0]
        self.assertEqual(
            target_flags,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        )
        self.assertIsNotNone(target_dir_fd)
        self.assertEqual(fsynced[1][0], target_dir_fd)

    def test_ancestor_symlink_swap_is_detected_and_cleanup_is_safe(
        self,
    ) -> None:
        original_ancestor = self.root / "branch"
        original_parent = original_ancestor / "receipts"
        original_parent.mkdir(parents=True)
        moved_ancestor = self.root / "branch-original"
        attacker_ancestor = self.root / "attacker"
        attacker_parent = attacker_ancestor / "receipts"
        attacker_parent.mkdir(parents=True)
        self.target = original_parent / "target.json"
        raw = self.write_source()
        attacker_target = attacker_parent / "target.json"
        attacker_bytes = b"attacker replacement must survive\n"
        attacker_target.write_bytes(attacker_bytes)
        real_fsync = os.fsync
        swapped = False

        def swapping_fsync(descriptor: int) -> None:
            nonlocal swapped
            real_fsync(descriptor)
            if stat.S_ISREG(os.fstat(descriptor).st_mode) and not swapped:
                swapped = True
                original_ancestor.rename(moved_ancestor)
                original_ancestor.symlink_to(
                    attacker_ancestor,
                    target_is_directory=True,
                )

        with (
            mock.patch.object(writer.os, "fsync", side_effect=swapping_fsync),
            self.assertRaisesRegex(ValueError, "symlink-free"),
        ):
            self.call_writer(raw)

        self.assertTrue(swapped)
        retained = moved_ancestor / "receipts" / "target.json"
        self.assertTrue(retained.is_file())
        self.assertEqual(retained.read_bytes(), raw)
        self.assertEqual(attacker_target.read_bytes(), attacker_bytes)

    def test_truncation_after_file_fsync_is_detected_and_retained(self) -> None:
        raw = self.write_source()
        real_fsync = os.fsync
        corrupted = False

        def truncating_fsync(descriptor: int) -> None:
            nonlocal corrupted
            real_fsync(descriptor)
            if stat.S_ISREG(os.fstat(descriptor).st_mode) and not corrupted:
                corrupted = True
                corrupt_descriptor = os.open(self.target, os.O_WRONLY | os.O_TRUNC)
                os.close(corrupt_descriptor)

        with (
            mock.patch.object(
                writer.os,
                "fsync",
                side_effect=truncating_fsync,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "size differs|metadata changed",
            ),
        ):
            self.call_writer(raw)

        self.assertTrue(corrupted)
        self.assertTrue(os.path.lexists(self.target))
        self.assertEqual(self.target.read_bytes(), b"")

    def test_repo_root_replacement_between_trust_and_open_is_rejected(
        self,
    ) -> None:
        raw = self.write_source()
        moved_root = self.sandbox / "repo-original"
        real_open = os.open
        swapped = False

        def replacing_root_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            *args: object,
            **kwargs: object,
        ) -> int:
            nonlocal swapped
            if os.fsdecode(os.fspath(path)) == str(self.root) and not swapped:
                swapped = True
                self.root.rename(moved_root)
                self.root.mkdir()
            return real_open(path, flags, *args, **kwargs)

        with (
            mock.patch.object(
                writer.os,
                "open",
                side_effect=replacing_root_open,
            ),
            self.assertRaisesRegex(RuntimeError, "trusted identity"),
        ):
            self.call_writer(raw)

        self.assertTrue(swapped)
        self.assertFalse((self.root / self.target.name).exists())
        self.assertFalse((moved_root / self.target.name).exists())

    def test_repo_root_replacement_during_target_fsync_fails_closed(
        self,
    ) -> None:
        raw = self.write_source()
        moved_root = self.sandbox / "repo-original"
        real_fsync = os.fsync
        swapped = False

        def replacing_root_fsync(descriptor: int) -> None:
            nonlocal swapped
            real_fsync(descriptor)
            if stat.S_ISREG(os.fstat(descriptor).st_mode) and not swapped:
                swapped = True
                self.root.rename(moved_root)
                self.root.mkdir()

        with (
            mock.patch.object(
                writer.os,
                "fsync",
                side_effect=replacing_root_fsync,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "Repository root path changed",
            ),
        ):
            self.call_writer(raw)

        self.assertTrue(swapped)
        self.assertFalse((self.root / self.target.name).exists())
        retained = moved_root / self.target.name
        self.assertTrue(retained.is_file())
        self.assertEqual(retained.read_bytes(), raw)

    def test_failure_never_unlinks_a_replacement_victim(self) -> None:
        raw = self.write_source()
        retained_creation = self.root / "retained-created-receipt.json"
        victim_bytes = b"unrelated victim must survive\n"
        replaced = False

        def replace_then_fail(
            descriptor: int,
            *,
            raw: bytes,
            expected_sha256: str,
            stable_reference: os.stat_result | None,
        ) -> os.stat_result:
            del descriptor, raw, expected_sha256, stable_reference
            nonlocal replaced
            self.target.rename(retained_creation)
            self.target.write_bytes(victim_bytes)
            replaced = True
            raise RuntimeError("synthetic verification failure after replacement")

        with (
            mock.patch.object(
                writer,
                "_verify_target_fd",
                side_effect=replace_then_fail,
            ),
            mock.patch.object(writer.os, "unlink", wraps=os.unlink) as unlink,
            self.assertRaisesRegex(RuntimeError, "synthetic verification"),
        ):
            self.call_writer(raw)

        self.assertTrue(replaced)
        self.assertEqual(self.target.read_bytes(), victim_bytes)
        self.assertEqual(retained_creation.read_bytes(), raw)
        unlink.assert_not_called()

    def test_payload_field_types_are_strict(self) -> None:
        raw = self.write_source(self.payload_bytes(write_once_target=123))
        with self.assertRaisesRegex(ValueError, "must be a string"):
            self.call_writer(raw)

        raw = self.write_source(
            self.payload_bytes(
                original_receipt_must_not_be_overwritten=1
            )
        )
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.call_writer(raw)

        raw = self.write_source(
            json.dumps(
                [
                    str(self.target),
                    True,
                ]
            ).encode("utf-8")
        )
        with self.assertRaisesRegex(ValueError, "top-level object"):
            self.call_writer(raw)

        raw = self.write_source(self.payload_bytes(terminal=1))
        with self.assertRaisesRegex(ValueError, "terminal must be boolean true"):
            self.call_writer(raw)

        raw = self.write_source(self.payload_bytes(terminal_exit_code=True))
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.call_writer(raw)

        raw = self.write_source(self.payload_bytes(transport=[]))
        with self.assertRaisesRegex(ValueError, "transport must be an object"):
            self.call_writer(raw)

    def test_payload_schema_enums_hashes_and_paths_are_strict(self) -> None:
        raw = self.write_source(self.payload_bytes(status="unknown"))
        with self.assertRaisesRegex(ValueError, "status must be either"):
            self.call_writer(raw)

        raw = self.write_source(
            self.payload_bytes(status="completed", terminal_exit_code=1)
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.call_writer(raw)

        payload = json.loads(self.payload_bytes())
        payload["bindings"]["effective_command_sha256"] = "ABC"
        raw = self.write_source(
            (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        )
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            self.call_writer(raw)

        payload = json.loads(self.payload_bytes())
        payload["bindings"]["transport_lock"]["path"] = "relative.json"
        raw = self.write_source(
            (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        )
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            self.call_writer(raw)

        payload = json.loads(self.payload_bytes())
        payload["artifacts"]["terminal_checkpoint"]["update"] = True
        raw = self.write_source(
            (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.call_writer(raw)

        raw = self.write_source(self.payload_bytes(unexpected="extra"))
        with self.assertRaisesRegex(ValueError, "contain exactly"):
            self.call_writer(raw)

    def test_failed_terminal_receipt_uses_null_checkpoint(self) -> None:
        payload = json.loads(
            self.payload_bytes(
                status="failed",
                terminal_exit_code=17,
            )
        )
        payload["artifacts"]["terminal_checkpoint"] = None
        raw = self.write_source(
            (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        )

        digest = self.call_writer(raw)

        self.assertEqual(digest, self.sha256(raw))
        self.assertEqual(self.target.read_bytes(), raw)

    def test_payload_target_and_overwrite_guard_are_required(self) -> None:
        wrong_target = self.root / "other.json"
        raw = self.write_source(
            self.payload_bytes(write_once_target=str(wrong_target))
        )
        with self.assertRaisesRegex(ValueError, "write_once_target"):
            self.call_writer(raw)

        raw = self.write_source(
            self.payload_bytes(
                original_receipt_must_not_be_overwritten=False
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "original_receipt_must_not_be_overwritten",
        ):
            self.call_writer(raw)


if __name__ == "__main__":
    unittest.main()
