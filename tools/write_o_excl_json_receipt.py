#!/usr/bin/env python3
"""Copy a validated JSON receipt to a write-once repository path.

All path traversal is anchored to a held repository-root directory descriptor.
The source is parsed only for validation; the target receives the exact raw
source bytes and is created exclusively, so an existing receipt is never
overwritten.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_SCHEMA_VERSION = (
    "ptcg-preregistered-ppo-training-terminal-receipt-v1"
)
RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "terminal",
        "terminal_exit_code",
        "attempt_consumed",
        "transport",
        "bindings",
        "artifacts",
        "write_once_target",
        "original_receipt_must_not_be_overwritten",
    }
)
TRANSPORT_KEYS = frozenset(
    {
        "tool",
        "sandbox_permissions",
        "login",
        "tty",
        "workdir",
        "shell",
    }
)
TRANSPORT_VALUES = {
    "tool": "functions.exec_command",
    "sandbox_permissions": "require_escalated",
    "login": False,
    "tty": False,
    "shell": "/bin/bash",
}
BOUND_FILE_KEYS = frozenset({"path", "sha256"})
CHECKPOINT_KEYS = frozenset({"path", "sha256", "update"})
BINDINGS_KEYS = frozenset(
    {"transport_lock", "effective_command_sha256"}
)
ARTIFACT_KEYS = frozenset({"stdout_stderr_log", "terminal_checkpoint"})
STABLE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-standard constant {value!r}")


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _same_stable_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field) == getattr(right, field)
        for field in STABLE_STAT_FIELDS
    )


def _trusted_repo_root() -> tuple[Path, os.stat_result]:
    root = REPO_ROOT
    raw = os.fspath(root)
    if not root.is_absolute() or os.path.normpath(raw) != raw:
        raise ValueError(
            "Repository root must be an absolute normalized path"
        )
    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Repository root does not exist: {root}") from error
    if resolved != root:
        raise ValueError("Repository root must be symlink-free")
    trusted_stat = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(trusted_stat.st_mode):
        raise ValueError("Repository root must be a directory")
    return root, trusted_stat


def _lexical_repo_path(
    raw_value: str,
    *,
    root: Path,
    label: str,
) -> tuple[Path, tuple[str, ...], str]:
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"{label} must be a non-empty path")
    if os.path.normpath(raw_value) != raw_value:
        raise ValueError(f"{label} must be normalized")

    path = Path(raw_value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository") from error
    if not relative.parts:
        raise ValueError(f"{label} must name a file below the repository root")
    return path, tuple(relative.parts[:-1]), relative.parts[-1]


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("O_NOFOLLOW and O_DIRECTORY are required")
    return os.O_RDONLY | directory | nofollow


def _open_root(root: Path, trusted_stat: os.stat_result) -> int:
    descriptor = os.open(root, _directory_flags())
    try:
        opened = os.fstat(descriptor)
        current = os.stat(root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not _same_inode(opened, trusted_stat)
            or not _same_inode(current, trusted_stat)
        ):
            raise RuntimeError(
                "Repository root changed after its trusted identity was read"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_root_identity_current(
    root: Path,
    trusted_stat: os.stat_result,
    root_descriptor: int,
) -> None:
    """Fail if the frozen absolute repository path no longer names held root."""

    held = os.fstat(root_descriptor)
    try:
        current = os.stat(root, follow_symlinks=False)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Repository root path disappeared during receipt creation"
        ) from error
    if (
        not stat.S_ISDIR(held.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or not _same_inode(held, trusted_stat)
        or not _same_inode(current, trusted_stat)
    ):
        raise RuntimeError(
            "Repository root path changed during receipt creation"
        )


def _open_directory_chain(
    root_descriptor: int,
    components: Sequence[str],
    *,
    label: str,
) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for component in components:
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        f"{label} must be symlink-free directories"
                    ) from error
                if error.errno == errno.ENOENT:
                    raise FileNotFoundError(
                        f"{label} directory does not exist: {component}"
                    ) from error
                raise
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise ValueError(f"{label} must contain only directories")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_source(
    parent_descriptor: int,
    basename: str,
    *,
    label: str = "source",
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(basename, flags, dir_fd=parent_descriptor)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"{label} must be symlink-free") from error
        if error.errno == errno.ENOENT:
            raise FileNotFoundError(
                f"{label} does not exist: {basename}"
            ) from error
        raise
    try:
        opened = os.fstat(descriptor)
        current = os.stat(
            basename,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if not _same_inode(opened, current):
            raise RuntimeError(f"{label} changed while it was opened")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_stable_source(
    descriptor: int,
    *,
    label: str = "source",
) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    raw = _read_all(descriptor)
    after = os.fstat(descriptor)
    if not _same_stable_stat(before, after):
        raise RuntimeError(f"{label} changed while it was being read")
    if len(raw) != after.st_size:
        raise RuntimeError(f"{label} size changed while it was being read")
    return raw


def _parse_json_object(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("source must contain UTF-8 JSON") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("source must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("source JSON must have a top-level object")
    return value


def _require_exact_object(
    value: object,
    *,
    label: str,
    expected_keys: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if set(value) != set(expected_keys):
        raise ValueError(
            f"{label} must contain exactly {sorted(expected_keys)}"
        )
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_bound_regular_file(
    value: object,
    *,
    root: Path,
    root_descriptor: int,
    label: str,
    expected_keys: frozenset[str] = BOUND_FILE_KEYS,
) -> tuple[Path, dict[str, Any]]:
    binding = _require_exact_object(
        value,
        label=label,
        expected_keys=expected_keys,
    )
    raw_path = binding.get("path")
    if not isinstance(raw_path, str):
        raise ValueError(f"{label}.path must be a string")
    path, parent_parts, basename = _lexical_repo_path(
        raw_path,
        root=root,
        label=f"{label}.path",
    )
    expected_sha256 = _require_sha256(
        binding.get("sha256"),
        label=f"{label}.sha256",
    )

    parent_descriptor = _open_directory_chain(
        root_descriptor,
        parent_parts,
        label=f"{label}.path",
    )
    descriptor: int | None = None
    try:
        descriptor = _open_source(
            parent_descriptor,
            basename,
            label=label,
        )
        raw = _read_stable_source(descriptor, label=label)
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(f"{label} SHA-256 mismatch")
        _postcheck_source_path(
            root_descriptor=root_descriptor,
            parent_components=parent_parts,
            held_parent_descriptor=parent_descriptor,
            basename=basename,
            source_descriptor=descriptor,
            label=label,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    return path, binding


def _validate_payload(
    payload: dict[str, Any],
    *,
    source: Path,
    target: Path,
    root: Path,
    root_descriptor: int,
) -> None:
    if set(payload) != set(RECEIPT_KEYS):
        raise ValueError(
            f"payload must contain exactly {sorted(RECEIPT_KEYS)}"
        )
    if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            f"payload.schema_version must be {RECEIPT_SCHEMA_VERSION!r}"
        )

    status = payload.get("status")
    if not isinstance(status, str) or status not in {"completed", "failed"}:
        raise ValueError(
            "payload.status must be either 'completed' or 'failed'"
        )
    terminal = payload.get("terminal")
    if not isinstance(terminal, bool) or terminal is not True:
        raise ValueError("payload.terminal must be boolean true")
    attempt_consumed = payload.get("attempt_consumed")
    if not isinstance(attempt_consumed, bool) or attempt_consumed is not True:
        raise ValueError("payload.attempt_consumed must be boolean true")
    exit_code = payload.get("terminal_exit_code")
    if (
        isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or not 0 <= exit_code <= 255
    ):
        raise ValueError(
            "payload.terminal_exit_code must be an integer from 0 to 255"
        )
    if (status == "completed") != (exit_code == 0):
        raise ValueError(
            "payload.status and terminal_exit_code are inconsistent"
        )

    transport = _require_exact_object(
        payload.get("transport"),
        label="payload.transport",
        expected_keys=TRANSPORT_KEYS,
    )
    for key, expected in TRANSPORT_VALUES.items():
        observed = transport.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"payload.transport.{key} must be {expected!r}"
            )
    workdir = transport.get("workdir")
    if not isinstance(workdir, str) or workdir != str(root):
        raise ValueError(
            "payload.transport.workdir must equal the repository root"
        )

    write_once_target = payload.get("write_once_target")
    if not isinstance(write_once_target, str):
        raise ValueError("payload.write_once_target must be a string")
    if write_once_target != str(target):
        raise ValueError("payload.write_once_target must equal target")

    overwrite_guard = payload.get("original_receipt_must_not_be_overwritten")
    if not isinstance(overwrite_guard, bool):
        raise ValueError(
            "payload.original_receipt_must_not_be_overwritten "
            "must be a boolean"
        )
    if overwrite_guard is not True:
        raise ValueError(
            "payload.original_receipt_must_not_be_overwritten must be true"
        )

    bindings = _require_exact_object(
        payload.get("bindings"),
        label="payload.bindings",
        expected_keys=BINDINGS_KEYS,
    )
    effective_command_sha256 = _require_sha256(
        bindings.get("effective_command_sha256"),
        label="payload.bindings.effective_command_sha256",
    )
    if effective_command_sha256 == "0" * 64:
        raise ValueError(
            "payload.bindings.effective_command_sha256 must not be all zero"
        )
    transport_lock_path, _ = _validate_bound_regular_file(
        bindings.get("transport_lock"),
        root=root,
        root_descriptor=root_descriptor,
        label="payload.bindings.transport_lock",
    )

    artifacts = _require_exact_object(
        payload.get("artifacts"),
        label="payload.artifacts",
        expected_keys=ARTIFACT_KEYS,
    )
    log_path, _ = _validate_bound_regular_file(
        artifacts.get("stdout_stderr_log"),
        root=root,
        root_descriptor=root_descriptor,
        label="payload.artifacts.stdout_stderr_log",
    )
    checkpoint_value = artifacts.get("terminal_checkpoint")
    checkpoint_path: Path | None = None
    if status == "completed":
        checkpoint_path, checkpoint = _validate_bound_regular_file(
            checkpoint_value,
            root=root,
            root_descriptor=root_descriptor,
            label="payload.artifacts.terminal_checkpoint",
            expected_keys=CHECKPOINT_KEYS,
        )
        update = checkpoint.get("update")
        if isinstance(update, bool) or not isinstance(update, int) or update <= 0:
            raise ValueError(
                "payload.artifacts.terminal_checkpoint.update "
                "must be a positive integer"
            )
    elif checkpoint_value is not None:
        raise ValueError(
            "payload.artifacts.terminal_checkpoint must be null "
            "when status is failed"
        )

    bound_paths = [source, target, transport_lock_path, log_path]
    if checkpoint_path is not None:
        bound_paths.append(checkpoint_path)
    if len(set(bound_paths)) != len(bound_paths):
        raise ValueError(
            "source, target, transport lock, log, and checkpoint "
            "paths must be distinct"
        )


def _assert_leaf_absent(parent_descriptor: int, basename: str) -> None:
    try:
        os.stat(
            basename,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise FileExistsError(f"target already exists: {basename}")


def _postcheck_source_path(
    *,
    root_descriptor: int,
    parent_components: Sequence[str],
    held_parent_descriptor: int,
    basename: str,
    source_descriptor: int,
    label: str = "source",
) -> None:
    current_parent = _open_directory_chain(
        root_descriptor,
        parent_components,
        label=label,
    )
    try:
        if not _same_inode(
            os.fstat(held_parent_descriptor),
            os.fstat(current_parent),
        ):
            raise RuntimeError(
                f"{label} parent path changed during validation"
            )
        current_source = os.stat(
            basename,
            dir_fd=current_parent,
            follow_symlinks=False,
        )
        if not _same_inode(current_source, os.fstat(source_descriptor)):
            raise RuntimeError(f"{label} path changed during validation")
    except FileNotFoundError as error:
        raise RuntimeError(
            f"{label} path disappeared during validation"
        ) from error
    finally:
        os.close(current_parent)


def _target_open_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW is required")
    return os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise OSError("short write while creating target")
        offset += written


def _verify_target_fd(
    descriptor: int,
    *,
    raw: bytes,
    expected_sha256: str,
    stable_reference: os.stat_result | None,
) -> os.stat_result:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("created target is not a regular file")
    if before.st_size != len(raw):
        raise RuntimeError("created target size differs from source")
    if stable_reference is not None and not _same_stable_stat(
        stable_reference,
        before,
    ):
        raise RuntimeError("created target metadata changed unexpectedly")

    os.lseek(descriptor, 0, os.SEEK_SET)
    observed = _read_all(descriptor)
    after = os.fstat(descriptor)
    if not _same_stable_stat(before, after):
        raise RuntimeError("created target changed while it was verified")
    if observed != raw:
        raise RuntimeError("created target bytes differ from source")
    observed_sha256 = hashlib.sha256(observed).hexdigest()
    if observed_sha256 != expected_sha256:
        raise RuntimeError("created target SHA-256 differs from source")
    return after


def _postcheck_target_path(
    *,
    root_descriptor: int,
    parent_components: Sequence[str],
    held_parent_descriptor: int,
    basename: str,
    target_descriptor: int,
) -> None:
    current_parent = _open_directory_chain(
        root_descriptor,
        parent_components,
        label="target",
    )
    try:
        if not _same_inode(
            os.fstat(held_parent_descriptor),
            os.fstat(current_parent),
        ):
            raise RuntimeError("target parent path changed during creation")
        current_target = os.stat(
            basename,
            dir_fd=current_parent,
            follow_symlinks=False,
        )
        if not _same_inode(current_target, os.fstat(target_descriptor)):
            raise RuntimeError("target path no longer names the created file")
    except FileNotFoundError as error:
        raise RuntimeError("target path disappeared during creation") from error
    finally:
        os.close(current_parent)


def write_receipt(
    *,
    source_value: str,
    expected_source_sha256: str,
    target_value: str,
) -> str:
    """Validate and copy a source receipt to an exclusively created target."""

    if not isinstance(expected_source_sha256, str) or (
        SHA256_RE.fullmatch(expected_source_sha256) is None
    ):
        raise ValueError(
            "expected-source-sha256 must be a lowercase SHA-256 digest"
        )

    root, trusted_root_stat = _trusted_repo_root()
    source, source_parent_parts, source_basename = _lexical_repo_path(
        source_value,
        root=root,
        label="source",
    )
    target, target_parent_parts, target_basename = _lexical_repo_path(
        target_value,
        root=root,
        label="target",
    )
    if source == target:
        raise ValueError("source and target must be distinct")

    root_descriptor = _open_root(root, trusted_root_stat)
    source_parent_descriptor: int | None = None
    source_descriptor: int | None = None
    target_parent_descriptor: int | None = None
    target_descriptor: int | None = None
    created_stat: os.stat_result | None = None
    try:
        source_parent_descriptor = _open_directory_chain(
            root_descriptor,
            source_parent_parts,
            label="source",
        )
        source_descriptor = _open_source(
            source_parent_descriptor,
            source_basename,
        )
        raw = _read_stable_source(source_descriptor)
        source_sha256 = hashlib.sha256(raw).hexdigest()
        if source_sha256 != expected_source_sha256:
            raise ValueError(
                "source SHA-256 does not match expected-source-sha256"
            )

        payload = _parse_json_object(raw)
        _validate_payload(
            payload,
            source=source,
            target=target,
            root=root,
            root_descriptor=root_descriptor,
        )
        _postcheck_source_path(
            root_descriptor=root_descriptor,
            parent_components=source_parent_parts,
            held_parent_descriptor=source_parent_descriptor,
            basename=source_basename,
            source_descriptor=source_descriptor,
        )

        target_parent_descriptor = _open_directory_chain(
            root_descriptor,
            target_parent_parts,
            label="target",
        )
        _assert_root_identity_current(
            root,
            trusted_root_stat,
            root_descriptor,
        )
        _assert_leaf_absent(target_parent_descriptor, target_basename)

        target_descriptor = os.open(
            target_basename,
            _target_open_flags(),
            0o600,
            dir_fd=target_parent_descriptor,
        )
        created_stat = os.fstat(target_descriptor)
        if not stat.S_ISREG(created_stat.st_mode):
            raise RuntimeError("exclusively created target is not a regular file")

        _write_all(target_descriptor, raw)
        verified_stat = _verify_target_fd(
            target_descriptor,
            raw=raw,
            expected_sha256=source_sha256,
            stable_reference=None,
        )
        os.fsync(target_descriptor)
        verified_stat = _verify_target_fd(
            target_descriptor,
            raw=raw,
            expected_sha256=source_sha256,
            stable_reference=verified_stat,
        )
        _postcheck_target_path(
            root_descriptor=root_descriptor,
            parent_components=target_parent_parts,
            held_parent_descriptor=target_parent_descriptor,
            basename=target_basename,
            target_descriptor=target_descriptor,
        )
        _assert_root_identity_current(
            root,
            trusted_root_stat,
            root_descriptor,
        )
        os.fsync(target_parent_descriptor)
        _verify_target_fd(
            target_descriptor,
            raw=raw,
            expected_sha256=source_sha256,
            stable_reference=verified_stat,
        )
        _postcheck_target_path(
            root_descriptor=root_descriptor,
            parent_components=target_parent_parts,
            held_parent_descriptor=target_parent_descriptor,
            basename=target_basename,
            target_descriptor=target_descriptor,
        )
        _assert_root_identity_current(
            root,
            trusted_root_stat,
            root_descriptor,
        )
        return source_sha256
    except BaseException:
        if (
            created_stat is not None
            and target_parent_descriptor is not None
        ):
            # Never perform lstat-then-unlink cleanup: another actor could
            # replace the basename between those operations and turn cleanup
            # into deletion of an unrelated victim.  A failed, exclusively
            # created inode is deliberately retained as a no-retry witness.
            try:
                os.fsync(target_parent_descriptor)
            except OSError:
                pass
        raise
    finally:
        for descriptor in (
            target_descriptor,
            target_parent_descriptor,
            source_descriptor,
            source_parent_descriptor,
            root_descriptor,
        ):
            if descriptor is not None:
                os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--target", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    digest = write_receipt(
        source_value=args.source,
        expected_source_sha256=args.expected_source_sha256,
        target_value=args.target,
    )
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
