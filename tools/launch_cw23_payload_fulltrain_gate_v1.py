#!/usr/bin/env python3
"""Hash-bound in-process launcher for the frozen CW23 full-train adapter."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
LAUNCHER = TOOLS / "launch_cw23_payload_fulltrain_gate_v1.py"
ADAPTER = TOOLS / "run_cw23_payload_fulltrain_gate_v1.py"
ADAPTER_SHA256 = "70f7a5dc10f6f0cc6e5dc863b9592a292c2802a62584b3ef7c2c17776fe39130"
FROZEN_MODE = 0o555
ATTEMPT = ROOT / "artifacts/.ptcg-cw23_payload_fulltrain_gate_20260803_v1-attempt.json"
OUTPUT = ROOT / "artifacts/cw23_payload_fulltrain_gate_20260803_v1.json"


class LaunchError(RuntimeError):
    """Fail-closed hash-bound launch error."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def lexists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def read_regular(
    path: Path, expected_sha256: str | None, expected_mode: int, label: str
) -> tuple[bytes, dict[str, Any]]:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or int(before.st_nlink) != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
    ):
        raise LaunchError(f"{label}: unsafe type/link/mode")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    chunks: list[bytes] = []
    try:
        opened = os.fstat(fd)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
        after_path = path.lstat()
    finally:
        os.close(fd)
    if not identity(before) == identity(opened) == identity(after_fd) == identity(
        after_path
    ):
        raise LaunchError(f"{label}: identity changed during held-FD read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    if (
        len(payload) != int(after_fd.st_size)
        or (expected_sha256 is not None and digest != expected_sha256)
    ):
        raise LaunchError(f"{label}: SHA/size drift")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(stat.S_IMODE(after_fd.st_mode), "04o"),
        "device": int(after_fd.st_dev),
        "inode": int(after_fd.st_ino),
        "nlink": int(after_fd.st_nlink),
    }


def open_adapter_held() -> tuple[bytes, dict[str, Any], int, tuple[int, ...]]:
    before = ADAPTER.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or int(before.st_nlink) != 1
        or stat.S_IMODE(before.st_mode) != FROZEN_MODE
    ):
        raise LaunchError("adapter: unsafe type/link/mode")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(ADAPTER, flags)
    try:
        opened = os.fstat(fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
        after_path = ADAPTER.lstat()
        locked_identity = identity(after_fd)
        if not identity(before) == identity(opened) == locked_identity == identity(
            after_path
        ):
            raise LaunchError("adapter: identity changed during held-FD read")
        payload = b"".join(chunks)
        digest = sha256_bytes(payload)
        if len(payload) != int(after_fd.st_size) or digest != ADAPTER_SHA256:
            raise LaunchError("adapter: SHA/size drift")
        record = {
            "path": str(ADAPTER.relative_to(ROOT)),
            "sha256": digest,
            "bytes": len(payload),
            "mode_octal": format(stat.S_IMODE(after_fd.st_mode), "04o"),
            "device": int(after_fd.st_dev),
            "inode": int(after_fd.st_ino),
            "nlink": int(after_fd.st_nlink),
        }
        return payload, record, fd, locked_identity
    except BaseException:
        os.close(fd)
        raise


def main() -> None:
    audit_only = sys.argv[1:] == ["--audit-only"]
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "launcher_file_exact": Path(__file__).resolve() == LAUNCHER.resolve(),
        "argv0_exact": Path(sys.argv[0]).resolve() == LAUNCHER.resolve(),
        "argv_exact": sys.argv
        in ([str(LAUNCHER)], [str(LAUNCHER), "--audit-only"]),
        "targets_absent": not lexists(ATTEMPT) and not lexists(OUTPUT),
    }
    if not all(checks.values()):
        raise LaunchError(f"launcher runtime/target drift: {checks}")
    launcher_source, launcher_record = read_regular(
        LAUNCHER, None, FROZEN_MODE, "launcher"
    )
    adapter_source, adapter_record, adapter_fd, adapter_identity = open_adapter_held()
    original_argv = list(sys.argv)
    binding = {
        "schema_version": "ptcg-cw23-fulltrain-hash-bound-launch-v1",
        "adapter_path": str(ADAPTER),
        "adapter_sha256": ADAPTER_SHA256,
        "adapter_bytes": len(adapter_source),
        "adapter_mode_octal": adapter_record["mode_octal"],
        "held_fd_identity_exact": True,
        "compile_exec_same_verified_bytes": True,
        "audit_only": audit_only,
        "launcher": launcher_record,
        "launcher_source_sha256": sha256_bytes(launcher_source),
        "original_argv": original_argv,
        "runtime_checks": checks,
    }
    code = compile(
        adapter_source, str(ADAPTER), "exec", dont_inherit=True, optimize=0
    )
    sys.argv = [str(ADAPTER)] + (["--audit-only"] if audit_only else [])
    namespace = {
        "__name__": "__main__",
        "__file__": str(ADAPTER),
        "__package__": None,
        "__loader__": None,
        "__spec__": None,
        "__cached__": None,
        "_PTCG_HASH_BOUND_EXECUTION": binding,
    }
    try:
        exec(code, namespace, namespace)
        if identity(os.fstat(adapter_fd)) != adapter_identity or identity(
            ADAPTER.lstat()
        ) != adapter_identity:
            raise LaunchError("held adapter identity changed during execution")
    finally:
        os.close(adapter_fd)
    _, adapter_after = read_regular(
        ADAPTER, ADAPTER_SHA256, FROZEN_MODE, "adapter after execution"
    )
    _, launcher_after = read_regular(
        LAUNCHER, launcher_record["sha256"], FROZEN_MODE, "launcher after execution"
    )
    if adapter_after != adapter_record or launcher_after != launcher_record:
        raise LaunchError("launcher/adapter identity changed during execution")


if __name__ == "__main__":
    main()
