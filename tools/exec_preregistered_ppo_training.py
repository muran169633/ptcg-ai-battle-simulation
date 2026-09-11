#!/usr/bin/env python3
"""One-shot direct executor for a hash-bound PPO training command.

The launcher performs only local validation, claims a write-once log, and
replaces itself with the exact command stored in the frozen branch
preregistration.  It deliberately has no shell or child-process fallback.

The transport lock binds the already-frozen launcher by path and digest.
Only the transport lock's *own* digest is supplied separately on the CLI,
which avoids making the lock hash self-referential.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "ptcg-preregistered-ppo-training-transport-lock-v4"
LOCKED_STATUS = "locked_before_training"
TRANSPORT_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "bindings",
        "child_command",
        "attempt",
        "expected_transport",
    }
)
BRANCH_PREREGISTRATION_KEYS = frozenset(
    {
        "binding",
        "binding_sha256",
        "created_at",
        "schema_version",
        "status",
    }
)
BRANCH_BINDING_COMMAND_KEYS = frozenset(
    {
        "command",
        "command_sha256",
        "output_dir",
        "seed",
        "source_train_script",
        "source_train_script_sha256",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_BINDINGS = frozenset(
    {
        "launcher",
        "source_protocol",
        "comprehensive_preregistration",
        "runner_preregistration",
        "branch_preregistration",
        "trainer",
    }
)
EXPECTED_TRANSPORT_CONSTANTS = {
    "tool": "functions.exec_command",
    "sandbox_permissions": "require_escalated",
    "login": False,
    "tty": False,
    "shell": "/bin/bash",
    "topology": "supervised_fork_exec_v1",
}
EXPECTED_TRANSPORT_KEYS = frozenset(
    set(EXPECTED_TRANSPORT_CONSTANTS) | {"workdir"}
)
CHILD_COMMAND_KEYS = frozenset(
    {
        "source_binding",
        "json_path",
        "token_count",
        "canonical_sha256",
        "effective_trainer_fd",
        "effective_trainer_path",
        "sealed_trainer_payload_fd",
        "sealed_trainer_payload_path",
        "sealed_bootstrap_sha256",
        "effective_root_fd",
        "effective_root_path",
        "effective_repo_path_token_indices",
        "effective_command_canonical_sha256",
    }
)
ATTEMPT_KEYS = frozenset(
    {
        "seed",
        "attempts_authorized",
        "output_dir",
        "log",
        "terminal_receipt",
        "terminal_checkpoint",
        "attempt_start_marker",
        "unpublished_log_witness",
        "expected_terminal_update",
        "absent_at_lock",
    }
)
LOCKED_ABSENCE_KEYS = frozenset(
    {
        "output_dir",
        "log",
        "terminal_receipt",
        "terminal_checkpoint",
        "attempt_start_marker",
        "unpublished_log_witness",
    }
)
EFFECTIVE_TRAINER_FD = 198
EFFECTIVE_TRAINER_PATH = f"/proc/self/fd/{EFFECTIVE_TRAINER_FD}"
SEALED_TRAINER_PAYLOAD_FD = 199
SEALED_TRAINER_PAYLOAD_PATH = (
    f"/proc/self/fd/{SEALED_TRAINER_PAYLOAD_FD}"
)
EFFECTIVE_ROOT_FD = 197
EFFECTIVE_ROOT_PATH = f"/proc/self/fd/{EFFECTIVE_ROOT_FD}"
CRITICAL_COMMAND_FLAGS = (
    "--seed",
    "--output-dir",
    "--updates",
)
# Linux UAPI constants.  The conda Python used by this project does not expose
# os.memfd_create or the fcntl seal names even though libc and the kernel
# support them, so the launcher uses libc.memfd_create plus numeric fcntl ABI.
MFD_CLOEXEC = 0x0001
MFD_ALLOW_SEALING = 0x0002
F_ADD_SEALS = 1033
F_GET_SEALS = 1034
F_SEAL_SEAL = 0x0001
F_SEAL_SHRINK = 0x0002
F_SEAL_GROW = 0x0004
F_SEAL_WRITE = 0x0008
AT_EMPTY_PATH = 0x1000
SUPERVISOR_CHILD_EXEC_FAILURE = 125


@dataclass(frozen=True)
class ValidatedLaunch:
    """All values needed after the no-side-effect validation phase."""

    root: Path
    root_fstat: dict[str, int]
    lock_path: Path
    lock_sha256: str
    launcher_sha256: str
    protocol: dict[str, Any]
    binding_paths: dict[str, Path]
    binding_sha256s: dict[str, str]
    command: tuple[str, ...]
    command_sha256: str
    effective_command: tuple[str, ...]
    effective_command_sha256: str
    effective_root_fd: int
    effective_root_path: str
    effective_repo_path_token_indices: tuple[int, ...]
    effective_trainer_fd: int
    effective_trainer_path: str
    sealed_trainer_payload_fd: int
    sealed_trainer_payload_path: str
    sealed_bootstrap_sha256: str
    seed: int
    output_dir: Path
    log: Path
    terminal_receipt: Path
    terminal_checkpoint: Path
    attempt_start_marker: Path
    unpublished_log_witness: Path


@dataclass(frozen=True)
class HeldTrainer:
    """Sealed bootstrap and source-identical payload inherited by the child."""

    fd: int
    payload_fd: int
    source_sha256: str
    source_fstat: dict[str, int]
    bootstrap_sha256: str
    bootstrap_fstat: dict[str, int]
    bootstrap_seals: int
    payload_sha256: str
    payload_fstat: dict[str, int]
    payload_seals: int


@dataclass(frozen=True)
class HeldRoot:
    fd: int
    fstat: dict[str, int]


class SupervisorInterrupted(RuntimeError):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"Supervisor received signal {signum}")


def _fstat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_uid),
        int(info.st_gid),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _fstat_record(info: os.stat_result) -> dict[str, int]:
    return {
        "st_ctime_ns": int(info.st_ctime_ns),
        "st_dev": int(info.st_dev),
        "st_gid": int(info.st_gid),
        "st_ino": int(info.st_ino),
        "st_mode": int(info.st_mode),
        "st_mtime_ns": int(info.st_mtime_ns),
        "st_nlink": int(info.st_nlink),
        "st_size": int(info.st_size),
        "st_uid": int(info.st_uid),
    }


def _same_root_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        int(left.st_dev),
        int(left.st_ino),
    ) == (
        int(right.st_dev),
        int(right.st_ino),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_attempt_start_marker(
    *,
    root: Path,
    log: Path,
    seed: int,
) -> Path:
    """Return the repo-root marker path bound to this log attempt."""

    if type(seed) is not int or seed < 0:
        raise ValueError("Attempt marker seed must be a non-negative integer")
    if (
        not root.is_absolute()
        or root != Path(os.path.normpath(os.fspath(root)))
    ):
        raise ValueError("Attempt marker root must be normalized and absolute")
    if (
        not log.is_absolute()
        or log != Path(os.path.normpath(os.fspath(log)))
    ):
        raise ValueError("Attempt marker log must be normalized and absolute")
    try:
        log.relative_to(root)
    except ValueError as error:
        raise ValueError("Attempt marker log must remain under root") from error
    token = canonical_json_sha256(
        {
            "log": str(log),
            "seed": seed,
        }
    )[:20]
    return root / f".ptcg-ppo-attempt-{seed}-{token}.json"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _normalized_absolute_path(
    value: Any,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    lexical = Path(os.path.normpath(os.fspath(path)))
    if path != lexical:
        raise ValueError(f"{label} must be lexically normalized")
    if not must_exist and os.path.lexists(path):
        # Existing write targets, including dangling symlinks, are returned
        # without following them so the explicit lstat-based absence gate can
        # reject them as consumed/reserved artifacts.
        return path
    try:
        resolved = path.resolve(strict=must_exist)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} does not exist: {path}") from error
    if path != resolved:
        raise ValueError(f"{label} must be normalized and symlink-free")
    return path


def _path_under_root(
    value: Any,
    *,
    root: Path,
    label: str,
    must_exist: bool,
) -> Path:
    path = _normalized_absolute_path(
        value,
        label=label,
        must_exist=must_exist,
    )
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain under required workdir") from error
    return path


def _single_command_value(command: Sequence[str], flag: str) -> str:
    indices = [index for index, token in enumerate(command) if token == flag]
    if len(indices) != 1:
        raise ValueError(f"binding.command must contain {flag} exactly once")
    index = indices[0]
    if index + 1 >= len(command):
        raise ValueError(f"binding.command has no value after {flag}")
    return command[index + 1]


def _reject_critical_flag_overrides(command: Sequence[str]) -> None:
    """Reject argparse abbreviations, spelling variants, and ``--flag=value``."""

    for token in command:
        if not token.startswith("--"):
            continue
        raw_name, separator, _ = token.partition("=")
        normalized = raw_name.casefold().replace("_", "-")
        for critical in CRITICAL_COMMAND_FLAGS:
            is_normalized_match = normalized == critical
            is_argparse_prefix = (
                len(normalized) > 2
                and normalized != critical
                and critical.startswith(normalized)
            )
            if is_argparse_prefix:
                raise ValueError(
                    f"binding.command contains forbidden abbreviation {token!r}"
                )
            if is_normalized_match and (
                separator or raw_name != critical
            ):
                raise ValueError(
                    f"binding.command contains forbidden critical-flag "
                    f"variant {token!r}"
                )


def _required_memfd_seals() -> int:
    """Return the mandatory irreversible seal set or reject this platform."""

    missing: list[str] = []
    if sys.platform != "linux":
        missing.append("Linux memfd/seal ABI")
    if not hasattr(fcntl, "fcntl"):
        missing.append("fcntl.fcntl")
    if not hasattr(os, "pread"):
        missing.append("os.pread")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        getattr(libc, "memfd_create")
    except (OSError, AttributeError):
        missing.append("libc.memfd_create")
    if missing:
        raise RuntimeError(
            "Sealed memfd transport is unavailable; missing "
            + ", ".join(missing)
        )
    exported_constants = (
        (os, "MFD_CLOEXEC", MFD_CLOEXEC),
        (os, "MFD_ALLOW_SEALING", MFD_ALLOW_SEALING),
        (fcntl, "F_ADD_SEALS", F_ADD_SEALS),
        (fcntl, "F_GET_SEALS", F_GET_SEALS),
        (fcntl, "F_SEAL_SEAL", F_SEAL_SEAL),
        (fcntl, "F_SEAL_SHRINK", F_SEAL_SHRINK),
        (fcntl, "F_SEAL_GROW", F_SEAL_GROW),
        (fcntl, "F_SEAL_WRITE", F_SEAL_WRITE),
    )
    for module, name, expected in exported_constants:
        if hasattr(module, name) and int(getattr(module, name)) != expected:
            raise RuntimeError(f"Linux ABI constant mismatch: {name}")
    return F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL


def _libc_memfd_create(name: str, flags: int) -> int:
    _required_memfd_seals()
    libc = ctypes.CDLL(None, use_errno=True)
    memfd_create = libc.memfd_create
    memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    memfd_create.restype = ctypes.c_int
    descriptor = int(memfd_create(name.encode("ascii"), flags))
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"memfd_create({name})",
        )
    return descriptor


def _require_supervisor_platform() -> None:
    missing: list[str] = []
    for name in (
        "fork",
        "waitpid",
        "WIFEXITED",
        "WEXITSTATUS",
        "WIFSIGNALED",
        "WTERMSIG",
        "O_TMPFILE",
    ):
        if not hasattr(os, name):
            missing.append(f"os.{name}")
    for name in ("pthread_sigmask", "SIG_BLOCK", "SIG_SETMASK"):
        if not hasattr(signal, name):
            missing.append(f"signal.{name}")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        getattr(libc, "linkat")
    except (OSError, AttributeError):
        missing.append("libc.linkat")
    if missing:
        raise RuntimeError(
            "Supervisor transport is unavailable; missing "
            + ", ".join(missing)
        )


def _link_fd_at_empty_path(
    *,
    source_fd: int,
    target_parent_fd: int,
    target_name: str,
) -> None:
    _require_supervisor_platform()
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    result = int(
        linkat(
            source_fd,
            b"",
            target_parent_fd,
            os.fsencode(target_name),
            AT_EMPTY_PATH,
        )
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            target_name,
        )


def build_trainer_bootstrap(
    *,
    canonical_trainer_path: Path,
    trainer_sha256: str,
    payload_fd: int = SEALED_TRAINER_PAYLOAD_FD,
) -> bytes:
    """Build the sealed shim that executes payload bytes as the real file."""

    _required_memfd_seals()
    if (
        not canonical_trainer_path.is_absolute()
        or canonical_trainer_path
        != Path(os.path.normpath(os.fspath(canonical_trainer_path)))
    ):
        raise ValueError("Canonical trainer path must be normalized and absolute")
    digest = _require_sha256(trainer_sha256, "trainer_sha256")
    if type(payload_fd) is not int or payload_fd != SEALED_TRAINER_PAYLOAD_FD:
        raise ValueError(
            f"Sealed trainer payload FD must equal "
            f"{SEALED_TRAINER_PAYLOAD_FD}"
        )
    path_literal = json.dumps(
        str(canonical_trainer_path),
        ensure_ascii=True,
    )
    digest_literal = json.dumps(digest)
    return (
        "#!/usr/bin/env python3\n"
        "import fcntl as _bootstrap_fcntl\n"
        "import hashlib as _bootstrap_hashlib\n"
        "import os as _bootstrap_os\n"
        "import stat as _bootstrap_stat\n"
        "import sys as _bootstrap_sys\n"
        f"_BOOTSTRAP_CANONICAL_PATH = {path_literal}\n"
        f"_BOOTSTRAP_EXPECTED_SHA256 = {digest_literal}\n"
        f"_BOOTSTRAP_PAYLOAD_FD = {payload_fd}\n"
        "_BOOTSTRAP_REQUIRED_SEALS = (\n"
        f"    {F_SEAL_WRITE}\n"
        f"    | {F_SEAL_SHRINK}\n"
        f"    | {F_SEAL_GROW}\n"
        f"    | {F_SEAL_SEAL}\n"
        ")\n"
        "_bootstrap_seals = _bootstrap_fcntl.fcntl(\n"
        "    _BOOTSTRAP_PAYLOAD_FD,\n"
        f"    {F_GET_SEALS},\n"
        ")\n"
        "if _bootstrap_seals != _BOOTSTRAP_REQUIRED_SEALS:\n"
        "    raise RuntimeError('Trainer payload memfd seal mismatch')\n"
        "_bootstrap_before = _bootstrap_os.fstat(_BOOTSTRAP_PAYLOAD_FD)\n"
        "if not _bootstrap_stat.S_ISREG(_bootstrap_before.st_mode):\n"
        "    raise RuntimeError('Trainer payload memfd is not regular')\n"
        "_bootstrap_chunks = []\n"
        "_bootstrap_offset = 0\n"
        "while _bootstrap_offset < _bootstrap_before.st_size:\n"
        "    _bootstrap_chunk = _bootstrap_os.pread(\n"
        "        _BOOTSTRAP_PAYLOAD_FD,\n"
        "        min(1048576, _bootstrap_before.st_size - _bootstrap_offset),\n"
        "        _bootstrap_offset,\n"
        "    )\n"
        "    if not _bootstrap_chunk:\n"
        "        raise RuntimeError('Short read from trainer payload memfd')\n"
        "    _bootstrap_chunks.append(_bootstrap_chunk)\n"
        "    _bootstrap_offset += len(_bootstrap_chunk)\n"
        "_bootstrap_payload = b''.join(_bootstrap_chunks)\n"
        "_bootstrap_after = _bootstrap_os.fstat(_BOOTSTRAP_PAYLOAD_FD)\n"
        "if (\n"
        "    _bootstrap_before.st_dev,\n"
        "    _bootstrap_before.st_ino,\n"
        "    _bootstrap_before.st_mode,\n"
        "    _bootstrap_before.st_nlink,\n"
        "    _bootstrap_before.st_size,\n"
        "    _bootstrap_before.st_mtime_ns,\n"
        "    _bootstrap_before.st_ctime_ns,\n"
        ") != (\n"
        "    _bootstrap_after.st_dev,\n"
        "    _bootstrap_after.st_ino,\n"
        "    _bootstrap_after.st_mode,\n"
        "    _bootstrap_after.st_nlink,\n"
        "    _bootstrap_after.st_size,\n"
        "    _bootstrap_after.st_mtime_ns,\n"
        "    _bootstrap_after.st_ctime_ns,\n"
        "):\n"
        "    raise RuntimeError('Trainer payload memfd changed while read')\n"
        "if _bootstrap_hashlib.sha256(_bootstrap_payload).hexdigest() != "
        "_BOOTSTRAP_EXPECTED_SHA256:\n"
        "    raise RuntimeError('Trainer payload memfd SHA-256 mismatch')\n"
        "_bootstrap_repo_root = _bootstrap_os.path.dirname(\n"
        "    _bootstrap_os.path.dirname(_BOOTSTRAP_CANONICAL_PATH)\n"
        ")\n"
        "if _bootstrap_repo_root not in _bootstrap_sys.path:\n"
        "    _bootstrap_sys.path.insert(0, _bootstrap_repo_root)\n"
        "_bootstrap_sys.argv[0] = _BOOTSTRAP_CANONICAL_PATH\n"
        "globals()['__file__'] = _BOOTSTRAP_CANONICAL_PATH\n"
        "globals()['__cached__'] = None\n"
        "globals()['__loader__'] = None\n"
        "globals()['__package__'] = None\n"
        "globals()['__spec__'] = None\n"
        "exec(\n"
        "    compile(\n"
        "        _bootstrap_payload,\n"
        "        _BOOTSTRAP_CANONICAL_PATH,\n"
        "        'exec',\n"
        "    ),\n"
        "    globals(),\n"
        "    globals(),\n"
        ")\n"
    ).encode("utf-8")


def build_effective_command(
    command: Sequence[str],
    *,
    root: Path,
    effective_root_path: str = EFFECTIVE_ROOT_PATH,
    effective_trainer_path: str = EFFECTIVE_TRAINER_PATH,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Build the FD-rooted argv and return its rewritten source-token indices."""

    if (
        len(command) < 2
        or not all(type(token) is str and token for token in command)
    ):
        raise ValueError("Source command must contain at least two string tokens")
    if not root.is_absolute() or root != Path(os.path.normpath(os.fspath(root))):
        raise ValueError("Effective-command root must be normalized and absolute")
    if (
        type(effective_root_path) is not str
        or effective_root_path != EFFECTIVE_ROOT_PATH
    ):
        raise ValueError(
            f"Effective root path must equal {EFFECTIVE_ROOT_PATH!r}"
        )
    if (
        type(effective_trainer_path) is not str
        or effective_trainer_path != EFFECTIVE_TRAINER_PATH
    ):
        raise ValueError(
            f"Effective trainer path must equal {EFFECTIVE_TRAINER_PATH!r}"
        )

    effective_command = list(command)
    repo_path_indices: list[int] = []
    for index, token in enumerate(command):
        if index == 1:
            continue
        token_path = Path(token)
        if not token_path.is_absolute():
            continue
        if token_path != Path(os.path.normpath(token)):
            raise ValueError(
                f"Repository command path token {index} is not normalized"
            )
        try:
            relative = token_path.relative_to(root)
        except ValueError:
            continue
        repo_path_indices.append(index)
        effective_command[index] = str(Path(effective_root_path) / relative)
    effective_command[1] = effective_trainer_path
    return tuple(effective_command), tuple(repo_path_indices)


def _validate_expected_transport(value: Any) -> tuple[dict[str, Any], Path]:
    if type(value) is not dict:
        raise ValueError("expected_transport must be an object")
    if set(value) != EXPECTED_TRANSPORT_KEYS:
        raise ValueError(
            "expected_transport must contain exactly tool, "
            "sandbox_permissions, login, tty, workdir, shell, and topology"
        )
    for key, expected in EXPECTED_TRANSPORT_CONSTANTS.items():
        actual = value.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"expected_transport.{key} must be {expected!r}")
    if type(value["workdir"]) is not str:
        raise ValueError("expected_transport.workdir must be a string")
    root = _normalized_absolute_path(
        value["workdir"],
        label="expected_transport.workdir",
        must_exist=True,
    )
    if not root.is_dir():
        raise ValueError("expected_transport.workdir must be a directory")
    if Path.cwd().resolve() != root:
        raise ValueError("Current working directory differs from locked workdir")
    return dict(value), root


def _validate_binding_files(
    value: Any,
    *,
    root: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    if type(value) is not dict:
        raise ValueError("bindings must be an object")
    missing = REQUIRED_BINDINGS - set(value)
    if missing:
        raise ValueError(f"bindings is missing required names: {sorted(missing)}")

    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for name, item in value.items():
        if type(name) is not str or not name:
            raise ValueError("Every binding name must be a non-empty string")
        if type(item) is not dict:
            raise ValueError(f"bindings.{name} must be an object")
        if set(item) != {"path", "sha256"}:
            raise ValueError(
                f"bindings.{name} must contain exactly path and sha256"
            )
        path = _path_under_root(
            item.get("path"),
            root=root,
            label=f"bindings.{name}.path",
            must_exist=True,
        )
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bindings.{name}.path must be a regular file")
        paths[name] = path
        digest = _require_sha256(
            item.get("sha256"),
            f"bindings.{name}.sha256",
        )
        if file_sha256(path) != digest:
            raise ValueError(f"Binding SHA-256 mismatch: {name}")
        digests[name] = digest
        if name == "launcher":
            expected_path = root / "tools" / "exec_preregistered_ppo_training.py"
            if path != expected_path or Path(__file__).resolve() != expected_path:
                raise ValueError("Training launcher path mismatch")
    if len(set(paths.values())) != len(paths):
        raise ValueError("Every named binding must refer to a distinct file")
    return paths, digests


def _validate_child_command(
    value: Any,
    *,
    branch_path: Path,
    trainer_path: Path,
    trainer_sha256: str,
    root: Path,
    branch_raw: bytes | None = None,
) -> tuple[
    tuple[str, ...],
    str,
    tuple[str, ...],
    str,
    int,
    str,
    int,
    str,
    str,
    int,
    str,
    tuple[int, ...],
]:
    if type(value) is not dict:
        raise ValueError("child_command must be an object")
    if set(value) != CHILD_COMMAND_KEYS:
        raise ValueError(
            "child_command must contain exactly source_binding, json_path, "
            "token_count, canonical_sha256, effective_trainer_fd, "
            "effective_trainer_path, sealed_trainer_payload_fd, "
            "sealed_trainer_payload_path, sealed_bootstrap_sha256, "
            "effective_root_fd, "
            "effective_root_path, effective_repo_path_token_indices, and "
            "effective_command_canonical_sha256"
        )
    if value["source_binding"] != "branch_preregistration":
        raise ValueError(
            "child_command.source_binding must be branch_preregistration"
        )
    if value["json_path"] != "binding.command":
        raise ValueError("child_command.json_path must be binding.command")
    token_count = value["token_count"]
    if type(token_count) is not int:
        raise ValueError("child_command.token_count must be an integer")
    if token_count < 2:
        raise ValueError("child_command.token_count must be at least two")
    expected_command_sha256 = _require_sha256(
        value["canonical_sha256"],
        "child_command.canonical_sha256",
    )
    effective_trainer_fd = value["effective_trainer_fd"]
    if (
        type(effective_trainer_fd) is not int
        or effective_trainer_fd != EFFECTIVE_TRAINER_FD
    ):
        raise ValueError(
            f"child_command.effective_trainer_fd must equal "
            f"{EFFECTIVE_TRAINER_FD}"
        )
    effective_trainer_path = value["effective_trainer_path"]
    if (
        type(effective_trainer_path) is not str
        or effective_trainer_path != EFFECTIVE_TRAINER_PATH
    ):
        raise ValueError(
            "child_command.effective_trainer_path must equal "
            f"{EFFECTIVE_TRAINER_PATH!r}"
        )
    sealed_trainer_payload_fd = value["sealed_trainer_payload_fd"]
    if (
        type(sealed_trainer_payload_fd) is not int
        or sealed_trainer_payload_fd != SEALED_TRAINER_PAYLOAD_FD
    ):
        raise ValueError(
            "child_command.sealed_trainer_payload_fd must equal "
            f"{SEALED_TRAINER_PAYLOAD_FD}"
        )
    sealed_trainer_payload_path = value["sealed_trainer_payload_path"]
    if (
        type(sealed_trainer_payload_path) is not str
        or sealed_trainer_payload_path != SEALED_TRAINER_PAYLOAD_PATH
    ):
        raise ValueError(
            "child_command.sealed_trainer_payload_path must equal "
            f"{SEALED_TRAINER_PAYLOAD_PATH!r}"
        )
    expected_bootstrap_sha256 = _require_sha256(
        value["sealed_bootstrap_sha256"],
        "child_command.sealed_bootstrap_sha256",
    )
    expected_effective_sha256 = _require_sha256(
        value["effective_command_canonical_sha256"],
        "child_command.effective_command_canonical_sha256",
    )
    effective_root_fd = value["effective_root_fd"]
    if type(effective_root_fd) is not int or effective_root_fd != EFFECTIVE_ROOT_FD:
        raise ValueError(
            f"child_command.effective_root_fd must equal {EFFECTIVE_ROOT_FD}"
        )
    effective_root_path = value["effective_root_path"]
    if (
        type(effective_root_path) is not str
        or effective_root_path != EFFECTIVE_ROOT_PATH
    ):
        raise ValueError(
            "child_command.effective_root_path must equal "
            f"{EFFECTIVE_ROOT_PATH!r}"
        )
    locked_repo_indices = value["effective_repo_path_token_indices"]
    if (
        type(locked_repo_indices) is not list
        or not all(type(index) is int for index in locked_repo_indices)
        or locked_repo_indices != sorted(set(locked_repo_indices))
    ):
        raise ValueError(
            "child_command.effective_repo_path_token_indices must be a "
            "sorted unique integer list"
        )

    if branch_raw is None:
        branch_raw = branch_path.read_bytes()
    branch = _parse_json_object(branch_raw, "branch_preregistration")
    if set(branch) != BRANCH_PREREGISTRATION_KEYS:
        raise ValueError(
            "branch_preregistration must contain exactly binding, "
            "binding_sha256, created_at, schema_version, and status"
        )
    if type(branch["created_at"]) is not str or not branch["created_at"]:
        raise ValueError("branch_preregistration.created_at must be a string")
    if (
        type(branch["schema_version"]) is not str
        or branch["schema_version"]
        != "ptcg-ppo-gold-branch-preregistration-v1"
    ):
        raise ValueError("branch_preregistration.schema_version mismatch")
    if (
        type(branch["status"]) is not str
        or branch["status"] != "preregistered_not_started"
    ):
        raise ValueError("branch_preregistration.status mismatch")
    binding = branch.get("binding")
    if type(binding) is not dict:
        raise ValueError("branch_preregistration.binding must be an object")
    missing_binding_keys = BRANCH_BINDING_COMMAND_KEYS - set(binding)
    if missing_binding_keys:
        raise ValueError(
            "branch binding is missing command-related keys: "
            f"{sorted(missing_binding_keys)}"
        )
    binding_sha256 = _require_sha256(
        branch["binding_sha256"],
        "branch_preregistration.binding_sha256",
    )
    if canonical_json_sha256(binding) != binding_sha256:
        raise ValueError("branch_preregistration.binding_sha256 mismatch")
    command = binding.get("command")
    if (
        type(command) is not list
        or not all(type(token) is str and token for token in command)
    ):
        raise ValueError("branch binding.command must be a non-empty string list")
    _reject_critical_flag_overrides(command)
    if len(command) != token_count:
        raise ValueError("binding.command token_count mismatch")
    command_sha256 = canonical_json_sha256(command)
    if command_sha256 != expected_command_sha256:
        raise ValueError("binding.command canonical SHA-256 mismatch")
    if binding.get("command_sha256") != command_sha256:
        raise ValueError("branch binding.command_sha256 mismatch")

    executable = Path(command[0])
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("binding.command Python executable must be an absolute file")
    if executable.resolve() != Path(sys.executable).resolve():
        raise ValueError("binding.command Python differs from launcher environment")
    if Path(command[1]) != trainer_path:
        raise ValueError("binding.command trainer differs from trainer binding")
    try:
        trainer_path.relative_to(root)
    except ValueError as error:
        raise ValueError("Trainer must remain under required workdir") from error
    if type(binding["seed"]) is not int:
        raise ValueError("branch binding.seed must be an integer")
    if type(binding["output_dir"]) is not str:
        raise ValueError("branch binding.output_dir must be a string")
    if binding["seed"] < 0 or str(binding["seed"]) != _single_command_value(
        command,
        "--seed",
    ):
        raise ValueError("branch binding.seed differs from binding.command")
    if binding["output_dir"] != _single_command_value(
        command,
        "--output-dir",
    ):
        raise ValueError("branch binding.output_dir differs from binding.command")
    if binding["source_train_script"] != str(trainer_path):
        raise ValueError("branch binding.source_train_script mismatch")
    if binding["source_train_script_sha256"] != trainer_sha256:
        raise ValueError("branch binding.source_train_script_sha256 mismatch")
    bootstrap_sha256 = hashlib.sha256(
        build_trainer_bootstrap(
            canonical_trainer_path=trainer_path,
            trainer_sha256=trainer_sha256,
            payload_fd=sealed_trainer_payload_fd,
        )
    ).hexdigest()
    if bootstrap_sha256 != expected_bootstrap_sha256:
        raise ValueError("Sealed trainer bootstrap SHA-256 mismatch")

    effective_command, computed_repo_indices = build_effective_command(
        command,
        root=root,
        effective_root_path=effective_root_path,
        effective_trainer_path=effective_trainer_path,
    )
    if tuple(locked_repo_indices) != computed_repo_indices:
        raise ValueError(
            "child_command.effective_repo_path_token_indices mismatch"
        )
    effective_command_sha256 = canonical_json_sha256(effective_command)
    if effective_command_sha256 != expected_effective_sha256:
        raise ValueError("Effective binding.command canonical SHA-256 mismatch")
    return (
        tuple(command),
        command_sha256,
        effective_command,
        effective_command_sha256,
        effective_trainer_fd,
        effective_trainer_path,
        sealed_trainer_payload_fd,
        sealed_trainer_payload_path,
        bootstrap_sha256,
        effective_root_fd,
        effective_root_path,
        computed_repo_indices,
    )


def _validate_attempt(
    value: Any,
    *,
    root: Path,
    command: Sequence[str],
) -> tuple[int, Path, Path, Path, Path, Path, Path]:
    if type(value) is not dict:
        raise ValueError("attempt must be an object")
    if set(value) != ATTEMPT_KEYS:
        raise ValueError(
            "attempt must contain exactly seed, attempts_authorized, "
            "output_dir, log, terminal_receipt, terminal_checkpoint, "
            "attempt_start_marker, unpublished_log_witness, "
            "expected_terminal_update, and absent_at_lock"
        )
    seed = value["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("attempt.seed must be a non-negative integer")
    if (
        type(value["attempts_authorized"]) is not int
        or value["attempts_authorized"] != 1
    ):
        raise ValueError("attempt.attempts_authorized must equal one")
    expected_update = value["expected_terminal_update"]
    if (
        type(expected_update) is not int
        or expected_update <= 0
    ):
        raise ValueError("attempt.expected_terminal_update must be positive")

    absent_at_lock = value["absent_at_lock"]
    if (
        type(absent_at_lock) is not dict
        or set(absent_at_lock) != LOCKED_ABSENCE_KEYS
        or any(absent_at_lock[key] is not True for key in LOCKED_ABSENCE_KEYS)
    ):
        raise ValueError(
            "attempt.absent_at_lock must assert all six target paths true"
        )

    output_dir = _path_under_root(
        value["output_dir"],
        root=root,
        label="attempt.output_dir",
        must_exist=False,
    )
    log = _path_under_root(
        value["log"],
        root=root,
        label="attempt.log",
        must_exist=False,
    )
    receipt = _path_under_root(
        value["terminal_receipt"],
        root=root,
        label="attempt.terminal_receipt",
        must_exist=False,
    )
    checkpoint = _path_under_root(
        value["terminal_checkpoint"],
        root=root,
        label="attempt.terminal_checkpoint",
        must_exist=False,
    )
    start_marker = _path_under_root(
        value["attempt_start_marker"],
        root=root,
        label="attempt.attempt_start_marker",
        must_exist=False,
    )
    unpublished_witness = _path_under_root(
        value["unpublished_log_witness"],
        root=root,
        label="attempt.unpublished_log_witness",
        must_exist=False,
    )
    targets = {
        output_dir,
        log,
        receipt,
        checkpoint,
        start_marker,
        unpublished_witness,
    }
    if len(targets) != 6:
        raise ValueError("All six attempt target paths must differ")
    expected_start_marker = canonical_attempt_start_marker(
        root=root,
        log=log,
        seed=seed,
    )
    expected_unpublished_witness = log.with_name(
        log.name + ".unpublished-witness"
    )
    if start_marker != expected_start_marker:
        raise ValueError("attempt.attempt_start_marker is not canonical")
    if unpublished_witness != expected_unpublished_witness:
        raise ValueError("attempt.unpublished_log_witness is not canonical")
    if start_marker.parent != root:
        raise ValueError("Attempt start marker must be directly under root")
    if unpublished_witness.parent != log.parent:
        raise ValueError("Unpublished witness must share the locked log parent")
    expected_checkpoint = (
        output_dir / "checkpoints" / f"update-{expected_update:04d}.pt"
    )
    if checkpoint != expected_checkpoint:
        raise ValueError("attempt.terminal_checkpoint is not canonical")
    if Path(_single_command_value(command, "--output-dir")) != output_dir:
        raise ValueError("binding.command --output-dir differs from locked output")
    if _single_command_value(command, "--seed") != str(seed):
        raise ValueError("binding.command --seed differs from attempt.seed")
    if _single_command_value(command, "--updates") != str(expected_update):
        raise ValueError(
            "binding.command --updates differs from expected_terminal_update"
        )

    for target, label in (
        (output_dir, "output directory"),
        (log, "stdout/stderr log"),
        (receipt, "terminal receipt"),
        (checkpoint, "terminal checkpoint"),
        (start_marker, "attempt start marker"),
        (unpublished_witness, "unpublished log witness"),
    ):
        if os.path.lexists(target):
            os.lstat(target)
            raise FileExistsError(f"Refusing existing {label}: {target}")
    for parent, label in (
        (output_dir.parent, "output directory"),
        (log.parent, "stdout/stderr log"),
        (receipt.parent, "terminal receipt"),
        (start_marker.parent, "attempt start marker"),
        (unpublished_witness.parent, "unpublished log witness"),
    ):
        if not parent.is_dir():
            raise FileNotFoundError(f"{label} parent does not exist: {parent}")
    return (
        seed,
        output_dir,
        log,
        receipt,
        checkpoint,
        start_marker,
        unpublished_witness,
    )


def _reject_existing_root_start_marker(
    attempt: Any,
    *,
    root: Path,
) -> None:
    """Reject a consumed attempt before validating mutable nested bindings."""

    if type(attempt) is not dict:
        return
    value = attempt.get("attempt_start_marker")
    if type(value) is not str or not value:
        return
    marker = Path(value)
    if (
        not marker.is_absolute()
        or marker != Path(os.path.normpath(value))
        or marker.parent != root
    ):
        return
    if os.path.lexists(marker):
        os.lstat(marker)
        raise FileExistsError(
            f"Refusing existing attempt start marker: {marker}"
        )


def read_and_validate_transport_lock(
    *,
    transport_lock: Path,
    expected_transport_lock_sha256: str,
) -> ValidatedLaunch:
    """Validate the complete lock without creating any training artifact."""

    lock_sha256 = _require_sha256(
        expected_transport_lock_sha256,
        "expected transport-lock SHA-256",
    )
    lock_path = _normalized_absolute_path(
        str(transport_lock),
        label="transport_lock",
        must_exist=True,
    )
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("transport_lock must be a regular symlink-free file")
    raw = lock_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != lock_sha256:
        raise ValueError("Transport-lock SHA-256 mismatch")
    protocol = _parse_json_object(raw, "transport_lock")
    if set(protocol) != TRANSPORT_LOCK_KEYS:
        raise ValueError(
            "transport_lock must contain exactly schema_version, status, "
            "bindings, child_command, attempt, and expected_transport"
        )
    if (
        type(protocol.get("schema_version")) is not str
        or protocol.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}")
    if (
        type(protocol.get("status")) is not str
        or protocol.get("status") != LOCKED_STATUS
    ):
        raise ValueError(f"status must equal {LOCKED_STATUS!r}")

    expected_transport, root = _validate_expected_transport(
        protocol.get("expected_transport")
    )
    trusted_root_stat = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(trusted_root_stat.st_mode):
        raise ValueError("Locked workdir must remain a directory")
    if not _same_root_identity(
        trusted_root_stat,
        os.stat(Path.cwd().resolve(), follow_symlinks=False),
    ):
        raise ValueError("Current directory inode differs from locked workdir")
    try:
        lock_path.relative_to(root)
    except ValueError as error:
        raise ValueError("transport_lock must remain under locked workdir") from error

    _reject_existing_root_start_marker(
        protocol.get("attempt"),
        root=root,
    )
    binding_paths, binding_sha256s = _validate_binding_files(
        protocol.get("bindings"),
        root=root,
    )
    launcher_sha256 = binding_sha256s["launcher"]
    (
        command,
        command_sha256,
        effective_command,
        effective_command_sha256,
        effective_trainer_fd,
        effective_trainer_path,
        sealed_trainer_payload_fd,
        sealed_trainer_payload_path,
        sealed_bootstrap_sha256,
        effective_root_fd,
        effective_root_path,
        effective_repo_path_token_indices,
    ) = _validate_child_command(
        protocol.get("child_command"),
        branch_path=binding_paths["branch_preregistration"],
        trainer_path=binding_paths["trainer"],
        trainer_sha256=binding_sha256s["trainer"],
        root=root,
    )
    (
        seed,
        output_dir,
        log,
        receipt,
        checkpoint,
        start_marker,
        unpublished_witness,
    ) = _validate_attempt(
        protocol.get("attempt"),
        root=root,
        command=command,
    )
    final_root_stat = os.stat(root, follow_symlinks=False)
    if _fstat_identity(final_root_stat) != _fstat_identity(trusted_root_stat):
        raise RuntimeError("Locked workdir changed during validation")

    # Keep the exact object used in the pre-exec header, not a reconstructed
    # approximation with potentially different fields.
    protocol["expected_transport"] = expected_transport
    return ValidatedLaunch(
        root=root,
        root_fstat=_fstat_record(trusted_root_stat),
        lock_path=lock_path,
        lock_sha256=lock_sha256,
        launcher_sha256=launcher_sha256,
        protocol=protocol,
        binding_paths=binding_paths,
        binding_sha256s=binding_sha256s,
        command=command,
        command_sha256=command_sha256,
        effective_command=effective_command,
        effective_command_sha256=effective_command_sha256,
        effective_root_fd=effective_root_fd,
        effective_root_path=effective_root_path,
        effective_repo_path_token_indices=effective_repo_path_token_indices,
        effective_trainer_fd=effective_trainer_fd,
        effective_trainer_path=effective_trainer_path,
        sealed_trainer_payload_fd=sealed_trainer_payload_fd,
        sealed_trainer_payload_path=sealed_trainer_payload_path,
        sealed_bootstrap_sha256=sealed_bootstrap_sha256,
        seed=seed,
        output_dir=output_dir,
        log=log,
        terminal_receipt=receipt,
        terminal_checkpoint=checkpoint,
        attempt_start_marker=start_marker,
        unpublished_log_witness=unpublished_witness,
    )


def _stat_matches_record(info: os.stat_result, record: dict[str, int]) -> bool:
    return _fstat_record(info) == record


def _fixed_fd_must_be_closed(fd: int, label: str) -> None:
    try:
        os.fstat(fd)
    except OSError as error:
        if error.errno == errno.EBADF:
            return
        raise
    raise RuntimeError(f"Preregistered {label} FD {fd} is already open")


def _held_directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("O_NOFOLLOW and O_DIRECTORY are required")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _assert_root_cwd_identity(
    launch: ValidatedLaunch,
    held_root: HeldRoot,
) -> None:
    held_stat = os.fstat(held_root.fd)
    current_stat = os.stat(launch.root, follow_symlinks=False)
    cwd_path = Path.cwd().resolve()
    cwd_stat = os.stat(cwd_path, follow_symlinks=False)
    if not stat.S_ISDIR(held_stat.st_mode):
        raise RuntimeError("Held workdir is no longer a directory")
    if not _stat_matches_record(held_stat, held_root.fstat):
        raise RuntimeError("Held workdir metadata changed after validation")
    if not _same_root_identity(held_stat, current_stat):
        raise RuntimeError("Locked workdir path no longer names held workdir")
    if not _same_root_identity(held_stat, cwd_stat):
        raise RuntimeError("Current directory no longer names held workdir")
    if cwd_path != launch.root:
        raise RuntimeError("Current working directory path changed")


def _refresh_held_root_after_authorized_marker(
    launch: ValidatedLaunch,
    held_root: HeldRoot,
) -> HeldRoot:
    """Accept only the root metadata change caused by creating the marker."""

    held_stat = os.fstat(held_root.fd)
    current_stat = os.stat(launch.root, follow_symlinks=False)
    cwd_path = Path.cwd().resolve()
    cwd_stat = os.stat(cwd_path, follow_symlinks=False)
    if not stat.S_ISDIR(held_stat.st_mode):
        raise RuntimeError("Held workdir is no longer a directory")
    if not _same_root_identity(held_stat, current_stat):
        raise RuntimeError("Locked workdir path no longer names held workdir")
    if not _same_root_identity(held_stat, cwd_stat):
        raise RuntimeError("Current directory no longer names held workdir")
    if cwd_path != launch.root:
        raise RuntimeError("Current working directory path changed")
    return HeldRoot(
        fd=held_root.fd,
        fstat=_fstat_record(held_stat),
    )


def open_held_root(launch: ValidatedLaunch) -> HeldRoot:
    target_fd = launch.effective_root_fd
    _fixed_fd_must_be_closed(target_fd, "root")
    source_fd = os.open(launch.root, _held_directory_flags())
    duplicated = False
    try:
        opened = os.fstat(source_fd)
        if not _stat_matches_record(opened, launch.root_fstat):
            raise RuntimeError("Workdir changed before its root FD was held")
        if source_fd != target_fd:
            os.dup2(source_fd, target_fd, inheritable=True)
            duplicated = True
        else:
            os.set_inheritable(target_fd, True)
        held = HeldRoot(
            fd=target_fd,
            fstat=_fstat_record(os.fstat(target_fd)),
        )
        if not os.get_inheritable(target_fd):
            raise RuntimeError("Held root FD is not inheritable")
        _assert_root_cwd_identity(launch, held)
        return held
    except BaseException:
        if duplicated or source_fd == target_fd:
            os.close(target_fd)
        raise
    finally:
        if source_fd != target_fd:
            os.close(source_fd)


def _close_held_root(held_root: HeldRoot) -> None:
    try:
        os.close(held_root.fd)
    except OSError as error:
        if error.errno != errno.EBADF:
            raise


def _relative_parts(path: Path, root: Path) -> tuple[str, ...]:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path is outside held workdir: {path}") from error
    if not relative.parts:
        raise ValueError("Held-root file path must be below workdir")
    return tuple(relative.parts)


def _open_held_directory_chain(
    held_root: HeldRoot,
    components: Sequence[str],
) -> int:
    descriptor = os.dup(held_root.fd)
    try:
        for component in components:
            child = os.open(
                component,
                _held_directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_held_directory_is_canonical(
    *,
    held_root: HeldRoot,
    relative_components: Sequence[str],
    held_directory_fd: int,
    label: str,
) -> None:
    """Rewalk from the held root and compare the canonical directory inode."""

    canonical_fd = _open_held_directory_chain(
        held_root,
        relative_components,
    )
    try:
        held_stat = os.fstat(held_directory_fd)
        canonical_stat = os.fstat(canonical_fd)
        if (
            not stat.S_ISDIR(held_stat.st_mode)
            or not stat.S_ISDIR(canonical_stat.st_mode)
        ):
            raise RuntimeError(f"{label} is no longer a directory")
        if not _same_root_identity(held_stat, canonical_stat):
            raise RuntimeError(
                f"Held {label} no longer names the canonical directory"
            )
    finally:
        os.close(canonical_fd)


def _read_held_repo_file(
    launch: ValidatedLaunch,
    held_root: HeldRoot,
    path: Path,
) -> tuple[bytes, os.stat_result]:
    parts = _relative_parts(path, launch.root)
    parent = _open_held_directory_chain(held_root, parts[:-1])
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(parts[-1], flags, dir_fd=parent)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"Held binding is not regular: {path}")
        raw = _read_all_fd(descriptor)
        after = os.fstat(descriptor)
        if _fstat_identity(before) != _fstat_identity(after):
            raise RuntimeError(f"Held binding changed while read: {path}")
        if len(raw) != after.st_size:
            raise RuntimeError(f"Held binding size changed while read: {path}")
        return raw, after
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _read_all_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def _held_leaf_stat(
    launch: ValidatedLaunch,
    held_root: HeldRoot,
    path: Path,
) -> os.stat_result | None:
    parts = _relative_parts(path, launch.root)
    try:
        parent = _open_held_directory_chain(held_root, parts[:-1])
    except FileNotFoundError:
        return None
    try:
        try:
            return os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return None
    finally:
        os.close(parent)


def _revalidate_before_claim(
    launch: ValidatedLaunch,
    held_root: HeldRoot,
) -> None:
    """Repeat every mutable filesystem check immediately before O_EXCL."""

    _assert_root_cwd_identity(launch, held_root)
    lock_raw, _ = _read_held_repo_file(
        launch,
        held_root,
        launch.lock_path,
    )
    if hashlib.sha256(lock_raw).hexdigest() != launch.lock_sha256:
        raise RuntimeError("Transport lock changed before log claim")
    branch_raw: bytes | None = None
    for name, path in launch.binding_paths.items():
        raw, _ = _read_held_repo_file(launch, held_root, path)
        if hashlib.sha256(raw).hexdigest() != launch.binding_sha256s[name]:
            raise RuntimeError(f"Binding hash changed before log claim: {name}")
        if name == "branch_preregistration":
            branch_raw = raw
    if branch_raw is None:
        raise RuntimeError("Held branch preregistration was not revalidated")

    (
        command,
        command_sha256,
        effective_command,
        effective_command_sha256,
        effective_trainer_fd,
        effective_trainer_path,
        sealed_trainer_payload_fd,
        sealed_trainer_payload_path,
        sealed_bootstrap_sha256,
        effective_root_fd,
        effective_root_path,
        effective_repo_path_token_indices,
    ) = _validate_child_command(
        launch.protocol["child_command"],
        branch_path=launch.binding_paths["branch_preregistration"],
        trainer_path=launch.binding_paths["trainer"],
        trainer_sha256=launch.binding_sha256s["trainer"],
        root=launch.root,
        branch_raw=branch_raw,
    )
    if command != launch.command or command_sha256 != launch.command_sha256:
        raise RuntimeError("Branch command changed before log claim")
    if (
        effective_command != launch.effective_command
        or effective_command_sha256 != launch.effective_command_sha256
        or effective_trainer_fd != launch.effective_trainer_fd
        or effective_trainer_path != launch.effective_trainer_path
        or sealed_trainer_payload_fd != launch.sealed_trainer_payload_fd
        or sealed_trainer_payload_path != launch.sealed_trainer_payload_path
        or sealed_bootstrap_sha256 != launch.sealed_bootstrap_sha256
        or effective_root_fd != launch.effective_root_fd
        or effective_root_path != launch.effective_root_path
        or effective_repo_path_token_indices
        != launch.effective_repo_path_token_indices
    ):
        raise RuntimeError("Effective branch command changed before log claim")

    for target, label in (
        (launch.output_dir, "output directory"),
        (launch.log, "stdout/stderr log"),
        (launch.terminal_receipt, "terminal receipt"),
        (launch.terminal_checkpoint, "terminal checkpoint"),
        (launch.attempt_start_marker, "attempt start marker"),
        (
            launch.unpublished_log_witness,
            "unpublished log witness",
        ),
    ):
        if _held_leaf_stat(launch, held_root, target) is not None:
            raise FileExistsError(f"Refusing existing {label}: {target}")
    _assert_root_cwd_identity(launch, held_root)


def _sha256_open_fd(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _reserve_fixed_fds(target_fds: Sequence[int]) -> None:
    for target_fd in target_fds:
        _fixed_fd_must_be_closed(target_fd, "sealed trainer transport")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    source_fd = os.open(os.devnull, flags)
    try:
        for target_fd in target_fds:
            if source_fd == target_fd:
                os.set_inheritable(target_fd, False)
            else:
                os.dup2(source_fd, target_fd, inheritable=False)
    except BaseException:
        for target_fd in target_fds:
            try:
                os.close(target_fd)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
        raise
    finally:
        if source_fd not in target_fds:
            os.close(source_fd)


def _close_fixed_fds(target_fds: Sequence[int]) -> None:
    first_error: OSError | None = None
    for target_fd in target_fds:
        try:
            os.close(target_fd)
        except OSError as error:
            if error.errno != errno.EBADF and first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _create_sealed_memfd(
    *,
    name: str,
    payload: bytes,
) -> tuple[int, str, dict[str, int], int]:
    required_seals = _required_memfd_seals()
    flags = MFD_ALLOW_SEALING | MFD_CLOEXEC
    descriptor = _libc_memfd_create(name, flags)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        before_seal = os.fstat(descriptor)
        if not stat.S_ISREG(before_seal.st_mode):
            raise RuntimeError(f"{name} memfd is not a regular file")
        if before_seal.st_size != len(payload):
            raise RuntimeError(f"{name} memfd size differs from payload")
        if _sha256_open_fd(descriptor) != hashlib.sha256(payload).hexdigest():
            raise RuntimeError(f"{name} memfd SHA-256 differs before sealing")
        fcntl.fcntl(
            descriptor,
            F_ADD_SEALS,
            required_seals,
        )
        actual_seals = int(
            fcntl.fcntl(descriptor, F_GET_SEALS)
        )
        if actual_seals != required_seals:
            raise RuntimeError(f"{name} memfd seal set mismatch")
        after_seal = os.fstat(descriptor)
        digest = _sha256_open_fd(descriptor)
        final_stat = os.fstat(descriptor)
        if _fstat_identity(after_seal) != _fstat_identity(final_stat):
            raise RuntimeError(f"{name} memfd changed after sealing")
        if digest != hashlib.sha256(payload).hexdigest():
            raise RuntimeError(f"{name} memfd SHA-256 differs after sealing")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return (
            descriptor,
            digest,
            _fstat_record(final_stat),
            actual_seals,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_sealed_fd(
    *,
    descriptor: int,
    expected_sha256: str,
    expected_fstat: dict[str, int],
    expected_seals: int,
    label: str,
) -> None:
    required_seals = _required_memfd_seals()
    if expected_seals != required_seals:
        raise RuntimeError(f"{label} expected seal set mismatch")
    if not os.get_inheritable(descriptor):
        raise RuntimeError(f"{label} FD is not inheritable")
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{label} FD is not regular")
    if not _stat_matches_record(before, expected_fstat):
        raise RuntimeError(f"{label} fstat changed before exec")
    actual_seals = int(
        fcntl.fcntl(descriptor, F_GET_SEALS)
    )
    if actual_seals != expected_seals:
        raise RuntimeError(f"{label} seals changed before exec")
    digest = _sha256_open_fd(descriptor)
    after = os.fstat(descriptor)
    if not _stat_matches_record(after, expected_fstat):
        raise RuntimeError(f"{label} changed while rehashed before exec")
    if digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 changed before exec")
    os.lseek(descriptor, 0, os.SEEK_SET)


def _revalidate_sealed_trainer(held: HeldTrainer) -> None:
    _revalidate_sealed_fd(
        descriptor=held.fd,
        expected_sha256=held.bootstrap_sha256,
        expected_fstat=held.bootstrap_fstat,
        expected_seals=held.bootstrap_seals,
        label="Sealed trainer bootstrap",
    )
    _revalidate_sealed_fd(
        descriptor=held.payload_fd,
        expected_sha256=held.payload_sha256,
        expected_fstat=held.payload_fstat,
        expected_seals=held.payload_seals,
        label="Sealed trainer payload",
    )
    if held.payload_sha256 != held.source_sha256:
        raise RuntimeError("Sealed trainer payload differs from verified source")


def open_held_trainer(
    launch: ValidatedLaunch,
    held_root: HeldRoot | None = None,
) -> HeldTrainer:
    """Copy the verified trainer into sealed memfds and pin their fixed FDs."""

    bootstrap_fd = launch.effective_trainer_fd
    payload_fd = launch.sealed_trainer_payload_fd
    target_fds = (bootstrap_fd, payload_fd)
    if bootstrap_fd == payload_fd:
        raise RuntimeError("Bootstrap and trainer payload FDs must differ")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required to hold the trainer safely")
    _required_memfd_seals()
    _reserve_fixed_fds(target_fds)

    owned_root: HeldRoot | None = None
    parent_fd: int | None = None
    source_fd: int | None = None
    raw_bootstrap_fd: int | None = None
    raw_payload_fd: int | None = None
    keep_fixed_fds = False
    try:
        if held_root is None:
            owned_root = open_held_root(launch)
            held_root = owned_root
        parts = _relative_parts(launch.binding_paths["trainer"], launch.root)
        parent_fd = _open_held_directory_chain(held_root, parts[:-1])
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        source_fd = os.open(parts[-1], flags, dir_fd=parent_fd)
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("Held trainer is not a regular file")
        source_bytes = _read_all_fd(source_fd)
        after = os.fstat(source_fd)
        if _fstat_identity(before) != _fstat_identity(after):
            raise RuntimeError("Trainer changed while its bytes were copied")
        if len(source_bytes) != after.st_size:
            raise RuntimeError("Trainer size changed while its bytes were copied")
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if source_sha256 != launch.binding_sha256s["trainer"]:
            raise RuntimeError("Held trainer SHA-256 differs from transport lock")

        bootstrap_bytes = build_trainer_bootstrap(
            canonical_trainer_path=launch.binding_paths["trainer"],
            trainer_sha256=source_sha256,
            payload_fd=payload_fd,
        )
        bootstrap_sha256 = hashlib.sha256(bootstrap_bytes).hexdigest()
        if bootstrap_sha256 != launch.sealed_bootstrap_sha256:
            raise RuntimeError(
                "Sealed trainer bootstrap differs from transport lock"
            )

        (
            raw_payload_fd,
            payload_sha256,
            raw_payload_fstat,
            payload_seals,
        ) = _create_sealed_memfd(
            name="ptcg-ppo-trainer-payload",
            payload=source_bytes,
        )
        (
            raw_bootstrap_fd,
            sealed_bootstrap_sha256,
            raw_bootstrap_fstat,
            bootstrap_seals,
        ) = _create_sealed_memfd(
            name="ptcg-ppo-trainer-bootstrap",
            payload=bootstrap_bytes,
        )
        os.dup2(raw_payload_fd, payload_fd, inheritable=True)
        os.dup2(raw_bootstrap_fd, bootstrap_fd, inheritable=True)
        payload_stat = os.fstat(payload_fd)
        bootstrap_stat = os.fstat(bootstrap_fd)
        if _fstat_identity(payload_stat) != _fstat_identity(
            os.fstat(raw_payload_fd)
        ):
            raise RuntimeError("Sealed trainer payload changed during fixed-FD dup")
        if _fstat_identity(bootstrap_stat) != _fstat_identity(
            os.fstat(raw_bootstrap_fd)
        ):
            raise RuntimeError(
                "Sealed trainer bootstrap changed during fixed-FD dup"
            )
        if not _stat_matches_record(payload_stat, raw_payload_fstat):
            raise RuntimeError("Sealed trainer payload fstat changed during dup")
        if not _stat_matches_record(bootstrap_stat, raw_bootstrap_fstat):
            raise RuntimeError("Sealed trainer bootstrap fstat changed during dup")

        held = HeldTrainer(
            fd=bootstrap_fd,
            payload_fd=payload_fd,
            source_sha256=source_sha256,
            source_fstat=_fstat_record(after),
            bootstrap_sha256=sealed_bootstrap_sha256,
            bootstrap_fstat=_fstat_record(bootstrap_stat),
            bootstrap_seals=bootstrap_seals,
            payload_sha256=payload_sha256,
            payload_fstat=_fstat_record(payload_stat),
            payload_seals=payload_seals,
        )
        _revalidate_sealed_trainer(held)
        keep_fixed_fds = True
        return held
    finally:
        if raw_bootstrap_fd is not None:
            os.close(raw_bootstrap_fd)
        if raw_payload_fd is not None:
            os.close(raw_payload_fd)
        if source_fd is not None:
            os.close(source_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        if owned_root is not None:
            _close_held_root(owned_root)
        if not keep_fixed_fds:
            _close_fixed_fds(target_fds)


def _close_held_trainer(held: HeldTrainer) -> None:
    _close_fixed_fds((held.fd, held.payload_fd))


def build_pre_exec_header(
    launch: ValidatedLaunch,
    held_root: HeldRoot,
    held_trainer: HeldTrainer,
) -> dict[str, Any]:
    return {
        "branch_preregistration_sha256": launch.binding_sha256s[
            "branch_preregistration"
        ],
        "attempt_start_marker": str(launch.attempt_start_marker),
        "child_command_sha256": launch.command_sha256,
        "comprehensive_preregistration_sha256": launch.binding_sha256s[
            "comprehensive_preregistration"
        ],
        "cwd": str(launch.root),
        "effective_command_sha256": launch.effective_command_sha256,
        "effective_repo_path_token_indices": list(
            launch.effective_repo_path_token_indices
        ),
        "effective_root_fd": held_root.fd,
        "effective_root_fstat": held_root.fstat,
        "effective_root_path": launch.effective_root_path,
        "effective_trainer_fd": held_trainer.fd,
        "effective_trainer_fstat": held_trainer.bootstrap_fstat,
        "effective_trainer_path": launch.effective_trainer_path,
        "event": "preregistered_ppo_training_pre_exec_checks_passed",
        "expected_transport": launch.protocol["expected_transport"],
        "held_trainer_fstat": held_trainer.payload_fstat,
        "held_trainer_sha256": held_trainer.payload_sha256,
        "launcher_sha256": launch.launcher_sha256,
        "runner_preregistration_sha256": launch.binding_sha256s[
            "runner_preregistration"
        ],
        "sealed_bootstrap_fstat": held_trainer.bootstrap_fstat,
        "sealed_bootstrap_seals": held_trainer.bootstrap_seals,
        "sealed_bootstrap_sha256": held_trainer.bootstrap_sha256,
        "sealed_trainer_payload_fd": held_trainer.payload_fd,
        "sealed_trainer_payload_fstat": held_trainer.payload_fstat,
        "sealed_trainer_payload_path": launch.sealed_trainer_payload_path,
        "sealed_trainer_payload_seals": held_trainer.payload_seals,
        "sealed_trainer_payload_sha256": held_trainer.payload_sha256,
        "seed": launch.seed,
        "source_command_sha256": launch.command_sha256,
        "source_protocol_sha256": launch.binding_sha256s["source_protocol"],
        "source_trainer_fstat": held_trainer.source_fstat,
        "source_trainer_sha256": held_trainer.source_sha256,
        "trainer_sha256": launch.binding_sha256s["trainer"],
        "topology": "supervised_fork_exec_v1",
        "transport_lock_sha256": launch.lock_sha256,
        "unpublished_log_witness": str(
            launch.unpublished_log_witness
        ),
    }


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("Short write while copying locked bytes")
        view = view[written:]


def _open_anonymous_log(log_parent_fd: int) -> int:
    _require_supervisor_platform()
    flags = os.O_TMPFILE | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(".", flags, 0o644, dir_fd=log_parent_fd)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 0:
        os.close(descriptor)
        raise RuntimeError("O_TMPFILE log is not an anonymous regular inode")
    os.set_inheritable(descriptor, False)
    return descriptor


def _read_stable_fd(fd: int, *, label: str) -> tuple[bytes, os.stat_result]:
    before = os.fstat(fd)
    raw = _read_all_fd(fd)
    after = os.fstat(fd)
    if _fstat_identity(before) != _fstat_identity(after):
        raise RuntimeError(f"{label} changed while read")
    if len(raw) != after.st_size:
        raise RuntimeError(f"{label} size changed while read")
    return raw, after


def _verify_named_fd(
    *,
    source_fd: int,
    parent_fd: int,
    basename: str,
    expected_raw: bytes,
    label: str,
) -> os.stat_result:
    source_raw, source_stat = _read_stable_fd(source_fd, label=label)
    if source_raw != expected_raw:
        raise RuntimeError(f"{label} bytes changed")
    try:
        path_stat = os.stat(
            basename,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise RuntimeError(f"{label} path is unavailable") from error
    if not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(f"{label} path is not regular")
    if not _same_root_identity(source_stat, path_stat):
        raise RuntimeError(f"{label} path inode differs from held FD")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    reopened = os.open(basename, flags, dir_fd=parent_fd)
    try:
        reopened_raw, reopened_stat = _read_stable_fd(
            reopened,
            label=f"reopened {label}",
        )
        if reopened_raw != expected_raw:
            raise RuntimeError(f"Reopened {label} bytes changed")
        if not _same_root_identity(source_stat, reopened_stat):
            raise RuntimeError(f"Reopened {label} inode differs from held FD")
    finally:
        os.close(reopened)
    final_path_stat = os.stat(
        basename,
        dir_fd=parent_fd,
        follow_symlinks=False,
    )
    if _fstat_identity(path_stat) != _fstat_identity(final_path_stat):
        raise RuntimeError(f"{label} path changed during verification")
    return source_stat


def _claim_attempt_start_marker(
    *,
    launch: ValidatedLaunch,
    marker_parent_fd: int,
    marker_name: str,
    anonymous_header_sha256: str,
) -> tuple[int, bytes]:
    payload = canonical_json_bytes(
        {
            "anonymous_log_header_sha256": anonymous_header_sha256,
            "event": "preregistered_ppo_training_attempt_consumed",
            "log": str(launch.log),
            "schema_version": SCHEMA_VERSION,
            "seed": launch.seed,
            "topology": "supervised_fork_exec_v1",
            "transport_lock_sha256": launch.lock_sha256,
            "unpublished_log_witness": str(
                launch.unpublished_log_witness
            ),
        }
    ) + b"\n"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(
        marker_name,
        flags,
        0o600,
        dir_fd=marker_parent_fd,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fsync(marker_parent_fd)
        _verify_named_fd(
            source_fd=descriptor,
            parent_fd=marker_parent_fd,
            basename=marker_name,
            expected_raw=payload,
            label="attempt start marker",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, payload
    except BaseException:
        os.close(descriptor)
        raise


def _append_log_record(log_fd: int, record: dict[str, Any]) -> bytes:
    info = os.fstat(log_fd)
    os.lseek(log_fd, 0, os.SEEK_END)
    if info.st_size and os.pread(log_fd, 1, info.st_size - 1) != b"\n":
        _write_all(log_fd, b"\n")
    payload = canonical_json_bytes(record) + b"\n"
    _write_all(log_fd, payload)
    return payload


def _wait_for_child(child_pid: int) -> tuple[int, dict[str, Any]]:
    while True:
        try:
            waited_pid, status = os.waitpid(child_pid, 0)
            break
        except InterruptedError:
            continue
    if waited_pid != child_pid:
        raise RuntimeError("waitpid returned a different child")
    if os.WIFEXITED(status):
        exit_code = int(os.WEXITSTATUS(status))
        return exit_code, {
            "child_exit_code": exit_code,
            "child_signal": None,
            "wait_status": int(status),
        }
    if os.WIFSIGNALED(status):
        child_signal = int(os.WTERMSIG(status))
        return 128 + child_signal, {
            "child_exit_code": None,
            "child_signal": child_signal,
            "wait_status": int(status),
        }
    raise RuntimeError(f"Child reached unsupported wait status {status}")


def _child_exec(
    *,
    launch: ValidatedLaunch,
    held_root: HeldRoot,
    held_trainer: HeldTrainer,
    log_fd: int,
    previous_signal_mask: set[signal.Signals],
) -> None:
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_DFL)
        signal.pthread_sigmask(
            signal.SIG_SETMASK,
            previous_signal_mask,
        )
        _assert_root_cwd_identity(launch, held_root)
        _revalidate_sealed_trainer(held_trainer)
        os.lseek(log_fd, 0, os.SEEK_END)
        os.dup2(log_fd, 1, inheritable=True)
        os.dup2(log_fd, 2, inheritable=True)
        command = list(launch.effective_command)
        os.execv(command[0], command)
        raise RuntimeError("os.execv unexpectedly returned")
    except BaseException as error:
        try:
            _append_log_record(
                log_fd,
                {
                    "error": f"{type(error).__name__}: {error}",
                    "event": "supervisor_child_exec_failed",
                    "exit_code": SUPERVISOR_CHILD_EXEC_FAILURE,
                },
            )
            os.fsync(log_fd)
        finally:
            os._exit(SUPERVISOR_CHILD_EXEC_FAILURE)


def _publish_log_fd(
    *,
    log_fd: int,
    log_parent_fd: int,
    target_name: str,
    expected_raw: bytes,
    label: str,
) -> str:
    digest = hashlib.sha256(expected_raw).hexdigest()
    _link_fd_at_empty_path(
        source_fd=log_fd,
        target_parent_fd=log_parent_fd,
        target_name=target_name,
    )
    os.fsync(log_parent_fd)
    _verify_named_fd(
        source_fd=log_fd,
        parent_fd=log_parent_fd,
        basename=target_name,
        expected_raw=expected_raw,
        label=label,
    )
    return digest


def _terminate_child_best_effort(child_pid: int) -> None:
    try:
        os.kill(child_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        _wait_for_child(child_pid)
    except (ChildProcessError, ProcessLookupError):
        pass


def _install_supervisor_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise SupervisorInterrupted(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    return previous


def _restore_supervisor_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def claim_log_and_exec(launch: ValidatedLaunch) -> int:
    """Run one sealed child and publish its anonymous log only at terminal."""

    _require_supervisor_platform()
    held_root = open_held_root(launch)
    try:
        previous_signal_handlers = _install_supervisor_signal_handlers()
    except BaseException:
        _close_held_root(held_root)
        raise
    held_trainer: HeldTrainer | None = None
    log_parent_fd: int | None = None
    log_fd: int | None = None
    marker_fd: int | None = None
    marker_raw: bytes | None = None
    child_pid: int | None = None
    child_reaped = False
    terminal_recorded = False
    canonical_published = False
    try:
        _revalidate_before_claim(launch, held_root)
        held_trainer = open_held_trainer(launch, held_root)
        _assert_root_cwd_identity(launch, held_root)

        log_parts = _relative_parts(launch.log, launch.root)
        marker_parts = _relative_parts(
            launch.attempt_start_marker,
            launch.root,
        )
        witness_parts = _relative_parts(
            launch.unpublished_log_witness,
            launch.root,
        )
        if len(marker_parts) != 1:
            raise RuntimeError(
                "Attempt start marker is not directly under held root"
            )
        if log_parts[:-1] != witness_parts[:-1]:
            raise RuntimeError("Log and witness parents differ")
        log_parent_fd = _open_held_directory_chain(
            held_root,
            log_parts[:-1],
        )
        _assert_held_directory_is_canonical(
            held_root=held_root,
            relative_components=log_parts[:-1],
            held_directory_fd=log_parent_fd,
            label="log parent",
        )
        for name, label in (
            (log_parts[-1], "stdout/stderr log"),
            (witness_parts[-1], "unpublished log witness"),
        ):
            try:
                os.stat(name, dir_fd=log_parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(f"Refusing existing {label}: {name}")
        try:
            os.stat(
                marker_parts[-1],
                dir_fd=held_root.fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                "Refusing existing attempt start marker: "
                f"{marker_parts[-1]}"
            )

        header = (
            canonical_json_bytes(
                build_pre_exec_header(launch, held_root, held_trainer)
            )
            + b"\n"
        )
        _assert_held_directory_is_canonical(
            held_root=held_root,
            relative_components=log_parts[:-1],
            held_directory_fd=log_parent_fd,
            label="log parent",
        )
        marker_fd, marker_raw = _claim_attempt_start_marker(
            launch=launch,
            marker_parent_fd=held_root.fd,
            marker_name=marker_parts[-1],
            anonymous_header_sha256=hashlib.sha256(header).hexdigest(),
        )
        held_root = _refresh_held_root_after_authorized_marker(
            launch,
            held_root,
        )
        _assert_held_directory_is_canonical(
            held_root=held_root,
            relative_components=log_parts[:-1],
            held_directory_fd=log_parent_fd,
            label="log parent",
        )
        _assert_root_cwd_identity(launch, held_root)

        _assert_held_directory_is_canonical(
            held_root=held_root,
            relative_components=log_parts[:-1],
            held_directory_fd=log_parent_fd,
            label="log parent",
        )
        log_fd = _open_anonymous_log(log_parent_fd)
        _write_all(log_fd, header)
        os.fsync(log_fd)
        anonymous_raw, anonymous_stat = _read_stable_fd(
            log_fd,
            label="anonymous training log header",
        )
        if anonymous_raw != header or anonymous_stat.st_nlink != 0:
            raise RuntimeError("Anonymous training log header verification failed")
        _assert_root_cwd_identity(launch, held_root)
        _revalidate_sealed_trainer(held_trainer)
        _assert_held_directory_is_canonical(
            held_root=held_root,
            relative_components=log_parts[:-1],
            held_directory_fd=log_parent_fd,
            label="log parent",
        )

        managed_signals = {
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGHUP,
        }
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            managed_signals,
        )
        try:
            child_pid = os.fork()
        except BaseException as error:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            _append_log_record(
                log_fd,
                {
                    "error": f"{type(error).__name__}: {error}",
                    "event": "supervisor_fork_failed",
                },
            )
            terminal_recorded = True
            raise
        if child_pid == 0:
            _child_exec(
                launch=launch,
                held_root=held_root,
                held_trainer=held_trainer,
                log_fd=log_fd,
                previous_signal_mask=previous_mask,
            )
            os._exit(SUPERVISOR_CHILD_EXEC_FAILURE)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

        return_code, terminal_fields = _wait_for_child(child_pid)
        child_reaped = True
        _append_log_record(
            log_fd,
            {
                **terminal_fields,
                "event": "supervisor_child_terminal",
                "mapped_return_code": return_code,
                "seed": launch.seed,
                "topology": "supervised_fork_exec_v1",
            },
        )
        terminal_recorded = True
        os.fsync(log_fd)
        final_raw, final_stat = _read_stable_fd(
            log_fd,
            label="terminal anonymous training log",
        )
        if final_stat.st_nlink != 0:
            raise RuntimeError("Anonymous training log was linked before publish")
        _verify_named_fd(
            source_fd=marker_fd,
            parent_fd=held_root.fd,
            basename=marker_parts[-1],
            expected_raw=marker_raw,
            label="attempt start marker",
        )
        _assert_held_directory_is_canonical(
            held_root=held_root,
            relative_components=log_parts[:-1],
            held_directory_fd=log_parent_fd,
            label="log parent",
        )
        _assert_root_cwd_identity(launch, held_root)
        _publish_log_fd(
            log_fd=log_fd,
            log_parent_fd=log_parent_fd,
            target_name=log_parts[-1],
            expected_raw=final_raw,
            label="canonical terminal training log",
        )
        canonical_published = True
        _assert_held_directory_is_canonical(
            held_root=held_root,
            relative_components=log_parts[:-1],
            held_directory_fd=log_parent_fd,
            label="log parent",
        )
        _assert_root_cwd_identity(launch, held_root)
        _assert_held_directory_is_canonical(
            held_root=held_root,
            relative_components=log_parts[:-1],
            held_directory_fd=log_parent_fd,
            label="log parent",
        )
        return return_code
    except BaseException as error:
        if child_pid not in (None, 0) and not child_reaped:
            _terminate_child_best_effort(child_pid)
            child_reaped = True
        if marker_fd is not None and log_fd is not None:
            try:
                if not terminal_recorded:
                    _append_log_record(
                        log_fd,
                        {
                            "error": f"{type(error).__name__}: {error}",
                            "event": "supervisor_launcher_failed",
                            "seed": launch.seed,
                        },
                    )
                else:
                    _append_log_record(
                        log_fd,
                        {
                            "error": f"{type(error).__name__}: {error}",
                            "event": "supervisor_publish_failed",
                            "seed": launch.seed,
                        },
                    )
                os.fsync(log_fd)
                failure_raw, _ = _read_stable_fd(
                    log_fd,
                    label="unpublished failure log",
                )
                if not canonical_published and log_parent_fd is not None:
                    witness_name = _relative_parts(
                        launch.unpublished_log_witness,
                        launch.root,
                    )[-1]
                    try:
                        _assert_held_directory_is_canonical(
                            held_root=held_root,
                            relative_components=log_parts[:-1],
                            held_directory_fd=log_parent_fd,
                            label="log parent",
                        )
                        _publish_log_fd(
                            log_fd=log_fd,
                            log_parent_fd=log_parent_fd,
                            target_name=witness_name,
                            expected_raw=failure_raw,
                            label="unpublished failure log witness",
                        )
                    except BaseException:
                        # The O_EXCL start marker remains the durable consumed
                        # attempt witness even if an attacker preclaims both
                        # publication names.
                        pass
            except BaseException:
                pass
        raise
    finally:
        try:
            for descriptor in (marker_fd, log_fd, log_parent_fd):
                if descriptor is not None:
                    os.close(descriptor)
            if held_trainer is not None:
                _close_held_trainer(held_trainer)
            _close_held_root(held_root)
        finally:
            _restore_supervisor_signal_handlers(previous_signal_handlers)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-exec one hash-bound PPO training command."
    )
    parser.add_argument("--transport-lock", type=Path, required=True)
    parser.add_argument("--expected-transport-lock-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    launch = read_and_validate_transport_lock(
        transport_lock=args.transport_lock,
        expected_transport_lock_sha256=args.expected_transport_lock_sha256,
    )
    return_code = claim_log_and_exec(launch)
    print(
        canonical_json_bytes(
            {
                "event": "supervisor_terminal_log_published",
                "log": str(launch.log),
                "return_code": return_code,
                "seed": launch.seed,
                "topology": "supervised_fork_exec_v1",
            }
        ).decode("utf-8"),
        flush=True,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
