#!/usr/bin/env python3
"""Run two frozen behavior panels in one CUDA-owning Python process.

The execution preregistration is the only source of paths, hashes, commands,
and runtime settings.  CUDA initialization and exact checkpoint loading happen
before the formal-attempt marker and before any dataset object is constructed.
After the marker is durably created, any exception is terminal and produces a
single exclusive failure record; a later invocation refuses to restart.

This module deliberately does not use a shell, subprocess, or exec boundary.
The model loaded during preflight is the same object passed to both panel
evaluations.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent

# Establish the repository trust anchor immediately after deriving the loaded
# runner's canonical path, before importing any mutable local dependency.
_STARTUP_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | os.O_NOFOLLOW
    | getattr(os, "O_CLOEXEC", 0)
)
_STARTUP_REPO_PARENT_FD = os.open(
    REPO_ROOT.parent,
    _STARTUP_DIRECTORY_FLAGS,
)
try:
    _STARTUP_REPO_ROOT_FD = os.open(
        REPO_ROOT.name,
        _STARTUP_DIRECTORY_FLAGS,
        dir_fd=_STARTUP_REPO_PARENT_FD,
    )
except BaseException:
    os.close(_STARTUP_REPO_PARENT_FD)
    raise
_STARTUP_REPO_PARENT_STAT = os.fstat(_STARTUP_REPO_PARENT_FD)
_STARTUP_REPO_ROOT_STAT = os.fstat(_STARTUP_REPO_ROOT_FD)
_STARTUP_REPO_LINK_STAT = os.stat(
    REPO_ROOT.name,
    dir_fd=_STARTUP_REPO_PARENT_FD,
    follow_symlinks=False,
)
if (
    not stat.S_ISDIR(_STARTUP_REPO_PARENT_STAT.st_mode)
    or not stat.S_ISDIR(_STARTUP_REPO_ROOT_STAT.st_mode)
    or stat.S_ISLNK(_STARTUP_REPO_LINK_STAT.st_mode)
    or (
        int(_STARTUP_REPO_ROOT_STAT.st_dev),
        int(_STARTUP_REPO_ROOT_STAT.st_ino),
    )
    != (
        int(_STARTUP_REPO_LINK_STAT.st_dev),
        int(_STARTUP_REPO_LINK_STAT.st_ino),
    )
):
    os.close(_STARTUP_REPO_ROOT_FD)
    os.close(_STARTUP_REPO_PARENT_FD)
    raise RuntimeError("could not establish the startup repository trust anchor")

if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import evaluate_policy_bc as policy_eval  # noqa: E402


RUNNER_SCHEMA = "ptcg-policy-behavior-panels-inprocess-v1"
EXECUTION_PREREGISTRATION_SCHEMA = "ptcg-policy-behavior-execution-v1"
CERTIFICATE_SCHEMA = "ptcg-policy-behavior-cuda-preflight-v1"
MARKER_SCHEMA = "ptcg-policy-behavior-formal-attempt-v1"
FAILURE_SCHEMA = "ptcg-policy-behavior-terminal-failure-v1"
PREFLIGHT_FAILURE_SCHEMA = "ptcg-policy-behavior-preflight-failure-v1"
SUCCESS_SCHEMA = "ptcg-policy-behavior-terminal-success-v1"
FROZEN_DEVICE = "cuda"
FROZEN_BATCH_SIZE = 256
FROZEN_REQUESTED_WORKERS = 8
FROZEN_EFFECTIVE_WORKERS = 0
FROZEN_PREDICTION_ORDER = "policy"
FROZEN_OUTPUT_PREDICTION_ORDER = "policy_greedy"
FROZEN_PROGRESS_INTERVAL = 0
FROZEN_SPLIT = "valid"
FROZEN_SPLIT_MODE = "archive"
FROZEN_SPLIT_SEED = 20260723
BEHAVIOR_GATE_OUTPUT_KEYS = {
    "rows": ("metrics.rows", "exact"),
    "set_exact": ("metrics.set_exact_correct", "minimum"),
    "hybrid_order_exact": (
        "metrics.hybrid_order_exact_correct",
        "minimum",
    ),
    "ordered_exact": ("metrics.ordered_exact_correct", "minimum"),
    "value": ("metrics.value_correct", "minimum"),
    "count": ("metrics.count_correct", "minimum"),
    "top1": ("metrics.top1_correct", "minimum"),
    "context34_rows": (
        'metrics.by_context["34"].rows',
        "exact",
    ),
    "context34_hybrid_order_exact": (
        'metrics.by_context["34"].hybrid_order_exact_correct',
        "minimum",
    ),
    "context34_ordered_exact": (
        'metrics.by_context["34"].ordered_exact_correct',
        "minimum",
    ),
}

SINGLE_VALUE_FLAGS = {
    "--checkpoint",
    "--data",
    "--split",
    "--split-mode",
    "--split-seed",
    "--batch-size",
    "--workers",
    "--prediction-order",
    "--device",
    "--json-output",
    "--progress-interval",
    "--max-rows",
}
REPEATED_VALUE_FLAGS = {"--deck-hash", "--team-name"}
BOOLEAN_FLAGS = {"--compact"}
RELEVANT_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "PYTORCH_NVML_BASED_CUDA_CHECK",
    "CUDA_DEVICE_ORDER",
    "LD_LIBRARY_PATH",
)


class ProtocolError(RuntimeError):
    """The bound execution request is malformed or fails integrity checks."""


class RestartRefusedError(ProtocolError):
    """An execution artifact already exists, so no-retry forbids a restart."""


class ConcurrentExecutionError(ProtocolError):
    """Another process already owns this preregistration's execution lock."""


class PanelExecutionAggregateError(RuntimeError):
    """One or more panels failed after every declared panel was attempted."""

    def __init__(self, failures: Sequence[Mapping[str, Any]]) -> None:
        self.failures = tuple(dict(item) for item in failures)
        super().__init__(
            f"{len(self.failures)} behavior panel(s) failed after all "
            "declared panels were attempted"
        )


class BehaviorGateFailure(RuntimeError):
    """All outputs exist, but one or more of the twenty frozen gates failed."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        failing = list(report.get("failing_gates", []))
        super().__init__(
            f"{len(failing)} of 20 frozen behavior gates failed"
        )


@dataclass(frozen=True)
class CodeDependency:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class PanelSpec:
    order: int
    name: str
    data_panel: str
    data_path: Path
    data_sha256: str
    split: str
    split_mode: str
    split_seed: int
    batch_size: int
    requested_workers: int
    effective_workers: int
    prediction_order: str
    device: str
    output: Path
    compact: bool
    progress_interval: int
    max_rows: int | None
    deck_hashes: tuple[str, ...]
    team_names: tuple[str, ...]
    command: tuple[str, ...]
    command_sha256: str


@dataclass(frozen=True)
class ExecutionSpec:
    preregistration_path: Path
    preregistration_sha256: str
    runner_path: Path
    runner_sha256: str
    evaluator_path: Path
    evaluator_sha256: str
    evaluator_dependencies: tuple[CodeDependency, ...]
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_update: int
    source_protocol_path: Path
    source_protocol_sha256: str
    training_integrity_decision_path: Path
    training_integrity_decision_sha256: str
    metric_mapping_sha256: str
    metric_mapping_and_gates: dict[str, Any]
    device: str
    expected_output_prediction_order: str
    panels: tuple[PanelSpec, PanelSpec]
    preflight_evidence_directory: Path
    marker: Path
    success_result: Path
    failure_result: Path


@dataclass(frozen=True)
class PreflightState:
    model: Any
    model_config: dict[str, Any]
    checkpoint_payload: dict[str, Any]
    checkpoint_kind: str
    device: Any
    pid: int
    process_start_ticks: str
    environment_sha256: str
    cuda_device_index: int
    cuda_device_name: str
    cuda_capability: tuple[int, int]
    model_devices: tuple[str, ...]
    checkpoint_held_file: dict[str, Any]


@dataclass
class HeldFile:
    """An immutable-by-name view of one preregistered source inode."""

    original_path: Path
    expected_sha256: str
    descriptor: int
    proc_path: Path
    device: int
    inode: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int
    link_count: int
    pre_sha256: str
    lock: str
    post_fstat: dict[str, int] | None = None

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass
class LockedPreregistration:
    """A raw-hash-verified preregistration held under an exclusive flock."""

    path: Path
    sha256: str
    raw: bytes
    descriptor: int

    def close(self) -> None:
        if self.descriptor >= 0:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(frozen=True)
class TrustedRootAnchor:
    """Startup-held identity for a repository root and its stable parent."""

    path: Path
    parent_path: Path
    name: str
    parent_descriptor: int
    root_descriptor: int
    parent_device: int
    parent_inode: int
    root_device: int
    root_inode: int


@dataclass
class ProcessExecutionLock:
    """Authoritative repo-root lock plus one digest sidecar audit identity."""

    preregistration_sha256: str
    root_anchor: TrustedRootAnchor
    repo_parent_descriptor: int
    repo_root_path: Path
    repo_root_descriptor: int
    path: Path
    descriptor: int

    def close(self) -> None:
        if self.descriptor >= 0:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = -1
        if self.repo_root_descriptor >= 0:
            fcntl.flock(self.repo_root_descriptor, fcntl.LOCK_UN)
            os.close(self.repo_root_descriptor)
            self.repo_root_descriptor = -1
        if self.repo_parent_descriptor >= 0:
            fcntl.flock(self.repo_parent_descriptor, fcntl.LOCK_UN)
            os.close(self.repo_parent_descriptor)
            self.repo_parent_descriptor = -1


@dataclass
class HeldArtifact:
    """An exclusively created artifact retained by descriptor until terminal."""

    path: Path
    descriptor: int
    sha256: str
    serialized: bytes
    device: int
    inode: int
    size: int
    mode: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def open_trusted_root_anchor(path: Path) -> TrustedRootAnchor:
    """Open and bind a root through its symlink-free parent directory."""
    root_path = Path(os.path.normpath(os.fspath(Path(path).absolute())))
    parent_path = root_path.parent
    parent_descriptor = os.open(parent_path, _STARTUP_DIRECTORY_FLAGS)
    try:
        root_descriptor = os.open(
            root_path.name,
            _STARTUP_DIRECTORY_FLAGS,
            dir_fd=parent_descriptor,
        )
    except BaseException:
        os.close(parent_descriptor)
        raise
    try:
        parent_stat = os.fstat(parent_descriptor)
        root_stat = os.fstat(root_descriptor)
        link_stat = os.stat(
            root_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or not stat.S_ISDIR(root_stat.st_mode)
            or stat.S_ISLNK(link_stat.st_mode)
            or (
                int(root_stat.st_dev),
                int(root_stat.st_ino),
            )
            != (
                int(link_stat.st_dev),
                int(link_stat.st_ino),
            )
        ):
            raise ProtocolError("trusted root path and opened inode differ")
        return TrustedRootAnchor(
            path=root_path,
            parent_path=parent_path,
            name=root_path.name,
            parent_descriptor=parent_descriptor,
            root_descriptor=root_descriptor,
            parent_device=int(parent_stat.st_dev),
            parent_inode=int(parent_stat.st_ino),
            root_device=int(root_stat.st_dev),
            root_inode=int(root_stat.st_ino),
        )
    except BaseException:
        os.close(root_descriptor)
        os.close(parent_descriptor)
        raise


TRUSTED_REPO_ROOT_ANCHOR = TrustedRootAnchor(
    path=REPO_ROOT,
    parent_path=REPO_ROOT.parent,
    name=REPO_ROOT.name,
    parent_descriptor=_STARTUP_REPO_PARENT_FD,
    root_descriptor=_STARTUP_REPO_ROOT_FD,
    parent_device=int(_STARTUP_REPO_PARENT_STAT.st_dev),
    parent_inode=int(_STARTUP_REPO_PARENT_STAT.st_ino),
    root_device=int(_STARTUP_REPO_ROOT_STAT.st_dev),
    root_inode=int(_STARTUP_REPO_ROOT_STAT.st_ino),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_bound_path(value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _lexical_artifact_path(
    value: object,
    *,
    anchor: TrustedRootAnchor = TRUSTED_REPO_ROOT_ANCHOR,
) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = anchor.path / path
    path = Path(os.path.normpath(os.fspath(path)))
    try:
        path.relative_to(anchor.path)
    except ValueError as error:
        raise ProtocolError(
            "execution artifact path must remain under trusted repository root"
        ) from error
    return path


def resolve_bound_artifact_path(value: object) -> Path:
    """Normalize lexically without resolving any artifact path component."""
    path = _lexical_artifact_path(value)
    if not path.name:
        raise ProtocolError("execution artifact path must name a file")
    return path


def resolve_bound_artifact_directory(value: object) -> Path:
    path = _lexical_artifact_path(value)
    if path == TRUSTED_REPO_ROOT_ANCHOR.path:
        raise ProtocolError("artifact directory must not be repository root")
    return path


def _artifact_relative(
    path: Path,
    *,
    anchor: TrustedRootAnchor = TRUSTED_REPO_ROOT_ANCHOR,
) -> Path:
    normalized = _lexical_artifact_path(path, anchor=anchor)
    return normalized.relative_to(anchor.path)


def _open_artifact_parent_dirfd(
    path: Path,
    *,
    create: bool,
    anchor: TrustedRootAnchor = TRUSTED_REPO_ROOT_ANCHOR,
) -> tuple[int, str]:
    """Open every parent component with openat and O_NOFOLLOW."""
    relative = _artifact_relative(path, anchor=anchor)
    if not relative.name:
        raise ProtocolError("artifact path must name a leaf")
    current = os.open(
        ".",
        _STARTUP_DIRECTORY_FLAGS,
        dir_fd=anchor.root_descriptor,
    )
    try:
        root_stat = os.fstat(current)
        if _root_identity(root_stat) != (
            anchor.root_device,
            anchor.root_inode,
        ):
            raise ProtocolError("artifact root descriptor changed identity")
        for component in relative.parent.parts:
            try:
                next_descriptor = os.open(
                    component,
                    _STARTUP_DIRECTORY_FLAGS,
                    dir_fd=current,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700, dir_fd=current)
                    os.fsync(current)
                except FileExistsError:
                    pass
                next_descriptor = os.open(
                    component,
                    _STARTUP_DIRECTORY_FLAGS,
                    dir_fd=current,
                )
            except OSError as error:
                raise ProtocolError(
                    "artifact ancestor is missing, non-directory, or symlink: "
                    f"{component}"
                ) from error
            next_stat = os.fstat(next_descriptor)
            if not stat.S_ISDIR(next_stat.st_mode):
                os.close(next_descriptor)
                raise ProtocolError(
                    f"artifact ancestor is not a directory: {component}"
                )
            os.close(current)
            current = next_descriptor
        return current, relative.name
    except BaseException:
        os.close(current)
        raise


def _artifact_lstat(
    path: Path,
    *,
    anchor: TrustedRootAnchor = TRUSTED_REPO_ROOT_ANCHOR,
) -> os.stat_result | None:
    try:
        parent_fd, leaf = _open_artifact_parent_dirfd(
            path,
            create=False,
            anchor=anchor,
        )
    except FileNotFoundError:
        return None
    try:
        try:
            return os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
    finally:
        os.close(parent_fd)


def artifact_path_lexists(path: Path) -> bool:
    """Return true for every directory entry, including dangling symlinks."""
    return _artifact_lstat(path) is not None


def reject_artifact_symlink(path: Path, label: str) -> None:
    """Reject an existing final-component symlink without dereferencing it."""
    path_stat = _artifact_lstat(path)
    if path_stat is None:
        return
    if stat.S_ISLNK(path_stat.st_mode):
        raise RestartRefusedError(
            f"{label} is an existing symlink; refusing execution: {path}"
        )


def existing_artifact_paths(paths: Sequence[Path]) -> list[Path]:
    """Find execution artifacts without overlooking dangling symlinks."""
    existing: list[Path] = []
    for path in paths:
        if artifact_path_lexists(path):
            reject_artifact_symlink(path, "execution artifact")
            existing.append(path)
    return existing


def bound_path_text(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def require_sha256(value: object, label: str) -> str:
    text = str(value).lower()
    is_hex = all(character in "0123456789abcdef" for character in text)
    if len(text) != 64 or not is_hex:
        raise ProtocolError(f"{label} must be a lowercase SHA256 hex digest")
    return text


def _root_identity(stat_result: os.stat_result) -> tuple[int, int]:
    return int(stat_result.st_dev), int(stat_result.st_ino)


def validate_trusted_root_anchor(anchor: TrustedRootAnchor) -> None:
    """Prove held root/parent FDs and the parent entry retain startup identity."""
    parent_stat = os.fstat(anchor.parent_descriptor)
    root_stat = os.fstat(anchor.root_descriptor)
    try:
        link_stat = os.stat(
            anchor.name,
            dir_fd=anchor.parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ProtocolError(
            "trusted repository root is no longer linked at its startup path"
        ) from error
    if (
        _root_identity(parent_stat)
        != (anchor.parent_device, anchor.parent_inode)
        or _root_identity(root_stat)
        != (anchor.root_device, anchor.root_inode)
        or _root_identity(link_stat)
        != (anchor.root_device, anchor.root_inode)
        or not stat.S_ISDIR(root_stat.st_mode)
        or not stat.S_ISDIR(link_stat.st_mode)
        or stat.S_ISLNK(link_stat.st_mode)
    ):
        raise ProtocolError(
            "trusted repository root identity or startup path changed"
        )


def validate_process_root_lock(lock: ProcessExecutionLock) -> None:
    """Revalidate both the startup anchor and execution-held lock descriptors."""
    validate_trusted_root_anchor(lock.root_anchor)
    parent_stat = os.fstat(lock.repo_parent_descriptor)
    root_stat = os.fstat(lock.repo_root_descriptor)
    if (
        _root_identity(parent_stat)
        != (
            lock.root_anchor.parent_device,
            lock.root_anchor.parent_inode,
        )
        or _root_identity(root_stat)
        != (
            lock.root_anchor.root_device,
            lock.root_anchor.root_inode,
        )
    ):
        raise ProtocolError("execution root-lock descriptors changed identity")


def acquire_process_execution_lock(
    preregistration_path: Path,
    expected_preregistration_sha256: str,
    *,
    root_anchor: TrustedRootAnchor = TRUSTED_REPO_ROOT_ANCHOR,
) -> ProcessExecutionLock:
    """Lock the startup parent/root identities before touching sidecars."""
    expected_hash = require_sha256(
        expected_preregistration_sha256,
        "expected preregistration SHA256",
    )
    repo_flags = _STARTUP_DIRECTORY_FLAGS
    try:
        repo_parent_descriptor = os.open(
            ".",
            repo_flags,
            dir_fd=root_anchor.parent_descriptor,
        )
    except OSError as error:
        raise ProtocolError(
            "could not open the authoritative repo-parent execution lock"
        ) from error
    repo_root_descriptor = -1
    try:
        try:
            fcntl.flock(
                repo_parent_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise ConcurrentExecutionError(
                "another process owns the authoritative repo-parent behavior "
                "execution lock"
            ) from error
        validate_trusted_root_anchor(root_anchor)
        repo_root_descriptor = os.open(
            ".",
            repo_flags,
            dir_fd=root_anchor.root_descriptor,
        )
        try:
            fcntl.flock(
                repo_root_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            os.close(repo_root_descriptor)
            repo_root_descriptor = -1
            raise ConcurrentExecutionError(
                "another process owns the authoritative repo-root behavior "
                "execution lock"
            ) from error
        if (
            _root_identity(os.fstat(repo_parent_descriptor))
            != (root_anchor.parent_device, root_anchor.parent_inode)
            or _root_identity(os.fstat(repo_root_descriptor))
            != (root_anchor.root_device, root_anchor.root_inode)
        ):
            os.close(repo_root_descriptor)
            repo_root_descriptor = -1
            raise ProtocolError(
                "authoritative execution lock descriptors differ from the "
                "startup root anchor"
            )

        # Parent/root flocks above are authoritative. Only now may the
        # digest-named sidecar be created as a non-authoritative audit identity.
        preregistration_path_value = Path(preregistration_path)
        preregistration_absolute = Path(
            os.path.normpath(
                os.fspath(
                    preregistration_path_value
                    if preregistration_path_value.is_absolute()
                    else Path.cwd() / preregistration_path_value
                )
            )
        )
        try:
            preregistration_absolute.relative_to(root_anchor.path)
        except ValueError as error:
            os.close(repo_root_descriptor)
            repo_root_descriptor = -1
            raise ProtocolError(
                "preregistration must remain under the trusted repository root"
            ) from error
        preregistration_parent = preregistration_absolute.parent
        lock_directory = (
            preregistration_parent
            / ".ptcg-policy-behavior-execution-locks"
        )
        lock_path = lock_directory / f"sha256-{expected_hash}.lock"
        lock_parent_fd, lock_leaf = _open_artifact_parent_dirfd(
            lock_path,
            create=True,
            anchor=root_anchor,
        )
        sidecar_flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                lock_leaf,
                sidecar_flags,
                0o600,
                dir_fd=lock_parent_fd,
            )
        except OSError as error:
            os.close(lock_parent_fd)
            raise ProtocolError(
                "could not open the digest-stable behavior lock sidecar"
            ) from error
        try:
            lock_stat = os.fstat(descriptor)
            path_stat = os.stat(
                lock_leaf,
                dir_fd=lock_parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(lock_stat.st_mode)
                or int(lock_stat.st_nlink) != 1
                or (
                    int(lock_stat.st_dev),
                    int(lock_stat.st_ino),
                )
                != (
                    int(path_stat.st_dev),
                    int(path_stat.st_ino),
                )
            ):
                raise ProtocolError(
                    "digest-stable behavior lock sidecar is not a private "
                    "stable regular inode"
                )
            canonical_stat = _artifact_lstat(
                lock_path,
                anchor=root_anchor,
            )
            if (
                canonical_stat is None
                or _root_identity(canonical_stat)
                != _root_identity(lock_stat)
            ):
                raise ProtocolError(
                    "digest-stable behavior lock sidecar canonical path changed"
                )
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            os.close(lock_parent_fd)
            return ProcessExecutionLock(
                preregistration_sha256=expected_hash,
                root_anchor=root_anchor,
                repo_parent_descriptor=repo_parent_descriptor,
                repo_root_path=root_anchor.path,
                repo_root_descriptor=repo_root_descriptor,
                path=lock_path,
                descriptor=descriptor,
            )
        except BaseException:
            os.close(lock_parent_fd)
            os.close(descriptor)
            raise
    except BaseException:
        if repo_root_descriptor >= 0:
            try:
                fcntl.flock(repo_root_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(repo_root_descriptor)
        try:
            fcntl.flock(repo_parent_descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(repo_parent_descriptor)
        raise


def verify_file_hash(path: Path, expected: str, label: str) -> None:
    try:
        path_stat = os.lstat(path)
    except FileNotFoundError:
        raise ProtocolError(f"{label} does not exist: {path}")
    except OSError as error:
        raise ProtocolError(f"{label} cannot be inspected: {path}") from error
    if stat.S_ISLNK(path_stat.st_mode):
        raise ProtocolError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(path_stat.st_mode):
        raise ProtocolError(f"{label} is not a regular file: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ProtocolError(
            f"{label} SHA256 mismatch: expected {expected}, observed {observed}"
        )


def parse_bound_command(
    command_value: object,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    if not isinstance(command_value, list) or not all(
        isinstance(token, str) for token in command_value
    ):
        raise ProtocolError("panel command must be a JSON string array")
    command = tuple(command_value)
    if len(command) < 2:
        raise ProtocolError("panel command is too short")

    parsed: dict[str, Any] = {
        "--deck-hash": [],
        "--team-name": [],
        "--compact": False,
    }
    index = 2
    while index < len(command):
        flag = command[index]
        if flag in BOOLEAN_FLAGS:
            if parsed.get(flag) is True:
                raise ProtocolError(f"duplicate command flag: {flag}")
            parsed[flag] = True
            index += 1
            continue
        if flag not in SINGLE_VALUE_FLAGS and flag not in REPEATED_VALUE_FLAGS:
            raise ProtocolError(f"unsupported command flag: {flag}")
        if index + 1 >= len(command):
            raise ProtocolError(f"command flag has no value: {flag}")
        value = command[index + 1]
        if flag in REPEATED_VALUE_FLAGS:
            parsed[flag].append(value)
        else:
            if flag in parsed:
                raise ProtocolError(f"duplicate command flag: {flag}")
            parsed[flag] = value
        index += 2
    return command, parsed


def command_required(parsed: Mapping[str, Any], flag: str) -> str:
    if flag not in parsed:
        raise ProtocolError(f"panel command lacks required flag {flag}")
    return str(parsed[flag])


def parse_positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"{label} must be an integer, not a boolean")
    if isinstance(value, float) and not value.is_integer():
        raise ProtocolError(f"{label} must be an integer")
    if not isinstance(value, (int, str, float)):
        raise ProtocolError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"{label} must be an integer") from error
    if isinstance(value, str) and str(parsed) != value.strip():
        raise ProtocolError(f"{label} must use canonical integer syntax")
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        raise ProtocolError(f"{label} must be >= {minimum}")
    return parsed


def command_path_matches(
    parsed_value: object,
    expected: Path,
    label: str,
    *,
    artifact: bool = False,
) -> None:
    observed = (
        resolve_bound_artifact_path(parsed_value)
        if artifact
        else resolve_bound_path(parsed_value)
    )
    if observed != expected:
        raise ProtocolError(f"{label} path differs from the bound preregistration")


def read_verified_preregistration(
    preregistration_path: Path,
    expected_preregistration_sha256: str,
) -> tuple[Path, str, bytes]:
    """Read the preregistration once and establish its raw-byte trust boundary."""
    path = preregistration_path.resolve()
    expected_hash = require_sha256(
        expected_preregistration_sha256,
        "expected preregistration SHA256",
    )
    if not path.is_file():
        raise ProtocolError(f"preregistration does not exist: {path}")
    raw = path.read_bytes()
    observed_hash = hashlib.sha256(raw).hexdigest()
    if observed_hash != expected_hash:
        raise ProtocolError(
            "preregistration SHA256 mismatch: "
            f"expected {expected_hash}, observed {observed_hash}"
        )
    return path, observed_hash, raw


def open_locked_verified_preregistration(
    preregistration_path: Path,
    expected_preregistration_sha256: str,
) -> LockedPreregistration:
    """Pin, exclusively lock, read, and hash the formal execution request."""
    path = preregistration_path.resolve()
    expected_hash = require_sha256(
        expected_preregistration_sha256,
        "expected preregistration SHA256",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProtocolError(f"preregistration does not exist: {path}") from error
    try:
        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise ConcurrentExecutionError(
                "another process owns the formal behavior execution lock"
            ) from error
        before = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        os.lseek(descriptor, 0, os.SEEK_SET)
        after = os.fstat(descriptor)
        if _stable_stat_fields(before) != _stable_stat_fields(after):
            raise ProtocolError(
                "preregistration inode changed while its bytes were read"
            )
        try:
            path_stat = path.stat()
        except OSError as error:
            raise ProtocolError(
                "preregistration path disappeared after it was locked"
            ) from error
        if (
            int(path_stat.st_dev),
            int(path_stat.st_ino),
        ) != (
            int(after.st_dev),
            int(after.st_ino),
        ):
            raise ProtocolError(
                "preregistration path was replaced after its inode was locked"
            )
        observed_hash = hashlib.sha256(raw).hexdigest()
        if observed_hash != expected_hash:
            raise ProtocolError(
                "preregistration SHA256 mismatch: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        return LockedPreregistration(
            path=path,
            sha256=observed_hash,
            raw=raw,
            descriptor=descriptor,
        )
    except BaseException:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)
        raise


def trusted_raw_execution_artifact_paths(raw: bytes) -> tuple[Path, ...]:
    """Extract only restart-guard paths from a raw-hash-verified preregistration."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(value, dict):
        return ()
    raw_runner = value.get("inprocess_runner")
    raw_evaluations = value.get("ordered_evaluations")
    candidates: list[object] = []
    if isinstance(raw_runner, dict):
        candidates.extend(
            raw_runner.get(key)
            for key in (
                "formal_attempt_marker",
                "success_result",
                "failure_result",
            )
        )
    if isinstance(raw_evaluations, list):
        candidates.extend(
            panel.get("output")
            for panel in raw_evaluations
            if isinstance(panel, dict)
        )
    paths: list[Path] = []
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            paths.append(resolve_bound_artifact_path(candidate))
    return tuple(paths)


def assert_trusted_raw_restart_artifacts_absent(raw: bytes) -> None:
    existing = existing_artifact_paths(
        trusted_raw_execution_artifact_paths(raw)
    )
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise RestartRefusedError(
            "no-retry execution artifacts already exist; refusing restart "
            "before external binding validation: "
            f"{rendered}"
        )


def read_bound_json_reference(
    reference_value: object,
    label: str,
) -> tuple[Path, str, dict[str, Any]]:
    reference = require_object(reference_value, label)
    path = resolve_bound_path(reference.get("path"))
    expected_sha256 = require_sha256(
        reference.get("sha256"),
        f"{label}.sha256",
    )
    if not path.is_file():
        raise ProtocolError(f"{label} does not exist: {path}")
    raw = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ProtocolError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"{label} is not valid JSON") from error
    return path, observed_sha256, require_object(value, label)


def decision_terminal_checkpoint(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    content = require_object(
        decision.get("training_content_integrity"),
        "training integrity decision.training_content_integrity",
    )
    return require_object(
        content.get("terminal_checkpoint"),
        "training integrity decision terminal checkpoint",
    )


def validate_training_authorization(
    preregistration: Mapping[str, Any],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_update: int,
    source_protocol_path: Path,
    source_protocol_sha256: str,
) -> tuple[Path, str]:
    authorization = require_object(
        preregistration.get("authorization"),
        "authorization",
    )
    reference_value = authorization.get("training_integrity_decision")
    reference = require_object(
        reference_value,
        "authorization.training_integrity_decision",
    )
    path, observed_sha256, decision = read_bound_json_reference(
        reference,
        "authorization.training_integrity_decision",
    )
    if reference.get("status") != "passed_and_behavior_authorized":
        raise ProtocolError(
            "training-integrity authorization status is not behavior-authorized"
        )
    if decision.get("status") != "passed_and_behavior_authorized":
        raise ProtocolError(
            "authoritative training-integrity decision status is not "
            "behavior-authorized"
        )
    if reference.get("status") != decision.get("status"):
        raise ProtocolError(
            "training-integrity status differs from the bound decision"
        )
    if reference.get("all_required_pass") is not True:
        raise ProtocolError(
            "training-integrity authorization does not attest all gates"
        )
    if reference.get("behavior_execution_authorized") is not True:
        raise ProtocolError(
            "behavior execution is not authorized by the execution binding"
        )
    if decision.get("pass") is not True:
        raise ProtocolError("authoritative training-integrity decision did not pass")
    decision_source = require_object(
        decision.get("authoritative_source_protocol"),
        "training integrity decision.authoritative_source_protocol",
    )
    decision_source_path = resolve_bound_path(decision_source.get("path"))
    decision_source_sha256 = require_sha256(
        decision_source.get("sha256"),
        "training integrity decision source protocol SHA256",
    )
    if (
        decision_source_path != source_protocol_path
        or decision_source_sha256 != source_protocol_sha256
    ):
        raise ProtocolError(
            "training-integrity decision authorizes a different source protocol"
        )
    content = require_object(
        decision.get("training_content_integrity"),
        "training integrity decision.training_content_integrity",
    )
    if content.get("pass") is not True:
        raise ProtocolError("authoritative training content integrity did not pass")
    transport = require_object(
        decision.get("transport_receipt_integrity"),
        "training integrity decision.transport_receipt_integrity",
    )
    if transport.get("pass") is not True:
        raise ProtocolError("authoritative training transport receipt did not pass")
    independent = require_object(
        content.get("independent_read_only_recomputations"),
        "independent read-only recomputations",
    )
    if int(independent.get("count", 0)) < 2:
        raise ProtocolError(
            "fewer than two independent training-integrity recomputations"
        )
    if int(independent.get("numeric_content_pass_count", 0)) < 2:
        raise ProtocolError(
            "fewer than two independent numeric-content passes"
        )
    decision_action = require_object(
        decision.get("decision"),
        "training integrity decision.decision",
    )
    if decision_action.get("behavior_execution_authorized") is not True:
        raise ProtocolError(
            "authoritative training-integrity decision blocks behavior execution"
        )
    terminal = decision_terminal_checkpoint(decision)
    terminal_path = resolve_bound_path(terminal.get("path"))
    terminal_sha256 = require_sha256(
        terminal.get("sha256"),
        "training integrity terminal checkpoint SHA256",
    )
    if terminal_path != checkpoint or terminal_sha256 != checkpoint_sha256:
        raise ProtocolError(
            "candidate checkpoint differs from the authoritative "
            "training-integrity terminal checkpoint"
        )
    if int(terminal.get("update", -1)) != checkpoint_update:
        raise ProtocolError(
            "candidate update differs from the training-integrity decision"
        )
    return path, observed_sha256


def expected_gate_mapping(
    source_panels: Sequence[object],
) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "metrics_object": "output.metrics",
        "context34_object": 'output.metrics.by_context["34"]',
    }
    for raw_panel in source_panels:
        panel = require_object(raw_panel, "source behavior panel")
        data_panel = str(panel.get("data_panel"))
        if (
            not data_panel
            or data_panel in mapping
            or data_panel
            in {
                "metrics_object",
                "context34_object",
                "all_required",
            }
        ):
            raise ProtocolError("source behavior panels have invalid data_panel names")
        minima = require_object(
            panel.get("minimum_correct"),
            f"source behavior panel {data_panel}.minimum_correct",
        )
        values = {
            "rows": panel.get("rows_exact"),
            "set_exact": minima.get("set_exact"),
            "hybrid_order_exact": minima.get("hybrid_order_exact"),
            "ordered_exact": minima.get("ordered_exact"),
            "value": minima.get("value"),
            "count": minima.get("count"),
            "top1": minima.get("top1"),
            "context34_rows": panel.get("context34_rows_exact"),
            "context34_hybrid_order_exact": minima.get(
                "context34_hybrid_order_exact"
            ),
            "context34_ordered_exact": minima.get(
                "context34_ordered_exact"
            ),
        }
        if set(values) != set(BEHAVIOR_GATE_OUTPUT_KEYS):
            raise ProtocolError("internal behavior gate mapping is incomplete")
        panel_mapping: dict[str, Any] = {}
        for gate_name, raw_value in values.items():
            value = parse_positive_int(
                raw_value,
                f"{data_panel}.{gate_name}",
                allow_zero=True,
            )
            output_key, comparison = BEHAVIOR_GATE_OUTPUT_KEYS[gate_name]
            panel_mapping[gate_name] = {
                "output_key": output_key,
                comparison: value,
            }
        mapping[data_panel] = panel_mapping
    mapping["all_required"] = True
    return mapping


def validate_source_behavior_protocol(
    preregistration: Mapping[str, Any],
    *,
    checkpoint: Path,
    runner_path: Path,
    runner_sha256: str,
    evaluator_path: Path,
    evaluator_sha256: str,
    evaluator_dependencies: Sequence[CodeDependency],
    device: str,
    batch_size: int,
    requested_workers: int,
    effective_workers: int,
    prediction_order: str,
    expected_output_prediction_order: str,
    progress_interval: int,
    compact: bool,
    panels: Sequence[PanelSpec],
) -> tuple[Path, str, str]:
    source_path, source_sha256, source = read_bound_json_reference(
        preregistration.get("source_protocol"),
        "source_protocol",
    )
    if source.get("status") != "locked_before_training":
        raise ProtocolError("source protocol was not locked before training")
    candidate_binding = require_object(
        source.get("candidate_binding"),
        "source protocol.candidate_binding",
    )
    if resolve_bound_path(candidate_binding.get("terminal_checkpoint")) != checkpoint:
        raise ProtocolError(
            "candidate checkpoint path differs from the source protocol"
        )
    behavior = require_object(
        source.get("behavior_protocol"),
        "source protocol.behavior_protocol",
    )
    implementation = require_object(
        behavior.get("implementation"),
        "source behavior implementation",
    )
    expected_implementation = {
        "python": bound_path_text(Path(sys.executable).resolve()),
        "inprocess_runner": bound_path_text(runner_path),
        "inprocess_runner_sha256": runner_sha256,
        "metric_evaluator": bound_path_text(evaluator_path),
        "metric_evaluator_sha256": evaluator_sha256,
        "local_dependencies": {
            dependency.name: {
                "path": bound_path_text(dependency.path),
                "sha256": dependency.sha256,
            }
            for dependency in evaluator_dependencies
        },
        "device": device,
        "batch_size": batch_size,
        "workers": requested_workers,
        "effective_workers": effective_workers,
        "prediction_order_cli_argument": prediction_order,
        "expected_output_prediction_order": expected_output_prediction_order,
        "compact": compact,
        "progress_interval": progress_interval,
        "same_process_two_panel_runner_required": True,
        "runner_path_and_sha256_bind_after_implementation_before_any_behavior_data_access": True,
    }
    for key, expected in expected_implementation.items():
        if implementation.get(key) != expected:
            raise ProtocolError(
                f"source behavior implementation differs for {key}"
            )
    boundary = require_object(
        behavior.get("cuda_preflight_and_formal_attempt_boundary"),
        "source behavior formal-attempt boundary",
    )
    for key in (
        "same_pid_and_cuda_context_for_preflight_and_both_panels",
        "preflight_requires_cuda_init_tensor_kernel_synchronize_and_exact_checkpoint_model_load",
        "preflight_must_not_open_behavior_archives_or_process_behavior_rows",
        "preflight_certificate_o_excl_and_fsync",
        "formal_attempt_marker_o_excl_and_fsync_after_certificate",
        "model_loaded_once_and_reused_for_both_panels",
        "no_exec_shell_or_subprocess_after_preflight",
    ):
        if boundary.get(key) is not True:
            raise ProtocolError(f"source behavior boundary does not require {key}")
    source_panels = behavior.get("ordered_panels")
    if not isinstance(source_panels, list) or len(source_panels) != 2:
        raise ProtocolError("source behavior protocol must contain two panels")
    for source_raw, panel in zip(source_panels, panels, strict=True):
        source_panel = require_object(source_raw, "source behavior panel")
        comparisons = {
            "order": panel.order,
            "name": panel.name,
            "data_panel": panel.data_panel,
            "data": bound_path_text(panel.data_path),
            "data_sha256": panel.data_sha256,
            "split": panel.split,
            "split_mode": panel.split_mode,
            "split_seed": panel.split_seed,
            "output": bound_path_text(panel.output),
        }
        for key, expected in comparisons.items():
            if source_panel.get(key) != expected:
                raise ProtocolError(
                    f"source behavior panel {panel.order} differs for {key}"
                )
    rules = require_object(behavior.get("rules"), "source behavior rules")
    if rules.get("all_20_gates_required") is not True:
        raise ProtocolError("source behavior protocol does not require all 20 gates")
    if rules.get("run_valid29_even_if_old_retention_gate_fails") is not True:
        raise ProtocolError("source behavior protocol does not require panel two")
    if (
        rules.get("run_panels_once_in_declared_order_after_formal_marker")
        is not True
    ):
        raise ProtocolError(
            "source behavior protocol does not require one ordered formal run"
        )
    if rules.get("no_alternate_decode") is not True:
        raise ProtocolError(
            "source behavior protocol does not prohibit alternate decode"
        )
    metric_mapping = require_object(
        preregistration.get("metric_mapping_and_gates"),
        "metric_mapping_and_gates",
    )
    expected_mapping = expected_gate_mapping(source_panels)
    if metric_mapping != expected_mapping:
        raise ProtocolError(
            "execution metric mapping or thresholds differ from source protocol"
        )
    return (
        source_path,
        source_sha256,
        canonical_json_sha256(metric_mapping),
    )


def parse_execution_spec(
    preregistration_path: Path,
    expected_preregistration_sha256: str,
    *,
    verified_raw: bytes | None = None,
) -> tuple[ExecutionSpec, dict[str, Any]]:
    if verified_raw is None:
        preregistration_path, observed_hash, raw = read_verified_preregistration(
            preregistration_path,
            expected_preregistration_sha256,
        )
    else:
        preregistration_path = preregistration_path.resolve()
        observed_hash = require_sha256(
            expected_preregistration_sha256,
            "expected preregistration SHA256",
        )
        raw = verified_raw
        if hashlib.sha256(raw).hexdigest() != observed_hash:
            raise ProtocolError("internally supplied preregistration bytes changed")
    try:
        preregistration = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProtocolError("preregistration is not valid JSON") from error
    preregistration = require_object(preregistration, "preregistration")
    if (
        preregistration.get("schema_version")
        != EXECUTION_PREREGISTRATION_SCHEMA
    ):
        raise ProtocolError("unsupported behavior execution schema_version")
    if preregistration.get("status") != "locked_before_behavior_evaluation":
        raise ProtocolError(
            "preregistration status must be locked_before_behavior_evaluation"
        )

    candidate = require_object(preregistration.get("candidate"), "candidate")
    checkpoint = resolve_bound_path(candidate.get("checkpoint"))
    checkpoint_sha256 = require_sha256(
        candidate.get("checkpoint_sha256"),
        "candidate checkpoint SHA256",
    )
    checkpoint_update_value = candidate.get("checkpoint_update")
    if checkpoint_update_value is None:
        raise ProtocolError("candidate checkpoint_update is required")
    checkpoint_update = parse_positive_int(
        checkpoint_update_value,
        "candidate checkpoint_update",
    )

    evaluator = require_object(preregistration.get("evaluator"), "evaluator")
    evaluator_path = resolve_bound_path(evaluator.get("path"))
    evaluator_sha256 = require_sha256(
        evaluator.get("sha256"),
        "evaluator SHA256",
    )
    raw_dependencies = require_object(
        evaluator.get("local_dependencies"),
        "evaluator.local_dependencies",
    )
    required_dependency_names = (
        "train_bc_orbit",
        "train_ppo",
        "cg/__init__.py",
        "cg/sim.py",
        "cg/libcg.so",
    )
    if set(raw_dependencies) != set(required_dependency_names):
        raise ProtocolError(
            "evaluator.local_dependencies must bind exactly train_bc_orbit, "
            "train_ppo, cg/__init__.py, cg/sim.py, and cg/libcg.so"
        )
    evaluator_dependencies = tuple(
        CodeDependency(
            name=name,
            path=resolve_bound_path(
                require_object(
                    raw_dependencies.get(name),
                    f"evaluator.local_dependencies.{name}",
                ).get("path")
            ),
            sha256=require_sha256(
                require_object(
                    raw_dependencies.get(name),
                    f"evaluator.local_dependencies.{name}",
                ).get("sha256"),
                f"evaluator.local_dependencies.{name}.sha256",
            ),
        )
        for name in required_dependency_names
    )
    evaluator_python = resolve_bound_path(evaluator.get("python"))
    if evaluator_python != Path(sys.executable).resolve():
        raise ProtocolError("evaluator.python differs from sys.executable")
    device = str(evaluator.get("device"))
    if not device.startswith("cuda"):
        raise ProtocolError("the in-process behavior runner requires a CUDA device")
    batch_size = parse_positive_int(evaluator.get("batch_size"), "batch_size")
    requested_workers = parse_positive_int(
        evaluator.get("workers"),
        "workers",
        allow_zero=True,
    )
    effective_workers = parse_positive_int(
        evaluator.get("effective_workers"),
        "effective_workers",
        allow_zero=True,
    )
    if effective_workers != 0:
        raise ProtocolError(
            "effective_workers must be 0 for the no-subprocess formal path"
        )
    prediction_order = str(evaluator.get("prediction_order_cli_argument"))
    if prediction_order not in {"auto", "policy", "canonical"}:
        raise ProtocolError("invalid prediction_order_cli_argument")
    expected_output_prediction_order = str(
        evaluator.get("expected_output_prediction_order")
    )
    if expected_output_prediction_order not in {
        "policy_greedy",
        "canonical_ascending",
    }:
        raise ProtocolError("invalid expected_output_prediction_order")
    progress_interval = parse_positive_int(
        evaluator.get("progress_interval"),
        "progress_interval",
        allow_zero=True,
    )
    compact = evaluator.get("compact")
    if not isinstance(compact, bool):
        raise ProtocolError("evaluator.compact must be boolean")
    frozen_settings = {
        "device": (device, FROZEN_DEVICE),
        "batch_size": (batch_size, FROZEN_BATCH_SIZE),
        "workers": (requested_workers, FROZEN_REQUESTED_WORKERS),
        "effective_workers": (
            effective_workers,
            FROZEN_EFFECTIVE_WORKERS,
        ),
        "prediction_order": (
            prediction_order,
            FROZEN_PREDICTION_ORDER,
        ),
        "output_prediction_order": (
            expected_output_prediction_order,
            FROZEN_OUTPUT_PREDICTION_ORDER,
        ),
        "progress_interval": (
            progress_interval,
            FROZEN_PROGRESS_INTERVAL,
        ),
        "compact": (compact, True),
    }
    for name, (observed, expected) in frozen_settings.items():
        if observed != expected:
            raise ProtocolError(
                f"{name} differs from the frozen behavior setting: "
                f"expected {expected!r}, observed {observed!r}"
            )

    data_entries = require_object(preregistration.get("data"), "data")
    evaluations = preregistration.get("ordered_evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 2:
        raise ProtocolError("ordered_evaluations must contain exactly two panels")

    panels: list[PanelSpec] = []
    for expected_order, raw_panel in enumerate(evaluations, start=1):
        panel = require_object(raw_panel, f"ordered_evaluations[{expected_order - 1}]")
        order = int(panel.get("order", -1))
        if order != expected_order:
            raise ProtocolError("panel orders must be exactly [1, 2]")
        name = str(panel.get("name", ""))
        if not name:
            raise ProtocolError("panel name must be non-empty")
        data_panel = str(panel.get("data_panel"))
        data_entry = require_object(
            data_entries.get(data_panel),
            f"data.{data_panel}",
        )
        data_path = resolve_bound_path(data_entry.get("path"))
        data_sha256 = require_sha256(
            data_entry.get("sha256"),
            f"data.{data_panel}.sha256",
        )
        split = str(data_entry.get("split"))
        split_mode = str(data_entry.get("split_mode"))
        split_seed = int(data_entry.get("split_seed"))
        if split not in {"train", "valid", "test"}:
            raise ProtocolError(f"invalid split: {split!r}")
        if split_mode not in {"archive", "episode_hash"}:
            raise ProtocolError(f"invalid split_mode: {split_mode!r}")
        if (
            split != FROZEN_SPLIT
            or split_mode != FROZEN_SPLIT_MODE
            or split_seed != FROZEN_SPLIT_SEED
        ):
            raise ProtocolError(
                "behavior data split differs from the frozen valid/archive/"
                f"{FROZEN_SPLIT_SEED} setting"
            )
        output = resolve_bound_artifact_path(panel.get("output"))
        if panel.get("output_absent_at_lock") is not True:
            raise ProtocolError("panel output_absent_at_lock must be true")
        command, parsed = parse_bound_command(panel.get("command"))

        if resolve_bound_path(command[0]) != Path(sys.executable).resolve():
            raise ProtocolError("panel command Python differs from sys.executable")
        if resolve_bound_path(command[1]) != evaluator_path:
            raise ProtocolError("panel command evaluator path differs")
        command_path_matches(
            command_required(parsed, "--checkpoint"),
            checkpoint,
            "--checkpoint",
        )
        command_path_matches(
            command_required(parsed, "--data"),
            data_path,
            "--data",
        )
        command_path_matches(
            command_required(parsed, "--json-output"),
            output,
            "--json-output",
            artifact=True,
        )
        comparisons = {
            "--split": split,
            "--split-mode": split_mode,
            "--split-seed": str(split_seed),
            "--batch-size": str(batch_size),
            "--workers": str(requested_workers),
            "--prediction-order": prediction_order,
            "--device": device,
            "--progress-interval": str(progress_interval),
        }
        for flag, expected_value in comparisons.items():
            if command_required(parsed, flag) != expected_value:
                raise ProtocolError(f"{flag} differs from the bound metadata")
        if bool(parsed["--compact"]) != compact:
            raise ProtocolError("--compact differs from the bound metadata")

        max_rows_value = parsed.get("--max-rows")
        max_rows = (
            parse_positive_int(max_rows_value, "--max-rows")
            if max_rows_value is not None
            else None
        )
        if max_rows is not None:
            raise ProtocolError("--max-rows is forbidden for formal behavior")
        if parsed["--deck-hash"] or parsed["--team-name"]:
            raise ProtocolError(
                "deck/team filters are forbidden for formal behavior"
            )
        panels.append(
            PanelSpec(
                order=order,
                name=name,
                data_panel=data_panel,
                data_path=data_path,
                data_sha256=data_sha256,
                split=split,
                split_mode=split_mode,
                split_seed=split_seed,
                batch_size=batch_size,
                requested_workers=requested_workers,
                effective_workers=effective_workers,
                prediction_order=prediction_order,
                device=device,
                output=output,
                compact=compact,
                progress_interval=progress_interval,
                max_rows=max_rows,
                deck_hashes=tuple(str(value) for value in parsed["--deck-hash"]),
                team_names=tuple(str(value) for value in parsed["--team-name"]),
                command=command,
                command_sha256=canonical_json_sha256(command),
            )
        )

    if panels[0].data_panel == panels[1].data_panel:
        raise ProtocolError("the two panels must refer to distinct data panels")
    if panels[0].output == panels[1].output:
        raise ProtocolError("the two panels must have distinct output paths")

    runner_evidence = require_object(
        preregistration.get("inprocess_runner"),
        "inprocess_runner",
    )
    runner_path = resolve_bound_path(runner_evidence.get("path"))
    runner_sha256 = require_sha256(
        runner_evidence.get("sha256"),
        "inprocess_runner.sha256",
    )
    dependency_by_name = {
        dependency.name: dependency
        for dependency in evaluator_dependencies
    }
    python_runtime_directories = {
        runner_path.parent,
        evaluator_path.parent,
        dependency_by_name["train_bc_orbit"].path.parent,
        dependency_by_name["train_ppo"].path.parent,
    }
    if len(python_runtime_directories) != 1:
        raise ProtocolError(
            "runner, evaluator, train_bc_orbit, and train_ppo must share one "
            "frozen Python runtime directory"
        )
    cg_dependencies = {
        name: dependency_by_name[name].path
        for name in ("cg/__init__.py", "cg/sim.py", "cg/libcg.so")
    }
    if (
        len({path.parent for path in cg_dependencies.values()}) != 1
        or {
            name: path.name
            for name, path in cg_dependencies.items()
        }
        != {
            "cg/__init__.py": "__init__.py",
            "cg/sim.py": "sim.py",
            "cg/libcg.so": "libcg.so",
        }
    ):
        raise ProtocolError(
            "cg/__init__.py, cg/sim.py, and cg/libcg.so must bind the "
            "expected files in one cg package directory"
        )
    preflight_evidence_directory = resolve_bound_artifact_directory(
        runner_evidence.get("preflight_evidence_directory")
    )
    marker = resolve_bound_artifact_path(
        runner_evidence.get("formal_attempt_marker")
    )
    success_result = resolve_bound_artifact_path(
        runner_evidence.get("success_result")
    )
    failure_result = resolve_bound_artifact_path(
        runner_evidence.get("failure_result")
    )
    execution_paths = {
        marker,
        success_result,
        failure_result,
        panels[0].output,
        panels[1].output,
    }
    if len(execution_paths) != 5:
        raise ProtocolError(
            "marker, success, failure, and outputs must be distinct"
        )
    if preflight_evidence_directory in execution_paths:
        raise ProtocolError(
            "preflight evidence directory must differ from execution files"
        )
    (
        source_protocol_path,
        source_protocol_sha256,
        metric_mapping_sha256,
    ) = validate_source_behavior_protocol(
        preregistration,
        checkpoint=checkpoint,
        runner_path=runner_path,
        runner_sha256=runner_sha256,
        evaluator_path=evaluator_path,
        evaluator_sha256=evaluator_sha256,
        evaluator_dependencies=evaluator_dependencies,
        device=device,
        batch_size=batch_size,
        requested_workers=requested_workers,
        effective_workers=effective_workers,
        prediction_order=prediction_order,
        expected_output_prediction_order=expected_output_prediction_order,
        progress_interval=progress_interval,
        compact=compact,
        panels=panels,
    )
    (
        training_integrity_decision_path,
        training_integrity_decision_sha256,
    ) = validate_training_authorization(
        preregistration,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_update=checkpoint_update,
        source_protocol_path=source_protocol_path,
        source_protocol_sha256=source_protocol_sha256,
    )

    return (
        ExecutionSpec(
            preregistration_path=preregistration_path,
            preregistration_sha256=observed_hash,
            runner_path=runner_path,
            runner_sha256=runner_sha256,
            evaluator_path=evaluator_path,
            evaluator_sha256=evaluator_sha256,
            evaluator_dependencies=evaluator_dependencies,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_update=checkpoint_update,
            source_protocol_path=source_protocol_path,
            source_protocol_sha256=source_protocol_sha256,
            training_integrity_decision_path=(
                training_integrity_decision_path
            ),
            training_integrity_decision_sha256=(
                training_integrity_decision_sha256
            ),
            metric_mapping_sha256=metric_mapping_sha256,
            metric_mapping_and_gates=require_object(
                preregistration.get("metric_mapping_and_gates"),
                "metric_mapping_and_gates",
            ),
            device=device,
            expected_output_prediction_order=(
                expected_output_prediction_order
            ),
            panels=(panels[0], panels[1]),
            preflight_evidence_directory=preflight_evidence_directory,
            marker=marker,
            success_result=success_result,
            failure_result=failure_result,
        ),
        preregistration,
    )


def assert_execution_artifacts_absent(spec: ExecutionSpec) -> None:
    paths = (
        spec.marker,
        spec.success_result,
        spec.failure_result,
        *(panel.output for panel in spec.panels),
    )
    existing = existing_artifact_paths(paths)
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise RestartRefusedError(
            "no-retry execution artifacts already exist; refusing restart: "
            f"{rendered}"
        )


def validate_static_bound_files(
    spec: ExecutionSpec,
    evaluator_module: Any,
) -> None:
    """Validate bound code without opening checkpoint or behavior sources."""
    loaded_runner_path = Path(__file__).resolve()
    if loaded_runner_path != spec.runner_path:
        raise ProtocolError(
            "loaded behavior runner differs from the bound runner path"
        )
    verify_file_hash(
        spec.runner_path,
        spec.runner_sha256,
        "in-process behavior runner",
    )
    module_path = Path(str(evaluator_module.__file__)).resolve()
    if module_path != spec.evaluator_path:
        raise ProtocolError(
            "loaded evaluate_policy_bc module differs from the bound evaluator path"
        )
    verify_file_hash(
        spec.evaluator_path,
        spec.evaluator_sha256,
        "evaluator",
    )
    loaded_bc = getattr(evaluator_module, "bc", None)
    loaded_ppo = getattr(evaluator_module, "ppo", None)
    loaded_cg = sys.modules.get("cg")
    loaded_cg_sim = sys.modules.get("cg.sim")
    if loaded_cg is None or loaded_cg_sim is None:
        raise ProtocolError("loaded evaluator lacks the cg package association")
    if getattr(loaded_cg, "sim", None) is not loaded_cg_sim:
        raise ProtocolError("loaded cg package is not associated with loaded cg.sim")
    loaded_lib = getattr(loaded_cg_sim, "lib", None)
    if loaded_lib is None:
        raise ProtocolError("loaded cg.sim lacks libcg")
    if getattr(loaded_ppo, "lib", None) is not loaded_lib:
        raise ProtocolError(
            "loaded train_ppo is not associated with the loaded cg.sim libcg"
        )
    cg_sim_lib_path = getattr(loaded_cg_sim, "lib_path", None)
    loaded_lib_name = getattr(loaded_lib, "_name", None)
    if cg_sim_lib_path is None or loaded_lib_name is None:
        raise ProtocolError("loaded cg.sim does not expose its loaded libcg path")
    if Path(str(cg_sim_lib_path)).resolve() != Path(str(loaded_lib_name)).resolve():
        raise ProtocolError(
            "loaded cg.sim libcg handle differs from cg.sim.lib_path"
        )
    loaded_dependency_paths: dict[str, object] = {
        "train_bc_orbit": getattr(loaded_bc, "__file__", None),
        "train_ppo": getattr(loaded_ppo, "__file__", None),
        "cg/__init__.py": getattr(loaded_cg, "__file__", None),
        "cg/sim.py": getattr(loaded_cg_sim, "__file__", None),
        "cg/libcg.so": loaded_lib_name,
    }
    for dependency in spec.evaluator_dependencies:
        loaded_path_value = loaded_dependency_paths.get(dependency.name)
        if loaded_path_value is None:
            raise ProtocolError(
                f"loaded evaluator lacks dependency {dependency.name}"
            )
        loaded_path = Path(str(loaded_path_value)).resolve()
        if loaded_path != dependency.path:
            raise ProtocolError(
                f"loaded {dependency.name} differs from the bound dependency"
            )
        verify_file_hash(
            dependency.path,
            dependency.sha256,
            f"evaluator dependency {dependency.name}",
        )


def hash_descriptor(descriptor: int) -> str:
    """Hash a held descriptor from byte zero and restore its offset to zero."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _stable_stat_fields(stat_result: os.stat_result) -> tuple[int, ...]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mode),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _held_use_stat_fields(stat_result: os.stat_result) -> tuple[int, ...]:
    """Fields that cannot change from pathname unlink/replacement alone."""
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mode),
        int(stat_result.st_mtime_ns),
    )


def _stat_payload(stat_result: os.stat_result) -> dict[str, int]:
    return {
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "size": int(stat_result.st_size),
        "mode": int(stat_result.st_mode),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "ctime_ns": int(stat_result.st_ctime_ns),
        "link_count": int(stat_result.st_nlink),
    }


def open_verified_held_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> HeldFile:
    """Pin, lock, and hash one inode so later path replacement is irrelevant."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProtocolError(f"{label} does not exist or cannot be opened: {path}") from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        before = os.fstat(descriptor)
        observed = hash_descriptor(descriptor)
        after = os.fstat(descriptor)
        if _stable_stat_fields(before) != _stable_stat_fields(after):
            raise ProtocolError(f"{label} changed while its SHA256 was computed")
        if observed != expected_sha256:
            raise ProtocolError(
                f"{label} SHA256 mismatch: expected {expected_sha256}, "
                f"observed {observed}"
            )
        return HeldFile(
            original_path=path,
            expected_sha256=expected_sha256,
            descriptor=descriptor,
            proc_path=Path(f"/proc/self/fd/{descriptor}"),
            device=int(after.st_dev),
            inode=int(after.st_ino),
            size=int(after.st_size),
            mode=int(after.st_mode),
            mtime_ns=int(after.st_mtime_ns),
            ctime_ns=int(after.st_ctime_ns),
            link_count=int(after.st_nlink),
            pre_sha256=observed,
            lock="shared_flock",
        )
    except BaseException:
        os.close(descriptor)
        raise


def verify_held_file_after_use(held: HeldFile, label: str) -> str:
    """Re-hash the same held inode and reject any in-place mutation."""
    observed = hash_descriptor(held.descriptor)
    after = os.fstat(held.descriptor)
    expected_stat = (
        held.device,
        held.inode,
        held.size,
        held.mode,
        held.mtime_ns,
    )
    if _held_use_stat_fields(after) != expected_stat:
        raise ProtocolError(f"{label} inode metadata changed while in use")
    if observed != held.expected_sha256 or observed != held.pre_sha256:
        raise ProtocolError(
            f"{label} changed while in use: expected {held.expected_sha256}, "
            f"observed {observed}"
        )
    held.post_fstat = _stat_payload(after)
    return observed


def held_file_payload(
    held: HeldFile,
    *,
    post_sha256: str | None,
) -> dict[str, Any]:
    return {
        "original_path": str(held.original_path),
        "expected_sha256": held.expected_sha256,
        "held_fd_path_during_use": str(held.proc_path),
        "device": held.device,
        "inode": held.inode,
        "size": held.size,
        "mode": held.mode,
        "mtime_ns": held.mtime_ns,
        "ctime_ns": held.ctime_ns,
        "lock": held.lock,
        "pre_sha256": held.pre_sha256,
        "post_sha256": post_sha256,
        "pre_fstat": {
            "device": held.device,
            "inode": held.inode,
            "size": held.size,
            "mode": held.mode,
            "mtime_ns": held.mtime_ns,
            "ctime_ns": held.ctime_ns,
            "link_count": held.link_count,
        },
        "post_fstat": held.post_fstat,
        "pre_post_sha256_match": (
            post_sha256 == held.pre_sha256
            if post_sha256 is not None
            else None
        ),
    }


def process_start_ticks() -> str:
    try:
        raw = Path("/proc/self/stat").read_text(encoding="utf-8")
        suffix = raw.rsplit(")", 1)[1].strip().split()
        return suffix[19]
    except (OSError, IndexError):
        return "unavailable"


def relevant_environment() -> dict[str, str | None]:
    return {
        key: os.environ.get(key)
        for key in RELEVANT_ENVIRONMENT_KEYS
    }


def environment_sha256() -> str:
    return canonical_json_sha256(relevant_environment())


def serialize_json_artifact(
    value: Mapping[str, Any],
    *,
    compact: bool = False,
) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=None if compact else 2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def json_artifact_sha256(
    value: Mapping[str, Any],
    *,
    compact: bool = False,
) -> str:
    return hashlib.sha256(
        serialize_json_artifact(value, compact=compact)
    ).hexdigest()


def atomic_write_json_exclusive(
    path: Path,
    value: Mapping[str, Any],
    *,
    compact: bool = False,
) -> str:
    """Create through a trusted openat chain and close after durable publish."""
    held = create_json_artifact_exclusive_held(
        path,
        value,
        compact=compact,
    )
    try:
        return held.sha256
    finally:
        held.close()


def create_json_artifact_exclusive_held(
    path: Path,
    value: Mapping[str, Any],
    *,
    compact: bool = False,
) -> HeldArtifact:
    """Durably create O_EXCL while retaining the exact published inode."""
    path = resolve_bound_artifact_path(path)
    reject_artifact_symlink(path, "execution artifact target")
    serialized = serialize_json_artifact(value, compact=compact)
    parent_fd, leaf = _open_artifact_parent_dirfd(path, create=True)
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        try:
            descriptor = os.open(
                leaf,
                flags,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError as error:
            raise RestartRefusedError(
                f"refusing to replace existing execution artifact: {path}"
            ) from error
        except OSError as error:
            raise ProtocolError(
                f"could not exclusively create execution artifact: {path}"
            ) from error
        try:
            view = memoryview(serialized)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while persisting JSON artifact")
                view = view[written:]
            os.fsync(descriptor)
            os.fsync(parent_fd)
            file_stat = os.fstat(descriptor)
            path_stat = os.stat(
                leaf,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or int(file_stat.st_nlink) != 1
                or _root_identity(file_stat) != _root_identity(path_stat)
                or int(file_stat.st_size) != len(serialized)
            ):
                raise ProtocolError(
                    "published execution artifact path or inode changed"
                )
            canonical_path_stat = _artifact_lstat(path)
            if (
                canonical_path_stat is None
                or _root_identity(canonical_path_stat)
                != _root_identity(file_stat)
            ):
                raise ProtocolError(
                    "artifact ancestor or canonical pathname changed during "
                    "exclusive publication"
                )
            return HeldArtifact(
                path=path,
                descriptor=descriptor,
                sha256=hashlib.sha256(serialized).hexdigest(),
                serialized=serialized,
                device=int(file_stat.st_dev),
                inode=int(file_stat.st_ino),
                size=int(file_stat.st_size),
                mode=int(file_stat.st_mode),
            )
        except BaseException:
            # The visible partial inode intentionally consumes this target.
            try:
                os.fsync(descriptor)
            except OSError:
                pass
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
            os.close(descriptor)
            raise
    finally:
        os.close(parent_fd)


def recover_expected_json_artifact_held(
    path: Path,
    value: Mapping[str, Any],
    *,
    compact: bool = False,
) -> HeldArtifact:
    """Securely recover an exact artifact after an indeterminate publish error.

    This is deliberately narrower than an ordinary reopen: only the exact
    canonical serialization that this process attempted to publish is
    accepted, and the recovered inode remains held for all later checks.
    """
    path = resolve_bound_artifact_path(path)
    serialized = serialize_json_artifact(value, compact=compact)
    expected_sha256 = hashlib.sha256(serialized).hexdigest()
    parent_fd, leaf = _open_artifact_parent_dirfd(path, create=False)
    flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(leaf, flags, dir_fd=parent_fd)
        try:
            os.fsync(descriptor)
            os.fsync(parent_fd)
            before = os.fstat(descriptor)
            observed_sha256 = hash_descriptor(descriptor)
            after = os.fstat(descriptor)
            path_stat = os.stat(
                leaf,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(after.st_mode)
                or int(after.st_nlink) != 1
                or _stable_stat_fields(before) != _stable_stat_fields(after)
                or _root_identity(after) != _root_identity(path_stat)
                or int(after.st_size) != len(serialized)
                or observed_sha256 != expected_sha256
            ):
                raise ProtocolError(
                    "indeterminate artifact publication did not leave the "
                    "exact expected regular-file inode"
                )
            canonical_path_stat = _artifact_lstat(path)
            if (
                canonical_path_stat is None
                or _root_identity(canonical_path_stat)
                != _root_identity(after)
            ):
                raise ProtocolError(
                    "artifact ancestor or canonical pathname changed while "
                    "recovering an indeterminate publication"
                )
            return HeldArtifact(
                path=path,
                descriptor=descriptor,
                sha256=expected_sha256,
                serialized=serialized,
                device=int(after.st_dev),
                inode=int(after.st_ino),
                size=int(after.st_size),
                mode=int(after.st_mode),
            )
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        os.close(parent_fd)


def verify_held_artifact(
    held: HeldArtifact,
    label: str,
) -> str:
    """Rehash held content and prove its trusted-root pathname still links it."""
    if held.descriptor < 0:
        raise ProtocolError(f"{label} held descriptor is closed")
    observed = hash_descriptor(held.descriptor)
    file_stat = os.fstat(held.descriptor)
    if (
        _root_identity(file_stat) != (held.device, held.inode)
        or int(file_stat.st_size) != held.size
        or int(file_stat.st_mode) != held.mode
        or int(file_stat.st_nlink) != 1
        or observed != held.sha256
        or observed != hashlib.sha256(held.serialized).hexdigest()
    ):
        raise ProtocolError(f"{label} held inode or content changed")
    try:
        parent_fd, leaf = _open_artifact_parent_dirfd(
            held.path,
            create=False,
        )
    except FileNotFoundError as error:
        raise ProtocolError(f"{label} pathname disappeared") from error
    try:
        try:
            path_stat = os.stat(
                leaf,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as error:
            raise ProtocolError(f"{label} pathname disappeared") from error
        if (
            stat.S_ISLNK(path_stat.st_mode)
            or _root_identity(path_stat) != (held.device, held.inode)
        ):
            raise ProtocolError(f"{label} pathname no longer links held inode")
    finally:
        os.close(parent_fd)
    return observed


def read_json_artifact_verified(
    path: Path,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Read a persisted artifact through the trusted root and bind its inode."""
    path = resolve_bound_artifact_path(path)
    parent_fd, leaf = _open_artifact_parent_dirfd(path, create=False)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(leaf, flags, dir_fd=parent_fd)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ProtocolError(f"{label} is not a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            path_stat = os.stat(
                leaf,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            canonical_path_stat = _artifact_lstat(path)
            if (
                not stat.S_ISREG(after.st_mode)
                or int(after.st_nlink) != 1
                or _stable_stat_fields(before) != _stable_stat_fields(after)
                or _root_identity(after) != _root_identity(path_stat)
                or canonical_path_stat is None
                or _root_identity(after)
                != _root_identity(canonical_path_stat)
                or hashlib.sha256(raw).hexdigest() != expected_sha256
            ):
                raise ProtocolError(f"{label} changed while it was read")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"{label} is not valid JSON") from error
    return require_object(value, label)


def parameter_devices(model: Any) -> tuple[str, ...]:
    devices = {
        str(tensor.device)
        for tensor in (*tuple(model.parameters()), *tuple(model.buffers()))
    }
    if not devices:
        raise ProtocolError("loaded policy has no parameters or buffers")
    if any(not device.startswith("cuda") for device in devices):
        raise ProtocolError(
            "loaded policy is not entirely on CUDA: "
            + ", ".join(sorted(devices))
        )
    return tuple(sorted(devices))


def run_cuda_preflight(
    spec: ExecutionSpec,
    evaluator_module: Any,
    torch_module: Any,
) -> PreflightState:
    device = torch_module.device(spec.device)
    if device.type != "cuda":
        raise ProtocolError("preflight device is not CUDA")
    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision("high")

    torch_module.cuda.init()
    probe = torch_module.ones(1, device=device)
    probe.add_(1)
    torch_module.cuda.synchronize(device)
    if float(probe.item()) != 2.0:
        raise ProtocolError("CUDA tensor-kernel probe returned an unexpected value")

    checkpoint_held = open_verified_held_file(
        spec.checkpoint,
        spec.checkpoint_sha256,
        "candidate checkpoint",
    )
    try:
        model, model_config, checkpoint_payload, checkpoint_kind = (
            evaluator_module.load_policy(checkpoint_held.proc_path, device)
        )
        checkpoint_post_sha256 = verify_held_file_after_use(
            checkpoint_held,
            "candidate checkpoint",
        )
        checkpoint_held_payload = held_file_payload(
            checkpoint_held,
            post_sha256=checkpoint_post_sha256,
        )
    finally:
        checkpoint_held.close()
    if not isinstance(model_config, dict):
        raise ProtocolError("evaluator load_policy returned invalid model_config")
    if not isinstance(checkpoint_payload, dict):
        raise ProtocolError("evaluator load_policy returned invalid checkpoint")
    if (
        spec.checkpoint_update is not None
        and int(checkpoint_payload.get("update", -1)) != spec.checkpoint_update
    ):
        raise ProtocolError("loaded checkpoint update differs from preregistration")
    observed_prediction_order = (
        "canonical_ascending"
        if canonicalize_prediction_order(
            spec.panels[0].prediction_order,
            str(checkpoint_kind),
        )
        else "policy_greedy"
    )
    if observed_prediction_order != spec.expected_output_prediction_order:
        raise ProtocolError(
            "checkpoint kind and prediction-order setting do not produce "
            "the preregistered output prediction order"
        )
    required_model_config = (
        "hash_size",
        "max_state_entities",
        "entity_fields",
        "option_fields",
    )
    missing_model_config = [
        key for key in required_model_config if key not in model_config
    ]
    if missing_model_config:
        raise ProtocolError(
            "loaded model_config lacks required fields: "
            + ", ".join(missing_model_config)
        )
    model.eval()
    devices = parameter_devices(model)
    count_classes(checkpoint_payload)
    inference_count_classes(model)
    torch_module.cuda.synchronize(device)

    device_index = (
        int(device.index)
        if getattr(device, "index", None) is not None
        else int(torch_module.cuda.current_device())
    )
    capability_raw = torch_module.cuda.get_device_capability(device_index)
    capability = (int(capability_raw[0]), int(capability_raw[1]))
    return PreflightState(
        model=model,
        model_config=model_config,
        checkpoint_payload=checkpoint_payload,
        checkpoint_kind=str(checkpoint_kind),
        device=device,
        pid=os.getpid(),
        process_start_ticks=process_start_ticks(),
        environment_sha256=environment_sha256(),
        cuda_device_index=device_index,
        cuda_device_name=str(torch_module.cuda.get_device_name(device_index)),
        cuda_capability=capability,
        model_devices=devices,
        checkpoint_held_file=checkpoint_held_payload,
    )


def preflight_certificate_payload(
    spec: ExecutionSpec,
    state: PreflightState,
    torch_module: Any,
) -> dict[str, Any]:
    return {
        "schema_version": CERTIFICATE_SCHEMA,
        "runner_schema_version": RUNNER_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "cuda_preflight_passed_before_formal_attempt",
        "preregistration": {
            "path": str(spec.preregistration_path),
            "sha256": spec.preregistration_sha256,
        },
        "source_protocol": {
            "path": str(spec.source_protocol_path),
            "sha256": spec.source_protocol_sha256,
        },
        "training_integrity_decision": {
            "path": str(spec.training_integrity_decision_path),
            "sha256": spec.training_integrity_decision_sha256,
            "behavior_execution_authorized": True,
        },
        "metric_mapping_sha256": spec.metric_mapping_sha256,
        "evaluator": {
            "path": str(spec.evaluator_path),
            "sha256": spec.evaluator_sha256,
            "local_dependencies": {
                dependency.name: {
                    "path": str(dependency.path),
                    "sha256": dependency.sha256,
                }
                for dependency in spec.evaluator_dependencies
            },
        },
        "runner": {
            "path": str(spec.runner_path),
            "sha256": spec.runner_sha256,
        },
        "checkpoint": {
            "original_path": str(spec.checkpoint),
            "sha256": spec.checkpoint_sha256,
            "update": spec.checkpoint_update,
            "kind": state.checkpoint_kind,
            "held_file": state.checkpoint_held_file,
        },
        "process": {
            "pid": state.pid,
            "start_ticks": state.process_start_ticks,
            "cwd": str(Path.cwd().resolve()),
            "python": str(Path(sys.executable).resolve()),
            "environment_sha256": state.environment_sha256,
            "relevant_environment": relevant_environment(),
        },
        "cuda": {
            "torch_version": str(torch_module.__version__),
            "torch_compiled_cuda": str(torch_module.version.cuda),
            "device_index": state.cuda_device_index,
            "device_name": state.cuda_device_name,
            "compute_capability": list(state.cuda_capability),
            "tensor_kernel_value": 2.0,
            "synchronize_passed": True,
            "model_devices": list(state.model_devices),
        },
        "formal_attempt_started": False,
        "dataset_objects_constructed": 0,
        "raw_data_files_read_for_sha256": 0,
        "raw_data_files_sha256_verified": 0,
        "zip_archive_members_opened": 0,
        "data_archives_semantically_opened": 0,
        "behavior_rows_evaluated": 0,
        "outputs_absent": all(
            not artifact_path_lexists(panel.output)
            for panel in spec.panels
        ),
    }


def formal_marker_payload(
    spec: ExecutionSpec,
    state: PreflightState,
    certificate_path: Path,
    certificate_sha256: str,
) -> dict[str, Any]:
    if os.getpid() != state.pid:
        raise ProtocolError("PID changed between preflight and formal marker")
    if process_start_ticks() != state.process_start_ticks:
        raise ProtocolError(
            "process start identity changed between preflight and marker"
        )
    current_environment_sha256 = environment_sha256()
    if current_environment_sha256 != state.environment_sha256:
        raise ProtocolError("CUDA-relevant environment changed after preflight")
    if any(artifact_path_lexists(panel.output) for panel in spec.panels):
        raise RestartRefusedError("behavior output appeared before formal marker")
    return {
        "schema_version": MARKER_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "formal_no_retry_attempt_started",
        "preregistration_sha256": spec.preregistration_sha256,
        "source_protocol_sha256": spec.source_protocol_sha256,
        "training_integrity_decision_sha256": (
            spec.training_integrity_decision_sha256
        ),
        "metric_mapping_sha256": spec.metric_mapping_sha256,
        "preflight_certificate": {
            "path": str(certificate_path),
            "sha256": certificate_sha256,
        },
        "process": {
            "pid": state.pid,
            "start_ticks": state.process_start_ticks,
            "environment_sha256": state.environment_sha256,
        },
        "ordered_panels": [
            {
                "order": panel.order,
                "name": panel.name,
                "data_panel": panel.data_panel,
                "command_sha256": panel.command_sha256,
                "output": str(panel.output),
                "requested_workers": panel.requested_workers,
                "effective_workers": panel.effective_workers,
            }
            for panel in spec.panels
        ],
        "no_retry_after_this_marker": True,
    }


def canonicalize_prediction_order(prediction_order: str, checkpoint_kind: str) -> bool:
    if prediction_order == "auto":
        return checkpoint_kind == "bc"
    return prediction_order == "canonical"


def count_classes(checkpoint_payload: Mapping[str, Any]) -> int:
    state_dict = checkpoint_payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ProtocolError("checkpoint has no model_state_dict")
    count_weight = state_dict.get("count_head.2.weight")
    shape = getattr(count_weight, "shape", None)
    if shape is None or len(shape) != 2:
        raise ProtocolError("checkpoint count_head.2.weight has invalid shape")
    return int(shape[0])


def inference_count_classes(model: Any) -> int:
    try:
        return int(model.count_head[-1].out_features)
    except (AttributeError, IndexError, TypeError) as error:
        raise ProtocolError("loaded model has an invalid count head") from error


def run_panel(
    spec: ExecutionSpec,
    panel: PanelSpec,
    state: PreflightState,
    evaluator_module: Any,
) -> tuple[dict[str, Any], str]:
    held = open_verified_held_file(
        panel.data_path,
        panel.data_sha256,
        f"{panel.data_panel} data",
    )
    try:
        dataset = evaluator_module.OrderedZipDecisionDataset(
            archive_path=held.proc_path,
            split=panel.split,
            split_mode=panel.split_mode,
            split_seed=panel.split_seed,
            hash_size=state.model_config["hash_size"],
            max_state_entities=state.model_config["max_state_entities"],
            deck_hashes=panel.deck_hashes,
            team_names=panel.team_names,
        )
        loader = evaluator_module.DataLoader(
            dataset,
            batch_size=panel.batch_size,
            num_workers=panel.effective_workers,
            collate_fn=partial(
                evaluator_module.collate_ordered,
                max_state_entities=state.model_config["max_state_entities"],
                entity_fields=state.model_config["entity_fields"],
                option_fields=state.model_config["option_fields"],
            ),
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None,
        )
        canonicalize_order = canonicalize_prediction_order(
            panel.prediction_order,
            state.checkpoint_kind,
        )
        metrics, seconds = evaluator_module.evaluate(
            state.model,
            loader,
            state.device,
            canonicalize_order=canonicalize_order,
            max_rows=panel.max_rows,
            progress_interval=panel.progress_interval,
        )
        data_post_sha256 = verify_held_file_after_use(
            held,
            f"{panel.data_panel} data",
        )
        rows: int | None = None
        if isinstance(metrics, Mapping):
            candidate_rows = metrics.get("rows")
            if type(candidate_rows) is int and candidate_rows >= 0:
                rows = candidate_rows
        result = {
            "evaluator": bound_path_text(spec.evaluator_path),
            "checkpoint": str(spec.checkpoint),
            "checkpoint_sha256": spec.checkpoint_sha256,
            "checkpoint_held_file": state.checkpoint_held_file,
            "checkpoint_kind": state.checkpoint_kind,
            "feature_version": state.checkpoint_payload.get("feature_version"),
            "checkpoint_update": state.checkpoint_payload.get("update"),
            "data": str(panel.data_path),
            "data_sha256": panel.data_sha256,
            "data_held_file": held_file_payload(
                held,
                post_sha256=data_post_sha256,
            ),
            "split": panel.split,
            "split_mode": panel.split_mode,
            "split_seed": panel.split_seed,
            "workers": {
                "requested_by_evaluator_command": panel.requested_workers,
                "effective_inprocess_dataloader": panel.effective_workers,
                "reconciliation": (
                    "formal runner forbids child processes; effective workers "
                    "is preregistered as zero"
                ),
            },
            "filters": {
                "deck_hashes": list(panel.deck_hashes),
                "team_names": list(panel.team_names),
            },
            "prediction_order": (
                "canonical_ascending"
                if canonicalize_order
                else "policy_greedy"
            ),
            "hybrid_prediction_order": (
                "policy_greedy_for_context_34_skill_order;"
                "canonical_ascending_otherwise"
            ),
            "device": str(state.device),
            "checkpoint_count_classes": count_classes(state.checkpoint_payload),
            "inference_count_classes": inference_count_classes(state.model),
            "max_rows": panel.max_rows,
            "metrics": metrics,
            "seconds": float(seconds),
            "rows_per_second": (
                rows / max(float(seconds), 1e-6)
                if rows is not None
                else None
            ),
        }
        output_sha256 = atomic_write_json_exclusive(
            panel.output,
            result,
            compact=panel.compact,
        )
        read_json_artifact_verified(
            panel.output,
            output_sha256,
            f"{panel.data_panel} persisted behavior output",
        )
        return result, output_sha256
    finally:
        held.close()


GATE_METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "rows": ("metrics", "rows"),
    "set_exact": ("metrics", "set_exact_correct"),
    "hybrid_order_exact": ("metrics", "hybrid_order_exact_correct"),
    "ordered_exact": ("metrics", "ordered_exact_correct"),
    "value": ("metrics", "value_correct"),
    "count": ("metrics", "count_correct"),
    "top1": ("metrics", "top1_correct"),
    "context34_rows": ("metrics", "by_context", "34", "rows"),
    "context34_hybrid_order_exact": (
        "metrics",
        "by_context",
        "34",
        "hybrid_order_exact_correct",
    ),
    "context34_ordered_exact": (
        "metrics",
        "by_context",
        "34",
        "ordered_exact_correct",
    ),
}


def _strict_output_integer_observation(
    output: Mapping[str, Any],
    path: Sequence[str],
) -> tuple[int | None, str | None]:
    """Return one strict metric observation without short-circuiting gates."""
    current: Any = output
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            return (
                None,
                f"missing output field {'.'.join(path)}",
            )
        current = current[component]
    if type(current) is not int:
        return (
            None,
            f"{'.'.join(path)} must be a strict integer",
        )
    if current < 0:
        return (
            None,
            f"{'.'.join(path)} must be non-negative",
        )
    return current, None


def evaluate_persisted_behavior_gates(
    spec: ExecutionSpec,
    completed_panels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Read both persisted outputs and evaluate every frozen gate internally."""
    if len(completed_panels) != 2:
        raise ProtocolError("all two panel outputs are required for gate evaluation")
    completed_by_panel = {
        str(item["data_panel"]): item
        for item in completed_panels
    }
    gate_results: list[dict[str, Any]] = []
    for panel in spec.panels:
        completed = completed_by_panel.get(panel.data_panel)
        if completed is None:
            raise ProtocolError(
                f"missing completed output for panel {panel.data_panel}"
            )
        output = read_json_artifact_verified(
            panel.output,
            str(completed["output_sha256"]),
            f"{panel.data_panel} output for frozen gate evaluation",
        )
        if output.get("checkpoint_sha256") != spec.checkpoint_sha256:
            raise ProtocolError(
                f"{panel.data_panel} output checkpoint binding differs"
            )
        if (
            output.get("prediction_order")
            != spec.expected_output_prediction_order
        ):
            raise ProtocolError(
                f"{panel.data_panel} output prediction order differs"
            )
        panel_mapping = require_object(
            spec.metric_mapping_and_gates.get(panel.data_panel),
            f"metric_mapping_and_gates.{panel.data_panel}",
        )
        if set(panel_mapping) != set(BEHAVIOR_GATE_OUTPUT_KEYS):
            raise ProtocolError(
                f"{panel.data_panel} gate mapping is not the frozen ten gates"
            )
        for gate_name in BEHAVIOR_GATE_OUTPUT_KEYS:
            gate_id = f"{panel.data_panel}.{gate_name}"
            mapping = require_object(
                panel_mapping.get(gate_name),
                f"gate mapping {gate_id}",
            )
            observed, validation_error = _strict_output_integer_observation(
                output,
                GATE_METRIC_PATHS[gate_name],
            )
            _, comparison = BEHAVIOR_GATE_OUTPUT_KEYS[gate_name]
            expected_output_key = BEHAVIOR_GATE_OUTPUT_KEYS[gate_name][0]
            if mapping.get("output_key") != expected_output_key:
                raise ProtocolError(
                    f"{gate_id} output_key differs from frozen mapping"
                )
            threshold = mapping.get(comparison)
            if type(threshold) is not int:
                raise ProtocolError(
                    f"{gate_id} threshold must be a strict integer"
                )
            passed = bool(
                validation_error is None
                and observed is not None
                and (
                    observed == threshold
                    if comparison == "exact"
                    else observed >= threshold
                )
            )
            gate_result = {
                "gate": gate_id,
                "output_key": mapping.get("output_key"),
                "comparison": comparison,
                "threshold": threshold,
                "observed": observed,
                "pass": passed,
            }
            if validation_error is not None:
                gate_result["validation_error"] = validation_error
            gate_results.append(gate_result)
    if len(gate_results) != 20:
        raise ProtocolError("frozen behavior evaluation did not produce 20 gates")
    failing_gates = [
        str(item["gate"])
        for item in gate_results
        if item["pass"] is not True
    ]
    report = {
        "all_20_gates_evaluated": True,
        "gate_count": 20,
        "passed_gate_count": 20 - len(failing_gates),
        "all_20_gates_passed": not failing_gates,
        "failing_gates": failing_gates,
        "gates": gate_results,
    }
    if failing_gates:
        raise BehaviorGateFailure(report)
    return report


def failure_payload(
    spec: ExecutionSpec,
    state: PreflightState,
    marker_sha256: str,
    current_panel: PanelSpec | None,
    completed_panels: Sequence[dict[str, Any]],
    error: BaseException,
) -> dict[str, Any]:
    panel_failures = (
        list(error.failures)
        if isinstance(error, PanelExecutionAggregateError)
        else []
    )
    gate_report = (
        error.report
        if isinstance(error, BehaviorGateFailure)
        else None
    )
    failure_class = (
        "behavior_gate_rejection"
        if isinstance(error, BehaviorGateFailure)
        else (
            "panel_runtime_failure"
            if isinstance(error, PanelExecutionAggregateError)
            else "execution_integrity_failure"
        )
    )
    return {
        "schema_version": FAILURE_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "terminal_failure_after_formal_attempt_marker",
        "failure_class": failure_class,
        "preregistration_sha256": spec.preregistration_sha256,
        "formal_attempt_marker": {
            "path": str(spec.marker),
            "sha256": marker_sha256,
        },
        "process": {
            "pid": state.pid,
            "start_ticks": state.process_start_ticks,
            "environment_sha256": state.environment_sha256,
        },
        "current_panel": (
            {
                "order": current_panel.order,
                "name": current_panel.name,
                "data_panel": current_panel.data_panel,
            }
            if current_panel is not None
            else None
        ),
        "completed_panels": list(completed_panels),
        "panel_failures": panel_failures,
        "quality_gates": gate_report,
        "quality_gates_evaluated": bool(
            gate_report
            and gate_report.get("all_20_gates_evaluated") is True
        ),
        "all_20_gates_passed": False,
        "gold19_authorized": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "no_retry": True,
    }


def preflight_evidence_path(
    spec: ExecutionSpec,
    kind: str,
    *,
    pid: int,
    start_ticks: str,
) -> Path:
    safe_ticks = "".join(
        character if character.isalnum() else "_"
        for character in start_ticks
    )
    return spec.preflight_evidence_directory / (
        f"{kind}.pid-{pid}.start-{safe_ticks}.ns-{time.time_ns()}.json"
    )


def preregistration_parse_failure_path(
    preregistration_path: Path,
) -> Path:
    directory = preregistration_path.parent / (
        f".{preregistration_path.name}.preflight-parse-evidence"
    )
    safe_ticks = "".join(
        character if character.isalnum() else "_"
        for character in process_start_ticks()
    )
    return directory / (
        "preregistration-parse-failure."
        f"pid-{os.getpid()}.start-{safe_ticks}.ns-{time.time_ns()}.json"
    )


def preregistration_parse_failure_payload(
    preregistration_path: Path,
    preregistration_sha256: str,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_FAILURE_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "trusted_preregistration_parse_failure_before_formal_attempt",
        "phase": "full_preregistration_parse",
        "preregistration": {
            "path": str(preregistration_path),
            "raw_bytes_sha256_verified": True,
            "sha256": preregistration_sha256,
        },
        "process": {
            "pid": os.getpid(),
            "start_ticks": process_start_ticks(),
            "cwd": str(Path.cwd().resolve()),
            "python": str(Path(sys.executable).resolve()),
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "formal_attempt_consumed": False,
        "same_frozen_raw_preregistration_may_retry_after_parser_repair": True,
    }


def preflight_failure_payload(
    spec: ExecutionSpec,
    *,
    phase: str,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_FAILURE_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "infrastructure_failure_before_formal_attempt_marker",
        "phase": phase,
        "preregistration": {
            "path": str(spec.preregistration_path),
            "sha256": spec.preregistration_sha256,
        },
        "process": {
            "pid": os.getpid(),
            "start_ticks": process_start_ticks(),
            "cwd": str(Path.cwd().resolve()),
            "python": str(Path(sys.executable).resolve()),
            "environment_sha256": environment_sha256(),
        },
        "formal_attempt_marker_exists": artifact_path_lexists(spec.marker),
        "behavior_outputs_exist": [
            str(panel.output)
            for panel in spec.panels
            if artifact_path_lexists(panel.output)
        ],
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "formal_attempt_consumed": False,
        "same_frozen_configuration_may_retry_after_infrastructure_repair": True,
    }


def success_payload(
    spec: ExecutionSpec,
    state: PreflightState,
    *,
    certificate_path: Path,
    certificate_sha256: str,
    marker_sha256: str,
    completed_panels: Sequence[dict[str, Any]],
    gate_report: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        gate_report.get("all_20_gates_evaluated") is not True
        or gate_report.get("all_20_gates_passed") is not True
        or gate_report.get("gate_count") != 20
    ):
        raise ProtocolError("terminal success requires all twenty frozen gates")
    return {
        "schema_version": SUCCESS_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "both_panels_completed",
        "preregistration_sha256": spec.preregistration_sha256,
        "source_protocol_sha256": spec.source_protocol_sha256,
        "training_integrity_decision_sha256": (
            spec.training_integrity_decision_sha256
        ),
        "metric_mapping_sha256": spec.metric_mapping_sha256,
        "preflight_certificate": {
            "path": str(certificate_path),
            "sha256": certificate_sha256,
        },
        "formal_attempt_marker": {
            "path": str(spec.marker),
            "sha256": marker_sha256,
        },
        "process": {
            "pid": state.pid,
            "start_ticks": state.process_start_ticks,
            "environment_sha256": state.environment_sha256,
        },
        "checkpoint": {
            "path": str(spec.checkpoint),
            "sha256": spec.checkpoint_sha256,
            "update": spec.checkpoint_update,
            "held_file": state.checkpoint_held_file,
        },
        "completed_panels": list(completed_panels),
        "formal_attempt_consumed": True,
        "quality_gates_evaluated": True,
        "all_20_gates_passed": True,
        "quality_gates": dict(gate_report),
        "gold19_authorized": False,
    }


def _execute_locked(
    locked: LockedPreregistration,
    process_lock: ProcessExecutionLock,
    *,
    evaluator_module: Any = policy_eval,
    torch_module: Any = torch,
) -> list[dict[str, Any]]:
    preregistration_path = locked.path
    preregistration_sha256 = locked.sha256
    verified_raw = locked.raw
    validate_process_root_lock(process_lock)
    # A consumed attempt must be refused without reading mutable external
    # bindings and without emitting a new preflight/parse-failure artifact.
    assert_trusted_raw_restart_artifacts_absent(verified_raw)
    try:
        spec, _ = parse_execution_spec(
            preregistration_path,
            preregistration_sha256,
            verified_raw=verified_raw,
        )
    except BaseException as error:
        failure_path = preregistration_parse_failure_path(
            preregistration_path
        )
        try:
            atomic_write_json_exclusive(
                failure_path,
                preregistration_parse_failure_payload(
                    preregistration_path,
                    preregistration_sha256,
                    error,
                ),
            )
        except BaseException as write_error:
            error.add_note(
                "trusted preregistration parse failure evidence could not be "
                "written exclusively: "
                f"{type(write_error).__name__}: {write_error}"
            )
        raise
    # This check intentionally precedes all hashing and CUDA/model work on a
    # restart.
    assert_execution_artifacts_absent(spec)
    phase = "static_bound_file_validation"
    state: PreflightState | None = None
    certificate_held: HeldArtifact | None = None
    marker_held: HeldArtifact | None = None
    marker_payload: dict[str, Any] | None = None
    try:
        validate_process_root_lock(process_lock)
        validate_static_bound_files(spec, evaluator_module)
        phase = "cuda_and_exact_model_preflight"
        validate_process_root_lock(process_lock)
        state = run_cuda_preflight(spec, evaluator_module, torch_module)
        validate_process_root_lock(process_lock)
        phase = "preflight_certificate_persistence"
        certificate_path = preflight_evidence_path(
            spec,
            "certificate",
            pid=state.pid,
            start_ticks=state.process_start_ticks,
        )
        certificate_held = create_json_artifact_exclusive_held(
            certificate_path,
            preflight_certificate_payload(spec, state, torch_module),
        )
        certificate_sha256 = verify_held_artifact(
            certificate_held,
            "preflight certificate before formal marker",
        )
        phase = "formal_attempt_marker_persistence"
        validate_process_root_lock(process_lock)
        verify_held_artifact(
            certificate_held,
            "preflight certificate immediately before formal marker",
        )
        marker_payload = formal_marker_payload(
            spec,
            state,
            certificate_path,
            certificate_sha256,
        )
        marker_held = create_json_artifact_exclusive_held(
            spec.marker,
            marker_payload,
        )
        marker_sha256 = verify_held_artifact(
            marker_held,
            "formal attempt marker after persistence",
        )
    except BaseException as error:
        if (
            marker_held is None
            and state is not None
            and marker_payload is not None
        ):
            try:
                marker_held = recover_expected_json_artifact_held(
                    spec.marker,
                    marker_payload,
                )
            except BaseException as recovery_error:
                error.add_note(
                    "formal marker publication was indeterminate and the "
                    "exact expected inode could not be recovered: "
                    f"{type(recovery_error).__name__}: {recovery_error}"
                )
        if marker_held is not None and state is not None:
            try:
                validate_process_root_lock(process_lock)
                observed_marker_sha256 = verify_held_artifact(
                    marker_held,
                    "formal marker before terminal preflight failure",
                )
                if certificate_held is not None:
                    verify_held_artifact(
                        certificate_held,
                        "preflight certificate before terminal failure",
                    )
                atomic_write_json_exclusive(
                    spec.failure_result,
                    failure_payload(
                        spec,
                        state,
                        observed_marker_sha256,
                        None,
                        (),
                        error,
                    ),
                )
            except BaseException as write_error:
                error.add_note(
                    "the formal attempt is consumed; terminal failure "
                    "evidence could not be completed: "
                    f"{type(write_error).__name__}: {write_error}"
                )
        else:
            failure_path = preflight_evidence_path(
                spec,
                "infrastructure-failure",
                pid=os.getpid(),
                start_ticks=process_start_ticks(),
            )
            try:
                validate_process_root_lock(process_lock)
                atomic_write_json_exclusive(
                    failure_path,
                    preflight_failure_payload(
                        spec,
                        phase=phase,
                        error=error,
                    ),
                )
            except BaseException as write_error:
                error.add_note(
                    "preflight infrastructure failure evidence could not be "
                    "written exclusively: "
                    f"{type(write_error).__name__}: {write_error}"
                )
        if marker_held is not None:
            marker_held.close()
        if certificate_held is not None:
            certificate_held.close()
        raise

    completed: list[dict[str, Any]] = []
    current_panel: PanelSpec | None = None
    panel_failures: list[dict[str, Any]] = []
    try:
        # Every declared panel is attempted exactly once. A runtime failure in
        # panel one is recorded, but cannot suppress the required second panel.
        for current_panel in spec.panels:
            validate_process_root_lock(process_lock)
            verify_held_artifact(
                certificate_held,
                f"preflight certificate before panel {current_panel.order}",
            )
            verify_held_artifact(
                marker_held,
                f"formal marker before panel {current_panel.order}",
            )
            try:
                result, output_sha256 = run_panel(
                    spec,
                    current_panel,
                    state,
                    evaluator_module,
                )
            except BaseException as panel_error:
                panel_failures.append(
                    {
                        "order": current_panel.order,
                        "name": current_panel.name,
                        "data_panel": current_panel.data_panel,
                        "error_type": type(panel_error).__name__,
                        "error_message": str(panel_error),
                        "traceback": traceback.format_exc(),
                    }
                )
                continue
            completed_rows, _ = _strict_output_integer_observation(
                result,
                GATE_METRIC_PATHS["rows"],
            )
            completed.append({
                    "order": current_panel.order,
                    "name": current_panel.name,
                    "data_panel": current_panel.data_panel,
                    "output": str(current_panel.output),
                    "output_sha256": output_sha256,
                    "rows": completed_rows,
                })
        validate_process_root_lock(process_lock)
        verify_held_artifact(
            certificate_held,
            "preflight certificate before terminal evaluation",
        )
        verify_held_artifact(
            marker_held,
            "formal marker before terminal evaluation",
        )
        if panel_failures:
            current_panel = None
            raise PanelExecutionAggregateError(panel_failures)
        current_panel = None
        gate_report = evaluate_persisted_behavior_gates(spec, completed)
        validate_process_root_lock(process_lock)
        verify_held_artifact(
            certificate_held,
            "preflight certificate immediately before terminal success",
        )
        verify_held_artifact(
            marker_held,
            "formal marker immediately before terminal success",
        )
        success_held = create_json_artifact_exclusive_held(
            spec.success_result,
            success_payload(
                spec,
                state,
                certificate_path=certificate_path,
                certificate_sha256=certificate_sha256,
                marker_sha256=marker_sha256,
                completed_panels=completed,
                gate_report=gate_report,
            ),
        )
        try:
            verify_held_artifact(success_held, "terminal success result")
            verify_held_artifact(
                certificate_held,
                "preflight certificate at terminal success",
            )
            validate_process_root_lock(process_lock)
        finally:
            success_held.close()
    except BaseException as error:
        terminal_error: BaseException = error
        try:
            validate_process_root_lock(process_lock)
            verify_held_artifact(
                certificate_held,
                "preflight certificate at terminal failure",
            )
            verify_held_artifact(
                marker_held,
                "formal marker at terminal failure",
            )
        except BaseException as integrity_error:
            integrity_error.add_note(
                "terminal failure was triggered while handling "
                f"{type(error).__name__}: {error}"
            )
            terminal_error = integrity_error
        failure = failure_payload(
            spec,
            state,
            marker_sha256,
            current_panel,
            completed,
            terminal_error,
        )
        try:
            failure_held = create_json_artifact_exclusive_held(
                spec.failure_result,
                failure,
            )
            try:
                verify_held_artifact(
                    failure_held,
                    "terminal failure result",
                )
            finally:
                failure_held.close()
        except BaseException as write_error:
            error.add_note(
                "terminal failure record could not be written exclusively: "
                f"{type(write_error).__name__}: {write_error}"
            )
        marker_held.close()
        certificate_held.close()
        if terminal_error is not error:
            raise terminal_error from error
        raise
    marker_held.close()
    certificate_held.close()
    return completed


def execute(
    preregistration_path: Path,
    expected_preregistration_sha256: str,
    *,
    evaluator_module: Any = policy_eval,
    torch_module: Any = torch,
) -> list[dict[str, Any]]:
    process_lock = acquire_process_execution_lock(
        preregistration_path,
        expected_preregistration_sha256,
    )
    try:
        locked = open_locked_verified_preregistration(
            preregistration_path,
            process_lock.preregistration_sha256,
        )
        try:
            return _execute_locked(
                locked,
                process_lock,
                evaluator_module=evaluator_module,
                torch_module=torch_module,
            )
        finally:
            locked.close()
    finally:
        process_lock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two preregistered policy behavior panels in one CUDA-owning "
            "process with a durable preflight/formal-attempt boundary."
        )
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        required=True,
        help="Checkpoint-bound behavior execution preregistration JSON.",
    )
    parser.add_argument(
        "--expected-preregistration-sha256",
        required=True,
        help="Expected raw-byte SHA256 of --preregistration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execute(
        args.preregistration,
        args.expected_preregistration_sha256,
    )


if __name__ == "__main__":
    main()
