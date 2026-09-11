#!/usr/bin/env python3
"""Frozen-beam, parent-only cohort atlas on one exact guarded-U468 B256 graph.

Run mode is not authorized by the master/correction alone.  A separate frozen
execution preregistration must bind the final mode-0555 runner SHA, correction,
exact invocation, absent O_EXCL targets, and sole attempt before run mode may
consume its marker.  The atlas only profiles the unchanged parent.  It never
writes parameters, forms a model overlay, performs a candidate forward, or
authorizes a candidate.

Static mode imports no ML runtime and writes nothing.  Formal run mode creates
the attempt marker before CUDA/model work, uses exactly one native-BF16 forward
of the exact frozen B256 batch, and publishes one immutable parent-only result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.machinery
import json
import math
import os
import random
import stat
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "profile_u456_g8_guarded_u468_currentparent_cohortatlas_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
BRANCH = "ppo_u456_g8_guarded3x4x96_currentparent_cohortatlas_design202608144"
SCHEMA = "ptcg-u456-g8-guarded-u468-currentparent-cohortatlas-v1"
ATTEMPT_SCHEMA = f"{SCHEMA}-attempt-v1"
SUMMARY_SCHEMA = f"{SCHEMA}-publication-summary-v1"
EXECUTION_SCHEMA = f"{SCHEMA}-execution-preregistration-v1"
ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-attempt.json"
RESULT = ROOT / f"artifacts/{BRANCH}.atlas.json"
EXECUTION_PREREG = ROOT / f"artifacts/{BRANCH}.execution_preregistration.json"

MASTER = ROOT / f"artifacts/{BRANCH}.master_preregistration.json"
MASTER_SHA256 = "178211b6c43e4a8b2510abee0fd5808caf4392a114bd62e5201bb32c057b35f2"
MASTER_SCHEMA = (
    "ptcg-u456-g8-guarded-u468-currentparent-cohortatlas-master-preregistration-v1"
)
CORRECTION = ROOT / f"artifacts/{BRANCH}.master_correction_v2.json"
CORRECTION_SHA256 = "67f7fac3229c3201cea6c5c8e3e5f8cf153de643b7d2f33c11136e5a7b95cc84"
CORRECTION_SCHEMA = (
    "ptcg-u456-g8-guarded-u468-currentparent-cohortatlas-master-correction-v2"
)
CORRECTION_V3 = ROOT / f"artifacts/{BRANCH}.master_correction_v3.json"
CORRECTION_V3_SHA256 = (
    "1a15366c929b358705774ef9d5952d4818de03051813bef49d6ad850667b864c"
)
CORRECTION_V3_SCHEMA = (
    "ptcg-u456-g8-guarded-u468-currentparent-cohortatlas-master-correction-v3"
)
CORRECTION_V4 = ROOT / f"artifacts/{BRANCH}.master_correction_v4.json"
CORRECTION_V4_SHA256 = (
    "1c0a3637c7647931a8d10522c2015ecc724a2c1dadc2ddb97934765a030f0378"
)
CORRECTION_V4_SCHEMA = (
    "ptcg-u456-g8-guarded-u468-currentparent-cohortatlas-master-correction-v4"
)
CORRECTION_V5 = ROOT / f"artifacts/{BRANCH}.master_correction_v5.json"
CORRECTION_V5_SHA256 = (
    "f5e3378fd0f236796d5b133abb83945a57527428ca0e4941b0851567412369c8"
)
CORRECTION_V5_SCHEMA = (
    "ptcg-u456-g8-guarded-u468-currentparent-cohortatlas-master-correction-v5"
)
CORRECTION_V6 = ROOT / f"artifacts/{BRANCH}.master_correction_v6.json"
CORRECTION_V6_SHA256 = (
    "2c2cd953b35be60508e196ea09e9fe02e03587db47ee2da45924154b896edf54"
)
CORRECTION_V6_SCHEMA = (
    "ptcg-u456-g8-guarded-u468-currentparent-cohortatlas-master-correction-v6"
)
CORRECTION_V7 = ROOT / f"artifacts/{BRANCH}.master_correction_v7.json"
CORRECTION_V7_SHA256 = (
    "28c65580524d7cd7e7acdc7a74c914c2fa3d22eafe449138211785a24ba0a1ce"
)
CORRECTION_V7_SCHEMA = (
    "ptcg-u456-g8-guarded-u468-currentparent-cohortatlas-master-correction-v7"
)
LEDGER = ROOT / (
    "artifacts/official_unique_changed_candidate_ledger_before_"
    "cohortatlas_design202608144.json"
)
LEDGER_SHA256 = "0b5d87b7f7bc3f794cdf4b8f386382e5287fe401629cb01ae2e5d162857e7fb3"
LEDGER_SCHEMA = "ptcg-official-unique-changed-candidate-ledger-before-cohortatlas-v1"
LEDGER_STATUS = "LOCKED_LEDGER_SNAPSHOT_BEFORE_ZERO_CANDIDATE_COHORT_ATLAS"
P12_DECISION = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141."
    "fulltrain24050_decision.json"
)
P12_DECISION_SHA256 = "5ea614fcb682b7d7321c894ed55e56b051d50db4a8667371e0b163b9b9541155"
BLOCK3_PROVENANCE = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141."
    "block3_training_integrity_decision.json"
)
BLOCK3_PROVENANCE_SHA256 = (
    "d57954cdaeed5d0a2ff5c2a8d459253ba03f1b9a0f8dd1b5c760b23fc6f026bf"
)
PROFILE = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_profile_selection_"
    "design202608143_v3.json"
)
PROFILE_SHA256 = "b7ef45722e50b2c71377b4dce8c7c2cee6ee1aaa16e99a1b874a0d932dab2d4e"
PROFILE_SCHEMA = "ptcg-u456-g8-guarded-u468-currentparent-train-profile-selection-v3"
PROFILE_STATUS = "completed_frozen_parent_only_train_profile_selection"

CURRENT = TOOLS / "run_u456_g8_guarded_u468_currentparent_boundaryqp_specialbc_v1.py"
CURRENT_SHA256 = "d19c5338329365cc4bb4ba8586df29aa1c616588c6ae5640265e0391e73b4c85"
FORMAL = TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal_v3.py"
FORMAL_SHA256 = "2d12ce9e76f6672911948dcc6458573899ec45fc39e748510544001025ede939"
CW20 = TOOLS / "probe_u468_raw_fixed_pcgrad_specialbc_cw20_v1.py"
CW20_SHA256 = "8f5b64c8a04b3de9a6d2890aac582967a50f4d708f3a11a64cbd395423d907d2"
TRAIN_BC = TOOLS / "train_bc_orbit.py"
TRAIN_BC_SHA256 = "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
TRAIN_PPO = TOOLS / "train_ppo.py"
TRAIN_PPO_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
EVALUATOR = TOOLS / "evaluate_policy_bc.py"
EVALUATOR_SHA256 = "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4"
HELPER = TOOLS / "run_u468_raw_balanced_train_only_endpoint_gate.py"
HELPER_SHA256 = "84b51d8ef51e2184271c4bf358e5d59b74ea631daa79b84d73fffaf44a37037d"
HELPER_PRIVATE_NAME = "cohort_atlas_hash_bound_helper"
HELPER_STACK_NAMES = (
    "train_bc_orbit",
    "cg",
    "cg.sim",
    "train_ppo",
    "evaluate_policy_bc",
    HELPER_PRIVATE_NAME,
)

RUNNER_SHA_PLACEHOLDER = "__FROZEN_RUNNER_SHA256__"
EXECUTION_PREREG_SHA_PLACEHOLDER = "__EXECUTION_PREREG_SHA256_SELF__"
ABSOLUTE_COMMAND_TOKEN_TEMPLATE = (
    str(EXPECTED_PYTHON),
    "-I",
    "-B",
    str(SCRIPT),
    "--mode",
    "run",
    "--profile",
    str(PROFILE),
    "--profile-sha256",
    PROFILE_SHA256,
    "--expected-self-sha256",
    RUNNER_SHA_PLACEHOLDER,
    "--execution-prereg-sha256",
    EXECUTION_PREREG_SHA_PLACEHOLDER,
)

PARENT = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141/ppo_stage/"
    "block3/B_gold_league/seed-202608141/checkpoints/update-0468.pt"
)
PARENT_FILE_SHA256 = "a9290745b8eb58704c4617594ac5ab49500c6e65ecd9f273a8285c12547fbb83"
PARENT_MODEL_SHA256 = "832c724277d6c2263d76ddbf41622f4ba06621ec3155fa8b7ae2591203db1c5e"
PARENT_ACTOR_BYTES_SHA256 = "03822d2e9dfd0984349384a22bc4ed22b608dbd7c4a05183ff8e507f773dc8a"
PARENT_ACTOR_STATE_SHA256 = "b8d28f848f33003c5031d622bf04ddbd6c674f998b8e9d0f34564f8011dd4e23"
PARENT_NONACTOR_SHA256 = "ce84a183fbe82d40861938ba3cc8f1fa93b19ff53aa14ecbc8eee18ccd00db87"

SEED = 202608144
B256_SIZE = 256
PF_POOL_SIZE = 64
SEARCH_POOL_SIZE = 63
RETENTION_SIZE = 160
DOMINIC_SIZE = 32
COHORT_SIZE = 5
BEAM_WIDTH = 512
ACTOR_DIMENSION = 65793
ACTOR_L2_CAP = 1.0e-3
ATLAS_L2_CEILING = 9.0e-4
NATIVE_TARGET_MARGIN = 1.0 / 512.0
LINEAR_TARGET_MARGIN = 3.0 / 1024.0
ZERO_GUARD_BUFFER = 1.0 / 2048.0
PF64_NLL_BUFFER = 5.0e-5
RETENTION160_NLL_BUFFER = 5.0e-5
DOMINIC32_NLL_BUFFER = 0.0
TARGET5_NLL_BUFFER = 1.0e-4
QP_FTOL = 1.0e-12
QP_MAXITER = 5000
CONSTRAINT_TOLERANCE = 1.0e-10
L2_TOLERANCE = 1.0e-12
CONDITION_LIMIT = 1.0e6
MAX_PRIMARY_CUTS = 160
MAX_FINALISTS = 512
EXPECTED_B256_SHA256 = "6f0d8bcdc10b782c676d743a00a2f9e2dcc0501ea1054398c880aee31212f325"
EXPECTED_B256_SAMPLE_WEIGHTS_F4_SHA256 = (
    "ee49d23c55e69f9b9b5a9ad853073c06326a9c4aff87c36f708f064cea29a5e6"
)
CONTEXT34_SAMPLE_WEIGHT_FLOAT32 = 0.3333333432674408
OLD_IMPOSSIBLE_LINE = "498a7f841b66e44756c04e93312d683c979c201d8eec7713823876e3cdc3fda7"
OLD_TARGET_SET = frozenset(
    {
        "d1d1aba7ae2c6949e30c6a4b6d1292cc9459f867b550e3b9b71b969fd74c0f54",
        OLD_IMPOSSIBLE_LINE,
        "5994ba14fd56d323743e2ba297621bb1e3385dde20116b5ae33f8ad9c7ffbb36",
        "2ed31b529dee80b049e76f753b2c4c5b2f33cc32a9f5bdace7566da616ca2cac",
        "6f1591a836cfa2858da431140641f6ee01db6684b4fc6a1a81caea003c5cfbd4",
    }
)
ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
CERTIFIED_STATUS = "CERTIFIED_COHORT_FOUND_BY_FROZEN_BEAM_ZERO_CANDIDATE"
NO_CERTIFICATE_STATUS = "NO_CERTIFIED_COHORT_FOUND_BY_FROZEN_BEAM_ZERO_CANDIDATE"
EXECUTION_FAILED_STATUS = "ATLAS_EXECUTION_FAILED_ATTEMPT_CONSUMED"
ENGINE_SENTINEL_MESSAGE = "cohort-atlas engine access is forbidden"


class ProtocolError(RuntimeError):
    """Fail-closed protocol error."""


class _EngineSentinel:
    """Unique process-local object that makes every engine access fail closed."""

    __slots__ = ()

    def __getattribute__(self, _name: str) -> Any:
        raise ProtocolError(ENGINE_SENTINEL_MESSAGE)


class PublicationDirectories:
    """Held directory descriptors for the two one-shot publications."""

    def __init__(
        self,
        root_fd: int,
        artifacts_fd: int,
        root_evidence: Mapping[str, Any],
        artifacts_evidence: Mapping[str, Any],
    ) -> None:
        self.root_fd = root_fd
        self.artifacts_fd = artifacts_fd
        self.root_evidence = dict(root_evidence)
        self.artifacts_evidence = dict(artifacts_evidence)
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        for descriptor in (self.artifacts_fd, self.root_fd):
            try:
                os.close(descriptor)
            except BaseException as error:
                errors.append(error)
        self.closed = True
        if errors:
            raise ProtocolError("held publication directory close failed")


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ProtocolError(f"{label}: nonfinite JSON constant {value}")

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ProtocolError(f"{label}: root must be an object")
    canonical_json(value)
    return value


def read_regular_bytes(
    path: Path,
    expected_sha256: str | None,
    label: str,
    *,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ProtocolError(f"{label}: not a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    visible = os.lstat(path)
    payload = b"".join(chunks)
    held = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    visible_identity = (
        visible.st_dev,
        visible.st_ino,
        visible.st_size,
        visible.st_mtime_ns,
    )
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != held
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or int(visible.st_nlink) != 1
        or visible_identity != held
        or len(payload) != int(after.st_size)
    ):
        raise ProtocolError(f"{label}: held-fd identity changed")
    digest = sha256_bytes(payload)
    mode = stat.S_IMODE(after.st_mode)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProtocolError(f"{label}: SHA-256 drift: {digest}")
    if expected_mode is not None and mode != expected_mode:
        raise ProtocolError(f"{label}: mode drift: {oct(mode)}")
    try:
        relative = str(path.relative_to(ROOT))
    except ValueError:
        relative = str(path)
    return payload, {
        "path": relative,
        "absolute_path": str(path),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "mtime_ns": int(after.st_mtime_ns),
        "held_fd_identity_exact": True,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def publication_target(path: Path, directories: PublicationDirectories) -> tuple[int, str]:
    if path == ATTEMPT_MARKER:
        return directories.root_fd, ATTEMPT_MARKER.name
    if path == RESULT:
        return directories.artifacts_fd, RESULT.name
    raise ProtocolError("publication target is not one of the two frozen paths")


def held_target_absent(directory_fd: int, leaf: str) -> bool:
    if not leaf or "/" in leaf or leaf in {".", ".."}:
        raise ProtocolError("invalid held-directory publication leaf")
    try:
        os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    return False


def publication_absence(
    directories: PublicationDirectories | None = None,
) -> dict[str, Any]:
    if directories is None:
        attempt_absent = not (
            ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink()
        )
        result_absent = not (RESULT.exists() or RESULT.is_symlink())
        mechanism = "path_lstat_static_only"
    else:
        attempt_absent = held_target_absent(
            directories.root_fd, ATTEMPT_MARKER.name
        )
        result_absent = held_target_absent(directories.artifacts_fd, RESULT.name)
        mechanism = "held_directory_fd_fstatat_no_follow"
    return {
        "attempt_absent": attempt_absent,
        "result_absent": result_absent,
        "both_absent": attempt_absent and result_absent,
        "mechanism": mechanism,
    }


def directory_evidence(
    descriptor: int,
    absolute_path: Path,
    relative_path: str,
) -> dict[str, Any]:
    held = os.fstat(descriptor)
    visible = os.lstat(absolute_path)
    held_identity = (
        held.st_dev,
        held.st_ino,
        held.st_mode,
        held.st_nlink,
    )
    visible_identity = (
        visible.st_dev,
        visible.st_ino,
        visible.st_mode,
        visible.st_nlink,
    )
    if (
        not stat.S_ISDIR(held.st_mode)
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISDIR(visible.st_mode)
        or held_identity != visible_identity
    ):
        raise ProtocolError("held publication directory identity drift")
    return {
        "relative_path": relative_path,
        "absolute_path": str(absolute_path),
        "device": int(held.st_dev),
        "inode": int(held.st_ino),
        "mode_octal": format(stat.S_IMODE(held.st_mode), "04o"),
        "nlink": int(held.st_nlink),
        "held_directory_identity_exact": True,
    }


def directory_binding_matches(
    expected: Any,
    actual: Mapping[str, Any],
) -> bool:
    if not isinstance(expected, Mapping):
        return False
    return all(
        expected.get(key) == actual.get(key)
        for key in (
            "relative_path",
            "absolute_path",
            "device",
            "inode",
            "mode_octal",
            "nlink",
        )
    )


def open_publication_directories(
    binding: Mapping[str, Any],
) -> PublicationDirectories:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_fd = os.open(ROOT, flags)
    try:
        artifacts_fd = os.open("artifacts", flags, dir_fd=root_fd)
    except BaseException:
        os.close(root_fd)
        raise
    try:
        root_evidence = directory_evidence(root_fd, ROOT, ".")
        artifacts_evidence = directory_evidence(
            artifacts_fd, ROOT / "artifacts", "artifacts"
        )
        checks = {
            "workspace_root": directory_binding_matches(
                binding.get("workspace_root"), root_evidence
            ),
            "artifacts_directory": directory_binding_matches(
                binding.get("artifacts_directory"), artifacts_evidence
            ),
        }
        if not all(checks.values()):
            raise ProtocolError("execution preregistration directory identity drift")
        return PublicationDirectories(
            root_fd,
            artifacts_fd,
            root_evidence,
            artifacts_evidence,
        )
    except BaseException:
        os.close(artifacts_fd)
        os.close(root_fd)
        raise


def read_regular_bytes_at(
    directory_fd: int,
    leaf: str,
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    expected_mode: int,
) -> tuple[bytes, dict[str, Any]]:
    if not leaf or "/" in leaf or leaf in {".", ".."}:
        raise ProtocolError(f"{label}: invalid held-directory leaf")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(leaf, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ProtocolError(f"{label}: not a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    visible = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    payload = b"".join(chunks)
    held_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != held_identity
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or int(visible.st_nlink) != 1
        or (
            visible.st_dev,
            visible.st_ino,
            visible.st_size,
            visible.st_mtime_ns,
        )
        != held_identity
        or len(payload) != int(after.st_size)
    ):
        raise ProtocolError(f"{label}: held-directory file identity drift")
    digest = sha256_bytes(payload)
    mode = stat.S_IMODE(after.st_mode)
    if digest != expected_sha256 or mode != expected_mode:
        raise ProtocolError(f"{label}: publication SHA or mode drift")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "absolute_path": str(path),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "mtime_ns": int(after.st_mtime_ns),
        "held_directory_fd_identity_exact": True,
    }


def publish_o_excl(
    path: Path,
    payload: bytes,
    directories: PublicationDirectories,
) -> dict[str, Any]:
    if canonical_json(strict_json(payload, f"publication {path.name}")) != payload:
        raise ProtocolError(f"publication is not canonical JSON: {path}")
    directory_fd, leaf = publication_target(path, directories)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(leaf, flags, 0o444, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise ProtocolError(f"short publication write: {path}")
            offset += written
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        final = os.fstat(fd)
    finally:
        os.close(fd)
    os.fsync(directory_fd)
    if stat.S_IMODE(final.st_mode) != 0o444 or int(final.st_size) != len(payload):
        raise ProtocolError(f"publication mode/size drift: {path}")
    _, evidence = read_regular_bytes_at(
        directory_fd,
        leaf,
        path,
        sha256_bytes(payload),
        f"published {path.name}",
        expected_mode=0o444,
    )
    return evidence


def audit_train_ppo_engine_safety(payload: bytes) -> dict[str, Any]:
    tree = ast.parse(payload.decode("utf-8"), filename=str(TRAIN_PPO))
    exact_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "cg.sim"
        and [(alias.name, alias.asname) for alias in node.names] == [("lib", None)]
    ]

    class LibUseVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_body_depth = 0
            self.load_lines: list[int] = []
            self.top_level_load_lines: list[int] = []

        def visit_Name(self, node: ast.Name) -> None:
            if node.id == "lib" and isinstance(node.ctx, ast.Load):
                self.load_lines.append(int(node.lineno))
                if self.function_body_depth == 0:
                    self.top_level_load_lines.append(int(node.lineno))

        def _visit_function_metadata(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            for decorator in node.decorator_list:
                self.visit(decorator)
            for default in node.args.defaults:
                self.visit(default)
            for default in node.args.kw_defaults:
                if default is not None:
                    self.visit(default)
            for argument in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            ):
                if argument.annotation is not None:
                    self.visit(argument.annotation)
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                self.visit(node.args.vararg.annotation)
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                self.visit(node.args.kwarg.annotation)
            if node.returns is not None:
                self.visit(node.returns)

        def _visit_function_body(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            self._visit_function_metadata(node)
            self.function_body_depth += 1
            try:
                for statement in node.body:
                    self.visit(statement)
            finally:
                self.function_body_depth -= 1

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function_body(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function_body(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            for default in node.args.defaults:
                self.visit(default)
            for default in node.args.kw_defaults:
                if default is not None:
                    self.visit(default)
            self.function_body_depth += 1
            try:
                self.visit(node.body)
            finally:
                self.function_body_depth -= 1

    visitor = LibUseVisitor()
    visitor.visit(tree)
    checks = {
        "exact_one_top_level_from_cg_sim_import_lib": len(exact_imports) == 1,
        "engine_uses_exist": bool(visitor.load_lines),
        "all_engine_loads_inside_function_or_method_body": not visitor.top_level_load_lines,
    }
    if not all(checks.values()):
        raise ProtocolError("held train_ppo engine AST gate failed")
    return {
        "pass": True,
        "checks": checks,
        "engine_global_load_count": len(visitor.load_lines),
        "module_top_level_engine_load_count": 0,
    }


def exec_held_module(
    path: Path,
    digest: str,
    name: str,
    expected_mode: int,
    installed: dict[str, ModuleType],
    installation_order: list[str],
    *,
    keep_registered: bool,
    payload_audit: Any = None,
) -> tuple[ModuleType, dict[str, Any]]:
    payload, evidence = read_regular_bytes(
        path, digest, name, expected_mode=expected_mode
    )
    if name in sys.modules:
        raise ProtocolError(f"verified execution name occupied: {name}")
    payload_audit_result = payload_audit(payload) if payload_audit is not None else None
    code = compile(
        payload,
        str(path),
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    spec = importlib.machinery.ModuleSpec(name, loader=None, origin=str(path))
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = spec
    module.__cached__ = None
    sys.modules[name] = module
    try:
        exec(code, module.__dict__, module.__dict__)
        if sys.modules.get(name) is not module:
            raise ProtocolError(f"verified module replaced its registration: {name}")
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise
    if keep_registered:
        installed[name] = module
        installation_order.append(name)
    else:
        if sys.modules.get(name) is not module:
            raise ProtocolError(f"verified module registration drift: {name}")
        sys.modules.pop(name)
    evidence = dict(evidence)
    evidence.update(
        {
            "compiled_from_exact_held_bytes": True,
            "executed_exact_held_bytes": True,
            "compile_optimize": 0,
            "repository_path_reopened_for_execution": False,
            "registered_after_exec": keep_registered,
        }
    )
    if payload_audit_result is not None:
        evidence["payload_static_audit"] = payload_audit_result
    return module, evidence


def install_engine_sentinel(
    installed: dict[str, ModuleType],
    installation_order: list[str],
) -> tuple[Any, dict[str, Any]]:
    for name in ("cg", "cg.sim"):
        if name in sys.modules:
            raise ProtocolError(f"engine sentinel module name occupied: {name}")
    sentinel = _EngineSentinel()
    package = ModuleType("cg")
    package_spec = importlib.machinery.ModuleSpec(
        "cg", loader=None, origin="<cohort-atlas-engine-sentinel>", is_package=True
    )
    package_spec.submodule_search_locations = []
    package.__file__ = None
    package.__package__ = "cg"
    package.__loader__ = None
    package.__spec__ = package_spec
    package.__path__ = []
    simulation = ModuleType("cg.sim")
    simulation_spec = importlib.machinery.ModuleSpec(
        "cg.sim", loader=None, origin="<cohort-atlas-engine-sentinel>"
    )
    simulation.__file__ = None
    simulation.__package__ = "cg"
    simulation.__loader__ = None
    simulation.__spec__ = simulation_spec
    simulation.lib = sentinel
    package.sim = simulation
    sys.modules["cg"] = package
    installed["cg"] = package
    installation_order.append("cg")
    try:
        sys.modules["cg.sim"] = simulation
        installed["cg.sim"] = simulation
        installation_order.append("cg.sim")
    except BaseException:
        if sys.modules.get("cg") is package:
            sys.modules.pop("cg", None)
        installed.pop("cg", None)
        if installation_order and installation_order[-1] == "cg":
            installation_order.pop()
        raise
    return sentinel, {
        "cg_package_process_local": True,
        "cg_sim_process_local": True,
        "lib_unique_sentinel": True,
        "sentinel_fixed_failure_message_sha256": sha256_bytes(
            ENGINE_SENTINEL_MESSAGE.encode("utf-8")
        ),
        "cg_source_bytecode_or_native_library_opened": False,
        "atlas_engine_calls": 0,
    }


def load_helper_stack(
    torch: Any,
    installed: dict[str, ModuleType],
    installation_order: list[str],
) -> tuple[ModuleType, dict[str, Any]]:
    collisions = [name for name in HELPER_STACK_NAMES if name in sys.modules]
    if collisions:
        raise ProtocolError("private helper-stack module collision")
    bc, bc_evidence = exec_held_module(
        TRAIN_BC,
        TRAIN_BC_SHA256,
        "train_bc_orbit",
        0o664,
        installed,
        installation_order,
        keep_registered=True,
    )
    sentinel, sentinel_evidence = install_engine_sentinel(
        installed, installation_order
    )
    ppo, ppo_evidence = exec_held_module(
        TRAIN_PPO,
        TRAIN_PPO_SHA256,
        "train_ppo",
        0o664,
        installed,
        installation_order,
        keep_registered=True,
        payload_audit=audit_train_ppo_engine_safety,
    )
    evaluator, evaluator_evidence = exec_held_module(
        EVALUATOR,
        EVALUATOR_SHA256,
        "evaluate_policy_bc",
        0o664,
        installed,
        installation_order,
        keep_registered=True,
    )
    helper, helper_evidence = exec_held_module(
        HELPER,
        HELPER_SHA256,
        HELPER_PRIVATE_NAME,
        0o555,
        installed,
        installation_order,
        keep_registered=True,
    )
    checks = {
        "six_private_modules_installed": tuple(installation_order)
        == HELPER_STACK_NAMES,
        "ppo_lib_is_unique_sentinel": getattr(ppo, "lib", None) is sentinel,
        "ppo_bc_entity_policy_identity": getattr(ppo, "EntityOptionPolicy", None)
        is getattr(bc, "EntityOptionPolicy", None),
        "evaluator_bc_identity": getattr(evaluator, "bc", None) is bc,
        "evaluator_ppo_identity": getattr(evaluator, "ppo", None) is ppo,
        "helper_bc_identity": getattr(helper, "bc", None) is bc,
        "helper_ppo_identity": getattr(helper, "ppo", None) is ppo,
        "helper_evaluator_identity": getattr(helper, "evaluator", None) is evaluator,
        "helper_torch_runtime_identity": getattr(helper, "torch", None) is torch,
        "checkpoint_from_bytes_present": callable(
            getattr(helper, "checkpoint_from_bytes", None)
        ),
        "instantiate_checkpoint_present": callable(
            getattr(helper, "instantiate_checkpoint", None)
        ),
        "model_state_sha256_present": callable(
            getattr(helper, "model_state_sha256", None)
        ),
        "model_forward_present": callable(getattr(ppo, "model_forward", None)),
        "sample_ordered_actions_present": callable(
            getattr(ppo, "sample_ordered_actions", None)
        ),
    }
    if not all(checks.values()):
        raise ProtocolError("verified helper-stack identity closure failed")
    return helper, {
        "pass": True,
        "exact_load_order": list(HELPER_STACK_NAMES),
        "checks": checks,
        "train_bc_orbit": bc_evidence,
        "engine_sentinel": sentinel_evidence,
        "train_ppo": ppo_evidence,
        "evaluate_policy_bc": evaluator_evidence,
        HELPER_PRIVATE_NAME: helper_evidence,
        "ordinary_sys_path_repository_imports": 0,
        "formal_load_helper_called": False,
    }


def validate_runtime(require_cuda: bool) -> dict[str, Any]:
    checks = {
        "cwd_absolute_spelling_exact": str(Path.cwd()) == str(ROOT),
        "cwd_resolved_exact": Path.cwd().resolve() == ROOT,
        "python_absolute_spelling_exact": str(sys.executable) == str(EXPECTED_PYTHON),
        "python_resolved_exact_my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1
        and sys.dont_write_bytecode is True,
        "cublas_workspace_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
    }
    if not all(checks.values()):
        raise ProtocolError(f"runtime drift: {checks}")
    result: dict[str, Any] = {"checks": checks, "pass": True}
    if require_cuda:
        import torch

        properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "native_bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            "device_count_exact_1": torch.cuda.device_count() == 1,
            "current_device_exact_0": bool(
                torch.cuda.is_available() and torch.cuda.current_device() == 0
            ),
            "device_name_exact": bool(properties and properties.name == "NVIDIA GeForce RTX 5090"),
            "compute_capability_exact_12_0": bool(
                properties and (int(properties.major), int(properties.minor)) == (12, 0)
            ),
            "torch_version_exact": torch.__version__ == "2.8.0+cu128",
            "cuda_runtime_exact": torch.version.cuda == "12.8",
        }
        if not all(cuda.values()):
            raise ProtocolError(f"CUDA runtime drift: {cuda}")
        result["cuda"] = cuda
    return result


def static_source_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"), filename=str(SCRIPT))
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def owner(node: ast.AST) -> str | None:
        cursor: ast.AST | None = node
        while cursor is not None:
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cursor.name
            cursor = parents.get(cursor)
        return None

    forbidden_imports = {"requests", "urllib", "http", "socket", "subprocess", "kaggle"}
    forbidden_calls = {
        "backward",
        "step",
        "save",
        "savez",
        "copy_",
        "load_state_dict",
        "apply_flat_actor",
        "restore_actor",
        "submit",
        "upload",
        "unlink",
        "rename",
        "replace",
        "rmtree",
    }
    imports: list[tuple[str, int, str | None]] = []
    calls: list[tuple[str, int, str | None, ast.Call]] = []
    attribute_writes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno, owner(node)) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", node.lineno, owner(node)))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            calls.append((name, node.lineno, owner(node), node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for nested in ast.walk(target):
                    if isinstance(nested, ast.Attribute) and nested.attr in {
                        "data",
                        "weight",
                        "bias",
                    }:
                        attribute_writes.append(f"{nested.attr}@{node.lineno}")
    import_hits = sorted(
        f"{name}@{line}"
        for name, line, _ in imports
        if name.split(".", 1)[0] in forbidden_imports
    )
    top_torch = [
        f"{name}@{line}" for name, line, function in imports if name == "torch" and function is None
    ]
    call_hits = sorted(
        f"{name}@{line}" for name, line, _, _ in calls if name in forbidden_calls
    )
    write_hits = [
        f"{name}@{line}:{function}"
        for name, line, function, _ in calls
        if name in {"write", "fchmod"} and function != "publish_o_excl"
    ]
    publish_sites = []
    for name, line, function, node in calls:
        if name == "publish_o_excl":
            target = node.args[0].id if node.args and isinstance(node.args[0], ast.Name) else ""
            publish_sites.append((target, line, function))
    marker = [line for target, line, function in publish_sites if target == "ATTEMPT_MARKER" and function == "main"]
    result = [line for target, line, function in publish_sites if target == "RESULT" and function == "main"]
    atlas = [line for name, line, function, _ in calls if name == "run_atlas" and function == "main"]
    cuda = []
    for name, line, function, node in calls:
        if name != "validate_runtime" or function != "main":
            continue
        if any(
            keyword.arg == "require_cuda"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        ):
            cuda.append(line)
    model_forwards = [line for name, line, _, _ in calls if name == "model_forward"]
    grad_sites = [line for name, line, _, _ in calls if name == "grad"]
    source_text = source.decode("utf-8")
    external_string_literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and owner(node) != "static_source_audit"
    ]
    external_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and owner(node) != "static_source_audit"
    }
    function_source = {
        node.name: ast.get_source_segment(source_text, node) or ""
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    nonaudit_call_names = [
        name for name, _, function, _ in calls if function != "static_source_audit"
    ]
    linalg_norm_sites = [
        line
        for name, line, function, node in calls
        if function != "static_source_audit"
        and name == "norm"
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "np"
        and node.func.value.attr == "linalg"
    ]
    solve_source = function_source.get("solve_active_qp", "")
    l2_source = function_source.get("authoritative_vector_l2", "")
    selected_nll_source = function_source.get("selected_tuple_nll_constraint", "")
    primary_atlas_source = function_source.get("build_primary_atlas", "")
    full_scan_source = function_source.get("full_threat_scan", "")
    loader_source = function_source.get("exec_held_module", "")
    publisher_source = function_source.get("publish_o_excl", "")
    prereg_source = function_source.get("audit_execution_preregistration", "")
    main_source = function_source.get("main", "")
    failure_source = function_source.get("execution_failure_result", "")
    raw_error_string_sites = [
        line
        for name, line, function, node in calls
        if name == "str"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "error"
        and function != "sanitized_exception"
    ]
    checks = {
        "no_forbidden_imports": not import_hits,
        "no_top_level_torch": not top_torch,
        "no_forbidden_calls": not call_hits,
        "no_parameter_attribute_writes": not attribute_writes,
        "writes_only_in_o_excl_publisher": not write_hits,
        "exact_one_model_forward_source_site": len(model_forwards) == 1,
        "exact_one_autograd_grad_source_site": len(grad_sites) == 1,
        "exact_one_attempt_publication": len(marker) == 1,
        "exact_one_result_publication": len(result) == 1,
        "exact_one_run_atlas_call": len(atlas) == 1,
        "exact_one_cuda_validation": len(cuda) == 1,
        "marker_before_cuda_before_atlas_before_result": (
            len(marker) == len(cuda) == len(atlas) == len(result) == 1
            and marker[0] < cuda[0] < atlas[0] < result[0]
        ),
        "no_closed_result_path_literal": (
            not any(
                "currentparent_boundaryqp_design202608143.result.json" in value
                for value in external_string_literals
            )
        ),
        "no_microbatch_or_repack_helpers": (
            "MICROBATCH" not in external_names
            and "take_batch" not in nonaudit_call_names
            and "index_select" not in nonaudit_call_names
        ),
        "corrected_pool_constants": PF_POOL_SIZE == 64 and SEARCH_POOL_SIZE == 63,
        "corrected_zero_buffer": ZERO_GUARD_BUFFER == 1.0 / 2048.0,
        "domininc_nll_constraint_present": DOMINIC32_NLL_BUFFER == 0.0,
        "v5_weight_vector_bound": (
            CORRECTION_V5_SHA256 in external_string_literals
            and EXPECTED_B256_SAMPLE_WEIGHTS_F4_SHA256 in external_string_literals
        ),
        "authoritative_dot_sqrt_l2_only": (
            not linalg_norm_sites
            and "np.sqrt(squared)" in l2_source
            and "np.dot(vector, vector)" in l2_source
        ),
        "exact_vh_reduced_matrix": "reduced = matrix @ vh[:rank].T" in solve_source,
        "slsqp_success_and_zero_status": (
            "not bool(solution.success)" in solve_source
            and "int(solution.status) != 0" in solve_source
        ),
        "v5_selected_nll_left_fold": (
            "numerator = numerator + weight_float64 * gradient_float64"
            in selected_nll_source
            and "denominator = np.float64(denominator + weight_float64)"
            in selected_nll_source
            and "target_SHA_order_float64_left_fold" in selected_nll_source
        ),
        "one_shared_float32_margin_logit_cast": (
            primary_atlas_source.count('outputs["policy_logits"].float()') == 1
            and 'outputs["policy_logits"].float()' not in full_scan_source
            and 'primary["margin_logits_float32"]' in function_source.get(
                "certify_finalists", ""
            )
        ),
        "float64_matrix_rhs_hashes_present": (
            "matrix_float64_le_sha256" in solve_source
            and "rhs_float64_le_sha256" in solve_source
        ),
        "corrected_status_vocabulary": (
            CERTIFIED_STATUS in external_string_literals
            and NO_CERTIFICATE_STATUS in external_string_literals
            and EXECUTION_FAILED_STATUS in external_string_literals
            and not any(
                "NO_FEASIBLE_COHORT" in value for value in external_string_literals
            )
        ),
        "held_bytes_compile_exec_only": (
            len(
                [
                    1
                    for name, _, function, _ in calls
                    if name == "compile" and function == "exec_held_module"
                ]
            )
            == 1
            and len(
                [
                    1
                    for name, _, function, _ in calls
                    if name == "exec" and function == "exec_held_module"
                ]
            )
            == 1
            and "compile(\n        payload," in loader_source
            and "exec(code, module.__dict__, module.__dict__)" in loader_source
        ),
        "no_path_reopening_import_loader": (
            not any(
                name == "spec_from_" + "file_location"
                for name in nonaudit_call_names
            )
            and not any(name == "exec_" + "module" for name in nonaudit_call_names)
            and not any(
                name == "importlib." + "util" for name, _, _ in imports
            )
        ),
        "formal_v3_never_executed": not any(
            name == "load_helper" for name in nonaudit_call_names
        ),
        "held_directory_fd_publication": (
            "dir_fd=directory_fd" in publisher_source
            and "os.fsync(directory_fd)" in publisher_source
            and "directories" in publisher_source
        ),
        "absolute_invocation_and_proc_binding": (
            "ABSOLUTE_COMMAND_TOKEN_TEMPLATE" in prereg_source
            and "read_proc_self_cmdline" in prereg_source
            and "original_argv == expanded" in prereg_source
            and "runtime_argv == expanded[3:]" in prereg_source
            and "actual_script_gate" in prereg_source
        ),
        "outer_restoration_order_present": (
            main_source.find("restore_model_context")
            < main_source.find("restore_global_state")
            < main_source.find("cleanup_helper_stack")
            < main_source.find("restore_sys_path")
        ),
        "third_failure_status_not_no_certificate": (
            '"status": EXECUTION_FAILED_STATUS' in failure_source
            and '"certificate_status": None' in failure_source
            and '"do_not_interpret_as_no_certificate": True' in failure_source
        ),
        "caught_error_message_redacted": not raw_error_string_sites,
        "execution_prereg_gate_present": sum(
            name == "audit_execution_preregistration" for name in nonaudit_call_names
        )
        == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(
            "static source audit failed: "
            f"checks={checks} imports={import_hits} torch={top_torch} calls={call_hits} "
            f"attributes={attribute_writes} writes={write_hits}"
        )
    return {
        "pass": True,
        "checks": checks,
        "formal_order": {
            "attempt": marker[0],
            "cuda": cuda[0],
            "atlas": atlas[0],
            "result": result[0],
        },
        "model_forward_source_line": model_forwards[0],
        "autograd_grad_source_line": grad_sites[0],
        "publish_sites": [
            {"target": target, "line": line, "owner": function}
            for target, line, function in publish_sites
        ],
    }


def audit_control_plane() -> dict[str, Any]:
    master_payload, master_evidence = read_regular_bytes(
        MASTER, MASTER_SHA256, "cohort-atlas master", expected_mode=0o444
    )
    correction_payload, correction_evidence = read_regular_bytes(
        CORRECTION, CORRECTION_SHA256, "cohort-atlas correction v2", expected_mode=0o444
    )
    correction_v3_payload, correction_v3_evidence = read_regular_bytes(
        CORRECTION_V3,
        CORRECTION_V3_SHA256,
        "cohort-atlas correction v3",
        expected_mode=0o444,
    )
    correction_v4_payload, correction_v4_evidence = read_regular_bytes(
        CORRECTION_V4,
        CORRECTION_V4_SHA256,
        "cohort-atlas correction v4",
        expected_mode=0o444,
    )
    correction_v5_payload, correction_v5_evidence = read_regular_bytes(
        CORRECTION_V5,
        CORRECTION_V5_SHA256,
        "cohort-atlas correction v5",
        expected_mode=0o444,
    )
    correction_v6_payload, correction_v6_evidence = read_regular_bytes(
        CORRECTION_V6,
        CORRECTION_V6_SHA256,
        "cohort-atlas correction v6",
        expected_mode=0o444,
    )
    correction_v7_payload, correction_v7_evidence = read_regular_bytes(
        CORRECTION_V7,
        CORRECTION_V7_SHA256,
        "cohort-atlas correction v7",
        expected_mode=0o444,
    )
    p12_payload, p12_evidence = read_regular_bytes(
        P12_DECISION, P12_DECISION_SHA256, "P12 fulltrain decision", expected_mode=0o444
    )
    _, block3_evidence = read_regular_bytes(
        BLOCK3_PROVENANCE,
        BLOCK3_PROVENANCE_SHA256,
        "consumed block3 provenance",
        expected_mode=0o444,
    )
    master = strict_json(master_payload, "cohort-atlas master")
    correction = strict_json(correction_payload, "cohort-atlas correction v2")
    correction_v3 = strict_json(correction_v3_payload, "cohort-atlas correction v3")
    correction_v4 = strict_json(correction_v4_payload, "cohort-atlas correction v4")
    correction_v5 = strict_json(correction_v5_payload, "cohort-atlas correction v5")
    correction_v6 = strict_json(correction_v6_payload, "cohort-atlas correction v6")
    correction_v7 = strict_json(correction_v7_payload, "cohort-atlas correction v7")
    p12 = strict_json(p12_payload, "P12 fulltrain decision")
    expected_helper_stack = [
        {
            "module": "train_bc_orbit",
            "path": str(TRAIN_BC.relative_to(ROOT)),
            "sha256": TRAIN_BC_SHA256,
            "mode_octal": "0664",
        },
        {
            "module": "train_ppo",
            "path": str(TRAIN_PPO.relative_to(ROOT)),
            "sha256": TRAIN_PPO_SHA256,
            "mode_octal": "0664",
        },
        {
            "module": "evaluate_policy_bc",
            "path": str(EVALUATOR.relative_to(ROOT)),
            "sha256": EVALUATOR_SHA256,
            "mode_octal": "0664",
        },
        {
            "module": HELPER_PRIVATE_NAME,
            "path": str(HELPER.relative_to(ROOT)),
            "sha256": HELPER_SHA256,
            "mode_octal": "0555",
        },
    ]
    checks = {
        "master_schema": master.get("schema_version") == MASTER_SCHEMA,
        "master_branch": master.get("branch") == BRANCH,
        "correction_schema": correction.get("schema_version") == CORRECTION_SCHEMA,
        "correction_status": correction.get("status")
        == "locked_before_atlas_runner_freeze_execution_preregistration_cuda_or_parent_gradient_execution",
        "correction_binds_master": correction.get("master", {}).get("sha256")
        == MASTER_SHA256,
        "correction_v3_schema": correction_v3.get("schema_version")
        == CORRECTION_V3_SCHEMA,
        "correction_v3_status": correction_v3.get("status")
        == "locked_before_atlas_runner_freeze_execution_preregistration_cuda_or_parent_gradient_execution",
        "correction_v3_binds_v2": correction_v3.get("correction_v2", {}).get(
            "sha256"
        )
        == CORRECTION_SHA256,
        "correction_v4_schema": correction_v4.get("schema_version")
        == CORRECTION_V4_SCHEMA,
        "correction_v4_status": correction_v4.get("status")
        == "locked_before_atlas_runner_freeze_execution_preregistration_cuda_or_parent_gradient_execution",
        "correction_v4_binds_v3": correction_v4.get("correction_v3", {}).get(
            "sha256"
        )
        == CORRECTION_V3_SHA256,
        "correction_v5_schema": correction_v5.get("schema_version")
        == CORRECTION_V5_SCHEMA,
        "correction_v5_status": correction_v5.get("status")
        == "locked_before_atlas_runner_freeze_execution_preregistration_cuda_or_parent_gradient_execution",
        "correction_v5_binds_v4": correction_v5.get("correction_v4", {}).get(
            "sha256"
        )
        == CORRECTION_V4_SHA256,
        "design_static_only": correction.get("execution_authorization", {}).get(
            "this_master_and_correction_authorize_design_and_static_audit_only"
        )
        is True,
        "run_not_authorized": correction.get("execution_authorization", {}).get(
            "atlas_run_authorized_now"
        )
        is False,
        "v4_run_not_authorized": correction_v4.get(
            "execution_authorization", {}
        ).get("atlas_run_authorized_now")
        is False,
        "v5_run_not_authorized": correction_v5.get(
            "execution_authorization", {}
        ).get("atlas_run_authorized_now")
        is False,
        "correction_v6_schema": correction_v6.get("schema_version")
        == CORRECTION_V6_SCHEMA,
        "correction_v6_status": correction_v6.get("status")
        == "locked_before_revised_runner_freeze_execution_preregistration_or_atlas_execution",
        "correction_v6_binds_v5": correction_v6.get("correction_v5", {}).get(
            "sha256"
        )
        == CORRECTION_V5_SHA256,
        "correction_v6_helper_stack_exact": correction_v6.get(
            "verified_bytes_execution", {}
        ).get("helper_stack_exact_order")
        == expected_helper_stack,
        "correction_v6_failure_status": correction_v6.get(
            "restoration_and_failure_contract", {}
        ).get("execution_failure_status")
        == EXECUTION_FAILED_STATUS,
        "correction_v6_run_not_authorized": correction_v6.get(
            "authorization", {}
        ).get("atlas_run_authorized_now")
        is False,
        "correction_v7_schema": correction_v7.get("schema_version")
        == CORRECTION_V7_SCHEMA,
        "correction_v7_status": correction_v7.get("status")
        == "locked_before_revised_runner_freeze_execution_preregistration_or_atlas_execution",
        "correction_v7_binds_v6": correction_v7.get("correction_v6", {}).get(
            "sha256"
        )
        == CORRECTION_V6_SHA256,
        "correction_v7_command_template_exact": correction_v7.get(
            "authoritative_process_invocation", {}
        ).get("absolute_command_token_template")
        == list(ABSOLUTE_COMMAND_TOKEN_TEMPLATE),
        "correction_v7_placeholders_exact": correction_v7.get(
            "authoritative_process_invocation", {}
        ).get("placeholder_counts")
        == {
            RUNNER_SHA_PLACEHOLDER: 1,
            EXECUTION_PREREG_SHA_PLACEHOLDER: 1,
        },
        "correction_v7_ledger_exact": correction_v7.get(
            "authoritative_ledger_snapshot", {}
        )
        == {
            "path": str(LEDGER.relative_to(ROOT)),
            "sha256": LEDGER_SHA256,
            "mode_octal": "0444",
            "bytes": 3266,
            "required_status": LEDGER_STATUS,
            "required_entry_count": 2,
            "required_ordinals": [1, 2],
            "required_consumed": 2,
            "required_remaining": 10,
            "required_budget": 12,
            "required_atlas_consumption": 0,
            "transitive_gate": (
                "runner verifies every attempt, manifest, decision, and result "
                "path/SHA/mode/bytes named by the two ledger entries and checks "
                "their required status, candidate hash, specialist-consumed flag, "
                "and before/this/after counters"
            ),
        },
        "correction_v7_run_not_authorized": correction_v7.get(
            "authorization", {}
        ).get("atlas_run_authorized_now")
        is False,
        "execution_prereg_required": correction.get("execution_authorization", {}).get(
            "runner_may_not_run_before_execution_preregistration"
        )
        is True,
        "single_exact_b256_forward": correction.get("canonical_batch_and_pool", {}).get(
            "forward_count"
        )
        == 1,
        "batch_shape256": correction.get("canonical_batch_and_pool", {}).get(
            "forward_batch_size"
        )
        == B256_SIZE,
        "pool64_search63": (
            correction.get("canonical_batch_and_pool", {}).get("base_pool_size")
            == PF_POOL_SIZE
            and correction.get("canonical_batch_and_pool", {}).get(
                "search_pool_size_after_fixed_exclusion"
            )
            == SEARCH_POOL_SIZE
        ),
        "zero_buffer_authoritative": correction.get(
            "authoritative_buffers_and_objectives", {}
        ).get("zero_margin_retention_primary_change_minimum")
        == ZERO_GUARD_BUFFER,
        "domininc32_nondegrade": correction.get(
            "authoritative_buffers_and_objectives", {}
        ).get("Dominic_hard32_first_order_ordered_NLL_improvement_minimum")
        == DOMINIC32_NLL_BUFFER,
        "corrected_statuses": (
            correction.get("corrected_result_contract", {}).get("certificate_status")
            == CERTIFIED_STATUS
            and correction.get("corrected_result_contract", {}).get(
                "no_certificate_status"
            )
            == NO_CERTIFICATE_STATUS
        ),
        "p12_closed": p12.get("status") == "FULLTRAIN_GATE_FAIL_CLOSE_P12_LINEAGE",
        "guarded_parent_new_design_allowed": p12.get("decision", {}).get(
            "guarded_u468_parent_remains_eligible_for_a_separately_preregistered_new_special_bc_design"
        )
        is True,
        "failed_p12_not_parent": p12.get("decision", {}).get(
            "failed_exact_p12_candidate_may_be_training_parent"
        )
        is False,
        "branch_local_zero": correction.get("candidate_accounting_correction", {}).get(
            "new_atlas_branch_changed_candidates_before_master"
        )
        == 0,
        "global_ledger_2_of_12_remaining10": (
            correction.get("candidate_accounting_correction", {}).get(
                "global_official_unique_changed_candidates_consumed"
            )
            == 2
            and correction.get("candidate_accounting_correction", {}).get(
                "global_official_budget"
            )
            == 12
            and correction.get("candidate_accounting_correction", {}).get(
                "global_official_unique_changed_candidates_remaining"
            )
            == 10
        ),
        "v4_branch_local_zero_global_2_of_12_remaining10": (
            correction_v4.get("authoritative_candidate_accounting", {}).get(
                "atlas_branch_changed_candidates_constructed_or_evaluated"
            )
            == 0
            and correction_v4.get("authoritative_candidate_accounting", {}).get(
                "atlas_branch_official_unique_changed_candidates_consumed"
            )
            == 0
            and correction_v4.get("authoritative_candidate_accounting", {}).get(
                "global_official_unique_changed_candidates_consumed_before_atlas"
            )
            == 2
            and correction_v4.get("authoritative_candidate_accounting", {}).get(
                "global_official_budget"
            )
            == 12
            and correction_v4.get("authoritative_candidate_accounting", {}).get(
                "global_official_unique_changed_candidates_remaining"
            )
            == 10
        ),
        "v4_exact_reduced_matrix": correction_v4.get(
            "authoritative_norm_and_condition", {}
        ).get("reduced_matrix_restatement")
        == (
            "construct A_reduced exactly as A @ Vh[:rank].T; the algebraically "
            "equivalent U[:, :rank] * singular_values[:rank] expression is forbidden "
            "because it can differ in floating-point bytes"
        ),
        "v4_weighted_selected_tuple": correction_v4.get(
            "authoritative_ordered_NLL", {}
        ).get("selected_tuple_A_row")
        == (
            "for each selected tuple, take each stored little-endian float32 per-row "
            "NLL gradient, convert it elementwise to float64, and form negative "
            "sum(weight_float64 * gradient_float64) / sum(weight_float64); weights "
            "are exact float32 values converted elementwise to float64, denominator "
            "is finite and strictly positive"
        ),
        "v5_weight_vector_sha": correction_v5.get(
            "authoritative_B256_sample_weights", {}
        ).get("float32_le_sha256")
        == EXPECTED_B256_SAMPLE_WEIGHTS_F4_SHA256,
        "v5_weight_counts": (
            correction_v5.get("authoritative_B256_sample_weights", {}).get(
                "context34_count"
            )
            == 16
            and correction_v5.get("authoritative_B256_sample_weights", {}).get(
                "other_count"
            )
            == 240
            and correction_v5.get("authoritative_B256_sample_weights", {}).get(
                "context34_float32_value"
            )
            == CONTEXT34_SAMPLE_WEIGHT_FLOAT32
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"control-plane drift: {checks}")
    return {
        "pass": True,
        "checks": checks,
        "master": master_evidence,
        "correction_v2": correction_evidence,
        "correction_v3": correction_v3_evidence,
        "correction_v4": correction_v4_evidence,
        "correction_v5": correction_v5_evidence,
        "correction_v6": correction_v6_evidence,
        "correction_v7": correction_v7_evidence,
        "authoritative_post_P12_decision": p12_evidence,
        "consumed_block3_provenance": block3_evidence,
        "closed_boundary_result_opened": False,
        "old_delta_or_payload_read": False,
    }


def audit_dependencies() -> dict[str, Any]:
    result = {}
    for name, path, digest, mode in (
        ("current_parent_profile_math", CURRENT, CURRENT_SHA256, 0o555),
        (
            "formal_v3_provenance_read_not_executed",
            FORMAL,
            FORMAL_SHA256,
            0o555,
        ),
        ("actor_codec_cw20", CW20, CW20_SHA256, 0o555),
        ("train_bc_orbit", TRAIN_BC, TRAIN_BC_SHA256, 0o664),
        ("train_ppo", TRAIN_PPO, TRAIN_PPO_SHA256, 0o664),
        ("evaluate_policy_bc", EVALUATOR, EVALUATOR_SHA256, 0o664),
        ("hash_bound_helper", HELPER, HELPER_SHA256, 0o555),
    ):
        _, result[name] = read_regular_bytes(
            path, digest, name, expected_mode=mode
        )
    return result


def ledger_artifact(
    specification: Mapping[str, Any],
    expected_path: str,
    expected_sha256: str,
    expected_bytes: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checks = {
        "path": specification.get("path") == expected_path,
        "sha256": specification.get("sha256") == expected_sha256,
        "mode": specification.get("mode_octal") == "0444",
        "bytes": specification.get("bytes") == expected_bytes,
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label}: frozen ledger artifact binding drift")
    path = ROOT / expected_path
    payload, evidence = read_regular_bytes(
        path,
        expected_sha256,
        label,
        expected_mode=0o444,
    )
    if evidence.get("bytes") != expected_bytes:
        raise ProtocolError(f"{label}: frozen ledger artifact size drift")
    return strict_json(payload, label), evidence


def audit_global_ledger() -> dict[str, Any]:
    payload, evidence = read_regular_bytes(
        LEDGER,
        LEDGER_SHA256,
        "global official candidate ledger snapshot",
        expected_mode=0o444,
    )
    if evidence.get("bytes") != 3266:
        raise ProtocolError("global official candidate ledger byte-count drift")
    ledger = strict_json(payload, "global official candidate ledger snapshot")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or len(entries) != 2:
        raise ProtocolError("global official candidate ledger entry drift")
    first, second = entries
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise ProtocolError("global official candidate ledger entry type drift")
    first_candidate = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
    second_candidate = "bf551805d807cb6a77f500a51a12204727ab5cbd371fb1d36a947c63299228cb"
    ledger_checks = {
        "schema": ledger.get("schema_version") == LEDGER_SCHEMA,
        "status": ledger.get("status") == LEDGER_STATUS,
        "budget": ledger.get("budget") == 12,
        "consumed": ledger.get("consumed_before_atlas") == 2,
        "remaining": ledger.get("remaining_before_atlas") == 10,
        "atlas_zero": ledger.get("atlas_branch_consumes") == 0,
        "ordinals": [entry.get("ordinal") for entry in entries] == [1, 2],
        "candidate_hashes": [
            entry.get("candidate_model_state_sha256") for entry in entries
        ]
        == [first_candidate, second_candidate],
        "candidate_hashes_unique": len(
            {entry.get("candidate_model_state_sha256") for entry in entries}
        )
        == 2,
        "classifications": all(
            entry.get("classification") == "specialist_valid_consumed"
            for entry in entries
        ),
        "checks_locked": ledger.get("checks")
        == {
            "ordinals_exact_1_2": True,
            "candidate_model_state_sha256_unique": True,
            "every_entry_has_immutable_attempt_and_completed_evidence": True,
            "entry_count_equals_consumed_before_atlas": True,
            "consumed_plus_remaining_equals_budget": True,
            "atlas_is_parent_only_and_constructs_no_changed_candidate": True,
        },
    }
    if not all(ledger_checks.values()):
        raise ProtocolError("global official candidate ledger semantic drift")

    first_attempt, first_attempt_evidence = ledger_artifact(
        first.get("attempt", {}),
        "artifacts/.ptcg-cw11_4317_specialist_devconsistency_20260802_v1-"
        "specialist-dev-consistency-attempt.json",
        "3ba4b90de83b3639b8b8671451524440cf829efa2f7d3a362af44a4bd98e803c",
        8599,
        "official ledger entry 1 attempt",
    )
    first_manifest, first_manifest_evidence = ledger_artifact(
        first.get("execution_manifest", {}),
        "artifacts/cw11_4317_specialist_devconsistency_20260802_v1."
        "specialist_dev_consistency/dev_consistency_execution_manifest.json",
        "47ebd3683b73f397ffff5c6874136c9f12d23ee8eb619a9705445fabd4b73a27",
        35469,
        "official ledger entry 1 execution manifest",
    )
    first_decision, first_decision_evidence = ledger_artifact(
        first.get("decision", {}),
        "artifacts/cw11_4317_specialist_devconsistency_20260802_v1."
        "specialist_dev_consistency/dev_consistency_decision.json",
        "9c925a66f116a10b0ac996a76fdcbe0c378f0e4d02d45c635a2d47228b189a35",
        18171,
        "official ledger entry 1 decision",
    )
    second_attempt, second_attempt_evidence = ledger_artifact(
        second.get("attempt", {}),
        "artifacts/.ptcg-cw23_specialist60_noharm_cw11_20260803_v1-attempt.json",
        "965e562cd52b81edfcd0067f58f099347b8fb2e4f1eef0c4a87e3fa3fad68ba5",
        2145,
        "official ledger entry 2 attempt",
    )
    second_result, second_result_evidence = ledger_artifact(
        second.get("result", {}),
        "artifacts/cw23_specialist60_noharm_cw11_20260803_v1.json",
        "426622c027ed4c41478c61663dea141e41ba24937b17fdf7607f5f36e2f3a51c",
        153165,
        "official ledger entry 2 result",
    )
    first_checks = {
        "attempt_schema": first_attempt.get("schema_version")
        == "ptcg-cw11-single-endpoint-specialist-dev-consistency-attempt-v1",
        "attempt_status": first_attempt.get("status")
        == "one_shot_dev_consistency_attempt_consumed",
        "attempt_candidate": first_attempt.get("candidate", {}).get(
            "model_state_sha256"
        )
        == first_candidate,
        "attempt_specialist_consumed": first_attempt.get("classification", {}).get(
            "specialist_valid_consumed"
        )
        is True,
        "manifest_schema": first_manifest.get("schema_version")
        == "ptcg-cw11-single-endpoint-specialist-dev-consistency-manifest-v1",
        "manifest_status": first_manifest.get("status")
        == first.get("execution_manifest", {}).get("required_status")
        == "six_dev_consistency_panels_completed",
        "manifest_candidate": first_manifest.get("candidate", {}).get(
            "model_state_sha256"
        )
        == first_candidate,
        "manifest_specialist_consumed": first_manifest.get("classification", {}).get(
            "specialist_valid_consumed"
        )
        is True,
        "decision_schema": first_decision.get("schema_version")
        == "ptcg-cw11-single-endpoint-specialist-dev-consistency-decision-v1",
        "decision_status": first_decision.get("status")
        == first.get("decision", {}).get("required_status")
        == "failed_dev_consistency",
        "decision_candidate": first_decision.get("candidate", {}).get(
            "model_state_sha256"
        )
        == first_candidate,
        "decision_specialist_consumed": first_decision.get("classification", {}).get(
            "specialist_valid_consumed"
        )
        is first.get("decision", {}).get("required_specialist_valid_consumed")
        is True,
        "derived_before_this_after": (0, 1, 1)
        == (int(first.get("ordinal", -1)) - 1, 1, int(first.get("ordinal", -1))),
    }
    second_ledger = second_result.get("official_candidate_budget_ledger", {})
    second_checks = {
        "attempt_schema": second_attempt.get("schema_version")
        == "ptcg-cw23-payload-specialist60-gate-v1",
        "attempt_status": second_attempt.get("status")
        == "specialist_attempt_committed_before_CUDA_or_valid_open",
        "attempt_candidate": second_attempt.get("candidate_model_state_sha256")
        == second_candidate,
        "attempt_before": second_attempt.get(
            "official_unique_changed_candidate_count_before_gate"
        )
        == 1,
        "attempt_this": second_attempt.get(
            "official_unique_changed_candidate_count_consumed_this_gate"
        )
        == 1,
        "attempt_after": second_attempt.get(
            "cumulative_official_unique_changed_candidate_count"
        )
        == 2,
        "attempt_budget_remaining": second_attempt.get(
            "official_unique_changed_candidate_budget"
        )
        == 12
        and second_attempt.get("remaining_official_unique_changed_candidate_budget")
        == 10,
        "result_schema": second_result.get("schema_version")
        == "ptcg-cw23-payload-specialist60-gate-v1",
        "result_status": second_result.get("status")
        == second.get("result", {}).get("required_status")
        == "NO_GO_CW23_SPECIALIST60_GATE",
        "result_decision_status": second_result.get("decision", {}).get("status")
        == "NO_GO_CW23_SPECIALIST60_GATE",
        "result_candidate": second_result.get("candidate_model_state_sha256")
        == second_candidate,
        "result_before": second_ledger.get("before_gate")
        == second.get("result", {}).get("required_before_gate")
        == 1,
        "result_this": second_ledger.get("consumed_this_gate")
        == second.get("result", {}).get("required_consumed_this_gate")
        == 1,
        "result_after": second_ledger.get("after_gate")
        == second.get("result", {}).get("required_after_gate")
        == 2,
        "result_budget_remaining": second_ledger.get("budget") == 12
        and second_ledger.get("remaining")
        == second.get("result", {}).get("required_remaining")
        == 10,
        "specialist_consumed_from_attempt_and_counter": second_attempt.get(
            "official_unique_changed_candidate_count_consumed_this_gate"
        )
        == 1,
        "result_specialist_consumed": second_result.get("scope", {}).get(
            "specialist_valid_consumed"
        )
        is True,
    }
    chain_checks = {
        "entry1_before_this_after": [0, 1, 1],
        "entry2_before_this_after": [1, 1, 2],
        "entry1_after_confirmed_by_entry2_before": second_attempt.get(
            "official_unique_changed_candidate_count_before_gate"
        )
        == 1,
        "continuous": second_ledger.get("before_gate") == 1,
        "final_consumed": second_ledger.get("after_gate"),
        "remaining": second_ledger.get("remaining"),
        "budget": second_ledger.get("budget"),
        "atlas_consumption": ledger.get("atlas_branch_consumes"),
    }
    if (
        not all(first_checks.values())
        or not all(second_checks.values())
        or chain_checks
        != {
            "entry1_before_this_after": [0, 1, 1],
            "entry2_before_this_after": [1, 1, 2],
            "entry1_after_confirmed_by_entry2_before": True,
            "continuous": True,
            "final_consumed": 2,
            "remaining": 10,
            "budget": 12,
            "atlas_consumption": 0,
        }
    ):
        raise ProtocolError("global official candidate ledger transitive drift")
    return {
        "pass": True,
        "snapshot": evidence,
        "checks": ledger_checks,
        "entries": [
            {
                "ordinal": 1,
                "candidate_model_state_sha256": first_candidate,
                "checks": first_checks,
                "attempt": first_attempt_evidence,
                "execution_manifest": first_manifest_evidence,
                "decision": first_decision_evidence,
            },
            {
                "ordinal": 2,
                "candidate_model_state_sha256": second_candidate,
                "checks": second_checks,
                "attempt": second_attempt_evidence,
                "result": second_result_evidence,
            },
        ],
        "counter_chain": chain_checks,
    }


def read_proc_self_cmdline() -> tuple[list[str], dict[str, Any]]:
    descriptor = os.open(
        "/proc/self/cmdline",
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            total += len(chunk)
            if total > 65536:
                raise ProtocolError("/proc/self/cmdline exceeds fixed limit")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if not payload or not payload.endswith(b"\0"):
        raise ProtocolError("/proc/self/cmdline framing drift")
    raw_tokens = payload[:-1].split(b"\0")
    if not raw_tokens or any(not token for token in raw_tokens):
        raise ProtocolError("/proc/self/cmdline empty-token drift")
    try:
        tokens = [token.decode("utf-8", errors="strict") for token in raw_tokens]
    except UnicodeDecodeError as error:
        raise ProtocolError("/proc/self/cmdline is not strict UTF-8") from error
    return tokens, {
        "source": "/proc/self/cmdline",
        "nul_terminated": True,
        "token_count": len(tokens),
        "raw_sha256": sha256_bytes(payload),
        "canonical_token_json_sha256": sha256_bytes(canonical_json(tokens)),
    }


def actual_script_gate(self_evidence: Mapping[str, Any]) -> dict[str, Any]:
    actual_file = str(__file__)
    visible = os.lstat(actual_file)
    checks = {
        "file_is_absolute": os.path.isabs(actual_file),
        "file_spelling_exact": actual_file == str(SCRIPT),
        "file_resolves_exact": Path(actual_file).resolve() == SCRIPT,
        "regular_not_symlink": stat.S_ISREG(visible.st_mode)
        and not stat.S_ISLNK(visible.st_mode),
        "single_link": int(visible.st_nlink) == 1,
        "device": int(visible.st_dev) == self_evidence.get("device"),
        "inode": int(visible.st_ino) == self_evidence.get("inode"),
        "bytes": int(visible.st_size) == self_evidence.get("bytes"),
        "mode": format(stat.S_IMODE(visible.st_mode), "04o")
        == self_evidence.get("mode_octal"),
        "nlink": int(visible.st_nlink) == self_evidence.get("nlink"),
        "mtime_ns": int(visible.st_mtime_ns) == self_evidence.get("mtime_ns"),
    }
    if not all(checks.values()):
        raise ProtocolError("actual runner __file__ identity drift")
    return {"pass": True, "absolute_path": actual_file, "checks": checks}


def target_absence_binding_matches(value: Any, path: Path) -> bool:
    return isinstance(value, Mapping) and value == {
        "relative_path": str(path.relative_to(ROOT)),
        "absolute_path": str(path),
        "absent": True,
        "publication": "held_dirfd_O_EXCL_mode_0444",
    }


def runner_binding_matches(value: Any, self_evidence: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected = {
        "path": str(SCRIPT.relative_to(ROOT)),
        "absolute_path": str(SCRIPT),
        "sha256": self_evidence.get("sha256"),
        "bytes": self_evidence.get("bytes"),
        "mode_octal": "0555",
        "device": self_evidence.get("device"),
        "inode": self_evidence.get("inode"),
        "nlink": self_evidence.get("nlink"),
        "mtime_ns": self_evidence.get("mtime_ns"),
    }
    return all(value.get(key) == expected_value for key, expected_value in expected.items())


def audit_execution_preregistration(
    expected_sha256: str,
    self_evidence: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    if not is_sha256(expected_sha256):
        raise ProtocolError("run requires --execution-prereg-sha256")
    payload, evidence = read_regular_bytes(
        EXECUTION_PREREG,
        expected_sha256,
        "cohort-atlas execution preregistration",
        expected_mode=0o444,
    )
    prereg = strict_json(payload, "cohort-atlas execution preregistration")
    invocation = prereg.get("invocation", {})
    absence = prereg.get("absent_at_lock", {})
    template = invocation.get("absolute_command_token_template")
    template_hash = sha256_bytes(canonical_json(list(ABSOLUTE_COMMAND_TOKEN_TEMPLATE)))
    expanded = [
        (
            str(self_evidence.get("sha256"))
            if token == RUNNER_SHA_PLACEHOLDER
            else expected_sha256
            if token == EXECUTION_PREREG_SHA_PLACEHOLDER
            else token
        )
        for token in ABSOLUTE_COMMAND_TOKEN_TEMPLATE
    ]
    expanded_hash = sha256_bytes(canonical_json(expanded))
    proc_tokens, proc_evidence = read_proc_self_cmdline()
    original_argv = list(getattr(sys, "orig_argv", []))
    runtime_argv = list(sys.argv)
    script_gate = actual_script_gate(self_evidence)
    placeholder_text = payload.decode("utf-8")
    ledger_binding = prereg.get("ledger_snapshot", {})
    checks = {
        "schema": prereg.get("schema_version") == EXECUTION_SCHEMA,
        "status": prereg.get("status") == "LOCKED_FOR_ONE_PARENT_ONLY_ATLAS_ATTEMPT",
        "branch": prereg.get("branch") == BRANCH,
        "correction_v2_sha": prereg.get("correction_v2_sha256")
        == CORRECTION_SHA256,
        "correction_v3_sha": prereg.get("correction_v3_sha256")
        == CORRECTION_V3_SHA256,
        "correction_v4_sha": prereg.get("correction_v4_sha256")
        == CORRECTION_V4_SHA256,
        "correction_v5_sha": prereg.get("correction_v5_sha256")
        == CORRECTION_V5_SHA256,
        "correction_v6_sha": prereg.get("correction_v6_sha256")
        == CORRECTION_V6_SHA256,
        "correction_v7_sha": prereg.get("correction_v7_sha256")
        == CORRECTION_V7_SHA256,
        "runner_identity": runner_binding_matches(prereg.get("runner"), self_evidence),
        "template_exact": template == list(ABSOLUTE_COMMAND_TOKEN_TEMPLATE),
        "template_hash": invocation.get("template_canonical_sha256")
        == template_hash,
        "runner_placeholder_once": isinstance(template, list)
        and template.count(RUNNER_SHA_PLACEHOLDER) == 1
        and placeholder_text.count(f'"{RUNNER_SHA_PLACEHOLDER}"') == 1,
        "prereg_placeholder_once": isinstance(template, list)
        and template.count(EXECUTION_PREREG_SHA_PLACEHOLDER) == 1
        and placeholder_text.count(f'"{EXECUTION_PREREG_SHA_PLACEHOLDER}"') == 1,
        "prereg_self_sha_not_embedded": expected_sha256 not in placeholder_text,
        "proc_cmdline_exact": proc_tokens == expanded,
        "sys_orig_argv_exact": original_argv == expanded,
        "sys_argv_suffix_exact": runtime_argv == expanded[3:],
        "sys_executable_spelling_exact": str(sys.executable) == expanded[0],
        "python_flags_exact": sys.flags.isolated == 1
        and sys.flags.dont_write_bytecode == 1,
        "actual_file_gate": script_gate.get("pass") is True,
        "attempt_absent_at_lock": target_absence_binding_matches(
            absence.get("attempt_marker"), ATTEMPT_MARKER
        ),
        "result_absent_at_lock": target_absence_binding_matches(
            absence.get("result"), RESULT
        ),
        "workspace_root_binding_present": isinstance(
            absence.get("workspace_root"), Mapping
        ),
        "artifacts_directory_binding_present": isinstance(
            absence.get("artifacts_directory"), Mapping
        ),
        "ledger_path": ledger_binding.get("path") == str(LEDGER.relative_to(ROOT)),
        "ledger_absolute_path": ledger_binding.get("absolute_path") == str(LEDGER),
        "ledger_sha": ledger_binding.get("sha256") == LEDGER_SHA256,
        "ledger_mode": ledger_binding.get("mode_octal") == "0444",
        "ledger_bytes": ledger_binding.get("bytes") == 3266,
        "ledger_status": ledger_binding.get("status") == LEDGER_STATUS,
        "ledger_entry_count": ledger_binding.get("entry_count") == 2,
        "ledger_consumed": ledger_binding.get("consumed_before_atlas") == 2,
        "ledger_remaining": ledger_binding.get("remaining_before_atlas") == 10,
        "ledger_budget": ledger_binding.get("budget") == 12,
        "ledger_atlas_zero": ledger_binding.get("atlas_branch_consumes") == 0,
        "ledger_transitive_pass": ledger.get("pass") is True,
        "one_attempt": prereg.get("formal_attempts_authorized") == 1,
        "candidate_forbidden": prereg.get("changed_candidates_authorized") == 0,
        "candidate_unauthorized": prereg.get("candidate_authorized") is False,
    }
    if not all(checks.values()):
        raise ProtocolError(f"execution preregistration drift: {checks}")
    return {
        "pass": True,
        "checks": checks,
        "evidence": evidence,
        "invocation": {
            "unexpanded_template_canonical_sha256": template_hash,
            "expanded_command_canonical_sha256": expanded_hash,
            "token_count": len(expanded),
            "proc_cmdline": proc_evidence,
            "sys_orig_argv_exact": True,
            "sys_argv_suffix_exact": True,
            "actual_script": script_gate,
            "substitution_in_memory_only": True,
        },
        "publication_binding": {
            "workspace_root": dict(absence["workspace_root"]),
            "artifacts_directory": dict(absence["artifacts_directory"]),
            "attempt_marker": dict(absence["attempt_marker"]),
            "result": dict(absence["result"]),
        },
        "ledger_snapshot": dict(ledger_binding),
    }


def validate_profile(
    profile: Mapping[str, Any], current: ModuleType
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if profile.get("schema_version") != PROFILE_SCHEMA or profile.get("status") != PROFILE_STATUS:
        raise ProtocolError("frozen profile schema/status drift")
    b256, _, _, b256_audit = current.validate_profile_selection(profile, PROFILE_SHA256)
    if [int(row["final_b256_slot_zero_based"]) for row in b256] != list(range(B256_SIZE)):
        raise ProtocolError("B256 canonical slot order drift")
    pf_slots = [
        index
        for index, row in enumerate(b256)
        if str(row["stratum"]) in {"pf_ctx0_hard", "pf_ctx7_hard"}
    ]
    retention_slots = [
        index for index, row in enumerate(b256) if str(row["category"]) != "hard"
    ]
    dominic_slots = [
        index for index, row in enumerate(b256) if str(row["stratum"]) == "dominic_ctx0_hard"
    ]
    search_slots = [
        index
        for index in pf_slots
        if str(b256[index]["line_sha256"]) != OLD_IMPOSSIBLE_LINE
    ]
    zero_retention_slots = [
        index for index in retention_slots if float(b256[index]["selection_margin"]) == 0.0
    ]
    checks = {
        "b256_sha": b256_audit.get("b256_selection_sha256") == EXPECTED_B256_SHA256,
        "b256_256": len(b256) == B256_SIZE,
        "pf_pool64": len(pf_slots) == PF_POOL_SIZE,
        "search_pool63": len(search_slots) == SEARCH_POOL_SIZE,
        "retention160": len(retention_slots) == RETENTION_SIZE,
        "dominic32": len(dominic_slots) == DOMINIC_SIZE,
        "zero_retention16": len(zero_retention_slots) == 16,
        "impossible_present_once_then_excluded": sum(
            str(b256[index]["line_sha256"]) == OLD_IMPOSSIBLE_LINE for index in pf_slots
        )
        == 1,
        "all_targets_singleton_wrong": all(
            len(b256[index]["expert_order"]) == 1
            and len(b256[index]["predicted_order"]) == 1
            and b256[index]["ordered_correct"] is False
            for index in pf_slots
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"corrected B256/pool drift: {checks}")
    return [dict(row) for row in b256], {
        "pass": True,
        "checks": checks,
        "b256": b256_audit,
        "pf_slots": pf_slots,
        "search_slots": search_slots,
        "retention_slots": retention_slots,
        "dominic_slots": dominic_slots,
        "zero_retention_slots": zero_retention_slots,
        "search_line_sha256": [str(b256[index]["line_sha256"]) for index in search_slots],
    }


def audit_b256_sample_weights(
    batch_cpu: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    weights = batch_cpu.get("sample_weights")
    expected = np.ascontiguousarray(
        [
            np.float32(CONTEXT34_SAMPLE_WEIGHT_FLOAT32)
            if int(row["context"]) == 34
            else np.float32(1.0)
            for row in rows
        ],
        dtype="<f4",
    )
    observed = (
        np.ascontiguousarray(weights.detach().cpu().numpy(), dtype="<f4")
        if isinstance(weights, torch.Tensor)
        else np.empty(0, dtype="<f4")
    )
    denominator = np.float64(0.0)
    for value in observed:
        denominator = np.float64(denominator + np.float64(value))
    checks = {
        "torch_float32": isinstance(weights, torch.Tensor)
        and weights.dtype == torch.float32,
        "cpu": isinstance(weights, torch.Tensor) and weights.device.type == "cpu",
        "contiguous": isinstance(weights, torch.Tensor) and bool(weights.is_contiguous()),
        "shape256": observed.shape == (B256_SIZE,),
        "context34_exact16": sum(int(row["context"]) == 34 for row in rows) == 16,
        "other_exact240": sum(int(row["context"]) != 34 for row in rows) == 240,
        "elementwise_exact": observed.shape == expected.shape
        and bool(np.array_equal(observed, expected)),
        "context34_scalar_hex": np.float32(
            CONTEXT34_SAMPLE_WEIGHT_FLOAT32
        ).tobytes().hex()
        == "abaaaa3e",
        "float32_le_sha256": f4_sha(observed, np)
        == EXPECTED_B256_SAMPLE_WEIGHTS_F4_SHA256,
        "float64_left_fold_sum": float(denominator) == 245.33333349227905,
    }
    if not all(checks.values()):
        raise ProtocolError(f"authoritative B256 sample-weight drift: {checks}")
    return {
        "pass": True,
        "checks": checks,
        "shape": [B256_SIZE],
        "dtype": "little-endian float32",
        "float32_le_sha256": f4_sha(observed, np),
        "float64_left_fold_sum": float(denominator),
        "archive_sample_weight_preserved": False,
    }


def f4_sha(value: Any, np: Any) -> str:
    return sha256_bytes(np.ascontiguousarray(value, dtype="<f4").tobytes())


def f8_sha(value: Any, np: Any) -> str:
    return sha256_bytes(np.ascontiguousarray(value, dtype="<f8").tobytes())


def stored_f4_to_f8(value: Any, np: Any) -> Any:
    stored = np.ascontiguousarray(value, dtype="<f4")
    converted = np.ascontiguousarray(stored, dtype="<f8")
    if not bool(np.isfinite(converted).all()):
        raise ProtocolError("stored float32 to float64 conversion is nonfinite")
    return converted


def authoritative_vector_l2(value: Any, np: Any) -> float:
    vector = np.ascontiguousarray(value, dtype="<f8")
    if vector.ndim != 1 or not bool(np.isfinite(vector).all()):
        raise ProtocolError("authoritative L2 input is malformed or nonfinite")
    squared = np.dot(vector, vector)
    if not bool(np.isfinite(squared)) or bool(squared < 0.0):
        raise ProtocolError("authoritative L2 squared norm is invalid")
    norm = float(np.sqrt(squared))
    if not math.isfinite(norm):
        raise ProtocolError("authoritative L2 is nonfinite")
    return norm


def flat_gradient(
    objective: Any,
    parameter_tuple: Sequence[Any],
    current: ModuleType,
    np: Any,
    torch: Any,
) -> Any:
    values = torch.autograd.grad(
        objective,
        parameter_tuple,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
        materialize_grads=False,
    )
    flat = current.flat_gradient(values, np, torch).astype("<f4", copy=False)
    if flat.shape != (ACTOR_DIMENSION,) or not bool(np.isfinite(flat).all()):
        raise ProtocolError("same-graph actor6 gradient shape/finite drift")
    return flat


def gradient_report(value: Any, np: Any) -> dict[str, Any]:
    vector = stored_f4_to_f8(value, np)
    return {
        "l2": authoritative_vector_l2(vector, np),
        "max_abs": float(np.max(np.abs(vector))),
        "nonzero": int(np.count_nonzero(vector)),
        "float32_le_sha256": f4_sha(value, np),
    }


def weighted_loss(
    per_row: Any,
    weights: Any,
    indices: Sequence[int],
    torch: Any,
) -> Any:
    selector = torch.tensor(list(indices), dtype=torch.long, device=per_row.device)
    selected_weights = weights[selector].float()
    denominator = torch.sum(selected_weights)
    if (
        not bool(torch.isfinite(selected_weights).all())
        or not bool(torch.isfinite(denominator))
        or float(denominator.detach().cpu()) <= 0.0
    ):
        raise ProtocolError("ordered-NLL sample-weight denominator is not finite positive")
    return torch.sum(per_row[selector] * selected_weights) / denominator


def build_primary_atlas(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    parameters: Mapping[str, Any],
    helper: ModuleType,
    current: ModuleType,
    np: Any,
) -> dict[str, Any]:
    torch = helper.torch
    if any(value.dtype != torch.bfloat16 for value in outputs.values()):
        raise ProtocolError("exact B256 parent outputs are not all native BF16")
    predictions, _, _, _ = helper.ppo.sample_ordered_actions(
        outputs,
        batch,
        deterministic=True,
        canonicalize_order=False,
    )
    normalized_predictions = [[int(value) for value in row] for row in predictions]
    if any(
        normalized_predictions[index] != [int(value) for value in row["predicted_order"]]
        for index, row in enumerate(rows)
    ):
        raise ProtocolError("exact B256 parent prediction drift")
    logits = outputs["policy_logits"].float()
    per_row_nll = current.ordered_nll_per_row(outputs, batch, torch)
    parameter_tuple = tuple(parameters[name] for name in ACTOR6_NAMES)
    pf_slots = list(selection["pf_slots"])
    retention_slots = list(selection["retention_slots"])
    dominic_slots = list(selection["dominic_slots"])

    target_margin = np.empty((PF_POOL_SIZE, ACTOR_DIMENSION), dtype="<f4")
    target_nll = np.empty((PF_POOL_SIZE, ACTOR_DIMENSION), dtype="<f4")
    target_reports = []
    for pool_slot, b256_index in enumerate(pf_slots):
        row = rows[b256_index]
        margin, detail = current.threat_margin_tensor(
            logits[b256_index],
            batch["option_mask"][b256_index],
            row["expert_order"],
            singleton_only=True,
        )
        if (
            float(detail["margin"]) != float(row["selection_margin"])
            or int(detail["positive_option"]) != int(row["expert_order"][0])
            or int(detail["negative_option"]) != int(row["predicted_order"][0])
        ):
            raise ProtocolError("PF hard64 primary-threat drift")
        margin_gradient = flat_gradient(
            margin, parameter_tuple, current, np, torch
        )
        nll_gradient = flat_gradient(
            per_row_nll[b256_index], parameter_tuple, current, np, torch
        )
        target_margin[pool_slot] = margin_gradient
        target_nll[pool_slot] = nll_gradient
        target_reports.append(
            {
                "pool_slot": pool_slot,
                "b256_index": b256_index,
                "line_sha256": str(row["line_sha256"]),
                "episode_id": str(row["episode_id"]),
                "context": int(row["context"]),
                "expert_option": int(row["expert_order"][0]),
                "primary_stage": int(detail["stage"]),
                "primary_competitor": int(row["predicted_order"][0]),
                "parent_primary_margin": float(detail["margin"]),
                "buffered_required_change": LINEAR_TARGET_MARGIN
                - float(detail["margin"]),
                "sample_weight_float32": float(
                    batch["sample_weights"][b256_index].float().detach().cpu()
                ),
                "margin_gradient": gradient_report(margin_gradient, np),
                "row_nll_gradient": gradient_report(nll_gradient, np),
            }
        )

    retention_margin = np.empty((RETENTION_SIZE, ACTOR_DIMENSION), dtype="<f4")
    retention_reports = []
    for guard_slot, b256_index in enumerate(retention_slots):
        row = rows[b256_index]
        margin, detail = current.threat_margin_tensor(
            logits[b256_index],
            batch["option_mask"][b256_index],
            row["expert_order"],
            singleton_only=False,
        )
        if float(detail["margin"]) != float(row["selection_margin"]):
            raise ProtocolError("retention160 primary-threat drift")
        gradient = flat_gradient(margin, parameter_tuple, current, np, torch)
        retention_margin[guard_slot] = gradient
        retention_reports.append(
            {
                "guard_slot": guard_slot,
                "b256_index": b256_index,
                "line_sha256": str(row["line_sha256"]),
                "source": str(row["source"]),
                "context": int(row["context"]),
                "parent_primary_margin": float(detail["margin"]),
                "zero_primary_margin": float(detail["margin"]) == 0.0,
                "primary_stage": int(detail["stage"]),
                "primary_positive_option": int(detail["positive_option"]),
                "primary_competitor": int(detail["negative_option"]),
                "gradient": gradient_report(gradient, np),
            }
        )
    pf_loss = weighted_loss(per_row_nll, batch["sample_weights"], pf_slots, torch)
    retention_loss = weighted_loss(
        per_row_nll, batch["sample_weights"], retention_slots, torch
    )
    dominic_loss = weighted_loss(
        per_row_nll, batch["sample_weights"], dominic_slots, torch
    )
    pf_descent = flat_gradient(-pf_loss, parameter_tuple, current, np, torch)
    retention_descent = flat_gradient(
        -retention_loss, parameter_tuple, current, np, torch
    )
    dominic_descent = flat_gradient(
        -dominic_loss, parameter_tuple, current, np, torch
    )
    base_matrix = np.concatenate(
        (
            target_margin,
            target_nll,
            retention_margin,
            pf_descent.reshape(1, -1),
            retention_descent.reshape(1, -1),
            dominic_descent.reshape(1, -1),
        ),
        axis=0,
    ).astype("<f4", copy=False)
    if base_matrix.shape != (291, ACTOR_DIMENSION):
        raise ProtocolError(f"primary atlas shape drift: {base_matrix.shape}")
    zero_count = sum(bool(row["zero_primary_margin"]) for row in retention_reports)
    if zero_count != 16:
        raise ProtocolError(f"same-graph zero-retention count drift: {zero_count}")
    return {
        "margin_logits_float32": logits,
        "base_matrix": base_matrix,
        "target_margin": target_margin,
        "target_nll": target_nll,
        "retention_margin": retention_margin,
        "target_reports": target_reports,
        "retention_reports": retention_reports,
        "aggregate_reports": {
            "pf_hard64_descent": gradient_report(pf_descent, np),
            "retention160_descent": gradient_report(retention_descent, np),
            "dominic32_descent": gradient_report(dominic_descent, np),
        },
        "parent_losses": {
            "pf_hard64": float(pf_loss.detach().cpu()),
            "retention160": float(retention_loss.detach().cpu()),
            "dominic32": float(dominic_loss.detach().cpu()),
        },
        "ordered_nll_contract": {
            "per_row_stage_reduction": "sum_without_action_count_division",
            "aggregate_weighting": "exact_float32_sample_weight_weighted_mean",
            "aggregate_denominator_clamp": False,
            "b256_sample_weights_float32_le_sha256": f4_sha(
                batch["sample_weights"].detach().cpu().numpy(), np
            ),
        },
        "autograd_grad_calls": 291,
    }


def solve_active_qp(
    active_matrix: Any,
    rhs: Any,
    np: Any,
    optimize: Any,
) -> dict[str, Any] | None:
    matrix = np.ascontiguousarray(active_matrix, dtype="<f8")
    rhs_f8 = np.ascontiguousarray(rhs, dtype="<f8")
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] != ACTOR_DIMENSION
        or rhs_f8.shape != (matrix.shape[0],)
        or not bool(np.isfinite(matrix).all())
        or not bool(np.isfinite(rhs_f8).all())
    ):
        return None
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or not bool(np.isfinite(singular).all()) or singular[0] <= 0.0:
        return None
    rank_tolerance = (
        max(matrix.shape) * np.finfo(np.float64).eps * float(singular[0])
    )
    rank = int((singular > rank_tolerance).sum())
    if rank <= 0:
        return None
    condition = float(singular[0] / singular[rank - 1])
    if not math.isfinite(condition) or condition > CONDITION_LIMIT:
        return None
    if u.shape[0] != matrix.shape[0]:
        return None
    reduced = matrix @ vh[:rank].T
    solution = optimize.minimize(
        lambda point: 0.5 * np.dot(point, point),
        np.zeros(rank, dtype=np.float64),
        jac=lambda point: point,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda point: reduced @ point - rhs_f8,
                "jac": lambda point: reduced,
            }
        ],
        method="SLSQP",
        bounds=None,
        options={"ftol": QP_FTOL, "maxiter": QP_MAXITER, "disp": False},
    )
    if (
        not bool(solution.success)
        or int(solution.status) != 0
        or not bool(np.isfinite(solution.x).all())
    ):
        return None
    delta = vh[:rank].T @ solution.x
    residual = matrix @ delta - rhs_f8
    try:
        norm = authoritative_vector_l2(delta, np)
    except ProtocolError:
        return None
    condition_recheck = float(singular[0] / singular[rank - 1])
    if (
        not bool(np.isfinite(delta).all())
        or not bool(np.isfinite(residual).all())
        or float(residual.min()) < -CONSTRAINT_TOLERANCE
        or not math.isfinite(norm)
        or norm > ATLAS_L2_CEILING + L2_TOLERANCE
        or condition_recheck != condition
        or condition_recheck > CONDITION_LIMIT
    ):
        return None
    return {
        "delta": delta,
        "norm": norm,
        "rank": rank,
        "condition": condition,
        "iterations": int(solution.nit),
        "solver_status": int(solution.status),
        "residual_min": float(residual.min()),
        "matrix_shape": [int(value) for value in matrix.shape],
        "matrix_float64_le_sha256": f8_sha(matrix, np),
        "rhs_float64_le_sha256": f8_sha(rhs_f8, np),
        "delta_float64_le_sha256": f8_sha(delta, np),
    }


def assemble_constraint_records(
    ordered_records: Sequence[Mapping[str, Any]],
    np: Any,
) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[tuple[Any, ...]] = set()
    vectors = []
    rhs_values = []
    manifests = []
    retained = []
    for raw in ordered_records:
        identity = tuple(raw["identity"])
        if identity in seen:
            continue
        vector = np.ascontiguousarray(raw["vector"], dtype="<f8")
        rhs = np.float64(raw["rhs"])
        if (
            vector.shape != (ACTOR_DIMENSION,)
            or not bool(np.isfinite(vector).all())
            or not bool(np.isfinite(rhs))
        ):
            raise ProtocolError("nonfinite or malformed constraint record")
        manifest = dict(raw["manifest"])
        manifest["constraint_identity"] = list(identity)
        seen.add(identity)
        vectors.append(vector)
        rhs_values.append(rhs)
        manifests.append(manifest)
        retained.append(dict(raw))
    if not vectors:
        raise ProtocolError("empty active constraint record list")
    matrix = np.ascontiguousarray(np.stack(vectors, axis=0), dtype="<f8")
    rhs = np.ascontiguousarray(rhs_values, dtype="<f8")
    return matrix, rhs, manifests, retained


def selected_tuple_nll_constraint(
    target_tuple: Sequence[int],
    target_reports: Sequence[Mapping[str, Any]],
    base_float64: Any,
    target_nll_offset: int,
    np: Any,
) -> tuple[Any, dict[str, Any]]:
    numerator = np.zeros(ACTOR_DIMENSION, dtype=np.float64)
    denominator = np.float64(0.0)
    tuple_weights = []
    tuple_sha = []
    for raw_slot in target_tuple:
        pool_slot = int(raw_slot)
        report = target_reports[pool_slot]
        weight_float32 = np.float32(report["sample_weight_float32"])
        weight_float64 = np.float64(weight_float32)
        gradient_float64 = np.ascontiguousarray(
            base_float64[target_nll_offset + pool_slot], dtype="<f8"
        )
        if not bool(np.isfinite(weight_float64)):
            raise ProtocolError("selected-target NLL weight is nonfinite")
        numerator = numerator + weight_float64 * gradient_float64
        denominator = np.float64(denominator + weight_float64)
        tuple_weights.append(weight_float32)
        tuple_sha.append(str(report["line_sha256"]))
    if tuple_sha != sorted(tuple_sha):
        raise ProtocolError("selected-target NLL reduction is not in target SHA order")
    if not bool(np.isfinite(denominator)) or float(denominator) <= 0.0:
        raise ProtocolError("selected-target NLL denominator is not finite positive")
    vector = np.ascontiguousarray(-numerator / denominator, dtype="<f8")
    return vector, {
        "target_line_sha256": tuple_sha,
        "sample_weights_float32_le_sha256": f4_sha(tuple_weights, np),
        "sample_weight_float64_left_fold_denominator": float(denominator),
        "reduction": "target_SHA_order_float64_left_fold",
        "vector_float64_le_sha256": f8_sha(vector, np),
    }


def primary_beam_search(
    primary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    np: Any,
    optimize: Any,
) -> dict[str, Any]:
    if len(rows) != B256_SIZE:
        raise ProtocolError("primary beam B256 row count drift")
    base = stored_f4_to_f8(primary["base_matrix"], np)
    target_reports = list(primary["target_reports"])
    guard_reports = list(primary["retention_reports"])
    target_margin_offset = 0
    target_nll_offset = PF_POOL_SIZE
    guard_offset = PF_POOL_SIZE * 2
    pf_descent_index = guard_offset + RETENTION_SIZE
    retention_descent_index = pf_descent_index + 1
    dominic_descent_index = retention_descent_index + 1
    pf_b256_to_pool = {
        int(report["b256_index"]): int(report["pool_slot"])
        for report in target_reports
    }
    search_pool_slots = [
        pf_b256_to_pool[int(index)] for index in selection["search_slots"]
    ]
    survivors = []
    for pool_slot in search_pool_slots:
        report = target_reports[pool_slot]
        norm = float(report["margin_gradient"]["l2"])
        if not math.isfinite(norm) or norm <= 0.0:
            lower_bound = None
        else:
            lower_bound = float(
                np.float64(report["buffered_required_change"]) / np.float64(norm)
            )
        report["individual_cauchy_lower_bound_l2"] = lower_bound
        report["passes_fixed_0p0009_necessary_filter"] = (
            lower_bound is not None
            and math.isfinite(lower_bound)
            and lower_bound <= ATLAS_L2_CEILING
        )
        if report["passes_fixed_0p0009_necessary_filter"]:
            survivors.append(pool_slot)
    survivors.sort(key=lambda slot: str(target_reports[slot]["line_sha256"]))
    guard_scan_slots = sorted(
        range(len(guard_reports)),
        key=lambda slot: (
            str(guard_reports[slot]["line_sha256"]),
            int(guard_reports[slot]["primary_stage"]),
            int(guard_reports[slot]["primary_competitor"]),
        ),
    )
    zero_guard_slots = [
        slot for slot in guard_scan_slots if guard_reports[slot]["zero_primary_margin"]
    ]
    qp_calls = 0
    primary_cuts = 0

    def target_record(pool_slot: int) -> dict[str, Any]:
        report = target_reports[pool_slot]
        line_sha = str(report["line_sha256"])
        stage = int(report["primary_stage"])
        competitor = int(report["primary_competitor"])
        identity = ("target", line_sha, stage, competitor)
        return {
            "identity": identity,
            "threat_identity": identity,
            "vector": base[target_margin_offset + pool_slot],
            "rhs": np.float64(report["buffered_required_change"]),
            "manifest": {
                "kind": "target_primary",
                "pool_slot": pool_slot,
                "line_sha256": line_sha,
                "stage": stage,
                "positive_option": int(report["expert_option"]),
                "competitor": competitor,
                "parent_margin": float(report["parent_primary_margin"]),
                "endpoint_threshold": LINEAR_TARGET_MARGIN,
            },
        }

    def zero_record(guard_slot: int) -> dict[str, Any]:
        report = guard_reports[guard_slot]
        line_sha = str(report["line_sha256"])
        stage = int(report["primary_stage"])
        competitor = int(report["primary_competitor"])
        return {
            "identity": ("zero_primary", line_sha, stage, competitor),
            "threat_identity": ("retention", line_sha, stage, competitor),
            "vector": base[guard_offset + guard_slot],
            "rhs": np.float64(ZERO_GUARD_BUFFER),
            "manifest": {
                "kind": "zero_retention_primary",
                "guard_slot": guard_slot,
                "line_sha256": line_sha,
                "stage": stage,
                "positive_option": int(report["primary_positive_option"]),
                "competitor": competitor,
                "parent_margin": float(report["parent_primary_margin"]),
                "required_change": ZERO_GUARD_BUFFER,
            },
        }

    def primary_cut_record(cut: Mapping[str, Any]) -> dict[str, Any]:
        guard_slot = int(cut["guard_slot"])
        report = guard_reports[guard_slot]
        line_sha = str(report["line_sha256"])
        stage = int(report["primary_stage"])
        competitor = int(report["primary_competitor"])
        identity = ("retention", line_sha, stage, competitor)
        return {
            "identity": identity,
            "threat_identity": identity,
            "vector": base[guard_offset + guard_slot],
            "rhs": np.float64(-float(report["parent_primary_margin"])),
            "manifest": {
                "kind": "retention_primary_cut",
                "guard_slot": guard_slot,
                "line_sha256": line_sha,
                "stage": stage,
                "positive_option": int(report["primary_positive_option"]),
                "competitor": competitor,
                "parent_margin": float(report["parent_primary_margin"]),
                "residual_before_cut": float(cut["residual_before_cut"]),
            },
        }

    def selected_nll_record(target_tuple: tuple[int, ...]) -> dict[str, Any]:
        vector, reduction_audit = selected_tuple_nll_constraint(
            target_tuple, target_reports, base, target_nll_offset, np
        )
        name = "selected_target5" if len(target_tuple) == COHORT_SIZE else "selected_target_partial"
        return {
            "identity": ("aggregate", name),
            "vector": vector,
            "rhs": np.float64(TARGET5_NLL_BUFFER),
            "manifest": {
                "kind": name,
                **reduction_audit,
            },
        }

    def ordered_records(
        target_tuple: tuple[int, ...],
        cuts: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        targets = [target_record(pool_slot) for pool_slot in target_tuple]
        targets.sort(
            key=lambda record: (
                str(record["manifest"]["line_sha256"]),
                int(record["manifest"]["stage"]),
                int(record["manifest"]["competitor"]),
            )
        )
        zeros = [zero_record(slot) for slot in zero_guard_slots]
        aggregates = [
            {
                "identity": ("aggregate", "PF_hard64"),
                "vector": base[pf_descent_index],
                "rhs": np.float64(PF64_NLL_BUFFER),
                "manifest": {"kind": "PF_hard64"},
            },
            {
                "identity": ("aggregate", "retention160"),
                "vector": base[retention_descent_index],
                "rhs": np.float64(RETENTION160_NLL_BUFFER),
                "manifest": {"kind": "retention160"},
            },
            {
                "identity": ("aggregate", "Dominic_hard32"),
                "vector": base[dominic_descent_index],
                "rhs": np.float64(DOMINIC32_NLL_BUFFER),
                "manifest": {"kind": "Dominic_hard32"},
            },
            selected_nll_record(target_tuple),
        ]
        primary_records = [primary_cut_record(cut) for cut in cuts]
        return targets + zeros + aggregates + primary_records

    def solve_tuple(target_tuple: tuple[int, ...]) -> dict[str, Any] | None:
        nonlocal qp_calls, primary_cuts
        target_sha = tuple(
            str(target_reports[slot]["line_sha256"]) for slot in target_tuple
        )
        if target_sha != tuple(sorted(target_sha)):
            raise ProtocolError("target tuple is not canonical SHA order")
        if len(set(target_sha)) != len(target_sha) or len(
            {str(target_reports[slot]["episode_id"]) for slot in target_tuple}
        ) != len(target_tuple):
            raise ProtocolError("target tuple identity or episode uniqueness drift")
        cut_additions: list[dict[str, Any]] = []
        active_guard_slots = set(zero_guard_slots)
        while True:
            matrix, rhs, active_names, _ = assemble_constraint_records(
                ordered_records(target_tuple, cut_additions), np
            )
            qp_calls += 1
            solution = solve_active_qp(matrix, rhs, np, optimize)
            if solution is None:
                return None
            delta = solution["delta"]
            guard_violations = []
            for guard_slot in guard_scan_slots:
                report = guard_reports[guard_slot]
                required_change = (
                    ZERO_GUARD_BUFFER
                    if report["zero_primary_margin"]
                    else -float(report["parent_primary_margin"])
                )
                residual = float(base[guard_offset + guard_slot] @ delta) - required_change
                if residual < -CONSTRAINT_TOLERANCE:
                    guard_violations.append(
                        (
                            residual,
                            str(report["line_sha256"]),
                            int(report["primary_stage"]),
                            int(report["primary_competitor"]),
                            guard_slot,
                        )
                    )
            if not guard_violations:
                target_slacks = [
                    float(target_reports[slot]["parent_primary_margin"])
                    + float(base[target_margin_offset + slot] @ delta)
                    - LINEAR_TARGET_MARGIN
                    for slot in target_tuple
                ]
                repairs = sum(
                    float(report["parent_primary_margin"]) < 0.0
                    and float(report["parent_primary_margin"])
                    + float(base[target_margin_offset + slot] @ delta)
                    >= 0.0
                    for slot, report in enumerate(target_reports)
                )
                if (
                    not all(math.isfinite(value) for value in target_slacks)
                    or not math.isfinite(float(solution["norm"]))
                ):
                    return None
                record = {
                    key: value for key, value in solution.items() if key != "delta"
                }
                record.update(
                    {
                        "target_tuple": target_tuple,
                        "predicted_pf64_repairs": int(repairs),
                        "worst_target_slack": min(target_slacks),
                        "primary_constraint_manifest": active_names,
                        "primary_cut_additions": [dict(value) for value in cut_additions],
                        "primary_active_guard_slots": sorted(
                            active_guard_slots,
                            key=lambda slot: str(guard_reports[slot]["line_sha256"]),
                        ),
                        "primary_cut_count": len(cut_additions),
                    }
                )
                return record
            residual, _, _, _, add_guard = min(guard_violations)
            if add_guard in active_guard_slots or len(cut_additions) >= MAX_PRIMARY_CUTS:
                return None
            cut_additions.append(
                {"guard_slot": add_guard, "residual_before_cut": residual}
            )
            active_guard_slots.add(add_guard)
            primary_cuts += 1

    def rank_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        values = (
            float(record["norm"]),
            -int(record["predicted_pf64_repairs"]),
            -float(record["worst_target_slack"]),
        )
        if not all(math.isfinite(value) for value in values):
            raise ProtocolError("nonfinite frozen-beam ranking value")
        return (
            *values,
            tuple(
                str(target_reports[int(slot)]["line_sha256"])
                for slot in record["target_tuple"]
            ),
        )

    beam = []
    evaluated_by_size = {}
    feasible_before_truncation = {}
    seed_tuples = [(int(slot),) for slot in survivors]
    for target_tuple in seed_tuples:
        record = solve_tuple(target_tuple)
        if record is not None:
            beam.append(record)
    beam = sorted(beam, key=rank_key)[:BEAM_WIDTH]
    evaluated_by_size[1] = len(seed_tuples)
    feasible_before_truncation[1] = len(beam)
    for size in range(2, COHORT_SIZE + 1):
        generated = []
        seen_sha_tuples: set[tuple[str, ...]] = set()
        for record in beam:
            existing = tuple(int(value) for value in record["target_tuple"])
            final_sha = str(target_reports[existing[-1]]["line_sha256"])
            existing_episodes = {
                str(target_reports[slot]["episode_id"]) for slot in existing
            }
            for slot in survivors:
                slot = int(slot)
                slot_sha = str(target_reports[slot]["line_sha256"])
                if (
                    slot_sha <= final_sha
                    or str(target_reports[slot]["episode_id"]) in existing_episodes
                ):
                    continue
                target_tuple = existing + (slot,)
                sha_tuple = tuple(
                    str(target_reports[value]["line_sha256"])
                    for value in target_tuple
                )
                if sha_tuple in seen_sha_tuples:
                    continue
                seen_sha_tuples.add(sha_tuple)
                generated.append(target_tuple)
        solved = []
        for target_tuple in generated:
            candidate = solve_tuple(target_tuple)
            if candidate is not None:
                solved.append(candidate)
        ordered = sorted(solved, key=rank_key)
        evaluated_by_size[size] = len(generated)
        feasible_before_truncation[size] = len(ordered)
        beam = ordered[:BEAM_WIDTH]
        if not beam:
            break
    finalists = (
        list(beam)
        if beam and len(tuple(beam[0]["target_tuple"])) == COHORT_SIZE
        else []
    )
    return {
        "survivors": survivors,
        "survivor_line_sha256": [
            str(target_reports[slot]["line_sha256"]) for slot in survivors
        ],
        "finalists": finalists,
        "audit": {
            "beam_width": BEAM_WIDTH,
            "ranking": [
                "QP norm ascending",
                "PF64 predicted repairs descending",
                "worst target slack descending",
                "target line-SHA tuple ascending",
            ],
            "nonfinite_scores": "reject",
            "duplicates_removed_before_truncation": True,
            "canonical_deduplication_before_solving": True,
            "base_filter_preserves_b256_slot_order_before_survivor_SHA_sort": True,
            "expansion_uses_greater_SHA_not_pool_slot": True,
            "evaluated_by_size": evaluated_by_size,
            "feasible_before_truncation": feasible_before_truncation,
            "finalist_count": len(finalists),
            "qp_calls": qp_calls,
            "primary_cutting_plane_additions": primary_cuts,
            "absence_is_not_global_infeasibility": True,
        },
    }


def full_threat_scan(
    delta: Any,
    logits_float32: Any,
    batch: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    target_pool_slots: Sequence[int],
    selection: Mapping[str, Any],
    primary: Mapping[str, Any],
    parameters: Mapping[str, Any],
    current: ModuleType,
    helper: ModuleType,
    np: Any,
    active_gradient_cache: Mapping[tuple[Any, ...], Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    torch = helper.torch
    logits = logits_float32
    if logits.dtype != torch.float32 or int(logits.shape[0]) != B256_SIZE:
        raise ProtocolError("shared float32 margin-logit tensor drift")
    parameter_tuple = tuple(parameters[name] for name in ACTOR6_NAMES)
    target_reports = list(primary["target_reports"])
    target_b256 = [int(target_reports[slot]["b256_index"]) for slot in target_pool_slots]
    retention_slots = list(selection["retention_slots"])
    best: tuple[tuple[Any, ...], dict[str, Any], Any] | None = None
    threats_scanned = 0
    vjp_calls = 0
    active_cache_hits = 0

    def consider(
        role: str,
        b256_index: int,
        stage: int,
        chosen: int,
        competitor: int,
        threshold: float,
    ) -> None:
        nonlocal best, threats_scanned, vjp_calls, active_cache_hits
        margin = logits[b256_index, chosen] - logits[b256_index, competitor]
        parent_margin = float(margin.detach().cpu())
        row_sha = str(rows[b256_index]["line_sha256"])
        identity = (role, row_sha, stage, competitor)
        if identity in active_gradient_cache:
            gradient = np.ascontiguousarray(
                active_gradient_cache[identity], dtype="<f8"
            )
            active_cache_hits += 1
        else:
            gradient = stored_f4_to_f8(
                flat_gradient(margin, parameter_tuple, current, np, torch), np
            )
            vjp_calls += 1
        residual = parent_margin + float(gradient.astype(np.float64) @ delta) - threshold
        if not math.isfinite(residual):
            raise ProtocolError("nonfinite full-threat residual")
        threats_scanned += 1
        if residual < -CONSTRAINT_TOLERANCE:
            key = (residual, row_sha, stage, competitor)
            record = {
                "constraint_identity": list(identity),
                "role": role,
                "b256_index": b256_index,
                "line_sha256": row_sha,
                "stage": stage,
                "chosen": chosen,
                "competitor": competitor,
                "parent_margin": parent_margin,
                "endpoint_threshold": threshold,
                "required_change": threshold - parent_margin,
                "residual_before_cut": residual,
                "gradient_float32_le_sha256": f4_sha(gradient, np),
                "gradient_float64_le_sha256": f8_sha(gradient, np),
            }
            if best is None or key < best[0]:
                best = (key, record, gradient)

    for b256_index in sorted(target_b256, key=lambda index: str(rows[index]["line_sha256"])):
        row = rows[b256_index]
        expert = int(row["expert_order"][0])
        competitors = [
            int(value)
            for value in batch["option_mask"][b256_index]
            .bool()
            .nonzero(as_tuple=False)
            .squeeze(1)
            .detach()
            .cpu()
            .tolist()
            if int(value) != expert
        ]
        for competitor in sorted(competitors):
            consider(
                "target",
                b256_index,
                0,
                expert,
                competitor,
                LINEAR_TARGET_MARGIN,
            )
    for b256_index in sorted(retention_slots, key=lambda index: str(rows[index]["line_sha256"])):
        row = rows[b256_index]
        remaining = batch["option_mask"][b256_index].bool().clone()
        for stage, chosen_value in enumerate(row["expert_order"]):
            chosen = int(chosen_value)
            if not bool(remaining[chosen]):
                raise ProtocolError("retention expert prefix is illegal")
            competitors = [
                int(value)
                for value in remaining.nonzero(as_tuple=False)
                .squeeze(1)
                .detach()
                .cpu()
                .tolist()
                if int(value) != chosen
            ]
            for competitor in sorted(competitors):
                consider(
                    "retention",
                    b256_index,
                    stage,
                    chosen,
                    competitor,
                    0.0,
                )
            remaining[chosen] = False
    return (
        (
            None
            if best is None
            else {
                "record": best[1],
                "gradient": best[2],
            }
        ),
        {
            "threats_scanned": threats_scanned,
            "vjp_calls": vjp_calls,
            "active_gradient_cache_hits": active_cache_hits,
            "scan_order": "targets by line SHA then competitor; retention by line SHA then stage then competitor",
            "violation_tie_break": "most negative residual then line SHA then stage then competitor",
            "inactive_gradients_released_except_most_violated": True,
        },
    )


def certify_finalists(
    beam: Mapping[str, Any],
    batch: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    primary: Mapping[str, Any],
    parameters: Mapping[str, Any],
    current: ModuleType,
    helper: ModuleType,
    np: Any,
    optimize: Any,
) -> dict[str, Any]:
    finalists = list(beam["finalists"][:MAX_FINALISTS])
    target_reports = list(primary["target_reports"])
    base = stored_f4_to_f8(primary["base_matrix"], np)
    guard_reports = list(primary["retention_reports"])
    target_margin_offset = 0
    target_nll_offset = PF_POOL_SIZE
    guard_offset = PF_POOL_SIZE * 2
    pf_descent_index = guard_offset + RETENTION_SIZE
    retention_descent_index = pf_descent_index + 1
    dominic_descent_index = retention_descent_index + 1
    guard_scan_slots = sorted(
        range(len(guard_reports)),
        key=lambda slot: (
            str(guard_reports[slot]["line_sha256"]),
            int(guard_reports[slot]["primary_stage"]),
            int(guard_reports[slot]["primary_competitor"]),
        ),
    )
    zero_guard_slots = [
        slot for slot in guard_scan_slots if guard_reports[slot]["zero_primary_margin"]
    ]
    examined = []
    total_full_vjps = 0
    total_full_scans = 0
    total_active_cache_hits = 0

    def target_primary_record(pool_slot: int) -> dict[str, Any]:
        report = target_reports[pool_slot]
        line_sha = str(report["line_sha256"])
        stage = int(report["primary_stage"])
        competitor = int(report["primary_competitor"])
        identity = ("target", line_sha, stage, competitor)
        return {
            "identity": identity,
            "threat_identity": identity,
            "vector": base[target_margin_offset + pool_slot],
            "rhs": np.float64(report["buffered_required_change"]),
            "manifest": {
                "kind": "target_primary",
                "pool_slot": pool_slot,
                "line_sha256": line_sha,
                "stage": stage,
                "positive_option": int(report["expert_option"]),
                "competitor": competitor,
                "parent_margin": float(report["parent_primary_margin"]),
                "endpoint_threshold": LINEAR_TARGET_MARGIN,
            },
        }

    def zero_record(guard_slot: int) -> dict[str, Any]:
        report = guard_reports[guard_slot]
        line_sha = str(report["line_sha256"])
        stage = int(report["primary_stage"])
        competitor = int(report["primary_competitor"])
        return {
            "identity": ("zero_primary", line_sha, stage, competitor),
            "threat_identity": ("retention", line_sha, stage, competitor),
            "vector": base[guard_offset + guard_slot],
            "rhs": np.float64(ZERO_GUARD_BUFFER),
            "manifest": {
                "kind": "zero_retention_primary",
                "guard_slot": guard_slot,
                "line_sha256": line_sha,
                "stage": stage,
                "positive_option": int(report["primary_positive_option"]),
                "competitor": competitor,
                "parent_margin": float(report["parent_primary_margin"]),
                "required_change": ZERO_GUARD_BUFFER,
            },
        }

    def primary_cut_record(cut: Mapping[str, Any]) -> dict[str, Any]:
        guard_slot = int(cut["guard_slot"])
        report = guard_reports[guard_slot]
        line_sha = str(report["line_sha256"])
        stage = int(report["primary_stage"])
        competitor = int(report["primary_competitor"])
        identity = ("retention", line_sha, stage, competitor)
        return {
            "identity": identity,
            "threat_identity": identity,
            "vector": base[guard_offset + guard_slot],
            "rhs": np.float64(-float(report["parent_primary_margin"])),
            "manifest": {
                "kind": "retention_primary_cut",
                "guard_slot": guard_slot,
                "line_sha256": line_sha,
                "stage": stage,
                "positive_option": int(report["primary_positive_option"]),
                "competitor": competitor,
                "parent_margin": float(report["parent_primary_margin"]),
                "residual_before_cut": float(cut["residual_before_cut"]),
                "added_during": str(cut.get("added_during", "primary_beam")),
            },
        }

    def aggregate_records(target_tuple: tuple[int, ...]) -> list[dict[str, Any]]:
        selected_vector, selected_audit = selected_tuple_nll_constraint(
            target_tuple, target_reports, base, target_nll_offset, np
        )
        return [
            {
                "identity": ("aggregate", "PF_hard64"),
                "vector": base[pf_descent_index],
                "rhs": np.float64(PF64_NLL_BUFFER),
                "manifest": {"kind": "PF_hard64"},
            },
            {
                "identity": ("aggregate", "retention160"),
                "vector": base[retention_descent_index],
                "rhs": np.float64(RETENTION160_NLL_BUFFER),
                "manifest": {"kind": "retention160"},
            },
            {
                "identity": ("aggregate", "Dominic_hard32"),
                "vector": base[dominic_descent_index],
                "rhs": np.float64(DOMINIC32_NLL_BUFFER),
                "manifest": {"kind": "Dominic_hard32"},
            },
            {
                "identity": ("aggregate", "selected_target5"),
                "vector": selected_vector,
                "rhs": np.float64(TARGET5_NLL_BUFFER),
                "manifest": {"kind": "selected_target5", **selected_audit},
            },
        ]

    def full_cut_record(
        violation: Mapping[str, Any],
    ) -> dict[str, Any]:
        report = dict(violation["record"])
        identity = tuple(report["constraint_identity"])
        role = str(report["role"])
        if role not in {"target", "retention"}:
            raise ProtocolError("unknown full-threat role")
        return {
            "identity": identity,
            "threat_identity": identity,
            "vector": np.ascontiguousarray(violation["gradient"], dtype="<f8"),
            "rhs": np.float64(report["required_change"]),
            "manifest": {
                "kind": (
                    "target_full_threat"
                    if role == "target"
                    else "retention_full_threat_cut"
                ),
                **report,
            },
        }

    def active_records(
        target_tuple: tuple[int, ...],
        primary_cuts_local: Sequence[Mapping[str, Any]],
        full_targets: Sequence[Mapping[str, Any]],
        full_retention: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        targets = [target_primary_record(slot) for slot in target_tuple]
        targets.extend(dict(record) for record in full_targets)
        targets.sort(
            key=lambda record: (
                str(record["manifest"]["line_sha256"]),
                int(record["manifest"]["stage"]),
                int(record["manifest"]["competitor"]),
            )
        )
        zeros = [zero_record(slot) for slot in zero_guard_slots]
        primary_records = [primary_cut_record(cut) for cut in primary_cuts_local]
        return (
            targets
            + zeros
            + aggregate_records(target_tuple)
            + primary_records
            + [dict(record) for record in full_retention]
        )

    for finalist_index, finalist in enumerate(finalists):
        target_tuple = tuple(int(value) for value in finalist["target_tuple"])
        target_sha = tuple(
            str(target_reports[slot]["line_sha256"]) for slot in target_tuple
        )
        if frozenset(target_sha) == OLD_TARGET_SET or OLD_IMPOSSIBLE_LINE in target_sha:
            raise ProtocolError("closed exact target set or impossible row reached finalist")
        if len({str(target_reports[slot]["episode_id"]) for slot in target_tuple}) != COHORT_SIZE:
            raise ProtocolError("finalist episode uniqueness drift")
        if target_sha != tuple(sorted(target_sha)):
            raise ProtocolError("finalist target SHA tuple is not canonical")
        primary_cut_additions = [
            dict(value) for value in finalist["primary_cut_additions"]
        ]
        active_primary_guards = set(zero_guard_slots)
        active_primary_guards.update(
            int(value["guard_slot"]) for value in primary_cut_additions
        )
        if active_primary_guards != set(
            int(value) for value in finalist["primary_active_guard_slots"]
        ):
            raise ProtocolError("finalist primary active-guard reconstruction drift")
        full_target_records: list[dict[str, Any]] = []
        full_retention_records: list[dict[str, Any]] = []
        full_cuts = 0
        full_target_cuts = 0
        full_retention_cuts = 0
        primary_recuts = 0
        scans = 0
        failure = None
        certified_solution = None
        active_names: list[dict[str, Any]] = []
        active_retained: list[dict[str, Any]] = []
        matrix = np.empty((0, ACTOR_DIMENSION), dtype="<f8")
        rhs = np.empty(0, dtype="<f8")
        first_assembly = True
        first_solution_recheck = True
        while True:
            matrix, rhs, active_names, active_retained = assemble_constraint_records(
                active_records(
                    target_tuple,
                    primary_cut_additions,
                    full_target_records,
                    full_retention_records,
                ),
                np,
            )
            if first_assembly:
                primary_hash_checks = {
                    "matrix": f8_sha(matrix, np)
                    == finalist["matrix_float64_le_sha256"],
                    "rhs": f8_sha(rhs, np) == finalist["rhs_float64_le_sha256"],
                    "shape": [int(value) for value in matrix.shape]
                    == finalist["matrix_shape"],
                }
                if not all(primary_hash_checks.values()):
                    raise ProtocolError(
                        f"finalist primary matrix reconstruction drift: {primary_hash_checks}"
                    )
                first_assembly = False
            solution = solve_active_qp(matrix, rhs, np, optimize)
            if solution is None:
                if first_solution_recheck:
                    raise ProtocolError("finalist primary QP replay unexpectedly rejected")
                failure = "ACTIVE_QP_SOLVER_NORM_RANK_CONDITION_OR_RESIDUAL_GATE"
                break
            if first_solution_recheck:
                primary_solution_checks = {
                    "delta_hash": solution["delta_float64_le_sha256"]
                    == finalist["delta_float64_le_sha256"],
                    "norm": float(solution["norm"]) == float(finalist["norm"]),
                    "rank": int(solution["rank"]) == int(finalist["rank"]),
                    "condition": float(solution["condition"])
                    == float(finalist["condition"]),
                    "residual_min": float(solution["residual_min"])
                    == float(finalist["residual_min"]),
                }
                if not all(primary_solution_checks.values()):
                    raise ProtocolError(
                        f"finalist primary QP replay drift: {primary_solution_checks}"
                    )
                first_solution_recheck = False
            delta = solution["delta"]
            primary_violations = []
            for guard_slot in guard_scan_slots:
                report = guard_reports[guard_slot]
                required = (
                    ZERO_GUARD_BUFFER
                    if report["zero_primary_margin"]
                    else -float(report["parent_primary_margin"])
                )
                residual = float(base[guard_offset + guard_slot] @ delta) - required
                if residual < -CONSTRAINT_TOLERANCE:
                    primary_violations.append(
                        (
                            residual,
                            str(report["line_sha256"]),
                            int(report["primary_stage"]),
                            int(report["primary_competitor"]),
                            guard_slot,
                        )
                    )
            if primary_violations:
                residual, _, _, _, guard_slot = min(primary_violations)
                if guard_slot in active_primary_guards or primary_recuts >= MAX_PRIMARY_CUTS:
                    failure = "PRIMARY_RETENTION_RECUT_STALLED"
                    break
                primary_cut_additions.append(
                    {
                        "guard_slot": guard_slot,
                        "residual_before_cut": residual,
                        "added_during": "full_finalist_certification",
                    }
                )
                active_primary_guards.add(guard_slot)
                primary_recuts += 1
                continue
            active_gradient_cache: dict[tuple[Any, ...], Any] = {}
            for active_record in active_retained:
                threat_identity = active_record.get("threat_identity")
                if threat_identity is None:
                    continue
                identity = tuple(threat_identity)
                if identity not in active_gradient_cache:
                    active_gradient_cache[identity] = active_record["vector"]
            violation, scan_audit = full_threat_scan(
                delta,
                primary["margin_logits_float32"],
                batch,
                rows,
                target_tuple,
                selection,
                primary,
                parameters,
                current,
                helper,
                np,
                active_gradient_cache,
            )
            scans += 1
            total_full_scans += 1
            total_full_vjps += int(scan_audit["vjp_calls"])
            total_active_cache_hits += int(scan_audit["active_gradient_cache_hits"])
            if violation is None:
                certified_solution = solution
                break
            record = dict(violation["record"])
            identity = tuple(record["constraint_identity"])
            active_identities = {
                tuple(value["identity"]) for value in active_retained
            }
            if identity in active_identities:
                failure = "FULL_THREAT_ACTIVE_CUT_STALLED"
                break
            new_record = full_cut_record(violation)
            if str(record["role"]) == "target":
                full_target_records.append(new_record)
                full_target_cuts += 1
            else:
                full_retention_records.append(new_record)
                full_retention_cuts += 1
            full_cuts += 1
        examined_record = {
            "frozen_finalist_index": finalist_index,
            "target_line_sha256": list(target_sha),
            "target_episode_id": [
                str(target_reports[slot]["episode_id"]) for slot in target_tuple
            ],
            "primary_plan_l2": float(finalist["norm"]),
            "primary_predicted_pf64_repairs": int(finalist["predicted_pf64_repairs"]),
            "primary_worst_target_slack": float(finalist["worst_target_slack"]),
            "full_threat_scans": scans,
            "full_threat_lazy_cuts": full_cuts,
            "full_target_lazy_cuts": full_target_cuts,
            "full_retention_lazy_cuts": full_retention_cuts,
            "primary_recuts": primary_recuts,
            "active_constraint_count": len(active_names),
            "last_active_matrix_float64_le_sha256": f8_sha(matrix, np),
            "last_active_rhs_float64_le_sha256": f8_sha(rhs, np),
            "failure": failure,
            "certified": certified_solution is not None,
        }
        examined.append(examined_record)
        if certified_solution is not None:
            selected = {
                **examined_record,
                "unscaled_nominal_l2": float(certified_solution["norm"]),
                "actor_l2_absolute_future_cap": ACTOR_L2_CAP,
                "atlas_l2_ceiling": ATLAS_L2_CEILING,
                "linear_plan_float64_le_sha256": certified_solution[
                    "delta_float64_le_sha256"
                ],
                "linear_plan_values_published": False,
                "actor_bytes_or_candidate_payload_published": False,
                "active_rank": int(certified_solution["rank"]),
                "active_condition": float(certified_solution["condition"]),
                "active_residual_min": float(certified_solution["residual_min"]),
                "active_matrix_shape": certified_solution["matrix_shape"],
                "active_matrix_float64_le_sha256": certified_solution[
                    "matrix_float64_le_sha256"
                ],
                "active_rhs_float64_le_sha256": certified_solution[
                    "rhs_float64_le_sha256"
                ],
                "active_constraint_manifest": active_names,
            }
            return {
                "status": CERTIFIED_STATUS,
                "selected": selected,
                "examined": examined,
                "audit": {
                    "frozen_finalist_order_preserved": True,
                    "first_certified_selected_without_reranking": True,
                    "maximum_finalists": MAX_FINALISTS,
                    "full_threat_scans": total_full_scans,
                    "full_threat_vjp_calls": total_full_vjps,
                    "active_gradient_cache_hits": total_active_cache_hits,
                    "active_gradients_only_retained": True,
                },
            }
    return {
        "status": NO_CERTIFICATE_STATUS,
        "selected": None,
        "examined": examined,
        "audit": {
            "frozen_finalist_order_preserved": True,
            "maximum_finalists": MAX_FINALISTS,
            "finalists_available": len(finalists),
            "finalists_examined": len(examined),
            "full_threat_scans": total_full_scans,
            "full_threat_vjp_calls": total_full_vjps,
            "active_gradient_cache_hits": total_active_cache_hits,
            "active_gradients_only_retained": True,
            "absence_is_not_global_infeasibility": True,
            "meaning": "the frozen width-512 beam produced no fully certified cohort",
        },
    }


def sanitized_exception(error: BaseException) -> dict[str, Any]:
    class_name = type(error).__name__
    if not class_name.isascii() or not class_name.isidentifier():
        class_name = "NonCanonicalExceptionClass"
    try:
        message_bytes = str(error).encode("utf-8", errors="strict")
    except BaseException:
        message_bytes = b"exception-message-unavailable"
    return {
        "exception_class": class_name,
        "message_utf8_bytes": len(message_bytes),
        "message_sha256": sha256_bytes(message_bytes),
        "message_published": False,
        "repr_traceback_or_raw_values_published": False,
    }


def capture_execution_snapshot(
    snapshot: dict[str, Any],
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    snapshot["python_random"] = random.getstate()
    snapshot["numpy_random"] = np.random.get_state()
    snapshot["torch_cpu_random"] = torch.get_rng_state()
    snapshot["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    snapshot["deterministic_algorithms"] = (
        torch.are_deterministic_algorithms_enabled()
    )
    snapshot["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    snapshot["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    snapshot["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    snapshot["sys_path"] = list(sys.path)
    snapshot["module_absence"] = {
        name: name not in sys.modules for name in HELPER_STACK_NAMES
    }
    checks = {
        "python_rng_snapshotted": "python_random" in snapshot,
        "numpy_rng_snapshotted": "numpy_random" in snapshot,
        "torch_cpu_rng_snapshotted": "torch_cpu_random" in snapshot,
        "torch_all_cuda_rng_snapshotted": "torch_cuda_random" in snapshot,
        "backend_flags_snapshotted": all(
            key in snapshot
            for key in (
                "deterministic_algorithms",
                "cudnn_benchmark",
                "cudnn_deterministic",
                "float32_matmul_precision",
            )
        ),
        "sys_path_snapshotted": "sys_path" in snapshot,
        "six_private_module_names_absent": all(
            snapshot["module_absence"].values()
        ),
    }
    if not all(checks.values()):
        raise ProtocolError("pre-helper execution snapshot gate failed")
    return {"pass": True, "checks": checks}


def numpy_rng_equal(left: Any, right: Any, np: Any) -> bool:
    return (
        isinstance(left, tuple)
        and isinstance(right, tuple)
        and len(left) == len(right) == 5
        and left[0] == right[0]
        and bool(np.array_equal(left[1], right[1]))
        and left[2:] == right[2:]
    )


def restore_model_context(context: Mapping[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "model_instantiated": context.get("model") is not None,
        "restoration_evidence_available": None,
        "restoration_status": None,
        "attempted": False,
        "parent_state_exact": None,
        "actor_bytes_exact": None,
        "requires_grad_restored": None,
        "training_mode_restored": None,
        "grad_buffers_restored": None,
        "pass": None,
        "errors": {},
    }
    model = context.get("model")
    if model is None:
        facts.update(
            {
                "restoration_evidence_available": False,
                "restoration_status": "NOT_APPLICABLE_MODEL_NOT_INSTANTIATED",
            }
        )
        return facts
    facts["restoration_evidence_available"] = True
    facts["restoration_status"] = "MODEL_RESTORATION_ATTEMPTED"
    facts["attempted"] = True
    helper = context.get("helper")
    cw20 = context.get("cw20")
    np = context.get("np")
    requires_grad_before = context.get("requires_grad_before")
    training_before = context.get("training_before")
    named_after: dict[str, Any] | None = None
    try:
        named_after = dict(model.named_parameters())
    except BaseException as error:
        facts["errors"]["named_parameters"] = sanitized_exception(error)
    if named_after is not None and isinstance(requires_grad_before, Mapping):
        try:
            exact_names = set(named_after) == set(requires_grad_before)
            if exact_names:
                for name, parameter in named_after.items():
                    parameter.requires_grad_(bool(requires_grad_before[name]))
            facts["requires_grad_restored"] = exact_names and all(
                bool(named_after[name].requires_grad)
                == bool(requires_grad_before[name])
                for name in named_after
            )
        except BaseException as error:
            facts["requires_grad_restored"] = False
            facts["errors"]["requires_grad"] = sanitized_exception(error)
        try:
            for parameter in named_after.values():
                parameter.grad = None
            facts["grad_buffers_restored"] = all(
                parameter.grad is None for parameter in named_after.values()
            )
        except BaseException as error:
            facts["grad_buffers_restored"] = False
            facts["errors"]["grad_buffers"] = sanitized_exception(error)
    if training_before is not None:
        try:
            model.train(bool(training_before))
            facts["training_mode_restored"] = bool(model.training) == bool(
                training_before
            )
        except BaseException as error:
            facts["training_mode_restored"] = False
            facts["errors"]["training_mode"] = sanitized_exception(error)
    if helper is not None and context.get("parent_state_before") is not None:
        try:
            facts["parent_state_exact"] = (
                helper.model_state_sha256(model.state_dict())
                == context.get("parent_state_before")
                == PARENT_MODEL_SHA256
            )
        except BaseException as error:
            facts["parent_state_exact"] = False
            facts["errors"]["parent_state"] = sanitized_exception(error)
    if (
        cw20 is not None
        and np is not None
        and context.get("parameters")
        and context.get("parent_actor_bytes") is not None
    ):
        try:
            facts["actor_bytes_exact"] = (
                cw20.actor_bytes(context["parameters"], ACTOR6_NAMES, np)
                == context["parent_actor_bytes"]
            )
        except BaseException as error:
            facts["actor_bytes_exact"] = False
            facts["errors"]["actor_bytes"] = sanitized_exception(error)
    required = (
        "parent_state_exact",
        "actor_bytes_exact",
        "requires_grad_restored",
        "training_mode_restored",
        "grad_buffers_restored",
    )
    facts["pass"] = all(facts.get(key) is True for key in required)
    facts["restoration_status"] = (
        "MODEL_RESTORATION_PASS"
        if facts["pass"] is True
        else "MODEL_RESTORATION_INCOMPLETE_OR_FAILED"
    )
    return facts


def restore_global_state(
    snapshot: Mapping[str, Any],
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "snapshot_complete": all(
            key in snapshot
            for key in (
                "python_random",
                "numpy_random",
                "torch_cpu_random",
                "torch_cuda_random",
                "deterministic_algorithms",
                "cudnn_benchmark",
                "cudnn_deterministic",
                "float32_matmul_precision",
            )
        ),
        "python_rng_restored": None,
        "numpy_rng_restored": None,
        "torch_cpu_rng_restored": None,
        "torch_all_cuda_rng_restored": None,
        "deterministic_algorithms_restored": None,
        "cudnn_benchmark_restored": None,
        "cudnn_deterministic_restored": None,
        "float32_matmul_precision_restored": None,
        "pass": None,
        "errors": {},
    }

    def attempt(name: str, setter: Any, checker: Any) -> None:
        try:
            setter()
            facts[name] = bool(checker())
        except BaseException as error:
            facts[name] = False
            facts["errors"][name] = sanitized_exception(error)

    if "python_random" in snapshot:
        attempt(
            "python_rng_restored",
            lambda: random.setstate(snapshot["python_random"]),
            lambda: random.getstate() == snapshot["python_random"],
        )
    if "numpy_random" in snapshot:
        attempt(
            "numpy_rng_restored",
            lambda: np.random.set_state(snapshot["numpy_random"]),
            lambda: numpy_rng_equal(
                np.random.get_state(), snapshot["numpy_random"], np
            ),
        )
    if "torch_cpu_random" in snapshot:
        attempt(
            "torch_cpu_rng_restored",
            lambda: torch.set_rng_state(snapshot["torch_cpu_random"]),
            lambda: bool(
                torch.equal(torch.get_rng_state(), snapshot["torch_cpu_random"])
            ),
        )
    if "torch_cuda_random" in snapshot:
        attempt(
            "torch_all_cuda_rng_restored",
            lambda: torch.cuda.set_rng_state_all(snapshot["torch_cuda_random"]),
            lambda: len(torch.cuda.get_rng_state_all())
            == len(snapshot["torch_cuda_random"])
            and all(
                bool(torch.equal(left, right))
                for left, right in zip(
                    torch.cuda.get_rng_state_all(), snapshot["torch_cuda_random"]
                )
            ),
        )
    if "deterministic_algorithms" in snapshot:
        attempt(
            "deterministic_algorithms_restored",
            lambda: torch.use_deterministic_algorithms(
                bool(snapshot["deterministic_algorithms"])
            ),
            lambda: torch.are_deterministic_algorithms_enabled()
            == bool(snapshot["deterministic_algorithms"]),
        )
    if "cudnn_benchmark" in snapshot:
        attempt(
            "cudnn_benchmark_restored",
            lambda: setattr(
                torch.backends.cudnn,
                "benchmark",
                bool(snapshot["cudnn_benchmark"]),
            ),
            lambda: torch.backends.cudnn.benchmark
            == bool(snapshot["cudnn_benchmark"]),
        )
    if "cudnn_deterministic" in snapshot:
        attempt(
            "cudnn_deterministic_restored",
            lambda: setattr(
                torch.backends.cudnn,
                "deterministic",
                bool(snapshot["cudnn_deterministic"]),
            ),
            lambda: torch.backends.cudnn.deterministic
            == bool(snapshot["cudnn_deterministic"]),
        )
    if "float32_matmul_precision" in snapshot:
        attempt(
            "float32_matmul_precision_restored",
            lambda: torch.set_float32_matmul_precision(
                str(snapshot["float32_matmul_precision"])
            ),
            lambda: torch.get_float32_matmul_precision()
            == str(snapshot["float32_matmul_precision"]),
        )
    required = (
        "python_rng_restored",
        "numpy_rng_restored",
        "torch_cpu_rng_restored",
        "torch_all_cuda_rng_restored",
        "deterministic_algorithms_restored",
        "cudnn_benchmark_restored",
        "cudnn_deterministic_restored",
        "float32_matmul_precision_restored",
    )
    facts["pass"] = facts["snapshot_complete"] is True and all(
        facts.get(key) is True for key in required
    )
    return facts


def cleanup_helper_stack(
    installed: Mapping[str, ModuleType],
    installation_order: Sequence[str],
) -> dict[str, Any]:
    removal: dict[str, bool | None] = {name: None for name in HELPER_STACK_NAMES}
    errors: dict[str, Any] = {}
    for name in reversed(list(installation_order)):
        module = installed.get(name)
        try:
            if module is None or sys.modules.get(name) is not module:
                removal[name] = False
            else:
                sys.modules.pop(name)
                removal[name] = name not in sys.modules
        except BaseException as error:
            removal[name] = False
            errors[name] = sanitized_exception(error)
    absence = {name: name not in sys.modules for name in HELPER_STACK_NAMES}
    order_is_prefix = list(installation_order) == list(
        HELPER_STACK_NAMES[: len(installation_order)]
    )
    installed_removed_exact = all(
        removal.get(name) is True for name in installation_order
    )
    return {
        "reverse_installation_order_used": True,
        "installation_order": list(installation_order),
        "installation_order_is_authoritative_prefix": order_is_prefix,
        "identity_exact_removal": removal,
        "all_six_absent_afterward": all(absence.values()),
        "absence_afterward": absence,
        "errors": errors,
        "pass": order_is_prefix
        and installed_removed_exact
        and all(absence.values()),
    }


def restore_sys_path(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if "sys_path" not in snapshot:
        return {
            "snapshot_available": False,
            "restored_exact": None,
            "pass": None,
            "error": None,
        }
    try:
        sys.path[:] = list(snapshot["sys_path"])
        exact = sys.path == list(snapshot["sys_path"])
        return {
            "snapshot_available": True,
            "restored_exact": exact,
            "pass": exact,
            "error": None,
        }
    except BaseException as error:
        return {
            "snapshot_available": True,
            "restored_exact": False,
            "pass": False,
            "error": sanitized_exception(error),
        }


def run_atlas(
    profile: Mapping[str, Any],
    profile_evidence: Mapping[str, Any],
    current: ModuleType,
    cw20: ModuleType,
    helper: ModuleType,
    dependencies: Mapping[str, Any],
    runtime: Mapping[str, Any],
    model_restore_context: dict[str, Any],
    np: Any,
    optimize: Any,
    torch: Any,
) -> dict[str, Any]:
    if helper.torch is not torch:
        raise ProtocolError("hash-bound helper torch runtime identity drift")
    model = None
    parameters: dict[str, Any] = {}
    parent_actor_bytes = None
    result: dict[str, Any] | None = None
    model_restore_context.update(
        {"helper": helper, "cw20": cw20, "np": np, "model": None}
    )
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("high")
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda:0")
        b256_rows, selection = validate_profile(profile, current)
        parent_payload, parent_evidence = read_regular_bytes(
            PARENT,
            PARENT_FILE_SHA256,
            "guarded U468 parent",
            expected_mode=0o444,
        )
        checkpoint = helper.checkpoint_from_bytes(parent_payload, "guarded U468 parent")
        state = checkpoint.get("model_state_dict")
        checkpoint_checks = {
            "update468": checkpoint.get("update") == 468,
            "kind_ppo": helper.evaluator.checkpoint_kind(checkpoint) == "ppo",
            "state80": isinstance(state, Mapping) and len(state) == 80,
            "model_sha_exact": isinstance(state, Mapping)
            and helper.model_state_sha256(state) == PARENT_MODEL_SHA256,
        }
        if not all(checkpoint_checks.values()):
            raise ProtocolError(f"guarded parent checkpoint drift: {checkpoint_checks}")
        model, model_config, kind = helper.instantiate_checkpoint(checkpoint, device)
        model_restore_context["model"] = model
        if kind != "ppo" or helper.model_state_sha256(model.state_dict()) != PARENT_MODEL_SHA256:
            raise ProtocolError("guarded parent runtime identity drift")
        parent_state_before = helper.model_state_sha256(model.state_dict())
        training_before = bool(model.training)
        named = dict(model.named_parameters())
        requires_grad_before = {name: bool(parameter.requires_grad) for name, parameter in named.items()}
        if any(parameter.grad is not None for parameter in named.values()):
            raise ProtocolError("fresh parent has gradient buffers")
        model_restore_context.update(
            {
                "parent_state_before": parent_state_before,
                "training_before": training_before,
                "requires_grad_before": dict(requires_grad_before),
                "grad_buffers_before_all_none": True,
            }
        )
        model.eval()
        for parameter in named.values():
            parameter.requires_grad_(False)
            parameter.grad = None
        parameters = {name: named[name] for name in ACTOR6_NAMES}
        for parameter in parameters.values():
            parameter.requires_grad_(True)
        parent_actor_bytes = cw20.actor_bytes(parameters, ACTOR6_NAMES, np)
        model_restore_context.update(
            {
                "parameters": dict(parameters),
                "parent_actor_bytes": parent_actor_bytes,
            }
        )
        nonactor_names = sorted(set(model.state_dict()).difference(ACTOR6_NAMES))
        actor_checks = {
            "actor_dimension": sum(int(value.numel()) for value in parameters.values())
            == ACTOR_DIMENSION,
            "actor_all_float32": all(str(value.dtype) == "torch.float32" for value in parameters.values()),
            "actor_bytes_sha": sha256_bytes(parent_actor_bytes) == PARENT_ACTOR_BYTES_SHA256,
            "actor_state_sha": helper.model_state_sha256(
                {name: model.state_dict()[name] for name in ACTOR6_NAMES}
            )
            == PARENT_ACTOR_STATE_SHA256,
            "nonactor74": len(nonactor_names) == 74,
            "nonactor_sha": helper.model_state_sha256(
                {name: model.state_dict()[name] for name in nonactor_names}
            )
            == PARENT_NONACTOR_SHA256,
            "only_actor6_requires_grad": all(parameters[name].requires_grad for name in ACTOR6_NAMES)
            and all(not named[name].requires_grad for name in nonactor_names),
        }
        if not all(actor_checks.values()):
            raise ProtocolError(f"parent actor/nonactor drift: {actor_checks}")
        batch_cpu, cache_audit = current.load_b256(
            b256_rows, model_config, helper
        )
        if int(batch_cpu["action_counts"].shape[0]) != B256_SIZE:
            raise ProtocolError("exact B256 batch shape drift")
        sample_weight_audit = audit_b256_sample_weights(
            batch_cpu, b256_rows, np, torch
        )
        batch = {name: value.to(device) for name, value in batch_cpu.items()}
        outputs = helper.ppo.model_forward(model, batch, device)
        primary = build_primary_atlas(
            outputs,
            batch,
            b256_rows,
            selection,
            parameters,
            helper,
            current,
            np,
        )
        beam = primary_beam_search(
            primary, b256_rows, selection, np, optimize
        )
        certificate = certify_finalists(
            beam,
            batch,
            b256_rows,
            selection,
            primary,
            parameters,
            current,
            helper,
            np,
            optimize,
        )
        parent_after = helper.model_state_sha256(model.state_dict())
        no_mutation_checks = {
            "parent_model_state_exact": parent_after
            == parent_state_before
            == PARENT_MODEL_SHA256,
            "actor_bytes_exact": cw20.actor_bytes(parameters, ACTOR6_NAMES, np)
            == parent_actor_bytes,
            "nonactor_exact": helper.model_state_sha256(
                {name: model.state_dict()[name] for name in nonactor_names}
            )
            == PARENT_NONACTOR_SHA256,
            "all_grad_buffers_none": all(parameter.grad is None for parameter in model.parameters()),
        }
        if not all(no_mutation_checks.values()):
            raise ProtocolError(f"parent-only atlas mutation detected: {no_mutation_checks}")
        primary_matrix = primary["base_matrix"]
        result = {
            "schema_version": SCHEMA,
            "status": str(certificate["status"]),
            "seed": SEED,
            "input_lock": {
                "parent": parent_evidence,
                "profile": dict(profile_evidence),
                "dependencies": dict(dependencies),
            },
            "runtime": dict(runtime),
            "checkpoint_checks": checkpoint_checks,
            "actor_nonactor_checks": actor_checks,
            "profile_selection": selection,
            "exact_b256_cache": cache_audit,
            "authoritative_b256_sample_weights": sample_weight_audit,
            "canonical_graph": {
                "batch_size": B256_SIZE,
                "model_forward_count": 1,
                "native_dtype": "torch.bfloat16",
                "repacking_reordering_microbatch_or_shape_change": False,
                "all_VJPs_from_same_graph": True,
            },
            "primary_atlas": {
                "matrix_shape": [int(value) for value in primary_matrix.shape],
                "matrix_float32_le_sha256": f4_sha(primary_matrix, np),
                "storage_mib": float(primary_matrix.nbytes / (1024 * 1024)),
                "target_rows": primary["target_reports"],
                "retention_rows": primary["retention_reports"],
                "aggregate_gradients": primary["aggregate_reports"],
                "parent_losses": primary["parent_losses"],
                "ordered_nll_contract": primary["ordered_nll_contract"],
                "autograd_grad_calls": primary["autograd_grad_calls"],
            },
            "frozen_beam": {
                "survivor_pool_slots": beam["survivors"],
                "survivor_line_sha256": beam["survivor_line_sha256"],
                "audit": beam["audit"],
                "size5_finalist_manifest": [
                    {
                        "frozen_index": index,
                        "target_line_sha256": [
                            str(primary["target_reports"][int(slot)]["line_sha256"])
                            for slot in record["target_tuple"]
                        ],
                        "primary_l2": float(record["norm"]),
                        "primary_plan_float64_le_sha256": record[
                            "delta_float64_le_sha256"
                        ],
                        "primary_matrix_shape": record["matrix_shape"],
                        "primary_matrix_float64_le_sha256": record[
                            "matrix_float64_le_sha256"
                        ],
                        "primary_rhs_float64_le_sha256": record[
                            "rhs_float64_le_sha256"
                        ],
                        "primary_rank": int(record["rank"]),
                        "primary_condition": float(record["condition"]),
                        "primary_residual_min": float(record["residual_min"]),
                        "primary_solver_status": int(record["solver_status"]),
                        "primary_constraint_count": len(
                            record["primary_constraint_manifest"]
                        ),
                        "selected_target5_NLL_constraint": next(
                            dict(value)
                            for value in record["primary_constraint_manifest"]
                            if value.get("kind") == "selected_target5"
                        ),
                        "predicted_pf64_repairs": int(record["predicted_pf64_repairs"]),
                        "worst_target_slack": float(record["worst_target_slack"]),
                    }
                    for index, record in enumerate(beam["finalists"])
                ],
            },
            "full_threat_certification": certificate,
            "parent_only_integrity": no_mutation_checks,
            "execution_counts": {
                "parent_native_BF16_B256_forwards": 1,
                "candidate_forwards": 0,
                "primary_autograd_grad_calls": primary["autograd_grad_calls"],
                "full_threat_autograd_grad_calls": certificate["audit"][
                    "full_threat_vjp_calls"
                ],
                "backward_calls": 0,
                "optimizer_instances": 0,
                "optimizer_steps": 0,
                "parameter_value_writes": 0,
                "model_overlays_constructed": 0,
                "changed_candidates_constructed_or_evaluated": 0,
            },
            "candidate_accounting": {
                "atlas_branch_changed_candidates_constructed_or_evaluated": 0,
                "atlas_branch_official_unique_changed_candidates_consumed": 0,
                "historical_closed_boundary_other_branch_train_shadow_candidates": 1,
                "global_official_unique_changed_candidates_consumed_before_atlas": 2,
                "global_official_budget": 12,
                "global_official_unique_changed_candidates_remaining": 10,
                "atlas_certificate_authorizes_or_constructs_candidate": False,
            },
            "candidate_payload": None,
            "candidate_authorized": False,
            "scope": {
                "train_only": True,
                "validation_test_broad_gold_opened": False,
                "network_package_upload_submission": False,
                "checkpoint_or_actor_payload_writes": 0,
                "changed_candidates_constructed_or_evaluated": 0,
            },
        }
        del outputs, primary_matrix, batch
    finally:
        model_restore_context["run_atlas_exited"] = True
    if result is None:
        raise ProtocolError("parent-only atlas result was not constructed")
    if result.get("status") not in {CERTIFIED_STATUS, NO_CERTIFICATE_STATUS}:
        raise ProtocolError("parent-only atlas returned a non-decision status")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="static")
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--profile-sha256")
    parser.add_argument("--expected-self-sha256", required=True)
    parser.add_argument("--execution-prereg-sha256")
    return parser.parse_args()


def execution_failure_result(
    error: BaseException,
    phase: str,
    self_evidence: Mapping[str, Any],
    pre_attempt_runtime: Mapping[str, Any],
    cuda_runtime: Mapping[str, Any] | None,
    control: Mapping[str, Any],
    execution: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    profile_evidence: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    ledger: Mapping[str, Any],
    snapshot_audit: Mapping[str, Any] | None,
    helper_stack_evidence: Mapping[str, Any] | None,
    restoration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": EXECUTION_FAILED_STATUS,
        "attempt_consumed": True,
        "retry_authorized": False,
        "certificate_evaluation_complete": False,
        "certificate_status": None,
        "do_not_interpret_as_no_certificate": True,
        "mathematical_feasibility_conclusion_published": False,
        "process_evidence_partial": True,
        "phase": phase,
        "failure": sanitized_exception(error),
        "self": dict(self_evidence),
        "runtime_before_attempt": dict(pre_attempt_runtime),
        "cuda_runtime_after_attempt": (
            None if cuda_runtime is None else dict(cuda_runtime)
        ),
        "control_plane": dict(control),
        "execution_preregistration": dict(execution),
        "source_audit": dict(source_audit),
        "input_lock": {
            "profile": dict(profile_evidence),
            "dependencies_before_attempt": dict(dependencies),
            "global_official_candidate_ledger": dict(ledger),
        },
        "execution_snapshot": (
            None if snapshot_audit is None else dict(snapshot_audit)
        ),
        "helper_stack": (
            None
            if helper_stack_evidence is None
            else dict(helper_stack_evidence)
        ),
        "restoration": dict(restoration),
        "candidate_payload": None,
        "candidate_authorized": False,
        "candidate_accounting": {
            "atlas_branch_changed_candidates_constructed_or_evaluated": 0,
            "atlas_branch_official_unique_changed_candidates_consumed": 0,
            "global_official_unique_changed_candidates_consumed_before_atlas": 2,
            "global_official_budget": 12,
            "global_official_unique_changed_candidates_remaining": 10,
            "atlas_certificate_authorizes_or_constructs_candidate": False,
        },
        "scope": {
            "train_only": True,
            "validation_test_broad_gold_opened": False,
            "network_package_upload_submission": False,
            "checkpoint_or_actor_payload_writes": 0,
            "changed_candidates_constructed_or_evaluated": 0,
            "partial_process_evidence_only": True,
        },
    }


def main() -> None:
    args = parse_args()
    if not is_sha256(args.expected_self_sha256):
        raise ProtocolError("--expected-self-sha256 must be a lowercase SHA-256")
    pre_attempt_runtime = validate_runtime(require_cuda=False)
    source, self_evidence = read_regular_bytes(
        SCRIPT,
        args.expected_self_sha256,
        "cohort-atlas runner",
        expected_mode=0o555,
    )
    source_audit = static_source_audit(source)
    control = audit_control_plane()
    dependencies = audit_dependencies()
    ledger = audit_global_ledger()
    if args.mode == "static":
        initial_absence = publication_absence()
        if initial_absence["both_absent"] is not True:
            raise ProtocolError(
                f"one-shot publication targets are not absent: {initial_absence}"
            )
        final_absence = publication_absence()
        if final_absence != initial_absence or final_absence["both_absent"] is not True:
            raise ProtocolError(f"static target state changed: {final_absence}")
        static_result = {
            "schema_version": SCHEMA,
            "status": "STATIC_AUDIT_PASS_DESIGN_ONLY_RUN_NOT_AUTHORIZED",
            "self": self_evidence,
            "runtime": pre_attempt_runtime,
            "control_plane": control,
            "dependencies": dependencies,
            "source_audit": source_audit,
            "corrected_constants": {
                "exact_B256_forward_count": 1,
                "PF_pool64": PF_POOL_SIZE,
                "search_pool63": SEARCH_POOL_SIZE,
                "retention160": RETENTION_SIZE,
                "Dominic32": DOMINIC_SIZE,
                "cohort5": COHORT_SIZE,
                "beam_width": BEAM_WIDTH,
                "zero_guard_buffer": ZERO_GUARD_BUFFER,
                "earlier_one_over_1024_prose_authoritative": False,
                "atlas_l2_ceiling": ATLAS_L2_CEILING,
                "future_absolute_cap": ACTOR_L2_CAP,
                "correction_v2_sha256": CORRECTION_SHA256,
                "correction_v3_sha256": CORRECTION_V3_SHA256,
                "correction_v4_sha256": CORRECTION_V4_SHA256,
                "correction_v5_sha256": CORRECTION_V5_SHA256,
                "correction_v6_sha256": CORRECTION_V6_SHA256,
                "correction_v7_sha256": CORRECTION_V7_SHA256,
                "global_ledger_sha256": LEDGER_SHA256,
                "B256_sample_weights_float32_le_sha256": (
                    EXPECTED_B256_SAMPLE_WEIGHTS_F4_SHA256
                ),
            },
            "global_official_candidate_ledger": ledger,
            "scope": {
                "profile_opened": False,
                "execution_prereg_opened": False,
                "cuda_accessed": False,
                "model_instances": 0,
                "parent_forwards": 0,
                "candidate_forwards": 0,
                "parameter_value_writes": 0,
                "model_overlays_constructed": 0,
                "changed_candidates_constructed_or_evaluated": 0,
                "total_writes": 0,
            },
            "one_shot_publication": {
                "pre_post_absence_exact": True,
                "targets": final_absence,
                "attempt_marker_writes": 0,
                "result_writes": 0,
            },
        }
        print(canonical_json(static_result).decode("utf-8"), end="")
        return

    execution = audit_execution_preregistration(
        str(args.execution_prereg_sha256 or ""), self_evidence, ledger
    )
    if str(args.profile) != str(PROFILE) or args.profile.resolve() != PROFILE:
        raise ProtocolError("--profile must equal the frozen exact absolute path")
    if args.profile_sha256 != PROFILE_SHA256:
        raise ProtocolError("--profile-sha256 must equal the frozen exact SHA")
    directories = open_publication_directories(execution["publication_binding"])
    try:
        initial_absence = publication_absence(directories)
        if initial_absence["both_absent"] is not True:
            raise ProtocolError("held one-shot publication targets are not absent")
        profile_payload, profile_evidence = read_regular_bytes(
            PROFILE,
            PROFILE_SHA256,
            "frozen parent profile",
            expected_mode=0o444,
        )
        profile = strict_json(profile_payload, "frozen parent profile")
        final_absence = publication_absence(directories)
        if final_absence != initial_absence or final_absence["both_absent"] is not True:
            raise ProtocolError("held targets changed before attempt")
        marker_payload = {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "PARENT_ONLY_ATLAS_ATTEMPT_CONSUMED_BEFORE_CUDA_OR_MODEL",
            "consumed_at_utc": utc_now(),
            "branch": BRANCH,
            "attempt_number": 1,
            "runner": self_evidence,
            "master": control["master"],
            "correction_v2": control["correction_v2"],
            "correction_v3": control["correction_v3"],
            "correction_v4": control["correction_v4"],
            "correction_v5": control["correction_v5"],
            "correction_v6": control["correction_v6"],
            "correction_v7": control["correction_v7"],
            "execution_preregistration": execution,
            "profile": profile_evidence,
            "runtime_before_attempt": pre_attempt_runtime,
            "dependencies": dependencies,
            "global_official_candidate_ledger": ledger,
            "held_publication_directories": {
                "workspace_root": directories.root_evidence,
                "artifacts_directory": directories.artifacts_evidence,
            },
            "reserved_result": str(RESULT.relative_to(ROOT)),
            "candidate_accounting_before_cuda": {
                "atlas_branch_changed_candidates_constructed_or_evaluated": 0,
                "atlas_branch_official_unique_changed_candidates_consumed": 0,
                "candidate_forwards": 0,
                "model_overlays_constructed": 0,
                "candidate_authorized": False,
                "global_official_unique_changed_candidates_consumed_before_atlas": 2,
                "global_official_unique_changed_candidates_remaining": 10,
                "global_official_budget": 12,
            },
        }
        marker_bytes = canonical_json(marker_payload)
        marker_evidence = publish_o_excl(
            ATTEMPT_MARKER, marker_bytes, directories
        )

        phase = "cuda_runtime_validation_after_attempt"
        cuda_runtime: dict[str, Any] | None = None
        np: Any = None
        torch: Any = None
        optimize: Any = None
        global_snapshot: dict[str, Any] = {}
        snapshot_audit: dict[str, Any] | None = None
        installed: dict[str, ModuleType] = {}
        installation_order: list[str] = []
        helper_stack_evidence: dict[str, Any] | None = None
        model_restore_context: dict[str, Any] = {}
        provisional_result: dict[str, Any] | None = None
        execution_error: BaseException | None = None
        restoration_error: BaseException | None = None
        model_restoration: dict[str, Any]
        global_restoration: dict[str, Any]
        module_cleanup: dict[str, Any]
        path_restoration: dict[str, Any]
        try:
            cuda_runtime = validate_runtime(require_cuda=True)
            phase = "external_runtime_imports_after_attempt"
            import numpy as np
            import torch
            from scipy import optimize

            phase = "pre_repository_execution_snapshot_after_attempt"
            snapshot_audit = capture_execution_snapshot(global_snapshot, np, torch)
            phase = "verified_helper_stack_execution_after_attempt"
            helper, helper_stack_evidence = load_helper_stack(
                torch, installed, installation_order
            )
            current, current_import = exec_held_module(
                CURRENT,
                CURRENT_SHA256,
                "cohort_atlas_current_parent_math",
                0o555,
                installed,
                installation_order,
                keep_registered=False,
            )
            cw20, cw20_import = exec_held_module(
                CW20,
                CW20_SHA256,
                "cohort_atlas_cw20_codec",
                0o555,
                installed,
                installation_order,
                keep_registered=False,
            )
            run_dependencies = dict(dependencies)
            run_dependencies.update(
                {
                    "helper_stack_exact_byte_execution": helper_stack_evidence,
                    "current_parent_profile_math_exact_byte_execution": current_import,
                    "actor_codec_cw20_exact_byte_execution": cw20_import,
                    "formal_v3_executed": False,
                }
            )
            phase = "single_graph_parent_only_gradient_atlas_after_attempt"
            provisional_result = run_atlas(
                profile,
                profile_evidence,
                current,
                cw20,
                helper,
                run_dependencies,
                cuda_runtime,
                model_restore_context,
                np,
                optimize,
                torch,
            )
            phase = "completed_math_pending_independent_outer_restoration"
        except BaseException as error:
            execution_error = error
        finally:
            failure_phase = phase
            try:
                model_restoration = restore_model_context(model_restore_context)
            except BaseException as error:
                restoration_error = error
                model_restoration = {
                    "model_instantiated": model_restore_context.get("model") is not None,
                    "restoration_evidence_available": False,
                    "restoration_status": "MODEL_RESTORATION_EVIDENCE_UNAVAILABLE",
                    "attempted": None,
                    "parent_state_exact": None,
                    "actor_bytes_exact": None,
                    "requires_grad_restored": None,
                    "training_mode_restored": None,
                    "grad_buffers_restored": None,
                    "pass": False,
                    "errors": {"outer_model_restoration": sanitized_exception(error)},
                }
            try:
                if np is None or torch is None:
                    global_restoration = {
                        "snapshot_complete": False,
                        "restoration_evidence_available": False,
                        "pass": None,
                        "errors": {},
                    }
                else:
                    global_restoration = restore_global_state(
                        global_snapshot, np, torch
                    )
            except BaseException as error:
                if restoration_error is None:
                    restoration_error = error
                global_restoration = {
                    "snapshot_complete": False,
                    "restoration_evidence_available": False,
                    "pass": False,
                    "errors": {"outer_global_restoration": sanitized_exception(error)},
                }
            try:
                module_cleanup = cleanup_helper_stack(
                    installed, installation_order
                )
            except BaseException as error:
                if restoration_error is None:
                    restoration_error = error
                module_cleanup = {
                    "all_six_absent_afterward": None,
                    "pass": False,
                    "errors": {"outer_module_cleanup": sanitized_exception(error)},
                }
            try:
                path_restoration = restore_sys_path(global_snapshot)
            except BaseException as error:
                if restoration_error is None:
                    restoration_error = error
                path_restoration = {
                    "snapshot_available": "sys_path" in global_snapshot,
                    "restored_exact": None,
                    "pass": False,
                    "error": sanitized_exception(error),
                }

        restoration = {
            "outer_finally_order": [
                "model",
                "global_rng_and_backend",
                "private_modules_reverse_order",
                "sys_path",
            ],
            "model": model_restoration,
            "global_rng_and_backend": global_restoration,
            "private_modules": module_cleanup,
            "sys_path": path_restoration,
        }
        restoration["pass"] = all(
            component.get("pass") is True
            for component in (
                model_restoration,
                global_restoration,
                module_cleanup,
                path_restoration,
            )
        )
        completed_normally = (
            execution_error is None
            and restoration_error is None
            and provisional_result is not None
            and helper_stack_evidence is not None
            and restoration["pass"] is True
            and provisional_result.get("status")
            in {CERTIFIED_STATUS, NO_CERTIFICATE_STATUS}
        )
        if completed_normally:
            result = provisional_result
            result["certificate_evaluation_complete"] = True
            result["certificate_status"] = result["status"]
            result["restoration"] = restoration
            result["self"] = self_evidence
            result["control_plane"] = control
            result["execution_preregistration"] = execution
            result["source_audit"] = source_audit
            result["global_official_candidate_ledger"] = ledger
        else:
            if execution_error is not None:
                failure_error = execution_error
            elif restoration_error is not None:
                failure_error = restoration_error
                failure_phase = "independent_outer_restoration_after_attempt"
            else:
                failure_error = ProtocolError(
                    "completed math or restoration gate was incomplete"
                )
                failure_phase = "independent_outer_restoration_after_attempt"
            provisional_result = None
            result = execution_failure_result(
                failure_error,
                failure_phase,
                self_evidence,
                pre_attempt_runtime,
                cuda_runtime,
                control,
                execution,
                source_audit,
                profile_evidence,
                dependencies,
                ledger,
                snapshot_audit,
                helper_stack_evidence,
                restoration,
            )
        result["formal_publication"] = {
            "branch": BRANCH,
            "attempt_marker": marker_evidence,
            "attempt_before_cuda_model": True,
            "result_path": str(RESULT.relative_to(ROOT)),
            "result_absolute_path": str(RESULT),
            "held_directory_fd_O_EXCL_mode_0444": True,
            "candidate_authorized": False,
            "retry_authorized": False,
        }
        result_scope = dict(result.get("scope", {}))
        result_scope["evidence_writes"] = [
            str(ATTEMPT_MARKER.relative_to(ROOT)),
            str(RESULT.relative_to(ROOT)),
        ]
        result_scope["evidence_write_count"] = 2
        result["scope"] = result_scope
        result_bytes = canonical_json(result)
        result_evidence = publish_o_excl(RESULT, result_bytes, directories)
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "publication_status": "PARENT_ONLY_ATLAS_RESULT_PUBLISHED",
            "decision_status": result["status"],
            "attempt_marker": marker_evidence,
            "result": result_evidence,
            "result_payload_canonical_sha256": sha256_bytes(result_bytes),
            "candidate_payload_present": False,
            "candidate_authorized": False,
            "atlas_branch_changed_candidates_constructed_or_evaluated": 0,
            "atlas_branch_official_unique_changed_candidates_consumed": 0,
            "global_official_unique_changed_candidates_consumed_before_atlas": 2,
            "global_official_unique_changed_candidates_remaining": 10,
            "global_official_budget": 12,
            "retry_authorized": False,
        }
    finally:
        directories.close()
    print(canonical_json(summary).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
