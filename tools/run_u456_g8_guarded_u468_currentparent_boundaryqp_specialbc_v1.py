#!/usr/bin/env python3
"""Sole current-parent actor6 boundary-QP special-BC B256 gate.

This runner consumes one hash-bound, parent-only, train-only profile/selection.
It constructs exactly one actor6 RAM overlay from guarded U468, audits that
overlay on the frozen B256 selection, and publishes an absolute actor6 payload
only when every gate passes.  It never trains with an optimizer, writes a model
checkpoint, opens a validation/test member, evaluates broad/Gold, or submits.

Static mode does not import torch or construct/evaluate a changed candidate.
Run mode consumes one O_EXCL attempt before CUDA/model work, publishes one fixed
O_EXCL canonical result, prints only a compact publication summary, and always
restores the live model to the guarded parent before returning.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import stat
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "run_u456_g8_guarded_u468_currentparent_boundaryqp_specialbc_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u456-g8-guarded-u468-currentparent-boundaryqp-specialbc-v1"
ATTEMPT_SCHEMA = f"{SCHEMA}-attempt-v1"
PUBLICATION_SUMMARY_SCHEMA = f"{SCHEMA}-publication-summary-v1"
BRANCH = "ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_design202608143"
ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-attempt.json"
RESULT = ROOT / f"artifacts/{BRANCH}.result.json"
PROFILE_SCHEMA = "ptcg-u456-g8-guarded-u468-currentparent-train-profile-selection-v3"
PROFILE_STATUS = "completed_frozen_parent_only_train_profile_selection"
SOURCE_PROFILE_V2_SHA256 = (
    "ad25c407d4c08e53fccf0c883ea0b09d224bf1ad25e4fe7dd9031196b50dd5ce"
)
PROFILE_CORRECTOR_V3_SHA256 = (
    "4cc1444a9fbae83d088d1dcbbf9debba92eaa0a68d30b38b3ec417604c0e5ee8"
)
PROFILE_SELECTION_ENGINE_SHA256 = (
    "a398749d606d706d4622cfdfcdd36b7e6ad0c717a8d346af9f524554c18fa869"
)
EXPECTED_B256_SELECTION_SHA256 = (
    "6f0d8bcdc10b782c676d743a00a2f9e2dcc0501ea1054398c880aee31212f325"
)
DEFAULT_PROFILE = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_profile_selection_"
    "design202608143_v3.json"
)
EXPECTED_PROFILE_SHA256 = (
    "b7ef45722e50b2c71377b4dce8c7c2cee6ee1aaa16e99a1b874a0d932dab2d4e"
)

SEED = 202608143
TARGET_MARGIN = 1.0 / 512.0
ACTOR_L2_HARD_CAP = 1.0e-3
BATCH_SIZE = 256
CONTEXT34_ROWS = 16
CONTEXT34_SAMPLE_WEIGHT = 1.0 / 3.0
TARGET_COUNT = 5
GUARD_COUNT = 14
OTHER_ROW_COUNT = BATCH_SIZE - TARGET_COUNT - GUARD_COUNT
QP_FTOL = 1.0e-12
QP_MAXITER = 5000
QP_RESIDUAL_TOLERANCE = 1.0e-8
SVD_RELATIVE_RANK_TOLERANCE = 1.0e-12
NLL_TOLERANCE = 1.0e-6

ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
EFFECTIVE_ACTOR_NAMES = ACTOR6_NAMES[:-1]
INVARIANT_COMMON_LOGIT_BIAS = ACTOR6_NAMES[-1]
ACTOR_DIMENSION = 65793

PARENT = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141/ppo_stage/"
    "block3/B_gold_league/seed-202608141/checkpoints/update-0468.pt"
)
PARENT_FILE_SHA256 = "a9290745b8eb58704c4617594ac5ab49500c6e65ecd9f273a8285c12547fbb83"
PARENT_MODEL_SHA256 = "832c724277d6c2263d76ddbf41622f4ba06621ec3155fa8b7ae2591203db1c5e"
PARENT_ACTOR_FLOAT32_LE_SHA256 = (
    "03822d2e9dfd09843493894a22bc4ed22b608dbd7c4a05183ff8e507f773dc8a"
)
PARENT_ACTOR_STATE_SHA256 = (
    "b8d28f848f33003c5031d622bf04ddbd6c674f998b8e9d0f34564f8011dd4e23"
)
PARENT_NONACTOR_SHA256 = (
    "ce84a183fbe82d40861938ba3cc8f1fa93b19ff53aa14ecbc8eee18ccd00db87"
)

MASTER = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_"
    "design202608143.master_preregistration.json"
)
MASTER_SHA256 = "0efc5a08ba1a83ed26b0b518c826c5744d67e98895d2da21449e56291116bfa8"
CORRECTION_V2 = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_"
    "design202608143.master_correction_v2.json"
)
CORRECTION_V2_SHA256 = (
    "6accc12201900aba1a50d64d4ffd21796d36180cdb44990694b768a5a351263a"
)
CORRECTION_V3 = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_"
    "design202608143.master_correction_v3.json"
)
CORRECTION_V3_SHA256 = (
    "709972d0774536fd79ab5627d7b970e0b6ab4b5f2987d3e0e1f5f762b200461d"
)
CORRECTION_V4 = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_"
    "design202608143.master_correction_v4.json"
)
CORRECTION_V4_SHA256 = (
    "060fd238feba40908cb90fee81c81d4e4a981dfade274faf416233deae886857"
)
AUTHORIZATION = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141."
    "block3_training_integrity_decision.json"
)
AUTHORIZATION_SHA256 = (
    "d57954cdaeed5d0a2ff5c2a8d459253ba03f1b9a0f8dd1b5c760b23fc6f026bf"
)

FORMAL = TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal_v3.py"
FORMAL_SHA256 = "2d12ce9e76f6672911948dcc6458573899ec45fc39e748510544001025ede939"
PROFILE_CORRECTOR_V3 = TOOLS / "correct_u456_g8_guarded_u468_currentparent_selection_v3.py"
PROFILE_SELECTION_ENGINE = TOOLS / "profile_u456_g8_guarded_u468_currentparent_selection_v1.py"
CW20 = TOOLS / "probe_u468_raw_fixed_pcgrad_specialbc_cw20_v1.py"
CW20_SHA256 = "8f5b64c8a04b3de9a6d2890aac582967a50f4d708f3a11a64cbd395423d907d2"
CW22 = TOOLS / "probe_u468_cw11_targeted_b256_pcgrad_specialbc_cw22_v1.py"
CW22_SHA256 = "f307f3f26355125f97c857f14b97a73d9a7a4f2ea39ca8b0a578e801055e82b4"
CW23 = TOOLS / "probe_u468_cw11_boundary_qp_specialbc_cw23_v1.py"
CW23_SHA256 = "4a2d209af71a7ea5d955bd373f1fbada936688fc5bc597615c41192dc360b4b8"
CURRENT_FULLTRAIN = TOOLS / "run_u456_g8_guarded3x4_u468_exactp12_fulltrain24050_gate_v1.py"
CURRENT_FULLTRAIN_SHA256 = (
    "c75829fe5bf7d5b2bcf4069fac8bd505a86e531c795f63caf3438a9171eaefc2"
)
CW23_PAYLOAD_GATE = TOOLS / "run_cw23_payload_fulltrain_gate_v1.py"
CW23_PAYLOAD_GATE_SHA256 = (
    "70f7a5dc10f6f0cc6e5dc863b9592a292c2802a62584b3ef7c2c17776fe39130"
)

DATASETS = {
    "flg": ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
    "pokemonfan": ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
    "core5": ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
}
DATA_SHA256 = {
    "flg": "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    "pokemonfan": "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    "core5": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}
DATA_MODE = {"flg": 0o664, "pokemonfan": 0o664, "core5": 0o600}

B256_QUOTAS = {
    "pf_ctx0_hard": 32,
    "pf_ctx7_hard": 32,
    "dominic_ctx0_hard": 32,
    "pf_ctx0_retention": 32,
    "pf_ctx7_retention": 32,
    "dominic_ctx0_retention": 32,
    "broad_flg_retention": 16,
    "broad_pf_other_retention": 16,
    "broad_core_non_dominic_retention": 16,
    "broad_context34_retention": 16,
}
FORCED_FLG_GUARDS = {
    "f6d1d86202a3010abb8736c70396029c35478a9f79ed9d8e53704c51bfaacd64": 7876,
    "20ba6315e9860d54183bab71e4e818bd9847c299c52120e3fbcbc8481972d73f": 8711,
}
TARGET_LINE_SHA256 = (
    "d1d1aba7ae2c6949e30c6a4b6d1292cc9459f867b550e3b9b71b969fd74c0f54",
    "498a7f841b66e44756c04e93312d683c979c201d8eec7713823876e3cdc3fda7",
    "5994ba14fd56d323743e2ba297621bb1e3385dde20116b5ae33f8ad9c7ffbb36",
    "2ed31b529dee80b049e76f753b2c4c5b2f33cc32a9f5bdace7566da616ca2cac",
    "6f1591a836cfa2858da431140641f6ee01db6684b4fc6a1a81caea003c5cfbd4",
)


class ProtocolError(RuntimeError):
    """Fail-closed protocol error."""


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
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
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
    identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    visible_identity = (
        visible.st_dev,
        visible.st_ino,
        visible.st_size,
        visible.st_mtime_ns,
    )
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != identity
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or int(visible.st_nlink) != 1
        or visible_identity != identity
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
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "held_fd_identity_exact": True,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def publication_absence() -> dict[str, Any]:
    attempt_absent = not (ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink())
    result_absent = not (RESULT.exists() or RESULT.is_symlink())
    return {
        "attempt_marker": {
            "path": str(ATTEMPT_MARKER.relative_to(ROOT)),
            "absent": attempt_absent,
        },
        "result": {
            "path": str(RESULT.relative_to(ROOT)),
            "absent": result_absent,
        },
        "both_absent": attempt_absent and result_absent,
    }


def fsync_directory(path: Path) -> None:
    visible = os.lstat(path)
    if stat.S_ISLNK(visible.st_mode) or not stat.S_ISDIR(visible.st_mode):
        raise ProtocolError(f"publication directory is unsafe: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    if path not in {ATTEMPT_MARKER, RESULT}:
        raise ProtocolError(f"publication outside exact evidence allowlist: {path}")
    if path.parent.resolve() not in {ROOT.resolve(), (ROOT / "artifacts").resolve()}:
        raise ProtocolError(f"publication parent outside allowlist: {path.parent}")
    parent = os.lstat(path.parent)
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise ProtocolError(f"publication parent is unsafe: {path.parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    parsed = strict_json(payload, f"publication {path.name}")
    if canonical_json(parsed) != payload:
        raise ProtocolError(f"publication is not canonical JSON: {path}")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            count = os.write(fd, view[offset:])
            if count <= 0:
                raise ProtocolError(f"short O_EXCL publication: {path}")
            offset += count
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        before = os.fstat(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        reloaded = b"".join(chunks)
        after = os.fstat(fd)
        visible = os.lstat(path)
        identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        if (
            reloaded != payload
            or not stat.S_ISREG(after.st_mode)
            or stat.S_IMODE(after.st_mode) != 0o444
            or int(after.st_nlink) != 1
            or (
                int(before.st_dev),
                int(before.st_ino),
                int(before.st_size),
                int(before.st_mtime_ns),
            )
            != identity
            or stat.S_ISLNK(visible.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or (
                int(visible.st_dev),
                int(visible.st_ino),
                int(visible.st_size),
                int(visible.st_mtime_ns),
            )
            != identity
        ):
            raise ProtocolError(f"unsafe O_EXCL publication: {path}")
    finally:
        os.close(fd)
    fsync_directory(path.parent)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_bytes(reloaded),
        "bytes": len(reloaded),
        "mode_octal": "0444",
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "held_fd_exact_reload": True,
        "o_excl": True,
    }


def import_frozen(path: Path, digest: str, name: str) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(path, digest, name, expected_mode=0o555)
    if name in sys.modules:
        raise ProtocolError(f"frozen module name already occupied: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot construct frozen import: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, evidence


def validate_runtime(require_cuda: bool) -> dict[str, Any]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact_my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1
        and sys.dont_write_bytecode is True,
        "cublas_workspace_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        == ":4096:8",
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
            "current_device_exact_0": bool(torch.cuda.is_available() and torch.cuda.current_device() == 0),
            "device_name_exact": bool(properties and properties.name == "NVIDIA GeForce RTX 5090"),
            "compute_capability_exact_12_0": bool(
                properties and (int(properties.major), int(properties.minor)) == (12, 0)
            ),
            "torch_version_exact": torch.__version__ == "2.8.0+cu128",
            "cuda_runtime_exact": torch.version.cuda == "12.8",
        }
        if not all(cuda.values()):
            raise ProtocolError(f"CUDA BF16 runtime drift: {cuda}")
        result["cuda"] = cuda
    return result


def static_source_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"), filename=str(SCRIPT))
    forbidden_import_roots = {
        "requests",
        "urllib",
        "http",
        "socket",
        "subprocess",
        "kaggle",
    }
    forbidden_calls = {
        "backward",
        "step",
        "save",
        "savez",
        "submit",
        "upload",
        "package_submission",
        "unlink",
        "rename",
        "replace",
        "rmtree",
    }
    imports: list[str] = []
    calls: list[tuple[str, int]] = []
    call_nodes: list[ast.Call] = []
    identifiers: set[str] = set()
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

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            imports.extend(names)
        if isinstance(node, ast.Call):
            call_nodes.append(node)
            if isinstance(node.func, ast.Name):
                calls.append((node.func.id, node.lineno))
            elif isinstance(node.func, ast.Attribute):
                calls.append((node.func.attr, node.lineno))
    import_hits = sorted(
        name for name in imports if name.split(".", 1)[0] in forbidden_import_roots
    )
    call_hits = sorted(
        f"{name}@{line}" for name, line in calls if name in forbidden_calls
    )
    grad_sites = sum(name == "grad" for name, _ in calls)
    minimize_sites = sum(name == "minimize" for name, _ in calls)
    write_nodes = [
        node
        for node in call_nodes
        if isinstance(node.func, ast.Attribute) and node.func.attr == "write"
    ]
    publish_sites = [
        (
            node.args[0].id
            if node.args and isinstance(node.args[0], ast.Name)
            else "",
            node.lineno,
            owner(node),
        )
        for node in call_nodes
        if isinstance(node.func, ast.Name) and node.func.id == "publish_o_excl"
    ]
    candidate_sites = [
        node.lineno
        for node in call_nodes
        if isinstance(node.func, ast.Name)
        and node.func.id == "run_candidate"
        and owner(node) == "main"
    ]
    cuda_runtime_sites = [
        node.lineno
        for node in call_nodes
        if isinstance(node.func, ast.Name)
        and node.func.id == "validate_runtime"
        and owner(node) == "main"
        and any(
            keyword.arg == "require_cuda"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    ]
    marker_sites = [line for target, line, function in publish_sites if target == "ATTEMPT_MARKER" and function == "main"]
    result_sites = [line for target, line, function in publish_sites if target == "RESULT" and function == "main"]
    formal_order_exact = (
        len(marker_sites)
        == len(cuda_runtime_sites)
        == len(candidate_sites)
        == len(result_sites)
        == 1
        and marker_sites[0]
        < cuda_runtime_sites[0]
        < candidate_sites[0]
        < result_sites[0]
    )
    checks = {
        "ast_parse": True,
        "no_network_submission_or_subprocess_imports": not import_hits,
        "no_optimizer_backward_checkpoint_write_or_destructive_calls": not call_hits,
        "one_autograd_grad_call_site": grad_sites == 1,
        "one_qp_minimize_call_site": minimize_sites == 1,
        "seed_fixed": SEED == 202608143,
        "target_margin_fixed": TARGET_MARGIN == 1.0 / 512.0,
        "l2_cap_corrected_v2": ACTOR_L2_HARD_CAP == 1.0e-3,
        "optimizer_steps_zero_and_lr_absent": not any(
            name == "torch.optim" or name.startswith("torch.optim.") for name in imports
        ),
        "actor6_dimension_exact": ACTOR_DIMENSION == 65793,
        "single_candidate_contract": not {
            "RADII",
            "ALPHAS",
            "LEARNING_RATES",
        }.intersection(identifiers),
        "evidence_write_only_inside_o_excl_publisher": len(write_nodes) == 1
        and owner(write_nodes[0]) == "publish_o_excl",
        "exact_attempt_and_result_publish_sites": sorted(
            (target, function) for target, _, function in publish_sites
        )
        == [("ATTEMPT_MARKER", "main"), ("RESULT", "main")],
        "marker_before_cuda_candidate_before_result": formal_order_exact,
        "static_and_compact_summary_print_sites_exact2": sum(
            name == "print" for name, _ in calls
        )
        == 2,
        "fixed_one_shot_paths": ATTEMPT_MARKER.parent == ROOT
        and RESULT.parent == ROOT / "artifacts"
        and ATTEMPT_MARKER.name == f".ptcg-{BRANCH}-attempt.json"
        and RESULT.name == f"{BRANCH}.result.json",
    }
    if not all(checks.values()):
        raise ProtocolError(
            f"source static audit failed: {checks}; imports={import_hits}; calls={call_hits}"
        )
    return {
        "pass": True,
        "checks": checks,
        "forbidden_import_hits": import_hits,
        "forbidden_call_hits": call_hits,
        "autograd_grad_call_sites": grad_sites,
        "qp_minimize_call_sites": minimize_sites,
        "publication_call_sites": [
            {"target": target, "line": line, "owner": function}
            for target, line, function in publish_sites
        ],
        "formal_control_flow_order": {
            "attempt_marker_line": marker_sites[0] if len(marker_sites) == 1 else None,
            "cuda_runtime_line": (
                cuda_runtime_sites[0] if len(cuda_runtime_sites) == 1 else None
            ),
            "candidate_line": candidate_sites[0] if len(candidate_sites) == 1 else None,
            "result_line": result_sites[0] if len(result_sites) == 1 else None,
        },
    }


def audit_control_plane() -> dict[str, Any]:
    inputs = {}
    parsed = {}
    for key, path, digest in (
        ("master", MASTER, MASTER_SHA256),
        ("correction_v2", CORRECTION_V2, CORRECTION_V2_SHA256),
        ("correction_v3", CORRECTION_V3, CORRECTION_V3_SHA256),
        ("correction_v4", CORRECTION_V4, CORRECTION_V4_SHA256),
        ("authorization", AUTHORIZATION, AUTHORIZATION_SHA256),
    ):
        payload, evidence = read_regular_bytes(
            path, digest, key, expected_mode=0o444
        )
        inputs[key] = evidence
        parsed[key] = strict_json(payload, key)
    master = parsed["master"]
    v2 = parsed["correction_v2"]
    v3 = parsed["correction_v3"]
    v4 = parsed["correction_v4"]
    authorization = parsed["authorization"]
    checks = {
        "master_schema": master.get("schema_version")
        == "ptcg-u456-g8-guarded-u468-currentparent-boundaryqp-master-preregistration-v1",
        "master_locked_before_candidate": master.get("status")
        == "locked_before_parent_profile_or_changed_candidate_construction",
        "master_parent_file": master.get("parent", {}).get("file_sha256")
        == PARENT_FILE_SHA256,
        "master_parent_model": master.get("parent", {}).get("runtime_model_state_sha256")
        == PARENT_MODEL_SHA256,
        "master_seed": master.get("sole_changed_candidate", {}).get("seed") == SEED,
        "master_actor6_scope": tuple(master.get("sole_changed_candidate", {}).get("scope", []))
        == ACTOR6_NAMES,
        "master_single_candidate": master.get("sole_changed_candidate", {}).get(
            "alpha_lr_radius_seed_or_endpoint_sweep"
        )
        is False,
        "master_optimizer_zero": master.get("sole_changed_candidate", {}).get(
            "optimizer_steps"
        )
        == 0,
        "master_learning_rate_null": master.get("sole_changed_candidate", {}).get(
            "learning_rate"
        )
        is None,
        "v2_schema": v2.get("schema_version")
        == "ptcg-u456-g8-guarded-u468-currentparent-boundaryqp-master-correction-v2",
        "v2_binds_master": v2.get("superseded_master", {}).get("sha256")
        == MASTER_SHA256,
        "v2_cap_exact": v2.get("corrected_value") == ACTOR_L2_HARD_CAP,
        "v2_no_retry": "without cap relaxation" in str(v2.get("fail_closed_contract", "")),
        "v3_schema": v3.get("schema_version")
        == "ptcg-u456-g8-guarded-u468-currentparent-boundaryqp-master-correction-v3",
        "v3_binds_master": v3.get("master", {}).get("sha256") == MASTER_SHA256,
        "v3_binds_v2": v3.get("prior_correction_v2", {}).get("sha256")
        == CORRECTION_V2_SHA256,
        "v3_net_semantics": "net at least zero" in str(v3.get("authoritative_gate", "")),
        "v4_schema": v4.get("schema_version")
        == "ptcg-u456-g8-guarded-u468-currentparent-boundaryqp-master-correction-v4",
        "v4_locked_before_candidate": v4.get("status")
        == "locked_after_parent_profile_before_any_changed_candidate_construction",
        "v4_binds_master": v4.get("master", {}).get("sha256") == MASTER_SHA256,
        "v4_binds_v2": v4.get("prior_corrections", {}).get("v2")
        == CORRECTION_V2_SHA256,
        "v4_binds_v3": v4.get("prior_corrections", {}).get("v3")
        == CORRECTION_V3_SHA256,
        "v4_binds_source_profile_v2": v4.get("parent_profile", {}).get("sha256")
        == SOURCE_PROFILE_V2_SHA256,
        "v4_source_profile_zero_candidates": v4.get("parent_profile", {}).get(
            "changed_candidates"
        )
        == 0,
        "v4_count_safe_semantics": "count_margin is finite and strictly positive"
        in str(v4.get("corrected_count_safe_singleton", "")),
        "v4_selection_only": v4.get("scope", {}).get("profile_rerun_authorized")
        is False
        and v4.get("scope", {}).get("selection_only_correction_authorized") is True,
        "v4_selection_only_no_candidate": v4.get("candidate_accounting", {}).get(
            "changed_candidates_constructed_before_v4"
        )
        == 0,
        "authorization_status": authorization.get("status") == "GO_P12",
        "authorization_block3": authorization.get("block") == 3,
        "authorization_parent_file": authorization.get("p12_authorization", {}).get(
            "parent_checkpoint_sha256"
        )
        == PARENT_FILE_SHA256,
        "authorization_parent_model": authorization.get("terminal_identity", {}).get(
            "runtime_model_state_sha256"
        )
        == PARENT_MODEL_SHA256,
        "authorization_update468": authorization.get("terminal_identity", {}).get("update")
        == 468,
        "authorization_finite": authorization.get("terminal_identity", {}).get(
            "all_model_and_optimizer_values_finite"
        )
        is True,
    }
    if not all(checks.values()):
        raise ProtocolError(f"control-plane contract drift: {checks}")
    return {"pass": True, "checks": checks, "inputs": inputs}


def audit_dependencies() -> dict[str, Any]:
    result = {}
    for key, path, digest in (
        ("formal_v3", FORMAL, FORMAL_SHA256),
        ("profile_corrector_v3", PROFILE_CORRECTOR_V3, PROFILE_CORRECTOR_V3_SHA256),
        (
            "profile_selection_engine",
            PROFILE_SELECTION_ENGINE,
            PROFILE_SELECTION_ENGINE_SHA256,
        ),
        ("actor_codec_cw20", CW20, CW20_SHA256),
        ("train_math_cw22_reference", CW22, CW22_SHA256),
        ("boundary_qp_cw23_reference", CW23, CW23_SHA256),
        ("current_fulltrain_reference", CURRENT_FULLTRAIN, CURRENT_FULLTRAIN_SHA256),
        ("cw23_payload_reference", CW23_PAYLOAD_GATE, CW23_PAYLOAD_GATE_SHA256),
    ):
        _, evidence = read_regular_bytes(path, digest, key, expected_mode=0o555)
        result[key] = evidence
    return result


def normalize_row(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    required = {
        "source",
        "stratum",
        "category",
        "member",
        "line_index",
        "line_sha256",
        "episode_id",
        "team_name",
        "context",
        "min_count",
        "max_count",
        "expert_order",
        "predicted_order",
        "set_correct",
        "ordered_correct",
        "selection_margin",
        "decision_margin",
    }
    if not isinstance(row, Mapping) or not required.issubset(row):
        raise ProtocolError(f"{label}: missing row fields {sorted(required - set(row))}")
    result = dict(row)
    if (
        not isinstance(result["source"], str)
        or result["source"] not in DATASETS
        or not isinstance(result["stratum"], str)
        or not isinstance(result["category"], str)
        or not isinstance(result["member"], str)
        or not result["member"].startswith("train/")
        or not result["member"].endswith(".jsonl")
        or ".." in result["member"].split("/")
        or type(result["line_index"]) is not int
        or int(result["line_index"]) < 0
        or not is_sha256(result["line_sha256"])
        or not isinstance(result["episode_id"], (str, int))
        or isinstance(result["episode_id"], bool)
        or not isinstance(result["team_name"], str)
        or not isinstance(result["expert_order"], list)
        or not isinstance(result["predicted_order"], list)
        or any(type(value) is not int or value < 0 for value in result["expert_order"])
        or any(type(value) is not int or value < 0 for value in result["predicted_order"])
        or len(set(result["expert_order"])) != len(result["expert_order"])
        or len(set(result["predicted_order"])) != len(result["predicted_order"])
        or type(result["context"]) is not int
        or type(result["min_count"]) is not int
        or type(result["max_count"]) is not int
        or int(result["min_count"]) < 0
        or int(result["max_count"]) < int(result["min_count"])
        or type(result["set_correct"]) is not bool
        or type(result["ordered_correct"]) is not bool
        or not isinstance(result["selection_margin"], (int, float))
        or isinstance(result["selection_margin"], bool)
        or not math.isfinite(float(result["selection_margin"]))
        or not isinstance(result["decision_margin"], (int, float))
        or isinstance(result["decision_margin"], bool)
        or not math.isfinite(float(result["decision_margin"]))
    ):
        raise ProtocolError(f"{label}: malformed row identity or scalar")
    return result


def count_safe_row(row: Mapping[str, Any], *, singleton: bool) -> bool:
    expert = row.get("expert_order")
    predicted = row.get("predicted_order")
    if (
        not isinstance(expert, list)
        or not isinstance(predicted, list)
        or not expert
        or len(expert) != len(predicted)
        or (singleton and len(expert) != 1)
    ):
        return False
    count = len(expert)
    minimum = row.get("min_count")
    maximum = row.get("max_count")
    if type(minimum) is not int or type(maximum) is not int:
        return False
    if not int(minimum) <= count <= int(maximum):
        return False
    if int(minimum) == int(maximum):
        return True
    margin = row.get("count_margin")
    return (
        isinstance(margin, (int, float))
        and not isinstance(margin, bool)
        and math.isfinite(float(margin))
        and float(margin) > 0.0
    )


def role_row_from_b256(
    raw: Any,
    role: str,
    row_by_sha: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    required = {
        "source",
        "member",
        "line_index",
        "line_sha256",
        "episode_id",
        "team_name",
        "context",
        "min_count",
        "max_count",
        "expert_order",
        "predicted_order",
        "set_correct",
        "ordered_correct",
        "selection_margin",
        "decision_margin",
        "role",
    }
    if not isinstance(raw, Mapping) or not required.issubset(raw):
        missing = sorted(required.difference(raw if isinstance(raw, Mapping) else {}))
        raise ProtocolError(f"{label}: missing role-row fields {missing}")
    line_sha = raw.get("line_sha256")
    if not is_sha256(line_sha) or str(line_sha) not in row_by_sha:
        raise ProtocolError(f"{label}: role row is absent from B256")
    canonical = dict(row_by_sha[str(line_sha)])
    if raw.get("role") != role or canonical.get("role") != role:
        raise ProtocolError(f"{label}: role drift")
    shared_keys = set(raw).intersection(canonical)
    mismatched = sorted(key for key in shared_keys if raw[key] != canonical[key])
    if mismatched:
        raise ProtocolError(f"{label}: role/B256 field drift {mismatched}")
    return canonical


def profile_base_checks(profile: Mapping[str, Any]) -> dict[str, bool]:
    base = profile.get("base")
    if not isinstance(base, Mapping):
        return {"base_mapping": False}
    return {
        "base_mapping": True,
        "checkpoint_file_sha256": base.get("checkpoint_file_sha256")
        == PARENT_FILE_SHA256,
        "model_state_sha256": base.get("model_state_sha256") == PARENT_MODEL_SHA256,
        "live_model_final_state_sha256": base.get("live_model_final_state_sha256")
        == PARENT_MODEL_SHA256,
        "checkpoint_update468": base.get("checkpoint_update") == 468,
        "profile_constructed_no_candidate": base.get(
            "changed_candidate_constructed_or_evaluated"
        )
        is False,
    }


def validate_profile_selection(
    profile: Mapping[str, Any], profile_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selection = profile.get("selection")
    if not isinstance(selection, Mapping):
        raise ProtocolError("profile selection is missing")
    raw_rows = selection.get("b256_rows")
    raw_targets = selection.get("targets")
    raw_guards = selection.get("guards")
    contracts = selection.get("contracts")
    correction = profile.get("selection_correction_v3")
    scope_audit = profile.get("scope_audit")
    if (
        not isinstance(raw_rows, list)
        or not isinstance(raw_targets, list)
        or not isinstance(raw_guards, list)
        or not isinstance(contracts, Mapping)
        or not isinstance(correction, Mapping)
        or not isinstance(scope_audit, Mapping)
    ):
        raise ProtocolError("profile selection arrays/contracts are malformed")
    rows = [normalize_row(row, f"b256[{index}]") for index, row in enumerate(raw_rows)]
    row_by_sha = {str(row["line_sha256"]): row for row in rows}
    targets = [
        role_row_from_b256(row, "target", row_by_sha, f"target[{index}]")
        for index, row in enumerate(raw_targets)
    ]
    guards = [
        role_row_from_b256(row, "guard", row_by_sha, f"guard[{index}]")
        for index, row in enumerate(raw_guards)
    ]
    target_sha = [str(row["line_sha256"]) for row in targets]
    guard_sha = [str(row["line_sha256"]) for row in guards]
    strata = Counter(str(row["stratum"]) for row in rows)
    target_contexts = Counter(int(row["context"]) for row in targets)
    guard_sources = Counter(str(row["source"]) for row in guards)
    hard_strata = {name for name in B256_QUOTAS if name.endswith("_hard")}
    hard_count = sum(str(row["stratum"]) in hard_strata for row in rows)
    context34 = sum(int(row["context"]) == 34 for row in rows)
    slot_values = [
        row.get("final_b256_slot_zero_based") for row in rows
    ]
    checks = {
        "schema_exact": profile.get("schema_version") == PROFILE_SCHEMA,
        "status_exact": profile.get("status") == PROFILE_STATUS,
        "selection_status_exact": selection.get("status")
        == "completed_count_safe_singleton_selection_correction_v3",
        "selection_target_margin_exact": selection.get("target_margin") == TARGET_MARGIN,
        "profile_sha_exact": profile_sha256 == EXPECTED_PROFILE_SHA256,
        "correction_binds_source_profile_v2": correction.get("source_profile", {}).get(
            "sha256"
        )
        == SOURCE_PROFILE_V2_SHA256,
        "correction_binds_master_v4": correction.get("master_correction_v4", {}).get(
            "sha256"
        )
        == CORRECTION_V4_SHA256,
        "correction_tool_exact": correction.get("tool", {}).get("sha256")
        == PROFILE_CORRECTOR_V3_SHA256,
        "correction_selection_engine_exact": correction.get(
            "frozen_selection_engine", {}
        ).get("sha256")
        == PROFILE_SELECTION_ENGINE_SHA256,
        "correction_zero_profile_rows_recomputed": correction.get(
            "profile_rows_recomputed"
        )
        == 0,
        "correction_zero_archives_opened": correction.get("archive_members_opened") == 0,
        "correction_zero_cuda": correction.get("cuda_accessed") is False,
        "correction_zero_candidates": correction.get(
            "changed_candidates_constructed_or_evaluated"
        )
        == 0,
        "scope_profile_and_correction_zero_candidates": scope_audit.get(
            "changed_candidates_constructed_or_evaluated"
        )
        == 0
        and scope_audit.get("selection_correction_changed_candidates") == 0,
        **profile_base_checks(profile),
        "rows_exact256": len(rows) == BATCH_SIZE,
        "targets_exact5": len(targets) == TARGET_COUNT,
        "guards_exact14": len(guards) == GUARD_COUNT,
        "row_sha_unique": len(row_by_sha) == BATCH_SIZE,
        "target_sha_unique": len(set(target_sha)) == TARGET_COUNT,
        "corrected_target_identity_and_order_exact": tuple(target_sha)
        == TARGET_LINE_SHA256,
        "guard_sha_unique": len(set(guard_sha)) == GUARD_COUNT,
        "target_guard_disjoint": not set(target_sha).intersection(guard_sha),
        "targets_subset_b256": set(target_sha).issubset(row_by_sha),
        "guards_subset_b256": set(guard_sha).issubset(row_by_sha),
        "strata_quotas_exact": dict(strata) == B256_QUOTAS,
        "hard96_retention160": hard_count == 96 and len(rows) - hard_count == 160,
        "context34_exact16": context34 == CONTEXT34_ROWS,
        "slots_exact": slot_values == list(range(BATCH_SIZE)),
        "slots_all_plain_int": all(type(value) is int for value in slot_values),
        "one_episode_per_stratum": all(
            len({str(row["episode_id"]) for row in rows if str(row["stratum"]) == name})
            == count
            for name, count in B256_QUOTAS.items()
        ),
        "target_context_3plus2": dict(target_contexts) == {0: 3, 7: 2},
        "targets_pf_singleton_parent_wrong": all(
            str(row["source"]) == "pokemonfan"
            and count_safe_row(row, singleton=True)
            and row["predicted_order"] != row["expert_order"]
            and row["set_correct"] is False
            and row["ordered_correct"] is False
            and float(row["decision_margin"]) < 0.0
            and float(row["selection_margin"]) == float(row["decision_margin"])
            and float(row.get("endpoint_margin_minimum", float("nan")))
            == TARGET_MARGIN
            and float(row.get("required_linear_margin_change", float("nan")))
            == TARGET_MARGIN - float(row["decision_margin"])
            for row in targets
        ),
        "corrected_context7_margins_exact": [
            float(row["decision_margin"]) for row in targets if int(row["context"]) == 7
        ]
        == [-1.0 / 512.0, -2.0 / 512.0],
        "corrected_context7_count_margins_exact": [
            float(row["count_margin"]) for row in targets if int(row["context"]) == 7
        ]
        == [8.4609375, 11.015625],
        "all_b256_rows_parent_count_safe": all(
            row.get("count_correct") is True and count_safe_row(row, singleton=False)
            for row in rows
        ),
        "guard_source_6_4_4": dict(guard_sources)
        == {"flg": 6, "pokemonfan": 4, "core5": 4},
        "guards_parent_ordered_correct": all(
            row["set_correct"] is True
            and row["ordered_correct"] is True
            and row["predicted_order"] == row["expert_order"]
            and float(row["decision_margin"]) >= 0.0
            and float(row["selection_margin"]) == float(row["decision_margin"])
            for row in guards
        ),
        "forced_flg_guards_present": set(FORCED_FLG_GUARDS).issubset(guard_sha),
        "forced_flg_guard_indices_exact": all(
            str(row["line_sha256"]) not in FORCED_FLG_GUARDS
            or (
                str(row["source"]) == "flg"
                and str(row["member"]) == "train/part-00000.jsonl"
                and int(row["line_index"]) == FORCED_FLG_GUARDS[str(row["line_sha256"])]
            )
            for row in guards
        ),
        "all_contract_booleans_true": bool(contracts)
        and all(type(value) is bool and value is True for value in contracts.values()),
        "no_nontrain_member": all(str(row["member"]).startswith("train/") for row in rows),
    }
    if not all(checks.values()):
        raise ProtocolError(f"profile/selection contract drift: {checks}")
    computed_b256_sha = sha256_bytes(canonical_json(rows))
    if (
        selection.get("b256_rows_canonical_sha256") != computed_b256_sha
        or computed_b256_sha != EXPECTED_B256_SELECTION_SHA256
    ):
        raise ProtocolError("profile selection B256 canonical SHA drift")
    return rows, targets, guards, {
        "pass": True,
        "checks": checks,
        "profile_sha256": profile_sha256,
        "b256_selection_sha256": computed_b256_sha,
        "target_line_sha256": target_sha,
        "guard_line_sha256": guard_sha,
        "strata": dict(strata),
    }


def tensor_bytes_sha(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    return sha256_bytes(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def tensor_batch_manifest(batch: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    import torch

    manifest = []
    digest = hashlib.sha256()
    for name, tensor in sorted(batch.items()):
        value = tensor.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        record = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        }
        manifest.append(record)
        digest.update(canonical_json(record))
    return digest.hexdigest(), manifest


def load_b256(
    rows: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
    helper: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wanted: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {
        source: {} for source in DATASETS
    }
    for row in rows:
        source = str(row["source"])
        key = (str(row["member"]), int(row["line_index"]))
        if source not in wanted or key in wanted[source]:
            raise ProtocolError("B256 source/key collision")
        wanted[source][key] = row
    inputs = {}
    archive_payloads = {}
    for source, path in DATASETS.items():
        payload, evidence = read_regular_bytes(
            path,
            DATA_SHA256[source],
            f"{source} train archive",
            expected_mode=DATA_MODE[source],
        )
        inputs[source] = evidence
        archive_payloads[source] = payload

    bc = helper.bc
    ppo = helper.ppo
    features: dict[str, dict[str, Any]] = {}
    opened: dict[str, list[str]] = {}
    for source, payload in archive_payloads.items():
        by_member: dict[str, set[int]] = defaultdict(set)
        for member, line_index in wanted[source]:
            by_member[member].add(line_index)
        opened[source] = []
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ProtocolError(f"{source}: duplicate ZIP member names")
            for member in sorted(by_member):
                if (
                    not member.startswith("train/")
                    or not member.endswith(".jsonl")
                    or ".." in member.split("/")
                ):
                    raise ProtocolError("attempted non-train/noncanonical ZIP member")
                archive.getinfo(member)
                opened[source].append(member)
                remaining = set(by_member[member])
                with archive.open(member) as handle:
                    for line_index, raw in enumerate(handle):
                        if line_index not in remaining:
                            continue
                        identity = wanted[source][(member, line_index)]
                        if sha256_bytes(raw) != str(identity["line_sha256"]):
                            raise ProtocolError("selected train row SHA drift")
                        source_row = strict_json(raw, f"{source}/{member}:{line_index}")
                        if str(source_row.get("split", "")) != "train":
                            raise ProtocolError("selected source row is not train")
                        if (
                            str(source_row.get("episode_id", ""))
                            != str(identity["episode_id"])
                            or str(source_row.get("team_name", ""))
                            != str(identity["team_name"])
                        ):
                            raise ProtocolError("selected source row metadata drift")
                        previous_max = bc.MAX_ACTION_COUNT
                        bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
                        try:
                            feature = bc.featurize_row(
                                source_row,
                                int(model_config["hash_size"]),
                                int(model_config["max_state_entities"]),
                            )
                        finally:
                            bc.MAX_ACTION_COUNT = previous_max
                        if feature is None:
                            raise ProtocolError("selected train row no longer featurizes")
                        expert = [int(value) for value in source_row.get("action", [])]
                        row_checks = {
                            "expert": expert == [int(value) for value in identity["expert_order"]],
                            "context": int(feature["context"]) == int(identity["context"]),
                            "min_count": int(feature["min_count"]) == int(identity["min_count"]),
                            "max_count": int(feature["max_count"]) == int(identity["max_count"]),
                            "nonempty": bool(expert),
                        }
                        if not all(row_checks.values()):
                            raise ProtocolError(f"selected row decision drift: {row_checks}")
                        feature["action_sequence"] = expert
                        feature["sample_weight"] = (
                            CONTEXT34_SAMPLE_WEIGHT
                            if int(feature["context"]) == ppo.SKILL_ORDER_CONTEXT
                            else 1.0
                        )
                        features[str(identity["line_sha256"])] = feature
                        remaining.remove(line_index)
                        if not remaining:
                            break
                if remaining:
                    raise ProtocolError(f"selected rows missing: {source}/{member}")
    if len(features) != BATCH_SIZE:
        raise ProtocolError("loaded feature cardinality drift")
    ordered_features = [features[str(row["line_sha256"])] for row in rows]
    batch = bc.collate_decisions(
        ordered_features,
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    context34_mask = batch["contexts"] == ppo.SKILL_ORDER_CONTEXT
    cache_checks = {
        "rows_exact256": int(batch["action_counts"].shape[0]) == BATCH_SIZE,
        "context34_exact16": int(context34_mask.sum()) == CONTEXT34_ROWS,
        "context34_weight_one_third": bool(
            helper.torch.allclose(
                batch["sample_weights"][context34_mask],
                helper.torch.full_like(
                    batch["sample_weights"][context34_mask], CONTEXT34_SAMPLE_WEIGHT
                ),
                rtol=0.0,
                atol=1.0e-7,
            )
        ),
        "other_weight_one": bool(
            helper.torch.equal(
                batch["sample_weights"][~context34_mask],
                helper.torch.ones_like(batch["sample_weights"][~context34_mask]),
            )
        ),
        "all_action_counts_positive": bool((batch["action_counts"] > 0).all()),
        "only_train_members_opened": all(
            members and all(member.startswith("train/") for member in members)
            for members in opened.values()
        ),
    }
    if not all(cache_checks.values()):
        raise ProtocolError(f"B256 cache contract failed: {cache_checks}")
    cache_sha, manifest = tensor_batch_manifest(batch)
    return batch, {
        "pass": True,
        "checks": cache_checks,
        "cache_sha256": cache_sha,
        "tensor_manifest": manifest,
        "opened_train_members": opened,
        "non_train_member_payloads_opened": 0,
        "archives": inputs,
    }


def ordered_nll_per_row(outputs: Mapping[str, Any], batch: Mapping[str, Any], torch: Any) -> Any:
    logits = outputs["policy_logits"].float()
    option_mask = batch["option_mask"].bool()
    action_counts = batch["action_counts"]
    sequences = batch["action_sequences"]
    selected = torch.zeros_like(option_mask)
    result = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    for stage in range(sequences.shape[1]):
        active = stage < action_counts
        if not bool(active.any()):
            break
        chosen = sequences[:, stage]
        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        active_chosen = chosen[active]
        if not bool(
            (option_mask[active_rows, active_chosen] & ~selected[active_rows, active_chosen]).all()
        ):
            raise ProtocolError("illegal or duplicate expert action in B256")
        allowed = option_mask & ~selected
        log_probs = torch.log_softmax(logits.masked_fill(~allowed, -1.0e9), dim=1)
        safe_chosen = chosen.clamp(0, logits.shape[1] - 1)
        selected_log_prob = log_probs.gather(1, safe_chosen.unsqueeze(1)).squeeze(1)
        result -= torch.where(active, selected_log_prob, torch.zeros_like(result))
        selected.scatter_(1, safe_chosen.unsqueeze(1), active.unsqueeze(1))
    return result


def threat_margin_tensor(
    logits: Any,
    option_mask: Any,
    expert_order: Sequence[int],
    *,
    singleton_only: bool,
) -> tuple[Any, dict[str, Any]]:
    if singleton_only and len(expert_order) != 1:
        raise ProtocolError("target threat margin is not singleton")
    remaining = option_mask.bool().clone()
    candidates: list[tuple[float, Any, int, int, int]] = []
    for stage, chosen_value in enumerate(expert_order):
        chosen = int(chosen_value)
        if chosen < 0 or chosen >= int(remaining.numel()) or not bool(remaining[chosen]):
            raise ProtocolError("expert action is outside remaining option mask")
        competitors = remaining.nonzero(as_tuple=False).squeeze(1)
        competitors = competitors[competitors != chosen]
        if int(competitors.numel()) > 0:
            competitor_logits = logits[competitors]
            local = int(competitor_logits.argmax().detach().cpu())
            negative = int(competitors[local].detach().cpu())
            margin = logits[chosen] - logits[negative]
            candidates.append((float(margin.detach().cpu()), margin, stage, chosen, negative))
        remaining[chosen] = False
    if not candidates:
        raise ProtocolError("selected decision has no threat pair")
    observed, margin, stage, positive, negative = min(
        candidates, key=lambda item: (item[0], item[2], item[3], item[4])
    )
    return margin, {
        "stage": stage,
        "positive_option": positive,
        "negative_option": negative,
        "margin": observed,
    }


def flat_gradient(values: Sequence[Any], np: Any, torch: Any) -> Any:
    flat = np.concatenate(
        [
            value.detach().cpu().to(dtype=torch.float64).contiguous().numpy().reshape(-1)
            for value in values
        ]
    )
    if flat.shape != (ACTOR_DIMENSION,) or not bool(np.isfinite(flat).all()):
        raise ProtocolError("actor6 gradient shape/finite drift")
    return flat


def array_sha(value: Any) -> str:
    import numpy as np

    return sha256_bytes(np.ascontiguousarray(value, dtype="<f8").tobytes())


def weighted_loss(per_row: Any, indices: Sequence[int], weights: Any, torch: Any) -> Any:
    mask = torch.tensor(list(indices), dtype=torch.long, device=per_row.device)
    selected_weights = weights[mask].float()
    return (per_row[mask] * selected_weights).sum() / selected_weights.sum().clamp_min(1.0)


def solve_minimum_norm_qp(matrix: Any, rhs: Any, np: Any, optimize: Any) -> tuple[Any, dict[str, Any]]:
    if matrix.ndim != 2 or matrix.shape[1] != ACTOR_DIMENSION or rhs.shape != (matrix.shape[0],):
        raise ProtocolError("boundary-QP matrix/vector shape drift")
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or not bool(np.isfinite(singular_values).all()):
        raise ProtocolError("boundary-QP singular values invalid")
    rank = int(
        (singular_values > singular_values[0] * SVD_RELATIVE_RANK_TOLERANCE).sum()
    )
    if rank <= 0:
        raise ProtocolError("boundary-QP gradient matrix has zero rank")
    reduced = u[:, :rank] * singular_values[:rank]

    solution = optimize.minimize(
        lambda value: 0.5 * float(value @ value),
        np.zeros(rank, dtype=np.float64),
        jac=lambda value: value,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda value: reduced @ value - rhs,
                "jac": lambda value: reduced,
            }
        ],
        method="SLSQP",
        options={"ftol": QP_FTOL, "maxiter": QP_MAXITER, "disp": False},
    )
    if not bool(solution.success):
        raise ProtocolError(
            f"boundary-QP SLSQP failed: {solution.status} {solution.message}"
        )
    nominal = vh[:rank].T @ solution.x
    residual = matrix @ nominal - rhs
    nominal_l2 = float(np.linalg.norm(nominal))
    if (
        not bool(np.isfinite(nominal).all())
        or not math.isfinite(nominal_l2)
        or nominal_l2 <= 0.0
        or float(residual.min()) < -QP_RESIDUAL_TOLERANCE
    ):
        raise ProtocolError("boundary-QP nominal solution audit failed")
    if nominal_l2 > ACTOR_L2_HARD_CAP:
        planned = nominal * (ACTOR_L2_HARD_CAP / nominal_l2)
        capped = True
    else:
        planned = nominal
        capped = False
    effective_condition = float(singular_values[0] / singular_values[rank - 1])
    effective_gram_condition = effective_condition * effective_condition
    if not math.isfinite(effective_gram_condition):
        raise ProtocolError("boundary-QP effective Gram condition is nonfinite")
    audit = {
        "objective": "minimum full-actor6 L2 under target guard and NLL inequalities",
        "matrix_shape": [int(value) for value in matrix.shape],
        "matrix_float64_le_sha256": array_sha(matrix),
        "rhs": [float(value) for value in rhs],
        "rhs_float64_le_sha256": array_sha(rhs),
        "svd_rank": rank,
        "singular_values": [float(value) for value in singular_values],
        "effective_gram_condition_number": effective_gram_condition,
        "full_row_gram_rank": rank,
        "full_row_gram_dimension": int(matrix.shape[0]),
        "solver": {
            "implementation": "scipy.optimize.minimize/SLSQP",
            "status": int(solution.status),
            "message": str(solution.message),
            "iterations": int(solution.nit),
            "ftol": QP_FTOL,
            "maxiter": QP_MAXITER,
            "success": True,
        },
        "nominal_l2": nominal_l2,
        "nominal_float64_le_sha256": array_sha(nominal),
        "nominal_residual_min": float(residual.min()),
        "radially_capped": capped,
        "hard_l2_cap": ACTOR_L2_HARD_CAP,
        "planned_l2": float(np.linalg.norm(planned)),
        "planned_float64_le_sha256": array_sha(planned),
        "predicted_planned_residual_min": float((matrix @ planned - rhs).min()),
        "single_candidate_only": True,
        "optimizer_instances": 0,
        "optimizer_steps": 0,
        "learning_rate": None,
    }
    return planned, audit


def policy_row_metrics(predicted: Sequence[int], expert: Sequence[int], context: int) -> dict[str, bool]:
    set_exact = set(predicted) == set(expert) and len(predicted) == len(expert)
    ordered = list(predicted) == list(expert)
    top1 = bool(predicted and expert and int(predicted[0]) == int(expert[0]))
    hybrid = ordered if context == 34 else set_exact
    return {
        "set_exact": set_exact,
        "hybrid_order_exact": hybrid,
        "ordered_exact": ordered,
        "top1_correct": top1,
        "context34_hybrid_order_exact": context != 34 or hybrid,
        "context34_ordered_exact": context != 34 or ordered,
        "count_correct": len(predicted) == len(expert),
    }


def evaluate_outputs(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    helper: ModuleType,
) -> dict[str, Any]:
    torch = helper.torch
    predictions, _, _, _ = helper.ppo.sample_ordered_actions(
        outputs,
        batch,
        deterministic=True,
        canonicalize_order=False,
    )
    per_row_nll = ordered_nll_per_row(outputs, batch, torch)
    records = []
    for index, row in enumerate(rows):
        count = int(batch["action_counts"][index].detach().cpu())
        expert = [
            int(value)
            for value in batch["action_sequences"][index, :count].detach().cpu().tolist()
        ]
        predicted = [int(value) for value in predictions[index]]
        records.append(
            {
                "line_sha256": str(row["line_sha256"]),
                "source": str(row["source"]),
                "stratum": str(row["stratum"]),
                "category": str(row["category"]),
                "context": int(row["context"]),
                "expert_order": expert,
                "predicted_order": predicted,
                "ordered_nll": float(per_row_nll[index].detach().cpu()),
                "sample_weight": float(batch["sample_weights"][index].detach().cpu()),
                "metrics": policy_row_metrics(predicted, expert, int(row["context"])),
            }
        )

    def aggregate(indices: Sequence[int]) -> dict[str, Any]:
        if not indices:
            raise ProtocolError("empty B256 aggregation")
        weight_sum = sum(records[index]["sample_weight"] for index in indices)
        nll = sum(
            records[index]["ordered_nll"] * records[index]["sample_weight"]
            for index in indices
        ) / weight_sum
        return {
            "rows": len(indices),
            "sample_weight_sum": weight_sum,
            "ordered_pl_nll": nll,
            "ordered_correct": sum(records[index]["metrics"]["ordered_exact"] for index in indices),
            "set_correct": sum(records[index]["metrics"]["set_exact"] for index in indices),
            "count_correct": sum(records[index]["metrics"]["count_correct"] for index in indices),
        }

    hard_indices = [index for index, row in enumerate(rows) if str(row["stratum"]).endswith("_hard")]
    retention_indices = [index for index in range(len(rows)) if index not in set(hard_indices)]
    by_stratum = {
        name: aggregate([index for index, row in enumerate(rows) if str(row["stratum"]) == name])
        for name in B256_QUOTAS
    }
    return {
        "rows": records,
        "hard": aggregate(hard_indices),
        "retention": aggregate(retention_indices),
        "by_stratum": by_stratum,
        "native_output": {
            "policy_logits_dtype": str(outputs["policy_logits"].dtype),
            "policy_logits_sha256": tensor_bytes_sha(outputs["policy_logits"]),
            "count_logits_dtype": str(outputs["count_logits"].dtype),
            "count_logits_sha256": tensor_bytes_sha(outputs["count_logits"]),
            "value_logits_dtype": str(outputs["value_logits"].dtype),
            "value_logits_sha256": tensor_bytes_sha(outputs["value_logits"]),
        },
    }


def compare_b256(
    parent_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    target_sha: set[str],
    guard_sha: set[str],
    target_margins: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, bool]]:
    parent_rows = {str(row["line_sha256"]): row for row in parent_report["rows"]}
    candidate_rows = {str(row["line_sha256"]): row for row in candidate_report["rows"]}
    if set(parent_rows) != set(candidate_rows) or len(parent_rows) != BATCH_SIZE:
        raise ProtocolError("B256 comparison identity drift")
    metric_names = tuple(next(iter(parent_rows.values()))["metrics"])
    other_sha = set(parent_rows).difference(target_sha, guard_sha)
    regressions = []
    for line_sha in sorted(other_sha):
        for metric in metric_names:
            if parent_rows[line_sha]["metrics"][metric] and not candidate_rows[line_sha]["metrics"][metric]:
                regressions.append({"line_sha256": line_sha, "metric": metric})
    guard_failures = []
    for line_sha in sorted(guard_sha):
        for metric in metric_names:
            if parent_rows[line_sha]["metrics"][metric] and not candidate_rows[line_sha]["metrics"][metric]:
                guard_failures.append({"line_sha256": line_sha, "metric": metric})
    target_coverage = {
        metric: sum(candidate_rows[line_sha]["metrics"][metric] for line_sha in target_sha)
        for metric in ("set_exact", "hybrid_order_exact", "ordered_exact", "top1_correct")
    }
    target_all_exact = all(
        candidate_rows[line_sha]["metrics"]["ordered_exact"] for line_sha in target_sha
    )
    target_margin_pass = all(
        float(target_margins[line_sha]) >= TARGET_MARGIN for line_sha in target_sha
    )
    hard_improvement = float(parent_report["hard"]["ordered_pl_nll"]) - float(
        candidate_report["hard"]["ordered_pl_nll"]
    )
    retention_improvement = float(parent_report["retention"]["ordered_pl_nll"]) - float(
        candidate_report["retention"]["ordered_pl_nll"]
    )
    target_indices = [
        index for index, row in enumerate(parent_report["rows"]) if str(row["line_sha256"]) in target_sha
    ]
    parent_target_nll = sum(parent_report["rows"][index]["ordered_nll"] for index in target_indices) / len(target_indices)
    candidate_target_nll = sum(candidate_report["rows"][index]["ordered_nll"] for index in target_indices) / len(target_indices)
    target_nll_improvement = parent_target_nll - candidate_target_nll
    stratum_improvements = {
        name: float(parent_report["by_stratum"][name]["ordered_pl_nll"])
        - float(candidate_report["by_stratum"][name]["ordered_pl_nll"])
        for name in B256_QUOTAS
    }
    checks = {
        "five_targets_native_exact": target_all_exact,
        "five_targets_margin_at_least_1_over_512": target_margin_pass,
        "target_coverage_set_at_least3": target_coverage["set_exact"] >= 3,
        "target_coverage_hybrid_at_least3": target_coverage["hybrid_order_exact"] >= 3,
        "target_coverage_ordered_at_least5": target_coverage["ordered_exact"] >= 5,
        "target_coverage_top1_at_least3": target_coverage["top1_correct"] >= 3,
        "fourteen_guards_zero_correct_to_wrong": not guard_failures,
        "other237_zero_correct_to_wrong": len(other_sha) == OTHER_ROW_COUNT and not regressions,
        "hard96_ordered_pl_nll_nondegrade": hard_improvement >= -NLL_TOLERANCE,
        "retention160_ordered_pl_nll_nondegrade": retention_improvement >= -NLL_TOLERANCE,
        "special_target5_ordered_pl_nll_nondegrade": target_nll_improvement >= -NLL_TOLERANCE,
        "count_logits_bit_exact": parent_report["native_output"]["count_logits_sha256"]
        == candidate_report["native_output"]["count_logits_sha256"],
        "value_logits_bit_exact": parent_report["native_output"]["value_logits_sha256"]
        == candidate_report["native_output"]["value_logits_sha256"],
        "native_bf16": parent_report["native_output"]["policy_logits_dtype"]
        == candidate_report["native_output"]["policy_logits_dtype"]
        == "torch.bfloat16",
    }
    return {
        "target_coverage": target_coverage,
        "target_margin_by_line_sha256": dict(target_margins),
        "guard_failures": guard_failures,
        "other237_regressions": regressions,
        "hard96_ordered_pl_nll_improvement": hard_improvement,
        "retention160_ordered_pl_nll_improvement": retention_improvement,
        "special_target5_ordered_pl_nll_improvement": target_nll_improvement,
        "retention_improvement_by_stratum": stratum_improvements,
    }, checks


def run_candidate(
    profile: Mapping[str, Any],
    profile_evidence: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    guards: Sequence[Mapping[str, Any]],
    selection_audit: Mapping[str, Any],
    formal: ModuleType,
    cw20: ModuleType,
    dependency_evidence: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    helper = formal.load_helper()
    torch = helper.torch
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    matmul_precision_before = torch.get_float32_matmul_precision()
    model = None
    parameters: dict[str, Any] = {}
    parent_actor: dict[str, Any] = {}
    requires_grad_before: dict[str, bool] = {}
    training_before = None
    parent_state_sha = None
    restore: dict[str, Any] = {"attempted": False, "pass": False}
    result: dict[str, Any] | None = None
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
            "model_config_mapping": isinstance(checkpoint.get("model_config"), dict),
            "state_mapping": isinstance(state, Mapping),
            "state80": isinstance(state, Mapping) and len(state) == 80,
            "model_sha_exact": isinstance(state, Mapping)
            and helper.model_state_sha256(state) == PARENT_MODEL_SHA256,
        }
        if not all(checkpoint_checks.values()):
            raise ProtocolError(f"guarded parent checkpoint drift: {checkpoint_checks}")
        model, model_config, kind = helper.instantiate_checkpoint(checkpoint, device)
        if kind != "ppo" or helper.model_state_sha256(model.state_dict()) != PARENT_MODEL_SHA256:
            raise ProtocolError("guarded parent runtime model drift")
        training_before = bool(model.training)
        parent_state_sha = helper.model_state_sha256(model.state_dict())
        named = dict(model.named_parameters())
        if not set(ACTOR6_NAMES).issubset(named):
            raise ProtocolError("actor6 parameters absent")
        if any(parameter.grad is not None for parameter in named.values()):
            raise ProtocolError("fresh guarded parent unexpectedly has gradient buffers")
        requires_grad_before = {name: bool(parameter.requires_grad) for name, parameter in named.items()}
        for parameter in named.values():
            parameter.requires_grad_(False)
            parameter.grad = None
        parameters = {name: named[name] for name in ACTOR6_NAMES}
        for parameter in parameters.values():
            parameter.requires_grad_(True)
        parent_actor = cw20.clone_actor(parameters, ACTOR6_NAMES)
        parent_actor_bytes = cw20.actor_bytes(parameters, ACTOR6_NAMES, np)
        parent_actor_state_sha = helper.model_state_sha256(
            {name: model.state_dict()[name] for name in ACTOR6_NAMES}
        )
        nonactor_names = sorted(set(model.state_dict()).difference(ACTOR6_NAMES))
        parent_nonactor_sha = helper.model_state_sha256(
            {name: model.state_dict()[name] for name in nonactor_names}
        )
        actor_checks = {
            "actor_names_exact": tuple(parameters) == ACTOR6_NAMES,
            "actor_dimension_exact": sum(int(value.numel()) for value in parameters.values())
            == ACTOR_DIMENSION,
            "actor_all_float32": all(str(value.dtype) == "torch.float32" for value in parameters.values()),
            "actor_bytes_sha_exact": sha256_bytes(parent_actor_bytes)
            == PARENT_ACTOR_FLOAT32_LE_SHA256,
            "actor_state_sha_exact": parent_actor_state_sha == PARENT_ACTOR_STATE_SHA256,
            "nonactor74": len(nonactor_names) == 74,
            "nonactor_sha_exact": parent_nonactor_sha == PARENT_NONACTOR_SHA256,
            "only_actor6_requires_grad": all(parameters[name].requires_grad for name in ACTOR6_NAMES)
            and all(not named[name].requires_grad for name in nonactor_names),
        }
        if not all(actor_checks.values()):
            raise ProtocolError(f"parent actor/nonactor contract drift: {actor_checks}")

        cpu_batch, cache_audit = load_b256(rows, model_config, helper)
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        outputs = helper.ppo.model_forward(model, batch, device)
        if any(value.dtype != torch.bfloat16 for value in outputs.values()):
            raise ProtocolError("parent forward is not native BF16")
        parent_report = evaluate_outputs(outputs, batch, rows, helper)
        parent_predictions = {
            str(row["line_sha256"]): parent_report["rows"][index]["predicted_order"]
            for index, row in enumerate(rows)
        }
        profile_prediction_checks = {
            str(row["line_sha256"]): parent_predictions[str(row["line_sha256"])]
            == [int(value) for value in row["predicted_order"]]
            for row in rows
        }
        if not all(profile_prediction_checks.values()):
            failures = [key for key, value in profile_prediction_checks.items() if not value]
            raise ProtocolError(f"profile/live parent prediction drift: {failures[:10]}")

        row_index = {str(row["line_sha256"]): index for index, row in enumerate(rows)}
        objectives: list[Any] = []
        objective_names: list[str] = []
        rhs_values: list[float] = []
        margin_contract: dict[str, Any] = {}
        logits = outputs["policy_logits"].float()
        for ordinal, row in enumerate(targets):
            line_sha = str(row["line_sha256"])
            index = row_index[line_sha]
            margin, detail = threat_margin_tensor(
                logits[index], batch["option_mask"][index], row["expert_order"], singleton_only=True
            )
            if (
                detail["margin"] != float(row["selection_margin"])
                or detail["stage"] != 0
                or detail["positive_option"] != int(row["expert_order"][0])
                or detail["negative_option"] != int(row["predicted_order"][0])
            ):
                raise ProtocolError(f"target profile/live margin drift: {line_sha}")
            objectives.append(margin)
            objective_names.append(f"target_{ordinal}")
            rhs_values.append(TARGET_MARGIN - detail["margin"])
            margin_contract[line_sha] = {
                "role": "target",
                **detail,
                "profile_selection_margin": float(row["selection_margin"]),
                "profile_decision_margin": float(row["decision_margin"]),
                "endpoint_margin_threshold": TARGET_MARGIN,
            }
        for ordinal, row in enumerate(guards):
            line_sha = str(row["line_sha256"])
            index = row_index[line_sha]
            margin, detail = threat_margin_tensor(
                logits[index], batch["option_mask"][index], row["expert_order"], singleton_only=False
            )
            if detail["margin"] != float(row["selection_margin"]):
                raise ProtocolError(f"guard profile/live margin drift: {line_sha}")
            objectives.append(margin)
            objective_names.append(f"guard_{ordinal}")
            rhs_values.append(-detail["margin"])
            margin_contract[line_sha] = {
                "role": "guard",
                **detail,
                "profile_selection_margin": float(row["selection_margin"]),
                "profile_decision_margin": float(row["decision_margin"]),
                "linear_constraint": "gradient_dot_delta_at_least_negative_parent_margin",
                "endpoint_margin_threshold": 0.0,
                "native_bf16_gate": "ordered_correct",
            }

        per_row_nll = ordered_nll_per_row(outputs, batch, torch)
        hard_indices = [index for index, row in enumerate(rows) if str(row["stratum"]).endswith("_hard")]
        retention_indices = [index for index in range(BATCH_SIZE) if index not in set(hard_indices)]
        target_indices = [row_index[str(row["line_sha256"])] for row in targets]
        for name, indices in (
            ("hard96_ordered_pl_nll", hard_indices),
            ("retention160_ordered_pl_nll", retention_indices),
            ("special_target5_ordered_pl_nll", target_indices),
        ):
            loss = weighted_loss(per_row_nll, indices, batch["sample_weights"], torch)
            objectives.append(-loss)
            objective_names.append(name)
            rhs_values.append(0.0)

        parameter_tuple = tuple(parameters[name] for name in ACTOR6_NAMES)
        gradients = []
        gradient_audit = {}
        for objective_index, (name, objective) in enumerate(zip(objective_names, objectives)):
            values = torch.autograd.grad(
                objective,
                parameter_tuple,
                retain_graph=objective_index + 1 < len(objectives),
                create_graph=False,
                allow_unused=False,
                materialize_grads=False,
            )
            flat = flat_gradient(values, np, torch)
            norm = float(np.linalg.norm(flat))
            if not math.isfinite(norm) or norm <= 0.0:
                raise ProtocolError(f"invalid actor6 gradient: {name}")
            gradients.append(flat)
            gradient_audit[name] = {
                "l2": norm,
                "max_abs": float(np.max(np.abs(flat))),
                "nonzero_elements": int(np.count_nonzero(flat)),
                "float64_le_sha256": array_sha(flat),
            }
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ProtocolError("autograd.grad materialized .grad buffers")
        matrix = np.stack(gradients, axis=0)
        rhs = np.asarray(rhs_values, dtype=np.float64)
        planned_delta, qp_audit = solve_minimum_norm_qp(matrix, rhs, np, optimize)

        parent_flat = cw20.flat_actor(parameters, ACTOR6_NAMES, np)
        cw20.apply_flat_actor(parameters, ACTOR6_NAMES, parent_flat + planned_delta, torch)
        candidate_flat = cw20.flat_actor(parameters, ACTOR6_NAMES, np)
        actual_delta = candidate_flat - parent_flat
        actual_l2 = float(np.linalg.norm(actual_delta))
        with torch.no_grad():
            candidate_outputs = helper.ppo.model_forward(model, batch, device)
        if any(value.dtype != torch.bfloat16 for value in candidate_outputs.values()):
            raise ProtocolError("candidate forward is not native BF16")
        candidate_report = evaluate_outputs(candidate_outputs, batch, rows, helper)

        target_margins = {}
        candidate_logits = candidate_outputs["policy_logits"].float()
        for row in targets:
            line_sha = str(row["line_sha256"])
            index = row_index[line_sha]
            _, detail = threat_margin_tensor(
                candidate_logits[index],
                batch["option_mask"][index],
                row["expert_order"],
                singleton_only=True,
            )
            target_margins[line_sha] = float(detail["margin"])

        target_sha = {str(row["line_sha256"]) for row in targets}
        guard_sha = {str(row["line_sha256"]) for row in guards}
        comparison, b256_checks = compare_b256(
            parent_report, candidate_report, target_sha, guard_sha, target_margins
        )
        candidate_state_sha = helper.model_state_sha256(model.state_dict())
        candidate_nonactor_sha = helper.model_state_sha256(
            {name: model.state_dict()[name] for name in nonactor_names}
        )
        changed_names = sorted(
            name
            for name in model.state_dict()
            if not torch.equal(checkpoint["model_state_dict"][name].to(model.state_dict()[name].device), model.state_dict()[name])
        )
        delta_checks = {
            "planned_finite": bool(np.isfinite(planned_delta).all()),
            "actual_finite": bool(np.isfinite(actual_delta).all()),
            "actual_l2_positive": actual_l2 > 0.0,
            "actual_l2_at_most_0p001": actual_l2 <= ACTOR_L2_HARD_CAP + 1.0e-12,
            "changed_exact_effective_actor5": changed_names == sorted(EFFECTIVE_ACTOR_NAMES),
            "common_logit_bias_bit_exact_parent": torch.equal(
                checkpoint["model_state_dict"][INVARIANT_COMMON_LOGIT_BIAS].to(
                    model.state_dict()[INVARIANT_COMMON_LOGIT_BIAS].device
                ),
                model.state_dict()[INVARIANT_COMMON_LOGIT_BIAS],
            ),
            "nonactor74_bit_exact": candidate_nonactor_sha == PARENT_NONACTOR_SHA256,
            "candidate_state_differs_parent": candidate_state_sha != PARENT_MODEL_SHA256,
            "all_grad_buffers_none": all(parameter.grad is None for parameter in model.parameters()),
        }
        all_pass = all(delta_checks.values()) and all(b256_checks.values())
        payload = None
        if all_pass:
            candidate_actor_bytes = cw20.actor_bytes(parameters, ACTOR6_NAMES, np)
            layout = cw20.actor_layout(parameters, ACTOR6_NAMES)
            layout_sha = sha256_bytes(canonical_json(layout))
            payload = {
                "formula": "guarded_current_U468_actor6_plus_sole_radially_capped_minimum_norm_boundary_QP",
                "parent_checkpoint_file_sha256": PARENT_FILE_SHA256,
                "parent_model_state_sha256": PARENT_MODEL_SHA256,
                "profile_selection_file_sha256": str(profile_evidence["sha256"]),
                "b256_cache_sha256": str(cache_audit["cache_sha256"]),
                "seed": SEED,
                "target_margin": TARGET_MARGIN,
                "actor_l2_hard_cap": ACTOR_L2_HARD_CAP,
                "nominal_delta_l2": float(qp_audit["nominal_l2"]),
                "planned_delta_float64_le_sha256": array_sha(planned_delta),
                "actual_float32_quantized_delta_l2": actual_l2,
                "actual_float32_quantized_delta_float64_le_sha256": array_sha(actual_delta),
                "actor_names": list(ACTOR6_NAMES),
                "changed_parameter_names": changed_names,
                "actor_layout": layout,
                "actor_layout_sha256": layout_sha,
                "candidate_actor_float32_le": cw20.xz_payload(candidate_actor_bytes),
                "candidate_actor_float32_le_sha256": sha256_bytes(candidate_actor_bytes),
                "candidate_model_state_sha256": candidate_state_sha,
                "candidate_nonactor_sha256": candidate_nonactor_sha,
                "optimizer_instances": 0,
                "optimizer_steps": 0,
                "learning_rate": None,
                "single_candidate_only": True,
            }
        result = {
            "schema_version": SCHEMA,
            "status": (
                "GO_CURRENT_PARENT_BOUNDARYQP_B256"
                if all_pass
                else "NO_GO_CURRENT_PARENT_BOUNDARYQP_B256_CLOSE_LINEAGE"
            ),
            "seed": SEED,
            "input_lock": {
                "parent": parent_evidence,
                "profile_selection": dict(profile_evidence),
                "dependencies": dict(dependency_evidence),
            },
            "runtime": dict(runtime),
            "profile_selection": dict(selection_audit),
            "cache": cache_audit,
            "parent_checkpoint_checks": checkpoint_checks,
            "parent_actor_nonactor_checks": actor_checks,
            "optimization": {
                "objective_order": objective_names,
                "margin_contract": margin_contract,
                "gradient_audit": gradient_audit,
                "boundary_qp": qp_audit,
                "actual_float32_quantized_delta_l2": actual_l2,
                "actual_float32_quantized_delta_float64_le_sha256": array_sha(actual_delta),
            },
            "b256": {
                "parent": parent_report,
                "candidate": candidate_report,
                "comparison": comparison,
                "checks": b256_checks,
            },
            "candidate_integrity": {
                "checks": delta_checks,
                "changed_parameter_names": changed_names,
                "candidate_model_state_sha256": candidate_state_sha,
                "candidate_nonactor_sha256": candidate_nonactor_sha,
            },
            "candidate_payload": payload,
            "candidate_accounting": {
                "parent_only_profile_changed_candidates": 0,
                "changed_candidate_train_shadow_count": 1,
                "official_unique_changed_candidate_count_consumed": 0,
                "endpoint_sweep": False,
                "retry": False,
            },
            "scope": {
                "train_only": True,
                "validation_or_test_members_opened": 0,
                "changed_candidates_constructed": 1,
                "optimizer_instances": 0,
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
                "model_artifact_writes": 0,
                "network_package_upload_submission": False,
                "fulltrain_evaluation_performed": False,
            },
        }
    finally:
        if model is not None and parameters and parent_actor:
            restore["attempted"] = True
            cw20.restore_actor(parameters, ACTOR6_NAMES, parent_actor, torch)
            named_after = dict(model.named_parameters())
            for name, parameter in named_after.items():
                parameter.requires_grad_(requires_grad_before[name])
                parameter.grad = None
            model.train(training_before)
            restored_sha = helper.model_state_sha256(model.state_dict())
            restore.update(
                {
                    "parent_model_state_sha256": restored_sha,
                    "requires_grad_restored": all(
                        bool(parameter.requires_grad) == requires_grad_before[name]
                        for name, parameter in named_after.items()
                    ),
                    "training_mode_restored": bool(model.training) == bool(training_before),
                    "all_grad_buffers_none": all(parameter.grad is None for parameter in model.parameters()),
                    "pass": restored_sha == parent_state_sha == PARENT_MODEL_SHA256,
                }
            )
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)
        torch.use_deterministic_algorithms(deterministic_before)
        torch.backends.cudnn.benchmark = cudnn_benchmark_before
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.set_float32_matmul_precision(matmul_precision_before)
    if result is None:
        raise ProtocolError("candidate result was not constructed")
    result["restore"] = restore
    if restore.get("pass") is not True:
        raise ProtocolError(f"guarded parent restoration failed: {restore}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="static")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--profile-sha256")
    parser.add_argument("--expected-self-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_absence = publication_absence()
    if initial_absence["both_absent"] is not True:
        raise ProtocolError(f"one-shot publication targets are not absent: {initial_absence}")
    pre_attempt_runtime = validate_runtime(require_cuda=False)
    runtime = pre_attempt_runtime
    source, self_evidence = read_regular_bytes(
        SCRIPT,
        args.expected_self_sha256,
        "current-parent boundary-QP solver",
        expected_mode=0o555,
    )
    source_audit = static_source_audit(source)
    control = audit_control_plane()
    dependencies = audit_dependencies()
    if args.mode == "static":
        final_static_absence = publication_absence()
        if (
            final_static_absence["both_absent"] is not True
            or final_static_absence != initial_absence
        ):
            raise ProtocolError(
                f"static one-shot target state changed: {final_static_absence}"
            )
        result = {
            "schema_version": SCHEMA,
            "status": "STATIC_AUDIT_PASS",
            "self": self_evidence,
            "runtime": runtime,
            "control_plane": control,
            "dependencies": dependencies,
            "constants": {
                "seed": SEED,
                "target_margin": TARGET_MARGIN,
                "actor_l2_hard_cap": ACTOR_L2_HARD_CAP,
                "actor6_names": list(ACTOR6_NAMES),
                "actor_dimension": ACTOR_DIMENSION,
                "targets": TARGET_COUNT,
                "guards": GUARD_COUNT,
                "b256_rows": BATCH_SIZE,
                "guard_linear_constraint": (
                    "gradient_dot_delta_at_least_negative_parent_policy_margin; "
                    "native_bf16_ordered_correct_endpoint_gate"
                ),
                "profile_selection_path": str(DEFAULT_PROFILE.relative_to(ROOT)),
                "profile_selection_sha256": EXPECTED_PROFILE_SHA256,
                "b256_selection_sha256": EXPECTED_B256_SELECTION_SHA256,
                "optimizer_instances": 0,
                "optimizer_steps": 0,
                "learning_rate": None,
                "alpha_lr_radius_seed_or_endpoint_sweep": False,
            },
            "audit": source_audit,
            "scope": {
                "candidate_constructed": False,
                "cuda_forward": False,
                "profile_opened": False,
                "model_or_evidence_writes": 0,
                "validation_test_network_submission": False,
            },
            "one_shot_publication": {
                "targets": final_static_absence,
                "pre_post_absence_exact": True,
                "attempt_marker_writes": 0,
                "result_writes": 0,
                "total_writes": 0,
            },
        }
        canonical_json(result)
        print(canonical_json(result).decode("utf-8"), end="")
        return

    if args.profile.resolve() != DEFAULT_PROFILE.resolve():
        raise ProtocolError("--profile must be the preregistered exact path")
    if not is_sha256(args.profile_sha256):
        raise ProtocolError("run mode requires --profile-sha256")
    if args.profile_sha256 != EXPECTED_PROFILE_SHA256:
        raise ProtocolError("--profile-sha256 must equal the frozen corrected-v3 SHA")
    if not is_sha256(args.expected_self_sha256):
        raise ProtocolError("run mode requires --expected-self-sha256")
    profile_payload, profile_evidence = read_regular_bytes(
        args.profile,
        args.profile_sha256,
        "parent-only profile/selection",
        expected_mode=0o444,
    )
    profile = strict_json(profile_payload, "parent-only profile/selection")
    rows, targets, guards, selection_audit = validate_profile_selection(
        profile, str(args.profile_sha256)
    )
    final_absence = publication_absence()
    if final_absence["both_absent"] is not True:
        raise ProtocolError(
            f"one-shot targets changed immediately before attempt: {final_absence}"
        )
    marker_payload = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "FORMAL_ATTEMPT_CONSUMED_BEFORE_CUDA_MODEL_FORWARD_OR_CANDIDATE",
        "consumed_at_utc": utc_now(),
        "branch": BRANCH,
        "attempt_number": 1,
        "solver": self_evidence,
        "runtime_before_attempt": pre_attempt_runtime,
        "profile_selection": profile_evidence,
        "b256_selection_sha256": selection_audit["b256_selection_sha256"],
        "target_line_sha256": selection_audit["target_line_sha256"],
        "guard_line_sha256": selection_audit["guard_line_sha256"],
        "control_plane": control,
        "dependencies": dependencies,
        "invocation": {
            "python": str(EXPECTED_PYTHON),
            "isolated": True,
            "dont_write_bytecode": True,
            "mode": "run",
            "profile": str(DEFAULT_PROFILE.relative_to(ROOT)),
            "profile_sha256": EXPECTED_PROFILE_SHA256,
            "expected_self_sha256": self_evidence["sha256"],
        },
        "reserved_result": {
            "path": str(RESULT.relative_to(ROOT)),
            "must_be_canonical_json_o_excl_mode_0444": True,
        },
        "candidate_accounting_before_marker": {
            "cuda_accessed": False,
            "model_instances": 0,
            "cuda_forwards": 0,
            "changed_candidates_constructed": 0,
            "optimizer_instances": 0,
            "optimizer_steps": 0,
            "retry_authorized": False,
        },
        "scope": {
            "train_only_profile_opened": True,
            "validation_test_broad_gold_opened": False,
            "network_package_upload_submission": False,
            "model_artifact_writes": 0,
        },
    }
    marker_bytes = canonical_json(marker_payload)
    marker_evidence = publish_o_excl(ATTEMPT_MARKER, marker_bytes)

    phase = "cuda_runtime_validation_after_attempt"
    cuda_runtime: dict[str, Any] | None = None
    try:
        cuda_runtime = validate_runtime(require_cuda=True)
        runtime = cuda_runtime
        phase = "frozen_module_import_after_attempt"
        formal, formal_import_evidence = import_frozen(
            FORMAL, FORMAL_SHA256, "currentparent_boundaryqp_formal_v3"
        )
        cw20, cw20_import_evidence = import_frozen(
            CW20, CW20_SHA256, "currentparent_boundaryqp_cw20_codec"
        )
        dependencies = dict(dependencies)
        dependencies["formal_v3_import"] = formal_import_evidence
        dependencies["actor_codec_cw20_import"] = cw20_import_evidence
        phase = "sole_candidate_construction_and_b256_after_attempt"
        result = run_candidate(
            profile,
            profile_evidence,
            rows,
            targets,
            guards,
            selection_audit,
            formal,
            cw20,
            dependencies,
            runtime,
        )
        phase = "sole_candidate_completed_and_restored"
        result["self"] = self_evidence
        result["control_plane"] = control
        result["source_audit"] = source_audit
    except Exception as error:
        result = {
            "schema_version": SCHEMA,
            "status": "NO_GO_CURRENT_PARENT_BOUNDARYQP_FORMAL_EXCEPTION_CLOSE_LINEAGE",
            "reason": "ONE_SHOT_FORMAL_EXCEPTION_AFTER_ATTEMPT_CONSUMED",
            "phase": phase,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "seed": SEED,
            "self": self_evidence,
            "runtime_before_attempt": pre_attempt_runtime,
            "cuda_runtime_after_attempt": cuda_runtime,
            "control_plane": control,
            "source_audit": source_audit,
            "input_lock": {
                "profile_selection": profile_evidence,
                "dependencies_before_attempt": dependencies,
            },
            "profile_selection": selection_audit,
            "candidate_payload": None,
            "candidate_accounting": {
                "attempt_consumed": 1,
                "changed_candidate_train_shadow_count_upper_bound": 1,
                "exact_shadow_count_unavailable_due_fail_closed_exception": True,
                "official_unique_changed_candidate_count_consumed": 0,
                "retry": False,
            },
            "scope": {
                "train_only": True,
                "validation_or_test_members_opened": 0,
                "model_artifact_writes": 0,
                "network_package_upload_submission": False,
            },
        }

    result["formal_publication"] = {
        "branch": BRANCH,
        "attempt_marker": marker_evidence,
        "attempt_payload_canonical_sha256": sha256_bytes(marker_bytes),
        "attempt_consumed_before_cuda_model_forward_or_candidate": True,
        "result_path": str(RESULT.relative_to(ROOT)),
        "result_o_excl_mode_0444": True,
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
    result_evidence = publish_o_excl(RESULT, result_bytes)
    candidate_payload = result.get("candidate_payload")
    summary = {
        "schema_version": PUBLICATION_SUMMARY_SCHEMA,
        "publication_status": "FORMAL_RESULT_PUBLISHED",
        "decision_status": result["status"],
        "attempt_marker": marker_evidence,
        "result": result_evidence,
        "result_payload_canonical_sha256": sha256_bytes(result_bytes),
        "candidate_payload_present": isinstance(candidate_payload, Mapping),
        "candidate_model_state_sha256": (
            candidate_payload.get("candidate_model_state_sha256")
            if isinstance(candidate_payload, Mapping)
            else None
        ),
        "candidate_nonactor_sha256": (
            candidate_payload.get("candidate_nonactor_sha256")
            if isinstance(candidate_payload, Mapping)
            else None
        ),
        "retry_authorized": False,
    }
    canonical_json(summary)
    print(canonical_json(summary).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
