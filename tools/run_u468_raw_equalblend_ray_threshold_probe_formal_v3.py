#!/usr/bin/env python3
"""One-shot formal train-only probe for the frozen U468 actor6 ray contract.

Static/cache/contract modes are zero-write.  Formal mode is CUDA-only and may
publish exactly two evidence JSON files: one O_EXCL attempt marker before any
checkpoint deserialization/model construction, and one O_EXCL final result.
It never opens validation members, trains, serializes a model, or uses network.
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
import struct
import sys
import time
import zipfile
from datetime import datetime, timezone
from fractions import Fraction
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal_v3.py"
SUPERSEDED_FORMAL_V2 = (
    TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal.py"
)
SUPERSEDED_FORMAL_V2_SHA256 = (
    "c1df4616334f9a3c6f3cf76c5a7ce22f555412edfc8f763fc6ec708e6729435a"
)
SUPERSEDED_FORMAL_V2_BRANCH = (
    "ppo_u468_raw_equalblend_ray_threshold_probe_design202608121"
)
SUPERSEDED_FORMAL_V2_PREREGISTRATION = ROOT / (
    f"artifacts/{SUPERSEDED_FORMAL_V2_BRANCH}.preregistration.json"
)
SUPERSEDED_FORMAL_V2_PREREGISTRATION_SHA256 = (
    "3801c2f47bbd2ab67c07d409bbe5d931b4b63e706f2c69f99353c4eae392ade1"
)
SUPERSEDED_FORMAL_V2_ATTEMPT_MARKER = ROOT / (
    f".ptcg-{SUPERSEDED_FORMAL_V2_BRANCH}-attempt.json"
)
SUPERSEDED_FORMAL_V2_RESULT = ROOT / (
    f"artifacts/{SUPERSEDED_FORMAL_V2_BRANCH}.result.json"
)
SUPERSEDED_FORMAL_V1 = (
    TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal_v1_superseded.py"
)
SUPERSEDED_FORMAL_V1_SHA256 = (
    "49197fd23fdf8baaac245a25b2d38c4ad0fa0d4017f3959dfe845396c1fe06a5"
)
DESIGN_TOOL = TOOLS / "probe_u468_raw_equalblend_ray_thresholds.py"
DESIGN_TOOL_SHA256 = (
    "7339184a12224302f3bf9701c1fbc722e317c3c5a9f68ae4cbfcccda87b8be52"
)
DESIGN_ARTIFACT = ROOT / (
    "artifacts/ppo_u468_raw_equalblend_ray_threshold_probe_contract_"
    "design202608121.json"
)
DESIGN_ARTIFACT_SHA256 = (
    "d2b2547cd03fd27b46a19e32499ba233280ed32eebe64cd98bf46570ce4604c4"
)
HELPER = TOOLS / "run_u468_raw_balanced_train_only_endpoint_gate.py"
HELPER_SHA256 = (
    "84b51d8ef51e2184271c4bf358e5d59b74ea631daa79b84d73fffaf44a37037d"
)

BRANCH = "ppo_u468_raw_equalblend_ray_threshold_probe_design202608122"
PREREGISTRATION = ROOT / f"artifacts/{BRANCH}.preregistration.json"
ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-attempt.json"
RESULT = ROOT / f"artifacts/{BRANCH}.result.json"
PREREGISTRATION_SCHEMA = (
    "ptcg-u468-raw-equalblend-ray-threshold-probe-formal-preregistration-v2"
)
ATTEMPT_SCHEMA = "ptcg-u468-raw-equalblend-ray-threshold-probe-formal-attempt-v2"
RESULT_SCHEMA = "ptcg-u468-raw-equalblend-ray-threshold-probe-result-v2"

BATCH_SIZE = 256
SEED = 202608121
MODEL_ORDER = (
    "raw",
    "alpha1",
    "alpha2",
    "alpha4",
    "alpha8",
    "alpha16",
    "alpha32",
    "alpha64",
    "alpha128",
    "alpha256",
)
ALPHAS = {"raw": 0, "alpha1": 1, **{f"alpha{x}": x for x in (2, 4, 8, 16, 32, 64, 128, 256)}}
PANEL_ORDER = ("flg", "pokemonfan", "core5")
MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)
SAFETY_METRICS = MAIN_METRICS + (
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
TRANSITION_CELLS = ("cc", "cw", "wc", "ww")
ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
FROZEN_METRICS_CANONICAL_SHA256 = {
    "flg": "ba06f1375af688cd141b4508230d5503d64fd1c0e16c13da50a573e9dbb4f5a3",
    "pokemonfan": "847bb45b3a713b8cc8ad3c66329b4c8150ecff7c18047f7d3b435a2cb98ce7c1",
    "core5": "50ea4fa91d76da89ac069c6e0afecc7bb90bd698ce37f0509a8581282546e3ba",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RuntimeError(f"{label} has duplicate key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise RuntimeError(f"{label} has forbidden JSON constant {value}")

    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=reject)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root is not an object")
    return value


def read_regular_bytes(
    path: Path,
    expected_sha256: str,
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
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"{label} is not a single-link regular file")
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
    identity = (after.st_dev, after.st_ino, after.st_size)
    if (
        (before.st_dev, before.st_ino, before.st_size) != identity
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or visible.st_nlink != 1
        or (visible.st_dev, visible.st_ino, visible.st_size) != identity
    ):
        raise RuntimeError(f"{label} changed during held-fd read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    if digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    mode = stat.S_IMODE(after.st_mode)
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode": oct(mode),
        "device": after.st_dev,
        "inode": after.st_ino,
        "nlink": after.st_nlink,
        "held_fd_pre_post_visible_identity_exact": True,
    }


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    if path not in {ATTEMPT_MARKER, RESULT}:
        raise RuntimeError(f"publication outside exact evidence allowlist: {path}")
    parent = os.lstat(path.parent)
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or path.parent.resolve()
        not in {ROOT.resolve(), (ROOT / "artifacts").resolve()}
    ):
        raise RuntimeError(f"publication parent identity is unsafe: {path.parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            count = os.write(fd, view[offset:])
            if count <= 0:
                raise RuntimeError(f"short publication write: {path}")
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
        identity = (after.st_dev, after.st_ino, after.st_size)
        if (
            reloaded != payload
            or not stat.S_ISREG(after.st_mode)
            or stat.S_IMODE(after.st_mode) != 0o444
            or after.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size) != identity
            or stat.S_ISLNK(visible.st_mode)
            or (visible.st_dev, visible.st_ino, visible.st_size) != identity
        ):
            raise RuntimeError(f"unsafe O_EXCL publication: {path}")
    finally:
        os.close(fd)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_bytes(reloaded),
        "bytes": len(reloaded),
        "mode": "0o444",
        "device": after.st_dev,
        "inode": after.st_ino,
        "nlink": after.st_nlink,
        "descriptor_held_exact_reload": True,
    }


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")


def load_module(path: Path, digest: str, name: str, mode: int) -> ModuleType:
    read_regular_bytes(path, digest, name, expected_mode=mode)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import hash-bound module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_frozen_design() -> tuple[ModuleType, dict[str, Any], dict[str, Any]]:
    design = load_module(DESIGN_TOOL, DESIGN_TOOL_SHA256, "u468_ray_design121", 0o555)
    artifact_bytes, artifact_evidence = read_regular_bytes(
        DESIGN_ARTIFACT,
        DESIGN_ARTIFACT_SHA256,
        "frozen ray design121 artifact",
        expected_mode=0o444,
    )
    artifact = strict_json_bytes(artifact_bytes, "frozen ray design121 artifact")
    fixed, profile = design.verify_fixed_inputs()
    cache = design.cache_audit(profile["profile"])
    expected = {
        "schema_version": design.PREREGISTRATION_SCHEMA,
        "status": "contract_only_future_formal_revision_required",
        "probe_contract": design.probe_contract(
            fixed["self"], fixed, cache, profile
        ),
    }
    if artifact != expected:
        raise RuntimeError("frozen design121 artifact/object drift")
    if artifact["probe_contract"]["runner"]["sha256"] != DESIGN_TOOL_SHA256:
        raise RuntimeError("design121 predecessor runner binding drift")
    return design, artifact, {
        "tool": fixed["self"],
        "artifact": artifact_evidence,
        "artifact_canonical_sha256": sha256_bytes(canonical_json(artifact)),
        "fixed_inputs": fixed,
        "cache": cache,
    }


def load_helper() -> ModuleType:
    return load_module(HELPER, HELPER_SHA256, "u468_ray_hash_bound_helper", 0o555)


def verify_superseded_formal_v1() -> dict[str, Any]:
    _, evidence = read_regular_bytes(
        SUPERSEDED_FORMAL_V1,
        SUPERSEDED_FORMAL_V1_SHA256,
        "superseded formal v1 archive",
        expected_mode=0o555,
    )
    return {
        "file": evidence,
        "status": "superseded_before_any_preregistration_or_formal_attempt",
        "reason": (
            "float-nextafter secant open-interval sampling could remain tied "
            "for exact BF16 dyadic logits"
        ),
        "preregistration_created": False,
        "formal_attempt_consumed": False,
    }


def verify_superseded_formal_v2() -> dict[str, Any]:
    _, runner = read_regular_bytes(
        SUPERSEDED_FORMAL_V2,
        SUPERSEDED_FORMAL_V2_SHA256,
        "superseded formal v2 runner",
        expected_mode=0o555,
    )
    preregistration_payload, preregistration = read_regular_bytes(
        SUPERSEDED_FORMAL_V2_PREREGISTRATION,
        SUPERSEDED_FORMAL_V2_PREREGISTRATION_SHA256,
        "superseded formal v2 invalid preregistration",
        expected_mode=0o444,
    )
    parsed = strict_json_bytes(
        preregistration_payload,
        "superseded formal v2 invalid preregistration",
    )
    contract = parsed.get("formal_contract")
    static = (
        contract.get("zero_write_guards", {}).get("static_ast_audit", {})
        if isinstance(contract, dict)
        else {}
    )
    nextafter_scope = static.get("math_nextafter_scope")
    if (
        parsed.get("schema_version")
        != "ptcg-u468-raw-equalblend-ray-threshold-probe-formal-preregistration-v1"
        or parsed.get("status") != "locked_before_formal"
        or not isinstance(contract, dict)
        or contract.get("branch") != SUPERSEDED_FORMAL_V2_BRANCH
        or contract.get("formal_runner", {}).get("sha256")
        != SUPERSEDED_FORMAL_V2_SHA256
        or contract.get("formal_runner", {}).get("path")
        != str(SUPERSEDED_FORMAL_V2.relative_to(ROOT))
        or nextafter_scope != [[943, "synthetic_self_test"]]
    ):
        raise RuntimeError("superseded formal v2 evidence semantic drift")
    old_marker_absent = not (
        SUPERSEDED_FORMAL_V2_ATTEMPT_MARKER.exists()
        or SUPERSEDED_FORMAL_V2_ATTEMPT_MARKER.is_symlink()
    )
    old_result_absent = not (
        SUPERSEDED_FORMAL_V2_RESULT.exists()
        or SUPERSEDED_FORMAL_V2_RESULT.is_symlink()
    )
    if not old_marker_absent or not old_result_absent:
        raise RuntimeError("superseded formal v2 unexpectedly consumed/published")
    return {
        "status": "superseded_before_formal_attempt",
        "reason": (
            "raw Python contract rebuilt math_nextafter_scope as a tuple inside "
            "a list, while JSON roundtrip froze it as a list inside a list; "
            "direct formal contract equality therefore failed before marker"
        ),
        "runner": runner,
        "invalid_preregistration": preregistration,
        "invalid_preregistration_json_scope": nextafter_scope,
        "attempt_marker": {
            "path": str(SUPERSEDED_FORMAL_V2_ATTEMPT_MARKER.relative_to(ROOT)),
            "absent": True,
        },
        "result": {
            "path": str(SUPERSEDED_FORMAL_V2_RESULT.relative_to(ROOT)),
            "absent": True,
        },
        "formal_attempt_consumed": False,
    }


def cuda_runtime(helper: ModuleType) -> dict[str, Any]:
    torch = helper.torch
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("requires exact CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("formal ray probe requires an available CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal ray probe requires CUDA BF16 support")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda:0",
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version()),
        "device_name": str(props.name),
        "compute_capability": [int(props.major), int(props.minor)],
        "total_memory": int(props.total_memory),
        "bf16_supported": True,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "float32_matmul_precision": "high",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": SEED,
    }


def output_absence() -> dict[str, bool]:
    return {
        "preregistration_absent": not (
            PREREGISTRATION.exists() or PREREGISTRATION.is_symlink()
        ),
        "attempt_marker_absent": not (
            ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink()
        ),
        "result_absent": not (RESULT.exists() or RESULT.is_symlink()),
    }


def fixed_ledger() -> list[dict[str, Any]]:
    return [
        {"ordinal": ordinal, "model": model, "panel": panel}
        for ordinal, (model, panel) in enumerate(
            ((model, panel) for model in MODEL_ORDER for panel in PANEL_ORDER),
            start=1,
        )
    ]


def formal_contract(
    runner_evidence: Mapping[str, Any],
    design_artifact: Mapping[str, Any],
    design_evidence: Mapping[str, Any],
    superseded_v1: Mapping[str, Any],
    superseded_v2: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    contract = design_artifact["probe_contract"]
    result = {
        "branch": BRANCH,
        "formal_implementation_revision": "v3",
        "formal_runner": {
            **runner_evidence,
            "python": str(EXPECTED_PYTHON),
            "flags": ["-I", "-B"],
        },
        "frozen_predecessor": {
            "contract_only_tool": design_evidence["tool"],
            "design_artifact": design_evidence["artifact"],
            "design_artifact_canonical_sha256": design_evidence[
                "artifact_canonical_sha256"
            ],
            "exact_probe_contract": contract,
            "superseded_formal_v1": dict(superseded_v1),
            "superseded_formal_v2": dict(superseded_v2),
            "design121_future_formal_revision_resolution": {
                "original_design_status": str(design_artifact["status"]),
                "original_future_path_superseded": True,
                "superseded_by": {
                    "formal_implementation_revision": "v2",
                    "branch": SUPERSEDED_FORMAL_V2_BRANCH,
                    "failure_boundary": "before_attempt_marker",
                    "formal_attempt_consumed": False,
                    "reason": str(superseded_v2["reason"]),
                },
                "replacement": {
                    "formal_implementation_revision": "v3",
                    "branch": BRANCH,
                    "preregistration_schema": PREREGISTRATION_SCHEMA,
                    "attempt_schema": ATTEMPT_SCHEMA,
                    "result_schema": RESULT_SCHEMA,
                },
            },
        },
        "runtime": dict(runtime),
        "zero_write_guards": {
            "static_ast_audit": ast_audit(),
            "zero_input_synthetic_self_test": synthetic_self_test(),
        },
        "evaluation": {
            "ledger": fixed_ledger(),
            "evaluation_count_exact": 30,
            "batch_size": BATCH_SIZE,
            "workers": 0,
            "physical_protocol": (
                "one panel loader; for each frozen batch evaluate all ten states "
                "sequentially before advancing"
            ),
            "all_30_complete_before_selection": True,
            "fingerprint_recipes": {
                "identity_fields": [
                    "panel",
                    "archive",
                    "member",
                    "line_index_zero_based",
                    "line_sha256",
                ],
                "prediction": (
                    "canonical identity JSON, NUL, canonical greedy-order JSON, NUL"
                ),
                "allowed_policy_logits": (
                    "canonical identity JSON, NUL, literal <f4, NUL, canonical "
                    "shape JSON, NUL, ascending original option-index JSON, NUL, "
                    "CPU contiguous IEEE-754 float32 little-endian bytes, NUL"
                ),
                "count_and_value_logits": (
                    "canonical identity JSON, NUL, literal <f4, NUL, canonical "
                    "shape JSON, NUL, CPU contiguous IEEE-754 float32 little-endian "
                    "bytes, NUL"
                ),
                "empty_allowed_slice": "shape [0], indices [], zero data bytes",
            },
            "secant_advisory_implementation": {
                "arithmetic": "Fraction.from_float exact rational arithmetic",
                "crossing_domain": "all pair roots alpha >= 0",
                "same_exact_root_pairs_grouped": True,
                "tie_evaluation": "exact root with lowest full option index",
                "following_open_interval": (
                    "midpoint to next distinct exact root; after final root use "
                    "root + max(1, root)"
                ),
                "action_rank_scope": "allowed option indices",
                "top1_rank_scope": "full padded policy vector as official argmax",
                "zero_boundary_crossings_included": True,
                "float_nextafter_forbidden_outside_synthetic_counterexample": True,
                "advisory_only_never_selection": True,
            },
        },
        "fixed_input_snapshot_canonical_sha256": sha256_bytes(
            canonical_json(design_evidence["fixed_inputs"])
        ),
        "train_cache_canonical_sha256": sha256_bytes(
            canonical_json(design_evidence["cache"])
        ),
        "publication": {
            "preregistration": str(PREREGISTRATION.relative_to(ROOT)),
            "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
            "result": str(RESULT.relative_to(ROOT)),
            "attempt_marker_before_checkpoint_deserialization_or_model_instance": True,
            "evidence_writes_exact": 2,
            "model_artifact_writes": 0,
            "all_publications_O_EXCL_mode_0444": True,
        },
        "scope": {
            "train_only": True,
            "validation_member_payloads_opened": False,
            "validation_results_read": False,
            "training_optimizer_backward": False,
            "model_checkpoint_writes": 0,
            "network_upload_submission": False,
        },
    }
    roundtripped = strict_json_bytes(
        canonical_json(result),
        "formal contract JSON-native roundtrip",
    )
    if roundtripped != result:
        raise RuntimeError(
            "formal contract is not object-exact after canonical JSON roundtrip"
        )
    return result


def canonical_fragment(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def fingerprint_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "panel": str(identity["panel"]),
        "archive": str(identity["archive"]),
        "member": str(identity["member"]),
        "line_index_zero_based": int(identity["line_index_zero_based"]),
        "line_sha256": str(identity["line_sha256"]),
    }


def transition_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "panel": str(identity["panel"]),
        "member": str(identity["member"]),
        "line_index_zero_based": int(identity["line_index_zero_based"]),
        "line_sha256": str(identity["line_sha256"]),
        "context": int(identity["context"]),
    }


def update_scalar_fingerprint(
    digest: Any,
    identity: Mapping[str, Any],
    name: str,
    value: Any,
) -> None:
    digest.update(canonical_fragment(fingerprint_identity(identity)))
    digest.update(b"\0")
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_fragment(value))
    digest.update(b"\0")


def update_float32_fingerprint_bytes(
    digest: Any,
    identity: Mapping[str, Any],
    shape: Sequence[int],
    raw_little_endian_float32: bytes,
    *,
    option_indices: Sequence[int] | None = None,
) -> None:
    expected_bytes = math.prod(int(value) for value in shape) * 4
    if len(raw_little_endian_float32) != expected_bytes:
        raise RuntimeError("float32 fingerprint byte length/shape mismatch")
    digest.update(canonical_fragment(fingerprint_identity(identity)))
    digest.update(b"\0<f4\0")
    digest.update(canonical_fragment([int(value) for value in shape]))
    digest.update(b"\0")
    if option_indices is not None:
        indices = [int(value) for value in option_indices]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise RuntimeError("allowed option fingerprint order is not ascending unique")
        if list(shape) != [len(indices)]:
            raise RuntimeError("allowed option indices/shape mismatch")
        digest.update(canonical_fragment(indices))
        digest.update(b"\0")
    digest.update(raw_little_endian_float32)
    digest.update(b"\0")


def tensor_float32_le_bytes(tensor: Any, torch: Any) -> tuple[list[int], bytes]:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    array = value.numpy()
    if sys.byteorder != "little":
        array = array.byteswap().newbyteorder("<")
    else:
        array = array.astype("<f4", copy=False)
    return [int(item) for item in value.shape], array.tobytes(order="C")


def new_state_fingerprints() -> dict[str, Any]:
    return {
        name: hashlib.sha256()
        for name in (
            "prediction",
            "allowed_policy_logits_float32_le",
            "count_logits_float32_le",
            "value_logits_float32_le",
            "predicted_count",
            "greedy_length",
            "count_correct",
            "value_sign",
            "value_correct",
            "top1_index",
        )
    }


def update_prediction_fingerprint(
    digest: Any,
    identity: Mapping[str, Any],
    order: Sequence[int],
) -> None:
    digest.update(canonical_fragment(fingerprint_identity(identity)))
    digest.update(b"\0")
    digest.update(canonical_fragment([int(value) for value in order]))
    digest.update(b"\0")


def update_state_fingerprints(
    digests: Mapping[str, Any],
    identity: Mapping[str, Any],
    order: Sequence[int],
    flags: Mapping[str, bool],
    policy_row: Any,
    count_row: Any,
    value_row: Any,
    option_mask_row: Any,
    top1_index: int,
    value_sign: bool,
    torch: Any,
) -> None:
    allowed = [
        index
        for index, allowed_value in enumerate(option_mask_row.tolist())
        if bool(allowed_value)
    ]
    policy_allowed = policy_row[allowed]
    shape, payload = tensor_float32_le_bytes(policy_allowed, torch)
    update_float32_fingerprint_bytes(
        digests["allowed_policy_logits_float32_le"],
        identity,
        shape,
        payload,
        option_indices=allowed,
    )
    for name, tensor in (
        ("count_logits_float32_le", count_row),
        ("value_logits_float32_le", value_row),
    ):
        shape, payload = tensor_float32_le_bytes(tensor, torch)
        update_float32_fingerprint_bytes(
            digests[name], identity, shape, payload
        )
    update_prediction_fingerprint(digests["prediction"], identity, order)
    update_scalar_fingerprint(
        digests["predicted_count"], identity, "predicted_count", len(order)
    )
    update_scalar_fingerprint(
        digests["greedy_length"], identity, "greedy_length", len(order)
    )
    update_scalar_fingerprint(
        digests["count_correct"], identity, "count_correct", flags["count_correct"]
    )
    update_scalar_fingerprint(
        digests["value_sign"], identity, "value_sign", value_sign
    )
    update_scalar_fingerprint(
        digests["value_correct"], identity, "value_correct", flags["value_correct"]
    )
    update_scalar_fingerprint(
        digests["top1_index"], identity, "top1_index", top1_index
    )


def finalize_fingerprints(digests: Mapping[str, Any]) -> dict[str, str]:
    return {name: digest.hexdigest() for name, digest in sorted(digests.items())}


def new_transition_book(metrics: Sequence[str]) -> dict[str, Any]:
    return {
        metric: {
            cell: {"count": 0, "identity_digest": hashlib.sha256(), "expanded": []}
            for cell in TRANSITION_CELLS
        }
        for metric in metrics
    }


def metric_applicable(
    metric: str,
    identity: Mapping[str, Any],
    flags: Mapping[str, bool],
) -> bool:
    if metric.startswith("context34_"):
        return int(identity["context"]) == 34
    if metric == "top1_correct":
        return bool(flags["nonempty"])
    return True


def update_transition_book(
    book: Mapping[str, Any],
    identity: Mapping[str, Any],
    raw_flags: Mapping[str, bool],
    candidate_flags: Mapping[str, bool],
    raw_order: Sequence[int],
    candidate_order: Sequence[int],
) -> None:
    for metric, cells in book.items():
        if not metric_applicable(metric, identity, raw_flags):
            if metric_applicable(metric, identity, candidate_flags):
                raise RuntimeError(f"transition applicability drift for {metric}")
            continue
        raw_key = metric.removeprefix("context34_")
        candidate_key = raw_key
        raw_correct = bool(raw_flags[raw_key])
        candidate_correct = bool(candidate_flags[candidate_key])
        cell = ("c" if raw_correct else "w") + ("c" if candidate_correct else "w")
        record = cells[cell]
        record["count"] += 1
        record["identity_digest"].update(canonical_json(transition_identity(identity)))
        if cell in {"cw", "wc"}:
            record["expanded"].append(
                {
                    **transition_identity(identity),
                    "raw_order": [int(value) for value in raw_order],
                    "candidate_order": [int(value) for value in candidate_order],
                }
            )


def finalize_transition_book(
    book: Mapping[str, Any],
    raw_correct_counts: Mapping[str, int],
    candidate_correct_counts: Mapping[str, int],
    denominators: Mapping[str, int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric, cells in book.items():
        serialized: dict[str, Any] = {}
        for cell in TRANSITION_CELLS:
            record = cells[cell]
            item = {
                "count": int(record["count"]),
                "canonical_identity_stream_sha256": record[
                    "identity_digest"
                ].hexdigest(),
            }
            if cell in {"cw", "wc"}:
                item["expanded_full_line_identities_and_orders"] = record["expanded"]
                item["expanded_count"] = len(record["expanded"])
                if item["expanded_count"] != item["count"]:
                    raise RuntimeError(f"{metric}/{cell} expansion count drift")
            serialized[cell] = item
        denominator = sum(serialized[cell]["count"] for cell in TRANSITION_CELLS)
        if denominator != int(denominators[metric]):
            raise RuntimeError(f"{metric} transition denominator drift: {denominator}")
        net = serialized["wc"]["count"] - serialized["cw"]["count"]
        observed_net = int(candidate_correct_counts[metric]) - int(
            raw_correct_counts[metric]
        )
        if net != observed_net:
            raise RuntimeError(f"{metric} WC-CW net identity failed")
        result[metric] = {
            "cells": serialized,
            "applicable_denominator": denominator,
            "four_cells_close_denominator": True,
            "candidate_correct_minus_raw_correct": observed_net,
            "wc_minus_cw": net,
            "net_identity_exact": True,
        }
    return result


def line_flags(
    cpu_batch: Mapping[str, Any],
    outputs_cpu: Mapping[str, Any],
    actions: Sequence[Sequence[int]],
    row_index: int,
) -> tuple[dict[str, bool], int, bool, list[int]]:
    targets = cpu_batch["targets"][row_index].bool()
    option_mask = cpu_batch["option_mask"][row_index].bool()
    prediction = targets.new_zeros(targets.shape)
    order = [int(value) for value in actions[row_index]]
    if order:
        prediction[order] = True
    set_exact = bool(((prediction == targets) | ~option_mask).all())
    expert_count = int(cpu_batch["expert_ordered_action_counts"][row_index])
    expert_order = cpu_batch["expert_ordered_actions"][
        row_index, :expert_count
    ].tolist()
    context = int(cpu_batch["contexts"][row_index])
    hybrid_order = order if context == 34 else sorted(order)
    top1_index = int(outputs_cpu["policy_logits"][row_index].argmax())
    value_sign = bool(outputs_cpu["value_logits"][row_index] >= 0)
    value_target = bool(cpu_batch["win_targets"][row_index])
    flags = {
        "set_exact": set_exact,
        "ordered_exact": order == expert_order,
        "hybrid_order_exact": hybrid_order == expert_order,
        "count_correct": len(order) == int(cpu_batch["action_counts"][row_index]),
        "nonempty": int(cpu_batch["action_counts"][row_index]) > 0,
        "top1_correct": bool(targets[top1_index]),
        "value_correct": value_sign == value_target,
    }
    return flags, top1_index, value_sign, expert_order


def metric_correct_counts(summary: Mapping[str, Any]) -> dict[str, int]:
    context34 = summary.get("by_context", {}).get("34")
    if not isinstance(context34, dict):
        raise RuntimeError("official summary lacks context 34")
    return {
        "set_exact": int(summary["set_exact_correct"]),
        "hybrid_order_exact": int(summary["hybrid_order_exact_correct"]),
        "ordered_exact": int(summary["ordered_exact_correct"]),
        "top1_correct": int(summary["top1_correct"]),
        "context34_hybrid_order_exact": int(context34["hybrid_order_exact_correct"]),
        "context34_ordered_exact": int(context34["ordered_exact_correct"]),
    }


def typed_leaf_map(value: Any, prefix: str = "") -> dict[str, tuple[str, Any]]:
    if not isinstance(value, dict):
        return {prefix: (type(value).__name__, value)}
    result: dict[str, tuple[str, Any]] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(typed_leaf_map(item, path))
    return result


def assert_frozen_summary_exact(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    panel: str,
    model_name: str,
) -> dict[str, Any]:
    observed_leaves = typed_leaf_map(observed)
    expected_leaves = typed_leaf_map(expected)
    if len(observed_leaves) != 572 or observed_leaves != expected_leaves:
        differing = sorted(
            key
            for key in set(observed_leaves) | set(expected_leaves)
            if observed_leaves.get(key) != expected_leaves.get(key)
        )
        raise RuntimeError(
            f"{model_name}/{panel} frozen 572-leaf summary drift: {differing[:20]}"
        )
    digest = sha256_bytes(canonical_json(observed))
    if digest != FROZEN_METRICS_CANONICAL_SHA256[panel]:
        raise RuntimeError(f"{model_name}/{panel} frozen metrics SHA drift")
    return {
        "typed_scalar_leaf_count": 572,
        "full_nested_metrics_typed_leaf_exact": True,
        "metrics_canonical_sha256": digest,
    }


def synthetic_self_test() -> dict[str, Any]:
    identities = [
        {
            "panel": "synthetic",
            "archive": "synthetic.zip",
            "member": "train/x.jsonl",
            "line_index_zero_based": index,
            "line_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "context": 34 if index == 4 else 0,
        }
        for index in range(5)
    ]
    patterns = [(True, False), (True, True), (True, False), (False, True), (False, False)]
    book = new_transition_book(("top1_correct",))
    for index, (raw_value, candidate_value) in enumerate(patterns):
        raw = {"top1_correct": raw_value, "nonempty": index != 0}
        candidate = {"top1_correct": candidate_value, "nonempty": index != 0}
        update_transition_book(book, identities[index], raw, candidate, [0], [1])
    finalized = finalize_transition_book(
        book,
        {"top1_correct": 2},
        {"top1_correct": 2},
        {"top1_correct": 4},
    )
    cells = finalized["top1_correct"]["cells"]
    if [cells[name]["count"] for name in TRANSITION_CELLS] != [1, 1, 1, 1]:
        raise RuntimeError("synthetic transition four-cell coverage failed")
    if "expanded_full_line_identities_and_orders" in cells["cc"] or len(
        cells["cw"]["expanded_full_line_identities_and_orders"]
    ) != 1:
        raise RuntimeError("synthetic CW/WC-only expansion failed")
    digest = hashlib.sha256()
    packed = struct.pack("<ff", 1.0, 2.0)
    if packed != b"\x00\x00\x80?\x00\x00\x00@":
        raise RuntimeError("synthetic little-endian float32 fixture failed")
    update_float32_fingerprint_bytes(digest, identities[0], [2], packed, option_indices=[0, 2])
    nonempty_sha = digest.hexdigest()
    empty = hashlib.sha256()
    update_float32_fingerprint_bytes(empty, identities[0], [0], b"", option_indices=[])
    if empty.hexdigest() == nonempty_sha:
        raise RuntimeError("synthetic empty fingerprint delimiter failed")
    tied = max(range(3), key=lambda index: (1.0, -index))
    if tied != 0:
        raise RuntimeError("synthetic lowest-index argmax tie rule failed")
    zero_raw = [Fraction(1), Fraction(1)]
    zero_alpha1 = [Fraction(0), Fraction(2)]
    zero_slopes = [
        alpha1 - raw for raw, alpha1 in zip(zero_raw, zero_alpha1)
    ]
    zero_root = (zero_raw[1] - zero_raw[0]) / (
        zero_slopes[0] - zero_slopes[1]
    )
    if (
        zero_root != 0
        or secant_order(zero_raw, zero_slopes, [0, 1], Fraction(0), 1) != [0]
        or secant_order(zero_raw, zero_slopes, [0, 1], Fraction(1), 1) != [1]
    ):
        raise RuntimeError("synthetic alpha-zero crossing/open interval failed")
    dyadic_raw = [
        Fraction.from_float(19.765625),
        Fraction.from_float(12.0078125),
    ]
    dyadic_alpha1 = [
        Fraction.from_float(13.21875),
        Fraction.from_float(26.765625),
    ]
    dyadic_slopes = [
        alpha1 - raw for raw, alpha1 in zip(dyadic_raw, dyadic_alpha1)
    ]
    dyadic_root = (dyadic_raw[1] - dyadic_raw[0]) / (
        dyadic_slopes[0] - dyadic_slopes[1]
    )
    dyadic_raw_float = [19.765625, 12.0078125]
    dyadic_alpha1_float = [13.21875, 26.765625]
    dyadic_slope_float = [
        alpha1 - raw
        for raw, alpha1 in zip(dyadic_raw_float, dyadic_alpha1_float)
    ]
    dyadic_root_float = (
        dyadic_raw_float[1] - dyadic_raw_float[0]
    ) / (dyadic_slope_float[0] - dyadic_slope_float[1])
    dyadic_nextafter_float = math.nextafter(dyadic_root_float, math.inf)
    float_tie_margin = (
        (dyadic_raw_float[0] + dyadic_root_float * dyadic_slope_float[0])
        - (dyadic_raw_float[1] + dyadic_root_float * dyadic_slope_float[1])
    )
    float_nextafter_margin = (
        (
            dyadic_raw_float[0]
            + dyadic_nextafter_float * dyadic_slope_float[0]
        )
        - (
            dyadic_raw_float[1]
            + dyadic_nextafter_float * dyadic_slope_float[1]
        )
    )
    dyadic_open = dyadic_root + max(Fraction(1), dyadic_root)
    dyadic_tie_margin = (
        dyadic_raw[0]
        + dyadic_root * dyadic_slopes[0]
        - dyadic_raw[1]
        - dyadic_root * dyadic_slopes[1]
    )
    dyadic_open_margin = (
        dyadic_raw[0]
        + dyadic_open * dyadic_slopes[0]
        - dyadic_raw[1]
        - dyadic_open * dyadic_slopes[1]
    )
    if (
        float_tie_margin != 0.0
        or float_nextafter_margin != 0.0
        or dyadic_tie_margin != 0
        or dyadic_open_margin == 0
    ):
        raise RuntimeError("synthetic BF16 dyadic open-interval exactness failed")
    padded_raw = [Fraction(-40), Fraction(-30)]
    padded_alpha1 = [Fraction(-20), Fraction(-30)]
    padded_slopes = [
        alpha1 - raw for raw, alpha1 in zip(padded_raw, padded_alpha1)
    ]
    if (
        secant_order(padded_raw, padded_slopes, [0], Fraction(0), 1) != [0]
        or secant_order(padded_raw, padded_slopes, [0, 1], Fraction(0), 1)
        != [1]
        or secant_order(padded_raw, padded_slopes, [0, 1], Fraction(1), 1)
        != [0]
    ):
        raise RuntimeError("synthetic allowed-action/full-vector-top1 separation failed")
    return {
        "status": "zero_input_synthetic_contract_self_test_passed",
        "top1_empty_row_excluded": True,
        "four_transition_cells_exercised": True,
        "four_cells_close_denominator": True,
        "wc_minus_cw_net_identity": True,
        "cw_wc_only_full_expansion": True,
        "float32_little_endian_fixture": True,
        "empty_slice_shape_indices_zero_bytes_rule": True,
        "argmax_tie_lowest_option_index": True,
        "alpha_zero_crossing_open_interval_detected": True,
        "bf16_dyadic_fraction_open_interval_non_tie": True,
        "bf16_dyadic_float_nextafter_remains_tied_fixture": True,
        "allowed_action_and_full_vector_top1_scopes_separated": True,
        "nonempty_fixture_fingerprint": nonempty_sha,
        "empty_fixture_fingerprint": empty.hexdigest(),
    }


def identity_collate(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    helper: ModuleType,
    model_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise RuntimeError("identity collate received an empty batch")
    features = [item[0] for item in rows]
    identities = [item[1] for item in rows]
    batch = helper.evaluator.collate_ordered(
        features,
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    if int(batch["contexts"].shape[0]) != len(identities):
        raise RuntimeError("identity/features collate row-count drift")
    for row_index, identity in enumerate(identities):
        expert_count = int(batch["expert_ordered_action_counts"][row_index])
        expert_order = batch["expert_ordered_actions"][
            row_index, :expert_count
        ].tolist()
        if (
            int(batch["contexts"][row_index]) != int(identity["context"])
            or expert_order != identity["expert_order"]
            or int(batch["option_mask"][row_index].sum())
            != int(identity["option_count"])
            or int(batch["action_counts"][row_index]) != len(identity["expert_order"])
        ):
            raise RuntimeError("identity/features synchronization drift")
    return batch, identities


def build_identity_loader(
    helper: ModuleType,
    design: ModuleType,
    archive_payload: bytes,
    panel: str,
    model_config: Mapping[str, Any],
) -> tuple[Any, Any]:
    torch = helper.torch

    class IdentityStrictTrainDataset(torch.utils.data.IterableDataset):
        def __init__(self) -> None:
            super().__init__()
            self.inventory_train_members: list[str] = []
            self.opened_members: list[str] = []
            self.raw_rows = 0
            self.emitted_rows = 0
            self.context34_rows = 0
            self.nonempty_rows = 0
            self.validation_member_payloads_opened = False
            self.identity_digest = hashlib.sha256()

        def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
            if helper.get_worker_info() is not None:
                raise RuntimeError("formal identity dataset requires workers=0")
            with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
                members = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("train/") and name.endswith(".jsonl")
                )
                if len(members) != int(design.EXPECTED_TRAIN[panel]["members"]):
                    raise RuntimeError(f"{panel} train member inventory drift")
                self.inventory_train_members = members
                for member in members:
                    if not member.startswith("train/") or not member.endswith(".jsonl"):
                        self.validation_member_payloads_opened = True
                        raise RuntimeError(f"refusing non-train member {member}")
                    self.opened_members.append(member)
                    with archive.open(member) as handle:
                        for line_index, raw_line in enumerate(handle):
                            self.raw_rows += 1
                            try:
                                row = helper.orjson.loads(raw_line)
                            except Exception as error:
                                raise RuntimeError(
                                    f"invalid JSON in {member}:{line_index}"
                                ) from error
                            if not isinstance(row, dict) or row.get("split") != "train":
                                raise RuntimeError(
                                    f"non-train row in {member}:{line_index}"
                                )
                            raw_action = row.get("action")
                            if not isinstance(raw_action, list):
                                raise RuntimeError(
                                    f"invalid action in {member}:{line_index}"
                                )
                            try:
                                expert_order = [int(value) for value in raw_action]
                            except (TypeError, ValueError) as error:
                                raise RuntimeError(
                                    f"invalid action index in {member}:{line_index}"
                                ) from error
                            previous_max = helper.bc.MAX_ACTION_COUNT
                            helper.bc.MAX_ACTION_COUNT = helper.ppo.MAX_ACTION_COUNT
                            try:
                                features = helper.bc.featurize_row(
                                    row,
                                    int(model_config["hash_size"]),
                                    int(model_config["max_state_entities"]),
                                )
                            finally:
                                helper.bc.MAX_ACTION_COUNT = previous_max
                            options = (
                                ((row.get("observation") or {}).get("select") or {}).get(
                                    "option"
                                )
                                or []
                            )
                            if (
                                features is None
                                or len(expert_order) > helper.ppo.MAX_ACTION_COUNT
                                or len(set(expert_order)) != len(expert_order)
                                or any(
                                    value < 0 or value >= len(options)
                                    for value in expert_order
                                )
                            ):
                                raise RuntimeError(
                                    f"non-evaluable train row in {member}:{line_index}"
                                )
                            features["expert_action_order"] = expert_order
                            context = int(features["context"])
                            identity = {
                                "panel": panel,
                                "archive": str(design.DATASETS[panel].relative_to(ROOT)),
                                "member": member,
                                "line_index_zero_based": line_index,
                                "line_sha256": sha256_bytes(raw_line),
                                "context": context,
                                "expert_order": expert_order,
                                "option_count": len(options),
                            }
                            cache_identity = {
                                "panel": panel,
                                "member": member,
                                "line_index_zero_based": line_index,
                                "line_sha256": identity["line_sha256"],
                            }
                            self.identity_digest.update(canonical_json(cache_identity))
                            self.emitted_rows += 1
                            self.context34_rows += int(context == 34)
                            self.nonempty_rows += int(bool(expert_order))
                            yield features, identity

    dataset = IdentityStrictTrainDataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=partial(
            identity_collate,
            helper=helper,
            model_config=model_config,
        ),
        pin_memory=True,
        persistent_workers=False,
    )
    return loader, dataset


def finalize_dataset_audit(dataset: Any, design: ModuleType, panel: str) -> dict[str, Any]:
    expected = design.EXPECTED_TRAIN[panel]
    if (
        dataset.validation_member_payloads_opened
        or dataset.opened_members != dataset.inventory_train_members
        or len(dataset.opened_members) != int(expected["members"])
        or dataset.raw_rows != int(expected["rows"])
        or dataset.emitted_rows != int(expected["rows"])
        or dataset.context34_rows != int(expected["context34_rows"])
        or dataset.nonempty_rows != int(expected["nonempty_rows"])
        or dataset.identity_digest.hexdigest()
        != design.EXPECTED_TRAIN_IDENTITY_SHA256[panel]
    ):
        raise RuntimeError(f"{panel} identity dataset structural audit failed")
    return {
        "rows": dataset.emitted_rows,
        "nonempty_rows": dataset.nonempty_rows,
        "context34_rows": dataset.context34_rows,
        "train_members": dataset.opened_members,
        "workers": 0,
        "identity_stream_sha256": dataset.identity_digest.hexdigest(),
        "validation_member_payloads_opened": False,
        "identity_feature_synchronization_exact": True,
    }


def all_state_tensors_finite(state: Mapping[str, Any], torch: Any) -> bool:
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"non-tensor model state entry {name}")
        if not bool(torch.isfinite(tensor).all()):
            return False
    return True


def displacement_summary(
    raw_state: Mapping[str, Any],
    state: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    squared = torch.zeros((), dtype=torch.float64, device="cpu")
    maximum = 0.0
    for name in ACTOR6_NAMES:
        delta = torch.sub(
            state[name].detach().to(device="cpu", dtype=torch.float32),
            raw_state[name].detach().to(device="cpu", dtype=torch.float32),
        )
        squared = squared + delta.to(dtype=torch.float64).square().sum()
        maximum = max(maximum, float(delta.abs().max()))
    return {"actor6_l2": float(squared.sqrt()), "actor6_max_abs": maximum}


def build_ray_states(
    helper: ModuleType,
    design: ModuleType,
    raw_payload: bytes,
    endpoint_payload: bytes,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    torch = helper.torch
    raw_checkpoint = helper.checkpoint_from_bytes(raw_payload, "raw U468")
    endpoint_checkpoint = helper.checkpoint_from_bytes(
        endpoint_payload, "serialized alpha1 equalblend endpoint"
    )
    if (
        raw_checkpoint.get("update") != 468
        or helper.evaluator.checkpoint_kind(raw_checkpoint) != "ppo"
        or not isinstance(raw_checkpoint.get("model_config"), dict)
    ):
        raise RuntimeError("raw checkpoint inference binding drift")
    if (
        helper.evaluator.checkpoint_kind(endpoint_checkpoint) != "ppo"
        or endpoint_checkpoint.get("model_config") != raw_checkpoint.get("model_config")
        or endpoint_checkpoint.get("feature_version")
        != raw_checkpoint.get("feature_version")
        or endpoint_checkpoint.get("bc_feature_version")
        != raw_checkpoint.get("bc_feature_version")
    ):
        raise RuntimeError("alpha1 endpoint inference schema drift")
    raw_state = raw_checkpoint["model_state_dict"]
    endpoint_state = endpoint_checkpoint["model_state_dict"]
    raw_hash = helper.model_state_sha256(raw_state)
    endpoint_hash = helper.model_state_sha256(endpoint_state)
    if raw_hash != design.PARENT_MODEL_STATE_SHA256:
        raise RuntimeError("raw model-state SHA drift")
    if endpoint_hash != design.ENDPOINT_MODEL_STATE_SHA256:
        raise RuntimeError("alpha1 endpoint model-state SHA drift")
    endpoint_actor_audit = helper.verify_actor6_only(
        raw_state, endpoint_state, "alpha1 endpoint"
    )
    reconstructed_alpha1: dict[str, Any] = {}
    for name in sorted(raw_state):
        raw_tensor = raw_state[name].detach().cpu()
        endpoint_tensor = endpoint_state[name].detach().cpu()
        if name in ACTOR6_NAMES:
            if raw_tensor.dtype != torch.float32 or endpoint_tensor.dtype != torch.float32:
                raise RuntimeError(f"actor ray requires CPU FP32 tensor {name}")
            displacement = torch.sub(endpoint_tensor, raw_tensor)
            scaled = torch.mul(displacement, float(1))
            candidate = torch.add(raw_tensor, scaled)
            reconstructed_alpha1[name] = candidate
        else:
            reconstructed_alpha1[name] = raw_tensor.clone()
    mismatched_alpha1 = sorted(
        name
        for name in endpoint_state
        if not torch.equal(reconstructed_alpha1[name], endpoint_state[name].detach().cpu())
    )
    if mismatched_alpha1:
        raise RuntimeError(f"alpha1 explicit three-op reconstruction drift: {mismatched_alpha1}")

    states: dict[str, dict[str, Any]] = {
        "raw": {name: tensor.detach().cpu().clone() for name, tensor in raw_state.items()},
        "alpha1": {
            name: tensor.detach().cpu().clone() for name, tensor in endpoint_state.items()
        },
    }
    for alpha in (2, 4, 8, 16, 32, 64, 128, 256):
        state: dict[str, Any] = {}
        for name in sorted(raw_state):
            raw_tensor = raw_state[name].detach().cpu()
            if name in ACTOR6_NAMES:
                endpoint_tensor = endpoint_state[name].detach().cpu()
                displacement = torch.sub(endpoint_tensor, raw_tensor)
                scaled = torch.mul(displacement, float(alpha))
                state[name] = torch.add(raw_tensor, scaled)
            else:
                state[name] = raw_tensor.clone()
        states[f"alpha{alpha}"] = state

    audits: dict[str, Any] = {}
    for model_name in MODEL_ORDER:
        state = states[model_name]
        if set(state) != set(raw_state) or not all_state_tensors_finite(state, torch):
            raise RuntimeError(f"{model_name} state schema/nonfinite failure")
        if model_name == "raw":
            actor_audit = {
                "changed_parameter_names": [],
                "changed_parameter_count": 0,
                "exact_raw_anchor": True,
            }
        else:
            actor_audit = helper.verify_actor6_only(raw_state, state, model_name)
        nonactor_mismatch = [
            name
            for name in raw_state
            if name not in ACTOR6_NAMES
            and not torch.equal(raw_state[name].detach().cpu(), state[name])
        ]
        if nonactor_mismatch:
            raise RuntimeError(f"{model_name} changed nonactor tensors")
        audits[model_name] = {
            "alpha": ALPHAS[model_name],
            "model_state_sha256": helper.model_state_sha256(state),
            "all_tensors_finite": True,
            "actor_scope": actor_audit,
            "nonactor_tensor_count": len(state) - len(ACTOR6_NAMES),
            "all_nonactor_tensors_bit_exact_raw": True,
            "constructed_directly_from_raw_and_serialized_alpha1": (
                model_name not in {"raw", "alpha1"}
            ),
            "explicit_CPU_FP32_operations": (
                ["D=E-R", "S=D*alpha", "C=R+S"]
                if model_name != "raw"
                else []
            ),
            "displacement": displacement_summary(raw_state, state, torch),
            "RAM_only": True,
        }
    if audits["raw"]["model_state_sha256"] != design.PARENT_MODEL_STATE_SHA256:
        raise RuntimeError("raw cloned state hash drift")
    if audits["alpha1"]["model_state_sha256"] != design.ENDPOINT_MODEL_STATE_SHA256:
        raise RuntimeError("alpha1 cloned state hash drift")
    audits["alpha1"]["explicit_three_op_reconstruction_tensor_exact_endpoint"] = True
    audits["alpha1"]["serialized_endpoint_actor_audit"] = endpoint_actor_audit
    alpha1_displacement = audits["alpha1"]["displacement"]
    if (
        not math.isclose(
            float(alpha1_displacement["actor6_l2"]),
            float(design.ALPHA1_DISPLACEMENT_L2),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or float(alpha1_displacement["actor6_max_abs"])
        != float(design.ALPHA1_DISPLACEMENT_MAX_ABS)
    ):
        raise RuntimeError("alpha1 displacement magnitude binding drift")
    audits["alpha1"]["frozen_displacement_magnitudes_exact_or_tolerance_bound"] = {
        "expected_l2": design.ALPHA1_DISPLACEMENT_L2,
        "l2_absolute_tolerance": 1e-12,
        "expected_max_abs": design.ALPHA1_DISPLACEMENT_MAX_ABS,
        "pass": True,
    }
    return states, audits, raw_checkpoint


def copy_actor_state_to_model(
    model: Any,
    state: Mapping[str, Any],
    helper: ModuleType,
) -> None:
    live = model.state_dict()
    if not set(ACTOR6_NAMES).issubset(live):
        raise RuntimeError("live model lacks actor6 state keys")
    with helper.torch.no_grad():
        for name in ACTOR6_NAMES:
            destination = live[name]
            source = state[name]
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise RuntimeError(f"live actor tensor schema drift at {name}")
            destination.copy_(source.to(device=destination.device))


def verify_live_actor_state(
    model: Any,
    state: Mapping[str, Any],
    helper: ModuleType,
) -> None:
    live = model.state_dict()
    for name in ACTOR6_NAMES:
        if not helper.torch.equal(live[name].detach().cpu(), state[name]):
            raise RuntimeError(f"live actor copy mismatch at {name}")


def metric_flags_from_order(
    order: Sequence[int],
    expert_order: Sequence[int],
    context: int,
    top1_index: int,
) -> dict[str, bool]:
    predicted = [int(value) for value in order]
    expert = [int(value) for value in expert_order]
    top1_correct = int(top1_index) in set(expert)
    return {
        "set_exact": set(predicted) == set(expert) and len(predicted) == len(expert),
        "ordered_exact": predicted == expert,
        "hybrid_order_exact": (
            predicted == expert if context == 34 else sorted(predicted) == expert
        ),
        "top1_correct": top1_correct,
        "context34_ordered_exact": context == 34 and predicted == expert,
        "context34_hybrid_order_exact": context == 34 and predicted == expert,
    }


def secant_order(
    raw_values: Sequence[Fraction],
    slopes: Sequence[Fraction],
    ranked_indices: Sequence[int],
    alpha: Fraction,
    count: int,
) -> list[int]:
    ranked = sorted(
        (int(index) for index in ranked_indices),
        key=lambda index: (
            -(raw_values[index] + alpha * slopes[index]),
            index,
        ),
    )
    return ranked[:count]


def new_secant_advisory() -> dict[str, Any]:
    return {
        "pokemonfan_wrong_to_correct": {metric: None for metric in MAIN_METRICS},
        "correct_to_wrong": {
            panel: {metric: None for metric in SAFETY_METRICS}
            for panel in PANEL_ORDER
        },
        "rows_examined": {panel: 0 for panel in PANEL_ORDER},
        "finite_positive_pair_crossings_examined": {panel: 0 for panel in PANEL_ORDER},
        "distinct_exact_rational_crossing_groups_examined": {
            panel: 0 for panel in PANEL_ORDER
        },
        "zero_boundary_pair_crossings_examined": {
            panel: 0 for panel in PANEL_ORDER
        },
    }


def maybe_keep_earliest(container: dict[str, Any], key: str, record: dict[str, Any]) -> None:
    current = container[key]
    record_root = Fraction(
        int(record["secant_root_numerator"]),
        int(record["secant_root_denominator"]),
    )
    current_root = (
        Fraction(
            int(current["secant_root_numerator"]),
            int(current["secant_root_denominator"]),
        )
        if current is not None
        else None
    )
    if current is None or (
        record_root,
        canonical_fragment(record["identity"]),
        record["limiting_option_pair"],
    ) < (
        current_root,
        canonical_fragment(current["identity"]),
        current["limiting_option_pair"],
    ):
        container[key] = record


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
        "float": float(value),
    }


def secant_pair_record(
    raw_values: Sequence[Fraction],
    alpha1_values: Sequence[Fraction],
    allowed_indices: Sequence[int],
    left: int,
    right: int,
) -> dict[str, Any]:
    raw_margin = raw_values[left] - raw_values[right]
    alpha1_margin = alpha1_values[left] - alpha1_values[right]
    slope = alpha1_margin - raw_margin
    return {
        "option_pair": [int(left), int(right)],
        "option_pair_allowed_mask": [
            int(left) in allowed_indices,
            int(right) in allowed_indices,
        ],
        "raw_pair_margin": float(raw_margin),
        "raw_pair_margin_fraction": fraction_record(raw_margin),
        "alpha1_pair_margin": float(alpha1_margin),
        "alpha1_pair_margin_fraction": fraction_record(alpha1_margin),
        "pair_margin_slope": float(slope),
        "pair_margin_slope_fraction": fraction_record(slope),
    }


def update_secant_advisory_for_row(
    advisory: dict[str, Any],
    panel: str,
    identity: Mapping[str, Any],
    raw_policy_row: Any,
    alpha1_policy_row: Any,
    option_mask_row: Any,
    predicted_count: int,
    expert_order: Sequence[int],
    raw_flags: Mapping[str, bool],
    raw_order: Sequence[int],
    raw_top1_index: int,
) -> None:
    allowed = [
        index
        for index, value in enumerate(option_mask_row.tolist())
        if bool(value)
    ]
    advisory["rows_examined"][panel] += 1
    if not allowed:
        raise RuntimeError("formal evaluator row has no allowed option")
    if tuple(raw_policy_row.shape) != tuple(alpha1_policy_row.shape) or len(
        raw_policy_row.shape
    ) != 1:
        raise RuntimeError("secant raw/alpha1 full policy vector shape drift")
    full_indices = list(range(int(raw_policy_row.shape[0])))
    if len(option_mask_row) != len(full_indices):
        raise RuntimeError("secant option mask/full policy vector length drift")
    raw_float_values = [float(raw_policy_row[index]) for index in full_indices]
    alpha1_float_values = [
        float(alpha1_policy_row[index]) for index in full_indices
    ]
    if not all(
        math.isfinite(value) for value in raw_float_values + alpha1_float_values
    ):
        raise RuntimeError("secant full policy logits must all be finite")
    raw_values = [Fraction.from_float(value) for value in raw_float_values]
    alpha1_values = [Fraction.from_float(value) for value in alpha1_float_values]
    slopes = [alpha1 - raw for raw, alpha1 in zip(raw_values, alpha1_values)]
    root_groups: dict[Fraction, list[tuple[int, int]]] = {}
    for left in full_indices:
        for right in range(left + 1, len(full_indices)):
            denominator = slopes[left] - slopes[right]
            if denominator == 0:
                continue
            root = (raw_values[right] - raw_values[left]) / denominator
            if root >= 0:
                root_groups.setdefault(root, []).append((left, right))
    grouped_roots = sorted(root_groups.items(), key=lambda item: item[0])
    advisory["finite_positive_pair_crossings_examined"][panel] += sum(
        len(pairs) for root, pairs in grouped_roots if root > 0
    )
    advisory["zero_boundary_pair_crossings_examined"][panel] += len(
        root_groups.get(Fraction(0), [])
    )
    advisory["distinct_exact_rational_crossing_groups_examined"][panel] += len(
        grouped_roots
    )
    context = int(identity["context"])
    nonempty = bool(expert_order)
    raw_expanded = {
        **{metric: bool(raw_flags[metric]) for metric in MAIN_METRICS},
        "context34_hybrid_order_exact": (
            bool(raw_flags["hybrid_order_exact"]) if context == 34 else False
        ),
        "context34_ordered_exact": (
            bool(raw_flags["ordered_exact"]) if context == 34 else False
        ),
    }
    alpha0_allowed_rank = secant_order(
        raw_values, slopes, allowed, Fraction(0), len(allowed)
    )
    alpha0_selected = alpha0_allowed_rank[: int(predicted_count)]
    alpha0_full_rank = secant_order(
        raw_values, slopes, full_indices, Fraction(0), len(full_indices)
    )
    if (
        alpha0_selected != [int(value) for value in raw_order]
        or not alpha0_full_rank
        or alpha0_full_rank[0] != int(raw_top1_index)
    ):
        raise RuntimeError("secant alpha0 ranking differs from official raw greedy output")
    for group_index, (root, same_root_pairs) in enumerate(grouped_roots):
        same_root_pairs = sorted(same_root_pairs)
        if group_index + 1 < len(grouped_roots):
            next_root = grouped_roots[group_index + 1][0]
            open_interval_alpha = (root + next_root) / 2
            open_interval_kind = "midpoint_to_next_distinct_crossing"
        else:
            next_root = None
            open_interval_alpha = root + max(Fraction(1), root)
            open_interval_kind = "deterministic_point_beyond_final_crossing"
        if not root < open_interval_alpha or (
            next_root is not None and not open_interval_alpha < next_root
        ):
            raise RuntimeError("exact secant open-interval sample construction failed")
        pair_records = [
            secant_pair_record(
                raw_values,
                alpha1_values,
                allowed,
                left,
                right,
            )
            for left, right in same_root_pairs
        ]
        limiting_pair = pair_records[0]
        for side, alpha in (
            ("at_tie_lowest_index", root),
            (open_interval_kind, open_interval_alpha),
        ):
            allowed_rank = secant_order(
                raw_values, slopes, allowed, alpha, len(allowed)
            )
            full_rank = secant_order(
                raw_values, slopes, full_indices, alpha, len(full_indices)
            )
            order = allowed_rank[: int(predicted_count)]
            estimated = metric_flags_from_order(
                order, expert_order, context, full_rank[0]
            )
            record_base = {
                "secant_estimated_alpha": float(root),
                "secant_root_numerator": int(root.numerator),
                "secant_root_denominator": int(root.denominator),
                "event_side": side,
                "evaluation_alpha": float(alpha),
                "evaluation_alpha_fraction": fraction_record(alpha),
                "next_distinct_crossing_fraction": (
                    fraction_record(next_root) if next_root is not None else None
                ),
                "identity": fingerprint_identity(identity),
                "context": context,
                "limiting_option_pair": limiting_pair["option_pair"],
                "raw_pair_margin": limiting_pair["raw_pair_margin"],
                "raw_pair_margin_fraction": limiting_pair[
                    "raw_pair_margin_fraction"
                ],
                "alpha1_pair_margin": limiting_pair["alpha1_pair_margin"],
                "alpha1_pair_margin_fraction": limiting_pair[
                    "alpha1_pair_margin_fraction"
                ],
                "pair_margin_slope": limiting_pair["pair_margin_slope"],
                "pair_margin_slope_fraction": limiting_pair[
                    "pair_margin_slope_fraction"
                ],
                "same_root_pairs": pair_records,
                "same_root_pair_count": len(pair_records),
                "raw_order": [int(value) for value in raw_order],
                "raw_top1_index": int(raw_top1_index),
                "estimated_top1_index": int(full_rank[0]),
                "estimated_order": order,
                "estimate_only_not_gate": True,
            }
            if panel == "pokemonfan":
                for metric in MAIN_METRICS:
                    if metric == "top1_correct" and not nonempty:
                        continue
                    if not raw_expanded[metric] and estimated[metric]:
                        maybe_keep_earliest(
                            advisory["pokemonfan_wrong_to_correct"],
                            metric,
                            {**record_base, "metric": metric, "direction": "wrong_to_correct"},
                        )
            for metric in SAFETY_METRICS:
                if metric.startswith("context34_") and context != 34:
                    continue
                if metric == "top1_correct" and not nonempty:
                    continue
                if raw_expanded[metric] and not estimated[metric]:
                    maybe_keep_earliest(
                        advisory["correct_to_wrong"][panel],
                        metric,
                        {**record_base, "metric": metric, "direction": "correct_to_wrong"},
                    )


def finalize_secant_advisory(
    advisory: dict[str, Any],
    transition_results: Mapping[str, Any],
) -> dict[str, Any]:
    for metric, value in list(advisory["pokemonfan_wrong_to_correct"].items()):
        if value is None:
            advisory["pokemonfan_wrong_to_correct"][metric] = {
                "secant_estimated_alpha": None,
                "reason": (
                    "no nonnegative exact rational crossing or following open "
                    "interval in the raw-to-alpha1 affine logit secant made any "
                    "raw-wrong PokemonFan row correct"
                ),
            }
    for panel in PANEL_ORDER:
        for metric, value in list(advisory["correct_to_wrong"][panel].items()):
            if value is None:
                advisory["correct_to_wrong"][panel][metric] = {
                    "secant_estimated_alpha": None,
                    "reason": (
                        "no nonnegative exact rational crossing or following open "
                        "interval in the raw-to-alpha1 affine logit secant made an "
                        "applicable raw-correct row wrong"
                    ),
                }
    observed_gain: dict[str, Any] = {}
    observed_regression: dict[str, Any] = {
        panel: {} for panel in PANEL_ORDER
    }
    observed_names = MODEL_ORDER[1:]
    for metric in MAIN_METRICS:
        bracket = None
        previous_alpha = 0
        for name in observed_names:
            transition = transition_results[name]["pokemonfan"][metric]
            wc_count = int(transition["cells"]["wc"]["count"])
            cw_count = int(transition["cells"]["cw"]["count"])
            net = int(transition["candidate_correct_minus_raw_correct"])
            if net >= 1:
                bracket = {
                    "lower_observed_alpha": previous_alpha,
                    "upper_observed_alpha": ALPHAS[name],
                    "upper_wc_count": wc_count,
                    "upper_cw_count": cw_count,
                    "upper_net_correct_gain": net,
                }
                break
            previous_alpha = ALPHAS[name]
        observed_gain[metric] = bracket or {
            "lower_observed_alpha": None,
            "upper_observed_alpha": None,
            "reason": "no net +1 correct gain on the bounded observed grid",
        }
    for panel in PANEL_ORDER:
        for metric in SAFETY_METRICS:
            bracket = None
            previous_alpha = 0
            for name in observed_names:
                count = transition_results[name][panel][metric]["cells"]["cw"][
                    "count"
                ]
                if count > 0:
                    bracket = {
                        "lower_observed_alpha": previous_alpha,
                        "upper_observed_alpha": ALPHAS[name],
                        "upper_cw_count": count,
                    }
                    break
                previous_alpha = ALPHAS[name]
            observed_regression[panel][metric] = bracket or {
                "lower_observed_alpha": None,
                "upper_observed_alpha": None,
                "reason": "no raw-correct to candidate-wrong row on the bounded grid",
            }
    advisory["observed_grid"] = {
        "pokemonfan_earliest_gain_bracket_by_main_metric": observed_gain,
        "earliest_regression_bracket_by_panel_and_safety_metric": observed_regression,
    }
    advisory["limitations"] = {
        "advisory_only": True,
        "participates_in_selection": False,
        "affine_logit_model": "z_hat(alpha)=z_raw+alpha*(z_alpha1-z_raw)",
        "does_not_claim_exact_or_continuous_root": True,
        "bounded_observed_grid_can_miss_flip_and_revert": True,
        "native_BF16_candidate_transitions_are_only_gate_authority": True,
        "empty_competitor_margin": "+Infinity",
        "argmax_tie_break": "lowest option index",
        "secant_arithmetic": (
            "exact Fraction.from_float rational logits, grouped equal roots, "
            "exact tie evaluation and exact open-interval samples"
        ),
        "zero_boundary_crossings_included": True,
        "action_ranking_scope": "allowed option indices only",
        "top1_ranking_scope": "full padded policy vector exactly as official argmax",
        "hybrid_context34": "ordered",
        "hybrid_non_context34": "ascending selected set versus expert order",
    }
    return advisory


def official_forward_and_sample(
    helper: ModuleType,
    model: Any,
    batch: Mapping[str, Any],
    device: Any,
) -> tuple[dict[str, Any], list[list[int]]]:
    outputs = helper.ppo.model_forward(model, batch, device)
    actions, _, _, _ = helper.ppo.sample_ordered_actions(
        outputs,
        batch,
        deterministic=True,
        canonicalize_order=False,
    )
    return outputs, actions


def load_frozen_metric_summaries(
    design: ModuleType,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {"raw": {}, "alpha1": {}}
    evidence: dict[str, Any] = {}
    for model_name, directory in (
        ("raw", "raw_full_u468"),
        ("alpha1", "sole_equalblend_endpoint"),
    ):
        for panel in PANEL_ORDER:
            relative = f"{directory}/{panel}.json"
            path = design.CLOSED_GATE_ROOT / relative
            payload, item = read_regular_bytes(
                path,
                design.CLOSED_EVALUATIONS[relative],
                f"frozen {model_name}/{panel} official evaluation",
                expected_mode=0o444,
            )
            parsed = strict_json_bytes(
                payload, f"frozen {model_name}/{panel} official evaluation"
            )
            metrics = parsed.get("metrics")
            if not isinstance(metrics, dict) or len(typed_leaf_map(metrics)) != 572:
                raise RuntimeError(f"frozen {model_name}/{panel} metrics malformed")
            digest = sha256_bytes(canonical_json(metrics))
            if digest != FROZEN_METRICS_CANONICAL_SHA256[panel]:
                raise RuntimeError(f"frozen {model_name}/{panel} metrics digest drift")
            summaries[model_name][panel] = metrics
            evidence[f"{model_name}/{panel}"] = {
                **item,
                "typed_scalar_leaf_count": 572,
                "metrics_canonical_sha256": digest,
            }
    return summaries, evidence


def transition_denominators(design: ModuleType, panel: str) -> dict[str, int]:
    expected = design.EXPECTED_TRAIN[panel]
    return {
        "set_exact": int(expected["rows"]),
        "hybrid_order_exact": int(expected["rows"]),
        "ordered_exact": int(expected["rows"]),
        "top1_correct": int(expected["nonempty_rows"]),
        "context34_hybrid_order_exact": int(expected["context34_rows"]),
        "context34_ordered_exact": int(expected["context34_rows"]),
    }


def new_change_audit() -> dict[str, Any]:
    return {
        "greedy_sequence": [],
        "greedy_length": [],
        "top1_index": [],
    }


def append_change(
    audit: Mapping[str, list[Any]],
    kind: str,
    identity: Mapping[str, Any],
    raw_order: Sequence[int],
    candidate_order: Sequence[int],
    *,
    raw_top1: int,
    candidate_top1: int,
) -> None:
    item = {
        **transition_identity(identity),
        "raw_order": [int(value) for value in raw_order],
        "candidate_order": [int(value) for value in candidate_order],
    }
    if kind == "top1_index":
        item["raw_top1_index"] = int(raw_top1)
        item["candidate_top1_index"] = int(candidate_top1)
    audit[kind].append(item)


def finalize_change_audit(audit: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    return {
        name: {
            "changed_count": len(records),
            "canonical_changed_identity_stream_sha256": sha256_bytes(
                b"".join(canonical_json(transition_identity(record)) for record in records)
            ),
            "full_changed_identities_and_orders": list(records),
            "untruncated": True,
        }
        for name, records in audit.items()
    }


def state_output_container(helper: ModuleType) -> dict[str, Any]:
    return {
        "accumulator": helper.evaluator.MetricAccumulator(),
        "fingerprints": new_state_fingerprints(),
        "rows": 0,
    }


def evaluate_all_states(
    helper: ModuleType,
    design: ModuleType,
    states: Mapping[str, Mapping[str, Any]],
    raw_checkpoint: Mapping[str, Any],
    archive_payloads: Mapping[str, bytes],
    frozen_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch = helper.torch
    device = torch.device("cuda:0")
    model, model_config, kind = helper.instantiate_checkpoint(
        dict(raw_checkpoint), device
    )
    if kind != "ppo" or helper.model_state_sha256(model.state_dict()) != design.PARENT_MODEL_STATE_SHA256:
        raise RuntimeError("live raw model construction/hash drift")
    outputs = {
        name: {panel: state_output_container(helper) for panel in PANEL_ORDER}
        for name in MODEL_ORDER
    }
    safety_books = {
        name: {panel: new_transition_book(SAFETY_METRICS) for panel in PANEL_ORDER}
        for name in MODEL_ORDER[1:]
    }
    immutable_books = {
        name: {
            panel: new_transition_book(("count_correct", "value_correct"))
            for panel in PANEL_ORDER
        }
        for name in MODEL_ORDER[1:]
    }
    changes = {
        name: {panel: new_change_audit() for panel in PANEL_ORDER}
        for name in MODEL_ORDER[1:]
    }
    immutable_mismatches = {
        name: {
            panel: {"count_logits": [], "value_logits": []}
            for panel in PANEL_ORDER
        }
        for name in MODEL_ORDER[1:]
    }
    secant = new_secant_advisory()
    dataset_audits: dict[str, Any] = {}
    native_dtypes: dict[str, dict[str, str]] = {}
    actor_copy_count = 0
    live_actor_states_verified: set[str] = set()

    model.eval()
    for panel in PANEL_ORDER:
        loader, dataset = build_identity_loader(
            helper,
            design,
            archive_payloads[panel],
            panel,
            model_config,
        )
        native_dtypes[panel] = {}
        for cpu_batch, identities in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in cpu_batch.items()
            }
            raw_snapshot: dict[str, Any] | None = None
            for model_name in MODEL_ORDER:
                copy_actor_state_to_model(model, states[model_name], helper)
                if model_name not in live_actor_states_verified:
                    verify_live_actor_state(model, states[model_name], helper)
                    live_actor_states_verified.add(model_name)
                actor_copy_count += 1
                with torch.inference_mode():
                    state_outputs, actions = official_forward_and_sample(
                        helper, model, batch, device
                    )
                if set(state_outputs) != {
                    "policy_logits",
                    "count_logits",
                    "value_logits",
                }:
                    raise RuntimeError(f"{model_name}/{panel} output key drift")
                dtype_record = {
                    key: str(value.dtype) for key, value in state_outputs.items()
                }
                if any(value.dtype != torch.bfloat16 for value in state_outputs.values()):
                    raise RuntimeError(f"{model_name}/{panel} is not native CUDA BF16")
                prior_dtype = native_dtypes[panel].get(model_name)
                encoded_dtype = canonical_fragment(dtype_record).decode("utf-8")
                if prior_dtype is not None and prior_dtype != encoded_dtype:
                    raise RuntimeError(f"{model_name}/{panel} output dtype drift")
                native_dtypes[panel][model_name] = encoded_dtype
                if len(actions) != len(identities):
                    raise RuntimeError(f"{model_name}/{panel} action row-count drift")
                outputs[model_name][panel]["accumulator"].update(
                    batch, state_outputs, actions, actions
                )
                outputs_cpu = {
                    key: value.detach().float().cpu().contiguous()
                    for key, value in state_outputs.items()
                }
                row_flags: list[dict[str, bool]] = []
                top1_indices: list[int] = []
                value_signs: list[bool] = []
                for row_index, identity in enumerate(identities):
                    flags, top1_index, value_sign, expert_order = line_flags(
                        cpu_batch, outputs_cpu, actions, row_index
                    )
                    if expert_order != identity["expert_order"]:
                        raise RuntimeError("expert identity drift after official forward")
                    row_flags.append(flags)
                    top1_indices.append(top1_index)
                    value_signs.append(value_sign)
                    update_state_fingerprints(
                        outputs[model_name][panel]["fingerprints"],
                        identity,
                        actions[row_index],
                        flags,
                        outputs_cpu["policy_logits"][row_index],
                        outputs_cpu["count_logits"][row_index],
                        outputs_cpu["value_logits"][row_index],
                        cpu_batch["option_mask"][row_index],
                        top1_index,
                        value_sign,
                        torch,
                    )
                outputs[model_name][panel]["rows"] += len(identities)
                if model_name == "raw":
                    raw_snapshot = {
                        "actions": [[int(value) for value in row] for row in actions],
                        "flags": row_flags,
                        "top1_indices": top1_indices,
                        "value_signs": value_signs,
                        "policy_cpu": outputs_cpu["policy_logits"],
                        "count_native": state_outputs["count_logits"].detach().clone(),
                        "value_native": state_outputs["value_logits"].detach().clone(),
                    }
                    continue
                if raw_snapshot is None:
                    raise RuntimeError("candidate evaluated before raw on same batch")
                if (
                    state_outputs["count_logits"].shape
                    != raw_snapshot["count_native"].shape
                    or state_outputs["value_logits"].shape
                    != raw_snapshot["value_native"].shape
                ):
                    raise RuntimeError(f"{model_name}/{panel} immutable output shape drift")
                batch_rows = len(identities)
                count_equal_rows = (
                    (state_outputs["count_logits"] == raw_snapshot["count_native"])
                    .reshape(batch_rows, -1)
                    .all(dim=1)
                    .cpu()
                    .tolist()
                )
                value_equal_rows = (
                    (state_outputs["value_logits"] == raw_snapshot["value_native"])
                    .reshape(batch_rows, -1)
                    .all(dim=1)
                    .cpu()
                    .tolist()
                )
                for row_index, identity in enumerate(identities):
                    raw_order = raw_snapshot["actions"][row_index]
                    candidate_order = actions[row_index]
                    raw_flags = raw_snapshot["flags"][row_index]
                    candidate_flags = row_flags[row_index]
                    update_transition_book(
                        safety_books[model_name][panel],
                        identity,
                        raw_flags,
                        candidate_flags,
                        raw_order,
                        candidate_order,
                    )
                    update_transition_book(
                        immutable_books[model_name][panel],
                        identity,
                        raw_flags,
                        candidate_flags,
                        raw_order,
                        candidate_order,
                    )
                    if candidate_order != raw_order:
                        append_change(
                            changes[model_name][panel],
                            "greedy_sequence",
                            identity,
                            raw_order,
                            candidate_order,
                            raw_top1=raw_snapshot["top1_indices"][row_index],
                            candidate_top1=top1_indices[row_index],
                        )
                    if len(candidate_order) != len(raw_order):
                        append_change(
                            changes[model_name][panel],
                            "greedy_length",
                            identity,
                            raw_order,
                            candidate_order,
                            raw_top1=raw_snapshot["top1_indices"][row_index],
                            candidate_top1=top1_indices[row_index],
                        )
                    if top1_indices[row_index] != raw_snapshot["top1_indices"][row_index]:
                        append_change(
                            changes[model_name][panel],
                            "top1_index",
                            identity,
                            raw_order,
                            candidate_order,
                            raw_top1=raw_snapshot["top1_indices"][row_index],
                            candidate_top1=top1_indices[row_index],
                        )
                    if not bool(count_equal_rows[row_index]):
                        immutable_mismatches[model_name][panel]["count_logits"].append(
                            transition_identity(identity)
                        )
                    if not bool(value_equal_rows[row_index]):
                        immutable_mismatches[model_name][panel]["value_logits"].append(
                            transition_identity(identity)
                        )
                    if model_name == "alpha1":
                        update_secant_advisory_for_row(
                            secant,
                            panel,
                            identity,
                            raw_snapshot["policy_cpu"][row_index],
                            outputs_cpu["policy_logits"][row_index],
                            cpu_batch["option_mask"][row_index],
                            len(raw_order),
                            identity["expert_order"],
                            raw_flags,
                            raw_order,
                            raw_snapshot["top1_indices"][row_index],
                        )
            del raw_snapshot
        dataset_audits[panel] = finalize_dataset_audit(dataset, design, panel)

    panel_results: dict[str, Any] = {name: {} for name in MODEL_ORDER}
    transition_results: dict[str, Any] = {name: {} for name in MODEL_ORDER[1:]}
    completion_ledger: list[dict[str, Any]] = []
    ordinal = 0
    for model_name in MODEL_ORDER:
        for panel in PANEL_ORDER:
            ordinal += 1
            container = outputs[model_name][panel]
            summary = container["accumulator"].summary()
            if (
                container["rows"] != int(design.EXPECTED_TRAIN[panel]["rows"])
                or len(typed_leaf_map(summary)) != 572
            ):
                raise RuntimeError(f"{model_name}/{panel} official summary structure drift")
            frozen_integrity = None
            if model_name in {"raw", "alpha1"}:
                frozen_integrity = assert_frozen_summary_exact(
                    summary,
                    frozen_summaries[model_name][panel],
                    panel,
                    model_name,
                )
            fingerprints = finalize_fingerprints(container["fingerprints"])
            item: dict[str, Any] = {
                "panel": panel,
                "model": model_name,
                "alpha": ALPHAS[model_name],
                "rows": container["rows"],
                "official_metrics": summary,
                "official_metrics_canonical_sha256": sha256_bytes(
                    canonical_json(summary)
                ),
                "official_metrics_typed_scalar_leaf_count": 572,
                "fingerprints": fingerprints,
                "native_output_dtypes": json.loads(native_dtypes[panel][model_name]),
                "frozen_gate_integrity": frozen_integrity,
                "full_per_row_records_published": False,
            }
            panel_results[model_name][panel] = item
            completion_ledger.append(
                {
                    "ordinal": ordinal,
                    "model": model_name,
                    "panel": panel,
                    "completed": True,
                    "rows": container["rows"],
                    "metrics_canonical_sha256": item[
                        "official_metrics_canonical_sha256"
                    ],
                }
            )
    if completion_ledger != [
        {**entry, **{
            "completed": True,
            "rows": panel_results[entry["model"]][entry["panel"]]["rows"],
            "metrics_canonical_sha256": panel_results[entry["model"]][entry["panel"]]["official_metrics_canonical_sha256"],
        }}
        for entry in fixed_ledger()
    ]:
        raise RuntimeError("fixed 30-evaluation completion ledger drift")

    for model_name in MODEL_ORDER[1:]:
        for panel in PANEL_ORDER:
            raw_summary = panel_results["raw"][panel]["official_metrics"]
            candidate_summary = panel_results[model_name][panel]["official_metrics"]
            safety = finalize_transition_book(
                safety_books[model_name][panel],
                metric_correct_counts(raw_summary),
                metric_correct_counts(candidate_summary),
                transition_denominators(design, panel),
            )
            immutable = finalize_transition_book(
                immutable_books[model_name][panel],
                {
                    "count_correct": int(raw_summary["count_correct"]),
                    "value_correct": int(raw_summary["value_correct"]),
                },
                {
                    "count_correct": int(candidate_summary["count_correct"]),
                    "value_correct": int(candidate_summary["value_correct"]),
                },
                {
                    "count_correct": int(design.EXPECTED_TRAIN[panel]["rows"]),
                    "value_correct": int(design.EXPECTED_TRAIN[panel]["rows"]),
                },
            )
            raw_fingerprints = panel_results["raw"][panel]["fingerprints"]
            candidate_fingerprints = panel_results[model_name][panel]["fingerprints"]
            immutable_names = (
                "count_logits_float32_le",
                "value_logits_float32_le",
                "predicted_count",
                "greedy_length",
                "count_correct",
                "value_sign",
                "value_correct",
            )
            exact_fingerprints = {
                name: candidate_fingerprints[name] == raw_fingerprints[name]
                for name in immutable_names
            }
            mismatch = immutable_mismatches[model_name][panel]
            transition_results[model_name][panel] = safety
            panel_results[model_name][panel]["raw_transition_evidence"] = safety
            panel_results[model_name][panel]["count_value_integrity"] = {
                "native_count_logits_mismatch_count": len(mismatch["count_logits"]),
                "native_count_logits_mismatch_full_line_identities": mismatch[
                    "count_logits"
                ],
                "native_value_logits_mismatch_count": len(mismatch["value_logits"]),
                "native_value_logits_mismatch_full_line_identities": mismatch[
                    "value_logits"
                ],
                "correctness_transitions": immutable,
                "fingerprint_exact_raw": exact_fingerprints,
                "all_mismatch_lists_untruncated": True,
            }
            panel_results[model_name][panel]["row_change_audits"] = finalize_change_audit(
                changes[model_name][panel]
            )
    advisory = finalize_secant_advisory(secant, transition_results)
    execution = {
        "device": "cuda:0",
        "model_kind": kind,
        "model_instance_count": 1,
        "actor6_copy_operations": actor_copy_count,
        "all_ten_live_actor_overlays_verified_after_copy": (
            live_actor_states_verified == set(MODEL_ORDER)
        ),
        "same_batch_all_ten_states": True,
        "workers": 0,
        "batch_size": BATCH_SIZE,
        "dataset_audits": dataset_audits,
        "completion_ledger": completion_ledger,
        "evaluation_count_exact": len(completion_ledger),
        "all_30_completed_before_decision": len(completion_ledger) == 30,
        "validation_member_payloads_opened": False,
    }
    return panel_results, advisory, execution


def decide_candidate(
    panel_results: Mapping[str, Any],
    state_audits: Mapping[str, Any],
) -> dict[str, Any]:
    def panel_integrity(model_name: str, panel: str) -> dict[str, Any]:
        result = panel_results[model_name][panel]
        transitions = result["raw_transition_evidence"]
        count_value = result["count_value_integrity"]
        safety_cw = {
            metric: int(transitions[metric]["cells"]["cw"]["count"])
            for metric in SAFETY_METRICS
        }
        immutable_transitions = count_value["correctness_transitions"]
        immutable_changes = {
            metric: {
                "cw": int(immutable_transitions[metric]["cells"]["cw"]["count"]),
                "wc": int(immutable_transitions[metric]["cells"]["wc"]["count"]),
            }
            for metric in ("count_correct", "value_correct")
        }
        checks = {
            "zero_cw_all_safety_metrics": all(value == 0 for value in safety_cw.values()),
            "native_count_logits_tensor_exact_raw_every_row": count_value[
                "native_count_logits_mismatch_count"
            ]
            == 0,
            "native_value_logits_tensor_exact_raw_every_row": count_value[
                "native_value_logits_mismatch_count"
            ]
            == 0,
            "all_count_value_fingerprints_exact_raw": all(
                count_value["fingerprint_exact_raw"].values()
            ),
            "count_value_correctness_has_no_transition": all(
                values["cw"] == 0 and values["wc"] == 0
                for values in immutable_changes.values()
            ),
        }
        return {
            "safety_cw_counts": safety_cw,
            "count_value_correctness_changed_cells": immutable_changes,
            "checks": checks,
            "pass": all(checks.values()),
        }

    alpha1_panels = {
        panel: panel_integrity("alpha1", panel) for panel in PANEL_ORDER
    }
    alpha1_pass = all(item["pass"] for item in alpha1_panels.values())
    candidates: dict[str, Any] = {}
    eligible: list[tuple[int, str]] = []
    for model_name in MODEL_ORDER[2:]:
        panel_gates = {
            panel: panel_integrity(model_name, panel) for panel in PANEL_ORDER
        }
        pokemonfan_transitions = panel_results[model_name]["pokemonfan"][
            "raw_transition_evidence"
        ]
        gains = {
            metric: int(
                pokemonfan_transitions[metric]["candidate_correct_minus_raw_correct"]
            )
            for metric in MAIN_METRICS
        }
        checks = {
            "alpha1_integrity_anchor_pass": alpha1_pass,
            "state_actor6_and_nonactor_integrity_pass": (
                state_audits[model_name]["all_tensors_finite"]
                and state_audits[model_name]["all_nonactor_tensors_bit_exact_raw"]
                and set(
                    state_audits[model_name]["actor_scope"][
                        "changed_parameter_names"
                    ]
                )
                == set(ACTOR6_NAMES)
            ),
            "all_three_panel_independent_safety_and_immutable_gates_pass": all(
                item["pass"] for item in panel_gates.values()
            ),
            "pokemonfan_at_least_one_main_metric_net_plus_one": any(
                value >= 1 for value in gains.values()
            ),
        }
        is_eligible = all(checks.values())
        candidates[model_name] = {
            "alpha": ALPHAS[model_name],
            "panel_gates": panel_gates,
            "pokemonfan_main_metric_net_gains": gains,
            "checks": checks,
            "eligible": is_eligible,
        }
        if is_eligible:
            eligible.append((ALPHAS[model_name], model_name))
    selected_name = min(eligible)[1] if eligible else None
    return {
        "status": "selected_one_candidate" if selected_name else "closed_no_candidate",
        "alpha1_integrity_hard_gate": {
            "selectable": False,
            "panel_gates": alpha1_panels,
            "pass": alpha1_pass,
        },
        "candidates": candidates,
        "eligible_candidates": [name for _, name in sorted(eligible)],
        "selected_candidate": selected_name,
        "selected_alpha": ALPHAS[selected_name] if selected_name else None,
        "selection_rule": "smallest independently eligible observed alpha",
        "selection_executed_after_all_30_evaluations": True,
        "maximum_selected_candidates": 1,
        "selected_candidate_materialized": False,
        "separate_materialization_preregistration_required": bool(selected_name),
        "secant_advisory_participated_in_selection": False,
    }


def validate_formal_preregistration(
    expected_sha256: str,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError("preregistration SHA-256 is not hexadecimal") from error
    payload, evidence = read_regular_bytes(
        PREREGISTRATION,
        expected_sha256,
        "formal preregistration",
        expected_mode=0o444,
    )
    parsed = strict_json_bytes(payload, "formal preregistration")
    expected = build_preregistration(contract)
    if parsed != expected:
        raise RuntimeError("formal preregistration payload is not the exact contract")
    return parsed, evidence


def build_preregistration(contract: Mapping[str, Any]) -> dict[str, Any]:
    preregistration = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "status": "locked_before_formal",
        "formal_contract": dict(contract),
    }
    roundtripped = strict_json_bytes(
        canonical_json(preregistration),
        "complete preregistration JSON-native roundtrip",
    )
    if roundtripped != preregistration:
        raise RuntimeError(
            "complete preregistration is not object-exact after canonical JSON "
            "roundtrip"
        )
    return preregistration


def reverify_after_evaluation(
    design: ModuleType,
    expected_preregistration_sha256: str,
    contract: Mapping[str, Any],
    marker_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    _, _, repeated_design = verify_frozen_design()
    if sha256_bytes(canonical_json(repeated_design["fixed_inputs"])) != contract[
        "fixed_input_snapshot_canonical_sha256"
    ]:
        raise RuntimeError("post-evaluation fixed input snapshot drift")
    if sha256_bytes(canonical_json(repeated_design["cache"])) != contract[
        "train_cache_canonical_sha256"
    ]:
        raise RuntimeError("post-evaluation train cache drift")
    _, runner = read_regular_bytes(
        SCRIPT,
        str(contract["formal_runner"]["sha256"]),
        "post-evaluation formal runner",
        expected_mode=0o555,
    )
    _, helper_evidence = read_regular_bytes(
        HELPER,
        HELPER_SHA256,
        "post-evaluation helper",
        expected_mode=0o555,
    )
    _, preregistration = read_regular_bytes(
        PREREGISTRATION,
        expected_preregistration_sha256,
        "post-evaluation formal preregistration",
        expected_mode=0o444,
    )
    _, marker = read_regular_bytes(
        ATTEMPT_MARKER,
        str(marker_evidence["sha256"]),
        "post-evaluation attempt marker",
        expected_mode=0o444,
    )
    superseded_v1 = verify_superseded_formal_v1()
    if superseded_v1 != contract["frozen_predecessor"]["superseded_formal_v1"]:
        raise RuntimeError("post-evaluation superseded formal v1 archive drift")
    superseded_v2 = verify_superseded_formal_v2()
    if superseded_v2 != contract["frozen_predecessor"]["superseded_formal_v2"]:
        raise RuntimeError("post-evaluation superseded formal v2 evidence drift")
    return {
        "frozen_design_tool_and_artifact_exact": True,
        "fixed_inputs_snapshot_exact": True,
        "train_cache_exact": True,
        "formal_runner": runner,
        "helper": helper_evidence,
        "preregistration": preregistration,
        "attempt_marker": marker,
        "superseded_formal_v1": superseded_v1,
        "superseded_formal_v2": superseded_v2,
    }


def ast_audit() -> dict[str, Any]:
    source = SCRIPT.read_text(encoding="utf-8")
    stub_marker = "formal implementation" + " is not yet complete"
    unfinished_markers = ("TO" + "DO", "FIX" + "ME")
    if stub_marker in source or any(marker in source for marker in unfinished_markers):
        raise RuntimeError("formal runner still contains an implementation stub")
    tree = ast.parse(source, filename=str(SCRIPT))

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def owner(node: ast.AST) -> str | None:
        cursor: ast.AST | None = node
        while cursor is not None:
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cursor.name
            cursor = parent.get(cursor)
        return None

    call_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    calls = [(dotted(node.func), owner(node)) for node in call_nodes]
    pass_nodes = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Pass)]
    not_implemented_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and dotted(node.func) == "NotImplementedError"
    ]
    allowed_os = {
        "os.open": {"read_regular_bytes", "publish_o_excl", "fsync_directory"},
        "os.read": {"read_regular_bytes", "publish_o_excl"},
        "os.write": {"publish_o_excl"},
        "os.fsync": {"publish_o_excl", "fsync_directory"},
        "os.fchmod": {"publish_o_excl"},
        "os.close": {"read_regular_bytes", "publish_o_excl", "fsync_directory"},
        "os.lseek": {"publish_o_excl"},
    }
    bad_os = sorted(
        f"{name}@{where}"
        for name, where in calls
        if name in allowed_os and where not in allowed_os[name]
    )
    forbidden = sorted(
        name
        for name, _ in calls
        if name
        in {
            "open",
            "torch.save",
            "os.mkdir",
            "os.makedirs",
            "os.unlink",
            "os.remove",
            "os.rmdir",
            "os.rename",
            "os.replace",
            "os.symlink",
            "os.link",
            "os.truncate",
            "os.ftruncate",
        }
        or name.endswith(".torch.save")
        or name.endswith(".backward")
        or name.endswith(".step")
        or name.endswith(".write_text")
        or name.endswith(".write_bytes")
    )
    duplicate_keys: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen: dict[tuple[type[Any], Any], int] = {}
        for key in node.keys:
            if not isinstance(key, ast.Constant):
                continue
            try:
                identity = (type(key.value), key.value)
                hash(identity)
            except TypeError:
                continue
            if identity in seen:
                duplicate_keys.append(
                    {"key": repr(key.value), "first_line": seen[identity], "duplicate_line": key.lineno}
                )
            else:
                seen[identity] = key.lineno
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden_imports = sorted(
        imported
        & {
            "requests",
            "urllib",
            "socket",
            "http",
            "ftplib",
            "ssl",
            "kaggle",
            "subprocess",
            "shutil",
            "torch",
        }
    )
    publish_calls = sum(name == "publish_o_excl" for name, _ in calls)
    forward_calls = sum(name == "helper.ppo.model_forward" for name, _ in calls)
    sample_calls = sum(name == "helper.ppo.sample_ordered_actions" for name, _ in calls)
    expected_calls = {
        "publish_o_excl": 2,
        "helper.ppo.model_forward": 1,
        "helper.ppo.sample_ordered_actions": 1,
    }
    observed_calls = {
        "publish_o_excl": publish_calls,
        "helper.ppo.model_forward": forward_calls,
        "helper.ppo.sample_ordered_actions": sample_calls,
    }
    nextafter_owners = sorted(
        [
            [int(node.lineno), owner(node)]
            for node in call_nodes
            if dotted(node.func) == "math.nextafter"
        ],
        key=lambda item: (item[0], str(item[1])),
    )
    nextafter_scope_exact = (
        len(nextafter_owners) == 1
        and nextafter_owners[0][1] == "synthetic_self_test"
    )
    marker_publish_lines = [
        int(node.lineno)
        for node in call_nodes
        if owner(node) == "run_formal"
        and dotted(node.func) == "publish_o_excl"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "ATTEMPT_MARKER"
    ]
    result_publish_lines = [
        int(node.lineno)
        for node in call_nodes
        if owner(node) == "run_formal"
        and dotted(node.func) == "publish_o_excl"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "RESULT"
    ]
    build_lines = [
        int(node.lineno)
        for node in call_nodes
        if owner(node) == "run_formal" and dotted(node.func) == "build_ray_states"
    ]
    evaluate_lines = [
        int(node.lineno)
        for node in call_nodes
        if owner(node) == "run_formal"
        and dotted(node.func) == "evaluate_all_states"
    ]
    checkpoint_deserialization_sites = [
        int(node.lineno)
        for node in call_nodes
        if owner(node) == "build_ray_states"
        and dotted(node.func) == "helper.checkpoint_from_bytes"
    ]
    model_instantiation_sites = [
        int(node.lineno)
        for node in call_nodes
        if owner(node) == "evaluate_all_states"
        and dotted(node.func) == "helper.instantiate_checkpoint"
    ]
    formal_order_exact = (
        len(marker_publish_lines) == 1
        and len(result_publish_lines) == 1
        and len(build_lines) == 1
        and len(evaluate_lines) == 1
        and len(checkpoint_deserialization_sites) == 2
        and len(model_instantiation_sites) == 1
        and marker_publish_lines[0] < build_lines[0] < evaluate_lines[0]
        < result_publish_lines[0]
    )
    if (
        bad_os
        or forbidden
        or duplicate_keys
        or forbidden_imports
        or pass_nodes
        or not_implemented_calls
        or not nextafter_scope_exact
        or observed_calls != expected_calls
        or not formal_order_exact
    ):
        raise RuntimeError(
            f"formal AST audit failed: os={bad_os}, calls={forbidden}, "
            f"duplicates={duplicate_keys}, imports={forbidden_imports}, "
            f"pass_nodes={pass_nodes}, not_implemented={not_implemented_calls}, "
            f"expected_calls={expected_calls}, observed_calls={observed_calls}, "
            f"nextafter_owners={nextafter_owners}, formal_order={formal_order_exact}"
        )
    return {
        "status": "formal_runner_static_audit_passed",
        "os_calls_outside_allowlisted_helpers": bad_os,
        "forbidden_calls": forbidden,
        "forbidden_imports": forbidden_imports,
        "duplicate_constant_dict_keys": duplicate_keys,
        "publish_o_excl_call_sites": publish_calls,
        "official_forward_call_sites": forward_calls,
        "official_sample_call_sites": sample_calls,
        "model_serialization_call_sites": 0,
        "network_call_sites": 0,
        "exact_expected_call_sites": True,
        "implementation_stub_absent": True,
        "pass_nodes": pass_nodes,
        "not_implemented_call_sites": not_implemented_calls,
        "math_nextafter_scope": nextafter_owners,
        "math_nextafter_synthetic_fixture_only": nextafter_scope_exact,
        "formal_control_flow_order": {
            "attempt_marker_publish_lines": marker_publish_lines,
            "build_ray_states_lines": build_lines,
            "checkpoint_deserialization_sites_inside_builder": (
                checkpoint_deserialization_sites
            ),
            "evaluate_all_states_lines": evaluate_lines,
            "model_instantiation_sites_inside_evaluator": model_instantiation_sites,
            "result_publish_lines": result_publish_lines,
            "marker_before_deserialize_before_model_eval_before_result": (
                formal_order_exact
            ),
        },
    }


def run_formal(
    expected_preregistration_sha256: str,
    contract: Mapping[str, Any],
    design: ModuleType,
    design_artifact: Mapping[str, Any],
    design_evidence: Mapping[str, Any],
    helper: ModuleType,
) -> dict[str, Any]:
    preregistration, preregistration_evidence = validate_formal_preregistration(
        expected_preregistration_sha256, contract
    )
    absence = output_absence()
    if (
        absence["preregistration_absent"]
        or not absence["attempt_marker_absent"]
        or not absence["result_absent"]
    ):
        raise FileExistsError(f"formal one-shot output state is invalid: {absence}")
    repeated_design, repeated_artifact, repeated_evidence = verify_frozen_design()
    if repeated_artifact != design_artifact:
        raise RuntimeError("design artifact changed at formal lock")
    if sha256_bytes(canonical_json(repeated_evidence["fixed_inputs"])) != contract[
        "fixed_input_snapshot_canonical_sha256"
    ]:
        raise RuntimeError("fixed input snapshot changed at formal lock")
    if sha256_bytes(canonical_json(repeated_evidence["cache"])) != contract[
        "train_cache_canonical_sha256"
    ]:
        raise RuntimeError("train identity cache changed at formal lock")
    _, runner_at_lock = read_regular_bytes(
        SCRIPT,
        str(contract["formal_runner"]["sha256"]),
        "formal runner at lock",
        expected_mode=0o555,
    )
    _, helper_at_lock = read_regular_bytes(
        HELPER,
        HELPER_SHA256,
        "formal helper at lock",
        expected_mode=0o555,
    )
    superseded_v1_at_lock = verify_superseded_formal_v1()
    if superseded_v1_at_lock != contract["frozen_predecessor"][
        "superseded_formal_v1"
    ]:
        raise RuntimeError("superseded formal v1 archive changed at lock")
    superseded_v2_at_lock = verify_superseded_formal_v2()
    if superseded_v2_at_lock != contract["frozen_predecessor"][
        "superseded_formal_v2"
    ]:
        raise RuntimeError("superseded formal v2 evidence changed at lock")
    repeated_runtime = cuda_runtime(helper)
    if repeated_runtime != contract["runtime"]:
        raise RuntimeError("CUDA runtime changed between preregistration and lock")
    expected_contract = formal_contract(
        runner_at_lock,
        repeated_artifact,
        repeated_evidence,
        superseded_v1_at_lock,
        superseded_v2_at_lock,
        repeated_runtime,
    )
    if expected_contract != contract or preregistration["formal_contract"] != contract:
        raise RuntimeError("formal contract reconstruction drift at lock")

    # Snapshot every large binary/control input through held O_NOFOLLOW fds at
    # lock, but do not deserialize a checkpoint or construct a model yet.
    raw_payload, raw_evidence = read_regular_bytes(
        design.PARENT,
        design.PARENT_FILE_SHA256,
        "raw U468 checkpoint byte snapshot at formal lock",
        expected_mode=int(
            str(repeated_evidence["fixed_inputs"]["parent"]["mode"]), 8
        ),
    )
    endpoint_payload, endpoint_evidence = read_regular_bytes(
        design.ENDPOINT,
        design.ENDPOINT_FILE_SHA256,
        "alpha1 endpoint checkpoint byte snapshot at formal lock",
        expected_mode=0o444,
    )
    archive_payloads: dict[str, bytes] = {}
    archive_evidence: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        payload, item = read_regular_bytes(
            design.DATASETS[panel],
            design.DATA_SHA256[panel],
            f"{panel} train archive byte snapshot at formal lock",
            expected_mode=int(
                str(
                    repeated_evidence["fixed_inputs"]["datasets"][panel]["mode"]
                ),
                8,
            ),
        )
        archive_payloads[panel] = payload
        archive_evidence[panel] = item
    frozen_summaries, frozen_summary_evidence = load_frozen_metric_summaries(design)

    final_absence = output_absence()
    if (
        final_absence["preregistration_absent"]
        or not final_absence["attempt_marker_absent"]
        or not final_absence["result_absent"]
    ):
        raise FileExistsError(
            f"one-shot outputs changed immediately before marker: {final_absence}"
        )

    marker_payload = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "formal_attempt_consumed_before_checkpoint_deserialization_or_model_instance",
        "consumed_at_utc": utc_now(),
        "branch": BRANCH,
        "formal_contract_canonical_sha256": sha256_bytes(canonical_json(contract)),
        "preregistration": preregistration_evidence,
        "runner": runner_at_lock,
        "helper": helper_at_lock,
        "frozen_design": {
            "tool": repeated_evidence["tool"],
            "artifact": repeated_evidence["artifact"],
        },
        "fixed_input_snapshot_canonical_sha256": contract[
            "fixed_input_snapshot_canonical_sha256"
        ],
        "train_cache_canonical_sha256": contract[
            "train_cache_canonical_sha256"
        ],
        "at_lock_binary_snapshots": {
            "raw_checkpoint": raw_evidence,
            "alpha1_endpoint_checkpoint": endpoint_evidence,
            "train_archives": archive_evidence,
            "frozen_official_summaries": frozen_summary_evidence,
        },
        "all_binary_snapshots_O_NOFOLLOW_pre_post_visible_identity_exact": True,
        "checkpoint_deserializations_before_marker": 0,
        "model_instances_before_marker": 0,
        "model_artifact_writes_before_marker": 0,
        "validation_member_payloads_opened": False,
        "network_upload_submission": False,
    }
    marker_evidence = publish_o_excl(ATTEMPT_MARKER, canonical_json(marker_payload))
    fsync_directory(ROOT)
    started = time.time()

    random.seed(SEED)
    helper.torch.manual_seed(SEED)
    helper.torch.cuda.manual_seed_all(SEED)
    states, state_audits, raw_checkpoint = build_ray_states(
        helper,
        design,
        raw_payload,
        endpoint_payload,
    )
    panel_results, advisory, execution = evaluate_all_states(
        helper,
        design,
        states,
        raw_checkpoint,
        archive_payloads,
        frozen_summaries,
    )
    if execution["evaluation_count_exact"] != 30 or not execution[
        "all_30_completed_before_decision"
    ]:
        raise RuntimeError("formal decision reached before fixed 30-ledger completion")
    decision = decide_candidate(panel_results, state_audits)
    postrun = reverify_after_evaluation(
        design,
        expected_preregistration_sha256,
        contract,
        marker_evidence,
    )
    if RESULT.exists() or RESULT.is_symlink():
        raise FileExistsError("reserved formal result appeared during evaluation")
    completed_at = utc_now()
    result_payload = {
        "schema_version": RESULT_SCHEMA,
        "status": decision["status"],
        "branch": BRANCH,
        "started_after_marker_utc": marker_payload["consumed_at_utc"],
        "completed_at_utc": completed_at,
        "seconds_after_marker": time.time() - started,
        "formal_contract": contract,
        "formal_contract_canonical_sha256": sha256_bytes(canonical_json(contract)),
        "preregistration": preregistration_evidence,
        "attempt_marker": marker_evidence,
        "input_lock": {
            "raw_checkpoint": raw_evidence,
            "alpha1_endpoint_checkpoint": endpoint_evidence,
            "train_archives": archive_evidence,
            "frozen_official_summaries": frozen_summary_evidence,
        },
        "state_integrity": state_audits,
        "execution": execution,
        "evaluations": panel_results,
        "secant_and_observed_grid_advisory": advisory,
        "decision": decision,
        "post_evaluation_reverification": postrun,
        "scope_audit": {
            "validation_member_payloads_opened": False,
            "validation_results_read": False,
            "training_optimizer_backward": False,
            "candidate_states_RAM_only": True,
            "model_artifact_writes": 0,
            "evidence_writes": [
                str(ATTEMPT_MARKER.relative_to(ROOT)),
                str(RESULT.relative_to(ROOT)),
            ],
            "network_upload_submission": False,
        },
    }
    result_bytes = canonical_json(result_payload)
    result_evidence = publish_o_excl(RESULT, result_bytes)
    fsync_directory(ROOT / "artifacts")
    del states
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "formal_result_published",
        "decision": decision,
        "attempt_marker": marker_evidence,
        "result": result_evidence,
        "result_payload_canonical_sha256": sha256_bytes(result_bytes),
        "evaluation_count_exact": 30,
        "all_30_completed_before_decision": True,
        "validation_member_payloads_opened": False,
        "model_artifact_writes": 0,
        "submission_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("static-audit", "cache-audit", "contract", "formal"),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--expected-preregistration-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    static = ast_audit()
    synthetic = synthetic_self_test()
    design, design_artifact, design_evidence = verify_frozen_design()
    superseded_v1 = verify_superseded_formal_v1()
    superseded_v2 = verify_superseded_formal_v2()
    if args.mode in {"static-audit", "cache-audit"}:
        if args.device != "cpu" or args.expected_preregistration_sha256 is not None:
            raise ValueError("static/cache modes require --device cpu and no prereg SHA")
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": f"zero_write_{args.mode.replace('-', '_')}_passed",
            "ast_audit": static,
            "synthetic_self_test": synthetic,
            "frozen_design": {
                "tool": design_evidence["tool"],
                "artifact": design_evidence["artifact"],
            },
            "superseded_formal_v1": superseded_v1,
            "superseded_formal_v2": superseded_v2,
            "cache": design_evidence["cache"] if args.mode == "cache-audit" else None,
            "outputs": output_absence(),
            "formal_executed": False,
            "writes_performed": False,
        }
    else:
        if args.device != "cuda":
            raise ValueError("contract/formal modes require --device cuda")
        helper = load_helper()
        runtime = cuda_runtime(helper)
        runner_payload, runner_evidence = read_regular_bytes(
            SCRIPT,
            sha256_bytes(SCRIPT.read_bytes()),
            "formal runner",
            expected_mode=0o555,
        )
        del runner_payload
        contract = formal_contract(
            runner_evidence,
            design_artifact,
            design_evidence,
            superseded_v1,
            superseded_v2,
            runtime,
        )
        if args.mode == "contract":
            if args.expected_preregistration_sha256 is not None:
                raise ValueError("contract mode forbids expected prereg SHA")
            if not all(output_absence().values()):
                raise FileExistsError("reserved formal output already exists")
            result = build_preregistration(contract)
        else:
            expected = args.expected_preregistration_sha256
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("formal mode requires exact preregistration SHA-256")
            result = run_formal(
                expected,
                contract,
                design,
                design_artifact,
                design_evidence,
                helper,
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
