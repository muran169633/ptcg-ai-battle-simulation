#!/usr/bin/env python3
"""Build one train-only CW19 special-BC endpoint from the CW11 bootstrap.

The production path replays the already-consumed exact CW11 model, applies the
frozen e381 bootstrap, computes one fixed-order PCGrad direction on the frozen
512-row training cache, and performs exactly one constrained projection.  It
never opens a changed-candidate validation stream, evaluates an official
candidate, writes a checkpoint, packages a submission, uploads, or submits.
The only production output is JSON on stdout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_bootstrap_pcgrad_specialbc_cw19_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-bootstrap-pcgrad-specialbc-cw19-v1"
SEED = 202608301

CW15 = TOOLS / "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py"
DUAL = TOOLS / "cw17_stage2_dual_math_v1.py"
GRADIENT = TOOLS / "probe_u468_raw_aggregate512_actor6_gradients.py"
RUNNER = TOOLS / "run_u468_raw_trainhard_actor6_balanced_mix_sweep.py"
FIXTURE = ROOT / "artifacts/cw18_direct_fixture_20260803_v1.stdout.json"

FROZEN: dict[Path, tuple[str, int]] = {
    CW15: ("2aeb770f8774b7f7407a93553405089511ab2d2f278653162274a3afb89a2f24", 0o555),
    DUAL: ("a14241d500f352e6ae1b5662d5d7ea51ca4da22b61126536997e69e39750a600", 0o555),
    GRADIENT: ("ed85b59bd69558ab6f027ba21df9cb56c744baf9793bbb6b550d46bffbc3389a", 0o555),
    RUNNER: ("419feeeb5644adf34a8d56142cfcd87b3a4b79446d95b9b566bf9fcabdb721b1", 0o555),
    FIXTURE: ("a0d3ecc5a1315160b8b06ed212af9c6b6179c13695488c44cf5e445e043cd929", 0o444),
}

EXPECTED_FIXTURE_INTERNAL_SHA = (
    "ef6c7bfa1017ea9ceb77b24543a2a9507bdd48873c7b3a21865691684987c8ec"
)
EXPECTED_CENTER_SHA = "e3814600afa6f1c7843ca073d89a86cb1e32cf162d89f8cbfaf8db378ad3183f"
EXPECTED_STAGE1_RAW_SHA = (
    "3f50bfe3fc49d5d23867db61982341000206cbb9cd5fc9eb924fb3e264a832c4"
)
EXPECTED_BOOTSTRAP_MODEL_SHA = (
    "ba5098d4e5c52bb6a02c4e69fa2015428d018f85953edc4fb6bc14330634d769"
)
EXPECTED_RAW_MODEL_SHA = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
EXPECTED_RAW_NONACTOR_SHA = (
    "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
)
EXPECTED_CW11_MODEL_SHA = (
    "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
)
EXPECTED_CW11_VECTOR_SHA = (
    "e7183412ec13e9b69b2b50c95b4ebc6fc603a3d934229ca47b00538abb95abcd"
)

ACTOR_DIMENSION = 65793
ANCHOR_ROWS = 50
LOCAL_ROWS = 41
TOTAL_CAP = 0.001
TRUST_RADIUS = 0.000125
LINEAR_GATE_TOLERANCE = 1e-8
PROJECTION_LINEAR_RESERVE = 5e-9
L2_TOLERANCE = 1e-12
COORDINATE_SCALE = TOTAL_CAP
RHO = TRUST_RADIUS / COORDINATE_SCALE
ORTHOGONAL_EXTENSION_RELATIVE_TOLERANCE = 1e-12
ORTHOGONALITY_ABSOLUTE_TOLERANCE = 1e-12
PCGRAD_COSINE_EPSILON = 1e-12
PREDICTED_TASK_IMPROVEMENT_MIN = 1e-8
OBSERVED_TASK_IMPROVEMENT_MIN = 1e-8
OBSERVED_RETENTION_LOSS_TOLERANCE = 0.0
SLSQP_FTOL = 1e-12
SLSQP_MAXITER = 2000

TASK_ORDER = (
    "union_mixed",
    "flg_hard",
    "pokemonfan_hard",
    "core5_hard",
)
RETENTION_OBJECTIVE = "union_retention"
EXTENSION_ORDER = TASK_ORDER + (RETENTION_OBJECTIVE,)


class ProtocolError(RuntimeError):
    """Fail-closed CW19 protocol error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def read_locked(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    expected_sha, expected_mode = FROZEN[path]
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
    ):
        raise ProtocolError(f"{label} identity/mode drift")
    payload = path.read_bytes()
    after = path.lstat()
    checks = {
        "regular": stat.S_ISREG(after.st_mode) and not stat.S_ISLNK(after.st_mode),
        "single_link": int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "sha_exact": sha256_bytes(payload) == expected_sha,
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} changed while reading: {checks}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": expected_sha,
        "bytes": len(payload),
        "mode_octal": format(expected_mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def import_locked(path: Path, name: str) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_locked(path, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, evidence


def runtime_audit(require_cuda: bool) -> dict[str, Any]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact_my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "cublas_workspace_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        == ":4096:8",
    }
    if not all(checks.values()):
        raise ProtocolError(f"runtime contract failed: {checks}")
    result: dict[str, Any] = {
        "checks": checks,
        "pass": True,
        "cuda_required": require_cuda,
    }
    if require_cuda:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "native_bf16": bool(torch.cuda.is_bf16_supported()),
            "device_count_positive": int(torch.cuda.device_count()) > 0,
        }
        if not all(cuda.values()):
            raise ProtocolError(f"CUDA BF16 unavailable: {cuda}")
        result["cuda"] = cuda
    return result


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_attributes = {
        "save",
        "savez",
        "savez_compressed",
        "evaluate_candidate_once",
        "evaluate_full_stream",
        "package_submission",
        "submit",
        "upload",
        "backward",
        "step",
    }
    forbidden_names = {
        "evaluate_candidate_once",
        "evaluate_full_stream",
        "package_submission",
        "submit",
        "upload",
    }
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attributes:
            hits.append(f"attribute:{node.func.attr}@{node.lineno}")
        if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
            hits.append(f"name:{node.func.id}@{node.lineno}")
    write_modes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name == "open" and len(node.args) >= 2:
            mode = node.args[1]
            if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
                if any(value in mode.value for value in "wax+"):
                    write_modes.append(f"open:{mode.value}@{node.lineno}")
    checks = {
        "no_forbidden_calls": not hits,
        "no_explicit_write_open": not write_modes,
        "stdout_print_present": any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            for node in ast.walk(tree)
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"source audit failed: {checks}, {hits}, {write_modes}")
    return {
        "checks": checks,
        "pass": True,
        "source_sha256": sha256_bytes(source),
        "source_bytes": len(source),
        "forbidden_call_hits": hits,
        "write_mode_hits": write_modes,
    }


def decode_fixture(dual: ModuleType) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, evidence = read_locked(FIXTURE, "CW18 fixture")
    document = json.loads(payload)
    del payload
    if (
        document.get("status") != "fixture_built"
        or document.get("pass") is not True
        or document.get("writes_performed") != 0
        or document.get("submission_performed") is not False
        or document.get("changed_model_official_evaluation_count") != 0
        or document.get("fixture", {}).get("fixture_sha256")
        != EXPECTED_FIXTURE_INTERNAL_SHA
    ):
        raise ProtocolError("CW18 fixture top-level contract drift")
    raw_contract = document["raw_space_contract"]
    reconstruction = document["reconstruction_map"]
    arrays = {
        name: dual.decode_array(record)
        for name, record in raw_contract["arrays"].items()
    }
    A = arrays["anchor_gradients_A"]
    b = arrays["anchor_rhs_b"]
    G = arrays["local_gradients_G"]
    residual = arrays["local_residual_at_center"]
    basis = dual.decode_array(reconstruction["basis_reduced_by_actor"])
    center = dual.decode_array(reconstruction["center_raw"])
    stage1_scaled = dual.decode_array(document["stage1"]["x_scaled"])
    stage1_raw = basis.T @ (COORDINATE_SCALE * stage1_scaled)
    sha_ledger = raw_contract["array_float64_le_sha256"]
    checks = {
        "raw_contract_pass": raw_contract.get("pass") is True,
        "A_shape": A.shape == (ANCHOR_ROWS, ACTOR_DIMENSION),
        "b_shape": b.shape == (ANCHOR_ROWS,),
        "G_shape": G.shape == (LOCAL_ROWS, ACTOR_DIMENSION),
        "residual_shape": residual.shape == (LOCAL_ROWS,),
        "basis_shape": basis.shape == (80, ACTOR_DIMENSION),
        "center_shape": center.shape == (ACTOR_DIMENSION,),
        "stage1_shape": stage1_scaled.shape == (80,),
        "arrays_finite": all(
            value.dtype.name == "float64" and bool(__import__("numpy").isfinite(value).all())
            for value in (A, b, G, residual, basis, center, stage1_scaled, stage1_raw)
        ),
        "raw_array_shas": all(
            dual.float64_sha256(arrays[name]) == sha_ledger[name]
            for name in arrays
        ),
        "basis_sha": dual.float64_sha256(basis)
        == document["basis_float64_le_sha256"],
        "center_sha_e381": dual.float64_sha256(center) == EXPECTED_CENTER_SHA,
        "stage1_raw_sha": dual.float64_sha256(stage1_raw)
        == EXPECTED_STAGE1_RAW_SHA,
        "caps_exact": float(raw_contract["TOTAL_L2_CAP"]) == TOTAL_CAP
        and float(raw_contract["trust_radius_relative_actual_center"])
        == TRUST_RADIUS,
        "linear_gate_tolerance_exact": float(raw_contract["LINEAR_RESIDUAL_TOL"])
        == LINEAR_GATE_TOLERANCE,
        "l2_tolerance_exact": float(raw_contract["L2_ABS_TOL"])
        == L2_TOLERANCE,
    }
    if not all(checks.values()):
        raise ProtocolError(f"decoded fixture drift: {checks}")
    return {
        "A": A,
        "b": b,
        "G": G,
        "residual": residual,
        "basis": basis,
        "center": center,
        "stage1_raw": stage1_raw,
        "z_star": float(raw_contract["z_star"]),
    }, {"checks": checks, "pass": True, "file": evidence}


def flatten_gradient(
    gradient: Mapping[str, Any], actor_names: Sequence[str], np: Any
) -> Any:
    value = np.concatenate(
        [
            gradient[name].detach().cpu().to(dtype=__import__("torch").float64)
            .contiguous()
            .numpy()
            .reshape(-1)
            for name in actor_names
        ]
    ).astype(np.float64, copy=False)
    if value.shape != (ACTOR_DIMENSION,) or not np.isfinite(value).all():
        raise ProtocolError("flattened actor6 gradient drift")
    return value


def extend_basis(
    base_basis: Any,
    vectors: Sequence[Any],
    np: Any,
    *,
    production_contract: bool = True,
) -> tuple[Any, dict[str, Any]]:
    basis = np.asarray(base_basis, dtype=np.float64).copy()
    base_gram_error = float(np.abs(basis @ basis.T - np.eye(basis.shape[0])).max())
    records = []
    for name, vector in zip(EXTENSION_ORDER, vectors, strict=True):
        original = np.asarray(vector, dtype=np.float64)
        residual = original - basis.T @ (basis @ original)
        residual = residual - basis.T @ (basis @ residual)
        original_l2 = float(np.linalg.norm(original))
        residual_l2 = float(np.linalg.norm(residual))
        threshold = ORTHOGONAL_EXTENSION_RELATIVE_TOLERANCE * original_l2
        appended = residual_l2 > threshold
        if appended:
            basis = np.concatenate([basis, (residual / residual_l2).reshape(1, -1)], axis=0)
        records.append(
            {
                "objective": name,
                "original_l2": original_l2,
                "orthogonal_residual_l2": residual_l2,
                "append_threshold": threshold,
                "appended": appended,
                "dimension_after": int(basis.shape[0]),
            }
        )
    gram_error = float(np.abs(basis @ basis.T - np.eye(basis.shape[0])).max())
    checks = {
        "base_shape": (
            base_basis.shape == (80, ACTOR_DIMENSION)
            if production_contract
            else base_basis.ndim == 2
        ),
        "base_orthonormal": base_gram_error <= ORTHOGONALITY_ABSOLUTE_TOLERANCE,
        "extension_dimension_bounded": base_basis.shape[0]
        <= basis.shape[0]
        <= min(base_basis.shape[0] + len(vectors), base_basis.shape[1]),
        "extended_orthonormal": gram_error <= ORTHOGONALITY_ABSOLUTE_TOLERANCE,
        "all_input_vectors_nonzero": all(record["original_l2"] > 0.0 for record in records),
    }
    if not all(checks.values()):
        raise ProtocolError(f"extended basis failed: {checks}")
    return basis, {
        "checks": checks,
        "pass": True,
        "base_gram_max_abs_error": base_gram_error,
        "extended_gram_max_abs_error": gram_error,
        "records": records,
        "dimension": int(basis.shape[0]),
    }


def project_fixed_endpoint(
    fixture: Mapping[str, Any],
    objective_vectors: Mapping[str, Any],
    pcgrad_direction: Any,
    np: Any,
    optimize: Any,
) -> tuple[Any | None, dict[str, Any]]:
    A = np.asarray(fixture["A"], dtype=np.float64)
    b = np.asarray(fixture["b"], dtype=np.float64)
    G = np.asarray(fixture["G"], dtype=np.float64)
    residual = np.asarray(fixture["residual"], dtype=np.float64)
    center = np.asarray(fixture["center"], dtype=np.float64)
    stage1_raw = np.asarray(fixture["stage1_raw"], dtype=np.float64)
    z_star = float(fixture["z_star"])
    ordered_vectors = [objective_vectors[name] for name in EXTENSION_ORDER]
    basis, basis_audit = extend_basis(fixture["basis"], ordered_vectors, np)

    direction = np.asarray(pcgrad_direction, dtype=np.float64)
    direction_l2 = float(np.linalg.norm(direction))
    if direction.shape != (ACTOR_DIMENSION,) or not math.isfinite(direction_l2) or direction_l2 <= 0.0:
        raise ProtocolError("PCGrad direction is zero/nonfinite")
    direction_unit = direction / direction_l2
    direction_projection = basis.T @ (basis @ direction_unit)
    direction_projection_error = float(np.linalg.norm(direction_projection - direction_unit))
    if direction_projection_error > 2e-10:
        raise ProtocolError("extended basis does not contain PCGrad direction")

    center_scaled = basis @ center / COORDINATE_SCALE
    seed_scaled = basis @ stage1_raw / COORDINATE_SCALE
    center_reconstruction_error = float(
        np.linalg.norm(basis.T @ (basis @ center) - center)
    )
    seed_reconstruction_error = float(
        np.linalg.norm(basis.T @ (basis @ stage1_raw) - stage1_raw)
    )
    objective_reconstruction_relative = {
        name: float(
            np.linalg.norm(
                basis.T @ (basis @ np.asarray(objective_vectors[name]))
                - np.asarray(objective_vectors[name])
            )
            / max(float(np.linalg.norm(objective_vectors[name])), 1e-300)
        )
        for name in EXTENSION_ORDER
    }
    reconstruction_checks = {
        "center_reconstruction": center_reconstruction_error <= L2_TOLERANCE,
        "stage1_seed_reconstruction": seed_reconstruction_error <= L2_TOLERANCE,
        "five_objective_reconstruction": max(
            objective_reconstruction_relative.values()
        )
        <= 2e-10,
    }
    if not all(reconstruction_checks.values()):
        raise ProtocolError(
            f"extended basis reconstruction failed: {reconstruction_checks}"
        )
    target_scaled = center_scaled - RHO * (basis @ direction_unit)
    A_scaled = (A @ basis.T) * COORDINATE_SCALE
    G_scaled = (G @ basis.T) * COORDINATE_SCALE
    objective_scaled = np.stack(
        [
            (np.asarray(objective_vectors[name], dtype=np.float64) @ basis.T)
            * COORDINATE_SCALE
            for name in EXTENSION_ORDER
        ],
        axis=0,
    )
    anchor_scales = np.maximum.reduce(
        [
            np.abs(b),
            np.linalg.norm(A_scaled, axis=1),
            np.full(b.shape, np.finfo(np.float64).eps),
        ]
    )
    z_scale = float(
        max(
            float(np.abs(residual).max()),
            float((np.linalg.norm(G_scaled, axis=1) * RHO).max()),
            np.finfo(np.float64).eps,
        )
    )
    prediction_thresholds = np.asarray(
        [PREDICTED_TASK_IMPROVEMENT_MIN] * len(TASK_ORDER) + [0.0],
        dtype=np.float64,
    )
    prediction_scales = np.maximum.reduce(
        [
            np.abs(prediction_thresholds),
            np.linalg.norm(objective_scaled, axis=1) * RHO,
            np.full(prediction_thresholds.shape, np.finfo(np.float64).eps),
        ]
    )

    def anchor_fun(value: Any) -> Any:
        return (A_scaled @ value - b + PROJECTION_LINEAR_RESERVE) / anchor_scales

    def local_fun(value: Any) -> Any:
        return (
            residual
            + G_scaled @ (value - center_scaled)
            - z_star
            + PROJECTION_LINEAR_RESERVE
        ) / z_scale

    def prediction_fun(value: Any) -> Any:
        return (
            -objective_scaled @ (value - center_scaled) - prediction_thresholds
        ) / prediction_scales

    constraints = [
        {
            "type": "ineq",
            "fun": anchor_fun,
            "jac": lambda value: A_scaled / anchor_scales[:, None],
        },
        {
            "type": "ineq",
            "fun": local_fun,
            "jac": lambda value: G_scaled / z_scale,
        },
        {
            "type": "ineq",
            "fun": prediction_fun,
            "jac": lambda value: -objective_scaled / prediction_scales[:, None],
        },
        {
            "type": "ineq",
            "fun": lambda value: float(1.0 - value @ value),
            "jac": lambda value: -2.0 * value,
        },
        {
            "type": "ineq",
            "fun": lambda value: float(
                RHO**2 - (value - center_scaled) @ (value - center_scaled)
            ),
            "jac": lambda value: -2.0 * (value - center_scaled),
        },
    ]
    projection = optimize.minimize(
        lambda value: 0.5 * float((value - target_scaled) @ (value - target_scaled)),
        seed_scaled,
        jac=lambda value: value - target_scaled,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER, "disp": False},
    )
    report: dict[str, Any] = {
        "method": "single_SLSQP_projection_no_restart_no_sweep_no_clip",
        "success": bool(projection.success),
        "status": int(projection.status),
        "message": str(projection.message),
        "iterations": int(projection.nit),
        "function_evaluations": int(projection.nfev),
        "jacobian_evaluations": int(projection.njev),
        "objective": float(projection.fun) if math.isfinite(float(projection.fun)) else None,
        "target_rule": "center_minus_full_trust_radius_times_normalized_fixed_PCGrad",
        "target_trust_l2": TRUST_RADIUS,
        "projection_linear_reserve": PROJECTION_LINEAR_RESERVE,
        "prediction_positive_row_scales": [
            float(value) for value in prediction_scales
        ],
        "final_linear_gate_tolerance": LINEAR_GATE_TOLERANCE,
        "basis": basis_audit,
        "basis_reconstruction": {
            "checks": reconstruction_checks,
            "pass": True,
            "center_l2_error": center_reconstruction_error,
            "stage1_seed_l2_error": seed_reconstruction_error,
            "objective_relative_l2_error": objective_reconstruction_relative,
        },
        "direction_l2": direction_l2,
        "direction_projection_error": direction_projection_error,
        "one_projection_call": True,
        "restart_count": 0,
        "alternate_start_count": 0,
        "clip_count": 0,
    }
    if not bool(projection.success):
        report["decision"] = "NO_ENDPOINT_PROJECTION_FAILED"
        return None, report
    candidate = basis.T @ (COORDINATE_SCALE * np.asarray(projection.x, dtype=np.float64))
    planned = planned_contract(candidate, fixture, objective_vectors, np)
    report["planned_contract"] = planned
    if planned["pass"] is not True:
        report["decision"] = "NO_ENDPOINT_PLANNED_CONTRACT_FAILED"
        return None, report
    report["decision"] = "ONE_PROJECTED_ENDPOINT"
    report["candidate_float64_le_sha256"] = float64_sha(candidate, np)
    return candidate, report


def float64_sha(value: Any, np: Any) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(value, dtype="<f8")).tobytes())


def planned_contract(
    candidate: Any,
    fixture: Mapping[str, Any],
    objective_vectors: Mapping[str, Any],
    np: Any,
) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    center = np.asarray(fixture["center"], dtype=np.float64)
    anchor = np.asarray(fixture["A"], dtype=np.float64) @ candidate - np.asarray(
        fixture["b"], dtype=np.float64
    )
    local = np.asarray(fixture["residual"], dtype=np.float64) + np.asarray(
        fixture["G"], dtype=np.float64
    ) @ (candidate - center)
    improvements = {
        name: -float(np.asarray(objective_vectors[name]) @ (candidate - center))
        for name in EXTENSION_ORDER
    }
    total_l2 = float(np.linalg.norm(candidate))
    trust_l2 = float(np.linalg.norm(candidate - center))
    checks = {
        "shape": candidate.shape == (ACTOR_DIMENSION,),
        "finite": bool(np.isfinite(candidate).all()),
        "anchor_original_raw_gate": float(anchor.min()) >= -LINEAR_GATE_TOLERANCE,
        "local_original_raw_gate": float(local.min())
        >= float(fixture["z_star"]) - LINEAR_GATE_TOLERANCE,
        "anchor_projection_half_tolerance_reserve": float(anchor.min())
        >= -PROJECTION_LINEAR_RESERVE,
        "local_projection_half_tolerance_reserve": float(local.min())
        >= float(fixture["z_star"]) - PROJECTION_LINEAR_RESERVE,
        "total_cap": total_l2 <= TOTAL_CAP + L2_TOLERANCE,
        "trust_cap": trust_l2 <= TRUST_RADIUS + L2_TOLERANCE,
        "four_task_predicted_improvement": all(
            improvements[name] >= PREDICTED_TASK_IMPROVEMENT_MIN - 1e-12
            for name in TASK_ORDER
        ),
        "retention_predicted_nonincrease": improvements[RETENTION_OBJECTIVE] >= -1e-12,
        "no_clip": True,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "anchor_residual_min": float(anchor.min()),
        "local_prediction_min": float(local.min()),
        "z_star": float(fixture["z_star"]),
        "total_l2": total_l2,
        "trust_l2": trust_l2,
        "predicted_loss_improvements": improvements,
        "candidate_float64_le_sha256": float64_sha(candidate, np),
    }


def flatten_actor_delta(parameters: Sequence[Any], reference: Sequence[Any], np: Any) -> Any:
    if len(parameters) != 6 or len(reference) != 6:
        raise ProtocolError("actor6 snapshot count drift")
    result = np.concatenate(
        [
            (
                parameter.detach().cpu().to(dtype=__import__("torch").float64)
                - anchor.detach().cpu().to(dtype=__import__("torch").float64)
            )
            .contiguous()
            .numpy()
            .reshape(-1)
            for parameter, anchor in zip(parameters, reference, strict=True)
        ]
    )
    if result.shape != (ACTOR_DIMENSION,) or not np.isfinite(result).all():
        raise ProtocolError("actual actor6 delta drift")
    return result


def retention_loss(report: Mapping[str, Any]) -> float:
    numerator = 0.0
    denominator = 0.0
    for source in ("flg", "pokemonfan", "core5"):
        for bucket in ("fragile", "c34"):
            record = report["by_source_and_bucket"][source][bucket]
            weight = float(record["effective_weight"])
            numerator += weight * float(record["ordered_loss"])
            denominator += weight
    if denominator <= 0.0:
        raise ProtocolError("retention report has no effective weight")
    return numerator / denominator


def train_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_correct: Mapping[str, bool],
    candidate_correct: Mapping[str, bool],
    selections: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    improvements = {
        "union_mixed": float(baseline["mixed_ordered_loss"])
        - float(candidate["mixed_ordered_loss"]),
        **{
            f"{source}_hard": float(
                baseline["by_source_and_bucket"][source]["hard"]["ordered_loss"]
            )
            - float(candidate["by_source_and_bucket"][source]["hard"]["ordered_loss"])
            for source in ("flg", "pokemonfan", "core5")
        },
    }
    base_retention_loss = retention_loss(baseline)
    candidate_retention_loss = retention_loss(candidate)
    retention_rows = [
        row
        for batch in selections
        for row in batch
        if str(row["category"]) in {"fragile", "c34"}
    ]
    flips = [
        str(row["line_sha256"])
        for row in retention_rows
        if baseline_correct[str(row["line_sha256"])]
        and not candidate_correct[str(row["line_sha256"])]
    ]
    hard_correct_checks = {}
    for source in ("flg", "pokemonfan", "core5"):
        base_count = int(
            baseline["by_source_and_bucket"][source]["hard"]["ordered_correct"]
        )
        candidate_count = int(
            candidate["by_source_and_bucket"][source]["hard"]["ordered_correct"]
        )
        hard_correct_checks[source] = {
            "baseline": base_count,
            "candidate": candidate_count,
            "nondegrade": candidate_count >= base_count,
        }
    checks = {
        "union_observed_improvement": improvements["union_mixed"]
        >= OBSERVED_TASK_IMPROVEMENT_MIN,
        "three_hard_observed_improvement": all(
            improvements[f"{source}_hard"] >= OBSERVED_TASK_IMPROVEMENT_MIN
            for source in ("flg", "pokemonfan", "core5")
        ),
        "retention_weighted_loss_nondegrade": candidate_retention_loss
        <= base_retention_loss + OBSERVED_RETENTION_LOSS_TOLERANCE,
        "retention_correct_to_wrong_zero": len(flips) == 0,
        "three_hard_correct_count_nondegrade": all(
            value["nondegrade"] for value in hard_correct_checks.values()
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "loss_improvements_baseline_minus_candidate": improvements,
        "minimum_required_task_improvement": OBSERVED_TASK_IMPROVEMENT_MIN,
        "baseline_retention_weighted_loss": base_retention_loss,
        "candidate_retention_weighted_loss": candidate_retention_loss,
        "retention_loss_tolerance": OBSERVED_RETENTION_LOSS_TOLERANCE,
        "retention_rows": len(retention_rows),
        "baseline_correct_retention_rows": sum(
            baseline_correct[str(row["line_sha256"])] for row in retention_rows
        ),
        "retention_correct_to_wrong_count": len(flips),
        "retention_correct_to_wrong_line_sha256": flips,
        "hard_correct_counts": hard_correct_checks,
    }


def actual_contract(
    context: Mapping[str, Any],
    cw15: ModuleType,
    parameters: Sequence[Any],
    cw11_actor: Sequence[Any],
    bootstrap_actor: Sequence[Any],
    fixture: Mapping[str, Any],
    objective_vectors: Mapping[str, Any],
    candidate_model_sha: str,
    np: Any,
) -> dict[str, Any]:
    actual_additional = flatten_actor_delta(parameters, cw11_actor, np)
    actual_step = flatten_actor_delta(parameters, bootstrap_actor, np)
    A = np.asarray(fixture["A"], dtype=np.float64)
    b = np.asarray(fixture["b"], dtype=np.float64)
    G = np.asarray(fixture["G"], dtype=np.float64)
    residual = np.asarray(fixture["residual"], dtype=np.float64)
    anchor = A @ actual_additional - b
    local = residual + G @ actual_step
    total_l2 = float(np.linalg.norm(actual_additional))
    trust_l2 = float(np.linalg.norm(actual_step))
    predicted_improvements = {
        name: -float(np.asarray(objective_vectors[name]) @ actual_step)
        for name in EXTENSION_ORDER
    }
    model = context["model"]
    state = model.state_dict()
    nonactor_names = sorted(set(state) - set(cw15.ACTOR6_NAMES))
    nonactor_sha = context["helper"].model_state_sha256(
        {name: state[name] for name in nonactor_names}
    )
    checks = {
        "actor6_tensor_count": len(parameters) == 6,
        "nonactor_tensor_count74": len(nonactor_names) == 74,
        "model_tensor_count80": len(state) == 80,
        "all_actor_float32": all(str(parameter.dtype) == "torch.float32" for parameter in parameters),
        "nonactor_exact_raw": nonactor_sha == EXPECTED_RAW_NONACTOR_SHA,
        "anchor_actual_shadow_raw_gate": float(anchor.min()) >= -LINEAR_GATE_TOLERANCE,
        "local_actual_shadow_raw_gate": float(local.min())
        >= float(fixture["z_star"]) - LINEAR_GATE_TOLERANCE,
        "actual_additional_total_cap": total_l2 <= TOTAL_CAP + L2_TOLERANCE,
        "actual_bootstrap_step_trust_cap": trust_l2 <= TRUST_RADIUS + L2_TOLERANCE,
        "actual_step_nonzero": trust_l2 > 0.0,
        "actual_four_task_predicted_improvement": all(
            predicted_improvements[name]
            >= PREDICTED_TASK_IMPROVEMENT_MIN - 1e-12
            for name in TASK_ORDER
        ),
        "actual_retention_predicted_nonincrease": predicted_improvements[
            RETENTION_OBJECTIVE
        ]
        >= -1e-12,
        "terminal_model_changed": candidate_model_sha
        not in {EXPECTED_RAW_MODEL_SHA, EXPECTED_CW11_MODEL_SHA, EXPECTED_BOOTSTRAP_MODEL_SHA},
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "actual_additional_from_CW11_l2": total_l2,
        "actual_step_from_bootstrap_l2": trust_l2,
        "actual_additional_float64_le_sha256": float64_sha(actual_additional, np),
        "actual_bootstrap_step_float64_le_sha256": float64_sha(actual_step, np),
        "anchor_actual_shadow_residual_min": float(anchor.min()),
        "local_actual_shadow_prediction_min": float(local.min()),
        "z_star": float(fixture["z_star"]),
        "actual_predicted_loss_improvements": predicted_improvements,
        "nonactor_sha256": nonactor_sha,
        "candidate_model_state_sha256": candidate_model_sha,
    }


def run_callback(
    context: Mapping[str, Any],
    cw15: ModuleType,
    dual: ModuleType,
    grad: ModuleType,
    fixture: Mapping[str, Any],
    cache: Sequence[Mapping[str, Any]],
    selections: Sequence[Sequence[Mapping[str, Any]]],
    inventory: Mapping[str, Mapping[str, Any]],
    modules: Mapping[str, ModuleType],
) -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    torch = context["helper"].torch
    device = next(context["model"].parameters()).device
    model = context["model"]
    context_checks = {
        "device_cuda": device.type == "cuda",
        "exact_CW11_model": context["helper"].model_state_sha256(model.state_dict())
        == EXPECTED_CW11_MODEL_SHA,
        "exact_CW11_vector": dual.float64_sha256(
            np.asarray(context["terminal_cumulative_float64"], dtype=np.float64)
        )
        == EXPECTED_CW11_VECTOR_SHA,
        "raw_model_anchor": str(context["raw_model_state_sha256"])
        == EXPECTED_RAW_MODEL_SHA,
        "raw_nonactor_anchor": str(context["raw_nonactor_sha256"])
        == EXPECTED_RAW_NONACTOR_SHA,
        "model_eval": model.training is False,
        "legacy_ledger34": len(context["active_pair_ledger"]) == 34,
    }
    if not all(context_checks.values()):
        raise ProtocolError(f"exact CW11 callback context drift: {context_checks}")

    parameters_sequence = modules["geometry"].configure_actor6(model)
    if tuple(cw15.ACTOR6_NAMES) != tuple(grad.sweep.ACTOR_NAMES):
        raise ProtocolError("CW15/gradient actor6 order drift")
    cw11_actor = [parameter.detach().clone() for parameter in parameters_sequence]
    cw11_actor_bytes = cw15.actor_float32_le_bytes(parameters_sequence, np)
    raw_actor = context["raw_actor"]
    cw11_total = np.asarray(context["terminal_cumulative_float64"], dtype=np.float64)
    result: dict[str, Any] | None = None
    precision_before = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        if torch.get_float32_matmul_precision() != "high":
            raise ProtocolError("failed to enter high matmul precision scope")
        total_bootstrap = cw15.apply_additional_from_cw11(
            modules,
            parameters_sequence,
            raw_actor,
            cw11_total,
            np.asarray(fixture["center"], dtype=np.float64),
            torch,
        )
        bootstrap_model_sha = context["helper"].model_state_sha256(model.state_dict())
        if bootstrap_model_sha != EXPECTED_BOOTSTRAP_MODEL_SHA:
            raise ProtocolError("e381 bootstrap model hash drift")
        bootstrap_actor = [parameter.detach().clone() for parameter in parameters_sequence]

        baseline_report, baseline_correct = grad.sweep.evaluate_selected_union(
            model, cache, selections, device
        )
        parameters_map = grad.sweep.configure_actor6(model)
        objective_gradients, baseline_losses, autograd_audit = (
            grad.compute_objective_gradients(
                model, parameters_map, cache, selections, inventory, device
            )
        )
        if context["helper"].model_state_sha256(model.state_dict()) != bootstrap_model_sha:
            raise ProtocolError("gradient pass changed bootstrap model")
        retention_gradient, retention_gradient_loss, retention_weight = (
            grad.derive_union_retention(objective_gradients, baseline_losses, inventory)
        )
        effect_gradients = {
            **objective_gradients,
            RETENTION_OBJECTIVE: retention_gradient,
        }
        pcgrad, pcgrad_audit = grad.fixed_order_pcgrad(objective_gradients)
        pcgrad_effects = grad.candidate_effects(pcgrad, effect_gradients)
        direction_gate = {
            "task_order_exact": tuple(pcgrad_audit["task_order"]) == TASK_ORDER,
            "original_unprojected_references": pcgrad_audit["reference_gradient_kind"]
            == "original_unprojected",
            "arithmetic_mean": pcgrad_audit["aggregation"]
            == "arithmetic_mean_of_projected_task_gradients",
            "four_tasks_robust_first_order_descent": pcgrad_effects[
                "all_required_numerically_robust_first_order_descent"
            ]
            is True,
            "minimum_cosine": float(pcgrad_effects["minimum_required_cosine"])
            > PCGRAD_COSINE_EPSILON,
        }
        objective_vectors = {
            name: flatten_gradient(effect_gradients[name], grad.sweep.ACTOR_NAMES, np)
            for name in EXTENSION_ORDER
        }
        pcgrad_vector = flatten_gradient(pcgrad, grad.sweep.ACTOR_NAMES, np)

        base_loss_alignment = {
            "union": abs(
                float(baseline_report["mixed_ordered_loss"])
                - float(baseline_losses["union_mixed"])
            ),
            **{
                f"{source}_hard": abs(
                    float(
                        baseline_report["by_source_and_bucket"][source]["hard"][
                            "ordered_loss"
                        ]
                    )
                    - float(baseline_losses[f"{source}_hard"])
                )
                for source in ("flg", "pokemonfan", "core5")
            },
            "retention": abs(retention_loss(baseline_report) - retention_gradient_loss),
        }
        if max(base_loss_alignment.values()) > 2e-6:
            raise ProtocolError(f"baseline loss definition drift: {base_loss_alignment}")

        candidate, projection = (None, {"decision": "NO_PCGRAD_DIRECTION"})
        if all(direction_gate.values()):
            candidate, projection = project_fixed_endpoint(
                fixture, objective_vectors, pcgrad_vector, np, optimize
            )
        common = {
            "context_checks": context_checks,
            "matmul_precision_scope": {
                "before": precision_before,
                "during": torch.get_float32_matmul_precision(),
                "high_exact": torch.get_float32_matmul_precision() == "high",
            },
            "bootstrap": {
                "center_float64_le_sha256": float64_sha(fixture["center"], np),
                "model_state_sha256": bootstrap_model_sha,
                "terminal_total_from_raw_float64_le_sha256": float64_sha(
                    total_bootstrap, np
                ),
            },
            "baseline_train512": baseline_report,
            "baseline_objective_losses": {
                **{name: float(value) for name, value in baseline_losses.items()},
                RETENTION_OBJECTIVE: float(retention_gradient_loss),
            },
            "baseline_loss_alignment_absolute": base_loss_alignment,
            "retention_effective_weight": float(retention_weight),
            "pcgrad": {
                "direction_gate": direction_gate,
                "direction_float64_le_sha256": float64_sha(pcgrad_vector, np),
                "direction_l2": float(np.linalg.norm(pcgrad_vector)),
                "audit": pcgrad_audit,
                "effects": pcgrad_effects,
            },
            "autograd_audit": autograd_audit,
            "projection": projection,
        }
        if candidate is None:
            result = {
                **common,
                "decision": "NO_GO_CW19_PREFLIGHT",
                "reason": projection["decision"],
                "candidate_payload": None,
                "changed_candidate_train_endpoint_count": 0,
            }
            return result

        total = cw15.apply_additional_from_cw11(
            modules, parameters_sequence, raw_actor, cw11_total, candidate, torch
        )
        candidate_model_sha = context["helper"].model_state_sha256(model.state_dict())
        actual = actual_contract(
            context,
            cw15,
            parameters_sequence,
            cw11_actor,
            bootstrap_actor,
            fixture,
            objective_vectors,
            candidate_model_sha,
            np,
        )
        candidate_report, candidate_correct = grad.sweep.evaluate_selected_union(
            model, cache, selections, device
        )
        observed_gate = train_gate(
            baseline_report,
            candidate_report,
            baseline_correct,
            candidate_correct,
            selections,
        )
        pre_payload_checks = {
            "planned_contract": projection["planned_contract"]["pass"] is True,
            "actual_FP32_contract": actual["pass"] is True,
            "train_gate": observed_gate["pass"] is True,
            "nonactor_still_exact": cw15.nonactor_sha256(context)
            == EXPECTED_RAW_NONACTOR_SHA,
        }
        payload = None
        reconstruction = None
        final_checks = dict(pre_payload_checks)
        if all(pre_payload_checks.values()):
            payload = cw15.encode_terminal_payload(
                context,
                modules,
                parameters_sequence,
                cw11_actor_bytes,
                candidate,
                total,
                candidate_model_sha,
            )
            reconstruction = cw15.reconstruct_terminal_candidate(
                model,
                context["helper"],
                raw_actor,
                cw11_total,
                payload,
                cutting=modules["cutting"],
                geometry=modules["geometry"],
                ram=modules["ram"],
            )
            final_checks["pure_payload_reconstruction"] = (
                reconstruction["checks"]["model_sha_exact"] is True
                and reconstruction["model_state_sha256"] == candidate_model_sha
            )
        passed = bool(final_checks) and all(final_checks.values())
        result = {
            **common,
            "decision": (
                "GO_CW19_SPECIALIST_PREFLIGHT"
                if passed
                else "NO_GO_CW19_PREFLIGHT"
            ),
            "reason": "all_train_only_gates_passed" if passed else "endpoint_gate_failed",
            "candidate_model_state_sha256": candidate_model_sha,
            "candidate_train512": candidate_report,
            "actual_FP32_contract": actual,
            "observed_train_gate": observed_gate,
            "pre_payload_checks": pre_payload_checks,
            "final_checks": final_checks,
            "pure_payload_reconstruction": reconstruction,
            "candidate_payload": payload if passed else None,
            "changed_candidate_train_endpoint_count": 1,
        }
        return result
    finally:
        torch.set_float32_matmul_precision(precision_before)
        precision_restored = torch.get_float32_matmul_precision() == precision_before
        modules["ram"].set_actor_overlay(
            parameters_sequence, cw11_actor, cw11_actor, 0.0, torch
        )
        restored_sha = context["helper"].model_state_sha256(model.state_dict())
        if restored_sha != EXPECTED_CW11_MODEL_SHA:
            raise ProtocolError(f"callback failed to restore exact CW11: {restored_sha}")
        if not precision_restored:
            raise ProtocolError("failed to restore incoming matmul precision")


def production_run() -> dict[str, Any]:
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    runner, runner_evidence = import_locked(
        RUNNER, "run_u468_raw_trainhard_actor6_balanced_mix_sweep"
    )
    grad, grad_evidence = import_locked(GRADIENT, "cw19_frozen_gradient_probe")
    if grad.sweep is not runner:
        raise ProtocolError("gradient probe did not bind the locked runner module")
    cw15, cw15_evidence = import_locked(CW15, "cw19_frozen_cw15")
    dual, dual_evidence = import_locked(DUAL, "cw19_frozen_dual")
    fixture, fixture_evidence = decode_fixture(dual)

    selections, selection_summary, selection_evidence = grad.load_frozen_selection()
    cache, cache_audit, cache_inputs = grad.load_frozen_cache(selections)
    inventory = grad.objective_inventory(cache, selections)
    modules = cw15.frozen_modules()
    primary_source, primary_evidence = modules["cw11"].read_regular_bytes(
        cw15.PRIMARY,
        cw15.MODULE_SHAS[cw15.PRIMARY],
        "CW19 frozen primary source",
        expected_mode=0o555,
    )
    holder: dict[str, Any] = {}

    def consume(context: Mapping[str, Any]) -> None:
        if holder:
            raise ProtocolError("CW19 callback called more than once")
        holder["endpoint"] = run_callback(
            context,
            cw15,
            dual,
            grad,
            fixture,
            cache,
            selections,
            inventory,
            modules,
        )

    historical = modules["cw11"].run_probe(
        modules["primary"],
        primary_source,
        primary_evidence,
        candidate_consumer=consume,
    )
    historical_checks = {
        "callback_once": "endpoint" in holder,
        "exact_historical_CW11_status": historical.get("status")
        == "exploratory_33row_specialist_valid_CW11_success",
        "outer_restore_pass": historical.get("final_integrity", {}).get("pass") is True,
    }
    if not all(historical_checks.values()):
        raise ProtocolError(f"historical CW11 replay chain failed: {historical_checks}")
    endpoint = holder["endpoint"]
    decision = endpoint["decision"]
    if decision not in {
        "GO_CW19_SPECIALIST_PREFLIGHT",
        "NO_GO_CW19_PREFLIGHT",
    }:
        raise ProtocolError(f"unexpected endpoint decision: {decision}")
    result = {
        "schema_version": SCHEMA,
        "status": decision,
        "decision": decision,
        "seed": SEED,
        "endpoint": endpoint,
        "historical_exact_CW11_replay": {
            "checks": historical_checks,
            "pass": True,
            "consumed_evidence_only": True,
            "status": historical["status"],
            "changed_candidate_official_or_validation_evaluation_count": 0,
        },
        "train_cache": {
            "selection_summary": selection_summary,
            "cache_audit": cache_audit,
            "objective_inventory": inventory,
            "CW19_non_train_members_opened": False,
            "rows": 512,
            "batches": 2,
        },
        "contract": {
            "base": "exact_CW11_then_frozen_e381_bootstrap",
            "training_direction": "fixed_order_PCGrad_union_plus_three_hards",
            "task_order": list(TASK_ORDER),
            "retention_role": "projection_and_observed_guard_only",
            "target": "full_trust_radius_normalized_PCGrad_step",
            "projection_calls": 1 if endpoint["projection"].get("one_projection_call") else 0,
            "hyperparameter_sweeps": 0,
            "changed_candidate_train_endpoint_count": endpoint[
                "changed_candidate_train_endpoint_count"
            ],
            "changed_candidate_official_or_validation_evaluation_count": 0,
            "official_candidate_budget_consumed": 0,
            "optimizer_instances_created": 0,
            "backward_calls": 0,
            "optimizer_step_calls": 0,
            "checkpoint_writes": 0,
            "model_writes": 0,
            "result_artifact_writes": 0,
            "submission_performed": False,
            "stdout_only": True,
        },
        "integrity": {
            "source": source_audit(),
            "frozen_inputs": {
                "runner": runner_evidence,
                "gradient": grad_evidence,
                "cw15": cw15_evidence,
                "dual": dual_evidence,
                "fixture": fixture_evidence,
                "selection": selection_evidence,
                "cache_inputs": cache_inputs,
                "primary": primary_evidence,
            },
            "runtime": runtime_audit(require_cuda=True),
        },
        "writes_performed": 0,
        "submission_performed": False,
    }
    if not grad.sweep.repair.finite_nested(result):
        raise ProtocolError("production result contains nonfinite values")
    return result


def selftest_result() -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    base = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    vectors = [
        np.asarray([1.0, float(index + 1), 1.0], dtype=np.float64)
        for index in range(len(EXTENSION_ORDER))
    ]
    extended, audit = extend_basis(
        base, vectors, np, production_contract=False
    )
    # A separate tiny convex projection tests the fixed SLSQP API and analytic
    # Jacobians without touching any model, cache, or production fixture.
    target = np.asarray([0.4, 0.3], dtype=np.float64)
    seed = np.asarray([0.0, 0.0], dtype=np.float64)
    tiny = optimize.minimize(
        lambda value: 0.5 * float((value - target) @ (value - target)),
        seed,
        jac=lambda value: value - target,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda value: float(0.25 - value @ value),
                "jac": lambda value: -2.0 * value,
            },
            {
                "type": "ineq",
                "fun": lambda value: np.asarray([value[0], value[1]]),
                "jac": lambda value: np.eye(2),
            },
        ],
        method="SLSQP",
        options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER, "disp": False},
    )
    checks = {
        "basis_extension_pass": audit["pass"] is True,
        "basis_reaches_dimension3": extended.shape == (3, 3),
        "tiny_projection_success": bool(tiny.success),
        "tiny_projection_matches_target": float(np.linalg.norm(tiny.x - target)) <= 1e-9,
        "tiny_projection_feasible": float(tiny.x @ tiny.x) <= 0.25 + 1e-12,
    }
    if not all(checks.values()):
        raise ProtocolError(f"selftest failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": "selftest_passed",
        "checks": checks,
        "pass": True,
        "basis_audit": audit,
        "tiny_projection": {
            "status": int(tiny.status),
            "iterations": int(tiny.nit),
            "x": [float(value) for value in tiny.x],
        },
        "source": source_audit(),
        "runtime": runtime_audit(require_cuda=False),
        "writes_performed": 0,
        "submission_performed": False,
    }


def static_result() -> dict[str, Any]:
    dependencies = {}
    for path in FROZEN:
        _, evidence = read_locked(path, str(path.relative_to(ROOT)))
        dependencies[str(path.relative_to(ROOT))] = evidence
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_passed",
        "pass": True,
        "source": source_audit(),
        "runtime": runtime_audit(require_cuda=False),
        "frozen_direct_dependencies": dependencies,
        "protocol": {
            "task_order": list(TASK_ORDER),
            "target_trust_radius": TRUST_RADIUS,
            "projection_linear_reserve": PROJECTION_LINEAR_RESERVE,
            "final_linear_gate_tolerance": LINEAR_GATE_TOLERANCE,
            "single_projection": True,
            "no_sweep": True,
            "no_official_evaluation": True,
            "stdout_only": True,
        },
        "writes_performed": 0,
        "submission_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "selftest", "run"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ProtocolError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise ProtocolError("requires Python -I -B")
    if args.mode == "run" and args.device != "cuda":
        raise ProtocolError("production run is CUDA-only")
    if args.mode != "run" and args.device != "cpu":
        raise ProtocolError("static/selftest are CPU-only")
    if args.mode == "static":
        result = static_result()
    elif args.mode == "selftest":
        result = selftest_result()
    else:
        result = production_run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
