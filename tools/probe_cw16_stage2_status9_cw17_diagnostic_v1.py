#!/usr/bin/env python3
"""One-shot numerical diagnostic for the consumed CW16 stage-2 status 9.

The run path replays only the already-counted CW15/CW16 bootstrap model.  It
patches the frozen CW16 optimizer at the fixed-z stage-2 boundary, captures the
discarded SLSQP result and a reduced CPU fixture, and reports primal/dual
certificates.  It cannot evaluate a changed model, call a candidate consumer,
materialize a checkpoint, access broad/Gold data, use the network, or submit.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
ARTIFACTS = ROOT / "artifacts"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "probe_cw16_stage2_status9_cw17_diagnostic_v1.py"
CW16_SOLVER = TOOLS / "probe_u468_cw11_consumed_valid_official6_cw16_local_trust_v1.py"
CW16_SOLVER_SHA256 = "a05df3df944451fabefa8e2a037ad541b1f90af7b966be6fc9ecb87bafe2ebbf"
MATH_SOURCE = TOOLS / "cw17_stage2_dual_math_v1.py"
EXPECTED_MATH_SHA256 = "a14241d500f352e6ae1b5662d5d7ea51ca4da22b61126536997e69e39750a600"

CW16_ATTEMPT = ARTIFACTS / ".ptcg-cw16_consumed_valid_official6_local_trust_20260802_v1-attempt.json"
CW16_ATTEMPT_SHA256 = "ecee069b30e7cb0b33b33365d58bbae048569c0e4c5b2f2b96307f3bb40102a1"
CW16_STDOUT = ARTIFACTS / "cw16_consumed_valid_official6_local_trust_20260802_v1.stdout.json"
CW16_STDOUT_SHA256 = "62eaf348a4c125c8da72ce15ed8904e4ddfd32320d1121a2bba1ca2ce34dbb4e"
CW16_STDERR = ARTIFACTS / "cw16_consumed_valid_official6_local_trust_20260802_v1.stderr-audit.json"
CW16_STDERR_SHA256 = "fa10668b9c51a2fc6047962decdfd33e864211e4ee38e78f6b68299a587e89c3"
CW16_LAUNCHER_SHA256 = "0aa701207f485572df130c9b83ad2bc6de2ffc076bb84d50e79675925ba28cb7"
CW16_RAW_STDERR_SHA256 = "5ceb096b8ac5376a3ee276bf70012f90cb2fd479e3fa2804f2c0a8145aff4512"
CW16_OLD40_ORDERED_SHA256 = "24f08ab5e2ea2dc9762adecc6b2488e1188fe6c0cf5d519de737dfb60c51217b"
CW16_ACTIVE50_SHA256 = "765b74c9a500699e65f43c27edfb7da4b568842dd2065cf55e46efb826d1ef09"
CW16_CLOSE_REASON = (
    "fail_closed_hard_anchor_local_trust_subproblem:RuntimeError:"
    "CW16 fixed-z minimum-norm stage2 failed: 9 Iteration limit reached"
)

SCHEMA = "ptcg-cw17-stage2-status9-diagnostic-v1"
STATIC_STATUS = "static_ready_CW17_stage2_diagnostic_implemented"
RUN_STATUS = "diagnostic_complete_CW17_stage2_certificate_reported"
FROZEN_MODE = 0o555
EVIDENCE_MODE = 0o444
COORDINATE_SCALE = 0.001
AFFINE_CAPTURE_ABS_TOL = 1e-12
CLASSIFICATION = {
    "specialist_valid_consumed": True,
    "numerical_diagnostic_only": True,
    "unique_official_model_count_before": 1,
    "unique_official_model_count_after": 1,
    "unique_official_model_count_delta": 0,
    "changed_model_official_evaluation": False,
    "candidate_consumer": False,
    "checkpoint_materialization": False,
    "broad_access": False,
    "gold_access": False,
    "network_upload_submission": False,
}


class DiagnosticError(RuntimeError):
    pass


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_evidence(
    path: Path,
    expected_sha256: str | None,
    expected_mode: int | None,
) -> tuple[bytes, dict[str, Any]]:
    observed = path.lstat()
    mode = stat.S_IMODE(observed.st_mode)
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    checks = {
        "regular": stat.S_ISREG(observed.st_mode),
        "one_link": observed.st_nlink == 1,
        "mode_exact": expected_mode is None or mode == expected_mode,
        "sha256_exact": expected_sha256 is None or digest == expected_sha256,
        "size_stable": len(payload) == observed.st_size,
    }
    if not all(checks.values()):
        raise DiagnosticError(f"frozen identity drift for {path}: {checks}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": f"{mode:04o}",
        "nlink": int(observed.st_nlink),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "checks": checks,
    }


def import_held(source: bytes, path: Path, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def load_modules(*, require_frozen_math: bool) -> tuple[ModuleType, ModuleType, dict[str, Any], dict[str, Any]]:
    cw16_source, cw16_evidence = regular_evidence(
        CW16_SOLVER, CW16_SOLVER_SHA256, FROZEN_MODE
    )
    math_expected = None if EXPECTED_MATH_SHA256.startswith("PENDING_") else EXPECTED_MATH_SHA256
    if require_frozen_math and math_expected is None:
        raise DiagnosticError("CW17 dual-math source lock is pending")
    math_source, math_evidence = regular_evidence(
        MATH_SOURCE,
        math_expected,
        FROZEN_MODE if require_frozen_math else None,
    )
    return (
        import_held(cw16_source, CW16_SOLVER, "cw17_diag_frozen_cw16"),
        import_held(math_source, MATH_SOURCE, "cw17_diag_dual_math"),
        cw16_evidence,
        math_evidence,
    )


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"{label} is not an object")
    return value


def validate_cw16_closure() -> dict[str, Any]:
    attempt_raw, attempt_evidence = regular_evidence(
        CW16_ATTEMPT, CW16_ATTEMPT_SHA256, EVIDENCE_MODE
    )
    stdout_raw, stdout_evidence = regular_evidence(
        CW16_STDOUT, CW16_STDOUT_SHA256, EVIDENCE_MODE
    )
    stderr_raw, stderr_evidence = regular_evidence(
        CW16_STDERR, CW16_STDERR_SHA256, EVIDENCE_MODE
    )
    attempt = strict_json(attempt_raw, "CW16 attempt")
    stdout = strict_json(stdout_raw, "CW16 stdout")
    stderr = strict_json(stderr_raw, "CW16 stderr audit")
    decision = stdout.get("second_stage", {}).get("decision", {})
    terminal = stdout.get("terminal", {})
    iterations = stdout.get("second_stage", {}).get("iterations", [])
    fail_rows = [
        value
        for value in iterations
        if isinstance(value, Mapping)
        and value.get("kind") == "fail_closed_before_official_proposal"
    ]
    try:
        raw_stderr = base64.b64decode(
            str(stderr.get("base64", "")).encode("ascii"), validate=True
        )
    except (ValueError, UnicodeEncodeError) as exc:
        raise DiagnosticError("CW16 stderr base64 invalid") from exc
    checks = {
        "attempt_schema_status_exact": attempt.get("schema_version")
        == "ptcg-cw16-consumed-valid-official6-one-shot-attempt-v1"
        and attempt.get("status") == "one_shot_attempt_consumed_before_cuda_and_official6",
        "attempt_one_no_retry_before_cuda": attempt.get("attempts_authorized") == 1
        and attempt.get("retry_authorized") is False
        and attempt.get("marker_created_before_cuda_and_official6") is True,
        "attempt_solver_exact": attempt.get("solver", {}).get("sha256")
        == CW16_SOLVER_SHA256
        and attempt.get("solver", {}).get("mode_octal") == "0555",
        "attempt_launcher_exact": attempt.get("launcher", {}).get("sha256")
        == CW16_LAUNCHER_SHA256
        and attempt.get("launcher", {}).get("mode_octal") == "0555",
        "attempt_old40_exact": attempt.get("independent_frozen_input_rehash", {}).get(
            "binding_count"
        )
        == 40
        and attempt.get("independent_frozen_input_rehash", {}).get(
            "ordered_identity_sha256"
        )
        == CW16_OLD40_ORDERED_SHA256
        and attempt.get("independent_frozen_input_rehash", {}).get("pass") is True,
        "stdout_schema_status_exact": stdout.get("schema_version")
        == "ptcg-cw16-consumed-valid-official-b256-local-trust-v1"
        and stdout.get("status") == "closed_no_CW16_candidate",
        "close_reason_exact_status9": decision.get("close_reason") == CW16_CLOSE_REASON
        and len(fail_rows) == 1
        and fail_rows[0].get("close_reason") == CW16_CLOSE_REASON,
        "official_count_exact1": decision.get("official_candidate_evaluation_count") == 1
        and decision.get("bootstrap_evaluation_count") == 1,
        "consumer_and_terminal_absent": decision.get("candidate_consumer_called") is False
        and decision.get("eligible_only_for_formal_fulltrain_revalidation") is False
        and terminal.get("reconstruction_payload") is None
        and terminal.get("reconstruction_audit") is None
        and terminal.get("optimization_affine_tangent_ledger") is None,
        "active50_exact": decision.get("terminal_active_cut_count") == 50
        and decision.get("terminal_active_CW11_anchor_count") == 50
        and decision.get("terminal_active_CW11_anchor_ledger_sha256")
        == CW16_ACTIVE50_SHA256,
        "final_integrity_pass": stdout.get("final_integrity", {}).get("pass") is True
        and all(stdout.get("final_integrity", {}).get("checks", {}).values()),
        "classification_no_broad_gold_write": stdout.get("classification", {}).get(
            "broad_access"
        )
        is False
        and stdout.get("classification", {}).get("gold_access") is False
        and stdout.get("classification", {}).get("package_upload_submission") is False
        and stdout.get("writes_performed") is False,
        "stderr_schema_status_exact": stderr.get("schema_version")
        == "ptcg-cw16-consumed-valid-official6-one-shot-stderr-audit-v1"
        and stderr.get("status") == "captured_losslessly_not_a_success_veto",
        "stderr_lossless_no_traceback": len(raw_stderr) == 347
        and sha256_bytes(raw_stderr)
        == stderr.get("sha256")
        == CW16_RAW_STDERR_SHA256
        and b"Traceback (most recent call last):" not in raw_stderr,
        "stderr_post_integrity_pass": stderr.get("post_child_integrity", {}).get("pass")
        is True
        and all(stderr.get("post_child_integrity", {}).get("checks", {}).values()),
    }
    if not all(checks.values()):
        raise DiagnosticError(f"CW16 closure drift: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "attempt": attempt_evidence,
        "stdout": stdout_evidence,
        "stderr_audit": stderr_evidence,
        "stdout_document": stdout,
    }


def validate_frozen_inputs_43(cw16: ModuleType) -> dict[str, Any]:
    cw15, _ = cw16.load_cw15_parent()
    old40 = cw16.validate_frozen_inputs_40(cw15)
    if old40.get("binding_count") != 40 or old40.get("all_exact") is not True:
        raise DiagnosticError("CW16 old40 manifest is not exact")
    closure = validate_cw16_closure()
    records = [
        *copy.deepcopy(old40["records"]),
        *[
            {
                **copy.deepcopy(closure[name]),
                "role": "consumed_CW16_closure",
            }
            for name in ("attempt", "stdout", "stderr_audit")
        ],
    ]
    if len(records) != 43 or len({value["path"] for value in records}) != 43:
        raise DiagnosticError("CW17 diagnostic 43-record manifest drift")
    ordered_identity = [
        [value["path"], value["sha256"], value.get("mode", value.get("mode_octal"))]
        for value in records
    ]
    return {
        "all_exact": True,
        "binding_count": 43,
        "old_input_count": 40,
        "new_CW16_closure_count": 3,
        "old40_ordered_identity_sha256": old40["ordered_identity_sha256"],
        "ordered_identity_sha256": sha256_bytes(canonical_json(ordered_identity)),
        "records": records,
    }


def runtime_checks(*, cuda_required: bool) -> dict[str, Any]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT.resolve(),
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated_exact": sys.flags.isolated == 1,
        "dont_write_bytecode_exact": sys.dont_write_bytecode is True,
        "cublas_workspace_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
    }
    if not all(checks.values()):
        raise DiagnosticError(f"runtime drift: {checks}")
    return {"checks": checks, "cuda_required": cuda_required}


def validate_dual_math_contract(dual_math: ModuleType) -> dict[str, Any]:
    checks = {
        "schema_exact": dual_math.SCHEMA == "ptcg-cw17-stage2-dual-math-v1",
        "coordinate_scale_exact": dual_math.COORDINATE_SCALE == COORDINATE_SCALE,
        "dual_gap_exact": dual_math.CERT_DUAL_GAP_TOL_SCALED == 5e-13,
        "raw_radius_exact": dual_math.CERT_RAW_OPTIMAL_RADIUS_MAX == 1e-9,
        "stationarity_exact": dual_math.CERT_STATIONARITY_INF_TOL_SCALED == 1e-10,
        "complementarity_exact": dual_math.CERT_COMPLEMENTARITY_TOL_SCALED
        == 1e-10,
        "linear_primal_tolerance_exact": dual_math.CERT_LINEAR_PRIMAL_TOL_SCALED
        == 1e-9,
        "trust_tolerance_exact": dual_math.CERT_TRUST_TOL_SCALED == 1e-9,
    }
    if not all(checks.values()):
        raise DiagnosticError(f"dual-math contract drift: {checks}")
    return {"checks": checks, "pass": True}


class OptimizeProxy:
    def __init__(self, real: Any):
        self.real = real
        self.calls: list[dict[str, Any]] = []

    def minimize(self, fun: Any, x0: Any, *args: Any, **kwargs: Any) -> Any:
        result = self.real.minimize(fun, x0, *args, **kwargs)
        self.calls.append(
            {
                "fun": fun,
                "x0": np.asarray(x0, dtype=np.float64).copy(),
                "args": args,
                "kwargs": kwargs,
                "result": result,
            }
        )
        return result


class LinalgCaptureProxy:
    def __init__(self, real: Any):
        self.real = real
        self.svd_calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)

    def svd(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        result = self.real.svd(value, *args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 3:
            raise DiagnosticError("captured SVD result structure drift")
        self.svd_calls.append(
            {
                "input": np.asarray(value, dtype=np.float64).copy(),
                "args": args,
                "kwargs": copy.deepcopy(kwargs),
                "u": np.asarray(result[0], dtype=np.float64).copy(),
                "singular_values": np.asarray(result[1], dtype=np.float64).copy(),
                "vh": np.asarray(result[2], dtype=np.float64).copy(),
            }
        )
        return result


class NumpyCaptureProxy:
    def __init__(self, real: Any):
        self.real = real
        self.linalg = LinalgCaptureProxy(real.linalg)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)


def conservative_basis_operator_norm(basis: Any) -> dict[str, Any]:
    value = np.asarray(basis, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) <= 0 or not np.isfinite(value).all():
        raise DiagnosticError("basis operator-norm input drift")

    def upward(item: float) -> float:
        if not math.isfinite(item):
            raise DiagnosticError("operator-norm bound became nonfinite")
        return math.nextafter(float(item), math.inf)

    def downward(item: float) -> float:
        if not math.isfinite(item):
            raise DiagnosticError("operator-norm denominator became nonfinite")
        return math.nextafter(float(item), -math.inf)

    dimension = int(value.shape[1])
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    numerator = upward(4.0 * dimension * unit_roundoff)
    denominator = downward(1.0 - numerator)
    if denominator <= 0.0:
        raise DiagnosticError("operator-norm dot-product gamma is invalid")
    gamma = upward(numerator / denominator)
    gram = value @ value.T
    absolute_product_sums = np.abs(value) @ np.abs(value).T
    if not np.isfinite(gram).all() or not np.isfinite(absolute_product_sums).all():
        raise DiagnosticError("operator-norm Gram audit became nonfinite")
    rank = int(value.shape[0])
    row_upper_bounds: list[float] = []
    entry_error_max = 0.0
    for row in range(rank):
        row_sum = 0.0
        for column in range(rank):
            absolute_sum_upper = upward(
                upward(float(absolute_product_sums[row, column]))
                / downward(1.0 - gamma)
            )
            error_upper = upward(gamma * absolute_sum_upper)
            entry_error_max = max(entry_error_max, error_upper)
            if row == column:
                entry_upper = upward(float(gram[row, column]) + error_upper)
            else:
                entry_upper = upward(abs(float(gram[row, column])) + error_upper)
            row_sum = upward(row_sum + entry_upper)
        row_upper_bounds.append(row_sum)
    gram_eigenvalue_upper = max(row_upper_bounds)
    if gram_eigenvalue_upper < 0.0:
        raise DiagnosticError("operator-norm Gram upper bound is negative")
    square_root_input = upward(gram_eigenvalue_upper)
    operator_norm_upper = math.sqrt(square_root_input)
    while Fraction.from_float(operator_norm_upper) ** 2 < Fraction.from_float(
        square_root_input
    ):
        operator_norm_upper = math.nextafter(operator_norm_upper, math.inf)
    operator_norm_upper = upward(operator_norm_upper)
    observed_spectral_norm = float(np.linalg.norm(value, ord=2))
    checks = {
        "finite_positive_upper_bound": math.isfinite(operator_norm_upper)
        and operator_norm_upper > 0.0,
        "upper_not_below_observed_spectral_norm": operator_norm_upper
        >= observed_spectral_norm,
        "gamma_strictly_between_zero_and_one": 0.0 < gamma < 1.0,
        "gram_shapes_exact": gram.shape == (rank, rank)
        and absolute_product_sums.shape == (rank, rank),
    }
    if not all(checks.values()):
        raise DiagnosticError(f"basis operator-norm certificate failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "method": "Gershgorin_with_gamma_4n_float64_dot_error_enclosure",
        "basis_shape": list(value.shape),
        "unit_roundoff": unit_roundoff,
        "dot_error_gamma_upper": gamma,
        "entry_error_upper_max": entry_error_max,
        "gram_eigenvalue_upper": gram_eigenvalue_upper,
        "operator_norm_upper": operator_norm_upper,
        "observed_spectral_norm_audit_only": observed_spectral_norm,
        "row_upper_bound_min": min(row_upper_bounds),
        "row_upper_bound_max": max(row_upper_bounds),
    }


def _rebuild_fixture_and_certificates(
    cw16: ModuleType,
    dual_math: ModuleType,
    anchor_gradients: Any,
    anchor_rhs: Any,
    local_gradients: Any,
    local_residual_at_center: Any,
    center: Any,
    trust_radius: float,
    np: Any,
    calls: list[dict[str, Any]],
    svd_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(calls) != 2:
        raise DiagnosticError(f"expected exactly two SLSQP calls, got {len(calls)}")
    if len(svd_calls) != 1:
        raise DiagnosticError(f"expected exactly one captured SVD, got {len(svd_calls)}")
    stage1 = calls[0]["result"]
    stage2 = calls[1]["result"]
    A = np.asarray(anchor_gradients, dtype=np.float64)
    b = np.asarray(anchor_rhs, dtype=np.float64)
    G = np.asarray(local_gradients, dtype=np.float64)
    residual = np.asarray(local_residual_at_center, dtype=np.float64)
    xcenter = np.asarray(center, dtype=np.float64)
    span_rows = np.concatenate([A, G, xcenter.reshape(1, -1)], axis=0)
    captured_svd = svd_calls[0]
    singular_values = np.asarray(
        captured_svd["singular_values"], dtype=np.float64
    )
    vh = np.asarray(captured_svd["vh"], dtype=np.float64)
    rank = int(
        (singular_values > singular_values[0] * cw16.SVD_RELATIVE_RANK_TOL).sum()
    )
    basis = vh[:rank]
    basis_operator_norm = conservative_basis_operator_norm(basis)
    A_reduced = A @ basis.T
    G_reduced = G @ basis.T
    center_reduced = basis @ xcenter
    center_reconstruction_error = float(np.linalg.norm(basis.T @ center_reduced - xcenter))
    coordinate_scale = cw16.TOTAL_L2_CAP
    center_scaled = center_reduced / coordinate_scale
    A_scaled = A_reduced * coordinate_scale
    G_scaled = G_reduced * coordinate_scale
    floor = np.finfo(np.float64).eps
    anchor_row_scales = np.maximum.reduce(
        [np.abs(b), np.linalg.norm(A_scaled, axis=1), np.full(b.shape, floor)]
    )
    rho = trust_radius / coordinate_scale
    local_motion_scales = np.linalg.norm(G_scaled, axis=1) * rho
    z_scale = float(
        max(float(np.abs(residual).max()), float(local_motion_scales.max()), floor)
    )
    local_scale = z_scale

    stage1_scaled = np.asarray(stage1.x[:-1], dtype=np.float64)
    stage1_raw = basis.T @ (coordinate_scale * stage1_scaled)
    stage1_z_solver_raw = float(stage1.x[-1]) * z_scale
    stage1_local_prediction_raw = residual + G @ (stage1_raw - xcenter)
    z_star = min(0.0, stage1_z_solver_raw, float(stage1_local_prediction_raw.min()))
    z_adjustment = stage1_z_solver_raw - z_star

    stage2_x = np.asarray(stage2.x, dtype=np.float64)
    stage2_constraints = calls[1]["kwargs"].get("constraints")
    if not isinstance(stage2_constraints, (list, tuple)) or len(stage2_constraints) != 4:
        raise DiagnosticError("captured stage2 constraint structure drift")
    linear_constraints = stage2_constraints[:2]
    quadratic_constraints = stage2_constraints[2:]
    if not all(
        isinstance(value, Mapping)
        and value.get("type") == "ineq"
        and callable(value.get("fun"))
        and callable(value.get("jac"))
        for value in linear_constraints
    ):
        raise DiagnosticError("captured linear constraint structure drift")
    if not all(
        isinstance(value, Mapping)
        and value.get("type") == "ineq"
        and callable(value.get("fun"))
        and callable(value.get("jac"))
        for value in quadratic_constraints
    ):
        raise DiagnosticError("captured ball constraint structure drift")
    zero_scaled = np.zeros(rank, dtype=np.float64)
    K_anchor = np.asarray(linear_constraints[0]["jac"](zero_scaled), dtype=np.float64)
    K_local = np.asarray(linear_constraints[1]["jac"](zero_scaled), dtype=np.float64)
    fun_zero_anchor = np.asarray(
        linear_constraints[0]["fun"](zero_scaled), dtype=np.float64
    )
    fun_zero_local = np.asarray(
        linear_constraints[1]["fun"](zero_scaled), dtype=np.float64
    )
    captured_trust_center = (
        np.asarray(quadratic_constraints[1]["jac"](zero_scaled), dtype=np.float64)
        / 2.0
    )
    captured_trust_radius_squared = float(
        quadratic_constraints[1]["fun"](captured_trust_center)
    )
    if (
        K_anchor.shape != (A.shape[0], rank)
        or K_local.shape != (G.shape[0], rank)
        or fun_zero_anchor.shape != (A.shape[0],)
        or fun_zero_local.shape != (G.shape[0],)
        or stage1_scaled.shape != (rank,)
        or stage2_x.shape != (rank,)
        or captured_trust_center.shape != (rank,)
        or not np.isfinite(captured_trust_center).all()
        or not math.isfinite(captured_trust_radius_squared)
        or captured_trust_radius_squared <= 0.0
    ):
        raise DiagnosticError("captured reduced constraint dimensions drift")
    captured_trust_rho = math.sqrt(captured_trust_radius_squared)
    K = np.concatenate([K_anchor, K_local], axis=0)
    q = -np.concatenate([fun_zero_anchor, fun_zero_local], axis=0)
    independent_K_anchor = A_scaled / anchor_row_scales[:, None]
    independent_q_anchor = b / anchor_row_scales
    independent_K_local = G_scaled / local_scale
    independent_q_local = (
        z_star - residual + G_scaled @ center_scaled
    ) / local_scale
    independent_K = np.concatenate(
        [independent_K_anchor, independent_K_local], axis=0
    )
    independent_q = np.concatenate(
        [independent_q_anchor, independent_q_local], axis=0
    )
    affine_errors: list[float] = []
    jacobian_constant_checks: list[bool] = []
    total_fun_checks: list[bool] = []
    total_jac_checks: list[bool] = []
    trust_fun_checks: list[bool] = []
    trust_jac_checks: list[bool] = []
    for point in (zero_scaled, stage1_scaled, stage2_x):
        for offset, constraint in (
            (0, linear_constraints[0]),
            (A.shape[0], linear_constraints[1]),
        ):
            row_count = A.shape[0] if offset == 0 else G.shape[0]
            expected = K[offset : offset + row_count] @ point - q[
                offset : offset + row_count
            ]
            observed = np.asarray(constraint["fun"](point), dtype=np.float64)
            affine_errors.append(float(np.abs(observed - expected).max()))
            jacobian_constant_checks.append(
                np.array_equal(
                    np.asarray(constraint["jac"](point), dtype=np.float64),
                    K[offset : offset + row_count],
                )
            )
        expected_total_fun = float(1.0 - point @ point)
        expected_total_jac = -2.0 * point
        trust_delta = point - center_scaled
        expected_trust_fun = float(rho**2 - trust_delta @ trust_delta)
        expected_trust_jac = -2.0 * trust_delta
        total_fun_checks.append(
            float(quadratic_constraints[0]["fun"](point)) == expected_total_fun
        )
        total_jac_checks.append(
            np.array_equal(
                np.asarray(quadratic_constraints[0]["jac"](point), dtype=np.float64),
                expected_total_jac,
            )
        )
        trust_fun_checks.append(
            float(quadratic_constraints[1]["fun"](point)) == expected_trust_fun
        )
        trust_jac_checks.append(
            np.array_equal(
                np.asarray(quadratic_constraints[1]["jac"](point), dtype=np.float64),
                expected_trust_jac,
            )
        )
    affine_error_max = max(affine_errors)
    constraint_capture_checks = {
        "captured_original_svd_input_exact": np.array_equal(
            np.asarray(captured_svd["input"], dtype=np.float64), span_rows
        ),
        "captured_original_svd_call_exact": captured_svd["args"] == ()
        and captured_svd["kwargs"] == {"full_matrices": False},
        "captured_original_svd_shapes_exact": singular_values.ndim == 1
        and vh.shape == (singular_values.size, xcenter.size),
        "stage2_method_options_exact": calls[1]["kwargs"].get("method") == "SLSQP"
        and calls[1]["kwargs"].get("options")
        == {"ftol": cw16.SLSQP_FTOL, "maxiter": cw16.SLSQP_MAXITER, "disp": False},
        "closure_jacobians_constant_exact": all(jacobian_constant_checks),
        "closure_jacobians_match_independent_exact": np.array_equal(
            K, independent_K
        ),
        "closure_rhs_matches_independent_within_1e_12": float(
            np.abs(q - independent_q).max()
        )
        <= AFFINE_CAPTURE_ABS_TOL,
        "closure_affine_evaluations_within_1e_12": affine_error_max
        <= AFFINE_CAPTURE_ABS_TOL,
        "captured_trust_center_exact": np.array_equal(
            captured_trust_center, center_scaled
        ),
        "captured_trust_radius_squared_exact": captured_trust_radius_squared
        == rho**2
        and captured_trust_rho == rho,
        "total_ball_fun_jac_exact_at_three_points": all(total_fun_checks)
        and all(total_jac_checks),
        "trust_ball_fun_jac_exact_at_three_points": all(trust_fun_checks)
        and all(trust_jac_checks),
        "total_ball_intentionally_omitted_from_dual": True,
        "trust_ball_captured_into_dual_fixture": True,
    }
    if not all(constraint_capture_checks.values()):
        raise DiagnosticError(
            f"captured numerical problem identity drift: {constraint_capture_checks}"
        )
    center_scaled = captured_trust_center
    rho = captured_trust_rho
    fixture = dual_math.encode_fixture(K, q, center_scaled, rho)
    decoded = dual_math.decode_fixture(fixture)
    if not (
        np.array_equal(decoded[0], K)
        and np.array_equal(decoded[1], q)
        and np.array_equal(decoded[2], center_scaled)
        and decoded[3] == rho
    ):
        raise DiagnosticError("reduced fixture roundtrip drift")

    raw_multipliers = np.asarray(
        getattr(stage2, "multipliers", np.asarray([], dtype=np.float64)),
        dtype=np.float64,
    )
    expected_multiplier_count = A.shape[0] + G.shape[0] + 2
    multiplier_shape_exact = raw_multipliers.shape == (expected_multiplier_count,)
    multiplier_finite = bool(np.isfinite(raw_multipliers).all())
    multiplier_usable = multiplier_shape_exact and multiplier_finite
    if multiplier_usable:
        warm = np.concatenate(
            [raw_multipliers[: A.shape[0] + G.shape[0]], [2.0 * raw_multipliers[-1]]]
        )
    else:
        warm = np.zeros(K.shape[0] + 1, dtype=np.float64)
    raw_multipliers_encoded = (
        dual_math.encode_array(raw_multipliers)
        if raw_multipliers.size > 0 and multiplier_finite
        else None
    )

    def original_space(candidate_scaled: Any) -> dict[str, Any]:
        candidate_scaled = np.asarray(candidate_scaled, dtype=np.float64)
        candidate_raw = basis.T @ (coordinate_scale * candidate_scaled)
        anchor_residual = A @ candidate_raw - b
        local_prediction = residual + G @ (candidate_raw - xcenter)
        total_l2 = float(np.linalg.norm(candidate_raw))
        trust_l2 = float(np.linalg.norm(candidate_raw - xcenter))
        checks = {
            "anchor_existing_hard_gate": float(anchor_residual.min())
            >= -cw16.LINEAR_RESIDUAL_TOL,
            "local_existing_hard_gate": float(local_prediction.min())
            >= z_star - cw16.LINEAR_RESIDUAL_TOL,
            "total_existing_hard_gate": total_l2
            <= cw16.TOTAL_L2_CAP + cw16.L2_ABS_TOL,
            "trust_existing_hard_gate": trust_l2 <= trust_radius + cw16.L2_ABS_TOL,
            "no_clip": True,
        }
        return {
            "checks": checks,
            "pass": all(checks.values()),
            "candidate_scaled": dual_math.encode_array(candidate_scaled),
            "candidate_raw": dual_math.encode_array(candidate_raw),
            "candidate_raw_float64_le_sha256": dual_math.float64_sha256(candidate_raw),
            "anchor_residual_min": float(anchor_residual.min()),
            "local_prediction_min": float(local_prediction.min()),
            "z_star": z_star,
            "total_l2": total_l2,
            "trust_l2": trust_l2,
        }

    def build_candidate_record(candidate_scaled: Any, candidate_warm: Any) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        value = np.asarray(candidate_scaled, dtype=np.float64)
        record: dict[str, Any] = {
            "candidate_available": True,
            "candidate_scaled": None,
            "certificate_available": False,
            "certificate": None,
            "mapped_raw_radius_certificate": None,
            "original_space_available": False,
            "original_space": None,
            "errors": errors,
        }
        try:
            record["candidate_scaled"] = dual_math.encode_array(value)
        except Exception as exc:
            errors.append(
                {
                    "phase": "encode_candidate",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            record["candidate_available"] = False
            return record
        try:
            record["original_space"] = original_space(value)
            record["original_space_available"] = True
        except Exception as exc:
            errors.append(
                {
                    "phase": "original_space_reconstruction",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        try:
            certificate = dual_math.certify_primal(
                value,
                K,
                q,
                center_scaled,
                rho,
                warm_start=candidate_warm,
            )
            reduced_radius_bound = float(certificate["raw_optimal_radius_bound"])
            mapped_product = (
                0.0
                if reduced_radius_bound == 0.0
                else math.nextafter(
                    reduced_radius_bound
                    * float(basis_operator_norm["operator_norm_upper"]),
                    math.inf,
                )
            )
            mapped_radius_certificate = {
                "checks": {
                    "basis_operator_norm_certificate_pass": basis_operator_norm[
                        "pass"
                    ]
                    is True,
                    "mapped_raw_optimal_radius_at_most_1e_9": mapped_product
                    <= 1e-9,
                    "reduced_certificate_raw_radius_gate_pass": certificate[
                        "checks"
                    ]["raw_optimal_radius_at_most_1e_9"]
                    is True,
                },
                "basis_operator_norm_upper": basis_operator_norm[
                    "operator_norm_upper"
                ],
                "reduced_radius_bound": reduced_radius_bound,
                "mapped_raw_radius_bound": mapped_product,
                "multiplication_rounding": "float64_product_then_nextafter_up",
                "pass": basis_operator_norm["pass"] is True
                and certificate["checks"][
                    "raw_optimal_radius_at_most_1e_9"
                ]
                is True
                and mapped_product <= 1e-9,
            }
            record["certificate"] = certificate
            record["mapped_raw_radius_certificate"] = mapped_radius_certificate
            record["certificate_available"] = True
        except Exception as exc:
            errors.append(
                {
                    "phase": "exact_certificate_backend",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        return record

    stage2_record = build_candidate_record(stage2_x, warm)
    if stage2_record["certificate_available"]:
        try:
            selected_dual = stage2_record["certificate"]["dual"]["selected"]
            dual_recovered = dual_math.decode_array(selected_dual["u_dual"])
            dual_recovered_variables = dual_math.decode_array(
                selected_dual["variables"]
            )
            dual_recovered_record = build_candidate_record(
                dual_recovered, dual_recovered_variables
            )
        except Exception as exc:
            dual_recovered_record = {
                "candidate_available": False,
                "candidate_scaled": None,
                "certificate_available": False,
                "certificate": None,
                "mapped_raw_radius_certificate": None,
                "original_space_available": False,
                "original_space": None,
                "errors": [
                    {
                        "phase": "dual_recovered_dependency",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                ],
            }
    else:
        dual_recovered_record = {
            "candidate_available": False,
            "candidate_scaled": None,
            "certificate_available": False,
            "certificate": None,
            "mapped_raw_radius_certificate": None,
            "original_space_available": False,
            "original_space": None,
            "errors": [
                {
                    "phase": "dual_recovered_dependency",
                    "error_type": "UnavailableDependency",
                    "error_message": "old stage2 certificate unavailable",
                }
            ],
        }
    stage1_record = build_candidate_record(stage1_scaled, warm)
    candidates = {
        "old_slsqp_stage2_last_x": stage2_record,
        "dual_recovered": dual_recovered_record,
        "stage1_feasible_upper_bound": stage1_record,
    }
    priority = [
        "old_slsqp_stage2_last_x",
        "dual_recovered",
        "stage1_feasible_upper_bound",
    ]
    formally_certified = [
        name
        for name in priority
        if candidates[name]["certificate_available"]
        and candidates[name]["original_space_available"]
        and isinstance(candidates[name]["certificate"], Mapping)
        and candidates[name]["certificate"].get("formal_certificate_pass") is True
        and candidates[name]["certificate"].get("exact_feasibility", {}).get(
            "total_squared_slack_sign"
        )
        >= 0
        and isinstance(candidates[name]["mapped_raw_radius_certificate"], Mapping)
        and candidates[name]["mapped_raw_radius_certificate"].get("pass") is True
        and isinstance(candidates[name]["original_space"], Mapping)
        and candidates[name]["original_space"].get("pass") is True
    ]
    certificate_backend_errors = [
        {"candidate": name, **error}
        for name in priority
        for error in candidates[name]["errors"]
    ]
    capture_checks = {
        **constraint_capture_checks,
        "basis_operator_norm_certificate_pass": basis_operator_norm["pass"] is True,
        "stage1_success_exact": bool(stage1.success) and int(stage1.status) == 0,
        "stage2_status9_exact": bool(stage2.success) is False
        and int(stage2.status) == 9
        and str(stage2.message) == "Iteration limit reached",
        "stage2_iteration_limit_exact2000": int(stage2.nit) == cw16.SLSQP_MAXITER,
        "row_counts_exact50_plus41": A.shape[0] == 50 and G.shape[0] == 41,
        "rank_in_preregistered_range": 43 <= rank <= 81,
        "center_reconstruction_exact": center_reconstruction_error <= cw16.L2_ABS_TOL,
        "coordinate_scale_exact": coordinate_scale == COORDINATE_SCALE,
        "trust_radius_exact_C_div_8": trust_radius == cw16.TRUST_RADII[0],
        "z_certification_adjustment_numeric_only": 0.0 <= z_adjustment
        <= cw16.LINEAR_RESIDUAL_TOL,
        "fixture_roundtrip_exact": True,
        "no_candidate_materialized_or_evaluated": True,
    }
    if not all(capture_checks.values()):
        raise DiagnosticError(f"live stage2 capture drift: {capture_checks}")
    return {
        "checks": capture_checks,
        "pass": True,
        "dimensions": {
            "actor_original": int(xcenter.size),
            "anchor_rows": int(A.shape[0]),
            "local_physical_rows": int(G.shape[0]),
            "span_shape": list(span_rows.shape),
            "reduced_rank": rank,
            "stage2_multiplier_count": int(raw_multipliers.size),
            "expected_stage2_multiplier_count": expected_multiplier_count,
        },
        "singular_values": dual_math.encode_array(singular_values),
        "basis_float64_le_sha256": dual_math.float64_sha256(basis),
        "basis_operator_norm_certificate": basis_operator_norm,
        "reconstruction_map": {
            "formula": "candidate_raw=basis_reduced_by_actor.T@(coordinate_scale*candidate_scaled)",
            "basis_orientation": "reduced_rank_by_actor_original",
            "basis_reduced_by_actor": dual_math.encode_array(basis),
            "coordinate_scale": coordinate_scale,
            "actor_original_dimension": int(xcenter.size),
            "reduced_rank": rank,
            "externally_bound_by_diagnostic_stdout_sha256": True,
        },
        "center_reconstruction_error": center_reconstruction_error,
        "captured_constraint_audit": {
            "checks": constraint_capture_checks,
            "pass": True,
            "affine_error_max": affine_error_max,
            "independent_rhs_error_max": float(
                np.abs(q - independent_q).max()
            ),
            "fixture_uses_captured_closure_jacobians_and_fun_at_zero": True,
            "captured_trust_center": dual_math.encode_array(center_scaled),
            "captured_trust_radius_squared": captured_trust_radius_squared,
            "captured_trust_rho": rho,
            "total_ball_omission_reason": (
                "minimum_norm_relaxation_optimum_is_inside_total_ball_when_accepted_"
                "candidate_is_exact_total_feasible"
            ),
        },
        "scaling": {
            "coordinate_scale": coordinate_scale,
            "trust_radius_raw": trust_radius,
            "rho_scaled": rho,
            "z_scale": z_scale,
            "local_global_scale": local_scale,
            "anchor_row_scales": dual_math.encode_array(anchor_row_scales),
        },
        "fixture": fixture,
        "stage1": {
            "success": bool(stage1.success),
            "status": int(stage1.status),
            "message": str(stage1.message),
            "iterations": int(stage1.nit),
            "x_scaled": dual_math.encode_array(stage1_scaled),
            "z_solver_raw": stage1_z_solver_raw,
            "z_star": z_star,
            "z_certification_adjustment": z_adjustment,
        },
        "old_stage2": {
            "success": bool(stage2.success),
            "status": int(stage2.status),
            "message": str(stage2.message),
            "iterations": int(stage2.nit),
            "function_evaluations": int(stage2.nfev),
            "x_scaled": dual_math.encode_array(stage2_x),
            "multipliers": raw_multipliers_encoded,
            "multiplier_capture_audit": {
                "shape_exact": multiplier_shape_exact,
                "finite": multiplier_finite,
                "usable_as_warm_start": multiplier_usable,
                "fallback_to_zero_if_unusable": True,
            },
            "warm_dual_without_redundant_total_ball": dual_math.encode_array(warm),
        },
        "candidate_priority": priority,
        "candidates": candidates,
        "certificate_backend_complete": all(
            candidates[name]["certificate_available"] for name in priority
        ),
        "certificate_backend_errors": certificate_backend_errors,
        "fixture_and_reconstruction_survive_certificate_backend_errors": True,
        "formally_certified_candidates": formally_certified,
        "first_formally_certified_candidate": formally_certified[0]
        if formally_certified
        else None,
        "formal_CW17_may_be_built": bool(formally_certified),
        "diagnostic_does_not_materialize_or_consume_candidate": True,
    }


def certificate_failure_recovery_selftest(dual_math: ModuleType) -> dict[str, Any]:
    dimension = 43
    anchor_rows = 50
    local_rows = 41
    A = np.concatenate(
        [np.eye(dimension, dtype=np.float64), np.eye(dimension, dtype=np.float64)[:7]],
        axis=0,
    )
    G = np.eye(dimension, dtype=np.float64)[:local_rows]
    b = np.full(anchor_rows, -0.1, dtype=np.float64)
    residual = np.zeros(local_rows, dtype=np.float64)
    center = np.zeros(dimension, dtype=np.float64)
    span = np.concatenate([A, G, center.reshape(1, -1)], axis=0)
    svd_result = np.linalg.svd(span, full_matrices=False)
    singular_values = svd_result[1]
    vh = svd_result[2]
    rank = int((singular_values > singular_values[0] * 1e-12).sum())
    basis = vh[:rank]
    coordinate_scale = 0.001
    trust_radius = 0.000125
    center_scaled = basis @ center / coordinate_scale
    A_scaled = (A @ basis.T) * coordinate_scale
    G_scaled = (G @ basis.T) * coordinate_scale
    anchor_scales = np.maximum.reduce(
        [
            np.abs(b),
            np.linalg.norm(A_scaled, axis=1),
            np.full(b.shape, np.finfo(np.float64).eps),
        ]
    )
    local_scale = float(
        max(
            float(np.abs(residual).max()),
            float(
                (
                    np.linalg.norm(G_scaled, axis=1)
                    * (trust_radius / coordinate_scale)
                ).max()
            ),
            np.finfo(np.float64).eps,
        )
    )
    z_star = 0.0
    K_anchor = A_scaled / anchor_scales[:, None]
    K_local = G_scaled / local_scale
    q_anchor = b / anchor_scales
    q_local = (
        z_star - residual + G_scaled @ center_scaled
    ) / local_scale

    def anchor_fun(value: Any) -> Any:
        return K_anchor @ value - q_anchor

    def anchor_jac(value: Any) -> Any:
        del value
        return K_anchor

    def local_fun(value: Any) -> Any:
        return K_local @ value - q_local

    def local_jac(value: Any) -> Any:
        del value
        return K_local

    constraints = [
        {"type": "ineq", "fun": anchor_fun, "jac": anchor_jac},
        {"type": "ineq", "fun": local_fun, "jac": local_jac},
        {
            "type": "ineq",
            "fun": lambda value: 1.0 - value @ value,
            "jac": lambda value: -2.0 * value,
        },
        {
            "type": "ineq",
            "fun": lambda value: (trust_radius / coordinate_scale) ** 2
            - (value - center_scaled) @ (value - center_scaled),
            "jac": lambda value: -2.0 * (value - center_scaled),
        },
    ]
    stage1 = argparse.Namespace(
        x=np.concatenate([np.zeros(rank, dtype=np.float64), np.asarray([0.0])]),
        success=True,
        status=0,
        message="synthetic success",
        nit=1,
        nfev=1,
    )
    stage2 = argparse.Namespace(
        x=np.zeros(rank, dtype=np.float64),
        success=False,
        status=9,
        message="Iteration limit reached",
        nit=2000,
        nfev=2001,
        multipliers=np.zeros(anchor_rows + local_rows + 2, dtype=np.float64),
    )
    cw16 = argparse.Namespace(
        SVD_RELATIVE_RANK_TOL=1e-12,
        TOTAL_L2_CAP=coordinate_scale,
        SLSQP_FTOL=1e-12,
        SLSQP_MAXITER=2000,
        LINEAR_RESIDUAL_TOL=1e-9,
        L2_ABS_TOL=1e-12,
        TRUST_RADII=(trust_radius,),
    )

    class FailingCertificateProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(dual_math, name)

        def certify_primal(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise dual_math.CertificateError("injected certificate backend failure")

    recovered = _rebuild_fixture_and_certificates(
        cw16,
        FailingCertificateProxy(),
        A,
        b,
        G,
        residual,
        center,
        trust_radius,
        np,
        [
            {"result": stage1, "kwargs": {}},
            {
                "result": stage2,
                "kwargs": {
                    "constraints": constraints,
                    "method": "SLSQP",
                    "options": {
                        "ftol": cw16.SLSQP_FTOL,
                        "maxiter": cw16.SLSQP_MAXITER,
                        "disp": False,
                    },
                },
            },
        ],
        [
            {
                "input": span.copy(),
                "args": (),
                "kwargs": {"full_matrices": False},
                "u": svd_result[0].copy(),
                "singular_values": singular_values.copy(),
                "vh": vh.copy(),
            }
        ],
    )
    checks = {
        "capture_still_passes": recovered["pass"] is True,
        "fixture_retained": isinstance(recovered.get("fixture"), Mapping),
        "reconstruction_map_retained": isinstance(
            recovered.get("reconstruction_map", {}).get("basis_reduced_by_actor"),
            Mapping,
        ),
        "backend_closed_without_formal_candidate": recovered[
            "certificate_backend_complete"
        ]
        is False
        and recovered["formal_CW17_may_be_built"] is False,
        "all_three_errors_serialized": len(recovered["certificate_backend_errors"])
        == 3,
        "stage1_and_stage2_raw_reconstruction_retained": recovered["candidates"][
            "old_slsqp_stage2_last_x"
        ]["original_space_available"]
        is True
        and recovered["candidates"]["stage1_feasible_upper_bound"][
            "original_space_available"
        ]
        is True,
    }
    if not all(checks.values()):
        raise DiagnosticError(f"certificate failure recovery selftest failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "fixture_sha256": recovered["fixture"]["fixture_sha256"],
        "error_count": len(recovered["certificate_backend_errors"]),
    }


def run_diagnostic(
    cw16: ModuleType,
    dual_math: ModuleType,
    cw16_evidence: Mapping[str, Any],
    math_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = runtime_checks(cuda_required=True)
    math_contract = validate_dual_math_contract(dual_math)
    frozen43 = validate_frozen_inputs_43(cw16)
    closure = validate_cw16_closure()
    cw15, parent_evidence = cw16.load_cw15_parent()
    original_solve = cw16.solve_local_trust_maximin
    capture: dict[str, Any] = {}

    def diagnostic_solve(
        anchor_gradients: Any,
        anchor_rhs: Any,
        local_gradients: Any,
        local_residual_at_center: Any,
        center: Any,
        trust_radius: float,
        np: Any,
        optimize: Any,
    ) -> Any:
        proxy = OptimizeProxy(optimize)
        numpy_proxy = NumpyCaptureProxy(np)
        try:
            return original_solve(
                anchor_gradients,
                anchor_rhs,
                local_gradients,
                local_residual_at_center,
                center,
                trust_radius,
                numpy_proxy,
                proxy,
            )
        except RuntimeError as exc:
            if str(exc) != "CW16 fixed-z minimum-norm stage2 failed: 9 Iteration limit reached":
                raise
            if capture:
                raise DiagnosticError("stage2 diagnostic capture repeated")
            capture.update(
                _rebuild_fixture_and_certificates(
                    cw16,
                    dual_math,
                    anchor_gradients,
                    anchor_rhs,
                    local_gradients,
                    local_residual_at_center,
                    center,
                    trust_radius,
                    np,
                    proxy.calls,
                    numpy_proxy.linalg.svd_calls,
                )
            )
            raise

    cw11 = cw15.import_frozen(
        cw15.CW11_PROBE,
        cw15.MODULE_SHAS[cw15.CW11_PROBE],
        cw15.EXPECTED_INPUT_MODES[cw15.CW11_PROBE],
        "cw17_diag_frozen_cw11",
    )
    cw16_source, cw16_self_evidence = cw11.read_regular_bytes(
        CW16_SOLVER,
        CW16_SOLVER_SHA256,
        "CW17 diagnostic frozen CW16 solver",
        expected_mode=FROZEN_MODE,
    )
    primary_source, primary_evidence = cw11.read_regular_bytes(
        cw15.PRIMARY,
        cw15.MODULE_SHAS[cw15.PRIMARY],
        "CW17 diagnostic frozen primary",
        expected_mode=FROZEN_MODE,
    )
    primary = cw15.import_frozen(
        cw15.PRIMARY,
        cw15.MODULE_SHAS[cw15.PRIMARY],
        cw15.EXPECTED_INPUT_MODES[cw15.PRIMARY],
        "cw17_diag_frozen_primary",
    )
    try:
        cw16.solve_local_trust_maximin = diagnostic_solve
        replay = cw16.run_probe(
            cw15,
            cw16_source,
            cw16_self_evidence,
            primary,
            primary_source,
            primary_evidence,
            candidate_consumer=None,
        )
    finally:
        cw16.solve_local_trust_maximin = original_solve
    replay_bytes = cw16.canonical_json(replay)
    replay_checks = {
        "canonical_stdout_sha_exact_CW16": sha256_bytes(replay_bytes)
        == CW16_STDOUT_SHA256,
        "status_closed_exact": replay.get("status") == "closed_no_CW16_candidate",
        "close_reason_exact": replay.get("second_stage", {}).get("decision", {}).get(
            "close_reason"
        )
        == CW16_CLOSE_REASON,
        "official_unique_model_count_still1": replay.get("second_stage", {}).get(
            "decision", {}
        ).get("official_candidate_evaluation_count")
        == 1,
        "candidate_consumer_false": replay.get("second_stage", {}).get("decision", {}).get(
            "candidate_consumer_called"
        )
        is False,
        "capture_exactly_once": bool(capture),
        "replay_writes_false": replay.get("writes_performed") is False,
    }
    if not all(replay_checks.values()):
        raise DiagnosticError(f"CW16 replay identity drift: {replay_checks}")
    source = SCRIPT.read_bytes()
    source_audit = static_source_audit(source)
    return {
        "schema_version": SCHEMA,
        "status": RUN_STATUS,
        "classification": dict(CLASSIFICATION),
        "contract": contract(),
        "input_lock": {
            "self": regular_evidence(SCRIPT, sha256_bytes(source), FROZEN_MODE)[1],
            "CW16_solver_source": dict(cw16_evidence),
            "dual_math_source": dict(math_evidence),
            "CW16_closure": {
                key: closure[key] for key in ("attempt", "stdout", "stderr_audit")
            },
        },
        "frozen_inputs_43": frozen43,
        "cw16_closure_evidence": {
            "checks": closure["checks"],
            "pass": closure["pass"],
        },
        "replay_identity": {
            "checks": replay_checks,
            "pass": True,
            "canonical_stdout_sha256": sha256_bytes(replay_bytes),
            "canonical_stdout_bytes": len(replay_bytes),
        },
        "diagnostic": capture,
        "dual_math_selftest": dual_math.selftest(),
        "dual_math_contract": math_contract,
        "runtime": runtime,
        "source_audit": source_audit,
        "run_executed": True,
        "cuda_accessed": True,
        "writes_performed": False,
    }


def contract() -> dict[str, Any]:
    return {
        "purpose": "capture_consumed_CW16_stage2_status9_without_new_model_evaluation",
        "frozen_input_count": 43,
        "old_input_count": 40,
        "new_CW16_closure_count": 3,
        "unique_official_model_count_before": 1,
        "unique_official_model_count_after": 1,
        "unique_official_model_count_delta": 0,
        "cumulative_official_budget_max": 12,
        "cumulative_official_budget_remaining": 11,
        "identical_bootstrap_reconstruction_access": True,
        "changed_model_official_evaluation": False,
        "candidate_consumer": False,
        "checkpoint_materialization": False,
        "dual_math_source_sha256": EXPECTED_MATH_SHA256,
        "fixture_coordinate_scale": COORDINATE_SCALE,
        "formal_dual_gap_scaled_max": 5e-13,
        "formal_raw_optimal_radius_max": 1e-9,
        "formal_stationarity_inf_max": 1e-10,
        "formal_complementarity_inf_max": 1e-10,
        "raw_radius_mapping_operator_norm_method":
        "Gershgorin_with_gamma_4n_float64_dot_error_enclosure",
        "formal_raw_radius_requires_mapped_bound": True,
        "total_ball_captured_and_intentionally_omitted_from_dual": True,
        "formal_candidate_requires_exact_total_feasibility": True,
        "optimizer_status_is_not_acceptance": True,
        "exact_float_and_protocol_tolerance_feasibility_separate": True,
        "no_clip": True,
        "network_broad_gold_submission": False,
    }


def static_source_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    imports: set[str] = set()
    calls: set[str] = set()
    functions: set[str] = set()
    open_modes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
            if (
                (isinstance(node.func, ast.Name) and node.func.id == "open")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "open")
            ):
                mode_node = None
                if isinstance(node.func, ast.Name) and len(node.args) >= 2:
                    mode_node = node.args[1]
                elif isinstance(node.func, ast.Attribute) and node.args:
                    mode_node = node.args[0]
                for keyword in node.keywords:
                    if keyword.arg == "mode":
                        mode_node = keyword.value
                open_modes.append(
                    mode_node.value
                    if isinstance(mode_node, ast.Constant)
                    and isinstance(mode_node.value, str)
                    else "UNKNOWN"
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
    required = {
        "validate_cw16_closure",
        "validate_frozen_inputs_43",
        "run_diagnostic",
        "_rebuild_fixture_and_certificates",
        "certificate_failure_recovery_selftest",
        "conservative_basis_operator_norm",
        "validate_dual_math_contract",
        "static_result",
    }
    checks = {
        "parse_pass": True,
        "required_functions_present": required.issubset(functions),
        "no_network_submission_imports": not bool(
            imports.intersection(
                {"requests", "socket", "urllib", "kaggle", "subprocess"}
            )
        ),
        "no_checkpoint_write_calls": not bool(
            calls.intersection({"save", "torch_save", "to_csv", "write_text"})
        ),
        "no_filesystem_mutation_calls": not bool(
            calls.intersection(
                {
                    "write",
                    "write_bytes",
                    "unlink",
                    "rename",
                    "replace",
                    "remove",
                    "rmdir",
                    "mkdir",
                    "makedirs",
                    "chmod",
                    "system",
                    "popen",
                    "Popen",
                }
            )
        ),
        "all_open_modes_read_only": all(mode in {"r", "rb"} for mode in open_modes),
        "dual_math_lock_installed_or_pending": (
            EXPECTED_MATH_SHA256.startswith("PENDING_")
            or len(EXPECTED_MATH_SHA256) == 64
        ),
        "cw16_solver_sha_literal_exact": CW16_SOLVER_SHA256
        == "a05df3df944451fabefa8e2a037ad541b1f90af7b966be6fc9ecb87bafe2ebbf",
        "formal_thresholds_literal_exact": contract()["formal_dual_gap_scaled_max"]
        == 5e-13
        and contract()["formal_raw_optimal_radius_max"] == 1e-9,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "imports": sorted(imports),
        "calls": sorted(calls),
        "functions": sorted(functions),
        "open_modes": open_modes,
        "source_sha256": sha256_bytes(source),
    }


def selftest_result(*, require_frozen: bool) -> dict[str, Any]:
    runtime = runtime_checks(cuda_required=False)
    cw16, dual_math, cw16_evidence, math_evidence = load_modules(
        require_frozen_math=require_frozen
    )
    source, self_evidence = regular_evidence(
        SCRIPT,
        None,
        FROZEN_MODE if require_frozen else None,
    )
    if require_frozen and self_evidence["mode_octal"] != "0555":
        raise DiagnosticError("diagnostic solver is not frozen")
    source_audit = static_source_audit(source)
    frozen43 = validate_frozen_inputs_43(cw16)
    closure = validate_cw16_closure()
    math_selftest = dual_math.selftest()
    math_contract = validate_dual_math_contract(dual_math)
    identity_operator = conservative_basis_operator_norm(
        np.eye(3, dtype=np.float64)
    )
    nonorthogonal_operator = conservative_basis_operator_norm(
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    )
    numpy_proxy = NumpyCaptureProxy(np)
    svd_input = np.asarray([[3.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    svd_output = numpy_proxy.linalg.svd(svd_input, full_matrices=False)
    svd_capture = numpy_proxy.linalg.svd_calls
    certificate_failure_recovery = certificate_failure_recovery_selftest(dual_math)
    checks = {
        "source_audit_pass": source_audit["pass"] is True,
        "frozen43_exact": frozen43["all_exact"] is True
        and frozen43["binding_count"] == 43,
        "cw16_closure_pass": closure["pass"] is True,
        "dual_math_selftest_pass": math_selftest["pass"] is True,
        "dual_math_contract_pass": math_contract["pass"] is True,
        "dual_math_frozen_identity_exact": math_evidence["sha256"]
        == EXPECTED_MATH_SHA256
        and math_evidence["mode_octal"] == "0555",
        "basis_operator_norm_selftests_pass": identity_operator["pass"] is True
        and identity_operator["operator_norm_upper"] >= 1.0
        and nonorthogonal_operator["operator_norm_upper"] >= math.sqrt(2.0),
        "original_svd_capture_selftest_pass": len(svd_capture) == 1
        and np.array_equal(svd_capture[0]["input"], svd_input)
        and np.array_equal(svd_capture[0]["singular_values"], svd_output[1])
        and np.array_equal(svd_capture[0]["vh"], svd_output[2]),
        "certificate_failure_recovery_selftest_pass":
        certificate_failure_recovery["pass"] is True,
        "classification_exact": CLASSIFICATION["unique_official_model_count_delta"] == 0
        and CLASSIFICATION["changed_model_official_evaluation"] is False,
    }
    if not all(checks.values()):
        raise DiagnosticError(f"diagnostic CPU selftest failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": STATIC_STATUS
        if require_frozen
        else "writable_CW17_diagnostic_CPU_selftests_passed_not_runnable",
        "classification": dict(CLASSIFICATION),
        "contract": contract(),
        "checks": checks,
        "self_source": self_evidence,
        "CW16_solver_source": cw16_evidence,
        "dual_math_source": math_evidence,
        "frozen_inputs_43": frozen43,
        "cw16_closure_evidence": {
            "checks": closure["checks"],
            "pass": closure["pass"],
        },
        "dual_math_selftest": math_selftest,
        "dual_math_contract": math_contract,
        "basis_operator_norm_selftest": {
            "identity": identity_operator,
            "nonorthogonal": nonorthogonal_operator,
        },
        "certificate_failure_recovery_selftest": certificate_failure_recovery,
        "runtime": runtime,
        "source_audit": source_audit,
        "run_executed": False,
        "cuda_accessed": False,
        "writes_performed": False,
    }


def static_result() -> dict[str, Any]:
    return selftest_result(require_frozen=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("selftest", "static", "run"), default="static")
    args = parser.parse_args()
    if args.mode == "selftest":
        result = selftest_result(require_frozen=False)
    elif args.mode == "static":
        result = static_result()
    else:
        if stat.S_IMODE(SCRIPT.lstat().st_mode) != FROZEN_MODE:
            raise DiagnosticError("CW17 diagnostic run requires frozen mode 0555")
        cw16, dual_math, cw16_evidence, math_evidence = load_modules(
            require_frozen_math=True
        )
        result = run_diagnostic(cw16, dual_math, cw16_evidence, math_evidence)
    print(canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
