#!/usr/bin/env python3
"""Solve a captured CW17 stage-2 fixture with frozen exact certificates.

This is a pure-CPU, read-only utility.  It reads one JSON fixture, writes only
one canonical JSON object to stdout, and has no model, CUDA, checkpoint,
network, packaging, or submission capability.

Accepted solve input is either a wrapper containing ``fixture`` and
``reconstruction_map`` or a complete CW17 diagnostic whose ``diagnostic``
member contains those fields.  The fixture itself is decoded by the frozen
``cw17_stage2_dual_math_v1.py`` implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

# Preserve the stdout-only/read-only contract even if the caller omits ``-B``.
sys.dont_write_bytecode = True

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "solve_cw18_exact_dual_v1.py"
DUAL_MATH_SOURCE = TOOLS / "cw17_stage2_dual_math_v1.py"
BUILDER_SOURCE = TOOLS / "build_cw18_direct_fixture_from_cw16_stdout_v1.py"
FIXED_FIXTURE_PATH = ROOT / "artifacts/cw18_direct_fixture_20260803_v1.stdout.json"
DUAL_MATH_SHA256 = (
    "a14241d500f352e6ae1b5662d5d7ea51ca4da22b61126536997e69e39750a600"
)
DUAL_MATH_MODE = 0o555
BUILDER_SHA256 = (
    "b2b86a6ce117bb5063eaf36c3a86603af1694f6dc9be05e40221e59ee886f2aa"
)
BUILDER_MODE = 0o555
FIXTURE_MODE = 0o444
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")

SCHEMA = "ptcg-cw18-exact-dual-solver-v1"
INPUT_SCHEMA = "ptcg-cw18-exact-dual-input-v1"
STATIC_STATUS = "static_ready_CW18_exact_dual_solver"
SELFTEST_STATUS = "selftest_passed_CW18_exact_dual_solver"
SOLVED_STATUS = "solved_formally_certified_CW18_exact_dual_candidate"
NO_CANDIDATE_STATUS = "complete_no_formally_certified_CW18_candidate"
FAILURE_STATUS = "terminal_failure_CW18_exact_dual_solver"
MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_PRIMAL_SEEDS = 8
MAX_DUAL_WARM_STARTS = 4
MAX_POLISH_CALLS = 16
TOTAL_CAP_RAW = 0.001
BUILDER_SCHEMA = "ptcg-cw18-direct-fixture-from-cw16-stdout-v1"
BUILDER_STATUS = "fixture_built"
RAW_CONTRACT_SCHEMA = "ptcg-cw18-raw-space-contract-v1"
EXPECTED_BOOTSTRAP_POINT_SHA256 = (
    "e3814600afa6f1c7843ca073d89a86cb1e32cf162d89f8cbfaf8db378ad3183f"
)
EXPECTED_LOCAL_RESIDUAL_SHA256 = (
    "28f8c97b07cb9b6d2c8096141b171cecb262b0b86973de6b1b57dd7efe1329dc"
)
EXPECTED_ACTOR_DIMENSION = 65793
EXPECTED_ANCHOR_ROWS = 50
EXPECTED_LOCAL_ROWS = 41
EXPECTED_REDUCED_RANK_MIN = 43
EXPECTED_REDUCED_RANK_MAX = 81
EXPECTED_RHO_SCALED = 0.125
EXPECTED_TRUST_RADIUS_RAW = 0.000125
EXPECTED_LINEAR_RESIDUAL_TOL = 1e-8
EXPECTED_L2_ABS_TOL = 1e-12

CLASSIFICATION = {
    "cpu_only": True,
    "fixture_read_only": True,
    "stdout_only": True,
    "filesystem_writes": False,
    "cuda_access": False,
    "model_access": False,
    "checkpoint_writes": False,
    "network_access": False,
    "broad_gold_access": False,
    "package_upload_submission": False,
}


class SolverError(RuntimeError):
    """Fail-closed CW18 input, source, or certificate error."""


def canonical_json(value: Any) -> bytes:
    def numpy_scalar(item: Any) -> Any:
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            result = float(item)
            if not math.isfinite(result):
                raise ValueError("nonfinite NumPy scalar")
            return result
        raise TypeError(f"unsupported JSON value: {type(item).__name__}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=numpy_scalar,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_regular_stable(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    expected_mode: int | None = None,
    maximum_bytes: int = MAX_INPUT_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or int(before.st_size) <= 0
        or int(before.st_size) > maximum_bytes
    ):
        raise SolverError(f"{label} is not one bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise SolverError(f"{label} changed before read")
        chunks: list[bytes] = []
        remaining = int(opened.st_size)
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise SolverError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SolverError(f"{label} grew during read")
        payload = b"".join(chunks)
        after = os.lstat(path)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            or stat.S_ISLNK(after.st_mode)
            or not stat.S_ISREG(after.st_mode)
            or int(after.st_nlink) != 1
        ):
            raise SolverError(f"{label} changed during read")
    finally:
        os.close(descriptor)
    observed_sha = sha256_bytes(payload)
    observed_mode = stat.S_IMODE(before.st_mode)
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise SolverError(f"{label} SHA256 drift")
    if expected_mode is not None and observed_mode != expected_mode:
        raise SolverError(f"{label} mode drift")
    return payload, {
        "path": root_relative(path),
        "sha256": observed_sha,
        "bytes": len(payload),
        "mode_octal": format(observed_mode, "04o"),
        "nlink": int(before.st_nlink),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
    }


def strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SolverError(f"{label} is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SolverError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise SolverError(f"{label} must be one JSON object")
    return value


def import_frozen_dual_math() -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = read_regular_stable(
        DUAL_MATH_SOURCE,
        "frozen CW17 dual math",
        expected_sha256=DUAL_MATH_SHA256,
        expected_mode=DUAL_MATH_MODE,
        maximum_bytes=2 * 1024 * 1024,
    )
    module = ModuleType("cw18_held_cw17_stage2_dual_math_v1")
    module.__file__ = str(DUAL_MATH_SOURCE)
    exec(compile(source, str(DUAL_MATH_SOURCE), "exec"), module.__dict__)
    return module, evidence


def validate_frozen_builder_source() -> dict[str, Any]:
    _, evidence = read_regular_stable(
        BUILDER_SOURCE,
        "frozen CW18 direct-fixture builder",
        expected_sha256=BUILDER_SHA256,
        expected_mode=BUILDER_MODE,
        maximum_bytes=2 * 1024 * 1024,
    )
    return evidence


def validate_dual_math_contract(dual_math: ModuleType) -> dict[str, Any]:
    checks = {
        "schema_exact": dual_math.SCHEMA == "ptcg-cw17-stage2-dual-math-v1",
        "coordinate_scale_exact": dual_math.COORDINATE_SCALE == 0.001,
        "linear_primal_tolerance_exact": dual_math.CERT_LINEAR_PRIMAL_TOL_SCALED
        == 1e-9,
        "trust_tolerance_exact": dual_math.CERT_TRUST_TOL_SCALED == 1e-9,
        "dual_gap_exact": dual_math.CERT_DUAL_GAP_TOL_SCALED == 5e-13,
        "raw_radius_exact": dual_math.CERT_RAW_OPTIMAL_RADIUS_MAX == 1e-9,
        "stationarity_exact": dual_math.CERT_STATIONARITY_INF_TOL_SCALED
        == 1e-10,
        "complementarity_exact": dual_math.CERT_COMPLEMENTARITY_TOL_SCALED
        == 1e-10,
        "critical_callables_present": all(
            callable(getattr(dual_math, name, None))
            for name in (
                "encode_array",
                "decode_array",
                "encode_fixture",
                "decode_fixture",
                "polish_dual",
                "certify_primal",
                "selftest",
            )
        ),
    }
    if not all(checks.values()):
        raise SolverError(f"frozen dual-math contract drift: {checks}")
    return {"checks": checks, "pass": True}


def runtime_checks() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    expected = EXPECTED_PYTHON.resolve()
    checks = {
        "my_project_env_python_exact": executable == expected,
        "numpy_float64_exact8": np.dtype(np.float64).itemsize == 8,
        "cuda_modules_not_imported": "torch" not in sys.modules
        and "cupy" not in sys.modules,
    }
    if not all(checks.values()):
        raise SolverError(f"CW18 runtime contract failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "python_executable": str(executable),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }


def conservative_basis_operator_norm(basis: Any) -> dict[str, Any]:
    """Upper-bound ||basis.T||_2 with the frozen CW17 Gershgorin method."""

    value = np.asarray(basis, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) <= 0 or not np.isfinite(value).all():
        raise SolverError("basis operator-norm input invalid")

    def upward(item: float) -> float:
        if not math.isfinite(item):
            raise SolverError("operator-norm upper bound became nonfinite")
        return math.nextafter(float(item), math.inf)

    def downward(item: float) -> float:
        if not math.isfinite(item):
            raise SolverError("operator-norm denominator became nonfinite")
        return math.nextafter(float(item), -math.inf)

    actor_dimension = int(value.shape[1])
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    numerator = upward(4.0 * actor_dimension * unit_roundoff)
    denominator = downward(1.0 - numerator)
    if denominator <= 0.0:
        raise SolverError("operator-norm dot-product gamma invalid")
    gamma = upward(numerator / denominator)
    gram = value @ value.T
    absolute_product_sums = np.abs(value) @ np.abs(value).T
    if not np.isfinite(gram).all() or not np.isfinite(absolute_product_sums).all():
        raise SolverError("operator-norm Gram audit became nonfinite")
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
        raise SolverError("operator-norm Gram upper bound is negative")
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
        raise SolverError(f"basis operator-norm certificate failed: {checks}")
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


def require_pass_checks(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SolverError(f"{label} is not an object")
    checks = value.get("checks")
    if (
        value.get("pass") is not True
        or not isinstance(checks, Mapping)
        or not checks
        or not all(item is True for item in checks.values())
    ):
        raise SolverError(f"{label} pass/checks drift")
    return value


def validate_project_provenance(
    payload: Mapping[str, Any], *, production: bool
) -> dict[str, Any]:
    runtime = require_pass_checks(payload.get("runtime"), "builder runtime")
    source_audit = require_pass_checks(payload.get("source_audit"), "builder source_audit")
    plan_audit = require_pass_checks(payload.get("plan_audit"), "builder plan_audit")
    captured = require_pass_checks(
        payload.get("captured_constraint_audit"), "builder captured_constraint_audit"
    )
    precision = require_pass_checks(payload.get("precision_scope"), "builder precision_scope")
    graph = require_pass_checks(
        payload.get("graph_fingerprint_audit"), "builder graph_fingerprint_audit"
    )
    reconstruction = payload.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise SolverError("builder reconstruction is not an object")
    dimensions = reconstruction.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise SolverError("builder reconstruction dimensions missing")
    checks = {
        "builder_schema_exact": payload.get("schema_version") == BUILDER_SCHEMA,
        "builder_status_exact": payload.get("status") == BUILDER_STATUS,
        "builder_pass_true": payload.get("pass") is True,
        "builder_runtime_pass": runtime.get("pass") is True,
        "builder_source_sha_exact": source_audit.get("source_sha256")
        == BUILDER_SHA256,
        "builder_source_checks_pass": source_audit.get("pass") is True,
        "plan_audit_pass": plan_audit.get("pass") is True,
        "captured_closure_pass": captured.get("pass") is True,
        "precision_high_scoped": precision.get("during") == "high"
        and precision.get("after") == precision.get("before"),
        "graph_fingerprint_pass": graph.get("pass") is True,
        "changed_model_official_count_zero": payload.get(
            "changed_model_official_evaluation_count"
        ) == 0,
        "changed_model_consumer_not_called": payload.get(
            "changed_model_candidate_consumer_called"
        ) is False,
        "exact_CW11_consumer_called": payload.get(
            "exact_CW11_fixture_consumer_called"
        ) is True,
        "writes_zero": payload.get("writes_performed") == 0,
        "submission_false": payload.get("submission_performed") is False,
        "reconstruction_pass": reconstruction.get("pass") is True,
        "no_changed_model_eval_in_reconstruction": reconstruction.get(
            "changed_model_officially_evaluated"
        ) is False,
        "selected_context_only": reconstruction.get(
            "selected_context_autograd_for_fixture"
        ) is True,
        "no_full_stream_snapshot": reconstruction.get(
            "full_stream_snapshot_constructed"
        ) is False,
    }
    if production:
        checks.update(
            {
                "builder_runtime_cuda_BF16_contract": runtime.get(
                    "cuda_required"
                ) is True
                and isinstance(runtime.get("cuda_checks"), Mapping)
                and all(value is True for value in runtime["cuda_checks"].values()),
                "actor_dimension_exact65793": dimensions.get("actor_original")
                == EXPECTED_ACTOR_DIMENSION,
                "anchor_rows_exact50": dimensions.get("anchor_rows")
                == EXPECTED_ANCHOR_ROWS,
                "local_rows_exact41": dimensions.get("local_physical_rows")
                == EXPECTED_LOCAL_ROWS,
                "bootstrap_literal_exact": payload.get(
                    "expected_bootstrap_point_sha256"
                ) == EXPECTED_BOOTSTRAP_POINT_SHA256,
                "local_residual_frozen_sha_exact": reconstruction.get(
                    "local_residual_float64_le_sha256"
                ) == EXPECTED_LOCAL_RESIDUAL_SHA256,
            }
        )
    if not all(checks.values()):
        raise SolverError(f"CW18 project provenance drift: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "builder_sha256": BUILDER_SHA256,
        "dimensions": dict(dimensions),
    }


def decode_raw_space_contract(
    payload: Mapping[str, Any], dual_math: ModuleType, *, production: bool
) -> dict[str, Any]:
    contract = payload.get("raw_space_contract")
    contract = require_pass_checks(contract, "raw_space_contract")
    arrays = contract.get("arrays")
    sha_ledger = contract.get("array_float64_le_sha256")
    if not isinstance(arrays, Mapping) or not isinstance(sha_ledger, Mapping):
        raise SolverError("raw-space array ledger missing")
    names = (
        "anchor_gradients_A",
        "anchor_rhs_b",
        "local_gradients_G",
        "local_residual_at_center",
    )
    if set(arrays) != set(names) or set(sha_ledger) != set(names):
        raise SolverError("raw-space array names drift")
    decoded = {name: dual_math.decode_array(arrays[name]) for name in names}
    A = decoded["anchor_gradients_A"]
    b = decoded["anchor_rhs_b"]
    G = decoded["local_gradients_G"]
    residual = decoded["local_residual_at_center"]
    sha_checks = {
        name: dual_math.float64_sha256(decoded[name]) == sha_ledger[name]
        == arrays[name].get("sha256")
        for name in names
    }
    checks = {
        "schema_exact": contract.get("schema_version") == RAW_CONTRACT_SCHEMA,
        "anchor_shapes": A.ndim == 2 and b.shape == (A.shape[0],),
        "local_shapes": G.ndim == 2 and residual.shape == (G.shape[0],)
        and G.shape[1] == A.shape[1],
        "all_finite": all(np.isfinite(value).all() for value in decoded.values()),
        "all_array_shas_exact": all(sha_checks.values()),
        "actor_dimension_self_consistent": contract.get(
            "actor_original_dimension"
        ) == A.shape[1],
        "anchor_count_self_consistent": contract.get("anchor_row_count")
        == A.shape[0],
        "local_count_self_consistent": contract.get("local_physical_row_count")
        == G.shape[0],
        "total_cap_exact": contract.get("TOTAL_L2_CAP") == TOTAL_CAP_RAW,
        "linear_tolerance_exact": contract.get("LINEAR_RESIDUAL_TOL")
        == EXPECTED_LINEAR_RESIDUAL_TOL,
        "l2_tolerance_exact": contract.get("L2_ABS_TOL")
        == EXPECTED_L2_ABS_TOL,
        "trust_radius_exact": contract.get(
            "trust_radius_relative_actual_center"
        ) == EXPECTED_TRUST_RADIUS_RAW,
        "z_star_finite_nonpositive": isinstance(contract.get("z_star"), (int, float))
        and math.isfinite(float(contract["z_star"]))
        and float(contract["z_star"]) <= 0.0,
        "raw_gate_formulas_exact": contract.get("raw_gate_formulas")
        == {
            "anchor": "A@candidate_raw-b>=-LINEAR_RESIDUAL_TOL",
            "local": (
                "residual+G@(candidate_raw-center_raw)>=z_star-"
                "LINEAR_RESIDUAL_TOL"
            ),
            "total": "norm(candidate_raw)<=TOTAL_L2_CAP+L2_ABS_TOL",
            "trust": (
                "norm(candidate_raw-center_raw)<=trust_radius+L2_ABS_TOL"
            ),
        },
    }
    if production:
        checks.update(
            {
                "actor_dimension_exact65793": A.shape[1]
                == EXPECTED_ACTOR_DIMENSION,
                "anchor_rows_exact50": A.shape[0] == EXPECTED_ANCHOR_ROWS,
                "local_rows_exact41": G.shape[0] == EXPECTED_LOCAL_ROWS,
                "residual_sha_frozen_exact": dual_math.float64_sha256(residual)
                == EXPECTED_LOCAL_RESIDUAL_SHA256,
            }
        )
    if not all(checks.values()):
        raise SolverError(f"raw-space contract drift: {checks}")
    return {
        "contract": contract,
        "A": A,
        "b": b,
        "G": G,
        "residual": residual,
        "z_star": float(contract["z_star"]),
        "linear_tolerance": float(contract["LINEAR_RESIDUAL_TOL"]),
        "l2_tolerance": float(contract["L2_ABS_TOL"]),
        "trust_radius_raw": float(contract["trust_radius_relative_actual_center"]),
        "checks": checks,
        "sha_checks": sha_checks,
    }


def problem_payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostic = document.get("diagnostic")
    if diagnostic is not None:
        if not isinstance(diagnostic, Mapping):
            raise SolverError("diagnostic member is not an object")
        return diagnostic
    return document


def decode_problem(
    document: Mapping[str, Any], dual_math: ModuleType, *, production: bool = False
) -> dict[str, Any]:
    if production and document.get("diagnostic") is not None:
        raise SolverError("production CW18 fixture must be the direct builder object")
    payload = problem_payload(document)
    provenance = validate_project_provenance(payload, production=production)
    raw_space = decode_raw_space_contract(
        payload, dual_math, production=production
    )
    fixture = payload.get("fixture")
    reconstruction = payload.get("reconstruction_map")
    if not isinstance(fixture, Mapping) or not isinstance(reconstruction, Mapping):
        raise SolverError("input lacks fixture plus reconstruction_map")
    K, q, center, rho = dual_math.decode_fixture(fixture)
    required_reconstruction = {
        "basis_orientation",
        "basis_reduced_by_actor",
        "basis_float64_le_sha256",
        "center_raw",
        "center_raw_float64_le_sha256",
        "coordinate_scale",
        "actor_original_dimension",
        "reduced_rank",
    }
    if not required_reconstruction.issubset(reconstruction):
        raise SolverError("reconstruction_map required fields missing")
    if reconstruction.get("basis_orientation") != "reduced_rank_by_actor_original":
        raise SolverError("reconstruction basis orientation drift")
    basis = dual_math.decode_array(reconstruction["basis_reduced_by_actor"])
    center_raw = dual_math.decode_array(reconstruction["center_raw"])
    if (
        basis.ndim != 2
        or basis.shape[0] != K.shape[1]
        or center_raw.shape != (basis.shape[1],)
        or not np.isfinite(center_raw).all()
        or reconstruction.get("coordinate_scale") != dual_math.COORDINATE_SCALE
        or type(reconstruction.get("actor_original_dimension")) is not int
        or reconstruction["actor_original_dimension"] != basis.shape[1]
        or type(reconstruction.get("reduced_rank")) is not int
        or reconstruction["reduced_rank"] != basis.shape[0]
    ):
        raise SolverError("reconstruction_map dimensions or scale drift")
    expected_basis_sha = payload.get("basis_float64_le_sha256")
    observed_basis_sha = dual_math.float64_sha256(basis)
    if (
        not isinstance(expected_basis_sha, str)
        or len(expected_basis_sha) != 64
        or expected_basis_sha != observed_basis_sha
        or reconstruction.get("basis_float64_le_sha256") != observed_basis_sha
    ):
        raise SolverError("external basis SHA256 drift")
    expected_center_sha = reconstruction.get("center_raw_float64_le_sha256")
    observed_center_sha = dual_math.float64_sha256(center_raw)
    if (
        not isinstance(expected_center_sha, str)
        or len(expected_center_sha) != 64
        or expected_center_sha != observed_center_sha
    ):
        raise SolverError("reconstruction raw-center SHA256 drift")
    expected_bootstrap_sha = payload.get("expected_bootstrap_point_sha256")
    if (
        not isinstance(expected_bootstrap_sha, str)
        or len(expected_bootstrap_sha) != 64
        or expected_bootstrap_sha != observed_center_sha
    ):
        raise SolverError("expected bootstrap point SHA256 is absent or mismatched")
    coordinate_scale = float(reconstruction["coordinate_scale"])
    center_scaled_from_raw = basis @ center_raw / coordinate_scale
    projected_center_raw = basis.T @ (coordinate_scale * center)
    center_projection_error = float(np.linalg.norm(projected_center_raw - center_raw))
    center_binding_checks = {
        "fixture_center_exact_from_raw_basis_and_scale": np.array_equal(
            center_scaled_from_raw, center
        ),
        "basis_projection_error_at_most_1e_12": math.isfinite(
            center_projection_error
        )
        and center_projection_error <= 1e-12,
        "raw_center_sha_exact_expected_bootstrap": observed_center_sha
        == expected_bootstrap_sha,
    }
    if not all(center_binding_checks.values()):
        raise SolverError(f"raw-center reconstruction binding failed: {center_binding_checks}")
    A = raw_space["A"]
    b = raw_space["b"]
    G = raw_space["G"]
    residual = raw_space["residual"]
    if center_raw.shape != (A.shape[1],):
        raise SolverError("raw center and raw-space actor dimensions differ")
    A_scaled = (A @ basis.T) * coordinate_scale
    G_scaled = (G @ basis.T) * coordinate_scale
    floor = np.finfo(np.float64).eps
    anchor_scales = np.maximum.reduce(
        [np.abs(b), np.linalg.norm(A_scaled, axis=1), np.full(b.shape, floor)]
    )
    local_motion = np.linalg.norm(G_scaled, axis=1) * rho
    z_scale = float(
        max(float(np.abs(residual).max()), float(local_motion.max()), floor)
    )
    zero_scaled = np.zeros(basis.shape[0], dtype=np.float64)
    K_anchor = A_scaled / anchor_scales[:, None]
    K_local = G_scaled / z_scale
    fun_zero_anchor = (A_scaled @ zero_scaled - b) / anchor_scales
    fun_zero_local = (
        residual
        + G_scaled @ (zero_scaled - center)
        - raw_space["z_star"]
    ) / z_scale
    rebuilt_K = np.concatenate([K_anchor, K_local], axis=0)
    rebuilt_q = -np.concatenate([fun_zero_anchor, fun_zero_local], axis=0)
    raw_fixture_binding_checks = {
        "K_shape_exact": rebuilt_K.shape == K.shape,
        "q_shape_exact": rebuilt_q.shape == q.shape,
        "K_bit_exact_from_raw_contract": np.array_equal(rebuilt_K, K),
        "q_bit_exact_from_stage2_closure": np.array_equal(rebuilt_q, q),
        "fixture_rows_equal_raw_rows": K.shape[0] == A.shape[0] + G.shape[0],
        "fixture_dimension_equal_basis_rank": K.shape[1] == basis.shape[0],
        "rho_matches_raw_trust_over_scale": rho
        == raw_space["trust_radius_raw"] / coordinate_scale,
    }
    stage1 = payload.get("stage1")
    if not isinstance(stage1, Mapping):
        raise SolverError("builder stage1 evidence missing")
    stage1_raw = require_pass_checks(
        stage1.get("raw_certification"), "builder stage1 raw certification"
    )
    raw_fixture_binding_checks.update(
        {
            "stage1_success_status0": stage1.get("success") is True
            and stage1.get("status") == 0,
            "stage1_z_star_exact_contract": stage1.get("z_star")
            == raw_space["z_star"],
            "stage1_raw_z_star_exact_contract": stage1_raw.get("z_star")
            == raw_space["z_star"],
        }
    )
    if production:
        raw_fixture_binding_checks.update(
            {
                "fixture_rho_exact_0p125": rho == EXPECTED_RHO_SCALED,
                "reduced_rank_preregistered": EXPECTED_REDUCED_RANK_MIN
                <= basis.shape[0]
                <= EXPECTED_REDUCED_RANK_MAX,
            }
        )
    if not all(raw_fixture_binding_checks.values()):
        raise SolverError(
            f"raw-space to reduced-fixture binding failed: {raw_fixture_binding_checks}"
        )
    operator_norm = conservative_basis_operator_norm(basis)
    return {
        "payload": payload,
        "fixture": fixture,
        "fixture_sha256": fixture["fixture_sha256"],
        "K": K,
        "q": q,
        "center": center,
        "rho": rho,
        "basis": basis,
        "basis_float64_le_sha256": observed_basis_sha,
        "center_raw": center_raw,
        "center_raw_float64_le_sha256": observed_center_sha,
        "expected_bootstrap_point_sha256": expected_bootstrap_sha,
        "center_projection_error": center_projection_error,
        "center_binding_checks": center_binding_checks,
        "operator_norm": operator_norm,
        "coordinate_scale": coordinate_scale,
        "actor_original_dimension": int(reconstruction["actor_original_dimension"]),
        "reduced_rank": int(reconstruction["reduced_rank"]),
        "project_provenance": provenance,
        "raw_space": raw_space,
        "raw_fixture_binding_checks": raw_fixture_binding_checks,
    }


def optional_encoded_array(
    container: Mapping[str, Any], key: str, dual_math: ModuleType
) -> np.ndarray | None:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SolverError(f"optional encoded array {key} has invalid structure")
    return dual_math.decode_array(value)


def add_unique_vector(
    records: list[tuple[str, np.ndarray]],
    seen: set[str],
    name: str,
    value: Any,
    expected_shape: tuple[int, ...],
    dual_math: ModuleType,
) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise SolverError(f"seed {name} shape or finiteness drift")
    identity = dual_math.float64_sha256(array)
    if identity not in seen:
        records.append((name, array.copy()))
        seen.add(identity)


def collect_starts(problem: Mapping[str, Any], dual_math: ModuleType) -> dict[str, Any]:
    payload = problem["payload"]
    K = problem["K"]
    q = problem["q"]
    center = problem["center"]
    dimension = int(K.shape[1])
    primal_records: list[tuple[str, np.ndarray]] = []
    primal_seen: set[str] = set()
    add_unique_vector(
        primal_records,
        primal_seen,
        "zero",
        np.zeros(dimension, dtype=np.float64),
        (dimension,),
        dual_math,
    )
    add_unique_vector(
        primal_records,
        primal_seen,
        "fixture_center",
        center,
        (dimension,),
        dual_math,
    )
    least_squares, _, _, _ = np.linalg.lstsq(K, q, rcond=None)
    add_unique_vector(
        primal_records,
        primal_seen,
        "all_rows_least_squares",
        least_squares,
        (dimension,),
        dual_math,
    )
    for section_name, key in (
        ("stage1", "x_scaled"),
        ("old_stage2", "x_scaled"),
    ):
        section = payload.get(section_name)
        if section is not None:
            if not isinstance(section, Mapping):
                raise SolverError(f"optional {section_name} section is not an object")
            value = optional_encoded_array(section, key, dual_math)
            if value is not None:
                add_unique_vector(
                    primal_records,
                    primal_seen,
                    f"{section_name}.{key}",
                    value,
                    (dimension,),
                    dual_math,
                )
    extra_primal = payload.get("primal_seeds")
    if extra_primal is not None:
        if not isinstance(extra_primal, list) or len(extra_primal) > MAX_PRIMAL_SEEDS:
            raise SolverError("primal_seeds must be a bounded list")
        for index, record in enumerate(extra_primal):
            if (
                not isinstance(record, Mapping)
                or not isinstance(record.get("name"), str)
                or not isinstance(record.get("array"), Mapping)
            ):
                raise SolverError("primal_seeds record schema drift")
            value = dual_math.decode_array(record["array"])
            add_unique_vector(
                primal_records,
                primal_seen,
                f"input.{index}.{record['name']}",
                value,
                (dimension,),
                dual_math,
            )
    if len(primal_records) > MAX_PRIMAL_SEEDS:
        primal_records = primal_records[:MAX_PRIMAL_SEEDS]

    dual_size = int(K.shape[0] + 1)
    warm_records: list[tuple[str, np.ndarray]] = []
    warm_seen: set[str] = set()
    old_stage2 = payload.get("old_stage2")
    if old_stage2 is not None:
        if not isinstance(old_stage2, Mapping):
            raise SolverError("optional old_stage2 section is not an object")
        warm = optional_encoded_array(
            old_stage2,
            "warm_dual_without_redundant_total_ball",
            dual_math,
        )
        if warm is not None:
            add_unique_vector(
                warm_records,
                warm_seen,
                "old_stage2.warm_dual_without_redundant_total_ball",
                warm,
                (dual_size,),
                dual_math,
            )
    extra_warm = payload.get("dual_warm_starts")
    if extra_warm is not None:
        if not isinstance(extra_warm, list) or len(extra_warm) > MAX_DUAL_WARM_STARTS:
            raise SolverError("dual_warm_starts must be a bounded list")
        for index, record in enumerate(extra_warm):
            if (
                not isinstance(record, Mapping)
                or not isinstance(record.get("name"), str)
                or not isinstance(record.get("array"), Mapping)
            ):
                raise SolverError("dual_warm_starts record schema drift")
            value = dual_math.decode_array(record["array"])
            add_unique_vector(
                warm_records,
                warm_seen,
                f"input.{index}.{record['name']}",
                value,
                (dual_size,),
                dual_math,
            )
    return {
        "primal": primal_records,
        "warm": warm_records[:MAX_DUAL_WARM_STARTS],
    }


def exact_squared_norm(value: np.ndarray) -> Fraction:
    return sum(
        (Fraction.from_float(float(item)) ** 2 for item in value), Fraction(0)
    )


def fraction_sign(value: Fraction) -> int:
    return int((value > 0) - (value < 0))


def float_down(value: Fraction) -> float:
    result = float(value)
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def reconstruct_original_space(
    candidate_scaled: np.ndarray,
    problem: Mapping[str, Any],
    dual_math: ModuleType,
) -> dict[str, Any]:
    basis = problem["basis"]
    center_raw = problem["center_raw"]
    scale = float(problem["coordinate_scale"])
    rho = float(problem["rho"])
    candidate_raw = basis.T @ (scale * candidate_scaled)
    delta_raw = candidate_raw - center_raw
    if not (
        np.isfinite(candidate_raw).all()
        and np.isfinite(center_raw).all()
        and np.isfinite(delta_raw).all()
    ):
        raise SolverError("original-space reconstruction became nonfinite")
    total_squared = exact_squared_norm(candidate_raw)
    trust_squared = exact_squared_norm(delta_raw)
    total_cap = Fraction.from_float(TOTAL_CAP_RAW) ** 2
    trust_radius_raw_float = float(scale * rho)
    trust_cap = Fraction.from_float(trust_radius_raw_float) ** 2
    total_slack = total_cap - total_squared
    trust_slack = trust_cap - trust_squared
    raw_space = problem["raw_space"]
    A = raw_space["A"]
    b = raw_space["b"]
    G = raw_space["G"]
    residual = raw_space["residual"]
    anchor_residual = A @ candidate_raw - b
    local_prediction = residual + G @ (candidate_raw - center_raw)
    total_l2 = float(np.linalg.norm(candidate_raw))
    trust_l2 = float(np.linalg.norm(delta_raw))
    linear_tolerance = float(raw_space["linear_tolerance"])
    l2_tolerance = float(raw_space["l2_tolerance"])
    raw_gate_checks = {
        "raw_arrays_and_results_finite": bool(
            np.isfinite(anchor_residual).all()
            and np.isfinite(local_prediction).all()
            and math.isfinite(total_l2)
            and math.isfinite(trust_l2)
        ),
        "anchor_raw_residual_gate": float(anchor_residual.min())
        >= -linear_tolerance,
        "local_raw_fixed_z_gate": float(local_prediction.min())
        >= raw_space["z_star"] - linear_tolerance,
        "total_raw_gate": total_l2 <= TOTAL_CAP_RAW + l2_tolerance,
        "trust_raw_gate": trust_l2
        <= raw_space["trust_radius_raw"] + l2_tolerance,
        "no_clip": True,
    }
    checks = {
        "candidate_scaled_shape_exact": candidate_scaled.shape
        == (problem["reduced_rank"],),
        "candidate_raw_shape_exact": candidate_raw.shape
        == (problem["actor_original_dimension"],),
        "exact_total_cap_feasible": total_slack >= 0,
        "exact_trust_cap_feasible": trust_slack >= 0,
        "canonical_raw_anchor_local_total_trust": all(raw_gate_checks.values()),
        "no_clip": True,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "candidate_scaled": dual_math.encode_array(candidate_scaled),
        "candidate_raw": dual_math.encode_array(candidate_raw),
        "candidate_raw_float64_le_sha256": dual_math.float64_sha256(candidate_raw),
        "center_raw_float64_le_sha256": dual_math.float64_sha256(center_raw),
        "total_l2_float64_audit_only": total_l2,
        "trust_l2_float64_audit_only": trust_l2,
        "raw_gate": {
            "checks": raw_gate_checks,
            "pass": all(raw_gate_checks.values()),
            "anchor_residual_min": float(anchor_residual.min()),
            "local_prediction_min": float(local_prediction.min()),
            "z_star": float(raw_space["z_star"]),
            "linear_residual_tolerance": linear_tolerance,
            "total_l2": total_l2,
            "trust_l2": trust_l2,
            "total_cap_plus_tolerance": TOTAL_CAP_RAW + l2_tolerance,
            "trust_cap_plus_tolerance": raw_space["trust_radius_raw"]
            + l2_tolerance,
            "arithmetic": "canonical_float64_raw_space_recomputation",
        },
        "total_squared_slack_sign": fraction_sign(total_slack),
        "trust_squared_slack_sign": fraction_sign(trust_slack),
        "total_squared_slack_lower_bound": float_down(total_slack),
        "trust_squared_slack_lower_bound": float_down(trust_slack),
        "total_cap_raw": TOTAL_CAP_RAW,
        "trust_radius_raw": trust_radius_raw_float,
        "exact_arithmetic": "binary_float_Fraction",
    }


def mapped_radius_certificate(
    certificate: Mapping[str, Any], operator_norm: Mapping[str, Any]
) -> dict[str, Any]:
    reduced = float(certificate["raw_optimal_radius_bound"])
    upper = float(operator_norm["operator_norm_upper"])
    mapped = (
        0.0
        if reduced == 0.0
        else math.nextafter(float(reduced * upper), math.inf)
    )
    checks = {
        "basis_operator_norm_certificate_pass": operator_norm.get("pass") is True,
        "reduced_certificate_raw_radius_gate_pass": certificate.get("checks", {}).get(
            "raw_optimal_radius_at_most_1e_9"
        )
        is True,
        "mapped_raw_optimal_radius_at_most_1e_9": mapped <= 1e-9,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "basis_operator_norm_upper": upper,
        "reduced_radius_bound": reduced,
        "mapped_raw_radius_bound": mapped,
        "multiplication_rounding": "float64_product_then_nextafter_up",
    }


def polish_records(
    problem: Mapping[str, Any], starts: Mapping[str, Any], dual_math: ModuleType
) -> list[dict[str, Any]]:
    K = problem["K"]
    q = problem["q"]
    center = problem["center"]
    rho = problem["rho"]
    calls: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for name, primal in starts["primal"]:
        calls.append((f"{name}/internal_zero_and_nnls", primal, None))
    witness_name, witness = starts["primal"][-1]
    for name, warm in starts["warm"]:
        calls.append((f"{witness_name}/warm:{name}", witness, warm))
    calls = calls[:MAX_POLISH_CALLS]
    records: list[dict[str, Any]] = []
    for index, (name, primal, warm) in enumerate(calls):
        try:
            polished = dual_math.polish_dual(
                primal,
                K,
                q,
                center,
                rho,
                warm_start=warm,
            )
            selected = polished["selected"]
            records.append(
                {
                    "index": index,
                    "name": name,
                    "primal_seed_sha256": dual_math.float64_sha256(primal),
                    "warm_start_sha256": dual_math.float64_sha256(warm)
                    if warm is not None
                    else None,
                    "available": True,
                    "selected_conservative_lower_bound": selected["terms"][
                        "conservative_lower_bound"
                    ],
                    "selected_variables_sha256": selected["variables"]["sha256"],
                    "selected_u_dual_sha256": selected["u_dual"]["sha256"],
                    "polish": polished,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "index": index,
                    "name": name,
                    "primal_seed_sha256": dual_math.float64_sha256(primal),
                    "warm_start_sha256": dual_math.float64_sha256(warm)
                    if warm is not None
                    else None,
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    return records


def certify_dual_recovered_candidates(
    problem: Mapping[str, Any],
    polish: Sequence[Mapping[str, Any]],
    dual_math: ModuleType,
) -> list[dict[str, Any]]:
    K = problem["K"]
    q = problem["q"]
    center = problem["center"]
    rho = problem["rho"]
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for record in polish:
        if record.get("available") is not True:
            continue
        selected = record["polish"]["selected"]
        candidate = dual_math.decode_array(selected["u_dual"])
        variables = dual_math.decode_array(selected["variables"])
        candidate_sha = dual_math.float64_sha256(candidate)
        if candidate_sha in seen:
            continue
        seen.add(candidate_sha)
        result: dict[str, Any] = {
            "index": len(candidates),
            "source_polish_index": record["index"],
            "source_polish_name": record["name"],
            "candidate_scaled_sha256": candidate_sha,
            "available": True,
            "certificate_available": False,
            "formal_pass": False,
            "errors": [],
        }
        try:
            certificate = dual_math.certify_primal(
                candidate,
                K,
                q,
                center,
                rho,
                warm_start=variables,
            )
            mapped = mapped_radius_certificate(certificate, problem["operator_norm"])
            original = reconstruct_original_space(candidate, problem, dual_math)
            identity_checks = {
                "certificate_primal_equals_candidate": certificate["primal"]
                == dual_math.encode_array(candidate),
                "certificate_formal_pass": certificate["formal_certificate_pass"]
                is True,
                "mapped_radius_pass": mapped["pass"] is True,
                "original_space_pass": original["pass"] is True,
                "canonical_raw_gate_pass": original.get("raw_gate", {}).get(
                    "pass"
                ) is True,
                "no_clip": certificate["checks"].get("no_clip") is True,
            }
            result.update(
                {
                    "candidate_scaled": dual_math.encode_array(candidate),
                    "certificate_available": True,
                    "certificate": certificate,
                    "mapped_raw_radius_certificate": mapped,
                    "original_space": original,
                    "identity_checks": identity_checks,
                    "formal_pass": all(identity_checks.values()),
                }
            )
        except Exception as exc:
            result["errors"].append(
                {
                    "phase": "strict_certify_dual_recovered_primal",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        candidates.append(result)
    return candidates


def solve_document(
    document: Mapping[str, Any],
    dual_math: ModuleType,
    *,
    input_evidence: Mapping[str, Any] | None,
    production: bool = False,
) -> dict[str, Any]:
    problem = decode_problem(document, dual_math, production=production)
    starts = collect_starts(problem, dual_math)
    polish = polish_records(problem, starts, dual_math)
    candidates = certify_dual_recovered_candidates(problem, polish, dual_math)
    formal = [record for record in candidates if record.get("formal_pass") is True]
    formal.sort(
        key=lambda record: (
            float(record["certificate"]["primal_objective_scaled"]),
            str(record["candidate_scaled_sha256"]),
            int(record["index"]),
        )
    )
    selected = formal[0] if formal else None
    checks = {
        "fixture_decoded_and_hash_checked": True,
        "reconstruction_map_checked": True,
        "project_provenance_checked": problem["project_provenance"]["pass"] is True,
        "raw_space_fixture_binding_checked": all(
            problem["raw_fixture_binding_checks"].values()
        ),
        "operator_norm_certificate_pass": problem["operator_norm"]["pass"] is True,
        "at_least_one_polish_call": bool(polish),
        "at_least_one_polish_evaluation_available": any(
            record.get("available") is True for record in polish
        ),
        "all_candidates_are_unique_dual_recoveries": len(candidates)
        == len({record["candidate_scaled_sha256"] for record in candidates}),
        "selected_is_formally_certified_or_absent": selected is None
        or selected.get("formal_pass") is True,
        "no_clip": True,
    }
    if not all(checks.values()):
        raise SolverError(f"CW18 solve invariant failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": SOLVED_STATUS if selected is not None else NO_CANDIDATE_STATUS,
        "classification": dict(CLASSIFICATION),
        "checks": checks,
        "pass": selected is not None,
        "input": dict(input_evidence) if input_evidence is not None else None,
        "fixture": {
            "schema_version": problem["fixture"]["schema_version"],
            "fixture_sha256": problem["fixture_sha256"],
            "row_count": int(problem["K"].shape[0]),
            "reduced_dimension": int(problem["K"].shape[1]),
            "rho": float(problem["rho"]),
            "coordinate_scale": float(problem["coordinate_scale"]),
        },
        "reconstruction": {
            "basis_float64_le_sha256": problem["basis_float64_le_sha256"],
            "basis_shape": list(problem["basis"].shape),
            "center_raw_float64_le_sha256": problem[
                "center_raw_float64_le_sha256"
            ],
            "expected_bootstrap_point_sha256": problem[
                "expected_bootstrap_point_sha256"
            ],
            "center_projection_error": problem["center_projection_error"],
            "center_binding_checks": problem["center_binding_checks"],
            "operator_norm_certificate": problem["operator_norm"],
            "formula": "candidate_raw=basis_reduced_by_actor.T@(coordinate_scale*candidate_scaled)",
            "raw_space_contract": {
                "array_float64_le_sha256": dict(
                    problem["raw_space"]["contract"]["array_float64_le_sha256"]
                ),
                "z_star": problem["raw_space"]["z_star"],
                "anchor_rows": int(problem["raw_space"]["A"].shape[0]),
                "local_rows": int(problem["raw_space"]["G"].shape[0]),
                "actor_dimension": int(problem["raw_space"]["A"].shape[1]),
                "binding_checks": problem["raw_fixture_binding_checks"],
            },
        },
        "start_manifest": {
            "primal": [
                {
                    "name": name,
                    "sha256": dual_math.float64_sha256(value),
                }
                for name, value in starts["primal"]
            ],
            "dual_warm": [
                {
                    "name": name,
                    "sha256": dual_math.float64_sha256(value),
                }
                for name, value in starts["warm"]
            ],
            "polish_call_count": len(polish),
            "max_polish_calls": MAX_POLISH_CALLS,
        },
        "dual_polish_records": polish,
        "dual_recovered_candidates": candidates,
        "formally_certified_candidate_count": len(formal),
        "selected_candidate_index": selected["index"] if selected is not None else None,
        "selected_candidate": selected,
        "acceptance": {
            "optimizer_status_is_not_acceptance": True,
            "frozen_exact_certificate_is_acceptance": True,
            "exact_float_feasibility_required": True,
            "mapped_raw_radius_required": True,
            "original_space_total_and_trust_required": True,
            "canonical_raw_anchor_and_local_required": True,
            "no_clip": True,
        },
        "run_executed": True,
        "cuda_accessed": False,
        "writes_performed": False,
    }


def source_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    imports: set[str] = set()
    calls: set[str] = set()
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
    required_functions = {
        "import_frozen_dual_math",
        "validate_frozen_builder_source",
        "validate_dual_math_contract",
        "conservative_basis_operator_norm",
        "decode_problem",
        "decode_raw_space_contract",
        "validate_project_provenance",
        "collect_starts",
        "polish_records",
        "certify_dual_recovered_candidates",
        "reconstruct_original_space",
        "solve_document",
        "selftest_result",
        "static_result",
    }
    observed_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    checks = {
        "parse_pass": True,
        "required_functions_present": required_functions.issubset(observed_functions),
        "no_cuda_model_network_submission_imports": not bool(
            imports.intersection(
                {
                    "torch",
                    "cupy",
                    "requests",
                    "socket",
                    "urllib",
                    "kaggle",
                    "subprocess",
                }
            )
        ),
        "no_filesystem_write_calls": not bool(
            calls.intersection(
                {
                    "write",
                    "write_bytes",
                    "write_text",
                    "save",
                    "torch_save",
                    "to_csv",
                    "unlink",
                    "rename",
                    "replace",
                    "mkdir",
                    "makedirs",
                }
            )
        ),
        "dual_math_sha_literal_exact": DUAL_MATH_SHA256 in source.decode("utf-8"),
        "builder_sha_literal_exact": BUILDER_SHA256 in source.decode("utf-8"),
        "fixed_fixture_path_literal_exact": str(
            FIXED_FIXTURE_PATH.relative_to(ROOT)
        ) in source.decode("utf-8"),
    }
    if not all(checks.values()):
        raise SolverError(f"CW18 source audit failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "imports": sorted(imports),
        "script_sha256": sha256_bytes(source),
        "script_bytes": len(source),
    }


def static_result() -> dict[str, Any]:
    runtime = runtime_checks()
    builder_evidence = validate_frozen_builder_source()
    dual_math, math_evidence = import_frozen_dual_math()
    contract = validate_dual_math_contract(dual_math)
    source, source_evidence = read_regular_stable(
        SCRIPT,
        "CW18 exact-dual solver source",
        maximum_bytes=2 * 1024 * 1024,
    )
    audit = source_audit(source)
    return {
        "schema_version": SCHEMA,
        "status": STATIC_STATUS,
        "classification": dict(CLASSIFICATION),
        "runtime": runtime,
        "builder_source": builder_evidence,
        "dual_math_source": math_evidence,
        "dual_math_contract": contract,
        "self_source": source_evidence,
        "source_audit": audit,
        "run_executed": False,
        "cuda_accessed": False,
        "writes_performed": False,
    }


def selftest_result() -> dict[str, Any]:
    static = static_result()
    dual_math, math_evidence = import_frozen_dual_math()
    basis = np.asarray([[1.0]], dtype=np.float64)
    center_raw = np.asarray([0.0], dtype=np.float64)
    A = np.asarray([[1000.0]], dtype=np.float64)
    b = np.asarray([0.05], dtype=np.float64)
    G = np.asarray([[0.0]], dtype=np.float64)
    residual = np.asarray([0.0], dtype=np.float64)
    z_star = 0.0
    center = basis @ center_raw / 0.001
    A_scaled = (A @ basis.T) * 0.001
    G_scaled = (G @ basis.T) * 0.001
    anchor_scale = np.maximum.reduce(
        [
            np.abs(b),
            np.linalg.norm(A_scaled, axis=1),
            np.full(b.shape, np.finfo(np.float64).eps),
        ]
    )
    z_scale = float(np.finfo(np.float64).eps)
    zero_scaled = np.zeros(1, dtype=np.float64)
    K = np.concatenate(
        [A_scaled / anchor_scale[:, None], G_scaled / z_scale], axis=0
    )
    q = -np.concatenate(
        [
            (A_scaled @ zero_scaled - b) / anchor_scale,
            (residual + G_scaled @ (zero_scaled - center) - z_star) / z_scale,
        ]
    )
    fixture = dual_math.encode_fixture(K, q, center, EXPECTED_RHO_SCALED)
    basis_sha = dual_math.float64_sha256(basis)
    center_raw_sha = dual_math.float64_sha256(center_raw)
    raw_arrays = {
        "anchor_gradients_A": dual_math.encode_array(A),
        "anchor_rhs_b": dual_math.encode_array(b),
        "local_gradients_G": dual_math.encode_array(G),
        "local_residual_at_center": dual_math.encode_array(residual),
    }
    pass_checks = {"synthetic": True}
    document = {
        "schema_version": BUILDER_SCHEMA,
        "status": BUILDER_STATUS,
        "pass": True,
        "source_audit": {
            "pass": True,
            "checks": pass_checks,
            "source_sha256": BUILDER_SHA256,
        },
        "runtime": {
            "pass": True,
            "checks": pass_checks,
            "cuda_required": False,
        },
        "plan_audit": {"pass": True, "checks": pass_checks},
        "captured_constraint_audit": {"pass": True, "checks": pass_checks},
        "precision_scope": {
            "pass": True,
            "checks": pass_checks,
            "before": "highest",
            "during": "high",
            "after": "highest",
        },
        "graph_fingerprint_audit": {"pass": True, "checks": pass_checks},
        "changed_model_official_evaluation_count": 0,
        "changed_model_candidate_consumer_called": False,
        "exact_CW11_fixture_consumer_called": True,
        "writes_performed": 0,
        "submission_performed": False,
        "fixture": fixture,
        "basis_float64_le_sha256": basis_sha,
        "expected_bootstrap_point_sha256": center_raw_sha,
        "reconstruction_map": {
            "basis_orientation": "reduced_rank_by_actor_original",
            "basis_reduced_by_actor": dual_math.encode_array(basis),
            "basis_float64_le_sha256": basis_sha,
            "center_raw": dual_math.encode_array(center_raw),
            "center_raw_float64_le_sha256": center_raw_sha,
            "coordinate_scale": 0.001,
            "actor_original_dimension": 1,
            "reduced_rank": 1,
        },
        "raw_space_contract": {
            "schema_version": RAW_CONTRACT_SCHEMA,
            "pass": True,
            "checks": pass_checks,
            "actor_original_dimension": 1,
            "anchor_row_count": 1,
            "local_physical_row_count": 1,
            "arrays": raw_arrays,
            "array_float64_le_sha256": {
                key: value["sha256"] for key, value in raw_arrays.items()
            },
            "z_star": z_star,
            "trust_radius_relative_actual_center": EXPECTED_TRUST_RADIUS_RAW,
            "TOTAL_L2_CAP": TOTAL_CAP_RAW,
            "L2_ABS_TOL": EXPECTED_L2_ABS_TOL,
            "LINEAR_RESIDUAL_TOL": EXPECTED_LINEAR_RESIDUAL_TOL,
            "raw_gate_formulas": {
                "anchor": "A@candidate_raw-b>=-LINEAR_RESIDUAL_TOL",
                "local": (
                    "residual+G@(candidate_raw-center_raw)>=z_star-"
                    "LINEAR_RESIDUAL_TOL"
                ),
                "total": "norm(candidate_raw)<=TOTAL_L2_CAP+L2_ABS_TOL",
                "trust": (
                    "norm(candidate_raw-center_raw)<=trust_radius+L2_ABS_TOL"
                ),
            },
        },
        "stage1": {
            "success": True,
            "status": 0,
            "z_star": z_star,
            "raw_certification": {
                "pass": True,
                "checks": pass_checks,
                "z_star": z_star,
            },
            "x_scaled": dual_math.encode_array(np.asarray([0.05])),
        },
        "reconstruction": {
            "pass": True,
            "dimensions": {
                "anchor_rows": 1,
                "local_physical_rows": 1,
                "actor_original": 1,
            },
            "changed_model_officially_evaluated": False,
            "selected_context_autograd_for_fixture": True,
            "full_stream_snapshot_constructed": False,
            "local_residual_float64_le_sha256": dual_math.float64_sha256(
                residual
            ),
        },
        "old_stage2": {
            "x_scaled": dual_math.encode_array(np.asarray([0.05])),
            "warm_dual_without_redundant_total_ball": dual_math.encode_array(
                np.asarray([0.05, 0.0, 0.0])
            ),
        },
    }
    solved_first = solve_document(document, dual_math, input_evidence=None)
    solved_second = solve_document(document, dual_math, input_evidence=None)
    tampered = json.loads(canonical_json(document).decode("utf-8"))
    tampered["fixture"]["rho"] = 0.5
    tamper_rejected = False
    try:
        solve_document(tampered, dual_math, input_evidence=None)
    except Exception:
        tamper_rejected = True
    missing_basis_sha = json.loads(canonical_json(document).decode("utf-8"))
    missing_basis_sha.pop("basis_float64_le_sha256")
    missing_basis_sha_rejected = False
    try:
        solve_document(missing_basis_sha, dual_math, input_evidence=None)
    except Exception:
        missing_basis_sha_rejected = True
    missing_bootstrap_sha = json.loads(canonical_json(document).decode("utf-8"))
    missing_bootstrap_sha.pop("expected_bootstrap_point_sha256")
    missing_bootstrap_sha_rejected = False
    try:
        solve_document(missing_bootstrap_sha, dual_math, input_evidence=None)
    except Exception:
        missing_bootstrap_sha_rejected = True
    wrong_center = json.loads(canonical_json(document).decode("utf-8"))
    wrong_center_raw = np.asarray([0.0001], dtype=np.float64)
    wrong_center_sha = dual_math.float64_sha256(wrong_center_raw)
    wrong_center["reconstruction_map"]["center_raw"] = dual_math.encode_array(
        wrong_center_raw
    )
    wrong_center["reconstruction_map"][
        "center_raw_float64_le_sha256"
    ] = wrong_center_sha
    wrong_center["expected_bootstrap_point_sha256"] = wrong_center_sha
    wrong_center_rejected = False
    try:
        solve_document(wrong_center, dual_math, input_evidence=None)
    except Exception:
        wrong_center_rejected = True
    missing_raw = json.loads(canonical_json(document).decode("utf-8"))
    missing_raw.pop("raw_space_contract")
    missing_raw_rejected = False
    try:
        solve_document(missing_raw, dual_math, input_evidence=None)
    except Exception:
        missing_raw_rejected = True
    synchronized_raw_tamper = json.loads(canonical_json(document).decode("utf-8"))
    changed_A = np.asarray([[1001.0]], dtype=np.float64)
    changed_A_record = dual_math.encode_array(changed_A)
    synchronized_raw_tamper["raw_space_contract"]["arrays"][
        "anchor_gradients_A"
    ] = changed_A_record
    synchronized_raw_tamper["raw_space_contract"]["array_float64_le_sha256"][
        "anchor_gradients_A"
    ] = changed_A_record["sha256"]
    synchronized_raw_tamper_rejected = False
    try:
        solve_document(synchronized_raw_tamper, dual_math, input_evidence=None)
    except Exception:
        synchronized_raw_tamper_rejected = True
    rehashed_q_tamper = json.loads(canonical_json(document).decode("utf-8"))
    changed_q = np.asarray([0.049, 0.0], dtype=np.float64)
    rehashed_q_tamper["fixture"] = dual_math.encode_fixture(
        K, changed_q, center, EXPECTED_RHO_SCALED
    )
    rehashed_q_tamper_rejected = False
    try:
        solve_document(rehashed_q_tamper, dual_math, input_evidence=None)
    except Exception:
        rehashed_q_tamper_rejected = True
    selected_first = solved_first.get("selected_candidate")
    selected_second = solved_second.get("selected_candidate")
    checks = {
        "static_ready": static.get("status") == STATIC_STATUS,
        "frozen_dual_math_selftest_pass": dual_math.selftest().get("pass") is True,
        "synthetic_fixture_formally_solved": solved_first.get("status")
        == SOLVED_STATUS
        and isinstance(selected_first, Mapping)
        and selected_first.get("formal_pass") is True,
        "synthetic_candidate_scaled_exact_target": isinstance(
            selected_first, Mapping
        )
        and np.array_equal(
            dual_math.decode_array(selected_first["candidate_scaled"]),
            np.asarray([0.05], dtype=np.float64),
        ),
        "synthetic_original_raw_exact": isinstance(selected_first, Mapping)
        and np.array_equal(
            dual_math.decode_array(selected_first["original_space"]["candidate_raw"]),
            np.asarray([0.00005], dtype=np.float64),
        ),
        "synthetic_raw_gate_pass": isinstance(selected_first, Mapping)
        and selected_first.get("original_space", {}).get("raw_gate", {}).get(
            "pass"
        ) is True,
        "synthetic_exact_center_binding": solved_first.get("reconstruction", {})
        .get("center_binding_checks", {})
        .get("fixture_center_exact_from_raw_basis_and_scale")
        is True
        and solved_first.get("reconstruction", {}).get("center_projection_error")
        == 0.0,
        "deterministic_selected_candidate_sha": isinstance(selected_first, Mapping)
        and isinstance(selected_second, Mapping)
        and selected_first.get("candidate_scaled_sha256")
        == selected_second.get("candidate_scaled_sha256"),
        "tampered_aggregate_fixture_rejected": tamper_rejected,
        "missing_top_level_basis_sha_rejected": missing_basis_sha_rejected,
        "missing_expected_bootstrap_sha_rejected": missing_bootstrap_sha_rejected,
        "fixture_center_not_bound_to_raw_center_rejected": wrong_center_rejected,
        "missing_raw_contract_rejected": missing_raw_rejected,
        "synchronized_raw_array_tamper_rejected": synchronized_raw_tamper_rejected,
        "validly_rehashed_fixture_q_tamper_rejected": rehashed_q_tamper_rejected,
        "no_writes_or_cuda": solved_first.get("writes_performed") is False
        and solved_first.get("cuda_accessed") is False,
    }
    if not all(checks.values()):
        raise SolverError(f"CW18 self-test failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": SELFTEST_STATUS,
        "classification": dict(CLASSIFICATION),
        "checks": checks,
        "pass": True,
        "dual_math_source": math_evidence,
        "synthetic_fixture_sha256": fixture["fixture_sha256"],
        "synthetic_selected_candidate_sha256": selected_first[
            "candidate_scaled_sha256"
        ],
        "run_executed": True,
        "cuda_accessed": False,
        "writes_performed": False,
    }


def solve_path(path: Path, expected_fixture_sha256: str) -> dict[str, Any]:
    runtime = runtime_checks()
    resolved = path.resolve()
    if resolved != FIXED_FIXTURE_PATH.resolve():
        raise SolverError("fixture path is not the fixed CW18 artifact path")
    if (
        not isinstance(expected_fixture_sha256, str)
        or len(expected_fixture_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_fixture_sha256)
    ):
        raise SolverError("--fixture-sha256 must be one lowercase SHA256")
    builder_evidence = validate_frozen_builder_source()
    dual_math, math_evidence = import_frozen_dual_math()
    contract = validate_dual_math_contract(dual_math)
    payload, evidence = read_regular_stable(
        resolved,
        "CW18 fixture input",
        expected_sha256=expected_fixture_sha256,
        expected_mode=FIXTURE_MODE,
    )
    document = strict_json_object(payload, "CW18 fixture input")
    result = solve_document(
        document, dual_math, input_evidence=evidence, production=True
    )
    result["runtime"] = runtime
    result["builder_source"] = builder_evidence
    result["dual_math_source"] = math_evidence
    result["dual_math_contract"] = contract
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--mode", choices=("static", "selftest", "solve"), default="static"
    )
    value.add_argument("--fixture", type=Path)
    value.add_argument("--fixture-sha256")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.mode == "static":
            if args.fixture is not None or args.fixture_sha256 is not None:
                raise SolverError("fixture arguments are only valid with --mode solve")
            result = static_result()
            return_code = 0
        elif args.mode == "selftest":
            if args.fixture is not None or args.fixture_sha256 is not None:
                raise SolverError("fixture arguments are only valid with --mode solve")
            result = selftest_result()
            return_code = 0
        else:
            fixture_path = args.fixture if args.fixture is not None else FIXED_FIXTURE_PATH
            if args.fixture_sha256 is None:
                raise SolverError("--mode solve requires --fixture-sha256")
            result = solve_path(fixture_path, args.fixture_sha256)
            return_code = 0 if result.get("status") == SOLVED_STATUS else 2
    except BaseException as exc:
        result = {
            "schema_version": SCHEMA,
            "status": FAILURE_STATUS,
            "classification": dict(CLASSIFICATION),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "run_executed": args.mode in {"selftest", "solve"},
            "cuda_accessed": False,
            "writes_performed": False,
        }
        return_code = 1
    print(canonical_json(result).decode("utf-8"), end="")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
