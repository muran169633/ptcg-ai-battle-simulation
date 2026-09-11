#!/usr/bin/env python3
"""Pure-CPU fixture and primal/dual certificate helpers for CW17.

This module has no model, CUDA, filesystem-write, network, packaging, or
submission capability.  It certifies the cap-normalized reduced problem

    min 0.5 ||u||^2  subject to  K u >= q, ||u-c|| <= rho.

The unchanged total cap is redundant for the numerical stage-2 solve because
the stage-1 feasible upper bound is already inside it.  Callers must still
recheck the original-space total and trust caps without clipping.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import optimize


SCHEMA = "ptcg-cw17-stage2-dual-math-v1"
FLOAT64_DTYPE = np.dtype("<f8")
COORDINATE_SCALE = 0.001
CERT_LINEAR_PRIMAL_TOL_SCALED = 1e-9
CERT_TRUST_TOL_SCALED = 1e-9
CERT_DUAL_GAP_TOL_SCALED = 5e-13
CERT_RAW_OPTIMAL_RADIUS_MAX = 1e-9
CERT_STATIONARITY_INF_TOL_SCALED = 1e-10
CERT_COMPLEMENTARITY_TOL_SCALED = 1e-10
DUAL_LB_ROUNDOFF_ULPS = 256
DUAL_GTOL = 1e-12
DUAL_FTOL = 16.0 * np.finfo(np.float64).eps
DUAL_MAXITER = 50_000
DUAL_MAXFUN = 500_000
DUAL_MAXLS = 100


class CertificateError(RuntimeError):
    """Raised when fixture or certificate structure fails closed."""


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


def float64_sha256(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=FLOAT64_DTYPE))
    return sha256_bytes(value.tobytes(order="C"))


def encode_array(array: Any) -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(array, dtype=FLOAT64_DTYPE))
    if not np.isfinite(value).all():
        raise CertificateError("fixture array is not finite")
    raw = value.tobytes(order="C")
    return {
        "dtype": "<f8",
        "shape": list(value.shape),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def decode_array(record: Mapping[str, Any]) -> np.ndarray:
    expected_keys = {"dtype", "shape", "bytes", "sha256", "base64"}
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_keys
        or record.get("dtype") != "<f8"
        or type(record.get("bytes")) is not int
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
        or not isinstance(record.get("base64"), str)
    ):
        raise CertificateError("fixture array schema drift")
    shape = record.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or len(shape) > 2
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise CertificateError("fixture array shape invalid")
    expected_items = math.prod(shape)
    if expected_items > 100_000_000:
        raise CertificateError("fixture array exceeds size bound")
    expected_bytes = expected_items * FLOAT64_DTYPE.itemsize
    encoded = record["base64"]
    expected_base64_characters = 4 * ((expected_bytes + 2) // 3)
    if (
        record["bytes"] != expected_bytes
        or len(encoded) != expected_base64_characters
    ):
        raise CertificateError("fixture encoded length drift")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise CertificateError("fixture base64 invalid") from exc
    if (
        len(raw) != expected_bytes
        or record.get("sha256") != sha256_bytes(raw)
    ):
        raise CertificateError("fixture array byte identity drift")
    value = np.frombuffer(raw, dtype=FLOAT64_DTYPE).reshape(tuple(shape)).copy()
    if not np.isfinite(value).all():
        raise CertificateError("decoded fixture array is not finite")
    return value


def encode_fixture(
    K: Any,
    q: Any,
    center: Any,
    rho: float,
    *,
    coordinate_scale: float = COORDINATE_SCALE,
) -> dict[str, Any]:
    K64 = np.asarray(K, dtype=np.float64)
    q64 = np.asarray(q, dtype=np.float64)
    c64 = np.asarray(center, dtype=np.float64)
    if (
        K64.ndim != 2
        or K64.shape[0] <= 0
        or K64.shape[1] <= 0
        or q64.shape != (K64.shape[0],)
        or c64.shape != (K64.shape[1],)
        or not np.isfinite(K64).all()
        or not np.isfinite(q64).all()
        or not np.isfinite(c64).all()
        or not math.isfinite(float(rho))
        or float(rho) <= 0.0
        or float(coordinate_scale) != COORDINATE_SCALE
    ):
        raise CertificateError("fixture dimensions or constants invalid")
    arrays = {
        "K": encode_array(K64),
        "q": encode_array(q64),
        "center": encode_array(c64),
    }
    identity = {
        "schema_version": SCHEMA,
        "coordinate_scale": float(coordinate_scale),
        "rho": float(rho),
        "row_count": int(K64.shape[0]),
        "reduced_dimension": int(K64.shape[1]),
        "arrays": arrays,
    }
    return {**identity, "fixture_sha256": sha256_bytes(canonical_json(identity))}


def decode_fixture(
    record: Mapping[str, Any],
    *,
    expected_fixture_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    expected_keys = {
        "schema_version",
        "coordinate_scale",
        "rho",
        "row_count",
        "reduced_dimension",
        "arrays",
        "fixture_sha256",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_keys
        or record.get("schema_version") != SCHEMA
        or type(record.get("coordinate_scale")) is not float
        or type(record.get("rho")) is not float
        or type(record.get("row_count")) is not int
        or type(record.get("reduced_dimension")) is not int
        or record["row_count"] <= 0
        or record["reduced_dimension"] <= 0
        or not isinstance(record.get("fixture_sha256"), str)
        or len(record["fixture_sha256"]) != 64
    ):
        raise CertificateError("fixture schema invalid")
    identity = {key: record[key] for key in expected_keys - {"fixture_sha256"}}
    if record.get("fixture_sha256") != sha256_bytes(canonical_json(identity)):
        raise CertificateError("fixture aggregate identity drift")
    if (
        expected_fixture_sha256 is not None
        and record["fixture_sha256"] != expected_fixture_sha256
    ):
        raise CertificateError("fixture differs from external frozen identity")
    if record.get("coordinate_scale") != COORDINATE_SCALE:
        raise CertificateError("fixture coordinate scale drift")
    arrays = record.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != {"K", "q", "center"}:
        raise CertificateError("fixture array mapping invalid")
    K = decode_array(arrays["K"])
    q = decode_array(arrays["q"])
    center = decode_array(arrays["center"])
    if (
        K.ndim != 2
        or q.shape != (K.shape[0],)
        or center.shape != (K.shape[1],)
        or record.get("row_count") != K.shape[0]
        or record.get("reduced_dimension") != K.shape[1]
    ):
        raise CertificateError("fixture decoded shape drift")
    rho = float(record["rho"])
    if not math.isfinite(rho) or rho <= 0.0:
        raise CertificateError("fixture rho invalid")
    return K, q, center, rho


def _validate_problem(
    K: Any,
    q: Any,
    center: Any,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    K64 = np.asarray(K, dtype=np.float64)
    q64 = np.asarray(q, dtype=np.float64)
    c64 = np.asarray(center, dtype=np.float64)
    if (
        K64.ndim != 2
        or K64.shape[0] <= 0
        or K64.shape[1] <= 0
        or q64.shape != (K64.shape[0],)
        or c64.shape != (K64.shape[1],)
        or not np.isfinite(K64).all()
        or not np.isfinite(q64).all()
        or not np.isfinite(c64).all()
        or not math.isfinite(float(rho))
        or float(rho) <= 0.0
    ):
        raise CertificateError("dual problem invalid")
    return K64, q64, c64, float(rho)


def dual_objective_gradient(
    variables: Any,
    K: Any,
    q: Any,
    center: Any,
    rho: float,
) -> tuple[float, np.ndarray]:
    K64, q64, c64, rho64 = _validate_problem(K, q, center, rho)
    values = np.asarray(variables, dtype=np.float64)
    if values.shape != (K64.shape[0] + 1,) or not np.isfinite(values).all():
        raise CertificateError("dual variable shape or finiteness invalid")
    lam = values[:-1]
    mu = float(values[-1])
    if (lam < 0.0).any() or mu < 0.0:
        raise CertificateError("dual variables must be nonnegative")
    vector = K64.T @ lam + mu * c64
    denominator = 1.0 + mu
    u_dual = vector / denominator
    value = float(
        lam @ q64
        + 0.5 * mu * (c64 @ c64 - rho64**2)
        - 0.5 * (vector @ vector) / denominator
    )
    gradient = np.concatenate(
        [q64 - K64 @ u_dual, np.asarray([0.5 * ((u_dual - c64) @ (u_dual - c64) - rho64**2)])]
    )
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise CertificateError("dual objective became nonfinite")
    return value, gradient


def _fraction(value: float) -> Fraction:
    if not math.isfinite(float(value)):
        raise CertificateError("nonfinite value cannot enter exact arithmetic")
    return Fraction.from_float(float(value))


def _float_down(value: Fraction) -> float:
    result = float(value)
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def _float_up(value: Fraction) -> float:
    result = float(value)
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _sqrt_float_up(value: Fraction) -> float:
    if value < 0:
        raise CertificateError("cannot upper-bound square root of negative value")
    if value == 0:
        return 0.0
    result = math.sqrt(_float_up(value))
    while Fraction.from_float(result) ** 2 < value:
        result = math.nextafter(result, math.inf)
    return result


def _exact_dual_terms(
    variables: np.ndarray,
    K: np.ndarray,
    q: np.ndarray,
    center: np.ndarray,
    rho: float,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    row_count, dimension = K.shape
    lam = [_fraction(value) for value in variables[:-1]]
    mu = _fraction(float(variables[-1]))
    q_exact = [_fraction(value) for value in q]
    center_exact = [_fraction(value) for value in center]
    term_linear = sum(
        (lam[index] * q_exact[index] for index in range(row_count)),
        Fraction(0),
    )
    vector: list[Fraction] = []
    for column in range(dimension):
        value = mu * center_exact[column]
        for row in range(row_count):
            value += _fraction(float(K[row, column])) * lam[row]
        vector.append(value)
    center_squared = sum((value * value for value in center_exact), Fraction(0))
    rho_exact = _fraction(rho)
    term_trust = Fraction(1, 2) * mu * (
        center_squared - rho_exact * rho_exact
    )
    vector_squared = sum((value * value for value in vector), Fraction(0))
    term_quadratic = -Fraction(1, 2) * vector_squared / (Fraction(1) + mu)
    direct = term_linear + term_trust + term_quadratic
    return term_linear, term_trust, term_quadratic, direct


def _exact_primal_and_feasibility(
    primal: np.ndarray,
    K: np.ndarray,
    q: np.ndarray,
    center: np.ndarray,
    rho: float,
) -> dict[str, Any]:
    u = [_fraction(value) for value in primal]
    c = [_fraction(value) for value in center]
    q_exact = [_fraction(value) for value in q]
    linear_slacks: list[Fraction] = []
    for row in range(K.shape[0]):
        value = -q_exact[row]
        for column in range(K.shape[1]):
            value += _fraction(float(K[row, column])) * u[column]
        linear_slacks.append(value)
    trust_squared_slack = _fraction(rho) ** 2 - sum(
        ((left - right) ** 2 for left, right in zip(u, c)), Fraction(0)
    )
    total_squared_slack = Fraction(1) - sum(
        (value * value for value in u), Fraction(0)
    )
    primal_objective = Fraction(1, 2) * sum(
        (value * value for value in u), Fraction(0)
    )
    return {
        "linear_slacks": linear_slacks,
        "linear_all_nonnegative": all(value >= 0 for value in linear_slacks),
        "linear_min_sign": min(linear_slacks),
        "trust_squared_slack": trust_squared_slack,
        "total_squared_slack": total_squared_slack,
        "primal_objective": primal_objective,
        "exact_float_feasible": all(value >= 0 for value in linear_slacks)
        and trust_squared_slack >= 0
        and total_squared_slack >= 0,
    }


def _exact_kkt_metrics(
    primal: np.ndarray,
    K: np.ndarray,
    center: np.ndarray,
    variables: np.ndarray,
    exact_primal: Mapping[str, Any],
) -> dict[str, Fraction]:
    u = [_fraction(value) for value in primal]
    c = [_fraction(value) for value in center]
    lam = [_fraction(value) for value in variables[:-1]]
    mu = _fraction(float(variables[-1]))
    vector: list[Fraction] = []
    for column in range(K.shape[1]):
        value = mu * c[column]
        for row in range(K.shape[0]):
            value += _fraction(float(K[row, column])) * lam[row]
        vector.append(value)
    stationarity = [
        (Fraction(1) + mu) * u[column] - vector[column]
        for column in range(K.shape[1])
    ]
    linear_slacks = exact_primal["linear_slacks"]
    linear_complementarity = [
        lam[row] * linear_slacks[row] for row in range(K.shape[0])
    ]
    trust_complementarity = (
        Fraction(1, 2) * mu * exact_primal["trust_squared_slack"]
    )
    u_dual = [value / (Fraction(1) + mu) for value in vector]
    decomposition = (
        Fraction(1, 2)
        * (Fraction(1) + mu)
        * sum(
            ((u[column] - u_dual[column]) ** 2 for column in range(K.shape[1])),
            Fraction(0),
        )
        + sum(
            (lam[row] * linear_slacks[row] for row in range(K.shape[0])),
            Fraction(0),
        )
        + trust_complementarity
    )
    return {
        "stationarity_abs_max": max(
            (abs(value) for value in stationarity), default=Fraction(0)
        ),
        "complementarity_abs_max": max(
            [abs(value) for value in linear_complementarity]
            + [abs(trust_complementarity)]
        ),
        "direct_gap_decomposition": decomposition,
    }


def conservative_dual_evaluation(
    variables: Any,
    K: Any,
    q: Any,
    center: Any,
    rho: float,
) -> dict[str, Any]:
    K64, q64, c64, rho64 = _validate_problem(K, q, center, rho)
    values = np.asarray(variables, dtype=np.float64)
    if values.shape != (K64.shape[0] + 1,) or not np.isfinite(values).all():
        raise CertificateError("dual variable shape invalid")
    lam = values[:-1]
    mu = float(values[-1])
    if (lam < 0.0).any() or mu < 0.0:
        raise CertificateError("dual variable is negative")
    term_linear_exact, term_trust_exact, term_quadratic_exact, direct_exact = (
        _exact_dual_terms(values, K64, q64, c64, rho64)
    )
    lower = _float_down(direct_exact)
    direct_nearest = float(direct_exact)
    pad = direct_nearest - lower
    vector = K64.T @ lam + mu * c64
    denominator = 1.0 + mu
    u_dual = np.asarray(vector / denominator, dtype=np.float64)
    value64, gradient = dual_objective_gradient(values, K64, q64, c64, rho64)
    checks = {
        "variables_finite_nonnegative": bool(np.isfinite(values).all() and (values >= 0.0).all()),
        "direct_finite": math.isfinite(direct_nearest),
        "downward_rounding_finite": math.isfinite(lower),
        "lower_not_above_exact_rational": Fraction.from_float(lower)
        <= direct_exact,
        "exact_binary_fraction_authoritative": Fraction.from_float(lower)
        <= direct_exact,
        "gradient_finite": bool(np.isfinite(gradient).all()),
        "u_dual_finite": bool(np.isfinite(u_dual).all()),
    }
    if not all(checks.values()):
        raise CertificateError(f"dual conservative evaluation failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "variables": encode_array(values),
        "lambda_count": int(lam.size),
        "mu": mu,
        "terms": {
            "linear": float(term_linear_exact),
            "trust": float(term_trust_exact),
            "quadratic": float(term_quadratic_exact),
            "direct": direct_nearest,
            "roundoff_pad": float(pad),
            "conservative_lower_bound": float(lower),
            "exact_binary_rational_sha256": sha256_bytes(
                f"{direct_exact.numerator}/{direct_exact.denominator}".encode("ascii")
            ),
            "exact_numerator_bits": int(abs(direct_exact.numerator).bit_length()),
            "exact_denominator_bits": int(direct_exact.denominator.bit_length()),
            "lower_bound_method": "exact_binary_fraction_then_float_down",
            "float64_recompute_error_audit_only": value64 - direct_nearest,
        },
        "u_dual": encode_array(u_dual),
        "gradient": encode_array(gradient),
    }


def _nnls_start(primal: np.ndarray, K: np.ndarray, center: np.ndarray) -> np.ndarray:
    matrix = np.concatenate([K.T, (center - primal).reshape(-1, 1)], axis=1)
    try:
        values, _ = optimize.nnls(matrix, primal, maxiter=100 * matrix.shape[1])
    except (RuntimeError, ValueError):
        values = np.zeros(K.shape[0] + 1, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def polish_dual(
    primal: Any,
    K: Any,
    q: Any,
    center: Any,
    rho: float,
    *,
    warm_start: Any | None = None,
) -> dict[str, Any]:
    K64, q64, c64, rho64 = _validate_problem(K, q, center, rho)
    u = np.asarray(primal, dtype=np.float64)
    if u.shape != (K64.shape[1],) or not np.isfinite(u).all():
        raise CertificateError("primal warm witness invalid")
    zero = np.zeros(K64.shape[0] + 1, dtype=np.float64)
    warm_policy: dict[str, Any] = {
        "provided": warm_start is not None,
        "accepted_without_projection": False,
        "projection_applied": False,
        "fallback_to_zero": False,
    }
    if warm_start is None:
        warm = zero.copy()
        warm_policy["fallback_to_zero"] = True
        warm_policy["reason"] = "not_provided"
    else:
        raw_warm = np.asarray(warm_start, dtype=np.float64)
        warm_policy["raw_shape"] = list(raw_warm.shape)
        warm_policy["raw_finite"] = bool(np.isfinite(raw_warm).all())
        warm_policy["raw_sha256"] = (
            float64_sha256(raw_warm) if np.isfinite(raw_warm).all() else None
        )
        if raw_warm.shape != zero.shape or not np.isfinite(raw_warm).all():
            warm = zero.copy()
            warm_policy["fallback_to_zero"] = True
            warm_policy["reason"] = "invalid_shape_or_nonfinite"
        elif (raw_warm < 0.0).any():
            warm = zero.copy()
            warm_policy["raw_min"] = float(raw_warm.min())
            warm_policy["fallback_to_zero"] = True
            warm_policy["reason"] = "negative_raw_warm_start"
        else:
            warm = raw_warm.copy()
            warm_policy["raw_min"] = float(raw_warm.min())
            warm_policy["accepted_without_projection"] = True
            warm_policy["reason"] = "accepted_exactly"
    starts = (
        ("zero", zero),
        ("slsqp_or_zero", warm),
        ("active_nnls", _nnls_start(u, K64, c64)),
    )
    records: list[dict[str, Any]] = []
    for index, (name, start) in enumerate(starts):
        try:
            result = optimize.minimize(
                lambda value: -dual_objective_gradient(value, K64, q64, c64, rho64)[0],
                start,
                jac=lambda value: -dual_objective_gradient(value, K64, q64, c64, rho64)[1],
                bounds=[(0.0, None)] * start.size,
                method="L-BFGS-B",
                options={
                    "gtol": DUAL_GTOL,
                    "ftol": DUAL_FTOL,
                    "maxiter": DUAL_MAXITER,
                    "maxfun": DUAL_MAXFUN,
                    "maxls": DUAL_MAXLS,
                },
            )
            raw_variables = np.asarray(result.x, dtype=np.float64)
            if (
                raw_variables.shape != start.shape
                or not np.isfinite(raw_variables).all()
                or (raw_variables < 0.0).any()
            ):
                raise CertificateError(
                    "optimizer dual point is not finite nonnegative with exact shape"
                )
            variables = raw_variables.copy()
            evaluation = conservative_dual_evaluation(
                variables, K64, q64, c64, rho64
            )
            records.append(
                {
                    "index": index,
                    "start": name,
                    "start_sha256": float64_sha256(start),
                    "evaluation_available": True,
                    "optimizer_success_audit_only": bool(result.success),
                    "optimizer_status_audit_only": int(result.status),
                    "optimizer_message_audit_only": str(result.message),
                    "iterations": int(result.nit),
                    "function_evaluations": int(result.nfev),
                    "optimizer_raw_variables_sha256": float64_sha256(raw_variables),
                    "optimizer_raw_variable_min": float(raw_variables.min()),
                    "optimizer_projection_applied": False,
                    "optimizer_projection_inf": 0.0,
                    "evaluation": evaluation,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "index": index,
                    "start": name,
                    "start_sha256": float64_sha256(start),
                    "evaluation_available": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    available_indices = [
        index for index, record in enumerate(records) if record["evaluation_available"]
    ]
    if not available_indices:
        raise CertificateError("all fixed dual starts failed before evaluation")
    selected_index = max(
        available_indices,
        key=lambda value: (
            records[value]["evaluation"]["terms"]["conservative_lower_bound"],
            -value,
        ),
    )
    return {
        "pass": True,
        "fixed_start_order": [value[0] for value in starts],
        "selected_index": selected_index,
        "selected_start": records[selected_index]["start"],
        "selected": records[selected_index]["evaluation"],
        "records": records,
        "optimizer_status_is_not_acceptance": True,
        "optimizer_point_projection_allowed": False,
        "warm_start_policy": warm_policy,
        "nnls_start_rows_used": "all_rows",
    }


def certify_primal(
    primal: Any,
    K: Any,
    q: Any,
    center: Any,
    rho: float,
    *,
    warm_start: Any | None = None,
    clipped_vector_applied: bool = False,
    coordinate_scale: float = COORDINATE_SCALE,
) -> dict[str, Any]:
    K64, q64, c64, rho64 = _validate_problem(K, q, center, rho)
    u = np.asarray(primal, dtype=np.float64)
    if (
        u.shape != (K64.shape[1],)
        or not np.isfinite(u).all()
        or float(coordinate_scale) != COORDINATE_SCALE
    ):
        raise CertificateError("primal certificate input invalid")
    linear_slack = K64 @ u - q64
    trust_delta = u - c64
    trust_norm = float(np.linalg.norm(trust_delta))
    trust_squared_slack = float(rho64**2 - trust_delta @ trust_delta)
    total_norm = float(np.linalg.norm(u))
    dual = polish_dual(u, K64, q64, c64, rho64, warm_start=warm_start)
    selected = dual["selected"]
    variables = decode_array(selected["variables"])
    lam = variables[:-1]
    mu = float(variables[-1])
    exact_primal = _exact_primal_and_feasibility(u, K64, q64, c64, rho64)
    _, _, _, exact_dual = _exact_dual_terms(variables, K64, q64, c64, rho64)
    exact_primal_objective = exact_primal["primal_objective"]
    exact_linear_minimum = exact_primal["linear_min_sign"]
    primal_objective = _float_up(exact_primal_objective)
    exact_direct_gap = exact_primal_objective - exact_dual
    gap_upper = _float_up(exact_direct_gap)
    signed_direct_gap = float(exact_direct_gap)
    nonnegative_exact_gap = max(exact_direct_gap, Fraction(0))
    raw_radius_squared_exact = (
        Fraction(2)
        * nonnegative_exact_gap
        * _fraction(float(coordinate_scale)) ** 2
    )
    raw_optimal_radius = _sqrt_float_up(raw_radius_squared_exact)
    exact_kkt = _exact_kkt_metrics(u, K64, c64, variables, exact_primal)
    stationarity_inf = _float_up(exact_kkt["stationarity_abs_max"])
    complementarity_inf = _float_up(exact_kkt["complementarity_abs_max"])
    decomposed_gap = _float_up(exact_kkt["direct_gap_decomposition"])

    # Nearest-float recomputations are audit-only.  Every formal threshold below
    # is evaluated against the exact binary rational represented by its inputs.
    vector = K64.T @ lam + mu * c64
    stationarity_float_audit = (1.0 + mu) * u - vector
    linear_complementarity = lam * linear_slack
    trust_complementarity = 0.5 * mu * trust_squared_slack
    complementarity_float_audit = float(
        max(
            float(np.abs(linear_complementarity).max(initial=0.0)),
            abs(trust_complementarity),
        )
    )
    u_dual = vector / (1.0 + mu)
    decomposed_gap_float_audit = float(
        0.5 * (1.0 + mu) * ((u - u_dual) @ (u - u_dual))
        + lam @ linear_slack
        + 0.5 * mu * trust_squared_slack
    )
    exact_float_feasible = bool(exact_primal["exact_float_feasible"])
    protocol_tolerance_feasible = bool(
        float(linear_slack.min()) >= -CERT_LINEAR_PRIMAL_TOL_SCALED
        and trust_norm <= rho64 + CERT_TRUST_TOL_SCALED
        and total_norm <= 1.0 + CERT_TRUST_TOL_SCALED
    )
    checks = {
        "protocol_tolerance_feasible": protocol_tolerance_feasible,
        "exact_float_feasible": exact_float_feasible,
        "dual_variables_nonnegative_finite": bool(
            np.isfinite(variables).all() and (variables >= 0.0).all()
        ),
        "conservative_gap_nonnegative": exact_direct_gap >= 0,
        "dual_gap_at_most_5e_13": exact_direct_gap
        <= _fraction(CERT_DUAL_GAP_TOL_SCALED),
        "raw_optimal_radius_at_most_1e_9": exact_direct_gap >= 0
        and raw_radius_squared_exact
        <= _fraction(CERT_RAW_OPTIMAL_RADIUS_MAX) ** 2,
        "stationarity_inf_at_most_1e_10": exact_kkt["stationarity_abs_max"]
        <= _fraction(CERT_STATIONARITY_INF_TOL_SCALED),
        "complementarity_inf_at_most_1e_10": exact_kkt[
            "complementarity_abs_max"
        ]
        <= _fraction(CERT_COMPLEMENTARITY_TOL_SCALED),
        "direct_gap_decomposition_matches": exact_direct_gap
        == exact_kkt["direct_gap_decomposition"],
        "no_clip": clipped_vector_applied is False,
    }
    formal_gate_names = {
        "exact_float_feasible",
        "dual_variables_nonnegative_finite",
        "conservative_gap_nonnegative",
        "dual_gap_at_most_5e_13",
        "raw_optimal_radius_at_most_1e_9",
        "stationarity_inf_at_most_1e_10",
        "complementarity_inf_at_most_1e_10",
        "direct_gap_decomposition_matches",
        "no_clip",
    }
    formal_pass = all(checks[name] for name in formal_gate_names)
    return {
        "checks": checks,
        "formal_gate_names": sorted(formal_gate_names),
        "formal_certificate_pass": formal_pass,
        "optimizer_status_not_used_for_acceptance": True,
        "clipped_vector_applied": clipped_vector_applied,
        "primal": encode_array(u),
        "primal_objective_scaled": primal_objective,
        "primal_objective_bound_method": "exact_binary_fraction_then_float_up",
        "exact_primal_objective_sha256": sha256_bytes(
            (
                f"{exact_primal_objective.numerator}/"
                f"{exact_primal_objective.denominator}"
            ).encode("ascii")
        ),
        "exact_feasibility": {
            "linear_all_nonnegative": exact_primal["linear_all_nonnegative"],
            "linear_min_sign": int(
                (exact_linear_minimum > 0)
                - (exact_linear_minimum < 0)
            ),
            "linear_min_lower_bound_scaled": _float_down(exact_linear_minimum),
            "linear_min_exact_sha256": sha256_bytes(
                (
                    f"{exact_linear_minimum.numerator}/"
                    f"{exact_linear_minimum.denominator}"
                ).encode("ascii")
            ),
            "trust_squared_slack_sign": int(
                (exact_primal["trust_squared_slack"] > 0)
                - (exact_primal["trust_squared_slack"] < 0)
            ),
            "total_squared_slack_sign": int(
                (exact_primal["total_squared_slack"] > 0)
                - (exact_primal["total_squared_slack"] < 0)
            ),
        },
        "linear_slack": encode_array(linear_slack),
        "linear_slack_min": float(linear_slack.min()),
        "trust_norm_scaled": trust_norm,
        "trust_squared_slack_scaled": trust_squared_slack,
        "total_norm_scaled": total_norm,
        "dual": dual,
        "signed_direct_gap_scaled": signed_direct_gap,
        "exact_direct_gap_sha256": sha256_bytes(
            (
                f"{exact_direct_gap.numerator}/{exact_direct_gap.denominator}"
            ).encode("ascii")
        ),
        "conservative_gap_scaled": gap_upper,
        "conservative_gap_bound_method": "exact_binary_fraction_then_float_up",
        "raw_optimal_radius_bound": raw_optimal_radius,
        "raw_optimal_radius_gate_method": "exact_squared_radius_comparison",
        "stationarity_inf_scaled": stationarity_inf,
        "stationarity_inf_float64_audit_only": float(
            np.abs(stationarity_float_audit).max()
        ),
        "complementarity_inf_scaled": complementarity_inf,
        "complementarity_inf_float64_audit_only": complementarity_float_audit,
        "decomposed_direct_gap_scaled": decomposed_gap,
        "decomposed_direct_gap_float64_audit_only": decomposed_gap_float_audit,
        "formal_threshold_arithmetic": "exact_binary_fraction",
    }


def selftest() -> dict[str, Any]:
    K = np.asarray([[1.0]], dtype=np.float64)
    q = np.asarray([1.0], dtype=np.float64)
    center = np.asarray([0.0], dtype=np.float64)
    rho = 2.0
    fixture = encode_fixture(K, q, center, rho)
    decoded = decode_fixture(fixture)
    certificate = certify_primal(
        np.asarray([1.0]), *decoded, warm_start=np.asarray([1.0, 0.0])
    )

    mutated = json.loads(canonical_json(fixture).decode("utf-8"))
    raw = bytearray(base64.b64decode(mutated["arrays"]["q"]["base64"]))
    raw[0] ^= 1
    mutated["arrays"]["q"]["base64"] = base64.b64encode(bytes(raw)).decode("ascii")
    mutation_rejected = False
    try:
        decode_fixture(mutated)
    except CertificateError:
        mutation_rejected = True

    recomputed_mutation = json.loads(canonical_json(fixture).decode("utf-8"))
    recomputed_mutation["arrays"]["q"] = encode_array(
        np.asarray([math.nextafter(1.0, math.inf)], dtype=np.float64)
    )
    recomputed_identity = {
        key: recomputed_mutation[key]
        for key in recomputed_mutation
        if key != "fixture_sha256"
    }
    recomputed_mutation["fixture_sha256"] = sha256_bytes(
        canonical_json(recomputed_identity)
    )
    externally_bound_mutation_rejected = False
    try:
        decode_fixture(
            recomputed_mutation,
            expected_fixture_sha256=fixture["fixture_sha256"],
        )
    except CertificateError:
        externally_bound_mutation_rejected = True

    cancellation = conservative_dual_evaluation(
        np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([2.0**65, -1.0, -(2.0**65)], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        1.0,
    )
    bool_shape_rejected = False
    malformed = encode_array(np.asarray([1.0], dtype=np.float64))
    malformed["shape"] = [True]
    try:
        decode_array(malformed)
    except CertificateError:
        bool_shape_rejected = True

    oversized = encode_array(np.asarray([1.0], dtype=np.float64))
    oversized["shape"] = [100_000_001]
    oversized["bytes"] = 800_000_008
    oversized["base64"] = "not-valid-base64"
    oversized_predecode_rejected = False
    try:
        decode_array(oversized)
    except CertificateError as exc:
        oversized_predecode_rejected = str(exc) == "fixture array exceeds size bound"

    original_minimize = optimize.minimize

    def status9_minimize(*args: Any, **kwargs: Any) -> Any:
        result = original_minimize(*args, **kwargs)
        result.status = 9
        result.success = False
        result.message = "injected status 9 for status-independence selftest"
        return result

    optimize.minimize = status9_minimize
    try:
        status9_certificate = certify_primal(
            np.asarray([1.0]), *decoded, warm_start=np.asarray([1.0, 0.0])
        )
    finally:
        optimize.minimize = original_minimize

    gap_edge_q = 1.4511769723235268e-07
    gap_edge_u = 7.83378647519179e-07

    def fixed_gap_edge_status9(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return optimize.OptimizeResult(
            x=np.asarray([gap_edge_u, 0.0], dtype=np.float64),
            status=9,
            success=False,
            message="injected status 9 at exact-gap threshold edge",
            nit=2000,
            nfev=2001,
        )

    optimize.minimize = fixed_gap_edge_status9
    try:
        gap_edge_certificate = certify_primal(
            np.asarray([gap_edge_u], dtype=np.float64),
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([gap_edge_q], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            2.0,
            warm_start=np.asarray([gap_edge_u, 0.0], dtype=np.float64),
        )
    finally:
        optimize.minimize = original_minimize
    legacy_gap_edge_float_subtraction = (
        gap_edge_certificate["primal_objective_scaled"]
        - gap_edge_certificate["dual"]["selected"]["terms"][
            "conservative_lower_bound"
        ]
    )

    def fixed_stationarity_edge_status9(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return optimize.OptimizeResult(
            x=np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float64),
            status=9,
            success=False,
            message="injected status 9 at exact-stationarity cancellation edge",
            nit=2000,
            nfev=2001,
        )

    optimize.minimize = fixed_stationarity_edge_status9
    try:
        stationarity_edge_certificate = certify_primal(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([[2.0**65], [-2e-10], [-(2.0**65)]], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            2.0,
            warm_start=np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float64),
        )
    finally:
        optimize.minimize = original_minimize

    bad_primal = certify_primal(
        np.asarray([0.9]), K, q, center, rho, warm_start=np.asarray([1.0, 0.0])
    )
    duplicate_K = np.asarray([[1.0], [1.0], [-1.0]], dtype=np.float64)
    duplicate_q = np.asarray([1.0, 1.0, -1.0], dtype=np.float64)
    duplicate = certify_primal(
        np.asarray([1.0]),
        duplicate_K,
        duplicate_q,
        center,
        rho,
        warm_start=np.asarray([0.5, 0.5, 0.0, 0.0]),
    )
    binary_scaled = certify_primal(
        np.asarray([1.0]),
        8.0 * K,
        8.0 * q,
        center,
        rho,
        warm_start=np.asarray([1.0 / 8.0, 0.0]),
    )
    rounded_primal = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    nonbinary_scales = np.asarray([1e-100, 1.0, 1.0, 1e100], dtype=np.float64)
    rounded_K = np.diag(nonbinary_scales)
    rounded_q = nonbinary_scales * rounded_primal
    rounded_warm = np.concatenate(
        [rounded_primal / nonbinary_scales, np.asarray([0.0])]
    )
    rounded_scale_certificate = certify_primal(
        rounded_primal,
        rounded_K,
        rounded_q,
        np.zeros(4, dtype=np.float64),
        2.0,
        warm_start=rounded_warm,
    )
    negative_rejected = False
    try:
        conservative_dual_evaluation(np.asarray([-1.0, 0.0]), K, q, center, rho)
    except CertificateError:
        negative_rejected = True
    checks = {
        "fixture_roundtrip_exact": all(
            np.array_equal(left, right)
            for left, right in zip(decoded[:3], (K, q, center))
        )
        and decoded[3] == rho,
        "fixture_byte_mutation_rejected": mutation_rejected,
        "externally_bound_rehashed_fixture_mutation_rejected":
        externally_bound_mutation_rejected,
        "catastrophic_cancellation_exact_lower_bound": cancellation["terms"][
            "conservative_lower_bound"
        ]
        <= -1.0
        and cancellation["terms"]["lower_bound_method"]
        == "exact_binary_fraction_then_float_down",
        "bool_shape_rejected": bool_shape_rejected,
        "oversized_payload_rejected_before_base64_decode":
        oversized_predecode_rejected,
        "status9_but_certificate_good_is_status_independent":
        status9_certificate["formal_certificate_pass"]
        and status9_certificate["optimizer_status_not_used_for_acceptance"]
        and all(
            record["optimizer_status_audit_only"] == 9
            and record["optimizer_success_audit_only"] is False
            for record in status9_certificate["dual"]["records"]
            if record["evaluation_available"]
        ),
        "exact_gap_threshold_edge_rejects_legacy_float_false_pass":
        legacy_gap_edge_float_subtraction == CERT_DUAL_GAP_TOL_SCALED
        and gap_edge_certificate["conservative_gap_scaled"]
        > CERT_DUAL_GAP_TOL_SCALED
        and not gap_edge_certificate["checks"]["dual_gap_at_most_5e_13"]
        and not gap_edge_certificate["formal_certificate_pass"],
        "exact_stationarity_cancellation_rejects_float_zero_false_pass":
        stationarity_edge_certificate["stationarity_inf_float64_audit_only"] == 0.0
        and stationarity_edge_certificate["stationarity_inf_scaled"] > 1e-10
        and not stationarity_edge_certificate["checks"][
            "stationarity_inf_at_most_1e_10"
        ]
        and not stationarity_edge_certificate["formal_certificate_pass"],
        "optimizer_points_never_projected": all(
            record.get("optimizer_projection_applied") is False
            and record.get("optimizer_projection_inf") == 0.0
            for record in certificate["dual"]["records"]
            if record["evaluation_available"]
        )
        and certificate["dual"]["optimizer_point_projection_allowed"] is False,
        "bad_primal_rejected_even_if_optimizer_status_were_zero": not bad_primal[
            "formal_certificate_pass"
        ]
        and not bad_primal["checks"]["protocol_tolerance_feasible"],
        "duplicate_rank_deficient_no_slater_certified": duplicate[
            "formal_certificate_pass"
        ],
        "binary_power_row_scale_preserves_exact_problem": binary_scaled[
            "formal_certificate_pass"
        ]
        and binary_scaled["exact_feasibility"]["linear_min_sign"] == 0
        and np.array_equal(
            decode_array(binary_scaled["primal"]), np.asarray([1.0])
        ),
        "nonbinary_row_scale_rounding_rejected_by_exact_gate":
        rounded_scale_certificate["checks"]["protocol_tolerance_feasible"]
        and not rounded_scale_certificate["checks"]["exact_float_feasible"]
        and not rounded_scale_certificate["formal_certificate_pass"]
        and rounded_scale_certificate["exact_feasibility"]["linear_min_sign"] < 0,
        "negative_multiplier_rejected": negative_rejected,
        "formal_thresholds_exact": CERT_DUAL_GAP_TOL_SCALED == 5e-13
        and CERT_RAW_OPTIMAL_RADIUS_MAX == 1e-9
        and CERT_STATIONARITY_INF_TOL_SCALED == 1e-10
        and CERT_COMPLEMENTARITY_TOL_SCALED == 1e-10,
    }
    if not all(checks.values()):
        raise CertificateError(f"CW17 dual math selftest failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": "pure_CPU_selftest_passed",
        "checks": checks,
        "pass": True,
        "writes_performed": False,
        "cuda_accessed": False,
        "fixture_sha256": fixture["fixture_sha256"],
        "certificate": certificate,
        "status9_certificate": status9_certificate,
        "gap_threshold_edge_certificate": gap_edge_certificate,
        "stationarity_cancellation_edge_certificate":
        stationarity_edge_certificate,
        "duplicate_certificate": duplicate,
        "binary_scaled_certificate": binary_scaled,
        "nonbinary_rounded_scale_certificate": rounded_scale_certificate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "selftest"), default="static")
    parser.parse_args()
    result = selftest()
    print(canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
