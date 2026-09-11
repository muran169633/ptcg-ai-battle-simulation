#!/usr/bin/env python3
"""Thin quantization-aware wrapper for the frozen beta100 PET probe v1.

The frozen v1 designs every endpoint in CPU float64, then writes it to FP32
actor6 parameters.  This wrapper changes only that RAM write boundary.  A
nonzero planned delta is written, the actual FP32 lattice delta is measured,
and its P/E/T component error is subtracted in float64 before another write.
At most eight writes are attempted.  A call succeeds only when the measured
FP32 endpoint satisfies both PET tolerances and the hard cumulative radius.

The wrapper also independently audits every frozen-v1 SLSQP failure.  It
row-scales the exact linearized constraints, solves the minimum-norm dual by
Cholesky plus NNLS, and accepts either a KKT/strong-duality radius certificate
or a certified feasible fallback.  A numerically unverified result remains
explicitly inconclusive.

The exact-zero restoration path always delegates directly to frozen v1.  No
checkpoint, result, cache, or evidence file is written.  Standalone actual
execution forbids a candidate consumer and emits only a JSON report to stdout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / (
    "probe_u468_beta100_actor6_pet_orthogonal_quantized_design20260813_v2.py"
)
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = (
    "ptcg-u468-beta100-actor6-pet-orthogonal-quantized-design20260813-v2"
)

V1_TOOL = TOOLS / "probe_u468_beta100_actor6_pet_orthogonal_cuttingplane_v1.py"
V1_TOOL_SHA256 = "9b36284b2cfa8a6307c5ecfe542fe76735d74bf9f1bc6702728f46e9e94c75b2"
V1_EXPECTED_MODE = 0o444
PET_TOOL_SHA256 = "4f233db368f0150ee68109ee7d974a5a647ce8a35c919a5217f1714dddd98b2f"
TRANSITION_TOOL_SHA256 = "5f25ffbd54705b282edf4512195bc83fe783ef300bc42d9361b48c627d0bffbb"

MAX_LATTICE_ITERATIONS = 8
PET_COMPONENT_ABS_MAX = 2.0e-9
PET_COMPONENT_RELATIVE_L2_MAX = 1.0e-5
HARD_RADIUS_TOLERANCE = 1.0e-12

DUAL_NNLS_MAXITER = 100_000
DUAL_ACTIVE_LAMBDA_TOLERANCE = 1.0e-10
DUAL_PRIMAL_RESIDUAL_TOLERANCE = 1.0e-9
DUAL_GRADIENT_TOLERANCE = 1.0e-9
DUAL_COMPLEMENTARITY_TOLERANCE = 1.0e-9
DUALITY_GAP_TOLERANCE = 1.0e-9
DUAL_MIN_GRAM_EIGENVALUE = 1.0e-14


def canonical_json_bytes(value: Any) -> bytes:
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


def vector_sha256(vector: np.ndarray, dtype: str) -> str:
    value = np.ascontiguousarray(np.asarray(vector, dtype=dtype))
    return hashlib.sha256(value.tobytes()).hexdigest()


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
    mode = stat.S_IMODE(after.st_mode)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 drift: expected {expected_sha256}, observed {digest}"
        )
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(
            f"{label} mode drift: expected {oct(expected_mode)}, observed {oct(mode)}"
        )
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
        "single_link_regular_held_fd_identity_exact": True,
    }


def import_locked(
    path: Path,
    expected_sha256: str,
    module_name: str,
    *,
    expected_mode: int | None = None,
) -> tuple[ModuleType, dict[str, Any]]:
    payload, evidence = read_regular_bytes(
        path,
        expected_sha256,
        module_name,
        expected_mode=expected_mode,
    )
    spec = importlib.util.spec_from_file_location(
        f"_{module_name}_{sha256_bytes(payload)[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct import spec for {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


V1, V1_EVIDENCE = import_locked(
    V1_TOOL,
    V1_TOOL_SHA256,
    "frozen_beta100_pet_v1",
    expected_mode=V1_EXPECTED_MODE,
)


class QuantizationClosureError(RuntimeError):
    """Raised only after all bounded FP32 lattice correction attempts fail."""


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")
    if V1.PET_TOOL_SHA256 != PET_TOOL_SHA256:
        raise RuntimeError("frozen v1 PET dependency SHA drift")
    if V1.TRANSITION_TOOL_SHA256 != TRANSITION_TOOL_SHA256:
        raise RuntimeError("frozen v1 transition dependency SHA drift")
    if V1.ORTHOGONAL_ABS_TOLERANCE != PET_COMPONENT_ABS_MAX:
        raise RuntimeError("frozen v1 PET absolute tolerance drift")
    if V1.ORTHOGONAL_RELATIVE_L2_TOLERANCE != PET_COMPONENT_RELATIVE_L2_MAX:
        raise RuntimeError("frozen v1 PET relative tolerance drift")
    if V1.RADIUS_ABS_TOLERANCE > HARD_RADIUS_TOLERANCE:
        raise RuntimeError("frozen v1 radius tolerance is too loose")


def static_audit() -> dict[str, Any]:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_call_names = {
        "backward",
        "step",
        "save",
        "savez",
        "write",
        "write_text",
        "write_bytes",
        "touch",
        "mkdir",
        "makedirs",
        "unlink",
        "remove",
        "rmtree",
        "rename",
        "replace",
        "copy_",
    }
    forbidden_import_roots = {
        "requests",
        "urllib",
        "http",
        "socket",
        "subprocess",
        "kaggle",
    }
    call_hits = []
    import_hits = []
    write_flags = []
    print_sites = []
    monkeypatch_assignments = []
    span_feedback_assignments = []
    dual_endpoint_assignments = []
    cholesky_sites = []
    nnls_sites = []
    source_name_ids = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".", 1)[0] in forbidden_import_roots:
                    import_hits.append({"line": node.lineno, "name": name})
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"}
        ):
            write_flags.append({"line": node.lineno, "name": node.attr})
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "V1"
                ):
                    monkeypatch_assignments.append(
                        {"line": node.lineno, "attribute": target.attr}
                    )
                if (
                    isinstance(target, ast.Name)
                    and target.id == "request"
                    and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.op, ast.Sub)
                    and isinstance(node.value.right, ast.Name)
                    and node.value.right.id == "span_correction"
                ):
                    span_feedback_assignments.append(node.lineno)
                if (
                    isinstance(target, ast.Name)
                    and target.id == "endpoint"
                    and ast.unparse(node.value) == "scaled_matrix.T @ dual_lambda"
                ):
                    dual_endpoint_assignments.append(node.lineno)
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            name = ""
        if name in forbidden_call_names:
            call_hits.append({"line": node.lineno, "name": name})
        if name == "print":
            print_sites.append(node.lineno)
        if name == "cholesky":
            cholesky_sites.append(node.lineno)
        if name == "nnls":
            nnls_sites.append(node.lineno)
    signature = inspect.signature(run_probe)
    checks = {
        "ast_parse": True,
        "no_write_optimizer_backward_step_save_network_or_direct_copy": not (
            call_hits or import_hits or write_flags
        ),
        "only_actor_write_and_slsqp_failure_are_monkeypatched": (
            len(monkeypatch_assignments) == 4
            and {item["attribute"] for item in monkeypatch_assignments}
            == {"set_actor_delta", "solve_minimum_endpoint_l2"}
        ),
        "single_stdout_print_site": len(print_sites) == 1,
        "candidate_consumer_default_none": (
            signature.parameters["candidate_consumer"].default is None
        ),
        "lattice_iterations_at_most_8": MAX_LATTICE_ITERATIONS <= 8,
        "pet_abs_exact_2e_9": PET_COMPONENT_ABS_MAX == 2.0e-9,
        "pet_relative_exact_1e_5": PET_COMPONENT_RELATIVE_L2_MAX == 1.0e-5,
        "hard_radius_tolerance_at_most_1e_12": (
            HARD_RADIUS_TOLERANCE <= 1.0e-12
        ),
        "dual_nnls_iterations_bounded": DUAL_NNLS_MAXITER == 100_000,
        "dual_certificate_tolerances_at_most_1e_9": max(
            DUAL_PRIMAL_RESIDUAL_TOLERANCE,
            DUAL_GRADIENT_TOLERANCE,
            DUAL_COMPLEMENTARITY_TOLERANCE,
            DUALITY_GAP_TOLERANCE,
        )
        <= 1.0e-9,
        "span_only_feedback_formula_exactly_once": (
            len(span_feedback_assignments) == 1
            and "feedback_eta" not in source_name_ids
            and "full_target_error" not in source_name_ids
        ),
        "dual_cholesky_nnls_formula_present": (
            len(cholesky_sites) == 1
            and len(nnls_sites) == 1
            and len(dual_endpoint_assignments) == 1
        ),
        "runtime_requires_full_state_nonactor_actor_restore": all(
            token in source
            for token in (
                "full_state_before = V1.ppo.model_state_sha256(model)",
                "full_state_after = V1.ppo.model_state_sha256(model)",
                "nonactor_sha_after == nonactor_sha_before",
                "actor_sha_after == actor_sha_before",
                "restore_cardinality_pass",
                "if not full_restore_pass:",
            )
        ),
        "standalone_candidate_consumer_none": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v2 static audit failed: {checks}")
    return {
        "status": "static_quantization_wrapper_audit_passed",
        "checks": checks,
        "forbidden_call_hits": call_hits,
        "forbidden_import_hits": import_hits,
        "forbidden_os_write_flags": write_flags,
        "monkeypatch_assignments": monkeypatch_assignments,
        "span_feedback_assignment_sites": span_feedback_assignments,
        "dual_endpoint_assignment_sites": dual_endpoint_assignments,
        "cholesky_call_sites": cholesky_sites,
        "nnls_call_sites": nnls_sites,
        "stdout_print_sites": print_sites,
        "frozen_v1_static_audit": V1.static_audit(),
    }


def hard_radius_for_planned(planned_l2: float) -> float:
    if planned_l2 <= V1.INITIAL_CUMULATIVE_L2 + HARD_RADIUS_TOLERANCE:
        return float(V1.INITIAL_CUMULATIVE_L2)
    if planned_l2 <= V1.EXPANDED_CUMULATIVE_L2 + HARD_RADIUS_TOLERANCE:
        return float(V1.EXPANDED_CUMULATIVE_L2)
    raise QuantizationClosureError(
        f"planned endpoint exceeds hard expanded radius: {planned_l2}"
    )


def measured_projection(vector: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    value = np.asarray(vector, dtype=np.float64)
    components = q.T @ value
    norm = float(np.linalg.norm(value))
    component_l2 = float(np.linalg.norm(components))
    relative = component_l2 / norm if norm > 0.0 else 0.0
    maximum = float(np.max(np.abs(components), initial=0.0))
    return {
        "l2": norm,
        "pet_components": [float(item) for item in components],
        "pet_component_l2": component_l2,
        "pet_component_max_abs": maximum,
        "pet_component_relative_l2": relative,
        "abs_pass": maximum <= PET_COMPONENT_ABS_MAX,
        "relative_pass": relative <= PET_COMPONENT_RELATIVE_L2_MAX,
    }


def state_subset_sha256(model: Any, names: list[str]) -> str:
    state = model.state_dict()
    return V1.PET.frozen.model_state_sha256({name: state[name] for name in names})


def dual_minimum_norm_certificate(
    pair_gradients: np.ndarray,
    pair_rhs: np.ndarray,
    loss_gradients: np.ndarray,
    loss_rhs: np.ndarray,
    cumulative: np.ndarray,
    radius: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Certify the exact local linear system by its row-scaled convex dual."""

    report: dict[str, Any] = {
        "schema_version": f"{SCHEMA}-row-scaled-dual-certificate-v1",
        "solver": "Cholesky(AA^T) plus scipy.optimize.nnls",
        "radius": float(radius),
        "hard_radius_tolerance": HARD_RADIUS_TOLERANCE,
        "classification": "solver_failure_certificate_inconclusive",
        "certificate_pass": False,
    }
    try:
        matrix = np.concatenate((pair_gradients, -loss_gradients), axis=0)
        constraint_rhs = np.concatenate((pair_rhs, loss_rhs)).astype(
            np.float64, copy=False
        )
        cumulative_value = np.asarray(cumulative, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != V1.PET.EXPECTED_ACTOR_ELEMENTS
            or constraint_rhs.shape != (matrix.shape[0],)
            or cumulative_value.shape != (matrix.shape[1],)
            or not bool(np.all(np.isfinite(matrix)))
            or not bool(np.all(np.isfinite(constraint_rhs)))
            or not bool(np.all(np.isfinite(cumulative_value)))
        ):
            raise RuntimeError("dual certificate input shape/finite gate failed")

        effective_rhs = constraint_rhs + matrix @ cumulative_value
        row_norms = np.linalg.norm(matrix, axis=1)
        if not bool(np.all(np.isfinite(effective_rhs))) or bool(
            np.any(row_norms <= 0.0)
        ):
            raise RuntimeError("dual certificate row norm/effective RHS gate failed")

        scaled_matrix = matrix / row_norms[:, None]
        scaled_rhs = effective_rhs / row_norms
        row_ledger = []
        pair_count = int(pair_gradients.shape[0])
        for index in range(matrix.shape[0]):
            is_pair = index < pair_count
            kind_index = index if is_pair else index - pair_count
            row_ledger.append(
                {
                    "row_index": index,
                    "kind": "pair" if is_pair else "loss",
                    "kind_index": kind_index,
                    "loss_name": (
                        None if is_pair else str(V1.PET.LOSS_NAMES[kind_index])
                    ),
                    "raw_row_sha256_float64": vector_sha256(
                        matrix[index], "<f8"
                    ),
                    "raw_constraint_rhs": float(constraint_rhs[index]),
                    "raw_cumulative_dot": float(matrix[index] @ cumulative_value),
                    "raw_effective_rhs": float(effective_rhs[index]),
                    "row_l2": float(row_norms[index]),
                    "scaled_effective_rhs": float(scaled_rhs[index]),
                }
            )

        gram = scaled_matrix @ scaled_matrix.T
        gram = 0.5 * (gram + gram.T)
        eigenvalues = np.linalg.eigvalsh(gram)
        if (
            not bool(np.all(np.isfinite(eigenvalues)))
            or float(eigenvalues[0]) <= DUAL_MIN_GRAM_EIGENVALUE
        ):
            raise RuntimeError("dual Gram matrix is not certifiably positive definite")

        cholesky = np.linalg.cholesky(gram)
        nnls_matrix = cholesky.T
        nnls_target = np.linalg.solve(cholesky, scaled_rhs)
        dual_lambda, nnls_rnorm = V1.optimize.nnls(
            nnls_matrix,
            nnls_target,
            maxiter=DUAL_NNLS_MAXITER,
        )
        dual_lambda = np.asarray(dual_lambda, dtype=np.float64)
        endpoint = scaled_matrix.T @ dual_lambda
        scaled_residual = scaled_matrix @ endpoint - scaled_rhs
        raw_residual = matrix @ endpoint - effective_rhs
        dual_gradient = gram @ dual_lambda - scaled_rhs
        active = dual_lambda > DUAL_ACTIVE_LAMBDA_TOLERANCE
        inactive = ~active
        complementarity = dual_lambda * scaled_residual
        stationarity = endpoint - scaled_matrix.T @ dual_lambda
        primal_objective = 0.5 * float(endpoint @ endpoint)
        dual_objective = float(scaled_rhs @ dual_lambda) - 0.5 * float(
            dual_lambda @ (gram @ dual_lambda)
        )
        duality_gap = primal_objective - dual_objective
        inactive_gradient_min = (
            float(dual_gradient[inactive].min()) if bool(inactive.any()) else None
        )
        active_gradient_max_abs = (
            float(np.abs(dual_gradient[active]).max())
            if bool(active.any())
            else 0.0
        )
        endpoint_l2 = float(np.linalg.norm(endpoint))
        radius_margin = float(radius) - endpoint_l2

        kkt_checks = {
            "dual_lambda_nonnegative": float(dual_lambda.min())
            >= -DUAL_GRADIENT_TOLERANCE,
            "scaled_primal_residual": float(scaled_residual.min())
            >= -DUAL_PRIMAL_RESIDUAL_TOLERANCE,
            "raw_primal_residual": float(raw_residual.min())
            >= -DUAL_PRIMAL_RESIDUAL_TOLERANCE,
            "active_dual_gradient": active_gradient_max_abs
            <= DUAL_GRADIENT_TOLERANCE,
            "inactive_dual_margin": (
                inactive_gradient_min is None
                or inactive_gradient_min >= -DUAL_GRADIENT_TOLERANCE
            ),
            "complementarity": float(np.abs(complementarity).max(initial=0.0))
            <= DUAL_COMPLEMENTARITY_TOLERANCE,
            "stationarity_by_construction": float(
                np.abs(stationarity).max(initial=0.0)
            )
            <= DUAL_GRADIENT_TOLERANCE,
            "strong_duality": abs(duality_gap) <= DUALITY_GAP_TOLERANCE,
        }
        certificate_pass = all(kkt_checks.values())
        if certificate_pass and endpoint_l2 > radius + HARD_RADIUS_TOLERANCE:
            classification = "certified_linearized_radius_infeasible"
        elif certificate_pass:
            classification = "certified_feasible_minimum_norm_fallback"
        else:
            classification = "solver_failure_certificate_inconclusive"

        report.update(
            {
                "classification": classification,
                "certificate_pass": certificate_pass,
                "constraint_rows": int(matrix.shape[0]),
                "actor6_columns": int(matrix.shape[1]),
                "pair_rows": pair_count,
                "loss_rows": int(loss_gradients.shape[0]),
                "raw_A_sha256_float64": vector_sha256(matrix, "<f8"),
                "raw_constraint_rhs_sha256_float64": vector_sha256(
                    constraint_rhs, "<f8"
                ),
                "raw_d_effective_rhs_sha256_float64": vector_sha256(
                    effective_rhs, "<f8"
                ),
                "cumulative_sha256_float64": vector_sha256(
                    cumulative_value, "<f8"
                ),
                "row_ledger_sha256": sha256_bytes(
                    canonical_json_bytes(row_ledger)
                ),
                "row_ledger": row_ledger,
                "row_l2_min": float(row_norms.min()),
                "row_l2_max": float(row_norms.max()),
                "scaled_A_sha256_float64": vector_sha256(
                    scaled_matrix, "<f8"
                ),
                "scaled_d_sha256_float64": vector_sha256(scaled_rhs, "<f8"),
                "scaled_d_min": float(scaled_rhs.min()),
                "scaled_d_max": float(scaled_rhs.max()),
                "gram_sha256_float64": vector_sha256(gram, "<f8"),
                "gram_min_eigenvalue": float(eigenvalues[0]),
                "gram_max_eigenvalue": float(eigenvalues[-1]),
                "gram_condition_number": float(
                    eigenvalues[-1] / eigenvalues[0]
                ),
                "nnls_max_iterations": DUAL_NNLS_MAXITER,
                "nnls_residual_l2": float(nnls_rnorm),
                "lambda_sha256_float64": vector_sha256(dual_lambda, "<f8"),
                "lambda_min": float(dual_lambda.min()),
                "lambda_max": float(dual_lambda.max()),
                "lambda_active_count": int(active.sum()),
                "primal_endpoint_sha256_float64": vector_sha256(
                    endpoint, "<f8"
                ),
                "primal_endpoint_l2": endpoint_l2,
                "radius_margin": radius_margin,
                "radius_excess": endpoint_l2 - float(radius),
                "inside_hard_radius": endpoint_l2
                <= float(radius) + HARD_RADIUS_TOLERANCE,
                "scaled_primal_residual_min": float(scaled_residual.min()),
                "scaled_primal_residual_max": float(scaled_residual.max()),
                "raw_primal_residual_min": float(raw_residual.min()),
                "raw_primal_residual_max": float(raw_residual.max()),
                "active_dual_gradient_max_abs": active_gradient_max_abs,
                "inactive_dual_margin_min": inactive_gradient_min,
                "complementarity_max_abs": float(
                    np.abs(complementarity).max(initial=0.0)
                ),
                "stationarity_max_abs": float(
                    np.abs(stationarity).max(initial=0.0)
                ),
                "primal_objective": primal_objective,
                "dual_objective": dual_objective,
                "duality_gap": duality_gap,
                "kkt_tolerances": {
                    "active_lambda": DUAL_ACTIVE_LAMBDA_TOLERANCE,
                    "primal_residual": DUAL_PRIMAL_RESIDUAL_TOLERANCE,
                    "dual_gradient": DUAL_GRADIENT_TOLERANCE,
                    "complementarity": DUAL_COMPLEMENTARITY_TOLERANCE,
                    "duality_gap": DUALITY_GAP_TOLERANCE,
                },
                "kkt_checks": kkt_checks,
            }
        )
        return endpoint, report
    except Exception as error:
        report.update(
            {
                "classification": "solver_failure_certificate_inconclusive",
                "certificate_pass": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        return None, report


def run_probe(
    transition_context: Mapping[str, Any] | None = None,
    candidate_consumer: Any | None = None,
) -> dict[str, Any]:
    if candidate_consumer is not None:
        raise ValueError("design20260813 v2 forbids candidate_consumer before review")

    if transition_context is None:
        transition, transition_evidence = V1.load_transition_module()
        effective_context = V1.build_actual_transition_context(transition)
        context_origin = "built_by_frozen_v1_transition_callback"
    else:
        effective_context = transition_context
        transition = effective_context.get("_transition_module")
        transition_evidence = {"injected_context": True}
        context_origin = "injected_context"
    if not isinstance(transition, ModuleType):
        raise RuntimeError("v2 requires the hash-bound transition module in context")

    model = effective_context["beta_model"]
    expected_beta_sha = str(effective_context["beta_model_state_sha256"])
    full_state_before = V1.ppo.model_state_sha256(model)
    if full_state_before != expected_beta_sha:
        raise RuntimeError("v2 beta100 model state SHA drift before wrapper")
    state_names = sorted(model.state_dict())
    nonactor_names = sorted(set(state_names) - set(V1.PET.ACTOR_NAMES))
    actor_names = list(V1.PET.ACTOR_NAMES)
    nonactor_sha_before = state_subset_sha256(model, nonactor_names)
    actor_sha_before = state_subset_sha256(model, actor_names)

    basis = V1.PET.build_basis_geometry()
    q = np.ascontiguousarray(basis["q"].numpy(), dtype=np.float64)
    original_set_actor_delta = V1.set_actor_delta
    original_solve = V1.solve_minimum_endpoint_l2
    lattice_calls: list[dict[str, Any]] = []
    restore_calls: list[dict[str, Any]] = []
    dual_certificates: list[dict[str, Any]] = []

    def quantized_set_actor_delta(
        parameters: Mapping[str, Any],
        beta_actor: Mapping[str, Any],
        delta: np.ndarray,
    ) -> None:
        planned = np.asarray(delta, dtype=np.float64)
        if planned.shape != (V1.PET.EXPECTED_ACTOR_ELEMENTS,):
            raise RuntimeError(f"planned actor6 shape drift: {planned.shape}")
        if not bool(np.all(np.isfinite(planned))):
            raise FloatingPointError("planned actor6 delta is nonfinite")
        if not bool(np.any(planned != 0.0)):
            original_set_actor_delta(parameters, beta_actor, planned)
            exact = all(
                V1.torch.equal(parameters[name].detach(), beta_actor[name])
                for name in V1.PET.ACTOR_NAMES
            )
            restore_calls.append(
                {
                    "restore_index": len(restore_calls) + 1,
                    "delegated_directly_to_frozen_v1": True,
                    "actor6_tensor_exact_beta100_after_delegate": exact,
                }
            )
            if not exact:
                raise RuntimeError("delegated frozen-v1 zero restoration was not exact")
            return

        call_audit: dict[str, Any] = {
            "call_index": len(lattice_calls) + 1,
            "planned_float64_sha256": vector_sha256(planned, "<f8"),
            "maximum_lattice_iterations": MAX_LATTICE_ITERATIONS,
            "update_rule": "request_next=request-Q(Q^T FP32_actual)",
            "equivalent_error_feedback_rule": (
                "request_next=planned-Q(Q^T(FP32_actual-request)) while "
                "request-planned remains in span(Q)"
            ),
            "cycle_detection": (
                "only an exactly repeated request_sha256 plus actual_sha256 "
                "state terminates early; repeated actual alone is diagnostic"
            ),
            "iterations": [],
            "converged": False,
        }
        try:
            planned_audit = measured_projection(planned, q)
            call_audit["planned_projection"] = planned_audit
            if not (planned_audit["abs_pass"] and planned_audit["relative_pass"]):
                raise QuantizationClosureError(
                    "incoming float64 endpoint is not PET-orthogonal"
                )
            hard_radius = hard_radius_for_planned(planned_audit["l2"])
            call_audit["hard_radius"] = hard_radius
            call_audit["hard_radius_tolerance"] = HARD_RADIUS_TOLERANCE
            request = planned.copy()
            seen_states: dict[tuple[str, str], int] = {}
            seen_actual: dict[str, int] = {}
            failure_reason = "maximum_lattice_iterations_reached"

            for iteration in range(1, MAX_LATTICE_ITERATIONS + 1):
                request_audit = measured_projection(request, q)
                request_sha = vector_sha256(request, "<f8")
                request_radius_pass = request_audit["l2"] <= (
                    hard_radius + HARD_RADIUS_TOLERANCE
                )
                if not request_radius_pass:
                    failure_reason = "lattice_command_exceeds_hard_radius"
                    call_audit["iterations"].append(
                        {
                            "iteration": iteration,
                            "requested_float64_sha256": request_sha,
                            "requested_projection": request_audit,
                            "request_minus_planned_l2": float(
                                np.linalg.norm(request - planned)
                            ),
                            "request_hard_radius_pass": False,
                            "write_performed": False,
                            "pass": False,
                        }
                    )
                    break

                original_set_actor_delta(parameters, beta_actor, request)
                actual = V1.actor_delta_from_model(parameters, beta_actor)
                actual_audit = measured_projection(actual, q)
                actual_sha = vector_sha256(actual, "<f4")
                state_key = (request_sha, actual_sha)
                repeated_state_from = seen_states.get(state_key)
                repeated_actual_from = seen_actual.get(actual_sha)
                seen_states[state_key] = iteration
                seen_actual[actual_sha] = iteration
                actual_radius_pass = actual_audit["l2"] <= (
                    hard_radius + HARD_RADIUS_TOLERANCE
                )
                span_correction = q @ np.asarray(
                    actual_audit["pet_components"], dtype=np.float64
                )
                iteration_record = {
                    "iteration": iteration,
                    "requested_float64_sha256": request_sha,
                    "requested_projection": request_audit,
                    "request_minus_planned_l2": float(
                        np.linalg.norm(request - planned)
                    ),
                    "request_hard_radius_pass": request_radius_pass,
                    "write_performed": True,
                    "actual_float32_sha256": actual_sha,
                    "actual_projection": actual_audit,
                    "actual_minus_planned_l2": float(
                        np.linalg.norm(actual - planned)
                    ),
                    "actual_minus_request_l2": float(
                        np.linalg.norm(actual - request)
                    ),
                    "span_correction_coordinates": list(
                        actual_audit["pet_components"]
                    ),
                    "span_correction_l2": float(np.linalg.norm(span_correction)),
                    "actual_lattice_point_repeated": repeated_actual_from
                    is not None,
                    "actual_lattice_point_first_seen_iteration": repeated_actual_from,
                    "request_actual_state_cycle": repeated_state_from is not None,
                    "request_actual_state_first_seen_iteration": repeated_state_from,
                    "hard_radius": hard_radius,
                    "hard_radius_tolerance": HARD_RADIUS_TOLERANCE,
                    "actual_hard_radius_pass": actual_radius_pass,
                    "pass": (
                        actual_audit["abs_pass"]
                        and actual_audit["relative_pass"]
                        and actual_radius_pass
                    ),
                }
                call_audit["iterations"].append(iteration_record)
                if iteration_record["pass"]:
                    call_audit["converged"] = True
                    call_audit["terminal_actual_float32_sha256"] = actual_sha
                    call_audit["terminal_actual_projection"] = actual_audit
                    failure_reason = "none"
                    break
                if repeated_state_from is not None:
                    failure_reason = "exact_request_actual_state_cycle"
                    break
                request = request - span_correction
                if not bool(np.all(np.isfinite(request))):
                    raise FloatingPointError(
                        "PET-span lattice-corrected request is nonfinite"
                    )

            call_audit["iteration_count"] = len(call_audit["iterations"])
            call_audit["failure_reason"] = failure_reason
        except Exception as error:
            call_audit["failure_reason"] = str(error)
            call_audit["error_type"] = type(error).__name__
            call_audit["iteration_count"] = len(call_audit["iterations"])
            lattice_calls.append(call_audit)
            raise

        lattice_calls.append(call_audit)
        if not call_audit["converged"]:
            raise QuantizationClosureError(
                "FP32 lattice PET projection did not converge within bounded attempts: "
                f"{call_audit['failure_reason']}"
            )

    def certified_solve_minimum_endpoint_l2(
        pair_gradients: np.ndarray,
        pair_rhs: np.ndarray,
        loss_gradients: np.ndarray,
        loss_rhs: np.ndarray,
        cumulative: np.ndarray,
        radius: float,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        correction, slsqp_audit = original_solve(
            pair_gradients,
            pair_rhs,
            loss_gradients,
            loss_rhs,
            cumulative,
            radius,
        )
        if correction is not None:
            return correction, slsqp_audit

        endpoint, certificate = dual_minimum_norm_certificate(
            pair_gradients,
            pair_rhs,
            loss_gradients,
            loss_rhs,
            cumulative,
            radius,
        )
        certificate["certificate_index"] = len(dual_certificates) + 1
        certificate["original_slsqp_failure"] = slsqp_audit
        dual_certificates.append(certificate)
        audit = {
            "objective": "minimum cumulative actor6 endpoint L2",
            "solver": "frozen-v1 SLSQP then independent row-scaled dual",
            "success": certificate["classification"]
            == "certified_feasible_minimum_norm_fallback",
            "radius": float(radius),
            "original_slsqp_failure": slsqp_audit,
            "independent_dual_certificate": certificate,
            "failure_classification": certificate["classification"],
        }
        if certificate["classification"] == "certified_feasible_minimum_norm_fallback":
            if endpoint is None:
                raise RuntimeError("certified feasible dual has no endpoint")
            fallback_correction = endpoint - np.asarray(cumulative, dtype=np.float64)
            audit.update(
                {
                    "candidate_l2": float(np.linalg.norm(endpoint)),
                    "correction_l2": float(np.linalg.norm(fallback_correction)),
                    "fallback_endpoint_sha256_float64": vector_sha256(
                        endpoint, "<f8"
                    ),
                }
            )
            return fallback_correction, audit
        audit["failure"] = certificate["classification"]
        return None, audit

    V1.set_actor_delta = quantized_set_actor_delta
    V1.solve_minimum_endpoint_l2 = certified_solve_minimum_endpoint_l2
    result: dict[str, Any] | None = None
    caught_quantization_error: QuantizationClosureError | None = None
    caught_unexpected_error: Exception | None = None
    try:
        try:
            result = V1.run_probe(
                transition_context=effective_context,
                candidate_consumer=None,
            )
        except QuantizationClosureError as error:
            caught_quantization_error = error
            result = {
                "schema_version": SCHEMA,
                "status": "closed_no_candidate",
                "decision": {
                    "status": "closed_no_candidate",
                    "close_reason": "quantized_fp32_pet_projection_failed_closed",
                    "error": str(error),
                    "model_materialized": False,
                },
            }
        except Exception as error:
            caught_unexpected_error = error
    finally:
        V1.set_actor_delta = original_set_actor_delta
        V1.solve_minimum_endpoint_l2 = original_solve

    full_state_after = V1.ppo.model_state_sha256(model)
    nonactor_sha_after = state_subset_sha256(model, nonactor_names)
    actor_sha_after = state_subset_sha256(model, actor_names)
    expected_restore_calls = 1 if lattice_calls else 0
    restore_cardinality_pass = len(restore_calls) == expected_restore_calls
    zero_restore_records_pass = all(
        call["actor6_tensor_exact_beta100_after_delegate"] for call in restore_calls
    )
    full_restore_pass = (
        full_state_after == full_state_before == expected_beta_sha
        and nonactor_sha_after == nonactor_sha_before
        and actor_sha_after == actor_sha_before
        and restore_cardinality_pass
        and zero_restore_records_pass
    )
    if not full_restore_pass:
        raise RuntimeError(
            "v2 terminal full/nonactor/actor restoration or cardinality gate failed"
        ) from caught_unexpected_error
    if caught_unexpected_error is not None:
        raise caught_unexpected_error
    if result is None:
        raise RuntimeError("v2 produced no result")

    classifications = [item["classification"] for item in dual_certificates]
    last_certificate = dual_certificates[-1] if dual_certificates else None
    underlying_close_reason = result.get("decision", {}).get("close_reason")
    if caught_quantization_error is not None:
        authoritative_close_reason = "quantized_fp32_pet_projection_failed_closed"
    elif (
        underlying_close_reason == "projected_qp_infeasible_inside_final_radius"
        and last_certificate is not None
    ):
        authoritative_close_reason = str(last_certificate["classification"])
    else:
        authoritative_close_reason = underlying_close_reason

    result["quantization_wrapper"] = {
        "schema_version": SCHEMA,
        "frozen_v1": V1_EVIDENCE,
        "transition": transition_evidence,
        "context_origin": context_origin,
        "monkeypatch_scope": [
            "set_actor_delta",
            "solve_minimum_endpoint_l2_on_slsqp_failure",
        ],
        "nonzero_write_contract": (
            "write the frozen-v1 continuous endpoint, measure the actual FP32 "
            "delta, subtract only Q(Q^T actual) from the next ephemeral command, "
            "and retain planned/command/actual evidence"
        ),
        "zero_restore_contract": (
            "delegate exact frozen-v1 zero restoration, then verify full, "
            "nonactor, and actor state hashes"
        ),
        "pet_abs_max": PET_COMPONENT_ABS_MAX,
        "pet_relative_l2_max": PET_COMPONENT_RELATIVE_L2_MAX,
        "hard_radius_tolerance": HARD_RADIUS_TOLERANCE,
        "nonzero_calls": lattice_calls,
        "nonzero_call_count": len(lattice_calls),
        "all_nonzero_calls_converged": bool(lattice_calls)
        and all(call["converged"] for call in lattice_calls),
        "zero_restore_calls": restore_calls,
        "zero_restore_call_count": len(restore_calls),
        "expected_zero_restore_call_count": expected_restore_calls,
        "zero_restore_call_cardinality_exact": restore_cardinality_pass,
        "all_zero_restore_calls_exact": zero_restore_records_pass,
        "terminal_state_restore": {
            "expected_beta100_model_state_sha256": expected_beta_sha,
            "full_model_state_sha256_before": full_state_before,
            "full_model_state_sha256_after": full_state_after,
            "full_model_state_restored_exact": full_state_after
            == full_state_before
            == expected_beta_sha,
            "nonactor_state_sha256_before": nonactor_sha_before,
            "nonactor_state_sha256_after": nonactor_sha_after,
            "nonactor_state_restored_exact": nonactor_sha_after
            == nonactor_sha_before,
            "actor6_state_sha256_before": actor_sha_before,
            "actor6_state_sha256_after": actor_sha_after,
            "actor6_state_restored_exact": actor_sha_after == actor_sha_before,
            "combined_restore_pass": full_restore_pass,
        },
        "slsqp_failure_dual_certificates": dual_certificates,
        "slsqp_failure_count": len(dual_certificates),
        "dual_certificate_classifications": classifications,
        "all_slsqp_failures_certified_or_feasible": all(
            value
            in {
                "certified_linearized_radius_infeasible",
                "certified_feasible_minimum_norm_fallback",
            }
            for value in classifications
        ),
        "authoritative_decision": {
            "underlying_frozen_v1_close_reason": underlying_close_reason,
            "authoritative_close_reason": authoritative_close_reason,
            "local_linearized_scope_only": authoritative_close_reason
            == "certified_linearized_radius_infeasible",
        },
        "candidate_consumer": None,
        "checkpoint_or_result_writes": 0,
    }
    return result


def cache_audit() -> dict[str, Any]:
    transition, transition_evidence = V1.load_transition_module()
    result = V1.cache_audit(transition)
    if result.get("status") != "cache_audit_passed_zero_write_train_only":
        raise RuntimeError("frozen-v1 cache audit status drift")
    return {
        "schema_version": SCHEMA,
        "status": "quantized_v2_cache_audit_passed",
        "frozen_v1": V1_EVIDENCE,
        "transition": transition_evidence,
        "frozen_v1_cache": result,
        "quantized_write_executed": False,
        "writes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("static-audit", "cache-audit", "actual"),
        required=True,
    )
    args = parser.parse_args()
    validate_runtime()
    static = static_audit()
    if args.mode == "static-audit":
        transition, transition_evidence = V1.load_transition_module()
        del transition
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "static_audit_passed",
            "self": read_regular_bytes(SCRIPT, None, "self")[1],
            "frozen_v1": V1_EVIDENCE,
            "transition": transition_evidence,
            "audit": static,
        }
    elif args.mode == "cache-audit":
        result = {**cache_audit(), "static_audit": static}
    else:
        result = run_probe(transition_context=None, candidate_consumer=None)
        result["static_audit"] = static
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
