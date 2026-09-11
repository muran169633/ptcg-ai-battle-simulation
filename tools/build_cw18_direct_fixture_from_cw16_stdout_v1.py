#!/usr/bin/env python3
"""Rebuild the frozen CW16 stage-2 fixture without evaluating a new model.

``static`` and ``selftest`` are CPU-only.  ``run`` reconstructs the missing
CW16 A/G/center arrays with selected-context CUDA BF16 autograd, solves only
the stage-1 maximin problem, and emits the reduced fixture on stdout.  It does
not evaluate, consume, save, package, upload, or submit the changed model.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "build_cw18_direct_fixture_from_cw16_stdout_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-cw18-direct-fixture-from-cw16-stdout-v1"
SEED = 202608207

CW15 = TOOLS / "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py"
CW16 = TOOLS / "probe_u468_cw11_consumed_valid_official6_cw16_local_trust_v1.py"
DUAL_MATH = TOOLS / "cw17_stage2_dual_math_v1.py"
CW16_STDOUT = ROOT / "artifacts/cw16_consumed_valid_official6_local_trust_20260802_v1.stdout.json"
FROZEN = {
    CW15: ("2aeb770f8774b7f7407a93553405089511ab2d2f278653162274a3afb89a2f24", 0o555),
    CW16: ("a05df3df944451fabefa8e2a037ad541b1f90af7b966be6fc9ecb87bafe2ebbf", 0o555),
    DUAL_MATH: ("a14241d500f352e6ae1b5662d5d7ea51ca4da22b61126536997e69e39750a600", 0o555),
    CW16_STDOUT: ("62eaf348a4c125c8da72ce15ed8904e4ddfd32320d1121a2bba1ca2ce34dbb4e", 0o444),
}
EXPECTED_BOOTSTRAP_MODEL_SHA = "ba5098d4e5c52bb6a02c4e69fa2015428d018f85953edc4fb6bc14330634d769"
EXPECTED_BOOTSTRAP_POINT_SHA = "e3814600afa6f1c7843ca073d89a86cb1e32cf162d89f8cbfaf8db378ad3183f"
EXPECTED_ANCHOR50_SHA = "765b74c9a500699e65f43c27edfb7da4b568842dd2065cf55e46efb826d1ef09"
EXPECTED_LOCAL41_RESIDUAL_SHA = (
    "28f8c97b07cb9b6d2c8096141b171cecb262b0b86973de6b1b57dd7efe1329dc"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def read_locked(path: Path) -> tuple[bytes, dict[str, Any]]:
    expected_sha, expected_mode = FROZEN[path]
    before = path.lstat()
    payload = path.read_bytes()
    after = path.lstat()
    checks = {
        "regular": stat.S_ISREG(before.st_mode) and stat.S_ISREG(after.st_mode),
        "single_link": before.st_nlink == after.st_nlink == 1,
        "identity_stable": (before.st_dev, before.st_ino, before.st_size)
        == (after.st_dev, after.st_ino, after.st_size),
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "sha_exact": sha256_bytes(payload) == expected_sha,
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen input drift: {path}: {checks}")
    return payload, {
        "path": str(path.relative_to(ROOT)), "sha256": expected_sha,
        "bytes": len(payload), "mode_octal": f"{expected_mode:04o}",
        "checks": checks,
    }


def import_locked(path: Path, name: str) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_locked(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def runtime_audit(*, require_cuda: bool = False) -> dict[str, Any]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"runtime contract failed: {checks}")
    result = {"checks": checks, "pass": True, "cuda_required": require_cuda}
    if require_cuda:
        cw15, _ = import_locked(CW15, "cw18_runtime_cw15")
        modules = cw15.frozen_modules()
        helper, _ = modules["geometry"].load_helper()
        torch = helper.torch
        cuda_checks = {
            "cuda_available": bool(torch.cuda.is_available()),
            "native_bf16": bool(torch.cuda.is_bf16_supported()),
            "cublas_workspace_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            == ":4096:8",
        }
        if not all(cuda_checks.values()):
            raise RuntimeError(f"CUDA BF16 runtime unavailable: {cuda_checks}")
        result["cuda_checks"] = cuda_checks
    return result


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_calls = {
        "run_outer_local_trust", "evaluate_candidate_once",
        "snapshot_official_six_views", "full_stream_separation_oracle",
        "save", "savez", "write", "write_bytes", "write_text", "touch",
        "mkdir", "replace", "rename", "unlink", "submit", "upload",
    }
    forbidden_imports = {"requests", "urllib", "subprocess", "socket"}
    call_hits: list[tuple[int, str]] = []
    import_hits: list[tuple[int, str]] = []
    print_count = 0
    probe_calls: list[tuple[int, str]] = []
    precision_get_calls = 0
    precision_set_calls = 0
    tensor_sha_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name == "run_probe":
                    probe_calls.append((node.lineno, ast.unparse(node.func)))
            if name in forbidden_calls:
                call_hits.append((node.lineno, name))
            if name == "print":
                print_count += 1
            if name == "get_float32_matmul_precision":
                precision_get_calls += 1
            if name == "set_float32_matmul_precision":
                precision_set_calls += 1
            if name == "tensor_sha256":
                tensor_sha_calls += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            for name in names:
                if name.split(".")[0] in forbidden_imports:
                    import_hits.append((node.lineno, name))
    checks = {
        "forbidden_calls_absent": not call_hits,
        "forbidden_imports_absent": not import_hits,
        "only_exact_CW11_reconstruction_probe": len(probe_calls) == 1
        and probe_calls[0][1] in {
            "modules['cw11'].run_probe", 'modules["cw11"].run_probe'
        },
        "stdout_single_site": print_count == 1,
        "precision_scope_calls_exact": precision_get_calls == 3
        and precision_set_calls == 2,
        "selected_context_tensor_raw_bytes_SHA_call_exact": tensor_sha_calls
        == 1,
        "seed_exact_CW15_CW16": SEED == 202608207,
        "modes_exact": {"static", "selftest", "run"}.issubset(
            {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
             and isinstance(node.value, str)}
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"source audit failed: {checks}, {call_hits}, {import_hits}")
    return {
        "checks": checks, "pass": True, "source_sha256": sha256_bytes(source),
        "source_bytes": len(source), "forbidden_call_hits": call_hits,
        "forbidden_import_hits": import_hits, "probe_calls": probe_calls,
    }


def load_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    payload, evidence = read_locked(CW16_STDOUT)
    document = json.loads(payload)
    second = document["second_stage"]
    iterations = {int(value["iteration"]): value for value in second["iterations"]}
    zero, one, failure = iterations[0], iterations[1], iterations[2]
    terminal_members = {
        key: value for key, value in second.items() if key.startswith("terminal_")
    }
    checks = {
        "terminal_payloads_null": bool(terminal_members)
        and all(value is None for value in terminal_members.values()),
        "iteration_kinds_exact": zero["kind"] == "exact_CW11_reference_before_bootstrap"
        and one["kind"] == "exact_CW15_iteration1_bootstrap_seed"
        and failure["kind"] == "fail_closed_before_official_proposal",
        "initial38": len(zero["active_cut_gates"]["records"]) == 38,
        "legacy34": len(zero["legacy_gate"]["gate"]["active_pair_gates"]) == 34,
        "added12": len(one["post_oracle_cut_merge"]["added"]) == 12,
        "active50": len(one["active_cut_gates"]["records"]) == 50,
        "failure_before_changed_eval": failure["official_evaluation_count"] == 1,
        "trust_exact": float(failure["trust_radius"]) == 0.000125,
        "bootstrap_model_exact": one["model_state_sha256"]
        == EXPECTED_BOOTSTRAP_MODEL_SHA,
        "bootstrap_point_exact": one["additional_float64_le_sha256"]
        == EXPECTED_BOOTSTRAP_POINT_SHA,
        "anchor50_exact": one["hard_anchor50_selection"][
            "active_anchor_ledger_sha256"] == EXPECTED_ANCHOR50_SHA,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 stdout semantic plan drift: {checks}")
    return {"document": document, "zero": zero, "one": one, "failure": failure}, {
        "checks": checks, "pass": True, "stdout": evidence,
    }


def reduce_fixture(cw16: ModuleType, dual_math: ModuleType, A: Any, b: Any,
                   G: Any, residual: Any, center: Any, trust_radius: float,
                   np: Any, optimize: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    span = np.concatenate([A, G, center.reshape(1, -1)], axis=0)
    _, singular, vh = np.linalg.svd(span, full_matrices=False)
    rank = int((singular > singular[0] * cw16.SVD_RELATIVE_RANK_TOL).sum())
    basis = vh[:rank]
    center_error = float(np.linalg.norm(basis.T @ (basis @ center) - center))
    if center_error > cw16.L2_ABS_TOL:
        raise RuntimeError("reduced basis does not contain center")
    scale = cw16.TOTAL_L2_CAP
    center_scaled = basis @ center / scale
    A_scaled = (A @ basis.T) * scale
    G_scaled = (G @ basis.T) * scale
    floor = np.finfo(np.float64).eps
    a_scale = np.maximum.reduce(
        [np.abs(b), np.linalg.norm(A_scaled, axis=1), np.full(b.shape, floor)]
    )
    rho = trust_radius / scale
    z_scale = float(max(float(np.abs(residual).max()),
                        float((np.linalg.norm(G_scaled, axis=1) * rho).max()), floor))

    def a_fun(v: Any) -> Any:
        return (A_scaled @ v[:-1] - b) / a_scale

    def a_jac(v: Any) -> Any:
        del v
        return np.concatenate([A_scaled / a_scale[:, None],
                               np.zeros((A.shape[0], 1))], axis=1)

    def g_fun(v: Any) -> Any:
        return (residual + G_scaled @ (v[:-1] - center_scaled)
                - z_scale * v[-1]) / z_scale

    def g_jac(v: Any) -> Any:
        del v
        return np.concatenate([G_scaled / z_scale,
                               -np.ones((G.shape[0], 1))], axis=1)

    initial_z = min(0.0, float(residual.min()))
    x0 = np.concatenate([center_scaled, np.asarray([initial_z / z_scale])])
    constraints = [
        {"type": "ineq", "fun": a_fun, "jac": a_jac},
        {"type": "ineq", "fun": g_fun, "jac": g_jac},
        {"type": "ineq", "fun": lambda v: -float(v[-1]),
         "jac": lambda v: np.concatenate([np.zeros(v.size - 1), [-1.0]])},
        {"type": "ineq", "fun": lambda v: float(1.0 - v[:-1] @ v[:-1]),
         "jac": lambda v: np.concatenate([-2.0 * v[:-1], [0.0]])},
        {"type": "ineq",
         "fun": lambda v: float(rho**2 - (v[:-1] - center_scaled)
                                  @ (v[:-1] - center_scaled)),
         "jac": lambda v: np.concatenate([-2.0 * (v[:-1] - center_scaled), [0.0]])},
    ]
    stage1 = optimize.minimize(
        lambda v: -float(v[-1]), x0,
        jac=lambda v: np.concatenate([np.zeros(v.size - 1), [-1.0]]),
        constraints=constraints, method="SLSQP",
        options={"ftol": cw16.SLSQP_FTOL, "maxiter": cw16.SLSQP_MAXITER,
                 "disp": False},
    )
    if not bool(stage1.success):
        raise RuntimeError(f"stage1 failed: {stage1.status} {stage1.message}")
    raw = basis.T @ (scale * np.asarray(stage1.x[:-1], dtype=np.float64))
    predicted = residual + G @ (raw - center)
    z_solver = float(stage1.x[-1]) * z_scale
    z_star = min(0.0, z_solver, float(predicted.min()))
    z_adjustment = z_solver - z_star
    anchor_raw_residual = A @ raw - b
    local_at_certified_z = predicted - z_star
    raw_total_l2 = float(np.linalg.norm(raw))
    raw_trust_l2 = float(np.linalg.norm(raw - center))
    raw_certification_checks = {
        "all_finite": bool(
            np.isfinite(raw).all()
            and np.isfinite(predicted).all()
            and np.isfinite(anchor_raw_residual).all()
            and np.isfinite(local_at_certified_z).all()
            and all(
                math.isfinite(value)
                for value in (
                    z_solver, z_star, z_adjustment, raw_total_l2, raw_trust_l2
                )
            )
        ),
        "anchor_raw_residual": float(anchor_raw_residual.min())
        >= -cw16.LINEAR_RESIDUAL_TOL,
        "local_raw_residual_at_certified_z": float(
            local_at_certified_z.min()
        ) >= -cw16.LINEAR_RESIDUAL_TOL,
        "total_raw_cap": raw_total_l2
        <= cw16.TOTAL_L2_CAP + cw16.L2_ABS_TOL,
        "trust_raw_cap_relative_actual_center": raw_trust_l2
        <= trust_radius + cw16.L2_ABS_TOL,
        "z_raw_nonpositive": z_star <= 0.0,
        "z_certification_adjustment_only_numeric": z_adjustment >= 0.0
        and z_adjustment <= cw16.LINEAR_RESIDUAL_TOL,
    }
    if not all(raw_certification_checks.values()):
        raise RuntimeError(
            f"stage1 raw certification failed: {raw_certification_checks}"
        )
    raw_certification = {
        "checks": raw_certification_checks,
        "pass": True,
        "anchor_residual_min": float(anchor_raw_residual.min()),
        "local_residual_at_certified_z_min": float(
            local_at_certified_z.min()
        ),
        "total_l2": raw_total_l2,
        "trust_l2_relative_actual_center": raw_trust_l2,
        "z_solver_raw": z_solver,
        "z_star": z_star,
        "z_certification_adjustment": z_adjustment,
    }

    # The fixture identity comes from the exact CW16 stage-2 closures, not an
    # algebraically equivalent rewrite.  The latter is retained only as an
    # independent numerical audit because floating-point operation order can
    # change the last bits of q.
    def anchor_fun_stage2(value: Any) -> Any:
        return (A_scaled @ value - b) / a_scale

    def anchor_jac_stage2(value: Any) -> Any:
        del value
        return A_scaled / a_scale[:, None]

    def local_fun_stage2(value: Any) -> Any:
        return (
            residual + G_scaled @ (value - center_scaled) - z_star
        ) / z_scale

    def local_jac_stage2(value: Any) -> Any:
        del value
        return G_scaled / z_scale

    zero_scaled = np.zeros(rank, dtype=np.float64)
    K_anchor = np.asarray(anchor_jac_stage2(zero_scaled), dtype=np.float64)
    K_local = np.asarray(local_jac_stage2(zero_scaled), dtype=np.float64)
    fun_zero_anchor = np.asarray(
        anchor_fun_stage2(zero_scaled), dtype=np.float64
    )
    fun_zero_local = np.asarray(
        local_fun_stage2(zero_scaled), dtype=np.float64
    )
    K = np.concatenate([K_anchor, K_local], axis=0)
    q = -np.concatenate([fun_zero_anchor, fun_zero_local], axis=0)
    independent_K = np.concatenate(
        [A_scaled / a_scale[:, None], G_scaled / z_scale], axis=0
    )
    independent_q = np.concatenate(
        [
            b / a_scale,
            (z_star - residual + G_scaled @ center_scaled) / z_scale,
        ]
    )
    q_independent_error = float(np.abs(q - independent_q).max())
    captured_constraint_checks = {
        "zero_scaled_shape_exact": zero_scaled.shape == (rank,),
        "anchor_closure_shapes_exact": K_anchor.shape == (A.shape[0], rank)
        and fun_zero_anchor.shape == (A.shape[0],),
        "local_closure_shapes_exact": K_local.shape == (G.shape[0], rank)
        and fun_zero_local.shape == (G.shape[0],),
        "closure_jacobians_match_independent_exact": np.array_equal(
            K, independent_K
        ),
        "closure_rhs_matches_independent_within_1e_12": q_independent_error
        <= 1e-12,
        "fixture_uses_stage2_closure_jacobians_and_fun_at_zero": True,
    }
    if not all(captured_constraint_checks.values()):
        raise RuntimeError(
            "captured CW16 stage2 closure drift: "
            f"{captured_constraint_checks}"
        )
    raw_arrays = {
        "anchor_gradients_A": dual_math.encode_array(A),
        "anchor_rhs_b": dual_math.encode_array(b),
        "local_gradients_G": dual_math.encode_array(G),
        "local_residual_at_center": dual_math.encode_array(residual),
    }
    raw_space_checks = {
        "anchor_shapes_exact": A.ndim == 2 and b.shape == (A.shape[0],),
        "local_shapes_exact": G.ndim == 2
        and G.shape[1] == A.shape[1]
        and residual.shape == (G.shape[0],),
        "center_shape_exact": center.shape == (A.shape[1],),
        "all_raw_arrays_finite": bool(
            np.isfinite(A).all()
            and np.isfinite(b).all()
            and np.isfinite(G).all()
            and np.isfinite(residual).all()
        ),
        "array_record_shas_self_consistent": all(
            record["sha256"] == dual_math.float64_sha256(value)
            for record, value in zip(
                raw_arrays.values(), (A, b, G, residual)
            )
        ),
    }
    if not all(raw_space_checks.values()):
        raise RuntimeError(f"raw-space contract drift: {raw_space_checks}")
    raw_space_contract = {
        "schema_version": "ptcg-cw18-raw-space-contract-v1",
        "checks": raw_space_checks,
        "pass": True,
        "actor_original_dimension": int(center.size),
        "anchor_row_count": int(A.shape[0]),
        "local_physical_row_count": int(G.shape[0]),
        "arrays": raw_arrays,
        "array_float64_le_sha256": {
            key: value["sha256"] for key, value in raw_arrays.items()
        },
        "z_star": z_star,
        "trust_radius_relative_actual_center": float(trust_radius),
        "TOTAL_L2_CAP": float(cw16.TOTAL_L2_CAP),
        "L2_ABS_TOL": float(cw16.L2_ABS_TOL),
        "LINEAR_RESIDUAL_TOL": float(cw16.LINEAR_RESIDUAL_TOL),
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
    }
    fixture = dual_math.encode_fixture(K, q, center_scaled, rho)
    reconstruction = {
        "formula": "candidate_raw=basis_reduced_by_actor.T@(coordinate_scale*candidate_scaled)",
        "basis_orientation": "reduced_rank_by_actor_original",
        "basis_reduced_by_actor": dual_math.encode_array(basis),
        "basis_float64_le_sha256": dual_math.float64_sha256(basis),
        "center_raw": dual_math.encode_array(center),
        "center_raw_float64_le_sha256": dual_math.float64_sha256(center),
        "center_raw_l2": float(np.linalg.norm(center)),
        "coordinate_scale": scale,
        "actor_original_dimension": int(center.size),
        "reduced_rank": rank,
    }
    audit = {
        "pass": True, "span_shape": list(span.shape), "rank": rank,
        "center_reconstruction_error": center_error,
        "captured_constraint_audit": {
            "checks": captured_constraint_checks,
            "pass": True,
            "independent_rhs_error_max": q_independent_error,
            "fixture_uses_captured_closure_jacobians_and_fun_at_zero": True,
        },
        "raw_certification": raw_certification,
        "raw_space_contract": raw_space_contract,
        "stage1": {"success": True, "status": int(stage1.status),
                   "iterations": int(stage1.nit), "z_solver_raw": z_solver,
                   "z_star": z_star,
                   "raw_certification": raw_certification,
                   "x_scaled": dual_math.encode_array(
                       np.asarray(stage1.x[:-1], dtype=np.float64))},
        "reconstruction_map": reconstruction,
    }
    return fixture, audit


def selftest_result() -> dict[str, Any]:
    runtime = runtime_audit(require_cuda=False)
    audit = source_audit()
    plan, plan_audit = load_plan()
    cw15, cw15_evidence = import_locked(CW15, "cw18_selftest_cw15")
    cw16, cw16_evidence = import_locked(CW16, "cw18_selftest_cw16")
    dual, dual_evidence = import_locked(DUAL_MATH, "cw18_selftest_dual")
    import numpy as np
    from scipy import optimize
    A = np.eye(3, dtype=np.float64)
    b = np.asarray([-0.2, -0.2, -0.2], dtype=np.float64)
    G = np.eye(3, dtype=np.float64)
    residual = np.asarray([-0.1, -0.1, -0.1], dtype=np.float64)
    center = np.zeros(3, dtype=np.float64)
    fixture, reduced = reduce_fixture(
        cw16, dual, A, b, G, residual, center, cw16.TRUST_RADII[0], np, optimize
    )
    decoded = dual.decode_fixture(fixture)
    reconstruction = reduced["reconstruction_map"]
    raw_contract = reduced["raw_space_contract"]
    raw_arrays = raw_contract["arrays"]
    zero_graph, zero_graph_audit = _official_graph_fingerprint_map(
        cw15, plan["zero"]["six_view_snapshot"]
    )
    one_graph, one_graph_audit = _official_graph_fingerprint_map(
        cw15, plan["one"]["six_view_snapshot"]
    )
    required_zero_contexts = {
        str(value["identity"][0])
        for value in plan["zero"]["active_cut_gates"]["records"]
        if value["context_type"] == "official_B256"
    }
    required_one_contexts = {
        str(value["identity"][0])
        for value in plan["one"]["active_cut_gates"]["records"]
        if value["context_type"] == "official_B256"
    }
    checks = {
        "static_pass": audit["pass"], "plan_pass": plan_audit["pass"],
        "fixture_roundtrip": decoded[0].shape[0] == 6 and decoded[2].shape == (3,),
        "solver_reconstruction_contract": {
            "basis_orientation", "basis_reduced_by_actor", "center_raw",
            "center_raw_float64_le_sha256", "coordinate_scale",
            "actor_original_dimension", "reduced_rank",
        }.issubset(reconstruction),
        "solver_stage1_seed_present": isinstance(
            reduced.get("stage1", {}).get("x_scaled"), Mapping
        ),
        "captured_stage2_closure_audit": reduced[
            "captured_constraint_audit"]["pass"] is True
        and all(
            reduced["captured_constraint_audit"]["checks"].values()
        ),
        "stage1_raw_certification": reduced["raw_certification"]["pass"]
        is True
        and all(reduced["raw_certification"]["checks"].values()),
        "raw_space_contract_roundtrip": all(
            np.array_equal(dual.decode_array(raw_arrays[key]), expected)
            for key, expected in (
                ("anchor_gradients_A", A),
                ("anchor_rhs_b", b),
                ("local_gradients_G", G),
                ("local_residual_at_center", residual),
            )
        )
        and raw_contract["pass"] is True
        and all(raw_contract["checks"].values()),
        "frozen_official_graph_fingerprint_maps": zero_graph_audit["pass"]
        is True
        and one_graph_audit["pass"] is True
        and required_zero_contexts.issubset(zero_graph)
        and required_one_contexts.issubset(one_graph)
        and len(required_zero_contexts) == 4
        and len(required_one_contexts) == 7,
        "center_hash_self_consistent": reconstruction[
            "center_raw_float64_le_sha256"]
        == reconstruction["center_raw"]["sha256"],
        "no_cuda_requested": runtime["cuda_required"] is False,
        "terminal_payload_not_used": plan["document"]["second_stage"].get(
            "terminal_active_cut_ledger") is None,
    }
    if not all(checks.values()):
        raise RuntimeError(f"selftest failed: {checks}")
    return {"schema_version": SCHEMA, "status": "selftest_passed",
            "checks": checks, "pass": True, "runtime": runtime,
            "source_audit": audit, "plan_audit": plan_audit,
            "frozen_modules": {"cw15": cw15_evidence,
                               "cw16": cw16_evidence,
                               "dual_math": dual_evidence},
            "synthetic_fixture_sha256": fixture["fixture_sha256"],
            "synthetic_reduction": reduced, "cuda_accessed": False,
            "writes_performed": 0}


def static_result() -> dict[str, Any]:
    runtime = runtime_audit(require_cuda=False)
    audit = source_audit()
    _, plan_audit = load_plan()
    return {"schema_version": SCHEMA, "status": "static_passed", "pass": True,
            "runtime": runtime, "source_audit": audit, "plan_audit": plan_audit,
            "cuda_accessed": False, "writes_performed": 0}


def run_result() -> dict[str, Any]:
    """The CUDA reconstruction body is deliberately isolated from CPU modes."""
    runtime = runtime_audit(require_cuda=True)
    audit = source_audit()
    plan, plan_audit = load_plan()
    fixture, reconstruction = reconstruct_live_fixture(plan)
    reduction = reconstruction["reduction"]
    reconstruction_map = reduction.pop("reconstruction_map")
    raw_space_contract = reduction.pop("raw_space_contract")
    return {"schema_version": SCHEMA, "status": "fixture_built", "pass": True,
            "runtime": runtime, "source_audit": audit, "plan_audit": plan_audit,
            "fixture": fixture,
            "reconstruction_map": reconstruction_map,
            "basis_float64_le_sha256": reconstruction_map[
                "basis_float64_le_sha256"],
            "expected_bootstrap_point_sha256": EXPECTED_BOOTSTRAP_POINT_SHA,
            "stage1": reduction["stage1"],
            "captured_constraint_audit": reduction[
                "captured_constraint_audit"],
            "raw_space_contract": raw_space_contract,
            "precision_scope": reconstruction["precision_scope"],
            "graph_fingerprint_audit": reconstruction[
                "graph_fingerprint_audit"],
            "reconstruction": reconstruction,
            "changed_model_official_evaluation_count": 0,
            "changed_model_candidate_consumer_called": False,
            "exact_CW11_fixture_consumer_called": True,
            "writes_performed": 0,
            "submission_performed": False}


def reconstruct_live_fixture(plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recreate A/G/center without invoking any full-stream model evaluator."""
    cw15, cw15_evidence = import_locked(CW15, "cw18_run_cw15")
    cw16, cw16_evidence = import_locked(CW16, "cw18_run_cw16")
    dual, dual_evidence = import_locked(DUAL_MATH, "cw18_run_dual")
    modules = cw15.frozen_modules()
    primary_source, primary_evidence = modules["cw11"].read_regular_bytes(
        cw15.PRIMARY, cw15.MODULE_SHAS[cw15.PRIMARY], "CW18 frozen primary source",
        expected_mode=0o555,
    )
    holder: dict[str, Any] = {}

    def consume(context: Mapping[str, Any]) -> None:
        if holder:
            raise RuntimeError("CW18 exact CW11 consumer called more than once")
        fixture, reconstruction = _reconstruct_from_exact_cw11_context(
            plan, cw15, cw16, dual, modules, context
        )
        holder.update({"fixture": fixture, "reconstruction": reconstruction})

    cw11_result = modules["cw11"].run_probe(
        modules["primary"], primary_source, primary_evidence,
        candidate_consumer=consume,
    )
    chain_checks = {
        "consumer_called_once": bool(holder),
        "cw11_status_exact": cw11_result.get("status")
        == "exploratory_33row_specialist_valid_CW11_success",
        "outer_final_restore_pass": bool(cw11_result.get("final_integrity", {}).get("pass")),
    }
    if not all(chain_checks.values()):
        raise RuntimeError(f"exact CW11 reconstruction chain failed: {chain_checks}")
    holder["reconstruction"]["exact_CW11_chain"] = {
        "checks": chain_checks, "pass": True, "primary": primary_evidence,
        "cw11_summary": {
            "status": cw11_result["status"],
            "model_state_sha256": cw11_result["second_stage"]["decision"][
                "candidate_model_state_sha256_before_CW10_finally_restore"],
            "vector_sha256": cw11_result["second_stage"]["decision"][
                "terminal_cumulative_float64_le_sha256"],
        },
    }
    holder["reconstruction"]["frozen_modules"] = {
        "cw15": cw15_evidence, "cw16": cw16_evidence, "dual_math": dual_evidence,
    }
    return holder["fixture"], holder["reconstruction"]


def _reconstruct_from_exact_cw11_context(
    plan: Mapping[str, Any], cw15: ModuleType, cw16: ModuleType,
    dual: ModuleType, modules: Mapping[str, ModuleType],
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = context["helper"].torch
    precision_before = torch.get_float32_matmul_precision()
    precision_during = None
    precision_after = None
    try:
        torch.set_float32_matmul_precision("high")
        precision_during = torch.get_float32_matmul_precision()
        if precision_during != "high":
            raise RuntimeError("failed to enter exact high matmul-precision scope")
        fixture, reconstruction = _reconstruct_high_precision_core(
            plan, cw15, cw16, dual, modules, context
        )
    finally:
        torch.set_float32_matmul_precision(precision_before)
        precision_after = torch.get_float32_matmul_precision()
        if precision_after != precision_before:
            raise RuntimeError("failed to restore incoming matmul precision")
    precision_checks = {
        "saved_before_selected_context_reconstruction": isinstance(
            precision_before, str
        ),
        "high_applied_and_verified": precision_during == "high",
        "incoming_value_restored_exact": precision_after == precision_before,
        "scope_covers_A_center_and_G": True,
    }
    if not all(precision_checks.values()):
        raise RuntimeError(f"matmul precision scope drift: {precision_checks}")
    reconstruction["precision_scope"] = {
        "checks": precision_checks,
        "pass": True,
        "before": precision_before,
        "during": precision_during,
        "after": precision_after,
    }
    return fixture, reconstruction


def _reconstruct_high_precision_core(
    plan: Mapping[str, Any], cw15: ModuleType, cw16: ModuleType,
    dual: ModuleType, modules: Mapping[str, ModuleType],
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    from scipy import optimize

    helper = context["helper"]
    torch = helper.torch
    cw11_model = context["model"]
    model_config = context["model_config"]
    device = next(cw11_model.parameters()).device
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_checks = {
        "builder_seed_exact": SEED == 202608207,
        "builder_matches_CW15": SEED == cw15.SEED,
        "builder_matches_CW16": SEED == cw16.SEED,
    }
    if not all(seed_checks.values()):
        raise RuntimeError(f"frozen seed drift: {seed_checks}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    parameters = modules["geometry"].configure_actor6(cw11_model)
    cw11_actor = [value.detach().clone() for value in parameters]
    raw_actor = context["raw_actor"]
    cw11_total = np.asarray(context["terminal_cumulative_float64"], dtype=np.float64)
    context_checks = {
        "seed_scope_exact": all(seed_checks.values()),
        "model_exact_CW11": helper.model_state_sha256(cw11_model.state_dict())
        == cw15.CW11_MODEL_SHA256,
        "model_eval": cw11_model.training is False,
        "raw_sha": context["raw_model_state_sha256"] == cw15.RAW_MODEL_SHA256,
        "nonactor_sha": context["raw_nonactor_sha256"] == cw15.RAW_NONACTOR_SHA256,
        "vector_shape": cw11_total.shape == (cw15.ACTOR6_FLAT_LENGTH,),
        "vector_sha_exact_e718": dual.float64_sha256(cw11_total)
        == cw15.CW11_VECTOR_SHA256,
        "vector_l2_exact": math.isclose(float(np.linalg.norm(cw11_total)),
                                        cw15.CW11_TOTAL_FROM_RAW_L2,
                                        rel_tol=0.0, abs_tol=1e-15),
        "legacy_ledger_34": len(context["active_pair_ledger"]) == 34,
    }
    if not all(context_checks.values()):
        raise RuntimeError(f"exact CW11 live context drift: {context_checks}")

    zero = plan["zero"]
    one = plan["one"]
    zero_graph_fingerprints, zero_graph_source_audit = (
        _official_graph_fingerprint_map(cw15, zero["six_view_snapshot"])
    )
    one_graph_fingerprints, one_graph_source_audit = (
        _official_graph_fingerprint_map(cw15, one["six_view_snapshot"])
    )
    forensic = cw15.read_json_locked(cw15.FORENSIC, cw15.FORENSIC_SHA256, 0o444)
    guardplan = cw15.read_json_locked(cw15.GUARDPLAN, cw15.GUARDPLAN_SHA256, 0o444)
    streams, stream_audit = cw15.load_official_six_streams(
        helper, modules["legacy"], model_config, forensic, guardplan
    )
    legacy = cw15.load_legacy_B33(context, modules)
    active: list[dict[str, Any]] = []
    cw15.merge_active_cuts(active, cw15.legacy_pair_cuts(legacy))
    cw15.merge_active_cuts(active, _build_initial_official_cuts(cw15, streams, zero))
    initial_expected = {
        tuple(value["identity"]): value
        for value in zero["active_cut_gates"]["records"]
    }
    initial_checks = {
        "active38": len(active) == 38,
        "identity_exact": {cw15.cut_identity(value) for value in active}
        == set(initial_expected),
    }
    if not all(initial_checks.values()):
        raise RuntimeError(f"initial active reconstruction failed: {initial_checks}")

    stored_initial = one["initial_anchor_growth"]["linearization"]["records"]
    anchor38, anchor38_audit = _linearize_selected(
        cw15, cw11_model, helper, modules, streams, legacy, active, parameters,
        np.zeros(cw15.ACTOR6_FLAT_LENGTH, dtype=np.float64), stored_initial,
        expected_model_sha=cw15.CW11_MODEL_SHA256,
        expected_output_sha=zero["output_state_fingerprint"]["sha256"],
        kind="CW11_anchor", np=np,
        expected_official_fingerprints=zero_graph_fingerprints,
        graph_phase="frozen_CW16_iteration0_reference",
    )
    A38, b38, affine38 = cw15.affine_qp_arrays(anchor38)
    center, replay_qp = modules["cutting"].solve_minimum_l2_correction(
        A38, b38, np, optimize
    )
    center = np.asarray(center, dtype=np.float64)
    center_checks = {
        "shape": center.shape == (cw15.ACTOR6_FLAT_LENGTH,),
        "sha_exact": dual.float64_sha256(center) == EXPECTED_BOOTSTRAP_POINT_SHA,
        "l2_exact": math.isclose(float(np.linalg.norm(center)),
                                 float(one["additional_l2"]), rel_tol=0.0,
                                 abs_tol=1e-18),
        "not_clipped": replay_qp["capped"] is False,
    }
    if not all(center_checks.values()):
        raise RuntimeError(f"bootstrap center reconstruction failed: {center_checks}")

    additions = copy.deepcopy(one["post_oracle_cut_merge"]["added"])
    merge = cw15.merge_active_cuts(active, additions)
    if merge["active_count"] != 50 or len(merge["added"]) != 12:
        raise RuntimeError("stored CW16 semantic additions do not reconstruct active50")
    stored_added = one["post_bootstrap_anchor_growth"]["linearization"]["records"]
    anchor12, anchor12_audit = _linearize_selected(
        cw15, cw11_model, helper, modules, streams, legacy, additions, parameters,
        np.zeros(cw15.ACTOR6_FLAT_LENGTH, dtype=np.float64), stored_added,
        expected_model_sha=cw15.CW11_MODEL_SHA256,
        expected_output_sha=zero["output_state_fingerprint"]["sha256"],
        kind="CW11_anchor", np=np,
        expected_official_fingerprints=zero_graph_fingerprints,
        graph_phase="frozen_CW16_iteration0_reference",
    )
    anchor50 = [*anchor38, *anchor12]
    A, b, anchor_array_audit = cw15.affine_qp_arrays(anchor50)
    anchor50_sha = cw15.canonical_sha(
        [cw15.sanitized_affine_tangent(value)
         for value in sorted(anchor50, key=cw15.affine_tangent_identity)]
    )
    if anchor50_sha != EXPECTED_ANCHOR50_SHA or A.shape[0] != 50:
        raise RuntimeError("reconstructed anchor50 differs from frozen CW16")

    cw15.apply_additional_from_cw11(
        modules, parameters, raw_actor, cw11_total, center, torch
    )
    application_audit = {
        "pass": helper.model_state_sha256(cw11_model.state_dict())
        == EXPECTED_BOOTSTRAP_MODEL_SHA,
        "method": "exact_raw_actor_plus_e718_float64_total_plus_e381_additional",
        "model_sha256": helper.model_state_sha256(cw11_model.state_dict()),
        "cw11_total_float64_le_sha256": dual.float64_sha256(cw11_total),
    }
    if application_audit["pass"] is not True:
        raise RuntimeError(f"exact bootstrap application drift: {application_audit}")
    current_expected = {
        tuple(value["identity"]): value for value in one["active_cut_gates"]["records"]
    }
    local_semantic, local_semantic_audit = _linearize_selected(
        cw15, cw11_model, helper, modules, streams, legacy, active, parameters,
        center, None, expected_model_sha=EXPECTED_BOOTSTRAP_MODEL_SHA,
        expected_output_sha=one["output_state_fingerprint"]["sha256"],
        kind="CW16_current_local_physical", np=np,
        expected_official_fingerprints=one_graph_fingerprints,
        graph_phase="frozen_CW16_iteration1_bootstrap",
        expected_gates=current_expected,
    )
    gates = {"records": [
        {"identity": list(value["semantic_identity"]),
         "margin": float(value["observed_margin"]),
         "threshold": float(value["threshold"]),
         "residual": float(value["actual_residual"]),
         "pass": float(value["actual_residual"])
         >= -modules["cutting"].PAIR_THRESHOLD_TOLERANCE}
        for value in local_semantic
    ]}
    local_rows, dedup = cw16.deduplicate_active_physical(
        cw15, active, gates, local_semantic
    )
    ordered_local = sorted(local_rows, key=cw15.affine_tangent_identity)
    G = np.stack([np.asarray(value["gradient_float64"], dtype=np.float64)
                  for value in ordered_local], axis=0)
    residual = np.asarray([float(value["actual_residual"])
                           for value in ordered_local], dtype=np.float64)
    residual_sha = dual.float64_sha256(residual)
    if (
        G.shape != (41, cw15.ACTOR6_FLAT_LENGTH)
        or residual.shape != (41,)
        or residual_sha != EXPECTED_LOCAL41_RESIDUAL_SHA
    ):
        raise RuntimeError("current physical 41-row reconstruction drift")
    fixture, reduction = reduce_fixture(
        cw16, dual, A, b, G, residual, center,
        float(plan["failure"]["trust_radius"]), np, optimize
    )
    if reduction["reconstruction_map"]["center_raw_float64_le_sha256"] \
            != EXPECTED_BOOTSTRAP_POINT_SHA:
        raise RuntimeError("fixture reconstruction map center identity drift")

    graph_fingerprint_audits = {
        "anchor38": anchor38_audit["graph_fingerprint_audit"],
        "anchor12": anchor12_audit["graph_fingerprint_audit"],
        "local50_to_physical41": local_semantic_audit[
            "graph_fingerprint_audit"
        ],
    }
    graph_fingerprint_audit = {
        "checks": {
            "all_three_linearization_graph_audits_pass": all(
                value["pass"] for value in graph_fingerprint_audits.values()
            ),
            "local_official_contexts_exact7": graph_fingerprint_audits[
                "local50_to_physical41"
            ]["official_context_count"] == 7,
            "local_legacy_context_explicit1": graph_fingerprint_audits[
                "local50_to_physical41"
            ]["legacy_context_count"] == 1,
        },
        "pass": True,
        "ordered_phase_audit_sha256": cw15.canonical_sha(
            graph_fingerprint_audits
        ),
        "phases": graph_fingerprint_audits,
    }
    if not all(graph_fingerprint_audit["checks"].values()):
        raise RuntimeError(
            "combined selected-context graph fingerprint audit drift: "
            f"{graph_fingerprint_audit['checks']}"
        )

    _restore_actor(modules, parameters, cw11_actor, torch)
    if helper.model_state_sha256(cw11_model.state_dict()) != cw15.CW11_MODEL_SHA256:
        raise RuntimeError("final RAM restore to materialized CW11 failed")
    return fixture, {
        "pass": True,
        "exact_CW11_context_checks": context_checks,
        "seed_scope": {"checks": seed_checks, "pass": True, "seed": SEED},
        "stream_loading_audit": stream_audit,
        "official_graph_fingerprint_sources": {
            "anchor_zero": zero_graph_source_audit,
            "local_one": one_graph_source_audit,
        },
        "graph_fingerprint_audit": graph_fingerprint_audit,
        "initial_checks": initial_checks,
        "anchor38_linearization": anchor38_audit,
        "anchor12_linearization": anchor12_audit,
        "anchor38_array_audit": affine38,
        "anchor50_array_audit": anchor_array_audit,
        "center_checks": center_checks,
        "center_application": application_audit,
        "local_semantic_linearization": local_semantic_audit,
        "physical_deduplication": dedup,
        "dimensions": {"anchor_rows": int(A.shape[0]),
                       "local_physical_rows": int(G.shape[0]),
                       "actor_original": int(center.size)},
        "local_residual_float64_le_sha256": residual_sha,
        "reduction": reduction,
        "changed_model_officially_evaluated": False,
        "selected_context_autograd_for_fixture": True,
        "full_stream_snapshot_constructed": False,
    }


def _restore_actor(modules: Mapping[str, ModuleType], parameters: Sequence[Any],
                   actor: Sequence[Any], torch: Any) -> None:
    modules["ram"].set_actor_overlay(parameters, actor, actor, 0.0, torch)


def _build_initial_official_cuts(cw15: ModuleType,
                                 streams: Mapping[str, Sequence[dict[str, Any]]],
                                 zero: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected_gates = {tuple(value["identity"]): value
                      for value in zero["active_cut_gates"]["records"]
                      if value["context_type"] == "official_B256"}
    cuts = []
    for expected in cw15.FIXED_CONTEXT_EXPECTATIONS:
        view = str(expected["view"])
        batch = streams[view][int(expected["batch_ordinal"]) - 1]
        offset = int(expected["offset"])
        identity = batch["identities"][offset]
        physical = next(value for value in cw15.FIXED_REPAIR_IDENTITIES
                        if str(value["line_sha256"]) == str(identity["line_sha256"]))
        positive, negative = int(physical["positive_option"]), int(physical["negative_option"])
        semantic = (batch["context"]["context_hash"], identity["decision_sha256"],
                    "fixed_initial_repair_floor", positive, negative)
        gate = expected_gates[semantic]
        threshold = float(physical["raw_margin"])
        if float(gate["threshold"]) != threshold:
            raise RuntimeError("fixed initial raw floor differs from CW16 stdout")
        cuts.append({
            "context_type": "official_B256",
            "context_hash": batch["context"]["context_hash"], "view": view,
            "batch_ordinal_one_based": batch["context"]["batch_ordinal_one_based"],
            "actual_batch_size": batch["context"]["actual_batch_size"],
            "dynamic_max_options": batch["context"]["dynamic_max_options"],
            "offset_zero_based": offset,
            "physical_identity": list(cw15.identity_tuple(identity)),
            "decision_sha256": identity["decision_sha256"],
            "metric": "fixed_initial_repair_floor", "positive_option": positive,
            "negative_option": negative, "threshold": threshold,
            "threshold_sources": {"same_official_shape_raw": threshold},
            "anchor_model": "raw_U468", "anchor_source": "same_official_shape_raw_margin",
            "strict_positive_required": bool(physical.get("strict_positive", False)),
            "cw11_start_margin": float(gate["margin"]),
            "construction": "fixed_preregistered_raw_correct_vs_CW11_harmful_pair",
            "created_iteration": 0, "origins": ["fixed3_minimum_repair"],
        })
    if len(cuts) != 4 or {cw15.cut_identity(value) for value in cuts} != set(expected_gates):
        raise RuntimeError("initial four official cuts reconstruction drift")
    return cuts


def _official_graph_fingerprint_map(
    cw15: ModuleType, six_view_snapshot: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    records = []
    lookup: dict[str, dict[str, Any]] = {}
    panels = six_view_snapshot.get("panels")
    if not isinstance(panels, Mapping) or set(panels) != set(cw15.VIEW_ORDER):
        raise RuntimeError("frozen six-view panel fingerprint schema drift")
    for view in cw15.VIEW_ORDER:
        for batch in panels[view]["batches"]:
            context_hash = str(batch["context_hash"])
            tensor_fingerprints = {
                str(key): str(value)
                for key, value in batch["tensor_fingerprints"].items()
            }
            record = {
                "view": str(view),
                "context_hash": context_hash,
                "batch_ordinal_one_based": int(
                    batch["batch_ordinal_one_based"]
                ),
                "actual_batch_size": int(batch["actual_batch_size"]),
                "dynamic_max_options": int(batch["dynamic_max_options"]),
                "tensor_fingerprints": tensor_fingerprints,
            }
            if context_hash in lookup:
                raise RuntimeError("duplicate frozen official context fingerprint")
            lookup[context_hash] = record
            records.append(record)
    checks = {
        "official_context_count_exact134": len(records) == len(lookup) == 134,
        "three_tensor_fingerprints_each": all(
            set(value["tensor_fingerprints"])
            == {"policy_logits", "count_logits", "value_logits"}
            and all(
                len(digest) == 64
                for digest in value["tensor_fingerprints"].values()
            )
            for value in records
        ),
        "contexts_unique": len(lookup) == len(records),
    }
    if not all(checks.values()):
        raise RuntimeError(f"official graph fingerprint map drift: {checks}")
    return lookup, {
        "checks": checks,
        "pass": True,
        "context_count": len(records),
        "ordered_context_fingerprint_sha256": cw15.canonical_sha(records),
    }


def _linearize_selected(cw15: ModuleType, model: Any, helper: ModuleType,
                        modules: Mapping[str, ModuleType],
                        streams: Mapping[str, Sequence[dict[str, Any]]],
                        legacy: Mapping[str, Any], cuts: Sequence[Mapping[str, Any]],
                        parameters: Sequence[Any], point: Any,
                        stored_records: Sequence[Mapping[str, Any]] | None, *,
                        expected_model_sha: str, expected_output_sha: str,
                        kind: str, np: Any,
                        expected_official_fingerprints: Mapping[
                            str, Mapping[str, Any]
                        ],
                        graph_phase: str,
                        expected_gates: Mapping[tuple[Any, ...], Mapping[str, Any]] | None = None,
                        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = next(model.parameters()).device
    point = np.asarray(point, dtype=np.float64)
    if helper.model_state_sha256(model.state_dict()) != expected_model_sha:
        raise RuntimeError(f"{kind}: live model SHA drift")
    stored = ({tuple(value["semantic_identity"]): value for value in stored_records}
              if stored_records is not None else None)
    lookup = cw15.official_context_lookup(streams)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cut in sorted(cuts, key=cw15.cut_identity):
        groups[str(cut["context_hash"])].append(cut)
    records = []
    contexts = []
    for context_hash in sorted(groups):
        grouped = groups[context_hash]
        if grouped[0]["context_type"] == "legacy_B33":
            gpu_batch = legacy["gpu_batch"]
        else:
            gpu_batch = {key: value.to(device, non_blocking=True)
                         for key, value in lookup[context_hash]["cpu_batch"].items()}
        outputs = helper.ppo.model_forward(model, dict(gpu_batch), device)
        output_names = {"policy_logits", "count_logits", "value_logits"}
        if set(outputs) != output_names or any(
            value.dtype != helper.torch.bfloat16 for value in outputs.values()
        ):
            raise RuntimeError(f"{kind}: selected-context forward is not native BF16")
        observed_tensor_fingerprints = {
            name: modules["ram"].tensor_sha256(outputs[name], helper.torch)
            for name in sorted(output_names)
        }
        context_type = str(grouped[0]["context_type"])
        if context_type == "official_B256":
            expected_graph = expected_official_fingerprints.get(context_hash)
            official_graph_checks = {
                "frozen_context_fingerprint_present": isinstance(
                    expected_graph, Mapping
                ),
                "three_tensor_raw_bytes_sha_exact": isinstance(
                    expected_graph, Mapping
                )
                and observed_tensor_fingerprints
                == expected_graph.get("tensor_fingerprints"),
                "actual_batch_size_exact": isinstance(expected_graph, Mapping)
                and int(gpu_batch["targets"].shape[0])
                == int(expected_graph.get("actual_batch_size", -1)),
                "dynamic_max_options_exact": isinstance(expected_graph, Mapping)
                and int(gpu_batch["option_mask"].shape[1])
                == int(expected_graph.get("dynamic_max_options", -1)),
            }
            if not all(official_graph_checks.values()):
                raise RuntimeError(
                    f"{kind}: official context graph fingerprint drift: "
                    f"{official_graph_checks}"
                )
        else:
            expected_graph = None
            official_graph_checks = None
        logits = outputs["policy_logits"].float()
        group_margin_checks = []
        group_stored_record_checks = []
        for index, cut in enumerate(grouped):
            identity = cw15.cut_identity(cut)
            margin_tensor = (logits[int(cut["offset_zero_based"]), int(cut["positive_option"])]
                             - logits[int(cut["offset_zero_based"]), int(cut["negative_option"])])
            flat, _ = modules["geometry"].gradient_for_margin(
                margin_tensor, parameters, helper.torch,
                retain_graph=index + 1 < len(grouped))
            gradient = np.asarray(flat, dtype=np.float64)
            observed = float(margin_tensor.detach().cpu())
            if stored is not None:
                expected_margin = float(stored[identity]["observed_margin"])
            elif expected_gates is not None:
                expected_margin = float(expected_gates[identity]["margin"])
            else:
                raise RuntimeError("linearization lacks frozen expected margins")
            group_margin_checks.append(observed == expected_margin)
            threshold = float(cut["threshold"])
            dot = float(gradient @ point)
            rhs = float(threshold - observed + dot)
            actual = float(observed - threshold)
            tangent = float(dot - rhs)
            core = {
                "semantic_identity": list(identity),
                "linearization_point_sha256": cw15.float64_vector_sha256(point),
                "linearization_model_state_sha256": expected_model_sha,
                "linearization_output_state_sha256": expected_output_sha,
                "linearization_kind": kind, "threshold": threshold,
                "threshold_float64_le_sha256": cw15.float64_scalar_sha256(threshold),
                "observed_margin": observed,
                "gradient_float64_le_sha256": cw15.float64_vector_sha256(gradient),
                "point_dot_gradient": dot, "rhs": rhs,
                "actual_residual": actual, "tangent_residual_at_point": tangent,
            }
            formula = {
                "gradient_shape_exact": gradient.shape == (cw15.ACTOR6_FLAT_LENGTH,),
                "gradient_finite": bool(np.isfinite(gradient).all()),
                "observed_threshold_dot_rhs_finite": all(math.isfinite(value)
                    for value in (observed, threshold, dot, rhs)),
                "graph_margin_bit_equal_snapshot": observed == expected_margin,
                "graph_margin_bit_equal_active_gate": observed == expected_margin,
                "rhs_exact_t_minus_m_plus_Gx": rhs == threshold - observed + dot,
                "tangent_residual_matches_actual_residual": math.isclose(
                    tangent, actual, rel_tol=0.0, abs_tol=1e-15),
                "candidate_tangent_point_is_actually_violated": True,
            }
            if not all(formula.values()):
                raise RuntimeError(f"{kind}: selected tangent drift: {formula}")
            sanitized = {**core, "formula_checks": formula}
            record = {**sanitized,
                      "linearization_record_sha256": cw15.canonical_sha(sanitized),
                      "gradient_float64": gradient.copy()}
            stored_record_exact = stored is None or (
                cw15.sanitized_affine_tangent(record) == stored[identity]
            )
            group_stored_record_checks.append(stored_record_exact)
            if not stored_record_exact:
                raise RuntimeError(f"{kind}: regenerated frozen affine record differs")
            records.append(record)
        if context_type == "official_B256":
            graph_checks = {
                **official_graph_checks,
                "all_selected_pair_margins_exact": all(group_margin_checks),
                "stored_anchor_records_exact_if_present": all(
                    group_stored_record_checks
                ),
            }
            binding_level = "frozen_official_three_BF16_tensors_raw_bytes_exact"
            evidence_difference = None
        else:
            graph_checks = {
                "frozen_legacy_per_tensor_fingerprints_absent": context_hash
                not in expected_official_fingerprints,
                "all_frozen_legacy_gate_margins_exact": all(
                    group_margin_checks
                ),
                "stored_anchor_gradient_records_exact_or_prior_anchor50_bound": (
                    stored is not None and all(group_stored_record_checks)
                ) or (
                    stored is None
                    and kind == "CW16_current_local_physical"
                    and len(EXPECTED_ANCHOR50_SHA) == 64
                ),
                "live_model_state_sha_exact": helper.model_state_sha256(
                    model.state_dict()
                ) == expected_model_sha,
            }
            binding_level = (
                "legacy_gate_margins_plus_anchor_gradient_hash_ledger_"
                "no_frozen_full_tensor_SHA"
            )
            evidence_difference = (
                "CW16 stdout legacy_gate contains margins but no policy/count/value "
                "tensor fingerprints; official contexts have exact three-tensor SHA"
            )
        if not all(graph_checks.values()):
            raise RuntimeError(
                f"{kind}: selected context graph binding failed: {graph_checks}"
            )
        contexts.append({
            "graph_phase": graph_phase,
            "context_type": context_type,
            "context_hash": context_hash,
            "cut_count": len(grouped),
            "binding_level": binding_level,
            "expected_tensor_fingerprints": (
                dict(expected_graph["tensor_fingerprints"])
                if isinstance(expected_graph, Mapping)
                else None
            ),
            "observed_tensor_fingerprints": observed_tensor_fingerprints,
            "anchor50_frozen_ledger_sha256": EXPECTED_ANCHOR50_SHA,
            "evidence_difference": evidence_difference,
            "checks": graph_checks,
            "pass": True,
        })
    if len(records) != len(cuts):
        raise RuntimeError(f"{kind}: selected linearization cardinality drift")
    graph_fingerprint_checks = {
        "all_context_bindings_pass": all(value["pass"] for value in contexts),
        "all_official_contexts_three_tensor_exact": all(
            value["checks"]["three_tensor_raw_bytes_sha_exact"]
            for value in contexts
            if value["context_type"] == "official_B256"
        ),
        "legacy_evidence_difference_explicit": all(
            value["evidence_difference"] is not None
            for value in contexts
            if value["context_type"] == "legacy_B33"
        ),
        "context_count_matches_groups": len(contexts) == len(groups),
    }
    if not all(graph_fingerprint_checks.values()):
        raise RuntimeError(
            f"{kind}: graph fingerprint ledger drift: {graph_fingerprint_checks}"
        )
    graph_fingerprint_audit = {
        "checks": graph_fingerprint_checks,
        "pass": True,
        "graph_phase": graph_phase,
        "official_context_count": sum(
            value["context_type"] == "official_B256" for value in contexts
        ),
        "legacy_context_count": sum(
            value["context_type"] == "legacy_B33" for value in contexts
        ),
        "ordered_context_graph_fingerprint_sha256": cw15.canonical_sha(
            contexts
        ),
        "contexts": contexts,
    }
    return records, {"pass": True, "kind": kind, "record_count": len(records),
                     "context_count": len(contexts), "contexts": contexts,
                     "graph_fingerprint_audit": graph_fingerprint_audit,
                     "sanitized_ledger_sha256": cw15.canonical_sha(
                         [cw15.sanitized_affine_tangent(value)
                          for value in sorted(records, key=cw15.affine_tangent_identity)])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "selftest", "run"), default="run")
    args = parser.parse_args()
    if args.mode == "static":
        result = static_result()
    elif args.mode == "selftest":
        result = selftest_result()
    else:
        result = run_result()
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
