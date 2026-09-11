#!/usr/bin/env python3
"""RAM-only guard-nullspace target repair probe for raw U468.

This wrapper SHA-binds the completed v1 tie-constrained probe and changes only
its linear solver.  The 19 exact tied margins belonging to 17 raw-correct rows
receive a zero directional derivative, while all seven tied margins belonging
to seven raw-wrong rows receive the same positive derivative.  The resulting
minimum-norm edit remains restricted to ``actor_residual.2.weight``.

All endpoint construction, canonical 256-row BF16 replay, direct512 gates,
count/value tensor checks, RAM restoration, and no-write behavior are inherited
from the hash-bound v1 implementation.  This exploratory shadow cannot
authorize checkpoint materialization, validation, promotion, or submission.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
V1_TOOL = TOOLS / "probe_u468_raw_residual2_tie_constrained_shadow.py"
V1_TOOL_SHA256 = (
    "37d44c06575c26c82fe6563eb013b15598499aca40868ba3f04db313b35c0f30"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_evidence(
    path: Path, expected_sha256: str | None, label: str
) -> dict[str, Any]:
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise RuntimeError(f"{label} is not a single-link regular file")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "mode": oct(observed.st_mode & 0o777),
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


V1_EVIDENCE = regular_evidence(V1_TOOL, V1_TOOL_SHA256, "frozen v1 probe")
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_SPEC = importlib.util.spec_from_file_location(
    "ptcg_u468_residual2_tie_v1_37d44c06", V1_TOOL
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot construct frozen v1 import spec")
v1: ModuleType = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v1)

torch = v1.torch


SCHEMA = "ptcg-u468-raw-residual2-guardnull-target-shadow-v1"
ALPHAS = (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256)
SOLVER_ABS_RESIDUAL_MAX = 1e-9
GUARD_SLOPE_ABS_MAX = 1e-15


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")


def self_evidence() -> dict[str, Any]:
    return regular_evidence(Path(__file__).resolve(), None, "guard-null probe")


def ast_audit() -> dict[str, Any]:
    source = Path(__file__).resolve().read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__).resolve()))

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [dotted(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden = sorted(
        name
        for name in calls
        if name.endswith((".backward", ".save", ".write_text", ".write_bytes"))
        or name.startswith("torch.optim")
    )
    if forbidden:
        raise RuntimeError(f"wrapper zero-write AST gate failed: {forbidden}")
    counts = {
        "torch.linalg.lstsq": calls.count("torch.linalg.lstsq"),
        "v1.run_actual": calls.count("v1.run_actual"),
        "torch.save": calls.count("torch.save"),
    }
    if counts != {"torch.linalg.lstsq": 1, "v1.run_actual": 1, "torch.save": 0}:
        raise RuntimeError(f"wrapper call-site drift: {counts}")
    inherited = v1.ast_audit()
    if inherited.get("status") != "zero_write_ast_audit_passed":
        raise RuntimeError("frozen v1 AST audit no longer passes")
    return {
        "status": "wrapper_and_frozen_v1_zero_write_ast_audit_passed",
        "wrapper_call_sites": counts,
        "inherited_v1": inherited,
        "optimizer_instances": 0,
        "backward_calls": 0,
        "checkpoint_or_artifact_write_calls": 0,
    }


def solve_guardnull_direction(
    gradients: Sequence[torch.Tensor],
    identities: Sequence[Mapping[str, Any]],
    parameter: torch.nn.Parameter,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(gradients) != len(identities) or not gradients:
        raise RuntimeError("constraint gradient/identity alignment drift")
    matrix = torch.stack([gradient / gradient.norm() for gradient in gradients])
    roles = [str(identity["role"]) for identity in identities]
    if roles.count("guard") != 19 or roles.count("target") != 7:
        raise RuntimeError(f"constraint role-count drift: {roles}")
    rhs = torch.tensor(
        [0.0 if role == "guard" else 1.0 for role in roles],
        dtype=torch.float64,
    )
    solution = torch.linalg.lstsq(matrix, rhs, driver="gelsd")
    raw_direction = solution.solution
    residual = matrix @ raw_direction - rhs
    max_abs_residual = float(residual.abs().max())
    if (
        not bool(torch.isfinite(raw_direction).all())
        or max_abs_residual > SOLVER_ABS_RESIDUAL_MAX
    ):
        raise RuntimeError(
            f"guard-null equality system infeasible/unstable: {max_abs_residual}"
        )
    raw_norm = float(raw_direction.norm())
    if raw_norm <= 0.0:
        raise RuntimeError("guard-null solution has zero norm")
    direction = raw_direction * (v1.REFERENCE_STEP_L2 / raw_norm)
    slopes = matrix @ direction
    guard_slopes = slopes[
        torch.tensor([role == "guard" for role in roles], dtype=torch.bool)
    ]
    target_slopes = slopes[
        torch.tensor([role == "target" for role in roles], dtype=torch.bool)
    ]
    guard_max_abs = float(guard_slopes.abs().max())
    target_min = float(target_slopes.min())
    if guard_max_abs > GUARD_SLOPE_ABS_MAX or target_min <= 0.0:
        raise RuntimeError(
            f"guard-null/target-positive directional gate failed: "
            f"guard={guard_max_abs}, target={target_min}"
        )
    singular_values = torch.linalg.svdvals(matrix)
    positive = singular_values[singular_values > torch.finfo(torch.float64).eps]
    condition = float(positive.max() / positive.min())
    payload = direction.contiguous().view(torch.uint8).numpy().tobytes()
    return direction.reshape(parameter.shape), {
        "method": "cpu_float64_minimum_norm_guard_zero_target_one_lstsq_gelsd",
        "parameter": v1.EDITED_PARAMETER,
        "parameter_elements": int(parameter.numel()),
        "constraint_count": len(gradients),
        "constraints_by_role": {"guard": 19, "target": 7},
        "matrix_rank": int(solution.rank),
        "condition_over_positive_singular_values": condition,
        "singular_values_max": float(singular_values.max()),
        "singular_values_min": float(singular_values.min()),
        "unscaled_solution_l2": raw_norm,
        "max_abs_equality_residual": max_abs_residual,
        "reference_step_l2": v1.REFERENCE_STEP_L2,
        "guard_max_abs_normalized_slope_at_reference_step": guard_max_abs,
        "target_min_normalized_slope_at_reference_step": target_min,
        "target_max_normalized_slope_at_reference_step": float(target_slopes.max()),
        "direction_sha256_float64": hashlib.sha256(payload).hexdigest(),
        "all_guard_first_order_slopes_zero_within_tolerance": True,
        "all_target_first_order_slopes_positive": True,
    }


def compact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": candidate["label"],
        "displacement": candidate["displacement"],
        "target_by_source": candidate["constraint_predictions"]["target_by_source"],
        "target_correct_total": candidate["constraint_predictions"][
            "target_correct_total"
        ],
        "target_correct_hashes": candidate["constraint_predictions"][
            "target_correct_hashes"
        ],
        "guard_wrong_count": candidate["constraint_predictions"][
            "guard_wrong_count"
        ],
        "guard_wrong_hashes": candidate["constraint_predictions"][
            "guard_wrong_hashes"
        ],
        "direct512_transitions": candidate["direct512_transitions"],
        "count_value_exact": candidate["count_value_exact"],
        "direct512_loss_deltas_candidate_minus_raw": candidate[
            "direct512_loss_deltas_candidate_minus_raw"
        ],
        "gate_checks": candidate["gate_checks"],
        "eligible": candidate["eligible"],
    }


def build_context() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    Sequence[Sequence[dict[str, Any]]],
    Mapping[str, torch.Tensor],
    dict[str, Any],
]:
    v1.aggregate.verify_fixed_inputs()
    profile, _ = v1.load_profile()
    rows = v1.select_constraint_rows(profile)
    parent = torch.load(v1.frozen.U468, map_location="cpu", weights_only=False)
    if int(parent.get("update", -1)) != 468:
        raise RuntimeError("raw U468 update drift")
    canonical_batches, constraint_cache = v1.load_canonical_constraint_batches(
        rows, parent["model_config"], profile
    )
    cache_audit, direct_selections, direct_union = v1.aggregate.build_cache_audit()
    cache_summary = {
        "constraint_rows": constraint_cache["rows"],
        "constraint_targets": constraint_cache["targets"],
        "constraint_guards": constraint_cache["guards"],
        "canonical_batch_size": constraint_cache["canonical_batch_size"],
        "canonical_source_rows": constraint_cache["canonical_source_rows"],
        "retained_batch_count": constraint_cache["retained_batch_count"],
        "constraint_identity_sha256": constraint_cache["line_identity_sha256"],
        "direct512_cache_sha256": cache_audit["aggregate512"]["cache_sha256"],
        "direct512_batch_sha256": cache_audit["aggregate512"]["batch_sha256"],
        "validation_member_payloads_opened": False,
    }
    return rows, canonical_batches, direct_selections, direct_union, cache_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("static-audit", "cache-audit", "actual"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    tool = self_evidence()
    static = ast_audit()
    common: dict[str, Any] = {
        "schema_version": SCHEMA,
        "tool": tool,
        "frozen_v1": V1_EVIDENCE,
        "static_audit": static,
        "contract": {
            "base": "raw full U468",
            "edited_parameter": v1.EDITED_PARAMETER,
            "solver_rhs": {"19_guard_ties": 0.0, "7_target_ties": 1.0},
            "reference_step_l2": v1.REFERENCE_STEP_L2,
            "alphas": list(ALPHAS),
            "endpoint_construction": "raw_fp32_plus_alpha_times_one_fixed_direction",
            "selection": "smallest endpoint passing every inherited shadow gate",
            "partial_exploratory_shadow_only": True,
            "train_only": True,
            "validation": False,
            "optimizer": False,
            "backward": False,
            "checkpoint_write": False,
            "model_artifact_write": False,
            "result_artifact_write": False,
            "submission": False,
            "stdout_only": True,
        },
    }
    if args.mode == "static-audit":
        if args.device != "cpu":
            raise ValueError("static-audit requires CPU")
        print(
            json.dumps(
                {**common, "status": "zero_write_static_audit_passed"},
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    rows, canonical_batches, direct_selections, direct_union, cache_summary = (
        build_context()
    )
    common["cache"] = cache_summary
    if args.mode == "cache-audit":
        if args.device != "cpu":
            raise ValueError("cache-audit requires CPU")
        print(
            json.dumps(
                {**common, "status": "zero_write_cache_audit_passed"},
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual requires CUDA")

    original_solver = v1.solve_direction
    original_alphas = v1.ALPHAS
    v1.solve_direction = solve_guardnull_direction
    v1.ALPHAS = ALPHAS
    try:
        result = v1.run_actual(
            rows,
            canonical_batches,
            direct_selections,
            direct_union,
            torch.device("cuda"),
        )
    finally:
        v1.solve_direction = original_solver
        v1.ALPHAS = original_alphas
    post = {
        "tool_sha256_unchanged": sha256_file(Path(__file__).resolve()) == tool["sha256"],
        "frozen_v1_sha256_unchanged": sha256_file(V1_TOOL) == V1_TOOL_SHA256,
        "profile_sha256_unchanged": sha256_file(v1.PROFILE) == v1.PROFILE_SHA256,
        "raw_u468_sha256_unchanged": (
            sha256_file(v1.frozen.U468) == v1.frozen.U468_SHA256
        ),
    }
    if not all(post.values()):
        raise RuntimeError(f"postrun binding drift: {post}")
    compact = {
        **common,
        "status": result["status"],
        "decision": result["decision"],
        "selected_smallest_eligible_alpha": result[
            "selected_smallest_eligible_alpha"
        ],
        "solver": result["solver"],
        "raw_constraint_predictions": result["raw_constraint_predictions"],
        "candidates": [compact_candidate(candidate) for candidate in result["candidates"]],
        "integrity": result["integrity"],
        "postrun_reverification": post,
    }
    print(
        json.dumps(
            compact,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
