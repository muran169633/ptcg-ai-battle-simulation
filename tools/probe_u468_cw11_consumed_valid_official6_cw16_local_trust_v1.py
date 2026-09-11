#!/usr/bin/env python3
"""CW16 local-trust consumed-valid official-B256 cutting-plane probe.

CW16 consumes the immutable CW15 closure and replays its iteration-1 proposal
bit-for-bit as one ineligible bootstrap centre.  Thereafter it keeps the
semantic ledger monotone while separating the optimization model into
permanent CW11-anchor rows and freshly relinearized current-point physical
rows.  Historical candidate tangents are audit-only and never enter a
subproblem.

The probe is stdout-only.  It has no checkpoint/result write path, no network,
upload, submission, broad, or Gold evaluation path.
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
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "probe_u468_cw11_consumed_valid_official6_cw16_local_trust_v1.py"
SCHEMA = "ptcg-cw16-consumed-valid-official-b256-local-trust-v1"
SEED = 202608207
FROZEN_MODE = 0o555

CW15_PARENT = TOOLS / (
    "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py"
)
CW15_PARENT_SHA256 = (
    "2aeb770f8774b7f7407a93553405089511ab2d2f278653162274a3afb89a2f24"
)
CW15_ATTEMPT = ROOT / (
    "artifacts/.ptcg-cw15_consumed_valid_official6_sequential_tangent_"
    "20260802_v1-attempt.json"
)
CW15_ATTEMPT_SHA256 = (
    "e395acf51486cceb68e7f9524b716e1172a644623d097e32053a68ea999d1967"
)
CW15_STDOUT = ROOT / (
    "artifacts/cw15_consumed_valid_official6_sequential_tangent_"
    "20260802_v1.stdout.json"
)
CW15_STDOUT_SHA256 = (
    "6e7f7bd58e0cf4465f9be3dfba4222d328ee8520d1050d99a008c4c543117b54"
)
CW15_STDERR = ROOT / (
    "artifacts/cw15_consumed_valid_official6_sequential_tangent_"
    "20260802_v1.stderr-audit.json"
)
CW15_STDERR_SHA256 = (
    "b05165d9ba1476afc6be133045029eb18b508ed3afea29ffb3716d3ca37b7864"
)
CW15_LAUNCHER_SHA256 = (
    "88f813644c132ffd3fdb3109d962d51826a8835705ac579c713021e4c0c508e8"
)

CW15_ITERATION1_ADDITIONAL_L2 = 0.0009424461480998953
CW15_ITERATION1_POINT_SHA256 = (
    "e3814600afa6f1c7843ca073d89a86cb1e32cf162d89f8cbfaf8db378ad3183f"
)
CW15_ITERATION1_MODEL_SHA256 = (
    "ba5098d4e5c52bb6a02c4e69fa2015428d018f85953edc4fb6bc14330634d769"
)
CW15_ITERATION1_OUTPUT_SHA256 = (
    "8d5fd5ced2091c2ecb3dcc8de4f18f317959742da443dd5c6812921bd56be9e6"
)
CW15_ITERATION1_ADDITIONS_SHA256 = (
    "e91917703f6573f90016444c07eb7350a565b69a8a3362b48d1c7ec4710f5dc0"
)
CW15_INITIAL_ACTIVE_SHA256 = (
    "e70e14b370ae22ec2ab88d4cca9d9337ca005aa68c67af02b7a581e5ae332682"
)
CW15_ITERATION1_ANCHOR_LEDGER_SHA256 = (
    "2d853f4fda11d5ef8a17d9ca343fbcfd0657eaf0416326e02efe6b0586cfcc97"
)
CW15_ITERATION1_ANCHOR_ORDER_SHA256 = (
    "6e66ae93fef82b59e0f82b85a71d8c7a6b2e5b812880729a7597cf25be6e9a3e"
)
CW15_ITERATION2_UNCAPPED_L2 = 0.001501167697632579
CW14_ANCHOR50_ONLY_L2 = 0.0009447829261258565
CW14_ANCHOR50_ONLY_POINT_SHA256 = (
    "08712b376218bbe223ff7a9de1373413af8f2bf32e6cf797ac1959b12e342721"
)
CW15_COMBINED_ANCHOR50_HISTORICAL12_ROW_COUNT = 62
CW11_PHI_ON_EXPANDED_50 = -0.0078125
CW15_ITERATION1_PHI_ON_EXPANDED_50 = -0.015625

TOTAL_L2_CAP = 0.001
MAX_OFFICIAL_EVALUATIONS = 12
TRUST_RADIUS_DIVISORS = (8, 16, 32, 64)
TRUST_RADII = tuple(TOTAL_L2_CAP / value for value in TRUST_RADIUS_DIVISORS)
L2_ABS_TOL = 1e-12
LINEAR_RESIDUAL_TOL = 1e-8
SVD_RELATIVE_RANK_TOL = 1e-12
SLSQP_FTOL = 1e-12
SLSQP_MAXITER = 2000
POLICY_METRICS = ("set", "hybrid", "ordered", "top1")
FIRST_FEASIBLE = True
CLASSIFICATION = {
    "specialist_valid_consumed": True,
    "dev_tuning_only": True,
    "promotion_evidence": False,
    "broad_access": False,
    "gold_access": False,
    "package_upload_submission": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def regular_file_evidence(
    path: Path, expected_sha: str, expected_mode: int
) -> dict[str, Any]:
    observed_stat = path.lstat()
    observed_mode = stat.S_IMODE(observed_stat.st_mode)
    observed_sha = sha256_file(path)
    checks = {
        "regular": stat.S_ISREG(observed_stat.st_mode),
        "one_link": observed_stat.st_nlink == 1,
        "mode_exact": observed_mode == expected_mode,
        "sha256_exact": observed_sha == expected_sha,
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen file drift: {path}: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": observed_sha,
        "mode": f"{observed_mode:04o}",
        "expected_mode": f"{expected_mode:04o}",
        "nlink": int(observed_stat.st_nlink),
        "bytes": int(observed_stat.st_size),
        "regular": True,
        "checks": checks,
    }


def load_cw15_parent() -> tuple[ModuleType, dict[str, Any]]:
    evidence = regular_file_evidence(CW15_PARENT, CW15_PARENT_SHA256, 0o555)
    spec = importlib.util.spec_from_file_location(
        "cw16_exact_frozen_cw15_parent", CW15_PARENT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create exact CW15 parent import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def read_json_exact(path: Path, expected_sha: str) -> dict[str, Any]:
    regular_file_evidence(path, expected_sha, 0o444)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_runtime(*, require_cuda: bool) -> dict[str, Any]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated_exact": sys.flags.isolated == 1,
        "dont_write_bytecode_exact": sys.flags.dont_write_bytecode == 1,
        "cublas_workspace_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        == ":4096:8",
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 runtime contract failed: {checks}")
    result: dict[str, Any] = {"checks": checks, "cuda_required": require_cuda}
    if require_cuda:
        import torch

        cuda_checks = {
            "available": torch.cuda.is_available(),
            "native_bf16": torch.cuda.is_available()
            and torch.cuda.is_bf16_supported(),
            "device_count_positive": torch.cuda.device_count() > 0,
        }
        if not all(cuda_checks.values()):
            raise RuntimeError(f"CW16 CUDA contract failed: {cuda_checks}")
        result["cuda"] = {
            "checks": cuda_checks,
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    return result


def validate_frozen_inputs_40(cw15: ModuleType) -> dict[str, Any]:
    inherited = cw15.validate_frozen_inputs()
    if inherited.get("binding_count") != 37 or inherited.get("all_exact") is not True:
        raise RuntimeError("CW16 inherited frozen input cardinality drift")
    extras = [
        regular_file_evidence(CW15_ATTEMPT, CW15_ATTEMPT_SHA256, 0o444),
        regular_file_evidence(CW15_STDOUT, CW15_STDOUT_SHA256, 0o444),
        regular_file_evidence(CW15_STDERR, CW15_STDERR_SHA256, 0o444),
    ]
    records = [*inherited["records"], *extras]
    if len(records) != 40 or len({value["path"] for value in records}) != 40:
        raise RuntimeError("CW16 exact old37 plus CW15-three binding drift")
    return {
        "all_exact": True,
        "binding_count": 40,
        "inherited_old_binding_count": 37,
        "new_CW15_binding_count": 3,
        "records": records,
        "ordered_identity_sha256": canonical_sha(
            [
                [value["path"], value["sha256"], value["mode"], value["nlink"]]
                for value in records
            ]
        ),
    }


def validate_cw15_closure_evidence(cw15: ModuleType) -> dict[str, Any]:
    marker = read_json_exact(CW15_ATTEMPT, CW15_ATTEMPT_SHA256)
    closed = read_json_exact(CW15_STDOUT, CW15_STDOUT_SHA256)
    stderr = read_json_exact(CW15_STDERR, CW15_STDERR_SHA256)
    second = closed.get("second_stage")
    terminal = closed.get("terminal")
    if not isinstance(second, dict) or not isinstance(terminal, dict):
        raise RuntimeError("CW15 closure missing second_stage/terminal")
    iterations = second.get("iterations")
    if not isinstance(iterations, list) or len(iterations) != 3:
        raise RuntimeError("CW15 closure iteration cardinality drift")
    iteration0, iteration1, iteration2 = iterations
    child_stderr = base64.b64decode(
        str(stderr["base64"]).encode("ascii"), validate=True
    )
    candidate_added = iteration1["post_merge_candidate_affine_tangent_merge"][
        "added"
    ]
    anchor_added = iteration1["post_merge_new_semantic_anchor_affine_merge"][
        "added"
    ]

    def physical_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
        identity = value["semantic_identity"]
        return (
            str(identity[0]),
            str(identity[1]),
            int(identity[3]),
            int(identity[4]),
            float(value["threshold"]),
            str(value["linearization_point_sha256"]),
            str(value["gradient_float64_le_sha256"]),
            float(value["rhs"]),
        )

    candidate_physical = {physical_signature(value) for value in candidate_added}
    anchor_physical = {physical_signature(value) for value in anchor_added}
    all_affine = second["optimization_affine_closure_ledger"]
    anchor50 = [
        value for value in all_affine if value["linearization_kind"] == "CW11_anchor"
    ]
    historical12 = [
        value
        for value in all_affine
        if value["linearization_kind"] != "CW11_anchor"
    ]
    anchor50_sha = canonical_sha(
        sorted(anchor50, key=lambda value: tuple(value["semantic_identity"]) + (
            value["linearization_point_sha256"],
            value["linearization_model_state_sha256"],
            value["threshold_float64_le_sha256"],
        ))
    )
    historical12_sha = canonical_sha(
        sorted(historical12, key=lambda value: tuple(value["semantic_identity"]) + (
            value["linearization_point_sha256"],
            value["linearization_model_state_sha256"],
            value["threshold_float64_le_sha256"],
        ))
    )
    checks = {
        "marker_schema_status_exact": marker.get("schema_version")
        == "ptcg-cw15-consumed-valid-official6-one-shot-attempt-v1"
        and marker.get("status")
        == "one_shot_attempt_consumed_before_cuda_and_official6",
        "marker_one_attempt_no_retry": marker.get("attempts_authorized") == 1
        and marker.get("retry_authorized") is False
        and marker.get("marker_created_before_cuda_and_official6") is True,
        "marker_solver_launcher_exact": marker.get("solver", {}).get("sha256")
        == CW15_PARENT_SHA256
        and marker.get("launcher", {}).get("sha256") == CW15_LAUNCHER_SHA256
        and marker.get("solver", {}).get("mode_octal") == "0555"
        and marker.get("launcher", {}).get("mode_octal") == "0555",
        "marker_input_count_exact37": marker.get(
            "independent_frozen_input_rehash", {}
        ).get("binding_count")
        == 37
        and marker.get("independent_frozen_input_rehash", {}).get("pass") is True,
        "closed_schema_status_exact": closed.get("schema_version")
        == "ptcg-cw15-consumed-valid-official-b256-sequential-affine-tangent-cuttingplane-v1"
        and closed.get("status") == "closed_no_CW15_candidate"
        and second.get("status") == "closed_no_CW15_candidate"
        and terminal.get("status") == "closed_no_CW15_candidate",
        "closed_run_integrity_exact": closed.get("run_executed") is True
        and closed.get("cuda_accessed") is True
        and closed.get("writes_performed") is False
        and closed.get("classification") == CLASSIFICATION
        and closed.get("final_integrity", {}).get("pass") is True
        and closed.get("source_audit", {}).get("pass") is True
        and closed.get("frozen_inputs", {}).get("binding_count") == 37,
        "closed_self_parent_exact": closed.get("input_lock", {}).get("self", {}).get(
            "sha256"
        )
        == CW15_PARENT_SHA256,
        "iteration_order_exact": [value.get("iteration") for value in iterations]
        == [0, 1, 2],
        "initial_active_exact38": iteration0.get("active_cut_count") == 38
        and iteration0.get("active_cut_identity_sha256")
        == CW15_INITIAL_ACTIVE_SHA256,
        "canonical_bytes_exact_parent": canonical_json(
            {"unicode": "Pokémon", "ordered": [1, 2, 3]}
        )
        == cw15.canonical_json({"unicode": "Pokémon", "ordered": [1, 2, 3]}),
        "canonical_recomputes_initial_active_sha": canonical_sha(
            [value["identity"] for value in iteration0["active_cut_gates"]["records"]]
        )
        == CW15_INITIAL_ACTIVE_SHA256,
        "iteration1_replay_identity_exact": iteration1.get(
            "additional_float64_le_sha256"
        )
        == CW15_ITERATION1_POINT_SHA256
        and iteration1.get("model_state_sha256") == CW15_ITERATION1_MODEL_SHA256
        and iteration1.get("output_state_fingerprint", {}).get("sha256")
        == CW15_ITERATION1_OUTPUT_SHA256
        and math.isclose(
            float(iteration1["additional_l2"]),
            CW15_ITERATION1_ADDITIONAL_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        ),
        "iteration1_anchor_QP_exact": iteration1.get("affine_qp_audit", {}).get(
            "affine_ledger_sha256"
        )
        == CW15_ITERATION1_ANCHOR_LEDGER_SHA256
        and iteration1.get("affine_qp_audit", {}).get(
            "ordered_affine_identity_sha256"
        )
        == CW15_ITERATION1_ANCHOR_ORDER_SHA256
        and iteration1.get("qp", {}).get("svd_rank") == 37,
        "iteration1_semantic_merge_exact": iteration1.get(
            "deterministic_addition_count"
        )
        == 12
        and iteration1.get("deterministic_addition_identity_sha256")
        == CW15_ITERATION1_ADDITIONS_SHA256
        and iteration1.get("post_oracle_cut_merge", {}).get("active_count") == 50
        and not iteration1.get("post_oracle_cut_merge", {}).get("strengthened"),
        "canonical_recomputes_iteration1_additions_sha": canonical_sha(
            [
                list(cw15.cut_identity(value))
                for value in iteration1["post_oracle_cut_merge"]["added"]
            ]
        )
        == CW15_ITERATION1_ADDITIONS_SHA256,
        "iteration1_phi_exact": iteration1.get("active_cut_gates", {}).get(
            "residual_min"
        )
        == CW15_ITERATION1_PHI_ON_EXPANDED_50,
        "iteration2_affine_closure_exact": iteration2.get("close_reason")
        == "fail_closed_anchored_minimum_total_exceeds_cap"
        and math.isclose(
            float(iteration2["anchored_minimum_total_l2"]),
            CW15_ITERATION2_UNCAPPED_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        and iteration2.get("clipped_vector_applied") is False
        and iteration2.get("optimization_affine_tangent_count")
        == CW15_COMBINED_ANCHOR50_HISTORICAL12_ROW_COUNT
        and iteration2.get("affine_qp_audit", {}).get(
            "affine_tangent_count"
        )
        == CW15_COMBINED_ANCHOR50_HISTORICAL12_ROW_COUNT,
        "new_rows_exact12_plus12": len(candidate_added) == 12
        and len(anchor_added) == 12,
        "new_rows_exact3_plus3_physical": len(candidate_physical) == 3
        and len(anchor_physical) == 3,
        "closure_ledger_exact50_anchor12_historical": len(anchor50) == 50
        and len(historical12) == 12
        and len(all_affine)
        == CW15_COMBINED_ANCHOR50_HISTORICAL12_ROW_COUNT,
        "anchor50_and_historical12_kind_separation_exact": all(
            value["linearization_kind"] == "CW11_anchor"
            and value["linearization_point_sha256"]
            == cw15.CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256
            and value["linearization_model_state_sha256"]
            == cw15.CW11_MODEL_SHA256
            for value in anchor50
        )
        and all(
            value["linearization_kind"] == "post_merge_actual_active_violation"
            and value["linearization_point_sha256"]
            == CW15_ITERATION1_POINT_SHA256
            and value["linearization_model_state_sha256"]
            == CW15_ITERATION1_MODEL_SHA256
            for value in historical12
        ),
        "historical12_archive_sanitized_no_private_gradient": all(
            "gradient_float64" not in value for value in historical12
        ),
        "immutable_anchor50_only_reference_exact": math.isclose(
            float(cw15.CW14_TERMINAL_ADDITIONAL_L2),
            CW14_ANCHOR50_ONLY_L2,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and cw15.CW14_STALLED_ADDITIONAL_SHA256
        == CW14_ANCHOR50_ONLY_POINT_SHA256
        and CW14_ANCHOR50_ONLY_L2 < TOTAL_L2_CAP
        and CW15_ITERATION2_UNCAPPED_L2 > TOTAL_L2_CAP,
        "terminal_absent_exact": second.get("decision", {}).get(
            "terminal_iteration"
        )
        is None
        and terminal.get("reconstruction_payload") is None,
        "stderr_lossless_exact": stderr.get("schema_version")
        == "ptcg-cw15-consumed-valid-official6-one-shot-stderr-audit-v1"
        and stderr.get("status") == "captured_losslessly_not_a_success_veto"
        and stderr.get("bytes") == 347
        and hashlib.sha256(child_stderr).hexdigest()
        == "5ceb096b8ac5376a3ee276bf70012f90cb2fd479e3fa2804f2c0a8145aff4512"
        == stderr.get("sha256")
        and stderr.get("post_child_integrity", {}).get("pass") is True,
        "threshold_function_shas_exact": closed.get("source_audit", {}).get(
            "threshold_function_shas"
        )
        == cw15.CW13_THRESHOLD_FUNCTION_SHAS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 consumed CW15 closure drift: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "parent_solver_sha256": CW15_PARENT_SHA256,
        "attempt_marker_sha256": CW15_ATTEMPT_SHA256,
        "closed_stdout_sha256": CW15_STDOUT_SHA256,
        "stderr_audit_sha256": CW15_STDERR_SHA256,
        "bootstrap": {
            "additional_l2": CW15_ITERATION1_ADDITIONAL_L2,
            "point_sha256": CW15_ITERATION1_POINT_SHA256,
            "model_sha256": CW15_ITERATION1_MODEL_SHA256,
            "output_sha256": CW15_ITERATION1_OUTPUT_SHA256,
            "additions_sha256": CW15_ITERATION1_ADDITIONS_SHA256,
            "semantic_count_after_merge": 50,
            "CW11_phi_expanded50": CW11_PHI_ON_EXPANDED_50,
            "bootstrap_phi_expanded50": CW15_ITERATION1_PHI_ON_EXPANDED_50,
        },
        "archive": {
            "CW11_anchor_count": len(anchor50),
            "CW11_anchor_sanitized_sha256": anchor50_sha,
            "CW11_anchor_only_reference_l2": CW14_ANCHOR50_ONLY_L2,
            "CW11_anchor_only_reference_point_sha256":
            CW14_ANCHOR50_ONLY_POINT_SHA256,
            "historical_candidate_count": len(historical12),
            "historical_candidate_sanitized_sha256": historical12_sha,
            "historical_candidate_records": historical12,
        },
    }


def physical_key(cut: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(cut["context_type"]),
        str(cut["context_hash"]),
        str(cut["decision_sha256"]),
        int(cut["offset_zero_based"]),
        int(cut["positive_option"]),
        int(cut["negative_option"]),
    )


def deduplicate_active_physical(
    cw15: ModuleType,
    active: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    affine_records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    active_identities = [cw15.cut_identity(value) for value in active]
    gate_identities = [tuple(value["identity"]) for value in gates["records"]]
    affine_identities = [tuple(value["semantic_identity"]) for value in affine_records]
    identity_checks = {
        "active_identity_unique": len(active_identities) == len(set(active_identities)),
        "gate_identity_unique": len(gate_identities) == len(set(gate_identities)),
        "affine_identity_unique": len(affine_identities)
        == len(set(affine_identities)),
        "active_gate_identity_exact": set(active_identities) == set(gate_identities),
        "active_affine_identity_exact": set(active_identities)
        == set(affine_identities),
        "cardinalities_exact": len(active) == len(gate_identities)
        == len(affine_identities),
    }
    if not all(identity_checks.values()):
        raise RuntimeError(f"CW16 physical dedup identity drift: {identity_checks}")
    gate_by_identity = {
        tuple(value["identity"]): value for value in gates["records"]
    }
    affine_by_identity = {
        tuple(value["semantic_identity"]): value for value in affine_records
    }
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for cut in active:
        key = physical_key(cut)
        groups[key].append(cut)
    optimization_records: list[Mapping[str, Any]] = []
    mappings = []
    four_to_one_count = 0
    for key in sorted(groups):
        values = sorted(groups[key], key=cw15.cut_identity)
        identities = [cw15.cut_identity(value) for value in values]
        metrics = [str(value[2]) for value in identities]
        margins = [float(gate_by_identity[value]["margin"]) for value in identities]
        residuals = [float(gate_by_identity[value]["residual"]) for value in identities]
        thresholds = [
            cw15.float64_scalar_sha256(float(value["threshold"]))
            for value in values
        ]
        margin_shas = [cw15.float64_scalar_sha256(value) for value in margins]
        residual_shas = [cw15.float64_scalar_sha256(value) for value in residuals]
        gradient_shas = [
            str(affine_by_identity[value]["gradient_float64_le_sha256"])
            for value in identities
        ]
        rhs_shas = [
            cw15.float64_scalar_sha256(float(affine_by_identity[value]["rhs"]))
            for value in identities
        ]
        exact_four_checks = {
            "exactly_four": len(values) == 4,
            "metrics_exact": len(set(metrics)) == 4
            and set(metrics) == set(POLICY_METRICS),
            "thresholds_bit_equal": len(set(thresholds)) == 1,
            "margins_bit_equal": len(set(margin_shas)) == 1,
            "residuals_bit_equal": len(set(residual_shas)) == 1,
            "gradients_hash_equal": len(set(gradient_shas)) == 1,
            "rhs_bit_equal": len(set(rhs_shas)) == 1,
        }
        exact_four_to_one = all(exact_four_checks.values())
        if exact_four_to_one:
            four_to_one_count += 1
            retained_identities = [identities[0]]
            reason = "exact_four_policy_metrics_all_numeric_hashes_equal"
        else:
            retained_identities = identities
            if len(values) in (1, 2, 3):
                reason = f"partial_{len(values)}_semantic_group_keep_all"
            elif len(values) == 4:
                reason = "four_semantics_not_all_exact_keep_all"
            else:
                reason = "noncanonical_semantic_group_keep_all"
        optimization_records.extend(
            affine_by_identity[value] for value in retained_identities
        )
        mappings.append(
            {
                "physical_key": list(key),
                "semantic_identities": [list(value) for value in identities],
                "semantic_count": len(values),
                "retained_optimization_identities": [
                    list(value) for value in retained_identities
                ],
                "retained_optimization_count": len(retained_identities),
                "exact_four_checks": exact_four_checks,
                "exact_four_to_one": exact_four_to_one,
                "reason": reason,
                "threshold_float64_le_sha256": thresholds,
                "margin_float64_le_sha256": margin_shas,
                "residual_float64_le_sha256": residual_shas,
                "gradient_float64_le_sha256": gradient_shas,
                "rhs_float64_le_sha256": rhs_shas,
            }
        )
    optimization_records.sort(
        key=lambda value: tuple(value["semantic_identity"])
    )
    mapped_semantics = [
        tuple(identity)
        for value in mappings
        for identity in value["semantic_identities"]
    ]
    retained_identities = [
        tuple(value["semantic_identity"]) for value in optimization_records
    ]
    checks = {
        **identity_checks,
        "semantic_mapping_bijective": len(mapped_semantics) == len(active_identities)
        and len(set(mapped_semantics)) == len(mapped_semantics)
        and set(mapped_semantics) == set(active_identities),
        "retained_identity_unique": len(retained_identities)
        == len(set(retained_identities)),
        "retained_identity_subset": set(retained_identities).issubset(
            set(active_identities)
        ),
        "only_exact_four_groups_are_deduplicated": all(
            value["retained_optimization_count"]
            == (1 if value["exact_four_to_one"] else value["semantic_count"])
            for value in mappings
        ),
        "cardinality_formula_exact": len(optimization_records)
        == len(active) - 3 * four_to_one_count,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 physical dedup audit failed: {checks}")
    return optimization_records, {
        "checks": checks,
        "pass": True,
        "semantic_count": len(active),
        "physical_group_count": len(groups),
        "physical_count": len(optimization_records),
        "optimization_row_count": len(optimization_records),
        "exact_four_to_one_group_count": four_to_one_count,
        "fallback_keep_all_group_count": len(groups) - four_to_one_count,
        "mapping_sha256": canonical_sha(mappings),
        "mappings": mappings,
    }


def physical_dedup_self_test(cw15: ModuleType) -> dict[str, Any]:
    import numpy as np

    shared = {
        "context_type": "official_B256",
        "context_hash": "a" * 64,
        "decision_sha256": "b" * 64,
        "offset_zero_based": 7,
        "positive_option": 1,
        "negative_option": 2,
        "threshold": 0.0009765625,
    }
    active = [{**shared, "metric": metric} for metric in POLICY_METRICS]
    active.append(
        {
            "context_type": "legacy_B33",
            "context_hash": "c" * 64,
            "decision_sha256": "legacy-line:" + "d" * 64,
            "offset_zero_based": 3,
            "positive_option": 0,
            "negative_option": 1,
            "threshold": 0.0,
            "metric": "legacy_pair_unit",
        }
    )
    gates = {
        "records": [
            {
                "identity": list(cw15.cut_identity(value)),
                "margin": 0.0,
                "threshold": float(value["threshold"]),
                "residual": -float(value["threshold"]),
                "pass": float(value["threshold"]) == 0.0,
            }
            for value in active
        ]
    }
    shared_gradient = np.asarray([1.0, -2.0], dtype=np.float64)

    def tangent_records(
        values: Sequence[Mapping[str, Any]],
        *,
        gradient_override: Mapping[tuple[Any, ...], str] | None = None,
    ) -> list[dict[str, Any]]:
        result = []
        for value in values:
            identity = cw15.cut_identity(value)
            gradient_sha = hashlib.sha256(
                np.ascontiguousarray(shared_gradient.astype("<f8")).tobytes()
            ).hexdigest()
            if gradient_override is not None:
                gradient_sha = gradient_override.get(identity, gradient_sha)
            result.append(
                {
                    "semantic_identity": list(identity),
                    "gradient_float64_le_sha256": gradient_sha,
                    "gradient_float64": shared_gradient.copy(),
                    "rhs": float(value["threshold"]),
                    "actual_residual": -float(value["threshold"]),
                }
            )
        return result

    affine = tangent_records(active)
    representatives, audit = deduplicate_active_physical(
        cw15, active, gates, affine
    )
    partial_audits = []
    for partial_count in (1, 2, 3):
        partial_active = active[:partial_count]
        partial_gates = {"records": gates["records"][:partial_count]}
        partial_affine = affine[:partial_count]
        partial_records, partial_audit = deduplicate_active_physical(
            cw15, partial_active, partial_gates, partial_affine
        )
        partial_audits.append(
            {
                "semantic_count": partial_count,
                "optimization_row_count": len(partial_records),
                "audit": partial_audit,
            }
        )

    def four_variant(
        *, threshold_change: bool = False, margin_change: bool = False,
        gradient_change: bool = False,
    ) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
        variant_active = copy.deepcopy(active[:4])
        if threshold_change:
            variant_active[-1]["threshold"] = 0.001953125
        variant_gates = {
            "records": [
                {
                    "identity": list(cw15.cut_identity(value)),
                    "margin": (
                        0.001953125
                        if margin_change and index == 3
                        else 0.0
                    ),
                    "threshold": float(value["threshold"]),
                    "residual": (
                        0.001953125
                        if margin_change and index == 3
                        else 0.0
                    )
                    - float(value["threshold"]),
                    "pass": False,
                }
                for index, value in enumerate(variant_active)
            ]
        }
        gradient_override = None
        if gradient_change:
            gradient_override = {
                cw15.cut_identity(variant_active[-1]): "f" * 64
            }
        variant_affine = tangent_records(
            variant_active, gradient_override=gradient_override
        )
        return deduplicate_active_physical(
            cw15, variant_active, variant_gates, variant_affine
        )

    threshold_variant, threshold_audit = four_variant(threshold_change=True)
    margin_variant, margin_audit = four_variant(margin_change=True)
    gradient_variant, gradient_audit = four_variant(gradient_change=True)
    duplicate_identity_fails_closed = False
    try:
        deduplicate_active_physical(
            cw15,
            [active[0], active[0]],
            {"records": [gates["records"][0], gates["records"][0]]},
            [affine[0], affine[0]],
        )
    except RuntimeError:
        duplicate_identity_fails_closed = True
    missing_identity_fails_closed = False
    try:
        deduplicate_active_physical(
            cw15, active[:2], {"records": gates["records"][:2]}, affine[:1]
        )
    except RuntimeError:
        missing_identity_fails_closed = True

    exact_four_full = affine[:4]
    exact_four_dedup = representatives[:1]
    witness_points = (
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([0.25, -0.5], dtype=np.float64),
        np.asarray([-1.0, 2.0], dtype=np.float64),
    )
    feasible_region_equivalent = all(
        min(
            float(np.asarray(value["gradient_float64"]) @ point - value["rhs"])
            for value in exact_four_full
        )
        == min(
            float(np.asarray(value["gradient_float64"]) @ point - value["rhs"])
            for value in exact_four_dedup
        )
        for point in witness_points
    )
    checks = {
        "five_semantics_to_two_physical": audit["semantic_count"] == 5
        and audit["optimization_row_count"] == 2,
        "exactly_one_four_to_one_group": audit["exact_four_to_one_group_count"]
        == 1,
        "representatives_exact2": len(representatives) == 2,
        "partial1_2_3_preserved_rowwise": all(
            value["semantic_count"] == value["optimization_row_count"]
            and value["audit"]["exact_four_to_one_group_count"] == 0
            for value in partial_audits
        ),
        "four_nonidentical_threshold_preserved_rowwise": len(threshold_variant)
        == 4
        and threshold_audit["exact_four_to_one_group_count"] == 0,
        "four_nonidentical_margin_preserved_rowwise": len(margin_variant) == 4
        and margin_audit["exact_four_to_one_group_count"] == 0,
        "four_nonidentical_gradient_preserved_rowwise": len(gradient_variant)
        == 4
        and gradient_audit["exact_four_to_one_group_count"] == 0,
        "semantic_mapping_bijective": audit["checks"][
            "semantic_mapping_bijective"
        ],
        "duplicate_identity_fails_closed": duplicate_identity_fails_closed,
        "missing_identity_fails_closed": missing_identity_fails_closed,
        "dedup_feasible_region_equivalent_to_unreduced":
        feasible_region_equivalent,
        "mapping_sha_hex64": len(audit["mapping_sha256"]) == 64,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 physical dedup self-test failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "audit": audit,
        "partial_audits": partial_audits,
        "nonidentical_four_audits": {
            "threshold": threshold_audit,
            "margin": margin_audit,
            "gradient": gradient_audit,
        },
    }


def solve_local_trust_maximin(
    anchor_gradients: Any,
    anchor_rhs: Any,
    local_gradients: Any,
    local_residual_at_center: Any,
    center: Any,
    trust_radius: float,
    np: Any,
    optimize: Any,
) -> tuple[Any, dict[str, Any]]:
    """Two-stage reduced-space SOCP-like SLSQP; never clips a solution."""

    anchor_gradients = np.asarray(anchor_gradients, dtype=np.float64)
    anchor_rhs = np.asarray(anchor_rhs, dtype=np.float64)
    local_gradients = np.asarray(local_gradients, dtype=np.float64)
    local_residual_at_center = np.asarray(
        local_residual_at_center, dtype=np.float64
    )
    center = np.asarray(center, dtype=np.float64)
    n = center.size
    shape_checks = {
        "anchor_matrix": anchor_gradients.ndim == 2
        and anchor_gradients.shape[1] == n
        and anchor_gradients.shape[0] > 0,
        "anchor_rhs": anchor_rhs.shape == (anchor_gradients.shape[0],),
        "local_matrix": local_gradients.ndim == 2
        and local_gradients.shape[1] == n
        and local_gradients.shape[0] > 0,
        "local_residual": local_residual_at_center.shape
        == (local_gradients.shape[0],),
        "finite": bool(
            np.isfinite(anchor_gradients).all()
            and np.isfinite(anchor_rhs).all()
            and np.isfinite(local_gradients).all()
            and np.isfinite(local_residual_at_center).all()
            and np.isfinite(center).all()
        ),
        "caps_exact": TOTAL_L2_CAP == 0.001
        and trust_radius in TRUST_RADII,
    }
    if not all(shape_checks.values()):
        raise RuntimeError(f"CW16 local-trust input drift: {shape_checks}")

    span_rows = np.concatenate(
        [anchor_gradients, local_gradients, center.reshape(1, -1)], axis=0
    )
    _, singular_values, vh = np.linalg.svd(span_rows, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise RuntimeError("CW16 local-trust row span has zero rank")
    rank = int(
        (singular_values > singular_values[0] * SVD_RELATIVE_RANK_TOL).sum()
    )
    basis = vh[:rank]
    anchor_reduced = anchor_gradients @ basis.T
    local_reduced = local_gradients @ basis.T
    center_reduced = basis @ center
    center_reconstruction = basis.T @ center_reduced
    center_reconstruction_error = float(np.linalg.norm(center_reconstruction - center))
    if center_reconstruction_error > L2_ABS_TOL:
        raise RuntimeError("CW16 reduced basis does not contain exact current center")

    # SLSQP is poorly conditioned when the ball constraints are expressed as
    # differences of ~1e-8 squared norms.  Normalize coordinates by the exact
    # unchanged total cap.  Anchor inequalities may use independent positive
    # row scales because they are hard constraints.  Every local residual and
    # z use one shared positive global scale so the raw max-min objective is
    # unchanged.  These transformations preserve the feasible set, z ordering,
    # and minimum-norm ordering exactly; final gates are always recomputed in
    # the original unscaled coordinates.
    coordinate_scale = TOTAL_L2_CAP
    trust_radius_scaled = trust_radius / coordinate_scale
    center_scaled = center_reduced / coordinate_scale
    anchor_gradients_scaled = anchor_reduced * coordinate_scale
    local_gradients_scaled = local_reduced * coordinate_scale
    numeric_scale_floor = np.finfo(np.float64).eps
    anchor_row_scales = np.maximum.reduce(
        [
            np.abs(anchor_rhs),
            np.linalg.norm(anchor_gradients_scaled, axis=1),
            np.full(anchor_rhs.shape, numeric_scale_floor, dtype=np.float64),
        ]
    )
    local_motion_scales = (
        np.linalg.norm(local_gradients_scaled, axis=1) * trust_radius_scaled
    )
    z_scale = float(
        max(
            float(np.abs(local_residual_at_center).max()),
            float(local_motion_scales.max()),
            numeric_scale_floor,
        )
    )
    local_global_scale = z_scale
    scaling_checks = {
        "coordinate_scale_exact_total_cap": coordinate_scale == TOTAL_L2_CAP,
        "trust_radius_scaled_exact": trust_radius_scaled
        == trust_radius / TOTAL_L2_CAP,
        "center_scaled_finite": bool(np.isfinite(center_scaled).all()),
        "anchor_row_scales_positive_finite": bool(
            np.isfinite(anchor_row_scales).all()
            and (anchor_row_scales > 0.0).all()
        ),
        "single_global_local_residual_scale": math.isfinite(
            local_global_scale
        )
        and local_global_scale > 0.0,
        "global_scale_preserves_raw_max_min_objective":
        local_global_scale == z_scale,
        "z_scale_positive_finite": math.isfinite(z_scale) and z_scale > 0.0,
    }
    if not all(scaling_checks.values()):
        raise RuntimeError(f"CW16 local-trust scaling drift: {scaling_checks}")

    def anchor_fun_stage1(value: Any) -> Any:
        return (
            anchor_gradients_scaled @ value[:-1] - anchor_rhs
        ) / anchor_row_scales

    def anchor_jac_stage1(value: Any) -> Any:
        del value
        return np.concatenate(
            [
                anchor_gradients_scaled / anchor_row_scales[:, None],
                np.zeros((anchor_reduced.shape[0], 1)),
            ],
            axis=1,
        )

    def local_fun_stage1(value: Any) -> Any:
        return (
            local_residual_at_center
            + local_gradients_scaled @ (value[:-1] - center_scaled)
            - z_scale * value[-1]
        ) / local_global_scale

    def local_jac_stage1(value: Any) -> Any:
        del value
        return np.concatenate(
            [
                local_gradients_scaled / local_global_scale,
                np.full(
                    (local_reduced.shape[0], 1),
                    -z_scale / local_global_scale,
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )

    def total_cap_fun_stage1(value: Any) -> float:
        return float(1.0 - value[:-1] @ value[:-1])

    def total_cap_jac_stage1(value: Any) -> Any:
        return np.concatenate([-2.0 * value[:-1], np.zeros(1)])

    def trust_fun_stage1(value: Any) -> float:
        delta = value[:-1] - center_scaled
        return float(trust_radius_scaled**2 - delta @ delta)

    def trust_jac_stage1(value: Any) -> Any:
        return np.concatenate(
            [-2.0 * (value[:-1] - center_scaled), np.zeros(1)]
        )

    initial_z = min(0.0, float(local_residual_at_center.min()))
    stage1_initial = np.concatenate(
        [center_scaled, np.asarray([initial_z / z_scale])]
    )
    stage1_constraints = [
        {"type": "ineq", "fun": anchor_fun_stage1, "jac": anchor_jac_stage1},
        {"type": "ineq", "fun": local_fun_stage1, "jac": local_jac_stage1},
        {
            "type": "ineq",
            "fun": lambda value: -float(value[-1]),
            "jac": lambda value: np.concatenate(
                [np.zeros(value.size - 1), np.asarray([-1.0])]
            ),
        },
        {"type": "ineq", "fun": total_cap_fun_stage1, "jac": total_cap_jac_stage1},
        {"type": "ineq", "fun": trust_fun_stage1, "jac": trust_jac_stage1},
    ]
    stage1 = optimize.minimize(
        lambda value: -float(value[-1]),
        stage1_initial,
        jac=lambda value: np.concatenate(
            [np.zeros(value.size - 1), np.asarray([-1.0])]
        ),
        constraints=stage1_constraints,
        method="SLSQP",
        options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER, "disp": False},
    )
    if not bool(stage1.success):
        raise RuntimeError(
            "CW16 hard-anchor local-trust stage1 failed: "
            f"{stage1.status} {stage1.message}"
        )
    stage1_scaled_point = np.asarray(stage1.x[:-1], dtype=np.float64)
    stage1_reduced_point = coordinate_scale * stage1_scaled_point
    stage1_raw_point = basis.T @ stage1_reduced_point
    stage1_z_solver_raw = float(stage1.x[-1]) * z_scale
    stage1_anchor_residual_raw = anchor_gradients @ stage1_raw_point - anchor_rhs
    stage1_local_prediction_raw = local_residual_at_center + local_gradients @ (
        stage1_raw_point - center
    )
    # Certify a raw fixed-z value from the actual stage1 point.  This can only
    # move z downward by numerical feasibility noise; a material adjustment
    # fails closed.
    z_star = min(
        0.0,
        stage1_z_solver_raw,
        float(stage1_local_prediction_raw.min()),
    )
    stage1_z_certification_adjustment = stage1_z_solver_raw - z_star
    stage1_certification_checks = {
        "anchor_raw_residual": float(stage1_anchor_residual_raw.min())
        >= -LINEAR_RESIDUAL_TOL,
        "local_raw_residual_at_certified_z": float(
            (stage1_local_prediction_raw - z_star).min()
        )
        >= -LINEAR_RESIDUAL_TOL,
        "total_raw_cap": float(np.linalg.norm(stage1_raw_point))
        <= TOTAL_L2_CAP + L2_ABS_TOL,
        "trust_raw_cap": float(np.linalg.norm(stage1_raw_point - center))
        <= trust_radius + L2_ABS_TOL,
        "z_raw_nonpositive": z_star <= 0.0,
        "z_certification_adjustment_only_numeric":
        stage1_z_certification_adjustment >= 0.0
        and stage1_z_certification_adjustment <= LINEAR_RESIDUAL_TOL,
    }
    if not all(stage1_certification_checks.values()):
        raise RuntimeError(
            "CW16 scaled stage1 raw certification failed: "
            f"{stage1_certification_checks}"
        )

    def anchor_fun_stage2(value: Any) -> Any:
        return (
            anchor_gradients_scaled @ value - anchor_rhs
        ) / anchor_row_scales

    def local_fun_stage2(value: Any) -> Any:
        return (
            local_residual_at_center
            + local_gradients_scaled @ (value - center_scaled)
            - z_star
        ) / local_global_scale

    def total_cap_fun_stage2(value: Any) -> float:
        return float(1.0 - value @ value)

    def trust_fun_stage2(value: Any) -> float:
        delta = value - center_scaled
        return float(trust_radius_scaled**2 - delta @ delta)

    stage2_constraints = [
        {
            "type": "ineq",
            "fun": anchor_fun_stage2,
            "jac": lambda value: anchor_gradients_scaled
            / anchor_row_scales[:, None],
        },
        {
            "type": "ineq",
            "fun": local_fun_stage2,
            "jac": lambda value: local_gradients_scaled / local_global_scale,
        },
        {
            "type": "ineq",
            "fun": total_cap_fun_stage2,
            "jac": lambda value: -2.0 * value,
        },
        {
            "type": "ineq",
            "fun": trust_fun_stage2,
            "jac": lambda value: -2.0 * (value - center_scaled),
        },
    ]
    stage2 = optimize.minimize(
        lambda value: 0.5 * float(value @ value),
        np.asarray(stage1.x[:-1], dtype=np.float64),
        jac=lambda value: value,
        constraints=stage2_constraints,
        method="SLSQP",
        options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER, "disp": False},
    )
    if not bool(stage2.success):
        raise RuntimeError(
            "CW16 fixed-z minimum-norm stage2 failed: "
            f"{stage2.status} {stage2.message}"
        )
    proposal = basis.T @ (
        coordinate_scale * np.asarray(stage2.x, dtype=np.float64)
    )
    anchor_residual = anchor_gradients @ proposal - anchor_rhs
    local_prediction = (
        local_residual_at_center + local_gradients @ (proposal - center)
    )
    total_l2 = float(np.linalg.norm(proposal))
    step_l2 = float(np.linalg.norm(proposal - center))
    checks = {
        "shape_checks": all(shape_checks.values()),
        "scaling_checks": all(scaling_checks.values()),
        "stage1_success": bool(stage1.success),
        "stage1_raw_certification": all(stage1_certification_checks.values()),
        "stage2_success": bool(stage2.success),
        "z_nonpositive": z_star <= 0.0,
        "anchor_hard_residual": float(anchor_residual.min())
        >= -LINEAR_RESIDUAL_TOL,
        "current_local_fixed_z_residual": float(local_prediction.min())
        >= z_star - LINEAR_RESIDUAL_TOL,
        "total_cap_hard": total_l2 <= TOTAL_L2_CAP + L2_ABS_TOL,
        "trust_cap_hard": step_l2 <= trust_radius + L2_ABS_TOL,
        "no_clip": True,
        "historical_candidate_active_count_zero": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 local-trust direct gate failed: {checks}")
    return proposal, {
        "checks": checks,
        "pass": True,
        "objective": "max z<=0 then fixed-z minimum total anchored L2",
        "span_shape": list(span_rows.shape),
        "svd_rank": rank,
        "singular_values": [float(value) for value in singular_values],
        "center_reconstruction_error": center_reconstruction_error,
        "anchor_row_count": int(anchor_gradients.shape[0]),
        "current_local_physical_row_count": int(local_gradients.shape[0]),
        "historical_candidate_active_count": 0,
        "trust_radius": trust_radius,
        "scaling": {
            "checks": scaling_checks,
            "pass": all(scaling_checks.values()),
            "coordinate_scale": coordinate_scale,
            "trust_radius_scaled": trust_radius_scaled,
            "single_global_local_residual_scale": local_global_scale,
            "anchor_row_scale_min": float(anchor_row_scales.min()),
            "anchor_row_scale_max": float(anchor_row_scales.max()),
            "raw_problem_direct_gates_after_inverse_scaling": True,
        },
        "z_star": z_star,
        "predicted_phi": float(local_prediction.min()),
        "anchor_residual_min": float(anchor_residual.min()),
        "current_local_residual_min": float(local_prediction.min()),
        "total_l2": total_l2,
        "center_to_proposal_l2": step_l2,
        "clipped_vector_applied": False,
        "stage1": {
            "success": bool(stage1.success),
            "status": int(stage1.status),
            "message": str(stage1.message),
            "iterations": int(stage1.nit),
            "z_scaled": float(stage1.x[-1]),
            "z_solver_raw": stage1_z_solver_raw,
            "z_certified_raw": z_star,
            "z_certification_adjustment": stage1_z_certification_adjustment,
            "raw_certification_checks": stage1_certification_checks,
        },
        "stage2": {
            "success": bool(stage2.success),
            "status": int(stage2.status),
            "message": str(stage2.message),
            "iterations": int(stage2.nit),
        },
    }


def decide_transition(
    *,
    phi_current_expanded: float,
    phi_candidate: float,
    predicted_phi: float,
    changed_model: bool,
    same_bf16_output: bool,
    semantic_changed: bool,
    plateau_already_used: bool,
) -> dict[str, Any]:
    strict_actual_improvement = phi_candidate > phi_current_expanded
    strict_predicted_improvement = predicted_phi > phi_current_expanded
    plateau_allowed = bool(
        not strict_actual_improvement
        and strict_predicted_improvement
        and changed_model
        and same_bf16_output
        and not semantic_changed
        and not plateau_already_used
    )
    if strict_actual_improvement:
        action = "accept_strict_phi_improvement"
    elif plateau_allowed:
        action = "move_once_same_BF16_plateau_and_shrink"
    else:
        action = "reject_and_shrink"
    return {
        "action": action,
        "strict_actual_phi_improvement": strict_actual_improvement,
        "strict_predicted_phi_improvement": strict_predicted_improvement,
        "changed_model": changed_model,
        "same_BF16_output": same_bf16_output,
        "semantic_changed": semantic_changed,
        "plateau_already_used": plateau_already_used,
        "plateau_exception_allowed": plateau_allowed,
    }


def local_trust_math_self_test() -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    # Anchor y0 >= 0.25C starts infeasible at the 0.2C centre but remains
    # reachable inside C/8; local rows prefer increasing both coordinates.
    anchor = np.asarray([[1.0, 0.0]], dtype=np.float64)
    rhs = np.asarray([0.00025], dtype=np.float64)
    local = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    residual = np.asarray([-0.0003, -0.0004], dtype=np.float64)
    center = np.asarray([0.0002, 0.0], dtype=np.float64)
    proposal, audit = solve_local_trust_maximin(
        anchor,
        rhs,
        local,
        residual,
        center,
        TRUST_RADII[0],
        np,
        optimize,
    )
    scaled_proposal, scaled_audit = solve_local_trust_maximin(
        7.0 * anchor,
        7.0 * rhs,
        13.0 * local,
        13.0 * residual,
        center,
        TRUST_RADII[0],
        np,
        optimize,
    )
    checks = {
        "solver_pass": audit["pass"] is True,
        "cap_exact": float(np.linalg.norm(proposal)) <= TOTAL_L2_CAP + L2_ABS_TOL,
        "trust_exact": float(np.linalg.norm(proposal - center))
        <= TRUST_RADII[0] + L2_ABS_TOL,
        "anchor_hard": float((anchor @ proposal - rhs).min())
        >= -LINEAR_RESIDUAL_TOL,
        "infeasible_center_reaches_hard_anchor_inside_trust": float(
            (anchor @ center - rhs).min()
        )
        < 0.0
        and float((anchor @ proposal - rhs).min()) >= -LINEAR_RESIDUAL_TOL,
        "predicted_phi_improves": audit["predicted_phi"] > float(residual.min()),
        "z_nonpositive": audit["z_star"] <= 0.0,
        "no_clip": audit["clipped_vector_applied"] is False,
        "historical_zero": audit["historical_candidate_active_count"] == 0,
        "single_global_local_scale_preserves_max_min": audit["scaling"][
            "checks"
        ]["global_scale_preserves_raw_max_min_objective"],
        "positive_constraint_scale_invariant_proposal": float(
            np.linalg.norm(scaled_proposal - proposal)
        )
        <= 1e-15,
        "global_local_scale_invariant_z_units": math.isclose(
            float(scaled_audit["z_star"]),
            13.0 * float(audit["z_star"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "scaled_witness_raw_direct_gates_pass": scaled_audit["pass"] is True
        and all(scaled_audit["checks"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 local trust math self-test failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "audit": audit,
        "positive_constraint_scale_invariance_audit": scaled_audit,
    }


def curvature_and_negative_self_test() -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    # Pure scalar witness: f(x)=x^2-0.09 >= 0 has feasible x=0.5 within cap 1.
    # Tangents at -0.5 and +0.5 demand x<=-0.34 and x>=+0.34.  Adding the
    # deliberately curved second local witness below makes the old bundle
    # inconsistent while latest-local replacement remains feasible.  This
    # checks ledger semantics, not the live neural model.
    old_bundle = [
        {"point": -0.5, "sense": "le", "bound": -0.34},
        {"point": +0.5, "sense": "ge", "bound": +0.34},
    ]
    old_bundle_feasible = old_bundle[1]["bound"] <= old_bundle[0]["bound"]
    latest_local = old_bundle[-1]
    latest_local_feasible = latest_local["bound"] <= 1.0
    first = decide_transition(
        phi_current_expanded=-1.0,
        phi_candidate=-1.0,
        predicted_phi=-0.5,
        changed_model=True,
        same_bf16_output=True,
        semantic_changed=False,
        plateau_already_used=False,
    )
    second = decide_transition(
        phi_current_expanded=-1.0,
        phi_candidate=-1.0,
        predicted_phi=-0.4,
        changed_model=True,
        same_bf16_output=True,
        semantic_changed=False,
        plateau_already_used=True,
    )
    strict = decide_transition(
        phi_current_expanded=-1.0,
        phi_candidate=-0.75,
        predicted_phi=-0.8,
        changed_model=True,
        same_bf16_output=False,
        semantic_changed=True,
        plateau_already_used=False,
    )
    hard_anchor_infeasible_fails_closed = False
    try:
        solve_local_trust_maximin(
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([0.0005], dtype=np.float64),
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([-0.0005], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            TRUST_RADII[0],
            np,
            optimize,
        )
    except RuntimeError:
        hard_anchor_infeasible_fails_closed = True
    checks = {
        "append_only_curved_bundle_infeasible": old_bundle_feasible is False,
        "latest_local_replacement_feasible": latest_local_feasible is True,
        "first_plateau_allowed_once": first["action"]
        == "move_once_same_BF16_plateau_and_shrink",
        "second_plateau_rejected": second["action"] == "reject_and_shrink",
        "strict_phi_improvement_accepts_even_with_new_semantic": strict["action"]
        == "accept_strict_phi_improvement",
        "equal_phi_without_plateau_never_accepts": decide_transition(
            phi_current_expanded=-1.0,
            phi_candidate=-1.0,
            predicted_phi=-1.0,
            changed_model=True,
            same_bf16_output=False,
            semantic_changed=False,
            plateau_already_used=False,
        )["action"]
        == "reject_and_shrink",
        "trust_schedule_exact_no_expand": TRUST_RADIUS_DIVISORS == (8, 16, 32, 64)
        and all(TRUST_RADII[index + 1] == TRUST_RADII[index] / 2.0 for index in range(3)),
        "budget_exact12": MAX_OFFICIAL_EVALUATIONS == 12,
        "cap_exact": TOTAL_L2_CAP == 0.001,
        "hard_anchor_outside_C_div_8_fails_closed":
        hard_anchor_infeasible_fails_closed,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 curvature/negative self-test failed: {checks}")
    return {"checks": checks, "pass": True}


def static_source_audit(source: bytes, cw15: ModuleType) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    imports = []
    calls = []
    function_names = set()
    cli_options = []
    radius_decrement_hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif node.module:
                imports.append(node.module)
        elif isinstance(node, ast.FunctionDef):
            function_names.add(node.name)
        elif isinstance(node, ast.Call):
            try:
                calls.append(ast.unparse(node.func))
            except Exception:
                calls.append(type(node.func).__name__)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                cli_options.append(node.args[0].value)
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "radius_index"
            and isinstance(node.op, ast.Sub)
        ):
            radius_decrement_hits.append(node.lineno)
    forbidden_import_roots = {
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "subprocess",
        "kaggle",
    }
    forbidden_imports = [
        value for value in imports if value.split(".", 1)[0] in forbidden_import_roots
    ]
    forbidden_call_fragments = (
        ".write_text",
        ".write_bytes",
        "torch.save",
        "os.remove",
        "os.unlink",
        "Path.unlink",
        "shutil.",
        "requests.",
        "urllib.",
        "socket.",
    )
    forbidden_calls = [
        value for value in calls if any(fragment in value for fragment in forbidden_call_fragments)
    ]
    required_functions = {
        "validate_cw15_closure_evidence",
        "deduplicate_active_physical",
        "solve_local_trust_maximin",
        "decide_transition",
        "run_outer_local_trust",
        "run_probe",
        "static_result",
        "main",
    }
    parent_source = CW15_PARENT.read_bytes()
    parent_source_audit = cw15.static_source_audit(parent_source)
    source_text = source.decode("utf-8")
    calls_by_function: dict[str, list[str]] = {}
    for function in (
        value for value in tree.body if isinstance(value, ast.FunctionDef)
    ):
        function_calls = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            try:
                function_calls.append(ast.unparse(node.func))
            except Exception:
                function_calls.append(type(node.func).__name__)
        calls_by_function[function.name] = function_calls
    checks = {
        "schema_literal_exact": SCHEMA.encode("ascii") in source,
        "required_functions_declared": required_functions.issubset(function_names),
        "no_network_submission_imports": not forbidden_imports,
        "no_forbidden_side_effect_calls": not forbidden_calls,
        "stdout_only_no_output_argument": "--output" not in cli_options,
        "caps_literal_exact": "TOTAL_L2_CAP = 0.001" in source_text,
        "budget_literal_exact": "MAX_OFFICIAL_EVALUATIONS = 12" in source_text,
        "trust_buckets_exact": "TRUST_RADIUS_DIVISORS = (8, 16, 32, 64)" in source_text,
        "no_radius_expansion_path": "TRUST_RADII[radius_index + 1]" in source_text
        and not radius_decrement_hits,
        "bootstrap_exception_named_exact":
        "bootstrap_seed_nonacceptance_exempt_once" in source_text,
        "bootstrap_ineligible_named_exact": '"eligible": False' in source_text,
        "historical_candidate_active_zero_named":
        "historical_candidate_active_count" in source_text,
        "strict_phi_comparison_present": "phi_candidate > phi_current_expanded"
        in source_text,
        "no_clip_contract_present": '"clipped_vector_applied": False' in source_text,
        "exact_two_candidate_evaluator_callsites_bootstrap_and_loop":
        calls_by_function.get("run_outer_local_trust", []).count(
            "evaluate_candidate_once"
        )
        == 2,
        "candidate_evaluator_owns_exact_one_full_six_callsite":
        calls_by_function.get("evaluate_candidate_once", []).count(
            "cw15.snapshot_official_six_views"
        )
        == 1,
        "physical_dedup_complete_four_metrics_only":
        "exact_four_to_one = all(exact_four_checks.values())" in source_text,
        "physical_dedup_threshold_margin_gradient_all_identical": all(
            value in source_text
            for value in (
                '"thresholds_bit_equal"',
                '"margins_bit_equal"',
                '"gradients_hash_equal"',
            )
        ),
        "physical_partial_or_nonidentical_preserved_rowwise":
        'reason = f"partial_{len(values)}_semantic_group_keep_all"'
        in source_text
        and 'reason = "four_semantics_not_all_exact_keep_all"' in source_text,
        "physical_mapping_bijective": '"semantic_mapping_bijective"'
        in source_text,
        "all_semantic_gradients_materialized_before_dedup": source_text.index(
            "semantic_records, linearization ="
        )
        < source_text.index(
            "records, dedup = deduplicate_active_physical(\n        cw15, active"
        ),
        "exact_self_lock_replaces_inherited_parent_self":
        'input_lock["self"] = dict(self_evidence)' in source_text,
        "parent_source_audit_pass": parent_source_audit["pass"] is True,
        "parent_source_sha_exact": hashlib.sha256(parent_source).hexdigest()
        == CW15_PARENT_SHA256,
        "threshold_functions_byte_exact_parent": parent_source_audit[
            "threshold_function_shas"
        ]
        == cw15.CW13_THRESHOLD_FUNCTION_SHAS,
        "terminal10_imported_not_redefined": "terminal_acceptance" not in function_names
        and "cw15.terminal_acceptance(" in source_text,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "forbidden_imports": forbidden_imports,
        "forbidden_calls": forbidden_calls,
        "cli_options": cli_options,
        "radius_decrement_hits": radius_decrement_hits,
        "declared_functions": sorted(function_names),
        "calls_by_function": calls_by_function,
        "threshold_function_shas": parent_source_audit["threshold_function_shas"],
        "parent_source_audit_sha256": canonical_sha(parent_source_audit),
    }


def capture_snapshot_cache(
    cw15: ModuleType,
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: Mapping[str, Any],
    torch: Any,
    *,
    model_state_sha256: str,
) -> dict[str, Any]:
    official: dict[str, Mapping[str, Any]] = {}
    batch_count = 0
    for view in cw15.VIEW_ORDER:
        for batch in streams[view]:
            snapshot = batch.get("candidate_snapshot")
            if snapshot is None:
                raise RuntimeError("CW16 cannot cache a missing official snapshot")
            context_hash = str(batch["context"]["context_hash"])
            prior = official.get(context_hash)
            if prior is not None and prior is not snapshot:
                raise RuntimeError("CW16 duplicate context hash aliases different snapshots")
            official[context_hash] = snapshot
            batch_count += 1
    legacy_snapshot = legacy.get("candidate_snapshot")
    if legacy_snapshot is None:
        raise RuntimeError("CW16 cannot cache a missing legacy snapshot")
    fingerprint = cw15.candidate_output_state_fingerprint(streams, legacy, torch)
    return {
        "model_state_sha256": model_state_sha256,
        "output_state_sha256": fingerprint["sha256"],
        "output_fingerprint": fingerprint,
        "official": official,
        "legacy": legacy_snapshot,
        "official_unique_context_count": len(official),
        "official_batch_count": batch_count,
        "context_identity_sha256": canonical_sha(sorted(official)),
    }


def snapshot_cache_audit(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_state_sha256": cache["model_state_sha256"],
        "output_state_sha256": cache["output_state_sha256"],
        "official_unique_context_count": cache["official_unique_context_count"],
        "official_batch_count": cache["official_batch_count"],
        "context_identity_sha256": cache["context_identity_sha256"],
    }


def install_snapshot_cache(
    cw15: ModuleType,
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: dict[str, Any],
    cache: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    installed = 0
    for view in cw15.VIEW_ORDER:
        for batch in streams[view]:
            context_hash = str(batch["context"]["context_hash"])
            batch["candidate_snapshot"] = cache["official"][context_hash]
            installed += 1
    legacy["candidate_snapshot"] = cache["legacy"]
    fingerprint = cw15.candidate_output_state_fingerprint(streams, legacy, torch)
    checks = {
        "official_batch_count_exact": installed == cache["official_batch_count"],
        "output_fingerprint_exact": fingerprint["sha256"]
        == cache["output_state_sha256"],
        "native_bf16_exact": fingerprint["native_bf16_exact"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 cached snapshot installation drift: {checks}")
    return {"checks": checks, "pass": True, "output_fingerprint": fingerprint}


def cached_active_cut_gates(
    cw15: ModuleType,
    active: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    records = []
    for cut in active:
        if cut["context_type"] == "legacy_B33":
            snapshot = cache["legacy"]
        else:
            snapshot = cache["official"][str(cut["context_hash"])]
        margin = cw15.pair_margin(
            snapshot,
            int(cut["offset_zero_based"]),
            int(cut["positive_option"]),
            int(cut["negative_option"]),
        )
        threshold = float(cut["threshold"])
        residual = margin - threshold
        records.append(
            {
                "identity": list(cw15.cut_identity(cut)),
                "context_type": cut["context_type"],
                "margin": margin,
                "threshold": threshold,
                "residual": residual,
                "pass": margin + tolerance >= threshold,
            }
        )
    if not records:
        raise RuntimeError("CW16 cached gate ledger unexpectedly empty")
    return {
        "count": len(records),
        "violated_count": sum(int(not value["pass"]) for value in records),
        "residual_min": min(float(value["residual"]) for value in records),
        "pass": all(bool(value["pass"]) for value in records),
        "records": records,
        "cache": snapshot_cache_audit(cache),
    }


def restore_current_center(
    cw15: ModuleType,
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: dict[str, Any],
    parameters: Sequence[Any],
    raw_actor: Sequence[Any],
    cw11_total: Any,
    additional: Any,
    cache: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    total = cw15.apply_additional_from_cw11(
        modules,
        parameters,
        raw_actor,
        cw11_total,
        additional,
        context["helper"].torch,
    )
    snapshot_install = install_snapshot_cache(
        cw15, streams, legacy, cache, context["helper"].torch
    )
    model_sha = context["helper"].model_state_sha256(context["model"].state_dict())
    checks = {
        "model_sha_exact_cache": model_sha == cache["model_state_sha256"],
        "output_sha_exact_cache": snapshot_install["output_fingerprint"]["sha256"]
        == cache["output_state_sha256"],
        "model_eval_exact": context["model"].training is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 current-center restore drift: {checks}")
    return total, {"checks": checks, "pass": True, "snapshot_install": snapshot_install}


def anchor_requirement_key(cw15: ModuleType, cut: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        *cw15.cut_identity(cut),
        cw15.CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256,
        cw15.CW11_MODEL_SHA256,
        cw15.float64_scalar_sha256(float(cut["threshold"])),
    )


def ensure_current_anchor_rows(
    cw15: ModuleType,
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: dict[str, Any],
    parameters: Sequence[Any],
    raw_actor: Sequence[Any],
    cw11_total: Any,
    active: Sequence[Mapping[str, Any]],
    anchor_archive: list[dict[str, Any]],
) -> dict[str, Any]:
    restore = cw15.restore_cw11_anchor_for_total_qp(
        context,
        modules,
        streams,
        legacy,
        parameters,
        raw_actor,
        cw11_total,
    )
    anchor_fingerprint = cw15.candidate_output_state_fingerprint(
        streams, legacy, context["helper"].torch
    )
    gates = cw15.active_cut_gates(
        active,
        streams,
        legacy,
        modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
    )
    margin_by_identity = {
        tuple(value["identity"]): float(value["margin"])
        for value in gates["records"]
    }
    present = {cw15.affine_tangent_identity(value) for value in anchor_archive}
    missing = [
        value for value in active if anchor_requirement_key(cw15, value) not in present
    ]
    if missing:
        records, linearization = cw15.linearize_semantic_cuts_at_current_point(
            context,
            modules,
            streams,
            legacy,
            missing,
            parameters,
            __import__("numpy").zeros(cw15.ACTOR6_FLAT_LENGTH, dtype="float64"),
            linearization_model_state_sha256=cw15.CW11_MODEL_SHA256,
            linearization_output_state_sha256=anchor_fingerprint["sha256"],
            linearization_kind="CW11_anchor",
            expected_gate_margins={
                cw15.cut_identity(value): margin_by_identity[cw15.cut_identity(value)]
                for value in missing
            },
            require_actual_violation=False,
        )
        merge = cw15.merge_affine_tangents(anchor_archive, records)
    else:
        linearization = {
            "pass": True,
            "semantic_cut_count": 0,
            "reason": "all_current_semantics_already_have_exact_threshold_anchor",
        }
        merge = {
            "added": [],
            "repeated": [],
            "added_count": 0,
            "repeated_count": 0,
            "affine_count": len(anchor_archive),
            "affine_ledger_sha256": canonical_sha(
                [
                    cw15.sanitized_affine_tangent(value)
                    for value in sorted(anchor_archive, key=cw15.affine_tangent_identity)
                ]
            ),
            "strict_growth": False,
        }
    coverage = {
        anchor_requirement_key(cw15, value) for value in active
    }.issubset({cw15.affine_tangent_identity(value) for value in anchor_archive})
    checks = {
        "restore_exact_CW11": restore["pass"] is True,
        "anchor_output_exact_CW11": anchor_fingerprint["sha256"]
        == cw15.candidate_output_state_fingerprint(streams, legacy, context["helper"].torch)[
            "sha256"
        ],
        "missing_identity_unique": len({cw15.cut_identity(value) for value in missing})
        == len(missing),
        "new_anchor_one_to_one": merge["added_count"] == len(missing),
        "new_anchor_no_repeat": merge["repeated_count"] == 0,
        "all_current_semantics_have_hard_anchor": coverage,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 hard-anchor coverage drift: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "restore": restore,
        "anchor_output_fingerprint": anchor_fingerprint,
        "anchor_gates": gates,
        "missing_semantic_count": len(missing),
        "linearization": linearization,
        "merge": merge,
    }


def select_current_hard_anchor_rows(
    cw15: ModuleType,
    active: Sequence[Mapping[str, Any]],
    anchor_archive: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    by_key = {
        cw15.affine_tangent_identity(value): value for value in anchor_archive
    }
    required = [anchor_requirement_key(cw15, value) for value in active]
    if len(set(required)) != len(required) or any(value not in by_key for value in required):
        raise RuntimeError("CW16 current hard-anchor selection coverage drift")
    selected = [by_key[value] for value in required]
    selected.sort(key=cw15.affine_tangent_identity)
    checks = {
        "one_anchor_per_current_semantic": len(selected) == len(active),
        "all_CW11_anchor_kind": all(
            value["linearization_kind"] == "CW11_anchor" for value in selected
        ),
        "identity_unique": len(
            {cw15.affine_tangent_identity(value) for value in selected}
        )
        == len(selected),
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 selected hard-anchor rows drift: {checks}")
    return selected, {
        "checks": checks,
        "pass": True,
        "active_anchor_count": len(selected),
        "archive_anchor_count": len(anchor_archive),
        "active_anchor_ledger_sha256": canonical_sha(
            [cw15.sanitized_affine_tangent(value) for value in selected]
        ),
    }


def linearize_current_physical_rows(
    cw15: ModuleType,
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: Mapping[str, Any],
    parameters: Sequence[Any],
    current_additional: Any,
    current_model_sha: str,
    current_output_sha: str,
    active: Sequence[Mapping[str, Any]],
    current_gates: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gate_by_identity = {
        tuple(value["identity"]): float(value["margin"])
        for value in current_gates["records"]
    }
    semantic_records, linearization = cw15.linearize_semantic_cuts_at_current_point(
        context,
        modules,
        streams,
        legacy,
        active,
        parameters,
        current_additional,
        linearization_model_state_sha256=current_model_sha,
        linearization_output_state_sha256=current_output_sha,
        linearization_kind="CW16_current_local_physical",
        expected_gate_margins={
            cw15.cut_identity(value): gate_by_identity[cw15.cut_identity(value)]
            for value in active
        },
        require_actual_violation=False,
    )
    records, dedup = deduplicate_active_physical(
        cw15, active, current_gates, semantic_records
    )
    checks = {
        "dedup_pass": dedup["pass"] is True,
        "linearization_pass": linearization["pass"] is True,
        "all_semantic_gradients_materialized_before_dedup":
        len(semantic_records) == len(active),
        "one_gradient_per_optimization_row": len(records)
        == dedup["optimization_row_count"],
        "strict_four_to_one_accounting": dedup["physical_count"]
        == dedup["semantic_count"] - 3 * dedup["exact_four_to_one_group_count"],
        "partial_or_nonidentical_groups_preserved_rowwise": all(
            value["retained_optimization_count"]
            == (1 if value["exact_four_to_one"] else value["semantic_count"])
            for value in dedup["mappings"]
        ),
        "all_current_point_kind": all(
            value["linearization_kind"] == "CW16_current_local_physical"
            for value in semantic_records
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 current physical tangent drift: {checks}")
    return records, {
        "checks": checks,
        "pass": True,
        "deduplication": dedup,
        "linearization": linearization,
        "all_semantic_sanitized_local_ledger_sha256": canonical_sha(
            [
                cw15.sanitized_affine_tangent(value)
                for value in sorted(
                    semantic_records, key=cw15.affine_tangent_identity
                )
            ]
        ),
        "sanitized_local_ledger_sha256": canonical_sha(
            [
                cw15.sanitized_affine_tangent(value)
                for value in sorted(records, key=cw15.affine_tangent_identity)
            ]
        ),
    }


def evaluate_candidate_once(
    cw15: ModuleType,
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: dict[str, Any],
    forensic: Mapping[str, Any],
    guardplan: Mapping[str, Any],
    prereg: Mapping[str, Any],
    *,
    model_state_sha256: str,
    official_iteration: int,
) -> dict[str, Any]:
    helper = context["helper"]
    model = context["model"]
    device = next(model.parameters()).device
    cw15.snapshot_legacy_B33(context, modules, legacy, "candidate")
    legacy_gate = cw15.legacy_B33_gate(context, modules, legacy)
    six_view = cw15.snapshot_official_six_views(
        helper,
        modules["ram"],
        model,
        streams,
        device,
        phase="candidate",
        model_sha256=model_state_sha256,
    )
    output_fingerprint = cw15.candidate_output_state_fingerprint(
        streams, legacy, helper.torch
    )
    oracle = cw15.full_stream_separation_oracle(
        helper,
        streams,
        forensic,
        guardplan,
        prereg,
        iteration=official_iteration,
    )
    fixed = cw15.fixed_repair_gates_and_cuts(
        helper, streams, iteration=official_iteration
    )
    cache = capture_snapshot_cache(
        cw15,
        streams,
        legacy,
        helper.torch,
        model_state_sha256=model_state_sha256,
    )
    checks = {
        "single_full_six_call_owned_here": True,
        "model_sha_exact": six_view["model_state_sha256"] == model_state_sha256,
        "output_sha_exact_cache": output_fingerprint["sha256"]
        == cache["output_state_sha256"],
        "native_bf16": output_fingerprint["native_bf16_exact"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW16 single candidate evaluation drift: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "legacy_gate": legacy_gate,
        "six_view_snapshot": six_view,
        "output_fingerprint": output_fingerprint,
        "oracle": oracle,
        "fixed": fixed,
        "cache": cache,
    }


def deterministic_additions(
    cw15: ModuleType,
    context: Mapping[str, Any],
    legacy: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    iteration: int,
) -> list[dict[str, Any]]:
    legacy_additions = cw15.legacy_dynamic_cuts(
        context,
        legacy,
        evaluation["legacy_gate"]["gate"]["false_obligations"],
        iteration,
    )
    return sorted(
        [
            *evaluation["oracle"]["deterministic_new_harm_cuts"],
            *evaluation["oracle"]["deterministic_restoration_cuts"],
            *evaluation["fixed"]["deterministic_repair_cuts"],
            *legacy_additions,
        ],
        key=cw15.cut_identity,
    )


def terminal_integrity(
    cw15: ModuleType,
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    parameters: Sequence[Any],
    layout: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    subproblem: Mapping[str, Any],
    *,
    official_evaluation_count: int,
    evaluated_model_count: int,
    physical_dedup_pass: bool,
    semantic_monotone_pass: bool,
    post_merge_anchor_coverage_pass: bool,
) -> dict[str, bool]:
    return {
        "actor6_trainable_scope_exact": tuple(
            name for name, value in context["model"].named_parameters() if value.requires_grad
        )
        == cw15.ACTOR6_NAMES,
        "actor6_layout_exact": cw15.actor6_layout(parameters) == list(layout),
        "nonactor_74_exact_raw": cw15.nonactor_sha256(context)
        == cw15.RAW_NONACTOR_SHA256,
        "count_value_logits_exact_raw": bool(
            evaluation["oracle"]["hard_checks"][
                "count_value_logits_exact_raw_all_batches"
            ]
        ),
        "native_BF16_six_outputs": evaluation["output_fingerprint"][
            "native_bf16_exact"
        ],
        "hard_CW11_anchor_residual": bool(
            subproblem["checks"]["anchor_hard_residual"]
        ),
        "current_local_residual_fixed_z": bool(
            subproblem["checks"]["current_local_fixed_z_residual"]
        ),
        "additional_total_L2_cap": bool(subproblem["checks"]["total_cap_hard"]),
        "local_trust_L2_cap": bool(subproblem["checks"]["trust_cap_hard"]),
        "uncapped_solution_only_never_clip": subproblem[
            "clipped_vector_applied"
        ]
        is False,
        "historical_candidate_tangent_active_count_zero": subproblem[
            "historical_candidate_active_count"
        ]
        == 0,
        "current_physical_four_to_one_dedup_exact": physical_dedup_pass,
        "semantic_ledger_monotone": semantic_monotone_pass,
        "all_post_merge_semantics_have_hard_CW11_anchor":
        post_merge_anchor_coverage_pass,
        "one_full_six_per_unique_proposal": official_evaluation_count
        == evaluated_model_count,
        "official_budget_at_most_12": official_evaluation_count
        <= MAX_OFFICIAL_EVALUATIONS,
        "no_checkpoint_written_eval_only_13key_deferred": True,
    }


def run_outer_local_trust(
    cw15: ModuleType,
    cw15_evidence: Mapping[str, Any],
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    *,
    user_candidate_consumer: Any = None,
) -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    helper = context["helper"]
    torch = helper.torch
    model = context["model"]
    device = next(model.parameters()).device
    forensic = cw15.read_json_locked(cw15.FORENSIC, cw15.FORENSIC_SHA256, 0o444)
    guardplan = cw15.read_json_locked(cw15.GUARDPLAN, cw15.GUARDPLAN_SHA256, 0o444)
    prereg = cw15.read_json_locked(cw15.PREREG, cw15.PREREG_SHA256, 0o444)
    required_context = {
        "model",
        "checkpoint",
        "helper",
        "model_config",
        "raw_actor",
        "raw_model_state_sha256",
        "raw_nonactor_sha256",
        "terminal_cumulative_float64",
        "terminal_cumulative_float64_le_sha256",
        "selected_row_gate",
        "active_pair_ledger",
        "expanded_row_count",
        "candidate_model_state_sha256",
    }
    if not required_context.issubset(context):
        raise RuntimeError("CW16 incoming CW11 consumer schema drift")
    incoming_checks = {
        "model_sha_exact": helper.model_state_sha256(model.state_dict())
        == cw15.CW11_MODEL_SHA256
        == str(context["candidate_model_state_sha256"]),
        "model_eval_exact": model.training is False,
        "raw_sha_exact": str(context["raw_model_state_sha256"])
        == cw15.RAW_MODEL_SHA256,
        "nonactor_sha_exact": str(context["raw_nonactor_sha256"])
        == cw15.RAW_NONACTOR_SHA256
        and cw15.nonactor_sha256(context) == cw15.RAW_NONACTOR_SHA256,
        "cw11_vector_sha_exact": str(context["terminal_cumulative_float64_le_sha256"])
        == cw15.CW11_VECTOR_SHA256,
        "cw11_vector_l2_exact": math.isclose(
            float(np.linalg.norm(context["terminal_cumulative_float64"])),
            cw15.CW11_TOTAL_FROM_RAW_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "legacy_33_34_exact": int(context["expanded_row_count"])
        == cw15.EXPECTED_LEGACY_ROWS
        and len(context["active_pair_ledger"])
        == cw15.EXPECTED_LEGACY_ACTIVE_PAIRS,
        "legacy_gate_pass": bool(context["selected_row_gate"]["pass"]),
    }
    if not all(incoming_checks.values()):
        raise RuntimeError(f"CW16 incoming CW11 binding failed: {incoming_checks}")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    parameters = modules["geometry"].configure_actor6(model)
    layout = cw15.actor6_layout(parameters)
    raw_actor = context["raw_actor"]
    if len(raw_actor) != len(parameters):
        raise RuntimeError("CW16 raw actor tensor cardinality drift")
    cw11_total = np.asarray(
        context["terminal_cumulative_float64"], dtype=np.float64
    ).copy()
    if cw11_total.shape != (cw15.ACTOR6_FLAT_LENGTH,):
        raise RuntimeError("CW16 CW11 total vector shape drift")

    materialized_model, materialized_config, _, materialized_kind = (
        helper.evaluator.load_policy(cw15.CW11_CHECKPOINT, device)
    )
    materialized_checks = {
        "kind_ppo": materialized_kind == "ppo",
        "model_eval_exact": materialized_model.training is False,
        "config_exact": materialized_config == context["model_config"],
        "model_sha_exact": helper.model_state_sha256(materialized_model.state_dict())
        == cw15.CW11_MODEL_SHA256,
        "state_tensors_exact_live": set(materialized_model.state_dict())
        == set(model.state_dict())
        and all(
            torch.equal(materialized_model.state_dict()[name], model.state_dict()[name])
            for name in model.state_dict()
        ),
    }
    del materialized_model
    torch.cuda.empty_cache()
    if not all(materialized_checks.values()):
        raise RuntimeError(f"CW16 materialized/live CW11 mismatch: {materialized_checks}")

    legacy = cw15.load_legacy_B33(context, modules)
    streams, stream_loading_audit = cw15.load_official_six_streams(
        helper, modules["legacy"], context["model_config"], forensic, guardplan
    )
    modules["cutting"].restore_raw_actor(
        modules["ram"], parameters, raw_actor, torch
    )
    if helper.model_state_sha256(model.state_dict()) != cw15.RAW_MODEL_SHA256:
        raise RuntimeError("CW16 temporary raw restoration failed")
    cw15.snapshot_legacy_B33(context, modules, legacy, "raw")
    raw_six = cw15.snapshot_official_six_views(
        helper,
        modules["ram"],
        model,
        streams,
        device,
        phase="raw",
        model_sha256=cw15.RAW_MODEL_SHA256,
    )
    raw_reproduction = cw15.raw_metric_reproduction(raw_six, forensic)
    if not all(raw_reproduction.values()):
        raise RuntimeError(f"CW16 raw official reproduction failed: {raw_reproduction}")

    modules["cutting"].apply_cumulative_from_raw(
        modules["ram"], parameters, raw_actor, cw11_total, torch
    )
    if helper.model_state_sha256(model.state_dict()) != cw15.CW11_MODEL_SHA256:
        raise RuntimeError("CW16 CW11 reference reapplication failed")
    cw11_actor_bytes = cw15.actor_float32_le_bytes(parameters, np)
    cw15.snapshot_legacy_B33(context, modules, legacy, "reference")
    legacy["candidate_snapshot"] = legacy["reference_snapshot"]
    reference_six = cw15.snapshot_official_six_views(
        helper,
        modules["ram"],
        model,
        streams,
        device,
        phase="reference",
        model_sha256=cw15.CW11_MODEL_SHA256,
    )
    for view in cw15.VIEW_ORDER:
        for batch in streams[view]:
            batch["candidate_snapshot"] = batch["reference_snapshot"]
    reference_legacy_gate = cw15.legacy_B33_gate(context, modules, legacy)
    initial_oracle = cw15.full_stream_separation_oracle(
        helper, streams, forensic, guardplan, prereg, iteration=0
    )
    initial_fixed = cw15.fixed_repair_gates_and_cuts(helper, streams, iteration=0)
    initial_checks = {
        "cw11_failed_exact_51_of_60": initial_oracle["authoritative_60_gates"][
            "passed_gate_count"
        ]
        == 51,
        "cw11_has_no_self_new_harm": initial_oracle["new_harm_count_vs_cw11"] == 0,
        "cw11_favorable_transitions_retained": initial_oracle[
            "transition_state_checks"
        ]["favorable_transitions_retained"],
        "cw11_has_no_restoration_cuts": initial_oracle[
            "favorable_transition_break_count"
        ]
        == 0
        and not initial_oracle["deterministic_restoration_cuts"],
        "semantic_total_cap_exact_0p001": TOTAL_L2_CAP
        == cw15.STEP_L2_CAP
        == cw15.ADDITIONAL_TOTAL_L2_CAP
        == 0.001,
        "fixed_repairs_initially_fail": not initial_fixed["pass"],
        "iteration0_has_no_dynamic_fixed_cuts": initial_fixed[
            "deterministic_repair_cut_count"
        ]
        == 0,
        "legacy_reference_pass": reference_legacy_gate["pass"],
    }
    if not all(initial_checks.values()):
        raise RuntimeError(f"CW16 initial-state audit failed: {initial_checks}")

    active: list[dict[str, Any]] = []
    legacy_merge = cw15.merge_active_cuts(active, cw15.legacy_pair_cuts(legacy))
    fixed_merge = cw15.merge_active_cuts(
        active, cw15.initial_official_repair_cuts(streams)
    )
    if (
        legacy_merge["active_count"] != 34
        or fixed_merge["active_count"] != 38
        or len({cw15.cut_identity(value) for value in active}) != 38
    ):
        raise RuntimeError("CW16 initial 34+4 semantic ledger drift")
    initial_active_gates = cw15.active_cut_gates(
        active,
        streams,
        legacy,
        modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
    )
    zero_additional = np.zeros(cw15.ACTOR6_FLAT_LENGTH, dtype=np.float64)
    initial_output_fingerprint = cw15.candidate_output_state_fingerprint(
        streams, legacy, torch
    )
    cw11_cache = capture_snapshot_cache(
        cw15,
        streams,
        legacy,
        torch,
        model_state_sha256=cw15.CW11_MODEL_SHA256,
    )
    if not (
        initial_active_gates["violated_count"] == 4
        and cw15.float64_vector_sha256(zero_additional)
        == cw15.CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256
        and canonical_sha([list(cw15.cut_identity(value)) for value in active])
        == CW15_INITIAL_ACTIVE_SHA256
    ):
        raise RuntimeError("CW16 initial CW11 semantic/point identity drift")

    iterations: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "kind": "exact_CW11_reference_before_bootstrap",
            "model_state_sha256": cw15.CW11_MODEL_SHA256,
            "output_state_fingerprint": initial_output_fingerprint,
            "additional_l2": 0.0,
            "additional_float64_le_sha256":
            cw15.CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256,
            "total_from_raw_l2": float(np.linalg.norm(cw11_total)),
            "total_from_raw_float64_le_sha256":
            cw15.CW11_VECTOR_SHA256,
            "active_cut_count": len(active),
            "active_cut_identity_sha256": CW15_INITIAL_ACTIVE_SHA256,
            "legacy_gate": reference_legacy_gate,
            "six_view_snapshot": reference_six,
            "oracle": initial_oracle,
            "fixed_repair_gates": initial_fixed,
            "active_cut_gates": initial_active_gates,
            "snapshot_cache": snapshot_cache_audit(cw11_cache),
        }
    ]

    # Exact CW15 iteration-1 replay is the sole bootstrap exception.  It is
    # counted as official evaluation 1 but is not an acceptance or an eligible
    # terminal candidate.
    anchor_archive: list[dict[str, Any]] = []
    initial_anchor_growth = ensure_current_anchor_rows(
        cw15,
        context,
        modules,
        streams,
        legacy,
        parameters,
        raw_actor,
        cw11_total,
        active,
        anchor_archive,
    )
    initial_anchor_rows, initial_anchor_selection = select_current_hard_anchor_rows(
        cw15, active, anchor_archive
    )
    replay_gradients, replay_rhs, replay_affine_audit = cw15.affine_qp_arrays(
        initial_anchor_rows
    )
    replay_additional, replay_qp = modules["cutting"].solve_minimum_l2_correction(
        replay_gradients, replay_rhs, np, optimize
    )
    replay_additional = np.asarray(replay_additional, dtype=np.float64)
    replay_checks_pre_oracle = {
        "anchor_count_exact38": len(initial_anchor_rows) == 38,
        "anchor_ledger_sha_exact": replay_affine_audit["affine_ledger_sha256"]
        == CW15_ITERATION1_ANCHOR_LEDGER_SHA256,
        "anchor_order_sha_exact": replay_affine_audit[
            "ordered_affine_identity_sha256"
        ]
        == CW15_ITERATION1_ANCHOR_ORDER_SHA256,
        "QP_rank_exact37": replay_qp["svd_rank"] == 37,
        "QP_not_clipped": replay_qp["capped"] is False,
        "point_sha_exact": cw15.float64_vector_sha256(replay_additional)
        == CW15_ITERATION1_POINT_SHA256,
        "additional_l2_exact": math.isclose(
            float(np.linalg.norm(replay_additional)),
            CW15_ITERATION1_ADDITIONAL_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        ),
        "cap_exact": float(np.linalg.norm(replay_additional))
        <= TOTAL_L2_CAP + L2_ABS_TOL,
    }
    if not all(replay_checks_pre_oracle.values()):
        raise RuntimeError(
            f"CW16 bootstrap replay QP drift: {replay_checks_pre_oracle}"
        )
    replay_total = cw15.apply_additional_from_cw11(
        modules,
        parameters,
        raw_actor,
        cw11_total,
        replay_additional,
        torch,
    )
    replay_model_sha = helper.model_state_sha256(model.state_dict())
    if replay_model_sha != CW15_ITERATION1_MODEL_SHA256:
        raise RuntimeError("CW16 bootstrap replay model SHA drift before oracle")
    replay_evaluation = evaluate_candidate_once(
        cw15,
        context,
        modules,
        streams,
        legacy,
        forensic,
        guardplan,
        prereg,
        model_state_sha256=replay_model_sha,
        official_iteration=1,
    )
    official_evaluation_count = 1
    evaluated_model_shas = {replay_model_sha}
    additions = deterministic_additions(
        cw15, context, legacy, replay_evaluation, iteration=1
    )
    replay_active_before = cw15.active_cut_gates(
        active,
        streams,
        legacy,
        modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
    )
    replay_semantic_merge = cw15.merge_active_cuts(active, additions)
    replay_active_after = cw15.active_cut_gates(
        active,
        streams,
        legacy,
        modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
    )
    cw11_expanded_gates = cached_active_cut_gates(
        cw15,
        active,
        cw11_cache,
        modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
    )
    replay_checks_post_oracle = {
        "output_sha_exact": replay_evaluation["output_fingerprint"]["sha256"]
        == CW15_ITERATION1_OUTPUT_SHA256,
        "old38_all_pass": replay_active_before["count"] == 38
        and replay_active_before["violated_count"] == 0,
        "additions_exact12": len(additions) == 12,
        "additions_sha_exact": canonical_sha(
            [list(cw15.cut_identity(value)) for value in additions]
        )
        == CW15_ITERATION1_ADDITIONS_SHA256,
        "semantic_merge_exact50": replay_semantic_merge["active_count"] == 50
        and len(replay_semantic_merge["added"]) == 12
        and not replay_semantic_merge["strengthened"],
        "CW11_phi_expanded50_exact": cw11_expanded_gates["residual_min"]
        == CW11_PHI_ON_EXPANDED_50,
        "bootstrap_phi_expanded50_exact": replay_active_after["residual_min"]
        == CW15_ITERATION1_PHI_ON_EXPANDED_50,
        "bootstrap_would_not_be_strict_improvement": not (
            replay_active_after["residual_min"]
            > cw11_expanded_gates["residual_min"]
        ),
        "official_budget_count_exact1": official_evaluation_count == 1,
    }
    if not all(replay_checks_post_oracle.values()):
        raise RuntimeError(
            f"CW16 bootstrap replay oracle drift: {replay_checks_post_oracle}"
        )

    current_additional = replay_additional.copy()
    current_total = replay_total.copy()
    current_model_sha = replay_model_sha
    current_cache = replay_evaluation["cache"]
    current_output_sha = str(current_cache["output_state_sha256"])
    current_phi = float(replay_active_after["residual_min"])
    current_evaluation = replay_evaluation
    historical_candidate_archive = copy.deepcopy(
        cw15_evidence["archive"]["historical_candidate_records"]
    )
    if len(historical_candidate_archive) != 12:
        raise RuntimeError("CW16 consumed historical candidate archive drift")
    post_bootstrap_anchor_growth = ensure_current_anchor_rows(
        cw15,
        context,
        modules,
        streams,
        legacy,
        parameters,
        raw_actor,
        cw11_total,
        active,
        anchor_archive,
    )
    current_total, current_restore = restore_current_center(
        cw15,
        context,
        modules,
        streams,
        legacy,
        parameters,
        raw_actor,
        cw11_total,
        current_additional,
        current_cache,
    )
    anchor50_rows, anchor50_selection = select_current_hard_anchor_rows(
        cw15, active, anchor_archive
    )
    anchor50_sanitized_sha = canonical_sha(
        [cw15.sanitized_affine_tangent(value) for value in anchor50_rows]
    )
    if not (
        len(anchor50_rows) == 50
        and anchor50_sanitized_sha
        == cw15_evidence["archive"]["CW11_anchor_sanitized_sha256"]
    ):
        raise RuntimeError("CW16 replay hard-anchor50 differs from consumed CW15")
    anchor50_gradients, anchor50_rhs, anchor50_array_audit = cw15.affine_qp_arrays(
        anchor50_rows
    )
    anchor50_only_point, anchor50_only_qp = (
        modules["cutting"].solve_minimum_l2_correction(
            anchor50_gradients, anchor50_rhs, np, optimize
        )
    )
    anchor50_only_point = np.asarray(anchor50_only_point, dtype=np.float64)
    anchor50_only_checks = {
        "selected_row_count_exact50": len(anchor50_rows) == 50,
        "selected_kinds_all_CW11_anchor": all(
            value["linearization_kind"] == "CW11_anchor"
            for value in anchor50_rows
        ),
        "selected_anchor_sha_exact_consumed": anchor50_sanitized_sha
        == cw15_evidence["archive"]["CW11_anchor_sanitized_sha256"],
        "historical_candidate_rows_excluded": not any(
            value["linearization_kind"]
            == "post_merge_actual_active_violation"
            for value in anchor50_rows
        ),
        "array_row_count_exact50": anchor50_gradients.shape[0] == 50
        and anchor50_array_audit["affine_tangent_count"] == 50,
        "anchor_only_uncapped_not_clipped": anchor50_only_qp["capped"] is False,
        "anchor_only_minimum_l2_exact_immutable": math.isclose(
            float(anchor50_only_qp["uncapped_l2"]),
            CW14_ANCHOR50_ONLY_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        ),
        "anchor_only_point_sha_exact_immutable": cw15.float64_vector_sha256(
            anchor50_only_point
        )
        == CW14_ANCHOR50_ONLY_POINT_SHA256,
        "anchor_only_inside_total_cap": float(anchor50_only_qp["uncapped_l2"])
        <= TOTAL_L2_CAP + L2_ABS_TOL,
        "combined62_failure_not_used_as_anchor50":
        CW15_ITERATION2_UNCAPPED_L2 > TOTAL_L2_CAP
        and len(historical_candidate_archive) == 12,
    }
    if not all(anchor50_only_checks.values()):
        raise RuntimeError(
            f"CW16 anchor50-only classification drift: {anchor50_only_checks}"
        )
    anchor50_only_audit = {
        "checks": anchor50_only_checks,
        "pass": True,
        "selected_anchor_row_count": len(anchor50_rows),
        "selected_anchor_kind": "CW11_anchor",
        "selected_anchor_ledger_sha256": anchor50_sanitized_sha,
        "historical_candidate_active_count": 0,
        "historical_candidate_archive_count": len(historical_candidate_archive),
        "anchor_array_audit": anchor50_array_audit,
        "anchor_only_qp": anchor50_only_qp,
        "anchor_only_point_sha256": cw15.float64_vector_sha256(
            anchor50_only_point
        ),
        "anchor_only_l2": float(anchor50_only_qp["uncapped_l2"]),
        "consumed_combined62_uncapped_l2": CW15_ITERATION2_UNCAPPED_L2,
    }
    bootstrap_integrity = {
        "actor6_trainable_scope_exact": tuple(
            name for name, value in model.named_parameters() if value.requires_grad
        )
        == cw15.ACTOR6_NAMES,
        "actor6_layout_exact": cw15.actor6_layout(parameters) == layout,
        "nonactor_74_exact_raw": cw15.nonactor_sha256(context)
        == cw15.RAW_NONACTOR_SHA256,
        "count_value_logits_exact_raw": bool(
            replay_evaluation["oracle"]["hard_checks"][
                "count_value_logits_exact_raw_all_batches"
            ]
        ),
        "native_BF16_six_outputs": replay_evaluation["output_fingerprint"][
            "native_bf16_exact"
        ],
        "exact_CW15_replay_point_model_output_additions": all(
            replay_checks_pre_oracle.values()
        )
        and all(replay_checks_post_oracle.values()),
        "additional_total_L2_cap": float(np.linalg.norm(current_additional))
        <= TOTAL_L2_CAP + L2_ABS_TOL,
        "historical_candidate_tangent_active_count_zero": True,
        "official_budget_count_exact1": official_evaluation_count == 1,
        "no_checkpoint_written_eval_only_13key_deferred": True,
    }
    bootstrap_acceptance = cw15.terminal_acceptance(
        snapshot=replay_evaluation["six_view_snapshot"],
        oracle=replay_evaluation["oracle"],
        fixed=replay_evaluation["fixed"],
        legacy_gate=replay_evaluation["legacy_gate"],
        active_gates=replay_active_after,
        integrity=bootstrap_integrity,
    )
    if len(bootstrap_acceptance["checks"]) != 10:
        raise RuntimeError("CW16 bootstrap imported terminal gate count is not exact10")
    bootstrap_false_acceptance_keys = {
        key for key, value in bootstrap_acceptance["checks"].items() if not value
    }
    if not (
        bootstrap_acceptance["pass"] is False
        and bootstrap_false_acceptance_keys
        == {
            "active_cut_violations_zero",
            "forensic_favorable_retention_pass",
            "new_harm_count_zero",
        }
    ):
        raise RuntimeError(
            "CW16 bootstrap exact terminal10 failure set drift: "
            f"{bootstrap_false_acceptance_keys}"
        )
    bootstrap_record = {
        "iteration": 1,
        "kind": "exact_CW15_iteration1_bootstrap_seed",
        "bootstrap_seed_nonacceptance_exempt_once": True,
        "eligible": False,
        "is_acceptance": False,
        "counts_toward_official_budget": True,
        "official_evaluation_count_after": official_evaluation_count,
        "model_state_sha256": current_model_sha,
        "output_state_fingerprint": replay_evaluation["output_fingerprint"],
        "additional_l2": float(np.linalg.norm(current_additional)),
        "additional_float64_le_sha256": cw15.float64_vector_sha256(
            current_additional
        ),
        "total_from_raw_l2": float(np.linalg.norm(current_total)),
        "total_from_raw_float64_le_sha256": modules[
            "geometry"
        ].vector_sha256_float64_le(current_total, np),
        "replay_checks_pre_oracle": replay_checks_pre_oracle,
        "replay_checks_post_oracle": replay_checks_post_oracle,
        "initial_anchor_growth": initial_anchor_growth,
        "initial_anchor_selection": initial_anchor_selection,
        "replay_affine_qp_audit": replay_affine_audit,
        "replay_qp": replay_qp,
        "legacy_gate": replay_evaluation["legacy_gate"],
        "six_view_snapshot": replay_evaluation["six_view_snapshot"],
        "oracle": replay_evaluation["oracle"],
        "fixed_repair_gates": replay_evaluation["fixed"],
        "active_cut_gates_before_merge": replay_active_before,
        "post_oracle_cut_merge": replay_semantic_merge,
        "active_cut_gates": replay_active_after,
        "CW11_phi_on_expanded_ledger": cw11_expanded_gates["residual_min"],
        "bootstrap_phi_on_expanded_ledger": current_phi,
        "historical_candidate_tangent_archive_count": len(
            historical_candidate_archive
        ),
        "historical_candidate_tangent_active_count": 0,
        "post_bootstrap_anchor_growth": post_bootstrap_anchor_growth,
        "hard_anchor50_selection": anchor50_selection,
        "hard_anchor50_only_audit": anchor50_only_audit,
        "current_restore": current_restore,
        "integrity": bootstrap_integrity,
        "acceptance": bootstrap_acceptance,
        "acceptance_false_keys": sorted(bootstrap_false_acceptance_keys),
    }
    iterations.append(bootstrap_record)

    terminal_iteration = None
    terminal_payload = None
    terminal_reconstruction_audit = None
    terminal_active_gates = None
    terminal_legacy_gate = None
    terminal_oracle = None
    terminal_optimization_ledger_semantics = None
    close_reason = "maximum_12_official_evaluations_without_first_feasible"
    radius_index = 0
    plateau_used = False
    local_tangent_archive: list[dict[str, Any]] = []
    local_archive_keys: set[str] = set()
    seen_points = {
        cw15.CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256,
        CW15_ITERATION1_POINT_SHA256,
    }
    seen_models = {cw15.CW11_MODEL_SHA256, CW15_ITERATION1_MODEL_SHA256}
    seen_states: set[tuple[Any, ...]] = set()
    last_local_records: list[dict[str, Any]] = []
    last_active_anchor_rows: list[Mapping[str, Any]] = anchor50_rows
    candidate_consumer_called = False

    for proposal_index in range(2, MAX_OFFICIAL_EVALUATIONS + 1):
        if radius_index >= len(TRUST_RADII):
            close_reason = "fail_closed_trust_radius_below_cap_div_64"
            break
        trust_radius = TRUST_RADII[radius_index]
        current_total, center_restore = restore_current_center(
            cw15,
            context,
            modules,
            streams,
            legacy,
            parameters,
            raw_actor,
            cw11_total,
            current_additional,
            current_cache,
        )
        current_gates = cached_active_cut_gates(
            cw15,
            active,
            current_cache,
            modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
        )
        if current_gates["residual_min"] != current_phi:
            raise RuntimeError("CW16 current exact Phi drift before local subproblem")
        active_anchor_rows, anchor_selection = select_current_hard_anchor_rows(
            cw15, active, anchor_archive
        )
        try:
            local_records, local_audit = linearize_current_physical_rows(
                cw15,
                context,
                modules,
                streams,
                legacy,
                parameters,
                current_additional,
                current_model_sha,
                current_output_sha,
                active,
                current_gates,
            )
            anchor_gradients, anchor_rhs, anchor_array_audit = cw15.affine_qp_arrays(
                active_anchor_rows
            )
            ordered_local = sorted(local_records, key=cw15.affine_tangent_identity)
            local_gradients = np.stack(
                [
                    np.asarray(value["gradient_float64"], dtype=np.float64)
                    for value in ordered_local
                ],
                axis=0,
            )
            local_residuals = np.asarray(
                [float(value["actual_residual"]) for value in ordered_local],
                dtype=np.float64,
            )
            proposal_additional, subproblem = solve_local_trust_maximin(
                anchor_gradients,
                anchor_rhs,
                local_gradients,
                local_residuals,
                current_additional,
                trust_radius,
                np,
                optimize,
            )
        except Exception as exc:
            close_reason = (
                "fail_closed_hard_anchor_local_trust_subproblem:"
                f"{type(exc).__name__}:{exc}"
            )
            iterations.append(
                {
                    "iteration": proposal_index,
                    "kind": "fail_closed_before_official_proposal",
                    "close_reason": close_reason,
                    "official_evaluation_count": official_evaluation_count,
                    "trust_radius_index": radius_index,
                    "trust_radius": trust_radius,
                    "active_cut_count": len(active),
                    "hard_anchor_count": len(active_anchor_rows),
                    "historical_candidate_tangent_active_count": 0,
                    "center_restore": center_restore,
                    "anchor_selection": anchor_selection,
                }
            )
            break

        last_local_records = local_records
        last_active_anchor_rows = active_anchor_rows
        proposal_point_sha = cw15.float64_vector_sha256(proposal_additional)
        proposal_total = cw15.apply_additional_from_cw11(
            modules,
            parameters,
            raw_actor,
            cw11_total,
            proposal_additional,
            torch,
        )
        proposal_model_sha = helper.model_state_sha256(model.state_dict())
        pre_oracle_repeat_checks = {
            "point_new": proposal_point_sha not in seen_points,
            "model_new": proposal_model_sha not in seen_models,
            "model_changed_from_center": proposal_model_sha != current_model_sha,
            "point_changed_from_center": proposal_point_sha
            != cw15.float64_vector_sha256(current_additional),
        }
        if not all(pre_oracle_repeat_checks.values()):
            close_reason = "fail_closed_repeated_or_noninjective_point_model_before_oracle"
            iterations.append(
                {
                    "iteration": proposal_index,
                    "kind": "fail_closed_before_official_proposal",
                    "close_reason": close_reason,
                    "official_evaluation_count": official_evaluation_count,
                    "trust_radius_index": radius_index,
                    "trust_radius": trust_radius,
                    "proposal_point_sha256": proposal_point_sha,
                    "proposal_model_sha256": proposal_model_sha,
                    "pre_oracle_repeat_checks": pre_oracle_repeat_checks,
                    "subproblem": subproblem,
                    "historical_candidate_tangent_active_count": 0,
                }
            )
            break

        proposal_evaluation = evaluate_candidate_once(
            cw15,
            context,
            modules,
            streams,
            legacy,
            forensic,
            guardplan,
            prereg,
            model_state_sha256=proposal_model_sha,
            official_iteration=proposal_index,
        )
        official_evaluation_count += 1
        evaluated_model_shas.add(proposal_model_sha)
        seen_points.add(proposal_point_sha)
        seen_models.add(proposal_model_sha)
        if official_evaluation_count != proposal_index:
            raise RuntimeError("CW16 official proposal budget/index drift")

        active_before_count = len(active)
        active_before_thresholds = {
            cw15.cut_identity(value): float(value["threshold"]) for value in active
        }
        active_gates_before_merge = cw15.active_cut_gates(
            active,
            streams,
            legacy,
            modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
        )
        additions = deterministic_additions(
            cw15, context, legacy, proposal_evaluation, iteration=proposal_index
        )
        semantic_merge = cw15.merge_active_cuts(active, additions)
        post_merge_gates = cw15.active_cut_gates(
            active,
            streams,
            legacy,
            modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
        )
        current_expanded_gates = cached_active_cut_gates(
            cw15,
            active,
            current_cache,
            modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
        )
        active_after_thresholds = {
            cw15.cut_identity(value): float(value["threshold"]) for value in active
        }
        semantic_monotone_checks = {
            "count_nondecreasing": len(active) >= active_before_count,
            "old_identity_subset": set(active_before_thresholds).issubset(
                active_after_thresholds
            ),
            "thresholds_nondecreasing": all(
                active_after_thresholds[key] >= value
                for key, value in active_before_thresholds.items()
            ),
            "merge_count_exact": semantic_merge["active_count"] == len(active),
        }
        if not all(semantic_monotone_checks.values()):
            raise RuntimeError(
                f"CW16 semantic monotonicity drift: {semantic_monotone_checks}"
            )
        semantic_changed = bool(
            semantic_merge["added"] or semantic_merge["strengthened"]
        )
        phi_current_expanded = float(current_expanded_gates["residual_min"])
        phi_candidate = float(post_merge_gates["residual_min"])
        same_bf16_output = (
            proposal_evaluation["output_fingerprint"]["sha256"]
            == current_output_sha
        )
        transition = decide_transition(
            phi_current_expanded=phi_current_expanded,
            phi_candidate=phi_candidate,
            predicted_phi=float(subproblem["predicted_phi"]),
            changed_model=proposal_model_sha != current_model_sha,
            same_bf16_output=same_bf16_output,
            semantic_changed=semantic_changed,
            plateau_already_used=plateau_used,
        )
        state_key = (
            proposal_point_sha,
            proposal_model_sha,
            proposal_evaluation["output_fingerprint"]["sha256"],
            canonical_sha(
                [
                    [*cw15.cut_identity(value), float(value["threshold"])]
                    for value in active
                ]
            ),
            radius_index,
        )
        state_repeated = state_key in seen_states
        seen_states.add(state_key)
        second_plateau = bool(
            same_bf16_output
            and proposal_model_sha != current_model_sha
            and not transition["strict_actual_phi_improvement"]
            and plateau_used
        )

        # Semantic knowledge is monotone even for a rejected/closing trial.
        # Immediately bind every new/strengthened threshold to its permanent
        # CW11 anchor before any terminal or stagnation branch can exit.
        post_merge_anchor_growth = ensure_current_anchor_rows(
            cw15,
            context,
            modules,
            streams,
            legacy,
            parameters,
            raw_actor,
            cw11_total,
            active,
            anchor_archive,
        )
        proposal_total, proposal_restore_after_anchor = restore_current_center(
            cw15,
            context,
            modules,
            streams,
            legacy,
            parameters,
            raw_actor,
            cw11_total,
            proposal_additional,
            proposal_evaluation["cache"],
        )
        post_merge_active_anchor_rows, post_merge_anchor_selection = (
            select_current_hard_anchor_rows(cw15, active, anchor_archive)
        )

        integrity = terminal_integrity(
            cw15,
            context,
            modules,
            parameters,
            layout,
            proposal_evaluation,
            subproblem,
            official_evaluation_count=official_evaluation_count,
            evaluated_model_count=len(evaluated_model_shas),
            physical_dedup_pass=local_audit["pass"],
            semantic_monotone_pass=all(semantic_monotone_checks.values()),
            post_merge_anchor_coverage_pass=post_merge_anchor_growth["pass"],
        )
        acceptance = cw15.terminal_acceptance(
            snapshot=proposal_evaluation["six_view_snapshot"],
            oracle=proposal_evaluation["oracle"],
            fixed=proposal_evaluation["fixed"],
            legacy_gate=proposal_evaluation["legacy_gate"],
            active_gates=post_merge_gates,
            integrity=integrity,
        )
        if len(acceptance["checks"]) != 10:
            raise RuntimeError("CW16 imported terminal gate count is not exact10")
        terminal_component_gates_pass = bool(
            proposal_evaluation["oracle"]["authoritative_60_gates"]["pass"]
            and proposal_evaluation["fixed"]["pass"]
            and proposal_evaluation["legacy_gate"]["pass"]
            and proposal_evaluation["oracle"]["new_harm_count_vs_cw11"] == 0
            and proposal_evaluation["oracle"]["transition_state_checks"][
                "favorable_transitions_retained"
            ]
        )
        terminal_semantic_quiescence = {
            "terminal_component_gates_pass": terminal_component_gates_pass,
            "deterministic_addition_count_zero": len(additions) == 0,
            "semantic_merge_added_zero": not semantic_merge["added"],
            "semantic_merge_strengthened_zero": not semantic_merge["strengthened"],
            "semantic_changed_false": not semantic_changed,
            "proposal_and_post_merge_anchor_count_equal": len(active_anchor_rows)
            == len(post_merge_active_anchor_rows),
            "proposal_and_post_merge_anchor_ledger_sha_equal": anchor_selection[
                "active_anchor_ledger_sha256"
            ]
            == post_merge_anchor_selection["active_anchor_ledger_sha256"],
        }
        if acceptance["pass"] and not all(terminal_semantic_quiescence.values()):
            raise RuntimeError(
                "CW16 terminal10 pass did not imply a quiescent semantic ledger: "
                f"{terminal_semantic_quiescence}"
            )
        eligible = bool(
            acceptance["pass"] and transition["strict_actual_phi_improvement"]
        )

        local_sanitized = [
            cw15.sanitized_affine_tangent(value)
            for value in sorted(local_records, key=cw15.affine_tangent_identity)
        ]
        local_archive_sha = canonical_sha(local_sanitized)
        if local_archive_sha not in local_archive_keys:
            local_archive_keys.add(local_archive_sha)
            local_tangent_archive.append(
                {
                    "center_point_sha256": cw15.float64_vector_sha256(
                        current_additional
                    ),
                    "center_model_sha256": current_model_sha,
                    "center_output_sha256": current_output_sha,
                    "semantic_count_at_linearization": local_audit[
                        "deduplication"
                    ]["semantic_count"],
                    "post_merge_semantic_count": len(active),
                    "physical_record_count": len(local_sanitized),
                    "sanitized_ledger_sha256": local_archive_sha,
                    "records": local_sanitized,
                    "active_in_future_subproblem": False,
                }
            )
        iteration_record = {
            "iteration": proposal_index,
            "kind": "CW16_unique_local_trust_official_proposal",
            "eligible": eligible,
            "bootstrap_seed_nonacceptance_exempt_once": False,
            "official_evaluation_count_after": official_evaluation_count,
            "model_state_sha256": proposal_model_sha,
            "output_state_fingerprint": proposal_evaluation["output_fingerprint"],
            "additional_l2": float(np.linalg.norm(proposal_additional)),
            "additional_float64_le_sha256": proposal_point_sha,
            "total_from_raw_l2": float(np.linalg.norm(proposal_total)),
            "total_from_raw_float64_le_sha256": modules[
                "geometry"
            ].vector_sha256_float64_le(proposal_total, np),
            "center_additional_l2": float(np.linalg.norm(current_additional)),
            "center_additional_float64_le_sha256": cw15.float64_vector_sha256(
                current_additional
            ),
            "center_model_state_sha256": current_model_sha,
            "center_output_state_sha256": current_output_sha,
            "trust_radius_index": radius_index,
            "trust_radius_divisor": TRUST_RADIUS_DIVISORS[radius_index],
            "trust_radius": trust_radius,
            "center_restore": center_restore,
            "hard_anchor_selection": anchor_selection,
            "proposal_subproblem_anchor_ledger": {
                "row_count": len(active_anchor_rows),
                "sanitized_ledger_sha256": anchor_selection[
                    "active_anchor_ledger_sha256"
                ],
                "linearization_kind": "CW11_anchor",
                "used_in_this_proposal_subproblem": True,
            },
            "hard_anchor_array_audit": anchor_array_audit,
            "current_local_audit": local_audit,
            "subproblem": subproblem,
            "historical_candidate_tangent_archive_count": len(
                historical_candidate_archive
            ),
            "historical_candidate_tangent_active_count": 0,
            "pre_oracle_repeat_checks": pre_oracle_repeat_checks,
            "single_evaluation_checks": proposal_evaluation["checks"],
            "legacy_gate": proposal_evaluation["legacy_gate"],
            "six_view_snapshot": proposal_evaluation["six_view_snapshot"],
            "oracle": proposal_evaluation["oracle"],
            "fixed_repair_gates": proposal_evaluation["fixed"],
            "active_cut_gates_before_merge": active_gates_before_merge,
            "deterministic_addition_count": len(additions),
            "deterministic_addition_identity_sha256": canonical_sha(
                [list(cw15.cut_identity(value)) for value in additions]
            ),
            "post_oracle_cut_merge": semantic_merge,
            "semantic_monotone_checks": semantic_monotone_checks,
            "active_cut_gates": post_merge_gates,
            "cached_current_active_cut_gates_on_expanded_ledger":
            current_expanded_gates,
            "exact_merit": {
                "definition": "Phi=min post-merge semantic residual",
                "current_on_expanded_ledger": phi_current_expanded,
                "candidate_on_expanded_ledger": phi_candidate,
                "strict_comparison_without_tolerance": phi_candidate
                > phi_current_expanded,
                "predicted_phi": subproblem["predicted_phi"],
            },
            "transition": transition,
            "state_repeated": state_repeated,
            "second_same_BF16_plateau": second_plateau,
            "post_merge_hard_anchor_growth": post_merge_anchor_growth,
            "post_merge_current_anchor_selection": post_merge_anchor_selection,
            "post_merge_current_anchor_ledger": {
                "row_count": len(post_merge_active_anchor_rows),
                "sanitized_ledger_sha256": post_merge_anchor_selection[
                    "active_anchor_ledger_sha256"
                ],
                "linearization_kind": "CW11_anchor",
                "used_in_already_evaluated_proposal_subproblem": False,
            },
            "terminal_semantic_quiescence": terminal_semantic_quiescence,
            "proposal_restore_after_anchor_growth":
            proposal_restore_after_anchor,
            "integrity": integrity,
            "acceptance": acceptance,
            "local_tangent_archived_after_subproblem": {
                "sanitized_ledger_sha256": local_archive_sha,
                "physical_record_count": len(local_sanitized),
                "active_in_future_subproblem": False,
            },
        }
        iterations.append(iteration_record)

        if acceptance["pass"] and not eligible:
            close_reason = "fail_closed_terminal_without_strict_post_merge_Phi_improvement"
            iteration_record["close_reason"] = close_reason
            break
        if eligible:
            current_additional = np.asarray(proposal_additional, dtype=np.float64).copy()
            current_total = np.asarray(proposal_total, dtype=np.float64).copy()
            current_model_sha = proposal_model_sha
            current_cache = proposal_evaluation["cache"]
            current_output_sha = proposal_evaluation["output_fingerprint"]["sha256"]
            current_phi = phi_candidate
            current_evaluation = proposal_evaluation
            terminal_iteration = proposal_index
            terminal_active_gates = post_merge_gates
            terminal_legacy_gate = proposal_evaluation["legacy_gate"]
            terminal_oracle = proposal_evaluation["oracle"]
            terminal_optimization_ledger_semantics = {
                "terminal_semantic_quiescence": copy.deepcopy(
                    terminal_semantic_quiescence
                ),
                "proposal_subproblem_anchor_ledger": copy.deepcopy(
                    iteration_record["proposal_subproblem_anchor_ledger"]
                ),
                "post_merge_current_anchor_ledger": copy.deepcopy(
                    iteration_record["post_merge_current_anchor_ledger"]
                ),
                "proposal_subproblem_local_row_count": len(local_records),
                "proposal_subproblem_local_sanitized_ledger_sha256":
                local_archive_sha,
                "post_merge_new_anchors_are_not_claimed_as_used_in_QP": True,
                "eligible_implies_semantic_changed_false": not semantic_changed,
            }
            terminal_payload = cw15.encode_terminal_payload(
                context,
                modules,
                parameters,
                cw11_actor_bytes,
                current_additional,
                current_total,
                current_model_sha,
            )
            terminal_reconstruction_audit = cw15.reconstruct_terminal_candidate(
                model,
                helper,
                raw_actor,
                cw11_total,
                terminal_payload,
                cutting=modules["cutting"],
                geometry=modules["geometry"],
                ram=modules["ram"],
            )
            if not (
                all(terminal_reconstruction_audit["checks"].values())
                and terminal_reconstruction_audit["model_state_sha256"]
                == current_model_sha
            ):
                raise RuntimeError("CW16 immediate terminal reconstruction audit failed")
            iteration_record["terminal_reconstruction_audit"] = copy.deepcopy(
                terminal_reconstruction_audit
            )
            close_reason = "first_feasible_strict_post_merge_Phi_candidate"
            break
        if state_repeated:
            close_reason = "fail_closed_repeated_point_model_output_semantic_radius_state"
            iteration_record["close_reason"] = close_reason
            break
        if second_plateau:
            close_reason = "fail_closed_second_changed_model_same_BF16_plateau"
            iteration_record["close_reason"] = close_reason
            break

        center_changed = transition["action"] in {
            "accept_strict_phi_improvement",
            "move_once_same_BF16_plateau_and_shrink",
        }
        shrink_required = transition["action"] in {
            "move_once_same_BF16_plateau_and_shrink",
            "reject_and_shrink",
        }
        if center_changed:
            current_additional = np.asarray(proposal_additional, dtype=np.float64).copy()
            current_total = np.asarray(proposal_total, dtype=np.float64).copy()
            current_model_sha = proposal_model_sha
            current_cache = proposal_evaluation["cache"]
            current_output_sha = proposal_evaluation["output_fingerprint"]["sha256"]
            current_phi = phi_candidate
            current_evaluation = proposal_evaluation
            if transition["plateau_exception_allowed"]:
                plateau_used = True
        else:
            current_phi = phi_current_expanded

        current_total, post_decision_restore = restore_current_center(
            cw15,
            context,
            modules,
            streams,
            legacy,
            parameters,
            raw_actor,
            cw11_total,
            current_additional,
            current_cache,
        )
        iteration_record["post_decision_current_restore"] = post_decision_restore
        iteration_record["center_changed"] = center_changed
        iteration_record["radius_shrink_required"] = shrink_required
        if shrink_required:
            if radius_index + 1 >= len(TRUST_RADII):
                close_reason = "fail_closed_trust_radius_below_cap_div_64"
                iteration_record["close_reason"] = close_reason
                break
            radius_index += 1
            iteration_record["next_trust_radius_index"] = radius_index
            iteration_record["next_trust_radius"] = TRUST_RADII[radius_index]
        else:
            iteration_record["next_trust_radius_index"] = radius_index
            iteration_record["next_trust_radius"] = TRUST_RADII[radius_index]

    success = terminal_iteration is not None
    terminal_active_cut_ledger = copy.deepcopy(active) if success else None
    terminal_active_cut_ledger_sha256 = (
        canonical_sha(terminal_active_cut_ledger) if success else None
    )
    terminal_optimization_affine_ledger = None
    terminal_optimization_affine_ledger_sha256 = None
    if success:
        terminal_optimization_affine_ledger = [
            *[
                cw15.sanitized_affine_tangent(value)
                for value in sorted(
                    last_active_anchor_rows, key=cw15.affine_tangent_identity
                )
            ],
            *[
                cw15.sanitized_affine_tangent(value)
                for value in sorted(last_local_records, key=cw15.affine_tangent_identity)
            ],
        ]
        terminal_optimization_affine_ledger_sha256 = canonical_sha(
            terminal_optimization_affine_ledger
        )
        terminal_optimization_ledger_semantics[
            "combined_proposal_subproblem_optimization_row_count"
        ] = len(terminal_optimization_affine_ledger)
        terminal_optimization_ledger_semantics[
            "combined_proposal_subproblem_optimization_ledger_sha256"
        ] = terminal_optimization_affine_ledger_sha256
    final_active_anchor_rows, final_active_anchor_selection = (
        select_current_hard_anchor_rows(cw15, active, anchor_archive)
    )
    optimization_archive = {
        "CW11_anchor_archive": [
            cw15.sanitized_affine_tangent(value)
            for value in sorted(anchor_archive, key=cw15.affine_tangent_identity)
        ],
        "historical_CW15_candidate_tangent_archive": historical_candidate_archive,
        "expired_current_local_tangent_archives": local_tangent_archive,
        "current_active_CW11_anchor_selection": {
            "row_count": len(final_active_anchor_rows),
            "sanitized_ledger_sha256": final_active_anchor_selection[
                "active_anchor_ledger_sha256"
            ],
            "archive_row_count": len(anchor_archive),
            "selection_audit": final_active_anchor_selection,
        },
        "historical_candidate_active_count": 0,
    }
    optimization_archive_sha = canonical_sha(optimization_archive)

    if success and user_candidate_consumer is not None:
        user_candidate_consumer(
            {
                "helper": helper,
                "model": model,
                "checkpoint": context["checkpoint"],
                "model_config": dict(context["model_config"]),
                "raw_actor": raw_actor,
                "raw_model_state_sha256": cw15.RAW_MODEL_SHA256,
                "raw_nonactor_sha256": cw15.RAW_NONACTOR_SHA256,
                "cw11_model_state_sha256": cw15.CW11_MODEL_SHA256,
                "cw11_total_float64": cw11_total.copy(),
                "cw11_total_float64_le_sha256": cw15.CW11_VECTOR_SHA256,
                "terminal_additional_float64": current_additional.copy(),
                "terminal_total_float64": current_total.copy(),
                "terminal_payload": copy.deepcopy(terminal_payload),
                "terminal_model_state_sha256": current_model_sha,
                "terminal_iteration": terminal_iteration,
                "active_cut_ledger": copy.deepcopy(terminal_active_cut_ledger),
                "active_cut_ledger_sha256": terminal_active_cut_ledger_sha256,
                "optimization_affine_tangent_ledger": copy.deepcopy(
                    terminal_optimization_affine_ledger
                ),
                "optimization_affine_tangent_ledger_sha256":
                terminal_optimization_affine_ledger_sha256,
                "optimization_affine_tangent_ledger_semantics": copy.deepcopy(
                    terminal_optimization_ledger_semantics
                ),
                "legacy_B33_gate": copy.deepcopy(terminal_legacy_gate),
                "terminal_consumed_valid_oracle": copy.deepcopy(terminal_oracle),
                "terminal_active_cut_gates": copy.deepcopy(terminal_active_gates),
                "reconstruction_helper": cw15.reconstruct_terminal_candidate,
                "consumed_specialist_data_must_not_be_reopened_downstream": True,
            }
        )
        candidate_consumer_called = True

    return {
        "status": (
            "consumed_valid_CW16_local_trust_first_feasible"
            if success
            else "closed_no_CW16_candidate"
        ),
        "classification": dict(CLASSIFICATION),
        "incoming_checks": incoming_checks,
        "materialized_live_binding": materialized_checks,
        "actor6_layout": layout,
        "stream_loading_audit": stream_loading_audit,
        "raw_six_view_reproduction": raw_reproduction,
        "initial_checks": initial_checks,
        "initial_active_cut_count": 38,
        "iterations": iterations,
        "decision": {
            "first_feasible_required": True,
            "terminal_iteration": terminal_iteration,
            "close_reason": close_reason,
            "terminal_model_state_sha256": (
                None if terminal_payload is None else current_model_sha
            ),
            "terminal_additional_l2": float(np.linalg.norm(current_additional)),
            "terminal_total_from_raw_l2": float(np.linalg.norm(current_total)),
            "terminal_active_cut_count": len(active),
            "terminal_active_CW11_anchor_count": len(final_active_anchor_rows),
            "terminal_active_CW11_anchor_ledger_sha256":
            final_active_anchor_selection["active_anchor_ledger_sha256"],
            "full_CW11_anchor_archive_count": len(anchor_archive),
            "official_candidate_evaluation_count": official_evaluation_count,
            "max_official_candidate_evaluations": MAX_OFFICIAL_EVALUATIONS,
            "bootstrap_evaluation_count": 1,
            "bootstrap_seed_nonacceptance_exempt_once": True,
            "bootstrap_eligible": False,
            "plateau_exception_used": plateau_used,
            "terminal_trust_radius_index": radius_index,
            "terminal_trust_radius": TRUST_RADII[
                min(radius_index, len(TRUST_RADII) - 1)
            ],
            "semantic_ledger_monotone": True,
            "CW11_anchor_always_hard": True,
            "historical_candidate_tangent_active_count": 0,
            "no_clip": True,
            "candidate_consumer_called": candidate_consumer_called,
            "eligible_only_for_formal_fulltrain_revalidation": success,
            "eligible_as_promotion_evidence": False,
        },
        "terminal_reconstruction_payload": terminal_payload,
        "terminal_reconstruction_audit": terminal_reconstruction_audit,
        "terminal_active_cut_ledger": terminal_active_cut_ledger,
        "terminal_active_cut_ledger_sha256": terminal_active_cut_ledger_sha256,
        "optimization_affine_closure_ledger": optimization_archive,
        "optimization_affine_closure_ledger_sha256": optimization_archive_sha,
        "terminal_optimization_affine_ledger":
        terminal_optimization_affine_ledger,
        "terminal_optimization_affine_ledger_sha256":
        terminal_optimization_affine_ledger_sha256,
        "terminal_optimization_affine_ledger_semantics":
        terminal_optimization_ledger_semantics,
        "downstream_contract": {
            "formal_must_not_call_official6_solver_again": True,
            "formal_reconstructs_from_payload_without_specialist_access": True,
            "formal_uses_original_raw_U468_checkpoint_template": True,
            "formal_then_runs_one_frozen_fulltrain_revalidation_only": True,
            "broad_requires_new_CW16_hash_bound_one_shot_runner": True,
        },
    }


def run_probe(
    cw15: ModuleType,
    self_source: bytes,
    self_evidence: Mapping[str, Any],
    primary: ModuleType,
    primary_source: bytes,
    primary_evidence: Mapping[str, Any],
    *,
    candidate_consumer: Any = None,
) -> dict[str, Any]:
    self_stat = SCRIPT.lstat()
    self_checks = {
        "source_sha_exact_evidence": hashlib.sha256(self_source).hexdigest()
        == self_evidence.get("sha256"),
        "source_bytes_still_exact": SCRIPT.read_bytes() == self_source,
        "path_exact": self_evidence.get("path") == str(SCRIPT.relative_to(ROOT)),
        "mode_exact0555": self_evidence.get("mode_octal") == "0555"
        and stat.S_IMODE(self_stat.st_mode) == FROZEN_MODE,
        "regular_one_link": stat.S_ISREG(self_stat.st_mode)
        and self_stat.st_nlink == 1,
        "held_fd_identity_exact": self_evidence.get(
            "single_link_regular_held_fd_identity_exact"
        )
        is True,
    }
    if not all(self_checks.values()):
        raise RuntimeError(f"CW16 exact self binding drift: {self_checks}")
    frozen_inputs = validate_frozen_inputs_40(cw15)
    consumed_cw15 = validate_cw15_closure_evidence(cw15)
    local_math = local_trust_math_self_test()
    negative_math = curvature_and_negative_self_test()
    dedup_math = physical_dedup_self_test(cw15)
    original_outer = cw15.run_outer_cutting_plane

    def patched_outer(
        context: Mapping[str, Any],
        modules: Mapping[str, ModuleType],
        *,
        user_candidate_consumer: Any = None,
    ) -> dict[str, Any]:
        return run_outer_local_trust(
            cw15,
            consumed_cw15,
            context,
            modules,
            user_candidate_consumer=user_candidate_consumer,
        )

    try:
        cw15.run_outer_cutting_plane = patched_outer
        inherited = cw15.run_probe(
            primary,
            primary_source,
            primary_evidence,
            candidate_consumer=candidate_consumer,
        )
    finally:
        cw15.run_outer_cutting_plane = original_outer
    second = inherited["second_stage"]
    decision = second["decision"]
    bootstrap_records = [
        value
        for value in second["iterations"]
        if value.get("bootstrap_seed_nonacceptance_exempt_once") is True
    ]
    proposal_records = [
        value
        for value in second["iterations"]
        if value.get("kind") == "CW16_unique_local_trust_official_proposal"
    ]
    success = second["status"] == "consumed_valid_CW16_local_trust_first_feasible"
    terminal_records = [value for value in proposal_records if value.get("eligible")]
    plateau_records = [
        value
        for value in proposal_records
        if value.get("transition", {}).get("plateau_exception_allowed") is True
        and value.get("center_changed") is True
    ]
    unapplied_plateau_records = [
        value
        for value in proposal_records
        if value.get("transition", {}).get("plateau_exception_allowed") is True
        and value.get("center_changed") is not True
    ]
    proposal_radius_indices = [
        int(value["trust_radius_index"]) for value in proposal_records
    ]
    evaluated_point_shas = [
        CW15_ITERATION1_POINT_SHA256,
        *[str(value["additional_float64_le_sha256"]) for value in proposal_records],
    ]
    evaluated_model_shas = [
        CW15_ITERATION1_MODEL_SHA256,
        *[str(value["model_state_sha256"]) for value in proposal_records],
    ]
    run_checks = {
        "status_exact_success_or_closed": second["status"] in {
            "consumed_valid_CW16_local_trust_first_feasible",
            "closed_no_CW16_candidate",
        }
        and success
        == (
            second["status"]
            == "consumed_valid_CW16_local_trust_first_feasible"
        ),
        "inherited_CW11_final_integrity": inherited["final_integrity"]["pass"]
        is True,
        "frozen_inputs_exact40": frozen_inputs["binding_count"] == 40,
        "consumed_CW15_exact": consumed_cw15["pass"] is True,
        "bootstrap_exception_exactly_once": len(bootstrap_records) == 1,
        "bootstrap_exact_iteration1_ineligible": len(bootstrap_records) == 1
        and bootstrap_records[0]["iteration"] == 1
        and bootstrap_records[0]["eligible"] is False
        and bootstrap_records[0]["is_acceptance"] is False,
        "bootstrap_counts_budget1": len(bootstrap_records) == 1
        and bootstrap_records[0]["official_evaluation_count_after"] == 1,
        "bootstrap_terminal10_exact_failure_set": len(bootstrap_records) == 1
        and bootstrap_records[0]["acceptance"]["pass"] is False
        and bootstrap_records[0]["acceptance_false_keys"]
        == [
            "active_cut_violations_zero",
            "forensic_favorable_retention_pass",
            "new_harm_count_zero",
        ],
        "no_later_bootstrap_exception": all(
            value.get("bootstrap_seed_nonacceptance_exempt_once") is False
            for value in proposal_records
        ),
        "official_budget_at_most12": 1
        <= decision["official_candidate_evaluation_count"]
        <= MAX_OFFICIAL_EVALUATIONS,
        "official_budget_equals_bootstrap_plus_evaluated_proposals": decision[
            "official_candidate_evaluation_count"
        ]
        == 1 + len(proposal_records),
        "evaluated_iteration_indices_match_official_counts": all(
            value["iteration"] == value["official_evaluation_count_after"]
            for value in proposal_records
        ),
        "evaluated_points_and_models_globally_unique": len(evaluated_point_shas)
        == len(set(evaluated_point_shas))
        and len(evaluated_model_shas) == len(set(evaluated_model_shas)),
        "each_evaluated_proposal_single_full_six_and_integrity": all(
            value["single_evaluation_checks"]["single_full_six_call_owned_here"]
            is True
            and all(value["single_evaluation_checks"].values())
            and all(value["integrity"].values())
            and len(value["acceptance"]["checks"]) == 10
            for value in proposal_records
        ),
        "each_proposal_semantic_and_anchor_coverage_exact": all(
            all(value["semantic_monotone_checks"].values())
            and value["post_merge_hard_anchor_growth"]["pass"] is True
            and value["current_local_audit"]["pass"] is True
            and value["current_local_audit"]["deduplication"]["checks"][
                "semantic_mapping_bijective"
            ]
            is True
            for value in proposal_records
        ),
        "eligible_iff_terminal10_and_strict_exact_Phi": all(
            bool(value["eligible"])
            == bool(
                value["acceptance"]["pass"]
                and value["exact_merit"]["strict_comparison_without_tolerance"]
            )
            for value in proposal_records
        ),
        "terminal10_pass_implies_semantic_quiescence": all(
            not value["acceptance"]["pass"]
            or (
                value["transition"]["semantic_changed"] is False
                and value["deterministic_addition_count"] == 0
                and all(value["terminal_semantic_quiescence"].values())
            )
            for value in proposal_records
        ),
        "at_most_one_plateau_and_immediate_halving": len(plateau_records) <= 1
        and all(
            value.get("center_changed") is True
            and value.get("radius_shrink_required") is True
            and value.get("next_trust_radius_index")
            == value["trust_radius_index"] + 1
            and value.get("next_trust_radius") == value["trust_radius"] / 2.0
            for value in plateau_records
        )
        and all(
            value.get("close_reason")
            == "fail_closed_terminal_without_strict_post_merge_Phi_improvement"
            and value.get("acceptance", {}).get("pass") is True
            for value in unapplied_plateau_records
        ),
        "trust_radius_indices_monotone_no_expansion": all(
            right in (left, left + 1)
            for left, right in zip(
                proposal_radius_indices, proposal_radius_indices[1:]
            )
        ),
        "success_terminal_exactly_one_and_first_feasible": (
            not success
            and not terminal_records
            and second["terminal_reconstruction_payload"] is None
            and second["terminal_reconstruction_audit"] is None
            and second["terminal_optimization_affine_ledger_semantics"] is None
        )
        or (
            success
            and len(terminal_records) == 1
            and terminal_records[-1] is proposal_records[-1]
            and terminal_records[-1]["acceptance"]["pass"] is True
            and decision["terminal_iteration"] == terminal_records[-1]["iteration"]
            and decision["close_reason"]
            == "first_feasible_strict_post_merge_Phi_candidate"
            and second["terminal_reconstruction_payload"] is not None
            and second["terminal_reconstruction_audit"] is not None
            and second["terminal_optimization_affine_ledger_semantics"][
                "eligible_implies_semantic_changed_false"
            ]
            is True
            and all(
                second["terminal_optimization_affine_ledger_semantics"][
                    "terminal_semantic_quiescence"
                ].values()
            )
        ),
        "historical_candidate_active_zero": decision[
            "historical_candidate_tangent_active_count"
        ]
        == 0,
        "final_active_semantic_to_CW11_anchor_one_to_one": decision[
            "terminal_active_CW11_anchor_count"
        ]
        == decision["terminal_active_cut_count"],
        "no_clip": decision["no_clip"] is True,
        "classification_exact": second["classification"] == CLASSIFICATION,
    }
    if not all(run_checks.values()):
        raise RuntimeError(f"CW16 run integrity failed: {run_checks}")
    input_lock = copy.deepcopy(inherited["input_lock"])
    input_lock["self"] = dict(self_evidence)
    input_lock["CW16_exact_self_binding_checks"] = self_checks
    input_lock["CW15_parent_source"] = {
        "path": str(CW15_PARENT.relative_to(ROOT)),
        "sha256": CW15_PARENT_SHA256,
        "mode": "0555",
        "counted_in_frozen_input_40": False,
        "role": "exact executable source dependency",
    }
    input_lock["consumed_CW15_closure"] = {
        "attempt_marker_path": str(CW15_ATTEMPT.relative_to(ROOT)),
        "attempt_marker_sha256": CW15_ATTEMPT_SHA256,
        "closed_stdout_path": str(CW15_STDOUT.relative_to(ROOT)),
        "closed_stdout_sha256": CW15_STDOUT_SHA256,
        "stderr_audit_path": str(CW15_STDERR.relative_to(ROOT)),
        "stderr_audit_sha256": CW15_STDERR_SHA256,
        "parent_solver_sha256": CW15_PARENT_SHA256,
    }
    return {
        "schema_version": SCHEMA,
        "status": second["status"],
        "classification": dict(CLASSIFICATION),
        "scope": {
            "exact_CW11_reconstruction": True,
            "exact_CW15_iteration1_bootstrap_replay": True,
            "bootstrap_seed_nonacceptance_exempt_once": True,
            "bootstrap_eligible": False,
            "semantic_active_cut_ledger_monotone": True,
            "CW11_anchor_hard_active": True,
            "CW11_anchor50_only_reference_l2": CW14_ANCHOR50_ONLY_L2,
            "CW15_combined_anchor50_historical12_rows_excluded": True,
            "historical_candidate_tangent_archive_only": True,
            "historical_candidate_tangent_active_count": 0,
            "current_point_all_active_physical_relinearized": True,
            "exact_metric_four_to_one_optimization_dedup": True,
            "physical_dedup_complete_four_metrics_only": True,
            "physical_dedup_threshold_margin_gradient_all_identical": True,
            "physical_partial_or_nonidentical_preserved_rowwise": True,
            "physical_mapping_bijective": True,
            "max_z_nonpositive_then_fixed_z_minimum_norm": True,
            "additional_total_L2_cap": TOTAL_L2_CAP,
            "trust_radius_divisors": list(TRUST_RADIUS_DIVISORS),
            "trust_radius_expansion": False,
            "strict_exact_post_merge_Phi_improvement": True,
            "single_changed_model_same_BF16_plateau": True,
            "max_official_candidate_evaluations_including_bootstrap":
            MAX_OFFICIAL_EVALUATIONS,
            "authoritative_full_six_official_B256_oracle_each_new_model_proposal":
            True,
            "terminal_gate_count": 10,
            "candidate_RAM_only": True,
            "optimizer_backward_training": False,
            "model_or_result_writes": 0,
            "network_upload_submission": False,
            "broad_or_gold_access": False,
            "specialist_valid_consumed_dev_only": True,
        },
        "input_lock": input_lock,
        "frozen_inputs": frozen_inputs,
        "cw15_consumed_closure_evidence": consumed_cw15,
        "local_trust_math_self_test": local_math,
        "curvature_and_negative_self_test": negative_math,
        "physical_dedup_self_test": dedup_math,
        "threshold_function_shas": dict(cw15.CW13_THRESHOLD_FUNCTION_SHAS),
        "runtime": inherited["runtime"],
        "cw11_reconstruction_summary": inherited["cw11_reconstruction_summary"],
        "second_stage": second,
        "terminal_reconstruction_audit": second[
            "terminal_reconstruction_audit"
        ],
        "terminal": {
            "status": second["status"],
            "decision": decision,
            "reconstruction_payload": second["terminal_reconstruction_payload"],
            "reconstruction_audit": second["terminal_reconstruction_audit"],
            "active_cut_ledger": second["terminal_active_cut_ledger"],
            "active_cut_ledger_sha256": second[
                "terminal_active_cut_ledger_sha256"
            ],
            "optimization_affine_tangent_ledger": second[
                "terminal_optimization_affine_ledger"
            ],
            "optimization_affine_tangent_ledger_sha256": second[
                "terminal_optimization_affine_ledger_sha256"
            ],
            "optimization_affine_tangent_ledger_semantics": second[
                "terminal_optimization_affine_ledger_semantics"
            ],
            "downstream_contract": second["downstream_contract"],
        },
        "final_integrity": {
            "checks": run_checks,
            "pass": all(run_checks.values()),
            "model_left_raw_after_outer_finally": True,
        },
        "run_executed": True,
        "cuda_accessed": True,
        "writes_performed": False,
    }


def selftest_result(*, require_frozen_self: bool) -> dict[str, Any]:
    cw15, parent_evidence = load_cw15_parent()
    source = SCRIPT.read_bytes()
    source_mode = stat.S_IMODE(SCRIPT.stat().st_mode)
    if require_frozen_self and source_mode != FROZEN_MODE:
        raise RuntimeError("CW16 static/run solver must be frozen mode 0555")
    source_audit = static_source_audit(source, cw15)
    if not source_audit["pass"]:
        raise RuntimeError(f"CW16 static source audit failed: {source_audit}")
    frozen_inputs = validate_frozen_inputs_40(cw15)
    consumed_cw15 = validate_cw15_closure_evidence(cw15)
    local_math = local_trust_math_self_test()
    negative_math = curvature_and_negative_self_test()
    dedup_math = physical_dedup_self_test(cw15)
    inherited_math = {
        "anchored_total_geometry": cw15.anchored_total_geometry_self_test(),
        "sequential_affine_tangent": cw15.sequential_affine_tangent_self_test(),
        "affine_ledger_integrity": cw15.affine_ledger_integrity_self_test(),
    }
    return {
        "schema_version": SCHEMA,
        "status": (
            "static_ready_CW16_local_trust_run_implemented"
            if require_frozen_self
            else "writable_CW16_CPU_selftests_passed_not_frozen_not_runnable"
        ),
        "classification": dict(CLASSIFICATION),
        "contract": {
            "starting_model_sha256": cw15.CW11_MODEL_SHA256,
            "CW15_parent_solver_sha256": CW15_PARENT_SHA256,
            "CW15_attempt_sha256": CW15_ATTEMPT_SHA256,
            "CW15_stdout_sha256": CW15_STDOUT_SHA256,
            "CW15_stderr_sha256": CW15_STDERR_SHA256,
            "frozen_input_count": 40,
            "old_input_count": 37,
            "new_CW15_input_count": 3,
            "bootstrap_point_sha256": CW15_ITERATION1_POINT_SHA256,
            "bootstrap_model_sha256": CW15_ITERATION1_MODEL_SHA256,
            "bootstrap_output_sha256": CW15_ITERATION1_OUTPUT_SHA256,
            "bootstrap_additions_sha256": CW15_ITERATION1_ADDITIONS_SHA256,
            "bootstrap_seed_nonacceptance_exempt_once": True,
            "bootstrap_eligible": False,
            "bootstrap_counts_official_budget": True,
            "semantic_threshold_change_from_CW15": False,
            "threshold_function_shas": dict(cw15.CW13_THRESHOLD_FUNCTION_SHAS),
            "terminal_function": "exact imported CW15 terminal_acceptance 10 checks",
            "total_additional_L2_cap": TOTAL_L2_CAP,
            "max_official_candidate_evaluations": MAX_OFFICIAL_EVALUATIONS,
            "trust_radius_divisors": list(TRUST_RADIUS_DIVISORS),
            "trust_radius_values": list(TRUST_RADII),
            "trust_radius_expansion": False,
            "semantic_ledger_monotone": True,
            "CW11_anchor_hard": True,
            "CW11_anchor50_only_reference_l2": CW14_ANCHOR50_ONLY_L2,
            "CW11_anchor50_only_reference_point_sha256":
            CW14_ANCHOR50_ONLY_POINT_SHA256,
            "CW15_combined_anchor50_historical12_row_count":
            CW15_COMBINED_ANCHOR50_HISTORICAL12_ROW_COUNT,
            "CW15_combined62_uncapped_l2_archive_only":
            CW15_ITERATION2_UNCAPPED_L2,
            "historical_candidate_tangent_active_count": 0,
            "current_local_physical_exact_four_to_one": True,
            "physical_dedup_complete_four_metrics_only": True,
            "physical_dedup_threshold_margin_gradient_all_identical": True,
            "physical_partial_or_nonidentical_preserved_rowwise": True,
            "physical_mapping_bijective": True,
            "local_objective": "max z<=0 then fixed-z minimum total anchored L2",
            "exact_merit": "Phi=min post-merge semantic residual",
            "strict_improvement_without_tolerance": True,
            "single_plateau_exception": True,
            "no_clip": True,
            "one_full_six_per_proposal": True,
            "first_feasible": FIRST_FEASIBLE,
            "candidate_ordering": "fixed sequential no best-of-N",
            "attempt_semantics": (
                "stdout-only probe has no marker; an audited one-shot launcher owns "
                "unique absent output and attempt-marker enforcement"
            ),
        },
        "runtime": validate_runtime(require_cuda=False),
        "self_source": {
            "path": str(SCRIPT.relative_to(ROOT)),
            "sha256": hashlib.sha256(source).hexdigest(),
            "mode": f"{source_mode:04o}",
            "nlink": int(SCRIPT.stat().st_nlink),
            "frozen_required": require_frozen_self,
        },
        "parent_source": parent_evidence,
        "frozen_inputs": frozen_inputs,
        "cw15_consumed_closure_evidence": consumed_cw15,
        "local_trust_math_self_test": local_math,
        "curvature_and_negative_self_test": negative_math,
        "physical_dedup_self_test": dedup_math,
        "inherited_pure_math_self_tests": inherited_math,
        "source_audit": source_audit,
        "run_executed": False,
        "cuda_accessed": False,
        "writes_performed": False,
    }


def static_result() -> dict[str, Any]:
    return selftest_result(require_frozen_self=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("selftest", "static", "run"), default="static"
    )
    args = parser.parse_args()
    if args.mode == "selftest":
        result = selftest_result(require_frozen_self=False)
    elif args.mode == "static":
        result = static_result()
    else:
        if stat.S_IMODE(SCRIPT.stat().st_mode) != FROZEN_MODE:
            raise RuntimeError("CW16 run solver must be frozen mode 0555")
        validate_runtime(require_cuda=False)
        cw15, parent_evidence = load_cw15_parent()
        source = SCRIPT.read_bytes()
        source_audit = static_source_audit(source, cw15)
        if not source_audit["pass"]:
            raise RuntimeError(f"CW16 run source audit failed: {source_audit}")
        cw11 = cw15.import_frozen(
            cw15.CW11_PROBE,
            cw15.MODULE_SHAS[cw15.CW11_PROBE],
            cw15.EXPECTED_INPUT_MODES[cw15.CW11_PROBE],
            "cw16_main_frozen_cw11",
        )
        self_source, self_evidence = cw11.read_regular_bytes(
            SCRIPT,
            None,
            "CW16 local-trust consumed-valid solver",
            expected_mode=FROZEN_MODE,
        )
        if self_source != source:
            raise RuntimeError("CW16 self source changed during run preflight")
        primary_source, primary_evidence = cw11.read_regular_bytes(
            cw15.PRIMARY,
            cw15.MODULE_SHAS[cw15.PRIMARY],
            "CW16 frozen primary source",
            expected_mode=FROZEN_MODE,
        )
        primary = cw15.import_frozen(
            cw15.PRIMARY,
            cw15.MODULE_SHAS[cw15.PRIMARY],
            cw15.EXPECTED_INPUT_MODES[cw15.PRIMARY],
            "cw16_main_frozen_primary",
        )
        result = run_probe(
            cw15,
            source,
            self_evidence,
            primary,
            primary_source,
            primary_evidence,
        )
        result["input_lock"]["CW15_parent_source"] = parent_evidence
        result["source_audit"] = source_audit
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
