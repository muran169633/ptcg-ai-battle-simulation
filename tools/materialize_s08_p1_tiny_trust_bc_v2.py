#!/usr/bin/env python3
"""Transactional v2 materializer for the frozen S8 tiny-P1 BC step.

This revision supersedes (but never mutates or executes) v1.  It keeps the
same frozen P1 direction and R50->R25 calibration priority while fixing two
audit findings:

* the acceptance denominator is now ``-g_frozen dot actual_float32_delta``;
  it is explicitly a frozen CUDA/BF16 geometry reference for both forward
  semantics, not a claimed CPU/FP32 Taylor derivative;
* a passing checkpoint is O_EXCL-reserved empty, inputs are revalidated, and
  the immutable result (including expected checkpoint hash and validity rule)
  is published and its parent directory fsynced before the reserved file is
  filled and fsynced.  A crash may leave an empty, partial, or opportunistically
  complete checkpoint, but never a contract-valid checkpoint without its
  durable result contract.

Calibration gradients are recomputed and hash-matched before radius
selection.  Final gradients and behavior rows are not evaluated until one
radius is locked.  Every first/second behavior evaluation must independently
pass the complete CPU/FP32 and CUDA/BF16 gates, in addition to matching each
other deterministically.  No gameplay, H2H, optimizer, package, upload, or
submission path exists in this tool.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
V1_TOOL = TOOLS / "materialize_s08_p1_tiny_trust_bc.py"
V1_TOOL_SHA256 = (
    "c2d10ca8aa790fbd4951d15967eb6df2a066e3894a3eabde6c85d8ec8042c8d3"
)
V1_PLAN = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_p1_tiny_trust_bc_v1.reviewed_plan.json"
)
V1_PLAN_SHA256 = (
    "fc161394425b1aef4f3ecf90c3c1d92ae8a59c3f72d4c3374cd340dd0b1d61e0"
)
RUN_PPO_BC_REPAIR = TOOLS / "run_ppo_bc_repair.py"
RUN_PPO_BC_REPAIR_SHA256 = (
    "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"
)
SIM_PY = ROOT / "dataset/sample_submission/sample_submission/cg/sim.py"
SIM_PY_SHA256 = (
    "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655"
)
LIBCG = ROOT / "dataset/sample_submission/sample_submission/cg/libcg.so"
LIBCG_SHA256 = (
    "feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887"
)

SCHEMA_VERSION = "ptcg-s08-p1-tiny-trust-bc-materializer-v2"
PLAN_PATH = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_p1_tiny_trust_bc_v2.reviewed_plan.json"
)
RESULT_PATH = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_p1_tiny_trust_bc_v2.result.json"
)
CHECKPOINT_PATH = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_p1_tiny_trust_bc_v2.pt"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, expected_sha256: str, label: str) -> Path:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {label}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} must be a non-symlink regular file")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    return path


# Authenticate the additional import closure before importing v1, whose
# frozen geometry dependency imports train_ppo and therefore cg.sim/libcg.
PREIMPORT_INPUTS: dict[str, tuple[Path, str]] = {
    "v1_tool": (V1_TOOL, V1_TOOL_SHA256),
    "run_ppo_bc_repair": (RUN_PPO_BC_REPAIR, RUN_PPO_BC_REPAIR_SHA256),
    "cg_sim_py": (SIM_PY, SIM_PY_SHA256),
    "cg_libcg": (LIBCG, LIBCG_SHA256),
}
for _name, (_path, _sha256) in PREIMPORT_INPUTS.items():
    require_regular_file(_path, _sha256, _name)

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_V1_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_s08_p1_v1_c2d10ca8", V1_TOOL
)
if _V1_SPEC is None or _V1_SPEC.loader is None:
    raise RuntimeError("Cannot construct authenticated v1 helper import")
base: ModuleType = importlib.util.module_from_spec(_V1_SPEC)
sys.modules[_V1_SPEC.name] = base
_V1_SPEC.loader.exec_module(base)

torch = base.torch
probe = base.probe
ppo = base.ppo
training_core = base.training_core

IMMUTABLE_INPUTS: dict[str, tuple[Path, str]] = {
    **base.IMMUTABLE_INPUTS,
    **PREIMPORT_INPUTS,
    "v1_frozen_plan_superseded": (V1_PLAN, V1_PLAN_SHA256),
}

ACTOR6 = base.ACTOR6
P1_NAME = base.P1_NAME
P1_VECTOR_SHA256 = base.P1_VECTOR_SHA256
P1_ELEMENTS = base.P1_ELEMENTS
RADIUS_PRIORITY = base.RADIUS_PRIORITY
RADIUS_LABELS = base.RADIUS_LABELS
SEMANTICS = base.SEMANTICS
FIT_SPECS = base.FIT_SPECS
CAL_SPECS = base.CAL_SPECS
FINAL_SPECS = base.FINAL_SPECS
SEED = 2026081059
REFERENCE_RATIO_MIN = 0.10
ABSOLUTE_LOSS_DECREASE_MIN = 2e-7
CONFIRMATION_LOSS_ABS_TOL = 1e-9


def canonical_json_bytes(value: Any) -> bytes:
    return base.canonical_json_bytes(value)


def sha256_json(value: Any) -> str:
    return base.sha256_json(value)


def assert_output_absent(path: Path, label: str) -> None:
    base.assert_output_absent(path, label)


def all_finite(value: Any) -> bool:
    return base.all_finite(value)


def static_scope_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source)

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [
        dotted(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]
    leaves = [name.rsplit(".", 1)[-1] for name in calls]
    forbidden = sorted(
        name
        for name in leaves
        if name in {
            "backward",
            "step",
            "package",
            "upload",
            "submit",
            "submission",
            "BattleStart",
        }
    )
    if forbidden or any(name.startswith("torch.optim") for name in calls):
        raise RuntimeError(f"Forbidden v2 call sites: {forbidden}")
    if "torch.save" in calls:
        raise RuntimeError("v2 must delegate its single serialization to bound v1")
    required = {
        "base.serialize_checkpoint": calls.count("base.serialize_checkpoint") == 1,
        "os.open": calls.count("os.open") >= 1,
        "os.fsync": calls.count("os.fsync") >= 1,
    }
    if not all(required.values()):
        raise RuntimeError(f"v2 transaction call-site drift: {required}")
    return {
        "no_local_torch_save": True,
        "one_serialization_delegated_to_hash_bound_v1": True,
        "no_backward_optimizer_gameplay_package_upload_or_submission": True,
        "battle_import_closure_bound_but_not_called": True,
        "transaction_primitives_present": required,
        "persistent_outputs": [str(PLAN_PATH), str(RESULT_PATH), str(CHECKPOINT_PATH)],
    }


def geometry_gradient_requirements(specs: Sequence[Any]) -> dict[str, Any]:
    report = base.load_geometry_report_cached()
    records: dict[str, Any] = {}
    for spec in specs:
        records[spec.name] = {}
        for mode in ("natural", "legacy_8_2"):
            source = report["gradient_reports"][spec.name][mode]
            records[spec.name][mode] = {
                "gradient": copy.deepcopy(source["gradient"]),
                "selection_effective_weight_sum": source[
                    "selection_effective_weight_sum"
                ],
            }
    return records


def snapshot_input_hashes(*, include_frozen_plan: bool) -> dict[str, str]:
    records = {
        name: file_sha256(path)
        for name, (path, _expected) in IMMUTABLE_INPUTS.items()
    }
    records["materializer_v2"] = file_sha256(Path(__file__).resolve())
    if include_frozen_plan:
        records["materializer_v2_frozen_plan"] = file_sha256(PLAN_PATH)
    return records


def build_plan() -> dict[str, Any]:
    assert_output_absent(RESULT_PATH, "v2 materialization result")
    assert_output_absent(CHECKPOINT_PATH, "v2 materialized checkpoint")
    inputs: dict[str, Any] = {}
    for name, (path, expected) in IMMUTABLE_INPUTS.items():
        require_regular_file(path, expected, name)
        inputs[name] = {"path": str(path), "sha256": expected}
    tool_path = Path(__file__).resolve()
    tool_source = tool_path.read_bytes()
    geometry = base.load_geometry_evidence()
    domains = {
        spec.name: {
            "source": spec.source,
            "role": spec.role,
            "archive": str(spec.archive),
            "member_split": spec.member_split,
            "dates": list(spec.dates),
            **copy.deepcopy(probe.EXPECTED_DOMAIN_STATS[spec.name]),
        }
        for spec in FIT_SPECS + CAL_SPECS + FINAL_SPECS
    }
    return {
        "schema_version": SCHEMA_VERSION + "-plan",
        "purpose": "transactional unique tiny actor6 P1 BC endpoint from S8",
        "supersedes_v1_without_execution": {
            "tool_sha256": V1_TOOL_SHA256,
            "frozen_plan_file_sha256": V1_PLAN_SHA256,
            "v1_result_absent": not base.RESULT_PATH.exists(),
            "v1_checkpoint_absent": not base.CHECKPOINT_PATH.exists(),
            "no_go_reasons_fixed": [
                "ideal-radius dot was mislabeled as a per-semantics Taylor rho",
                "checkpoint preceded terminal input/result evidence",
                "second deterministic evaluation was not independently re-gated",
                "train_ppo dynamic import closure was incompletely bound",
            ],
        },
        "inputs": inputs
        | {
            "materializer_v2": {
                "path": str(tool_path),
                "sha256": hashlib.sha256(tool_source).hexdigest(),
            }
        },
        "runtime_dependency_closure": {
            "run_ppo_bc_repair": inputs["run_ppo_bc_repair"],
            "cg_sim_py": inputs["cg_sim_py"],
            "cg_libcg": inputs["cg_libcg"],
            "reason": "transitive import closure of train_ppo",
            "shared_library_may_be_loaded_by_transitive_import": True,
            "battle_API_function_invoked_by_materializer": False,
        },
        "geometry_lineage": geometry,
        "domains": domains,
        "runtime": {
            "python": str(base.EXPECTED_PYTHON),
            "torch_version": base.EXPECTED_TORCH_VERSION,
            "torch_cuda_version": base.EXPECTED_TORCH_CUDA_VERSION,
            "cudnn_version": base.EXPECTED_CUDNN_VERSION,
            "orjson_version": base.EXPECTED_ORJSON_VERSION,
            "isolated": True,
            "dont_write_bytecode": True,
            "CUBLAS_WORKSPACE_CONFIG": base.EXPECTED_CUBLAS,
            "CUDA_VISIBLE_DEVICES": base.EXPECTED_CUDA_VISIBLE_DEVICES,
            "deterministic_algorithms": True,
            "tf32": False,
            "batch_size": base.BATCH_SIZE,
            "forward_semantics": {
                "cpu_fp32": {
                    "device": "cpu",
                    "dtype": "float32",
                    "autocast": False,
                    "batch_size": base.BATCH_SIZE,
                    "meaning": "batched proxy for CPU/FP32 deployment arithmetic",
                },
                "cuda_bf16": {
                    "device": "cuda:0",
                    "dtype": "bfloat16",
                    "autocast": True,
                    "batch_size": base.BATCH_SIZE,
                    "meaning": "strict local H2H forward arithmetic",
                },
            },
        },
        "direction_reproduction": {
            "fit_sources": list(probe.FIT_TARGETS),
            "all_rows_once": True,
            "natural_ordered_selection_weights": [1.0, 1.0],
            "global_gradient_aggregation": (
                "sum(batch_gradient * selection_effective_weight_sum) / "
                "sum(selection_effective_weight_sum)"
            ),
            "P1_formula": "unit(0.5*g_anti+0.25*g_fros+0.25*g_general)",
            "required_P1_float64_le_sha256": P1_VECTOR_SHA256,
            "required_fit_gradients": copy.deepcopy(base.EXPECTED_FIT_GRADIENTS),
        },
        "frozen_geometry_reference": {
            "gradient_semantics": "S8 CUDA/BF16 frozen geometry",
            "calibration_gradients_recomputed_before_radius_selection": True,
            "calibration_required_records": geometry_gradient_requirements(CAL_SPECS),
            "final_gradients_recomputed_only_after_unique_radius_lock": True,
            "final_required_records": geometry_gradient_requirements(FINAL_SPECS),
            "reference_prediction": "-dot(g_frozen, actual_float32_candidate_delta)",
            "actual_delta": "theta_candidate_float32 - theta_S8_float32",
            "ratio_name": "frozen_geometry_reference_ratio",
            "ratio_definition": (
                "(S8_observed_global_loss-candidate_observed_global_loss) / "
                "(-dot(g_frozen, actual_float32_candidate_delta))"
            ),
            "ratio_min": REFERENCE_RATIO_MIN,
            "cpu_fp32_is_only_compared_to_shared_reference": True,
            "cpu_fp32_Taylor_derivative_claimed": False,
            "cuda_bf16_Taylor_derivative_claimed": False,
            "observed_absolute_loss_decrease_min_each_semantics": (
                ABSOLUTE_LOSS_DECREASE_MIN
            ),
        },
        "candidate_protocol": {
            "parameterization": "theta_candidate=round_fp32(theta_S8-radius*P1)",
            "radius_priority": [
                {"label": RADIUS_LABELS[radius], "l2": radius}
                for radius in RADIUS_PRIORITY
            ],
            "R25_evaluated_only_after_complete_R50_failure": True,
            "actor_scope": list(ACTOR6),
            "all_nonactor_and_count_head_tensors_bit_identical": True,
        },
        "behavior_protocol": {
            "same_run_S8_baseline": True,
            "semantics_required": list(SEMANTICS),
            "whole_domain_loss_aggregation": (
                "fsum(batch_loss*effective_weight_sum)/fsum(effective_weight_sum)"
            ),
            "row_reaverage_forbidden": True,
            "per_semantics_source_mode_loss_gates": {
                "observed_decrease_min": ABSOLUTE_LOSS_DECREASE_MIN,
                "frozen_geometry_reference_ratio_min": REFERENCE_RATIO_MIN,
            },
            "safety_gates": {
                "anti_fros_set_hybrid_ordered_correct_counts": "non-decrease",
                "general_three_accuracy_drop_max": base.GENERAL_ACCURACY_DROP_MAX,
                "each_team_seat_hybrid_drop_max": base.GROUP_HYBRID_DROP_MAX,
                "general_team_macro_hybrid": "non-decrease",
                "count_logits_action_counts_count_correct": "exact identical to S8",
            },
            "confirmation": {
                "second_complete_evaluation": True,
                "first_second_digests_and_metrics_identical": True,
                "first_second_loss_abs_difference_max": CONFIRMATION_LOSS_ABS_TOL,
                "second_evaluation_independently_compared_to_S8": True,
                "second_complete_gate_must_pass": True,
            },
        },
        "selection_protocol": {
            "calibration_views": list(probe.CAL_TARGETS),
            "R50_then_R25": True,
            "complete_gate_requires_first_second_and_confirmation": True,
            "final_behavior_rows_not_iterated_before_selection": True,
            "final_gradients_not_recomputed_before_selection": True,
            "final_archive_bytes_only_SHA_authenticated_before_selection": True,
            "final_views": list(probe.FINAL_TARGETS),
            "final_failure_fallback": False,
        },
        "publication_transaction": {
            "passing_checkpoint_serialized_and_verified_in_memory": True,
            "checkpoint_O_EXCL_reserved_empty_before_terminal_check": True,
            "input_hashes_rechecked_before_result": True,
            "result_O_EXCL_written_before_checkpoint_payload": True,
            "result_parent_directory_fsynced_before_checkpoint_payload": True,
            "result_contains_expected_checkpoint_sha256_and_bytes": True,
            "checkpoint_valid_iff_file_sha256_and_bytes_match_result": True,
            "input_hashes_rechecked_after_result_before_checkpoint_fill": True,
            "reserved_fd_then_filled_and_fsynced": True,
            "contract_valid_checkpoint_cannot_exist_without_durable_result": True,
        },
        "scope_audit": static_scope_audit(tool_source),
        "outputs": {
            "frozen_plan": str(PLAN_PATH),
            "result": str(RESULT_PATH),
            "checkpoint": str(CHECKPOINT_PATH),
            "exclusive_create": True,
            "checkpoint_count_max": 1,
        },
        "scope": {
            "optimizer": False,
            "backward": False,
            "battle_API_function_call": False,
            "gameplay": False,
            "H2H": False,
            "package": False,
            "upload": False,
            "submission": False,
            "local_only": True,
        },
    }


def plan_envelope(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION + "-frozen-plan-envelope",
        "plan_sha256": sha256_json(plan),
        "plan": dict(plan),
    }


def load_frozen_plan(
    path: Path,
    expected_plan_sha256: str,
    expected_plan_file_sha256: str,
) -> dict[str, Any]:
    if path.resolve() != PLAN_PATH.resolve():
        raise RuntimeError(f"Frozen v2 plan path must be {PLAN_PATH}")
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing frozen v2 plan: {path}") from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o444
    ):
        raise RuntimeError("Frozen v2 plan must be a single-link 0444 regular file")
    if file_sha256(path) != expected_plan_file_sha256:
        raise RuntimeError("Frozen v2 outer plan file SHA mismatch")
    raw = path.read_bytes()
    envelope = base.strict_json_bytes(raw, "frozen v2 plan")
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"schema_version", "plan_sha256", "plan"}
        or envelope.get("schema_version")
        != SCHEMA_VERSION + "-frozen-plan-envelope"
        or not isinstance(envelope.get("plan"), dict)
        or raw != canonical_json_bytes(envelope)
    ):
        raise RuntimeError("Frozen v2 plan envelope drift")
    plan = envelope["plan"]
    if plan.get("schema_version") != SCHEMA_VERSION + "-plan":
        raise RuntimeError("Frozen v2 inner plan schema drift")
    observed = sha256_json(plan)
    if envelope.get("plan_sha256") != observed or observed != expected_plan_sha256:
        raise RuntimeError("Frozen v2 plan SHA mismatch")
    return plan


def recompute_reference_gradients(
    model: torch.nn.Module,
    parent: Mapping[str, Any],
    specs: Sequence[Any],
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    parameters = training_core.configure_actor6(model)
    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if trainable != ACTOR6 or len(parameters) != len(ACTOR6):
        raise RuntimeError("Reference-gradient scope differs from actor6")
    expected = geometry_gradient_requirements(specs)
    state_before = ppo.model_state_sha256(model)
    gradients_by_mode: dict[str, dict[str, torch.Tensor]] = {
        "natural": {},
        "legacy_8_2": {},
    }
    reports: dict[str, Any] = {}
    for spec in specs:
        gradients, domain_report = probe.gradient_for_domain(
            model,
            parameters,
            spec,
            parent["model_config"],
            device,
        )
        reports[spec.name] = {
            "rows": domain_report["rows"],
            "batches": domain_report["batches"],
            "count_actor6_dependency": domain_report[
                "count_actor6_dependency"
            ],
            "modes": {},
        }
        for mode in ("natural", "legacy_8_2"):
            gradient = gradients[mode].detach().cpu().double()
            record = probe.vector_record(gradient)
            required = expected[spec.name][mode]
            checks = {
                "gradient_hash_exact": record["float64_le_sha256"]
                == required["gradient"]["float64_le_sha256"],
                "gradient_elements_exact": record["elements"]
                == required["gradient"]["elements"],
                "gradient_l2_exact_within_1e_15": math.isclose(
                    record["l2"],
                    required["gradient"]["l2"],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "effective_weight_sum_exact": domain_report[mode][
                    "selection_effective_weight_sum"
                ]
                == required["selection_effective_weight_sum"],
                "all_rows_once": domain_report["feature_checks"][
                    "all_rows_covered_once"
                ]
                is True,
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"Frozen reference gradient drift for {spec.name}/{mode}: "
                    f"{checks}"
                )
            gradients_by_mode[mode][spec.name] = gradient
            reports[spec.name]["modes"][mode] = {
                "gradient": record,
                "selection_effective_weight_sum": domain_report[mode][
                    "selection_effective_weight_sum"
                ],
                "checks": checks,
            }
    state_after = ppo.model_state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("Reference-gradient recomputation changed model state")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Reference-gradient recomputation left .grad buffers")
    return gradients_by_mode, {
        "domains": reports,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_bit_identical": True,
    }


def materialize_candidate_state_v2(
    parent_state: Mapping[str, torch.Tensor],
    direction: torch.Tensor,
    radius: float,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    # These are global lineage/schema invariants and remain fail-hard.
    if radius not in RADIUS_PRIORITY:
        raise RuntimeError(f"Unregistered trust radius: {radius}")
    flat = direction.detach().cpu().double().reshape(-1)
    if (
        flat.numel() != P1_ELEMENTS
        or probe.vector_sha256(flat) != P1_VECTOR_SHA256
        or not math.isclose(
            float(torch.linalg.vector_norm(flat)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("Global frozen P1 vector invariant failed")
    if any(name not in parent_state for name in ACTOR6):
        raise RuntimeError("Parent actor6 schema drift")
    candidate = base.clone_state(parent_state)
    offset = 0
    for name in ACTOR6:
        parent_tensor = parent_state[name].detach().cpu()
        elements = parent_tensor.numel()
        chunk = flat[offset : offset + elements].reshape(parent_tensor.shape)
        candidate[name] = (
            parent_tensor.double() - float(radius) * chunk
        ).to(dtype=parent_tensor.dtype).contiguous()
        offset += elements
    if offset != flat.numel():
        raise RuntimeError("P1 actor mapping/order failed to consume vector")
    if set(candidate) != set(parent_state):
        raise RuntimeError("Candidate/parent state schema drift")
    for name in parent_state:
        if (
            candidate[name].shape != parent_state[name].shape
            or candidate[name].dtype != parent_state[name].dtype
        ):
            raise RuntimeError(f"Candidate tensor schema drift at {name}")
    try:
        integrity = base.state_integrity(
            parent_state, candidate, flat, radius
        )
    except (RuntimeError, FloatingPointError) as error:
        return None, {
            "pass": False,
            "radius_local_integrity_failure": True,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    if not integrity["pass"]:
        return None, {
            **integrity,
            "radius_local_integrity_failure": True,
        }
    integrity["radius_local_integrity_failure"] = False
    return candidate, integrity


def actual_delta_reference_predictions(
    parent_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
    specs: Sequence[Any],
) -> dict[str, Any]:
    delta = probe.flatten_actor6(candidate_state) - probe.flatten_actor6(parent_state)
    if delta.numel() != P1_ELEMENTS or not torch.isfinite(delta).all():
        raise RuntimeError("Actual candidate delta invariant failed")
    records: dict[str, Any] = {}
    for spec in specs:
        records[spec.name] = {}
        for mode in ("natural", "legacy_8_2"):
            gradient = gradients[mode][spec.name].detach().cpu().double()
            prediction = float(torch.dot(gradient, -delta))
            records[spec.name][mode] = {
                "frozen_gradient": probe.vector_record(gradient),
                "actual_float32_delta": probe.vector_record(delta),
                "reference_prediction": prediction,
                "formula": "-dot(g_frozen_cuda_bf16, actual_float32_delta)",
                "reference_gradient_semantics": "S8 CUDA/BF16 frozen geometry",
                "is_CPU_Taylor_prediction": False,
                "is_per_forward_semantics_gradient": False,
            }
    return records


def compare_behavior_view_v2(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    target: str,
    source: str,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    if baseline.get("semantics") != candidate.get("semantics"):
        raise RuntimeError("Baseline/candidate semantics drift")
    if baseline.get("rows") != candidate.get("rows"):
        raise RuntimeError("Baseline/candidate row drift")
    metrics_base = baseline.get("metrics")
    metrics_candidate = candidate.get("metrics")
    if not isinstance(metrics_base, Mapping) or not isinstance(
        metrics_candidate, Mapping
    ):
        raise RuntimeError("Behavior comparison lacks metrics")
    losses: dict[str, Any] = {}
    loss_gates: dict[str, bool] = {}
    for mode in ("natural", "legacy_8_2"):
        base_loss = float(baseline["losses"][mode]["selection_loss"])
        candidate_loss = float(candidate["losses"][mode]["selection_loss"])
        decrease = base_loss - candidate_loss
        prediction = float(reference[mode]["reference_prediction"])
        ratio = decrease / prediction if prediction > 0.0 else None
        losses[mode] = {
            "baseline_global_selection_loss": base_loss,
            "candidate_global_selection_loss": candidate_loss,
            "observed_absolute_decrease": decrease,
            "frozen_geometry_reference_prediction": prediction,
            "frozen_geometry_reference_ratio": ratio,
            "reference_formula": reference[mode]["formula"],
            "reference_gradient_semantics": reference[mode][
                "reference_gradient_semantics"
            ],
            "CPU_Taylor_derivative_claimed": False,
            "per_forward_semantics_gradient_claimed": False,
        }
        loss_gates[f"{mode}_observed_decrease_at_least_2e_7"] = (
            decrease >= ABSOLUTE_LOSS_DECREASE_MIN
        )
        loss_gates[f"{mode}_reference_prediction_positive"] = prediction > 0.0
        loss_gates[f"{mode}_frozen_reference_ratio_at_least_0_10"] = (
            ratio is not None and ratio >= REFERENCE_RATIO_MIN
        )

    accuracy_keys = (
        "set_exact_accuracy",
        "hybrid_order_exact_accuracy",
        "ordered_exact_accuracy",
    )
    accuracy_deltas = {
        key: base._accuracy_delta(metrics_base, metrics_candidate, key)
        for key in accuracy_keys
    }
    correct_keys = (
        "set_exact_correct",
        "hybrid_order_exact_correct",
        "ordered_exact_correct",
    )
    correct_deltas = {
        key: int(metrics_candidate[key]) - int(metrics_base[key])
        for key in correct_keys
    }
    allowed_drop = (
        base.GENERAL_ACCURACY_DROP_MAX if source == "general" else 0.0
    )
    accuracy_gates = {
        f"{key}_drop_within_source_limit": value >= -allowed_drop
        for key, value in accuracy_deltas.items()
    }
    correct_gates = {
        f"{key}_non_decrease_for_specialist": (
            source == "general" or value >= 0
        )
        for key, value in correct_deltas.items()
    }
    team_deltas = base._group_hybrid_deltas(baseline, candidate, "by_team")
    seat_deltas = base._group_hybrid_deltas(baseline, candidate, "by_seat")
    team_macro_base = math.fsum(
        float(record["hybrid_order_exact_accuracy"])
        for record in baseline["by_team"].values()
    ) / len(baseline["by_team"])
    team_macro_candidate = math.fsum(
        float(record["hybrid_order_exact_accuracy"])
        for record in candidate["by_team"].values()
    ) / len(candidate["by_team"])
    subgroup_gates = {
        "each_team_hybrid_drop_at_most_0_005": all(
            value >= -base.GROUP_HYBRID_DROP_MAX for value in team_deltas.values()
        ),
        "each_seat_hybrid_drop_at_most_0_005": all(
            value >= -base.GROUP_HYBRID_DROP_MAX for value in seat_deltas.values()
        ),
        "general_team_macro_hybrid_non_decrease": (
            source != "general" or team_macro_candidate >= team_macro_base
        ),
    }
    identity_gates = {
        "row_keys_identical": baseline.get("row_key_sha256")
        == candidate.get("row_key_sha256"),
        "count_logits_bit_identical": baseline.get(
            "count_logits_float32_le_sha256"
        )
        == candidate.get("count_logits_float32_le_sha256"),
        "action_count_predictions_identical": baseline.get(
            "action_count_prediction_sha256"
        )
        == candidate.get("action_count_prediction_sha256"),
        "count_correct_identical": metrics_base.get("count_correct")
        == metrics_candidate.get("count_correct"),
    }
    gates = (
        loss_gates
        | accuracy_gates
        | correct_gates
        | subgroup_gates
        | identity_gates
    )
    result = {
        "target": target,
        "source": source,
        "semantics": baseline["semantics"],
        "losses": losses,
        "accuracy_deltas": accuracy_deltas,
        "correct_count_deltas": correct_deltas,
        "allowed_accuracy_drop": allowed_drop,
        "team_hybrid_deltas": team_deltas,
        "seat_hybrid_deltas": seat_deltas,
        "team_macro_hybrid": {
            "baseline": team_macro_base,
            "candidate": team_macro_candidate,
            "delta": team_macro_candidate - team_macro_base,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }
    if not all_finite(result):
        raise FloatingPointError("Non-finite v2 behavior gate result")
    return result


def compare_behavior_bundle_v2(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    specs: Sequence[Any],
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_names = {spec.name for spec in specs}
    if set(baseline) != set(SEMANTICS) or set(candidate) != set(SEMANTICS):
        raise RuntimeError("Dual forward semantics are incomplete")
    if set(references) != expected_names:
        raise RuntimeError("Actual-delta reference domains are incomplete")
    reports: dict[str, Any] = {}
    for semantics in SEMANTICS:
        if (
            set(baseline[semantics]) != expected_names
            or set(candidate[semantics]) != expected_names
        ):
            raise RuntimeError("Behavior bundle domain drift")
        reports[semantics] = {
            spec.name: compare_behavior_view_v2(
                baseline[semantics][spec.name],
                candidate[semantics][spec.name],
                target=spec.name,
                source=spec.source,
                reference=references[spec.name],
            )
            for spec in specs
        }
    view_gates = {
        f"{semantics}__{spec.name}": reports[semantics][spec.name]["pass"]
        for semantics in SEMANTICS
        for spec in specs
    }
    loss_items = {
        f"{semantics}__{spec.name}__{mode}": all(
            reports[semantics][spec.name]["gates"][key]
            for key in (
                f"{mode}_observed_decrease_at_least_2e_7",
                f"{mode}_reference_prediction_positive",
                f"{mode}_frozen_reference_ratio_at_least_0_10",
            )
        )
        for semantics in SEMANTICS
        for spec in specs
        for mode in ("natural", "legacy_8_2")
    }
    if len(loss_items) != 12:
        raise RuntimeError("v2 dual-semantics loss items must equal 12")
    return {
        "views": reports,
        "view_gates": view_gates,
        "loss_item_gates": loss_items,
        "loss_item_gate_count": len(loss_items),
        "pass": all(view_gates.values()) and all(loss_items.values()),
    }


def complete_two_pass_gate(
    baseline: Mapping[str, Mapping[str, Any]],
    first: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]],
    specs: Sequence[Any],
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    first_gate = compare_behavior_bundle_v2(
        baseline, first, specs, references
    )
    confirmation = base.confirm_behavior_bundle(first, second, specs)
    second_gate = compare_behavior_bundle_v2(
        baseline, second, specs, references
    )
    gates = {
        "first_complete_gate_pass": first_gate["pass"],
        "first_second_deterministic_confirmation_pass": confirmation["pass"],
        "second_complete_gate_pass": second_gate["pass"],
    }
    return {
        "first_gate": first_gate,
        "confirmation": confirmation,
        "second_gate": second_gate,
        "gates": gates,
        "pass": all(gates.values()),
    }


def build_checkpoint_v2(
    parent: Mapping[str, Any],
    candidate_state: Mapping[str, torch.Tensor],
    *,
    frozen_plan_sha256: str,
    frozen_plan_file_sha256: str,
    radius_label: str,
    radius: float,
    integrity: Mapping[str, Any],
    calibration_record: Mapping[str, Any],
    final_record: Mapping[str, Any],
) -> dict[str, Any]:
    retained = {
        key: copy.deepcopy(parent[key])
        for key in base.CHECKPOINT_RETAIN_KEYS
        if key in parent
    }
    retained.update(
        {
            "model_state_dict": base.clone_state(candidate_state),
            "update": 0,
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_states_omitted": sorted(base.FORBIDDEN_CHECKPOINT_KEYS),
            "s08_p1_tiny_trust_bc_v2": {
                "schema_version": SCHEMA_VERSION,
                "frozen_plan_sha256": frozen_plan_sha256,
                "frozen_plan_file_sha256": frozen_plan_file_sha256,
                "parent": "S8",
                "parent_sha256": base.IMMUTABLE_INPUTS["s8_parent"][1],
                "geometry_plan_sha256": base.GEOMETRY_PLAN_SHA256,
                "geometry_report_sha256": base.GEOMETRY_REPORT_SHA256,
                "P1_float64_le_sha256": P1_VECTOR_SHA256,
                "radius_label": radius_label,
                "radius_l2": radius,
                "trainable_scope": list(ACTOR6),
                "integrity": copy.deepcopy(dict(integrity)),
                "calibration_record_canonical_sha256": sha256_json(
                    calibration_record
                ),
                "final_record_canonical_sha256": sha256_json(final_record),
                "reference_ratio_name": "frozen_geometry_reference_ratio",
                "CPU_Taylor_derivative_claimed": False,
                "first_and_second_complete_gates_pass": True,
                "dual_forward_semantics_pass": True,
                "H2H_eligible": True,
                "H2H_not_performed": True,
                "promotion_eligible": False,
                "package_not_performed": True,
                "upload_not_performed": True,
                "submission_authorized": False,
                "submission_not_performed": True,
            },
        }
    )
    if base.FORBIDDEN_CHECKPOINT_KEYS.intersection(retained):
        raise RuntimeError("v2 checkpoint retained optimizer state")
    if set(retained["model_state_dict"]) != set(parent["model_state_dict"]):
        raise RuntimeError("v2 checkpoint state schema drift")
    return retained


def verify_checkpoint_payload_v2(
    payload: bytes,
    expected_state: Mapping[str, torch.Tensor],
    expected_plan_sha256: str,
    expected_plan_file_sha256: str,
    expected_radius_label: str,
) -> dict[str, Any]:
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Serialized v2 checkpoint root drift")
    metadata = checkpoint.get("s08_p1_tiny_trust_bc_v2")
    state = checkpoint.get("model_state_dict")
    if (
        not isinstance(metadata, dict)
        or not isinstance(state, dict)
        or metadata.get("frozen_plan_sha256") != expected_plan_sha256
        or metadata.get("frozen_plan_file_sha256")
        != expected_plan_file_sha256
        or metadata.get("radius_label") != expected_radius_label
        or metadata.get("first_and_second_complete_gates_pass") is not True
        or metadata.get("H2H_not_performed") is not True
        or metadata.get("submission_not_performed") is not True
        or checkpoint.get("evaluation_only") is not True
        or checkpoint.get("resume_forbidden") is not True
        or base.FORBIDDEN_CHECKPOINT_KEYS.intersection(checkpoint)
    ):
        raise RuntimeError("Serialized v2 checkpoint metadata drift")
    state_sha = base.state_dict_sha256(state)
    if state_sha != base.state_dict_sha256(expected_state):
        raise RuntimeError("Serialized v2 checkpoint state drift")
    return {
        "expected_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_bytes": len(payload),
        "model_state_sha256": state_sha,
        "metadata_canonical_sha256": sha256_json(metadata),
    }


def reserve_empty_checkpoint(path: Path) -> int:
    assert_output_absent(path, "v2 checkpoint reservation")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    observed = os.fstat(descriptor)
    if observed.st_size != 0 or stat.S_IMODE(observed.st_mode) != 0o444:
        os.close(descriptor)
        raise RuntimeError("v2 checkpoint reservation is not empty mode 0444")
    return descriptor


def fill_reserved_checkpoint(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    try:
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("Reserved checkpoint write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_parent_directory(path: Path) -> None:
    parent = path.parent
    try:
        parent_info = os.lstat(parent)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing output parent directory: {parent}"
        ) from error
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise RuntimeError("Output parent must be a non-symlink directory")
    if not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("Directory fsync requires os.O_DIRECTORY")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(parent, flags)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError("Opened output parent is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_rejection_result(
    result: dict[str, Any], initial_hashes: Mapping[str, str]
) -> None:
    final_hashes = snapshot_input_hashes(include_frozen_plan=True)
    if dict(initial_hashes) != final_hashes:
        raise RuntimeError("A v2 frozen input changed before rejection evidence")
    result["final_input_hashes"] = final_hashes
    result["inputs_unchanged_before_result"] = True
    result["checkpoint_validity"] = {
        "valid": False,
        "reason": "candidate rejected before checkpoint reservation",
    }
    if not all_finite(result):
        raise FloatingPointError("v2 rejection result contains non-finite values")
    base.write_exclusive(RESULT_PATH, canonical_json_bytes(result))


def publish_passing_result_then_checkpoint(
    result: dict[str, Any],
    payload: bytes,
    checkpoint_audit: Mapping[str, Any],
    initial_hashes: Mapping[str, str],
) -> dict[str, Any]:
    descriptor = reserve_empty_checkpoint(CHECKPOINT_PATH)
    descriptor_open = True
    try:
        pre_result_hashes = snapshot_input_hashes(include_frozen_plan=True)
        if dict(initial_hashes) != pre_result_hashes:
            raise RuntimeError("A v2 frozen input changed before pass result")
        result["final_input_hashes"] = pre_result_hashes
        result["inputs_unchanged_before_result"] = True
        result["checkpoint_publication"] = {
            "path": str(CHECKPOINT_PATH),
            **dict(checkpoint_audit),
            "result_written_before_checkpoint_payload": True,
            "result_parent_directory_fsync_required_before_checkpoint_payload": True,
            "valid_iff": (
                "checkpoint is a single-link 0444 regular file whose byte size "
                "equals expected_bytes and SHA-256 equals expected_payload_sha256"
            ),
            "H2H_eligible_iff_valid": True,
        }
        result["status"] = "candidate_passed_checkpoint_valid_iff_result_contract"
        if not all_finite(result):
            raise FloatingPointError("v2 passing result contains non-finite values")
        base.write_exclusive(RESULT_PATH, canonical_json_bytes(result))
        # Persist both the immutable result directory entry and the previously
        # reserved empty checkpoint entry before any valid checkpoint bytes can
        # become durable.  File fsync alone does not order a newly created
        # directory entry across power loss or a kernel crash.
        fsync_parent_directory(RESULT_PATH)

        # A second terminal check after the immutable result ensures any drift
        # leaves the reserved checkpoint empty and therefore invalid by that
        # result's own contract.
        post_result_hashes = snapshot_input_hashes(include_frozen_plan=True)
        if dict(initial_hashes) != post_result_hashes:
            raise RuntimeError("A v2 frozen input changed after pass result")
        fill_reserved_checkpoint(descriptor, payload)
        descriptor_open = False
    finally:
        if descriptor_open:
            try:
                os.close(descriptor)
            except OSError:
                pass
    info = os.lstat(CHECKPOINT_PATH)
    observed = {
        "sha256": file_sha256(CHECKPOINT_PATH),
        "bytes": info.st_size,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
        "regular": stat.S_ISREG(info.st_mode),
        "symlink": stat.S_ISLNK(info.st_mode),
    }
    valid = (
        observed["sha256"] == checkpoint_audit["expected_payload_sha256"]
        and observed["bytes"] == checkpoint_audit["expected_bytes"]
        and observed["mode"] == 0o444
        and observed["nlink"] == 1
        and observed["regular"]
        and not observed["symlink"]
    )
    if not valid:
        raise RuntimeError(f"Published v2 checkpoint violates result contract: {observed}")
    return {"observed": observed, "valid_by_result_contract": True}


def make_result_base(
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    initial_hashes: Mapping[str, str],
    direction_report: Mapping[str, Any],
    calibration_gradient_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION + "-result",
        "frozen_plan_sha256": sha256_json(plan),
        "frozen_plan_file_sha256": plan_file_sha256,
        "geometry_plan_sha256": base.GEOMETRY_PLAN_SHA256,
        "geometry_report_sha256": base.GEOMETRY_REPORT_SHA256,
        "direction_reproduction": copy.deepcopy(dict(direction_report)),
        "calibration_reference_gradients": copy.deepcopy(
            dict(calibration_gradient_report)
        ),
        "reference_ratio_name": "frozen_geometry_reference_ratio",
        "CPU_Taylor_derivative_claimed": False,
        "initial_input_hashes": dict(initial_hashes),
        "radius_priority": [RADIUS_LABELS[radius] for radius in RADIUS_PRIORITY],
        "calibration": {},
        "selected_radius_label": None,
        "selected_radius": None,
        "final": None,
        "H2H_performed": False,
        "gameplay_performed": False,
        "package_performed": False,
        "upload_performed": False,
        "submission_performed": False,
    }


def execute(
    plan: dict[str, Any],
    *,
    frozen_plan_file_sha256: str,
) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    assert_output_absent(RESULT_PATH, "v2 materialization result")
    assert_output_absent(CHECKPOINT_PATH, "v2 materialized checkpoint")
    if plan != build_plan():
        raise RuntimeError("Frozen v2 plan no longer matches rebuilt plan")
    initial_hashes = snapshot_input_hashes(include_frozen_plan=True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal v2 materialization requires cuda:0")
    parent = torch.load(base.S8_PARENT, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(
        base.BC_ARCHITECTURE, map_location="cpu", weights_only=False
    )
    if not isinstance(parent, dict) or not isinstance(bc_checkpoint, dict):
        raise RuntimeError("Bound checkpoint roots must be dictionaries")
    parent_state_raw = parent.get("model_state_dict")
    if not isinstance(parent_state_raw, dict):
        raise RuntimeError("S8 lacks model_state_dict")
    parent_state = base.clone_state(parent_state_raw)

    direction, direction_report, cuda_model = base.recompute_p1_direction(
        parent, bc_checkpoint, cuda_device
    )
    cpu_model = ppo.instantiate_model_from_checkpoint(
        parent, bc_checkpoint, cpu_device
    )
    models = {"cpu_fp32": cpu_model, "cuda_bf16": cuda_model}
    devices = {"cpu_fp32": cpu_device, "cuda_bf16": cuda_device}

    base.apply_state(cuda_model, parent_state)
    calibration_gradients, calibration_gradient_report = (
        recompute_reference_gradients(
            cuda_model, parent, CAL_SPECS, cuda_device
        )
    )
    result = make_result_base(
        plan,
        frozen_plan_file_sha256,
        initial_hashes,
        direction_report,
        calibration_gradient_report,
    )
    result["parent_model_state_sha256"] = base.state_dict_sha256(parent_state)
    calibration_baseline = base.evaluate_bundle(
        models,
        devices,
        parent_state,
        parent["model_config"],
        CAL_SPECS,
    )
    result["calibration"]["S8_baseline"] = calibration_baseline
    candidate_states: dict[str, dict[str, torch.Tensor]] = {}
    radius_records: dict[str, dict[str, Any]] = {}

    for radius in RADIUS_PRIORITY:
        label = RADIUS_LABELS[radius]
        if label == "R25" and radius_records["R50"][
            "complete_calibration_pass"
        ]:
            raise RuntimeError("R25 evaluation attempted after R50 passed")
        state, integrity = materialize_candidate_state_v2(
            parent_state, direction, radius
        )
        if state is None:
            radius_records[label] = {
                "radius": radius,
                "integrity": integrity,
                "reference_predictions": None,
                "first_evaluation": None,
                "first_gate": None,
                "second_evaluation": None,
                "complete_two_pass_gate": None,
                "complete_calibration_pass": False,
                "failure_stage": "radius_local_materialization_integrity",
            }
            result["calibration"][label] = radius_records[label]
            continue
        references = actual_delta_reference_predictions(
            parent_state, state, calibration_gradients, CAL_SPECS
        )
        first = base.evaluate_bundle(
            models,
            devices,
            state,
            parent["model_config"],
            CAL_SPECS,
        )
        first_gate = compare_behavior_bundle_v2(
            calibration_baseline, first, CAL_SPECS, references
        )
        second: dict[str, Any] | None = None
        complete: dict[str, Any] | None = None
        if first_gate["pass"]:
            second = base.evaluate_bundle(
                models,
                devices,
                state,
                parent["model_config"],
                CAL_SPECS,
            )
            complete = complete_two_pass_gate(
                calibration_baseline,
                first,
                second,
                CAL_SPECS,
                references,
            )
        complete_pass = bool(
            integrity["pass"] and complete is not None and complete["pass"]
        )
        radius_records[label] = {
            "radius": radius,
            "integrity": integrity,
            "reference_predictions": references,
            "first_evaluation": first,
            "first_gate": first_gate,
            "second_evaluation": second,
            "complete_two_pass_gate": complete,
            "complete_calibration_pass": complete_pass,
            "failure_stage": None if complete_pass else (
                "first_gate" if not first_gate["pass"] else "second_or_confirmation_gate"
            ),
        }
        result["calibration"][label] = radius_records[label]
        if complete_pass:
            candidate_states[label] = state
            break

    selected_label = base.select_radius_from_records(radius_records)
    if selected_label is None:
        result.update(
            {
                "status": "calibration_rejected",
                "final": {
                    "opened": False,
                    "reason": "no radius passed complete dual-semantics two-pass calibration",
                },
                "checkpoint_reserved": False,
            }
        )
        publish_rejection_result(result, initial_hashes)
        return 42, result, None

    selected_radius = next(
        radius for radius in RADIUS_PRIORITY if RADIUS_LABELS[radius] == selected_label
    )
    selected_state = candidate_states[selected_label]
    result["selected_radius_label"] = selected_label
    result["selected_radius"] = selected_radius
    result["calibration_selection"] = {
        "selected": selected_label,
        "priority": [RADIUS_LABELS[radius] for radius in RADIUS_PRIORITY],
        "R25_evaluated": "R25" in radius_records,
        "selected_before_final_gradient_or_behavior_iteration": True,
        "final_used_for_selection": False,
    }

    # Only now may the final domains participate.  They can reject this one
    # locked radius but can never select or revive another radius.
    base.apply_state(cuda_model, parent_state)
    final_gradients, final_gradient_report = recompute_reference_gradients(
        cuda_model, parent, FINAL_SPECS, cuda_device
    )
    final_references = actual_delta_reference_predictions(
        parent_state, selected_state, final_gradients, FINAL_SPECS
    )
    final_baseline = base.evaluate_bundle(
        models,
        devices,
        parent_state,
        parent["model_config"],
        FINAL_SPECS,
    )
    final_first = base.evaluate_bundle(
        models,
        devices,
        selected_state,
        parent["model_config"],
        FINAL_SPECS,
    )
    final_first_gate = compare_behavior_bundle_v2(
        final_baseline, final_first, FINAL_SPECS, final_references
    )
    final_second: dict[str, Any] | None = None
    final_complete: dict[str, Any] | None = None
    if final_first_gate["pass"]:
        final_second = base.evaluate_bundle(
            models,
            devices,
            selected_state,
            parent["model_config"],
            FINAL_SPECS,
        )
        final_complete = complete_two_pass_gate(
            final_baseline,
            final_first,
            final_second,
            FINAL_SPECS,
            final_references,
        )
    final_pass = bool(final_complete is not None and final_complete["pass"])
    final_record = {
        "opened": True,
        "opened_after_unique_radius_lock": True,
        "selected_radius_label": selected_label,
        "reference_gradients": final_gradient_report,
        "reference_predictions": final_references,
        "S8_baseline": final_baseline,
        "first_evaluation": final_first,
        "first_gate": final_first_gate,
        "second_evaluation": final_second,
        "complete_two_pass_gate": final_complete,
        "pass": final_pass,
        "fallback_allowed": False,
    }
    result["final"] = final_record
    if not final_pass:
        result.update(
            {
                "status": "final_rejected",
                "checkpoint_reserved": False,
                "final_failure_did_not_trigger_radius_fallback": True,
            }
        )
        publish_rejection_result(result, initial_hashes)
        return 42, result, None

    checkpoint = build_checkpoint_v2(
        parent,
        selected_state,
        frozen_plan_sha256=sha256_json(plan),
        frozen_plan_file_sha256=frozen_plan_file_sha256,
        radius_label=selected_label,
        radius=selected_radius,
        integrity=radius_records[selected_label]["integrity"],
        calibration_record=radius_records[selected_label],
        final_record=final_record,
    )
    checkpoint_payload = base.serialize_checkpoint(checkpoint)
    checkpoint_audit = verify_checkpoint_payload_v2(
        checkpoint_payload,
        selected_state,
        sha256_json(plan),
        frozen_plan_file_sha256,
        selected_label,
    )
    result["checkpoint_reserved"] = True
    publication = publish_passing_result_then_checkpoint(
        result,
        checkpoint_payload,
        checkpoint_audit,
        initial_hashes,
    )
    return 0, result, publication


def enforce_runtime(*, formal_cuda: bool) -> None:
    base.enforce_runtime(formal_cuda=formal_cuda)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--print-plan", action="store_true")
    actions.add_argument("--freeze-plan", action="store_true")
    actions.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--expected-plan-file-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    enforce_runtime(formal_cuda=args.execute)
    if args.print_plan:
        plan = build_plan()
        print(canonical_json_bytes(plan).decode("utf-8"), end="")
        return 0
    if args.freeze_plan:
        if args.plan.resolve() != PLAN_PATH.resolve():
            raise RuntimeError(f"Frozen v2 plan path must be {PLAN_PATH}")
        assert_output_absent(PLAN_PATH, "frozen v2 plan")
        plan = build_plan()
        envelope = plan_envelope(plan)
        base.write_exclusive(PLAN_PATH, canonical_json_bytes(envelope))
        info = os.lstat(PLAN_PATH)
        if stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
            raise RuntimeError("Published v2 plan mode/link drift")
        print(
            json.dumps(
                {
                    "path": str(PLAN_PATH),
                    "plan_sha256": envelope["plan_sha256"],
                    "file_sha256": file_sha256(PLAN_PATH),
                    "mode": oct(stat.S_IMODE(info.st_mode)),
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.expected_plan_sha256 or not args.expected_plan_file_sha256:
        raise RuntimeError(
            "--execute requires --expected-plan-sha256 and "
            "--expected-plan-file-sha256"
        )
    plan = load_frozen_plan(
        args.plan,
        args.expected_plan_sha256,
        args.expected_plan_file_sha256,
    )
    code, result, publication = execute(
        plan,
        frozen_plan_file_sha256=args.expected_plan_file_sha256,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_radius_label": result.get("selected_radius_label"),
                "checkpoint_contract_valid": (
                    publication.get("valid_by_result_contract")
                    if publication is not None
                    else False
                ),
                "result": str(RESULT_PATH),
            },
            sort_keys=True,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
