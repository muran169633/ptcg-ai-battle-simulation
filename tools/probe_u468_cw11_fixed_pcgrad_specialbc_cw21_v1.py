#!/usr/bin/env python3
"""One exact-CW11 fixed-PCGrad cache screen, train-only and stdout-only.

The exact CW11 state is reconstructed only inside the already-consumed historical
callback chain.  This script never resumes from the materialized evaluation-only
CW11 checkpoint and never opens changed-candidate validation data.
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
sys.pycache_prefix = "/dev/null"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_fixed_pcgrad_specialbc_cw21_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-fixed-pcgrad-specialbc-cw21-v1"
SEED = 202608303

CW20 = TOOLS / "probe_u468_raw_fixed_pcgrad_specialbc_cw20_v1.py"
CW20_SHA256 = "8f5b64c8a04b3de9a6d2890aac582967a50f4d708f3a11a64cbd395423d907d2"
CW20_MODE = 0o555
CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
CW11_VECTOR_SHA256 = "e7183412ec13e9b69b2b50c95b4ebc6fc603a3d934229ca47b00538abb95abcd"
CW11_LEDGER_SHA256 = "6ae0c07a94627cb6fd0155f265f75d13bd7505b0e5f62cd62c3e26ceb7714436"
CW11_ACTOR_FLOAT32_LE_SHA256 = (
    "fc13661fec1801768a11d2c0366602727df08d6773c982f2ba273e194b8fafa6"
)
RAW_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
RAW_NONACTOR_SHA256 = "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"

TASK_ORDER = ("union_mixed", "flg_hard", "pokemonfan_hard", "core5_hard")
EXPECTED_ACTOR_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
RETENTION_OBJECTIVE = "union_retention"
PLANNED_STEP_L2 = 1.25e-4
ACTUAL_STEP_L2_MIN = 1.245e-4
ACTUAL_STEP_L2_MAX = 1.255e-4
ADDITIONAL_FROM_CW11_CAP = 1.0e-3
FIRST_ORDER_COSINE_EPSILON = 1e-12


class ProtocolError(RuntimeError):
    """Fail-closed CW21 protocol error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_evidence(path: Path, expected_sha: str, expected_mode: int) -> dict[str, Any]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
    ):
        raise ProtocolError(f"identity/mode drift: {path}")
    digest = sha256_file(path)
    after = path.lstat()
    checks = {
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "sha_exact": digest == expected_sha,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"file drift: {path}: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": int(after.st_size),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def import_cw20() -> tuple[ModuleType, dict[str, Any]]:
    evidence = regular_evidence(CW20, CW20_SHA256, CW20_MODE)
    spec = importlib.util.spec_from_file_location("cw21_frozen_cw20", CW20)
    if spec is None or spec.loader is None:
        raise ProtocolError("cannot import frozen CW20 helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, evidence


def runtime_audit(cw20: ModuleType, require_cuda: bool) -> dict[str, Any]:
    result = cw20.runtime_audit(require_cuda=require_cuda)
    device = Path("/dev/null").stat()
    cache_checks = {
        "pycache_prefix_exact_dev_null": sys.pycache_prefix == "/dev/null",
        "dont_write_bytecode": sys.dont_write_bytecode is True,
        "dev_null_character_device": stat.S_ISCHR(device.st_mode),
        "dev_null_device_exact_1_3": (os.major(device.st_rdev), os.minor(device.st_rdev))
        == (1, 3),
    }
    if not all(cache_checks.values()):
        raise ProtocolError(f"bytecode-cache isolation drift: {cache_checks}")
    result["bytecode_cache_isolation"] = {
        "checks": cache_checks,
        "pass": True,
        "policy": "repo_imports_compile_from_source_because_cache_prefix_is_ENOTDIR",
    }
    return result


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden = {
        "save",
        "load_state_dict",
        "backward",
        "step",
        "evaluate_candidate_once",
        "package_submission",
        "upload",
        "submit",
    }
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else None
        )
        if name in forbidden:
            hits.append(f"{name}@{node.lineno}")
    direct_torch_loads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    ]
    train_cache_eval_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate_selected_union"
    ]
    historical_run_probe_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_probe"
    ]
    checks = {
        "no_forbidden_named_calls": not hits,
        "no_direct_torch_checkpoint_load": not direct_torch_loads,
        "exact_two_train_cache_evaluation_sites": len(train_cache_eval_lines) == 2,
        "exact_one_historical_run_probe_site": len(historical_run_probe_lines) == 1,
        "stdout_print_present": any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            for node in ast.walk(tree)
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"source audit failed: {checks}: {hits}")
    return {
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_bytes": len(source),
        "checks": checks,
        "forbidden_hits": hits,
        "direct_torch_load_lines": direct_torch_loads,
        "train_cache_evaluation_lines": train_cache_eval_lines,
        "historical_run_probe_lines": historical_run_probe_lines,
        "pass": True,
    }


def _run_callback_core(
    context: Mapping[str, Any],
    cw20: ModuleType,
    cw19: ModuleType,
    cw15: ModuleType,
    grad: ModuleType,
    cache: Sequence[Mapping[str, Any]],
    selections: Sequence[Sequence[Mapping[str, Any]]],
    inventory: Mapping[str, Mapping[str, Any]],
    modules: Mapping[str, ModuleType],
) -> dict[str, Any]:
    import numpy as np

    required_context = {
        "helper",
        "model",
        "checkpoint",
        "model_config",
        "raw_actor",
        "raw_model_state_sha256",
        "raw_nonactor_sha256",
        "terminal_cumulative_float64",
        "terminal_cumulative_float64_le_sha256",
        "success_iteration",
        "cw10_success_iteration",
        "selected_row_gate",
        "active_pair_ledger",
        "expanded_row_count",
        "candidate_model_state_sha256",
    }
    if not required_context.issubset(context):
        raise ProtocolError("historical CW11 callback context schema drift")
    torch = context["helper"].torch
    model = context["model"]
    device = next(model.parameters()).device
    state = model.state_dict()
    cw11_model_hash = context["helper"].model_state_sha256(model.state_dict())
    terminal_vector = np.asarray(
        context["terminal_cumulative_float64"], dtype=np.float64
    )
    ledger_hash = hashlib.sha256(
        cw15.canonical_json(context["active_pair_ledger"])
    ).hexdigest()
    raw_checkpoint = context["checkpoint"]
    raw_checkpoint_state = (
        raw_checkpoint.get("model_state_dict")
        if isinstance(raw_checkpoint, Mapping)
        else None
    )
    raw_checkpoint_hash = (
        context["helper"].model_state_sha256(raw_checkpoint_state)
        if isinstance(raw_checkpoint_state, Mapping)
        else None
    )
    context_checks = {
        "device_cuda": device.type == "cuda",
        "exact_CW11_model": cw11_model_hash == CW11_MODEL_SHA256,
        "context_candidate_model_exact_CW11": str(
            context["candidate_model_state_sha256"]
        )
        == CW11_MODEL_SHA256,
        "exact_CW11_vector_field": str(
            context["terminal_cumulative_float64_le_sha256"]
        )
        == CW11_VECTOR_SHA256,
        "exact_CW11_vector_recomputed": cw15.float64_vector_sha256(
            terminal_vector
        )
        == CW11_VECTOR_SHA256,
        "CW11_vector_shape_finite": terminal_vector.shape
        == (cw15.ACTOR6_FLAT_LENGTH,)
        and bool(np.isfinite(terminal_vector).all()),
        "CW11_vector_l2_exact": math.isclose(
            float(np.linalg.norm(terminal_vector)),
            0.00792176975336988,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "exact_CW11_ledger": ledger_hash == CW11_LEDGER_SHA256,
        "raw_model_anchor": str(context["raw_model_state_sha256"])
        == RAW_MODEL_SHA256,
        "raw_nonactor_anchor": str(context["raw_nonactor_sha256"])
        == RAW_NONACTOR_SHA256,
        "model_eval": model.training is False,
        "legacy_ledger34": len(context["active_pair_ledger"]) == 34,
        "success_iteration_exact3": int(context["success_iteration"]) == 3,
        "cw10_success_iteration_exact9": int(context["cw10_success_iteration"]) == 9,
        "expanded_row_count_exact33": int(context["expanded_row_count"]) == 33,
        "selected_row_gate_pass": bool(context["selected_row_gate"].get("pass")),
        "model_tensor_count80": len(state) == 80,
        "raw_parent_checkpoint_dictionary": isinstance(raw_checkpoint, Mapping),
        "raw_parent_checkpoint_update468": isinstance(raw_checkpoint, Mapping)
        and int(raw_checkpoint.get("update", -1)) == 468,
        "raw_parent_checkpoint_model_exact": raw_checkpoint_hash
        == RAW_MODEL_SHA256,
        "raw_parent_is_not_eval_only_materialization": isinstance(
            raw_checkpoint, Mapping
        )
        and "evaluation_only" not in raw_checkpoint
        and "resume_forbidden" not in raw_checkpoint,
    }
    if not all(context_checks.values()):
        raise ProtocolError(f"exact CW11 callback context drift: {context_checks}")

    named_parameters = dict(model.named_parameters())
    requires_grad_before = {
        name: bool(parameter.requires_grad)
        for name, parameter in named_parameters.items()
    }
    if any(parameter.grad is not None for parameter in named_parameters.values()):
        raise ProtocolError("incoming exact CW11 unexpectedly has gradient buffers")
    names = tuple(grad.sweep.ACTOR_NAMES)
    parameters = grad.sweep.configure_actor6(model)
    parameters_sequence = modules["geometry"].configure_actor6(model)
    anchor_checks = {
        "actor_names_exact": names == EXPECTED_ACTOR_NAMES == tuple(cw15.ACTOR6_NAMES),
        "actor_implementations_same_tensors": all(
            parameters[name] is parameter
            for name, parameter in zip(names, parameters_sequence)
        ),
        "actor_dimension_exact": sum(
            int(parameters[name].numel()) for name in names
        )
        == cw20.ACTOR_DIMENSION,
        "actor6_all_float32": all(
            parameters[name].dtype == torch.float32 for name in names
        ),
        "nonactor_tensor_count74": len(state) - len(names) == 74,
        "CW11_nonactor_exact_raw": cw20.nonactor_sha(
            model, names, grad.sweep.ppo
        )
        == RAW_NONACTOR_SHA256,
    }
    if not all(anchor_checks.values()):
        raise ProtocolError(f"exact CW11 actor anchor drift: {anchor_checks}")
    cw11_actor = cw20.clone_actor(parameters, names)
    cw11_flat = cw20.flat_actor(parameters, names, np)
    cw11_actor_bytes = cw20.actor_bytes(parameters, names, np)
    if hashlib.sha256(cw11_actor_bytes).hexdigest() != CW11_ACTOR_FLOAT32_LE_SHA256:
        raise ProtocolError("exact CW11 actor bytes drift")
    precision_before = torch.get_float32_matmul_precision()
    result: dict[str, Any]
    try:
        torch.set_float32_matmul_precision("high")
        if torch.get_float32_matmul_precision() != "high":
            raise ProtocolError("failed to enter high matmul precision scope")
        baseline_report, baseline_correct = grad.sweep.evaluate_selected_union(
            model, cache, selections, device
        )
        objective_gradients, baseline_losses, autograd_audit = (
            grad.compute_objective_gradients(
                model, parameters, cache, selections, inventory, device
            )
        )
        if context["helper"].model_state_sha256(model.state_dict()) != cw11_model_hash:
            raise ProtocolError("gradient computation changed CW11")
        retention_gradient, retention_loss, retention_weight = (
            grad.derive_union_retention(objective_gradients, baseline_losses, inventory)
        )
        effect_gradients = {
            **objective_gradients,
            RETENTION_OBJECTIVE: retention_gradient,
        }
        pcgrad, pcgrad_audit = grad.fixed_order_pcgrad(objective_gradients)
        effects = grad.candidate_effects(pcgrad, effect_gradients)
        direction = cw19.flatten_gradient(pcgrad, names, np)
        direction_l2 = float(np.linalg.norm(direction))
        direction_gate = {
            "task_order_exact": tuple(pcgrad_audit["task_order"]) == TASK_ORDER,
            "original_unprojected_references": pcgrad_audit["reference_gradient_kind"]
            == "original_unprojected",
            "arithmetic_mean": pcgrad_audit["aggregation"]
            == "arithmetic_mean_of_projected_task_gradients",
            "four_tasks_plus_retention_robust_descent": effects[
                "all_directional_gates_numerically_robust_first_order_descent"
            ]
            is True,
            "minimum_directional_gate_cosine": float(
                effects["minimum_directional_gate_cosine"]
            )
            > FIRST_ORDER_COSINE_EPSILON,
            "direction_finite_nonzero": math.isfinite(direction_l2)
            and direction_l2 > 0.0,
        }
        common = {
            "context_checks": context_checks,
            "anchor_checks": anchor_checks,
            "baseline_train512": baseline_report,
            "baseline_losses": {
                **{name: float(value) for name, value in baseline_losses.items()},
                RETENTION_OBJECTIVE: float(retention_loss),
            },
            "autograd_audit": autograd_audit,
            "pcgrad": {
                "direction_gate": direction_gate,
                "direction_l2": direction_l2,
                "direction_float64_le_sha256": cw19.float64_sha(direction, np),
                "audit": pcgrad_audit,
                "effects": effects,
                "retention_effective_weight": float(retention_weight),
            },
        }
        if not all(direction_gate.values()):
            result = {
                **common,
                "decision": "NO_GO_CW21_CACHE512_PREFLIGHT",
                "reason": "PCGRAD_DIRECTION_GATE_FAILED",
                "candidate_payload": None,
                "changed_candidate_train_endpoint_count": 0,
            }
        else:
            planned_delta = -PLANNED_STEP_L2 * direction / direction_l2
            cw20.apply_flat_actor(
                parameters, names, cw11_flat + planned_delta, torch
            )
            candidate_hash = context["helper"].model_state_sha256(model.state_dict())
            actual_flat = cw20.flat_actor(parameters, names, np)
            actual_delta = actual_flat - cw11_flat
            actual_l2 = float(np.linalg.norm(actual_delta))
            changed_names = [
                name
                for name in names
                if not torch.equal(parameters[name].detach(), cw11_actor[name])
            ]
            actual_predicted = {
                name: -float(
                    cw19.flatten_gradient(effect_gradients[name], names, np)
                    @ actual_delta
                )
                for name in (*TASK_ORDER, RETENTION_OBJECTIVE)
            }
            integrity_checks = {
                "planned_step_l2_exact": math.isclose(
                    float(np.linalg.norm(planned_delta)),
                    PLANNED_STEP_L2,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "actual_step_l2_in_frozen_interval": ACTUAL_STEP_L2_MIN
                <= actual_l2
                <= ACTUAL_STEP_L2_MAX,
                "additional_from_CW11_below_total_cap": actual_l2
                <= ADDITIONAL_FROM_CW11_CAP,
                "changed_scope_exact_actor6": tuple(changed_names) == names,
                "nonactor_exact_raw": cw20.nonactor_sha(
                    model, names, grad.sweep.ppo
                )
                == RAW_NONACTOR_SHA256,
                "candidate_model_changed": candidate_hash != cw11_model_hash,
                "all_finite": bool(np.isfinite(actual_flat).all()),
                "no_clip_or_parameter_space_projection": True,
                "actual_four_tasks_plus_retention_first_order_improvement": all(
                    actual_predicted[name] > 0.0
                    for name in (*TASK_ORDER, RETENTION_OBJECTIVE)
                ),
            }
            candidate_report, candidate_correct = grad.sweep.evaluate_selected_union(
                model, cache, selections, device
            )
            train_gate = cw20.endpoint_gate(
                cw19,
                baseline_report,
                candidate_report,
                baseline_correct,
                candidate_correct,
                selections,
            )
            pre_payload_checks = {
                "integrity": all(integrity_checks.values()),
                "CW11_baseline_retention_contract": train_gate["checks"][
                    "raw_retention_contract_exact"
                ]
                is True,
                "cache512_train_gate": train_gate["pass"] is True,
            }
            final_checks = dict(pre_payload_checks)
            payload = None
            reconstruction = None
            if all(pre_payload_checks.values()):
                layout = cw20.actor_layout(parameters, names)
                candidate_actor_bytes = cw20.actor_bytes(parameters, names, np)
                payload = {
                    "anchor": {
                        "reconstruction_base": "original_raw_U468",
                        "raw_checkpoint": str(grad.sweep.U468.relative_to(ROOT)),
                        "raw_checkpoint_sha256": grad.sweep.U468_SHA256,
                        "raw_model_state_sha256": RAW_MODEL_SHA256,
                        "CW11_provenance_model_state_sha256": cw11_model_hash,
                        "CW11_provenance_vector_float64_le_sha256": CW11_VECTOR_SHA256,
                        "CW11_provenance_active_pair_ledger_sha256": CW11_LEDGER_SHA256,
                        "CW11_provenance_actor_float32_le_sha256": (
                            CW11_ACTOR_FLOAT32_LE_SHA256
                        ),
                        "CW11_materialized_eval_only_checkpoint_used": False,
                        "terminal_model_state_sha256": candidate_hash,
                    },
                    "formula": (
                        "load_original_raw_U468_then_replace_only_six_absolute_"
                        "actor_float32_tensors_from_frozen_payload"
                    ),
                    "actor_names": list(names),
                    "actor_layout": layout,
                    "actor_layout_sha256": hashlib.sha256(
                        json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "CW11_actor_float32_le_sha256": hashlib.sha256(
                        cw11_actor_bytes
                    ).hexdigest(),
                    "candidate_actor_float32_le": cw20.xz_payload(
                        candidate_actor_bytes
                    ),
                    "planned_step_l2": PLANNED_STEP_L2,
                    "actual_additional_from_CW11_l2": actual_l2,
                    "actual_delta_float64_le_sha256": cw19.float64_sha(
                        actual_delta, np
                    ),
                    "pcgrad_direction_float64_le_sha256": cw19.float64_sha(
                        direction, np
                    ),
                }
                modules["cutting"].restore_raw_actor(
                    modules["ram"], parameters_sequence, context["raw_actor"], torch
                )
                restored_hash = context["helper"].model_state_sha256(model.state_dict())
                decoded = cw20.decode_xz(payload["candidate_actor_float32_le"])
                cw20.copy_actor_bytes(parameters, names, layout, decoded, np, torch)
                reconstructed_hash = context["helper"].model_state_sha256(
                    model.state_dict()
                )
                reconstruction_checks = {
                    "raw_U468_restore_exact": restored_hash == RAW_MODEL_SHA256,
                    "terminal_hash_exact": reconstructed_hash == candidate_hash,
                    "actor_bytes_exact": cw20.actor_bytes(parameters, names, np)
                    == candidate_actor_bytes,
                    "nonactor_still_exact_raw": cw20.nonactor_sha(
                        model, names, grad.sweep.ppo
                    )
                    == RAW_NONACTOR_SHA256,
                }
                reconstruction = {
                    "checks": reconstruction_checks,
                    "pass": all(reconstruction_checks.values()),
                    "reconstruction_base_model_state_sha256": restored_hash,
                    "model_state_sha256": reconstructed_hash,
                }
                final_checks["pure_payload_reconstruction"] = reconstruction["pass"]
            passed = bool(final_checks) and all(final_checks.values())
            reason = (
                "all_cache512_train_only_gates_passed"
                if passed
                else "integrity_gate_failed"
                if not pre_payload_checks["integrity"]
                else "cache512_train_gate_failed"
                if not pre_payload_checks["cache512_train_gate"]
                else "payload_reconstruction_gate_failed"
            )
            result = {
                **common,
                "decision": (
                    "GO_CW21_CACHE512_PREFLIGHT"
                    if passed
                    else "NO_GO_CW21_CACHE512_PREFLIGHT"
                ),
                "reason": reason,
                "candidate_model_state_sha256": candidate_hash,
                "candidate_train512": candidate_report,
                "planned_step": {
                    "l2": float(np.linalg.norm(planned_delta)),
                    "float64_le_sha256": cw19.float64_sha(planned_delta, np),
                },
                "actual_step": {
                    "l2": actual_l2,
                    "float64_le_sha256": cw19.float64_sha(actual_delta, np),
                    "changed_parameter_names": changed_names,
                    "predicted_loss_improvements": actual_predicted,
                },
                "integrity_checks": integrity_checks,
                "train_gate": train_gate,
                "pre_payload_checks": pre_payload_checks,
                "final_checks": final_checks,
                "pure_payload_reconstruction": reconstruction,
                "candidate_payload": payload if passed else None,
                "changed_candidate_train_endpoint_count": 1,
            }
    finally:
        cw20.restore_actor(parameters, names, cw11_actor, torch)
        for name, parameter in named_parameters.items():
            parameter.grad = None
            parameter.requires_grad_(requires_grad_before[name])
        torch.set_float32_matmul_precision(precision_before)
        restore_checks = {
            "exact_CW11_model": context["helper"].model_state_sha256(
                model.state_dict()
            )
            == cw11_model_hash,
            "nonactor_exact_raw": cw20.nonactor_sha(
                model, names, grad.sweep.ppo
            )
            == RAW_NONACTOR_SHA256,
            "model_eval": model.training is False,
            "gradient_buffers_none": all(
                parameter.grad is None for parameter in named_parameters.values()
            ),
            "requires_grad_flags_restored": all(
                bool(parameter.requires_grad) == requires_grad_before[name]
                for name, parameter in named_parameters.items()
            ),
            "matmul_precision_restored": torch.get_float32_matmul_precision()
            == precision_before,
        }
        if not all(restore_checks.values()):
            raise ProtocolError(f"failed final exact-CW11 restore: {restore_checks}")
    return result


def run_callback(
    context: Mapping[str, Any],
    cw20: ModuleType,
    cw19: ModuleType,
    cw15: ModuleType,
    grad: ModuleType,
    cache: Sequence[Mapping[str, Any]],
    selections: Sequence[Sequence[Mapping[str, Any]]],
    inventory: Mapping[str, Mapping[str, Any]],
    modules: Mapping[str, ModuleType],
) -> dict[str, Any]:
    """Outer restore shield, including failures before the core's local try."""

    if "helper" not in context or "model" not in context:
        raise ProtocolError("historical callback lacks helper/model")
    torch = context["helper"].torch
    model = context["model"]
    named = dict(model.named_parameters())
    if any(name not in named for name in EXPECTED_ACTOR_NAMES):
        raise ProtocolError("historical callback lacks exact actor6")
    incoming_model_hash = context["helper"].model_state_sha256(model.state_dict())
    actor_before = {
        name: named[name].detach().clone() for name in EXPECTED_ACTOR_NAMES
    }
    requires_grad_before = {
        name: bool(parameter.requires_grad) for name, parameter in named.items()
    }
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in named.items()
    }
    precision_before = torch.get_float32_matmul_precision()
    training_before = bool(model.training)
    try:
        return _run_callback_core(
            context,
            cw20,
            cw19,
            cw15,
            grad,
            cache,
            selections,
            inventory,
            modules,
        )
    finally:
        with torch.no_grad():
            for name in EXPECTED_ACTOR_NAMES:
                named[name].copy_(actor_before[name])
        for name, parameter in named.items():
            parameter.requires_grad_(requires_grad_before[name])
            parameter.grad = (
                None
                if gradients_before[name] is None
                else gradients_before[name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
            )
        model.train(training_before)
        torch.set_float32_matmul_precision(precision_before)
        outer_restore_checks = {
            "model_hash_exact": context["helper"].model_state_sha256(
                model.state_dict()
            )
            == incoming_model_hash,
            "training_flag_exact": bool(model.training) == training_before,
            "requires_grad_exact": all(
                bool(parameter.requires_grad) == requires_grad_before[name]
                for name, parameter in named.items()
            ),
            "gradient_presence_exact": all(
                (parameter.grad is None) == (gradients_before[name] is None)
                for name, parameter in named.items()
            ),
            "precision_exact": torch.get_float32_matmul_precision()
            == precision_before,
        }
        if not all(outer_restore_checks.values()):
            raise ProtocolError(
                f"outer callback restore shield failed: {outer_restore_checks}"
            )


def production_run() -> dict[str, Any]:
    import numpy as np
    import torch

    cw20, cw20_evidence = import_cw20()
    runtime = runtime_audit(cw20, require_cuda=True)
    source = source_audit()
    cw19, cw19_evidence = cw20.import_cw19()
    runner, runner_evidence = cw19.import_locked(
        cw19.RUNNER, "run_u468_raw_trainhard_actor6_balanced_mix_sweep"
    )
    grad, grad_evidence = cw19.import_locked(
        cw19.GRADIENT, "cw21_frozen_gradient_probe"
    )
    cw15, cw15_evidence = cw19.import_locked(cw19.CW15, "cw21_frozen_cw15")
    if grad.sweep is not runner:
        raise ProtocolError("gradient module did not bind locked runner")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    selections, selection_summary, selection_evidence = grad.load_frozen_selection()
    cache, cache_audit, cache_inputs = grad.load_frozen_cache(selections)
    inventory = grad.objective_inventory(cache, selections)
    modules = cw15.frozen_modules()
    primary_source, primary_evidence = modules["cw11"].read_regular_bytes(
        cw15.PRIMARY,
        cw15.MODULE_SHAS[cw15.PRIMARY],
        "CW21 frozen primary source",
        expected_mode=0o555,
    )
    holder: dict[str, Any] = {}

    def consume(context: Mapping[str, Any]) -> None:
        if holder:
            raise ProtocolError("CW21 callback called more than once")
        holder["endpoint"] = run_callback(
            context,
            cw20,
            cw19,
            cw15,
            grad,
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
        "historical_consumer_called": historical.get("second_stage", {})
        .get("decision", {})
        .get("candidate_consumer_called")
        is True,
        "historical_CW11_model_exact": historical.get("second_stage", {})
        .get("decision", {})
        .get("candidate_model_state_sha256_before_CW10_finally_restore")
        == CW11_MODEL_SHA256,
        "historical_CW11_vector_exact": historical.get("second_stage", {})
        .get("decision", {})
        .get("terminal_cumulative_float64_le_sha256")
        == CW11_VECTOR_SHA256,
        "historical_CW11_ledger_exact": historical.get("second_stage", {})
        .get("active_pair_contract", {})
        .get("final_canonical_ledger_sha256")
        == CW11_LEDGER_SHA256,
        "outer_restore_pass": historical.get("final_integrity", {}).get("pass")
        is True,
    }
    if not all(historical_checks.values()):
        raise ProtocolError(f"historical CW11 replay chain failed: {historical_checks}")
    endpoint = holder["endpoint"]
    decision = endpoint["decision"]
    if decision not in {
        "GO_CW21_CACHE512_PREFLIGHT",
        "NO_GO_CW21_CACHE512_PREFLIGHT",
    }:
        raise ProtocolError(f"unexpected endpoint decision: {decision}")
    output = {
        "schema_version": SCHEMA,
        "status": decision,
        "decision": decision,
        "seed": SEED,
        "endpoint": endpoint,
        "historical_exact_CW11_replay": {
            "checks": historical_checks,
            "pass": True,
            "historical_replay_count": 1,
            "consumed_evidence_only": True,
            "historical_specialist_valid_rows_replayed": 4,
            "specialist_valid_consumed_for_optimization": True,
            "promotion_evidence": False,
            "status": historical["status"],
            "changed_candidate_official_or_validation_evaluation_count": 0,
        },
        "train_cache": {
            "selection": selection_summary,
            "cache": cache_audit,
            "inventory": inventory,
            "profile_anchor": "raw_U468_frozen_selection_not_reprofiled_at_CW11",
            "non_train_members_opened": False,
        },
        "contract": {
            "base": "exact_CW11_historical_RAM_replay_after_generalBC_plus_PPO",
            "special_BC": (
                "generic_frozen_balanced512_fixed_order_PCGrad_screen; "
                "not_targeted_special_BC"
            ),
            "task_order": list(TASK_ORDER),
            "planned_step_l2": PLANNED_STEP_L2,
            "actual_step_l2_interval": [ACTUAL_STEP_L2_MIN, ACTUAL_STEP_L2_MAX],
            "additional_from_CW11_cap": ADDITIONAL_FROM_CW11_CAP,
            "candidate_count": 1,
            "promotion_scope": "cache512_train_only_preflight_not_fulltrain_or_specialist",
            "CW11_materialized_eval_only_checkpoint_opened": False,
            "CW11_resume_forbidden_respected": True,
            "CW11_baseline_retention_gate_required": True,
            "pokemonfan_hard_discrete_repair_required": True,
            "cw11_no_harm_proven": False,
            "next_gate_if_GO": "frozen_fulltrain_before_official_specialist",
            "sweeps": 0,
            "line_searches": 0,
            "parameter_space_projections": 0,
            "pcgrad_gradient_conflict_projection": True,
            "clips": 0,
            "optimizer_instances": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "changed_candidate_train_endpoint_count": endpoint[
                "changed_candidate_train_endpoint_count"
            ],
            "changed_candidate_fulltrain_evaluation_count": 0,
            "changed_candidate_specialist_or_validation_evaluation_count": 0,
            "changed_candidate_official6_evaluation_count": 0,
            "changed_candidate_official_or_validation_evaluation_count": 0,
            "official_candidate_budget_consumed": 0,
            "checkpoint_writes": 0,
            "model_writes": 0,
            "result_artifact_writes": 0,
            "stdout_only": True,
        },
        "integrity": {
            "runtime": runtime,
            "source": source,
            "frozen_inputs": {
                "cw20_helper": cw20_evidence,
                "cw19_helper": cw19_evidence,
                "cw15_helper": cw15_evidence,
                "runner": runner_evidence,
                "gradient": grad_evidence,
                "primary": primary_evidence,
                "selection": selection_evidence,
                "cache_inputs": cache_inputs,
            },
        },
        "writes_performed": 0,
        "submission_performed": False,
    }
    if not grad.sweep.repair.finite_nested(output):
        raise ProtocolError("CW21 output contains nonfinite values")
    return output


def selftest_result(cw20: ModuleType) -> dict[str, Any]:
    raw = bytes((index * 29) % 256 for index in range(4096))
    encoded = cw20.xz_payload(raw)
    checks = {
        "payload_roundtrip": cw20.decode_xz(encoded) == raw,
        "planned_step_literal": PLANNED_STEP_L2 == 1.25e-4,
        "actual_interval_contains_planned": ACTUAL_STEP_L2_MIN
        <= PLANNED_STEP_L2
        <= ACTUAL_STEP_L2_MAX,
        "step_below_additional_cap": PLANNED_STEP_L2 < ADDITIONAL_FROM_CW11_CAP,
        "task_order_exact": TASK_ORDER
        == ("union_mixed", "flg_hard", "pokemonfan_hard", "core5_hard"),
    }
    if not all(checks.values()):
        raise ProtocolError(f"selftest failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": "selftest_passed",
        "pass": True,
        "checks": checks,
        "source": source_audit(),
        "runtime": runtime_audit(cw20, require_cuda=False),
        "writes_performed": 0,
        "submission_performed": False,
    }


def static_result(cw20: ModuleType, cw20_evidence: dict[str, Any]) -> dict[str, Any]:
    cw19, cw19_evidence = cw20.import_cw19()
    cw15, cw15_evidence = cw19.import_locked(cw19.CW15, "cw21_static_cw15")
    primary_evidence = regular_evidence(
        cw15.PRIMARY, cw15.MODULE_SHAS[cw15.PRIMARY], 0o555
    )
    anchor_checks = {
        "CW11_model_constant_exact": cw15.CW11_MODEL_SHA256
        == CW11_MODEL_SHA256,
        "CW11_vector_constant_exact": cw15.CW11_VECTOR_SHA256
        == CW11_VECTOR_SHA256,
        "CW11_ledger_constant_exact": cw15.CW11_ACTIVE_LEDGER_SHA256
        == CW11_LEDGER_SHA256,
        "raw_model_constant_exact": cw15.RAW_MODEL_SHA256 == RAW_MODEL_SHA256,
        "raw_nonactor_constant_exact": cw15.RAW_NONACTOR_SHA256
        == RAW_NONACTOR_SHA256,
        "actor_dimension_exact": cw15.ACTOR6_FLAT_LENGTH == 65793,
        "actor_names_exact": tuple(cw15.ACTOR6_NAMES) == EXPECTED_ACTOR_NAMES,
    }
    if not all(anchor_checks.values()):
        raise ProtocolError(f"static historical anchor drift: {anchor_checks}")
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_passed",
        "pass": True,
        "source": source_audit(),
        "runtime": runtime_audit(cw20, require_cuda=False),
        "frozen_CW20_helper": cw20_evidence,
        "frozen_CW19_helper": cw19_evidence,
        "frozen_CW15_helper": cw15_evidence,
        "frozen_primary": primary_evidence,
        "historical_anchor_checks": anchor_checks,
        "protocol": {
            "base": "historical_RAM_replayed_exact_CW11",
            "planned_step_l2": PLANNED_STEP_L2,
            "single_endpoint": True,
            "train_only": True,
            "materialized_eval_only_checkpoint_opened": False,
            "resume_forbidden_respected": True,
            "no_parameter_space_projection": True,
            "pcgrad_gradient_conflict_projection": True,
            "no_sweep": True,
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
    if args.mode == "run":
        result = production_run()
    else:
        cw20, evidence = import_cw20()
        result = (
            static_result(cw20, evidence)
            if args.mode == "static"
            else selftest_result(cw20)
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
