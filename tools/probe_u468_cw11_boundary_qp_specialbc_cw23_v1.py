#!/usr/bin/env python3
"""Frozen-candidate CW23 boundary-QP special-BC train-only preflight.

This probe reconstructs exact CW11 only inside the already-consumed historical
callback.  It opens the frozen CW22 train-only B256 cache, constructs one
deterministic actor-only update in full actor6 geometry, evaluates that one
RAM-only endpoint, and restores the callback state.  It never opens a new
validation/test member and never writes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_boundary_qp_specialbc_cw23_v1.py"
CW22 = TOOLS / "probe_u468_cw11_targeted_b256_pcgrad_specialbc_cw22_v1.py"
CW22_SHA256 = "f307f3f26355125f97c857f14b97a73d9a7a4f2ea39ca8b0a578e801055e82b4"
CW22_MODE = 0o555
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
OUTPUT = ROOT / "artifacts/cw23_cw11_boundary_qp_specialbc_trainonly_20260803_v1.json"
SCHEMA = "ptcg-u468-cw11-boundary-qp-specialbc-cw23-v1"
SEED = 202608031
RAW = (
    ROOT
    / "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
RAW_SHA256 = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"

TARGET_MARGIN = 1.0 / 512.0
SPECIAL9_LINEAR_NLL_DELTA = -2.0e-4
ADDITIONAL_FROM_CW11_CAP = 1.0e-3
AGGREGATE_RETENTION_TOLERANCE = 1.0e-6
DEFAULT_STRATUM_TOLERANCE = 1.0 / 2048.0
CONTEXT34_STRATUM_TOLERANCE = 1.0 / 1024.0
PINV_RCOND = 1.0e-12

PF0_TARGET_SHA256 = "c691171c4e7093f498f29fda034e456a52e277c1d27defffc886217f08ca935b"
PF7_TARGET_SHA256 = "d35040093be24508b97bbfb4a9f0ea98d0f2ef2d7f6af1c9d40119d31de8f123"
ZERO_MARGIN_SHA256 = (
    "061627395d4bb484f2e21d1a647f6d69fad043016efc102fe093aa55815f4a7f",
    "51eb642eb4c5a82cad4ac22c82bca296841ee8e348a19028aa847b9b658b4305",
    "7bf6fa80288ce4f38874d11da426030af291cdd3880a0242458cd111632e537d",
    "3b8e123d82ea6b8233d3d6088830527ec635f0ac363e72bb95bbd8a99d99cd21",
    "78f8d0275a2f38f5a043ffee28df5901001d66ab964422bbfaba93c1403a65dc",
    "10192657ae504eed3bc9e3fd59f80f7e9499d6022e38bdddd6c5da69e78afd81",
)


class ProtocolError(RuntimeError):
    """Fail-closed CW23 protocol error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    def reject_nonfinite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ProtocolError("nonfinite JSON value")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ProtocolError("non-string JSON key")
                reject_nonfinite(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                reject_nonfinite(child)

    reject_nonfinite(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def regular_source(path: Path, expected_sha: str, expected_mode: int, label: str) -> tuple[bytes, dict[str, Any]]:
    before = path.lstat()
    source = path.read_bytes()
    after = path.lstat()
    digest = hashlib.sha256(source).hexdigest()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "sha_exact": digest == expected_sha,
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} drift: {checks}")
    return source, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(source),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def load_cw22() -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = regular_source(CW22, CW22_SHA256, CW22_MODE, "frozen CW22")
    name = "cw23_frozen_cw22"
    if name in sys.modules:
        raise ProtocolError("CW22 module name already occupied")
    module = types.ModuleType(name)
    module.__file__ = str(CW22)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, str(CW22), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, evidence


def self_evidence(require_frozen: bool) -> dict[str, Any]:
    before = SCRIPT.lstat()
    source = SCRIPT.read_bytes()
    after = SCRIPT.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "mode_0555": stat.S_IMODE(after.st_mode) == 0o555,
        "no_validation_or_test_literal_member_open": (
            b"open_validation" + b"_member"
        )
        not in source
        and (b"open_test" + b"_member") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "one_candidate_formula": (b"for radius" + b" in") not in source
        and (b"learning_rate" + b"_candidates") not in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"CW23 source audit failed: {checks}")
    return {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise ProtocolError("short CW23 result write")
            written += count
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    observed = path.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and int(observed.st_nlink) == 1,
        "mode_0444": stat.S_IMODE(observed.st_mode) == 0o444,
        "size_exact": int(observed.st_size) == len(payload),
        "sha_exact": sha256_file(path) == hashlib.sha256(payload).hexdigest(),
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW23 result publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "nlink": int(observed.st_nlink),
        "checks": checks,
    }
def fixed_threat_margin(
    logits: Any,
    option_mask: Any,
    expert_order: Sequence[int],
) -> tuple[Any, dict[str, Any]]:
    """Return the weakest exact-CW11 expert-vs-current-threat margin."""

    remaining = option_mask.bool().clone()
    candidates: list[tuple[Any, int, int, int]] = []
    for step, chosen_value in enumerate(expert_order):
        chosen = int(chosen_value)
        competitors_mask = remaining.clone()
        competitors_mask[chosen] = False
        competitors = competitors_mask.nonzero(as_tuple=False).squeeze(1)
        if int(competitors.numel()) > 0:
            threat_offset = int(
                logits[competitors].detach().float().argmax().detach().cpu()
            )
            threat = int(competitors[threat_offset].detach().cpu())
            candidates.append((logits[chosen].float() - logits[threat].float(), step, chosen, threat))
        remaining[chosen] = False
    if not candidates:
        raise ProtocolError("row has no comparable ordered-selection step")
    selected = min(candidates, key=lambda item: float(item[0].detach().cpu()))
    return selected[0], {
        "step": selected[1],
        "expert": selected[2],
        "fixed_threat": selected[3],
        "margin": float(selected[0].detach().cpu()),
    }


def flat_gradient(values: Sequence[Any], names: Sequence[str], np: Any) -> Any:
    return np.concatenate(
        [
            value.detach().cpu().float().numpy().astype(np.float64, copy=False).reshape(-1)
            for name, value in zip(names, values)
        ]
    )


def array_sha(value: Any) -> str:
    contiguous = value.astype("<f8", copy=False)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def special_margin_report(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    cw22: ModuleType,
) -> dict[str, Any]:
    logits = outputs["policy_logits"].float().detach().cpu()
    masks = batch["option_mask"].bool().detach().cpu()
    margins = [
        cw22.ordered_margin(logits[index], masks[index], rows[index]["expert_order"])
        for index in indices
    ]
    return {
        "rows": len(margins),
        "minimum": min(margins),
        "maximum": max(margins),
        "mean": sum(margins) / len(margins),
        "values": margins,
    }


def run_boundary_core(
    context: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cw20: ModuleType,
    cw19: ModuleType,
    cw15: ModuleType,
    modules: Mapping[str, ModuleType],
    *,
    cw22: ModuleType,
) -> dict[str, Any]:
    import numpy as np
    import torch

    context_checks = cw22.validate_exact_context(context, cw20, cw15)
    helper = context["helper"]
    model = context["model"]
    ppo = helper.ppo
    dependency_checks = {
        "BC_source_exact_CW15": sha256_file(Path(helper.bc.__file__))
        == cw15.MODULE_SHAS[cw15.BC],
        "PPO_source_exact_CW15": sha256_file(Path(ppo.__file__))
        == cw15.MODULE_SHAS[cw15.PPO],
    }
    if not all(dependency_checks.values()):
        raise ProtocolError(f"BC/PPO dependency drift: {dependency_checks}")
    batch_cpu, cache_audit = cw22.load_train_b256(
        rows, context["model_config"], helper.bc, ppo
    )
    device = next(model.parameters()).device
    named = dict(model.named_parameters())
    requires_grad_before = {
        name: bool(parameter.requires_grad) for name, parameter in named.items()
    }
    for parameter in named.values():
        parameter.requires_grad_(False)
        parameter.grad = None
    for name in cw22.EXPECTED_ACTOR_NAMES:
        named[name].requires_grad_(True)
    parameters = {name: named[name] for name in cw22.EXPECTED_ACTOR_NAMES}
    parameter_sequence = modules["geometry"].configure_actor6(model)
    if any(
        parameters[name] is not value
        for name, value in zip(cw22.EXPECTED_ACTOR_NAMES, parameter_sequence)
    ):
        raise ProtocolError("actor6 implementation tensor identity drift")
    cw11_actor = cw20.clone_actor(parameters, cw22.EXPECTED_ACTOR_NAMES)
    cw11_flat = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
    try:
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch_cpu.items()
        }
        outputs = ppo.model_forward(model, batch, device)
        baseline, baseline_correct, baseline_predictions = cw22.evaluate_outputs(
            outputs, batch, rows, ppo
        )
        baseline_native = {
            "count_logits": outputs["count_logits"].detach().clone(),
            "value_logits": outputs["value_logits"].detach().clone(),
        }
        baseline_checks = {
            "rows_exact256": baseline["rows"] == 256,
            "hard96_all_ordered_wrong": baseline["hard"]["rows"] == 96
            and baseline["hard"]["ordered_correct"] == 0,
            "hard96_all_set_wrong": baseline["hard"]["set_correct"] == 0,
            "hard96_all_count_correct": baseline["hard"]["count_correct"] == 96,
            "retention160_all_ordered_correct": baseline["retention"]["rows"] == 160
            and baseline["retention"]["ordered_correct"] == 160,
            "retention160_all_set_correct": baseline["retention"]["set_correct"] == 160,
            "retention160_all_count_correct": baseline["retention"]["count_correct"] == 160,
            "native_policy_BF16": baseline["native_output"]["policy_logits_dtype"]
            == "torch.bfloat16",
        }
        if not all(baseline_checks.values()):
            return {
                "decision": "NO_GO_CW23_BOUNDARY_QP_BASELINE",
                "reason": "FINAL_B256_BASELINE_CERTIFICATION_FAILED",
                "context_checks": context_checks,
                "dependency_checks": dependency_checks,
                "cache": cache_audit,
                "baseline": baseline,
                "baseline_checks": baseline_checks,
                "changed_candidate_train_shadow_count": 0,
                "candidate_payload": None,
            }

        row_by_sha = {str(row["line_sha256"]): index for index, row in enumerate(rows)}
        identity_checks = {
            "all_rows_unique": len(row_by_sha) == 256,
            "PF0_target_present": PF0_TARGET_SHA256 in row_by_sha,
            "PF7_target_present": PF7_TARGET_SHA256 in row_by_sha,
            "zero_rows_present": all(value in row_by_sha for value in ZERO_MARGIN_SHA256),
        }
        if not all(identity_checks.values()):
            raise ProtocolError(f"fixed row identity drift: {identity_checks}")
        pf0_index = row_by_sha[PF0_TARGET_SHA256]
        pf7_index = row_by_sha[PF7_TARGET_SHA256]
        zero_indices = [row_by_sha[value] for value in ZERO_MARGIN_SHA256]
        target_contract_checks = {
            "PF0_stratum": rows[pf0_index]["stratum"] == "pf_ctx0_hard",
            "PF7_stratum": rows[pf7_index]["stratum"] == "pf_ctx7_hard",
            "PF0_profile_margin": float(rows[pf0_index]["selection_margin"])
            == -1.0 / 512.0,
            "PF7_profile_margin": float(rows[pf7_index]["selection_margin"])
            == -3.0 / 512.0,
            "zero_profile_margins": all(
                float(rows[index]["selection_margin"]) == 0.0
                and rows[index]["category"] != "hard"
                for index in zero_indices
            ),
            "zero_count6": len(zero_indices) == 6,
        }
        if not all(target_contract_checks.values()):
            raise ProtocolError(f"boundary target contract drift: {target_contract_checks}")

        logits = outputs["policy_logits"].float()
        objectives: list[Any] = []
        objective_names: list[str] = []
        threat_contract: dict[str, Any] = {}
        for name, index in (("pf0_target_margin", pf0_index), ("pf7_target_margin", pf7_index)):
            objective, detail = fixed_threat_margin(
                logits[index], batch["option_mask"][index], rows[index]["expert_order"]
            )
            objectives.append(objective)
            objective_names.append(name)
            threat_contract[name] = {"line_sha256": rows[index]["line_sha256"], **detail}
        for offset, index in enumerate(zero_indices):
            name = f"zero_margin_guard_{offset}"
            objective, detail = fixed_threat_margin(
                logits[index], batch["option_mask"][index], rows[index]["expert_order"]
            )
            objectives.append(objective)
            objective_names.append(name)
            threat_contract[name] = {"line_sha256": rows[index]["line_sha256"], **detail}

        baseline_linearization_checks = {
            "PF0_native_margin_exact": threat_contract["pf0_target_margin"]["margin"]
            == -1.0 / 512.0,
            "PF7_native_margin_exact": threat_contract["pf7_target_margin"]["margin"]
            == -3.0 / 512.0,
            "six_zero_native_margins_exact": all(
                threat_contract[f"zero_margin_guard_{offset}"]["margin"] == 0.0
                for offset in range(len(zero_indices))
            ),
        }
        if not all(baseline_linearization_checks.values()):
            raise ProtocolError(
                f"native boundary linearization drift: {baseline_linearization_checks}"
            )

        special_indices = [
            index
            for index, row in enumerate(rows)
            if row["stratum"] == "dominic_ctx0_hard" and cw22.is_dominic_special(row)
        ]
        if len(special_indices) != 9:
            raise ProtocolError("special9 row count drift")
        per_row_nll = cw22.ordered_nll_per_row(outputs, batch)
        special_tensor = torch.tensor(special_indices, dtype=torch.long, device=device)
        special9_loss = per_row_nll[special_tensor].mean()
        objectives.append(special9_loss)
        objective_names.append("special9_exact_ordered_nll")

        parameter_tuple = tuple(parameters[name] for name in cw22.EXPECTED_ACTOR_NAMES)
        gradient_rows: list[Any] = []
        gradient_audit: dict[str, Any] = {}
        for objective_index, (name, objective) in enumerate(zip(objective_names, objectives)):
            values = torch.autograd.grad(
                objective,
                parameter_tuple,
                retain_graph=objective_index + 1 < len(objectives),
                create_graph=False,
                allow_unused=False,
                materialize_grads=False,
            )
            flat = flat_gradient(values, cw22.EXPECTED_ACTOR_NAMES, np)
            if not bool(np.isfinite(flat).all()) or float(np.linalg.norm(flat)) <= 0.0:
                raise ProtocolError(f"{name}: invalid gradient")
            gradient_rows.append(flat)
            gradient_audit[name] = {
                "l2": float(np.linalg.norm(flat)),
                "max_abs": float(np.max(np.abs(flat))),
                "nonzero_elements": int(np.count_nonzero(flat)),
                "float64_le_sha256": array_sha(flat),
            }
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ProtocolError("autograd.grad materialized parameter .grad buffers")

        matrix = np.stack(gradient_rows, axis=0)
        requested = np.array(
            [
                TARGET_MARGIN - threat_contract["pf0_target_margin"]["margin"],
                TARGET_MARGIN - threat_contract["pf7_target_margin"]["margin"],
                *([0.0] * len(zero_indices)),
                SPECIAL9_LINEAR_NLL_DELTA,
            ],
            dtype=np.float64,
        )
        gram = matrix @ matrix.T
        coefficients = np.linalg.pinv(gram, rcond=PINV_RCOND) @ requested
        nominal_delta = matrix.T @ coefficients
        nominal_l2 = float(np.linalg.norm(nominal_delta))
        if not math.isfinite(nominal_l2) or nominal_l2 <= ADDITIONAL_FROM_CW11_CAP:
            raise ProtocolError("nominal boundary-QP solution did not require radial cap")
        planned_delta = nominal_delta * (ADDITIONAL_FROM_CW11_CAP / nominal_l2)
        planned_l2 = float(np.linalg.norm(planned_delta))
        predicted_nominal = matrix @ nominal_delta
        predicted_planned = matrix @ planned_delta
        qp_audit = {
            "objective_order": objective_names,
            "requested_linear_changes": requested.tolist(),
            "gram_float64_le_sha256": array_sha(gram),
            "gram_rank": int(np.linalg.matrix_rank(gram)),
            "gram_condition_number": float(np.linalg.cond(gram)),
            "coefficients": coefficients.tolist(),
            "nominal_delta_l2": nominal_l2,
            "nominal_delta_float64_le_sha256": array_sha(nominal_delta),
            "nominal_residual_linf": float(np.max(np.abs(predicted_nominal - requested))),
            "radial_cap": ADDITIONAL_FROM_CW11_CAP,
            "planned_delta_l2": planned_l2,
            "planned_delta_float64_le_sha256": array_sha(planned_delta),
            "predicted_nominal_changes": predicted_nominal.tolist(),
            "predicted_planned_changes": predicted_planned.tolist(),
            "single_candidate_only": True,
            "optimizer_instances_created": 0,
            "optimizer_step_calls": 0,
            "autograd_grad_calls": len(objectives),
            "backward_calls": 0,
        }
        qp_checks = {
            "gram_full_rank9": int(np.linalg.matrix_rank(gram)) == len(objectives),
            "nominal_residual_linf_at_most_1e-9": float(
                np.max(np.abs(predicted_nominal - requested))
            )
            <= 1.0e-9,
            "nominal_l2_requires_radial_cap": nominal_l2
            > ADDITIONAL_FROM_CW11_CAP,
            "planned_l2_exact_cap": abs(
                planned_l2 - ADDITIONAL_FROM_CW11_CAP
            )
            <= 1.0e-12,
        }
        if not all(qp_checks.values()):
            raise ProtocolError(f"boundary-QP numerical audit failed: {qp_checks}")
        qp_audit["checks"] = qp_checks

        cw20.apply_flat_actor(
            parameters,
            cw22.EXPECTED_ACTOR_NAMES,
            cw11_flat + planned_delta,
            torch,
        )
        actual_flat = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
        actual_delta = actual_flat - cw11_flat
        actual_l2 = float(np.linalg.norm(actual_delta))
        candidate_hash = helper.model_state_sha256(model.state_dict())
        selected_actor_bytes = cw20.actor_bytes(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
        with torch.no_grad():
            candidate_outputs = ppo.model_forward(model, batch, device)
            candidate, candidate_correct, candidate_predictions = cw22.evaluate_outputs(
                candidate_outputs, batch, rows, ppo
            )
            candidate_per_row_nll = cw22.ordered_nll_per_row(candidate_outputs, batch)
            immutable_native = {
                "count_logits_native_exact": bool(
                    torch.equal(candidate_outputs["count_logits"], baseline_native["count_logits"])
                ),
                "value_logits_native_exact": bool(
                    torch.equal(candidate_outputs["value_logits"], baseline_native["value_logits"])
                ),
                "policy_logits_native_BF16": candidate_outputs["policy_logits"].dtype
                == torch.bfloat16,
            }

        hard_rows = [row for row in rows if row["category"] == "hard"]
        retention_rows = [row for row in rows if row["category"] != "hard"]
        repairs = [
            str(row["line_sha256"])
            for row in hard_rows
            if not baseline_correct[str(row["line_sha256"])]
            and candidate_correct[str(row["line_sha256"])]
        ]
        retention_flips = [
            str(row["line_sha256"])
            for row in retention_rows
            if baseline_correct[str(row["line_sha256"])]
            and not candidate_correct[str(row["line_sha256"])]
        ]
        hard_strata = ("pf_ctx0_hard", "pf_ctx7_hard", "dominic_ctx0_hard")
        hard_repairs_by_stratum = {
            stratum: sum(
                str(row["line_sha256"]) in repairs
                for row in hard_rows
                if row["stratum"] == stratum
            )
            for stratum in hard_strata
        }
        hard_nll_improvements = {
            stratum: float(baseline["by_stratum"][stratum]["ordered_nll"])
            - float(candidate["by_stratum"][stratum]["ordered_nll"])
            for stratum in hard_strata
        }
        retention_strata = sorted({str(row["stratum"]) for row in retention_rows})
        retention_nll_improvements = {
            stratum: float(baseline["by_stratum"][stratum]["ordered_nll"])
            - float(candidate["by_stratum"][stratum]["ordered_nll"])
            for stratum in retention_strata
        }
        special9_baseline_nll = float(special9_loss.detach().cpu())
        special9_candidate_nll = float(candidate_per_row_nll[special_tensor].mean().cpu())
        special9_nll_improvement = special9_baseline_nll - special9_candidate_nll
        baseline_special_margins = special_margin_report(outputs, batch, rows, special_indices, cw22)
        candidate_special_margins = special_margin_report(
            candidate_outputs, batch, rows, special_indices, cw22
        )
        target_observed = {}
        for name, index in (("pf0_target_margin", pf0_index), ("pf7_target_margin", pf7_index)):
            detail = threat_contract[name]
            observed = float(
                candidate_outputs["policy_logits"][index].float()[detail["expert"]].cpu()
                - candidate_outputs["policy_logits"][index]
                .float()[detail["fixed_threat"]]
                .cpu()
            )
            target_observed[name] = {
                "line_sha256": rows[index]["line_sha256"],
                "baseline_margin": detail["margin"],
                "candidate_fixed_threat_margin": observed,
                "ordered_correct": candidate_correct[str(rows[index]["line_sha256"])],
                "prediction": candidate_predictions[index],
            }
        zero_guard_observed = []
        for offset, index in enumerate(zero_indices):
            detail = threat_contract[f"zero_margin_guard_{offset}"]
            observed = float(
                candidate_outputs["policy_logits"][index].float()[detail["expert"]].cpu()
                - candidate_outputs["policy_logits"][index]
                .float()[detail["fixed_threat"]]
                .cpu()
            )
            zero_guard_observed.append(
                {
                    "line_sha256": rows[index]["line_sha256"],
                    "stratum": rows[index]["stratum"],
                    "baseline_margin": detail["margin"],
                    "candidate_fixed_threat_margin": observed,
                    "ordered_correct": candidate_correct[str(rows[index]["line_sha256"])],
                }
            )

        stratum_tolerances = {
            stratum: (
                CONTEXT34_STRATUM_TOLERANCE
                if stratum == "broad_context34_retention"
                else DEFAULT_STRATUM_TOLERANCE
            )
            for stratum in retention_strata
        }
        train_checks = {
            "PF0_target_native_ordered_correct": target_observed["pf0_target_margin"][
                "ordered_correct"
            ],
            "PF7_target_native_ordered_correct": target_observed["pf7_target_margin"][
                "ordered_correct"
            ],
            "PF0_repairs_at_least1": hard_repairs_by_stratum["pf_ctx0_hard"] >= 1,
            "PF7_repairs_at_least1": hard_repairs_by_stratum["pf_ctx7_hard"] >= 1,
            "total_hard_repairs_at_least2": len(repairs) >= 2,
            "all_three_hard_exact_NLL_improve": all(
                value > 0.0 for value in hard_nll_improvements.values()
            ),
            "special9_exact_NLL_improves": special9_nll_improvement > 0.0,
            "special9_first_order_constraint_is_descent": float(
                np.dot(gradient_rows[-1], actual_delta)
            )
            < 0.0,
            "six_zero_margin_guards_stay_correct": all(
                row["ordered_correct"] for row in zero_guard_observed
            ),
            "retention160_zero_correct_to_wrong": not retention_flips,
            "retention160_all_ordered_correct": candidate["retention"]["ordered_correct"]
            == 160,
            "retention160_aggregate_NLL_nondegrade": float(
                candidate["retention"]["ordered_nll"]
            )
            <= float(baseline["retention"]["ordered_nll"])
            + AGGREGATE_RETENTION_TOLERANCE,
            "each_retention_stratum_within_quantization_tolerance": all(
                retention_nll_improvements[stratum] >= -stratum_tolerances[stratum]
                for stratum in retention_strata
            ),
            "count_logits_native_exact": immutable_native["count_logits_native_exact"],
            "value_logits_native_exact": immutable_native["value_logits_native_exact"],
            "policy_logits_native_BF16": immutable_native["policy_logits_native_BF16"],
            "candidate_all256_count_correct": candidate["hard"]["count_correct"] == 96
            and candidate["retention"]["count_correct"] == 160,
        }
        changed_names = [
            name
            for name in cw22.EXPECTED_ACTOR_NAMES
            if not torch.equal(parameters[name], cw11_actor[name])
        ]
        integrity = {
            "nominal_solution_radially_capped": nominal_l2 > ADDITIONAL_FROM_CW11_CAP,
            "planned_l2_exact_cap": abs(planned_l2 - ADDITIONAL_FROM_CW11_CAP) <= 1e-12,
            "actual_additional_from_CW11_cap": actual_l2 <= ADDITIONAL_FROM_CW11_CAP,
            "candidate_model_changed": candidate_hash != cw22.CW11_MODEL_SHA256,
            "changed_parameter_names_nonempty": bool(changed_names),
            "changed_parameter_scope_subset_actor6": set(changed_names).issubset(
                set(cw22.EXPECTED_ACTOR_NAMES)
            ),
            "nonactor_exact_raw": cw20.nonactor_sha(
                model, cw22.EXPECTED_ACTOR_NAMES, helper
            )
            == cw22.RAW_NONACTOR_SHA256,
            "actual_delta_finite": bool(np.isfinite(actual_delta).all()),
            "candidate_model_all_finite": all(
                bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()
            ),
        }
        candidate_pass = all(train_checks.values()) and all(integrity.values())
        trial = {
            "candidate_model_state_sha256": candidate_hash,
            "actual_additional_from_CW11_l2": actual_l2,
            "actual_delta_float64_le_sha256": array_sha(actual_delta),
            "changed_parameter_names": changed_names,
            "integrity": integrity,
            "candidate": candidate,
            "target_observed": target_observed,
            "zero_guard_observed": zero_guard_observed,
            "hard_repairs": len(repairs),
            "hard_repairs_by_stratum": hard_repairs_by_stratum,
            "hard_repair_line_sha256": repairs,
            "hard_exact_NLL_improvements": hard_nll_improvements,
            "special9": {
                "baseline_exact_ordered_NLL": special9_baseline_nll,
                "candidate_exact_ordered_NLL": special9_candidate_nll,
                "exact_ordered_NLL_improvement": special9_nll_improvement,
                "baseline_margin": baseline_special_margins,
                "candidate_margin": candidate_special_margins,
            },
            "retention_correct_to_wrong_count": len(retention_flips),
            "retention_correct_to_wrong_line_sha256": retention_flips,
            "retention_exact_NLL_improvement": float(
                baseline["retention"]["ordered_nll"]
            )
            - float(candidate["retention"]["ordered_nll"]),
            "retention_exact_NLL_improvements_by_stratum": retention_nll_improvements,
            "retention_stratum_tolerances": stratum_tolerances,
            "immutable_native": immutable_native,
            "train_checks": train_checks,
            "pass": candidate_pass,
        }
        if not candidate_pass:
            cw20.restore_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, cw11_actor, torch)
            return {
                "decision": "NO_GO_CW23_BOUNDARY_QP_PREFLIGHT",
                "reason": "SOLE_BOUNDARY_QP_CANDIDATE_FAILED_TARGETED_TRAIN_GATE",
                "context_checks": context_checks,
                "dependency_checks": dependency_checks,
                "cache": cache_audit,
                "baseline": baseline,
                "baseline_checks": baseline_checks,
                "identity_checks": identity_checks,
                "target_contract_checks": target_contract_checks,
                "baseline_linearization_checks": baseline_linearization_checks,
                "threat_contract": threat_contract,
                "gradient_audit": gradient_audit,
                "boundary_qp": qp_audit,
                "trial": trial,
                "changed_candidate_train_shadow_count": 1,
                "candidate_payload": None,
            }

        layout = cw20.actor_layout(parameters, cw22.EXPECTED_ACTOR_NAMES)
        payload = {
            "anchor": {
                "reconstruction_base": "original_raw_U468",
                "raw_checkpoint": str(RAW.relative_to(ROOT)),
                "raw_checkpoint_file_sha256": RAW_SHA256,
                "raw_model_state_sha256": cw22.RAW_MODEL_SHA256,
                "CW11_provenance_model_state_sha256": cw22.CW11_MODEL_SHA256,
                "CW11_provenance_vector_float64_le_sha256": cw22.CW11_VECTOR_SHA256,
                "CW11_provenance_active_pair_ledger_sha256": cw22.CW11_LEDGER_SHA256,
                "CW11_materialized_eval_only_checkpoint_used": False,
                "terminal_model_state_sha256": candidate_hash,
            },
            "formula": (
                "load_original_raw_U468_then_replace_all_six_absolute_actor_"
                "float32_tensors_from_frozen_payload"
            ),
            "actor_names": list(cw22.EXPECTED_ACTOR_NAMES),
            "changed_actor_names": changed_names,
            "actor_layout": layout,
            "actor_layout_sha256": hashlib.sha256(
                json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "candidate_actor_float32_le": cw20.xz_payload(selected_actor_bytes),
            "actual_additional_from_CW11_l2": actual_l2,
            "boundary_qp_planned_delta_float64_le_sha256": array_sha(planned_delta),
            "selection_sha256": cw22.SELECTION_PAYLOAD_SHA256,
            "cache_sha256": cache_audit["cache_sha256"],
        }
        modules["cutting"].restore_raw_actor(
            modules["ram"], parameter_sequence, context["raw_actor"], torch
        )
        raw_restored_hash = helper.model_state_sha256(model.state_dict())
        decoded = cw20.decode_xz(payload["candidate_actor_float32_le"])
        cw20.copy_actor_bytes(
            parameters, cw22.EXPECTED_ACTOR_NAMES, layout, decoded, np, torch
        )
        reconstructed_hash = helper.model_state_sha256(model.state_dict())
        reconstruction_checks = {
            "raw_U468_restore_exact": raw_restored_hash == cw22.RAW_MODEL_SHA256,
            "candidate_hash_exact": reconstructed_hash == candidate_hash,
            "candidate_actor_bytes_exact": cw20.actor_bytes(
                parameters, cw22.EXPECTED_ACTOR_NAMES, np
            )
            == selected_actor_bytes,
            "nonactor_exact_raw": cw20.nonactor_sha(
                model, cw22.EXPECTED_ACTOR_NAMES, helper
            )
            == cw22.RAW_NONACTOR_SHA256,
        }
        cw20.restore_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, cw11_actor, torch)
        final_checks = {
            "exact_CW11_context": all(context_checks.values()),
            "dependency_sources": all(dependency_checks.values()),
            "final_B256_baseline": all(baseline_checks.values()),
            "sole_boundary_QP_trial": trial["pass"],
            "absolute_payload_reconstruction": all(reconstruction_checks.values()),
        }
        passed = all(final_checks.values())
        return {
            "decision": (
                "GO_CW23_BOUNDARY_QP_PREFLIGHT"
                if passed
                else "NO_GO_CW23_BOUNDARY_QP_PREFLIGHT"
            ),
            "reason": (
                "SOLE_PREREGISTERED_BOUNDARY_QP_CANDIDATE_PASSED_TARGETED_TRAIN_GATE"
                if passed
                else "ABSOLUTE_PAYLOAD_RECONSTRUCTION_FAILED"
            ),
            "context_checks": context_checks,
            "dependency_checks": dependency_checks,
            "cache": cache_audit,
            "baseline": baseline,
            "baseline_checks": baseline_checks,
            "identity_checks": identity_checks,
            "target_contract_checks": target_contract_checks,
            "baseline_linearization_checks": baseline_linearization_checks,
            "threat_contract": threat_contract,
            "gradient_audit": gradient_audit,
            "boundary_qp": qp_audit,
            "trial": trial,
            "pure_payload_reconstruction": {
                "checks": reconstruction_checks,
                "pass": all(reconstruction_checks.values()),
                "raw_model_state_sha256": raw_restored_hash,
                "candidate_model_state_sha256": reconstructed_hash,
            },
            "final_checks": final_checks,
            "changed_candidate_train_shadow_count": 1,
            "candidate_payload": payload if passed else None,
        }
    finally:
        cw20.restore_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, cw11_actor, torch)
        for name, parameter in named.items():
            parameter.requires_grad_(requires_grad_before[name])
            parameter.grad = None


def production_run() -> dict[str, Any]:
    source = self_evidence(require_frozen=True)
    cw22, cw22_evidence = load_cw22()
    original = cw22.run_targeted_core
    original_seed = cw22.SEED

    def bound_core(
        context: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        cw20: ModuleType,
        cw19: ModuleType,
        cw15: ModuleType,
        modules: Mapping[str, ModuleType],
    ) -> dict[str, Any]:
        return run_boundary_core(
            context, rows, cw20, cw19, cw15, modules, cw22=cw22
        )

    cw22.run_targeted_core = bound_core
    cw22.SEED = SEED
    try:
        result = cw22.production_run()
    finally:
        cw22.run_targeted_core = original
        cw22.SEED = original_seed
    result["schema_version"] = SCHEMA
    result["status"] = result["endpoint"]["decision"]
    result["decision"] = result["endpoint"]["decision"]
    result["seed"] = SEED
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_boundary_QP_special_BC"
    )
    result["selection"]["optimization_contract"] = {
        "single_candidate_only": True,
        "geometry": "full_actor6_minimum_norm_Gram_solution_then_radial_cap",
        "PF0_and_PF7_targets": (
            "nominal fixed exact-CW11 threat margins to +1/512 before radial cap; "
            "native BF16 correctness is authoritative"
        ),
        "zero_margin_retention_guards": "six fixed native-BF16 correct rows, linear delta zero",
        "special9_objective": "exact ordered PL NLL first-order delta -2e-4",
        "native_gate": "deterministic BF16 actions plus exact ordered PL NLL",
        "additional_from_CW11_cap": ADDITIONAL_FROM_CW11_CAP,
        "broad_context34_retention_tolerance": CONTEXT34_STRATUM_TOLERANCE,
        "other_retention_stratum_tolerance": DEFAULT_STRATUM_TOLERANCE,
    }
    result["historical_exact_CW11_replay"].pop(
        "new_validation_rows_opened_for_CW22_selection_or_candidate", None
    )
    result["historical_exact_CW11_replay"][
        "new_validation_rows_opened_for_CW23_selection_or_candidate"
    ] = 0
    result["inputs"]["CW23_source"] = source
    result["inputs"]["frozen_CW22_engine"] = cw22_evidence
    frozen_engine_contract = result["audit"].pop("source")
    frozen_engine_contract.pop("autograd_grad_runtime_calls", None)
    frozen_engine_contract["not_executed_as_CW22_runtime_contract"] = True
    frozen_engine_contract["role"] = "static imported engine source audit only"
    result["audit"]["frozen_CW22_engine_source_contract"] = frozen_engine_contract
    result["audit"]["source"] = source
    result["audit"]["CW23_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "train_only": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "optimizer_instances_created": 0,
        "optimizer_step_calls": 0,
        "changed_candidate_train_shadow_count": result["endpoint"].get(
            "changed_candidate_train_shadow_count"
        ),
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "recipe_selected_after_train_only_boundary_VJP_exploration": True,
        "exploratory_results_are_not_promotion_evidence": True,
        "Dominic_discrete_repair_not_required_under_0p001_cap": True,
        "special9_exact_NLL_improvement_required": True,
        "fulltrain_then_specialist_then_broad_then_Gold_required_after_GO": True,
    }
    result["official_unique_changed_candidate_count_consumed"] = 0
    result["cumulative_official_unique_changed_candidate_count"] = 1
    result["submission_performed"] = False
    result["package_upload_performed"] = False
    canonical_json(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        cw22_source, cw22_evidence = regular_source(
            CW22, CW22_SHA256, CW22_MODE, "frozen CW22"
        )
        del cw22_source
        print(
            canonical_json(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_pass",
                    "source": self_evidence(require_frozen=True),
                    "CW22": cw22_evidence,
                    "output_absent": not OUTPUT.exists(),
                    "python_exact": Path(sys.executable).resolve()
                    == EXPECTED_PYTHON.resolve(),
                }
            ).decode(),
            end="",
        )
        return
    output = args.output.resolve()
    if output != OUTPUT.resolve():
        raise ProtocolError("output path is not the frozen CW23 target")
    if OUTPUT.exists():
        raise ProtocolError("frozen CW23 output already exists")
    result = production_run()
    payload = canonical_json(result)
    publication = publish_o_excl(OUTPUT, payload)
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "decision": result["decision"],
                "candidate_model_state_sha256": result["endpoint"].get("trial", {}).get(
                    "candidate_model_state_sha256"
                ),
                "candidate_payload_present": result["endpoint"].get("candidate_payload")
                is not None,
                "changed_candidate_train_shadow_count": result["endpoint"].get(
                    "changed_candidate_train_shadow_count"
                ),
                "official_unique_changed_candidate_count_consumed": 0,
                "submission_performed": False,
                "output": publication,
            }
        ).decode(),
        end="",
    )


if __name__ == "__main__":
    main()
