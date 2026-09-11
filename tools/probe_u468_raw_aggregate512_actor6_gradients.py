#!/usr/bin/env python3
"""Probe frozen aggregate-512 actor6 gradient geometry at raw full U468.

This tool is deliberately not a trainer.  It imports the final frozen
balanced-sweep runner, reconstructs its exact two 256-row train batches, and
uses ``torch.autograd.grad`` at the unchanged raw-U468 point.  It never creates
an optimizer, calls ``backward``/``step``, opens a validation member, or writes
a model/checkpoint/result artifact.  JSON is emitted to stdout only.

The finite candidate set and selection order are fixed in source before the
probe is run:

1. analytic fresh-AdamW first-step direction from the row-weighted union;
2. raw row-weighted union gradient (plain SGD reference only);
3. equal-source mixed-loss gradient (plain SGD reference only);
4. fixed-order PCGrad over union plus the three source-hard objectives.

Fresh AdamW has first priority.  PCGrad is eligible only when none of the three
non-PCGrad directions gives numerically robust first-order descent for the
union mixed loss, every source-hard loss, and aggregate retention loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_u468_raw_trainhard_actor6_balanced_mix_sweep as sweep  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-raw-aggregate512-actor6-gradient-probe-v1"
FINAL_RUNNER = TOOLS / "run_u468_raw_trainhard_actor6_balanced_mix_sweep.py"
FINAL_RUNNER_SHA256 = (
    "419feeeb5644adf34a8d56142cfcd87b3a4b79446d95b9b566bf9fcabdb721b1"
)
EXPECTED_PROFILE_SHA256 = (
    "17af98e175167398a80cd2e1f8930c3d1d85cfe6ceea3143dfe2ba44dc89f1f9"
)
EXPECTED_CACHE_SHA256 = (
    "39c34d393eec077e9e2b005f4477b94a9141504d126cc702891af711a99a390f"
)
EXPECTED_BATCH_SHA256 = (
    "0dbdfb804fe2c1ca0187d4b1a1178ca3128b9ed53c447e12639fbf75c198953a",
    "aa8fa4714da5c26396f5b1a95e5197dc74a133ea0e2319dbea59e21ebc23dc00",
)
EXPECTED_SELECTION_SHA256 = (
    "6ff7becab756a4ed528195b22828e4abbaedbd3603b77ac87867d0310f875b6e",
    "6475bf85e4fa204811acc1f4628b8e28a6b15081c0e2ce8c6b0e08413a02cdd1",
)

PROBE_SEED = 202608113
SOURCES = ("flg", "pokemonfan", "core5")
OBJECTIVES = (
    "union_mixed",
    "flg_hard",
    "flg_retention",
    "pokemonfan_hard",
    "pokemonfan_retention",
    "core5_hard",
    "core5_retention",
)
REQUIRED_DESCENT_OBJECTIVES = (
    "union_mixed",
    "flg_hard",
    "pokemonfan_hard",
    "core5_hard",
)
RETENTION_GUARD_OBJECTIVE = "union_retention"
DIRECTIONAL_GATE_OBJECTIVES = REQUIRED_DESCENT_OBJECTIVES + (
    RETENTION_GUARD_OBJECTIVE,
)
FRESH_ADAMW_CANDIDATE = "union_fresh_adamw_step_direction"
NON_PCGRAD_CANDIDATES = (
    FRESH_ADAMW_CANDIDATE,
    "row_weighted_union",
    "source_balanced_union",
)
PCGRAD_CANDIDATE = "pcgrad_union_plus_three_hards_fixed_order"
CANDIDATE_PRIORITY = NON_PCGRAD_CANDIDATES + (PCGRAD_CANDIDATE,)

# Scale-free guard against classifying floating-point orthogonality as descent.
# Raw dot>0 is also reported separately.
FIRST_ORDER_COSINE_EPS = 1e-12
UNION_RECONSTRUCTION_RELATIVE_TOLERANCE = 5e-5


Gradient = dict[str, torch.Tensor]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_single_link_regular(
    path: Path, expected_sha256: str | None, label: str
) -> dict[str, Any]:
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise ValueError(f"{label} must be a single-link regular file")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 drift: {digest}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


def assert_frozen_runner_constants() -> None:
    expected = {
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "batch_sha256": EXPECTED_BATCH_SHA256,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "actor_names": tuple(sweep.ACTOR_NAMES),
        "sources": tuple(sweep.DATASETS),
        "batch_size": 256,
        "steps": 2,
        "context34_sample_weight": 1.0 / 3.0,
        "order_context_weight": 8.0,
    }
    observed = {
        "profile_sha256": sweep.PROFILE_SHA256,
        "cache_sha256": sweep.EXPECTED_CACHE_SHA256,
        "batch_sha256": tuple(sweep.EXPECTED_BATCH_SHA256),
        "selection_sha256": tuple(sweep.EXPECTED_STEP_SELECTION_SHA256),
        "actor_names": tuple(sweep.ACTOR_NAMES),
        "sources": tuple(sweep.DATASETS),
        "batch_size": sweep.BATCH_SIZE,
        "steps": sweep.STEPS,
        "context34_sample_weight": sweep.CONTEXT34_SAMPLE_WEIGHT,
        "order_context_weight": sweep.ORDER_CONTEXT_WEIGHT,
    }
    if observed != expected:
        raise ValueError(
            "final runner constant drift: "
            + json.dumps(observed, ensure_ascii=False, sort_keys=True)
        )
    if SOURCES != tuple(sweep.DATASETS):
        raise ValueError("source order drift")


def load_frozen_selection() -> tuple[
    list[list[dict[str, Any]]], dict[str, Any], dict[str, Any]
]:
    runner_evidence = require_single_link_regular(
        FINAL_RUNNER, FINAL_RUNNER_SHA256, "final balanced runner"
    )
    imported_runner = Path(sweep.__file__).resolve()
    if imported_runner != FINAL_RUNNER.resolve():
        raise ValueError(f"wrong runner module imported: {imported_runner}")
    assert_frozen_runner_constants()
    profile_evidence = require_single_link_regular(
        sweep.PROFILE, EXPECTED_PROFILE_SHA256, "raw-U468 train profile"
    )
    # The frozen profiler intentionally emits JSON ``Infinity`` for rows with
    # no finite runner-up margin.  Match the final runner's profile loader;
    # selection only consumes the finite hard/fragile records it validates.
    profile = json.loads(sweep.PROFILE.read_text(encoding="utf-8"))
    selections, selection_summary = sweep.select_batches(profile)
    observed_selection = tuple(
        str(row["selection_sha256"])
        for row in selection_summary["per_step"]
    )
    if observed_selection != EXPECTED_SELECTION_SHA256:
        raise RuntimeError(f"frozen selection SHA-256 drift: {observed_selection}")
    return selections, selection_summary, {
        "runner": runner_evidence,
        "profile": profile_evidence,
    }


def load_frozen_cache(
    selections: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any], dict[str, Any]]:
    inputs = {
        "u468": require_single_link_regular(
            sweep.U468, sweep.U468_SHA256, "raw U468"
        ),
        "general_bc": require_single_link_regular(
            sweep.GENERAL_BC, sweep.GENERAL_BC_SHA256, "general BC"
        ),
        "datasets": {
            source: require_single_link_regular(
                path, sweep.DATA_SHA256[source], f"{source} train archive"
            )
            for source, path in sweep.DATASETS.items()
        },
        "dependencies": {
            str(path.relative_to(ROOT)): require_single_link_regular(
                path, expected_sha256, str(path.relative_to(ROOT))
            )
            for path, expected_sha256 in sweep.DEPENDENCY_SHA256.items()
        },
    }
    parent = torch.load(sweep.U468, map_location="cpu", weights_only=False)
    cache = sweep.load_selected_features(selections, parent["model_config"])
    cache_sha256, batch_sha256 = sweep.repair.replay_cache_manifest(cache)
    if cache_sha256 != EXPECTED_CACHE_SHA256:
        raise RuntimeError(f"frozen aggregate cache SHA-256 drift: {cache_sha256}")
    if tuple(batch_sha256) != EXPECTED_BATCH_SHA256:
        raise RuntimeError(f"frozen batch cache SHA-256 drift: {batch_sha256}")
    rows = sum(int(batch["action_counts"].shape[0]) for batch in cache)
    context34_rows = sum(
        int((batch["contexts"] == sweep.ppo.SKILL_ORDER_CONTEXT).sum())
        for batch in cache
    )
    context34_sample_weight = sum(
        float(
            batch["sample_weights"][
                batch["contexts"] == sweep.ppo.SKILL_ORDER_CONTEXT
            ].sum()
        )
        for batch in cache
    )
    if (
        rows != 512
        or len(cache) != 2
        or context34_rows != 6
        or not math.isclose(
            context34_sample_weight, 2.0, rel_tol=0.0, abs_tol=1e-6
        )
    ):
        raise RuntimeError("aggregate512 cache shape/context contract drift")
    cache_audit = {
        "cache_sha256": cache_sha256,
        "batch_sha256": list(batch_sha256),
        "rows": rows,
        "batches": len(cache),
        "physical_context34_rows": context34_rows,
        "context34_sample_weight_before_order_multiplier": (
            context34_sample_weight
        ),
    }
    return cache, cache_audit, inputs


def belongs_to_objective(identity: Mapping[str, Any], objective: str) -> bool:
    if objective == "union_mixed":
        return True
    source, bucket = objective.rsplit("_", 1)
    if str(identity["source"]) != source:
        return False
    category = str(identity["category"])
    if bucket == "hard":
        return category == "hard"
    if bucket == "retention":
        return category in {"fragile", "c34"}
    raise ValueError(f"unknown objective: {objective}")


def effective_weights(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    weights = batch["sample_weights"].float()
    order_multiplier = torch.where(
        batch["contexts"] == sweep.ppo.SKILL_ORDER_CONTEXT,
        torch.full_like(weights, sweep.ORDER_CONTEXT_WEIGHT),
        torch.ones_like(weights),
    )
    return weights * order_multiplier


def objective_inventory(
    cache: Sequence[dict[str, torch.Tensor]],
    selections: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, dict[str, float | int]]:
    result = {
        objective: {"rows": 0, "effective_weight": 0.0}
        for objective in OBJECTIVES
    }
    for batch, identities in zip(cache, selections):
        weights = effective_weights(batch).cpu()
        if len(identities) != int(weights.shape[0]):
            raise RuntimeError("selection/cache row alignment drift")
        for row_index, identity in enumerate(identities):
            for objective in OBJECTIVES:
                if belongs_to_objective(identity, objective):
                    result[objective]["rows"] += 1
                    result[objective]["effective_weight"] += float(
                        weights[row_index]
                    )
    expected_rows = {
        "union_mixed": 512,
        "flg_hard": 96,
        "flg_retention": 32,
        "pokemonfan_hard": 64,
        "pokemonfan_retention": 128,
        "core5_hard": 64,
        "core5_retention": 128,
    }
    observed_rows = {
        objective: int(record["rows"]) for objective, record in result.items()
    }
    if observed_rows != expected_rows:
        raise RuntimeError(f"objective row inventory drift: {observed_rows}")
    if any(float(record["effective_weight"]) <= 0.0 for record in result.values()):
        raise RuntimeError("objective has nonpositive effective weight")
    return result


def zero_gradient_like(parameters: Mapping[str, torch.nn.Parameter]) -> Gradient:
    return {
        name: torch.zeros_like(
            parameter.detach(), device="cpu", dtype=torch.float64
        )
        for name, parameter in parameters.items()
    }


def clone_gradient(gradient: Mapping[str, torch.Tensor]) -> Gradient:
    return {name: tensor.clone() for name, tensor in gradient.items()}


def scaled_add(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    scale: float,
) -> Gradient:
    return {
        name: left[name] + float(scale) * right[name]
        for name in sweep.ACTOR_NAMES
    }


def scale_gradient(gradient: Mapping[str, torch.Tensor], scale: float) -> Gradient:
    return {
        name: gradient[name] * float(scale) for name in sweep.ACTOR_NAMES
    }


def sum_gradients(
    gradients: Sequence[Mapping[str, torch.Tensor]],
) -> Gradient:
    if not gradients:
        raise ValueError("cannot sum an empty gradient sequence")
    result = {
        name: torch.zeros_like(gradients[0][name]) for name in sweep.ACTOR_NAMES
    }
    for gradient in gradients:
        result = scaled_add(result, gradient, 1.0)
    return result


def dot_gradient(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    return float(
        sum(
            torch.dot(left[name].reshape(-1), right[name].reshape(-1))
            for name in sweep.ACTOR_NAMES
        )
    )


def norm_gradient(gradient: Mapping[str, torch.Tensor]) -> float:
    square = dot_gradient(gradient, gradient)
    if square < 0.0:
        raise FloatingPointError(f"negative gradient squared norm: {square}")
    return math.sqrt(square)


def cosine_gradient(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    denominator = norm_gradient(left) * norm_gradient(right)
    if denominator <= 0.0:
        raise RuntimeError("zero gradient prevents cosine calculation")
    return dot_gradient(left, right) / denominator


def gradient_sha256(gradient: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sweep.ACTOR_NAMES:
        tensor = gradient[name].detach().cpu().contiguous().to(torch.float64)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0float64\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def gradient_report(gradient: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    expected_names = tuple(gradient)
    if expected_names != tuple(sweep.ACTOR_NAMES):
        raise RuntimeError(f"gradient actor6 order/scope drift: {expected_names}")
    per_tensor: dict[str, Any] = {}
    for name in sweep.ACTOR_NAMES:
        tensor = gradient[name]
        per_tensor[name] = {
            "l2": norm_gradient({name_: tensor if name_ == name else torch.zeros_like(
                gradient[name_]
            ) for name_ in sweep.ACTOR_NAMES}),
            "max_abs": float(tensor.abs().max()),
            "elements": int(tensor.numel()),
            "nonzero_elements": int(torch.count_nonzero(tensor)),
            "finite": bool(torch.isfinite(tensor).all()),
        }
    return {
        "l2": norm_gradient(gradient),
        "finite": all(record["finite"] for record in per_tensor.values()),
        "all_six_tensors_nonzero": all(
            record["nonzero_elements"] > 0 for record in per_tensor.values()
        ),
        "parameter_names": list(sweep.ACTOR_NAMES),
        "vector_sha256_float64": gradient_sha256(gradient),
        "per_tensor": per_tensor,
    }


def compute_objective_gradients(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.nn.Parameter],
    cache: Sequence[dict[str, torch.Tensor]],
    selections: Sequence[Sequence[dict[str, Any]]],
    inventory: Mapping[str, Mapping[str, float | int]],
    device: torch.device,
) -> tuple[dict[str, Gradient], dict[str, float], dict[str, Any]]:
    gradients = {
        objective: zero_gradient_like(parameters) for objective in OBJECTIVES
    }
    numerators = {objective: 0.0 for objective in OBJECTIVES}
    parameter_tuple = tuple(parameters[name] for name in sweep.ACTOR_NAMES)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("parameter .grad buffers existed before probe")

    per_batch: list[dict[str, Any]] = []
    for batch_index, (cpu_batch, identities) in enumerate(
        zip(cache, selections), start=1
    ):
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in cpu_batch.items()
        }
        outputs = sweep.ppo.model_forward(model, batch, device)
        per_row_nll = sweep.ordered_nll_per_row(outputs, batch)
        weights = effective_weights(batch)
        batch_report: dict[str, Any] = {
            "batch": batch_index,
            "rows": int(per_row_nll.shape[0]),
            "objective_numerator": {},
        }
        for objective_index, objective in enumerate(OBJECTIVES):
            mask = torch.tensor(
                [belongs_to_objective(row, objective) for row in identities],
                dtype=torch.bool,
                device=device,
            )
            if not bool(mask.any()):
                raise RuntimeError(f"{objective} absent from batch {batch_index}")
            numerator = (per_row_nll[mask] * weights[mask]).sum()
            denominator = float(inventory[objective]["effective_weight"])
            component = numerator / denominator
            retain_graph = objective_index + 1 < len(OBJECTIVES)
            values = torch.autograd.grad(
                component,
                parameter_tuple,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=False,
                materialize_grads=False,
            )
            if len(values) != len(sweep.ACTOR_NAMES):
                raise RuntimeError("autograd actor6 result count drift")
            for name, value in zip(sweep.ACTOR_NAMES, values):
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(
                        f"nonfinite {objective}/{name} gradient"
                    )
                gradients[objective][name].add_(
                    value.detach().cpu().to(torch.float64)
                )
            numerator_value = float(numerator.detach().cpu())
            numerators[objective] += numerator_value
            batch_report["objective_numerator"][objective] = numerator_value
        per_batch.append(batch_report)
        del outputs, per_row_nll

    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("autograd.grad unexpectedly materialized .grad buffers")
    losses = {
        objective: numerators[objective]
        / float(inventory[objective]["effective_weight"])
        for objective in OBJECTIVES
    }
    if not all(math.isfinite(value) for value in losses.values()):
        raise FloatingPointError("nonfinite objective loss")
    return gradients, losses, {
        "api": "torch.autograd.grad",
        "backward_calls": 0,
        "optimizer_instances_created": 0,
        "optimizer_step_calls": 0,
        "parameter_grad_buffers_before": 0,
        "parameter_grad_buffers_after": 0,
        "forward_batches": len(per_batch),
        "per_batch": per_batch,
    }


def pairwise_geometry(
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            dot = dot_gradient(gradients[left], gradients[right])
            cosine = cosine_gradient(gradients[left], gradients[right])
            result.append(
                {
                    "left": left,
                    "right": right,
                    "dot": dot,
                    "cosine": cosine,
                    "conflict": dot < 0.0,
                }
            )
    return result


def derive_source_mixed_gradients(
    objective_gradients: Mapping[str, Mapping[str, torch.Tensor]],
    losses: Mapping[str, float],
    inventory: Mapping[str, Mapping[str, float | int]],
) -> tuple[dict[str, Gradient], dict[str, float], dict[str, float]]:
    gradients: dict[str, Gradient] = {}
    source_losses: dict[str, float] = {}
    source_weights: dict[str, float] = {}
    for source in SOURCES:
        hard_name = f"{source}_hard"
        retention_name = f"{source}_retention"
        hard_weight = float(inventory[hard_name]["effective_weight"])
        retention_weight = float(
            inventory[retention_name]["effective_weight"]
        )
        total_weight = hard_weight + retention_weight
        gradients[source] = scale_gradient(
            scaled_add(
                scale_gradient(objective_gradients[hard_name], hard_weight),
                objective_gradients[retention_name],
                retention_weight,
            ),
            1.0 / total_weight,
        )
        source_losses[source] = (
            hard_weight * losses[hard_name]
            + retention_weight * losses[retention_name]
        ) / total_weight
        source_weights[source] = total_weight
    return gradients, source_losses, source_weights


def derive_union_retention(
    objective_gradients: Mapping[str, Mapping[str, torch.Tensor]],
    losses: Mapping[str, float],
    inventory: Mapping[str, Mapping[str, float | int]],
) -> tuple[Gradient, float, float]:
    retention_names = [f"{source}_retention" for source in SOURCES]
    total_weight = sum(
        float(inventory[name]["effective_weight"]) for name in retention_names
    )
    gradient = scale_gradient(
        sum_gradients(
            [
                scale_gradient(
                    objective_gradients[name],
                    float(inventory[name]["effective_weight"]),
                )
                for name in retention_names
            ]
        ),
        1.0 / total_weight,
    )
    loss = sum(
        float(inventory[name]["effective_weight"]) * losses[name]
        for name in retention_names
    ) / total_weight
    return gradient, loss, total_weight


def relative_difference(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    delta = scaled_add(left, right, -1.0)
    absolute = norm_gradient(delta)
    reference = max(norm_gradient(left), norm_gradient(right), 1e-300)
    return {"absolute_l2": absolute, "relative_l2": absolute / reference}


def analytic_fresh_adamw_first_step_direction(
    union_gradient: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.nn.Parameter],
) -> tuple[Gradient, dict[str, Any]]:
    """Return the no-write direction of the frozen fresh AdamW first step.

    For ``theta_new = theta - lr * direction``, decoupled AdamW gives
    ``direction = m_hat / (sqrt(v_hat) + eps) + weight_decay * theta``.
    The moment calculation below is retained explicitly even though its first
    bias-corrected value algebraically reduces to the clipped gradient.
    Operations use each parameter's float32 dtype before the returned audit
    vector is promoted to float64, matching the intended optimizer arithmetic
    more closely than a float64 sign approximation would.
    """

    preclip_norm = norm_gradient(union_gradient)
    clip_factor = min(
        1.0, sweep.MAX_GRAD_NORM / (preclip_norm + 1e-6)
    )
    beta1, beta2 = sweep.ADAM_BETAS
    adaptive: Gradient = {}
    decay: Gradient = {}
    direction: Gradient = {}
    per_tensor: dict[str, Any] = {}
    for name in sweep.ACTOR_NAMES:
        parameter = parameters[name].detach().cpu()
        clipped = (
            union_gradient[name]
            .to(dtype=parameter.dtype, device="cpu")
            .mul(float(clip_factor))
        )
        first_moment = clipped.mul(1.0 - beta1)
        first_second_moment = clipped.square().mul(1.0 - beta2)
        corrected_moment = first_moment.div(1.0 - beta1)
        corrected_second_moment = first_second_moment.div(1.0 - beta2)
        adaptive_value = corrected_moment.div(
            corrected_second_moment.sqrt().add(sweep.ADAM_EPS)
        )
        decay_value = parameter.mul(sweep.WEIGHT_DECAY)
        direction_value = adaptive_value.add(decay_value)
        adaptive[name] = adaptive_value.to(torch.float64)
        decay[name] = decay_value.to(torch.float64)
        direction[name] = direction_value.to(torch.float64)
        per_tensor[name] = {
            "clipped_gradient_l2": float(clipped.double().norm()),
            "adaptive_direction_l2": float(adaptive_value.double().norm()),
            "decoupled_weight_decay_direction_l2": float(
                decay_value.double().norm()
            ),
            "combined_direction_l2": float(direction_value.double().norm()),
        }
    direction_norm = norm_gradient(direction)
    return direction, {
        "optimizer": "AdamW",
        "fresh_state": True,
        "gradient_source": "row_weighted_union_mixed",
        "betas": list(sweep.ADAM_BETAS),
        "eps": sweep.ADAM_EPS,
        "weight_decay": sweep.WEIGHT_DECAY,
        "decoupled_weight_decay": True,
        "max_grad_norm": sweep.MAX_GRAD_NORM,
        "preclip_gradient_l2": preclip_norm,
        "clip_denominator_epsilon": 1e-6,
        "clip_factor": clip_factor,
        "clipping_active": clip_factor < 1.0,
        "bias_correction_step": 1,
        "adaptive_component": gradient_report(adaptive),
        "decoupled_weight_decay_component": gradient_report(decay),
        "combined_direction_l2": direction_norm,
        "learning_rate": sweep.LEARNING_RATE,
        "predicted_update_l2": sweep.LEARNING_RATE * direction_norm,
        "predicted_update_convention": "delta_theta = -lr * direction",
        "per_tensor": per_tensor,
        "optimizer_instances_created": 0,
        "optimizer_step_calls": 0,
        "analytic_only": True,
    }


def fixed_order_pcgrad(
    objective_gradients: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[Gradient, dict[str, Any]]:
    task_order = list(REQUIRED_DESCENT_OBJECTIVES)
    projected: dict[str, Gradient] = {}
    audit: list[dict[str, Any]] = []
    for task_index, task in enumerate(task_order):
        current = clone_gradient(objective_gradients[task])
        # A deterministic cyclic order removes any post-result randomization.
        other_order = (
            task_order[task_index + 1 :] + task_order[:task_index]
        )
        for other in other_order:
            reference = objective_gradients[other]
            before = dot_gradient(current, reference)
            reference_norm_square = dot_gradient(reference, reference)
            if reference_norm_square <= 0.0:
                raise RuntimeError(f"zero PCGrad reference gradient: {other}")
            projected_conflict = before < 0.0
            coefficient = (
                before / reference_norm_square if projected_conflict else 0.0
            )
            if projected_conflict:
                current = scaled_add(current, reference, -coefficient)
            after = dot_gradient(current, reference)
            audit.append(
                {
                    "task": task,
                    "reference": other,
                    "dot_before": before,
                    "projection_applied": projected_conflict,
                    "projection_coefficient": coefficient,
                    "dot_after": after,
                }
            )
        projected[task] = current
    candidate = scale_gradient(
        sum_gradients([projected[name] for name in task_order]),
        1.0 / len(task_order),
    )
    return candidate, {
        "task_order": task_order,
        "other_order_rule": "cyclic_after_task_then_before_task",
        "reference_gradient_kind": "original_unprojected",
        "aggregation": "arithmetic_mean_of_projected_task_gradients",
        "projection_trace": audit,
    }


def candidate_effects(
    candidate: Mapping[str, torch.Tensor],
    objective_gradients: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    direction_norm = norm_gradient(candidate)
    if direction_norm <= 0.0 or not math.isfinite(direction_norm):
        raise FloatingPointError("candidate direction is zero/nonfinite")
    effects: dict[str, Any] = {}
    effect_objectives = OBJECTIVES + (RETENTION_GUARD_OBJECTIVE,)
    if tuple(objective_gradients) != effect_objectives:
        raise RuntimeError(
            f"candidate effect objective order drift: {tuple(objective_gradients)}"
        )
    for objective in effect_objectives:
        objective_norm = norm_gradient(objective_gradients[objective])
        dot = dot_gradient(objective_gradients[objective], candidate)
        cosine = dot / (objective_norm * direction_norm)
        effects[objective] = {
            "dot_gradient_with_direction": dot,
            "cosine": cosine,
            "d_loss_d_eta_for_theta_minus_eta_direction": -dot,
            "strict_first_order_descent": dot > 0.0,
            "numerically_robust_first_order_descent": (
                dot > 0.0 and cosine > FIRST_ORDER_COSINE_EPS
            ),
        }
    all_required_strict = all(
        effects[name]["strict_first_order_descent"]
        for name in REQUIRED_DESCENT_OBJECTIVES
    )
    all_required_robust = all(
        effects[name]["numerically_robust_first_order_descent"]
        for name in REQUIRED_DESCENT_OBJECTIVES
    )
    all_gate_strict = all(
        effects[name]["strict_first_order_descent"]
        for name in DIRECTIONAL_GATE_OBJECTIVES
    )
    all_gate_robust = all(
        effects[name]["numerically_robust_first_order_descent"]
        for name in DIRECTIONAL_GATE_OBJECTIVES
    )
    return {
        "gradient": gradient_report(candidate),
        "objective_effects": effects,
        "required_objectives": list(REQUIRED_DESCENT_OBJECTIVES),
        "retention_guard_objective": RETENTION_GUARD_OBJECTIVE,
        "directional_gate_objectives": list(DIRECTIONAL_GATE_OBJECTIVES),
        "all_required_strict_first_order_descent": all_required_strict,
        "all_required_numerically_robust_first_order_descent": (
            all_required_robust
        ),
        "all_directional_gates_strict_first_order_descent": all_gate_strict,
        "all_directional_gates_numerically_robust_first_order_descent": (
            all_gate_robust
        ),
        "minimum_required_cosine": min(
            effects[name]["cosine"] for name in REQUIRED_DESCENT_OBJECTIVES
        ),
        "minimum_directional_gate_cosine": min(
            effects[name]["cosine"] for name in DIRECTIONAL_GATE_OBJECTIVES
        ),
    }


def run_probe(
    cache: Sequence[dict[str, torch.Tensor]],
    selections: Sequence[Sequence[dict[str, Any]]],
    inventory: Mapping[str, Mapping[str, float | int]],
    device: torch.device,
) -> dict[str, Any]:
    random.seed(PROBE_SEED)
    torch.manual_seed(PROBE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PROBE_SEED)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)

    model, _ = sweep.load_raw_u468(device)
    model.eval()
    parameters = sweep.configure_actor6(model)
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if trainable_names != tuple(sweep.ACTOR_NAMES):
        raise RuntimeError(f"actor6 trainable scope drift: {trainable_names}")
    model_hash_before = sweep.ppo.model_state_sha256(model)
    if model_hash_before != sweep.BASE_MODEL_SHA256:
        raise RuntimeError("raw U468 model hash drift before probe")

    objective_gradients, losses, autograd_audit = compute_objective_gradients(
        model, parameters, cache, selections, inventory, device
    )
    model_hash_after = sweep.ppo.model_state_sha256(model)
    if model_hash_after != model_hash_before:
        raise RuntimeError("gradient probe changed model state tensors")

    objective_reports = {
        objective: {
            "loss": losses[objective],
            "rows": int(inventory[objective]["rows"]),
            "effective_weight": float(
                inventory[objective]["effective_weight"]
            ),
            "gradient": gradient_report(objective_gradients[objective]),
        }
        for objective in OBJECTIVES
    }
    if not all(
        report["gradient"]["finite"]
        and report["gradient"]["all_six_tensors_nonzero"]
        for report in objective_reports.values()
    ):
        raise FloatingPointError("objective gradient scope/finite gate failed")

    source_gradients, source_losses, source_weights = (
        derive_source_mixed_gradients(objective_gradients, losses, inventory)
    )
    union_retention_gradient, union_retention_loss, union_retention_weight = (
        derive_union_retention(objective_gradients, losses, inventory)
    )
    effect_gradients = {
        **objective_gradients,
        RETENTION_GUARD_OBJECTIVE: union_retention_gradient,
    }
    union_reconstruction = scale_gradient(
        sum_gradients(
            [
                scale_gradient(source_gradients[source], source_weights[source])
                for source in SOURCES
            ]
        ),
        1.0 / float(inventory["union_mixed"]["effective_weight"]),
    )
    reconstruction_error = relative_difference(
        objective_gradients["union_mixed"], union_reconstruction
    )
    if (
        reconstruction_error["relative_l2"]
        > UNION_RECONSTRUCTION_RELATIVE_TOLERANCE
    ):
        raise RuntimeError(
            f"union/source gradient reconstruction drift: {reconstruction_error}"
        )

    fresh_adamw_direction, fresh_adamw_audit = (
        analytic_fresh_adamw_first_step_direction(
            objective_gradients["union_mixed"], parameters
        )
    )
    candidates: dict[str, Gradient] = {
        FRESH_ADAMW_CANDIDATE: fresh_adamw_direction,
        "row_weighted_union": clone_gradient(
            objective_gradients["union_mixed"]
        ),
        "source_balanced_union": scale_gradient(
            sum_gradients([source_gradients[source] for source in SOURCES]),
            1.0 / len(SOURCES),
        ),
    }
    pcgrad, pcgrad_audit = fixed_order_pcgrad(objective_gradients)
    candidates[PCGRAD_CANDIDATE] = pcgrad
    candidate_reports = {
        name: candidate_effects(candidate, effect_gradients)
        for name, candidate in candidates.items()
    }
    if tuple(candidate_reports) != CANDIDATE_PRIORITY:
        raise RuntimeError("candidate set/order drift")

    non_pcgrad_passes = [
        name
        for name in NON_PCGRAD_CANDIDATES
        if candidate_reports[name][
            "all_directional_gates_numerically_robust_first_order_descent"
        ]
    ]
    pcgrad_needed = not non_pcgrad_passes
    selected = next(
        (
            name
            for name in CANDIDATE_PRIORITY
            if candidate_reports[name][
                "all_directional_gates_numerically_robust_first_order_descent"
            ]
            and (name != PCGRAD_CANDIDATE or pcgrad_needed)
        ),
        None,
    )

    result = {
        "status": "completed_zero_weight_write_gradient_probe",
        "decision": "DIRECTION_FOUND" if selected is not None else "NO_DIRECTION",
        "device": str(device),
        "seed": PROBE_SEED,
        "aggregate_point": {
            "checkpoint": str(sweep.U468.relative_to(ROOT)),
            "update": 468,
            "model_state_sha256": model_hash_before,
            "trajectory": "single_raw_u468_point_no_b1_b2_parameter_updates",
            "rows": 512,
            "batch_partition_only": [256, 256],
        },
        "loss_definition": {
            "kind": "ordered_plackett_luce_nll",
            "sample_weight": "frozen_cache_sample_weights",
            "context34_order_multiplier": sweep.ORDER_CONTEXT_WEIGHT,
            "non_context34_order_multiplier": 1.0,
            "mean_denominator": "sum_of_effective_row_weights_per_objective",
        },
        "objective_order": list(OBJECTIVES),
        "objectives": objective_reports,
        "pairwise_objective_geometry": pairwise_geometry(
            objective_gradients, OBJECTIVES
        ),
        "derived_source_mixed": {
            source: {
                "loss": source_losses[source],
                "effective_weight": source_weights[source],
                "gradient": gradient_report(source_gradients[source]),
            }
            for source in SOURCES
        },
        "derived_union_retention": {
            "loss": union_retention_loss,
            "rows": sum(
                int(inventory[f"{source}_retention"]["rows"])
                for source in SOURCES
            ),
            "effective_weight": union_retention_weight,
            "gradient": gradient_report(union_retention_gradient),
            "derivation": (
                "effective-weighted mean of three source retention objectives"
            ),
        },
        "union_from_sources_reconstruction": {
            **reconstruction_error,
            "relative_tolerance": UNION_RECONSTRUCTION_RELATIVE_TOLERANCE,
            "pass": reconstruction_error["relative_l2"]
            <= UNION_RECONSTRUCTION_RELATIVE_TOLERANCE,
        },
        "candidate_contract": {
            "finite_set": list(CANDIDATE_PRIORITY),
            "priority": list(CANDIDATE_PRIORITY),
            FRESH_ADAMW_CANDIDATE: (
                "analytic no-write fresh AdamW first-step direction from exact "
                "row-weighted union gradient, including clipping, eps, bias "
                "correction, and decoupled weight decay"
            ),
            "row_weighted_union": "exact union_mixed mean-loss gradient",
            "source_balanced_union": (
                "arithmetic mean of the three source mixed mean-loss gradients"
            ),
            PCGRAD_CANDIDATE: (
                "fixed cyclic-order PCGrad over union and three hard gradients"
            ),
            "pcgrad_eligible_only_if_no_non_pcgrad_candidate_passes": True,
            "update_convention": "theta_new = theta - eta * direction",
            "required_descent_objectives": list(REQUIRED_DESCENT_OBJECTIVES),
            "retention_guard_objective": RETENTION_GUARD_OBJECTIVE,
            "directional_gate_objectives": list(DIRECTIONAL_GATE_OBJECTIVES),
            "strict_descent_test": "dot(objective_gradient,direction) > 0",
            "robust_descent_test": (
                "strict test and cosine > first_order_cosine_epsilon"
            ),
            "first_order_cosine_epsilon": FIRST_ORDER_COSINE_EPS,
        },
        "candidates": candidate_reports,
        "fresh_adamw_first_step": fresh_adamw_audit,
        "pcgrad": {
            "needed_for_selection": pcgrad_needed,
            **pcgrad_audit,
        },
        "selection": {
            "non_pcgrad_candidates_passing": non_pcgrad_passes,
            "selected_direction": selected,
            "all_passing_directions": [
                name
                for name in CANDIDATE_PRIORITY
                if candidate_reports[name][
                    "all_directional_gates_numerically_robust_first_order_descent"
                ]
            ],
            "rule": (
                "fresh union AdamW first; then fixed non-PCGrad references; "
                "PCGrad only if no non-PCGrad candidate passes"
            ),
        },
        "integrity": {
            "actor6_scope_exact": trainable_names == tuple(sweep.ACTOR_NAMES),
            "all_objective_gradients_finite": all(
                report["gradient"]["finite"]
                for report in objective_reports.values()
            ),
            "all_objectives_have_six_nonzero_actor_tensors": all(
                report["gradient"]["all_six_tensors_nonzero"]
                for report in objective_reports.values()
            ),
            "model_state_sha256_before": model_hash_before,
            "model_state_sha256_after": model_hash_after,
            "model_weights_unchanged": model_hash_after == model_hash_before,
            "validation_member_payloads_opened": False,
            "train_only": True,
            "optimizer_instances_created": 0,
            "optimizer_step_calls": 0,
            "backward_calls": 0,
            "parameter_grad_buffers_materialized": 0,
            "checkpoint_writes": 0,
            "model_writes": 0,
            "result_artifact_writes": 0,
            "stdout_only": True,
        },
        "autograd_audit": autograd_audit,
    }
    if not sweep.repair.finite_nested(result):
        raise FloatingPointError("probe result contains a nonfinite value")
    return result


def common_report(
    mode: str,
    self_evidence: Mapping[str, Any],
    frozen_evidence: Mapping[str, Any],
    selection_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "zero_write_audit_passed",
        "mode": mode,
        "tool": dict(self_evidence),
        "frozen_inputs": dict(frozen_evidence),
        "selection": dict(selection_summary),
        "contract": {
            "aggregate_rows": 512,
            "batch_partition_only": [256, 256],
            "base": "raw full U468",
            "actor_parameter_names": list(sweep.ACTOR_NAMES),
            "train_only": True,
            "validation": False,
            "optimizer": False,
            "backward": False,
            "optimizer_step": False,
            "parameter_update": False,
            "checkpoint_write": False,
            "model_write": False,
            "result_artifact_write": False,
            "stdout_only": True,
            "candidate_priority": list(CANDIDATE_PRIORITY),
            "required_descent_objectives": list(REQUIRED_DESCENT_OBJECTIVES),
            "retention_guard_objective": RETENTION_GUARD_OBJECTIVE,
            "directional_gate_objectives": list(DIRECTIONAL_GATE_OBJECTIVES),
            "fresh_adamw_first_priority": True,
        },
        "writes_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("static-audit", "cache-audit", "probe"),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    if args.mode == "probe" and args.device != "cuda":
        raise ValueError("formal aggregate gradient probe is CUDA-only")

    self_path = Path(__file__).resolve()
    if self_path.parent != TOOLS:
        raise ValueError("probe tool must remain in repository tools directory")
    self_evidence = require_single_link_regular(self_path, None, "probe tool")
    selections, selection_summary, frozen_evidence = load_frozen_selection()
    common = common_report(
        args.mode, self_evidence, frozen_evidence, selection_summary
    )
    if args.mode == "static-audit":
        print(json.dumps(common, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return

    cache, cache_audit, inputs = load_frozen_cache(selections)
    inventory = objective_inventory(cache, selections)
    cache_payload = common | {
        "inputs": inputs,
        "cache": cache_audit,
        "objective_inventory": inventory,
        "validation_member_payloads_opened": False,
    }
    if args.mode == "cache-audit":
        print(
            json.dumps(
                cache_payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    probe = run_probe(cache, selections, inventory, device)
    payload = cache_payload | {
        "status": "completed_zero_weight_write_gradient_probe",
        "probe": probe,
    }
    if not sweep.repair.finite_nested(payload):
        raise FloatingPointError("final payload contains a nonfinite value")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
