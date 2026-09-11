#!/usr/bin/env python3
"""Zero-write direct512 actor6 multi-objective gradient probe.

This probe is deliberately separate from every earlier aggregate-gradient
probe and result.  It SHA-binds the frozen aggregate512 runner, consumes that
runner's independently collated direct512 tensor batch and exact flat row
identities, and never updates a parameter.

The native CUDA path performs one BF16-autocast direct512 forward.  Every
reported objective is then formed by a fresh call to the complete ordered BC
composite loss and differentiated directly with ``torch.autograd.grad``.
There is no BF16 post-hoc construction of the union, source-hard, aggregate
retention, or source-balanced gradients used for candidate selection.

An independent FP32 structural audit bypasses ``ppo.model_forward`` and calls
``model(batch)`` with CUDA autocast disabled.  Its union and six disjoint
source/category partition gradients are all direct composite gradients.  The
effective-weighted six-partition reconstruction of the FP32 union is a hard
gate.  The analogous native-BF16 reconstruction is diagnostic only.

The sole selectable suggestion is frozen in source: the native-BF16 gradient
of the direct scalar ``0.5 * L_union + (L_flg_hard + L_pokemonfan_hard +
L_core5_hard) / 6``, followed (outside this tool) by one plain-SGD step with
learning rate 5e-5, momentum zero, and weight decay zero.  Its scalar is
differentiated in one VJP; it is never reconstructed from extracted gradient
vectors.  Raw-union SGD geometry is reported only as a non-selectable
diagnostic, so this probe cannot authorize multiple endpoints.

The tool has no optimizer, backward, step, checkpoint, model, or result-write
path.  All evidence is emitted as one JSON object on stdout.
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


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
AGGREGATE_RUNNER = TOOLS / "run_u468_raw_actor6_aggregate512_shadow.py"
AGGREGATE_RUNNER_SHA256 = (
    "db7d6ca06b83d9d035e42476a4282c0ab25b8d532a589f3be80bb2299748b641"
)


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
        raise RuntimeError(f"{label} is not a single-link regular file")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


# The aggregate runner is authenticated before any of its code is imported.
AGGREGATE_RUNNER_EVIDENCE = require_single_link_regular(
    AGGREGATE_RUNNER,
    AGGREGATE_RUNNER_SHA256,
    "frozen aggregate512 runner",
)
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_u468_direct512_db7d6ca0", AGGREGATE_RUNNER
)
if _AGGREGATE_SPEC is None or _AGGREGATE_SPEC.loader is None:
    raise RuntimeError("cannot construct aggregate512 runner import spec")
aggregate: ModuleType = importlib.util.module_from_spec(_AGGREGATE_SPEC)
_AGGREGATE_SPEC.loader.exec_module(aggregate)

torch = aggregate.torch
ppo = aggregate.ppo
frozen = aggregate.frozen


SCHEMA = "ptcg-u468-raw-direct512-actor6-multiobjective-probe-v1"
PROBE_SEED = 202608114
ROWS = 512
SOURCES = ("flg", "pokemonfan", "core5")
PARTITIONS = (
    "flg_hard",
    "flg_retention",
    "pokemonfan_hard",
    "pokemonfan_retention",
    "core5_hard",
    "core5_retention",
)
EQUAL_BLEND_OBJECTIVE = "equal_blend_union_and_hard_balanced"
NATIVE_BASE_DIRECT_OBJECTIVES = (
    "union",
    *PARTITIONS,
    "retention",
)
NATIVE_DIRECT_OBJECTIVES = (EQUAL_BLEND_OBJECTIVE, *NATIVE_BASE_DIRECT_OBJECTIVES)
FP32_DIRECT_OBJECTIVES = ("union", *PARTITIONS)
GATE_OBJECTIVES = (
    "union",
    "flg_hard",
    "pokemonfan_hard",
    "core5_hard",
    "retention",
)

EQUAL_BLEND_CANDIDATE = "native_bf16_equal_blend_direct_composite_plain_sgd"
RAW_UNION_DIAGNOSTIC = "native_bf16_raw_union_plain_sgd_diagnostic"
SELECTABLE_CANDIDATES = (EQUAL_BLEND_CANDIDATE,)
DIAGNOSTIC_DIRECTIONS = (RAW_UNION_DIAGNOSTIC,)

FIRST_ORDER_COSINE_EPS = 1e-8
EQUAL_BLEND_ALPHA = 0.5
SUGGESTED_SGD_LEARNING_RATE = 5e-5
SUGGESTED_SGD_MOMENTUM = 0.0
SUGGESTED_SGD_WEIGHT_DECAY = 0.0
FP32_RECONSTRUCTION_RELATIVE_L2_MAX = 1e-6
FP32_RECONSTRUCTION_MAX_ABS_MAX = 5e-7
WEIGHT_ABS_TOLERANCE = 5e-5
EXPECTED_DIRECT_CACHE_SHA256 = (
    "ef5ab1b8f7e7162316e8ae80a6621f902e9a8e8cf73bb54daaf99e1355d0114a"
)
EXPECTED_DIRECT_BATCH_SHA256 = (
    "1c0bd23912b85dcbc64318d42ae41ba1fd7217f80e9a741e8970d91adeb1db05"
)
EXPECTED_FLAT_IDENTITY_SHA256 = (
    "d0c148e1c992030a9c5442b7a407b50816a30fdec27488f4f79b26b9e5916ce8"
)
IDENTITY_FIELDS = (
    "source",
    "member",
    "line_index",
    "line_sha256",
    "episode_id",
    "team_name",
    "category",
    "context",
    "min_count",
    "max_count",
    "expert_order",
)
EXPECTED_PARTITION_ROWS = {
    "flg_hard": 96,
    "flg_retention": 32,
    "pokemonfan_hard": 64,
    "pokemonfan_retention": 128,
    "core5_hard": 64,
    "core5_retention": 128,
}
# Rational targets implied by the frozen direct512 row/sample-weight contract.
EXPECTED_EFFECTIVE_WEIGHTS = {
    "flg_hard": 78.75,
    "flg_retention": 30.333333333333332,
    "pokemonfan_hard": 47.25,
    "pokemonfan_retention": 98.33333333333333,
    "core5_hard": 64.0,
    "core5_retention": 131.33333333333334,
    "union": 450.0,
}


Gradient = dict[str, torch.Tensor]


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


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    if Path(aggregate.__file__).resolve() != AGGREGATE_RUNNER.resolve():
        raise RuntimeError("wrong aggregate512 runner module imported")
    if sha256_file(AGGREGATE_RUNNER) != AGGREGATE_RUNNER_SHA256:
        raise RuntimeError("aggregate512 runner changed after hash-bound import")


def assert_frozen_aggregate_contract() -> None:
    observed = {
        "rows": aggregate.ROWS,
        "actor_names": tuple(aggregate.ACTOR_NAMES),
        "order_context_weight": aggregate.ORDER_CONTEXT_WEIGHT,
        "direct_cache_sha256": aggregate.EXPECTED_UNION_CACHE_SHA256,
        "direct_batch_sha256": aggregate.EXPECTED_UNION_BATCH_SHA256,
        "source_counts": dict(aggregate.EXPECTED_SOURCE_COUNTS),
        "category_counts": {
            source: dict(counts)
            for source, counts in aggregate.EXPECTED_CATEGORY_COUNTS.items()
        },
        "selection_sha256": tuple(frozen.EXPECTED_STEP_SELECTION_SHA256),
    }
    expected = {
        "rows": ROWS,
        "actor_names": tuple(frozen.ACTOR_NAMES),
        "order_context_weight": 8.0,
        "direct_cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
        "direct_batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
        "source_counts": {"flg": 128, "pokemonfan": 192, "core5": 192},
        "category_counts": {
            "flg": {"hard": 96, "fragile": 30, "c34": 2},
            "pokemonfan": {"hard": 64, "fragile": 126, "c34": 2},
            "core5": {"hard": 64, "fragile": 126, "c34": 2},
        },
        "selection_sha256": (
            "6ff7becab756a4ed528195b22828e4abbaedbd3603b77ac87867d0310f875b6e",
            "6475bf85e4fa204811acc1f4628b8e28a6b15081c0e2ce8c6b0e08413a02cdd1",
        ),
    }
    if observed != expected:
        raise RuntimeError(
            "frozen aggregate512 contract drift: "
            + json.dumps(observed, ensure_ascii=False, sort_keys=True)
        )


def ast_audit() -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    def dotted_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [
        dotted_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    forbidden_suffixes = (
        ".backward",
        ".step",
        ".zero_grad",
        ".write_text",
        ".write_bytes",
        ".save",
    )
    forbidden_calls = sorted(
        name for name in calls if any(name.endswith(suffix) for suffix in forbidden_suffixes)
    )
    optimizer_calls = sorted(name for name in calls if name.startswith("torch.optim"))
    if forbidden_calls or optimizer_calls:
        raise RuntimeError(
            f"zero-write/zero-update AST gate failed: {forbidden_calls + optimizer_calls}"
        )
    if calls.count("torch.autograd.grad") != 1:
        raise RuntimeError("probe must have exactly one autograd.grad call site")
    if calls.count("ppo.bc_expert_actor_loss") != 1:
        raise RuntimeError("probe must have exactly one composite-loss call site")
    if calls.count("ppo.model_forward") != 1:
        raise RuntimeError("native BF16 forward call-site drift")
    return {
        "status": "zero_write_static_ast_audit_passed",
        "torch_autograd_grad_call_sites": 1,
        "composite_loss_call_sites": 1,
        "ppo_model_forward_call_sites": 1,
        "forbidden_update_or_write_calls": [],
        "optimizer_constructor_calls": [],
        "aggregate_runner_hash_checked_before_import": True,
        "stdout_only": True,
    }


def flat_identities(
    selections: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    flat = [row for batch in selections for row in batch]
    if len(flat) != ROWS:
        raise RuntimeError("direct512 flat identity row-count drift")
    payload = [{key: row[key] for key in IDENTITY_FIELDS} for row in flat]
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if digest != EXPECTED_FLAT_IDENTITY_SHA256:
        raise RuntimeError(f"direct512 flat identity SHA-256 drift: {digest}")
    if len({(row["source"], row["member"], row["line_index"]) for row in payload}) != ROWS:
        raise RuntimeError("direct512 flat identities are not unique")
    return payload


def partition_name(identity: Mapping[str, Any]) -> str:
    source = str(identity["source"])
    category = str(identity["category"])
    if source not in SOURCES:
        raise RuntimeError(f"unexpected source: {source}")
    if category == "hard":
        return f"{source}_hard"
    if category in {"fragile", "c34"}:
        return f"{source}_retention"
    raise RuntimeError(f"unexpected category: {category}")


def build_masks_and_weights(
    union: Mapping[str, torch.Tensor],
    identities: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor]:
    if int(union["action_counts"].shape[0]) != ROWS or len(identities) != ROWS:
        raise RuntimeError("direct512 tensor/identity alignment drift")
    if not bool((union["action_counts"] > 0).all()):
        raise RuntimeError("direct512 contains an inactive expert row")
    cpu_masks = {
        name: torch.tensor(
            [partition_name(identity) == name for identity in identities],
            dtype=torch.bool,
        )
        for name in PARTITIONS
    }
    cover = torch.stack([cpu_masks[name] for name in PARTITIONS]).sum(dim=0)
    if not torch.equal(cover, torch.ones_like(cover)):
        raise RuntimeError("six direct512 partitions are not disjoint/exhaustive")
    cpu_masks["union"] = torch.ones(ROWS, dtype=torch.bool)
    cpu_masks["retention"] = torch.stack(
        [cpu_masks[f"{source}_retention"] for source in SOURCES]
    ).any(dim=0)

    sample_weights = union["sample_weights"].float().cpu()
    effective_weights = sample_weights * torch.where(
        union["contexts"].cpu() == ppo.SKILL_ORDER_CONTEXT,
        torch.full_like(sample_weights, aggregate.ORDER_CONTEXT_WEIGHT),
        torch.ones_like(sample_weights),
    )
    observed_rows = {
        name: int(cpu_masks[name].sum()) for name in PARTITIONS
    }
    observed_weights = {
        name: float(effective_weights[cpu_masks[name]].sum())
        for name in PARTITIONS
    }
    observed_weights["union"] = float(effective_weights.sum())
    row_checks = {
        name: observed_rows[name] == expected
        for name, expected in EXPECTED_PARTITION_ROWS.items()
    }
    weight_checks = {
        name: math.isclose(
            observed_weights[name],
            expected,
            rel_tol=0.0,
            abs_tol=WEIGHT_ABS_TOLERANCE,
        )
        for name, expected in EXPECTED_EFFECTIVE_WEIGHTS.items()
    }
    partition_weight_sum = sum(observed_weights[name] for name in PARTITIONS)
    partition_sum_close = math.isclose(
        partition_weight_sum,
        observed_weights["union"],
        rel_tol=0.0,
        abs_tol=WEIGHT_ABS_TOLERANCE,
    )
    if not all(row_checks.values()):
        raise RuntimeError(f"partition row-count gate failed: {observed_rows}")
    if not all(weight_checks.values()) or not partition_sum_close:
        raise RuntimeError(f"partition effective-weight gate failed: {observed_weights}")

    source_weights = {
        source: observed_weights[f"{source}_hard"]
        + observed_weights[f"{source}_retention"]
        for source in SOURCES
    }
    return cpu_masks, {
        "partition_rows": observed_rows,
        "expected_partition_rows": dict(EXPECTED_PARTITION_ROWS),
        "row_counts_exact": all(row_checks.values()),
        "effective_weights": observed_weights,
        "expected_effective_weights": dict(EXPECTED_EFFECTIVE_WEIGHTS),
        "effective_weight_abs_tolerance": WEIGHT_ABS_TOLERANCE,
        "effective_weights_close": all(weight_checks.values()),
        "partition_effective_weight_sum": partition_weight_sum,
        "partition_weight_sum_close_to_union": partition_sum_close,
        "source_effective_weights": source_weights,
        "six_partitions_disjoint_and_exhaustive": True,
        "aggregate_retention_rows": int(cpu_masks["retention"].sum()),
    }, effective_weights


def zero_gradient(parameters: Mapping[str, torch.nn.Parameter]) -> Gradient:
    return {
        name: torch.zeros_like(parameter.detach(), device="cpu", dtype=torch.float64)
        for name, parameter in parameters.items()
    }


def clone_gradient(gradient: Mapping[str, torch.Tensor]) -> Gradient:
    return {name: gradient[name].clone() for name in aggregate.ACTOR_NAMES}


def scale_gradient(gradient: Mapping[str, torch.Tensor], scale: float) -> Gradient:
    return {
        name: gradient[name] * float(scale) for name in aggregate.ACTOR_NAMES
    }


def add_scaled(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    scale: float,
) -> Gradient:
    return {
        name: left[name] + float(scale) * right[name]
        for name in aggregate.ACTOR_NAMES
    }


def sum_gradients(gradients: Sequence[Mapping[str, torch.Tensor]]) -> Gradient:
    if not gradients:
        raise ValueError("cannot sum zero gradients")
    result = {
        name: torch.zeros_like(gradients[0][name]) for name in aggregate.ACTOR_NAMES
    }
    for gradient in gradients:
        result = add_scaled(result, gradient, 1.0)
    return result


def dot_gradient(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    return float(
        sum(
            torch.dot(left[name].reshape(-1), right[name].reshape(-1))
            for name in aggregate.ACTOR_NAMES
        )
    )


def norm_gradient(gradient: Mapping[str, torch.Tensor]) -> float:
    square = dot_gradient(gradient, gradient)
    if square < 0.0 or not math.isfinite(square):
        raise FloatingPointError(f"invalid squared norm: {square}")
    return math.sqrt(square)


def normalize_gradient(gradient: Mapping[str, torch.Tensor]) -> Gradient:
    norm = norm_gradient(gradient)
    if norm <= 0.0:
        raise FloatingPointError("cannot normalize zero gradient")
    return scale_gradient(gradient, 1.0 / norm)


def gradient_sha256(gradient: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in aggregate.ACTOR_NAMES:
        value = gradient[name].detach().cpu().contiguous().to(torch.float64)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0float64\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def gradient_report(gradient: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if tuple(gradient) != tuple(aggregate.ACTOR_NAMES):
        raise RuntimeError("gradient actor6 order/scope drift")
    per_tensor = {
        name: {
            "shape": list(gradient[name].shape),
            "l2": float(gradient[name].norm()),
            "max_abs": float(gradient[name].abs().max()),
            "elements": int(gradient[name].numel()),
            "nonzero_elements": int(torch.count_nonzero(gradient[name])),
            "finite": bool(torch.isfinite(gradient[name]).all()),
        }
        for name in aggregate.ACTOR_NAMES
    }
    return {
        "l2": norm_gradient(gradient),
        "max_abs": max(record["max_abs"] for record in per_tensor.values()),
        "finite": all(record["finite"] for record in per_tensor.values()),
        "all_six_tensors_nonzero": all(
            record["nonzero_elements"] > 0 for record in per_tensor.values()
        ),
        "parameter_names": list(aggregate.ACTOR_NAMES),
        "vector_sha256_float64": gradient_sha256(gradient),
        "per_tensor": per_tensor,
    }


def masked_composite_batch(
    batch: Mapping[str, torch.Tensor], multiplier: torch.Tensor
) -> dict[str, torch.Tensor]:
    if multiplier.shape != batch["sample_weights"].shape:
        raise RuntimeError("objective multiplier shape drift")
    result = dict(batch)
    result["sample_weights"] = batch["sample_weights"] * multiplier.to(
        device=batch["sample_weights"].device,
        dtype=batch["sample_weights"].dtype,
    )
    if not bool(torch.isfinite(result["sample_weights"]).all()):
        raise FloatingPointError("objective sample weights are nonfinite")
    return result


def compute_direct_composite_gradients(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.nn.Parameter],
    multipliers: Mapping[str, torch.Tensor],
    objective_order: Sequence[str],
) -> tuple[dict[str, Gradient], dict[str, Any]]:
    if any(parameter.grad is not None for parameter in parameters.values()):
        raise RuntimeError("actor6 .grad buffer existed before autograd probe")
    scalars: dict[str, torch.Tensor] = {}
    parts: dict[str, dict[str, torch.Tensor]] = {}
    base_objective_order = [
        objective for objective in objective_order if objective != EQUAL_BLEND_OBJECTIVE
    ]
    for objective in base_objective_order:
        objective_batch = masked_composite_batch(batch, multipliers[objective])
        scalar, scalar_parts = ppo.bc_expert_actor_loss(
            dict(outputs),
            objective_batch,
            loss_mode="ordered",
            order_context_weight=aggregate.ORDER_CONTEXT_WEIGHT,
            non_context34_fixed_multi_action_order_weight=1.0,
        )
        if scalar.ndim != 0 or not bool(torch.isfinite(scalar)):
            raise FloatingPointError(f"nonfinite/non-scalar composite: {objective}")
        scalars[objective] = scalar
        parts[objective] = scalar_parts

    if EQUAL_BLEND_OBJECTIVE in objective_order:
        required = ("union", "flg_hard", "pokemonfan_hard", "core5_hard")
        if any(name not in scalars for name in required):
            raise RuntimeError("equal-blend scalar is missing a direct base composite")
        hard_component = (
            scalars["flg_hard"]
            + scalars["pokemonfan_hard"]
            + scalars["core5_hard"]
        ) / 6.0
        union_component = EQUAL_BLEND_ALPHA * scalars["union"]
        equal_blend = union_component + hard_component
        if equal_blend.ndim != 0 or not bool(torch.isfinite(equal_blend)):
            raise FloatingPointError("equal-blend direct composite is nonfinite")
        scalars[EQUAL_BLEND_OBJECTIVE] = equal_blend
        parts[EQUAL_BLEND_OBJECTIVE] = {
            "equal_blend_total": equal_blend,
            "half_union_component": union_component,
            "one_sixth_each_three_hards_component": hard_component,
        }

    parameter_tuple = tuple(parameters[name] for name in aggregate.ACTOR_NAMES)
    gradients: dict[str, Gradient] = {}
    for index, objective in enumerate(objective_order):
        values = torch.autograd.grad(
            scalars[objective],
            parameter_tuple,
            retain_graph=index + 1 < len(objective_order),
            create_graph=False,
            allow_unused=False,
            materialize_grads=False,
        )
        gradient = zero_gradient(parameters)
        for name, value in zip(aggregate.ACTOR_NAMES, values):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"nonfinite gradient: {objective}/{name}")
            gradient[name] = value.detach().cpu().to(torch.float64)
        gradients[objective] = gradient

    if any(parameter.grad is not None for parameter in parameters.values()):
        raise RuntimeError("autograd.grad unexpectedly materialized actor6 .grad")
    reports = {
        objective: {
            "loss": float(scalars[objective].detach().cpu()),
            "loss_parts": {
                key: float(value.detach().cpu())
                for key, value in parts[objective].items()
            },
            "gradient": gradient_report(gradients[objective]),
            "gradient_origin": (
                "direct bc_expert_actor_loss scalar then torch.autograd.grad"
            ),
        }
        for objective in objective_order
    }
    return gradients, reports


def difference_report(
    reference: Mapping[str, torch.Tensor],
    reconstruction: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    difference = add_scaled(reconstruction, reference, -1.0)
    absolute_l2 = norm_gradient(difference)
    reference_l2 = norm_gradient(reference)
    max_abs = max(float(value.abs().max()) for value in difference.values())
    return {
        "absolute_l2": absolute_l2,
        "relative_l2": absolute_l2 / max(reference_l2, 1e-30),
        "max_abs": max_abs,
        "reference_l2": reference_l2,
    }


def effective_weighted_partition_reconstruction(
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
    effective_weights: Mapping[str, float],
) -> Gradient:
    union_weight = float(effective_weights["union"])
    if union_weight <= 0.0:
        raise RuntimeError("union effective weight is nonpositive")
    return scale_gradient(
        sum_gradients(
            [
                scale_gradient(gradients[name], float(effective_weights[name]))
                for name in PARTITIONS
            ]
        ),
        1.0 / union_weight,
    )


def flat_unflat_layout_audit(
    gradient: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    shapes = {name: list(gradient[name].shape) for name in aggregate.ACTOR_NAMES}
    flat = torch.cat(
        [gradient[name].to(torch.float32).reshape(-1) for name in aggregate.ACTOR_NAMES]
    )
    rebuilt: Gradient = {}
    offset = 0
    offsets: dict[str, list[int]] = {}
    for name in aggregate.ACTOR_NAMES:
        count = gradient[name].numel()
        rebuilt[name] = flat[offset : offset + count].reshape(gradient[name].shape).to(
            torch.float64
        )
        offsets[name] = [offset, offset + count]
        offset += count
    if offset != int(flat.numel()):
        raise RuntimeError("flat/unflat element-count drift")
    fp32 = difference_report(
        {name: gradient[name].to(torch.float32).to(torch.float64) for name in aggregate.ACTOR_NAMES},
        rebuilt,
    )
    bf16_flat = flat.to(torch.bfloat16).to(torch.float32)
    bf16_rebuilt: Gradient = {}
    offset = 0
    for name in aggregate.ACTOR_NAMES:
        count = gradient[name].numel()
        bf16_rebuilt[name] = bf16_flat[offset : offset + count].reshape(
            gradient[name].shape
        ).to(torch.float64)
        offset += count
    bf16 = difference_report(
        {name: gradient[name].to(torch.float32).to(torch.float64) for name in aggregate.ACTOR_NAMES},
        bf16_rebuilt,
    )
    return {
        "parameter_order": list(aggregate.ACTOR_NAMES),
        "shapes": shapes,
        "offsets_half_open": offsets,
        "total_elements": int(flat.numel()),
        "fp32_flat_unflat": {
            **fp32,
            "pass": fp32["relative_l2"] <= 1e-6 and fp32["max_abs"] <= 5e-7,
        },
        "bf16_roundtrip_diagnostic_only": bf16,
    }


def candidate_effects(
    direction: Mapping[str, torch.Tensor],
    objective_gradients: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    direction_norm = norm_gradient(direction)
    if direction_norm <= 0.0 or not math.isfinite(direction_norm):
        raise RuntimeError(f"candidate direction is zero/nonfinite: {direction_norm}")
    effects: dict[str, Any] = {}
    for objective in GATE_OBJECTIVES:
        gradient = objective_gradients[objective]
        objective_norm = norm_gradient(gradient)
        dot = dot_gradient(gradient, direction)
        cosine = dot / (objective_norm * direction_norm)
        effects[objective] = {
            "dot_gradient_with_direction": dot,
            "cosine": cosine,
            "predicted_loss_derivative_for_theta_minus_lr_direction": -dot,
            "strict_descent": dot > 0.0,
            "robust_descent": dot > 0.0 and cosine > FIRST_ORDER_COSINE_EPS,
        }
    all_robust = all(effects[name]["robust_descent"] for name in GATE_OBJECTIVES)
    return {
        "direction": gradient_report(direction),
        "objective_effects": effects,
        "all_gate_objectives_robust_descent": all_robust,
        "minimum_gate_cosine": min(
            effects[name]["cosine"] for name in GATE_OBJECTIVES
        ),
        "frozen_plain_sgd_suggestion": {
            "learning_rate": SUGGESTED_SGD_LEARNING_RATE,
            "momentum": SUGGESTED_SGD_MOMENTUM,
            "weight_decay": SUGGESTED_SGD_WEIGHT_DECAY,
            "predicted_displacement_l2": (
                SUGGESTED_SGD_LEARNING_RATE * direction_norm
            ),
            "convention": "theta_new = theta - lr * raw_direction",
        },
    }


def pairwise_geometry(
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_norm = norm_gradient(gradients[left])
            right_norm = norm_gradient(gradients[right])
            dot = dot_gradient(gradients[left], gradients[right])
            result.append(
                {
                    "left": left,
                    "right": right,
                    "dot": dot,
                    "cosine": dot / (left_norm * right_norm),
                }
            )
    return result


def run_probe(
    union_cpu: Mapping[str, torch.Tensor],
    identities: Sequence[Mapping[str, Any]],
    masks_cpu: Mapping[str, torch.Tensor],
    weight_audit: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual direct512 multi-objective probe is CUDA-only")
    random.seed(PROBE_SEED)
    torch.manual_seed(PROBE_SEED)
    torch.cuda.manual_seed_all(PROBE_SEED)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)

    model, _ = frozen.load_raw_u468(device)
    model.eval()
    parameters = frozen.configure_actor6(model)
    if tuple(parameters) != tuple(aggregate.ACTOR_NAMES):
        raise RuntimeError("actor6 parameter order drift")
    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if trainable != tuple(aggregate.ACTOR_NAMES):
        raise RuntimeError(f"actor6 trainable scope drift: {trainable}")
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        raise RuntimeError("raw U468 actor6 storage dtype is not FP32")
    model_hash_before = ppo.model_state_sha256(model)
    if model_hash_before != frozen.BASE_MODEL_SHA256:
        raise RuntimeError("raw U468 model state hash drift")

    batch = {
        key: value.to(device, non_blocking=True) for key, value in union_cpu.items()
    }
    multipliers = {
        name: value.to(device, non_blocking=True) for name, value in masks_cpu.items()
    }

    # Native path: this exact helper enables CUDA BF16 autocast.  Every
    # objective below remains a direct composite scalar from the same outputs.
    native_outputs = ppo.model_forward(model, batch, device)
    native_output_dtypes = {
        key: str(value.dtype) for key, value in native_outputs.items()
    }
    if native_outputs["policy_logits"].dtype != torch.bfloat16:
        raise RuntimeError(
            f"native policy logits are not BF16: {native_outputs['policy_logits'].dtype}"
        )
    native_gradients, native_reports = compute_direct_composite_gradients(
        native_outputs,
        batch,
        parameters,
        multipliers,
        NATIVE_DIRECT_OBJECTIVES,
    )
    del native_outputs

    native_reconstruction = effective_weighted_partition_reconstruction(
        native_gradients, weight_audit["effective_weights"]
    )
    native_reconstruction_report = difference_report(
        native_gradients["union"], native_reconstruction
    )

    # Structural path: explicitly bypass ppo.model_forward/autocast.
    with torch.autocast(device_type="cuda", enabled=False):
        fp32_outputs = model(batch)
    fp32_output_dtypes = {key: str(value.dtype) for key, value in fp32_outputs.items()}
    if any(value.dtype != torch.float32 for value in fp32_outputs.values()):
        raise RuntimeError(f"FP32 direct model output dtype drift: {fp32_output_dtypes}")
    fp32_gradients, fp32_reports = compute_direct_composite_gradients(
        fp32_outputs,
        batch,
        parameters,
        multipliers,
        FP32_DIRECT_OBJECTIVES,
    )
    del fp32_outputs

    fp32_reconstruction = effective_weighted_partition_reconstruction(
        fp32_gradients, weight_audit["effective_weights"]
    )
    fp32_reconstruction_report = difference_report(
        fp32_gradients["union"], fp32_reconstruction
    )
    fp32_reconstruction_pass = (
        fp32_reconstruction_report["relative_l2"]
        <= FP32_RECONSTRUCTION_RELATIVE_L2_MAX
        and fp32_reconstruction_report["max_abs"]
        <= FP32_RECONSTRUCTION_MAX_ABS_MAX
    )
    if not fp32_reconstruction_pass:
        raise RuntimeError(
            "FP32 direct union/six-partition reconstruction gate failed: "
            + json.dumps(fp32_reconstruction_report, sort_keys=True)
        )

    directions = {
        EQUAL_BLEND_CANDIDATE: clone_gradient(
            native_gradients[EQUAL_BLEND_OBJECTIVE]
        ),
        RAW_UNION_DIAGNOSTIC: clone_gradient(native_gradients["union"]),
    }
    if tuple(directions) != SELECTABLE_CANDIDATES + DIAGNOSTIC_DIRECTIONS:
        raise RuntimeError("selectable/diagnostic direction set drift")
    direction_reports = {
        name: candidate_effects(direction, native_gradients)
        for name, direction in directions.items()
    }
    primary_pass = direction_reports[EQUAL_BLEND_CANDIDATE][
        "all_gate_objectives_robust_descent"
    ]
    selected = EQUAL_BLEND_CANDIDATE if primary_pass else None

    model_hash_after = ppo.model_state_sha256(model)
    if model_hash_after != model_hash_before:
        raise RuntimeError("zero-write probe changed model state")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("zero-write probe left parameter .grad buffers")

    result = {
        "status": "completed_zero_weight_write_direct512_multiobjective_probe",
        "decision": "DIRECTION_FOUND" if selected is not None else "NO_DIRECTION",
        "device": "cuda",
        "seed": PROBE_SEED,
        "base": {
            "checkpoint": str(frozen.U468.relative_to(ROOT)),
            "update": 468,
            "model_state_sha256": model_hash_before,
            "actor6_storage_dtype": "torch.float32",
        },
        "direct512": {
            "rows": ROWS,
            "cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
            "batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
            "flat_identity_sha256": EXPECTED_FLAT_IDENTITY_SHA256,
            "flat_identity_fields": list(IDENTITY_FIELDS),
            "single_forward_batch": True,
        },
        "objective_contract": {
            "loss": "complete ordered bc_expert_actor_loss composite",
            "order_context_weight": aggregate.ORDER_CONTEXT_WEIGHT,
            "non_context34_fixed_multi_action_order_weight": 1.0,
            "native_direct_objective_order": list(NATIVE_DIRECT_OBJECTIVES),
            "fp32_direct_objective_order": list(FP32_DIRECT_OBJECTIVES),
            "gate_objectives": list(GATE_OBJECTIVES),
            "each_objective_has_own_direct_scalar_autograd": True,
            "equal_blend_formula": (
                "0.5*L_union + "
                "(L_flg_hard+L_pokemonfan_hard+L_core5_hard)/6"
            ),
            "equal_blend_single_scalar_vjp": True,
            "no_native_bf16_objective_gradient_algebraic_substitution": True,
        },
        "weight_and_partition_audit": dict(weight_audit),
        "native_bf16": {
            "forward": "ppo.model_forward with CUDA BF16 autocast",
            "output_dtypes": native_output_dtypes,
            "objectives": native_reports,
            "pairwise_gate_geometry": pairwise_geometry(
                native_gradients, GATE_OBJECTIVES
            ),
            "six_partition_union_reconstruction_diagnostic_only": {
                **native_reconstruction_report,
                "gate_enforced": False,
                "reason": (
                    "BF16 forward rounding is not required to obey a post-hoc "
                    "partition algebra identity; candidates use direct scalars"
                ),
            },
        },
        "fp32_structural_reconstruction": {
            "forward": "direct model(batch), CUDA autocast disabled",
            "output_dtypes": fp32_output_dtypes,
            "objectives": fp32_reports,
            "formula": (
                "sum(partition_effective_weight * direct_partition_gradient) "
                "/ union_effective_weight"
            ),
            **fp32_reconstruction_report,
            "relative_l2_max": FP32_RECONSTRUCTION_RELATIVE_L2_MAX,
            "max_abs_max": FP32_RECONSTRUCTION_MAX_ABS_MAX,
            "pass": fp32_reconstruction_pass,
        },
        "additional_flat_unflat_layout_audit": flat_unflat_layout_audit(
            fp32_gradients["union"]
        ),
        "candidate_contract": {
            "selectable_candidates": list(SELECTABLE_CANDIDATES),
            "diagnostic_only_directions": list(DIAGNOSTIC_DIRECTIONS),
            "automatic_fallback": False,
            "maximum_authorized_endpoints": 1,
            "equal_blend_scalar": (
                "0.5*L_union + "
                "(L_flg_hard+L_pokemonfan_hard+L_core5_hard)/6"
            ),
            "equal_blend_alpha": EQUAL_BLEND_ALPHA,
            "equal_blend_single_scalar_vjp": True,
            "equal_blend_not_reconstructed_from_extracted_gradients": True,
            "update_convention": "theta_new = theta - lr * raw_direction",
            "gate_objectives": list(GATE_OBJECTIVES),
            "robust_descent": (
                "dot(objective_gradient,direction)>0 and cosine>1e-8"
            ),
            "first_order_cosine_epsilon": FIRST_ORDER_COSINE_EPS,
            "sole_optimizer_suggestion": {
                "name": "SGD",
                "learning_rate": SUGGESTED_SGD_LEARNING_RATE,
                "momentum": SUGGESTED_SGD_MOMENTUM,
                "weight_decay": SUGGESTED_SGD_WEIGHT_DECAY,
            },
        },
        "directions": direction_reports,
        "selection": {
            "equal_blend_passes": primary_pass,
            "selected_direction": selected,
            "rule": (
                "select the sole equal-blend direction iff all five robust "
                "descent gates pass; raw union never authorizes an endpoint"
            ),
        },
        "integrity": {
            "aggregate_runner_sha256": AGGREGATE_RUNNER_SHA256,
            "actor6_scope_exact": trainable == tuple(aggregate.ACTOR_NAMES),
            "model_state_sha256_before": model_hash_before,
            "model_state_sha256_after": model_hash_after,
            "model_weights_unchanged": model_hash_after == model_hash_before,
            "native_forward_calls": 1,
            "fp32_direct_forward_calls": 1,
            "native_direct_composite_autograd_calls": len(NATIVE_DIRECT_OBJECTIVES),
            "fp32_direct_composite_autograd_calls": len(FP32_DIRECT_OBJECTIVES),
            "equal_blend_scalar_vjp_calls": 1,
            "backward_calls": 0,
            "optimizer_instances_created": 0,
            "optimizer_step_calls": 0,
            "parameter_grad_buffers_materialized": 0,
            "validation_member_payloads_opened": False,
            "checkpoint_writes": 0,
            "model_writes": 0,
            "result_artifact_writes": 0,
            "stdout_only": True,
        },
    }
    if not frozen.repair.finite_nested(result):
        raise FloatingPointError("probe result contains a nonfinite value")
    return result


def common_report(
    mode: str,
    self_evidence: Mapping[str, Any],
    ast_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "zero_write_audit_passed",
        "mode": mode,
        "tool": dict(self_evidence),
        "aggregate_runner": dict(AGGREGATE_RUNNER_EVIDENCE),
        "contract": {
            "base": "raw full U468",
            "direct512_cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
            "direct512_batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
            "flat_identity_sha256": EXPECTED_FLAT_IDENTITY_SHA256,
            "rows": ROWS,
            "actor_parameter_names": list(aggregate.ACTOR_NAMES),
            "native_forward": "CUDA BF16 autocast",
            "fp32_reconstruction_forward": "direct model(batch), autocast disabled",
            "fp32_reconstruction_relative_l2_max": (
                FP32_RECONSTRUCTION_RELATIVE_L2_MAX
            ),
            "fp32_reconstruction_max_abs_max": FP32_RECONSTRUCTION_MAX_ABS_MAX,
            "selectable_candidates": list(SELECTABLE_CANDIDATES),
            "diagnostic_only_directions": list(DIAGNOSTIC_DIRECTIONS),
            "gate_objectives": list(GATE_OBJECTIVES),
            "equal_blend_alpha": EQUAL_BLEND_ALPHA,
            "sole_optimizer_suggestion": {
                "name": "SGD",
                "learning_rate": SUGGESTED_SGD_LEARNING_RATE,
                "momentum": SUGGESTED_SGD_MOMENTUM,
                "weight_decay": SUGGESTED_SGD_WEIGHT_DECAY,
            },
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
        },
        "ast_audit": dict(ast_result),
        "writes_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("static-audit", "cache-audit", "probe"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    assert_frozen_aggregate_contract()
    self_path = Path(__file__).resolve()
    if self_path.parent != TOOLS:
        raise RuntimeError("probe must remain in repository tools directory")
    self_evidence = require_single_link_regular(self_path, None, "probe tool")
    ast_result = ast_audit()
    common = common_report(args.mode, self_evidence, ast_result)

    if args.mode == "static-audit":
        if args.device != "cpu":
            raise ValueError("static-audit uses the default CPU device argument")
        print(json.dumps(common, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return

    cache_audit, selections, union = aggregate.build_cache_audit()
    identities = flat_identities(selections)
    if (
        cache_audit["aggregate512"]["cache_sha256"]
        != EXPECTED_DIRECT_CACHE_SHA256
        or cache_audit["aggregate512"]["batch_sha256"]
        != EXPECTED_DIRECT_BATCH_SHA256
    ):
        raise RuntimeError("aggregate direct512 cache identity drift")
    masks, weight_audit, _ = build_masks_and_weights(union, identities)
    cache_report = common | {
        "status": "zero_write_direct512_cache_audit_passed",
        "aggregate_cache_audit": cache_audit,
        "flat_identity": {
            "rows": len(identities),
            "fields": list(IDENTITY_FIELDS),
            "sha256": EXPECTED_FLAT_IDENTITY_SHA256,
            "unique": True,
        },
        "weight_and_partition_audit": weight_audit,
    }
    if args.mode == "cache-audit":
        if args.device != "cpu":
            raise ValueError("cache-audit uses the default CPU device argument")
        print(
            json.dumps(
                cache_report, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        )
        return

    if args.device != "cuda":
        raise ValueError("formal probe requires --device cuda")
    probe = run_probe(union, identities, masks, weight_audit, torch.device("cuda"))
    final = cache_report | {
        "status": probe["status"],
        "mode": "probe",
        "probe": probe,
    }
    print(json.dumps(final, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
