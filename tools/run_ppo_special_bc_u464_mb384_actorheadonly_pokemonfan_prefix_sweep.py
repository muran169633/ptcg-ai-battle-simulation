#!/usr/bin/env python3
"""Run the frozen U464 actor-head-only PokemonFan P4/P8/P12/P16 sweep.

This standalone executor binds both the superseded v1 design and authoritative
v2 correction, plus the audited U464 parent, general-BC architecture
checkpoint, PokemonFan archive, replay cache, seed, batch order, AdamW state,
learning rate, and loss.  The complete parent 24-parameter replay AdamW is
loaded directly.  Every model tensor is frozen before enabling exactly the ten
actor/count-head tensors, so the fourteen shared optimizer parameters retain
``grad is None`` and their complete AdamW states remain byte-identical.

Audit-only mode constructs and validates every frozen input and optimizer
mapping, but executes zero optimizer steps and writes no checkpoint.  Formal
mode is a single continuous 16-step trajectory that publishes exactly four
checkpoints: P4, P8, P12, and P16.  Neither mode authorizes evaluation,
packaging, upload, or submission.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import run_ppo_bc_repair as repair_audit
import train_ppo as ppo


EXPECTED_PARENT_CONFIG: dict[str, Any] = {
    "updates": 464,
    "minibatch_size": 384,
    "actor_reduction": "episode_mean",
    "trainable_scope": "last_block_heads",
    "learning_rate": 3.6e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 0.5,
    "bc_replay_split": "train",
    "bc_replay_batches": 72,
    "bc_replay_batch_size": 256,
    "bc_replay_workers": 8,
    "bc_replay_steps": 2,
    "bc_replay_lr_scale": 0.05,
    "bc_replay_loss": "ordered",
    "bc_replay_order_context_weight": 8.0,
    "bc_replay_context34_rows_per_batch": 4,
    "bc_replay_non_context34_fixed_multi_action_order_weight": 1.0,
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_ID = 202608060
EXPECTED_PYTHON = Path(
    "/home/xxc/miniconda3/envs/my_project_env/bin/python"
)

EXPECTED_DESIGN_ROOT = REPO_ROOT / (
    "artifacts/ppo_u464mb384_actorheadonly_pokemonfan_sweep_"
    "p4_p8_p12_p16_design202608060"
)
EXPECTED_FORMAL_OUTPUT_DIR = EXPECTED_DESIGN_ROOT / "sweep_stage"
EXPECTED_CPU_AUDIT_OUTPUT_DIR = Path(str(EXPECTED_DESIGN_ROOT) + ".preflight_cpu")
EXPECTED_CUDA_AUDIT_OUTPUT_DIR = Path(str(EXPECTED_DESIGN_ROOT) + ".preflight_cuda")
EXPECTED_DESIGN_PREREGISTRATION_V1 = REPO_ROOT / (
    "artifacts/ppo_u464mb384_actorheadonly_pokemonfan_sweep_"
    "p4_p8_p12_p16_design202608060.design_preregistration.json"
)
EXPECTED_DESIGN_PREREGISTRATION_V1_SHA256 = (
    "e2ffbd3a7bc6ae8d37c70d422d2f8116e15a7982792dfb289e2849266677e115"
)
EXPECTED_DESIGN_PREREGISTRATION_V2 = REPO_ROOT / (
    "artifacts/ppo_u464mb384_actorheadonly_pokemonfan_sweep_"
    "p4_p8_p12_p16_design202608060.design_preregistration_v2.json"
)
EXPECTED_DESIGN_PREREGISTRATION_V2_SHA256 = (
    "29edbe578122b81f05a02b3428d0e24c804f86add2734e3972c9d15f75ea0ca4"
)

EXPECTED_PARENT_CHECKPOINT = REPO_ROOT / (
    "artifacts/ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_localtransport_u456_to_u464_"
    "seed202607336/B_gold_league/seed-202607336/checkpoints/update-0464.pt"
)
EXPECTED_PARENT_SHA256 = (
    "fe51f40f37fca329cd6b0c94f7431001bb909b92624df33fcd7cf6f0da976264"
)
EXPECTED_PARENT_UPDATE = 464
EXPECTED_PARENT_MODEL_SHA256 = (
    "fa6e42da654f872f6f24836838654286d7cd64e10cacff554bdec36b6aa58f0a"
)
EXPECTED_PARENT_REPLAY_SHA256 = (
    "23f775207b0d216c1ffd19bf9213456026b71b786727a4d2ff0f889b6b30341d"
)
EXPECTED_PARENT_PPO_SHA256 = (
    "904a3d2c8ac63c8a60af2404c216830b8ae30a69fbe972818e05c0136058a54f"
)
EXPECTED_PARENT_REPLAY_STATE_COUNT = 24
EXPECTED_PARENT_REPLAY_STATE_STEP = 16
EXPECTED_PARENT_PPO_STATE_COUNT = 28
EXPECTED_PARENT_PPO_STATE_STEP = 276

EXPECTED_GENERAL_BC_CHECKPOINT = REPO_ROOT / (
    "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_"
    "seed20260922_20260731/best.pt"
)
EXPECTED_GENERAL_BC_SHA256 = (
    "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
)
EXPECTED_SPECIAL_DATA = REPO_ROOT / (
    "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_"
    "special_20260801.zip"
)
EXPECTED_SPECIAL_DATA_SHA256 = (
    "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598"
)
EXPECTED_TRAIN_PPO_SHA256 = (
    "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
)
TRAIN_BC_ORBIT_PATH = REPO_ROOT / "tools/train_bc_orbit.py"
EXPECTED_TRAIN_BC_ORBIT_SHA256 = (
    "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
)
CG_SIM_PATH = (
    REPO_ROOT / "dataset/sample_submission/sample_submission/cg/sim.py"
)
EXPECTED_CG_SIM_SHA256 = (
    "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655"
)
CG_LIB_PATH = (
    REPO_ROOT / "dataset/sample_submission/sample_submission/cg/libcg.so"
)
EXPECTED_CG_LIB_SHA256 = (
    "feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887"
)
REPAIR_AUDIT_PATH = REPO_ROOT / "tools/run_ppo_bc_repair.py"
EXPECTED_REPAIR_AUDIT_SHA256 = (
    "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"
)

SPECIAL_CACHE_BATCHES = 32
SPECIAL_BATCH_SIZE = 256
SPECIAL_WORKERS = 8
SPECIAL_CONTEXT34_ROWS_PER_BATCH = 1
SPECIAL_SEED = 202608012
SPECIAL_STEPS = 16
PUBLISHED_ENDPOINT_STEPS = (4, 8, 12, 16)
SPECIAL_BATCH_INDICES = (
    0,
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    15,
    16,
    18,
    19,
    22,
    26,
    27,
    31,
)
EXPECTED_CACHE_SHA256 = (
    "a1c16b8b2d6fbf45cf1dfd38ba4eb5527ee444d4d90e09b4f57ca185bdce3021"
)
# Preserve the exact binary float produced by the frozen base LR and scale.
EXPECTED_SPECIAL_LEARNING_RATE = 3.6e-5 * 0.05

EXPECTED_FROZEN_SHARED_PARAMETER_NAMES = (
    "transformer.layers.3.self_attn.in_proj_weight",
    "transformer.layers.3.self_attn.in_proj_bias",
    "transformer.layers.3.self_attn.out_proj.weight",
    "transformer.layers.3.self_attn.out_proj.bias",
    "transformer.layers.3.linear1.weight",
    "transformer.layers.3.linear1.bias",
    "transformer.layers.3.linear2.weight",
    "transformer.layers.3.linear2.bias",
    "transformer.layers.3.norm1.weight",
    "transformer.layers.3.norm1.bias",
    "transformer.layers.3.norm2.weight",
    "transformer.layers.3.norm2.bias",
    "transformer.norm.weight",
    "transformer.norm.bias",
)
EXPECTED_MUTABLE_PARAMETER_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
    "count_head.0.weight",
    "count_head.0.bias",
    "count_head.2.weight",
    "count_head.2.bias",
)
EXPECTED_ACTOR_PARAMETER_NAMES = (
    EXPECTED_FROZEN_SHARED_PARAMETER_NAMES + EXPECTED_MUTABLE_PARAMETER_NAMES
)
EXPECTED_VALUE_PARAMETER_NAMES = (
    "value_head.0.weight",
    "value_head.0.bias",
    "value_head.2.weight",
    "value_head.2.bias",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def clone_nested_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is absent or symlinked: {path}")
    actual = repair_audit.file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def all_files_unchanged(files: dict[str, str]) -> bool:
    return all(
        Path(path).is_file()
        and not Path(path).is_symlink()
        and repair_audit.file_sha256(Path(path)) == expected
        for path, expected in files.items()
    )


def state_step(entry: dict[str, Any]) -> int:
    raw_step = entry.get("step")
    if isinstance(raw_step, torch.Tensor):
        if raw_step.numel() != 1:
            raise ValueError("AdamW step tensor is not scalar")
        return int(raw_step.item())
    if raw_step is None:
        raise ValueError("AdamW state entry has no step")
    return int(raw_step)


def validate_parent_config(raw_config: Any) -> ppo.PPOConfig:
    if not isinstance(raw_config, dict):
        raise ValueError("Parent checkpoint has no PPO config")
    for name, expected in EXPECTED_PARENT_CONFIG.items():
        actual = raw_config.get(name)
        if actual != expected:
            raise ValueError(
                f"Parent config {name!r} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )
    return ppo.PPOConfig(**raw_config)


def validate_request() -> None:
    if PUBLISHED_ENDPOINT_STEPS != (4, 8, 12, 16):
        raise RuntimeError("Published endpoints drifted from P4/P8/P12/P16")
    if SPECIAL_STEPS != PUBLISHED_ENDPOINT_STEPS[-1]:
        raise RuntimeError("The continuous trajectory must stop exactly at P16")
    if len(SPECIAL_BATCH_INDICES) != SPECIAL_STEPS:
        raise RuntimeError("Frozen batch order length must equal 16 steps")
    if len(set(SPECIAL_BATCH_INDICES)) != SPECIAL_STEPS:
        raise RuntimeError("Frozen batch order must be without replacement")
    if min(SPECIAL_BATCH_INDICES) < 0 or max(SPECIAL_BATCH_INDICES) >= SPECIAL_CACHE_BATCHES:
        raise RuntimeError("Frozen batch order addresses an invalid cache batch")
    if len(EXPECTED_MUTABLE_PARAMETER_NAMES) != 10:
        raise RuntimeError("Mutable actor/count-head scope must contain 10 tensors")
    if len(EXPECTED_FROZEN_SHARED_PARAMETER_NAMES) != 14:
        raise RuntimeError("Frozen transformer/norm scope must contain 14 tensors")
    if len(EXPECTED_VALUE_PARAMETER_NAMES) != 4:
        raise RuntimeError("Frozen value scope must contain 4 tensors")
    if len(EXPECTED_ACTOR_PARAMETER_NAMES) != 24:
        raise RuntimeError("Full replay actor manifest must contain 24 tensors")


def full_state_mapping(
    replay_state: dict[str, Any],
    actor_names: list[str],
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    state = replay_state.get("state")
    groups = replay_state.get("param_groups")
    if not isinstance(state, dict) or not isinstance(groups, list):
        raise ValueError("Parent replay optimizer state is malformed")
    if len(groups) != 1 or not isinstance(groups[0], dict):
        raise ValueError("Parent replay optimizer must have exactly one group")
    group = groups[0]
    parameter_ids = list(group.get("params", []))
    if len(parameter_ids) != len(actor_names) or len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("Parent replay optimizer parameter IDs are malformed")
    if set(state) != set(parameter_ids):
        raise ValueError("Parent replay optimizer state does not cover its group exactly")
    if len(state) != EXPECTED_PARENT_REPLAY_STATE_COUNT:
        raise ValueError("Parent replay optimizer state count is not 24")
    id_by_name = dict(zip(actor_names, parameter_ids, strict=True))
    return state, parameter_ids, id_by_name


class FullAdamWWithFrozenGradientGuard(torch.optim.AdamW):
    """Full parent AdamW that refuses a step if any frozen grad exists."""

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        *,
        learning_rate: float,
        eps: float,
        weight_decay: float,
        mutable_named: list[tuple[str, torch.nn.Parameter]],
        frozen_shared_named: list[tuple[str, torch.nn.Parameter]],
        frozen_value_named: list[tuple[str, torch.nn.Parameter]],
    ) -> None:
        super().__init__(
            parameters,
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.mutable_named = mutable_named
        self.frozen_shared_named = frozen_shared_named
        self.frozen_value_named = frozen_value_named
        self.step_audits: list[dict[str, Any]] = []

    def step(self, closure: Any = None) -> Any:
        frozen_shared_with_grad = [
            name
            for name, parameter in self.frozen_shared_named
            if parameter.grad is not None
        ]
        frozen_value_with_grad = [
            name
            for name, parameter in self.frozen_value_named
            if parameter.grad is not None
        ]
        mutable_without_grad = [
            name
            for name, parameter in self.mutable_named
            if parameter.grad is None
        ]
        mutable_nonfinite_grad = [
            name
            for name, parameter in self.mutable_named
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        audit = {
            "step_ordinal": len(self.step_audits) + 1,
            "frozen_shared_with_grad": frozen_shared_with_grad,
            "frozen_value_with_grad": frozen_value_with_grad,
            "mutable_without_grad": mutable_without_grad,
            "mutable_nonfinite_grad": mutable_nonfinite_grad,
            "frozen_shared_14_grad_none_before_step": (
                not frozen_shared_with_grad
            ),
            "frozen_value_4_grad_none_before_step": (
                not frozen_value_with_grad
            ),
            "mutable_10_grad_present_before_step": not mutable_without_grad,
            "mutable_10_grad_finite_before_step": not mutable_nonfinite_grad,
        }
        if any(
            (
                frozen_shared_with_grad,
                frozen_value_with_grad,
                mutable_without_grad,
                mutable_nonfinite_grad,
            )
        ):
            raise RuntimeError(
                "Full replay AdamW gradient guard rejected a step: "
                + json.dumps(audit, sort_keys=True)
            )
        result = super().step(closure)
        self.step_audits.append(audit)
        return result


def build_full_optimizer_with_head_only_grad(
    model: torch.nn.Module,
    config: ppo.PPOConfig,
    parent_replay_state: dict[str, Any],
    actor_names: list[str],
    full_actor_parameters: list[torch.nn.Parameter],
) -> tuple[FullAdamWWithFrozenGradientGuard, dict[str, Any]]:
    _, parent_ids, full_id_by_name = full_state_mapping(
        parent_replay_state,
        actor_names,
    )

    named_parameters = dict(model.named_parameters())
    missing = [
        name
        for name in EXPECTED_ACTOR_PARAMETER_NAMES + EXPECTED_VALUE_PARAMETER_NAMES
        if name not in named_parameters
    ]
    if missing:
        raise ValueError("Required model parameters are absent: " + json.dumps(missing))

    model.requires_grad_(False)
    mutable_named: list[tuple[str, torch.nn.Parameter]] = []
    for name in EXPECTED_MUTABLE_PARAMETER_NAMES:
        parameter = named_parameters[name]
        parameter.requires_grad_(True)
        mutable_named.append((name, parameter))
    frozen_shared_named = [
        (name, named_parameters[name])
        for name in EXPECTED_FROZEN_SHARED_PARAMETER_NAMES
    ]
    frozen_value_named = [
        (name, named_parameters[name]) for name in EXPECTED_VALUE_PARAMETER_NAMES
    ]
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if tuple(trainable_names) != EXPECTED_MUTABLE_PARAMETER_NAMES:
        raise RuntimeError("Model trainable scope is not exactly the frozen 10-head set")
    if any(parameter.requires_grad for _, parameter in frozen_shared_named):
        raise RuntimeError("A frozen transformer/norm parameter remains trainable")
    if any(parameter.requires_grad for _, parameter in frozen_value_named):
        raise RuntimeError("A frozen value parameter remains trainable")

    optimizer = FullAdamWWithFrozenGradientGuard(
        full_actor_parameters,
        learning_rate=config.learning_rate * config.bc_replay_lr_scale,
        eps=1e-5,
        weight_decay=config.weight_decay,
        mutable_named=mutable_named,
        frozen_shared_named=frozen_shared_named,
        frozen_value_named=frozen_value_named,
    )
    optimizer.load_state_dict(clone_nested_to_cpu(parent_replay_state))
    loaded = clone_nested_to_cpu(optimizer.state_dict())
    loaded_ids = list(loaded["param_groups"][0]["params"])
    if loaded_ids != parent_ids:
        raise RuntimeError("Direct-loaded full AdamW parameter order drifted")
    loaded_hash = repair_audit.nested_sha256(loaded)
    if loaded_hash != EXPECTED_PARENT_REPLAY_SHA256:
        raise RuntimeError("Direct-loaded full AdamW state does not equal parent")
    if len(loaded.get("state", {})) != EXPECTED_PARENT_REPLAY_STATE_COUNT:
        raise RuntimeError("Direct-loaded full AdamW state count is not 24")

    return optimizer, {
        "mutable_parameter_names": list(EXPECTED_MUTABLE_PARAMETER_NAMES),
        "mutable_parameter_count": len(EXPECTED_MUTABLE_PARAMETER_NAMES),
        "mutable_parameter_numel": sum(
            parameter.numel() for _, parameter in mutable_named
        ),
        "optimizer_parameter_set": "complete_parent_actor_manifest_24",
        "optimizer_parameter_ids": loaded_ids,
        "parent_full_parameter_ids_by_name": full_id_by_name,
        "parent_state_loaded_directly": True,
        "subset_projection_or_state_merge_used": False,
        "loaded_state_sha256": loaded_hash,
        "loaded_state_exact_parent": True,
        "step_before_exact": EXPECTED_PARENT_REPLAY_STATE_STEP,
        "trainable_parameter_names_after_freeze": trainable_names,
        "frozen_shared_requires_grad_false": True,
        "frozen_value_requires_grad_false": True,
    }


def validate_model_endpoint(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, Any]:
    changed_names = repair_audit.changed_tensor_names(before, after)
    changed_exact = (
        len(changed_names) == len(EXPECTED_MUTABLE_PARAMETER_NAMES)
        and set(changed_names) == set(EXPECTED_MUTABLE_PARAMETER_NAMES)
    )
    frozen_shared_equal = all(
        torch.equal(before[name], after[name])
        for name in EXPECTED_FROZEN_SHARED_PARAMETER_NAMES
    )
    frozen_value_equal = all(
        torch.equal(before[name], after[name])
        for name in EXPECTED_VALUE_PARAMETER_NAMES
    )
    mutable = set(EXPECTED_MUTABLE_PARAMETER_NAMES)
    all_outside_mutable_equal = all(
        torch.equal(before[name], after[name])
        for name in before
        if name not in mutable
    )
    finite = repair_audit.finite_nested(after)
    if not all(
        (
            changed_exact,
            frozen_shared_equal,
            frozen_value_equal,
            all_outside_mutable_equal,
            finite,
        )
    ):
        raise RuntimeError(
            "Actor-head-only model integrity failed: "
            + json.dumps(
                {
                    "changed_names": changed_names,
                    "changed_exact": changed_exact,
                    "frozen_shared_equal": frozen_shared_equal,
                    "frozen_value_equal": frozen_value_equal,
                    "all_outside_mutable_equal": all_outside_mutable_equal,
                    "finite": finite,
                },
                sort_keys=True,
            )
        )
    return {
        "changed_parameter_names": changed_names,
        "changed_parameter_count": len(changed_names),
        "changed_exactly_mutable_10": True,
        "frozen_shared_14_byte_equal_parent": True,
        "frozen_value_4_byte_equal_parent": True,
        "all_model_tensors_outside_mutable_byte_equal_parent": True,
        "all_model_tensors_finite": True,
    }


def validate_full_replay_state(
    parent_replay_state: dict[str, Any],
    full_optimizer: torch.optim.Optimizer,
    actor_names: list[str],
    full_actor_parameters: list[torch.nn.Parameter],
    config: ppo.PPOConfig,
    special_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_state, parent_ids, full_id_by_name = full_state_mapping(
        parent_replay_state,
        actor_names,
    )
    current = clone_nested_to_cpu(full_optimizer.state_dict())

    if (
        repair_audit.nested_sha256(current["param_groups"])
        != repair_audit.nested_sha256(parent_replay_state["param_groups"])
    ):
        raise RuntimeError("Full replay optimizer param groups changed")
    if list(current["param_groups"][0]["params"]) != parent_ids:
        raise RuntimeError("Full replay optimizer parameter order changed")
    if len(current.get("state", {})) != EXPECTED_PARENT_REPLAY_STATE_COUNT:
        raise RuntimeError("Full replay optimizer does not retain all 24 states")
    if not repair_audit.finite_nested(current):
        raise FloatingPointError("Full replay optimizer contains non-finite state")

    frozen_state_exact_by_name: dict[str, bool] = {}
    frozen_steps_by_name: dict[str, int] = {}
    for name in EXPECTED_FROZEN_SHARED_PARAMETER_NAMES:
        parameter_id = full_id_by_name[name]
        exact = (
            repair_audit.nested_sha256(current["state"][parameter_id])
            == repair_audit.nested_sha256(parent_state[parameter_id])
        )
        frozen_state_exact_by_name[name] = exact
        frozen_steps_by_name[name] = state_step(current["state"][parameter_id])
        if not exact or frozen_steps_by_name[name] != EXPECTED_PARENT_REPLAY_STATE_STEP:
            raise RuntimeError(f"Frozen replay AdamW state drifted for {name}")

    expected_mutable_step = EXPECTED_PARENT_REPLAY_STATE_STEP + special_steps
    mutable_steps_by_name: dict[str, int] = {}
    for name in EXPECTED_MUTABLE_PARAMETER_NAMES:
        parameter_id = full_id_by_name[name]
        step = state_step(current["state"][parameter_id])
        mutable_steps_by_name[name] = step
        if step != expected_mutable_step:
            raise RuntimeError(
                f"Mutable replay AdamW step drifted for {name}: "
                f"expected {expected_mutable_step}, got {step}"
            )

    # Prove train_ppo can later dry-load this complete mixed-step state.
    compatibility_optimizer = torch.optim.AdamW(
        full_actor_parameters,
        lr=config.learning_rate * config.bc_replay_lr_scale,
        eps=1e-5,
        weight_decay=config.weight_decay,
    )
    compatibility_optimizer.load_state_dict(clone_nested_to_cpu(current))
    roundtrip_state = clone_nested_to_cpu(compatibility_optimizer.state_dict())
    roundtrip_exact = (
        repair_audit.nested_sha256(roundtrip_state)
        == repair_audit.nested_sha256(current)
    )
    del compatibility_optimizer
    if not roundtrip_exact:
        raise RuntimeError("Full 24-state AdamW did not dry-load exactly")

    return current, {
        "full_state_count": len(current["state"]),
        "full_parameter_order_retained": True,
        "full_param_groups_byte_equal_parent": True,
        "parent_state_loaded_directly": True,
        "subset_projection_or_state_merge_used": False,
        "frozen_state_exact_by_name": frozen_state_exact_by_name,
        "frozen_state_all_byte_equal_parent": all(
            frozen_state_exact_by_name.values()
        ),
        "frozen_steps_by_name": frozen_steps_by_name,
        "frozen_step_exact": EXPECTED_PARENT_REPLAY_STATE_STEP,
        "mutable_steps_by_name": mutable_steps_by_name,
        "mutable_step_exact": expected_mutable_step,
        "mutable_state_count": len(EXPECTED_MUTABLE_PARAMETER_NAMES),
        "full_24_state_roundtrip_load_exact": roundtrip_exact,
        "state_sha256": repair_audit.nested_sha256(current),
        "all_state_finite": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen U464 actor-head-only PokemonFan continuous "
            "P4/P8/P12/P16 prefix sweep."
        )
    )
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_request()

    executable = Path(sys.executable).resolve()
    if executable != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            "This executor must use my_project_env Python: "
            f"expected {EXPECTED_PYTHON.resolve()}, got {executable}"
        )

    tool_path = Path(__file__).resolve()
    train_path = tool_path.with_name("train_ppo.py")
    parent_path = EXPECTED_PARENT_CHECKPOINT.resolve()
    special_data_path = EXPECTED_SPECIAL_DATA.resolve()
    raw_output_dir = args.output_dir
    output_dir = raw_output_dir.resolve()
    formal_output_dir = EXPECTED_FORMAL_OUTPUT_DIR.resolve()

    if raw_output_dir.is_symlink():
        raise ValueError("Output directory may not be a symlink")
    if args.audit_only:
        expected_audit_dir = (
            EXPECTED_CPU_AUDIT_OUTPUT_DIR
            if args.device == "cpu"
            else EXPECTED_CUDA_AUDIT_OUTPUT_DIR
        ).resolve()
        if output_dir != expected_audit_dir:
            raise ValueError(
                f"{args.device.upper()} audit output must use its frozen sibling path: "
                f"{expected_audit_dir}"
            )
        if EXPECTED_DESIGN_ROOT.exists():
            raise FileExistsError(
                "Audit-only mode requires the entire formal output root to remain absent"
            )
    elif output_dir != formal_output_dir:
        raise ValueError("Formal output must be the frozen sweep_stage directory")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")

    bound_files: dict[str, str] = {}
    for path, expected, label in (
        (tool_path, args.expected_tool_sha256, "executor"),
        (train_path, EXPECTED_TRAIN_PPO_SHA256, "train_ppo dependency"),
        (
            TRAIN_BC_ORBIT_PATH,
            EXPECTED_TRAIN_BC_ORBIT_SHA256,
            "train_bc_orbit dependency",
        ),
        (CG_SIM_PATH, EXPECTED_CG_SIM_SHA256, "cg.sim dependency"),
        (CG_LIB_PATH, EXPECTED_CG_LIB_SHA256, "cg shared-library dependency"),
        (REPAIR_AUDIT_PATH, EXPECTED_REPAIR_AUDIT_SHA256, "repair-audit dependency"),
        (
            EXPECTED_DESIGN_PREREGISTRATION_V1,
            EXPECTED_DESIGN_PREREGISTRATION_V1_SHA256,
            "superseded v1 design preregistration",
        ),
        (
            EXPECTED_DESIGN_PREREGISTRATION_V2,
            EXPECTED_DESIGN_PREREGISTRATION_V2_SHA256,
            "authoritative v2 design preregistration",
        ),
        (parent_path, EXPECTED_PARENT_SHA256, "U464 parent checkpoint"),
        (special_data_path, EXPECTED_SPECIAL_DATA_SHA256, "PokemonFan archive"),
    ):
        resolved = path.resolve()
        bound_files[str(resolved)] = validate_sha(resolved, expected, label)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_probe = torch.ones(1, device=device) + 1.0
    if float(device_probe.item()) != 2.0:
        raise RuntimeError("Device tensor-kernel preflight failed")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    device_preflight = {
        "device": str(device),
        "tensor_kernel_result": float(device_probe.item()),
        "cuda_synchronized": device.type == "cuda",
        "pass": True,
    }
    del device_probe

    torch.use_deterministic_algorithms(True)
    random.seed(SPECIAL_SEED)
    torch.manual_seed(SPECIAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SPECIAL_SEED)

    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise ValueError("Parent checkpoint is not a compatible PPO checkpoint")
    if int(parent.get("update", -1)) != EXPECTED_PARENT_UPDATE:
        raise ValueError("Parent checkpoint update is not the frozen U464 update")
    config = validate_parent_config(parent.get("config"))

    bc_checkpoint_path = Path(config.bc_checkpoint).resolve()
    if bc_checkpoint_path != EXPECTED_GENERAL_BC_CHECKPOINT.resolve():
        raise ValueError("Parent general-BC path is not the frozen checkpoint")
    bc_checkpoint_hash = validate_sha(
        bc_checkpoint_path,
        EXPECTED_GENERAL_BC_SHA256,
        "general-BC architecture checkpoint",
    )
    bound_files[str(bc_checkpoint_path)] = bc_checkpoint_hash
    bc_checkpoint = torch.load(
        bc_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = ppo.instantiate_model_from_checkpoint(
        parent,
        bc_checkpoint,
        device,
    )
    (
        full_actor_parameters,
        _,
        trainable_manifest,
    ) = ppo.configure_trainable_scope(model, config.trainable_scope)
    actor_names = list(trainable_manifest["actor_parameter_names"])
    value_names = list(trainable_manifest["value_parameter_names"])
    if tuple(actor_names) != EXPECTED_ACTOR_PARAMETER_NAMES:
        raise ValueError("Actor manifest is not the frozen ordered 24-tensor set")
    if tuple(value_names) != EXPECTED_VALUE_PARAMETER_NAMES:
        raise ValueError("Value manifest is not the frozen ordered 4-tensor set")

    parent_parameter_names = parent.get("optimizer_parameter_names")
    if not isinstance(parent_parameter_names, dict):
        raise ValueError("Parent checkpoint has no optimizer parameter manifest")
    if actor_names != parent_parameter_names.get("actor"):
        raise ValueError("Actor manifest does not match the parent optimizer")
    if value_names != parent_parameter_names.get("value"):
        raise ValueError("Value manifest does not match the parent optimizer")

    raw_parent_replay_state = parent.get("bc_replay_optimizer_state_dict")
    if not isinstance(raw_parent_replay_state, dict):
        raise ValueError("Parent checkpoint has no replay optimizer state")
    parent_replay_state = clone_nested_to_cpu(raw_parent_replay_state)
    parent_replay_hash = repair_audit.nested_sha256(parent_replay_state)
    if parent_replay_hash != EXPECTED_PARENT_REPLAY_SHA256:
        raise ValueError("Parent replay optimizer SHA256 is not frozen")
    parent_replay_steps = repair_audit.optimizer_steps(parent_replay_state)
    if (
        len(parent_replay_steps) != EXPECTED_PARENT_REPLAY_STATE_COUNT
        or set(parent_replay_steps) != {EXPECTED_PARENT_REPLAY_STATE_STEP}
    ):
        raise ValueError("Parent replay optimizer step/count is not frozen")

    ppo_optimizer_state_before = clone_nested_to_cpu(
        parent.get("optimizer_state_dict")
    )
    ppo_optimizer_hash_before = repair_audit.nested_sha256(
        ppo_optimizer_state_before
    )
    if ppo_optimizer_hash_before != EXPECTED_PARENT_PPO_SHA256:
        raise ValueError("Parent PPO optimizer SHA256 is not frozen")
    ppo_optimizer_steps = repair_audit.optimizer_steps(
        ppo_optimizer_state_before
    )
    if (
        len(ppo_optimizer_steps) != EXPECTED_PARENT_PPO_STATE_COUNT
        or set(ppo_optimizer_steps) != {EXPECTED_PARENT_PPO_STATE_STEP}
    ):
        raise ValueError("Parent PPO optimizer step/count is not frozen")

    model_state_before = repair_audit.clone_model_state(model)
    model_hash_before = ppo.model_state_sha256(model)
    if model_hash_before != EXPECTED_PARENT_MODEL_SHA256:
        raise ValueError("Parent model-state SHA256 is not frozen")
    if not all(
        (
            repair_audit.finite_nested(model_state_before),
            repair_audit.finite_nested(parent_replay_state),
            repair_audit.finite_nested(ppo_optimizer_state_before),
        )
    ):
        raise FloatingPointError("Frozen parent contains non-finite state")

    replay_optimizer, optimizer_scope_audit = (
        build_full_optimizer_with_head_only_grad(
            model,
            config,
            parent_replay_state,
            actor_names,
            full_actor_parameters,
        )
    )

    special_config = copy.deepcopy(config)
    special_config.bc_replay_data = str(special_data_path)
    special_config.bc_replay_split = "train"
    special_config.bc_replay_batches = SPECIAL_CACHE_BATCHES
    special_config.bc_replay_batch_size = SPECIAL_BATCH_SIZE
    special_config.bc_replay_workers = SPECIAL_WORKERS
    special_config.bc_replay_steps = 1
    special_config.bc_replay_lr_scale = 0.05
    special_config.bc_replay_loss = "ordered"
    special_config.bc_replay_order_context_weight = 8.0
    special_config.bc_replay_context34_rows_per_batch = (
        SPECIAL_CONTEXT34_ROWS_PER_BATCH
    )
    special_config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    special_config.seed = SPECIAL_SEED
    special_learning_rate = (
        config.learning_rate * special_config.bc_replay_lr_scale
    )
    if special_learning_rate != EXPECTED_SPECIAL_LEARNING_RATE:
        raise ValueError("Special-BC learning rate is not the frozen 1.8e-6")

    replay_batches = ppo.build_bc_replay_batches(
        special_config,
        parent["model_config"],
    )
    if len(replay_batches) != SPECIAL_CACHE_BATCHES:
        raise ValueError("Special replay cache batch count is not frozen")
    if any(
        int(batch["contexts"].shape[0]) != SPECIAL_BATCH_SIZE
        for batch in replay_batches
    ):
        raise ValueError("Special replay cache contains a short batch")
    context34_per_batch = [
        int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum())
        for batch in replay_batches
    ]
    if set(context34_per_batch) != {SPECIAL_CONTEXT34_ROWS_PER_BATCH}:
        raise ValueError("Special replay cache context-34 quota drifted")
    cache_hash, batch_hashes = repair_audit.replay_cache_manifest(
        replay_batches
    )
    if cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError(
            "Special replay cache SHA256 mismatch: "
            f"expected {EXPECTED_CACHE_SHA256}, got {cache_hash}"
        )

    common: dict[str, Any] = {
        "schema_version": (
            "ptcg-u464mb384-actor-head-only-pokemonfan-prefix-sweep-v2"
        ),
        "mode": "audit_only" if args.audit_only else "special_bc",
        "frozen_protocol": {
            "design_id": DESIGN_ID,
            "superseded_design_v1_sha256": (
                EXPECTED_DESIGN_PREREGISTRATION_V1_SHA256
            ),
            "authoritative_design_v2_sha256": (
                EXPECTED_DESIGN_PREREGISTRATION_V2_SHA256
            ),
            "authoritative_design_version": 2,
            "training_rng_seed": SPECIAL_SEED,
            "special_optimizer_steps_exact": SPECIAL_STEPS,
            "published_endpoint_steps": list(PUBLISHED_ENDPOINT_STEPS),
            "single_continuous_in_memory_trajectory": True,
            "no_resume": True,
            "no_step_beyond_16": True,
            "formal_training_started_by_this_mode": not args.audit_only,
            "evaluation_authorized": False,
            "broad_or_gold19_authorized": False,
            "package_upload_or_submission_authorized": False,
        },
        "parent": {
            "path": str(parent_path),
            "sha256": EXPECTED_PARENT_SHA256,
            "update": EXPECTED_PARENT_UPDATE,
            "model_state_sha256": model_hash_before,
            "replay_state_sha256": parent_replay_hash,
            "ppo_state_sha256": ppo_optimizer_hash_before,
        },
        "sources": {
            "tool": {
                "path": str(tool_path),
                "sha256": bound_files[str(tool_path)],
            },
            "train_ppo": {
                "path": str(train_path),
                "sha256": EXPECTED_TRAIN_PPO_SHA256,
            },
            "train_bc_orbit": {
                "path": str(TRAIN_BC_ORBIT_PATH.resolve()),
                "sha256": EXPECTED_TRAIN_BC_ORBIT_SHA256,
            },
            "cg_sim": {
                "path": str(CG_SIM_PATH.resolve()),
                "sha256": EXPECTED_CG_SIM_SHA256,
            },
            "cg_shared_library": {
                "path": str(CG_LIB_PATH.resolve()),
                "sha256": EXPECTED_CG_LIB_SHA256,
            },
            "repair_audit": {
                "path": str(REPAIR_AUDIT_PATH.resolve()),
                "sha256": EXPECTED_REPAIR_AUDIT_SHA256,
            },
            "superseded_design_v1": {
                "path": str(EXPECTED_DESIGN_PREREGISTRATION_V1.resolve()),
                "sha256": EXPECTED_DESIGN_PREREGISTRATION_V1_SHA256,
                "authoritative": False,
            },
            "authoritative_design_v2": {
                "path": str(EXPECTED_DESIGN_PREREGISTRATION_V2.resolve()),
                "sha256": EXPECTED_DESIGN_PREREGISTRATION_V2_SHA256,
                "authoritative": True,
            },
            "general_bc_checkpoint": {
                "path": str(bc_checkpoint_path),
                "sha256": bc_checkpoint_hash,
            },
            "special_bc_archive": {
                "path": str(special_data_path),
                "sha256": EXPECTED_SPECIAL_DATA_SHA256,
                "split": "train",
                "team": "Pokemon Fan",
                "valid_rows_used_for_training": 0,
            },
        },
        "special_bc": {
            "seed": SPECIAL_SEED,
            "steps_requested": SPECIAL_STEPS,
            "batch_indices": list(SPECIAL_BATCH_INDICES),
            "batch_selection": "explicit_without_replacement",
            "batch_sha256": [
                batch_hashes[index] for index in SPECIAL_BATCH_INDICES
            ],
            "rows_per_batch": SPECIAL_BATCH_SIZE,
            "rows_exact": SPECIAL_STEPS * SPECIAL_BATCH_SIZE,
            "context34_rows_per_batch": SPECIAL_CONTEXT34_ROWS_PER_BATCH,
            "context34_rows_exact": (
                SPECIAL_STEPS * SPECIAL_CONTEXT34_ROWS_PER_BATCH
            ),
            "loss": special_config.bc_replay_loss,
            "order_context_weight": (
                special_config.bc_replay_order_context_weight
            ),
            "learning_rate": special_learning_rate,
            "weight_decay": config.weight_decay,
            "adamw_eps": 1e-5,
            "max_grad_norm": config.max_grad_norm,
            "active_parameter_scope": "actor_query_key_residual_and_count_head",
        },
        "replay_cache": {
            "sha256": cache_hash,
            "batches": len(replay_batches),
            "rows": len(replay_batches) * SPECIAL_BATCH_SIZE,
            "context34_rows": sum(context34_per_batch),
            "all_batches_have_frozen_context34_quota": True,
        },
        "optimizer": {
            "parent_actor_parameter_names": actor_names,
            "parent_value_parameter_names": value_names,
            "frozen_shared_parameter_names": list(
                EXPECTED_FROZEN_SHARED_PARAMETER_NAMES
            ),
            "mutable_parameter_names": list(EXPECTED_MUTABLE_PARAMETER_NAMES),
            "parent_replay_state_count": len(parent_replay_steps),
            "parent_replay_steps": parent_replay_steps,
            "parent_replay_state_sha256": parent_replay_hash,
            "parent_ppo_state_count": len(ppo_optimizer_steps),
            "parent_ppo_steps": ppo_optimizer_steps,
            "parent_ppo_state_sha256": ppo_optimizer_hash_before,
            "full_24_optimizer_head_only_grad_scope": optimizer_scope_audit,
            "subset_optimizer_projection_or_state_merge_forbidden": True,
        },
        "determinism": {
            "python_executable": str(executable),
            "python_prefix": str(Path(sys.prefix).resolve()),
            "expected_python": str(EXPECTED_PYTHON.resolve()),
            "device": str(device),
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "python_seed": SPECIAL_SEED,
            "torch_seed": SPECIAL_SEED,
        },
        "device_preflight": device_preflight,
    }

    if args.audit_only:
        zero_state, zero_state_audit = validate_full_replay_state(
            parent_replay_state,
            replay_optimizer,
            actor_names,
            full_actor_parameters,
            config,
            special_steps=0,
        )
        parent_hash_after = repair_audit.file_sha256(parent_path)
        result = common | {
            "status": "audit_passed",
            "optimizer_steps": 0,
            "training_rows": 0,
            "training_context34_rows": 0,
            "checkpoint_writes": 0,
            "formal_stage_created": False,
            "formal_output_root_absent": not EXPECTED_DESIGN_ROOT.exists(),
            "gradient_guard_optimizer_steps": len(
                replay_optimizer.step_audits
            ),
            "model_state_unchanged": (
                repair_audit.changed_tensor_names(
                    model_state_before,
                    repair_audit.clone_model_state(model),
                )
                == []
            ),
            "parent_full_replay_state_unchanged": (
                repair_audit.nested_sha256(parent_replay_state)
                == EXPECTED_PARENT_REPLAY_SHA256
            ),
            "zero_step_full_state_exact_parent": (
                repair_audit.nested_sha256(zero_state)
                == EXPECTED_PARENT_REPLAY_SHA256
            ),
            "zero_step_full_24_state_protocol": zero_state_audit,
            "ppo_optimizer_state_unchanged": (
                repair_audit.nested_sha256(parent.get("optimizer_state_dict"))
                == ppo_optimizer_hash_before
            ),
            "parent_checkpoint_unchanged": (
                parent_hash_after == EXPECTED_PARENT_SHA256
            ),
            "all_frozen_files_unchanged": all_files_unchanged(bound_files),
            "all_state_finite": all(
                (
                    repair_audit.finite_nested(model_state_before),
                    repair_audit.finite_nested(zero_state),
                    repair_audit.finite_nested(ppo_optimizer_state_before),
                )
            ),
        }
        required = (
            "model_state_unchanged",
            "formal_output_root_absent",
            "parent_full_replay_state_unchanged",
            "zero_step_full_state_exact_parent",
            "ppo_optimizer_state_unchanged",
            "parent_checkpoint_unchanged",
            "all_frozen_files_unchanged",
            "all_state_finite",
        )
        if not all(result[key] for key in required):
            raise RuntimeError("Audit-only mode failed a zero-step integrity gate")
        if result["gradient_guard_optimizer_steps"] != 0:
            raise RuntimeError("Audit-only mode executed an optimizer step")
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(output_dir / "preflight_audit.json", result)
        if sorted(path.name for path in output_dir.iterdir()) != [
            "preflight_audit.json"
        ]:
            raise RuntimeError("Audit-only output contains an unexpected file")
        if EXPECTED_DESIGN_ROOT.exists():
            raise RuntimeError("Audit-only mode created the formal output root")
        print(json.dumps(result, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    per_step: list[dict[str, Any]] = []
    endpoint_snapshots: list[dict[str, Any]] = []

    for special_step, batch_index in enumerate(SPECIAL_BATCH_INDICES, start=1):
        if any(
            parameter.grad is not None
            for _, parameter in (
                replay_optimizer.frozen_shared_named
                + replay_optimizer.frozen_value_named
            )
        ):
            raise RuntimeError(
                f"A frozen parameter had a gradient before step {special_step}"
            )
        metrics = ppo.bc_replay_update(
            model,
            replay_optimizer,
            [replay_batches[batch_index]],
            special_config,
            device,
            config.learning_rate,
        )
        if (
            not isinstance(metrics, dict)
            or metrics.get("steps") != 1
            or int(metrics.get("rows", -1)) != SPECIAL_BATCH_SIZE
            or int(metrics.get("context_34_rows", -1))
            != SPECIAL_CONTEXT34_ROWS_PER_BATCH
            or metrics.get("selected_batch_indices") != [0]
        ):
            raise RuntimeError("Special-BC step integrity failed")
        if not repair_audit.finite_nested(metrics):
            raise FloatingPointError("Special-BC metrics contain non-finite values")
        if len(replay_optimizer.step_audits) != special_step:
            raise RuntimeError("Full AdamW gradient guard did not audit every step")
        gradient_guard = copy.deepcopy(replay_optimizer.step_audits[-1])
        if gradient_guard["step_ordinal"] != special_step:
            raise RuntimeError("Full AdamW gradient-guard step order drifted")
        per_step.append(
            {
                "special_step": special_step,
                "batch_index": batch_index,
                "batch_sha256": batch_hashes[batch_index],
                "metrics": metrics,
                "gradient_guard": gradient_guard,
            }
        )

        if special_step in PUBLISHED_ENDPOINT_STEPS:
            endpoint_model_state = repair_audit.clone_model_state(model)
            model_integrity = validate_model_endpoint(
                model_state_before,
                endpoint_model_state,
            )
            endpoint_replay_state, replay_integrity = (
                validate_full_replay_state(
                    parent_replay_state,
                    replay_optimizer,
                    actor_names,
                    full_actor_parameters,
                    config,
                    special_steps=special_step,
                )
            )
            if (
                repair_audit.nested_sha256(parent.get("optimizer_state_dict"))
                != ppo_optimizer_hash_before
            ):
                raise RuntimeError(
                    f"PPO optimizer changed before P{special_step} snapshot"
                )
            if repair_audit.file_sha256(parent_path) != EXPECTED_PARENT_SHA256:
                raise RuntimeError(
                    f"Parent checkpoint changed before P{special_step} snapshot"
                )
            if not all_files_unchanged(bound_files):
                raise RuntimeError(
                    f"A frozen source changed before P{special_step} snapshot"
                )
            endpoint_snapshots.append(
                {
                    "special_steps": special_step,
                    "model_state_dict": endpoint_model_state,
                    "model_state_sha256": ppo.model_state_sha256(model),
                    "model_integrity": model_integrity,
                    "replay_state_dict": endpoint_replay_state,
                    "replay_state_sha256": replay_integrity["state_sha256"],
                    "replay_integrity": replay_integrity,
                }
            )

    if len(per_step) != SPECIAL_STEPS:
        raise RuntimeError("Continuous actor-head-only trajectory did not reach P16")
    if tuple(
        int(snapshot["special_steps"]) for snapshot in endpoint_snapshots
    ) != PUBLISHED_ENDPOINT_STEPS:
        raise RuntimeError("P4/P8/P12/P16 endpoint snapshots are incomplete")

    final_model_state = repair_audit.clone_model_state(model)
    final_model_hash = ppo.model_state_sha256(model)
    final_model_integrity = validate_model_endpoint(
        model_state_before,
        final_model_state,
    )
    final_replay_state, final_replay_integrity = validate_full_replay_state(
        parent_replay_state,
        replay_optimizer,
        actor_names,
        full_actor_parameters,
        config,
        special_steps=SPECIAL_STEPS,
    )
    if (
        final_model_hash != endpoint_snapshots[-1]["model_state_sha256"]
        or repair_audit.nested_sha256(final_replay_state)
        != endpoint_snapshots[-1]["replay_state_sha256"]
    ):
        raise RuntimeError("Final P16 state does not match the P16 snapshot")
    if repair_audit.nested_sha256(parent_replay_state) != EXPECTED_PARENT_REPLAY_SHA256:
        raise RuntimeError("In-memory parent replay optimizer state changed")
    if (
        repair_audit.nested_sha256(parent.get("optimizer_state_dict"))
        != ppo_optimizer_hash_before
    ):
        raise RuntimeError("PPO optimizer changed during actor-head-only sweep")
    parent_hash_after = repair_audit.file_sha256(parent_path)
    if parent_hash_after != EXPECTED_PARENT_SHA256:
        raise RuntimeError("Parent checkpoint changed during actor-head-only sweep")
    if not all_files_unchanged(bound_files):
        raise RuntimeError("A frozen source or dependency changed during sweep")

    trajectory_integrity = {
        "optimizer_steps": len(per_step),
        "rows": len(per_step) * SPECIAL_BATCH_SIZE,
        "context34_rows": (
            len(per_step) * SPECIAL_CONTEXT34_ROWS_PER_BATCH
        ),
        "batch_indices_consumed": [
            record["batch_index"] for record in per_step
        ],
        "batch_order_exact": tuple(
            record["batch_index"] for record in per_step
        )
        == SPECIAL_BATCH_INDICES,
        "no_step_beyond_16": True,
        "all_metrics_finite": all(
            repair_audit.finite_nested(record["metrics"])
            for record in per_step
        ),
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": final_model_hash,
        "model_integrity": final_model_integrity,
        "replay_state_sha256_before": parent_replay_hash,
        "replay_state_sha256_after": final_replay_integrity["state_sha256"],
        "replay_integrity": final_replay_integrity,
        "ppo_state_sha256_after": ppo_optimizer_hash_before,
        "ppo_optimizer_state_unchanged": True,
        "parent_checkpoint_sha256_after": parent_hash_after,
        "parent_checkpoint_unchanged": True,
        "all_frozen_files_unchanged": True,
    }
    if (
        trajectory_integrity["optimizer_steps"] != SPECIAL_STEPS
        or trajectory_integrity["rows"] != 4096
        or trajectory_integrity["context34_rows"] != 16
        or not trajectory_integrity["batch_order_exact"]
        or not trajectory_integrity["all_metrics_finite"]
    ):
        raise RuntimeError("Full P16 trajectory failed a frozen integrity gate")

    endpoint_records: list[dict[str, Any]] = []
    for snapshot in endpoint_snapshots:
        special_steps = int(snapshot["special_steps"])
        endpoint_special = copy.deepcopy(common["special_bc"])
        endpoint_special.update(
            {
                "steps_requested": special_steps,
                "batch_indices": list(SPECIAL_BATCH_INDICES[:special_steps]),
                "batch_sha256": [
                    batch_hashes[index]
                    for index in SPECIAL_BATCH_INDICES[:special_steps]
                ],
                "rows_exact": special_steps * SPECIAL_BATCH_SIZE,
                "context34_rows_exact": (
                    special_steps * SPECIAL_CONTEXT34_ROWS_PER_BATCH
                ),
                "trajectory_reconstruction_seed": SPECIAL_SEED,
            }
        )
        endpoint_integrity = {
            "optimizer_steps": special_steps,
            "rows": special_steps * SPECIAL_BATCH_SIZE,
            "context34_rows": (
                special_steps * SPECIAL_CONTEXT34_ROWS_PER_BATCH
            ),
            "all_metrics_finite": all(
                repair_audit.finite_nested(record["metrics"])
                for record in per_step[:special_steps]
            ),
            "model_state_sha256_before": model_hash_before,
            "model_state_sha256_after": snapshot["model_state_sha256"],
            "model_integrity": snapshot["model_integrity"],
            "replay_state_sha256_before": parent_replay_hash,
            "replay_state_sha256_after": snapshot["replay_state_sha256"],
            "replay_integrity": snapshot["replay_integrity"],
            "ppo_state_sha256_after": ppo_optimizer_hash_before,
            "ppo_optimizer_state_unchanged": True,
            "parent_checkpoint_sha256_after": parent_hash_after,
            "parent_checkpoint_unchanged": True,
            "all_frozen_files_unchanged": True,
        }
        endpoint_provenance = common | {
            "status": "actor_head_only_prefix_endpoint_completed",
            "special_bc": endpoint_special,
            "per_step": per_step[:special_steps],
            "integrity": endpoint_integrity,
            "full_continuous_trajectory": {
                "steps_executed_in_memory": SPECIAL_STEPS,
                "final_p16_model_state_sha256": final_model_hash,
                "final_p16_replay_state_sha256": (
                    final_replay_integrity["state_sha256"]
                ),
                "no_step_beyond_16": True,
            },
            "checkpoint_update_label": EXPECTED_PARENT_UPDATE,
            "not_a_new_ppo_update": True,
        }
        output_checkpoint = output_dir / (
            "special-bc-actorheadonly-pokemonfan-prefix-"
            f"{special_steps:04d}.pt"
        )
        payload = copy.deepcopy(parent)
        payload["model_state_dict"] = snapshot["model_state_dict"]
        payload["bc_replay_optimizer_state_dict"] = snapshot[
            "replay_state_dict"
        ]
        payload["post_ppo_special_bc"] = endpoint_provenance
        torch.save(payload, output_checkpoint)
        endpoint_records.append(
            {
                "name": f"p{special_steps}",
                "path": str(output_checkpoint),
                "sha256": repair_audit.file_sha256(output_checkpoint),
                "update": EXPECTED_PARENT_UPDATE,
                "special_bc_steps": special_steps,
                "model_state_sha256": snapshot["model_state_sha256"],
                "replay_state_sha256": snapshot["replay_state_sha256"],
                "mutable_replay_adamw_step": (
                    EXPECTED_PARENT_REPLAY_STATE_STEP + special_steps
                ),
                "frozen_replay_adamw_step": (
                    EXPECTED_PARENT_REPLAY_STATE_STEP
                ),
                "replay_state_count": EXPECTED_PARENT_REPLAY_STATE_COUNT,
            }
        )

    result = common | {
        "status": "actor_head_only_prefix_sweep_completed",
        "per_step": per_step,
        "trajectory_integrity": trajectory_integrity,
        "endpoints": endpoint_records,
        "published_endpoint_count": len(endpoint_records),
        "checkpoint_writes": len(endpoint_records),
        "checkpoint_update_label": EXPECTED_PARENT_UPDATE,
        "not_a_new_ppo_update": True,
    }
    manifest_path = output_dir / "actorheadonly_prefix_sweep_manifest.json"
    write_json(manifest_path, result)

    expected_files = [
        "special-bc-actorheadonly-pokemonfan-prefix-"
        f"{step:04d}.pt"
        for step in PUBLISHED_ENDPOINT_STEPS
    ] + [manifest_path.name]
    if sorted(path.name for path in output_dir.iterdir()) != sorted(expected_files):
        raise RuntimeError("Formal sweep output contains unexpected files")
    if len(endpoint_records) != 4:
        raise RuntimeError("Formal sweep did not publish exactly four checkpoints")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
