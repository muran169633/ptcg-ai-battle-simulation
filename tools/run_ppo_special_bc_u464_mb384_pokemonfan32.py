#!/usr/bin/env python3
"""Apply the frozen PokemonFan P32 specialist BC stage after audited U464 PPO.

This standalone executor binds the exact audited mb384 U464 parent, general BC,
PokemonFan time-forward archive, cache, batch order, seed, optimizer states,
and U464-specific 16-step witness.  The parent PPO checkpoint remains
immutable.  Only the actor parameter set owned by the embedded BC-replay
optimizer is updated; the PPO optimizer and value-head tensors are retained
byte-for-byte.  A separate audit-only run binds the deterministic replay cache
without authorizing or executing an optimizer step.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
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

SPECIAL_CACHE_BATCHES = 32
SPECIAL_BATCH_SIZE = 256
SPECIAL_WORKERS = 8
SPECIAL_CONTEXT34_ROWS_PER_BATCH = 1
SPECIAL_REPAIR_STEPS = 32
PREFIX_STEPS = 16
EXPECTED_PREFIX_MODEL_SHA256 = (
    "dd3ee4ce1f7fb29d8a2f381d79d2360af80fa00ee5f30e1c30dbdd4c5f9fd06c"
)
EXPECTED_PREFIX_REPLAY_SHA256 = (
    "7059a2c250e08a158314b735f6d611ebadd0f10f1975a4cc4cdb7cca6b48f34f"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_ID = 202608040
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
EXPECTED_REPLAY_STATE_STEP = 16
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
EXPECTED_CACHE_SHA256 = (
    "a1c16b8b2d6fbf45cf1dfd38ba4eb5527ee444d4d90e09b4f57ca185bdce3021"
)
SPECIAL_SEED = 202608012
SPECIAL_BATCH_INDICES = (
    0, 1, 2, 7, 8, 9, 10, 14, 15, 16, 18, 19, 22, 26, 27, 31,
    3, 4, 5, 6, 11, 12, 13, 17, 20, 21, 23, 24, 25, 28, 29, 30,
)
# Preserve the exact binary float produced by the frozen base LR and scale.
EXPECTED_SPECIAL_LEARNING_RATE = 3.6e-5 * 0.05
EXPECTED_ACTOR_PARAMETER_NAMES = (
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


def validate_sha(path: Path, expected: str, label: str) -> str:
    actual = repair_audit.file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


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


def validate_request(repair_steps: int, batch_indices: list[int]) -> None:
    if repair_steps != SPECIAL_REPAIR_STEPS:
        raise ValueError(
            f"This protocol requires exactly {SPECIAL_REPAIR_STEPS} steps"
        )
    if len(batch_indices) != repair_steps:
        raise ValueError("--batch-indices length must equal --repair-steps")
    if len(set(batch_indices)) != len(batch_indices):
        raise ValueError("Special-BC batch indices must be without replacement")
    if min(batch_indices) < 0 or max(batch_indices) >= SPECIAL_CACHE_BATCHES:
        raise ValueError("Special-BC batch index is outside the frozen cache")
    if tuple(batch_indices) != SPECIAL_BATCH_INDICES:
        raise ValueError("Special-BC batch order is not the frozen exact P32 order")
    if set(batch_indices) != set(range(SPECIAL_CACHE_BATCHES)):
        raise ValueError("Exact P32 must consume every frozen cache batch once")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue the exact audited U464 embedded actor BC optimizer on the frozen PokemonFan "
            "specialist archive."
        )
    )
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    # These values are deliberately not exposed as CLI knobs.  The design ID
    # is independent of the training RNG seed: exact P32 requires 202608012.
    args.parent_checkpoint = EXPECTED_PARENT_CHECKPOINT
    args.expected_parent_sha256 = EXPECTED_PARENT_SHA256
    args.expected_parent_update = EXPECTED_PARENT_UPDATE
    args.expected_bc_checkpoint_sha256 = EXPECTED_GENERAL_BC_SHA256
    args.special_data = EXPECTED_SPECIAL_DATA
    args.expected_special_data_sha256 = EXPECTED_SPECIAL_DATA_SHA256
    args.expected_train_ppo_sha256 = EXPECTED_TRAIN_PPO_SHA256
    args.expected_cache_sha256 = EXPECTED_CACHE_SHA256
    args.expected_replay_state_step = EXPECTED_REPLAY_STATE_STEP
    args.special_seed = SPECIAL_SEED
    args.repair_steps = SPECIAL_REPAIR_STEPS
    args.batch_indices = list(SPECIAL_BATCH_INDICES)
    return args


def main() -> None:
    args = parse_args()
    validate_request(args.repair_steps, args.batch_indices)

    tool_path = Path(__file__).resolve()
    train_path = tool_path.with_name("train_ppo.py")
    parent_path = args.parent_checkpoint.resolve()
    special_data_path = args.special_data.resolve()
    output_dir = args.output_dir.resolve()
    if parent_path != EXPECTED_PARENT_CHECKPOINT.resolve():
        raise ValueError("Parent checkpoint path is not the frozen U464 path")
    if special_data_path != EXPECTED_SPECIAL_DATA.resolve():
        raise ValueError("Special archive path is not the frozen PokemonFan path")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse existing output directory: {output_dir}"
        )

    tool_hash = validate_sha(tool_path, args.expected_tool_sha256, "tool")
    train_hash = validate_sha(
        train_path,
        args.expected_train_ppo_sha256,
        "train_ppo",
    )
    parent_hash_before = validate_sha(
        parent_path,
        args.expected_parent_sha256,
        "parent checkpoint",
    )
    special_data_hash = validate_sha(
        special_data_path,
        args.expected_special_data_sha256,
        "special BC archive",
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    random.seed(args.special_seed)
    torch.manual_seed(args.special_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.special_seed)

    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise ValueError("Parent checkpoint is not a compatible PPO checkpoint")
    if int(parent.get("update", -1)) != args.expected_parent_update:
        raise ValueError("Parent checkpoint update does not match the request")
    config = validate_parent_config(parent.get("config"))

    bc_checkpoint_path = Path(config.bc_checkpoint).resolve()
    if bc_checkpoint_path != EXPECTED_GENERAL_BC_CHECKPOINT.resolve():
        raise ValueError("Parent general-BC path is not the frozen checkpoint")
    bc_checkpoint_hash = validate_sha(
        bc_checkpoint_path,
        args.expected_bc_checkpoint_sha256,
        "general BC architecture checkpoint",
    )
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
    actor_parameters, _, trainable_manifest = ppo.configure_trainable_scope(
        model,
        config.trainable_scope,
    )
    actor_names = list(trainable_manifest["actor_parameter_names"])
    value_names = list(trainable_manifest["value_parameter_names"])
    if tuple(actor_names) != EXPECTED_ACTOR_PARAMETER_NAMES:
        raise ValueError("Actor parameter manifest is not the frozen 24-tensor set")
    if tuple(value_names) != EXPECTED_VALUE_PARAMETER_NAMES:
        raise ValueError("Value parameter manifest is not the frozen 4-tensor set")
    parent_parameter_names = parent.get("optimizer_parameter_names")
    if not isinstance(parent_parameter_names, dict):
        raise ValueError("Parent checkpoint has no optimizer parameter manifest")
    if actor_names != parent_parameter_names.get("actor"):
        raise ValueError("Actor parameter names do not match the parent optimizer")

    replay_state = parent.get("bc_replay_optimizer_state_dict")
    if not isinstance(replay_state, dict):
        raise ValueError("Parent checkpoint has no replay optimizer state")
    replay_optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=config.learning_rate * config.bc_replay_lr_scale,
        eps=1e-5,
        weight_decay=config.weight_decay,
    )
    replay_optimizer.load_state_dict(replay_state)
    replay_steps_before = repair_audit.optimizer_steps(
        replay_optimizer.state_dict()
    )
    if len(replay_steps_before) != len(actor_names):
        raise ValueError("Replay optimizer does not cover every actor tensor")
    if set(replay_steps_before) != {args.expected_replay_state_step}:
        raise ValueError("Replay optimizer step does not match the request")

    ppo_optimizer_state_before = copy.deepcopy(
        parent.get("optimizer_state_dict")
    )
    ppo_optimizer_hash_before = repair_audit.nested_sha256(
        ppo_optimizer_state_before
    )
    replay_optimizer_hash_before = repair_audit.nested_sha256(
        replay_optimizer.state_dict()
    )
    model_state_before = repair_audit.clone_model_state(model)
    model_hash_before = ppo.model_state_sha256(model)
    if model_hash_before != EXPECTED_PARENT_MODEL_SHA256:
        raise ValueError("Parent model-state SHA256 is not the frozen U464 state")
    if replay_optimizer_hash_before != EXPECTED_PARENT_REPLAY_SHA256:
        raise ValueError("Parent replay-optimizer SHA256 is not frozen")
    if ppo_optimizer_hash_before != EXPECTED_PARENT_PPO_SHA256:
        raise ValueError("Parent PPO-optimizer SHA256 is not frozen")

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
    special_config.seed = args.special_seed
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
    if cache_hash != args.expected_cache_sha256:
        raise ValueError(
            "Special replay cache SHA256 mismatch: "
            f"expected {args.expected_cache_sha256}, got {cache_hash}"
        )

    common: dict[str, Any] = {
        "schema_version": "ptcg-u464-mb384-pokemonfan-p32-v1",
        "mode": "audit_only" if args.audit_only else "special_bc",
        "frozen_protocol": {
            "design_id": DESIGN_ID,
            "training_rng_seed": SPECIAL_SEED,
            "design_id_is_not_training_rng_seed": True,
            "parent_model_state_sha256": EXPECTED_PARENT_MODEL_SHA256,
            "parent_replay_state_sha256": EXPECTED_PARENT_REPLAY_SHA256,
            "parent_ppo_state_sha256": EXPECTED_PARENT_PPO_SHA256,
            "expected_replay_step_before": EXPECTED_REPLAY_STATE_STEP,
            "expected_replay_step_after": (
                EXPECTED_REPLAY_STATE_STEP + SPECIAL_REPAIR_STEPS
            ),
            "prefix_steps": PREFIX_STEPS,
            "prefix_model_state_sha256": EXPECTED_PREFIX_MODEL_SHA256,
            "prefix_replay_state_sha256": EXPECTED_PREFIX_REPLAY_SHA256,
            "formal_training_started_by_this_mode": not args.audit_only,
        },
        "parent": {
            "path": str(parent_path),
            "sha256": parent_hash_before,
            "update": int(parent["update"]),
            "model_state_sha256": model_hash_before,
        },
        "sources": {
            "tool": {"path": str(tool_path), "sha256": tool_hash},
            "train_ppo": {"path": str(train_path), "sha256": train_hash},
            "general_bc_checkpoint": {
                "path": str(bc_checkpoint_path),
                "sha256": bc_checkpoint_hash,
            },
            "special_bc_archive": {
                "path": str(special_data_path),
                "sha256": special_data_hash,
                "split": "train",
            },
        },
        "special_bc": {
            "seed": args.special_seed,
            "steps_requested": args.repair_steps,
            "batch_indices": list(args.batch_indices),
            "batch_selection": "explicit_without_replacement",
            "batch_sha256": [
                batch_hashes[index] for index in args.batch_indices
            ],
            "rows_per_batch": SPECIAL_BATCH_SIZE,
            "context34_rows_per_batch": SPECIAL_CONTEXT34_ROWS_PER_BATCH,
            "loss": special_config.bc_replay_loss,
            "order_context_weight": (
                special_config.bc_replay_order_context_weight
            ),
            "learning_rate": special_learning_rate,
            "max_grad_norm": config.max_grad_norm,
            "trainable_scope": config.trainable_scope,
        },
        "replay_cache": {
            "sha256": cache_hash,
            "batches": len(replay_batches),
            "rows": len(replay_batches) * SPECIAL_BATCH_SIZE,
            "context34_rows": sum(context34_per_batch),
            "all_batches_have_frozen_context34_quota": True,
        },
        "optimizer": {
            "actor_parameter_names": actor_names,
            "value_parameter_names": value_names,
            "replay_state_tensor_count": len(replay_steps_before),
            "replay_steps_before": replay_steps_before,
            "replay_state_sha256_before": replay_optimizer_hash_before,
            "ppo_state_sha256_before": ppo_optimizer_hash_before,
            "replay_state_loaded_from_parent": True,
        },
        "determinism": {
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "python_seed": args.special_seed,
            "torch_seed": args.special_seed,
        },
    }

    output_dir.mkdir(parents=False)
    if args.audit_only:
        parent_hash_after = repair_audit.file_sha256(parent_path)
        result = common | {
            "status": "audit_passed",
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "model_state_unchanged": (
                ppo.model_state_sha256(model) == model_hash_before
            ),
            "replay_optimizer_state_unchanged": (
                repair_audit.nested_sha256(replay_optimizer.state_dict())
                == replay_optimizer_hash_before
            ),
            "ppo_optimizer_state_unchanged": (
                repair_audit.nested_sha256(parent.get("optimizer_state_dict"))
                == ppo_optimizer_hash_before
            ),
            "parent_checkpoint_unchanged": (
                parent_hash_after == parent_hash_before
            ),
        }
        if not all(
            result[key]
            for key in (
                "model_state_unchanged",
                "replay_optimizer_state_unchanged",
                "ppo_optimizer_state_unchanged",
                "parent_checkpoint_unchanged",
            )
        ):
            raise RuntimeError("Audit-only mode mutated frozen state")
        write_json(output_dir / "preflight_audit.json", result)
        print(json.dumps(result, sort_keys=True))
        return

    per_step: list[dict[str, Any]] = []
    prefix_integrity: dict[str, Any] | None = None
    for special_step, batch_index in enumerate(args.batch_indices, start=1):
        metrics = ppo.bc_replay_update(
            model,
            replay_optimizer,
            [replay_batches[batch_index]],
            special_config,
            device,
            config.learning_rate,
        )
        if not isinstance(metrics, dict) or metrics.get("steps") != 1:
            raise RuntimeError("Special BC core did not execute one step")
        if int(metrics.get("rows", -1)) != SPECIAL_BATCH_SIZE:
            raise RuntimeError("Special BC core consumed an unexpected row count")
        if int(metrics.get("context_34_rows", -1)) != (
            SPECIAL_CONTEXT34_ROWS_PER_BATCH
        ):
            raise RuntimeError("Special BC core consumed a wrong context-34 quota")
        if not repair_audit.finite_nested(metrics):
            raise FloatingPointError("Non-finite special-BC metric")
        per_step.append(
            {
                "special_step": special_step,
                "batch_index": batch_index,
                "batch_sha256": batch_hashes[batch_index],
                "metrics": metrics,
            }
        )
        if special_step == PREFIX_STEPS:
            prefix_model_hash = ppo.model_state_sha256(model)
            prefix_replay_hash = repair_audit.nested_sha256(
                replay_optimizer.state_dict()
            )
            prefix_integrity = {
                "steps": PREFIX_STEPS,
                "model_state_sha256": prefix_model_hash,
                "replay_state_sha256": prefix_replay_hash,
                "expected_model_state_sha256": (
                    EXPECTED_PREFIX_MODEL_SHA256
                ),
                "expected_replay_state_sha256": (
                    EXPECTED_PREFIX_REPLAY_SHA256
                ),
                "exact_match": (
                    prefix_model_hash == EXPECTED_PREFIX_MODEL_SHA256
                    and prefix_replay_hash == EXPECTED_PREFIX_REPLAY_SHA256
                ),
            }
            if not prefix_integrity["exact_match"]:
                raise RuntimeError(
                    "The 32-step run did not reproduce the frozen 16-step "
                    "prefix exactly"
                )

    if prefix_integrity is None:
        raise RuntimeError("The frozen 16-step prefix was not reached")

    model_state_after = repair_audit.clone_model_state(model)
    model_hash_after = ppo.model_state_sha256(model)
    if not repair_audit.finite_nested(model_state_after):
        raise FloatingPointError("Special-BC model contains non-finite tensors")
    changed_names = repair_audit.changed_tensor_names(
        model_state_before,
        model_state_after,
    )
    if not changed_names:
        raise RuntimeError("Special BC did not change any model tensor")
    unexpected_changes = sorted(set(changed_names) - set(actor_names))
    if unexpected_changes:
        raise RuntimeError(
            "Special BC changed tensors outside actor scope: "
            + json.dumps(unexpected_changes)
        )
    value_parameters_unchanged = all(
        torch.equal(model_state_before[name], model_state_after[name])
        for name in value_names
    )
    if not value_parameters_unchanged:
        raise RuntimeError("Special BC changed value-head tensors")

    replay_state_after = replay_optimizer.state_dict()
    replay_steps_after = repair_audit.optimizer_steps(replay_state_after)
    expected_final_step = args.expected_replay_state_step + args.repair_steps
    if set(replay_steps_after) != {expected_final_step}:
        raise RuntimeError(
            "Replay AdamW step did not advance exactly "
            f"{args.repair_steps} times"
        )
    if (
        repair_audit.nested_sha256(parent.get("optimizer_state_dict"))
        != ppo_optimizer_hash_before
    ):
        raise RuntimeError("PPO optimizer state changed during special BC")
    parent_hash_after = repair_audit.file_sha256(parent_path)
    if parent_hash_after != parent_hash_before:
        raise RuntimeError("Parent checkpoint changed during special BC")

    integrity = {
        "optimizer_steps": len(per_step),
        "rows": len(per_step) * SPECIAL_BATCH_SIZE,
        "context34_rows": (
            len(per_step) * SPECIAL_CONTEXT34_ROWS_PER_BATCH
        ),
        "all_metrics_and_model_tensors_finite": True,
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "model_state_changed": model_hash_after != model_hash_before,
        "changed_parameter_names": changed_names,
        "changed_parameters_subset_of_actor": True,
        "value_head_parameters_unchanged": value_parameters_unchanged,
        "replay_steps_after": replay_steps_after,
        "replay_state_sha256_after": repair_audit.nested_sha256(
            replay_state_after
        ),
        "ppo_state_sha256_after": ppo_optimizer_hash_before,
        "ppo_optimizer_state_unchanged": True,
        "parent_checkpoint_sha256_after": parent_hash_after,
        "parent_checkpoint_unchanged": True,
        "frozen_16_step_prefix": prefix_integrity,
    }
    provenance = common | {
        "status": "special_bc_completed",
        "per_step": per_step,
        "integrity": integrity,
        "checkpoint_update_label": int(parent["update"]),
        "not_a_new_ppo_update": True,
    }
    output_checkpoint = output_dir / "special-bc-0032.pt"
    payload = copy.deepcopy(parent)
    payload["model_state_dict"] = model_state_after
    payload["bc_replay_optimizer_state_dict"] = replay_state_after
    payload["post_ppo_special_bc"] = provenance
    torch.save(payload, output_checkpoint)
    output_checkpoint_hash = repair_audit.file_sha256(output_checkpoint)

    result = provenance | {
        "checkpoint": {
            "path": str(output_checkpoint),
            "sha256": output_checkpoint_hash,
            "update": int(parent["update"]),
        },
        "checkpoint_writes": 1,
    }
    write_json(output_dir / "special_bc_manifest.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
