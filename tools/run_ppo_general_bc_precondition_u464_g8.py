#!/usr/bin/env python3
"""Apply the frozen eight-step general-BC preconditioner to audited U464.

The operation continues the complete 24-tensor replay AdamW state embedded in
U464, leaves the PPO optimizer, value head, quota state, and parent checkpoint
unchanged, and writes exactly one update-464 checkpoint.  Audit mode performs
all input/cache/state checks without an optimizer step or checkpoint write.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import torch

import run_ppo_bc_repair as audit
import train_ppo as ppo


REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_ID = 202608090
G8_SEED = 202608013
G8_STEPS = 8
G8_BATCH_INDICES = (1, 3, 4, 5, 6, 7, 8, 11)
G8_LR_SCALE = 0.05
G8_LEARNING_RATE = 3.6e-5 * G8_LR_SCALE
G8_DISPLACEMENT_MIN = 0.00145
G8_DISPLACEMENT_MAX = 0.00340

PARENT_PATH = REPO_ROOT / (
    "artifacts/ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_localtransport_u456_to_u464_"
    "seed202607336/B_gold_league/seed-202607336/checkpoints/update-0464.pt"
)
PARENT_SHA256 = "fe51f40f37fca329cd6b0c94f7431001bb909b92624df33fcd7cf6f0da976264"
PARENT_UPDATE = 464
PARENT_MODEL_SHA256 = "fa6e42da654f872f6f24836838654286d7cd64e10cacff554bdec36b6aa58f0a"
PARENT_PPO_SHA256 = "904a3d2c8ac63c8a60af2404c216830b8ae30a69fbe972818e05c0136058a54f"
PARENT_REPLAY_SHA256 = "23f775207b0d216c1ffd19bf9213456026b71b786727a4d2ff0f889b6b30341d"
PARENT_QUOTA_SHA256 = "7139320983450f2d2b9d50a65684c31550c409ed8adde9feb84839db24e1fc27"
PARENT_PPO_STEP = 276
PARENT_REPLAY_STEP = 16

GENERAL_BC_PATH = REPO_ROOT / (
    "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_"
    "seed20260922_20260731/best.pt"
)
GENERAL_BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
GENERAL_REPLAY_PATH = REPO_ROOT / (
    "data/bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip"
)
GENERAL_REPLAY_SHA256 = "a3b9d572bfc0a784b5b140d3dbe09314252c46dd0d9c1afa3423236cda54543c"
TRAIN_PPO_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
AUDIT_HELPER_SHA256 = "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"
GENERAL_CACHE_SHA256 = "d5ab8b662915160e0cd655fa75e845a44f539b647729901a0a21f6d2eb81cd59"

EXPECTED_PARENT_CONFIG: dict[str, Any] = {
    "updates": 464,
    "minibatch_size": 384,
    "actor_reduction": "episode_mean",
    "trainable_scope": "last_block_heads",
    "learning_rate": 3.6e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 0.5,
    "policy_temperature": 0.8,
    "bc_replay_data": str(GENERAL_REPLAY_PATH),
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
    "seed": 202607336,
}

EXPECTED_ACTOR_NAMES = (
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
EXPECTED_VALUE_NAMES = (
    "value_head.0.weight",
    "value_head.0.bias",
    "value_head.2.weight",
    "value_head.2.bias",
)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    actual = audit.file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def validate_parent_config(raw: Any) -> ppo.PPOConfig:
    if not isinstance(raw, dict):
        raise ValueError("Parent checkpoint has no PPO config")
    for key, expected in EXPECTED_PARENT_CONFIG.items():
        if raw.get(key) != expected:
            raise ValueError(
                f"Parent config {key!r} mismatch: expected {expected!r}, "
                f"got {raw.get(key)!r}"
            )
    return ppo.PPOConfig(**raw)


def state_steps(state: dict[str, Any]) -> set[int]:
    return set(audit.optimizer_steps(state))


def actor_l2(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    names: list[str],
) -> float:
    squared = 0.0
    for name in names:
        delta = after[name].detach().cpu().double() - before[name].detach().cpu().double()
        squared += float(delta.square().sum())
    return math.sqrt(squared)


@torch.inference_mode()
def general_cache_reference_kl(
    model: torch.nn.Module,
    reference_model: torch.nn.Module,
    replay_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    temperature: float,
) -> float:
    model.eval()
    reference_model.eval()
    total = 0.0
    rows = 0
    for cpu_batch in replay_batches:
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        outputs = ppo.model_forward(model, batch, device)
        reference = ppo.model_forward(reference_model, batch, device)
        values = ppo.reference_policy_kl(
            outputs,
            reference,
            batch,
            batch["action_sequences"],
            batch["action_counts"],
            temperature=temperature,
        )
        total += float(values.double().sum())
        rows += int(values.numel())
    if rows != 72 * 256:
        raise RuntimeError(f"Unexpected general-cache row count: {rows}")
    result = total / rows
    if not math.isfinite(result):
        raise FloatingPointError("General-cache reference KL is non-finite")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tool_path = Path(__file__).resolve()
    trainer_path = tool_path.with_name("train_ppo.py")
    helper_path = tool_path.with_name("run_ppo_bc_repair.py")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    try:
        output_dir.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("Output directory must stay inside the repository") from error
    if output_dir.parent == REPO_ROOT or not output_dir.parent.is_dir():
        raise ValueError("Output parent must be an existing task-specific directory")

    tool_hash = require_hash(tool_path, args.expected_tool_sha256, "G8 tool")
    trainer_hash = require_hash(trainer_path, TRAIN_PPO_SHA256, "PPO trainer")
    helper_hash = require_hash(helper_path, AUDIT_HELPER_SHA256, "audit helper")
    parent_hash_before = require_hash(PARENT_PATH, PARENT_SHA256, "U464 parent")
    bc_hash = require_hash(GENERAL_BC_PATH, GENERAL_BC_SHA256, "general BC")
    replay_hash = require_hash(
        GENERAL_REPLAY_PATH, GENERAL_REPLAY_SHA256, "general replay archive"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    random.seed(G8_SEED)
    torch.manual_seed(G8_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(G8_SEED)

    parent = torch.load(PARENT_PATH, map_location="cpu", weights_only=False)
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise ValueError("Parent is not a compatible PPO checkpoint")
    if int(parent.get("update", -1)) != PARENT_UPDATE:
        raise ValueError("Parent update is not exactly 464")
    config = validate_parent_config(parent.get("config"))
    if Path(config.bc_checkpoint).resolve() != GENERAL_BC_PATH.resolve():
        raise ValueError("Parent general-BC path drifted")
    if Path(str(config.bc_replay_data)).resolve() != GENERAL_REPLAY_PATH.resolve():
        raise ValueError("Parent general replay path drifted")

    bc_checkpoint = torch.load(GENERAL_BC_PATH, map_location="cpu", weights_only=False)
    model = ppo.instantiate_model_from_checkpoint(parent, bc_checkpoint, device)
    reference_model = ppo.instantiate_model_from_checkpoint(
        bc_checkpoint, bc_checkpoint, device
    )
    actor_parameters, _, manifest = ppo.configure_trainable_scope(
        model, config.trainable_scope
    )
    actor_names = list(manifest["actor_parameter_names"])
    value_names = list(manifest["value_parameter_names"])
    if tuple(actor_names) != EXPECTED_ACTOR_NAMES:
        raise ValueError("Actor parameter manifest drifted")
    if tuple(value_names) != EXPECTED_VALUE_NAMES:
        raise ValueError("Value parameter manifest drifted")
    optimizer_names = parent.get("optimizer_parameter_names")
    if not isinstance(optimizer_names, dict) or optimizer_names.get("actor") != actor_names:
        raise ValueError("Parent replay optimizer parameter order drifted")

    ppo_state = copy.deepcopy(parent.get("optimizer_state_dict"))
    replay_state = copy.deepcopy(parent.get("bc_replay_optimizer_state_dict"))
    quota_state = copy.deepcopy(parent.get("opponent_quota_state"))
    if not isinstance(ppo_state, dict) or not isinstance(replay_state, dict):
        raise ValueError("Parent optimizer state is missing")
    if audit.nested_sha256(ppo_state) != PARENT_PPO_SHA256:
        raise ValueError("Parent PPO optimizer hash drifted")
    if audit.nested_sha256(replay_state) != PARENT_REPLAY_SHA256:
        raise ValueError("Parent replay optimizer hash drifted")
    if audit.nested_sha256(quota_state) != PARENT_QUOTA_SHA256:
        raise ValueError("Parent quota-state hash drifted")
    if state_steps(ppo_state) != {PARENT_PPO_STEP}:
        raise ValueError("Parent PPO optimizer step is not 276")
    if state_steps(replay_state) != {PARENT_REPLAY_STEP}:
        raise ValueError("Parent replay optimizer step is not 16")

    replay_optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=G8_LEARNING_RATE,
        eps=1e-5,
        weight_decay=config.weight_decay,
    )
    replay_optimizer.load_state_dict(replay_state)
    if audit.nested_sha256(replay_optimizer.state_dict()) != PARENT_REPLAY_SHA256:
        raise ValueError("Replay optimizer direct-load roundtrip drifted")

    model_hash_before = ppo.model_state_sha256(model)
    if model_hash_before != PARENT_MODEL_SHA256:
        raise ValueError("Parent runtime model hash drifted")
    model_state_before = audit.clone_model_state(model)

    replay_batches = ppo.build_bc_replay_batches(config, parent["model_config"])
    cache_hash, batch_hashes = audit.replay_cache_manifest(replay_batches)
    if len(replay_batches) != 72 or cache_hash != GENERAL_CACHE_SHA256:
        raise ValueError("Frozen general replay cache drifted")
    if any(int(batch["contexts"].shape[0]) != 256 for batch in replay_batches):
        raise ValueError("General replay cache has a short batch")
    context34 = [
        int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum())
        for batch in replay_batches
    ]
    if set(context34) != {4}:
        raise ValueError("General replay context-34 quota drifted")
    if len(set(G8_BATCH_INDICES)) != G8_STEPS:
        raise RuntimeError("Frozen G8 batch list is not unique")

    kl_before = general_cache_reference_kl(
        model, reference_model, replay_batches, device, config.policy_temperature
    )
    common: dict[str, Any] = {
        "schema_version": "ptcg-u464-g8-general-bc-precondition-v1",
        "mode": "audit_only" if args.audit_only else "general_bc",
        "frozen_protocol": {
            "design_id": DESIGN_ID,
            "training_rng_seed": G8_SEED,
            "cache_seed": config.seed,
            "steps": G8_STEPS,
            "batch_indices": list(G8_BATCH_INDICES),
            "learning_rate": G8_LEARNING_RATE,
            "displacement_interval": [G8_DISPLACEMENT_MIN, G8_DISPLACEMENT_MAX],
        },
        "parent": {
            "path": str(PARENT_PATH),
            "sha256": parent_hash_before,
            "update": PARENT_UPDATE,
            "model_state_sha256": model_hash_before,
        },
        "sources": {
            "tool": {"path": str(tool_path), "sha256": tool_hash},
            "trainer": {"path": str(trainer_path), "sha256": trainer_hash},
            "audit_helper": {"path": str(helper_path), "sha256": helper_hash},
            "general_bc": {"path": str(GENERAL_BC_PATH), "sha256": bc_hash},
            "general_replay": {
                "path": str(GENERAL_REPLAY_PATH),
                "sha256": replay_hash,
            },
        },
        "cache": {
            "sha256": cache_hash,
            "batches": 72,
            "rows": 72 * 256,
            "context34_rows_per_batch": 4,
            "selected_batch_sha256": [batch_hashes[i] for i in G8_BATCH_INDICES],
        },
        "optimizer_before": {
            "ppo_state_sha256": PARENT_PPO_SHA256,
            "ppo_state_count": len(ppo_state["state"]),
            "ppo_step": PARENT_PPO_STEP,
            "replay_state_sha256": PARENT_REPLAY_SHA256,
            "replay_state_count": len(replay_state["state"]),
            "replay_step": PARENT_REPLAY_STEP,
            "quota_state_sha256": PARENT_QUOTA_SHA256,
        },
        "general_cache_reference_kl_before": kl_before,
    }

    output_dir.mkdir(parents=False)
    if args.audit_only:
        result = common | {
            "status": "audit_passed",
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "model_state_unchanged": ppo.model_state_sha256(model) == model_hash_before,
            "ppo_optimizer_state_unchanged": audit.nested_sha256(ppo_state) == PARENT_PPO_SHA256,
            "replay_optimizer_state_unchanged": (
                audit.nested_sha256(replay_optimizer.state_dict()) == PARENT_REPLAY_SHA256
            ),
            "quota_state_unchanged": audit.nested_sha256(quota_state) == PARENT_QUOTA_SHA256,
            "parent_checkpoint_unchanged": audit.file_sha256(PARENT_PATH) == PARENT_SHA256,
        }
        required = (
            "model_state_unchanged",
            "ppo_optimizer_state_unchanged",
            "replay_optimizer_state_unchanged",
            "quota_state_unchanged",
            "parent_checkpoint_unchanged",
        )
        if not all(result[key] for key in required):
            raise RuntimeError("Audit-only mode mutated frozen state")
        write_json_exclusive(output_dir / "preflight_audit.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0

    train_config = copy.deepcopy(config)
    train_config.bc_replay_steps = 1
    train_config.bc_replay_lr_scale = G8_LR_SCALE
    per_step: list[dict[str, Any]] = []
    for step, batch_index in enumerate(G8_BATCH_INDICES, 1):
        metrics = ppo.bc_replay_update(
            model,
            replay_optimizer,
            [replay_batches[batch_index]],
            train_config,
            device,
            config.learning_rate,
        )
        if not isinstance(metrics, dict):
            raise RuntimeError("General-BC core returned no metrics")
        if metrics.get("steps") != 1 or int(metrics.get("rows", -1)) != 256:
            raise RuntimeError("General-BC core did not consume one full batch")
        if int(metrics.get("context_34_rows", -1)) != 4:
            raise RuntimeError("General-BC context-34 quota drifted")
        if metrics.get("learning_rate") != G8_LEARNING_RATE:
            raise RuntimeError("General-BC learning rate drifted")
        if not audit.finite_nested(metrics):
            raise FloatingPointError("General-BC metrics are non-finite")
        per_step.append(
            {
                "step": step,
                "batch_index": batch_index,
                "batch_sha256": batch_hashes[batch_index],
                "metrics": metrics,
            }
        )

    model_state_after = audit.clone_model_state(model)
    changed = audit.changed_tensor_names(model_state_before, model_state_after)
    if changed != sorted(actor_names):
        raise RuntimeError("G8 did not change exactly the frozen actor24 tensor set")
    if not audit.finite_nested(model_state_after):
        raise FloatingPointError("G8 model contains non-finite tensors")
    if not all(
        torch.equal(model_state_before[name], model_state_after[name])
        for name in value_names
    ):
        raise RuntimeError("G8 changed value-head tensors")

    displacement = actor_l2(model_state_before, model_state_after, actor_names)
    if not G8_DISPLACEMENT_MIN <= displacement <= G8_DISPLACEMENT_MAX:
        raise RuntimeError(
            f"G8 actor displacement {displacement} is outside the frozen interval"
        )
    kl_after = general_cache_reference_kl(
        model, reference_model, replay_batches, device, config.policy_temperature
    )
    if not kl_after < kl_before:
        raise RuntimeError(
            f"G8 did not lower general-cache reference KL: {kl_before} -> {kl_after}"
        )

    replay_after = replay_optimizer.state_dict()
    if len(replay_after["state"]) != 24 or state_steps(replay_after) != {24}:
        raise RuntimeError("G8 replay optimizer did not advance 24 states to step 24")
    if audit.nested_sha256(ppo_state) != PARENT_PPO_SHA256:
        raise RuntimeError("G8 changed the PPO optimizer state")
    if audit.nested_sha256(quota_state) != PARENT_QUOTA_SHA256:
        raise RuntimeError("G8 changed the opponent quota state")
    if audit.file_sha256(PARENT_PATH) != PARENT_SHA256:
        raise RuntimeError("G8 changed the frozen U464 parent")

    model_hash_after = ppo.model_state_sha256(model)
    provenance = common | {
        "status": "general_bc_completed",
        "per_step": per_step,
        "integrity": {
            "optimizer_steps": G8_STEPS,
            "rows": G8_STEPS * 256,
            "context34_rows": G8_STEPS * 4,
            "changed_parameter_names": changed,
            "changed_parameters_exact_actor24": True,
            "value_head_parameters_unchanged": True,
            "model_state_sha256_after": model_hash_after,
            "actor24_float64_l2_displacement": displacement,
            "general_cache_reference_kl_after": kl_after,
            "general_cache_reference_kl_strictly_lower": True,
            "replay_state_sha256_after": audit.nested_sha256(replay_after),
            "replay_state_count": len(replay_after["state"]),
            "replay_step_after": 24,
            "ppo_optimizer_state_unchanged": True,
            "quota_state_unchanged": True,
            "parent_checkpoint_unchanged": True,
            "all_metrics_model_and_optimizer_tensors_finite": (
                audit.finite_nested(replay_after) and audit.finite_nested(ppo_state)
            ),
        },
        "checkpoint_update_label": PARENT_UPDATE,
        "not_a_new_ppo_update": True,
    }
    if not provenance["integrity"]["all_metrics_model_and_optimizer_tensors_finite"]:
        raise FloatingPointError("G8 optimizer state contains non-finite tensors")

    payload = copy.deepcopy(parent)
    payload["model_state_dict"] = model_state_after
    payload["bc_replay_optimizer_state_dict"] = replay_after
    payload["post_ppo_general_bc"] = provenance
    checkpoint_path = output_dir / "general-bc-0008.pt"
    torch.save(payload, checkpoint_path)
    result = provenance | {
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": audit.file_sha256(checkpoint_path),
            "update": PARENT_UPDATE,
        },
        "checkpoint_writes": 1,
    }
    write_json_exclusive(output_dir / "general_bc_manifest.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
