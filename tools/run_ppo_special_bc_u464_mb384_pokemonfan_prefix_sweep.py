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
DESIGN_ID = 202608050
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
PUBLISHED_ENDPOINT_STEPS = (16, 20, 24, 28)
EXPECTED_P32_MODEL_SHA256 = (
    "fae8224ecd3e31598cb440de77fc5dd282630a92ca462399a624dd720b7b0748"
)
EXPECTED_P32_REPLAY_SHA256 = (
    "ee4123ed2c9cdf423d193a448fda730b72f2154f16d04787f9093b0caf1b0fa1"
)
EXPECTED_DESIGN_ROOT = REPO_ROOT / (
    "artifacts/ppo_u464mb384_pokemonfan_prefix_sweep_"
    "p16_p20_p24_p28_design202608050"
)
EXPECTED_FORMAL_OUTPUT_DIR = EXPECTED_DESIGN_ROOT / "sweep_stage"
EXPECTED_DESIGN_PREREGISTRATION = REPO_ROOT / (
    "artifacts/ppo_u464mb384_pokemonfan_prefix_sweep_"
    "p16_p20_p24_p28_design202608050.design_preregistration_v2.json"
)
EXPECTED_DESIGN_PREREGISTRATION_SHA256 = (
    "d43b3cbfe71a974aad87cffb857ca4578d864948cb6d37da278da7a10924d7db"
)
EXPECTED_P32_REFERENCE_CHECKPOINT = REPO_ROOT / (
    "artifacts/ppo_u464mb384_generalbc_ppo_specialbc_"
    "pokemonfan32_design202608040/special_stage/special-bc-0032.pt"
)
EXPECTED_P32_REFERENCE_CHECKPOINT_SHA256 = (
    "88d62f08a42a20b146641a53aa315b7d171cbdb051bcbd43271d6a5b154ddb17"
)
EXPECTED_P32_REFERENCE_MANIFEST = REPO_ROOT / (
    "artifacts/ppo_u464mb384_generalbc_ppo_specialbc_"
    "pokemonfan32_design202608040/special_stage/special_bc_manifest.json"
)
EXPECTED_P32_REFERENCE_MANIFEST_SHA256 = (
    "5af84e468bc3a92f84639747a32e2b70b795143182fd8322c27d42b41372970a"
)
EXPECTED_P32_REFERENCE_EXECUTOR = (
    REPO_ROOT / "tools/run_ppo_special_bc_u464_mb384_pokemonfan32.py"
)
EXPECTED_P32_REFERENCE_EXECUTOR_SHA256 = (
    "7b4fa3380053f5d3a0b4e1fbc2aa7782fc24736eac26cb9b9e74734f259082a5"
)
REPAIR_AUDIT_PATH = REPO_ROOT / "tools/run_ppo_bc_repair.py"
EXPECTED_REPAIR_AUDIT_SHA256 = (
    "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"
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


def validate_reference_p32() -> dict[str, Any]:
    payload = json.loads(EXPECTED_P32_REFERENCE_MANIFEST.read_text("utf-8"))
    integrity = payload.get("integrity")
    checkpoint = payload.get("checkpoint")
    sources = payload.get("sources")
    if not all(isinstance(item, dict) for item in (integrity, checkpoint, sources)):
        raise ValueError("Frozen P32 reference manifest is incomplete")
    prefix = integrity.get("frozen_16_step_prefix")
    tool = sources.get("tool")
    if not isinstance(prefix, dict) or not isinstance(tool, dict):
        raise ValueError("Frozen P32 reference provenance is incomplete")
    if (
        integrity.get("optimizer_steps") != SPECIAL_REPAIR_STEPS
        or integrity.get("model_state_sha256_before")
        != EXPECTED_PARENT_MODEL_SHA256
        or integrity.get("model_state_sha256_after")
        != EXPECTED_P32_MODEL_SHA256
        or integrity.get("replay_state_sha256_after")
        != EXPECTED_P32_REPLAY_SHA256
        or integrity.get("ppo_state_sha256_after")
        != EXPECTED_PARENT_PPO_SHA256
        or set(integrity.get("replay_steps_after", [])) != {48}
        or integrity.get("value_head_parameters_unchanged") is not True
        or integrity.get("ppo_optimizer_state_unchanged") is not True
        or integrity.get("parent_checkpoint_unchanged") is not True
        or prefix.get("steps") != PREFIX_STEPS
        or prefix.get("model_state_sha256") != EXPECTED_PREFIX_MODEL_SHA256
        or prefix.get("replay_state_sha256") != EXPECTED_PREFIX_REPLAY_SHA256
        or prefix.get("exact_match") is not True
        or checkpoint.get("sha256")
        != EXPECTED_P32_REFERENCE_CHECKPOINT_SHA256
        or tool.get("sha256") != EXPECTED_P32_REFERENCE_EXECUTOR_SHA256
    ):
        raise ValueError("Frozen P32 reference hashes or integrity drifted")
    return {
        "validated": True,
        "checkpoint_sha256": EXPECTED_P32_REFERENCE_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_P32_REFERENCE_MANIFEST_SHA256,
        "executor_sha256": EXPECTED_P32_REFERENCE_EXECUTOR_SHA256,
        "p16_model_state_sha256": EXPECTED_PREFIX_MODEL_SHA256,
        "p16_replay_state_sha256": EXPECTED_PREFIX_REPLAY_SHA256,
        "p32_model_state_sha256": EXPECTED_P32_MODEL_SHA256,
        "p32_replay_state_sha256": EXPECTED_P32_REPLAY_SHA256,
    }


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
            "Reconstruct the exact U464 PokemonFan P32 trajectory and publish "
            "only P16/P20/P24/P28 prefix checkpoints."
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
    if output_dir.is_symlink():
        raise ValueError("Output directory may not be a symlink")
    if not args.audit_only and output_dir != EXPECTED_FORMAL_OUTPUT_DIR.resolve():
        raise ValueError("Formal sweep output must use the frozen design root")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse existing output directory: {output_dir}"
        )

    bound_files: dict[str, str] = {}
    tool_hash = validate_sha(tool_path, args.expected_tool_sha256, "tool")
    bound_files[str(tool_path)] = tool_hash
    train_hash = validate_sha(
        train_path,
        args.expected_train_ppo_sha256,
        "train_ppo",
    )
    bound_files[str(train_path)] = train_hash
    parent_hash_before = validate_sha(
        parent_path,
        args.expected_parent_sha256,
        "parent checkpoint",
    )
    bound_files[str(parent_path)] = parent_hash_before
    special_data_hash = validate_sha(
        special_data_path,
        args.expected_special_data_sha256,
        "special BC archive",
    )
    bound_files[str(special_data_path)] = special_data_hash
    for path, expected, label in (
        (REPAIR_AUDIT_PATH, EXPECTED_REPAIR_AUDIT_SHA256, "repair audit"),
        (
            EXPECTED_P32_REFERENCE_EXECUTOR,
            EXPECTED_P32_REFERENCE_EXECUTOR_SHA256,
            "P32 reference executor",
        ),
        (
            EXPECTED_P32_REFERENCE_CHECKPOINT,
            EXPECTED_P32_REFERENCE_CHECKPOINT_SHA256,
            "P32 reference checkpoint",
        ),
        (
            EXPECTED_P32_REFERENCE_MANIFEST,
            EXPECTED_P32_REFERENCE_MANIFEST_SHA256,
            "P32 reference manifest",
        ),
        (
            EXPECTED_DESIGN_PREREGISTRATION,
            EXPECTED_DESIGN_PREREGISTRATION_SHA256,
            "authoritative sweep design v2",
        ),
    ):
        bound_files[str(path.resolve())] = validate_sha(path, expected, label)
    p32_reference = validate_reference_p32()

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
        "schema_version": "ptcg-u464-mb384-pokemonfan-prefix-sweep-v1",
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
            "published_endpoint_steps": list(PUBLISHED_ENDPOINT_STEPS),
            "p32_model_state_sha256": EXPECTED_P32_MODEL_SHA256,
            "p32_replay_state_sha256": EXPECTED_P32_REPLAY_SHA256,
            "p32_executed_only_as_in_memory_sentinel": True,
            "p32_checkpoint_publish_forbidden": True,
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
            "repair_audit": {
                "path": str(REPAIR_AUDIT_PATH.resolve()),
                "sha256": EXPECTED_REPAIR_AUDIT_SHA256,
            },
            "authoritative_design_v2": {
                "path": str(EXPECTED_DESIGN_PREREGISTRATION.resolve()),
                "sha256": EXPECTED_DESIGN_PREREGISTRATION_SHA256,
            },
            "general_bc_checkpoint": {
                "path": str(bc_checkpoint_path),
                "sha256": bc_checkpoint_hash,
            },
            "special_bc_archive": {
                "path": str(special_data_path),
                "sha256": special_data_hash,
                "split": "train",
            },
            "p32_reference": {
                "checkpoint": str(EXPECTED_P32_REFERENCE_CHECKPOINT.resolve()),
                "checkpoint_sha256": (
                    EXPECTED_P32_REFERENCE_CHECKPOINT_SHA256
                ),
                "manifest": str(EXPECTED_P32_REFERENCE_MANIFEST.resolve()),
                "manifest_sha256": EXPECTED_P32_REFERENCE_MANIFEST_SHA256,
                "executor": str(EXPECTED_P32_REFERENCE_EXECUTOR.resolve()),
                "executor_sha256": EXPECTED_P32_REFERENCE_EXECUTOR_SHA256,
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
        "p32_reference_validation": p32_reference,
        "determinism": {
            "device": str(device),
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "python_seed": args.special_seed,
            "torch_seed": args.special_seed,
        },
        "device_preflight": device_preflight,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
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
            "all_frozen_files_unchanged": all_files_unchanged(bound_files),
        }
        if not all(
            result[key]
            for key in (
                "model_state_unchanged",
                "replay_optimizer_state_unchanged",
                "ppo_optimizer_state_unchanged",
                "parent_checkpoint_unchanged",
                "all_frozen_files_unchanged",
            )
        ):
            raise RuntimeError("Audit-only mode mutated frozen state")
        write_json(output_dir / "preflight_audit.json", result)
        if sorted(path.name for path in output_dir.iterdir()) != [
            "preflight_audit.json"
        ]:
            raise RuntimeError("Audit-only output contains an unexpected file")
        print(json.dumps(result, sort_keys=True))
        return

    per_step: list[dict[str, Any]] = []
    endpoint_snapshots: list[dict[str, Any]] = []
    p16_integrity: dict[str, Any] | None = None
    for special_step, batch_index in enumerate(args.batch_indices, start=1):
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
            raise FloatingPointError("Non-finite special-BC metric")
        per_step.append(
            {
                "special_step": special_step,
                "batch_index": batch_index,
                "batch_sha256": batch_hashes[batch_index],
                "metrics": metrics,
            }
        )

        if special_step in PUBLISHED_ENDPOINT_STEPS:
            endpoint_model_state = repair_audit.clone_model_state(model)
            endpoint_replay_state = clone_nested_to_cpu(
                replay_optimizer.state_dict()
            )
            endpoint_model_hash = ppo.model_state_sha256(model)
            endpoint_replay_hash = repair_audit.nested_sha256(
                endpoint_replay_state
            )
            endpoint_changed_names = repair_audit.changed_tensor_names(
                model_state_before,
                endpoint_model_state,
            )
            endpoint_replay_steps = repair_audit.optimizer_steps(
                endpoint_replay_state
            )
            expected_replay_step = EXPECTED_REPLAY_STATE_STEP + special_step
            if (
                len(endpoint_changed_names) != len(EXPECTED_ACTOR_PARAMETER_NAMES)
                or set(endpoint_changed_names)
                != set(EXPECTED_ACTOR_PARAMETER_NAMES)
                or len(endpoint_replay_steps)
                != len(EXPECTED_ACTOR_PARAMETER_NAMES)
                or set(endpoint_replay_steps) != {expected_replay_step}
                or not repair_audit.finite_nested(endpoint_model_state)
                or not repair_audit.finite_nested(endpoint_replay_state)
                or any(
                    not torch.equal(
                        model_state_before[name],
                        endpoint_model_state[name],
                    )
                    for name in value_names
                )
            ):
                raise RuntimeError(f"P{special_step} endpoint integrity failed")
            if special_step == PREFIX_STEPS:
                p16_integrity = {
                    "steps": PREFIX_STEPS,
                    "model_state_sha256": endpoint_model_hash,
                    "replay_state_sha256": endpoint_replay_hash,
                    "expected_model_state_sha256": (
                        EXPECTED_PREFIX_MODEL_SHA256
                    ),
                    "expected_replay_state_sha256": (
                        EXPECTED_PREFIX_REPLAY_SHA256
                    ),
                    "exact_match": (
                        endpoint_model_hash == EXPECTED_PREFIX_MODEL_SHA256
                        and endpoint_replay_hash
                        == EXPECTED_PREFIX_REPLAY_SHA256
                    ),
                }
                if not p16_integrity["exact_match"]:
                    raise RuntimeError(
                        "P16 did not reproduce the frozen U464 witness"
                    )
            endpoint_snapshots.append(
                {
                    "special_steps": special_step,
                    "model_state_dict": endpoint_model_state,
                    "replay_state_dict": endpoint_replay_state,
                    "model_state_sha256": endpoint_model_hash,
                    "replay_state_sha256": endpoint_replay_hash,
                    "replay_steps_after": endpoint_replay_steps,
                    "changed_parameter_names": endpoint_changed_names,
                }
            )

    if p16_integrity is None:
        raise RuntimeError("The frozen P16 witness was not reached")
    if tuple(
        int(snapshot["special_steps"]) for snapshot in endpoint_snapshots
    ) != PUBLISHED_ENDPOINT_STEPS:
        raise RuntimeError("Required P16/P20/P24/P28 snapshots are incomplete")

    model_state_after = repair_audit.clone_model_state(model)
    model_hash_after = ppo.model_state_sha256(model)
    replay_state_after = clone_nested_to_cpu(replay_optimizer.state_dict())
    replay_hash_after = repair_audit.nested_sha256(replay_state_after)
    replay_steps_after = repair_audit.optimizer_steps(replay_state_after)
    changed_names = repair_audit.changed_tensor_names(
        model_state_before,
        model_state_after,
    )
    if (
        not repair_audit.finite_nested(model_state_after)
        or not repair_audit.finite_nested(replay_state_after)
        or len(changed_names) != len(EXPECTED_ACTOR_PARAMETER_NAMES)
        or set(changed_names) != set(EXPECTED_ACTOR_PARAMETER_NAMES)
        or len(replay_steps_after) != len(EXPECTED_ACTOR_PARAMETER_NAMES)
        or set(replay_steps_after)
        != {EXPECTED_REPLAY_STATE_STEP + SPECIAL_REPAIR_STEPS}
        or any(
            not torch.equal(model_state_before[name], model_state_after[name])
            for name in value_names
        )
    ):
        raise RuntimeError("Full in-memory P32 trajectory integrity failed")
    if (
        model_hash_after != EXPECTED_P32_MODEL_SHA256
        or replay_hash_after != EXPECTED_P32_REPLAY_SHA256
    ):
        raise RuntimeError(
            "P32 sentinel did not reproduce the frozen model/replay hashes"
        )
    if (
        repair_audit.nested_sha256(parent.get("optimizer_state_dict"))
        != ppo_optimizer_hash_before
    ):
        raise RuntimeError("PPO optimizer state changed during prefix sweep")
    parent_hash_after = repair_audit.file_sha256(parent_path)
    if parent_hash_after != parent_hash_before:
        raise RuntimeError("Parent checkpoint changed during prefix sweep")
    if not all_files_unchanged(bound_files):
        raise RuntimeError("A frozen source, dependency, or design file changed")

    trajectory_integrity = {
        "optimizer_steps": len(per_step),
        "rows": len(per_step) * SPECIAL_BATCH_SIZE,
        "context34_rows": (
            len(per_step) * SPECIAL_CONTEXT34_ROWS_PER_BATCH
        ),
        "all_metrics_and_model_tensors_finite": True,
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "replay_state_sha256_before": replay_optimizer_hash_before,
        "replay_state_sha256_after": replay_hash_after,
        "replay_steps_after": replay_steps_after,
        "changed_parameter_names": changed_names,
        "changed_exactly_24_actor_parameters": True,
        "value_head_parameters_unchanged": True,
        "ppo_state_sha256_after": ppo_optimizer_hash_before,
        "ppo_optimizer_state_unchanged": True,
        "parent_checkpoint_sha256_after": parent_hash_after,
        "parent_checkpoint_unchanged": True,
        "all_frozen_files_unchanged": True,
        "p16": p16_integrity,
        "p32_model_state_sha256_expected": EXPECTED_P32_MODEL_SHA256,
        "p32_replay_state_sha256_expected": EXPECTED_P32_REPLAY_SHA256,
        "p32_matches_frozen_reference": True,
        "p32_checkpoint_published": False,
    }

    endpoint_records: list[dict[str, Any]] = []
    for snapshot in endpoint_snapshots:
        special_steps = int(snapshot["special_steps"])
        endpoint_special = copy.deepcopy(common["special_bc"])
        endpoint_special.update(
            {
                "steps_requested": special_steps,
                "batch_indices": args.batch_indices[:special_steps],
                "batch_sha256": [
                    batch_hashes[index]
                    for index in args.batch_indices[:special_steps]
                ],
                "trajectory_reconstruction_seed": SPECIAL_SEED,
            }
        )
        endpoint_integrity = {
            "optimizer_steps": special_steps,
            "rows": special_steps * SPECIAL_BATCH_SIZE,
            "context34_rows": (
                special_steps * SPECIAL_CONTEXT34_ROWS_PER_BATCH
            ),
            "all_metrics_and_model_tensors_finite": True,
            "model_state_sha256_before": model_hash_before,
            "model_state_sha256_after": snapshot["model_state_sha256"],
            "model_state_changed": True,
            "changed_parameter_names": snapshot["changed_parameter_names"],
            "changed_exactly_24_actor_parameters": True,
            "value_head_parameters_unchanged": True,
            "replay_steps_after": snapshot["replay_steps_after"],
            "replay_state_sha256_after": snapshot["replay_state_sha256"],
            "ppo_state_sha256_after": ppo_optimizer_hash_before,
            "ppo_optimizer_state_unchanged": True,
            "parent_checkpoint_sha256_after": parent_hash_after,
            "parent_checkpoint_unchanged": True,
            "all_frozen_files_unchanged": True,
        }
        endpoint_provenance = common | {
            "status": "prefix_endpoint_completed",
            "special_bc": endpoint_special,
            "per_step": per_step[:special_steps],
            "integrity": endpoint_integrity,
            "trajectory_sentinel": {
                "full_steps_executed_in_memory": SPECIAL_REPAIR_STEPS,
                "p32_model_state_sha256": model_hash_after,
                "p32_replay_state_sha256": replay_hash_after,
                "exact_match": True,
                "p32_checkpoint_published": False,
            },
            "checkpoint_update_label": EXPECTED_PARENT_UPDATE,
            "not_a_new_ppo_update": True,
        }
        output_checkpoint = output_dir / (
            f"special-bc-pokemonfan-prefix-{special_steps:04d}.pt"
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
                "replay_adamw_step": (
                    EXPECTED_REPLAY_STATE_STEP + special_steps
                ),
            }
        )

    result = common | {
        "status": "prefix_sweep_completed",
        "per_step": per_step,
        "trajectory_integrity": trajectory_integrity,
        "endpoints": endpoint_records,
        "published_endpoint_count": len(endpoint_records),
        "checkpoint_writes": len(endpoint_records),
        "p32_checkpoint_writes": 0,
        "checkpoint_update_label": EXPECTED_PARENT_UPDATE,
        "not_a_new_ppo_update": True,
    }
    write_json(output_dir / "prefix_sweep_manifest.json", result)
    expected_files = [
        f"special-bc-pokemonfan-prefix-{step:04d}.pt"
        for step in PUBLISHED_ENDPOINT_STEPS
    ] + ["prefix_sweep_manifest.json"]
    if sorted(path.name for path in output_dir.iterdir()) != sorted(expected_files):
        raise RuntimeError("Prefix-sweep output contains unexpected files")
    if any("0032" in path.name for path in output_dir.iterdir()):
        raise RuntimeError("P32 publication is forbidden")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
