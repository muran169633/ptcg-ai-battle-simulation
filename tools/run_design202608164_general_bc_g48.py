#!/usr/bin/env python3
"""Run a frozen 48-batch updated-replay general-BC stage on raw U468."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import stat
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import torch  # noqa: E402
import run_ppo_bc_repair as audit  # noqa: E402
import train_ppo as ppo  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
PARENT = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
PARENT_SHA256 = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
BC = ROOT / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/best.pt"
BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
REPLAY = ROOT / "data/bc_marnie_top50_current14_timeforward_train0802_valid0803_design202608147.zip"
REPLAY_SHA256 = "bf01cfc7e9c0f616f157ae36ed68c76619ae45448cdc17562befe889a4d5ae93"
TRAINER = TOOLS / "train_ppo.py"
TRAINER_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
HELPER = TOOLS / "run_ppo_bc_repair.py"
HELPER_SHA256 = "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"
SEED = 202608164
STEPS = 48
BATCH_INDICES = (
    27, 29, 56, 8, 41, 69, 23, 11, 17, 13, 24, 68,
    51, 45, 32, 54, 67, 61, 62, 1, 18, 39, 57, 9,
    63, 14, 31, 59, 25, 30, 53, 35, 20, 15, 46, 4,
    21, 48, 44, 70, 5, 65, 16, 55, 19, 7, 49, 12,
)
LR_SCALE = 0.075
ACTOR_LR = 3.6e-5
LEARNING_RATE = ACTOR_LR * LR_SCALE
OUTPUT_ROOT = ROOT / "artifacts/design202608164_general_bc_g48"
CHECKPOINT = OUTPUT_ROOT / "general-bc-0048.pt"
MANIFEST = OUTPUT_ROOT / "general_bc_manifest.json"
PREFLIGHT_ROOT = ROOT / "artifacts/design202608164_general_bc_g48_preflight"
PREFLIGHT_MANIFEST = PREFLIGHT_ROOT / "preflight_audit.json"


def require(path: Path, expected: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"input is not a regular file: {path}")
    if audit.file_sha256(path) != expected:
        raise RuntimeError(f"input SHA-256 mismatch: {path}")


def write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def actor_l2(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], names: list[str]) -> float:
    return math.sqrt(sum(float((after[name].double() - before[name].double()).square().sum()) for name in names))


@torch.inference_mode()
def cache_loss(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], config: ppo.PPOConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    totals: dict[str, float] = {}
    for cpu_batch in batches:
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        outputs = ppo.model_forward(model, batch, device)
        loss, parts = ppo.bc_expert_actor_loss(
            outputs,
            batch,
            loss_mode=config.bc_replay_loss,
            order_context_weight=config.bc_replay_order_context_weight,
            non_context34_fixed_multi_action_order_weight=(
                config.bc_replay_non_context34_fixed_multi_action_order_weight
            ),
        )
        rows = int(batch["action_counts"].shape[0])
        total_loss += float(loss) * rows
        total_rows += rows
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value) * rows
    result = {"loss": total_loss / total_rows}
    result.update({key: value / total_rows for key, value in totals.items()})
    if not audit.finite_nested(result):
        raise FloatingPointError("non-finite cache loss")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tool = Path(__file__).resolve()
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires repository cwd and my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    require(tool, args.expected_tool_sha256)
    for path, digest in (
        (PARENT, PARENT_SHA256), (BC, BC_SHA256), (REPLAY, REPLAY_SHA256),
        (TRAINER, TRAINER_SHA256), (HELPER, HELPER_SHA256),
    ):
        require(path, digest)
    target_root = PREFLIGHT_ROOT if args.audit_only else OUTPUT_ROOT
    if target_root.exists() or target_root.is_symlink():
        raise FileExistsError(target_root)
    if len(BATCH_INDICES) != STEPS or len(set(BATCH_INDICES)) != STEPS or not all(0 <= value < 72 for value in BATCH_INDICES):
        raise RuntimeError("frozen batch order is invalid")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(BC, map_location="cpu", weights_only=False)
    if parent.get("update") != 468 or parent.get("resume_forbidden") is True:
        raise RuntimeError("raw U468 parent contract failed")
    config = ppo.PPOConfig(**parent["config"])
    config.bc_replay_data = str(REPLAY)
    config.bc_replay_split = "train"
    config.bc_replay_batches = 72
    config.bc_replay_batch_size = 256
    config.bc_replay_workers = 8
    config.bc_replay_steps = 1
    config.bc_replay_lr_scale = LR_SCALE
    config.bc_replay_loss = "ordered"
    config.bc_replay_order_context_weight = 8.0
    config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    config.bc_replay_context34_rows_per_batch = 4
    config.seed = SEED

    model = ppo.instantiate_model_from_checkpoint(parent, bc_checkpoint, device)
    actor_parameters, _, trainable = ppo.configure_trainable_scope(model, "last_block_heads")
    actor_names = list(trainable["actor_parameter_names"])
    value_names = list(trainable["value_parameter_names"])
    before = audit.clone_model_state(model)
    batches = ppo.build_bc_replay_batches(config, parent["model_config"])
    cache_hash, batch_hashes = audit.replay_cache_manifest(batches)
    if len(batches) != 72 or any(int(batch["action_counts"].shape[0]) != 256 for batch in batches):
        raise RuntimeError("general replay cache shape drifted")
    if set(int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum()) for batch in batches) != {4}:
        raise RuntimeError("context-34 stratification drifted")
    loss_before = cache_loss(model, batches, config, device)
    common: dict[str, Any] = {
        "schema_version": "ptcg-design202608164-general-bc-g48-v1",
        "mode": "audit_only" if args.audit_only else "general_bc",
        "parent": {"path": str(PARENT), "sha256": PARENT_SHA256, "update": 468},
        "sources": {
            "bc": {"path": str(BC), "sha256": BC_SHA256},
            "replay": {"path": str(REPLAY), "sha256": REPLAY_SHA256},
            "trainer": {"path": str(TRAINER), "sha256": TRAINER_SHA256},
            "helper": {"path": str(HELPER), "sha256": HELPER_SHA256},
            "tool": {"path": str(tool), "sha256": args.expected_tool_sha256},
        },
        "protocol": {
            "seed": SEED, "steps": STEPS, "rows": STEPS * 256,
            "batch_indices": list(BATCH_INDICES), "learning_rate": LEARNING_RATE,
            "trainable_scope": "last_block_heads", "optimizer": "fresh_adamw",
        },
        "cache": {"sha256": cache_hash, "batches": 72, "rows": 72 * 256, "loss_before": loss_before},
    }
    os.mkdir(target_root, mode=0o700)
    if args.audit_only:
        result = common | {
            "status": "audit_passed", "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "model_state_unchanged": ppo.model_state_sha256(model) == ppo.model_state_sha256(
                ppo.instantiate_model_from_checkpoint(parent, bc_checkpoint, device)
            ),
        }
        write_exclusive(PREFLIGHT_MANIFEST, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print(json.dumps(result, sort_keys=True))
        return 0

    optimizer = torch.optim.AdamW(actor_parameters, lr=LEARNING_RATE, eps=1e-5, weight_decay=config.weight_decay)
    per_step: list[dict[str, Any]] = []
    for step, batch_index in enumerate(BATCH_INDICES, 1):
        metrics = ppo.bc_replay_update(model, optimizer, [batches[batch_index]], config, device, ACTOR_LR)
        if not isinstance(metrics, dict) or metrics.get("rows") != 256 or metrics.get("steps") != 1:
            raise RuntimeError("general-BC update contract failed")
        per_step.append({"step": step, "batch_index": batch_index, "batch_sha256": batch_hashes[batch_index], "metrics": metrics})
    after = audit.clone_model_state(model)
    changed = audit.changed_tensor_names(before, after)
    if changed != sorted(actor_names) or any(not torch.equal(before[name], after[name]) for name in value_names):
        raise RuntimeError("general BC changed unexpected tensors")
    loss_after = cache_loss(model, batches, config, device)
    if not loss_after["loss"] < loss_before["loss"]:
        raise RuntimeError("general cache loss did not improve")
    payload = copy.deepcopy(parent)
    payload["model_state_dict"] = after
    payload["config"] = copy.deepcopy(config.__dict__)
    payload["bc_replay_optimizer_state_dict"] = optimizer.state_dict()
    payload["post_ppo_general_bc"] = {
        "schema_version": common["schema_version"],
        "steps": STEPS, "rows": STEPS * 256, "cache_sha256": cache_hash,
        "fresh_optimizer": True, "optimizer_reset_required_before_ppo": True,
    }
    torch.save(payload, CHECKPOINT)
    result = common | {
        "status": "general_bc_completed",
        "per_step": per_step,
        "integrity": {
            "changed_parameter_names": changed,
            "changed_exactly_actor24": True,
            "value_head_unchanged": True,
            "actor24_l2_displacement": actor_l2(before, after, actor_names),
            "model_state_sha256_after": ppo.model_state_sha256(model),
            "loss_after": loss_after,
            "loss_strictly_lower": True,
            "parent_unchanged": audit.file_sha256(PARENT) == PARENT_SHA256,
            "all_finite": audit.finite_nested(after) and audit.finite_nested(optimizer.state_dict()),
        },
        "checkpoint": {"path": str(CHECKPOINT), "sha256": audit.file_sha256(CHECKPOINT), "update": 468},
        "checkpoint_writes": 1,
        "not_a_new_ppo_update": True,
    }
    if not result["integrity"]["all_finite"] or not result["integrity"]["parent_unchanged"]:
        raise RuntimeError("terminal integrity check failed")
    write_exclusive(MANIFEST, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
