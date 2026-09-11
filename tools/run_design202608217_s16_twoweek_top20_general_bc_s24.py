#!/usr/bin/env python3
"""Continue S16 with complementary latest-two-week general-BC batches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
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
PARENT = ROOT / "artifacts/design202608195_alpha075_latestreplay_general_bc_s24/general-bc-s16.pt"
PARENT_SHA256 = "4bf6e80448e76965853580b92d0524e3afff0e685d629fd3bfce8aad4328acb8"
PARENT_MANIFEST = ROOT / "artifacts/design202608195_alpha075_latestreplay_general_bc_s24/general_bc_manifest.json"
PARENT_MANIFEST_SHA256 = "0ee5549739f900a293fd6902f3ee593e4afebba6fd8bd178bd59811f736becc5"
REPLAY = ROOT / "data/bc_marnie_top20_current14_latesttop50_trainvalid_through0804_design202608194.zip"
REPLAY_SHA256 = "d17ac0739b09193f0e7ae66107e02352599c2c75fbb7ca3deda3e22f6ff47152"
BC = ROOT / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/best.pt"
BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
TRAINER = TOOLS / "train_ppo.py"
TRAINER_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
HELPER = TOOLS / "run_ppo_bc_repair.py"
HELPER_SHA256 = "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"
SEED = 202608205
BATCH_INDICES = (2, 39, 17, 32, 13, 44, 40, 25, 38, 71, 45, 51, 23, 12, 6, 60, 33, 7, 29, 64, 8, 16, 4, 37)
PARENT_CONSUMED = (26, 5, 55, 22, 30, 20, 65, 19, 9, 57, 61, 35, 0, 56, 31, 53)
ENDPOINTS = (8, 16, 24)
TRAIN_BATCHES = 72
VALID_BATCHES = 16
BATCH_SIZE = 256
ACTOR_LR = 3.2e-5
LR_SCALE = 0.025
LEARNING_RATE = ACTOR_LR * LR_SCALE
OUTPUT_ROOT = ROOT / "artifacts/design202608217_s16_twoweek_top20_general_bc_s24"
PREFLIGHT_ROOT = ROOT / "artifacts/design202608217_s16_twoweek_top20_general_bc_s24_preflight"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(path: Path, expected: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or file_sha(path) != expected:
        raise RuntimeError(f"frozen input mismatch: {path}")


def publish(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload); os.fsync(fd)
    finally:
        os.close(fd)


@torch.inference_mode()
def cache_loss(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], config: ppo.PPOConfig, device: torch.device) -> dict[str, float]:
    model.eval(); total = 0.0; rows = 0; parts_total: dict[str, float] = {}
    for cpu_batch in batches:
        batch = {k: v.to(device) for k, v in cpu_batch.items()}
        outputs = ppo.model_forward(model, batch, device)
        loss, parts = ppo.bc_expert_actor_loss(
            outputs, batch, loss_mode=config.bc_replay_loss,
            order_context_weight=config.bc_replay_order_context_weight,
            non_context34_fixed_multi_action_order_weight=config.bc_replay_non_context34_fixed_multi_action_order_weight,
        )
        count = int(batch["action_counts"].shape[0]); total += float(loss) * count; rows += count
        for key, value in parts.items():
            parts_total[key] = parts_total.get(key, 0.0) + float(value) * count
    result = {"loss": total / rows, "rows": rows} | {k: v / rows for k, v in parts_total.items()}
    if not audit.finite_nested(result):
        raise FloatingPointError("non-finite cache metric")
    return result


def actor_l2(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], names: list[str]) -> float:
    return math.sqrt(sum(float((after[n].double() - before[n].double()).square().sum()) for n in names))


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    buffer = io.BytesIO(); torch.save(payload, buffer); raw = buffer.getvalue(); publish(path, raw)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    tool = Path(__file__).resolve()
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires repository cwd and my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    for path, expected in (
        (tool, args.expected_tool_sha256), (PARENT, PARENT_SHA256),
        (PARENT_MANIFEST, PARENT_MANIFEST_SHA256), (REPLAY, REPLAY_SHA256),
        (BC, BC_SHA256), (TRAINER, TRAINER_SHA256), (HELPER, HELPER_SHA256),
    ):
        require(path, expected)
    target = PREFLIGHT_ROOT if args.audit_only else OUTPUT_ROOT
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if len(BATCH_INDICES) != 24 or len(set(BATCH_INDICES)) != 24 or set(BATCH_INDICES) & set(PARENT_CONSUMED):
        raise RuntimeError("complementary batch contract failed")
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(BC, map_location="cpu", weights_only=False)
    if parent.get("evaluation_only") is not True or parent.get("resume_forbidden") is not True:
        raise RuntimeError("S16 parent contract failed")
    omitted = parent.get("optimizer_states_omitted")
    forbidden_state = ("optimizer_state_dict", "bc_replay_optimizer_state_dict", "opponent_quota_state")
    if omitted is not True or any(key in parent for key in forbidden_state):
        raise RuntimeError("S16 omitted-state contract failed")
    config = ppo.PPOConfig(**parent["config"])
    config.bc_replay_data = str(REPLAY); config.bc_replay_split = "train"
    config.bc_replay_batches = TRAIN_BATCHES; config.bc_replay_batch_size = BATCH_SIZE
    config.bc_replay_workers = 8; config.bc_replay_steps = 1; config.bc_replay_lr_scale = LR_SCALE
    config.bc_replay_loss = "ordered"; config.bc_replay_order_context_weight = 8.0
    config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    config.bc_replay_context34_rows_per_batch = 4; config.seed = SEED
    valid_config = copy.deepcopy(config); valid_config.bc_replay_split = "valid"
    valid_config.bc_replay_batches = VALID_BATCHES; valid_config.seed = SEED + 1
    model = ppo.instantiate_model_from_checkpoint(parent, bc_checkpoint, device)
    actor_parameters, _, trainable = ppo.configure_trainable_scope(model, "last_block_heads")
    actor_names = list(trainable["actor_parameter_names"]); value_names = list(trainable["value_parameter_names"])
    before = audit.clone_model_state(model)
    train_batches = ppo.build_bc_replay_batches(config, parent["model_config"])
    valid_batches = ppo.build_bc_replay_batches(valid_config, parent["model_config"])
    train_cache_sha, train_batch_shas = audit.replay_cache_manifest(train_batches)
    valid_cache_sha, _ = audit.replay_cache_manifest(valid_batches)
    if len(train_batches) != TRAIN_BATCHES or len(valid_batches) != VALID_BATCHES:
        raise RuntimeError("cache count drift")
    if set(int((b["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum()) for b in train_batches + valid_batches) != {4}:
        raise RuntimeError("context34 cache drift")
    train_before = cache_loss(model, train_batches, config, device)
    valid_before = cache_loss(model, valid_batches, valid_config, device)
    common = {
        "schema_version": "ptcg-design202608217-s16-twoweek-top20-general-bc-s24-v1",
        "mode": "audit_only" if args.audit_only else "general_bc",
        "parent": {"path": str(PARENT.relative_to(ROOT)), "sha256": PARENT_SHA256, "prior_general_bc_steps": 16},
        "replay": {"path": str(REPLAY.relative_to(ROOT)), "sha256": REPLAY_SHA256, "dates": ["2026-07-22", "2026-08-04"], "days": 14},
        "protocol": {"seed": SEED, "steps": 24, "endpoints": list(ENDPOINTS), "batch_indices": list(BATCH_INDICES), "excluded_parent_batch_indices": list(PARENT_CONSUMED), "learning_rate": LEARNING_RATE, "trainable_scope": "last_block_heads", "optimizer": "fresh_adamw"},
        "caches": {"train": {"sha256": train_cache_sha, "rows": TRAIN_BATCHES * BATCH_SIZE, "loss_before": train_before}, "valid": {"sha256": valid_cache_sha, "rows": VALID_BATCHES * BATCH_SIZE, "loss_before": valid_before, "used_for_optimizer_steps": False}},
        "scope": {"local_only": True, "package": False, "upload": False, "submission": False},
    }
    os.mkdir(target, mode=0o700)
    if args.audit_only:
        result = common | {"status": "audit_passed", "optimizer_steps": 0, "checkpoint_writes": 0, "model_state_unchanged": audit.changed_tensor_names(before, audit.clone_model_state(model)) == [], "validation_rows_used_for_training": 0}
        publish(target / "preflight_audit.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print(json.dumps(result, sort_keys=True)); return 0
    optimizer = torch.optim.AdamW(actor_parameters, lr=LEARNING_RATE, eps=1e-5, weight_decay=config.weight_decay)
    per_step = []; endpoints = {}
    for step, batch_index in enumerate(BATCH_INDICES, 1):
        metrics = ppo.bc_replay_update(model, optimizer, [train_batches[batch_index]], config, device, ACTOR_LR)
        if metrics.get("rows") != BATCH_SIZE or metrics.get("steps") != 1:
            raise RuntimeError("BC step contract failed")
        per_step.append({"step": step, "batch_index": batch_index, "batch_sha256": train_batch_shas[batch_index], "metrics": metrics})
        if step not in ENDPOINTS:
            continue
        state = audit.clone_model_state(model); changed = audit.changed_tensor_names(before, state)
        if changed != sorted(actor_names) or any(not torch.equal(before[n], state[n]) for n in value_names):
            raise RuntimeError(f"S{step} scope integrity failed")
        train_loss = cache_loss(model, train_batches, config, device)
        valid_loss = cache_loss(model, valid_batches, valid_config, device)
        if train_loss["loss"] >= train_before["loss"] or not audit.finite_nested(state):
            raise RuntimeError(f"S{step} metric integrity failed")
        payload = {k: copy.deepcopy(parent[k]) for k in ("feature_version", "bc_feature_version", "config", "model_config", "learner_deck_hash", "reward", "action_distribution")}
        payload.update({"model_state_dict": state, "update": 476, "evaluation_only": True, "resume_forbidden": True, "optimizer_states_omitted": ["optimizer_state_dict", "bc_replay_optimizer_state_dict", "opponent_quota_state"], "post_general_bc": {"schema_version": common["schema_version"], "steps_from_s16": step, "total_general_bc_steps": 16 + step, "fresh_optimizer": True, "validation_rows_used_for_training": 0}})
        checkpoint = OUTPUT_ROOT / f"general-bc-from-s16-s{step:02d}.pt"
        checkpoint_sha = save_checkpoint(checkpoint, payload)
        endpoints[str(step)] = {"checkpoint": str(checkpoint.relative_to(ROOT)), "sha256": checkpoint_sha, "actor_l2_from_s16": actor_l2(before, state, actor_names), "train_loss": train_loss, "valid_loss": valid_loss, "changed_parameter_names": changed, "value_head_unchanged": True, "all_finite": True}
    final = audit.clone_model_state(model)
    result = common | {"status": "general_bc_completed", "per_step": per_step, "endpoints": endpoints, "integrity": {"changed_exactly_actor24": audit.changed_tensor_names(before, final) == sorted(actor_names), "value_head_unchanged": all(torch.equal(before[n], final[n]) for n in value_names), "parent_unchanged": file_sha(PARENT) == PARENT_SHA256, "replay_unchanged": file_sha(REPLAY) == REPLAY_SHA256, "validation_rows_used_for_training": 0, "checkpoint_writes": len(endpoints)}}
    if not (
        result["integrity"]["changed_exactly_actor24"]
        and result["integrity"]["value_head_unchanged"]
        and result["integrity"]["parent_unchanged"]
        and result["integrity"]["replay_unchanged"]
        and result["integrity"]["validation_rows_used_for_training"] == 0
        and result["integrity"]["checkpoint_writes"] == len(ENDPOINTS)
    ):
        raise RuntimeError("terminal integrity failed")
    publish(OUTPUT_ROOT / "general_bc_manifest.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
