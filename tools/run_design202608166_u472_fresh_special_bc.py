#!/usr/bin/env python3
"""Train fresh actor-head-only PokemonFan special-BC S4/S8/S12 on U472."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import random
import stat
import sys
from datetime import datetime, timezone
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
PARENT = ROOT / "artifacts/design202608165_generalbc_g48_ppo6x192_u474/B_gold_league/seed-202608165/checkpoints/update-0472.pt"
PARENT_SHA256 = "75b4e79ab6b1ceb590c0517c3b0a2d9aa45c4b36b3b36e72bba0c6d8a6c89041"
BC = ROOT / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/best.pt"
BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
SPECIAL = ROOT / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip"
SPECIAL_SHA256 = "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598"
TRAINER = TOOLS / "train_ppo.py"
TRAINER_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
SEED = 202608166
BATCH_INDICES = (0, 1, 2, 7, 8, 9, 10, 14, 15, 16, 18, 19)
MILESTONES = (4, 8, 12)
ACTOR_LR = 4.8e-5
LR_SCALE = 0.025
LEARNING_RATE = ACTOR_LR * LR_SCALE
OUTPUT_ROOT = ROOT / "artifacts/design202608166_u472_fresh_special_bc"
MANIFEST = OUTPUT_ROOT / "special_bc_manifest.json"
PREFLIGHT_ROOT = ROOT / "artifacts/design202608166_u472_fresh_special_bc_preflight"
MUTABLE_NAMES = (
    "actor_query.weight", "actor_key.weight",
    "actor_residual.0.weight", "actor_residual.0.bias",
    "actor_residual.2.weight", "actor_residual.2.bias",
    "count_head.0.weight", "count_head.0.bias",
    "count_head.2.weight", "count_head.2.bias",
)
REQUIRED_SLIM_KEYS = (
    "feature_version", "bc_feature_version", "config", "model_config",
    "learner_deck_hash", "reward", "action_distribution",
)


def require(path: Path, expected: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or audit.file_sha256(path) != expected:
        raise RuntimeError(f"frozen input mismatch: {path}")


def write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload); os.fsync(fd)
    finally:
        os.close(fd)


@torch.inference_mode()
def cache_loss(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], config: ppo.PPOConfig, device: torch.device) -> float:
    total = 0.0; rows = 0
    for cpu_batch in batches:
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        outputs = ppo.model_forward(model, batch, device)
        loss, _ = ppo.bc_expert_actor_loss(
            outputs, batch, loss_mode=config.bc_replay_loss,
            order_context_weight=config.bc_replay_order_context_weight,
            non_context34_fixed_multi_action_order_weight=config.bc_replay_non_context34_fixed_multi_action_order_weight,
        )
        count = int(batch["action_counts"].shape[0]); total += float(loss) * count; rows += count
    value = total / rows
    if not __import__("math").isfinite(value):
        raise FloatingPointError("non-finite special cache loss")
    return value


def model_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape)).encode() + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes() + b"\0")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); tool = Path(__file__).resolve()
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve() or sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires repository cwd and my_project_env Python -I -B")
    require(tool, args.expected_tool_sha256)
    for path, digest in ((PARENT, PARENT_SHA256), (BC, BC_SHA256), (SPECIAL, SPECIAL_SHA256), (TRAINER, TRAINER_SHA256)):
        require(path, digest)
    target = PREFLIGHT_ROOT if args.audit_only else OUTPUT_ROOT
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(BC, map_location="cpu", weights_only=False)
    if parent.get("update") != 472 or parent.get("resume_forbidden") is True:
        raise RuntimeError("U472 parent contract failed")
    config = ppo.PPOConfig(**parent["config"])
    config.bc_replay_data = str(SPECIAL); config.bc_replay_split = "train"
    config.bc_replay_batches = 32; config.bc_replay_batch_size = 256
    config.bc_replay_workers = 8; config.bc_replay_steps = 1
    config.bc_replay_lr_scale = LR_SCALE; config.bc_replay_loss = "ordered"
    config.bc_replay_order_context_weight = 8.0
    config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    config.bc_replay_context34_rows_per_batch = 1; config.seed = SEED
    model = ppo.instantiate_model_from_checkpoint(parent, bc_checkpoint, device)
    actor_parameters, _, trainable = ppo.configure_trainable_scope(model, "heads")
    actor_names = list(trainable["actor_parameter_names"])
    if sorted(actor_names) != sorted(MUTABLE_NAMES):
        raise RuntimeError("actor-head mutable manifest drifted")
    before = audit.clone_model_state(model)
    batches = ppo.build_bc_replay_batches(config, parent["model_config"])
    cache_hash, batch_hashes = audit.replay_cache_manifest(batches)
    if len(batches) != 32 or set(int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum()) for batch in batches) != {1}:
        raise RuntimeError("special replay cache contract failed")
    loss_before = cache_loss(model, batches, config, device)
    common: dict[str, Any] = {
        "schema_version": "ptcg-design202608166-u472-fresh-special-bc-v1",
        "mode": "audit_only" if args.audit_only else "special_bc",
        "parent": {"path": str(PARENT), "sha256": PARENT_SHA256, "update": 472},
        "sources": {"bc": {"path": str(BC), "sha256": BC_SHA256}, "special": {"path": str(SPECIAL), "sha256": SPECIAL_SHA256}, "trainer": {"path": str(TRAINER), "sha256": TRAINER_SHA256}, "tool": {"path": str(tool), "sha256": args.expected_tool_sha256}},
        "protocol": {"seed": SEED, "batch_indices": list(BATCH_INDICES), "milestones": list(MILESTONES), "learning_rate": LEARNING_RATE, "optimizer": "fresh_adamw", "mutable_parameter_names": list(MUTABLE_NAMES)},
        "cache": {"sha256": cache_hash, "batches": 32, "rows": 8192, "context34_rows_per_batch": 1, "loss_before": loss_before},
    }
    os.mkdir(target, mode=0o700)
    if args.audit_only:
        result = common | {"status": "audit_passed", "optimizer_steps": 0, "checkpoint_writes": 0, "model_state_unchanged": model_state_hash(before) == model_state_hash(audit.clone_model_state(model))}
        write_exclusive(PREFLIGHT_ROOT / "preflight_audit.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print(json.dumps(result, sort_keys=True)); return 0
    optimizer = torch.optim.AdamW(actor_parameters, lr=LEARNING_RATE, eps=1e-5, weight_decay=config.weight_decay)
    per_step: list[dict[str, Any]] = []; endpoints: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()
    for step, batch_index in enumerate(BATCH_INDICES, 1):
        metrics = ppo.bc_replay_update(model, optimizer, [batches[batch_index]], config, device, ACTOR_LR)
        if not isinstance(metrics, dict) or metrics.get("rows") != 256:
            raise RuntimeError("special update contract failed")
        per_step.append({"step": step, "batch_index": batch_index, "batch_sha256": batch_hashes[batch_index], "metrics": metrics})
        if step in MILESTONES:
            state = audit.clone_model_state(model)
            changed = audit.changed_tensor_names(before, state)
            if changed != sorted(MUTABLE_NAMES):
                raise RuntimeError(f"S{step} changed unexpected tensors")
            slim = {key: copy.deepcopy(parent[key]) for key in REQUIRED_SLIM_KEYS}
            slim.update({
                "model_state_dict": state, "update": 472,
                "evaluation_only": True, "resume_forbidden": True,
                "optimizer_states_omitted": ["optimizer_state_dict", "bc_replay_optimizer_state_dict", "opponent_quota_state"],
                "fresh_special_bc": {"schema_version": common["schema_version"], "created_at_utc": created_at, "steps": step, "model_state_sha256": model_state_hash(state), "mutable_parameter_names": list(MUTABLE_NAMES), "resume_forbidden": True},
            })
            buffer = io.BytesIO(); torch.save(slim, buffer); payload = buffer.getvalue()
            filename = f"u472-fresh-special-s{step:02d}.pt"; write_exclusive(OUTPUT_ROOT / filename, payload)
            endpoints.append({"steps": step, "path": str((OUTPUT_ROOT / filename).relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest(), "model_state_sha256": model_state_hash(state), "cache_loss": cache_loss(model, batches, config, device)})
    result = common | {
        "status": "special_bc_completed", "per_step": per_step, "endpoints": endpoints,
        "integrity": {"checkpoint_writes": len(endpoints), "changed_exactly_mutable10": True, "all_finite": audit.finite_nested(audit.clone_model_state(model)) and audit.finite_nested(optimizer.state_dict()), "parent_unchanged": audit.file_sha256(PARENT) == PARENT_SHA256},
        "scope": {"evaluation_only": True, "resume_forbidden": True, "package": False, "upload": False, "submission": False},
    }
    if not all(result["integrity"][key] for key in ("changed_exactly_mutable10", "all_finite", "parent_unchanged")):
        raise RuntimeError("special terminal integrity failed")
    write_exclusive(MANIFEST, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
