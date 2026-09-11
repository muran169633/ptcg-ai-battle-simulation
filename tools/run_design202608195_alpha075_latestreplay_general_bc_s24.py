#!/usr/bin/env python3
"""Train conservative S8/S16/S24 general-BC endpoints on the August 5 replay."""

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
import tempfile
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
    "artifacts/design202608170_parent_plus_u476_actor_delta/"
    "parent-plus-u476-actor-alpha075.pt"
)
PARENT_SHA256 = "e01d9161245e3559f4ce21a3f14f6e09a1a4ba2b84c0f2ba1a59f15eda7c85eb"
BC = ROOT / (
    "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_"
    "seed20260922_20260731/best.pt"
)
BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
REPLAY = ROOT / (
    "data/bc_marnie_top50_current14_latesttop50_"
    "trainvalid_through0804_design202608194.zip"
)
REPLAY_SHA256 = "74ebd086d065f0e2ab95dde2192e463c8ba1eb04ab4e40c1876d484f66475c15"
TRAINER = TOOLS / "train_ppo.py"
TRAINER_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
HELPER = TOOLS / "run_ppo_bc_repair.py"
HELPER_SHA256 = "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"

SEED = 202608195
STEPS = 24
ENDPOINTS = (8, 16, 24)
BATCH_INDICES = (
    26, 5, 55, 22, 30, 20, 65, 19,
    9, 57, 61, 35, 0, 56, 31, 53,
    48, 33, 66, 27, 51, 50, 52, 42,
)
ACTOR_LR = 2.4e-5
LR_SCALE = 0.05
LEARNING_RATE = ACTOR_LR * LR_SCALE
TRAIN_BATCHES = 72
VALID_BATCHES = 16
BATCH_SIZE = 256
OUTPUT_ROOT = ROOT / "artifacts/design202608195_alpha075_latestreplay_general_bc_s24"
MANIFEST = OUTPUT_ROOT / "general_bc_manifest.json"
PREFLIGHT_ROOT = ROOT / (
    "artifacts/design202608195_alpha075_latestreplay_general_bc_s24_preflight"
)
PREFLIGHT_MANIFEST = PREFLIGHT_ROOT / "preflight_audit.json"


def require(path: Path, expected: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"input is not a regular file: {path}")
    actual = audit.file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"input SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def actor_l2(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    names: list[str],
) -> float:
    return math.sqrt(
        sum(
            float((after[name].double() - before[name].double()).square().sum())
            for name in names
        )
    )


@torch.inference_mode()
def cache_loss(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    config: ppo.PPOConfig,
    device: torch.device,
) -> dict[str, float]:
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
    result = {"loss": total_loss / total_rows, "rows": total_rows}
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
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("requires repository cwd")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    require(tool, args.expected_tool_sha256)
    for path, digest in (
        (PARENT, PARENT_SHA256),
        (BC, BC_SHA256),
        (REPLAY, REPLAY_SHA256),
        (TRAINER, TRAINER_SHA256),
        (HELPER, HELPER_SHA256),
    ):
        require(path, digest)

    target_root = PREFLIGHT_ROOT if args.audit_only else OUTPUT_ROOT
    if target_root.exists() or target_root.is_symlink():
        raise FileExistsError(target_root)
    if (
        len(BATCH_INDICES) != STEPS
        or len(set(BATCH_INDICES)) != STEPS
        or not all(0 <= value < TRAIN_BATCHES for value in BATCH_INDICES)
    ):
        raise RuntimeError("frozen batch order is invalid")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")

    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(BC, map_location="cpu", weights_only=False)
    if (
        parent.get("update") != 476
        or parent.get("evaluation_only") is not True
        or parent.get("resume_forbidden") is not True
        or set(parent.get("optimizer_states_omitted", []))
        != {
            "optimizer_state_dict",
            "bc_replay_optimizer_state_dict",
            "opponent_quota_state",
        }
    ):
        raise RuntimeError("alpha075 evaluation-parent contract failed")

    train_config = ppo.PPOConfig(**parent["config"])
    train_config.bc_replay_data = str(REPLAY)
    train_config.bc_replay_split = "train"
    train_config.bc_replay_batches = TRAIN_BATCHES
    train_config.bc_replay_batch_size = BATCH_SIZE
    train_config.bc_replay_workers = 8
    train_config.bc_replay_steps = 1
    train_config.bc_replay_lr_scale = LR_SCALE
    train_config.bc_replay_loss = "ordered"
    train_config.bc_replay_order_context_weight = 8.0
    train_config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    train_config.bc_replay_context34_rows_per_batch = 4
    train_config.seed = SEED

    valid_config = copy.deepcopy(train_config)
    valid_config.bc_replay_split = "valid"
    valid_config.bc_replay_batches = VALID_BATCHES
    valid_config.seed = SEED + 1

    model = ppo.instantiate_model_from_checkpoint(parent, bc_checkpoint, device)
    actor_parameters, _, trainable = ppo.configure_trainable_scope(
        model, "last_block_heads"
    )
    actor_names = list(trainable["actor_parameter_names"])
    value_names = list(trainable["value_parameter_names"])
    before = audit.clone_model_state(model)
    parent_model_sha256 = ppo.model_state_sha256(model)

    train_batches = ppo.build_bc_replay_batches(
        train_config, parent["model_config"]
    )
    valid_batches = ppo.build_bc_replay_batches(
        valid_config, parent["model_config"]
    )
    train_cache_sha256, train_batch_sha256 = audit.replay_cache_manifest(
        train_batches
    )
    valid_cache_sha256, valid_batch_sha256 = audit.replay_cache_manifest(
        valid_batches
    )
    if len(train_batches) != TRAIN_BATCHES or len(valid_batches) != VALID_BATCHES:
        raise RuntimeError("replay cache batch-count drift")
    if any(int(batch["action_counts"].shape[0]) != BATCH_SIZE for batch in train_batches):
        raise RuntimeError("train cache batch-size drift")
    if any(int(batch["action_counts"].shape[0]) != BATCH_SIZE for batch in valid_batches):
        raise RuntimeError("valid cache batch-size drift")
    if set(
        int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum())
        for batch in train_batches + valid_batches
    ) != {4}:
        raise RuntimeError("context-34 stratification drift")

    train_loss_before = cache_loss(model, train_batches, train_config, device)
    valid_loss_before = cache_loss(model, valid_batches, valid_config, device)
    common: dict[str, Any] = {
        "schema_version": "ptcg-design202608195-alpha075-latestreplay-general-bc-s24-v1",
        "mode": "audit_only" if args.audit_only else "general_bc",
        "parent": {
            "path": str(PARENT),
            "sha256": PARENT_SHA256,
            "update": 476,
            "model_state_sha256": parent_model_sha256,
            "evaluation_only": True,
            "resume_forbidden": True,
        },
        "sources": {
            "bc": {"path": str(BC), "sha256": BC_SHA256},
            "replay": {"path": str(REPLAY), "sha256": REPLAY_SHA256},
            "trainer": {"path": str(TRAINER), "sha256": TRAINER_SHA256},
            "helper": {"path": str(HELPER), "sha256": HELPER_SHA256},
            "tool": {"path": str(tool), "sha256": args.expected_tool_sha256},
        },
        "protocol": {
            "seed": SEED,
            "steps": STEPS,
            "endpoints": list(ENDPOINTS),
            "batch_indices": list(BATCH_INDICES),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "actor_learning_rate_reference": ACTOR_LR,
            "lr_scale": LR_SCALE,
            "trainable_scope": "last_block_heads",
            "optimizer": "fresh_adamw",
            "train_split": "train",
            "validation_split": "valid_inference_only",
        },
        "caches": {
            "train": {
                "sha256": train_cache_sha256,
                "batches": TRAIN_BATCHES,
                "rows": TRAIN_BATCHES * BATCH_SIZE,
                "loss_before": train_loss_before,
            },
            "valid": {
                "sha256": valid_cache_sha256,
                "batch_sha256": valid_batch_sha256,
                "batches": VALID_BATCHES,
                "rows": VALID_BATCHES * BATCH_SIZE,
                "loss_before": valid_loss_before,
                "used_for_optimizer_steps": False,
            },
        },
    }

    os.mkdir(target_root, mode=0o700)
    if args.audit_only:
        unchanged = ppo.model_state_sha256(model) == parent_model_sha256
        result = common | {
            "status": "audit_passed",
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "model_state_unchanged": unchanged,
            "validation_rows_used_for_training": 0,
        }
        if not unchanged:
            raise RuntimeError("audit-only mode changed model state")
        write_exclusive(
            PREFLIGHT_MANIFEST,
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=LEARNING_RATE,
        eps=1e-5,
        weight_decay=train_config.weight_decay,
    )
    per_step: list[dict[str, Any]] = []
    endpoint_records: dict[str, Any] = {}
    for step, batch_index in enumerate(BATCH_INDICES, 1):
        metrics = ppo.bc_replay_update(
            model,
            optimizer,
            [train_batches[batch_index]],
            train_config,
            device,
            ACTOR_LR,
        )
        if (
            not isinstance(metrics, dict)
            or metrics.get("rows") != BATCH_SIZE
            or metrics.get("steps") != 1
        ):
            raise RuntimeError("general-BC update contract failed")
        per_step.append(
            {
                "step": step,
                "batch_index": batch_index,
                "batch_sha256": train_batch_sha256[batch_index],
                "metrics": metrics,
            }
        )
        if step not in ENDPOINTS:
            continue

        after = audit.clone_model_state(model)
        changed = audit.changed_tensor_names(before, after)
        if changed != sorted(actor_names):
            raise RuntimeError(f"S{step} changed unexpected tensors")
        if any(not torch.equal(before[name], after[name]) for name in value_names):
            raise RuntimeError(f"S{step} changed value-head tensors")
        train_loss = cache_loss(model, train_batches, train_config, device)
        valid_loss = cache_loss(model, valid_batches, valid_config, device)
        if not train_loss["loss"] < train_loss_before["loss"]:
            raise RuntimeError(f"S{step} did not lower train cache loss")
        if not audit.finite_nested(after) or not audit.finite_nested(
            optimizer.state_dict()
        ):
            raise RuntimeError(f"S{step} contains non-finite state")

        payload = copy.deepcopy(parent)
        payload["model_state_dict"] = after
        payload["config"] = copy.deepcopy(train_config.__dict__)
        payload["evaluation_only"] = True
        payload["resume_forbidden"] = True
        payload["optimizer_states_omitted"] = True
        payload.pop("optimizer_state_dict", None)
        payload.pop("bc_replay_optimizer_state_dict", None)
        payload["post_ppo_general_bc"] = {
            "schema_version": common["schema_version"],
            "steps": step,
            "rows": step * BATCH_SIZE,
            "train_cache_sha256": train_cache_sha256,
            "valid_cache_sha256": valid_cache_sha256,
            "fresh_optimizer": True,
            "optimizer_state_intentionally_omitted": True,
            "fresh_ppo_bootstrap_required": True,
            "validation_rows_used_for_training": 0,
        }
        checkpoint = OUTPUT_ROOT / f"general-bc-s{step:02d}.pt"
        atomic_torch_save(checkpoint, payload)
        endpoint_records[str(step)] = {
            "checkpoint": str(checkpoint),
            "sha256": audit.file_sha256(checkpoint),
            "model_state_sha256": ppo.model_state_sha256(model),
            "actor24_l2_displacement": actor_l2(before, after, actor_names),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "changed_parameter_names": changed,
            "value_head_unchanged": True,
            "all_finite": True,
        }

    final_state = audit.clone_model_state(model)
    result = common | {
        "status": "general_bc_completed",
        "per_step": per_step,
        "endpoints": endpoint_records,
        "integrity": {
            "changed_parameter_names": audit.changed_tensor_names(
                before, final_state
            ),
            "changed_exactly_actor24": audit.changed_tensor_names(
                before, final_state
            )
            == sorted(actor_names),
            "value_head_unchanged": all(
                torch.equal(before[name], final_state[name]) for name in value_names
            ),
            "parent_unchanged": audit.file_sha256(PARENT) == PARENT_SHA256,
            "replay_unchanged": audit.file_sha256(REPLAY) == REPLAY_SHA256,
            "validation_rows_used_for_training": 0,
            "checkpoint_writes": len(endpoint_records),
        },
        "not_a_new_ppo_update": True,
        "fresh_ppo_bootstrap_required": True,
    }
    if not all(
        (
            result["integrity"]["changed_exactly_actor24"],
            result["integrity"]["value_head_unchanged"],
            result["integrity"]["parent_unchanged"],
            result["integrity"]["replay_unchanged"],
            len(endpoint_records) == len(ENDPOINTS),
        )
    ):
        raise RuntimeError("terminal integrity check failed")
    write_exclusive(
        MANIFEST, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
