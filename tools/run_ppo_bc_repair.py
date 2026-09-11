#!/usr/bin/env python3
"""Apply a deterministic, actor-only BC repair to a frozen PPO checkpoint.

This tool deliberately does not run PPO rollouts or evaluation.  It continues
the replay AdamW state embedded in a PPO checkpoint, consumes an explicit
sequence of cached replay batches, and writes only one final repaired
checkpoint.  Audit mode validates every input and optimizer mapping while
performing zero optimizer steps and writing no checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import train_ppo as ppo


EXPECTED_PARENT_CONFIG: dict[str, Any] = {
    "actor_reduction": "quota_group_mean",
    "trainable_scope": "last_block_heads",
    "learning_rate": 1.8e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 0.5,
    "bc_replay_split": "train",
    "bc_replay_batches": 72,
    "bc_replay_batch_size": 256,
    "bc_replay_workers": 8,
    "bc_replay_steps": 1,
    "bc_replay_lr_scale": 0.05,
    "bc_replay_loss": "ordered",
    "bc_replay_order_context_weight": 8.0,
    "bc_replay_context34_rows_per_batch": 2,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def update_digest(digest: Any, value: Any) -> None:
    """Hash nested optimizer state without relying on torch serialization."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            update_digest(digest, key)
            update_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        for item in value:
            update_digest(digest, item)
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bool):
        digest.update(b"bool\0")
        digest.update(b"1\0" if value else b"0\0")
        return
    if isinstance(value, int):
        digest.update(b"int\0")
        digest.update(str(value).encode("ascii"))
        digest.update(b"\0")
        return
    if isinstance(value, float):
        digest.update(b"float\0")
        digest.update(value.hex().encode("ascii"))
        digest.update(b"\0")
        return
    if isinstance(value, str):
        digest.update(b"str\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
        return
    raise TypeError(f"Unsupported nested hash value: {type(value).__name__}")


def nested_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    update_digest(digest, value)
    return digest.hexdigest()


def replay_batch_sha256(batch: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(batch):
        tensor = batch[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def replay_cache_manifest(
    batches: list[dict[str, torch.Tensor]],
) -> tuple[str, list[str]]:
    per_batch = [replay_batch_sha256(batch) for batch in batches]
    digest = hashlib.sha256()
    for index, batch_hash in enumerate(per_batch):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(batch_hash))
    return digest.hexdigest(), per_batch


def optimizer_steps(state_dict: dict[str, Any]) -> list[int]:
    state = state_dict.get("state")
    if not isinstance(state, dict) or not state:
        raise ValueError("Replay optimizer state is absent or empty")
    steps: list[int] = []
    for parameter_id in sorted(state):
        entry = state[parameter_id]
        if not isinstance(entry, dict) or "step" not in entry:
            raise ValueError("Replay optimizer parameter has no AdamW step")
        raw_step = entry["step"]
        if isinstance(raw_step, torch.Tensor):
            if raw_step.numel() != 1:
                raise ValueError("Replay optimizer AdamW step is not scalar")
            step = int(raw_step.item())
        else:
            step = int(raw_step)
        steps.append(step)
    return steps


def finite_nested(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_nested(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_nested(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def changed_tensor_names(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> list[str]:
    if set(before) != set(after):
        raise ValueError("Model state tensor names changed during BC repair")
    return sorted(
        name for name in before if not torch.equal(before[name], after[name])
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_sha(path: Path, expected: str, label: str) -> str:
    actual = file_sha256(path)
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


def validate_repair_request(
    repair_steps: int,
    batch_indices: list[int],
) -> None:
    if repair_steps != 8:
        raise ValueError("This preregistered repair tool requires exactly 8 steps")
    if len(batch_indices) != repair_steps:
        raise ValueError(
            "--batch-indices length must equal --repair-steps"
        )
    if len(set(batch_indices)) != len(batch_indices):
        raise ValueError("Repair batch indices must be without replacement")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue the embedded BC replay optimizer for an explicit number "
            "of deterministic actor-only post-PPO repair steps."
        )
    )
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--expected-parent-update", type=int, required=True)
    parser.add_argument("--expected-bc-checkpoint-sha256", required=True)
    parser.add_argument("--expected-replay-data-sha256", required=True)
    parser.add_argument("--expected-train-ppo-sha256", required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--expected-replay-state-step", type=int, required=True)
    parser.add_argument("--repair-seed", type=int, required=True)
    parser.add_argument("--repair-steps", type=int, required=True)
    parser.add_argument(
        "--batch-indices",
        type=int,
        nargs="+",
        required=True,
        help="Explicit zero-based replay cache batch indices.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tool_path = Path(__file__).resolve()
    train_path = tool_path.with_name("train_ppo.py")
    parent_path = args.parent_checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse existing output directory: {output_dir}"
        )
    validate_repair_request(args.repair_steps, args.batch_indices)

    tool_hash = validate_sha(
        tool_path,
        args.expected_tool_sha256,
        "repair tool",
    )
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

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    random.seed(args.repair_seed)
    torch.manual_seed(args.repair_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.repair_seed)

    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise ValueError("Parent checkpoint is not a compatible PPO checkpoint")
    if int(parent.get("update", -1)) != args.expected_parent_update:
        raise ValueError(
            "Parent checkpoint update does not match --expected-parent-update"
        )
    raw_config = parent.get("config")
    config = validate_parent_config(raw_config)

    bc_checkpoint_path = Path(config.bc_checkpoint).resolve()
    bc_checkpoint_hash = validate_sha(
        bc_checkpoint_path,
        args.expected_bc_checkpoint_sha256,
        "BC architecture checkpoint",
    )
    replay_data_path = Path(str(config.bc_replay_data)).resolve()
    replay_data_hash = validate_sha(
        replay_data_path,
        args.expected_replay_data_sha256,
        "BC replay archive",
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
    parent_parameter_names = parent.get("optimizer_parameter_names")
    if not isinstance(parent_parameter_names, dict):
        raise ValueError("Parent checkpoint has no optimizer parameter manifest")
    parent_actor_names = parent_parameter_names.get("actor")
    if actor_names != parent_actor_names:
        raise ValueError(
            "Actor parameter names or order do not match the parent optimizer"
        )
    value_names = list(trainable_manifest["value_parameter_names"])

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
    steps_before = optimizer_steps(replay_optimizer.state_dict())
    if len(steps_before) != len(actor_names):
        raise ValueError("Replay optimizer state does not cover every actor tensor")
    if set(steps_before) != {args.expected_replay_state_step}:
        raise ValueError(
            "Replay optimizer AdamW step does not match "
            "--expected-replay-state-step"
        )

    ppo_optimizer_state_before = copy.deepcopy(
        parent.get("optimizer_state_dict")
    )
    ppo_optimizer_hash_before = nested_sha256(
        ppo_optimizer_state_before
    )
    replay_optimizer_hash_before = nested_sha256(
        replay_optimizer.state_dict()
    )
    model_state_before = clone_model_state(model)
    model_hash_before = ppo.model_state_sha256(model)

    replay_batches = ppo.build_bc_replay_batches(
        config,
        parent["model_config"],
    )
    if len(replay_batches) != EXPECTED_PARENT_CONFIG["bc_replay_batches"]:
        raise ValueError("Replay cache batch count is not exactly 72")
    if any(
        int(batch["contexts"].shape[0])
        != EXPECTED_PARENT_CONFIG["bc_replay_batch_size"]
        for batch in replay_batches
    ):
        raise ValueError("Replay cache has a non-256-row batch")
    context34_per_batch = [
        int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum())
        for batch in replay_batches
    ]
    if set(context34_per_batch) != {
        EXPECTED_PARENT_CONFIG["bc_replay_context34_rows_per_batch"]
    }:
        raise ValueError(
            "Replay cache does not have exactly two context-34 rows per batch"
        )
    if min(args.batch_indices) < 0 or max(args.batch_indices) >= len(
        replay_batches
    ):
        raise ValueError("Repair batch index is outside the replay cache")
    cache_hash, batch_hashes = replay_cache_manifest(replay_batches)
    if cache_hash != args.expected_cache_sha256:
        raise ValueError(
            "Replay cache SHA256 mismatch: "
            f"expected {args.expected_cache_sha256}, got {cache_hash}"
        )

    selected_context34 = [
        context34_per_batch[index] for index in args.batch_indices
    ]
    common = {
        "schema_version": "ptcg-ppo-post-bc-repair-v1",
        "mode": "audit_only" if args.audit_only else "repair",
        "parent": {
            "path": str(parent_path),
            "sha256": parent_hash_before,
            "update": int(parent["update"]),
            "model_state_sha256": model_hash_before,
        },
        "sources": {
            "tool": {"path": str(tool_path), "sha256": tool_hash},
            "train_ppo": {"path": str(train_path), "sha256": train_hash},
            "bc_checkpoint": {
                "path": str(bc_checkpoint_path),
                "sha256": bc_checkpoint_hash,
            },
            "replay_data": {
                "path": str(replay_data_path),
                "sha256": replay_data_hash,
                "split": config.bc_replay_split,
            },
        },
        "repair": {
            "seed": args.repair_seed,
            "steps_requested": args.repair_steps,
            "batch_indices": list(args.batch_indices),
            "batch_selection": "explicit_without_replacement",
            "batch_sha256": [
                batch_hashes[index] for index in args.batch_indices
            ],
            "rows_per_batch": config.bc_replay_batch_size,
            "context34_rows_per_batch": selected_context34,
            "loss": config.bc_replay_loss,
            "order_context_weight": config.bc_replay_order_context_weight,
            "learning_rate": (
                config.learning_rate * config.bc_replay_lr_scale
            ),
            "max_grad_norm": config.max_grad_norm,
            "trainable_scope": config.trainable_scope,
        },
        "replay_cache": {
            "sha256": cache_hash,
            "batches": len(replay_batches),
            "rows": sum(
                int(batch["contexts"].shape[0])
                for batch in replay_batches
            ),
            "context34_rows": sum(context34_per_batch),
            "all_batches_have_exact_context34_quota": True,
        },
        "optimizer": {
            "actor_parameter_names": actor_names,
            "value_parameter_names": value_names,
            "replay_state_tensor_count": len(steps_before),
            "replay_steps_before": steps_before,
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
            "python_seed": args.repair_seed,
            "torch_seed": args.repair_seed,
        },
    }

    output_dir.mkdir(parents=False)
    if args.audit_only:
        parent_hash_after = file_sha256(parent_path)
        result = common | {
            "status": "audit_passed",
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "model_state_sha256_after": ppo.model_state_sha256(model),
            "model_state_unchanged": (
                ppo.model_state_sha256(model) == model_hash_before
            ),
            "replay_optimizer_state_unchanged": (
                nested_sha256(replay_optimizer.state_dict())
                == replay_optimizer_hash_before
            ),
            "ppo_optimizer_state_unchanged": (
                nested_sha256(parent.get("optimizer_state_dict"))
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
    for repair_step, batch_index in enumerate(args.batch_indices, start=1):
        metrics = ppo.bc_replay_update(
            model,
            replay_optimizer,
            [replay_batches[batch_index]],
            config,
            device,
            config.learning_rate,
        )
        if not isinstance(metrics, dict) or metrics.get("steps") != 1:
            raise RuntimeError("BC replay core did not execute exactly one step")
        if int(metrics.get("rows", -1)) != config.bc_replay_batch_size:
            raise RuntimeError("BC replay core consumed an unexpected row count")
        if int(metrics.get("context_34_rows", -1)) != 2:
            raise RuntimeError(
                "BC replay core consumed an unexpected context-34 count"
            )
        if not finite_nested(metrics):
            raise FloatingPointError("Non-finite repair metric")
        per_step.append(
            {
                "repair_step": repair_step,
                "batch_index": batch_index,
                "batch_sha256": batch_hashes[batch_index],
                "metrics": metrics,
            }
        )

    model_state_after = clone_model_state(model)
    model_hash_after = ppo.model_state_sha256(model)
    if not finite_nested(model_state_after):
        raise FloatingPointError("Repaired model contains non-finite tensors")
    changed_names = changed_tensor_names(
        model_state_before,
        model_state_after,
    )
    actor_name_set = set(actor_names)
    if not changed_names:
        raise RuntimeError("BC repair did not change any model tensor")
    unexpected_changes = sorted(set(changed_names) - actor_name_set)
    if unexpected_changes:
        raise RuntimeError(
            "BC repair changed tensors outside the actor scope: "
            + json.dumps(unexpected_changes)
        )
    value_parameters_unchanged = all(
        torch.equal(model_state_before[name], model_state_after[name])
        for name in value_names
    )
    if not value_parameters_unchanged:
        raise RuntimeError("BC repair changed value-head parameters")

    replay_state_after = replay_optimizer.state_dict()
    steps_after = optimizer_steps(replay_state_after)
    expected_final_step = (
        args.expected_replay_state_step + args.repair_steps
    )
    if set(steps_after) != {expected_final_step}:
        raise RuntimeError(
            "Replay AdamW state did not continue by exactly repair_steps"
        )
    ppo_optimizer_hash_after = nested_sha256(
        parent.get("optimizer_state_dict")
    )
    if ppo_optimizer_hash_after != ppo_optimizer_hash_before:
        raise RuntimeError("PPO optimizer state changed during BC repair")
    parent_hash_after = file_sha256(parent_path)
    if parent_hash_after != parent_hash_before:
        raise RuntimeError("Parent checkpoint changed during BC repair")

    integrity = {
        "optimizer_steps": len(per_step),
        "rows": sum(int(item["metrics"]["rows"]) for item in per_step),
        "context34_rows": sum(
            int(item["metrics"]["context_34_rows"])
            for item in per_step
        ),
        "all_metrics_and_model_tensors_finite": True,
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "model_state_changed": model_hash_after != model_hash_before,
        "changed_parameter_names": changed_names,
        "changed_parameters_subset_of_actor": True,
        "value_head_parameters_unchanged": value_parameters_unchanged,
        "replay_steps_after": steps_after,
        "replay_state_sha256_after": nested_sha256(replay_state_after),
        "ppo_state_sha256_after": ppo_optimizer_hash_after,
        "ppo_optimizer_state_unchanged": True,
        "parent_checkpoint_sha256_after": parent_hash_after,
        "parent_checkpoint_unchanged": True,
    }
    if integrity["rows"] != args.repair_steps * config.bc_replay_batch_size:
        raise RuntimeError("Repair row count failed the frozen integrity gate")
    if integrity["context34_rows"] != args.repair_steps * 2:
        raise RuntimeError(
            "Repair context-34 row count failed the frozen integrity gate"
        )

    provenance = common | {
        "status": "repair_completed",
        "per_step": per_step,
        "integrity": integrity,
        "checkpoint_update_label": int(parent["update"]),
        "not_a_new_ppo_update": True,
    }
    output_checkpoint = output_dir / "repair-0008.pt"
    payload = copy.deepcopy(parent)
    payload["model_state_dict"] = model_state_after
    payload["bc_replay_optimizer_state_dict"] = replay_state_after
    payload["post_ppo_bc_repair"] = provenance
    torch.save(payload, output_checkpoint)
    output_checkpoint_hash = file_sha256(output_checkpoint)

    result = provenance | {
        "checkpoint": {
            "path": str(output_checkpoint),
            "sha256": output_checkpoint_hash,
            "update": int(parent["update"]),
        },
        "checkpoint_writes": 1,
    }
    write_json(output_dir / "repair_manifest.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
