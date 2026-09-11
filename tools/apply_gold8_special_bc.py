#!/usr/bin/env python3
"""Apply a small, actor-only special-BC pass after a PPO checkpoint.

The input archive must declare the same exact deck hash as the PPO learner.
Only the PPO actor scope is optimized; the value head and every frozen tensor
must remain bit-identical.  The output checkpoint deliberately drops stale
PPO/replay optimizer state and records a reproducible provenance manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import train_ppo as ppo


SCHEMA_VERSION = "ptcg-gold8-post-ppo-special-bc-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def archive_deck_hash(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except KeyError as exc:
            raise ValueError(f"Special-BC archive has no manifest.json: {path}") from exc
    profile = manifest.get("profile")
    if isinstance(profile, dict) and profile.get("deck_hash"):
        return str(profile["deck_hash"])
    deck_hash_filter = manifest.get("deck_hash_filter")
    if isinstance(deck_hash_filter, str) and deck_hash_filter:
        return deck_hash_filter
    deck_hashes = manifest.get("deck_hashes")
    if isinstance(deck_hashes, list) and len(deck_hashes) == 1:
        return str(deck_hashes[0])
    raise ValueError(
        "Special-BC archive must declare exactly one deck hash in its manifest"
    )


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def changed_tensor_names(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> list[str]:
    if set(before) != set(after):
        raise RuntimeError("Model tensor schema changed during special BC")
    return sorted(
        name for name in before if not torch.equal(before[name], after[name])
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--special-data", type=Path, required=True)
    parser.add_argument("--expected-deck-hash", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cache-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate-scale", type=float, default=0.05)
    parser.add_argument("--context34-rows-per-batch", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.cache_batches < args.steps:
        raise ValueError("--cache-batches must be at least --steps")
    if args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("--batch-size and --workers must be positive")
    if not 0.0 < args.learning_rate_scale <= 0.1:
        raise ValueError("--learning-rate-scale must be in (0, 0.1]")

    parent_path = args.parent_checkpoint.resolve()
    special_data = args.special_data.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    if not special_data.is_file():
        raise FileNotFoundError(special_data)

    declared_hash = archive_deck_hash(special_data)
    if declared_hash != args.expected_deck_hash:
        raise ValueError(
            "Special-data deck hash mismatch: "
            f"expected {args.expected_deck_hash}, got {declared_hash}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    parent_hash_before = file_sha256(parent_path)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise ValueError("Parent is not a compatible PPO checkpoint")
    parent_deck_hash = str(parent.get("learner_deck_hash") or "")
    if parent_deck_hash != args.expected_deck_hash:
        raise ValueError(
            "Parent learner deck hash mismatch: "
            f"expected {args.expected_deck_hash}, got {parent_deck_hash!r}"
        )
    raw_config = parent.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Parent checkpoint has no PPO config")
    parent_config = ppo.PPOConfig(**raw_config)
    bc_checkpoint_path = Path(parent_config.bc_checkpoint).resolve()
    if not bc_checkpoint_path.is_file():
        raise FileNotFoundError(bc_checkpoint_path)
    bc_checkpoint = torch.load(
        bc_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = ppo.instantiate_model_from_checkpoint(parent, bc_checkpoint, device)
    actor_parameters, _, trainable_manifest = ppo.configure_trainable_scope(
        model,
        parent_config.trainable_scope,
    )
    actor_names = set(trainable_manifest["actor_parameter_names"])
    value_names = set(trainable_manifest["value_parameter_names"])

    special_config = replace(
        parent_config,
        bc_replay_data=str(special_data),
        bc_replay_split="train",
        bc_replay_batches=args.cache_batches,
        bc_replay_batch_size=args.batch_size,
        bc_replay_workers=args.workers,
        bc_replay_steps=1,
        bc_replay_lr_scale=args.learning_rate_scale,
        bc_replay_loss="ordered",
        bc_replay_order_context_weight=8.0,
        bc_replay_context34_rows_per_batch=args.context34_rows_per_batch,
        bc_replay_non_context34_fixed_multi_action_order_weight=1.0,
        seed=args.seed,
        device=str(device),
    )
    replay_batches = ppo.build_bc_replay_batches(
        special_config,
        parent["model_config"],
    )
    if len(replay_batches) < args.steps:
        raise RuntimeError(
            f"Special data produced {len(replay_batches)} batches for {args.steps} steps"
        )
    selected_indices = random.Random(args.seed + 101).sample(
        range(len(replay_batches)),
        args.steps,
    )

    replay_optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=parent_config.learning_rate * args.learning_rate_scale,
        eps=1e-5,
        weight_decay=parent_config.weight_decay,
    )
    before = clone_state(model)
    per_step: list[dict[str, Any]] = []
    for step, batch_index in enumerate(selected_indices, 1):
        metrics = ppo.bc_replay_update(
            model,
            replay_optimizer,
            [replay_batches[batch_index]],
            special_config,
            device,
            actor_learning_rate=parent_config.learning_rate,
        )
        if metrics is None or not finite_nested(metrics):
            raise RuntimeError(f"Special-BC step {step} produced invalid metrics")
        per_step.append(
            {"step": step, "source_batch_index": batch_index, "metrics": metrics}
        )

    after = clone_state(model)
    changed = changed_tensor_names(before, after)
    if not changed:
        raise RuntimeError("Special BC did not change any model tensor")
    unexpected = sorted(set(changed) - actor_names)
    if unexpected:
        raise RuntimeError(
            "Special BC changed tensors outside actor scope: "
            + json.dumps(unexpected)
        )
    if any(not torch.equal(before[name], after[name]) for name in value_names):
        raise RuntimeError("Special BC changed value-head tensors")
    if not finite_nested(after):
        raise RuntimeError("Special BC produced non-finite model tensors")
    if file_sha256(parent_path) != parent_hash_before:
        raise RuntimeError("Parent checkpoint changed during special BC")

    output_dir.mkdir(parents=True)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": {
            "path": str(parent_path),
            "sha256": parent_hash_before,
            "update": int(parent.get("update", -1)),
        },
        "bc_checkpoint": {
            "path": str(bc_checkpoint_path),
            "sha256": file_sha256(bc_checkpoint_path),
        },
        "special_data": {
            "path": str(special_data),
            "sha256": file_sha256(special_data),
            "deck_hash": declared_hash,
            "split": "train",
        },
        "training": {
            "seed": args.seed,
            "steps": args.steps,
            "cache_batches": len(replay_batches),
            "batch_size": args.batch_size,
            "selected_batch_indices": selected_indices,
            "learning_rate": (
                parent_config.learning_rate * args.learning_rate_scale
            ),
            "loss": "ordered",
            "order_context_weight": 8.0,
            "context34_rows_per_batch": args.context34_rows_per_batch,
            "trainable_scope": parent_config.trainable_scope,
        },
        "integrity": {
            "changed_parameter_names": changed,
            "changed_parameters_subset_of_actor": True,
            "value_head_parameters_unchanged": True,
            "all_metrics_and_model_tensors_finite": True,
            "parent_checkpoint_unchanged": True,
        },
        "per_step": per_step,
        "checkpoint_update_label": int(parent.get("update", -1)),
        "not_a_new_ppo_update": True,
    }

    payload = copy.deepcopy(parent)
    payload["model_state_dict"] = after
    payload.pop("optimizer_state_dict", None)
    payload.pop("bc_replay_optimizer_state_dict", None)
    payload.pop("opponent_quota_state", None)
    payload.pop("ppo_objective_state", None)
    payload["post_ppo_special_bc"] = provenance
    payload["optimizer_state_policy"] = (
        "stale PPO and replay optimizer states removed after actor-only BC"
    )
    output_checkpoint = output_dir / f"special-bc-{args.steps:04d}.pt"
    torch.save(payload, output_checkpoint)
    result = provenance | {
        "checkpoint": {
            "path": str(output_checkpoint),
            "sha256": file_sha256(output_checkpoint),
            "update": int(parent.get("update", -1)),
        }
    }
    atomic_write_json(output_dir / "manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
