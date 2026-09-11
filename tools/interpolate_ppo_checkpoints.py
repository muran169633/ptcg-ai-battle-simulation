#!/usr/bin/env python3
"""Interpolate two compatible PTCG PPO checkpoints into a slim checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PPO_FEATURE_VERSION = "ptcg-selfplay-ppo-terminal01-v1"
REQUIRED_PASSTHROUGH_KEYS = (
    "bc_feature_version",
    "reward",
    "action_distribution",
)
OPTIMIZER_KEYS = (
    "optimizer_state_dict",
    "bc_replay_optimizer_state_dict",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint {label} does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint {label} must contain a dict, got "
            f"{type(checkpoint).__name__}"
        )
    return checkpoint


def require_equal_metadata(
    checkpoint_a: Mapping[str, Any],
    checkpoint_b: Mapping[str, Any],
    key: str,
) -> Any:
    if key not in checkpoint_a:
        raise KeyError(f"Checkpoint A is missing required key {key!r}")
    if key not in checkpoint_b:
        raise KeyError(f"Checkpoint B is missing required key {key!r}")
    value_a = checkpoint_a[key]
    value_b = checkpoint_b[key]
    if value_a != value_b:
        raise ValueError(
            f"Checkpoint metadata mismatch for {key!r}: "
            f"A={value_a!r}, B={value_b!r}"
        )
    return copy.deepcopy(value_a)


def checkpoint_deck_hash(checkpoint: Mapping[str, Any], label: str) -> str:
    hashes: list[str] = []
    embedded_hash = checkpoint.get("learner_deck_hash")
    if isinstance(embedded_hash, str) and embedded_hash:
        hashes.append(embedded_hash)

    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        configured_hashes = config.get("deck_hashes")
        if isinstance(configured_hashes, (list, tuple)):
            hashes.extend(
                value for value in configured_hashes
                if isinstance(value, str) and value
            )

    unique_hashes = set(hashes)
    if not unique_hashes:
        raise KeyError(
            f"Checkpoint {label} has no learner_deck_hash or config.deck_hashes"
        )
    if len(unique_hashes) != 1:
        raise ValueError(
            f"Checkpoint {label} contains conflicting deck hashes: "
            f"{sorted(unique_hashes)}"
        )
    return unique_hashes.pop()


def interpolate_state_dict(
    state_a: Mapping[str, Any],
    state_b: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    keys_a = set(state_a)
    keys_b = set(state_b)
    if keys_a != keys_b:
        missing_from_b = sorted(keys_a - keys_b)
        missing_from_a = sorted(keys_b - keys_a)
        raise ValueError(
            "model_state_dict keys differ: "
            f"missing_from_B={missing_from_b[:10]}, "
            f"missing_from_A={missing_from_a[:10]}"
        )

    output: dict[str, Any] = {}
    for name in state_a:
        value_a = state_a[name]
        value_b = state_b[name]
        if not isinstance(value_a, torch.Tensor) or not isinstance(
            value_b, torch.Tensor
        ):
            if type(value_a) is not type(value_b) or value_a != value_b:
                raise ValueError(
                    f"Non-tensor state entry {name!r} differs between checkpoints"
                )
            output[name] = copy.deepcopy(value_a)
            continue

        if value_a.shape != value_b.shape:
            raise ValueError(
                f"Tensor shape mismatch for {name!r}: "
                f"A={tuple(value_a.shape)}, B={tuple(value_b.shape)}"
            )
        if value_a.dtype != value_b.dtype:
            raise ValueError(
                f"Tensor dtype mismatch for {name!r}: "
                f"A={value_a.dtype}, B={value_b.dtype}"
            )
        if value_a.layout != value_b.layout:
            raise ValueError(
                f"Tensor layout mismatch for {name!r}: "
                f"A={value_a.layout}, B={value_b.layout}"
            )

        if value_a.is_floating_point() or value_a.is_complex():
            if value_a.layout != torch.strided:
                raise ValueError(
                    f"Cannot interpolate non-strided tensor {name!r} "
                    f"with layout {value_a.layout}"
                )
            output[name] = torch.lerp(value_a, value_b, alpha)
        else:
            if not torch.equal(value_a, value_b):
                raise ValueError(
                    f"Non-floating tensor {name!r} differs between checkpoints"
                )
            output[name] = value_a.clone()
    return output


def interpolated_update(
    checkpoint_a: Mapping[str, Any],
    checkpoint_b: Mapping[str, Any],
    alpha: float,
) -> int:
    update_a = checkpoint_a.get("update")
    update_b = checkpoint_b.get("update")
    if (
        isinstance(update_a, bool)
        or not isinstance(update_a, int)
        or isinstance(update_b, bool)
        or not isinstance(update_b, int)
    ):
        raise TypeError(
            "Both checkpoints must have integer update metadata; "
            f"got A={update_a!r}, B={update_b!r}"
        )
    return round((1.0 - alpha) * update_a + alpha * update_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a slim PPO checkpoint whose parameters are "
            "(1 - alpha) * A + alpha * B. Optimizer states are omitted."
        )
    )
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Interpolation coefficient in [0, 1].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New checkpoint path; an existing file is never overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.alpha) or not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be finite and in [0, 1]")
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output}"
        )

    checkpoint_a = load_checkpoint(args.checkpoint_a, "A")
    checkpoint_b = load_checkpoint(args.checkpoint_b, "B")

    feature_version = require_equal_metadata(
        checkpoint_a, checkpoint_b, "feature_version"
    )
    if feature_version != PPO_FEATURE_VERSION:
        raise ValueError(
            f"Unsupported feature_version {feature_version!r}; "
            f"expected {PPO_FEATURE_VERSION!r}"
        )
    model_config = require_equal_metadata(
        checkpoint_a, checkpoint_b, "model_config"
    )
    if not isinstance(model_config, dict):
        raise TypeError("model_config must be a dict")

    deck_hash_a = checkpoint_deck_hash(checkpoint_a, "A")
    deck_hash_b = checkpoint_deck_hash(checkpoint_b, "B")
    if deck_hash_a != deck_hash_b:
        raise ValueError(
            f"Checkpoint deck hash mismatch: A={deck_hash_a}, B={deck_hash_b}"
        )

    passthrough = {
        key: require_equal_metadata(checkpoint_a, checkpoint_b, key)
        for key in REQUIRED_PASSTHROUGH_KEYS
    }
    state_a = checkpoint_a.get("model_state_dict")
    state_b = checkpoint_b.get("model_state_dict")
    if not isinstance(state_a, Mapping):
        raise TypeError("Checkpoint A model_state_dict must be a mapping")
    if not isinstance(state_b, Mapping):
        raise TypeError("Checkpoint B model_state_dict must be a mapping")
    interpolated_state = interpolate_state_dict(state_a, state_b, args.alpha)

    source_a = args.checkpoint_a.resolve()
    source_b = args.checkpoint_b.resolve()
    source_a_sha256 = sha256_file(args.checkpoint_a)
    source_b_sha256 = sha256_file(args.checkpoint_b)
    update = interpolated_update(checkpoint_a, checkpoint_b, args.alpha)
    stripped_keys = sorted(
        key for key in OPTIMIZER_KEYS
        if key in checkpoint_a or key in checkpoint_b
    )
    output_checkpoint: dict[str, Any] = {
        "feature_version": feature_version,
        **passthrough,
        "config": copy.deepcopy(checkpoint_a.get("config", {})),
        "model_config": model_config,
        "learner_deck_hash": deck_hash_a,
        "model_state_dict": interpolated_state,
        "update": update,
        "interpolation": {
            "formula": "(1-alpha)*A + alpha*B",
            "alpha": args.alpha,
            "checkpoint_a": str(source_a),
            "checkpoint_a_sha256": source_a_sha256,
            "checkpoint_a_update": checkpoint_a.get("update"),
            "checkpoint_b": str(source_b),
            "checkpoint_b_sha256": source_b_sha256,
            "checkpoint_b_update": checkpoint_b.get("update"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stripped_keys": stripped_keys,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)

    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
    reloaded_state = reloaded.get("model_state_dict")
    if not isinstance(reloaded_state, Mapping):
        raise RuntimeError("Saved checkpoint has no model_state_dict mapping")
    if reloaded_state.keys() != interpolated_state.keys():
        raise RuntimeError("Saved model_state_dict keys changed after reload")
    changed = [
        name for name in interpolated_state
        if isinstance(interpolated_state[name], torch.Tensor)
        and not torch.equal(interpolated_state[name], reloaded_state[name])
    ]
    if changed:
        raise RuntimeError(
            f"Saved checkpoint tensors changed after reload: {changed[:10]}"
        )

    summary = {
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "output_bytes": args.output.stat().st_size,
        "alpha": args.alpha,
        "update": update,
        "learner_deck_hash": deck_hash_a,
        "model_tensor_count": sum(
            isinstance(value, torch.Tensor)
            for value in interpolated_state.values()
        ),
        "optimizer_states_omitted": stripped_keys,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
