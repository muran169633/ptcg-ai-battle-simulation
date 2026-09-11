#!/usr/bin/env python3
"""Interpolate two architecture-compatible PTCG BC checkpoints safely."""

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


ARCHITECTURE_KEYS = (
    "categorical_dim",
    "model_dim",
    "layers",
    "heads",
    "dropout",
    "hash_size",
    "max_state_entities",
    "entity_fields",
    "option_fields",
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
        raise TypeError(f"Checkpoint {label} must contain a dict")
    return checkpoint


def require_equal(checkpoint_a: Mapping[str, Any], checkpoint_b: Mapping[str, Any], key: str) -> Any:
    if key not in checkpoint_a or key not in checkpoint_b:
        raise KeyError(f"Both checkpoints must contain {key!r}")
    if checkpoint_a[key] != checkpoint_b[key]:
        raise ValueError(
            f"Checkpoint metadata mismatch for {key!r}: "
            f"A={checkpoint_a[key]!r}, B={checkpoint_b[key]!r}"
        )
    return copy.deepcopy(checkpoint_a[key])


def architecture(checkpoint: Mapping[str, Any], label: str) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise TypeError(f"Checkpoint {label} config must be a mapping")
    missing = [key for key in ARCHITECTURE_KEYS if key not in config]
    if missing:
        raise KeyError(f"Checkpoint {label} is missing architecture keys: {missing}")
    return {key: config[key] for key in ARCHITECTURE_KEYS}


def interpolate_state(
    state_a: Mapping[str, Any], state_b: Mapping[str, Any], alpha: float
) -> tuple[dict[str, Any], list[str]]:
    if set(state_a) != set(state_b):
        raise ValueError("model_state_dict keys differ")
    output: dict[str, Any] = {}
    differing: list[str] = []
    for name in state_a:
        value_a = state_a[name]
        value_b = state_b[name]
        if not isinstance(value_a, torch.Tensor) or not isinstance(value_b, torch.Tensor):
            if type(value_a) is not type(value_b) or value_a != value_b:
                raise ValueError(f"Non-tensor state entry {name!r} differs")
            output[name] = copy.deepcopy(value_a)
            continue
        if value_a.shape != value_b.shape or value_a.dtype != value_b.dtype:
            raise ValueError(f"Tensor schema mismatch for {name!r}")
        if torch.equal(value_a, value_b):
            output[name] = value_a.clone()
        elif value_a.is_floating_point() or value_a.is_complex():
            if value_a.layout != torch.strided or value_b.layout != torch.strided:
                raise ValueError(f"Cannot interpolate non-strided tensor {name!r}")
            output[name] = torch.lerp(value_a, value_b, alpha)
            differing.append(name)
        else:
            raise ValueError(f"Non-floating tensor {name!r} differs")
    return output, differing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create (1-alpha)*A + alpha*B from compatible BC checkpoints."
    )
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--expected-differing-tensors", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.alpha) or not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be finite and in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    checkpoint_a = load_checkpoint(args.checkpoint_a, "A")
    checkpoint_b = load_checkpoint(args.checkpoint_b, "B")
    feature_version = require_equal(checkpoint_a, checkpoint_b, "feature_version")
    if not isinstance(feature_version, str) or not feature_version.startswith("ptcg-bc-"):
        raise ValueError(f"Unsupported BC feature_version: {feature_version!r}")
    if architecture(checkpoint_a, "A") != architecture(checkpoint_b, "B"):
        raise ValueError("Checkpoint architectures differ")
    ppo_interface = require_equal(checkpoint_a, checkpoint_b, "ppo_interface")

    state_a = checkpoint_a.get("model_state_dict")
    state_b = checkpoint_b.get("model_state_dict")
    if not isinstance(state_a, Mapping) or not isinstance(state_b, Mapping):
        raise TypeError("Both checkpoints must contain model_state_dict mappings")
    state, differing = interpolate_state(state_a, state_b, args.alpha)
    if (
        args.expected_differing_tensors is not None
        and len(differing) != args.expected_differing_tensors
    ):
        raise ValueError(
            "Differing tensor count mismatch: "
            f"observed={len(differing)} expected={args.expected_differing_tensors}"
        )

    output = {
        "feature_version": feature_version,
        "config": copy.deepcopy(checkpoint_a["config"]),
        "epoch": checkpoint_a.get("epoch"),
        "valid_metrics": {},
        "ppo_interface": ppo_interface,
        "model_state_dict": state,
        "evaluation_only": True,
        "resume_forbidden": True,
        "interpolation": {
            "formula": "(1-alpha)*A + alpha*B",
            "alpha": args.alpha,
            "checkpoint_a": str(args.checkpoint_a.resolve()),
            "checkpoint_a_sha256": sha256_file(args.checkpoint_a),
            "checkpoint_b": str(args.checkpoint_b.resolve()),
            "checkpoint_b_sha256": sha256_file(args.checkpoint_b),
            "differing_tensors": differing,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)

    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
    reloaded_state = reloaded.get("model_state_dict")
    if not isinstance(reloaded_state, Mapping) or reloaded_state.keys() != state.keys():
        raise RuntimeError("Saved checkpoint state schema changed after reload")
    changed = [name for name in state if not torch.equal(state[name], reloaded_state[name])]
    if changed:
        raise RuntimeError(f"Saved tensors changed after reload: {changed[:10]}")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "alpha": args.alpha,
                "differing_tensor_count": len(differing),
                "differing_tensors": differing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
