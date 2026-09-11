#!/usr/bin/env python3
"""Create a weighted average of two architecture-compatible BC checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--weight-b", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.weight_b <= 1.0:
        raise ValueError("--weight-b must be in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    a_path = args.checkpoint_a.resolve()
    b_path = args.checkpoint_b.resolve()
    checkpoint_a = torch.load(a_path, map_location="cpu", weights_only=False)
    checkpoint_b = torch.load(b_path, map_location="cpu", weights_only=False)
    if checkpoint_a.get("feature_version") != checkpoint_b.get("feature_version"):
        raise ValueError("feature_version mismatch")
    state_a = checkpoint_a.get("model_state_dict")
    state_b = checkpoint_b.get("model_state_dict")
    if not isinstance(state_a, dict) or not isinstance(state_b, dict):
        raise TypeError("Both checkpoints must contain model_state_dict mappings")
    if state_a.keys() != state_b.keys():
        raise ValueError("model_state_dict keys mismatch")

    weight_b = float(args.weight_b)
    weight_a = 1.0 - weight_b
    averaged: dict[str, torch.Tensor] = {}
    for name in state_a:
        tensor_a = state_a[name]
        tensor_b = state_b[name]
        if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
            raise TypeError(f"{name}: state value is not a tensor")
        if tensor_a.shape != tensor_b.shape or tensor_a.dtype != tensor_b.dtype:
            raise ValueError(f"{name}: tensor shape/dtype mismatch")
        if tensor_a.is_floating_point():
            averaged[name] = (
                tensor_a.to(torch.float64) * weight_a
                + tensor_b.to(torch.float64) * weight_b
            ).to(tensor_a.dtype)
        else:
            if not torch.equal(tensor_a, tensor_b):
                raise ValueError(f"{name}: non-floating tensors differ")
            averaged[name] = tensor_a.clone()

    output = copy.deepcopy(checkpoint_b)
    output["model_state_dict"] = averaged
    output["epoch"] = None
    output["valid_metrics"] = None
    output["bc_checkpoint_average"] = {
        "schema_version": "ptcg-bc-checkpoint-average-v1",
        "formula": "(1-weight_b)*checkpoint_a + weight_b*checkpoint_b",
        "weight_a": weight_a,
        "weight_b": weight_b,
        "checkpoint_a": {"path": str(a_path), "sha256": sha256_file(a_path)},
        "checkpoint_b": {"path": str(b_path), "sha256": sha256_file(b_path)},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_training": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    for name, tensor in averaged.items():
        if not torch.equal(reloaded["model_state_dict"][name], tensor):
            raise RuntimeError(f"{name}: saved tensor changed")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "weight_a": weight_a,
                "weight_b": weight_b,
                "tensor_count": len(averaged),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
