#!/usr/bin/env python3
"""Compose a deployment BC checkpoint from audited parameter blocks.

The ``transformer_actor`` composition keeps the stable source checkpoint for
all encoders, embeddings, count parameters, and value parameters.  It copies
only the complete Transformer stack and the six actor tensors from a learned
overlay checkpoint.  The output is deployment-only and cannot be resumed as a
training checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import interpolate_bc_checkpoints as checkpoint_utils  # noqa: E402


COMPOSITION = "transformer_actor"
ACTOR_KEYS = frozenset(
    {
        "actor_query.weight",
        "actor_key.weight",
        "actor_residual.0.weight",
        "actor_residual.0.bias",
        "actor_residual.2.weight",
        "actor_residual.2.bias",
    }
)
FORMULA = "A except transformer.* and actor6 copied bitwise from B"


def selected_by_transformer_actor(name: str) -> bool:
    return name.startswith("transformer.") or name in ACTOR_KEYS


def require_optional_expected(
    *,
    label: str,
    actual: int,
    expected: int | None,
) -> None:
    if expected is None:
        return
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise TypeError(f"{label} must be a non-bool integer")
    if expected < 1:
        raise ValueError(f"{label} must be positive")
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected}, got {actual}")


def compose_transformer_actor_state(
    state_a: Mapping[str, Any],
    state_b: Mapping[str, Any],
    *,
    expected_selected_tensors: int | None = None,
    expected_selected_parameters: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    # alpha=0 performs the shared strict key/shape/dtype/layout/finite audit
    # and gives us independent clones of every source tensor.
    output = checkpoint_utils.interpolate_state_dict(state_a, state_b, 0.0)
    missing_actor = sorted(ACTOR_KEYS - set(output))
    if missing_actor:
        raise KeyError(f"Missing required actor tensors: {missing_actor}")

    selected_keys = sorted(name for name in output if selected_by_transformer_actor(name))
    transformer_keys = [name for name in selected_keys if name.startswith("transformer.")]
    actor_keys = [name for name in selected_keys if name in ACTOR_KEYS]
    if not transformer_keys:
        raise ValueError("No transformer.* tensors were selected")
    if set(actor_keys) != ACTOR_KEYS:
        raise RuntimeError("Actor selection did not resolve to the exact actor6 set")

    for name in selected_keys:
        overlay = state_b[name]
        if not isinstance(overlay, torch.Tensor):
            raise TypeError(f"Overlay state entry {name!r} is not a tensor")
        output[name] = overlay.clone()

    selected_parameters = sum(output[name].numel() for name in selected_keys)
    transformer_parameters = sum(output[name].numel() for name in transformer_keys)
    actor_parameters = sum(output[name].numel() for name in actor_keys)
    total_parameters = sum(value.numel() for value in output.values())
    complement_keys = sorted(set(output) - set(selected_keys))
    complement_parameters = total_parameters - selected_parameters

    require_optional_expected(
        label="expected_selected_tensors",
        actual=len(selected_keys),
        expected=expected_selected_tensors,
    )
    require_optional_expected(
        label="expected_selected_parameters",
        actual=selected_parameters,
        expected=expected_selected_parameters,
    )

    for name in selected_keys:
        if not torch.equal(output[name], state_b[name]):
            raise RuntimeError(f"Selected tensor {name!r} is not bitwise equal to B")
    for name in complement_keys:
        if not torch.equal(output[name], state_a[name]):
            raise RuntimeError(f"Complement tensor {name!r} changed from A")

    audit = {
        "composition": COMPOSITION,
        "selected_tensor_count": len(selected_keys),
        "selected_parameter_count": selected_parameters,
        "transformer_tensor_count": len(transformer_keys),
        "transformer_parameter_count": transformer_parameters,
        "actor_tensor_count": len(actor_keys),
        "actor_parameter_count": actor_parameters,
        "complement_tensor_count": len(complement_keys),
        "complement_parameter_count": complement_parameters,
        "total_tensor_count": len(output),
        "total_parameter_count": total_parameters,
        "selected_keys": selected_keys,
        "actor_keys": sorted(ACTOR_KEYS),
    }
    return output, audit


def create_composed_checkpoint(
    checkpoint_a_path: Path,
    checkpoint_b_path: Path,
    output_path: Path,
    *,
    allow_epoch_mismatch: bool = False,
    expected_selected_tensors: int | None = None,
    expected_selected_parameters: int | None = None,
) -> dict[str, Any]:
    checkpoint_a_path = Path(checkpoint_a_path)
    checkpoint_b_path = Path(checkpoint_b_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    checkpoint_a = checkpoint_utils.load_checkpoint(checkpoint_a_path, "A")
    checkpoint_b = checkpoint_utils.load_checkpoint(checkpoint_b_path, "B")
    checkpoint_utils.require_bc_feature_version(checkpoint_a, "A")
    checkpoint_utils.require_bc_feature_version(checkpoint_b, "B")

    config_a, config_source_a = checkpoint_utils.runtime_model_config(checkpoint_a, "A")
    config_b, config_source_b = checkpoint_utils.runtime_model_config(checkpoint_b, "B")
    if config_a != config_b:
        differing = {
            key: {"A": config_a[key], "B": config_b[key]}
            for key in checkpoint_utils.RUNTIME_MODEL_CONFIG_KEYS
            if config_a[key] != config_b[key]
        }
        raise ValueError("Runtime model config mismatch: " + json.dumps(differing, sort_keys=True))

    deck_hash_a = checkpoint_utils.single_deck_hash(checkpoint_a, "A")
    deck_hash_b = checkpoint_utils.single_deck_hash(checkpoint_b, "B")
    if deck_hash_a != deck_hash_b:
        raise ValueError(f"Checkpoint deck hash mismatch: A={deck_hash_a!r}, B={deck_hash_b!r}")
    ppo_interface = checkpoint_utils.shared_ppo_interface(checkpoint_a, checkpoint_b)
    epoch = checkpoint_utils.shared_epoch(
        checkpoint_a,
        checkpoint_b,
        allow_epoch_mismatch=allow_epoch_mismatch,
    )

    state_a = checkpoint_a.get("model_state_dict")
    state_b = checkpoint_b.get("model_state_dict")
    if not isinstance(state_a, Mapping):
        raise TypeError("Checkpoint A model_state_dict must be a mapping")
    if not isinstance(state_b, Mapping):
        raise TypeError("Checkpoint B model_state_dict must be a mapping")
    composed_state, block_audit = compose_transformer_actor_state(
        state_a,
        state_b,
        expected_selected_tensors=expected_selected_tensors,
        expected_selected_parameters=expected_selected_parameters,
    )

    source_a = checkpoint_a_path.resolve()
    source_b = checkpoint_b_path.resolve()
    minimal_config: dict[str, Any] = {**config_a, "deck_hashes": [deck_hash_a]}
    output_checkpoint: dict[str, Any] = {
        "feature_version": checkpoint_utils.bc.FEATURE_VERSION,
        "config": minimal_config,
        "model_state_dict": composed_state,
        "epoch": epoch,
        "valid_metrics": None,
        "ppo_interface": copy.deepcopy(ppo_interface),
        "block_composition": {
            "schema_version": "ptcg-bc-block-composition-v1",
            "formula": FORMULA,
            "checkpoint_a": str(source_a),
            "checkpoint_a_sha256": checkpoint_utils.sha256_file(checkpoint_a_path),
            "checkpoint_a_epoch": checkpoint_a["epoch"],
            "checkpoint_b": str(source_b),
            "checkpoint_b_sha256": checkpoint_utils.sha256_file(checkpoint_b_path),
            "checkpoint_b_epoch": checkpoint_b["epoch"],
            "epoch_mismatch_allowed": allow_epoch_mismatch,
            "output_epoch_policy": "max_source_epoch_metadata_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_provenance": {
                "strategy": "validated_shared_runtime_fields_only",
                "runtime_model_keys": list(checkpoint_utils.RUNTIME_MODEL_CONFIG_KEYS),
                "deck_hashes": "validated_equal_singleton",
                "checkpoint_a_model_config_source": config_source_a,
                "checkpoint_b_model_config_source": config_source_b,
            },
            **block_audit,
            "optimizer_state_omitted": True,
            "resume_training": False,
            "deployment_only": True,
        },
    }
    checkpoint_utils.atomic_save_checkpoint(output_checkpoint, composed_state, output_path)
    return {
        "output": str(output_path.resolve()),
        "output_sha256": checkpoint_utils.sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "epoch": epoch,
        "deck_hash": deck_hash_a,
        **block_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deployment BC checkpoint using source A except for the "
            "complete Transformer stack and actor6 tensors copied from B."
        )
    )
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-epoch-mismatch", action="store_true")
    parser.add_argument("--expected-selected-tensors", type=int)
    parser.add_argument("--expected-selected-parameters", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_composed_checkpoint(
        args.checkpoint_a,
        args.checkpoint_b,
        args.output,
        allow_epoch_mismatch=args.allow_epoch_mismatch,
        expected_selected_tensors=args.expected_selected_tensors,
        expected_selected_parameters=args.expected_selected_parameters,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
