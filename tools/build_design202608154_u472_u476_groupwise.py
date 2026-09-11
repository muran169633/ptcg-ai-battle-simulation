#!/usr/bin/env python3
"""Build frozen group-wise U472-to-U476 interpolation candidates."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_A = ROOT / (
    "artifacts/ppo_guardedu468_updatedreplay_g8_ppo4x96_u472_"
    "freshjointactor6_s8_design202608148/ppo_stage/B_gold_league/"
    "seed-202608148/checkpoints/update-0472.pt"
)
SOURCE_B = ROOT / (
    "artifacts/ppo_u472_exactgold12_crossdeck_ppo4x96_u476_design202608152/"
    "B_gold_league/seed-202608152/checkpoints/update-0476.pt"
)
SOURCE_A_SHA256 = "c0436d54fff5e4da62d94c3002775c9e4b9ee848c5c59dd4422943cd71ad7018"
SOURCE_B_SHA256 = "efc107ff4a65243d540ccc2712c42b4680f725abcee717e41aeeee986a82fc09"
OUTPUT_DIR = ROOT / "artifacts/design202608154_u472_u476_groupwise_alpha025"
ALPHA = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_for(name: str) -> str | None:
    if name.startswith("transformer.layers.3") or name.startswith("transformer.norm"):
        return "trunk"
    if name.startswith(("actor_query", "actor_key", "actor_residual")):
        return "pointer"
    if name.startswith("count_head"):
        return "count"
    if name.startswith("value_head"):
        return "value"
    return None


def main() -> None:
    if sha256_file(SOURCE_A) != SOURCE_A_SHA256 or sha256_file(SOURCE_B) != SOURCE_B_SHA256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing existing output directory: {OUTPUT_DIR}")
    a = torch.load(SOURCE_A, map_location="cpu", weights_only=False)
    b = torch.load(SOURCE_B, map_location="cpu", weights_only=False)
    state_a = a["model_state_dict"]
    state_b = b["model_state_dict"]
    if list(state_a) != list(state_b):
        raise RuntimeError("model state schema mismatch")
    candidates = {
        "trunk025": {"trunk"},
        "heads025": {"pointer", "count"},
        "pointer025": {"pointer"},
        "count025": {"count"},
    }
    OUTPUT_DIR.mkdir(parents=True)
    manifest = {
        "schema_version": "ptcg-design202608154-u472-u476-groupwise-v1",
        "formula": "selected groups use (1-alpha)*U472 + alpha*U476; all other tensors use U472",
        "alpha": ALPHA,
        "source_a": {"path": str(SOURCE_A), "sha256": SOURCE_A_SHA256},
        "source_b": {"path": str(SOURCE_B), "sha256": SOURCE_B_SHA256},
        "candidates": {},
        "scope": {"evaluation_only": True, "optimizer_states_omitted": True, "submission": False},
    }
    passthrough = {
        key: copy.deepcopy(a[key])
        for key in (
            "feature_version", "bc_feature_version", "config", "model_config",
            "learner_deck_hash", "reward", "action_distribution",
        )
    }
    for label, selected_groups in candidates.items():
        output_state = {}
        changed_names = []
        for name, tensor_a in state_a.items():
            tensor_b = state_b[name]
            group = group_for(name)
            if group in selected_groups and (tensor_a.is_floating_point() or tensor_a.is_complex()):
                output_state[name] = torch.lerp(tensor_a, tensor_b, ALPHA)
                if not torch.equal(output_state[name], tensor_a):
                    changed_names.append(name)
            else:
                output_state[name] = tensor_a.detach().cpu().clone()
        output = {
            **passthrough,
            "model_state_dict": output_state,
            "update": 473,
            "groupwise_interpolation": {
                "alpha": ALPHA,
                "selected_groups": sorted(selected_groups),
                "changed_tensor_names": changed_names,
                "source_a_sha256": SOURCE_A_SHA256,
                "source_b_sha256": SOURCE_B_SHA256,
            },
        }
        path = OUTPUT_DIR / f"{label}.pt"
        torch.save(output, path)
        reloaded = torch.load(path, map_location="cpu", weights_only=True)
        for name in output_state:
            if not torch.equal(output_state[name], reloaded["model_state_dict"][name]):
                raise RuntimeError(f"reload mismatch for {label}:{name}")
        manifest["candidates"][label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "changed_tensor_count": len(changed_names),
            "selected_groups": sorted(selected_groups),
        }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
