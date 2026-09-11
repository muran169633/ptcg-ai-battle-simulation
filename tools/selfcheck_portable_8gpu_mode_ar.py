#!/usr/bin/env python3
"""Validate the frozen mode-AR update40 portable package before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

FROZEN_REL = Path(
    "artifacts/frozen_incumbents/"
    "dragapult_mode_ar_submit55527088_20260815_v1"
)
EXPECTED_ANCHOR_SHA256 = (
    "5c8e2659a9a6528202e4f24a75cb0058321584ba4160b0514aca59fb3ffb8e0e"
)
EXPECTED_BC_SHA256 = (
    "377f71c3153f1261c9b1879611352cd978c610e10251405d7f672dbafe498eb7"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpus", type=int, default=8)
    parser.add_argument("--skip-engine-smoke", action="store_true")
    args = parser.parse_args()
    if args.require_gpus < 0:
        raise ValueError("--require-gpus must be non-negative")

    manifest_path = ROOT / "PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for row in manifest.get("files", []):
        path = ROOT / row["path"]
        if not path.is_file():
            mismatches.append({"path": row["path"], "error": "missing"})
            continue
        actual = sha256(path)
        if actual != row["sha256"]:
            mismatches.append(
                {
                    "path": row["path"],
                    "error": "sha256",
                    "expected": row["sha256"],
                    "actual": actual,
                }
            )
    if mismatches:
        raise RuntimeError("Package integrity failure: " + json.dumps(mismatches))
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Bundled official engine requires Linux x86_64")
    gpu_count = torch.cuda.device_count()
    if gpu_count < args.require_gpus:
        raise RuntimeError(
            f"Need at least {args.require_gpus} visible GPUs, found {gpu_count}"
        )
    if args.require_gpus > 1 and (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        raise RuntimeError("PyTorch NCCL distributed backend is unavailable")

    import train_mode_ar_ppo as mode_ar
    import parallel_rollout  # noqa: F401
    import train_mode_ar_ppo_ddp as mode_ar_ddp  # noqa: F401

    anchor_path = ROOT / FROZEN_REL / "submitted_policy_update0040.pt"
    bc_path = ROOT / FROZEN_REL / "bc_base_best.pt"
    deck_path = ROOT / FROZEN_REL / "deck.csv"
    if sha256(anchor_path) != EXPECTED_ANCHOR_SHA256:
        raise RuntimeError("Submitted update40 SHA-256 mismatch")
    if sha256(bc_path) != EXPECTED_BC_SHA256:
        raise RuntimeError("Original mode-AR BC SHA-256 mismatch")
    checkpoint, model, model_config = mode_ar.load_anchor(
        anchor_path,
        torch.device("cpu"),
    )
    if checkpoint.get("feature_version") != mode_ar.FEATURE_VERSION:
        raise RuntimeError("Unexpected submitted policy feature version")
    if int(checkpoint.get("update", -1)) != 40:
        raise RuntimeError("Portable anchor is not submitted update40")
    bc_anchor = checkpoint.get("bc_anchor") or {}
    if bc_anchor.get("sha256") != EXPECTED_BC_SHA256:
        raise RuntimeError("Submitted policy does not reference the frozen BC")
    del model

    meta_path = (
        ROOT
        / "data"
        / "recent_day_meta_pool_20260813_top23_v1"
        / "meta_pool.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_decks = []
    for row in meta.get("opponents", []):
        candidate = Path(str(row["deck_path"]))
        candidate = candidate if candidate.is_absolute() else ROOT / candidate
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        meta_decks.append(candidate)
    if len(meta_decks) != 23:
        raise RuntimeError(f"Expected 23 recent-meta decks, found {len(meta_decks)}")

    engine_smoke = "skipped"
    if not args.skip_engine_smoke:
        deck = mode_ar.legacy.read_deck(deck_path)
        battle = mode_ar.legacy.RawBattle(deck, deck)
        try:
            if not isinstance(battle.observation, dict):
                raise RuntimeError("Official engine returned a non-dict observation")
            engine_smoke = "passed"
        finally:
            battle.close()

    result = {
        "status": "passed",
        "package_schema": manifest.get("schema_version"),
        "verified_files": len(manifest.get("files", [])),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        "anchor_sha256": sha256(anchor_path),
        "anchor_update": checkpoint.get("update"),
        "bc_sha256": sha256(bc_path),
        "model_config": model_config,
        "meta_decks": len(meta_decks),
        "engine_smoke": engine_smoke,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
