#!/usr/bin/env python3
"""Validate the portable package before starting an expensive PPO run."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpus", type=int, default=8)
    parser.add_argument("--skip-engine-smoke", action="store_true")
    args = parser.parse_args()

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
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        raise RuntimeError("PyTorch NCCL distributed backend is unavailable")

    # Import all runtime adapters before allocating a long training job.  This
    # catches a package that contains the DDP launcher but omitted the local
    # process-parallel rollout implementation.
    import parallel_rollout  # noqa: F401
    import train_nonar_league_ppo_ddp  # noqa: F401

    bc_path = (
        ROOT
        / "artifacts"
        / "bc_top100_recent14_nonar_order_v7_end0813_b2048_20260816_v1"
        / "best.pt"
    )
    checkpoint = torch.load(bc_path, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_version") != "ptcg-bc-orbit-nonar-v7-ordered-rank":
        raise RuntimeError("Unexpected BC feature version")
    if int(checkpoint.get("epoch", -1)) < 1:
        raise RuntimeError("BC checkpoint epoch metadata is invalid")

    engine_smoke = "skipped"
    if not args.skip_engine_smoke:
        from train_ppo import RawBattle, read_deck

        deck = read_deck(
            ROOT
            / "data"
            / "recent_day_meta_pool_20260813_top23_v1"
            / "decks"
            / "rank02_07bedfffbfad.csv"
        )
        battle = RawBattle(deck, deck)
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
        "nccl": torch.cuda.nccl.version(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        "bc_sha256": sha256(bc_path),
        "bc_epoch": checkpoint.get("epoch"),
        "engine_smoke": engine_smoke,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
