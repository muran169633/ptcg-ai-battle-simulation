#!/usr/bin/env python3
"""Build and publish the frozen train-only replay cache identity for design202608147.

This profiler performs no model construction, forward pass, optimizer step,
checkpoint write, evaluation, network access, packaging, or submission.  It
authenticates the updated archive and the held U468 run configuration, then
uses the repository's exact PPO replay-cache builder on ``train/`` members.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
RUN_CONFIG = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141/"
    "ppo_stage/block3/B_gold_league/seed-202608141/run_config.json"
)
RUN_CONFIG_SHA256 = "1de013a90d6ad6b05669d6da24bbcf36ead4a9c9c6cb8875713a16a8a6a2c867"
TRAIN_PPO = TOOLS / "train_ppo.py"
TRAIN_PPO_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
CACHE_HELPER = TOOLS / "run_ppo_bc_repair.py"
CACHE_HELPER_SHA256 = "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"
SEED = 202608147
SELECTED_BATCH_INDICES = (1, 3, 4, 5, 6, 7, 8, 11)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, digest: str, label: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    actual = sha256_file(path)
    if actual != digest:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual}")


def archive_layout(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate member names")
        train = [name for name in names if name.startswith("train/") and name.endswith(".jsonl")]
        valid = [name for name in names if name.startswith("valid/") and name.endswith(".jsonl")]
        test = [name for name in names if name.startswith("test/") and name.endswith(".jsonl")]
        if not train:
            raise RuntimeError("archive has no train shards")
        manifest = json.loads(archive.read("manifest.json"))
    split = manifest.get("split_policy", {})
    if split.get("test_dates", []) != [] or int(manifest.get("shards", {}).get("test", 0)) != 0:
        raise RuntimeError("updated replay must have an empty test split")
    if int(manifest.get("split_decisions", {}).get("train", 0)) < 18_432:
        raise RuntimeError("updated replay has too few train rows for the cache")
    rolled_output = manifest.get("roll_forward", {}).get("output", {})
    train_dates = split.get("train_dates") or rolled_output.get("train", {}).get("dataset_dates", [])
    valid_dates = split.get("valid_dates") or rolled_output.get("valid", {}).get("dataset_dates", [])
    test_dates = split.get("test_dates", [])
    return {
        "train_members": train,
        "valid_members": valid,
        "test_members": test,
        "train_rows": int(manifest["split_decisions"]["train"]),
        "valid_rows": int(manifest["split_decisions"]["valid"]),
        "train_episodes": int(manifest["split_episodes"]["train"]),
        "valid_episodes": int(manifest["split_episodes"]["valid"]),
        "train_dates": list(train_dates),
        "valid_dates": list(valid_dates),
        "test_dates": list(test_dates),
        "deck_hash_filter": manifest.get("deck_hash_filter"),
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"wrong Python runtime: {sys.executable}")
    archive = args.archive.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    if output.parent != (ROOT / "artifacts").resolve():
        raise RuntimeError("profile output must be directly under artifacts/")
    require_file(RUN_CONFIG, RUN_CONFIG_SHA256, "guarded U468 run config")
    require_file(TRAIN_PPO, TRAIN_PPO_SHA256, "PPO trainer")
    require_file(CACHE_HELPER, CACHE_HELPER_SHA256, "cache hash helper")
    require_file(archive, args.expected_archive_sha256, "updated replay")
    layout = archive_layout(archive)
    expected_deck = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
    if layout["deck_hash_filter"] != expected_deck:
        raise RuntimeError("updated replay deck identity mismatch")

    sys.path.insert(0, str(TOOLS))
    import torch  # noqa: PLC0415
    import train_ppo as ppo  # noqa: PLC0415
    import run_ppo_bc_repair as cache_audit  # noqa: PLC0415

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before train-only cache construction")
    run_config = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    raw_config = run_config["config"]
    model_config = run_config["model_config"]
    if set(raw_config) != set(ppo.PPOConfig.__dataclass_fields__):
        raise RuntimeError("held PPO config field set drifted")
    cloned = copy.deepcopy(raw_config)
    cloned["bc_replay_data"] = str(archive)
    cloned["seed"] = SEED
    changed = sorted(key for key in cloned if cloned[key] != raw_config[key])
    if changed != ["bc_replay_data", "seed"]:
        raise RuntimeError(f"unexpected config changes: {changed}")
    config = ppo.PPOConfig(**cloned)

    train_index = {name: index for index, name in enumerate(layout["train_members"])}
    allowed = frozenset(train_index)
    opened_bits = multiprocessing.get_context().Array(
        "b", len(train_index), lock=True
    )
    original_open = zipfile.ZipFile.open

    def guarded_open(self: zipfile.ZipFile, name: Any, mode: str = "r", *pos: Any, **kw: Any) -> Any:
        member = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        if mode != "r" or member not in allowed:
            raise RuntimeError(f"cache builder attempted forbidden member: {member}")
        with opened_bits.get_lock():
            opened_bits[train_index[member]] = 1
        return original_open(self, name, mode, *pos, **kw)

    zipfile.ZipFile.open = guarded_open
    try:
        replay_batches = ppo.build_bc_replay_batches(config, model_config)
    finally:
        zipfile.ZipFile.open = original_open
    cache_sha256, batch_sha256 = cache_audit.replay_cache_manifest(replay_batches)
    rows = [int(batch["contexts"].shape[0]) for batch in replay_batches]
    context34 = [int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum()) for batch in replay_batches]
    if len(replay_batches) != 72 or set(rows) != {256} or set(context34) != {4}:
        raise RuntimeError("replay cache shape or context-34 quota drifted")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during cache-only profiling")

    result = {
        "schema_version": "ptcg-design202608147-updated-replay-cache-profile-v1",
        "status": "CACHE_PROFILE_COMPLETE_ZERO_STEP",
        "design_id": "design202608147",
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_initialized": False,
        },
        "archive": {
            "path": str(archive.relative_to(ROOT)),
            "sha256": args.expected_archive_sha256,
            **layout,
            "opened_train_member_count": sum(int(value) for value in opened_bits),
            "non_train_members_opened": False,
        },
        "config": {
            "seed": SEED,
            "shuffle_seed": SEED + 41,
            "stratification_seed": SEED + 73,
            "batches": 72,
            "batch_size": 256,
            "rows": sum(rows),
            "workers": config.bc_replay_workers,
            "context34_rows_per_batch": 4,
        },
        "cache": {
            "aggregate_sha256": cache_sha256,
            "batch_sha256": batch_sha256,
            "selected_batch_indices": list(SELECTED_BATCH_INDICES),
            "selected_batch_sha256": [batch_sha256[index] for index in SELECTED_BATCH_INDICES],
            "per_batch_rows": rows,
            "per_batch_context34_rows": context34,
        },
        "scope": {
            "optimizer_steps": 0,
            "model_instances": 0,
            "model_forwards": 0,
            "checkpoint_writes": 0,
            "validation_or_test_rows_opened": 0,
            "network_upload_package_submission": False,
        },
    }
    write_exclusive(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
