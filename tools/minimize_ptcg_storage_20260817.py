#!/usr/bin/env python3
"""Reduce data/ and artifacts/ to the active Dragapult lineage only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"

KEEP_ARTIFACT_NAMES = {
    "frozen_incumbents",
    "ppo_mode_ar_dragapult_from_u40_parallel_gate5_u120_20260816_v1",
    "ppo_mode_ar_dragapult_from_u90_online_targeted_u12_20260817_v4",
}

KEEP_DATA_NAMES = {
    "decks",
    "recent_day_meta_pool_20260813_top23_v1",
    "retrain_top100_recent14_mode_ar_end0813_20260815_v1",
    "update90_live_episodes_20260817_v1",
    "update90_targeted_online_20260817_v1",
}

CRITICAL_FILES = {
    ARTIFACTS / "frozen_incumbents" / "dragapult_mode_ar_submit55527088_20260815_v1"
              / "submitted_policy_update0040.pt":
        "5c8e2659a9a6528202e4f24a75cb0058321584ba4160b0514aca59fb3ffb8e0e",
    ARTIFACTS / "ppo_mode_ar_dragapult_from_u40_parallel_gate5_u120_20260816_v1"
              / "champions" / "update-0090.pt":
        "f060322d60929b2abec0e3bc3b77eb3b0ee3adbd3f963ff8a438243d47f60e9a",
    ARTIFACTS / "ppo_mode_ar_dragapult_from_u40_parallel_gate5_u120_20260816_v1"
              / "checkpoints" / "update-0090.pt":
        "f060322d60929b2abec0e3bc3b77eb3b0ee3adbd3f963ff8a438243d47f60e9a",
    ARTIFACTS / "ppo_mode_ar_dragapult_from_u90_online_targeted_u12_20260817_v4"
              / "candidates" / "update-0004.pt":
        "1ba570f84edfde032a953c2111284a1817f1c05a20cc9d6bc886a17eaadfa48f",
}

CRITICAL_DATA = (
    DATA / "retrain_top100_recent14_mode_ar_end0813_20260815_v1"
         / "daily_parquet_v6_split90_10" / "manifest.json",
    DATA / "recent_day_meta_pool_20260813_top23_v1",
    DATA / "update90_live_episodes_20260817_v1",
    DATA / "update90_targeted_online_20260817_v1",
)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_critical() -> dict[str, str]:
    result: dict[str, str] = {}
    for path, expected in CRITICAL_FILES.items():
        if not path.is_file():
            raise RuntimeError(f"critical file missing: {relative(path)}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"critical SHA mismatch: {relative(path)}")
        result[relative(path)] = actual
    for path in CRITICAL_DATA:
        if not path.exists():
            raise RuntimeError(f"critical data missing: {relative(path)}")
    return result


def running_training_processes() -> list[str]:
    matches: list[str] = []
    own_pid = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == own_pid:
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        lowered = cmd.lower()
        if ("train_" in lowered and "ppo" in lowered) or "torchrun" in lowered:
            matches.append(f"pid={proc.name} {cmd.strip()}")
    return matches


def allocated_bytes(path: Path) -> int:
    try:
        total = path.lstat().st_blocks * 512
    except FileNotFoundError:
        return 0
    if not path.is_dir() or path.is_symlink():
        return total
    seen: set[tuple[int, int]] = set()
    for base, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            child = Path(base) / name
            try:
                info = child.lstat()
            except FileNotFoundError:
                continue
            identity = (info.st_dev, info.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            total += info.st_blocks * 512
    return total


def targets(root: Path, keep_names: set[str]) -> list[Path]:
    return sorted(
        (path for path in root.iterdir() if path.name not in keep_names),
        key=lambda path: path.name,
    )


def make_owner_deletable(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        for base, dirs, _files in os.walk(path, followlinks=False):
            for item in [Path(base), *(Path(base) / name for name in dirs)]:
                try:
                    item.chmod(item.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
                except FileNotFoundError:
                    pass
    try:
        path.parent.chmod(path.parent.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
    except FileNotFoundError:
        pass


def delete(path: Path) -> None:
    make_owner_deletable(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "storage_minimize_20260817.json",
    )
    args = parser.parse_args()

    hashes_before = verify_critical()
    running = running_training_processes()
    if args.execute and running:
        raise RuntimeError("refusing cleanup while training is running:\n" + "\n".join(running))

    artifact_targets = targets(ARTIFACTS, KEEP_ARTIFACT_NAMES)
    data_targets = targets(DATA, KEEP_DATA_NAMES)
    all_targets = artifact_targets + data_targets
    entries = [
        {
            "path": relative(path),
            "kind": "directory" if path.is_dir() else "file",
            "allocated_bytes": allocated_bytes(path),
        }
        for path in all_targets
    ]
    before = shutil.disk_usage(ROOT)

    if args.execute:
        for path in all_targets:
            delete(path)
        hashes_after = verify_critical()
    else:
        hashes_after = hashes_before

    after = shutil.disk_usage(ROOT)
    report = {
        "schema": "ptcg-storage-minimal-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry-run",
        "workspace": str(ROOT),
        "kept_artifacts": sorted(KEEP_ARTIFACT_NAMES),
        "kept_data": sorted(KEEP_DATA_NAMES),
        "critical_sha256_before": hashes_before,
        "critical_sha256_after": hashes_after,
        "summary": {
            "artifact_targets": len(artifact_targets),
            "data_targets": len(data_targets),
            "estimated_allocated_bytes": sum(x["allocated_bytes"] for x in entries),
            "disk_free_before": before.free,
            "disk_free_after": after.free,
            "observed_freed_bytes": after.free - before.free,
        },
        "targets": entries,
    }
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    temp = report_path.with_suffix(report_path.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temp.replace(report_path)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
