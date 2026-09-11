#!/usr/bin/env python3
"""Conservative storage cleanup for the 2026-08-17 PTCG training lineage.

Dry-run is the default.  --execute permanently deletes only the explicitly
listed old data trees and intermediate artifact checkpoint files.  Historical
best.pt files, logs, submissions, and the active Dragapult lineage are kept.
"""

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

ACTIVE_U90_RUN = ARTIFACTS / "ppo_mode_ar_dragapult_from_u40_parallel_gate5_u120_20260816_v1"
ACTIVE_U4_RUN = ARTIFACTS / "ppo_mode_ar_dragapult_from_u90_online_targeted_u12_20260817_v4"
FROZEN_RUN = ARTIFACTS / "frozen_incumbents" / "dragapult_mode_ar_submit55527088_20260815_v1"

CRITICAL_FILES = {
    FROZEN_RUN / "submitted_policy_update0040.pt":
        "5c8e2659a9a6528202e4f24a75cb0058321584ba4160b0514aca59fb3ffb8e0e",
    ACTIVE_U90_RUN / "champions" / "update-0090.pt":
        "f060322d60929b2abec0e3bc3b77eb3b0ee3adbd3f963ff8a438243d47f60e9a",
    ACTIVE_U90_RUN / "checkpoints" / "update-0090.pt":
        "f060322d60929b2abec0e3bc3b77eb3b0ee3adbd3f963ff8a438243d47f60e9a",
    ACTIVE_U4_RUN / "candidates" / "update-0004.pt":
        "1ba570f84edfde032a953c2111284a1817f1c05a20cc9d6bc886a17eaadfa48f",
}

CRITICAL_DATA = (
    DATA / "retrain_top100_recent14_mode_ar_end0813_20260815_v1"
           / "daily_parquet_v6_split90_10" / "manifest.json",
    DATA / "recent_day_meta_pool_20260813_top23_v1",
    DATA / "update90_live_episodes_20260817_v1",
    DATA / "update90_targeted_online_20260817_v1",
)

# These are superseded raw replay caches, Top50 datasets, or earlier derived
# datasets.  Current Mode-AR Top100 parquet and current opponent/eval pools are
# intentionally absent from this list.
OLD_DATA_TREES = tuple(DATA / name for name in (
    "episodes_cache",
    "retrain_top50_recent14_20260812_v1",
    "retrain_top100_recent14_alltrain_20260813_v1",
    "retrain_top50_recent9_20260813_v1",
    "retrain_top50_recent14_alltrain_20260813_v1",
    "retrain_top50_recent7_fresh_20260812_v1",
    "top100_proxy_recent14_20260811_v1",
    "retrain_top50_recent7_20260812_v2",
    "gold_push_recent7_20260810_v1",
    "retrain_top50_recent14_alltrain_20260814_v2",
    "gold8_recent7_20260808",
    "public_counter_policies_20260814_v1",
    "gold_push_marnie_top20union14_20260811_v1",
    "live_sixth_sense_dragapult_55439076_20260812_v2",
    "live_sixth_sense_dragapult_55439076_20260812_v1",
    "retrain_top50_recent14_strict_20260813_v1",
    "retrain_top100_refresh_20260814_v1",
    "gold_push_marnie_top50_recent7_20260811_v1",
    "yanz_alakazam_ppo_dual_anchor_20260810_v1",
    "yanz_alakazam_live_20260810_v1",
    "gold_league",
    "yanz_alakazam_actor6_pcgrad_specialbc_20260810_v2_deployment_policyboundary",
))

INTERMEDIATE_DIR_NAMES = {"checkpoints", "candidates", "policies", "snapshots"}


def under(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=True))
        return True
    except (FileNotFoundError, ValueError):
        return False


def allocated_bytes(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_file() or path.is_symlink():
        return path.lstat().st_blocks * 512
    total = path.lstat().st_blocks * 512
    for base, dirs, files in os.walk(path, followlinks=False):
        base_path = Path(base)
        for name in dirs + files:
            child = base_path / name
            try:
                total += child.lstat().st_blocks * 512
            except FileNotFoundError:
                pass
    return total


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def verify_critical() -> dict[str, str]:
    results: dict[str, str] = {}
    for path, expected in CRITICAL_FILES.items():
        if not path.is_file():
            raise RuntimeError(f"critical file is missing: {relative(path)}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"critical SHA mismatch: {relative(path)} expected={expected} actual={actual}"
            )
        results[relative(path)] = actual
    for path in CRITICAL_DATA:
        if not path.exists():
            raise RuntimeError(f"critical data is missing: {relative(path)}")
    return results


def running_training_processes() -> list[str]:
    matches: list[str] = []
    self_pid = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == self_pid:
            continue
        try:
            raw = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        lowered = raw.lower()
        if ("train_" in lowered and "ppo" in lowered) or "torchrun" in lowered:
            matches.append(f"pid={proc.name} {raw.strip()}")
    return matches


def artifact_intermediates() -> list[Path]:
    targets: list[Path] = []
    preserved_u90_checkpoint = ACTIVE_U90_RUN / "checkpoints" / "update-0090.pt"
    for path in ARTIFACTS.rglob("*.pt"):
        if not path.is_file():
            continue
        if under(path, FROZEN_RUN) or under(path, ACTIVE_U4_RUN):
            continue
        if path == preserved_u90_checkpoint:
            continue
        relative_parts = path.relative_to(ARTIFACTS).parts
        if any(part in INTERMEDIATE_DIR_NAMES for part in relative_parts[:-1]):
            targets.append(path)
    return sorted(targets)


def validate_targets(paths: list[Path]) -> None:
    critical = set(CRITICAL_FILES)
    for path in paths:
        if path in critical:
            raise RuntimeError(f"refusing to delete critical path: {relative(path)}")
        if not (under(path, ARTIFACTS) or under(path, DATA)):
            raise RuntimeError(f"target escaped allowed roots: {path}")
        if path.resolve(strict=False) in {ROOT.resolve(), ARTIFACTS.resolve(), DATA.resolve()}:
            raise RuntimeError(f"refusing broad target: {path}")


def remove_empty_intermediate_dirs() -> int:
    removed = 0
    dirs = [p for p in ARTIFACTS.rglob("*") if p.is_dir() and p.name in INTERMEDIATE_DIR_NAMES]
    for path in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def make_owner_deletable(path: Path) -> None:
    """Add owner write/search bits only where deletion requires them."""
    parent = path if path.is_dir() else path.parent
    while under(parent, ARTIFACTS) or under(parent, DATA):
        try:
            mode = parent.stat().st_mode
            parent.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
        except FileNotFoundError:
            pass
        if parent in {ARTIFACTS, DATA}:
            break
        parent = parent.parent


def make_tree_owner_deletable(path: Path) -> None:
    if not path.exists():
        return
    make_owner_deletable(path)
    for base, dirs, _files in os.walk(path, followlinks=False):
        base_path = Path(base)
        make_owner_deletable(base_path)
        for name in dirs:
            child = base_path / name
            try:
                mode = child.stat().st_mode
                child.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
            except FileNotFoundError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="permanently delete selected files")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "storage_cleanup_20260817.json",
    )
    args = parser.parse_args()

    critical_before = verify_critical()
    running = running_training_processes()
    if args.execute and running:
        raise RuntimeError("refusing cleanup while training is running:\n" + "\n".join(running))

    artifact_files = artifact_intermediates()
    data_trees = [path for path in OLD_DATA_TREES if path.exists()]
    targets = artifact_files + data_trees
    validate_targets(targets)

    entries = [
        {
            "path": relative(path),
            "kind": "directory" if path.is_dir() else "file",
            "allocated_bytes": allocated_bytes(path),
        }
        for path in targets
    ]
    estimated = sum(entry["allocated_bytes"] for entry in entries)
    disk_before = shutil.disk_usage(ROOT)

    if args.execute:
        for path in artifact_files:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                make_owner_deletable(path)
                path.unlink(missing_ok=True)
        for path in data_trees:
            make_tree_owner_deletable(path)
            shutil.rmtree(path)
        empty_dirs_removed = remove_empty_intermediate_dirs()
        critical_after = verify_critical()
    else:
        empty_dirs_removed = 0
        critical_after = critical_before

    disk_after = shutil.disk_usage(ROOT)
    report = {
        "schema": "ptcg-storage-cleanup-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry-run",
        "workspace": str(ROOT),
        "training_processes": running,
        "critical_sha256_before": critical_before,
        "critical_sha256_after": critical_after,
        "critical_data": [relative(path) for path in CRITICAL_DATA],
        "summary": {
            "artifact_files": len(artifact_files),
            "data_trees": len(data_trees),
            "estimated_allocated_bytes": estimated,
            "empty_intermediate_dirs_removed": empty_dirs_removed,
            "disk_free_before": disk_before.free,
            "disk_free_after": disk_after.free,
            "observed_freed_bytes": disk_after.free - disk_before.free,
        },
        "targets": entries,
    }
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
