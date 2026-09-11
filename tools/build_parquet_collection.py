#!/usr/bin/env python3
"""Build a reusable manifest over independently cached daily Parquet datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--team-snapshot", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not args.team_snapshot.is_file():
        raise FileNotFoundError(args.team_snapshot)

    children: list[dict[str, Any]] = []
    dates: set[str] = set()
    total_rows = 0
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("feature_version") != args.feature_version:
            raise ValueError(
                f"{manifest_path}: feature version mismatch: "
                f"{manifest.get('feature_version')!r}"
            )
        source_manifest = (manifest.get("source") or {}).get("manifest") or {}
        child_dates = source_manifest.get("dates")
        if not isinstance(child_dates, list) or len(child_dates) != 1:
            raise ValueError(f"{manifest_path}: expected exactly one source date")
        dataset_date = str(child_dates[0])
        if dataset_date in dates:
            raise ValueError(f"duplicate cached date: {dataset_date}")
        dates.add(dataset_date)
        rows = int((manifest.get("stats") or {}).get("output_rows", -1))
        if rows < 1:
            raise ValueError(f"{manifest_path}: invalid output row count {rows}")
        parquet_files = sorted(manifest_path.parent.rglob("*.parquet"))
        if len(parquet_files) != int((manifest.get("stats") or {}).get("files", -1)):
            raise ValueError(f"{manifest_path}: Parquet file count drift")
        total_rows += rows
        children.append(
            {
                "date": dataset_date,
                "directory": str(manifest_path.parent),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "source_archive": (manifest.get("source") or {}).get("path"),
                "source_archive_sha256": (manifest.get("source") or {}).get("sha256"),
                "rows": rows,
                "parquet_files": len(parquet_files),
                "parquet_bytes": sum(path.stat().st_size for path in parquet_files),
            }
        )

    if not children:
        raise ValueError(f"No daily Parquet manifests found under {root}")
    children.sort(key=lambda row: row["date"])
    fingerprint_payload = {
        "feature_version": args.feature_version,
        "team_snapshot_sha256": sha256_file(args.team_snapshot),
        "children": [
            {
                "date": row["date"],
                "manifest_sha256": row["manifest_sha256"],
                "source_archive_sha256": row["source_archive_sha256"],
            }
            for row in children
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    collection = {
        "schema_version": "ptcg-bc-parquet-daily-collection-v1",
        "feature_version": args.feature_version,
        "storage": {
            "format": "parquet",
            "layout": "daily_cache/date/source_split/shard.parquet",
            "reuse": "immutable daily caches; rebuild only changed dates",
        },
        "dates": [row["date"] for row in children],
        "split_policy": {
            "mode": "all_train",
            "train_dates": [row["date"] for row in children],
            "valid_dates": [],
            "test_dates": [],
        },
        "split_decisions": {"train": total_rows},
        "stats": {
            "output_rows": total_rows,
            "dates": len(children),
            "files": sum(row["parquet_files"] for row in children),
            "bytes": sum(row["parquet_bytes"] for row in children),
        },
        "team_snapshot": {
            "path": str(args.team_snapshot.resolve()),
            "sha256": sha256_file(args.team_snapshot),
        },
        "children": children,
        "cache_fingerprint": fingerprint,
    }
    output = root / "manifest.json"
    temporary = root / "manifest.json.partial"
    temporary.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(collection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
