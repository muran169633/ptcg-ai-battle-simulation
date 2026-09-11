#!/usr/bin/env python3
"""Move one dated shard into another split without changing decision payloads.

The historical default still promotes a holdout date to ``train``.  ``--to-split``
also permits the latest sealed ``test`` date to become ``valid`` after an older
validation date has been promoted to training.  Other rows keep their split.
The output manifest is rebuilt from the actual rows so the archive remains
self-auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import prepare_bc_week as base


SPLITS = ("train", "valid", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--from-split", choices=("valid", "test"), default="test")
    parser.add_argument(
        "--to-split",
        choices=SPLITS,
        default="train",
        help="Destination split for the selected date (default: train)",
    )
    parser.add_argument("--frames-per-shard", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("--input and --output must differ")
    if args.from_split == args.to_split:
        raise ValueError("--from-split and --to-split must differ")
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    counts: Counter[str] = Counter()
    episodes: dict[str, set[str]] = {split: set() for split in SPLITS}
    dates: dict[str, set[str]] = {split: set() for split in SPLITS}
    promoted_rows = 0
    promoted_episodes: set[str] = set()

    with zipfile.ZipFile(source) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".jsonl") and name.split("/", 1)[0] in SPLITS
        )
        with base.DecisionArchiveWriter(
            output, args.frames_per_shard, overwrite=False
        ) as writer:
            for member in members:
                member_split = member.split("/", 1)[0]
                with archive.open(member) as rows:
                    for line_number, line in enumerate(rows, 1):
                        row: dict[str, Any] = json.loads(line)
                        row_split = str(row.get("split") or member_split)
                        dataset_date = str(row.get("dataset_date") or "")
                        episode_id = str(row.get("episode_id") or "")
                        if row_split != member_split:
                            raise RuntimeError(
                                f"{member}:{line_number}: row split {row_split!r} "
                                f"does not match member split {member_split!r}"
                            )
                        if dataset_date == args.date:
                            if row_split != args.from_split:
                                raise RuntimeError(
                                    f"{member}:{line_number}: date {args.date} was "
                                    f"found in {row_split}, expected {args.from_split}"
                                )
                            row_split = args.to_split
                            row["split"] = args.to_split
                            promoted_rows += 1
                            if episode_id:
                                promoted_episodes.add(episode_id)
                        writer.add(row_split, row)
                        counts[row_split] += 1
                        dates[row_split].add(dataset_date)
                        if episode_id:
                            episodes[row_split].add(episode_id)

            if promoted_rows <= 0:
                raise RuntimeError(
                    f"No rows dated {args.date} were found in {args.from_split}"
                )
            overlap = {
                f"{left}_{right}": len(episodes[left] & episodes[right])
                for index, left in enumerate(SPLITS)
                for right in SPLITS[index + 1 :]
            }
            if any(overlap.values()):
                raise RuntimeError(f"Episode overlap after promotion: {overlap}")

            ordered_dates = [
                value
                for split in SPLITS
                for value in sorted(dates[split])
            ]
            chronological = (
                ordered_dates == sorted(ordered_dates)
                and len(ordered_dates) == len(set(ordered_dates))
            )

            manifest["split_policy"] = {
                "mode": (
                    "time"
                    if chronological
                    else "latest_date_finetune_with_older_monitor"
                ),
                "train_dates": sorted(dates["train"]),
                "valid_dates": sorted(dates["valid"]),
                "test_dates": sorted(dates["test"]),
                "promoted_date": args.date,
                "promoted_from_split": args.from_split,
                "promoted_to_split": args.to_split,
                "reason": (
                    "User requested a strict latest-date validation stage followed "
                    "by consumption of that date in final BC/PPO training"
                ),
            }
            manifest["split_decisions"] = {
                split: counts[split] for split in SPLITS
            }
            manifest["split_episodes"] = {
                split: len(episodes[split]) for split in SPLITS
            }
            manifest["stats"] = {
                "decisions": sum(counts.values()),
                "episodes": len(set().union(*episodes.values())),
            }
            manifest["promotion_audit"] = {
                "date": args.date,
                "from_split": args.from_split,
                "to_split": args.to_split,
                "rows": promoted_rows,
                "episodes": len(promoted_episodes),
                "episode_overlap": overlap,
                "source_archive": str(source),
                "source_sha256": sha256_file(source),
            }
            manifest["shards"] = {
                split: writer.shard_index[split] for split in SPLITS
            }
            writer.finish(manifest)

    verified = base.verify_archive(output)
    result = {
        "output": str(output),
        "sha256": sha256_file(output),
        "promoted_rows": promoted_rows,
        "promoted_episodes": len(promoted_episodes),
        "split_decisions": verified["split_decisions"],
        "split_episodes": verified["split_episodes"],
        "split_policy": verified["split_policy"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
