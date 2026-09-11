#!/usr/bin/env python3
"""Stream BC archives into one archive with a chronological split.

Every input date must occur in exactly one archive.  Existing row-level split
labels are ignored.  By default, the latest date becomes test, the penultimate
date becomes valid, and every earlier date becomes train.  With
``--latest-as-valid``, the latest date becomes valid, every earlier date becomes
train, and test is empty.
"""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO

import orjson


SPLITS = ("train", "valid", "test")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ArchiveInput:
    path: Path
    manifest: dict[str, Any]
    dates: tuple[str, ...]


def load_input(path: Path) -> ArchiveInput:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "manifest.json" not in names:
            raise ValueError(f"{path}: missing manifest.json")
        if not any(name.endswith(".jsonl") for name in names):
            raise ValueError(f"{path}: no JSONL decision shards")
        manifest = orjson.loads(archive.read("manifest.json"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest.json must contain an object")
    raw_dates = manifest.get("dates")
    if not isinstance(raw_dates, list) or not raw_dates:
        raise ValueError(f"{path}: manifest dates must be a non-empty list")
    if len(raw_dates) != len(set(raw_dates)):
        raise ValueError(f"{path}: manifest contains duplicate dates")
    for value in raw_dates:
        if not isinstance(value, str):
            raise ValueError(f"{path}: non-string date in manifest: {value!r}")
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{path}: invalid ISO date {value!r}") from error
    return ArchiveInput(path, manifest, tuple(sorted(raw_dates)))


def zip_info(name: str, compression: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


class ArchiveWriter:
    def __init__(
        self,
        output: Path,
        rows_per_shard: int,
        compression: int,
        overwrite: bool,
    ) -> None:
        self.output = output.resolve()
        self.partial = self.output.with_suffix(self.output.suffix + ".partial")
        self.rows_per_shard = rows_per_shard
        self.compression = compression
        self.overwrite = overwrite
        self.archive: zipfile.ZipFile | None = None
        self.handle: BinaryIO | None = None
        self.current_split: str | None = None
        self.current_rows = 0
        self.shards: Counter[str] = Counter()

    def __enter__(self) -> "ArchiveWriter":
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists() and not self.overwrite:
            raise FileExistsError(
                f"{self.output} already exists; pass --overwrite to replace it"
            )
        if self.partial.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"{self.partial} already exists; remove it or pass --overwrite"
                )
            self.partial.unlink()
        self.archive = zipfile.ZipFile(
            self.partial,
            mode="w",
            compression=self.compression,
            compresslevel=6 if self.compression == zipfile.ZIP_DEFLATED else None,
            allowZip64=True,
        )
        return self

    def _close_shard(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.current_split = None
        self.current_rows = 0

    def write(self, split: str, payload: bytes) -> None:
        assert self.archive is not None
        if (
            self.handle is None
            or self.current_split != split
            or self.current_rows >= self.rows_per_shard
        ):
            self._close_shard()
            member = f"{split}/part-{self.shards[split]:05d}.jsonl"
            self.handle = self.archive.open(
                zip_info(member, self.compression),
                mode="w",
                force_zip64=True,
            )
            self.current_split = split
            self.current_rows = 0
            self.shards[split] += 1
        self.handle.write(payload)
        self.current_rows += 1

    def finish(self, manifest: dict[str, Any]) -> None:
        assert self.archive is not None
        self._close_shard()
        payload = (
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        self.archive.writestr(zip_info("manifest.json", self.compression), payload)
        self.archive.close()
        self.archive = None
        os.replace(self.partial, self.output)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._close_shard()
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        if exc_type is not None and self.partial.exists():
            self.partial.unlink()


def member_sort_key(info: zipfile.ZipInfo) -> tuple[int, str]:
    prefix = info.filename.split("/", 1)[0]
    order = {"train": 0, "valid": 1, "test": 2}.get(prefix, 3)
    return order, info.filename


def canonical_team_display_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def merged_team_filter(inputs: list[ArchiveInput]) -> dict[str, Any] | None:
    """Return a canonical merged filter when every input declares its scope.

    Older archives may omit ``team_filter``.  In that case ``None`` preserves
    the historical behavior of retaining the base manifest's value.
    """

    filters = [item.manifest.get("team_filter") for item in inputs]
    if any(
        isinstance(team_filter, dict)
        and team_filter.get("all_teams") is True
        for team_filter in filters
    ):
        return {
            "all_teams": True,
            "global_team_count": 0,
            "dated_team_counts": {},
            "display_names": [],
        }
    if not all(
        isinstance(team_filter, dict)
        and team_filter.get("all_teams") is False
        for team_filter in filters
    ):
        return None

    representatives: dict[str, str] = {}
    for item, team_filter in zip(inputs, filters):
        assert isinstance(team_filter, dict)
        display_names = team_filter.get("display_names")
        if not isinstance(display_names, list):
            raise ValueError(
                f"{item.path}: team_filter.display_names must be a list"
            )
        for value in display_names:
            if not isinstance(value, str):
                raise ValueError(
                    f"{item.path}: team_filter.display_names contains "
                    "a non-string value"
                )
            display = canonical_team_display_name(value)
            normalized = display.casefold()
            if not normalized:
                continue
            previous = representatives.get(normalized)
            if previous is None or display < previous:
                representatives[normalized] = display
    names = [representatives[key] for key in sorted(representatives)]
    return {
        "all_teams": False,
        "global_team_count": len(names),
        "dated_team_counts": {},
        "display_names": names,
    }


def merge(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [load_input(args.base)] + [
        load_input(path) for path in args.supplement
    ]
    base_input = inputs[0]
    team_filter = merged_team_filter(inputs)
    input_paths = [item.path for item in inputs]
    if len(input_paths) != len(set(input_paths)):
        raise ValueError("Each input archive may be specified only once")

    output = args.output.resolve()
    if output in set(input_paths):
        raise ValueError("--output must differ from every input archive")

    expected_schema = inputs[0].manifest.get("schema_version")
    if not isinstance(expected_schema, str) or not expected_schema:
        raise ValueError("Base manifest has no valid schema_version")
    manifest_hashes = {
        item.manifest.get("deck_hash_filter")
        for item in inputs
        if item.manifest.get("deck_hash_filter") is not None
    }
    if len(manifest_hashes) > 1:
        raise ValueError(
            f"Input manifests have inconsistent deck_hash_filter values: "
            f"{sorted(manifest_hashes)}"
        )
    expected_deck_hash = next(iter(manifest_hashes), None)

    owners: dict[str, Path] = {}
    for item in inputs:
        schema = item.manifest.get("schema_version")
        if schema != expected_schema:
            raise ValueError(
                f"{item.path}: schema_version {schema!r} != {expected_schema!r}"
            )
        for dataset_date in item.dates:
            if dataset_date in owners:
                raise ValueError(
                    f"Date {dataset_date} overlaps between "
                    f"{owners[dataset_date]} and {item.path}"
                )
            owners[dataset_date] = item.path

    all_dates = sorted(owners)
    latest_as_valid = bool(getattr(args, "latest_as_valid", False))
    minimum_dates = 2 if latest_as_valid else 3
    if len(all_dates) < minimum_dates:
        if latest_as_valid:
            raise ValueError(
                "Latest-as-valid time split requires at least two distinct dates"
            )
        raise ValueError("Strict time split requires at least three distinct dates")
    if latest_as_valid:
        train_dates = all_dates[:-1]
        valid_dates = all_dates[-1:]
        test_dates: list[str] = []
    else:
        train_dates = all_dates[:-2]
        valid_dates = all_dates[-2:-1]
        test_dates = all_dates[-1:]
    date_to_split = {
        **{dataset_date: "train" for dataset_date in train_dates},
        **{dataset_date: "valid" for dataset_date in valid_dates},
        **{dataset_date: "test" for dataset_date in test_dates},
    }

    inputs.sort(key=lambda item: (item.dates[0], item.dates, str(item.path)))
    compression = zipfile.ZIP_DEFLATED if args.compress else zipfile.ZIP_STORED
    split_counts: Counter[str] = Counter()
    episode_ids_by_split: dict[str, set[str]] = {
        split: set() for split in SPLITS
    }
    team_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    total_rows = 0

    with ArchiveWriter(
        output,
        args.rows_per_shard,
        compression,
        args.overwrite,
    ) as writer:
        for item in inputs:
            allowed_dates = set(item.dates)
            print(f"Reading {item.path} ({', '.join(item.dates)})", flush=True)
            with zipfile.ZipFile(item.path) as source:
                members = sorted(
                    (
                        info
                        for info in source.infolist()
                        if info.filename.endswith(".jsonl")
                    ),
                    key=member_sort_key,
                )
                for member in members:
                    with source.open(member) as handle:
                        for line_number, line in enumerate(handle, 1):
                            if not line.strip():
                                continue
                            try:
                                row = orjson.loads(line)
                            except orjson.JSONDecodeError as error:
                                raise ValueError(
                                    f"{item.path}:{member.filename}:{line_number}: "
                                    "invalid JSON"
                                ) from error
                            if not isinstance(row, dict):
                                raise ValueError(
                                    f"{item.path}:{member.filename}:{line_number}: "
                                    "decision row is not an object"
                                )
                            if row.get("schema_version") != expected_schema:
                                raise ValueError(
                                    f"{item.path}:{member.filename}:{line_number}: "
                                    "row schema_version is inconsistent"
                                )
                            row_hash = row.get("deck_hash")
                            if not isinstance(row_hash, str) or not row_hash:
                                raise ValueError(
                                    f"{item.path}:{member.filename}:{line_number}: "
                                    "row has no valid deck_hash"
                                )
                            if expected_deck_hash is None:
                                expected_deck_hash = row_hash
                            elif row_hash != expected_deck_hash:
                                raise ValueError(
                                    f"{item.path}:{member.filename}:{line_number}: "
                                    f"deck_hash {row_hash!r} != "
                                    f"{expected_deck_hash!r}"
                                )
                            dataset_date = row.get("dataset_date")
                            if dataset_date not in allowed_dates:
                                raise ValueError(
                                    f"{item.path}:{member.filename}:{line_number}: "
                                    f"dataset_date {dataset_date!r} is not declared "
                                    "by this archive manifest"
                                )
                            episode_id = row.get("episode_id")
                            if (
                                not isinstance(episode_id, (str, int))
                                or isinstance(episode_id, bool)
                                or not str(episode_id).strip()
                            ):
                                raise ValueError(
                                    f"{item.path}:{member.filename}:{line_number}: "
                                    "row has no valid episode_id"
                                )
                            try:
                                hash(episode_id)
                            except TypeError as error:
                                raise ValueError(
                                    f"{item.path}:{member.filename}:{line_number}: "
                                    "row episode_id is not hashable"
                                ) from error
                            split = date_to_split[dataset_date]
                            row["split"] = split
                            writer.write(split, orjson.dumps(row) + b"\n")
                            split_counts[split] += 1
                            episode_ids_by_split[split].add(str(episode_id))
                            team_counts[str(row.get("team_name", ""))] += 1
                            context_counts[str(row.get("select_context", ""))] += 1
                            total_rows += 1
                            if total_rows % 100_000 == 0:
                                print(
                                    f"  merged {total_rows:,} decisions",
                                    flush=True,
                                )

        if total_rows == 0:
            raise RuntimeError("Input archives contain no decision rows")

        sources = [
            source
            for item in inputs
            for source in item.manifest.get("sources", [])
            if isinstance(source, dict)
        ]
        sources.sort(key=lambda value: (value.get("date", ""), value.get("path", "")))
        manifest = {
            **base_input.manifest,
            "dates": all_dates,
            "split_policy": {
                "mode": "time",
                "train_dates": train_dates,
                "valid_dates": valid_dates,
                "test_dates": test_dates,
            },
            "stats": {
                "decisions": total_rows,
                "input_archives": len(inputs),
            },
            "split_decisions": {
                split: split_counts[split] for split in SPLITS
            },
            "split_episodes": {
                split: len(episode_ids_by_split[split]) for split in SPLITS
            },
            "team_decisions": dict(team_counts.most_common()),
            "context_decisions": dict(context_counts.most_common()),
            "shards": {split: writer.shards[split] for split in SPLITS},
            "sources": sources,
            "merged_from": [
                {
                    "path": str(item.path),
                    "dates": list(item.dates),
                }
                for item in inputs
            ],
            "deck_hash_filter": expected_deck_hash,
            "rows_per_shard": args.rows_per_shard,
            "compression": "deflated" if args.compress else "stored",
        }
        if team_filter is not None:
            manifest["team_filter"] = team_filter
        writer.finish(manifest)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a base BC ZIP and supplemental non-overlapping date ZIPs "
            "into a chronological BC archive."
        )
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--supplement",
        type=Path,
        action="append",
        required=True,
        help="Supplemental archive; repeat this option for multiple archives",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=25_000)
    parser.add_argument(
        "--latest-as-valid",
        action="store_true",
        help=(
            "Put the latest date in valid and all earlier dates in train, "
            "leaving test empty (default: latest=test, penultimate=valid)"
        ),
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Deflate output members (default stores them without recompression)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output after a successful merge",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.rows_per_shard <= 0:
        parser.error("--rows-per-shard must be positive")
    manifest = merge(args)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "dates": manifest["dates"],
                "split_decisions": manifest["split_decisions"],
                "shards": manifest["shards"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
