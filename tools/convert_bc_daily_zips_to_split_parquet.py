#!/usr/bin/env python3
"""Convert filtered daily BC ZIPs into a reusable 90/10 Parquet dataset.

Every source ZIP is read exactly once.  The source archive's historical split
label is ignored and each episode is deterministically assigned with a hash of
``seed:dataset_date:episode_id``.  This keeps both seats and all decisions from
one episode together, while ensuring every date contributes to training and
validation.  Raw action order is preserved by the v6 featurizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

import convert_bc_zip_to_parquet as converter


PARQUET_SCHEMA_VERSION = "ptcg-bc-daily-hash-features-parquet-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_split(
    dataset_date: str,
    episode_id: str,
    seed: int,
    train_fraction: float,
) -> str:
    digest = hashlib.sha256(
        f"{seed}:{dataset_date}:{episode_id}".encode()
    ).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "train" if value < train_fraction else "valid"


def archive_date(source: Path) -> tuple[str, dict[str, Any]]:
    with zipfile.ZipFile(source) as archive:
        manifest = orjson.loads(archive.read("manifest.json"))
    dates = manifest.get("dates")
    if not isinstance(dates, list) or len(dates) != 1:
        raise ValueError(f"daily archive must contain exactly one date: {source}")
    return str(dates[0]), manifest


def canonical_team_filter(manifest: dict[str, Any]) -> tuple[str, ...]:
    team_filter = manifest.get("team_filter")
    if not isinstance(team_filter, dict):
        raise ValueError("source manifest is missing team_filter")
    names = team_filter.get("display_names")
    if not isinstance(names, list) or not names:
        raise ValueError("source manifest has no filtered team names")
    canonical = tuple(
        sorted(" ".join(str(name).split()).casefold() for name in names)
    )
    if len(canonical) != len(set(canonical)):
        raise ValueError("source manifest team filter contains duplicates")
    return canonical


def make_writer(
    partial: Path,
    compression: str,
    compression_level: int,
) -> pq.ParquetWriter:
    partial.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        raise FileExistsError(partial)
    return pq.ParquetWriter(
        partial,
        converter.SCHEMA,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=True,
        write_statistics=True,
    )


def convert_daily(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        source_text,
        output_text,
        dataset_date,
        seed,
        train_fraction,
        hash_size,
        max_entities,
        rows_per_group,
        compression,
        compression_level,
    ) = task
    source = Path(source_text)
    output_dir = Path(output_text)
    output_paths = {
        split: output_dir / split / f"{dataset_date}.parquet"
        for split in ("train", "valid")
    }
    partial_paths = {
        split: path.with_suffix(path.suffix + ".partial")
        for split, path in output_paths.items()
    }
    for path in (*output_paths.values(), *partial_paths.values()):
        if path.exists():
            raise FileExistsError(path)

    writers = {
        split: make_writer(
            partial_paths[split], compression, compression_level
        )
        for split in ("train", "valid")
    }
    buffers: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    split_rows: Counter[str] = Counter()
    context_rows: dict[str, Counter[str]] = {
        "train": Counter(),
        "valid": Counter(),
    }
    split_episodes: dict[str, set[str]] = {"train": set(), "valid": set()}
    input_rows = rejected_rows = 0

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        table = pa.Table.from_pylist(buffers[split], schema=converter.SCHEMA)
        writers[split].write_table(table, row_group_size=rows_per_group)
        buffers[split].clear()

    try:
        with zipfile.ZipFile(source) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.endswith(".jsonl")
            )
            if not members:
                raise ValueError(f"no JSONL members in {source}")
            for member in members:
                with archive.open(member) as handle:
                    for line in handle:
                        input_rows += 1
                        row = orjson.loads(line)
                        row_date = str(row.get("dataset_date", ""))
                        if row_date != dataset_date:
                            raise ValueError(
                                f"row date mismatch in {source}: "
                                f"{row_date!r} != {dataset_date!r}"
                            )
                        episode_id = str(row.get("episode_id", ""))
                        if not episode_id:
                            raise ValueError(f"empty episode_id in {source}")
                        split = episode_split(
                            dataset_date,
                            episode_id,
                            seed,
                            train_fraction,
                        )
                        record = converter.feature_record(
                            row, hash_size, max_entities
                        )
                        if record is None:
                            rejected_rows += 1
                            continue
                        record["split"] = split
                        buffers[split].append(record)
                        split_rows[split] += 1
                        split_episodes[split].add(episode_id)
                        context_rows[split][str(record["context"])] += 1
                        if len(buffers[split]) >= rows_per_group:
                            flush(split)
        flush("train")
        flush("valid")
    finally:
        for writer in writers.values():
            writer.close()

    leakage = split_episodes["train"] & split_episodes["valid"]
    if leakage:
        raise RuntimeError(
            f"episode leakage for {dataset_date}: {sorted(leakage)[:5]}"
        )
    for split in ("train", "valid"):
        if split_rows[split] == 0:
            raise RuntimeError(f"{dataset_date} has zero {split} rows")
        os.replace(partial_paths[split], output_paths[split])
        parquet_rows = pq.ParquetFile(output_paths[split]).metadata.num_rows
        if parquet_rows != split_rows[split]:
            raise RuntimeError(
                f"Parquet row mismatch {output_paths[split]}: "
                f"{parquet_rows} != {split_rows[split]}"
            )

    output_rows = sum(split_rows.values())
    if input_rows != output_rows + rejected_rows:
        raise RuntimeError("input/output/rejected row conservation failed")
    return {
        "dataset_date": dataset_date,
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "rejected_rows": rejected_rows,
        "split_rows": dict(split_rows),
        "split_episodes": {
            split: len(values) for split, values in split_episodes.items()
        },
        "episode_leakage": 0,
        "context_rows": {
            split: dict(values) for split, values in context_rows.items()
        },
        "outputs": {
            split: {
                "path": str(output_paths[split]),
                "bytes": output_paths[split].stat().st_size,
                "sha256": sha256_file(output_paths[split]),
                "row_groups": pq.ParquetFile(
                    output_paths[split]
                ).metadata.num_row_groups,
            }
            for split in ("train", "valid")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rows-per-group", type=int, default=4096)
    parser.add_argument("--hash-size", type=int, default=65_536)
    parser.add_argument("--max-state-entities", type=int, default=80)
    parser.add_argument("--split-seed", type=int, default=20260815)
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--compression", choices=("zstd", "snappy"), default="zstd")
    parser.add_argument("--compression-level", type=int, default=6)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.rows_per_group < 1:
        raise ValueError("workers and rows-per-group must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    sources = [path.resolve() for path in args.input]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    date_manifests: dict[str, dict[str, Any]] = {}
    source_by_date: dict[str, Path] = {}
    for source in sources:
        dataset_date, manifest = archive_date(source)
        if dataset_date in source_by_date:
            raise ValueError(f"duplicate source date: {dataset_date}")
        source_by_date[dataset_date] = source
        date_manifests[dataset_date] = manifest
    team_filters = {
        dataset_date: canonical_team_filter(manifest)
        for dataset_date, manifest in date_manifests.items()
    }
    reference_date = min(team_filters)
    reference_filter = team_filters[reference_date]
    mismatched_filters = [
        dataset_date
        for dataset_date, team_filter in team_filters.items()
        if team_filter != reference_filter
    ]
    if mismatched_filters:
        raise ValueError(
            "daily sources do not share one frozen team filter: "
            f"{mismatched_filters}"
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    tasks = [
        (
            str(source_by_date[dataset_date]),
            str(output_dir),
            dataset_date,
            args.split_seed,
            args.train_fraction,
            args.hash_size,
            args.max_state_entities,
            args.rows_per_group,
            args.compression,
            args.compression_level,
        )
        for dataset_date in sorted(source_by_date)
    ]
    results: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(convert_daily, task): task[2] for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"date={result['dataset_date']} "
                    f"train={result['split_rows']['train']:,} "
                    f"valid={result['split_rows']['valid']:,} "
                    f"rejected={result['rejected_rows']:,}",
                    flush=True,
                )
    except Exception:
        (output_dir / "FAILED").write_text(
            "Conversion failed; this directory is incomplete.\n",
            encoding="utf-8",
        )
        raise

    results.sort(key=lambda value: value["dataset_date"])
    dates = [value["dataset_date"] for value in results]
    split_decisions = {
        split: sum(value["split_rows"][split] for value in results)
        for split in ("train", "valid")
    }
    manifest = {
        "schema_version": PARQUET_SCHEMA_VERSION,
        "feature_version": converter.FEATURE_VERSION,
        "dates": dates,
        "split_policy": {
            "mode": "daily_episode_hash",
            "seed": args.split_seed,
            "hash_key": "seed:dataset_date:episode_id",
            "fractions": {
                "train": args.train_fraction,
                "valid": 1.0 - args.train_fraction,
                "test": 0.0,
            },
            "episode_disjoint": True,
        },
        "split_decisions": split_decisions,
        "input_rows": sum(value["input_rows"] for value in results),
        "output_rows": sum(value["output_rows"] for value in results),
        "rejected_rows": sum(value["rejected_rows"] for value in results),
        "action_sequence_policy": "preserve exact replay action list",
        "daily_results": results,
        "source_team_filters": {
            dataset_date: date_manifests[dataset_date].get("team_filter")
            for dataset_date in dates
        },
        "frozen_team_filter": {
            "team_count": len(reference_filter),
            "normalized_names_sha256": hashlib.sha256(
                "\n".join(reference_filter).encode()
            ).hexdigest(),
            "reference_date": reference_date,
            "identical_across_all_dates": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if (output_dir / "FAILED").exists():
        (output_dir / "FAILED").unlink()
    print(json.dumps(split_decisions, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
