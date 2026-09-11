#!/usr/bin/env python3
"""Convert one cached daily BC ZIP into one reusable all-train Parquet file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from convert_bc_zip_to_parquet import FEATURE_VERSION, SCHEMA, feature_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-group", type=int, default=2048)
    parser.add_argument("--hash-size", type=int, default=65_536)
    parser.add_argument("--max-state-entities", type=int, default=80)
    parser.add_argument("--compression", choices=("zstd", "snappy"), default="zstd")
    parser.add_argument("--compression-level", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if args.rows_per_group < 1:
        raise ValueError("--rows-per-group must be positive")

    output_dir.mkdir(parents=True)
    train_dir = output_dir / "train"
    train_dir.mkdir()
    output = train_dir / "part-00000.parquet"
    partial = output.with_suffix(".parquet.partial")
    input_rows = 0
    output_rows = 0
    rejected_rows = 0
    source_splits: Counter[str] = Counter()
    buffer: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    try:
        with zipfile.ZipFile(source) as archive:
            source_manifest = orjson.loads(archive.read("manifest.json"))
            members = sorted(
                name for name in archive.namelist() if name.endswith(".jsonl")
            )
            if not members:
                raise ValueError(f"No JSONL members found in {source}")
            for member_index, member in enumerate(members, 1):
                with archive.open(member) as handle:
                    for line in handle:
                        input_rows += 1
                        row = orjson.loads(line)
                        source_splits[str(row.get("split", ""))] += 1
                        record = feature_record(
                            row,
                            args.hash_size,
                            args.max_state_entities,
                        )
                        if record is None:
                            rejected_rows += 1
                            continue
                        record["split"] = "train"
                        buffer.append(record)
                        if len(buffer) >= args.rows_per_group:
                            table = pa.Table.from_pylist(buffer, schema=SCHEMA)
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    partial,
                                    SCHEMA,
                                    compression=args.compression,
                                    compression_level=args.compression_level,
                                    use_dictionary=True,
                                    write_statistics=True,
                                )
                            writer.write_table(table, row_group_size=args.rows_per_group)
                            output_rows += len(buffer)
                            buffer.clear()
                if member_index % 100 == 0 or member_index == len(members):
                    print(
                        f"members={member_index}/{len(members)} "
                        f"input_rows={input_rows:,} output_rows={output_rows:,}",
                        flush=True,
                    )
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(
                    partial,
                    SCHEMA,
                    compression=args.compression,
                    compression_level=args.compression_level,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(table, row_group_size=args.rows_per_group)
            output_rows += len(buffer)
            buffer.clear()
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No rows could be converted")
    os.replace(partial, output)

    parquet = pq.ParquetFile(output)
    if parquet.metadata.num_rows != output_rows:
        raise RuntimeError("Parquet row count mismatch")
    manifest = {
        "schema_version": "ptcg-bc-orbit-daily-alltrain-parquet-v1",
        "feature_version": FEATURE_VERSION,
        "storage": {
            "format": "parquet",
            "layout": "one immutable file per source date",
            "row_group_rows": args.rows_per_group,
            "compression": args.compression,
            "compression_level": args.compression_level,
            "pre_featurized": True,
        },
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "manifest": source_manifest,
            "original_split_rows": dict(source_splits),
        },
        "model_features": {
            "hash_size": args.hash_size,
            "max_state_entities": args.max_state_entities,
        },
        "split_policy": {
            "mode": "all_train",
            "train_dates": source_manifest.get("dates", []),
            "valid_dates": [],
            "test_dates": [],
        },
        "split_decisions": {"train": output_rows},
        "stats": {
            "input_rows": input_rows,
            "output_rows": output_rows,
            "rejected_rows": rejected_rows,
            "bytes": output.stat().st_size,
            "files": 1,
        },
        "files": [
            {
                "output": str(output),
                "rows": output_rows,
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "row_groups": parquet.metadata.num_row_groups,
            }
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
