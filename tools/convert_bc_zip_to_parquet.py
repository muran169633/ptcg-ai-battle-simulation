#!/usr/bin/env python3
"""Convert a sharded BC JSONL ZIP into streaming, featurized Parquet shards.

The output stores exactly the tensors consumed by ``train_bc_orbit.py`` plus
the small amount of row metadata required for filtering and audit.  Raw battle
observations are deliberately not duplicated, so repeated training runs avoid
JSON parsing and feature construction while keeping bounded row-group memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

_feature_module = importlib.import_module(
    os.environ.get("PTCG_BC_FEATURE_MODULE", "train_bc_orbit")
)
ENTITY_NUMERIC_SIZE = _feature_module.ENTITY_NUMERIC_SIZE
FEATURE_VERSION = _feature_module.FEATURE_VERSION
GLOBAL_NUMERIC_SIZE = _feature_module.GLOBAL_NUMERIC_SIZE
OPTION_NUMERIC_SIZE = _feature_module.OPTION_NUMERIC_SIZE
featurize_row = _feature_module.featurize_row


PARQUET_SCHEMA_VERSION = "ptcg-bc-orbit-features-parquet-v1"
SCHEMA = pa.schema(
    [
        ("global_fields", pa.list_(pa.int32())),
        ("global_numeric", pa.list_(pa.float32(), GLOBAL_NUMERIC_SIZE)),
        ("entity_fields", pa.list_(pa.list_(pa.int32()))),
        (
            "entity_numeric",
            pa.list_(pa.list_(pa.float32(), ENTITY_NUMERIC_SIZE)),
        ),
        ("option_fields", pa.list_(pa.list_(pa.int32()))),
        (
            "option_numeric",
            pa.list_(pa.list_(pa.float32(), OPTION_NUMERIC_SIZE)),
        ),
        ("targets", pa.list_(pa.float32())),
        ("action_count", pa.int16()),
        ("action_sequence", pa.list_(pa.int16())),
        ("min_count", pa.int16()),
        ("max_count", pa.int16()),
        ("context", pa.int16()),
        ("sample_weight", pa.float32()),
        ("win_target", pa.float32()),
        ("split", pa.string()),
        ("episode_id", pa.string()),
        ("dataset_date", pa.string()),
        ("deck_hash", pa.string()),
        ("team_name", pa.string()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_record(row: dict[str, Any], hash_size: int, max_entities: int) -> dict[str, Any] | None:
    features = featurize_row(row, hash_size, max_entities)
    if features is None:
        return None
    features.update(
        {
            "split": str(row.get("split", "")),
            "episode_id": str(row.get("episode_id", "")),
            "dataset_date": str(row.get("dataset_date", "")),
            "deck_hash": str(row.get("deck_hash", "")),
            "team_name": str(row.get("team_name", "")),
        }
    )
    return features


def convert_member(task: tuple[str, str, str, int, int, int, str, int]) -> dict[str, Any]:
    (
        source_text,
        member,
        output_text,
        hash_size,
        max_entities,
        rows_per_group,
        compression,
        compression_level,
    ) = task
    source = Path(source_text)
    output = Path(output_text)
    partial = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or partial.exists():
        raise FileExistsError(output)

    input_rows = 0
    output_rows = 0
    buffer: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    try:
        with zipfile.ZipFile(source) as archive, archive.open(member) as handle:
            for line in handle:
                input_rows += 1
                row = orjson.loads(line)
                record = feature_record(row, hash_size, max_entities)
                if record is None:
                    continue
                buffer.append(record)
                if len(buffer) >= rows_per_group:
                    table = pa.Table.from_pylist(buffer, schema=SCHEMA)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            partial,
                            SCHEMA,
                            compression=compression,
                            compression_level=compression_level,
                            use_dictionary=True,
                            write_statistics=True,
                        )
                    writer.write_table(table, row_group_size=rows_per_group)
                    output_rows += len(buffer)
                    buffer.clear()
            if buffer:
                table = pa.Table.from_pylist(buffer, schema=SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(
                        partial,
                        SCHEMA,
                        compression=compression,
                        compression_level=compression_level,
                        use_dictionary=True,
                        write_statistics=True,
                    )
                writer.write_table(table, row_group_size=rows_per_group)
                output_rows += len(buffer)
                buffer.clear()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # Preserve the shard mapping even if every raw row is rejected.
        pq.write_table(
            pa.Table.from_pylist([], schema=SCHEMA),
            partial,
            compression=compression,
            compression_level=compression_level,
        )
    os.replace(partial, output)
    parquet_file = pq.ParquetFile(output)
    if parquet_file.metadata.num_rows != output_rows:
        raise RuntimeError(
            f"Parquet row count mismatch for {output}: "
            f"{parquet_file.metadata.num_rows} != {output_rows}"
        )
    return {
        "member": member,
        "output": str(output),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "rejected_rows": input_rows - output_rows,
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "row_groups": parquet_file.metadata.num_row_groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rows-per-group", type=int, default=2048)
    parser.add_argument("--hash-size", type=int, default=65_536)
    parser.add_argument("--max-state-entities", type=int, default=80)
    parser.add_argument("--compression", choices=("zstd", "snappy"), default="zstd")
    parser.add_argument("--compression-level", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if args.workers < 1 or args.rows_per_group < 1:
        raise ValueError("--workers and --rows-per-group must be positive")

    output_dir.mkdir(parents=True)
    with zipfile.ZipFile(source) as archive:
        source_manifest = orjson.loads(archive.read("manifest.json"))
        members = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".jsonl")
            and name.split("/", 1)[0] in {"train", "valid", "test"}
        )
    if not members:
        raise ValueError(f"No BC JSONL shards found in {source}")

    tasks = []
    for member in members:
        relative = Path(member).with_suffix(".parquet")
        tasks.append(
            (
                str(source),
                member,
                str(output_dir / relative),
                args.hash_size,
                args.max_state_entities,
                args.rows_per_group,
                args.compression,
                args.compression_level,
            )
        )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_member, task): task[1] for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed}/{len(tasks)}] {result['member']} "
                f"rows={result['output_rows']:,} "
                f"bytes={result['bytes']:,}",
                flush=True,
            )

    results.sort(key=lambda row: row["member"])
    split_rows: dict[str, int] = {}
    split_bytes: dict[str, int] = {}
    for row in results:
        split = row["member"].split("/", 1)[0]
        split_rows[split] = split_rows.get(split, 0) + int(row["output_rows"])
        split_bytes[split] = split_bytes.get(split, 0) + int(row["bytes"])
    manifest = {
        "schema_version": PARQUET_SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "storage": {
            "format": "parquet",
            "layout": "one parquet file per source JSONL shard",
            "row_group_rows": args.rows_per_group,
            "compression": args.compression,
            "compression_level": args.compression_level,
            "pre_featurized": True,
        },
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "manifest": source_manifest,
        },
        "model_features": {
            "hash_size": args.hash_size,
            "max_state_entities": args.max_state_entities,
        },
        "split_decisions": split_rows,
        "split_bytes": split_bytes,
        "stats": {
            "input_rows": sum(int(row["input_rows"]) for row in results),
            "output_rows": sum(int(row["output_rows"]) for row in results),
            "rejected_rows": sum(int(row["rejected_rows"]) for row in results),
            "bytes": sum(int(row["bytes"]) for row in results),
            "files": len(results),
        },
        "files": results,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
