#!/usr/bin/env python3
"""Split one archive partition into episode-disjoint replay/gate partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

import orjson


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def episode_hash_value(episode_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assigned_split(
    episode_id: str,
    seed: int,
    train_threshold: int,
) -> str:
    return (
        "train"
        if episode_hash_value(episode_id, seed) < train_threshold
        else "valid"
    )


def zip_info(name: str, compression: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o644 << 16
    return info


def source_members(
    archive: zipfile.ZipFile,
    source_split: str,
) -> list[str]:
    members = sorted(
        name
        for name in archive.namelist()
        if name.startswith(f"{source_split}/") and name.endswith(".jsonl")
    )
    if not members:
        raise RuntimeError(
            f"Input archive contains no {source_split!r} JSONL members"
        )
    return members


def write_partition(
    source: zipfile.ZipFile,
    destination: zipfile.ZipFile,
    members: list[str],
    source_split: str,
    target_split: str,
    seed: int,
    train_threshold: int,
    rows_per_shard: int,
    compression: int,
) -> dict[str, Any]:
    rows = 0
    shard_count = 0
    episodes: set[str] = set()
    episode_uuids: set[str] = set()
    dataset_dates: set[str] = set()
    teams: Counter[str] = Counter()
    writer: BinaryIO | None = None
    try:
        for member in members:
            with source.open(member) as handle:
                for raw_line in handle:
                    row = orjson.loads(raw_line)
                    if str(row.get("split", "")) != source_split:
                        raise RuntimeError(
                            f"Row/member split mismatch in {member}: "
                            f"{row.get('split')!r}"
                        )
                    episode_id = str(row.get("episode_id", ""))
                    if not episode_id:
                        raise RuntimeError(f"Missing episode_id in {member}")
                    if (
                        assigned_split(episode_id, seed, train_threshold)
                        != target_split
                    ):
                        continue
                    if rows % rows_per_shard == 0:
                        if writer is not None:
                            writer.close()
                        output_member = (
                            f"{target_split}/part-{shard_count:05d}.jsonl"
                        )
                        writer = destination.open(
                            zip_info(output_member, compression),
                            mode="w",
                            force_zip64=True,
                        )
                        shard_count += 1
                    row["split"] = target_split
                    writer.write(orjson.dumps(row) + b"\n")
                    rows += 1
                    episodes.add(episode_id)
                    episode_uuid = str(row.get("episode_uuid", ""))
                    if episode_uuid:
                        episode_uuids.add(episode_uuid)
                    dataset_date = str(row.get("dataset_date", ""))
                    if dataset_date:
                        dataset_dates.add(dataset_date)
                    teams[str(row.get("team_name", ""))] += 1
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        raise RuntimeError(f"Generated {target_split!r} partition is empty")
    return {
        "rows": rows,
        "episodes": len(episodes),
        "episode_uuids": len(episode_uuids),
        "shards": shard_count,
        "dataset_dates": sorted(dataset_dates),
        "team_rows": dict(sorted(teams.items())),
        "_episode_ids": episodes,
    }


def verify_archive(
    path: Path,
    seed: int,
    train_threshold: int,
    expected_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP integrity failure in {bad_member}")
        for split in ("train", "valid"):
            rows = 0
            episodes: set[str] = set()
            episode_uuids: set[str] = set()
            dates: set[str] = set()
            members = source_members(archive, split)
            for member in members:
                with archive.open(member) as handle:
                    for raw_line in handle:
                        row = orjson.loads(raw_line)
                        episode_id = str(row.get("episode_id", ""))
                        if str(row.get("split", "")) != split:
                            raise RuntimeError(
                                f"Output row/member mismatch in {member}"
                            )
                        if (
                            assigned_split(
                                episode_id,
                                seed,
                                train_threshold,
                            )
                            != split
                        ):
                            raise RuntimeError(
                                f"Episode hash mismatch for {episode_id}"
                            )
                        rows += 1
                        episodes.add(episode_id)
                        episode_uuid = str(row.get("episode_uuid", ""))
                        if episode_uuid:
                            episode_uuids.add(episode_uuid)
                        dataset_date = str(row.get("dataset_date", ""))
                        if dataset_date:
                            dates.add(dataset_date)
            expected = expected_stats[split]
            if rows != expected["rows"] or len(episodes) != expected["episodes"]:
                raise RuntimeError(
                    f"{split} verification mismatch: "
                    f"rows={rows}/{expected['rows']} "
                    f"episodes={len(episodes)}/{expected['episodes']}"
                )
            observed[split] = {
                "rows": rows,
                "episodes": len(episodes),
                "episode_uuids": len(episode_uuids),
                "shards": len(members),
                "dataset_dates": sorted(dates),
                "_episode_ids": episodes,
            }
    overlap = (
        observed["train"]["_episode_ids"]
        & observed["valid"]["_episode_ids"]
    )
    if overlap:
        raise RuntimeError(
            f"Episode leakage across output splits: {len(overlap)}"
        )
    for split in ("train", "valid"):
        observed[split].pop("_episode_ids")
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-split", default="valid")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--rows-per-shard", type=int, default=25000)
    parser.add_argument(
        "--compression",
        choices=("stored", "deflated"),
        default="stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("--input and --output must be different")
    if output_path.exists():
        raise FileExistsError(output_path)
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.rows_per_shard < 1:
        raise ValueError("--rows-per-shard must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.name}.tmp-{os.getpid()}"
    )
    if temporary_path.exists():
        raise FileExistsError(temporary_path)

    train_threshold = int(args.train_fraction * (2**64))
    compression = (
        zipfile.ZIP_STORED
        if args.compression == "stored"
        else zipfile.ZIP_DEFLATED
    )
    source_sha256 = sha256_file(input_path)
    try:
        with zipfile.ZipFile(input_path) as source:
            members = source_members(source, args.source_split)
            source_manifest = (
                json.loads(source.read("manifest.json"))
                if "manifest.json" in source.namelist()
                else None
            )
            with zipfile.ZipFile(
                temporary_path,
                mode="w",
                compression=compression,
                allowZip64=True,
            ) as destination:
                stats = {
                    split: write_partition(
                        source,
                        destination,
                        members,
                        args.source_split,
                        split,
                        args.seed,
                        train_threshold,
                        args.rows_per_shard,
                        compression,
                    )
                    for split in ("train", "valid")
                }
                overlap = (
                    stats["train"]["_episode_ids"]
                    & stats["valid"]["_episode_ids"]
                )
                if overlap:
                    raise RuntimeError(
                        f"Episode leakage before write completion: {len(overlap)}"
                    )
                manifest_stats = {}
                for split, split_stats in stats.items():
                    manifest_stats[split] = {
                        key: value
                        for key, value in split_stats.items()
                        if key != "_episode_ids"
                    }
                manifest = {
                    "schema_version": "ptcg-bc-episode-replay-gate-v1",
                    "source": {
                        "path": str(input_path),
                        "sha256": source_sha256,
                        "split": args.source_split,
                        "members": members,
                        "manifest_schema_version": (
                            source_manifest.get("schema_version")
                            if isinstance(source_manifest, dict)
                            else None
                        ),
                    },
                    "split_policy": {
                        "mode": "episode_sha256",
                        "hash_input": "{seed}:{episode_id}",
                        "seed": args.seed,
                        "train_fraction": args.train_fraction,
                        "train_threshold_uint64": train_threshold,
                    },
                    "rows_per_shard": args.rows_per_shard,
                    "compression": args.compression,
                    "stats": manifest_stats,
                    "leakage": {
                        "episode_id_overlap": 0,
                        "row_total": sum(
                            split_stats["rows"]
                            for split_stats in manifest_stats.values()
                        ),
                        "episode_total": sum(
                            split_stats["episodes"]
                            for split_stats in manifest_stats.values()
                        ),
                    },
                }
                destination.writestr(
                    zip_info("manifest.json", compression),
                    json.dumps(
                        manifest,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ).encode()
                    + b"\n",
                )

        verified = verify_archive(
            temporary_path,
            args.seed,
            train_threshold,
            stats,
        )
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    result = {
        "output": str(output_path),
        "source_sha256": source_sha256,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "verified": verified,
        "episode_id_overlap": 0,
        "rows_total": sum(item["rows"] for item in verified.values()),
        "episodes_total": sum(
            item["episodes"] for item in verified.values()
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
