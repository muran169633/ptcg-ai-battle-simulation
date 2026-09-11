#!/usr/bin/env python3
"""Roll a strict time-split BC archive forward by one evaluation boundary.

The source ``train`` and ``valid`` partitions are promoted into the new
``train`` partition.  Source ``test`` episodes are deterministically assigned
to new ``train``/``valid`` partitions with SHA-256 over
``"{seed}:{episode_id}"``.  Rows from an episode can therefore never cross the
new boundary.

The output is written to a temporary file in the destination directory,
verified in full, and atomically installed only after every check succeeds.
The source archive is always read-only and can never be the output target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import orjson


SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = ("train", "valid")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
UINT64_SPACE = 2**64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def episode_hash_value(episode_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).digest()
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
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o644 << 16
    return info


def canonical_row_bytes(row: dict[str, Any]) -> bytes:
    return orjson.dumps(row, option=orjson.OPT_SORT_KEYS) + b"\n"


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@dataclass
class PartitionStats:
    rows: int = 0
    episode_ids: set[str] = field(default_factory=set)
    episode_uuids: set[str] = field(default_factory=set)
    dataset_dates: set[str] = field(default_factory=set)
    team_rows: Counter[str] = field(default_factory=Counter)
    context_rows: Counter[str] = field(default_factory=Counter)

    def add(self, row: dict[str, Any], episode_id: str) -> None:
        self.rows += 1
        self.episode_ids.add(episode_id)
        episode_uuid = row.get("episode_uuid")
        if episode_uuid not in (None, ""):
            self.episode_uuids.add(str(episode_uuid))
        self.dataset_dates.add(str(row["dataset_date"]))
        self.team_rows[str(row.get("team_name", ""))] += 1
        self.context_rows[str(row.get("select_context", ""))] += 1

    def public(self, shards: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rows": self.rows,
            "episodes": len(self.episode_ids),
            "episode_uuids": len(self.episode_uuids),
            "dataset_dates": sorted(self.dataset_dates),
        }
        if shards is not None:
            result["shards"] = shards
        return result


@dataclass(frozen=True)
class SourceLayout:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    members: dict[str, tuple[str, ...]]
    dates: tuple[str, ...]
    dates_by_split: dict[str, frozenset[str]]
    schema_version: str


class DeterministicArchiveWriter:
    def __init__(
        self,
        path: Path,
        rows_per_shard: int,
        compression: int,
    ) -> None:
        self.path = path
        self.rows_per_shard = rows_per_shard
        self.compression = compression
        self.archive: zipfile.ZipFile | None = None
        self.handle: BinaryIO | None = None
        self.current_split: str | None = None
        self.rows_in_shard = 0
        self.shards: Counter[str] = Counter()
        self.last_split_index = -1

    def __enter__(self) -> "DeterministicArchiveWriter":
        self.archive = zipfile.ZipFile(
            self.path,
            mode="w",
            compression=self.compression,
            compresslevel=(
                6 if self.compression == zipfile.ZIP_DEFLATED else None
            ),
            allowZip64=True,
        )
        return self

    def _close_shard(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.current_split = None
        self.rows_in_shard = 0

    def write(self, split: str, payload: bytes) -> None:
        if split not in OUTPUT_SPLITS:
            raise ValueError(f"Unsupported output split: {split!r}")
        split_index = OUTPUT_SPLITS.index(split)
        if split_index < self.last_split_index:
            raise RuntimeError("Output partitions must be written in split order")
        if (
            self.handle is None
            or self.current_split != split
            or self.rows_in_shard >= self.rows_per_shard
        ):
            self._close_shard()
            assert self.archive is not None
            member = f"{split}/part-{self.shards[split]:05d}.jsonl"
            self.handle = self.archive.open(
                zip_info(member, self.compression),
                mode="w",
                force_zip64=True,
            )
            self.current_split = split
            self.rows_in_shard = 0
            self.shards[split] += 1
        self.handle.write(payload)
        self.rows_in_shard += 1
        self.last_split_index = split_index

    def finish(self, manifest: dict[str, Any]) -> None:
        self._close_shard()
        assert self.archive is not None
        self.archive.writestr(
            zip_info("manifest.json", self.compression),
            canonical_manifest_bytes(manifest),
        )
        self.archive.close()
        self.archive = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._close_shard()
        if self.archive is not None:
            self.archive.close()
            self.archive = None


def _validated_date_list(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        requirement = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"manifest {label} must be {requirement}")
    if len(value) != len(set(value)):
        raise ValueError(f"manifest {label} contains duplicate dates")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"manifest {label} contains a non-string date")
        try:
            date.fromisoformat(item)
        except ValueError as error:
            raise ValueError(
                f"manifest {label} contains invalid ISO date {item!r}"
            ) from error
        result.append(item)
    return tuple(result)


def inspect_source_layout(source: zipfile.ZipFile) -> SourceLayout:
    names = source.namelist()
    duplicate_names = sorted(
        name for name, count in Counter(names).items() if count > 1
    )
    if duplicate_names:
        raise ValueError(
            f"source ZIP contains duplicate members: {duplicate_names[:3]}"
        )
    if names.count("manifest.json") != 1:
        raise ValueError("source ZIP must contain exactly one manifest.json")
    manifest_bytes = source.read("manifest.json")
    try:
        manifest = orjson.loads(manifest_bytes)
    except orjson.JSONDecodeError as error:
        raise ValueError("source manifest.json is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("source manifest.json must contain an object")

    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("source manifest has no valid schema_version")
    dates = _validated_date_list(manifest.get("dates"), "dates")
    if tuple(sorted(dates)) != dates:
        raise ValueError("manifest dates must be sorted")

    policy = manifest.get("split_policy")
    if not isinstance(policy, dict) or policy.get("mode") != "time":
        raise ValueError(
            "source manifest split_policy.mode must be 'time'"
        )
    dates_by_split = {
        split: frozenset(
            _validated_date_list(
                policy.get(f"{split}_dates"),
                f"split_policy.{split}_dates",
            )
        )
        for split in SOURCE_SPLITS
    }
    all_policy_dates: set[str] = set()
    for split in SOURCE_SPLITS:
        overlap = all_policy_dates & dates_by_split[split]
        if overlap:
            raise ValueError(
                "source split date lists overlap at "
                f"{sorted(overlap)}"
            )
        all_policy_dates.update(dates_by_split[split])
    if all_policy_dates != set(dates):
        raise ValueError(
            "source split date lists do not exactly cover manifest dates"
        )

    members: dict[str, tuple[str, ...]] = {}
    for split in SOURCE_SPLITS:
        split_members = tuple(
            sorted(
                name
                for name in names
                if name.startswith(f"{split}/") and name.endswith(".jsonl")
            )
        )
        if not split_members:
            raise ValueError(
                f"source ZIP contains no {split!r} JSONL members"
            )
        members[split] = split_members
    known_members = {
        member for split_members in members.values() for member in split_members
    }
    unexpected_jsonl = sorted(
        name
        for name in names
        if name.endswith(".jsonl") and name not in known_members
    )
    if unexpected_jsonl:
        raise ValueError(
            "source ZIP contains JSONL members outside train/valid/test: "
            f"{unexpected_jsonl[:3]}"
        )
    return SourceLayout(
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        members=members,
        dates=dates,
        dates_by_split=dates_by_split,
        schema_version=schema_version,
    )


def iter_source_rows(
    source: zipfile.ZipFile,
    layout: SourceLayout,
    split: str,
) -> Iterator[tuple[dict[str, Any], str, str, int]]:
    for member in layout.members[split]:
        with source.open(member) as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    row = orjson.loads(raw_line)
                except orjson.JSONDecodeError as error:
                    raise ValueError(
                        f"{member}:{line_number}: invalid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{member}:{line_number}: row is not an object"
                    )
                if row.get("split") != split:
                    raise ValueError(
                        f"{member}:{line_number}: row/member split mismatch "
                        f"({row.get('split')!r} != {split!r})"
                    )
                if row.get("schema_version") != layout.schema_version:
                    raise ValueError(
                        f"{member}:{line_number}: schema_version mismatch"
                    )
                raw_episode_id = row.get("episode_id")
                if raw_episode_id in (None, ""):
                    raise ValueError(
                        f"{member}:{line_number}: missing episode_id"
                    )
                episode_id = str(raw_episode_id)
                dataset_date = row.get("dataset_date")
                if (
                    not isinstance(dataset_date, str)
                    or dataset_date not in layout.dates_by_split[split]
                ):
                    raise ValueError(
                        f"{member}:{line_number}: dataset_date "
                        f"{dataset_date!r} is not valid for source {split!r}"
                    )
                yield row, episode_id, member, line_number


def _validate_source_manifest_counts(
    layout: SourceLayout,
    stats: dict[str, PartitionStats],
) -> None:
    declared = layout.manifest.get("split_decisions")
    if not isinstance(declared, dict):
        raise ValueError("source manifest has no split_decisions object")
    for split in SOURCE_SPLITS:
        expected = declared.get(split)
        if (
            not isinstance(expected, int)
            or isinstance(expected, bool)
            or expected < 0
        ):
            raise ValueError(
                f"source manifest split_decisions.{split} is invalid"
            )
        if stats[split].rows != expected:
            raise ValueError(
                f"source {split} row count mismatch: "
                f"{stats[split].rows} observed, {expected} declared"
            )
    declared_shards = layout.manifest.get("shards")
    if isinstance(declared_shards, dict):
        for split in SOURCE_SPLITS:
            expected = declared_shards.get(split)
            if expected is not None and expected != len(layout.members[split]):
                raise ValueError(
                    f"source {split} shard count mismatch: "
                    f"{len(layout.members[split])} observed, "
                    f"{expected} declared"
                )


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        key: value
        for key, value in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def build_manifest(
    layout: SourceLayout,
    source_path: Path,
    source_sha256: str,
    input_stats: dict[str, PartitionStats],
    output_stats: dict[str, PartitionStats],
    shards: Counter[str],
    seed: int,
    train_fraction: float,
    train_threshold: int,
    rows_per_shard: int,
    compression_name: str,
) -> dict[str, Any]:
    manifest = deepcopy(layout.manifest)
    total_rows = sum(stats.rows for stats in output_stats.values())
    total_episodes = len(
        output_stats["train"].episode_ids
        | output_stats["valid"].episode_ids
    )
    original_stats = manifest.get("stats")
    retained_stats = (
        deepcopy(original_stats) if isinstance(original_stats, dict) else {}
    )
    retained_stats["decisions"] = total_rows
    retained_stats["episodes_in_output"] = total_episodes

    team_rows = output_stats["train"].team_rows + output_stats["valid"].team_rows
    context_rows = (
        output_stats["train"].context_rows
        + output_stats["valid"].context_rows
    )
    manifest.update(
        {
            "dates": list(layout.dates),
            "split_policy": {
                "mode": "roll_forward_episode_hash",
                "promoted_source_splits": ["train", "valid"],
                "partitioned_source_split": "test",
                "hash": "sha256",
                "hash_input": "{seed}:{episode_id}",
                "seed": seed,
                "train_fraction": train_fraction,
                "train_threshold_uint64": train_threshold,
                "source_split_dates": {
                    split: sorted(layout.dates_by_split[split])
                    for split in SOURCE_SPLITS
                },
            },
            "stats": retained_stats,
            "split_decisions": {
                split: output_stats[split].rows for split in OUTPUT_SPLITS
            },
            "split_episodes": {
                split: len(output_stats[split].episode_ids)
                for split in OUTPUT_SPLITS
            },
            "team_decisions": _counter_dict(team_rows),
            "context_decisions": _counter_dict(context_rows),
            "shards": {
                split: shards[split] for split in OUTPUT_SPLITS
            },
            "rows_per_shard": rows_per_shard,
            "compression": compression_name,
            "roll_forward": {
                "source": {
                    "path": str(source_path),
                    "sha256": source_sha256,
                    "manifest_sha256": hashlib.sha256(
                        layout.manifest_bytes
                    ).hexdigest(),
                    "manifest_schema_version": layout.schema_version,
                },
                "input": {
                    split: input_stats[split].public(
                        shards=len(layout.members[split])
                    )
                    for split in SOURCE_SPLITS
                },
                "output": {
                    split: output_stats[split].public(
                        shards=shards[split]
                    )
                    for split in OUTPUT_SPLITS
                },
                "validation": {
                    "input_row_total": sum(
                        stats.rows for stats in input_stats.values()
                    ),
                    "output_row_total": total_rows,
                    "rows_preserved": True,
                    "output_episode_id_overlap": 0,
                    "row_member_split_mismatches": 0,
                },
            },
        }
    )
    return manifest


def verify_output_archive(
    path: Path,
    expected_stats: dict[str, PartitionStats],
    expected_manifest: dict[str, Any],
    schema_version: str,
) -> dict[str, Any]:
    observed = {split: PartitionStats() for split in OUTPUT_SPLITS}
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"output ZIP integrity failure in {bad_member}")
        names = archive.namelist()
        if names.count("manifest.json") != 1:
            raise RuntimeError(
                "output ZIP must contain exactly one manifest.json"
            )
        unexpected_jsonl = [
            name
            for name in names
            if name.endswith(".jsonl")
            and not any(
                name.startswith(f"{split}/") for split in OUTPUT_SPLITS
            )
        ]
        if unexpected_jsonl:
            raise RuntimeError(
                f"unexpected output JSONL member: {unexpected_jsonl[0]}"
            )
        for split in OUTPUT_SPLITS:
            members = sorted(
                name
                for name in names
                if name.startswith(f"{split}/") and name.endswith(".jsonl")
            )
            if not members:
                raise RuntimeError(
                    f"output contains no {split!r} JSONL members"
                )
            for member in members:
                with archive.open(member) as handle:
                    for line_number, raw_line in enumerate(handle, 1):
                        if not raw_line.strip():
                            continue
                        try:
                            row = orjson.loads(raw_line)
                        except orjson.JSONDecodeError as error:
                            raise RuntimeError(
                                f"{member}:{line_number}: invalid output JSON"
                            ) from error
                        if not isinstance(row, dict):
                            raise RuntimeError(
                                f"{member}:{line_number}: output row is not "
                                "an object"
                            )
                        if row.get("split") != split:
                            raise RuntimeError(
                                f"{member}:{line_number}: output row/member "
                                "split mismatch"
                            )
                        if row.get("schema_version") != schema_version:
                            raise RuntimeError(
                                f"{member}:{line_number}: output schema mismatch"
                            )
                        raw_episode_id = row.get("episode_id")
                        if raw_episode_id in (None, ""):
                            raise RuntimeError(
                                f"{member}:{line_number}: missing episode_id"
                            )
                        if not isinstance(row.get("dataset_date"), str):
                            raise RuntimeError(
                                f"{member}:{line_number}: missing dataset_date"
                            )
                        observed[split].add(row, str(raw_episode_id))

        output_manifest = orjson.loads(archive.read("manifest.json"))
        if output_manifest != expected_manifest:
            raise RuntimeError(
                "output manifest does not match the generated manifest"
            )

    for split in OUTPUT_SPLITS:
        expected = expected_stats[split]
        actual = observed[split]
        if actual.rows != expected.rows:
            raise RuntimeError(
                f"output {split} row count mismatch: "
                f"{actual.rows} != {expected.rows}"
            )
        if actual.episode_ids != expected.episode_ids:
            raise RuntimeError(
                f"output {split} episode set mismatch"
            )
        if actual.dataset_dates != expected.dataset_dates:
            raise RuntimeError(
                f"output {split} dataset date set mismatch"
            )
    overlap = observed["train"].episode_ids & observed["valid"].episode_ids
    if overlap:
        raise RuntimeError(
            f"output episode leakage across splits: {len(overlap)}"
        )
    if sum(item.rows for item in observed.values()) != sum(
        item.rows for item in expected_stats.values()
    ):
        raise RuntimeError("output row total does not match input row total")
    return {
        split: observed[split].public()
        for split in OUTPUT_SPLITS
    }


def _same_file(first: Path, second: Path) -> bool:
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    if first.exists() and second.exists():
        return os.path.samefile(first, second)
    return False


def roll_archive(
    input_path: Path,
    output_path: Path,
    *,
    seed: int = 20260820,
    train_fraction: float = 0.8,
    rows_per_shard: int = 25_000,
    compression_name: str = "stored",
    overwrite: bool = False,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve(strict=True)
    output_path = Path(os.path.abspath(output_path.expanduser()))
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if _same_file(input_path, output_path):
        raise ValueError("input and output must be different files")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    if not 0.0 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0, 1)")
    if rows_per_shard < 1:
        raise ValueError("rows_per_shard must be positive")
    if compression_name not in {"stored", "deflated"}:
        raise ValueError("compression_name must be 'stored' or 'deflated'")

    train_threshold = int(train_fraction * UINT64_SPACE)
    compression = (
        zipfile.ZIP_STORED
        if compression_name == "stored"
        else zipfile.ZIP_DEFLATED
    )
    source_sha256 = sha256_file(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".partial",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()

    try:
        input_stats = {
            split: PartitionStats() for split in SOURCE_SPLITS
        }
        output_stats = {
            split: PartitionStats() for split in OUTPUT_SPLITS
        }
        episode_owner: dict[str, str] = {}
        episode_date: dict[str, str] = {}

        with zipfile.ZipFile(input_path) as source:
            layout = inspect_source_layout(source)
            with DeterministicArchiveWriter(
                temporary_path,
                rows_per_shard,
                compression,
            ) as writer:
                # Output train is written first.  This single pass also fully
                # validates and counts every source row.
                for source_split in SOURCE_SPLITS:
                    for row, episode_id, member, line_number in iter_source_rows(
                        source,
                        layout,
                        source_split,
                    ):
                        previous_owner = episode_owner.setdefault(
                            episode_id, source_split
                        )
                        if previous_owner != source_split:
                            raise ValueError(
                                f"{member}:{line_number}: episode {episode_id!r} "
                                f"crosses source splits {previous_owner!r} and "
                                f"{source_split!r}"
                            )
                        dataset_date = str(row["dataset_date"])
                        previous_date = episode_date.setdefault(
                            episode_id, dataset_date
                        )
                        if previous_date != dataset_date:
                            raise ValueError(
                                f"{member}:{line_number}: episode {episode_id!r} "
                                f"crosses dates {previous_date!r} and "
                                f"{dataset_date!r}"
                            )
                        input_stats[source_split].add(row, episode_id)
                        target_split = (
                            "train"
                            if source_split in ("train", "valid")
                            else assigned_split(
                                episode_id,
                                seed,
                                train_threshold,
                            )
                        )
                        if target_split == "train":
                            row["split"] = "train"
                            writer.write("train", canonical_row_bytes(row))
                            output_stats["train"].add(row, episode_id)

                _validate_source_manifest_counts(layout, input_stats)
                input_total = sum(
                    item.rows for item in input_stats.values()
                )
                if input_total == 0:
                    raise RuntimeError("source archive contains no decision rows")

                # Re-read only source test to emit the held-out hash partition
                # after all train shards.  The hash makes the second pass
                # deterministic and episode-consistent.
                for row, episode_id, _, _ in iter_source_rows(
                    source,
                    layout,
                    "test",
                ):
                    if (
                        assigned_split(
                            episode_id,
                            seed,
                            train_threshold,
                        )
                        != "valid"
                    ):
                        continue
                    row["split"] = "valid"
                    writer.write("valid", canonical_row_bytes(row))
                    output_stats["valid"].add(row, episode_id)

                if not output_stats["valid"].episode_ids:
                    raise RuntimeError(
                        "source test episodes produced an empty output valid "
                        "partition; choose another seed or fraction"
                    )
                source_test_train = (
                    output_stats["train"].episode_ids
                    & input_stats["test"].episode_ids
                )
                if train_fraction > 0.0 and not source_test_train:
                    raise RuntimeError(
                        "source test episodes produced an empty hashed train "
                        "partition; choose another seed or fraction"
                    )
                output_overlap = (
                    output_stats["train"].episode_ids
                    & output_stats["valid"].episode_ids
                )
                if output_overlap:
                    raise RuntimeError(
                        f"episode leakage before output completion: "
                        f"{len(output_overlap)}"
                    )
                output_total = sum(
                    item.rows for item in output_stats.values()
                )
                if output_total != input_total:
                    raise RuntimeError(
                        f"row preservation failure: output={output_total}, "
                        f"input={input_total}"
                    )

                manifest = build_manifest(
                    layout,
                    input_path,
                    source_sha256,
                    input_stats,
                    output_stats,
                    writer.shards,
                    seed,
                    train_fraction,
                    train_threshold,
                    rows_per_shard,
                    compression_name,
                )
                writer.finish(manifest)

        verified = verify_output_archive(
            temporary_path,
            output_stats,
            manifest,
            layout.schema_version,
        )
        # Flush the verified archive before making it visible at the final
        # path.  Existing output remains untouched until this os.replace.
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if _same_file(input_path, output_path):
            raise ValueError("refusing to replace the input archive")
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"{output_path} appeared during generation; refusing to "
                "replace it without --overwrite"
            )
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "output": str(output_path),
        "sha256": sha256_file(output_path),
        "source": str(input_path),
        "source_sha256": source_sha256,
        "seed": seed,
        "train_fraction": train_fraction,
        "verified": verified,
        "episode_id_overlap": 0,
        "rows_total": sum(item["rows"] for item in verified.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote source train+valid into new train and split source test "
            "by seeded episode SHA-256 into new train/valid."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--rows-per-shard", type=int, default=25_000)
    parser.add_argument(
        "--compression",
        choices=("stored", "deflated"),
        default="stored",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output (never the input)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = roll_archive(
        args.input,
        args.output,
        seed=args.seed,
        train_fraction=args.train_fraction,
        rows_per_shard=args.rows_per_shard,
        compression_name=args.compression,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
