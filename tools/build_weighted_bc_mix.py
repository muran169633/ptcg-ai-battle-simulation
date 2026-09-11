#!/usr/bin/env python3
"""Build the frozen old-BC/yanz weighted Alakazam training view.

The output is deliberately *not* a generic archive merge:

* old gold8 Alakazam ``train`` rows become output ``train`` with their source
  ``sample_weight`` multiplied by 0.25;
* yanz live ``train`` rows become output ``train`` with factor 1.0;
* yanz live ``valid`` rows become output ``valid`` unchanged (factor 1.0);
* old ``valid`` and ``test`` rows are never copied.

Input archive hashes and row counts are a frozen contract.  Publication uses a
same-directory hard link so an output that appears concurrently is never
overwritten.  ZIP member names, ordering and metadata are deterministic.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

import orjson


OUTPUT_MANIFEST_SCHEMA = "ptcg-weighted-bc-mix-manifest-v1"
DECISION_KEY_FORMAT = "dataset_date|episode_id|seat|action_step_index"
DECK_HASH = "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
OLD_ARCHIVE_SHA256 = (
    "4bd0de193cfb88b060435f84bbb5dc83a9eff3ab1aa4a498ef0859d2346f3c6c"
)
YANZ_ARCHIVE_SHA256 = (
    "bbbb6f69b809c60e900f7d2992333b645d117b2a6c836642fecca0e1f4226720"
)
REJECTED_ALL_LOSSES_ARTIFACT = "old025_yanz1.zip"
REJECTED_ALL_LOSSES_SHA256 = (
    "89067ab5fa2a868e805931c34788dceb7e98d52dc299c7b5de5a63dcfddbb8d2"
)
OLD_ROW_SCHEMA = "ptcg-bc-visible-decisions-v1"
YANZ_ROW_SCHEMA = "ptcg-bc-visible-decisions-v1"
OLD_MANIFEST_SCHEMA = "ptcg-bc-visible-decisions-v1"
YANZ_MANIFEST_SCHEMA = "ptcg-gold-policy-visible-decisions-v1"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_EXTERNAL_ATTR = 0o600 << 16
ZIP_COMPRESSION = zipfile.ZIP_DEFLATED
ZIP_COMPRESSLEVEL = 6
DEFAULT_ROWS_PER_SHARD = 25_000


class ContractError(RuntimeError):
    """The frozen source or materialized output violated its contract."""


@dataclass(frozen=True)
class BuildContract:
    deck_hash: str
    old_archive_sha256: str
    yanz_archive_sha256: str
    old_manifest_schema: str
    yanz_manifest_schema: str
    old_row_schema: str
    yanz_row_schema: str
    old_split_rows: Mapping[str, int]
    yanz_split_rows: Mapping[str, int]
    old_split_episodes: Mapping[str, int]
    yanz_split_episodes: Mapping[str, int]
    old_train_factor: float = 0.25
    yanz_factor: float = 1.0


DEFAULT_CONTRACT = BuildContract(
    deck_hash=DECK_HASH,
    old_archive_sha256=OLD_ARCHIVE_SHA256,
    yanz_archive_sha256=YANZ_ARCHIVE_SHA256,
    old_manifest_schema=OLD_MANIFEST_SCHEMA,
    yanz_manifest_schema=YANZ_MANIFEST_SCHEMA,
    old_row_schema=OLD_ROW_SCHEMA,
    yanz_row_schema=YANZ_ROW_SCHEMA,
    old_split_rows={"train": 59_875, "valid": 8_116, "test": 3_747},
    yanz_split_rows={"train": 1_725, "valid": 525},
    old_split_episodes={"train": 771, "valid": 88, "test": 48},
    yanz_split_episodes={"train": 26, "valid": 7},
)


@dataclass(frozen=True)
class SourceSpec:
    role: str
    path: Path
    archive_sha256: str
    manifest_schema: str
    row_schema: str
    expected_split_rows: Mapping[str, int]
    expected_split_episodes: Mapping[str, int]
    selected_splits: tuple[str, ...]
    split_factors: Mapping[str, float]
    manifest: dict[str, Any]
    manifest_sha256: str
    members_by_split: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class MemberRecord:
    name: str
    split: str
    rows: int
    uncompressed_bytes: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_deck_hash(manifest: Mapping[str, Any], role: str) -> Any:
    if role == "old_gold8_alakazam":
        profile = manifest.get("profile")
        return profile.get("deck_hash") if isinstance(profile, dict) else None
    policy = manifest.get("policy")
    return policy.get("deck_hash") if isinstance(policy, dict) else None


def _expected_members(
    names: Iterable[str], expected_splits: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    by_split: dict[str, list[str]] = {split: [] for split in expected_splits}
    jsonl_names: list[str] = []
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        jsonl_names.append(name)
        prefix = name.split("/", 1)[0]
        if prefix not in by_split:
            raise ContractError(f"unexpected JSONL member {name!r}")
        by_split[prefix].append(name)
    if not jsonl_names:
        raise ContractError("source archive contains no JSONL members")
    result: dict[str, tuple[str, ...]] = {}
    for split, split_names in by_split.items():
        ordered = tuple(sorted(split_names))
        wanted = tuple(
            f"{split}/part-{index:05d}.jsonl" for index in range(len(ordered))
        )
        if ordered != wanted:
            raise ContractError(
                f"{split} members are not contiguous deterministic shards: {ordered!r}"
            )
        result[split] = ordered
    return result


def _preflight_source(
    *,
    role: str,
    path: Path,
    expected_sha256: str,
    manifest_schema: str,
    row_schema: str,
    expected_split_rows: Mapping[str, int],
    expected_split_episodes: Mapping[str, int],
    selected_splits: tuple[str, ...],
    split_factors: Mapping[str, float],
    deck_hash: str,
) -> SourceSpec:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ContractError(
            f"{role} archive SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    try:
        with zipfile.ZipFile(resolved) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ContractError(f"{role} archive has duplicate member names")
            if names.count("manifest.json") != 1:
                raise ContractError(f"{role} archive must contain one manifest.json")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ContractError(
                    f"{role} archive failed CRC/integrity at {bad_member!r}"
                )
            manifest_payload = archive.read("manifest.json")
            try:
                manifest = orjson.loads(manifest_payload)
            except orjson.JSONDecodeError as exc:
                raise ContractError(f"{role} manifest is invalid JSON") from exc
            if not isinstance(manifest, dict):
                raise ContractError(f"{role} manifest must be an object")
            if manifest.get("schema_version") != manifest_schema:
                raise ContractError(
                    f"{role} manifest schema_version drift: "
                    f"{manifest.get('schema_version')!r} != {manifest_schema!r}"
                )
            actual_deck_hash = _manifest_deck_hash(manifest, role)
            if actual_deck_hash != deck_hash:
                raise ContractError(
                    f"{role} manifest deck hash drift: "
                    f"{actual_deck_hash!r} != {deck_hash!r}"
                )
            raw_rows = manifest.get("split_decisions")
            raw_episodes = manifest.get("split_episodes")
            if not isinstance(raw_rows, dict) or not isinstance(raw_episodes, dict):
                raise ContractError(
                    f"{role} manifest lacks split_decisions/split_episodes objects"
                )
            for split, wanted in expected_split_rows.items():
                actual = raw_rows.get(split)
                if actual != wanted:
                    raise ContractError(
                        f"{role} manifest {split} rows drift: {actual!r} != {wanted}"
                    )
            for split, wanted in expected_split_episodes.items():
                actual = raw_episodes.get(split)
                if actual != wanted:
                    raise ContractError(
                        f"{role} manifest {split} episodes drift: "
                        f"{actual!r} != {wanted}"
                    )
            members_by_split = _expected_members(names, expected_split_rows)
            raw_shards = manifest.get("shards")
            if raw_shards is not None:
                if not isinstance(raw_shards, dict):
                    raise ContractError(f"{role} manifest shards must be an object")
                for split, members in members_by_split.items():
                    if raw_shards.get(split) != len(members):
                        raise ContractError(
                            f"{role} manifest {split} shard count drift"
                        )
    except zipfile.BadZipFile as exc:
        raise ContractError(f"{role} is not an intact ZIP archive") from exc

    return SourceSpec(
        role=role,
        path=resolved,
        archive_sha256=actual_sha256,
        manifest_schema=manifest_schema,
        row_schema=row_schema,
        expected_split_rows=dict(expected_split_rows),
        expected_split_episodes=dict(expected_split_episodes),
        selected_splits=selected_splits,
        split_factors=dict(split_factors),
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        members_by_split=members_by_split,
    )


def _zip_info(name: str, compression: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = ZIP_EXTERNAL_ATTR
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    if compression == zipfile.ZIP_DEFLATED:
        info._compresslevel = ZIP_COMPRESSLEVEL
    return info


class DeterministicArchiveWriter:
    def __init__(
        self,
        archive: zipfile.ZipFile,
        *,
        rows_per_shard: int,
        compression: int,
    ) -> None:
        self.archive = archive
        self.rows_per_shard = rows_per_shard
        self.compression = compression
        self.handle: BinaryIO | None = None
        self.current_split: str | None = None
        self.current_rows = 0
        self.current_bytes = 0
        self.current_digest: hashlib._Hash | None = None
        self.current_name: str | None = None
        self.shard_counts: Counter[str] = Counter()
        self.member_records: list[MemberRecord] = []

    def _close_member(self) -> None:
        if self.handle is None:
            return
        self.handle.close()
        assert self.current_name is not None
        assert self.current_split is not None
        assert self.current_digest is not None
        self.member_records.append(
            MemberRecord(
                name=self.current_name,
                split=self.current_split,
                rows=self.current_rows,
                uncompressed_bytes=self.current_bytes,
                sha256=self.current_digest.hexdigest(),
            )
        )
        self.handle = None
        self.current_split = None
        self.current_rows = 0
        self.current_bytes = 0
        self.current_digest = None
        self.current_name = None

    def write_row(self, split: str, payload: bytes) -> None:
        if split not in ("train", "valid"):
            raise ContractError(f"unsupported output split {split!r}")
        if (
            self.handle is None
            or self.current_split != split
            or self.current_rows >= self.rows_per_shard
        ):
            self._close_member()
            index = self.shard_counts[split]
            name = f"{split}/part-{index:05d}.jsonl"
            self.handle = self.archive.open(
                _zip_info(name, self.compression), mode="w", force_zip64=True
            )
            self.current_split = split
            self.current_name = name
            self.current_digest = hashlib.sha256()
            self.shard_counts[split] += 1
        assert self.handle is not None
        assert self.current_digest is not None
        self.handle.write(payload)
        self.current_digest.update(payload)
        self.current_rows += 1
        self.current_bytes += len(payload)

    def finish_rows(self) -> tuple[MemberRecord, ...]:
        self._close_member()
        return tuple(self.member_records)

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        payload = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                separators=(",", ": "),
            )
            + "\n"
        ).encode("utf-8")
        self.archive.writestr(_zip_info("manifest.json", self.compression), payload)


def _valid_episode_id(value: Any) -> bool:
    return (
        isinstance(value, (str, int))
        and not isinstance(value, bool)
        and bool(str(value).strip())
        and "|" not in str(value)
    )


def _validate_source_row(
    value: Any,
    *,
    source: SourceSpec,
    source_split: str,
    member: str,
    line_number: int,
    deck_hash: str,
) -> tuple[dict[str, Any], str, str]:
    where = f"{source.role}:{member}:{line_number}"
    if not isinstance(value, dict):
        raise ContractError(f"{where}: row is not an object")
    checks = {
        "schema_version": (value.get("schema_version"), source.row_schema),
        "split": (value.get("split"), source_split),
        "deck_hash": (value.get("deck_hash"), deck_hash),
    }
    failures = [
        f"{field}={actual!r}, expected {wanted!r}"
        for field, (actual, wanted) in checks.items()
        if actual != wanted
    ]
    if failures:
        raise ContractError(f"{where}: row contract drift: {'; '.join(failures)}")
    original_weight = value.get("sample_weight")
    if (
        isinstance(original_weight, bool)
        or not isinstance(original_weight, (int, float))
        or not math.isfinite(float(original_weight))
        or float(original_weight) <= 0.0
    ):
        raise ContractError(
            f"{where}: source sample_weight is not finite and positive"
        )
    dataset_date = value.get("dataset_date")
    episode_id = value.get("episode_id")
    seat = value.get("seat")
    action_step = value.get("action_step_index")
    observation_step = value.get("observation_step_index")
    if not isinstance(dataset_date, str) or not dataset_date or "|" in dataset_date:
        raise ContractError(f"{where}: invalid dataset_date {dataset_date!r}")
    if not _valid_episode_id(episode_id):
        raise ContractError(f"{where}: invalid episode_id {episode_id!r}")
    if isinstance(seat, bool) or not isinstance(seat, int) or seat not in (0, 1):
        raise ContractError(f"{where}: seat must be 0 or 1")
    if (
        isinstance(action_step, bool)
        or not isinstance(action_step, int)
        or action_step <= 0
        or isinstance(observation_step, bool)
        or not isinstance(observation_step, int)
        or observation_step != action_step - 1
    ):
        raise ContractError(f"{where}: observation/action step alignment drift")
    if "visualize" in value or (
        isinstance(value.get("observation"), dict)
        and "visualize" in value["observation"]
    ):
        raise ContractError(f"{where}: hidden visualize payload is forbidden")
    decision_key = f"{dataset_date}|{episode_id}|{seat}|{action_step}"
    return value, decision_key, str(episode_id)


def _preserved_payload(row: Mapping[str, Any]) -> bytes:
    preserved = {
        key: value
        for key, value in row.items()
        if key not in ("split", "sample_weight")
    }
    return orjson.dumps(preserved, option=orjson.OPT_SORT_KEYS) + b"\n"


def _digest_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _weight_histogram(values: Iterable[float]) -> dict[str, int]:
    counts = Counter(float(value).hex() for value in values)
    return {key: counts[key] for key in sorted(counts)}


def _weight_digest(values: Iterable[float]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(float(value).hex().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _weight_audit(
    original_weights: list[float],
    output_weights: list[float],
    factor: float,
) -> dict[str, Any]:
    if len(original_weights) != len(output_weights) or not original_weights:
        raise ContractError("weight audit requires paired non-empty weight vectors")
    for original, output in zip(original_weights, output_weights):
        if output != original * factor:
            raise ContractError("output sample_weight is not source weight times factor")
    return {
        "rows": len(original_weights),
        "factor": factor,
        "original_weight_sum": math.fsum(original_weights),
        "output_weight_sum": math.fsum(output_weights),
        "original_weight_min": min(original_weights),
        "original_weight_max": max(original_weights),
        "output_weight_min": min(output_weights),
        "output_weight_max": max(output_weights),
        "original_weight_histogram_float_hex": _weight_histogram(original_weights),
        "output_weight_histogram_float_hex": _weight_histogram(output_weights),
        "output_weight_sha256_in_row_order": _weight_digest(output_weights),
    }


def _source_manifest_record(
    source: SourceSpec,
    *,
    member_audits: list[dict[str, Any]],
    selected_rows: Mapping[str, int],
    selected_episodes: Mapping[str, int],
    preserved_digests: Mapping[str, str],
) -> dict[str, Any]:
    excluded = {
        split: {
            "rows": rows,
            "episodes": source.expected_split_episodes[split],
        }
        for split, rows in source.expected_split_rows.items()
        if split not in source.selected_splits
    }
    return {
        "role": source.role,
        "archive_sha256": source.archive_sha256,
        "manifest_sha256": source.manifest_sha256,
        "manifest_schema_version": source.manifest_schema,
        "row_schema_version": source.row_schema,
        "declared_split_decisions": dict(source.expected_split_rows),
        "declared_split_episodes": dict(source.expected_split_episodes),
        "selected_splits": list(source.selected_splits),
        "selected_rows": dict(selected_rows),
        "selected_episodes": dict(selected_episodes),
        "selected_member_audit": member_audits,
        "preserved_fields_sha256_by_source_split": dict(preserved_digests),
        "excluded_splits": excluded,
    }


def _assert_zip_metadata(info: zipfile.ZipInfo, compression: int) -> None:
    if info.date_time != ZIP_TIMESTAMP:
        raise ContractError(f"{info.filename}: nondeterministic ZIP timestamp")
    if info.create_system != 3:
        raise ContractError(f"{info.filename}: ZIP create_system drift")
    if info.external_attr != ZIP_EXTERNAL_ATTR:
        raise ContractError(f"{info.filename}: ZIP permissions drift")
    if info.internal_attr != 0 or info.extra != b"" or info.comment != b"":
        raise ContractError(f"{info.filename}: ZIP metadata is not canonical")
    if info.compress_type != compression:
        raise ContractError(f"{info.filename}: ZIP compression drift")


def _verify_output(
    path: Path,
    *,
    expected_manifest: Mapping[str, Any],
    member_records: tuple[MemberRecord, ...],
    contract: BuildContract,
    compression: int,
) -> dict[str, Any]:
    expected_names = [record.name for record in member_records] + ["manifest.json"]
    expected_member_records = {record.name: record for record in member_records}
    counts: Counter[tuple[str, str]] = Counter()
    split_ordinals: Counter[str] = Counter()
    output_weights: dict[tuple[str, str], list[float]] = defaultdict(list)
    episodes: dict[str, set[str]] = defaultdict(set)
    decision_keys: set[str] = set()
    preserved_digests: dict[tuple[str, str], hashlib._Hash] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if names != expected_names or len(names) != len(set(names)):
                raise ContractError(
                    f"output member order/names drift: {names!r} != {expected_names!r}"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ContractError(f"output ZIP integrity failed at {bad_member!r}")
            for info in archive.infolist():
                _assert_zip_metadata(info, compression)
            installed_manifest = orjson.loads(archive.read("manifest.json"))
            if installed_manifest != expected_manifest:
                raise ContractError("installed manifest differs from frozen manifest")

            for info in archive.infolist():
                if not info.filename.endswith(".jsonl"):
                    continue
                split = info.filename.split("/", 1)[0]
                record = expected_member_records[info.filename]
                member_digest = hashlib.sha256()
                member_rows = 0
                member_bytes = 0
                with archive.open(info) as handle:
                    for line_number, line in enumerate(handle, 1):
                        member_digest.update(line)
                        member_bytes += len(line)
                        if not line.strip():
                            raise ContractError(
                                f"{info.filename}:{line_number}: blank output row"
                            )
                        try:
                            row = orjson.loads(line)
                        except orjson.JSONDecodeError as exc:
                            raise ContractError(
                                f"{info.filename}:{line_number}: invalid output JSON"
                            ) from exc
                        if not isinstance(row, dict) or row.get("split") != split:
                            raise ContractError(
                                f"{info.filename}:{line_number}: member/row split mismatch"
                            )
                        ordinal = split_ordinals[split]
                        split_ordinals[split] += 1
                        if split == "train" and ordinal < contract.old_split_rows["train"]:
                            role = "old_gold8_alakazam"
                            expected_schema = contract.old_row_schema
                            factor = contract.old_train_factor
                        elif split in ("train", "valid"):
                            role = "yanz_live_alakazam"
                            expected_schema = contract.yanz_row_schema
                            factor = contract.yanz_factor
                        else:
                            raise ContractError(
                                f"{info.filename}:{line_number}: unknown output split"
                            )
                        schema = row.get("schema_version")
                        if (
                            schema != expected_schema
                            or row.get("deck_hash") != contract.deck_hash
                        ):
                            raise ContractError(
                                f"{info.filename}:{line_number}: output identity drift"
                            )
                        output_weight = row.get("sample_weight")
                        if (
                            isinstance(output_weight, bool)
                            or not isinstance(output_weight, (int, float))
                            or not math.isfinite(float(output_weight))
                            or float(output_weight) <= 0.0
                        ):
                            raise ContractError(
                                f"{info.filename}:{line_number}: invalid output weight"
                            )
                        _, key, episode_id = _validate_source_row(
                            row,
                            source=SourceSpec(
                                role=role,
                                path=path,
                                archive_sha256="",
                                manifest_schema="",
                                row_schema=str(schema),
                                expected_split_rows={},
                                expected_split_episodes={},
                                selected_splits=(split,),
                                split_factors={split: factor},
                                manifest={},
                                manifest_sha256="",
                                members_by_split={},
                            ),
                            source_split=split,
                            member=info.filename,
                            line_number=line_number,
                            deck_hash=contract.deck_hash,
                        )
                        if key in decision_keys:
                            raise ContractError(f"duplicate output decision key {key}")
                        decision_keys.add(key)
                        episodes[role].add(episode_id)
                        counts[(role, split)] += 1
                        output_weights[(role, split)].append(float(output_weight))
                        digest_key = (role, split)
                        digest = preserved_digests.setdefault(
                            digest_key, hashlib.sha256()
                        )
                        digest.update(_preserved_payload(row))
                        member_rows += 1
                if (
                    member_rows != record.rows
                    or member_bytes != record.uncompressed_bytes
                    or member_digest.hexdigest() != record.sha256
                ):
                    raise ContractError(f"{info.filename}: member content audit drift")
    except zipfile.BadZipFile as exc:
        raise ContractError("materialized output is not an intact ZIP") from exc

    expected_counts = {
        ("old_gold8_alakazam", "train"): contract.old_split_rows["train"],
        ("yanz_live_alakazam", "train"): contract.yanz_split_rows["train"],
        ("yanz_live_alakazam", "valid"): contract.yanz_split_rows["valid"],
    }
    if dict(counts) != expected_counts:
        raise ContractError(f"output source/split row counts drift: {dict(counts)!r}")
    overlap = episodes["old_gold8_alakazam"] & episodes["yanz_live_alakazam"]
    if overlap:
        raise ContractError(f"old/yanz output episodes overlap: {sorted(overlap)[:5]}")
    verified_weight_audit: dict[str, dict[str, Any]] = {}
    expected_weight_audit = expected_manifest.get("weight_audit")
    if not isinstance(expected_weight_audit, dict):
        raise ContractError("output manifest has no weight_audit object")
    for (role, split), weights in sorted(output_weights.items()):
        key = f"{role}.{split}"
        wanted = expected_weight_audit.get(key)
        if not isinstance(wanted, dict):
            raise ContractError(f"output manifest has no weight audit for {key}")
        actual = {
            "rows": len(weights),
            "output_weight_sum": math.fsum(weights),
            "output_weight_min": min(weights),
            "output_weight_max": max(weights),
            "output_weight_histogram_float_hex": _weight_histogram(weights),
            "output_weight_sha256_in_row_order": _weight_digest(weights),
        }
        for field, value in actual.items():
            if wanted.get(field) != value:
                raise ContractError(f"output weight audit drift for {key}.{field}")
        verified_weight_audit[key] = actual
    return {
        "zip_integrity": "passed",
        "member_metadata": "deterministic",
        "member_row_counts": {
            record.name: record.rows for record in member_records
        },
        "source_split_rows": {
            f"{role}.{split}": rows
            for (role, split), rows in sorted(counts.items())
        },
        "weight_audit": verified_weight_audit,
        "decision_key_count": len(decision_keys),
        "decision_key_duplicate_count": 0,
        "old_yanz_episode_overlap_count": 0,
        "preserved_fields_sha256": {
            f"{role}.{split}": digest.hexdigest()
            for (role, split), digest in sorted(preserved_digests.items())
        },
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_noreplace(partial: Path, output: Path) -> None:
    try:
        os.link(partial, output)
    except FileExistsError as exc:
        raise FileExistsError(f"{output} exists; refusing to overwrite") from exc
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise ContractError(
                "partial and output must share a filesystem for atomic publication"
            ) from exc
        raise
    os.unlink(partial)
    _fsync_directory(output.parent)


def build_weighted_mix(
    old_archive: Path,
    yanz_archive: Path,
    output: Path,
    *,
    contract: BuildContract = DEFAULT_CONTRACT,
    rows_per_shard: int = DEFAULT_ROWS_PER_SHARD,
    compression: int = ZIP_COMPRESSION,
) -> dict[str, Any]:
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")
    if compression not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError("compression must be ZIP_STORED or ZIP_DEFLATED")
    for label, factor in (
        ("old_train_factor", contract.old_train_factor),
        ("yanz_factor", contract.yanz_factor),
    ):
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"{label} must be finite and positive")

    output = output.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    old_resolved = old_archive.resolve()
    yanz_resolved = yanz_archive.resolve()
    if old_resolved == yanz_resolved:
        raise ValueError("old and yanz inputs must be distinct archives")
    if output in (old_resolved, yanz_resolved):
        raise ValueError("output must differ from both source archives")
    if output.exists():
        raise FileExistsError(f"{output} exists; refusing to overwrite")
    if partial.exists():
        raise FileExistsError(f"{partial} exists; refusing to overwrite")

    old = _preflight_source(
        role="old_gold8_alakazam",
        path=old_resolved,
        expected_sha256=contract.old_archive_sha256,
        manifest_schema=contract.old_manifest_schema,
        row_schema=contract.old_row_schema,
        expected_split_rows=contract.old_split_rows,
        expected_split_episodes=contract.old_split_episodes,
        selected_splits=("train",),
        split_factors={"train": contract.old_train_factor},
        deck_hash=contract.deck_hash,
    )
    yanz = _preflight_source(
        role="yanz_live_alakazam",
        path=yanz_resolved,
        expected_sha256=contract.yanz_archive_sha256,
        manifest_schema=contract.yanz_manifest_schema,
        row_schema=contract.yanz_row_schema,
        expected_split_rows=contract.yanz_split_rows,
        expected_split_episodes=contract.yanz_split_episodes,
        selected_splits=("train", "valid"),
        split_factors={
            "train": contract.yanz_factor,
            "valid": contract.yanz_factor,
        },
        deck_hash=contract.deck_hash,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    created_partial = False
    published = False
    source_row_counts: Counter[tuple[str, str]] = Counter()
    output_split_counts: Counter[str] = Counter()
    source_episode_ids: dict[str, set[str]] = defaultdict(set)
    source_split_episode_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    episode_split_owners: dict[str, str] = {}
    seen_decision_keys: set[str] = set()
    preserved_digests: dict[tuple[str, str], hashlib._Hash] = {}
    original_weights: dict[tuple[str, str], list[float]] = defaultdict(list)
    materialized_weights: dict[tuple[str, str], list[float]] = defaultdict(list)
    member_audits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output_member_records: tuple[MemberRecord, ...] = ()
    manifest: dict[str, Any]

    try:
        raw_handle = partial.open("xb")
        created_partial = True
        try:
            archive = zipfile.ZipFile(
                raw_handle,
                mode="w",
                compression=compression,
                compresslevel=(
                    ZIP_COMPRESSLEVEL
                    if compression == zipfile.ZIP_DEFLATED
                    else None
                ),
                allowZip64=True,
            )
            writer = DeterministicArchiveWriter(
                archive,
                rows_per_shard=rows_per_shard,
                compression=compression,
            )
            try:
                for source in (old, yanz):
                    with zipfile.ZipFile(source.path) as source_zip:
                        for source_split in source.selected_splits:
                            output_split = source_split
                            factor = source.split_factors[source_split]
                            for member in source.members_by_split[source_split]:
                                member_digest = hashlib.sha256()
                                member_bytes = 0
                                member_rows = 0
                                with source_zip.open(member) as handle:
                                    for line_number, line in enumerate(handle, 1):
                                        member_digest.update(line)
                                        member_bytes += len(line)
                                        if not line.strip():
                                            raise ContractError(
                                                f"{source.role}:{member}:{line_number}: "
                                                "blank source row"
                                            )
                                        try:
                                            raw_row = orjson.loads(line)
                                        except orjson.JSONDecodeError as exc:
                                            raise ContractError(
                                                f"{source.role}:{member}:{line_number}: "
                                                "invalid source JSON"
                                            ) from exc
                                        row, decision_key, episode_id = (
                                            _validate_source_row(
                                                raw_row,
                                                source=source,
                                                source_split=source_split,
                                                member=member,
                                                line_number=line_number,
                                                deck_hash=contract.deck_hash,
                                            )
                                        )
                                        if decision_key in seen_decision_keys:
                                            raise ContractError(
                                                f"duplicate decision key {decision_key}"
                                            )
                                        seen_decision_keys.add(decision_key)
                                        previous_split = episode_split_owners.setdefault(
                                            episode_id, output_split
                                        )
                                        if previous_split != output_split:
                                            raise ContractError(
                                                f"episode {episode_id} crosses output "
                                                f"splits {previous_split}/{output_split}"
                                            )
                                        if (
                                            source.role == "yanz_live_alakazam"
                                            and episode_id
                                            in source_episode_ids[
                                                "old_gold8_alakazam"
                                            ]
                                        ):
                                            raise ContractError(
                                                f"old/yanz episode overlap: {episode_id}"
                                            )
                                        source_episode_ids[source.role].add(episode_id)
                                        source_split_episode_ids[
                                            (source.role, source_split)
                                        ].add(episode_id)
                                        digest_key = (source.role, source_split)
                                        digest = preserved_digests.setdefault(
                                            digest_key, hashlib.sha256()
                                        )
                                        digest.update(_preserved_payload(row))
                                        original_weight = float(row["sample_weight"])
                                        output_weight = original_weight * factor
                                        if (
                                            not math.isfinite(output_weight)
                                            or output_weight <= 0.0
                                        ):
                                            raise ContractError(
                                                f"{source.role}:{member}:{line_number}: "
                                                "scaled sample_weight is invalid"
                                            )
                                        original_weights[digest_key].append(
                                            original_weight
                                        )
                                        materialized_weights[digest_key].append(
                                            output_weight
                                        )
                                        output_row = dict(row)
                                        output_row["split"] = output_split
                                        output_row["sample_weight"] = output_weight
                                        writer.write_row(
                                            output_split,
                                            orjson.dumps(output_row) + b"\n",
                                        )
                                        source_row_counts[
                                            (source.role, source_split)
                                        ] += 1
                                        output_split_counts[output_split] += 1
                                        member_rows += 1
                                member_audits[source.role].append(
                                    {
                                        "member": member,
                                        "rows": member_rows,
                                        "uncompressed_bytes": member_bytes,
                                        "sha256": member_digest.hexdigest(),
                                    }
                                )

                expected_source_counts = {
                    ("old_gold8_alakazam", "train"): contract.old_split_rows[
                        "train"
                    ],
                    ("yanz_live_alakazam", "train"): contract.yanz_split_rows[
                        "train"
                    ],
                    ("yanz_live_alakazam", "valid"): contract.yanz_split_rows[
                        "valid"
                    ],
                }
                if dict(source_row_counts) != expected_source_counts:
                    raise ContractError(
                        "selected source row counts drift: "
                        f"{dict(source_row_counts)!r} != {expected_source_counts!r}"
                    )
                for source in (old, yanz):
                    for split in source.selected_splits:
                        actual = len(
                            source_split_episode_ids[(source.role, split)]
                        )
                        wanted = source.expected_split_episodes[split]
                        if actual != wanted:
                            raise ContractError(
                                f"{source.role}.{split} episode count drift: "
                                f"{actual} != {wanted}"
                            )
                old_yanz_overlap = (
                    source_episode_ids["old_gold8_alakazam"]
                    & source_episode_ids["yanz_live_alakazam"]
                )
                if old_yanz_overlap:
                    raise ContractError(
                        "old/yanz episode overlap is non-zero: "
                        f"{sorted(old_yanz_overlap)[:5]}"
                    )

                output_member_records = writer.finish_rows()
                source_records = []
                for source in (old, yanz):
                    selected_rows = {
                        split: source_row_counts[(source.role, split)]
                        for split in source.selected_splits
                    }
                    selected_episodes = {
                        split: len(
                            source_split_episode_ids[(source.role, split)]
                        )
                        for split in source.selected_splits
                    }
                    digest_map = {
                        split: preserved_digests[(source.role, split)].hexdigest()
                        for split in source.selected_splits
                    }
                    source_records.append(
                        _source_manifest_record(
                            source,
                            member_audits=member_audits[source.role],
                            selected_rows=selected_rows,
                            selected_episodes=selected_episodes,
                            preserved_digests=digest_map,
                        )
                    )

                weight_audit = {
                    f"{source.role}.{split}": _weight_audit(
                        original_weights[(source.role, split)],
                        materialized_weights[(source.role, split)],
                        source.split_factors[split],
                    )
                    for source in (old, yanz)
                    for split in source.selected_splits
                }
                old_weight_mass = weight_audit[
                    "old_gold8_alakazam.train"
                ]["output_weight_sum"]
                yanz_weight_mass = weight_audit[
                    "yanz_live_alakazam.train"
                ]["output_weight_sum"]
                total_weight_mass = old_weight_mass + yanz_weight_mass
                output_split_episodes = {
                    "train": sum(
                        1
                        for split in episode_split_owners.values()
                        if split == "train"
                    ),
                    "valid": sum(
                        1
                        for split in episode_split_owners.values()
                        if split == "valid"
                    ),
                }
                compression_name = (
                    "deflated" if compression == zipfile.ZIP_DEFLATED else "stored"
                )
                manifest = {
                    "schema_version": OUTPUT_MANIFEST_SCHEMA,
                    "competition": "pokemon-tcg-ai-battle",
                    "deck_hash": contract.deck_hash,
                    "purpose": (
                        "BC-dominant Alakazam policy/count adaptation: retain the "
                        "frozen old gold8 train support at a 0.25 source-weight "
                        "factor, retain yanz live source weights at factor 1.0, "
                        "and keep value loss uniformly weighted at 1.0 per row."
                    ),
                    "provenance": {
                        "supersedes": {
                            "artifact": REJECTED_ALL_LOSSES_ARTIFACT,
                            "sha256": REJECTED_ALL_LOSSES_SHA256,
                            "status": "rejected_metadata_drift",
                            "reason": (
                                "predecessor manifest incorrectly declared "
                                "trajectory_weight_scope=all_losses instead of "
                                "the preregistered policy_only scope"
                            ),
                        },
                        "row_payload_change_from_superseded_artifact": False,
                        "manifest_contract_change": (
                            "policy_only weighting with uniform value loss"
                        ),
                    },
                    "row_schema_contract": {
                        "mode": "preserve_source_schema_per_row",
                        "mixed_row_schemas": (
                            contract.old_row_schema != contract.yanz_row_schema
                        ),
                        "row_schema_versions": {
                            "old_gold8_alakazam": contract.old_row_schema,
                            "yanz_live_alakazam": contract.yanz_row_schema,
                        },
                        "mutated_fields_only": ["split", "sample_weight"],
                        "all_other_row_fields": "preserved exactly by value",
                    },
                    "source_manifest_schema_contract": {
                        "mixed_source_manifest_schemas": (
                            contract.old_manifest_schema
                            != contract.yanz_manifest_schema
                        ),
                        "manifest_schema_versions": {
                            "old_gold8_alakazam": contract.old_manifest_schema,
                            "yanz_live_alakazam": contract.yanz_manifest_schema,
                        },
                        "note": (
                            "source ZIP manifest schemas differ independently "
                            "from the decision-row schema"
                        ),
                    },
                    "split_policy": {
                        "mode": "frozen_source_view",
                        "train": [
                            {
                                "source": "old_gold8_alakazam.train",
                                "rows": contract.old_split_rows["train"],
                                "sample_weight_factor": contract.old_train_factor,
                            },
                            {
                                "source": "yanz_live_alakazam.train",
                                "rows": contract.yanz_split_rows["train"],
                                "sample_weight_factor": contract.yanz_factor,
                            },
                        ],
                        "valid": [
                            {
                                "source": "yanz_live_alakazam.valid",
                                "rows": contract.yanz_split_rows["valid"],
                                "sample_weight_factor": contract.yanz_factor,
                            }
                        ],
                        "excluded": [
                            "old_gold8_alakazam.valid",
                            "old_gold8_alakazam.test",
                        ],
                        "test": [],
                    },
                    "training_semantics": {
                        "physical_row_sampling": "without duplication or resampling",
                        "sample_weight_meaning": (
                            "source sample_weight multiplied by the declared "
                            "source factor; result weights pointer, set-BCE, and "
                            "flexible-count policy losses only"
                        ),
                        "value_loss_weighting": {
                            "mode": "uniform",
                            "per_row_weight": 1.0,
                            "archive_sample_weight_ignored": True,
                        },
                        "required_train_bc_orbit_flags": [
                            "--use-trajectory-weights",
                            "--trajectory-weight-scope",
                            "policy_only",
                        ],
                        "policy_team_balance": (
                            "must remain disabled because train_bc_orbit rejects its "
                            "combination with --use-trajectory-weights"
                        ),
                        "effective_train_weight_mass": {
                            "old_gold8_alakazam": old_weight_mass,
                            "yanz_live_alakazam": yanz_weight_mass,
                            "total": total_weight_mass,
                            "yanz_fraction": yanz_weight_mass / total_weight_mass,
                        },
                        "validation": {
                            "dataset_role": (
                                "yanz valid is a consumed hash holdout/dev set, not "
                                "an independent time-forward final set"
                            ),
                            "value_loss": (
                                "uniform per-row weight 1.0; archive sample_weight "
                                "must not affect value validation or value training"
                            ),
                        },
                    },
                    "source_archives": source_records,
                    "weight_audit": weight_audit,
                    "split_decisions": {
                        "train": output_split_counts["train"],
                        "valid": output_split_counts["valid"],
                        "test": 0,
                    },
                    "split_episodes": {
                        **output_split_episodes,
                        "test": 0,
                    },
                    "episode_audit": {
                        "old_gold8_selected": len(
                            source_episode_ids["old_gold8_alakazam"]
                        ),
                        "yanz_selected": len(
                            source_episode_ids["yanz_live_alakazam"]
                        ),
                        "old_yanz_overlap_count": 0,
                        "cross_output_split_overlap_count": 0,
                    },
                    "decision_keys": {
                        "format": DECISION_KEY_FORMAT,
                        "count": len(seen_decision_keys),
                        "duplicate_count": 0,
                        "sha256_sorted": _digest_lines(seen_decision_keys),
                    },
                    "output_members": [
                        {
                            "name": record.name,
                            "split": record.split,
                            "rows": record.rows,
                            "uncompressed_bytes": record.uncompressed_bytes,
                            "sha256": record.sha256,
                        }
                        for record in output_member_records
                    ],
                    "zip_contract": {
                        "member_order": "train shards, valid shards, manifest.json",
                        "member_timestamp": "1980-01-01T00:00:00",
                        "create_system": 3,
                        "external_attr_octal": "0600",
                        "compression": compression_name,
                        "compresslevel": (
                            ZIP_COMPRESSLEVEL
                            if compression == zipfile.ZIP_DEFLATED
                            else None
                        ),
                        "rows_per_shard": rows_per_shard,
                        "atomic_publish": "same-directory hard-link no-replace",
                        "overwrite_policy": "forbidden",
                    },
                }
                writer.write_manifest(manifest)
            finally:
                writer._close_member()
                archive.close()
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        finally:
            raw_handle.close()

        verification = _verify_output(
            partial,
            expected_manifest=manifest,
            member_records=output_member_records,
            contract=contract,
            compression=compression,
        )
        expected_preserved = {
            f"{role}.{split}": digest.hexdigest()
            for (role, split), digest in sorted(preserved_digests.items())
        }
        if verification["preserved_fields_sha256"] != expected_preserved:
            raise ContractError("non-mutated row fields were not preserved")
        _publish_noreplace(partial, output)
        published = True
    finally:
        if created_partial and not published and partial.exists():
            partial.unlink()

    output_sha256 = sha256_file(output)
    return {
        "output": str(output),
        "output_sha256": output_sha256,
        "output_size_bytes": output.stat().st_size,
        "manifest_schema_version": OUTPUT_MANIFEST_SCHEMA,
        "split_decisions": manifest["split_decisions"],
        "split_episodes": manifest["split_episodes"],
        "source_archive_sha256": {
            "old_gold8_alakazam": old.archive_sha256,
            "yanz_live_alakazam": yanz.archive_sha256,
        },
        "verification": verification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen old025+yanz1 Alakazam BC mix without overwriting."
        )
    )
    parser.add_argument("--old-archive", type=Path, required=True)
    parser.add_argument("--yanz-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_weighted_mix(
        args.old_archive,
        args.yanz_archive,
        args.output,
        rows_per_shard=args.rows_per_shard,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
