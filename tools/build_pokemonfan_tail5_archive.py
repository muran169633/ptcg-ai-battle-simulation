#!/usr/bin/env python3
"""Build the frozen Pokemon Fan fresh-tail5 archive after the P32 prefix.

The source archive contains one train shard in source order.  The P32 special
BC cache consumed its first 8,192 eligible rows.  This builder selects, without
replacement, the first five context-34 rows and the first 1,275 ordinary rows
from the remaining source-order tail.  It preserves the selected JSONL bytes
exactly, publishes one deterministic train shard, and fails closed on any
source, identity, count, digest, or output-path drift.

``--dry-run`` performs the complete scan and deterministic ZIP construction in
memory, but never creates the output path.  A real build uses a same-directory
temporary file plus an atomic hard link so an output that appears concurrently
cannot be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import orjson


SCHEMA_VERSION = "ptcg-pokemonfan-fresh-tail5-archive-v1"
SOURCE_SCHEMA_VERSION = "ptcg-bc-visible-decisions-v1"
ROW_SCHEMA_VERSION = "ptcg-bc-visible-decisions-v1"
DECISION_KEY_FORMAT = "dataset_date|episode_id|seat|action_step_index"
EPISODE_KEY_FORMAT = "dataset_date|episode_id"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_"
    "train28_valid29_special_20260801.zip"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_"
    "special_tail5_after_p32_20260801.zip"
)

SOURCE_SHA256 = "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598"
SOURCE_MANIFEST_SHA256 = (
    "884f6ef27847b9f333c66a2b888e8708cf1139e61d3c667ce6d2bc3d9f3c9b6c"
)
SOURCE_TRAIN_MEMBER = "train/part-00000.jsonl"
SOURCE_TRAIN_MEMBER_SHA256 = (
    "1cd6e76e7e60446bccb2966d5d63bf44c30eb73956cc3986721351b6e966dbde"
)
SOURCE_VALID_MEMBER = "valid/part-00000.jsonl"
SOURCE_VALID_MEMBER_SHA256 = (
    "576040903f386b7dc59570e644eaaf0993c3f068909b93e012e8a5787d3947e3"
)
SOURCE_MEMBER_NAMES = {
    SOURCE_TRAIN_MEMBER,
    SOURCE_VALID_MEMBER,
    "manifest.json",
}

TEAM_NAME = "Pokemon Fan"
DECK_HASH = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
CONTEXT34 = "34"
SOURCE_TRAIN_ROWS = 9_487
SOURCE_TRAIN_CONTEXT34_ROWS = 38
SOURCE_VALID_ROWS = 15_152
SOURCE_VALID_CONTEXT34_ROWS = 58
P32_PREFIX_ROWS = 8_192
P32_PREFIX_ORDINARY_ROWS = 8_160
P32_PREFIX_CONTEXT34_ROWS = 32
TAIL_REMAINDER_ROWS = 1_295
TAIL_REMAINDER_ORDINARY_ROWS = 1_289
TAIL_REMAINDER_CONTEXT34_ROWS = 6
SELECTED_ORDINARY_ROWS = 1_275
SELECTED_CONTEXT34_ROWS = 5
SELECTED_ROWS = 1_280
UNUSED_ORDINARY_ROWS = 14
UNUSED_CONTEXT34_ROWS = 1
UNUSED_ROWS = 15
TRAIN_MEMBER = "train/part-00000.jsonl"

P32_PREFIX_DECISION_KEYS_DIGEST = (
    "0aa87ca9da7ed669f359706f3ed6e96dcc5c11d86df349e9f16493806074a636"
)
TAIL_DECISION_KEYS_DIGEST = (
    "5a0d3dc6987edb4def344054f6ce7deb554ebc6d48006f6726aaa5dea4d62837"
)
COMBINED37_DECISION_KEYS_DIGEST = (
    "d1a43d1f113c67133bc17de09fa782665badb5d58924ff5744368b10f341dde7"
)
RAW_SELECTED_CONTENT_SHA256 = (
    "505fca08a6ecccca3e7bb46f165d8bc26c8efd442a00ace3b387b45f429c54e7"
)

ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_COMPRESSION = zipfile.ZIP_DEFLATED
ZIP_COMPRESSLEVEL = 6
SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class MemberAudit:
    name: str
    sha256: str
    rows: int
    ordinary_rows: int
    context34_rows: int
    decision_keys: tuple[str, ...]
    episode_keys: frozenset[str]


@dataclass(frozen=True)
class SelectionAudit:
    source_manifest: dict[str, Any]
    train: MemberAudit
    valid: MemberAudit
    p32_keys: tuple[str, ...]
    selected_keys: tuple[str, ...]
    selected_lines: tuple[bytes, ...]
    unused_keys: tuple[str, ...]
    selected_context34_keys: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def digest_bytes(values: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = ZIP_COMPRESSION
    info.create_system = 3
    info.external_attr = 0o644 << 16
    return info


def require_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer, got {value!r}")
    return value


def decision_identity(row: dict[str, Any], where: str) -> tuple[str, str]:
    dataset_date = row.get("dataset_date")
    episode_id = row.get("episode_id")
    if (
        not isinstance(dataset_date, str)
        or not dataset_date
        or "|" in dataset_date
    ):
        raise RuntimeError(f"{where}: invalid dataset_date {dataset_date!r}")
    if (
        not isinstance(episode_id, (str, int))
        or not str(episode_id)
        or "|" in str(episode_id)
    ):
        raise RuntimeError(f"{where}: invalid episode_id {episode_id!r}")
    seat = require_integer(row.get("seat"), f"{where}: seat")
    action_step = require_integer(
        row.get("action_step_index"),
        f"{where}: action_step_index",
    )
    if seat < 0 or action_step < 0:
        raise RuntimeError(f"{where}: seat and action_step_index must be non-negative")
    episode_key = f"{dataset_date}|{episode_id}"
    return f"{episode_key}|{seat}|{action_step}", episode_key


def validate_row(
    value: Any,
    *,
    split: str,
    member: str,
    line_number: int,
) -> tuple[str, str, bool]:
    where = f"{member}:{line_number}"
    if not isinstance(value, dict):
        raise RuntimeError(f"{where}: row is not an object")
    expected = {
        "schema_version": ROW_SCHEMA_VERSION,
        "split": split,
        "team_name": TEAM_NAME,
        "deck_hash": DECK_HASH,
    }
    mismatches = [
        f"{name}={value.get(name)!r} expected {wanted!r}"
        for name, wanted in expected.items()
        if value.get(name) != wanted
    ]
    if mismatches:
        raise RuntimeError(f"{where}: row identity drift: {'; '.join(mismatches)}")
    if "visualize" in value or (
        isinstance(value.get("observation"), dict)
        and "visualize" in value["observation"]
    ):
        raise RuntimeError(f"{where}: hidden visualize payload is forbidden")
    decision_key, episode_key = decision_identity(value, where)
    return decision_key, episode_key, str(value.get("select_context", "")) == CONTEXT34


def validate_source_manifest(payload: bytes) -> dict[str, Any]:
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            "source manifest SHA256 mismatch: "
            f"expected {SOURCE_MANIFEST_SHA256}, got {actual_sha}"
        )
    try:
        manifest = orjson.loads(payload)
    except orjson.JSONDecodeError as exc:
        raise RuntimeError("source manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("source manifest is not an object")
    checks = {
        "schema_version": (manifest.get("schema_version"), SOURCE_SCHEMA_VERSION),
        "team_name_filter": (manifest.get("team_name_filter"), TEAM_NAME),
        "deck_hash_filter": (manifest.get("deck_hash_filter"), DECK_HASH),
        "split_decisions.train": (
            (manifest.get("split_decisions") or {}).get("train"),
            SOURCE_TRAIN_ROWS,
        ),
        "split_decisions.valid": (
            (manifest.get("split_decisions") or {}).get("valid"),
            SOURCE_VALID_ROWS,
        ),
        "split_decisions.test": (
            (manifest.get("split_decisions") or {}).get("test"),
            0,
        ),
        "shards.train": ((manifest.get("shards") or {}).get("train"), 1),
        "shards.valid": ((manifest.get("shards") or {}).get("valid"), 1),
        "shards.test": ((manifest.get("shards") or {}).get("test"), 0),
    }
    failures = [
        f"{name}={actual!r} expected {wanted!r}"
        for name, (actual, wanted) in checks.items()
        if actual != wanted
    ]
    if failures:
        raise RuntimeError("source manifest drift: " + "; ".join(failures))
    return manifest


def scan_member(
    archive: zipfile.ZipFile,
    name: str,
    split: str,
    expected_sha256: str,
) -> tuple[MemberAudit, list[tuple[bytes, str, bool]]]:
    digest = hashlib.sha256()
    keys: list[str] = []
    episode_keys: set[str] = set()
    rows: list[tuple[bytes, str, bool]] = []
    context34_rows = 0
    with archive.open(name) as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.endswith(b"\n"):
                raise RuntimeError(f"{name}:{line_number}: JSONL row lacks LF terminator")
            digest.update(raw_line)
            try:
                row = orjson.loads(raw_line)
            except orjson.JSONDecodeError as exc:
                raise RuntimeError(f"{name}:{line_number}: invalid JSON") from exc
            key, episode_key, is_context34 = validate_row(
                row,
                split=split,
                member=name,
                line_number=line_number,
            )
            keys.append(key)
            episode_keys.add(episode_key)
            context34_rows += int(is_context34)
            if split == "train":
                rows.append((raw_line, key, is_context34))
    member_sha256 = digest.hexdigest()
    if member_sha256 != expected_sha256:
        raise RuntimeError(
            f"{name} SHA256 mismatch: expected {expected_sha256}, got {member_sha256}"
        )
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        raise RuntimeError(f"{name} has {duplicate_count} duplicate decision keys")
    return (
        MemberAudit(
            name=name,
            sha256=member_sha256,
            rows=len(keys),
            ordinary_rows=len(keys) - context34_rows,
            context34_rows=context34_rows,
            decision_keys=tuple(keys),
            episode_keys=frozenset(episode_keys),
        ),
        rows,
    )


def scan_and_select(source: Path, expected_source_sha256: str) -> SelectionAudit:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"source must be a regular non-symlink file: {source}")
    actual_source_sha256 = sha256_file(source)
    if actual_source_sha256 != expected_source_sha256:
        raise RuntimeError(
            "source archive SHA256 mismatch: "
            f"expected {expected_source_sha256}, got {actual_source_sha256}"
        )
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("source archive has duplicate member names")
        if set(names) != SOURCE_MEMBER_NAMES:
            raise RuntimeError(
                "source member set drift: "
                f"expected {sorted(SOURCE_MEMBER_NAMES)!r}, got {sorted(names)!r}"
            )
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"source ZIP CRC failure in {corrupt_member}")
        source_manifest = validate_source_manifest(archive.read("manifest.json"))
        train, raw_train_rows = scan_member(
            archive,
            SOURCE_TRAIN_MEMBER,
            "train",
            SOURCE_TRAIN_MEMBER_SHA256,
        )
        valid, _ = scan_member(
            archive,
            SOURCE_VALID_MEMBER,
            "valid",
            SOURCE_VALID_MEMBER_SHA256,
        )

    expected_counts = {
        "train.rows": (train.rows, SOURCE_TRAIN_ROWS),
        "train.context34_rows": (
            train.context34_rows,
            SOURCE_TRAIN_CONTEXT34_ROWS,
        ),
        "valid.rows": (valid.rows, SOURCE_VALID_ROWS),
        "valid.context34_rows": (
            valid.context34_rows,
            SOURCE_VALID_CONTEXT34_ROWS,
        ),
    }
    failures = [
        f"{name}={actual} expected {wanted}"
        for name, (actual, wanted) in expected_counts.items()
        if actual != wanted
    ]
    if failures:
        raise RuntimeError("source row-count drift: " + "; ".join(failures))
    decision_overlap = set(train.decision_keys) & set(valid.decision_keys)
    if decision_overlap:
        raise RuntimeError("source train/valid decision-key overlap is non-zero")
    episode_overlap = train.episode_keys & valid.episode_keys
    if episode_overlap:
        raise RuntimeError("source train/valid episode-key overlap is non-zero")

    prefix = raw_train_rows[:P32_PREFIX_ROWS]
    remainder = raw_train_rows[P32_PREFIX_ROWS:]
    prefix_context34 = sum(int(is_context34) for _raw, _key, is_context34 in prefix)
    remainder_context34 = sum(
        int(is_context34) for _raw, _key, is_context34 in remainder
    )
    prefix_checks = {
        "P32 prefix rows": (len(prefix), P32_PREFIX_ROWS),
        "P32 prefix ordinary rows": (
            len(prefix) - prefix_context34,
            P32_PREFIX_ORDINARY_ROWS,
        ),
        "P32 prefix context34 rows": (
            prefix_context34,
            P32_PREFIX_CONTEXT34_ROWS,
        ),
        "tail remainder rows": (len(remainder), TAIL_REMAINDER_ROWS),
        "tail remainder ordinary rows": (
            len(remainder) - remainder_context34,
            TAIL_REMAINDER_ORDINARY_ROWS,
        ),
        "tail remainder context34 rows": (
            remainder_context34,
            TAIL_REMAINDER_CONTEXT34_ROWS,
        ),
    }
    prefix_failures = [
        f"{name}={actual} expected {wanted}"
        for name, (actual, wanted) in prefix_checks.items()
        if actual != wanted
    ]
    if prefix_failures:
        raise RuntimeError("P32/tail lineage drift: " + "; ".join(prefix_failures))

    selected: list[tuple[bytes, str, bool]] = []
    unused: list[tuple[bytes, str, bool]] = []
    selected_ordinary = 0
    selected_context34 = 0
    for row in remainder:
        is_context34 = row[2]
        take = (
            selected_context34 < SELECTED_CONTEXT34_ROWS
            if is_context34
            else selected_ordinary < SELECTED_ORDINARY_ROWS
        )
        if take:
            selected.append(row)
            selected_context34 += int(is_context34)
            selected_ordinary += int(not is_context34)
        else:
            unused.append(row)

    selection_checks = {
        "selected rows": (len(selected), SELECTED_ROWS),
        "selected ordinary rows": (selected_ordinary, SELECTED_ORDINARY_ROWS),
        "selected context34 rows": (selected_context34, SELECTED_CONTEXT34_ROWS),
        "unused rows": (len(unused), UNUSED_ROWS),
        "unused ordinary rows": (
            sum(int(not row[2]) for row in unused),
            UNUSED_ORDINARY_ROWS,
        ),
        "unused context34 rows": (
            sum(int(row[2]) for row in unused),
            UNUSED_CONTEXT34_ROWS,
        ),
    }
    selection_failures = [
        f"{name}={actual} expected {wanted}"
        for name, (actual, wanted) in selection_checks.items()
        if actual != wanted
    ]
    if selection_failures:
        raise RuntimeError("fresh-tail5 selection drift: " + "; ".join(selection_failures))

    p32_keys = tuple(key for _raw, key, _context in prefix)
    selected_keys = tuple(key for _raw, key, _context in selected)
    selected_lines = tuple(raw for raw, _key, _context in selected)
    unused_keys = tuple(key for _raw, key, _context in unused)
    selected_context34_keys = tuple(
        key for _raw, key, is_context34 in selected if is_context34
    )
    observed_digests = {
        "P32 prefix decision keys": (
            digest_lines(p32_keys),
            P32_PREFIX_DECISION_KEYS_DIGEST,
        ),
        "tail5 decision keys": (
            digest_lines(selected_keys),
            TAIL_DECISION_KEYS_DIGEST,
        ),
        "combined37 decision keys": (
            digest_lines((*p32_keys, *selected_keys)),
            COMBINED37_DECISION_KEYS_DIGEST,
        ),
        "raw selected content": (
            digest_bytes(selected_lines),
            RAW_SELECTED_CONTENT_SHA256,
        ),
    }
    digest_failures = [
        f"{name}={actual} expected {wanted}"
        for name, (actual, wanted) in observed_digests.items()
        if actual != wanted
    ]
    if digest_failures:
        raise RuntimeError("frozen digest drift: " + "; ".join(digest_failures))
    if set(p32_keys) & set(selected_keys):
        raise RuntimeError("fresh tail reuses a P32 decision key")
    if set(selected_keys) & set(unused_keys):
        raise RuntimeError("selected and unused tail decision keys overlap")
    if len(set((*p32_keys, *selected_keys))) != P32_PREFIX_ROWS + SELECTED_ROWS:
        raise RuntimeError("combined37 decision keys are not unique")

    return SelectionAudit(
        source_manifest=source_manifest,
        train=train,
        valid=valid,
        p32_keys=p32_keys,
        selected_keys=selected_keys,
        selected_lines=selected_lines,
        unused_keys=unused_keys,
        selected_context34_keys=selected_context34_keys,
    )


def build_manifest(source: Path, audit: SelectionAudit) -> dict[str, Any]:
    source_split_policy = audit.source_manifest.get("split_policy") or {}
    selected_episode_keys = {
        "|".join(key.split("|")[:2]) for key in audit.selected_keys
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": display_path(source),
            "sha256": SOURCE_SHA256,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "schema_version": SOURCE_SCHEMA_VERSION,
            "train_member": {
                "name": SOURCE_TRAIN_MEMBER,
                "sha256": SOURCE_TRAIN_MEMBER_SHA256,
                "rows": audit.train.rows,
                "ordinary_rows": audit.train.ordinary_rows,
                "context34_rows": audit.train.context34_rows,
            },
            "valid_member": {
                "name": SOURCE_VALID_MEMBER,
                "sha256": SOURCE_VALID_MEMBER_SHA256,
                "rows": audit.valid.rows,
                "ordinary_rows": audit.valid.ordinary_rows,
                "context34_rows": audit.valid.context34_rows,
            },
            "team_name": TEAM_NAME,
            "deck_hash": DECK_HASH,
            "train_dates": source_split_policy.get("train_dates"),
            "valid_dates": source_split_policy.get("valid_dates"),
        },
        "lineage": {
            "parent_stage": "PokemonFan special BC P32",
            "p32_source_prefix": {
                "selection": (
                    "first 8192 eligible source-order train rows; the frozen "
                    "source has one train shard and every row is retained"
                ),
                "rows": P32_PREFIX_ROWS,
                "ordinary_rows": P32_PREFIX_ORDINARY_ROWS,
                "context34_rows": P32_PREFIX_CONTEXT34_ROWS,
                "decision_keys_digest": P32_PREFIX_DECISION_KEYS_DIGEST,
            },
            "source_tail_after_p32": {
                "rows": TAIL_REMAINDER_ROWS,
                "ordinary_rows": TAIL_REMAINDER_ORDINARY_ROWS,
                "context34_rows": TAIL_REMAINDER_CONTEXT34_ROWS,
            },
            "p32_tail_decision_key_overlap_count": 0,
        },
        "selection": {
            "algorithm": "source_order_remaining_after_exact_p32_prefix",
            "seed": None,
            "without_replacement": True,
            "source_order_preserved": True,
            "ordinary_rule": (
                "first 1275 ordinary rows after the exact P32 source prefix"
            ),
            "context34_rule": (
                "first 5 context-34 rows after the exact P32 source prefix"
            ),
            "stable_decision_key_format": DECISION_KEY_FORMAT,
            "ordinary_rows": SELECTED_ORDINARY_ROWS,
            "context34_rows": SELECTED_CONTEXT34_ROWS,
            "rows": SELECTED_ROWS,
            "context34_source_order_decision_keys": list(
                audit.selected_context34_keys
            ),
        },
        "train": {
            "members": [
                {
                    "member": TRAIN_MEMBER,
                    "sha256": RAW_SELECTED_CONTENT_SHA256,
                    "rows": SELECTED_ROWS,
                    "ordinary_rows": SELECTED_ORDINARY_ROWS,
                    "context34_rows": SELECTED_CONTEXT34_ROWS,
                    "source_order_preserved": True,
                }
            ],
            "member_count": 1,
            "total_rows": SELECTED_ROWS,
            "ordinary_rows": SELECTED_ORDINARY_ROWS,
            "context34_rows": SELECTED_CONTEXT34_ROWS,
            "episodes": len(selected_episode_keys),
            "raw_selected_content_sha256": RAW_SELECTED_CONTENT_SHA256,
        },
        "unused_tail": {
            "rows": UNUSED_ROWS,
            "ordinary_rows": UNUSED_ORDINARY_ROWS,
            "context34_rows": UNUSED_CONTEXT34_ROWS,
            "decision_keys_digest": digest_lines(audit.unused_keys),
        },
        "decision_keys": {
            "scope": "selected fresh-tail5 train decisions",
            "format": DECISION_KEY_FORMAT,
            "digest_algorithm": "sha256(sorted(key + LF))",
            "digest": TAIL_DECISION_KEYS_DIGEST,
            "count": SELECTED_ROWS,
            "duplicate_count": 0,
            "without_replacement": True,
            "p32_prefix": {
                "digest": P32_PREFIX_DECISION_KEYS_DIGEST,
                "count": P32_PREFIX_ROWS,
                "duplicate_count": 0,
            },
            "combined_p32_plus_tail": {
                "digest": COMBINED37_DECISION_KEYS_DIGEST,
                "count": P32_PREFIX_ROWS + SELECTED_ROWS,
                "duplicate_count": 0,
                "overlap_count": 0,
            },
        },
        "split_integrity": {
            "episode_key_format": EPISODE_KEY_FORMAT,
            "source_train_valid_episode_overlap_count": 0,
            "source_train_valid_decision_overlap_count": 0,
            "valid_rows_copied": 0,
            "validation_rows_used_for_training": 0,
        },
        "byte_preservation": {
            "selected_rows_copied_verbatim": True,
            "row_schema_version": ROW_SCHEMA_VERSION,
            "sample_weight_modified": False,
            "raw_selected_content_sha256": RAW_SELECTED_CONTENT_SHA256,
        },
        "intended_replay": {
            "batches": 5,
            "rows_per_batch": 256,
            "context34_rows_per_batch": 1,
            "ordinary_rows_per_batch": 255,
            "note": (
                "the executor must stratify the one source-order shard; raw "
                "contiguous 256-row slices are not the frozen replay batches"
            ),
        },
        "zip": {
            "timestamp": list(ZIP_TIMESTAMP),
            "compression": "deflate",
            "compresslevel": ZIP_COMPRESSLEVEL,
            "member_mode": "0644",
            "member_order": [TRAIN_MEMBER, "manifest.json"],
        },
    }


def create_archive_bytes(
    manifest: dict[str, Any],
    selected_lines: tuple[bytes, ...],
) -> tuple[bytes, str]:
    member_payload = b"".join(selected_lines)
    if hashlib.sha256(member_payload).hexdigest() != RAW_SELECTED_CONTENT_SHA256:
        raise RuntimeError("selected member payload changed before ZIP serialization")
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=ZIP_COMPRESSION,
        compresslevel=ZIP_COMPRESSLEVEL,
        allowZip64=True,
    ) as archive:
        archive.writestr(zip_info(TRAIN_MEMBER), member_payload)
        archive.writestr(zip_info("manifest.json"), manifest_payload)
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != [TRAIN_MEMBER, "manifest.json"]:
            raise RuntimeError("deterministic output member order drifted")
        if archive.testzip() is not None:
            raise RuntimeError("deterministic output ZIP failed its CRC audit")
        if hashlib.sha256(archive.read(TRAIN_MEMBER)).hexdigest() != (
            RAW_SELECTED_CONTENT_SHA256
        ):
            raise RuntimeError("output train member did not preserve selected bytes")
        if archive.read("manifest.json") != manifest_payload:
            raise RuntimeError("output manifest bytes changed during serialization")
    return payload, hashlib.sha256(manifest_payload).hexdigest()


def publish_no_clobber(output: Path, payload: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".partial",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing output: {output}"
            ) from exc
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build(
    *,
    source: Path,
    output: Path,
    expected_source_sha256: str,
    expected_output_sha256: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to reuse existing output: {output}")
    if source == output:
        raise ValueError("source and output paths must differ")
    audit = scan_and_select(source, expected_source_sha256)
    manifest = build_manifest(source, audit)
    archive_payload, manifest_sha256 = create_archive_bytes(
        manifest,
        audit.selected_lines,
    )
    archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
    if (
        expected_output_sha256 is not None
        and archive_sha256 != expected_output_sha256
    ):
        raise RuntimeError(
            "output archive SHA256 mismatch: "
            f"expected {expected_output_sha256}, got {archive_sha256}"
        )
    if not dry_run:
        publish_no_clobber(output, archive_payload)
        if sha256_file(output) != archive_sha256:
            raise RuntimeError("published archive SHA256 differs from in-memory audit")
    return {
        "status": "dry_run_passed" if dry_run else "built",
        "dry_run": dry_run,
        "output_written": not dry_run,
        "source": display_path(source),
        "source_sha256": expected_source_sha256,
        "output": display_path(output),
        "archive_sha256": archive_sha256,
        "manifest_sha256": manifest_sha256,
        "raw_selected_content_sha256": RAW_SELECTED_CONTENT_SHA256,
        "train_member_sha256": RAW_SELECTED_CONTENT_SHA256,
        "train_rows": SELECTED_ROWS,
        "ordinary_rows": SELECTED_ORDINARY_ROWS,
        "context34_rows": SELECTED_CONTEXT34_ROWS,
        "decision_keys_digest": TAIL_DECISION_KEYS_DIGEST,
        "p32_prefix_decision_keys_digest": P32_PREFIX_DECISION_KEYS_DIGEST,
        "combined_p32_plus_tail_decision_keys_digest": (
            COMBINED37_DECISION_KEYS_DIGEST
        ),
        "p32_tail_overlap_count": 0,
        "source_train_valid_episode_overlap_count": 0,
        "valid_rows_used_for_training": 0,
    }


def parse_sha256(value: str, label: str) -> str:
    lowered = value.lower()
    if not SHA256_RE.fullmatch(lowered):
        raise argparse.ArgumentTypeError(f"{label} must be a SHA256")
    return lowered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="frozen Pokemon Fan source archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="new no-clobber derived archive",
    )
    parser.add_argument(
        "--expected-source-sha256",
        default=SOURCE_SHA256,
        type=lambda value: parse_sha256(value, "--expected-source-sha256"),
    )
    parser.add_argument(
        "--expected-output-sha256",
        default=None,
        type=lambda value: parse_sha256(value, "--expected-output-sha256"),
        help="optional deterministic output-archive hash gate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fully audit and serialize in memory without creating --output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        source=args.source,
        output=args.output,
        expected_source_sha256=args.expected_source_sha256,
        expected_output_sha256=args.expected_output_sha256,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
