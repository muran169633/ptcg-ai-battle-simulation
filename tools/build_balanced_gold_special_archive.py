#!/usr/bin/env python3
"""Build the frozen Gold-core balanced special-BC replay archive.

The builder is deliberately narrow and fail closed.  It accepts exactly the
five frozen Gold policy archives below, verifies their content hashes and
policy identities, removes every train episode that occurs in any source
validation split, and publishes a deterministic no-clobber ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import orjson


SCHEMA_VERSION = "ptcg-balanced-gold-special-archive-v1"
SOURCE_SCHEMA_VERSION = "ptcg-gold-policy-visible-decisions-v1"
ROW_SCHEMA_VERSION = "ptcg-bc-visible-decisions-v1"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_COMPRESSION = zipfile.ZIP_DEFLATED
ZIP_COMPRESSLEVEL = 6
TRAIN_BATCHES = 20
ROWS_PER_BATCH = 256
ORDINARY_PER_POLICY_PER_BATCH = 51
CONTEXT34_PER_BATCH = 1
CONTEXT34 = "34"
DECK_HASH = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    REPO_ROOT / "data/gold_league/top21_exact_20260727/policies"
)


@dataclass(frozen=True)
class FrozenPolicy:
    policy_id: str
    filename: str
    rank: int
    team_name: str
    submission_id: int


POLICIES = (
    FrozenPolicy("rank02_dominic", "rank02_dominic.zip", 2, "Dominic Peel", 55001357),
    FrozenPolicy("rank04_liam", "rank04_liam.zip", 4, "LiamK", 55011514),
    FrozenPolicy("rank11_luca", "rank11_luca.zip", 11, "Luca", 55002097),
    FrozenPolicy("rank12_taichicchi", "rank12_taichicchi.zip", 12, "__Taichicchi__", 55003706),
    FrozenPolicy("rank19_szlachetny", "rank19_szlachetny.zip", 19, "szlachetny snieg", 54973765),
)


@dataclass
class Source:
    policy: FrozenPolicy
    path: Path
    display_path: str
    sha256: str
    manifest_sha256: str
    manifest: dict[str, Any]
    members: dict[str, list[str]]
    valid_rows: int = 0
    valid_context34_rows: int = 0
    valid_episodes: set[str] | None = None
    train_rows: int = 0
    train_episodes: set[str] | None = None
    excluded_rows: int = 0
    excluded_context34_rows: int = 0
    excluded_episodes: set[str] | None = None
    eligible_ordinary: int = 0
    eligible_context34: int = 0


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


def parse_expected_source_hashes(values: list[str]) -> dict[str, str]:
    expected_ids = {policy.policy_id for policy in POLICIES}
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(
                "--expected-source-sha must use POLICY_ID=SHA256 syntax"
            )
        policy_id, digest = raw.split("=", 1)
        if policy_id not in expected_ids:
            raise ValueError(f"unexpected policy in hash gate: {policy_id!r}")
        digest = digest.lower()
        if not SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid SHA256 for {policy_id}: {digest!r}")
        if policy_id in result:
            raise ValueError(f"duplicate hash gate for {policy_id}")
        result[policy_id] = digest
    missing = expected_ids - result.keys()
    if missing:
        raise ValueError(
            "missing --expected-source-sha gates for: " + ", ".join(sorted(missing))
        )
    return result


def split_members(archive: zipfile.ZipFile, split: str) -> list[str]:
    members = sorted(
        name
        for name in archive.namelist()
        if name.startswith(f"{split}/") and name.endswith(".jsonl")
    )
    if not members:
        raise RuntimeError(f"source has no {split} JSONL members")
    return members


def require_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer, got {value!r}")
    return value


def stable_decision_key(row: dict[str, Any]) -> str:
    dataset_date = row.get("dataset_date")
    episode_id = row.get("episode_id")
    if not isinstance(dataset_date, str) or not dataset_date or "|" in dataset_date:
        raise RuntimeError(f"invalid dataset_date: {dataset_date!r}")
    if not isinstance(episode_id, (str, int)) or not str(episode_id) or "|" in str(episode_id):
        raise RuntimeError(f"invalid episode_id: {episode_id!r}")
    seat = require_integer(row.get("seat"), "seat")
    action_step = require_integer(row.get("action_step_index"), "action_step_index")
    if seat < 0 or action_step < 0:
        raise RuntimeError("seat and action_step_index must be non-negative")
    return f"{dataset_date}|{episode_id}|{seat}|{action_step}"


def validate_row(
    row: Any,
    source: Source,
    split: str,
    member: str,
    line_number: int,
) -> tuple[str, str, bool]:
    where = f"{source.display_path}:{member}:{line_number}"
    if not isinstance(row, dict):
        raise RuntimeError(f"{where}: row is not an object")
    expected = source.policy
    checks = {
        "schema_version": (row.get("schema_version"), ROW_SCHEMA_VERSION),
        "split": (row.get("split"), split),
        "policy_id": (row.get("policy_id"), expected.policy_id),
        "team_name": (row.get("team_name"), expected.team_name),
        "deck_hash": (row.get("deck_hash"), DECK_HASH),
        "source_submission_id": (
            row.get("source_submission_id"),
            expected.submission_id,
        ),
    }
    mismatches = [
        f"{key}={actual!r} expected {wanted!r}"
        for key, (actual, wanted) in checks.items()
        if actual != wanted
    ]
    if mismatches:
        raise RuntimeError(f"{where}: identity mismatch: {'; '.join(mismatches)}")
    if "visualize" in row or (
        isinstance(row.get("observation"), dict)
        and "visualize" in row["observation"]
    ):
        raise RuntimeError(f"{where}: hidden visualize payload is forbidden")
    key = stable_decision_key(row)
    episode_id = str(row["episode_id"])
    return key, episode_id, str(row.get("select_context", "")) == CONTEXT34


def iter_rows(
    source: Source,
    split: str,
) -> Iterable[tuple[dict[str, Any], str, str, bool]]:
    with zipfile.ZipFile(source.path) as archive:
        for member in source.members[split]:
            with archive.open(member) as handle:
                for line_number, raw_line in enumerate(handle, 1):
                    try:
                        row = orjson.loads(raw_line)
                    except orjson.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"{source.display_path}:{member}:{line_number}: invalid JSON"
                        ) from exc
                    key, episode_id, is_context34 = validate_row(
                        row, source, split, member, line_number
                    )
                    yield row, key, episode_id, is_context34


def validate_source_manifest(
    policy: FrozenPolicy,
    path: Path,
    expected_sha: str,
) -> Source:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing or symlinked source archive: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"source hash mismatch for {policy.policy_id}: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"ZIP integrity failure: {path}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate ZIP member names: {path}")
        if "manifest.json" not in names:
            raise RuntimeError(f"missing manifest.json: {path}")
        manifest_payload = archive.read("manifest.json")
        manifest = orjson.loads(manifest_payload)
        members = {
            split: split_members(archive, split) for split in ("train", "valid")
        }
    if not isinstance(manifest, dict):
        raise RuntimeError(f"source manifest is not an object: {path}")
    identity = manifest.get("policy")
    if not isinstance(identity, dict):
        raise RuntimeError(f"source manifest has no policy object: {path}")
    expected_identity = {
        "policy_id": policy.policy_id,
        "rank": policy.rank,
        "team_name": policy.team_name,
        "submission_id": policy.submission_id,
        "deck_hash": DECK_HASH,
    }
    observed_identity = {
        key: identity.get(key) for key in expected_identity
    }
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise RuntimeError(
            f"source schema mismatch for {policy.policy_id}: "
            f"{manifest.get('schema_version')!r}"
        )
    if observed_identity != expected_identity:
        raise RuntimeError(
            f"source policy mismatch for {policy.policy_id}: "
            f"{observed_identity!r} != {expected_identity!r}"
        )
    return Source(
        policy=policy,
        path=path,
        display_path=display_path(path),
        sha256=actual_sha,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        manifest=manifest,
        members=members,
        valid_episodes=set(),
        train_episodes=set(),
        excluded_episodes=set(),
    )


def verify_source_counts(source: Source) -> None:
    manifest_counts = source.manifest.get("split_decisions")
    manifest_episodes = source.manifest.get("episode_ids")
    if not isinstance(manifest_counts, dict) or not isinstance(manifest_episodes, dict):
        raise RuntimeError(f"source split metadata is missing: {source.display_path}")
    observed = {"train": source.train_rows, "valid": source.valid_rows}
    for split in ("train", "valid"):
        if manifest_counts.get(split) != observed[split]:
            raise RuntimeError(
                f"{source.policy.policy_id} {split} row count mismatch: "
                f"manifest={manifest_counts.get(split)!r}, observed={observed[split]}"
            )
        expected_episodes = {str(value) for value in manifest_episodes.get(split, [])}
        actual_episodes = (
            source.train_episodes if split == "train" else source.valid_episodes
        )
        assert actual_episodes is not None
        if expected_episodes != actual_episodes:
            raise RuntimeError(
                f"{source.policy.policy_id} {split} episode set mismatch"
            )
    assert source.train_episodes is not None and source.valid_episodes is not None
    if source.train_episodes & source.valid_episodes:
        raise RuntimeError(
            f"source split episode overlap: {source.policy.policy_id}"
        )


def selection_score(seed: int, policy_id: str, key: str) -> str:
    payload = (
        str(seed).encode("utf-8")
        + b"\0"
        + policy_id.encode("utf-8")
        + b"\0"
        + key.encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def normalized_row_bytes(row: dict[str, Any], split: str) -> bytes:
    output = dict(row)
    output["split"] = split
    output["sample_weight"] = 1.0
    return orjson.dumps(output, option=orjson.OPT_SORT_KEYS) + b"\n"


def write_member(
    archive: zipfile.ZipFile,
    member: str,
    rows: Iterable[bytes],
    content_digest: hashlib._Hash,
) -> tuple[str, int]:
    member_digest = hashlib.sha256()
    row_count = 0
    content_digest.update(member.encode("utf-8"))
    content_digest.update(b"\0")
    with archive.open(zip_info(member), mode="w", force_zip64=True) as handle:
        for payload in rows:
            handle.write(payload)
            member_digest.update(payload)
            content_digest.update(payload)
            row_count += 1
    return member_digest.hexdigest(), row_count


def build_archive(
    source_dir: Path,
    output: Path,
    seed: int,
    expected_hashes: dict[str, str],
    expected_output_sha: str | None,
) -> dict[str, Any]:
    if seed < 0:
        raise ValueError("--seed must be non-negative")
    output = output.resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    sources = [
        validate_source_manifest(
            policy,
            source_dir.resolve() / policy.filename,
            expected_hashes[policy.policy_id],
        )
        for policy in POLICIES
    ]

    global_valid_episodes: set[str] = set()
    global_valid_episode_keys: set[str] = set()
    valid_keys: set[str] = set()
    for source in sources:
        assert source.valid_episodes is not None
        local_keys: set[str] = set()
        for _row, key, episode_id, is_context34 in iter_rows(source, "valid"):
            if key in local_keys or key in valid_keys:
                raise RuntimeError(f"duplicate valid decision key: {key}")
            local_keys.add(key)
            valid_keys.add(key)
            source.valid_rows += 1
            source.valid_context34_rows += int(is_context34)
            source.valid_episodes.add(episode_id)
            key_parts = key.split("|")
            global_valid_episode_keys.add(f"{key_parts[0]}|{key_parts[1]}")
        global_valid_episodes.update(source.valid_episodes)

    ranked: dict[str, dict[str, list[tuple[str, str]]]] = {
        policy.policy_id: {"ordinary": [], "context34": []}
        for policy in POLICIES
    }
    all_source_train_keys: set[str] = set()
    for source in sources:
        assert source.train_episodes is not None
        assert source.excluded_episodes is not None
        local_keys: set[str] = set()
        for _row, key, episode_id, is_context34 in iter_rows(source, "train"):
            if key in local_keys or key in all_source_train_keys:
                raise RuntimeError(f"duplicate source train decision key: {key}")
            local_keys.add(key)
            all_source_train_keys.add(key)
            source.train_rows += 1
            source.train_episodes.add(episode_id)
            if episode_id in global_valid_episodes:
                source.excluded_rows += 1
                source.excluded_context34_rows += int(is_context34)
                source.excluded_episodes.add(episode_id)
                continue
            category = "context34" if is_context34 else "ordinary"
            ranked[source.policy.policy_id][category].append(
                (selection_score(seed, source.policy.policy_id, key), key)
            )
            if is_context34:
                source.eligible_context34 += 1
            else:
                source.eligible_ordinary += 1
        verify_source_counts(source)

    ordinary_needed = TRAIN_BATCHES * ORDINARY_PER_POLICY_PER_BATCH
    context34_needed = TRAIN_BATCHES // len(POLICIES)
    if TRAIN_BATCHES % len(POLICIES):
        raise AssertionError("train batch count must be divisible by policy count")
    selected_order: dict[str, dict[str, list[str]]] = {}
    selected_keys: set[str] = set()
    for policy in POLICIES:
        by_category = ranked[policy.policy_id]
        for values in by_category.values():
            values.sort()
        if len(by_category["ordinary"]) < ordinary_needed:
            raise RuntimeError(
                f"{policy.policy_id} has only {len(by_category['ordinary'])} "
                f"eligible ordinary rows; need {ordinary_needed}"
            )
        if len(by_category["context34"]) < context34_needed:
            raise RuntimeError(
                f"{policy.policy_id} has only {len(by_category['context34'])} "
                f"eligible context34 rows; need {context34_needed}"
            )
        selected_order[policy.policy_id] = {
            "ordinary": [key for _score, key in by_category["ordinary"][:ordinary_needed]],
            "context34": [key for _score, key in by_category["context34"][:context34_needed]],
        }
        for category in ("ordinary", "context34"):
            for key in selected_order[policy.policy_id][category]:
                if key in selected_keys:
                    raise RuntimeError(f"global selection reused decision key: {key}")
                selected_keys.add(key)

    expected_selected_rows = TRAIN_BATCHES * ROWS_PER_BATCH
    if len(selected_keys) != expected_selected_rows:
        raise AssertionError(
            f"selected {len(selected_keys)} decisions, expected {expected_selected_rows}"
        )
    if any(key.split("|", 2)[1] in global_valid_episodes for key in selected_keys):
        raise RuntimeError("selected train decision belongs to global valid episode union")

    selected_payloads: dict[str, bytes] = {}
    selected_policy: dict[str, str] = {}
    for source in sources:
        wanted = set(selected_order[source.policy.policy_id]["ordinary"])
        wanted.update(selected_order[source.policy.policy_id]["context34"])
        for row, key, episode_id, _is_context34 in iter_rows(source, "train"):
            if episode_id in global_valid_episodes or key not in wanted:
                continue
            if key in selected_payloads:
                raise RuntimeError(f"selected payload repeated: {key}")
            selected_payloads[key] = normalized_row_bytes(row, "train")
            selected_policy[key] = source.policy.policy_id
    if selected_payloads.keys() != selected_keys:
        missing = sorted(selected_keys - selected_payloads.keys())
        raise RuntimeError(f"failed to materialize selected decisions: {missing[:3]}")

    train_members_payloads: list[tuple[str, list[tuple[str, bytes]]]] = []
    for batch_index in range(TRAIN_BATCHES):
        rotating_policy_index = batch_index % len(POLICIES)
        context_index = batch_index // len(POLICIES)
        items: list[tuple[str, bytes]] = []
        for policy_index, policy in enumerate(POLICIES):
            start = batch_index * ORDINARY_PER_POLICY_PER_BATCH
            stop = start + ORDINARY_PER_POLICY_PER_BATCH
            for key in selected_order[policy.policy_id]["ordinary"][start:stop]:
                items.append((key, selected_payloads[key]))
            if policy_index == rotating_policy_index:
                key = selected_order[policy.policy_id]["context34"][context_index]
                items.append((key, selected_payloads[key]))
        if len(items) != ROWS_PER_BATCH:
            raise AssertionError(f"batch {batch_index} has {len(items)} rows")
        train_members_payloads.append(
            (f"train/part-{batch_index:05d}.jsonl", items)
        )

    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".partial",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        content_digest = hashlib.sha256()
        train_member_stats: list[dict[str, Any]] = []
        valid_member_stats: list[dict[str, Any]] = []
        train_episodes: set[str] = set()
        train_episode_keys: set[str] = set()
        train_dates: set[str] = set()
        train_team_counts: Counter[str] = Counter()
        train_context_counts: Counter[str] = Counter()
        valid_team_counts: Counter[str] = Counter()
        valid_context_counts: Counter[str] = Counter()

        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=ZIP_COMPRESSION,
            compresslevel=ZIP_COMPRESSLEVEL,
            allowZip64=True,
        ) as archive:
            for batch_index, (member, items) in enumerate(train_members_payloads):
                member_team_counts = Counter(selected_policy[key] for key, _payload in items)
                member_context_counts: Counter[str] = Counter()
                member_episodes: set[str] = set()
                for key, _payload in items:
                    policy_id = selected_policy[key]
                    if key in selected_order[policy_id]["context34"]:
                        member_context_counts[policy_id] += 1
                    parts = key.split("|")
                    train_dates.add(parts[0])
                    train_episodes.add(parts[1])
                    train_episode_keys.add(f"{parts[0]}|{parts[1]}")
                    member_episodes.add(parts[1])
                member_sha, rows = write_member(
                    archive,
                    member,
                    (payload for _key, payload in items),
                    content_digest,
                )
                train_team_counts.update(member_team_counts)
                train_context_counts.update(member_context_counts)
                train_member_stats.append(
                    {
                        "member": member,
                        "sha256": member_sha,
                        "rows": rows,
                        "context34_rows": sum(member_context_counts.values()),
                        "team_counts": {
                            policy.policy_id: member_team_counts[policy.policy_id]
                            for policy in POLICIES
                        },
                        "context34_team_counts": {
                            policy.policy_id: member_context_counts[policy.policy_id]
                            for policy in POLICIES
                        },
                        "episodes": len(member_episodes),
                        "context34_policy_id": POLICIES[
                            batch_index % len(POLICIES)
                        ].policy_id,
                    }
                )

            written_valid_keys: set[str] = set()
            for source_index, source in enumerate(sources):
                member = f"valid/part-{source_index:05d}.jsonl"
                member_keys: list[str] = []
                member_episodes: set[str] = set()
                member_context_count = 0

                def valid_payloads() -> Iterable[bytes]:
                    nonlocal member_context_count
                    for row, key, episode_id, is_context34 in iter_rows(source, "valid"):
                        if key in written_valid_keys:
                            raise RuntimeError(f"valid decision key reused: {key}")
                        written_valid_keys.add(key)
                        member_keys.append(key)
                        member_episodes.add(episode_id)
                        member_context_count += int(is_context34)
                        yield normalized_row_bytes(row, "valid")

                member_sha, rows = write_member(
                    archive, member, valid_payloads(), content_digest
                )
                valid_team_counts[source.policy.policy_id] += rows
                valid_context_counts[source.policy.policy_id] += member_context_count
                valid_member_stats.append(
                    {
                        "member": member,
                        "sha256": member_sha,
                        "rows": rows,
                        "context34_rows": member_context_count,
                        "team_counts": {
                            policy.policy_id: (
                                rows if policy.policy_id == source.policy.policy_id else 0
                            )
                            for policy in POLICIES
                        },
                        "context34_team_counts": {
                            policy.policy_id: (
                                member_context_count
                                if policy.policy_id == source.policy.policy_id
                                else 0
                            )
                            for policy in POLICIES
                        },
                        "episodes": len(member_episodes),
                        "policy_id": source.policy.policy_id,
                        "decision_keys_digest": digest_lines(member_keys),
                    }
                )
            if written_valid_keys != valid_keys:
                raise RuntimeError("written valid decision keys do not match source scan")
            if train_episodes & global_valid_episodes:
                raise RuntimeError("output train/valid episode overlap is non-zero")

            source_records: list[dict[str, Any]] = []
            for source in sources:
                assert source.valid_episodes is not None
                assert source.train_episodes is not None
                assert source.excluded_episodes is not None
                selected_ordinary = len(
                    selected_order[source.policy.policy_id]["ordinary"]
                )
                selected_context34 = len(
                    selected_order[source.policy.policy_id]["context34"]
                )
                source_records.append(
                    {
                        "path": source.display_path,
                        "sha256": source.sha256,
                        "manifest_sha256": source.manifest_sha256,
                        "schema_version": source.manifest["schema_version"],
                        "policy_id": source.policy.policy_id,
                        "team_name": source.policy.team_name,
                        "source_submission_id": source.policy.submission_id,
                        "deck_hash": DECK_HASH,
                        "policy": {
                            "policy_id": source.policy.policy_id,
                            "rank": source.policy.rank,
                            "team_name": source.policy.team_name,
                            "submission_id": source.policy.submission_id,
                            "deck_hash": DECK_HASH,
                        },
                        "source_counts": {
                            "train_rows": source.train_rows,
                            "train_episodes": len(source.train_episodes),
                            "valid_rows": source.valid_rows,
                            "valid_episodes": len(source.valid_episodes),
                            "valid_context34_rows": source.valid_context34_rows,
                        },
                        "global_valid_exclusion": {
                            "rows": source.excluded_rows,
                            "ordinary_rows": (
                                source.excluded_rows
                                - source.excluded_context34_rows
                            ),
                            "context34_rows": source.excluded_context34_rows,
                            "episodes": len(source.excluded_episodes),
                        },
                        "eligible_train": {
                            "ordinary_rows": source.eligible_ordinary,
                            "context34_rows": source.eligible_context34,
                        },
                        "selected_train": {
                            "ordinary_rows": selected_ordinary,
                            "context34_rows": selected_context34,
                            "rows": selected_ordinary + selected_context34,
                        },
                    }
                )
            sources_sha256 = hashlib.sha256(
                canonical_json_bytes(source_records)
            ).hexdigest()
            all_output_keys = selected_keys | valid_keys
            if len(all_output_keys) != len(selected_keys) + len(valid_keys):
                raise RuntimeError("train and valid decision keys overlap")
            train_digest = digest_lines(selected_keys)
            valid_digest = digest_lines(valid_keys)
            all_digest = digest_lines(all_output_keys)
            manifest: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "seed": seed,
                "policies": [policy.policy_id for policy in POLICIES],
                "sources": source_records,
                "sources_sha256": sources_sha256,
                "selection": {
                    "algorithm": "sha256_lowest_without_replacement",
                    "hash": "sha256",
                    "hash_input": (
                        "utf8(seed) + NUL + utf8(policy_id) + NUL + "
                        "utf8(stable_decision_key)"
                    ),
                    "stable_decision_key_format": (
                        "dataset_date|episode_id|seat|action_step_index"
                    ),
                    "train_batches": TRAIN_BATCHES,
                    "rows_per_batch": ROWS_PER_BATCH,
                    "ordinary_rows_per_policy_per_batch": (
                        ORDINARY_PER_POLICY_PER_BATCH
                    ),
                    "context34_rows_per_batch": CONTEXT34_PER_BATCH,
                    "context34_rotation": [
                        policy.policy_id for policy in POLICIES
                    ],
                    "sample_weight": 1.0,
                },
                "train": {
                    "members": train_member_stats,
                    "member_count": len(train_member_stats),
                    "total_rows": len(selected_keys),
                    "context34_rows": sum(train_context_counts.values()),
                    "total_context34_rows": sum(train_context_counts.values()),
                    "team_counts": {
                        policy.policy_id: train_team_counts[policy.policy_id]
                        for policy in POLICIES
                    },
                    "total_team_counts": {
                        policy.policy_id: train_team_counts[policy.policy_id]
                        for policy in POLICIES
                    },
                    "context34_team_counts": {
                        policy.policy_id: train_context_counts[policy.policy_id]
                        for policy in POLICIES
                    },
                    "total_context34_team_counts": {
                        policy.policy_id: train_context_counts[policy.policy_id]
                        for policy in POLICIES
                    },
                    "episodes": len(train_episodes),
                    "dataset_dates": sorted(train_dates),
                },
                "valid": {
                    "members": valid_member_stats,
                    "member_count": len(valid_member_stats),
                    "total_rows": len(valid_keys),
                    "context34_rows": sum(valid_context_counts.values()),
                    "total_context34_rows": sum(valid_context_counts.values()),
                    "team_counts": {
                        policy.policy_id: valid_team_counts[policy.policy_id]
                        for policy in POLICIES
                    },
                    "total_team_counts": {
                        policy.policy_id: valid_team_counts[policy.policy_id]
                        for policy in POLICIES
                    },
                    "context34_team_counts": {
                        policy.policy_id: valid_context_counts[policy.policy_id]
                        for policy in POLICIES
                    },
                    "total_context34_team_counts": {
                        policy.policy_id: valid_context_counts[policy.policy_id]
                        for policy in POLICIES
                    },
                    "episodes": len(global_valid_episodes),
                },
                "episode_isolation": {
                    "global_valid_episode_count": len(global_valid_episodes),
                    "global_valid_episode_digest": digest_lines(global_valid_episodes),
                    "selected_train_episode_count": len(train_episodes),
                    "train_valid_overlap_count": len(
                        train_episodes & global_valid_episodes
                    ),
                },
                "split_integrity": {
                    "episode_key_format": "dataset_date|episode_id",
                    "train_episode_count": len(train_episode_keys),
                    "valid_episode_count": len(global_valid_episode_keys),
                    "episode_overlap_count": len(
                        train_episode_keys & global_valid_episode_keys
                    ),
                    "global_valid_episode_digest": digest_lines(
                        global_valid_episode_keys
                    ),
                },
                "exclusion": {
                    "rule": (
                        "exclude a source train row when its episode_id occurs "
                        "in the union of all five source valid partitions"
                    ),
                    "global_valid_episode_union_count": len(
                        global_valid_episodes
                    ),
                    "excluded_train_rows": sum(
                        source.excluded_rows for source in sources
                    ),
                    "excluded_train_context34_rows": sum(
                        source.excluded_context34_rows for source in sources
                    ),
                    "by_policy": {
                        source.policy.policy_id: {
                            "rows": source.excluded_rows,
                            "ordinary_rows": (
                                source.excluded_rows
                                - source.excluded_context34_rows
                            ),
                            "context34_rows": source.excluded_context34_rows,
                            "episodes": len(source.excluded_episodes or set()),
                        }
                        for source in sources
                    },
                },
                "decision_keys": {
                    "scope": "train selected decisions across all policies",
                    "format": "dataset_date|episode_id|seat|action_step_index",
                    "digest_algorithm": "sha256(sorted(key + LF))",
                    "digest": train_digest,
                    "count": len(selected_keys),
                    "duplicate_count": 0,
                    "without_replacement": True,
                    "train": {
                        "digest": train_digest,
                        "count": len(selected_keys),
                        "duplicate_count": 0,
                    },
                    "valid": {
                        "digest": valid_digest,
                        "count": len(valid_keys),
                        "duplicate_count": 0,
                    },
                    "global": {
                        "digest": all_digest,
                        "count": len(all_output_keys),
                        "duplicate_count": 0,
                    },
                },
                "content_sha256": content_digest.hexdigest(),
                "zip": {
                    "timestamp": list(ZIP_TIMESTAMP),
                    "compression": "deflate",
                    "compresslevel": ZIP_COMPRESSLEVEL,
                    "member_mode": "0644",
                    "content_digest_input": (
                        "lexicographic members: utf8(member_name) + NUL + "
                        "raw_member_bytes; manifest.json excluded"
                    ),
                },
            }
            manifest_payload = canonical_json_bytes(manifest) + b"\n"
            archive.writestr(zip_info("manifest.json"), manifest_payload)

        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        archive_sha256 = sha256_file(temporary)
        if expected_output_sha is not None and archive_sha256 != expected_output_sha:
            raise RuntimeError(
                f"output hash mismatch: expected {expected_output_sha}, "
                f"got {archive_sha256}"
            )
        if os.path.lexists(output):
            raise FileExistsError(
                f"output appeared during build; refusing to overwrite: {output}"
            )
        os.replace(temporary, output)
        temporary = None
        return {
            "status": "ok",
            "output": str(output),
            "archive_sha256": archive_sha256,
            "manifest_sha256": manifest_sha256,
            "content_sha256": manifest["content_sha256"],
            "sources_sha256": manifest["sources_sha256"],
            "decision_keys_digest": manifest["decision_keys"]["digest"],
            "decision_key_count": manifest["decision_keys"]["count"],
            "train_rows": manifest["train"]["total_rows"],
            "valid_rows": manifest["valid"]["total_rows"],
            "train_valid_episode_overlap": manifest["episode_isolation"][
                "train_valid_overlap_count"
            ],
        }
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="directory containing the five frozen policy ZIPs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--expected-source-sha",
        action="append",
        default=[],
        metavar="POLICY_ID=SHA256",
        help="required once for every frozen source policy",
    )
    parser.add_argument(
        "--expected-output-sha",
        type=str,
        default=None,
        help="optional deterministic archive hash gate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_hashes = parse_expected_source_hashes(args.expected_source_sha)
    expected_output_sha = args.expected_output_sha
    if expected_output_sha is not None:
        expected_output_sha = expected_output_sha.lower()
        if not SHA256_RE.fullmatch(expected_output_sha):
            raise ValueError("--expected-output-sha must be a lowercase SHA256")
    result = build_archive(
        args.source_dir,
        args.output,
        args.seed,
        expected_hashes,
        expected_output_sha,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
