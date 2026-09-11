#!/usr/bin/env python3
"""Stream-audit a read-only, time-forward PTCG BC ZIP archive.

The auditor never extracts or rewrites the input archive.  It reads every
JSONL member once, checks the manifest against a caller-frozen date layout and
team allowlist, and reports row-level defects and cross-split episode leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import orjson


SPLITS = ("train", "valid", "test")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_SCHEMA = "ptcg-bc-visible-decisions-v1"
EXPECTED_COMPETITION = "pokemon-tcg-ai-battle"
EXPECTED_ALIGNMENT = "steps[t-1].observation -> steps[t].action"
SOURCE_KINDS = {"zip", "directory", "json"}
TRAINER_MAX_ACTION_COUNT = 16


def normalize_team_name(value: Any) -> str:
    """Match the normalization used by ``prepare_bc_week.py``."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def valid_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ExpectedLayout:
    dates_by_split: dict[str, tuple[str, ...]]

    @property
    def dates(self) -> tuple[str, ...]:
        return tuple(
            item
            for split in SPLITS
            for item in self.dates_by_split[split]
        )

    def public(self) -> dict[str, Any]:
        return {
            "dates": list(self.dates),
            "dates_by_split": {
                split: list(self.dates_by_split[split])
                for split in SPLITS
            },
        }


@dataclass(frozen=True)
class TeamAllowlist:
    global_names: frozenset[str]
    names_by_date: dict[str, frozenset[str]]
    display_names: tuple[str, ...]

    def allows(self, dataset_date: str, team_name: Any) -> bool:
        normalized = normalize_team_name(team_name)
        if not normalized:
            return False
        # This intentionally mirrors prepare_bc_week.TeamFilter.allows:
        # a dated CSV is interpreted as a date-indexed allowlist.
        if self.names_by_date:
            return normalized in self.names_by_date.get(
                dataset_date, frozenset()
            )
        return normalized in self.global_names

    @property
    def all_names(self) -> frozenset[str]:
        result = set(self.global_names)
        for names in self.names_by_date.values():
            result.update(names)
        return frozenset(result)


@dataclass
class IssueLog:
    max_samples: int
    counts: Counter[str] = field(default_factory=Counter)
    samples: list[dict[str, str]] = field(default_factory=list)

    def add(self, code: str, message: str) -> None:
        self.counts[code] += 1
        if len(self.samples) < self.max_samples:
            self.samples.append({"code": code, "message": message})

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def public(self) -> dict[str, Any]:
        return {
            "count": self.total,
            "counts_by_code": dict(sorted(self.counts.items())),
            "samples": self.samples,
            "samples_truncated": self.total > len(self.samples),
        }


@dataclass
class SplitStats:
    rows_scanned: int = 0
    valid_rows: int = 0
    trainable_rows: int = 0
    invalid_rows: int = 0
    episode_ids: set[str] = field(default_factory=set)
    episode_uuids: set[str] = field(default_factory=set)
    dataset_dates: set[str] = field(default_factory=set)
    team_rows: Counter[str] = field(default_factory=Counter)
    context_rows: Counter[str] = field(default_factory=Counter)

    def public(self, shards: int) -> dict[str, Any]:
        return {
            "rows_scanned": self.rows_scanned,
            "valid_rows": self.valid_rows,
            "trainable_rows": self.trainable_rows,
            "invalid_rows": self.invalid_rows,
            "unique_episode_ids": len(self.episode_ids),
            "unique_episode_uuids": len(self.episode_uuids),
            "dataset_dates": sorted(self.dataset_dates),
            "shards": shards,
            "team_rows": dict(
                sorted(
                    self.team_rows.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "context_rows": dict(
                sorted(
                    self.context_rows.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        }


class AuditIdentityIndex:
    """Exact disk-backed decision and episode-identity index."""

    def __init__(self) -> None:
        # An empty SQLite filename creates a temporary on-disk database that
        # SQLite removes when this connection closes.
        self.connection = sqlite3.connect("")
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript(
            """
            CREATE TABLE decisions (
                split TEXT NOT NULL,
                dataset_date TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                seat INTEGER NOT NULL,
                action_step_index INTEGER NOT NULL,
                PRIMARY KEY (
                    split,
                    dataset_date,
                    episode_id,
                    seat,
                    action_step_index
                )
            ) WITHOUT ROWID;
            CREATE TABLE episode_identities (
                dataset_date TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                episode_uuid TEXT,
                PRIMARY KEY (dataset_date, episode_id)
            ) WITHOUT ROWID;
            CREATE UNIQUE INDEX unique_nonempty_episode_uuid
            ON episode_identities(episode_uuid)
            WHERE episode_uuid IS NOT NULL;
            """
        )

    def check(
        self,
        row: dict[str, Any],
        split: str,
    ) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        dataset_date = row.get("dataset_date")
        episode_id = row.get("episode_id")
        seat = row.get("seat")
        action_step = row.get("action_step_index")
        if (
            not isinstance(dataset_date, str)
            or not dataset_date
            or not _valid_episode_id(episode_id)
            or not _integer_not_bool(seat)
            or not _integer_not_bool(action_step)
        ):
            return result

        key = (
            split,
            dataset_date,
            str(episode_id),
            seat,
            action_step,
        )
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO decisions(
                split, dataset_date, episode_id, seat, action_step_index
            ) VALUES (?, ?, ?, ?, ?)
            """,
            key,
        )
        if cursor.rowcount == 0:
            result.append(
                (
                    "row_duplicate_decision_key",
                    "duplicate (dataset_date, episode_id, seat, "
                    f"action_step_index) within {split}",
                )
            )

        raw_uuid = row.get("episode_uuid")
        episode_uuid = (
            raw_uuid
            if isinstance(raw_uuid, str) and raw_uuid.strip()
            else None
        )
        identity = (dataset_date, str(episode_id))
        existing = self.connection.execute(
            """
            SELECT episode_uuid
            FROM episode_identities
            WHERE dataset_date = ? AND episode_id = ?
            """,
            identity,
        ).fetchone()
        if existing is not None:
            if existing[0] != episode_uuid:
                result.append(
                    (
                        "row_episode_identity_inconsistent",
                        "(dataset_date, episode_id) maps to multiple "
                        "episode_uuid values",
                    )
                )
            return result

        if episode_uuid is not None:
            uuid_owner = self.connection.execute(
                """
                SELECT dataset_date, episode_id
                FROM episode_identities
                WHERE episode_uuid = ?
                """,
                (episode_uuid,),
            ).fetchone()
            if uuid_owner is not None and uuid_owner != identity:
                result.append(
                    (
                        "row_episode_identity_inconsistent",
                        "episode_uuid maps to multiple "
                        "(dataset_date, episode_id) values",
                    )
                )
                return result
        self.connection.execute(
            """
            INSERT INTO episode_identities(
                dataset_date, episode_id, episode_uuid
            ) VALUES (?, ?, ?)
            """,
            (*identity, episode_uuid),
        )
        return result

    def close(self) -> None:
        self.connection.rollback()
        self.connection.close()


def parse_team_file(path: Path) -> TeamAllowlist:
    path = path.expanduser().resolve(strict=True)
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"team allowlist is empty: {path}")

    global_names: set[str] = set()
    names_by_date: dict[str, set[str]] = defaultdict(set)
    display_names: list[str] = []

    if "," not in lines[0]:
        for line in lines:
            display = line.strip()
            if not display:
                continue
            normalized = normalize_team_name(display)
            if normalized:
                global_names.add(normalized)
                display_names.append(display)
    else:
        reader = csv.DictReader(lines)
        fields = {
            field.casefold(): field for field in (reader.fieldnames or [])
        }
        team_field = next(
            (
                fields[key]
                for key in ("team_name", "teamname", "team", "name")
                if key in fields
            ),
            None,
        )
        date_field = fields.get("date")
        if team_field is None:
            raise ValueError(
                f"{path} must contain team_name/TeamName/team/name"
            )
        for row_number, row in enumerate(reader, 2):
            display = str(row.get(team_field, "")).strip()
            if not display:
                continue
            normalized = normalize_team_name(display)
            display_names.append(display)
            raw_date = (
                str(row.get(date_field, "")).strip() if date_field else ""
            )
            if raw_date:
                if not valid_iso_date(raw_date):
                    raise ValueError(
                        f"{path}:{row_number}: invalid date {raw_date!r}"
                    )
                names_by_date[raw_date].add(normalized)
            else:
                global_names.add(normalized)

    if not global_names and not names_by_date:
        raise ValueError(f"team allowlist has no team names: {path}")
    return TeamAllowlist(
        global_names=frozenset(global_names),
        names_by_date={
            key: frozenset(value)
            for key, value in sorted(names_by_date.items())
        },
        display_names=tuple(display_names),
    )


def build_expected_layout(
    *,
    expected_dates: list[str] | None = None,
    train_dates: list[str] | None = None,
    valid_dates: list[str] | None = None,
    test_dates: list[str] | None = None,
    expect_empty_test: bool = False,
) -> ExpectedLayout:
    explicit = (train_dates, valid_dates, test_dates)
    if expected_dates is not None:
        if expect_empty_test or any(value is not None for value in explicit):
            raise ValueError(
                "--expected-dates cannot be combined with split-specific "
                "dates or --expect-empty-test"
            )
        if len(expected_dates) != 12:
            raise ValueError("--expected-dates must contain exactly 12 dates")
        groups = {
            "train": tuple(expected_dates[:10]),
            "valid": (expected_dates[10],),
            "test": (expected_dates[11],),
        }
    else:
        if train_dates is None or valid_dates is None:
            raise ValueError(
                "provide --expected-dates or all of "
                "--expected-train-dates/--expected-valid-dates/"
                "--expected-test-dates, or replace --expected-test-dates "
                "with --expect-empty-test"
            )
        if expect_empty_test:
            if test_dates is not None:
                raise ValueError(
                    "--expect-empty-test cannot be combined with "
                    "--expected-test-dates"
                )
            resolved_test_dates: tuple[str, ...] = ()
        else:
            if test_dates is None:
                raise ValueError(
                    "provide --expected-test-dates or --expect-empty-test"
                )
            resolved_test_dates = tuple(test_dates)
        groups = {
            "train": tuple(train_dates),
            "valid": tuple(valid_dates),
            "test": resolved_test_dates,
        }

    for split in SPLITS:
        values = groups[split]
        if not values and not (split == "test" and expect_empty_test):
            raise ValueError(f"expected {split} dates cannot be empty")
        if len(values) != len(set(values)):
            raise ValueError(f"expected {split} dates contain duplicates")
        for value in values:
            if not valid_iso_date(value):
                raise ValueError(
                    f"expected {split} dates contain invalid date {value!r}"
                )

    combined = tuple(item for split in SPLITS for item in groups[split])
    if len(combined) != len(set(combined)):
        raise ValueError("expected split date lists overlap")
    if tuple(sorted(combined)) != combined:
        raise ValueError(
            "expected dates must be chronological train, then valid, then test"
        )
    return ExpectedLayout(dates_by_split=groups)


def contains_visualize(value: Any) -> bool:
    """Return whether a ``visualize`` key occurs anywhere below ``value``."""

    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if "visualize" in current:
                return True
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return False


def _manifest_object(
    archive: zipfile.ZipFile,
    manifest_infos: list[zipfile.ZipInfo],
    issues: IssueLog,
) -> dict[str, Any]:
    if len(manifest_infos) != 1:
        issues.add(
            "manifest_member_count",
            "ZIP must contain exactly one manifest.json; "
            f"observed {len(manifest_infos)}",
        )
    if not manifest_infos:
        return {}
    try:
        with archive.open(manifest_infos[0]) as handle:
            payload = orjson.loads(handle.read())
    except (OSError, RuntimeError, zipfile.BadZipFile, orjson.JSONDecodeError) as error:
        issues.add(
            "manifest_invalid",
            f"manifest.json could not be read as JSON: {error}",
        )
        return {}
    if not isinstance(payload, dict):
        issues.add(
            "manifest_invalid",
            "manifest.json must contain a JSON object",
        )
        return {}
    return payload


def _require_exact(
    manifest: dict[str, Any],
    key: str,
    expected: Any,
    issues: IssueLog,
) -> None:
    actual = manifest.get(key)
    if actual != expected:
        issues.add(
            f"manifest_{key}_mismatch",
            f"manifest {key} is {actual!r}, expected {expected!r}",
        )


def _validate_manifest(
    manifest: dict[str, Any],
    expected: ExpectedLayout,
    teams: TeamAllowlist,
    expected_deck_hash: str | None,
    expected_episodes_scanned: int | None,
    split_stats: dict[str, SplitStats],
    shard_counts: Counter[str],
    issues: IssueLog,
) -> dict[str, Any]:
    filtered_from = manifest.get("filtered_from")
    declared_filter_hash = manifest.get("deck_hash_filter")
    has_filtered_from = isinstance(filtered_from, str) and bool(filtered_from)
    has_filter_hash = (
        isinstance(declared_filter_hash, str)
        and SHA256_RE.fullmatch(declared_filter_hash) is not None
    )
    filter_lineage = manifest.get("filter_lineage")
    if manifest.get("filtered_from") is not None and not (
        has_filtered_from and has_filter_hash
    ):
        issues.add(
            "manifest_filter_markers_invalid",
            "a filtered_from declaration requires a non-empty path and a "
            "lowercase SHA-256 deck_hash_filter",
        )
    is_filtered = has_filtered_from and has_filter_hash
    is_legacy_filtered = is_filtered and filter_lineage is None
    is_current_filtered = is_filtered and filter_lineage is not None
    if is_legacy_filtered:
        issues.add(
            "manifest_filter_lineage_missing",
            "filtered archives without filter_lineage are not promotion "
            "eligible; rebuild with the current filter_bc_archive.py",
        )
    lineage_report: dict[str, Any] = {
        "is_filtered_archive": is_filtered,
        "mode": (
            "legacy_missing_filter_lineage"
            if is_legacy_filtered
            else "materialized_filter_lineage"
            if is_current_filtered
            else "unfiltered"
        ),
        "filtered_from": filtered_from if has_filtered_from else None,
        "deck_hash_filter": (
            declared_filter_hash if has_filter_hash else None
        ),
        "stats_decisions_declaration": None,
        "shards_declaration": None,
        "team_context_declaration": None,
        "source_stats_location": (
            "filter_lineage.source_stats"
            if is_current_filtered
            else "stats"
        ),
    }
    if is_current_filtered:
        if not isinstance(filter_lineage, dict):
            issues.add(
                "manifest_filter_lineage_invalid",
                "manifest filter_lineage must be an object",
            )
        else:
            for key in (
                "source_split_decisions",
                "source_shards",
                "source_stats",
            ):
                if not isinstance(filter_lineage.get(key), dict):
                    issues.add(
                        "manifest_filter_lineage_invalid",
                        f"manifest filter_lineage.{key} must be an object",
                    )

    _require_exact(
        manifest, "schema_version", EXPECTED_SCHEMA, issues
    )
    _require_exact(
        manifest, "competition", EXPECTED_COMPETITION, issues
    )
    _require_exact(manifest, "dates", list(expected.dates), issues)
    _require_exact(manifest, "label_alignment", EXPECTED_ALIGNMENT, issues)

    hidden_policy = manifest.get("hidden_information_policy")
    if (
        not isinstance(hidden_policy, str)
        or "visualize" not in hidden_policy.casefold()
        or not any(
            word in hidden_policy.casefold()
            for word in ("exclude", "never", "not saved")
        )
    ):
        issues.add(
            "manifest_hidden_information_policy_invalid",
            "manifest hidden_information_policy must state that visualize "
            "is excluded",
        )

    policy = manifest.get("split_policy")
    if not isinstance(policy, dict):
        issues.add(
            "manifest_split_policy_invalid",
            "manifest split_policy must be an object",
        )
    else:
        if policy.get("mode") != "time":
            issues.add(
                "manifest_split_policy_invalid",
                "manifest split_policy.mode must be 'time'",
            )
        for split in SPLITS:
            actual = policy.get(f"{split}_dates")
            wanted = list(expected.dates_by_split[split])
            if actual != wanted:
                issues.add(
                    "manifest_split_dates_mismatch",
                    f"manifest split_policy.{split}_dates is {actual!r}, "
                    f"expected {wanted!r}",
                )

    sources = manifest.get("sources")
    if not isinstance(sources, list):
        issues.add(
            "manifest_sources_invalid",
            "manifest sources must be a list",
        )
    else:
        source_dates: list[str] = []
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                issues.add(
                    "manifest_source_invalid",
                    f"manifest sources[{index}] is not an object",
                )
                continue
            source_date = source.get("date")
            if not isinstance(source_date, str):
                issues.add(
                    "manifest_source_invalid",
                    f"manifest sources[{index}].date is invalid",
                )
            else:
                source_dates.append(source_date)
            if source.get("kind") not in SOURCE_KINDS:
                issues.add(
                    "manifest_source_invalid",
                    f"manifest sources[{index}].kind is invalid",
                )
            if not isinstance(source.get("path"), str) or not source["path"]:
                issues.add(
                    "manifest_source_invalid",
                    f"manifest sources[{index}].path is invalid",
                )
        if source_dates != list(expected.dates):
            issues.add(
                "manifest_source_dates_mismatch",
                f"manifest source dates are {source_dates!r}, "
                f"expected {list(expected.dates)!r}",
            )

    team_filter = manifest.get("team_filter")
    if not isinstance(team_filter, dict):
        issues.add(
            "manifest_team_filter_invalid",
            "manifest team_filter must be an object",
        )
    else:
        if team_filter.get("all_teams") is not False:
            issues.add(
                "manifest_team_filter_invalid",
                "manifest team_filter.all_teams must be false",
            )
        display = team_filter.get("display_names")
        normalized_display = (
            {
                normalize_team_name(item)
                for item in display
                if isinstance(item, str) and normalize_team_name(item)
            }
            if isinstance(display, list)
            else set()
        )
        if normalized_display != set(teams.all_names):
            issues.add(
                "manifest_team_filter_mismatch",
                "manifest team_filter.display_names does not match the "
                "frozen team allowlist",
            )
        if team_filter.get("global_team_count") != len(teams.global_names):
            issues.add(
                "manifest_team_filter_mismatch",
                "manifest team_filter.global_team_count does not match the "
                "frozen team allowlist",
            )
        expected_dated_counts = {
            key: len(value)
            for key, value in sorted(teams.names_by_date.items())
        }
        if team_filter.get("dated_team_counts") != expected_dated_counts:
            issues.add(
                "manifest_team_filter_mismatch",
                "manifest team_filter.dated_team_counts does not match the "
                "frozen team allowlist",
            )

    if expected_deck_hash is not None:
        declared_hash = declared_filter_hash
        # prepare_bc_week writes per-row hashes but does not necessarily write
        # a top-level filter.  If present, however, it must be the frozen hash.
        if declared_hash is not None and declared_hash != expected_deck_hash:
            issues.add(
                "manifest_deck_hash_mismatch",
                f"manifest deck_hash_filter is {declared_hash!r}, "
                f"expected {expected_deck_hash!r}",
            )

    declared_splits = manifest.get("split_decisions")
    if not isinstance(declared_splits, dict):
        issues.add(
            "manifest_split_decisions_invalid",
            "manifest split_decisions must be an object",
        )
    else:
        for split in SPLITS:
            observed = split_stats[split].rows_scanned
            if declared_splits.get(split) != observed:
                issues.add(
                    "manifest_split_decisions_mismatch",
                    f"manifest split_decisions.{split} is "
                    f"{declared_splits.get(split)!r}, observed {observed}",
                )

    declared_shards = manifest.get("shards")
    if not isinstance(declared_shards, dict):
        issues.add(
            "manifest_shards_invalid",
            "manifest shards must be an object",
        )
    else:
        shard_values_valid = True
        shard_values_match = True
        for split in SPLITS:
            observed = shard_counts[split]
            declared = declared_shards.get(split)
            if (
                not isinstance(declared, int)
                or isinstance(declared, bool)
                or declared < 0
            ):
                shard_values_valid = False
                issues.add(
                    "manifest_shards_invalid",
                    f"manifest shards.{split} is invalid: {declared!r}",
                )
                continue
            if declared != observed:
                shard_values_match = False
                issues.add(
                    "manifest_shards_mismatch",
                    f"manifest shards.{split} is {declared!r}, "
                    f"observed {observed}",
                )
        if shard_values_valid:
            lineage_report["shards_declaration"] = (
                "current_verified"
                if shard_values_match
                else "mismatch"
            )
            lineage_report["declared_shards"] = {
                split: declared_shards[split] for split in SPLITS
            }
            lineage_report["observed_shards"] = {
                split: shard_counts[split] for split in SPLITS
            }

    stats = manifest.get("stats")
    if not isinstance(stats, dict):
        issues.add("manifest_stats_invalid", "manifest stats must be an object")
    else:
        total_rows = sum(
            split_stats[split].rows_scanned for split in SPLITS
        )
        declared_decisions = stats.get("decisions")
        decisions_valid = (
            isinstance(declared_decisions, int)
            and not isinstance(declared_decisions, bool)
            and declared_decisions >= 0
        )
        if not decisions_valid:
            issues.add(
                "manifest_decisions_invalid",
                "manifest stats.decisions must be a non-negative integer",
            )
        elif declared_decisions == total_rows:
            lineage_report["stats_decisions_declaration"] = (
                "current_verified"
            )
        else:
            issues.add(
                "manifest_decisions_mismatch",
                f"manifest stats.decisions is {declared_decisions!r}, "
                f"observed {total_rows}",
            )
        lineage_report["declared_stats_decisions"] = declared_decisions
        lineage_report["observed_decisions"] = total_rows

        unique_by_split = {
            split: len(split_stats[split].episode_ids)
            for split in SPLITS
        }
        if "split_episodes" in manifest:
            if manifest.get("split_episodes") != unique_by_split:
                issues.add(
                    "manifest_split_episodes_mismatch",
                    f"manifest split_episodes is "
                    f"{manifest.get('split_episodes')!r}, observed "
                    f"{unique_by_split!r}",
                )
        unique_ids = len(
            set().union(
                *(split_stats[split].episode_ids for split in SPLITS)
            )
        )
        if (
            "episodes_in_output" in stats
            and stats.get("episodes_in_output") != unique_ids
        ):
            issues.add(
                "manifest_episodes_in_output_mismatch",
                f"manifest stats.episodes_in_output is "
                f"{stats.get('episodes_in_output')!r}, observed {unique_ids}",
            )

        observed_team_rows: Counter[str] = Counter()
        observed_context_rows: Counter[str] = Counter()
        for split in SPLITS:
            observed_team_rows.update(split_stats[split].team_rows)
            observed_context_rows.update(split_stats[split].context_rows)
        team_matches = manifest.get("team_decisions") == dict(
            observed_team_rows
        )
        context_matches = manifest.get("context_decisions") == dict(
            observed_context_rows
        )
        if team_matches and context_matches:
            lineage_report["team_context_declaration"] = "current_verified"
        else:
            if not team_matches:
                issues.add(
                    "manifest_team_decisions_mismatch",
                    "manifest team_decisions does not match streamed rows",
                )
            if not context_matches:
                issues.add(
                    "manifest_context_decisions_mismatch",
                    "manifest context_decisions does not match streamed rows",
                )

        source_stats = stats
        if (
            is_current_filtered
            and isinstance(filter_lineage, dict)
            and isinstance(filter_lineage.get("source_stats"), dict)
        ):
            source_stats = filter_lineage["source_stats"]
        lineage_report["authoritative_source_stats"] = {
            "episodes_scanned": source_stats.get("episodes_scanned"),
            "duplicate_episodes": source_stats.get("duplicate_episodes", 0),
            "invalid_json_files": source_stats.get("invalid_json_files", 0),
        }
        for zero_field in ("duplicate_episodes", "invalid_json_files"):
            declared_zero = source_stats.get(zero_field, 0)
            if (
                not isinstance(declared_zero, int)
                or isinstance(declared_zero, bool)
                or declared_zero != 0
            ):
                issues.add(
                    f"manifest_{zero_field}_nonzero",
                    f"authoritative source stats.{zero_field} must be zero "
                    f"(missing means zero), observed {declared_zero!r}",
                )
        scanned = source_stats.get("episodes_scanned")
        if (
            not isinstance(scanned, int)
            or isinstance(scanned, bool)
            or scanned < 0
        ):
            issues.add(
                "manifest_episodes_scanned_invalid",
                "authoritative source stats.episodes_scanned must be a "
                "non-negative integer",
            )
        else:
            if scanned < unique_ids:
                issues.add(
                    "manifest_episodes_scanned_invalid",
                    f"manifest episodes_scanned {scanned} is smaller than "
                    f"{unique_ids} kept episode IDs",
                )
            if (
                expected_episodes_scanned is not None
                and scanned != expected_episodes_scanned
            ):
                issues.add(
                    "manifest_episodes_scanned_mismatch",
                    f"manifest episodes_scanned is {scanned}, expected "
                    f"{expected_episodes_scanned}",
                )
    return lineage_report


def _member_split(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if len(parts) >= 2 and parts[0] in SPLITS:
        return parts[0]
    return None


def _valid_episode_id(value: Any) -> bool:
    return (
        isinstance(value, (str, int))
        and not isinstance(value, bool)
        and bool(str(value).strip())
    )


def _integer_not_bool(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _trainer_int_convertible(value: Any, *, default: int = 0) -> bool:
    try:
        int(value or default)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _trainer_float_convertible(value: Any) -> bool:
    try:
        number = float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number)


def _trainer_card_issues(
    card: Any,
    label: str,
) -> list[tuple[str, str]]:
    if not isinstance(card, dict):
        return []
    result: list[tuple[str, str]] = []
    for key in ("hp", "maxHp"):
        if not _trainer_float_convertible(card.get(key, 0)):
            result.append(
                (
                    "row_trainer_card_numeric_invalid",
                    f"{label}.{key} is not finite/float-convertible",
                )
            )
    for key in ("energies", "energyCards", "tools", "preEvolution"):
        value = card.get(key)
        if value is not None and not isinstance(value, list):
            result.append(
                (
                    "row_trainer_card_collection_invalid",
                    f"{label}.{key} must be a list when present",
                )
            )
    return result


def _trainer_current_issues(
    observation: dict[str, Any],
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    raw_current = observation.get("current")
    if raw_current is None:
        current: dict[str, Any] = {}
    elif not isinstance(raw_current, dict):
        return [
            (
                "row_trainer_current_invalid",
                "observation.current must be an object when present",
            )
        ]
    else:
        current = raw_current

    for key in ("yourIndex", "turn", "turnActionCount"):
        if not _trainer_int_convertible(current.get(key, 0)):
            result.append(
                (
                    "row_trainer_current_numeric_invalid",
                    f"observation.current.{key} is not int-convertible",
                )
            )
    players = current.get("players")
    if players is None:
        players = []
    if not isinstance(players, list):
        result.append(
            (
                "row_trainer_players_invalid",
                "observation.current.players must be a list when present",
            )
        )
        players = []
    for player_index, player in enumerate(players[:2]):
        if not isinstance(player, dict):
            continue
        for key in ("deckCount", "handCount", "benchMax"):
            if not _trainer_int_convertible(player.get(key, 0)):
                result.append(
                    (
                        "row_trainer_player_numeric_invalid",
                        f"current.players[{player_index}].{key} is not "
                        "int-convertible",
                    )
                )
        for zone in (
            "hand",
            "active",
            "bench",
            "discard",
            "prize",
        ):
            cards = player.get(zone)
            if cards is None:
                continue
            if not isinstance(cards, list):
                result.append(
                    (
                        "row_trainer_player_zone_invalid",
                        f"current.players[{player_index}].{zone} must be "
                        "a list",
                    )
                )
                continue
            for card_index, card in enumerate(cards):
                result.extend(
                    _trainer_card_issues(
                        card,
                        f"current.players[{player_index}]."
                        f"{zone}[{card_index}]",
                    )
                )
    for zone in ("stadium", "looking"):
        cards = current.get(zone)
        if cards is None:
            continue
        if not isinstance(cards, list):
            result.append(
                (
                    "row_trainer_current_zone_invalid",
                    f"observation.current.{zone} must be a list",
                )
            )
            continue
        for card_index, card in enumerate(cards):
            result.extend(
                _trainer_card_issues(
                    card, f"current.{zone}[{card_index}]"
                )
            )

    logs = observation.get("logs")
    if logs is not None and not isinstance(logs, list):
        result.append(
            (
                "row_trainer_logs_invalid",
                "observation.logs must be a list when present",
            )
        )
    elif isinstance(logs, list):
        for log_index, event in enumerate(logs[-12:]):
            if not isinstance(event, dict):
                continue
            for key in ("type", "playerIndex"):
                if not _trainer_int_convertible(event.get(key, -1), default=-1):
                    result.append(
                        (
                            "row_trainer_log_numeric_invalid",
                            f"observation.logs[{log_index}].{key} is not "
                            "int-convertible",
                        )
                    )
    return result


def _trainer_schema_issues(
    row: dict[str, Any],
) -> list[tuple[str, str]]:
    observation = row.get("observation")
    if not isinstance(observation, dict):
        return [
            (
                "row_trainer_observation_invalid",
                "trainer requires observation to be an object",
            )
        ]
    select = observation.get("select")
    if not isinstance(select, dict):
        return [
            (
                "row_select_invalid",
                "observation.select must be an object",
            )
        ]
    options = select.get("option")
    if not isinstance(options, list) or not options:
        return [
            (
                "row_select_invalid",
                "observation.select.option must be a non-empty list",
            )
        ]
    result: list[tuple[str, str]] = []
    for key in ("type", "context", "minCount", "maxCount"):
        if not _trainer_int_convertible(select.get(key, 0)):
            result.append(
                (
                    "row_trainer_select_numeric_invalid",
                    f"observation.select.{key} is not int-convertible",
                )
            )
    for key in ("remainDamageCounter", "remainEnergyCost"):
        if not _trainer_float_convertible(select.get(key, 0)):
            result.append(
                (
                    "row_trainer_select_numeric_invalid",
                    f"observation.select.{key} is not numeric",
                )
            )
    integer_option_fields = (
        "type",
        "area",
        "index",
        "playerIndex",
        "inPlayArea",
        "inPlayIndex",
        "toolIndex",
        "energyIndex",
    )
    for option_index, option in enumerate(options):
        if not isinstance(option, dict):
            result.append(
                (
                    "row_trainer_option_invalid",
                    f"observation.select.option[{option_index}] must be "
                    "an object",
                )
            )
            continue
        for key in integer_option_fields:
            if key not in option or option[key] is None:
                continue
            if key == "index":
                try:
                    int(option[key])
                except (TypeError, ValueError, OverflowError):
                    convertible = False
                else:
                    convertible = True
            else:
                convertible = _trainer_int_convertible(option[key])
            if not convertible:
                result.append(
                    (
                        "row_trainer_option_numeric_invalid",
                        f"option[{option_index}].{key} is not "
                        "int-convertible",
                    )
                )
        for key in ("attackId", "number", "count"):
            if key in option and not _trainer_float_convertible(option[key]):
                result.append(
                    (
                        "row_trainer_option_numeric_invalid",
                        f"option[{option_index}].{key} is not numeric",
                    )
                )

    try:
        min_count = int(select.get("minCount", 0) or 0)
        max_count = int(select.get("maxCount", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        min_count = max_count = 0
    action = row.get("action")
    indices_valid = isinstance(action, list) and all(
        _integer_not_bool(index) and 0 <= index < len(options)
        for index in action
    )
    action_valid = bool(
        indices_valid
        and isinstance(action, list)
        and min_count <= len(action) <= max_count
        and len(set(action)) == len(action)
    )
    if not action_valid:
        result.append(
            (
                "row_action_illegal",
                "action must contain unique in-range option indices and "
                "satisfy select minCount/maxCount",
            )
        )
    if isinstance(action, list) and len(action) > TRAINER_MAX_ACTION_COUNT:
        result.append(
            (
                "row_action_too_long",
                f"action length {len(action)} exceeds trainer limit "
                f"{TRAINER_MAX_ACTION_COUNT}",
            )
        )
    if (
        not _integer_not_bool(row.get("min_count"))
        or row.get("min_count") != min_count
    ):
        result.append(
            (
                "row_select_metadata_mismatch",
                f"min_count {row.get('min_count')!r} != {min_count}",
            )
        )
    if (
        not _integer_not_bool(row.get("max_count"))
        or row.get("max_count") != max_count
    ):
        result.append(
            (
                "row_select_metadata_mismatch",
                f"max_count {row.get('max_count')!r} != {max_count}",
            )
        )
    if (
        not _integer_not_bool(row.get("option_count"))
        or row.get("option_count") != len(options)
    ):
        result.append(
            (
                "row_select_metadata_mismatch",
                f"option_count {row.get('option_count')!r} != "
                f"{len(options)}",
            )
        )
    if (
        isinstance(action, list)
        and (
            not isinstance(row.get("no_action"), bool)
            or row.get("no_action") != (len(action) == 0)
        )
    ):
        result.append(
            (
                "row_select_metadata_mismatch",
                "no_action does not match whether action is empty",
            )
        )
    if row.get("select_context") != str(select.get("context", "unknown")):
        result.append(
            (
                "row_select_metadata_mismatch",
                "select_context does not match observation.select.context",
            )
        )
    if row.get("select_type") != select.get("type"):
        result.append(
            (
                "row_select_metadata_mismatch",
                "select_type does not match observation.select.type",
            )
        )
    deck_cards = select.get("deck")
    if deck_cards is not None and not isinstance(deck_cards, list):
        result.append(
            (
                "row_trainer_select_deck_invalid",
                "observation.select.deck must be a list when present",
            )
        )
    elif isinstance(deck_cards, list):
        for card_index, card in enumerate(deck_cards):
            result.extend(
                _trainer_card_issues(
                    card, f"observation.select.deck[{card_index}]"
                )
            )
    for key in ("contextCard", "effect"):
        card = select.get(key)
        if card is not None and not isinstance(card, dict):
            result.append(
                (
                    "row_trainer_select_card_invalid",
                    f"observation.select.{key} must be an object when present",
                )
            )
        elif isinstance(card, dict):
            result.extend(
                _trainer_card_issues(
                    card, f"observation.select.{key}"
                )
            )
    result.extend(_trainer_current_issues(observation))

    terminal_reward = row.get("terminal_reward")
    if (
        not isinstance(terminal_reward, (int, float))
        or isinstance(terminal_reward, bool)
        or not math.isfinite(float(terminal_reward))
    ):
        result.append(
            (
                "row_numeric_field_invalid",
                "terminal_reward must be a finite number",
            )
        )
    sample_weight = row.get("sample_weight")
    if (
        not isinstance(sample_weight, (int, float))
        or isinstance(sample_weight, bool)
        or not math.isfinite(float(sample_weight))
        or not 0.0 < float(sample_weight) <= 1.0
    ):
        result.append(
            (
                "row_numeric_field_invalid",
                "sample_weight must be finite and in (0, 1]",
            )
        )
    return result


def _record_row_issues(
    row: dict[str, Any],
    member_split: str | None,
    expected: ExpectedLayout,
    teams: TeamAllowlist,
    expected_deck_hash: str | None,
    schema_version: str | None,
    trainer_issues: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    row_split = row.get("split")
    if member_split is None or row_split != member_split:
        result.append(
            (
                "row_member_split_mismatch",
                f"row split {row_split!r} does not match member split "
                f"{member_split!r}",
            )
        )

    dataset_date = row.get("dataset_date")
    if (
        member_split not in SPLITS
        or not isinstance(dataset_date, str)
        or dataset_date not in expected.dates_by_split[member_split]
    ):
        allowed = (
            list(expected.dates_by_split[member_split])
            if member_split in SPLITS
            else []
        )
        result.append(
            (
                "row_dataset_date_invalid",
                f"dataset_date {dataset_date!r} is not allowed for "
                f"{member_split!r}; expected {allowed!r}",
            )
        )

    if not _valid_episode_id(row.get("episode_id")):
        result.append(
            ("row_episode_id_invalid", "episode_id is missing or invalid")
        )

    episode_uuid = row.get("episode_uuid")
    if episode_uuid == "" or (
        isinstance(episode_uuid, str) and not episode_uuid.strip()
    ):
        result.append(
            (
                "row_episode_uuid_invalid",
                "episode_uuid cannot be blank; use null when unavailable",
            )
        )
    elif episode_uuid is not None and not isinstance(episode_uuid, str):
        result.append(
            (
                "row_episode_uuid_invalid",
                "non-empty episode_uuid must be a string",
            )
        )

    if "visualize" in row:
        result.append(
            (
                "row_top_level_visualize",
                "top-level visualize key is forbidden",
            )
        )
    observation = row.get("observation")
    if not isinstance(observation, dict):
        result.append(
            ("row_observation_invalid", "observation must be an object")
        )
    elif contains_visualize(observation):
        result.append(
            (
                "row_observation_visualize",
                "visualize key is forbidden anywhere in observation",
            )
        )

    seat = row.get("seat")
    if not _integer_not_bool(seat) or seat not in (0, 1):
        result.append(
            ("row_seat_invalid", f"seat must be 0 or 1, observed {seat!r}")
        )
    observation_step = row.get("observation_step_index")
    action_step = row.get("action_step_index")
    if (
        not _integer_not_bool(observation_step)
        or observation_step < 0
        or not _integer_not_bool(action_step)
        or action_step != observation_step + 1
    ):
        result.append(
            (
                "row_step_alignment_invalid",
                "observation_step_index must be non-negative and "
                "action_step_index must equal observation_step_index + 1",
            )
        )
    result.extend(
        trainer_issues
        if trainer_issues is not None
        else _trainer_schema_issues(row)
    )

    if not isinstance(dataset_date, str) or not teams.allows(
        dataset_date, row.get("team_name")
    ):
        result.append(
            (
                "row_team_not_allowed",
                f"team_name {row.get('team_name')!r} is not in the frozen "
                f"allowlist for {dataset_date!r}",
            )
        )

    if (
        expected_deck_hash is not None
        and row.get("deck_hash") != expected_deck_hash
    ):
        result.append(
            (
                "row_deck_hash_mismatch",
                f"deck_hash {row.get('deck_hash')!r} does not match "
                f"{expected_deck_hash!r}",
            )
        )

    if schema_version is not None and row.get("schema_version") != schema_version:
        result.append(
            (
                "row_schema_version_mismatch",
                f"schema_version {row.get('schema_version')!r} does not "
                f"match manifest {schema_version!r}",
            )
        )
    return result


def _leakage_report(
    split_stats: dict[str, SplitStats],
    issues: IssueLog,
) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    leaked_ids: set[str] = set()
    leaked_uuids: set[str] = set()
    for first_index, first in enumerate(SPLITS):
        for second in SPLITS[first_index + 1 :]:
            ids = (
                split_stats[first].episode_ids
                & split_stats[second].episode_ids
            )
            uuids = (
                split_stats[first].episode_uuids
                & split_stats[second].episode_uuids
            )
            key = f"{first}_{second}"
            pairs[key] = {
                "episode_id_count": len(ids),
                "episode_id_samples": sorted(ids)[:10],
                "episode_uuid_count": len(uuids),
                "episode_uuid_samples": sorted(uuids)[:10],
            }
            if ids:
                issues.add(
                    "episode_id_cross_split_leakage",
                    f"{len(ids)} episode_id value(s) overlap between "
                    f"{first} and {second}: {sorted(ids)[:5]}",
                )
                leaked_ids.update(ids)
            if uuids:
                issues.add(
                    "episode_uuid_cross_split_leakage",
                    f"{len(uuids)} non-empty episode_uuid value(s) overlap "
                    f"between {first} and {second}: {sorted(uuids)[:5]}",
                )
                leaked_uuids.update(uuids)
    return {
        "pass": not leaked_ids and not leaked_uuids,
        "episode_ids_in_multiple_splits": len(leaked_ids),
        "episode_uuids_in_multiple_splits": len(leaked_uuids),
        "pairs": pairs,
    }


def audit_archive(
    archive_path: Path,
    teams_path: Path,
    expected: ExpectedLayout,
    *,
    expected_deck_hash: str | None = None,
    expected_episodes_scanned: int | None = None,
    max_error_samples: int = 50,
) -> dict[str, Any]:
    archive_path = archive_path.expanduser().resolve(strict=True)
    if not archive_path.is_file():
        raise ValueError(f"archive is not a file: {archive_path}")
    if expected_deck_hash is not None and not SHA256_RE.fullmatch(
        expected_deck_hash
    ):
        raise ValueError(
            "--expected-deck-hash must be a lowercase 64-character SHA-256"
        )
    if (
        expected_episodes_scanned is not None
        and expected_episodes_scanned < 0
    ):
        raise ValueError("--expected-episodes-scanned must be non-negative")
    if max_error_samples < 1:
        raise ValueError("--max-error-samples must be positive")

    teams_path = teams_path.expanduser().resolve(strict=True)
    teams = parse_team_file(teams_path)
    issues = IssueLog(max_samples=max_error_samples)
    split_stats = {split: SplitStats() for split in SPLITS}
    identity_index = AuditIdentityIndex()
    unknown_stats = SplitStats()
    shard_counts: Counter[str] = Counter()
    manifest: dict[str, Any] = {}
    before = archive_path.stat()

    try:
        with zipfile.ZipFile(archive_path, mode="r", allowZip64=True) as archive:
            infos = archive.infolist()
            duplicate_names = sorted(
                name
                for name, count in Counter(
                    info.filename for info in infos
                ).items()
                if count > 1
            )
            if duplicate_names:
                issues.add(
                    "zip_duplicate_members",
                    f"ZIP contains duplicate member names: "
                    f"{duplicate_names[:5]}",
                )

            manifest_infos = [
                info for info in infos if info.filename == "manifest.json"
            ]
            manifest = _manifest_object(archive, manifest_infos, issues)
            schema_version = manifest.get("schema_version")
            if not isinstance(schema_version, str):
                schema_version = None
            declared_filter_hash = manifest.get("deck_hash_filter")
            row_expected_deck_hash = expected_deck_hash
            if (
                row_expected_deck_hash is None
                and isinstance(declared_filter_hash, str)
                and SHA256_RE.fullmatch(declared_filter_hash) is not None
            ):
                row_expected_deck_hash = declared_filter_hash

            jsonl_infos: list[tuple[zipfile.ZipInfo, str | None]] = []
            for info in infos:
                if info.is_dir() or info.filename == "manifest.json":
                    continue
                if info.filename.endswith(".jsonl"):
                    split = _member_split(info.filename)
                    jsonl_infos.append((info, split))
                    if split is None:
                        issues.add(
                            "jsonl_member_outside_split",
                            f"JSONL member is outside train/valid/test: "
                            f"{info.filename}",
                        )
                    else:
                        shard_counts[split] += 1
                    continue
                issues.add(
                    "zip_unexpected_member",
                    f"unexpected non-JSONL ZIP member: {info.filename}",
                )

            for split in SPLITS:
                split_expected_empty = not expected.dates_by_split[split]
                if shard_counts[split] == 0 and not split_expected_empty:
                    issues.add(
                        "split_has_no_members",
                        f"ZIP has no JSONL members for {split}",
                    )
                elif shard_counts[split] != 0 and split_expected_empty:
                    issues.add(
                        "split_expected_empty_has_members",
                        f"ZIP has {shard_counts[split]} JSONL member(s) for "
                        f"expected-empty {split}; expected zero shards",
                    )

            for info, member_split in jsonl_infos:
                stats = (
                    split_stats[member_split]
                    if member_split in SPLITS
                    else unknown_stats
                )
                try:
                    with archive.open(info) as handle:
                        for line_number, raw_line in enumerate(handle, 1):
                            if not raw_line.strip():
                                continue
                            stats.rows_scanned += 1
                            location = f"{info.filename}:{line_number}"
                            try:
                                row = orjson.loads(raw_line)
                            except orjson.JSONDecodeError as error:
                                stats.invalid_rows += 1
                                issues.add(
                                    "row_json_invalid",
                                    f"{location}: invalid JSON: {error}",
                                )
                                continue
                            if not isinstance(row, dict):
                                stats.invalid_rows += 1
                                issues.add(
                                    "row_not_object",
                                    f"{location}: row is not an object",
                                )
                                continue

                            trainer_issues = _trainer_schema_issues(row)
                            if not trainer_issues:
                                stats.trainable_rows += 1
                            row_issues = _record_row_issues(
                                row,
                                member_split,
                                expected,
                                teams,
                                row_expected_deck_hash,
                                schema_version,
                                trainer_issues,
                            )
                            if member_split in SPLITS:
                                row_issues.extend(
                                    identity_index.check(row, member_split)
                                )
                            if row_issues:
                                stats.invalid_rows += 1
                                for code, message in row_issues:
                                    issues.add(code, f"{location}: {message}")
                            else:
                                stats.valid_rows += 1

                            if member_split in SPLITS:
                                episode_id = row.get("episode_id")
                                if _valid_episode_id(episode_id):
                                    stats.episode_ids.add(str(episode_id))
                                episode_uuid = row.get("episode_uuid")
                                if isinstance(episode_uuid, str) and episode_uuid:
                                    stats.episode_uuids.add(episode_uuid)
                                dataset_date = row.get("dataset_date")
                                if isinstance(dataset_date, str):
                                    stats.dataset_dates.add(dataset_date)
                                team_name = row.get("team_name")
                                if isinstance(team_name, str):
                                    stats.team_rows[team_name] += 1
                                select_context = row.get("select_context")
                                if isinstance(select_context, str):
                                    stats.context_rows[select_context] += 1
                except (
                    OSError,
                    RuntimeError,
                    zipfile.BadZipFile,
                    EOFError,
                ) as error:
                    issues.add(
                        "zip_member_read_failure",
                        f"failed while streaming {info.filename}: {error}",
                    )
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        issues.add(
            "zip_open_failure",
            f"archive could not be fully read: {error}",
        )
    finally:
        identity_index.close()

    after = archive_path.stat()
    archive_stable = (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
    if not archive_stable:
        issues.add(
            "archive_changed_during_audit",
            "archive identity, size, or mtime changed while it was being read",
        )

    manifest_is_filtered = (
        isinstance(manifest.get("filtered_from"), str)
        and bool(manifest["filtered_from"])
        and isinstance(manifest.get("deck_hash_filter"), str)
        and SHA256_RE.fullmatch(manifest["deck_hash_filter"]) is not None
    )
    for split in SPLITS:
        split_expected_empty = not expected.dates_by_split[split]
        if split_stats[split].rows_scanned == 0 and not split_expected_empty:
            issues.add(
                "split_empty",
                f"{split} contains no non-empty JSONL rows",
            )
        elif split_stats[split].rows_scanned != 0 and split_expected_empty:
            issues.add(
                "split_expected_empty_has_rows",
                f"{split} contains {split_stats[split].rows_scanned} "
                "non-empty JSONL row(s), expected zero",
            )
        observed_dates = split_stats[split].dataset_dates
        wanted_dates = set(expected.dates_by_split[split])
        unexpected_dates = observed_dates - wanted_dates
        if unexpected_dates:
            issues.add(
                "split_observed_dates_mismatch",
                f"{split} contains unexpected dates "
                f"{sorted(unexpected_dates)!r}; allowed "
                f"{sorted(wanted_dates)!r}",
            )
        if not manifest_is_filtered and observed_dates != wanted_dates:
            missing_dates = wanted_dates - observed_dates
            if missing_dates:
                issues.add(
                    "split_observed_dates_incomplete",
                    f"unfiltered {split} is missing expected dates "
                    f"{sorted(missing_dates)!r}",
                )

    leakage = _leakage_report(split_stats, issues)
    filter_lineage_report = _validate_manifest(
        manifest,
        expected,
        teams,
        expected_deck_hash,
        expected_episodes_scanned,
        split_stats,
        shard_counts,
        issues,
    )

    total_rows = sum(
        split_stats[split].rows_scanned for split in SPLITS
    ) + unknown_stats.rows_scanned
    total_invalid = sum(
        split_stats[split].invalid_rows for split in SPLITS
    ) + unknown_stats.invalid_rows
    total_valid = sum(
        split_stats[split].valid_rows for split in SPLITS
    ) + unknown_stats.valid_rows
    total_trainable = sum(
        split_stats[split].trainable_rows for split in SPLITS
    ) + unknown_stats.trainable_rows
    unique_ids = set().union(
        *(split_stats[split].episode_ids for split in SPLITS)
    )
    unique_uuids = set().union(
        *(split_stats[split].episode_uuids for split in SPLITS)
    )
    audit_passed = issues.total == 0
    legacy_filter_manifest = (
        filter_lineage_report["mode"] == "legacy_missing_filter_lineage"
    )
    report = {
        "schema_version": "ptcg-bc-timeforward-audit-v1",
        "pass": audit_passed,
        "diagnostic_only": legacy_filter_manifest,
        "promotion_eligible": audit_passed and not legacy_filter_manifest,
        "archive": str(archive_path),
        "archive_read_only": True,
        "archive_stable_during_audit": archive_stable,
        "teams_file": str(teams_path),
        "expected": {
            **expected.public(),
            "deck_hash": expected_deck_hash,
            "episodes_scanned": expected_episodes_scanned,
            "team_count": len(teams.all_names),
        },
        "statistics": {
            "rows_scanned": total_rows,
            "valid_rows": total_valid,
            "trainable_rows": total_trainable,
            "invalid_rows": total_invalid,
            "unique_episode_ids": len(unique_ids),
            "unique_episode_uuids": len(unique_uuids),
            "by_split": {
                split: split_stats[split].public(shard_counts[split])
                for split in SPLITS
            },
            "outside_split_members": unknown_stats.public(0),
        },
        "leakage": leakage,
        "filter_lineage": filter_lineage_report,
        "manifest_summary": {
            "schema_version": manifest.get("schema_version"),
            "competition": manifest.get("competition"),
            "dates": manifest.get("dates"),
            "split_policy": manifest.get("split_policy"),
            "stats": manifest.get("stats"),
            "split_decisions": manifest.get("split_decisions"),
            "shards": manifest.get("shards"),
            "source_count": (
                len(manifest["sources"])
                if isinstance(manifest.get("sources"), list)
                else None
            ),
        },
        "issues": issues.public(),
    }
    return report


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = Path(os.path.abspath(path.expanduser()))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-audit a time-forward BC ZIP without modifying it."
        )
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--teams-file", type=Path, required=True)
    parser.add_argument(
        "--expected-deck-hash",
        help="Optional lowercase SHA-256 required on every row",
    )
    parser.add_argument(
        "--expected-dates",
        nargs="+",
        help=(
            "Exactly 12 chronological dates; first 10 train, then valid/test"
        ),
    )
    parser.add_argument("--expected-train-dates", nargs="+")
    parser.add_argument("--expected-valid-dates", nargs="+")
    test_layout = parser.add_mutually_exclusive_group()
    test_layout.add_argument("--expected-test-dates", nargs="+")
    test_layout.add_argument(
        "--expect-empty-test",
        action="store_true",
        help=(
            "For split-specific layouts, require test_dates=[], zero test "
            "rows, and zero test shards"
        ),
    )
    parser.add_argument("--expected-episodes-scanned", type=int)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the complete JSON report",
    )
    parser.add_argument("--max-error-samples", type=int, default=50)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        expected = build_expected_layout(
            expected_dates=args.expected_dates,
            train_dates=args.expected_train_dates,
            valid_dates=args.expected_valid_dates,
            test_dates=args.expected_test_dates,
            expect_empty_test=args.expect_empty_test,
        )
        report = audit_archive(
            args.archive,
            args.teams_file,
            expected,
            expected_deck_hash=args.expected_deck_hash,
            expected_episodes_scanned=args.expected_episodes_scanned,
            max_error_samples=args.max_error_samples,
        )
    except (OSError, ValueError) as error:
        report = {
            "schema_version": "ptcg-bc-timeforward-audit-v1",
            "pass": False,
            "diagnostic_only": True,
            "promotion_eligible": False,
            "archive": str(args.archive),
            "configuration_error": f"{type(error).__name__}: {error}",
        }
        if args.json_output is not None:
            write_json_atomic(args.json_output, report)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    if args.json_output is not None:
        write_json_atomic(args.json_output, report)
        print(
            f"Audit {'PASS' if report['pass'] else 'FAIL'}: "
            f"{args.json_output.resolve()}"
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
