#!/usr/bin/env python3
"""Profile exact deck hashes in a local PTCG BC decision archive.

BC archives contain one JSONL row per visible decision, so a single episode
normally appears many times.  Episode outcomes and the requested seat/date/team
counts are deduplicated by ``(dataset_date, episode_id, seat)``.  The
``decision_rows`` counters deliberately retain every JSONL decision row.

The input ZIP is opened read-only and is never extracted or modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import orjson


BC_SCHEMA_VERSION = "ptcg-bc-visible-decisions-v1"
PROFILE_SCHEMA_VERSION = "ptcg-bc-exact-deck-profile-v1"
DECK_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class EpisodeIdentity:
    """Fields that must agree on every decision row for one episode seat."""

    deck_hash: str
    team_name: str
    reward: float
    split: str


@dataclass
class OutcomeStats:
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    team_counts: Counter[str] = field(default_factory=Counter)
    date_counts: Counter[str] = field(default_factory=Counter)

    def add(self, reward: float, team_name: str, dataset_date: str) -> None:
        self.episodes += 1
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1
        self.team_counts[team_name] += 1
        self.date_counts[dataset_date] += 1

    def public(self) -> dict[str, Any]:
        if self.episodes != self.wins + self.losses + self.draws:
            raise RuntimeError("Internal outcome-count mismatch")
        return {
            "episodes": self.episodes,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / self.episodes if self.episodes else None,
            "team_count": len(self.team_counts),
            "team_counts": dict(
                sorted(
                    self.team_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "date_counts": dict(sorted(self.date_counts.items())),
        }


@dataclass
class DeckStats:
    decision_rows: int = 0
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    seat_counts: Counter[str] = field(default_factory=Counter)
    date_counts: Counter[str] = field(default_factory=Counter)
    team_counts: Counter[str] = field(default_factory=Counter)
    split_stats: dict[str, OutcomeStats] = field(
        default_factory=lambda: {split: OutcomeStats() for split in SPLITS}
    )

    def add_episode(
        self,
        *,
        reward: float,
        seat: int,
        dataset_date: str,
        team_name: str,
        split: str,
    ) -> None:
        self.episodes += 1
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1
        self.seat_counts[str(seat)] += 1
        self.date_counts[dataset_date] += 1
        self.team_counts[team_name] += 1
        self.split_stats[split].add(reward, team_name, dataset_date)

    def public(
        self,
        deck_hash: str,
        profiled_splits: tuple[str, ...],
    ) -> dict[str, Any]:
        if self.episodes != self.wins + self.losses + self.draws:
            raise RuntimeError(f"Internal outcome-count mismatch for {deck_hash}")
        return {
            "deck_hash": deck_hash,
            "episodes": self.episodes,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / self.episodes,
            "decision_rows": self.decision_rows,
            "split_stats": {
                split: (
                    self.split_stats[split].public()
                    if split in profiled_splits
                    else {"profiled": False}
                )
                for split in SPLITS
            },
            "seat_counts": dict(sorted(self.seat_counts.items())),
            "date_counts": dict(sorted(self.date_counts.items())),
            "team_counts": dict(
                sorted(
                    self.team_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        }


def _required_text(row: dict[str, Any], field_name: str, location: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field_name} must be a non-empty string")
    return value


def _episode_id(row: dict[str, Any], location: str) -> str:
    value = row.get("episode_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{location}: episode_id must be a non-empty string or int")
    result = str(value)
    if not result:
        raise ValueError(f"{location}: episode_id must not be empty")
    return result


def _validated_row(
    row: Any,
    location: str,
) -> tuple[str, str, int, str, str, float, str]:
    if not isinstance(row, dict):
        raise ValueError(f"{location}: JSONL value must be an object")
    if row.get("schema_version") != BC_SCHEMA_VERSION:
        raise ValueError(
            f"{location}: unexpected schema_version "
            f"{row.get('schema_version')!r}"
        )

    dataset_date = _required_text(row, "dataset_date", location)
    try:
        date.fromisoformat(dataset_date)
    except ValueError as error:
        raise ValueError(
            f"{location}: dataset_date is not an ISO date: {dataset_date!r}"
        ) from error

    episode_id = _episode_id(row, location)
    seat = row.get("seat")
    if isinstance(seat, bool) or not isinstance(seat, int) or seat not in (0, 1):
        raise ValueError(f"{location}: seat must be integer 0 or 1")

    deck_hash = _required_text(row, "deck_hash", location)
    if DECK_HASH_RE.fullmatch(deck_hash) is None:
        raise ValueError(
            f"{location}: deck_hash must be a lowercase SHA-256 hex digest"
        )
    team_name = _required_text(row, "team_name", location)
    split = _required_text(row, "split", location)
    if split not in SPLITS:
        raise ValueError(
            f"{location}: split must be one of {', '.join(SPLITS)}"
        )

    raw_reward = row.get("terminal_reward")
    if isinstance(raw_reward, bool) or not isinstance(raw_reward, (int, float)):
        raise ValueError(f"{location}: terminal_reward must be numeric")
    reward = float(raw_reward)
    if not math.isfinite(reward):
        raise ValueError(f"{location}: terminal_reward must be finite")
    return dataset_date, episode_id, seat, deck_hash, team_name, reward, split


def _manifest(archive: zipfile.ZipFile, input_path: Path) -> dict[str, Any]:
    try:
        raw = archive.read("manifest.json")
    except KeyError as error:
        raise ValueError(f"{input_path}: BC archive has no manifest.json") from error
    try:
        manifest = orjson.loads(raw)
    except orjson.JSONDecodeError as error:
        raise ValueError(f"{input_path}: invalid manifest.json") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{input_path}: manifest.json must be an object")
    if manifest.get("schema_version") != BC_SCHEMA_VERSION:
        raise ValueError(
            f"{input_path}: unexpected manifest schema_version "
            f"{manifest.get('schema_version')!r}"
        )
    return manifest


def _profiled_splits(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("profile_splits must contain at least one split")
    result: list[str] = []
    for value in values:
        if value not in SPLITS:
            raise ValueError(f"Unknown profile split: {value!r}")
        if value in result:
            raise ValueError(f"Duplicate profile split: {value!r}")
        result.append(value)
    return tuple(split for split in SPLITS if split in result)


def _jsonl_member_split(member: zipfile.ZipInfo, input_path: Path) -> str:
    parts = PurePosixPath(member.filename).parts
    if not parts or parts[0] not in SPLITS:
        raise ValueError(
            f"{input_path}: JSONL member is outside train/valid/test: "
            f"{member.filename!r}"
        )
    return parts[0]


def profile_archive(
    input_path: Path,
    min_episodes: int = 1,
    profile_splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    """Stream-profile a read-only BC ZIP archive and return JSON-safe data."""

    if min_episodes < 1:
        raise ValueError("min_episodes must be at least 1")
    profile_splits = _profiled_splits(profile_splits)
    sealed_splits = tuple(
        split for split in SPLITS if split not in profile_splits
    )
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not zipfile.is_zipfile(input_path):
        raise ValueError(f"{input_path}: input is not a ZIP archive")

    identities: dict[tuple[str, str, int], EpisodeIdentity] = {}
    stats_by_deck: dict[str, DeckStats] = {}
    split_date_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    decision_rows = 0

    with zipfile.ZipFile(input_path, "r") as archive:
        manifest = _manifest(archive, input_path)
        all_members = sorted(
            (
                member
                for member in archive.infolist()
                if not member.is_dir() and member.filename.endswith(".jsonl")
            ),
            key=lambda member: member.filename,
        )
        if not all_members:
            raise ValueError(f"{input_path}: BC archive has no JSONL shards")
        members = [
            member
            for member in all_members
            if _jsonl_member_split(member, input_path) in profile_splits
        ]

        for member in members:
            with archive.open(member, "r") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    location = f"{member.filename}:{line_number}"
                    try:
                        row = orjson.loads(line)
                    except orjson.JSONDecodeError as error:
                        raise ValueError(f"{location}: invalid JSON") from error
                    (
                        dataset_date,
                        episode_id,
                        seat,
                        deck_hash,
                        team_name,
                        reward,
                        split,
                    ) = _validated_row(row, location)

                    key = (dataset_date, episode_id, seat)
                    identity = EpisodeIdentity(deck_hash, team_name, reward, split)
                    existing = identities.get(key)
                    if existing is not None and existing != identity:
                        raise ValueError(
                            f"{location}: inconsistent rows for episode key {key!r}; "
                            f"first={existing!r}, current={identity!r}"
                        )

                    deck_stats = stats_by_deck.setdefault(deck_hash, DeckStats())
                    deck_stats.decision_rows += 1
                    decision_rows += 1
                    if existing is None:
                        identities[key] = identity
                        split_date_counts[split][dataset_date] += 1
                        deck_stats.add_episode(
                            reward=reward,
                            seat=seat,
                            dataset_date=dataset_date,
                            team_name=team_name,
                            split=split,
                        )

    selected = [
        deck_stats.public(deck_hash, profile_splits)
        for deck_hash, deck_stats in stats_by_deck.items()
        if deck_stats.episodes >= min_episodes
    ]
    selected.sort(key=lambda item: (-item["episodes"], item["deck_hash"]))
    selected_decision_rows = sum(item["decision_rows"] for item in selected)
    selected_episode_seats = sum(item["episodes"] for item in selected)

    manifest_rows = manifest.get("stats", {}).get("decisions")
    complete_profile = set(profile_splits) == set(SPLITS)
    manifest_rows_is_integer = (
        isinstance(manifest_rows, int) and not isinstance(manifest_rows, bool)
    )
    manifest_match = (
        manifest_rows == decision_rows
        if complete_profile and manifest_rows_is_integer
        else None
    )
    if complete_profile and manifest_rows_is_integer:
        manifest_match_note = (
            "Compared because train, valid, and test JSONL members were profiled."
        )
    elif complete_profile:
        manifest_match_note = (
            "Not compared because manifest stats.decisions is not an integer."
        )
    else:
        manifest_match_note = (
            "Not compared because manifest decisions cover all splits while "
            f"this profile opened only {', '.join(profile_splits)}; sealed "
            f"splits: {', '.join(sealed_splits)}."
        )
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "source": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "bc_schema_version": manifest.get("schema_version"),
            "jsonl_members": len(all_members),
            "profiled_jsonl_members": len(members),
            "sealed_jsonl_members": len(all_members) - len(members),
            "manifest_decision_rows": manifest_rows,
            "manifest_decision_rows_match": manifest_match,
            "manifest_decision_rows_match_note": manifest_match_note,
        },
        "deduplication_key": ["dataset_date", "episode_id", "seat"],
        "profiled_splits": list(profile_splits),
        "sealed_splits": list(sealed_splits),
        "split_date_counts": {
            split: dict(sorted(split_date_counts[split].items()))
            for split in profile_splits
        },
        "min_episodes": min_episodes,
        "statistics": {
            "decision_rows": decision_rows,
            "unique_episode_seats": len(identities),
            "exact_decks": len(stats_by_deck),
            "selected_decision_rows": selected_decision_rows,
            "selected_episode_seats": selected_episode_seats,
            "selected_exact_decks": len(selected),
            "filtered_exact_decks": len(stats_by_deck) - len(selected),
        },
        "decks": selected,
    }


def atomic_write_json(
    output_path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Atomically install JSON without replacing a file unless authorized."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists; pass --overwrite to replace it"
        )
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    installed = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"{output_path} exists; pass --overwrite to replace it"
                ) from error
            temporary_path.unlink()
        installed = True
    finally:
        if not installed and temporary_path.exists():
            temporary_path.unlink()


def run(
    input_path: Path,
    output_path: Path,
    *,
    min_episodes: int,
    overwrite: bool,
    profile_splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must differ")
    if output_path.resolve().exists() and not overwrite:
        raise FileExistsError(
            f"{output_path.resolve()} exists; pass --overwrite to replace it"
        )
    report = profile_archive(
        input_path,
        min_episodes=min_episodes,
        profile_splits=profile_splits,
    )
    atomic_write_json(output_path, report, overwrite=overwrite)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument(
        "--profile-splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        metavar="SPLIT",
        help=(
            "JSONL splits to open and profile. For sealed route selection use "
            "'--profile-splits train valid'."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        args.input,
        args.output,
        min_episodes=args.min_episodes,
        overwrite=args.overwrite,
        profile_splits=tuple(args.profile_splits),
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "decision_rows": report["statistics"]["decision_rows"],
                "unique_episode_seats": report["statistics"][
                    "unique_episode_seats"
                ],
                "selected_exact_decks": report["statistics"][
                    "selected_exact_decks"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
