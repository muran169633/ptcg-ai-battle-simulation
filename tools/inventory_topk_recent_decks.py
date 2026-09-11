#!/usr/bin/env python3
"""Inventory exact deck clusters used by a frozen leaderboard Top-K.

The scanner reads official daily episode ZIPs without extracting them.  It
uses the frozen leaderboard only as a team allowlist and records exact deck
hashes, outcomes, dates, teams, and matchup frequencies.  No replay decision
is copied into the output, so this is an inventory/preselection step rather
than a BC-data builder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import prepare_bc_week as base
import prepare_gold8_week as gold8


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ptcg-topk-recent-exact-deck-inventory-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_leaderboard(path: Path, top_k: int) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        if len(members) != 1:
            raise ValueError(f"Expected one leaderboard CSV in {path}, got {members}")
        with archive.open(members[0]) as raw:
            rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")))
    rows.sort(key=lambda row: int(row.get("Rank", 10**9)))
    selected = rows[:top_k]
    if len(selected) != top_k:
        raise ValueError(f"Requested Top {top_k}, found {len(selected)} rows")
    normalized = [base.normalize_team_name(row.get("TeamName")) for row in selected]
    if any(not name for name in normalized):
        raise ValueError("Top-K contains an empty normalized team name")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Top-K contains duplicate normalized team names")
    return selected


def outcome_name(reward: float) -> str:
    if reward > 0:
        return "wins"
    if reward < 0:
        return "losses"
    return "draws"


def scan_source(
    source_path: str,
    dataset_date: str,
    team_lookup: dict[str, dict[str, Any]],
    max_members: int | None,
) -> dict[str, Any]:
    path = Path(source_path)
    stats: Counter[str] = Counter()
    deck_counts: dict[str, Counter[str]] = defaultdict(Counter)
    deck_teams: dict[str, Counter[str]] = defaultdict(Counter)
    deck_ranks: dict[str, Counter[str]] = defaultdict(Counter)
    deck_matchups: dict[str, Counter[str]] = defaultdict(Counter)
    team_decks: dict[str, Counter[str]] = defaultdict(Counter)
    captured_decks: dict[str, list[int]] = {}

    with zipfile.ZipFile(path) as archive:
        members = sorted(
            (
                info
                for info in archive.infolist()
                if not info.is_dir() and base.NUMERIC_JSON_RE.search(info.filename)
            ),
            key=lambda info: info.filename,
        )
        if max_members is not None:
            members = members[:max_members]
        stats["archive_members"] = len(members)
        for info in members:
            stats["episodes_scanned"] += 1
            try:
                with archive.open(info) as handle:
                    episode = base.json_load(handle)
            except Exception:
                stats["json_errors"] += 1
                continue
            if not isinstance(episode, dict):
                stats["invalid_episode_objects"] += 1
                continue
            names = base.episode_names(episode)
            if len(names) < 2:
                stats["unresolved_team_names"] += 1
                continue
            selected_seats = [
                seat
                for seat, display in enumerate(names[:2])
                if base.normalize_team_name(display) in team_lookup
            ]
            if not selected_seats:
                continue
            stats["episodes_with_topk_team"] += 1
            deck_hashes, decks = gold8.extract_replay_decks(episode)
            rewards = base.final_rewards(episode)
            for seat in selected_seats:
                stats["topk_seat_appearances"] += 1
                display = names[seat]
                normalized = base.normalize_team_name(display)
                team = team_lookup[normalized]
                deck_hash = deck_hashes[seat] if seat < len(deck_hashes) else None
                deck = decks[seat] if seat < len(decks) else None
                if deck_hash is None or deck is None:
                    stats["topk_seats_without_deck"] += 1
                    continue
                stats["topk_seats_with_deck"] += 1
                previous = captured_decks.get(deck_hash)
                if previous is not None and previous != deck:
                    raise RuntimeError(f"Deck hash collision for {deck_hash}")
                captured_decks[deck_hash] = deck
                reward = rewards[seat] if seat < len(rewards) else 0.0
                outcome = outcome_name(reward)
                deck_counts[deck_hash]["appearances"] += 1
                deck_counts[deck_hash][outcome] += 1
                deck_counts[deck_hash][f"date::{dataset_date}"] += 1
                deck_teams[deck_hash][display] += 1
                deck_ranks[deck_hash][str(team["rank"])] += 1
                team_decks[display][deck_hash] += 1
                opponent_hash = (
                    deck_hashes[1 - seat]
                    if len(deck_hashes) >= 2 and deck_hashes[1 - seat]
                    else "unknown"
                )
                deck_matchups[deck_hash][str(opponent_hash)] += 1

    return {
        "date": dataset_date,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "stats": dict(stats),
        "deck_counts": {key: dict(value) for key, value in deck_counts.items()},
        "deck_teams": {key: dict(value) for key, value in deck_teams.items()},
        "deck_ranks": {key: dict(value) for key, value in deck_ranks.items()},
        "deck_matchups": {key: dict(value) for key, value in deck_matchups.items()},
        "team_decks": {key: dict(value) for key, value in team_decks.items()},
        "captured_decks": captured_decks,
    }


def merge_counter_maps(
    target: dict[str, Counter[str]], source: dict[str, dict[str, int]]
) -> None:
    for key, values in source.items():
        target[key].update(values)


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    leaderboard = read_leaderboard(args.leaderboard_zip, args.top_k)
    team_lookup = {
        base.normalize_team_name(row["TeamName"]): {
            "rank": int(row["Rank"]),
            "team_name": row["TeamName"],
            "score": float(row["Score"]),
            "last_submission_date": row.get("LastSubmissionDate"),
        }
        for row in leaderboard
    }
    sources: list[tuple[str, Path]] = []
    for path in args.input:
        dataset_date = base.parse_date_from_path(path)
        if not dataset_date:
            raise ValueError(f"Cannot parse date from {path}")
        sources.append((dataset_date, path.resolve()))
    sources.sort()
    dates = [date for date, _ in sources]
    if len(dates) != len(set(dates)):
        raise ValueError(f"Duplicate source dates: {dates}")
    if len(sources) != args.days:
        raise ValueError(f"Expected {args.days} sources, got {len(sources)}")
    if args.date_end and dates[-1] != args.date_end:
        raise ValueError(f"Latest source is {dates[-1]}, expected {args.date_end}")
    for _, path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                scan_source,
                str(path),
                dataset_date,
                team_lookup,
                args.max_members_per_source,
            ): dataset_date
            for dataset_date, path in sources
        }
        for future in as_completed(futures):
            result = future.result()
            print(
                f"scanned {result['date']}: "
                f"episodes={result['stats'].get('episodes_scanned', 0)} "
                f"topk_seats={result['stats'].get('topk_seats_with_deck', 0)}",
                flush=True,
            )
            results.append(result)
    results.sort(key=lambda row: row["date"])

    global_stats: Counter[str] = Counter()
    deck_counts: dict[str, Counter[str]] = defaultdict(Counter)
    deck_teams: dict[str, Counter[str]] = defaultdict(Counter)
    deck_ranks: dict[str, Counter[str]] = defaultdict(Counter)
    deck_matchups: dict[str, Counter[str]] = defaultdict(Counter)
    team_decks: dict[str, Counter[str]] = defaultdict(Counter)
    captured_decks: dict[str, list[int]] = {}
    for result in results:
        global_stats.update(result["stats"])
        merge_counter_maps(deck_counts, result["deck_counts"])
        merge_counter_maps(deck_teams, result["deck_teams"])
        merge_counter_maps(deck_ranks, result["deck_ranks"])
        merge_counter_maps(deck_matchups, result["deck_matchups"])
        merge_counter_maps(team_decks, result["team_decks"])
        for deck_hash, deck in result["captured_decks"].items():
            previous = captured_decks.get(deck_hash)
            if previous is not None and previous != deck:
                raise RuntimeError(f"Deck hash collision for {deck_hash}")
            captured_decks[deck_hash] = deck

    known_labels = {profile.deck_hash: profile.label for profile in gold8.PROFILES}
    top_decks: list[dict[str, Any]] = []
    for deck_hash, counts in deck_counts.items():
        teams = deck_teams[deck_hash]
        ranks = deck_ranks[deck_hash]
        appearances = int(counts["appearances"])
        weighted_rank_sum = sum(int(rank) * count for rank, count in ranks.items())
        dates_seen = sorted(
            key.removeprefix("date::")
            for key, value in counts.items()
            if key.startswith("date::") and value
        )
        top_decks.append(
            {
                "deck_hash": deck_hash,
                "known_label": known_labels.get(deck_hash),
                "appearances": appearances,
                "unique_topk_teams": len(teams),
                "best_rank": min(int(rank) for rank in ranks),
                "appearance_weighted_mean_rank": weighted_rank_sum / appearances,
                "wins": int(counts["wins"]),
                "losses": int(counts["losses"]),
                "draws": int(counts["draws"]),
                "win_rate": counts["wins"] / appearances,
                "first_seen": dates_seen[0],
                "last_seen": dates_seen[-1],
                "dates": {
                    key.removeprefix("date::"): int(value)
                    for key, value in sorted(counts.items())
                    if key.startswith("date::")
                },
                "teams": [
                    {"team_name": team, "appearances": count}
                    for team, count in teams.most_common()
                ],
                "opponent_decks": [
                    {"deck_hash": opponent, "appearances": count}
                    for opponent, count in deck_matchups[deck_hash].most_common(20)
                ],
                "cards": captured_decks[deck_hash],
            }
        )
    top_decks.sort(
        key=lambda row: (
            -int(row["unique_topk_teams"]),
            -int(row["appearances"]),
            int(row["best_rank"]),
            str(row["deck_hash"]),
        )
    )

    team_rows = []
    for row in leaderboard:
        team = row["TeamName"]
        decks = team_decks.get(team, Counter())
        team_rows.append(
            {
                "rank": int(row["Rank"]),
                "team_name": team,
                "score": float(row["Score"]),
                "last_submission_date": row.get("LastSubmissionDate"),
                "observed_appearances": sum(decks.values()),
                "observed_decks": [
                    {"deck_hash": deck_hash, "appearances": count}
                    for deck_hash, count in decks.most_common()
                ],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_semantics": (
            "current frozen Top-K team names applied to every episode in the "
            "selected recent window; this is not a historical daily Top-K"
        ),
        "leaderboard": {
            "path": str(args.leaderboard_zip.resolve()),
            "bytes": args.leaderboard_zip.stat().st_size,
            "sha256": sha256_file(args.leaderboard_zip),
            "top_k": args.top_k,
            "rows": team_rows,
        },
        "window": {
            "days": len(dates),
            "date_start": dates[0],
            "date_end": dates[-1],
            "dates": dates,
        },
        "sources": [
            {
                "date": result["date"],
                "path": result["path"],
                "bytes": result["bytes"],
                "stats": result["stats"],
            }
            for result in results
        ],
        "stats": dict(global_stats),
        "exact_deck_cluster_count": len(top_decks),
        "top_decks": top_decks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard-zip", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--date-end")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-members-per-source", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1 or args.days < 1 or args.workers < 1:
        raise ValueError("--top-k, --days, and --workers must be positive")
    if args.max_members_per_source is not None and args.max_members_per_source < 1:
        raise ValueError("--max-members-per-source must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    inventory = build_inventory(args)
    atomic_write_text(
        args.output,
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "exact_deck_clusters": inventory["exact_deck_cluster_count"],
                "stats": inventory["stats"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
