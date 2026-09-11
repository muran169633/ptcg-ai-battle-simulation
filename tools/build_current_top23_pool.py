#!/usr/bin/env python3
"""Freeze the current leaderboard Top 23 as a deck-frequency opponent pool.

The script resolves every leaderboard team's active public submission through
Kaggle's public episode index, then verifies the exact 60-card deck from recent
completed public replays.  One output CSV is written per leaderboard slot, so
duplicate decks retain their observed Top-23 frequency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import prepare_live_submission_bc as live  # noqa: E402


_KAGGLE_API: Any | None = None


def normalize_team_name(value: str) -> str:
    return " ".join(value.casefold().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_leaderboard(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("teamId,teamName,submissionDate,score")
        )
    except StopIteration as error:
        raise ValueError(f"Leaderboard CSV header is missing in {path}") from error
    rows: list[dict[str, Any]] = []
    for rank, raw in enumerate(csv.DictReader(lines[header_index:]), 1):
        rows.append(
            {
                "rank": rank,
                "team_id": int(raw["teamId"]),
                "team_name": raw["teamName"],
                "submission_date": raw["submissionDate"],
                "score": float(raw["score"]),
            }
        )
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} leaderboard rows, found {len(rows)}"
        )
    return rows


def sdk_agent_payload(value: Any) -> dict[str, Any]:
    """Normalize Kaggle 2.0.x model objects and newer dict payloads."""
    if isinstance(value, dict):
        return value
    return {
        "submissionId": getattr(value, "submission_id", None),
        "index": getattr(value, "index", None),
        "reward": getattr(value, "reward", None),
        "state": str(getattr(value, "state", "") or ""),
        "teamName": getattr(value, "team_name", None),
        "teamId": getattr(value, "team_id", None),
    }


def sdk_listing_payload(submission_id: int) -> dict[str, Any]:
    global _KAGGLE_API
    if _KAGGLE_API is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        _KAGGLE_API = KaggleApi()
        _KAGGLE_API.authenticate()
    episodes = []
    for value in _KAGGLE_API.competition_list_episodes(submission_id):
        episode_type = getattr(value, "type", None)
        type_name = getattr(episode_type, "name", None) or str(episode_type or "")
        create_time = getattr(value, "create_time", None)
        episodes.append(
            {
                "id": int(value.id),
                "createTime": (
                    create_time.isoformat()
                    if hasattr(create_time, "isoformat")
                    else str(create_time or "")
                ),
                "type": type_name,
                "agents": [
                    sdk_agent_payload(agent)
                    for agent in (getattr(value, "agents", None) or [])
                ],
            }
        )
    return {"episodes": episodes, "teams": []}


def listing_payload(submission_id: int, backend: str) -> dict[str, Any]:
    if backend == "kaggle_api":
        return sdk_listing_payload(submission_id)
    payload, _ = live.fetch_json(
        live.LIST_EPISODES_URL,
        {
            "ids": [],
            "submissionId": submission_id,
            "successfulOnly": True,
            "includeInProgress": False,
        },
    )
    return payload


def crawl_active_submissions(
    rows: list[dict[str, Any]],
    seed_submission_ids: list[int],
    max_rounds: int,
    backend: str,
) -> tuple[dict[int, int], dict[int, dict[str, Any]], dict[str, Any]]:
    targets = {int(row["team_id"]) for row in rows}
    resolved: dict[int, int] = {}
    listings: dict[int, dict[str, Any]] = {}
    queue = list(dict.fromkeys(seed_submission_ids))
    seen: set[int] = set()
    latest_episode_by_team: dict[int, str] = {}
    rounds: list[dict[str, Any]] = []
    for round_index in range(max_rounds):
        current = [submission for submission in queue if submission not in seen]
        queue = []
        for submission_id in current:
            seen.add(submission_id)
            listing = listing_payload(submission_id, backend)
            listings[submission_id] = listing
            for team in listing.get("teams") or []:
                if not isinstance(team, dict):
                    continue
                team_id = int(team.get("id", -1))
                active = team.get("publicLeaderboardSubmissionId")
                if team_id not in targets or active is None:
                    continue
                active_id = int(active)
                resolved[team_id] = active_id
                if active_id not in seen:
                    queue.append(active_id)
            for episode in listing.get("episodes") or []:
                if not isinstance(episode, dict):
                    continue
                created = episode_time(episode)
                for agent in episode.get("agents") or []:
                    if not isinstance(agent, dict):
                        continue
                    team_id = int(agent.get("teamId", -1))
                    candidate = agent.get("submissionId")
                    if team_id not in targets or candidate is None:
                        continue
                    if created >= latest_episode_by_team.get(team_id, ""):
                        latest_episode_by_team[team_id] = created
                        active_id = int(candidate)
                        resolved[team_id] = active_id
                        if active_id not in seen:
                            queue.append(active_id)
            time.sleep(0.05)
        missing = sorted(targets - set(resolved))
        rounds.append(
            {
                "round": round_index,
                "queried_submissions": current,
                "resolved_team_count": len(resolved),
                "missing_team_ids": missing,
            }
        )
        print(
            f"crawl_round={round_index} queried={len(current)} "
            f"resolved={len(resolved)}/{len(targets)}",
            flush=True,
        )
        if not missing:
            break
    return resolved, listings, {
        "backend": backend,
        "rounds": rounds,
        "queries": len(seen),
    }


def exact_agent(episode: dict[str, Any], submission_id: int) -> dict[str, Any]:
    matches = [
        agent
        for agent in episode.get("agents") or []
        if isinstance(agent, dict)
        and int(agent.get("submissionId", -1)) == submission_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one agent for submission {submission_id}, got {len(matches)}"
        )
    return matches[0]


def episode_time(episode: dict[str, Any]) -> str:
    return str(episode.get("createTime") or "")


def verified_current_deck(
    row: dict[str, Any],
    submission_id: int,
    listing: dict[str, Any],
    replay_count: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for episode in listing.get("episodes") or []:
        if not isinstance(episode, dict):
            continue
        if str(episode.get("type", "")).upper() != "EPISODE_TYPE_PUBLIC":
            continue
        try:
            agent = exact_agent(episode, submission_id)
        except ValueError:
            continue
        candidates.append({**episode, "_seat": int(agent.get("index", 0) or 0)})
    candidates.sort(key=episode_time, reverse=True)
    if not candidates:
        raise RuntimeError(
            f"Top-{row['rank']} submission {submission_id} has no public episodes"
        )

    observed: Counter[str] = Counter()
    decks: dict[str, list[int]] = {}
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for episode in candidates[: max(replay_count * 3, replay_count)]:
        if len(records) >= replay_count:
            break
        episode_id = int(episode["id"])
        try:
            replay, raw = live.fetch_json(
                live.REPLAY_URL.format(episode_id=episode_id)
            )
            seat = int(episode["_seat"])
            names = live.base.episode_names(replay)
            if normalize_team_name(names[seat]) != normalize_team_name(
                str(row["team_name"])
            ):
                raise ValueError(
                    f"team mismatch at seat {seat}: {names[seat]!r}"
                )
            cards = sorted(live.deck_from_replay(replay, seat))
            deck_hash = live.deck_hash(cards)
            observed[deck_hash] += 1
            decks.setdefault(deck_hash, cards)
            records.append(
                {
                    "episode_id": episode_id,
                    "create_time": episode_time(episode),
                    "seat": seat,
                    "deck_hash": deck_hash,
                    "replay_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        except Exception as error:  # bounded fallback over recent episodes
            errors.append({"episode_id": episode_id, "error": str(error)})
        time.sleep(0.05)
    if not records:
        raise RuntimeError(
            f"No recent replay deck could be verified for {row['team_name']}"
        )
    winner_hash, winner_count = observed.most_common(1)[0]
    return {
        "deck_hash": winner_hash,
        "cards": decks[winner_hash],
        "observed_hash_counts": dict(observed),
        "selected_hash_count": winner_count,
        "verified_replays": records,
        "replay_errors": errors,
    }


def safe_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=23)
    parser.add_argument(
        "--seed-submission-id",
        action="append",
        type=int,
        default=[],
        help="Known public submission used to enter the episode matchmaking graph.",
    )
    parser.add_argument("--max-crawl-rounds", type=int, default=4)
    parser.add_argument("--verify-replays-per-team", type=int, default=3)
    parser.add_argument(
        "--listing-backend",
        choices=("kaggle_api", "public_endpoint"),
        default="kaggle_api",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.max_crawl_rounds < 1:
        raise ValueError("--top-k/--max-crawl-rounds must be positive")
    if args.verify_replays_per_team < 1:
        raise ValueError("--verify-replays-per-team must be positive")
    seeds = args.seed_submission_id or [55439076]
    rows = read_leaderboard(args.leaderboard_csv, args.top_k)
    output_dir = safe_output_dir(args.output_dir)
    deck_dir = output_dir / "decks"
    deck_dir.mkdir()

    active, listings, crawl = crawl_active_submissions(
        rows,
        seeds,
        args.max_crawl_rounds,
        args.listing_backend,
    )
    missing = [row for row in rows if int(row["team_id"]) not in active]
    if missing:
        raise RuntimeError(
            "Could not resolve current submissions for: "
            + ", ".join(str(row["team_name"]) for row in missing)
        )

    clean_csv = output_dir / "leaderboard_top23.csv"
    with clean_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("rank", "team_id", "team_name", "submission_date", "score"),
        )
        writer.writeheader()
        writer.writerows(rows)

    entries: list[dict[str, Any]] = []
    for row in rows:
        submission_id = active[int(row["team_id"])]
        listing = listings.get(submission_id)
        if listing is None:
            listing = listing_payload(submission_id, args.listing_backend)
            listings[submission_id] = listing
        verified = verified_current_deck(
            row,
            submission_id,
            listing,
            args.verify_replays_per_team,
        )
        deck_path = deck_dir / (
            f"rank{int(row['rank']):02d}_{int(row['team_id'])}_"
            f"{verified['deck_hash'][:12]}.csv"
        )
        deck_path.write_text(
            "".join(f"{card}\n" for card in verified.pop("cards")),
            encoding="utf-8",
        )
        entry = {
            **row,
            "active_submission_id": submission_id,
            "deck_path": str(deck_path),
            "deck_sha256": sha256_file(deck_path),
            "source": "current_public_submission_replay",
            **verified,
        }
        entries.append(entry)
        print(
            f"rank={row['rank']:02d} team={row['team_name']} "
            f"submission={submission_id} deck={entry['deck_hash'][:12]}",
            flush=True,
        )

    hash_frequency = Counter(entry["deck_hash"] for entry in entries)
    manifest = {
        "schema_version": "ptcg-current-top23-opponent-pool-v1",
        "competition": "pokemon-tcg-ai-battle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "top_k": args.top_k,
            "one_equal_weight_slot_per_rank": True,
            "duplicate_decks_preserve_rank_frequency": True,
            "discussion_signals_used": False,
            "notebooks_or_competitor_code_used": False,
        },
        "leaderboard_source": {
            "input": str(args.leaderboard_csv.resolve()),
            "frozen_csv": str(clean_csv),
            "sha256": sha256_file(clean_csv),
        },
        "crawl": crawl,
        "deck_frequency": dict(hash_frequency.most_common()),
        "unique_decks": len(hash_frequency),
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"complete entries={len(entries)} unique_decks={len(hash_frequency)} "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
