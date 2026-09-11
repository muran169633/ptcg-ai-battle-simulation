#!/usr/bin/env python3
"""Build a time-forward BC archive from one live Kaggle submission.

The Kaggle simulation leaderboard exposes public game history through
``EpisodeService/ListEpisodes`` and immutable ``replay.json`` files.  This
tool freezes that live surface, verifies that every retained replay belongs to
the requested submission and exact 60-card deck, then emits only player-visible
observations with the canonical next-step action alignment.

The oldest games are used for training and the newest games are held out.  The
replay-only ``visualize`` payload is used solely to verify the deck identity;
it is never copied into a BC row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import prepare_bc_week as base  # noqa: E402


LIST_EPISODES_URL = (
    "https://www.kaggle.com/api/i/"
    "competitions.EpisodeService/ListEpisodes"
)
REPLAY_URL = "https://www.kaggle.com/competitions/episodes/{episode_id}/replay.json"
USER_AGENT = "ptcg-live-bc-freezer/1.0"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str, payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], bytes]:
    data = None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return parsed, raw


def parse_timestamp(value: Any) -> datetime:
    text = str(value or "")
    if not text:
        raise ValueError("episode has no createTime")
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def exact_submission_agent(episode: dict[str, Any], submission_id: int) -> dict[str, Any]:
    agents = episode.get("agents")
    if not isinstance(agents, list):
        raise ValueError("episode agents must be a list")
    matches = [
        agent
        for agent in agents
        if isinstance(agent, dict) and int(agent.get("submissionId", -1)) == submission_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one agent for submission {submission_id}, got {len(matches)}"
        )
    return matches[0]


def deck_from_replay(replay: dict[str, Any], seat: int) -> list[int]:
    steps = replay.get("steps")
    if not isinstance(steps, list):
        raise ValueError("replay steps must be a list")
    for frame in steps[:3]:
        if not isinstance(frame, list):
            continue
        for entry in frame[:2]:
            if not isinstance(entry, dict):
                continue
            visualize = entry.get("visualize")
            if not isinstance(visualize, list):
                continue
            for item in visualize:
                decks = item.get("action") if isinstance(item, dict) else None
                if (
                    isinstance(decks, list)
                    and len(decks) >= 2
                    and all(isinstance(deck, list) and len(deck) == 60 for deck in decks[:2])
                ):
                    try:
                        return [int(card) for card in decks[seat]]
                    except (TypeError, ValueError) as error:
                        raise ValueError("deck contains a non-integer card id") from error
    raise ValueError("no complete two-player deck payload found in replay visualize")


def deck_hash(cards: list[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_counts(total: int, valid_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    if total < 5:
        raise ValueError("at least five exact-deck episodes are required")
    valid = max(1, math.floor(total * valid_fraction))
    test = max(1, math.floor(total * test_fraction))
    train = total - valid - test
    if train < 3:
        raise ValueError("split fractions leave fewer than three training episodes")
    return train, valid, test


def assign_splits(
    episodes: list[dict[str, Any]], valid_fraction: float, test_fraction: float
) -> dict[int, str]:
    ordered = sorted(episodes, key=lambda row: (parse_timestamp(row["createTime"]), int(row["id"])))
    train, valid, _ = split_counts(len(ordered), valid_fraction, test_fraction)
    result: dict[int, str] = {}
    for index, episode in enumerate(ordered):
        split = "train" if index < train else "valid" if index < train + valid else "test"
        result[int(episode["id"])] = split
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--expected-deck-hash")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--frames-per-shard", type=int, default=25_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.submission_id <= 0:
        raise ValueError("--submission-id must be positive")
    if not 0.0 < args.valid_fraction < 0.5:
        raise ValueError("--valid-fraction must be between zero and 0.5")
    if not 0.0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction must be between zero and 0.5")
    if args.valid_fraction + args.test_fraction >= 0.8:
        raise ValueError("validation and test fractions leave too little training data")

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    replay_dir = output_root / "replays"
    replay_dir.mkdir(parents=True)

    listing, listing_raw = fetch_json(
        LIST_EPISODES_URL,
        {
            "ids": [],
            "submissionId": args.submission_id,
            "successfulOnly": True,
            "includeInProgress": False,
        },
    )
    listing_path = output_root / "list_episodes.json"
    listing_path.write_bytes(listing_raw)
    raw_episodes = listing.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise RuntimeError("ListEpisodes returned no completed public episodes")

    episodes: list[dict[str, Any]] = []
    replays: dict[int, dict[str, Any]] = {}
    replay_records: list[dict[str, Any]] = []
    observed_decks: Counter[str] = Counter()
    canonical_decks: dict[str, list[int]] = {}
    failures: list[dict[str, Any]] = []
    excluded_validation_episodes: list[dict[str, Any]] = []

    for index, episode in enumerate(raw_episodes, 1):
        if not isinstance(episode, dict):
            failures.append({"reason": "episode_not_object"})
            continue
        if str(episode.get("type", "")).upper() == "EPISODE_TYPE_VALIDATION":
            excluded_validation_episodes.append(
                {
                    "episode_id": episode.get("id"),
                    "reason": "submission_self_play_validation",
                }
            )
            continue
        try:
            agent = exact_submission_agent(episode, args.submission_id)
            seat = int(agent.get("index", 0) or 0)
            if seat not in (0, 1):
                raise ValueError(f"invalid seat {seat}")
            episode_id = int(episode["id"])
            replay, replay_raw = fetch_json(REPLAY_URL.format(episode_id=episode_id))
            team_names = base.episode_names(replay)
            if team_names[seat] != args.team_name:
                raise ValueError(
                    f"seat/team mismatch: seat={seat}, replay={team_names}, expected={args.team_name!r}"
                )
            cards = deck_from_replay(replay, seat)
            observed_hash = deck_hash(cards)
            observed_decks[observed_hash] += 1
            canonical_decks.setdefault(observed_hash, sorted(cards))
            replay_path = replay_dir / f"{episode_id}.json"
            replay_path.write_bytes(replay_raw)
            episodes.append(episode)
            replays[episode_id] = replay
            replay_records.append(
                {
                    "episode_id": episode_id,
                    "create_time": episode.get("createTime"),
                    "end_time": episode.get("endTime"),
                    "seat": seat,
                    "reward": agent.get("reward"),
                    "deck_hash": observed_hash,
                    "replay": str(replay_path),
                    "replay_sha256": sha256_bytes(replay_raw),
                }
            )
        except Exception as error:
            failures.append(
                {
                    "episode_id": episode.get("id"),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
        if index % 20 == 0 or index == len(raw_episodes):
            print(
                f"downloaded={index}/{len(raw_episodes)} valid={len(episodes)} failures={len(failures)}",
                flush=True,
            )

    if failures:
        raise RuntimeError(
            f"failed closed because {len(failures)} listed episodes could not be frozen: "
            f"{failures[:3]}"
        )
    if len(observed_decks) != 1:
        raise RuntimeError(f"submission exposed multiple deck hashes: {dict(observed_decks)}")
    target_hash = next(iter(observed_decks))
    if args.expected_deck_hash and target_hash != args.expected_deck_hash:
        raise RuntimeError(
            f"live deck hash mismatch: expected {args.expected_deck_hash}, observed {target_hash}"
        )

    deck_path = output_root / "deck.csv"
    deck_path.write_text(
        "".join(f"{card}\n" for card in canonical_decks[target_hash]),
        encoding="utf-8",
    )
    split_by_episode = assign_splits(episodes, args.valid_fraction, args.test_fraction)

    archive_path = output_root / "bc_live_exact.zip"
    stats: Counter[str] = Counter()
    split_decisions: Counter[str] = Counter()
    split_episodes: Counter[str] = Counter()
    split_rewards: dict[str, Counter[str]] = {
        split: Counter() for split in ("train", "valid", "test")
    }
    team_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    team_filter = base.TeamFilter(
        all_teams=False,
        global_names={base.normalize_team_name(args.team_name)},
        names_by_date={},
        display_names=[args.team_name],
    )

    with base.DecisionArchiveWriter(
        archive_path, args.frames_per_shard, overwrite=False
    ) as writer:
        for episode in sorted(
            episodes, key=lambda row: (parse_timestamp(row["createTime"]), int(row["id"]))
        ):
            episode_id = int(episode["id"])
            split = split_by_episode[episode_id]
            agent = exact_submission_agent(episode, args.submission_id)
            reward = float(agent.get("reward", 0) or 0)
            outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
            split_episodes[split] += 1
            split_rewards[split][outcome] += 1
            replay = replays[episode_id]
            dataset_date = parse_timestamp(episode["createTime"]).date().isoformat()
            before = stats["decisions"]
            for row in base.iter_episode_decisions(
                replay,
                str(episode_id),
                dataset_date,
                team_filter,
                split,
                stats,
                team_counts,
                context_counts,
            ):
                if row.get("deck_hash") != target_hash:
                    raise RuntimeError(
                        f"row deck mismatch in episode {episode_id}: {row.get('deck_hash')}"
                    )
                writer.add(split, row)
                split_decisions[split] += 1
                stats["decisions"] += 1
            if stats["decisions"] == before:
                raise RuntimeError(f"episode {episode_id} emitted no legal BC decisions")

        split_time_ranges: dict[str, dict[str, str]] = {}
        for split in ("train", "valid", "test"):
            times = [
                parse_timestamp(episode["createTime"])
                for episode in episodes
                if split_by_episode[int(episode["id"])] == split
            ]
            split_time_ranges[split] = {
                "first_create_time": min(times).isoformat(),
                "last_create_time": max(times).isoformat(),
            }
        manifest = {
            "schema_version": base.SCHEMA_VERSION,
            "source_schema_version": "ptcg-live-submission-bc-freeze-v1",
            "competition": base.COMPETITION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "submission_id": args.submission_id,
            "team_name": args.team_name,
            "deck_hash": target_hash,
            "deck_path": str(deck_path),
            "deck_sha256": sha256_file(deck_path),
            "listed_episode_count": len(raw_episodes),
            "retained_episode_count": len(episodes),
            "excluded_validation_episode_count": len(excluded_validation_episodes),
            "excluded_validation_episodes": excluded_validation_episodes,
            "split_policy": {
                "mode": "episode_chronological",
                "oldest_to_train_newest_to_holdout": True,
                "valid_fraction": args.valid_fraction,
                "test_fraction": args.test_fraction,
                "time_ranges": split_time_ranges,
            },
            "label_alignment": "steps[t-1].observation -> steps[t].action",
            "hidden_information_policy": (
                "visualize used only for exact deck verification and excluded from rows"
            ),
            "loss_trajectory_weights": {
                "win": 1.0,
                "draw": 0.85,
                "loss_before_60_percent": 0.75,
                "loss_after_60_percent": 0.25,
            },
            "split_episodes": dict(split_episodes),
            "split_decisions": dict(split_decisions),
            "split_rewards": {
                split: dict(values) for split, values in split_rewards.items()
            },
            "stats": dict(stats),
            "team_decisions": dict(team_counts),
            "context_decisions": dict(context_counts),
            "shards": dict(writer.shard_index),
            "list_episodes": {
                "path": str(listing_path),
                "sha256": sha256_file(listing_path),
            },
            "replays": replay_records,
        }
        writer.finish(manifest)

    verified = base.verify_archive(archive_path)
    result = {
        "output_root": str(output_root),
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "deck": str(deck_path),
        "deck_hash": target_hash,
        "episodes": dict(split_episodes),
        "decisions": dict(split_decisions),
        "verified": verified,
    }
    result_path = output_root / "freeze_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
