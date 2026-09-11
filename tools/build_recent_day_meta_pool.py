#!/usr/bin/env python3
"""Build an exact-deck opponent distribution from one public replay day.

The output is a deployment/training pool, not a BC decision archive. Replay
``visualize`` is inspected only for the two submitted 60-card lists. No hidden
in-game state is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import orjson


SCHEMA_VERSION = "ptcg-recent-day-exact-deck-meta-pool-v1"
NUMERIC_JSON_RE = re.compile(r"(?:^|/)(\d+)\.json$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deck_hash(cards: list[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode()).hexdigest()


def extract_decks(replay: Mapping[str, Any]) -> list[list[int] | None]:
    steps = replay.get("steps")
    if not isinstance(steps, list):
        return [None, None]
    for frame in steps[:3]:
        if not isinstance(frame, list):
            continue
        for entry in frame[:2]:
            if not isinstance(entry, Mapping):
                continue
            visualize = entry.get("visualize")
            if not isinstance(visualize, list):
                continue
            for item in visualize:
                action = item.get("action") if isinstance(item, Mapping) else None
                if (
                    not isinstance(action, list)
                    or len(action) < 2
                    or not all(
                        isinstance(value, list) and len(value) == 60
                        for value in action[:2]
                    )
                ):
                    continue
                decks: list[list[int] | None] = []
                for raw_deck in action[:2]:
                    try:
                        cards = sorted(int(card) for card in raw_deck)
                    except (TypeError, ValueError):
                        decks.append(None)
                    else:
                        decks.append(cards)
                return decks
    return [None, None]


def replay_team_names(replay: Mapping[str, Any]) -> list[str]:
    info = replay.get("info")
    info = info if isinstance(info, Mapping) else {}
    names = info.get("TeamNames")
    if isinstance(names, list) and len(names) >= 2:
        return [str(value or "").strip() for value in names[:2]]
    return ["", ""]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_pool(
    replay_zip: Path,
    output_dir: Path,
    *,
    dataset_date: str,
    top_n: int,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    replay_zip = replay_zip.resolve()
    output_dir = output_dir.resolve()
    if not replay_zip.is_file() or not zipfile.is_zipfile(replay_zip):
        raise FileNotFoundError(replay_zip)

    counts: Counter[str] = Counter()
    teams: dict[str, Counter[str]] = defaultdict(Counter)
    decks: dict[str, list[int]] = {}
    stats: Counter[str] = Counter()
    with zipfile.ZipFile(replay_zip) as archive:
        members = sorted(
            (
                info
                for info in archive.infolist()
                if not info.is_dir() and NUMERIC_JSON_RE.search(info.filename)
            ),
            key=lambda info: info.filename,
        )
        if max_episodes is not None:
            members = members[:max_episodes]
        stats["archive_episode_members"] = len(members)
        for index, info in enumerate(members, 1):
            stats["episodes_scanned"] += 1
            try:
                with archive.open(info) as handle:
                    replay = orjson.loads(handle.read())
            except (OSError, orjson.JSONDecodeError):
                stats["invalid_replays"] += 1
                continue
            if not isinstance(replay, Mapping):
                stats["invalid_replays"] += 1
                continue
            names = replay_team_names(replay)
            replay_decks = extract_decks(replay)
            if any(deck is not None for deck in replay_decks):
                stats["episodes_with_decks"] += 1
            else:
                stats["episodes_without_decks"] += 1
            for seat, cards in enumerate(replay_decks):
                stats["seat_opportunities"] += 1
                if cards is None:
                    stats["seats_without_deck"] += 1
                    continue
                digest = deck_hash(cards)
                previous = decks.get(digest)
                if previous is not None and previous != cards:
                    raise RuntimeError(f"Deck hash collision for {digest}")
                decks[digest] = cards
                counts[digest] += 1
                teams[digest][names[seat]] += 1
                stats["seats_with_deck"] += 1
            if index % 250 == 0:
                print(
                    f"episodes={index:,}/{len(members):,} "
                    f"deck_seats={stats['seats_with_deck']:,}",
                    flush=True,
                )

    ordered = sorted(counts, key=lambda digest: (-counts[digest], digest))
    selected = ordered[:top_n]
    selected_seats = sum(counts[digest] for digest in selected)
    all_seats = sum(counts.values())
    if not selected or selected_seats <= 0 or all_seats <= 0:
        raise RuntimeError("No valid replay deck seats were found")

    deck_dir = output_dir / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, Any]] = []
    for rank, digest in enumerate(selected, 1):
        deck_path = deck_dir / f"rank{rank:02d}_{digest[:12]}.csv"
        atomic_write_text(
            deck_path,
            "".join(f"{card}\n" for card in decks[digest]),
        )
        appearances = counts[digest]
        selected_rows.append(
            {
                "rank": rank,
                "deck_hash": digest,
                "appearances": appearances,
                "arena_share": appearances / all_seats,
                "pool_base_probability": appearances / selected_seats,
                "deck_path": str(deck_path),
                "deck_sha256": sha256_file(deck_path),
                "top_teams": [
                    {"team_name": name, "appearances": count}
                    for name, count in teams[digest].most_common(10)
                ],
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_date": dataset_date,
        "source": {
            "path": str(replay_zip),
            "sha256": sha256_file(replay_zip),
            "bytes": replay_zip.stat().st_size,
        },
        "selection": {
            "unit": "replay_seat_exact_deck",
            "order": "appearance_desc_then_deck_hash_asc",
            "top_n": top_n,
            "selected_appearances": selected_seats,
            "all_valid_deck_appearances": all_seats,
            "coverage": selected_seats / all_seats,
            "normalization": "within_selected_pool",
            "maximum_episodes": max_episodes,
        },
        "stats": {
            **dict(stats),
            "unique_exact_decks": len(counts),
        },
        "opponents": selected_rows,
        "unselected_exact_decks": [
            {"deck_hash": digest, "appearances": counts[digest]}
            for digest in ordered[top_n:]
        ],
        "information_policy": (
            "Replay visualize was used only to read submitted 60-card lists; "
            "no hidden in-game state or decision observation is retained."
        ),
    }
    atomic_write_text(
        output_dir / "meta_pool.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-date", required=True)
    parser.add_argument("--top-n", type=int, default=23)
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_pool(
        args.replay_zip,
        args.output_dir,
        dataset_date=args.dataset_date,
        top_n=args.top_n,
        max_episodes=args.max_episodes,
    )
    print(
        json.dumps(
            {
                "output": str((args.output_dir / "meta_pool.json").resolve()),
                "dataset_date": payload["dataset_date"],
                "opponents": len(payload["opponents"]),
                "coverage": payload["selection"]["coverage"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
