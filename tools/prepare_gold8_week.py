#!/usr/bin/env python3
"""Build selected exact-deck BC archives from a frozen recent Top-K window.

The tool reads each official episode once and routes visible decision rows to
one of the selected exact deck hashes.  Replay ``visualize`` is used only to identify
the two submitted 60-card lists; it is never copied into a training row.

The default profile set mirrors the 2026-08-08 Gold-zone archetypes, while
``--profile`` can restrict a run to a smaller portfolio.  Every output archive
uses a strict time split: the last available day is test, the preceding day is
validation, and all earlier days are training data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import prepare_bc_week as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data/episodes_cache/refresh_20260808_recent7_gold8"
DEFAULT_OUTPUT = ROOT / "data/gold8_recent7_20260808"
SCHEMA_VERSION = "ptcg-gold8-exact-deck-week-v1"


@dataclass(frozen=True)
class DeckProfile:
    slug: str
    label: str
    deck_hash: str
    primary: bool = False


PROFILES = (
    DeckProfile(
        "marnie",
        "Marnie Grimmsnarl/Froslass",
        "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af",
        primary=True,
    ),
    DeckProfile(
        "mega_froslass_lopunny",
        "Mega Froslass/Mega Lopunny",
        "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc",
    ),
    DeckProfile(
        "mega_kangaskhan_crustle",
        "Mega Kangaskhan/Crustle",
        "4bf59ca589c2d685d74e3535424c2dbe3c11389dffba59eddba4567be7e437df",
    ),
    DeckProfile(
        "mega_lucario",
        "Mega Lucario",
        "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8",
    ),
    DeckProfile(
        "alakazam_control",
        "Alakazam control",
        "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf",
    ),
    DeckProfile(
        "dragapult_ex",
        "Dragapult ex",
        "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725",
    ),
    DeckProfile(
        "hydrapple_ogerpon",
        "Hydrapple/Ogerpon",
        "0a6ca2ca3e72f1d6c0860cb653f16b05c4ba9b8d7774d5669c608e43f4c154f2",
    ),
    DeckProfile(
        "ns_zoroark_ex",
        "N's Zoroark ex",
        "b529ad012ed92882c28ff4cabc81ba4890546431d6e294ade2edca9c4ce8839e",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
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


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def extract_replay_decks(
    episode: dict[str, Any],
) -> tuple[list[str | None], list[list[int] | None]]:
    """Return canonical hashes and sorted card IDs for both seats."""

    steps = episode.get("steps")
    if not isinstance(steps, list):
        return [None, None], [None, None]
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
                action = item.get("action") if isinstance(item, dict) else None
                if not isinstance(action, list) or len(action) < 2:
                    continue
                if not all(
                    isinstance(deck, list) and len(deck) == 60
                    for deck in action[:2]
                ):
                    continue
                hashes: list[str | None] = []
                decks: list[list[int] | None] = []
                for raw_deck in action[:2]:
                    try:
                        deck = sorted(int(card) for card in raw_deck)
                    except (TypeError, ValueError):
                        hashes.append(None)
                        decks.append(None)
                        continue
                    canonical = ",".join(str(card) for card in deck)
                    hashes.append(hashlib.sha256(canonical.encode()).hexdigest())
                    decks.append(deck)
                return hashes, decks
    return [None, None], [None, None]


def newest_leaderboard_rows(cache_dir: Path, top_k: int) -> list[dict[str, str]]:
    archives = sorted(
        (cache_dir / "leaderboard").glob("*.zip"),
        key=lambda path: path.stat().st_mtime,
    )
    if not archives:
        return []
    with zipfile.ZipFile(archives[-1]) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not names:
            return []
        with archive.open(names[0]) as raw:
            rows = list(
                csv.DictReader(line.decode("utf-8-sig") for line in raw)
            )
    rows.sort(key=lambda row: int(row.get("Rank", 10**9)))
    return rows[:top_k]


def source_inventory(sources: list[base.DailySource]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for index, source in enumerate(sources, 1):
        base.log(
            f"hashing source [{index}/{len(sources)}] "
            f"{source.dataset_date}: {source.path}"
        )
        inventory.append(
            {
                "date": source.dataset_date,
                "kind": source.kind,
                "path": str(source.path.resolve()),
                "bytes": source.path.stat().st_size if source.path.is_file() else None,
                "sha256": file_sha256(source.path) if source.path.is_file() else None,
            }
        )
    return inventory


def resplit_archive_by_episode_hash(
    archive_path: Path,
    manifest: dict[str, Any],
    frames_per_shard: int,
    split_seed: int,
    fallback_reason: list[str],
) -> dict[str, Any]:
    """Replace an incomplete time split with a deterministic 80/10/10 split."""

    replacement = archive_path.with_name(f".{archive_path.name}.resplit.zip")
    if replacement.exists():
        replacement.unlink()
    split_counts: Counter[str] = Counter()
    split_episodes = {
        split: set() for split in ("train", "valid", "test")
    }
    team_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(
        dir=archive_path.parent,
        prefix=f".{archive_path.stem}.resplit.",
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        raw_paths = {
            split: temporary_root / f"{split}.jsonl"
            for split in ("train", "valid", "test")
        }
        handles = {
            split: path.open("wb") for split, path in raw_paths.items()
        }
        try:
            with zipfile.ZipFile(archive_path) as source:
                members = sorted(
                    name for name in source.namelist() if name.endswith(".jsonl")
                )
                for member in members:
                    with source.open(member) as rows:
                        for line in rows:
                            row = json.loads(line)
                            episode_id = str(row.get("episode_id") or "")
                            if not episode_id:
                                raise ValueError(
                                    f"{archive_path}: row has no episode_id"
                                )
                            split = base.split_for_episode(
                                episode_id,
                                str(row.get("dataset_date") or ""),
                                [],
                                "hash",
                                split_seed,
                            )
                            row["split"] = split
                            handles[split].write(base.json_bytes(row))
                            split_counts[split] += 1
                            split_episodes[split].add(episode_id)
                            team_counts[str(row.get("team_name", ""))] += 1
                            context_counts[
                                str(row.get("select_context", "unknown"))
                            ] += 1
        finally:
            for handle in handles.values():
                handle.close()

        missing = [
            split
            for split in ("train", "valid", "test")
            if split_counts[split] <= 0
        ]
        if missing:
            raise RuntimeError(
                f"{archive_path}: episode-hash fallback still lacks {missing}"
            )
        manifest["split_policy"] = {
            "mode": "episode_hash_fallback",
            "train_fraction": 0.8,
            "valid_fraction": 0.1,
            "test_fraction": 0.1,
            "seed": split_seed,
            "reason": fallback_reason,
            "evidence_note": (
                "This holdout is not comparable to the strict time holdout."
            ),
        }
        manifest["stats"] = {
            "decisions": sum(split_counts.values()),
            "episodes": sum(len(values) for values in split_episodes.values()),
        }
        manifest["split_decisions"] = dict(split_counts)
        manifest["split_episodes"] = {
            split: len(values) for split, values in split_episodes.items()
        }
        manifest["team_decisions"] = dict(team_counts.most_common())
        manifest["context_decisions"] = dict(context_counts.most_common())
        with base.DecisionArchiveWriter(
            replacement,
            frames_per_shard,
            overwrite=False,
        ) as writer:
            for split in ("train", "valid", "test"):
                with raw_paths[split].open("rb") as rows:
                    for line in rows:
                        writer.add(split, json.loads(line))
            manifest["shards"] = dict(writer.shard_index)
            writer.finish(manifest)
    os.replace(replacement, archive_path)
    return manifest


def build_gold8(
    sources: list[base.DailySource],
    output_root: Path,
    team_filter: base.TeamFilter,
    excluded_team_display_names: list[str],
    excluded_team_names: set[str],
    leaderboard_rows: list[dict[str, str]],
    frames_per_shard: int,
    overwrite: bool,
    max_episodes: int | None,
    profiles: tuple[DeckProfile, ...] = PROFILES,
    target_accuracy: float = 0.75,
) -> dict[str, Any]:
    dates = sorted({source.dataset_date for source in sources})
    if len(dates) < 3:
        raise ValueError("Gold8 time split requires at least three dates")
    if len(dates) != len(sources):
        raise ValueError("Gold8 expects exactly one source per selected date")

    if not profiles:
        raise ValueError("At least one deck profile must be selected")
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in (0, 1]")
    if len({profile.slug for profile in profiles}) != len(profiles):
        raise ValueError("Selected deck profile slugs must be unique")
    by_hash = {profile.deck_hash: profile for profile in profiles}
    if len(by_hash) != len(profiles):
        raise ValueError("Selected deck profile hashes must be unique")
    archives_dir = output_root / "archives"
    decks_dir = output_root / "decks"
    archives_dir.mkdir(parents=True, exist_ok=True)
    decks_dir.mkdir(parents=True, exist_ok=True)

    global_stats: Counter[str] = Counter()
    profile_splits = {profile.slug: Counter() for profile in profiles}
    profile_teams = {profile.slug: Counter() for profile in profiles}
    profile_contexts = {profile.slug: Counter() for profile in profiles}
    profile_episodes = {
        profile.slug: {split: set() for split in ("train", "valid", "test")}
        for profile in profiles
    }
    captured_decks: dict[str, list[int]] = {}
    seen_episode_ids: set[str] = set()
    source_rows: Counter[str] = Counter()

    archive_paths = {
        profile.slug: archives_dir / f"{profile.slug}.zip"
        for profile in profiles
    }
    with ExitStack() as stack:
        writers = {
            profile.slug: stack.enter_context(
                base.DecisionArchiveWriter(
                    archive_paths[profile.slug],
                    frames_per_shard,
                    overwrite,
                )
            )
            for profile in profiles
        }

        stop = False
        for source_index, source in enumerate(sources, 1):
            scanned = routed = 0
            base.log(
                f"[{source_index}/{len(sources)}] {source.dataset_date} "
                f"{source.kind}: {source.path}"
            )
            for fallback_id, episode in base.iter_source_episodes(source):
                if (
                    max_episodes is not None
                    and global_stats["episodes_scanned"] >= max_episodes
                ):
                    stop = True
                    break
                global_stats["episodes_scanned"] += 1
                scanned += 1
                if episode is None:
                    global_stats["invalid_json_files"] += 1
                    continue
                episode_id, _ = base.episode_identifiers(episode, fallback_id)
                dedupe_key = f"{source.dataset_date}:{episode_id}"
                if dedupe_key in seen_episode_ids:
                    global_stats["duplicate_episodes"] += 1
                    continue
                seen_episode_ids.add(dedupe_key)

                names = base.episode_names(episode)
                if any(
                    base.normalize_team_name(name) in excluded_team_names
                    for name in names
                ):
                    global_stats["episodes_excluded_by_team_name"] += 1
                    continue
                deck_hashes, decks = extract_replay_decks(episode)
                target_seats = [
                    seat
                    for seat in range(min(2, len(names)))
                    if team_filter.allows(source.dataset_date, names[seat])
                    and seat < len(deck_hashes)
                    and deck_hashes[seat] in by_hash
                ]
                if not target_seats:
                    global_stats["episodes_without_target_topk_deck"] += 1
                    continue
                for seat in target_seats:
                    deck_hash = deck_hashes[seat]
                    deck = decks[seat]
                    if deck_hash is None or deck is None:
                        continue
                    previous = captured_decks.get(deck_hash)
                    if previous is not None and previous != deck:
                        raise RuntimeError(
                            f"Deck hash collision or replay drift for {deck_hash}"
                        )
                    captured_decks[deck_hash] = deck

                target_names = [names[seat] for seat in target_seats]
                episode_filter = base.TeamFilter(
                    all_teams=False,
                    global_names={
                        base.normalize_team_name(name) for name in target_names
                    },
                    names_by_date={},
                    display_names=target_names,
                )
                split = base.split_for_episode(
                    episode_id,
                    source.dataset_date,
                    dates,
                    "time",
                    0,
                )
                episode_routed = False
                scratch_teams: Counter[str] = Counter()
                scratch_contexts: Counter[str] = Counter()
                for row in base.iter_episode_decisions(
                    episode,
                    fallback_id,
                    source.dataset_date,
                    episode_filter,
                    split,
                    global_stats,
                    scratch_teams,
                    scratch_contexts,
                ):
                    deck_hash = str(row.get("deck_hash") or "")
                    profile = by_hash.get(deck_hash)
                    if profile is None:
                        continue
                    writers[profile.slug].add(split, row)
                    profile_splits[profile.slug][split] += 1
                    profile_teams[profile.slug][str(row.get("team_name", ""))] += 1
                    profile_contexts[profile.slug][
                        str(row.get("select_context", "unknown"))
                    ] += 1
                    profile_episodes[profile.slug][split].add(episode_id)
                    global_stats["decisions_routed"] += 1
                    source_rows[source.dataset_date] += 1
                    routed += 1
                    episode_routed = True
                if episode_routed:
                    global_stats["episodes_routed"] += 1
                if scanned % 250 == 0:
                    base.log(f"  scanned={scanned} routed_decisions={routed}")
            base.log(f"  done scanned={scanned} routed_decisions={routed}")
            if stop:
                break

        incomplete_time_splits: dict[str, list[str]] = {}
        unusable: dict[str, list[str]] = {}
        for profile in profiles:
            missing = [
                split
                for split in ("train", "valid", "test")
                if profile_splits[profile.slug][split] <= 0
            ]
            if missing:
                incomplete_time_splits[profile.slug] = missing
            if profile.deck_hash not in captured_decks:
                unusable.setdefault(profile.slug, []).append("deck")
            if sum(profile_splits[profile.slug].values()) <= 0:
                unusable.setdefault(profile.slug, []).append("decisions")
        if unusable:
            raise RuntimeError(
                "One or more Gold8 profiles have no usable weekly data: "
                + json.dumps(unusable, sort_keys=True)
            )

        profile_results: dict[str, Any] = {}
        for profile in profiles:
            deck_path = decks_dir / f"{profile.deck_hash}.csv"
            atomic_write_text(
                deck_path,
                "".join(f"{card}\n" for card in captured_decks[profile.deck_hash]),
            )
            manifest = {
                "schema_version": base.SCHEMA_VERSION,
                "gold8_schema_version": SCHEMA_VERSION,
                "competition": base.COMPETITION,
                "profile": {
                    "slug": profile.slug,
                    "label": profile.label,
                    "primary": profile.primary,
                    "deck_hash": profile.deck_hash,
                    "deck": str(deck_path.resolve()),
                },
                "dates": dates,
                "split_policy": {
                    "mode": "time",
                    "train_dates": dates[:-2],
                    "valid_dates": dates[-2:-1],
                    "test_dates": dates[-1:],
                },
                "team_filter": {
                    "all_teams": False,
                    "global_team_count": len(team_filter.global_names),
                    "display_names": sorted(set(team_filter.display_names)),
                },
                "excluded_team_names": excluded_team_display_names,
                "excluded_team_policy": "drop_whole_episode_if_either_seat_matches",
                "label_alignment": "steps[t-1].observation -> steps[t].action",
                "hidden_information_policy": (
                    "Replay visualize is used only to identify exact deck hashes; "
                    "only agent-visible observation is saved."
                ),
                "loss_trajectory_weights": {
                    "win": 1.0,
                    "draw": 0.85,
                    "loss_before_60_percent": 0.75,
                    "loss_after_60_percent": 0.25,
                },
                "target_bc_exact_accuracy": target_accuracy,
                "stats": {
                    "decisions": sum(profile_splits[profile.slug].values()),
                    "episodes": sum(
                        len(values)
                        for values in profile_episodes[profile.slug].values()
                    ),
                },
                "split_decisions": dict(profile_splits[profile.slug]),
                "split_episodes": {
                    split: len(values)
                    for split, values in profile_episodes[profile.slug].items()
                },
                "team_decisions": dict(profile_teams[profile.slug].most_common()),
                "context_decisions": dict(
                    profile_contexts[profile.slug].most_common()
                ),
                "shards": dict(writers[profile.slug].shard_index),
                "sources": [
                    {
                        "date": source.dataset_date,
                        "kind": source.kind,
                        "path": str(source.path.resolve()),
                    }
                    for source in sources
                ],
            }
            writers[profile.slug].finish(manifest)
            fallback_reason = incomplete_time_splits.get(profile.slug)
            if fallback_reason:
                base.log(
                    f"{profile.slug}: time split lacks {fallback_reason}; "
                    "using deterministic episode-hash fallback"
                )
                manifest = resplit_archive_by_episode_hash(
                    archive_paths[profile.slug],
                    manifest,
                    frames_per_shard,
                    split_seed=20260808,
                    fallback_reason=fallback_reason,
                )
            verified = base.verify_archive(archive_paths[profile.slug])
            profile_results[profile.slug] = {
                "label": profile.label,
                "primary": profile.primary,
                "deck_hash": profile.deck_hash,
                "deck": str(deck_path.resolve()),
                "deck_sha256": file_sha256(deck_path),
                "archive": str(archive_paths[profile.slug].resolve()),
                "archive_sha256": file_sha256(archive_paths[profile.slug]),
                "split_decisions": verified["split_decisions"],
                "split_episodes": verified["split_episodes"],
                "team_decisions": verified["team_decisions"],
                "split_policy": verified["split_policy"],
            }

    root_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "competition": base.COMPETITION,
        "date_window": {
            "dates": dates,
            "default_time_train": dates[:-2],
            "default_time_valid": dates[-2:-1],
            "default_time_test": dates[-1:],
        },
        "top_k": len(team_filter.global_names),
        "top_teams": list(team_filter.display_names),
        "excluded_team_names": excluded_team_display_names,
        "excluded_team_policy": "drop_whole_episode_if_either_seat_matches",
        "selected_profiles": [profile.slug for profile in profiles],
        "target_bc_exact_accuracy": target_accuracy,
        "leaderboard_snapshot": leaderboard_rows,
        "profiles": profile_results,
        "global_stats": dict(global_stats),
        "decisions_by_source_date": dict(source_rows),
        "source_inventory": source_inventory(sources),
        "contracts": {
            "exact_deck_hash_routing": True,
            "one_pass_profile_outputs": len(profiles),
            "one_pass_eight_outputs": len(profiles) == len(PROFILES),
            "observation_action_alignment": (
                "steps[t-1].observation -> steps[t].action"
            ),
            "visualize_not_in_training_rows": True,
            "test_used_for_training": False,
            "sparse_profile_fallback_is_explicit": True,
        },
    }
    atomic_write_json(output_root / "manifest.json", root_manifest)
    return root_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--date-end", default="2026-08-07")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--profile",
        action="append",
        choices=tuple(profile.slug for profile in PROFILES),
        help="Exact-deck profile to build; repeat for multiple profiles",
    )
    parser.add_argument("--target-accuracy", type=float, default=0.75)
    parser.add_argument("--teams-file", type=Path)
    parser.add_argument(
        "--exclude-team-name",
        action="append",
        default=[],
        help=(
            "Drop any episode containing this normalized team name in either "
            "seat (repeatable)"
        ),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--frames-per-shard", type=int, default=50_000)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.days < 3:
        raise ValueError("A strict time split requires at least three days")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if not 0.0 < args.target_accuracy <= 1.0:
        raise ValueError("--target-accuracy must be in (0, 1]")
    cache_dir = args.cache_dir.resolve()
    output_root = args.output_root.resolve()

    if args.download:
        sources = base.download_recent_sources(
            cache_dir,
            args.days,
            args.date_end,
            args.refresh,
        )
    else:
        sources = base.discover_sources(
            [cache_dir / "daily"],
            [],
            args.days,
            args.date_end,
        )
    if len(sources) != args.days:
        raise RuntimeError(
            f"Expected {args.days} daily sources, discovered {len(sources)}"
        )

    if args.teams_file is not None:
        team_filter = base.parse_team_file(args.teams_file)
        leaderboard_rows: list[dict[str, str]] = []
    else:
        team_filter = base.fetch_current_top_teams(cache_dir, args.top_k)
        leaderboard_rows = newest_leaderboard_rows(cache_dir, args.top_k)
    if len(team_filter.global_names) != args.top_k:
        raise RuntimeError(
            f"Expected exactly Top {args.top_k}, got {len(team_filter.global_names)}"
        )

    profile_by_slug = {profile.slug: profile for profile in PROFILES}
    selected_profiles = (
        tuple(profile_by_slug[slug] for slug in dict.fromkeys(args.profile))
        if args.profile
        else PROFILES
    )
    excluded_team_display_names, excluded_team_names = (
        base.prepare_excluded_team_names(args.exclude_team_name)
    )
    result = build_gold8(
        sources=sources,
        output_root=output_root,
        team_filter=team_filter,
        excluded_team_display_names=excluded_team_display_names,
        excluded_team_names=excluded_team_names,
        leaderboard_rows=leaderboard_rows,
        frames_per_shard=args.frames_per_shard,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        profiles=selected_profiles,
        target_accuracy=args.target_accuracy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
