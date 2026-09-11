#!/usr/bin/env python3
"""Finalize retained Gold8 partial ZIPs after an interrupted/strict scan.

This recovery path does not rescan decision rows.  It validates the eight
closed ``*.zip.partial`` files, extracts the exact 60-card lists from the last
daily source, preserves complete time splits, and applies the same explicit
episode-hash fallback used by ``prepare_gold8_week.py`` to sparse profiles.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import prepare_bc_week as base
import prepare_gold8_week as gold


def scan_partial(path: Path, expected_hash: str) -> dict[str, Any]:
    split_decisions: Counter[str] = Counter()
    split_episodes = {
        split: set() for split in ("train", "valid", "test")
    }
    team_decisions: Counter[str] = Counter()
    context_decisions: Counter[str] = Counter()
    date_decisions: Counter[str] = Counter()
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"{path}: ZIP failure at {bad_member}")
        members = sorted(
            name for name in archive.namelist() if name.endswith(".jsonl")
        )
        if not members:
            raise RuntimeError(f"{path}: no JSONL decision members")
        for member in members:
            with archive.open(member) as rows:
                for line in rows:
                    row = json.loads(line)
                    if str(row.get("deck_hash") or "") != expected_hash:
                        raise ValueError(f"{path}: row deck hash drifted")
                    if "visualize" in row or "visualize" in row.get(
                        "observation", {}
                    ):
                        raise ValueError(f"{path}: hidden visualize leaked")
                    split = str(row.get("split") or "")
                    if split not in split_episodes:
                        raise ValueError(f"{path}: invalid split {split!r}")
                    episode_id = str(row.get("episode_id") or "")
                    split_decisions[split] += 1
                    split_episodes[split].add(episode_id)
                    team_decisions[str(row.get("team_name", ""))] += 1
                    context_decisions[
                        str(row.get("select_context", "unknown"))
                    ] += 1
                    date_decisions[str(row.get("dataset_date", ""))] += 1
    return {
        "split_decisions": split_decisions,
        "split_episodes": split_episodes,
        "team_decisions": team_decisions,
        "context_decisions": context_decisions,
        "date_decisions": date_decisions,
    }


def extract_decks(last_source: base.DailySource) -> dict[str, list[int]]:
    wanted = {profile.deck_hash for profile in gold.PROFILES}
    found: dict[str, list[int]] = {}
    scanned = 0
    for _, episode in base.iter_source_episodes(last_source):
        scanned += 1
        if episode is None:
            continue
        hashes, decks = gold.extract_replay_decks(episode)
        for deck_hash, deck in zip(hashes, decks):
            if deck_hash in wanted and deck is not None:
                found[deck_hash] = deck
        if found.keys() == wanted:
            break
        if scanned % 250 == 0:
            base.log(
                f"deck recovery scanned={scanned} found={len(found)}/{len(wanted)}"
            )
    missing = sorted(wanted - found.keys())
    if missing:
        raise RuntimeError(f"Last daily source lacks deck lists: {missing}")
    return found


def profile_manifest(
    profile: gold.DeckProfile,
    deck_path: Path,
    stats: dict[str, Any],
    sources: list[base.DailySource],
    top_names: list[str],
) -> dict[str, Any]:
    dates = [source.dataset_date for source in sources]
    return {
        "schema_version": base.SCHEMA_VERSION,
        "gold8_schema_version": gold.SCHEMA_VERSION,
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
            "global_team_count": len(top_names),
            "display_names": top_names,
        },
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
        "target_bc_exact_accuracy": 0.75,
        "stats": {
            "decisions": sum(stats["split_decisions"].values()),
            "episodes": sum(
                len(values) for values in stats["split_episodes"].values()
            ),
        },
        "split_decisions": dict(stats["split_decisions"]),
        "split_episodes": {
            split: len(values)
            for split, values in stats["split_episodes"].items()
        },
        "team_decisions": dict(stats["team_decisions"].most_common()),
        "context_decisions": dict(stats["context_decisions"].most_common()),
        "sources": [
            {
                "date": source.dataset_date,
                "kind": source.kind,
                "path": str(source.path.resolve()),
            }
            for source in sources
        ],
        "recovery": {
            "source": "closed single-pass .zip.partial",
            "decision_rows_rescanned": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=gold.DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=gold.DEFAULT_OUTPUT)
    parser.add_argument("--frames-per-shard", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    output_root = args.output_root.resolve()
    archives_dir = output_root / "archives"
    decks_dir = output_root / "decks"
    root_manifest_path = output_root / "manifest.json"
    if root_manifest_path.exists():
        raise FileExistsError(root_manifest_path)

    sources = base.discover_sources(
        [cache_dir / "daily"],
        [],
        days=7,
        date_end="2026-08-07",
    )
    if len(sources) != 7:
        raise RuntimeError(f"Expected seven daily sources, got {len(sources)}")
    leaderboard_rows = gold.newest_leaderboard_rows(cache_dir, 20)
    if len(leaderboard_rows) != 20:
        raise RuntimeError("Leaderboard recovery requires a frozen Top20 ZIP")
    top_names = [str(row["TeamName"]) for row in leaderboard_rows]

    partials: dict[str, Path] = {}
    stats_by_slug: dict[str, dict[str, Any]] = {}
    for profile in gold.PROFILES:
        archive_path = archives_dir / f"{profile.slug}.zip"
        partial = archive_path.with_suffix(".zip.partial")
        if archive_path.exists():
            raise FileExistsError(archive_path)
        if not partial.is_file():
            raise FileNotFoundError(partial)
        partials[profile.slug] = partial
        stats_by_slug[profile.slug] = scan_partial(partial, profile.deck_hash)
    decks = extract_decks(sources[-1])

    decks_dir.mkdir(parents=True, exist_ok=True)
    profile_results: dict[str, Any] = {}
    decisions_by_date: Counter[str] = Counter()
    for profile in gold.PROFILES:
        stats = stats_by_slug[profile.slug]
        decisions_by_date.update(stats["date_decisions"])
        deck_path = decks_dir / f"{profile.deck_hash}.csv"
        gold.atomic_write_text(
            deck_path,
            "".join(f"{card}\n" for card in decks[profile.deck_hash]),
        )
        manifest = profile_manifest(
            profile,
            deck_path,
            stats,
            sources,
            top_names,
        )
        archive_path = archives_dir / f"{profile.slug}.zip"
        partial = partials[profile.slug]
        missing_time = [
            split
            for split in ("train", "valid", "test")
            if stats["split_decisions"][split] <= 0
        ]
        if missing_time:
            os.replace(partial, archive_path)
            manifest = gold.resplit_archive_by_episode_hash(
                archive_path,
                manifest,
                args.frames_per_shard,
                split_seed=20260808,
                fallback_reason=missing_time,
            )
        else:
            with zipfile.ZipFile(
                partial,
                mode="a",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2).encode(
                        "utf-8"
                    ),
                )
            os.replace(partial, archive_path)
        verified = base.verify_archive(archive_path)
        profile_results[profile.slug] = {
            "label": profile.label,
            "primary": profile.primary,
            "deck_hash": profile.deck_hash,
            "deck": str(deck_path.resolve()),
            "deck_sha256": gold.file_sha256(deck_path),
            "archive": str(archive_path.resolve()),
            "archive_sha256": gold.file_sha256(archive_path),
            "split_decisions": verified["split_decisions"],
            "split_episodes": verified["split_episodes"],
            "team_decisions": verified["team_decisions"],
            "split_policy": verified["split_policy"],
        }

    dates = [source.dataset_date for source in sources]
    root_manifest = {
        "schema_version": gold.SCHEMA_VERSION,
        "created_at_utc": gold.utc_now(),
        "competition": base.COMPETITION,
        "date_window": {
            "dates": dates,
            "default_time_train": dates[:-2],
            "default_time_valid": dates[-2:-1],
            "default_time_test": dates[-1:],
        },
        "top_k": 20,
        "top_teams": top_names,
        "leaderboard_snapshot": leaderboard_rows,
        "profiles": profile_results,
        "global_stats": {
            "recovered_from_closed_single_pass_partials": True,
            "decisions_routed": sum(decisions_by_date.values()),
        },
        "decisions_by_source_date": dict(decisions_by_date),
        "source_inventory": gold.source_inventory(sources),
        "contracts": {
            "exact_deck_hash_routing": True,
            "one_pass_eight_outputs": True,
            "observation_action_alignment": (
                "steps[t-1].observation -> steps[t].action"
            ),
            "visualize_not_in_training_rows": True,
            "test_used_for_training": False,
            "sparse_profile_fallback_is_explicit": True,
            "recovered_decision_rows_without_source_rescan": True,
        },
    }
    gold.atomic_write_json(root_manifest_path, root_manifest)
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
