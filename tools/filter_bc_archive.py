#!/usr/bin/env python3
"""Create a small, sharded BC archive for a deck and/or demonstrator."""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import orjson


SPLITS = ("train", "valid", "test")


def normalize_team_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def episode_key(row: dict[str, Any]) -> tuple[str, str] | None:
    dataset_date = row.get("dataset_date")
    episode_id = row.get("episode_id")
    if dataset_date in (None, "") or episode_id in (None, ""):
        return None
    return str(dataset_date), str(episode_id)


def collect_excluded_episodes(
    source: zipfile.ZipFile,
    excluded_names: set[str],
) -> tuple[set[tuple[str, str]], Counter[str]]:
    """Return episode identities containing an excluded team in either seat."""

    excluded_episodes: set[tuple[str, str]] = set()
    hit_counts: Counter[str] = Counter()
    if not excluded_names:
        return excluded_episodes, hit_counts

    names = sorted(
        name for name in source.namelist() if name.endswith(".jsonl")
    )
    for name in names:
        with source.open(name) as handle:
            for line_number, line in enumerate(handle, 1):
                row = orjson.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{name}:{line_number}: row is not an object")
                hits = {
                    normalized
                    for field in ("team_name", "opponent_team_name")
                    if (
                        normalized := normalize_team_name(row.get(field))
                    ) in excluded_names
                }
                if not hits:
                    continue
                key = episode_key(row)
                if key is None:
                    raise ValueError(
                        f"{name}:{line_number}: excluded-team row has no stable "
                        "dataset_date/episode_id"
                    )
                excluded_episodes.add(key)
                hit_counts.update(hits)
    return excluded_episodes, hit_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--deck-hash",
        help=(
            "Keep rows with this exact deck hash. May be omitted when a "
            "demonstrator filter such as --team-name is supplied."
        ),
    )
    parser.add_argument("--team-name")
    parser.add_argument(
        "--include-team-name",
        action="append",
        default=[],
        help=(
            "Keep rows demonstrated by this exact team name (repeatable). "
            "Use --team-filter-splits=train to preserve unbiased valid/test."
        ),
    )
    parser.add_argument(
        "--team-filter-splits",
        choices=("all", "train", "train_valid"),
        default="all",
        help="Splits on which --include-team-name is applied.",
    )
    parser.add_argument(
        "--exclude-team-name",
        action="append",
        default=[],
        help=(
            "Drop the entire episode if either team_name or "
            "opponent_team_name matches this normalized name (repeatable)."
        ),
    )
    parser.add_argument(
        "--max-action-count",
        type=int,
        help=(
            "Drop rows whose action list is longer than this value. Use 16 "
            "to materialize a train_bc_orbit-compatible archive."
        ),
    )
    parser.add_argument(
        "--dataset-date",
        action="append",
        default=[],
        help=(
            "Keep only this ISO dataset date; repeat for multiple dates. "
            "Existing split labels are preserved."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=("all", "wins", "nonlosses"),
        default="all",
        help=(
            "Terminal-outcome filter. 'wins' keeps reward > 0, "
            "'nonlosses' keeps reward >= 0, and 'all' preserves every row."
        ),
    )
    parser.add_argument(
        "--reward-filter-splits",
        choices=("all", "train", "train_valid"),
        default="all",
        help=(
            "Splits on which --reward-mode is applied. Use 'train' for a "
            "winning-demonstration training view with unbiased validation "
            "and a pass-through sealed test split."
        ),
    )
    parser.add_argument("--rows-per-shard", type=int, default=25_000)
    parser.add_argument("--compress", action="store_true")
    args = parser.parse_args()

    if not args.deck_hash and not args.team_name and not args.include_team_name:
        parser.error(
            "at least one of --deck-hash, --team-name, or "
            "--include-team-name is required"
        )

    excluded_display_names: list[str] = []
    excluded_names: set[str] = set()
    for value in args.exclude_team_name:
        display = " ".join(unicodedata.normalize("NFKC", value).split())
        normalized = normalize_team_name(display)
        if not normalized:
            parser.error("--exclude-team-name must not be empty")
        if normalized not in excluded_names:
            excluded_names.add(normalized)
            excluded_display_names.append(display)
    included_team_names = tuple(dict.fromkeys(args.include_team_name))
    if any(not value for value in included_team_names):
        parser.error("--include-team-name must not be empty")
    included_team_name_set = set(included_team_names)
    if args.max_action_count is not None and args.max_action_count < 0:
        parser.error("--max-action-count must be non-negative")

    requested_dates = tuple(dict.fromkeys(args.dataset_date))
    for value in requested_dates:
        try:
            date.fromisoformat(value)
        except ValueError as error:
            parser.error(f"invalid --dataset-date {value!r}: {error}")
    requested_date_set = set(requested_dates)

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--input and --output must differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_suffix(args.output.suffix + ".partial")
    for candidate in (args.output, partial_output):
        if candidate.exists():
            raise FileExistsError(
                f"{candidate} already exists; refusing to overwrite"
            )
    compression = zipfile.ZIP_DEFLATED if args.compress else zipfile.ZIP_STORED
    counts: Counter[str] = Counter()
    shard_counts: Counter[str] = Counter()
    team_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    split_dates: dict[str, set[str]] = {split: set() for split in SPLITS}
    episode_ids_by_split: dict[str, set[str]] = {
        split: set() for split in SPLITS
    }
    source_episode_keys: set[tuple[str, str]] = set()

    with zipfile.ZipFile(args.input) as source:
        source_manifest = orjson.loads(source.read("manifest.json"))
        excluded_episode_keys, excluded_hit_counts = collect_excluded_episodes(
            source, excluded_names
        )
        with zipfile.ZipFile(
            partial_output,
            "w",
            compression=compression,
            allowZip64=True,
        ) as target:
            for split in SPLITS:
                names = [
                    name
                    for name in source.namelist()
                    if name.startswith(f"{split}/") and name.endswith(".jsonl")
                ]
                shard_index = 0
                shard_rows = 0
                shard = None
                try:
                    for name in names:
                        with source.open(name) as handle:
                            for line in handle:
                                row = orjson.loads(line)
                                key = episode_key(row)
                                if key is not None:
                                    source_episode_keys.add(key)
                                if key in excluded_episode_keys:
                                    counts["rows_filtered_by_team_exclusion"] += 1
                                    counts[
                                        f"{split}_rows_filtered_by_team_exclusion"
                                    ] += 1
                                    continue
                                if (
                                    args.deck_hash
                                    and row.get("deck_hash") != args.deck_hash
                                ):
                                    continue
                                if (
                                    args.team_name
                                    and row.get("team_name") != args.team_name
                                ):
                                    continue
                                team_filter_applies = (
                                    bool(included_team_name_set)
                                    and (
                                        args.team_filter_splits == "all"
                                        or args.team_filter_splits == split
                                        or (
                                            args.team_filter_splits
                                            == "train_valid"
                                            and split in ("train", "valid")
                                        )
                                    )
                                )
                                if (
                                    team_filter_applies
                                    and row.get("team_name")
                                    not in included_team_name_set
                                ):
                                    counts["rows_filtered_by_team_inclusion"] += 1
                                    counts[
                                        f"{split}_rows_filtered_by_team_inclusion"
                                    ] += 1
                                    continue
                                if args.max_action_count is not None:
                                    action = row.get("action")
                                    if (
                                        isinstance(action, list)
                                        and len(action) > args.max_action_count
                                    ):
                                        counts[
                                            "rows_filtered_by_action_count"
                                        ] += 1
                                        counts[
                                            f"{split}_rows_filtered_by_action_count"
                                        ] += 1
                                        continue
                                dataset_date = row.get("dataset_date")
                                if (
                                    requested_date_set
                                    and dataset_date not in requested_date_set
                                ):
                                    counts["rows_filtered_by_date"] += 1
                                    counts[
                                        f"{split}_rows_filtered_by_date"
                                    ] += 1
                                    continue
                                reward_filter_applies = (
                                    args.reward_mode != "all"
                                    and (
                                        args.reward_filter_splits == "all"
                                        or args.reward_filter_splits == split
                                        or (
                                            args.reward_filter_splits
                                            == "train_valid"
                                            and split in ("train", "valid")
                                        )
                                    )
                                )
                                if reward_filter_applies:
                                    try:
                                        terminal_reward = float(
                                            row["terminal_reward"]
                                        )
                                    except (KeyError, TypeError, ValueError):
                                        counts["rows_missing_terminal_reward"] += 1
                                        counts[
                                            f"{split}_rows_missing_terminal_reward"
                                        ] += 1
                                        continue
                                    keep_reward = (
                                        terminal_reward > 0.0
                                        if args.reward_mode == "wins"
                                        else terminal_reward >= 0.0
                                    )
                                    if not keep_reward:
                                        counts["rows_filtered_by_reward"] += 1
                                        counts[
                                            f"{split}_rows_filtered_by_reward"
                                        ] += 1
                                        continue
                                if shard is None or shard_rows >= args.rows_per_shard:
                                    if shard is not None:
                                        shard.close()
                                    output_name = (
                                        f"{split}/part-{shard_index:05d}.jsonl"
                                    )
                                    shard = target.open(
                                        output_name,
                                        "w",
                                        force_zip64=True,
                                    )
                                    shard_index += 1
                                    shard_rows = 0
                                shard.write(line)
                                shard_rows += 1
                                counts[split] += 1
                                episode_id = row.get("episode_id")
                                if episode_id not in (None, ""):
                                    dataset_date = str(row.get("dataset_date") or "")
                                    episode_ids_by_split[split].add(
                                        f"{dataset_date}:{episode_id}"
                                    )
                                if dataset_date not in (None, ""):
                                    date_counts[str(dataset_date)] += 1
                                    split_dates[split].add(str(dataset_date))
                                team_name = row.get("team_name")
                                if team_name not in (None, ""):
                                    team_counts[str(team_name)] += 1
                                select_context = row.get("select_context")
                                if select_context not in (None, ""):
                                    context_counts[str(select_context)] += 1
                                if counts[split] % 50_000 == 0:
                                    print(
                                        f"{split}: {counts[split]:,} rows",
                                        flush=True,
                                    )
                finally:
                    if shard is not None:
                        shard.close()
                shard_counts[split] = shard_index

            # Diagnostic counters share ``counts`` with the split counters, so
            # summing every value would incorrectly count rejected/malformed
            # rows as output decisions.
            filtered_decisions = sum(
                counts[split] for split in SPLITS
            )
            filtered_episode_ids = set().union(
                *episode_ids_by_split.values()
            )
            filtered_dates = sorted(date_counts)
            sources_by_date: dict[str, dict[str, object]] = {}
            for source_entry in source_manifest.get("sources", []):
                if not isinstance(source_entry, dict):
                    continue
                source_date = source_entry.get("date")
                if source_date in date_counts:
                    # Later entries win.  Merged archives append newer source
                    # material after older supplements for overlapping source
                    # declarations, while their output rows remain disjoint.
                    sources_by_date[str(source_date)] = source_entry
            filtered_sources = [
                sources_by_date[value]
                for value in filtered_dates
                if value in sources_by_date
            ]
            lineage_source_stats = dict(source_manifest.get("stats") or {})
            lineage_source_stats.setdefault(
                "episodes_scanned", len(source_episode_keys)
            )
            manifest = {
                **source_manifest,
                "hidden_information_policy": (
                    "Replay visualize may be used only to identify exact deck "
                    "hashes; visualize is excluded from every saved decision "
                    "and only agent-visible observation is saved."
                ),
                "sources": filtered_sources,
                "filtered_from": str(args.input.resolve()),
                "deck_hash_filter": args.deck_hash,
                "team_name_filter": args.team_name,
                "team_inclusion_filter": {
                    "display_names": list(included_team_names),
                    "splits": args.team_filter_splits,
                    "rows_filtered": counts[
                        "rows_filtered_by_team_inclusion"
                    ],
                    "rows_filtered_by_split": {
                        split: counts[
                            f"{split}_rows_filtered_by_team_inclusion"
                        ]
                        for split in SPLITS
                    },
                },
                "team_exclusions": {
                    "display_names": excluded_display_names,
                    "normalized_names": sorted(excluded_names),
                    "scope": "drop_entire_episode_if_either_seat_matches",
                    "episode_hits": len(excluded_episode_keys),
                    "seat_hits": sum(excluded_hit_counts.values()),
                    "unresolved_episode_hits": 0,
                    "unresolved_seat_hits": 0,
                    "hits_by_normalized_name": {
                        name: excluded_hit_counts[name]
                        for name in sorted(excluded_names)
                    },
                    "rows_filtered": counts[
                        "rows_filtered_by_team_exclusion"
                    ],
                    "rows_filtered_by_split": {
                        split: counts[
                            f"{split}_rows_filtered_by_team_exclusion"
                        ]
                        for split in SPLITS
                    },
                },
                "trainer_compatibility_filter": {
                    "max_action_count": args.max_action_count,
                    "rows_filtered_by_action_count": counts[
                        "rows_filtered_by_action_count"
                    ],
                    "rows_filtered_by_action_count_by_split": {
                        split: counts[
                            f"{split}_rows_filtered_by_action_count"
                        ]
                        for split in SPLITS
                    },
                },
                "terminal_reward_filter": {
                    "mode": args.reward_mode,
                    "splits": args.reward_filter_splits,
                    "predicate": {
                        "all": "unfiltered",
                        "wins": "terminal_reward > 0",
                        "nonlosses": "terminal_reward >= 0",
                    }[args.reward_mode],
                    "rows_filtered_by_reward": counts[
                        "rows_filtered_by_reward"
                    ],
                    "rows_missing_terminal_reward": counts[
                        "rows_missing_terminal_reward"
                    ],
                    "rows_filtered_by_reward_by_split": {
                        split: counts[f"{split}_rows_filtered_by_reward"]
                        for split in SPLITS
                    },
                    "rows_missing_terminal_reward_by_split": {
                        split: counts[
                            f"{split}_rows_missing_terminal_reward"
                        ]
                        for split in SPLITS
                    },
                },
                "split_decisions": {
                    split: counts[split]
                    for split in SPLITS
                },
                "shards": {
                    split: shard_counts[split]
                    for split in SPLITS
                },
                "stats": {
                    "decisions": filtered_decisions,
                    "episodes_in_output": len(filtered_episode_ids),
                    "episodes_scanned": len(source_episode_keys),
                    "input_archives": 1,
                },
                "split_episodes": {
                    split: len(episode_ids_by_split[split])
                    for split in ("train", "valid", "test")
                },
                "team_decisions": dict(team_counts.most_common()),
                "context_decisions": dict(context_counts.most_common()),
                "date_decisions": dict(sorted(date_counts.items())),
                **(
                    {
                        "dates": filtered_dates,
                        "split_policy": {
                            "mode": "preserved_split_date_filter",
                            "train_dates": sorted(split_dates["train"]),
                            "valid_dates": sorted(split_dates["valid"]),
                            "test_dates": sorted(split_dates["test"]),
                        },
                    }
                    if requested_date_set
                    else {}
                ),
                "dataset_date_filter": {
                    "requested_dates": list(requested_dates),
                    "observed_dates": filtered_dates,
                    "missing_requested_dates": sorted(
                        requested_date_set - set(filtered_dates)
                    ),
                    "rows_filtered_by_date": counts[
                        "rows_filtered_by_date"
                    ],
                    "rows_filtered_by_date_by_split": {
                        split: counts[f"{split}_rows_filtered_by_date"]
                        for split in SPLITS
                    },
                },
                "filter_lineage": {
                    "source_split_decisions": source_manifest.get(
                        "split_decisions"
                    ),
                    "source_shards": source_manifest.get("shards"),
                    "source_stats": lineage_source_stats,
                },
                "rows_per_shard": args.rows_per_shard,
                "compression": "deflated" if args.compress else "stored",
            }
            target.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            )

    os.replace(partial_output, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "counts": dict(counts),
                "bytes": args.output.stat().st_size,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
