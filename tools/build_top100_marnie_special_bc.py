#!/usr/bin/env python3
"""Derive a bounded, recent, winning-only Top100 Marnie special-BC archive.

The source archive must already contain only current-Top100 demonstrations for
the exact learner deck with a strict time split.  Training episodes are capped
per demonstrator rank band to prevent a few very active teams from dominating:
Top20=24 wins, ranks 21-50=16 wins, ranks 51-100=8 wins.  Within each team the
most recent wins are retained, with a stable SHA-256 tie-break.  Validation and
test retain every winning episode from their sealed day.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import orjson

import prepare_bc_week as base
from prepare_gold8_week import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/top100_proxy_recent14_20260811_v1/archives/marnie.zip"
DEFAULT_INVENTORY = ROOT / "data/top100_recent14_20260811_v1/inventory.json"
DEFAULT_OUTPUT = ROOT / "data/top100_marnie_special_recent14_20260811_v1/marnie_top100_rankcapped_wins.zip"


def cap_for_rank(rank: int) -> int:
    if rank <= 20:
        return 24
    if rank <= 50:
        return 16
    return 8


def episode_key(row: dict[str, Any]) -> str:
    return f"{row.get('dataset_date', '')}:{row.get('episode_id', '')}:{row.get('seat', '')}"


def stable_tiebreak(key: str) -> str:
    return hashlib.sha256(f"top100-special-bc-v1:{key}".encode()).hexdigest()


def jsonl_members(archive: zipfile.ZipFile) -> list[str]:
    return sorted(name for name in archive.namelist() if name.endswith(".jsonl"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames-per-shard", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    inventory_path = args.inventory.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    ranks = {
        base.normalize_team_name(str(row["team_name"])): int(row["rank"])
        for row in inventory["leaderboard"]["rows"]
    }
    if len(ranks) != 100:
        raise RuntimeError("Inventory leaderboard is not an exact Top100 snapshot")

    episode_meta: dict[str, dict[str, Any]] = {}
    source_rows = Counter()
    with zipfile.ZipFile(source_path) as source:
        source_manifest = orjson.loads(source.read("manifest.json"))
        for member in jsonl_members(source):
            with source.open(member) as rows:
                for line in rows:
                    row = orjson.loads(line)
                    split = str(row.get("split") or "")
                    source_rows[split] += 1
                    if float(row.get("terminal_reward", 0.0)) <= 0.0:
                        continue
                    key = episode_key(row)
                    if key in episode_meta:
                        continue
                    normalized = base.normalize_team_name(str(row.get("team_name") or ""))
                    rank = ranks.get(normalized)
                    if rank is None:
                        raise RuntimeError(f"Source row team is outside frozen Top100: {row.get('team_name')!r}")
                    episode_meta[key] = {
                        "key": key,
                        "split": split,
                        "date": str(row.get("dataset_date") or ""),
                        "team_name": str(row.get("team_name") or ""),
                        "rank": rank,
                        "tie": stable_tiebreak(key),
                    }

    train_by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected: set[str] = set()
    for meta in episode_meta.values():
        if meta["split"] == "train":
            train_by_team[meta["team_name"]].append(meta)
        elif meta["split"] in {"valid", "test"}:
            selected.add(meta["key"])
        else:
            raise RuntimeError(f"Unexpected archive split: {meta['split']!r}")
    train_selection: dict[str, dict[str, Any]] = {}
    for team_name, candidates in train_by_team.items():
        rank = int(candidates[0]["rank"])
        candidates.sort(key=lambda row: (row["date"], row["tie"]), reverse=True)
        retained = candidates[: cap_for_rank(rank)]
        selected.update(row["key"] for row in retained)
        train_selection[team_name] = {
            "rank": rank,
            "cap": cap_for_rank(rank),
            "winning_episode_candidates": len(candidates),
            "selected_episodes": len(retained),
        }

    split_rows = Counter()
    split_episodes: dict[str, set[str]] = defaultdict(set)
    team_rows = Counter()
    date_rows = Counter()
    context_rows = Counter()
    with zipfile.ZipFile(source_path) as source:
        with base.DecisionArchiveWriter(output_path, args.frames_per_shard, args.overwrite) as writer:
            for member in jsonl_members(source):
                with source.open(member) as rows:
                    for line in rows:
                        row = orjson.loads(line)
                        key = episode_key(row)
                        if key not in selected:
                            continue
                        if float(row.get("terminal_reward", 0.0)) <= 0.0:
                            raise RuntimeError("Selected episode contains a non-winning row")
                        split = str(row["split"])
                        writer.add(split, row)
                        split_rows[split] += 1
                        split_episodes[split].add(key)
                        team_rows[str(row.get("team_name") or "")] += 1
                        date_rows[str(row.get("dataset_date") or "")] += 1
                        context_rows[str(row.get("select_context") or "unknown")] += 1
            missing = [name for name in ("train", "valid", "test") if not split_rows[name]]
            if missing:
                raise RuntimeError(f"Special-BC archive has empty splits: {missing}")
            manifest = {
                "schema_version": base.SCHEMA_VERSION,
                "special_bc_schema_version": "ptcg-top100-rankcapped-winning-marnie-v1",
                "competition": source_manifest.get("competition"),
                "profile": source_manifest.get("profile"),
                "dates": source_manifest.get("dates"),
                "split_policy": source_manifest.get("split_policy"),
                "team_filter": source_manifest.get("team_filter"),
                "label_alignment": source_manifest.get("label_alignment"),
                "hidden_information_policy": source_manifest.get("hidden_information_policy"),
                "target_bc_exact_accuracy": 0.77,
                "selection_policy": {
                    "terminal_reward": "strictly_positive_only",
                    "train_rank_band_episode_caps": {"1-20": 24, "21-50": 16, "51-100": 8},
                    "within_team_priority": "most_recent_then_sha256_descending",
                    "valid_test_policy": "all_winning_episodes_on_sealed_split_days",
                    "opponent_model_outputs_used": False,
                },
                "stats": {
                    "decisions": sum(split_rows.values()),
                    "episodes": sum(len(values) for values in split_episodes.values()),
                },
                "split_decisions": dict(split_rows),
                "split_episodes": {name: len(values) for name, values in split_episodes.items()},
                "team_decisions": dict(team_rows.most_common()),
                "date_decisions": dict(sorted(date_rows.items())),
                "context_decisions": dict(context_rows.most_common()),
                "train_team_selection": dict(sorted(train_selection.items())),
                "source_archive": {"path": str(source_path), "sha256": file_sha256(source_path)},
                "inventory": {"path": str(inventory_path), "sha256": file_sha256(inventory_path)},
                "source_split_decisions": dict(source_rows),
                "shards": dict(writer.shard_index),
                "contracts": {
                    "test_used_for_training": False,
                    "only_agent_visible_observations": True,
                    "whole_episode_selection": True,
                    "current_top100_snapshot_filter": True,
                },
            }
            writer.finish(manifest)

    verified = base.verify_archive(output_path)
    print(json.dumps({
        "output": str(output_path),
        "sha256": file_sha256(output_path),
        "split_decisions": verified["split_decisions"],
        "split_episodes": verified["split_episodes"],
        "train_teams": len(train_selection),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
