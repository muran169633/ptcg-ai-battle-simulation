#!/usr/bin/env python3
"""Cluster recent Top-K deck lists and freeze representative proxy decks.

The inventory is produced by ``inventory_topk_recent_decks.py``.  Clustering is
deliberately based on the 60-card multiset rather than deck names: two lists are
treated as variants when at least ``--similarity-threshold`` of their cards
(including multiplicity) overlap.  Only lists used by multiple current Top-K
teams and seen near the end of the window are eligible for proxy selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KNOWN_ARCHETYPES = {
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": "Marnie Grimmsnarl/Froslass",
    "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf": "Alakazam control",
    "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc": "Mega Froslass/Mega Lopunny",
    "01501d644249c08144b169d9af115042260d89f0990e04b81c5aeadfcb7d7b84": "Mega Lopunny",
    "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8": "Mega Lucario",
    "e40278fd83d971c280b0fb9cd14d5e45cfe63a45c647741eb937440b4b19be34": "Teal Mask Ogerpon",
    "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725": "Dragapult ex",
    "747769779b60ad8db730b0b238414abb1e47936349cc72c83ff79c2ffacf31bc": "Mega Kangaskhan/Crustle",
    "c04e65de725b8bb51b25f5dc87ae1c56a01316177c501ee2cfc62b3c60ac5712": "Cynthia's Garchomp ex",
    "d6c573dd89bd1319494e25d56e66536c0b68c3db2cf4694f4a63f7423ee2f0bd": "Festival Lead Dipplin",
    "df6f7443719675711c2dcead3559087de996f3b8c43192d55ebfe6182e43cb7c": "Slowking/Mega Kangaskhan",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def multiset_overlap(left: list[int], right: list[int]) -> float:
    """Return intersection cardinality divided by the larger deck size."""
    lc, rc = Counter(left), Counter(right)
    denom = max(sum(lc.values()), sum(rc.values()))
    return sum((lc & rc).values()) / denom if denom else 1.0


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        lroot, rroot = self.find(left), self.find(right)
        if lroot != rroot:
            self.parent[rroot] = lroot


def card_summary(cards: list[int], catalog: dict[int, dict[str, str]]) -> dict[str, list[str]]:
    pokemon: list[str] = []
    energy: list[str] = []
    for card_id, count in sorted(Counter(cards).items()):
        row = catalog.get(card_id, {})
        name = row.get("Card Name", f"Card {card_id}")
        stage = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
        rendered = f"{name} x{count}"
        if "Energy" in stage:
            energy.append(rendered)
        elif row.get("Category") != "Trainer" and stage not in {
            "Item", "Supporter", "Stadium", "Pokémon Tool"
        }:
            pokemon.append(rendered)
    return {"pokemon": pokemon, "energy": energy}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--card-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--min-teams", type=int, default=2)
    parser.add_argument("--min-last-seen", default="2026-08-09")
    parser.add_argument("--proxy-count", type=int, default=8)
    parser.add_argument(
        "--learner-deck-hash",
        default="c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    with args.card_catalog.open(encoding="utf-8-sig", newline="") as handle:
        catalog = {int(row["Card ID"]): row for row in csv.DictReader(handle)}

    decks = [
        deck
        for deck in inventory["top_decks"]
        if deck["unique_topk_teams"] >= args.min_teams
        and deck["last_seen"] >= args.min_last_seen
    ]
    union_find = UnionFind(len(decks))
    for left in range(len(decks)):
        for right in range(left + 1, len(decks)):
            if multiset_overlap(decks[left]["cards"], decks[right]["cards"]) >= args.similarity_threshold:
                union_find.union(left, right)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, deck in enumerate(decks):
        grouped.setdefault(union_find.find(index), []).append(deck)

    clusters: list[dict[str, Any]] = []
    for members in grouped.values():
        members.sort(
            key=lambda row: (
                row["appearances"], row["unique_topk_teams"], -row["best_rank"], row["deck_hash"]
            ),
            reverse=True,
        )
        representative = members[0]
        cluster = {
            "archetype": KNOWN_ARCHETYPES.get(
                representative["deck_hash"], representative.get("known_label") or "unlabeled"
            ),
            "representative_deck_hash": representative["deck_hash"],
            "representative_appearances": representative["appearances"],
            "representative_unique_top100_teams": representative["unique_topk_teams"],
            "aggregate_appearances": sum(row["appearances"] for row in members),
            "variant_count": len(members),
            "best_rank": min(row["best_rank"] for row in members),
            "last_seen": max(row["last_seen"] for row in members),
            "representative_win_rate": representative["win_rate"],
            "cards": representative["cards"],
            "card_summary": card_summary(representative["cards"], catalog),
            "variants": [
                {
                    "deck_hash": row["deck_hash"],
                    "appearances": row["appearances"],
                    "unique_top100_teams": row["unique_topk_teams"],
                    "best_rank": row["best_rank"],
                    "last_seen": row["last_seen"],
                    "overlap_with_representative": multiset_overlap(
                        representative["cards"], row["cards"]
                    ),
                }
                for row in members
            ],
        }
        clusters.append(cluster)

    clusters.sort(
        key=lambda row: (
            row["aggregate_appearances"],
            row["representative_unique_top100_teams"],
            -row["best_rank"],
        ),
        reverse=True,
    )
    learner_cluster = next(
        (
            row
            for row in clusters
            if any(v["deck_hash"] == args.learner_deck_hash for v in row["variants"])
        ),
        None,
    )
    opponents = [
        row
        for row in clusters
        if not any(v["deck_hash"] == args.learner_deck_hash for v in row["variants"])
    ][: args.proxy_count]

    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "inventory": str(args.inventory.resolve()),
            "inventory_sha256": sha256(args.inventory),
            "card_catalog": str(args.card_catalog.resolve()),
            "card_catalog_sha256": sha256(args.card_catalog),
        },
        "selection": {
            "similarity_metric": "60-card multiset intersection / max deck size",
            "similarity_threshold": args.similarity_threshold,
            "minimum_current_top100_teams": args.min_teams,
            "minimum_last_seen": args.min_last_seen,
            "learner_deck_hash": args.learner_deck_hash,
            "proxy_count": args.proxy_count,
            "candidate_exact_decks": len(decks),
            "candidate_clusters": len(clusters),
        },
        "learner_cluster": learner_cluster,
        "selected_opponent_clusters": opponents,
        "all_eligible_clusters": clusters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "eligible_exact_decks": len(decks),
        "eligible_clusters": len(clusters),
        "selected_opponents": [
            {
                "archetype": row["archetype"],
                "deck_hash": row["representative_deck_hash"],
                "aggregate_appearances": row["aggregate_appearances"],
            }
            for row in opponents
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
