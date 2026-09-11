#!/usr/bin/env python3
"""Build frozen recent-14-day Top100 exact-deck proxy BC archives.

This is a thin audited adapter around ``prepare_gold8_week.build_gold8``.  It
uses the source paths and leaderboard snapshot already frozen by the inventory
pass, so it does not redownload data or silently switch leaderboard versions.
The learner archive is included in the same one-pass scan so a later special-BC
view can be selected without parsing the 10+ GB replay window again.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import prepare_bc_week as base
from prepare_gold8_week import DeckProfile, build_gold8


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "data/top100_recent14_20260811_v1/inventory.json"
DEFAULT_SELECTION = ROOT / "data/top100_recent14_20260811_v1/proxy_selection.json"
DEFAULT_OUTPUT = ROOT / "data/top100_proxy_recent14_20260811_v1"


SLUGS = {
    "Marnie Grimmsnarl/Froslass": "marnie",
    "Alakazam control": "alakazam_control",
    "Mega Froslass/Mega Lopunny": "mega_froslass_lopunny",
    "Mega Lopunny": "mega_lopunny",
    "Mega Lucario": "mega_lucario",
    "Teal Mask Ogerpon": "teal_mask_ogerpon",
    "Dragapult ex": "dragapult_ex",
    "Mega Kangaskhan/Crustle": "mega_kangaskhan_crustle_current",
    "Cynthia's Garchomp ex": "cynthias_garchomp_ex",
}


def slugify(label: str) -> str:
    known = SLUGS.get(label)
    if known:
        return known
    value = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    return value or "deck"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames-per-shard", type=int, default=50_000)
    parser.add_argument("--target-accuracy", type=float, default=0.77)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))

    sources = [
        base.DailySource(str(row["date"]), Path(row["path"]).resolve(), "zip")
        for row in inventory["sources"]
    ]
    if len(sources) != 14 or len({row.dataset_date for row in sources}) != 14:
        raise RuntimeError("Frozen inventory must contain exactly 14 distinct days")
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(source.path)

    leaderboard_rows = inventory["leaderboard"]["rows"]
    top_names = [str(row["team_name"]) for row in leaderboard_rows]
    if len(top_names) != 100 or len(set(top_names)) != 100:
        raise RuntimeError("Frozen leaderboard must contain exactly 100 unique teams")
    team_filter = base.TeamFilter(
        all_teams=False,
        global_names={base.normalize_team_name(name) for name in top_names},
        names_by_date={},
        display_names=top_names,
    )

    learner = selection["learner_cluster"]
    profiles = [
        DeckProfile(
            "marnie",
            str(learner["archetype"]),
            str(learner["representative_deck_hash"]),
            primary=True,
        )
    ]
    for cluster in selection["selected_opponent_clusters"]:
        label = str(cluster["archetype"])
        profiles.append(
            DeckProfile(
                slugify(label),
                label,
                str(cluster["representative_deck_hash"]),
            )
        )
    if len(profiles) != 9 or len({row.deck_hash for row in profiles}) != 9:
        raise RuntimeError("Expected one learner plus eight unique proxy profiles")

    result = build_gold8(
        sources=sources,
        output_root=args.output_root.resolve(),
        team_filter=team_filter,
        excluded_team_display_names=[],
        excluded_team_names=set(),
        leaderboard_rows=leaderboard_rows,
        frames_per_shard=args.frames_per_shard,
        overwrite=args.overwrite,
        max_episodes=None,
        profiles=tuple(profiles),
        target_accuracy=args.target_accuracy,
    )
    result["frozen_inputs"] = {
        "inventory": str(args.inventory.resolve()),
        "selection": str(args.selection.resolve()),
    }
    # The underlying builder already writes the complete root manifest.  Keep
    # stdout compact; callers can inspect that manifest for all hashes/counts.
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "dates": result["date_window"]["dates"],
        "profiles": {
            slug: {
                "deck_hash": row["deck_hash"],
                "split_decisions": row["split_decisions"],
                "split_episodes": row["split_episodes"],
            }
            for slug, row in result["profiles"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
