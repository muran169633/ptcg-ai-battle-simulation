#!/usr/bin/env python3
"""Build the update90 online-failure targeted PPO opponent pool.

The resulting fixed stream has these marginal shares over all games:
45% two observed online-loss Dragapult variants, 5% frozen update90 mirror,
25% the remaining 2026-08-13 Top23 environment, and 15% split equally
between Marnie and Garchomp.  The other 10% of games is current-policy
self-play and is configured by the PPO launcher, not this manifest.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/recent_day_meta_pool_20260813_top23_v1/meta_pool.json"
OUT_DIR = ROOT / "data/update90_targeted_online_20260817_v1"
DECK_DIR = OUT_DIR / "decks"
OUT = OUT_DIR / "meta_pool.json"

LEARNER_HASH = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
MARNIE_HASH = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
CF11_HASH = "cf11ddaf76048f8734b2236115d44f3ed8f87d022bb7b881764f023026276f98"
B306_HASH = "b306db817db4e89bb1eb02a0a9d4d67071aa625c96be7d599c0cee3f639acdee"
GARCHOMP_HASH = "c04e65de725b8bb51b25f5dc87ae1c56a01316177c501ee2cfc62b3c60ac5712"


def deck_hash(path: Path) -> str:
    cards = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(cards)}")
    payload = ",".join(map(str, sorted(cards))).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads(SOURCE.read_text())
    source_rows = source["opponents"]
    by_hash = {row["deck_hash"]: row for row in source_rows}
    for required in (LEARNER_HASH, MARNIE_HASH):
        if required not in by_hash:
            raise ValueError(f"required source deck missing: {required}")

    # Pool values are conditional on the 90% frozen-opponent branch.
    weights = {
        CF11_HASH: 0.25,
        B306_HASH: 0.25,
        LEARNER_HASH: 1.0 / 18.0,
        MARNIE_HASH: 1.0 / 12.0,
        GARCHOMP_HASH: 1.0 / 12.0,
    }
    excluded = {LEARNER_HASH, MARNIE_HASH}
    broad = [row for row in source_rows if row["deck_hash"] not in excluded]
    broad_source_total = sum(float(row["pool_base_probability"]) for row in broad)
    broad_target_total = 5.0 / 18.0

    rows = []
    for row in broad:
        new = dict(row)
        new["pool_base_probability"] = (
            float(row["pool_base_probability"]) / broad_source_total * broad_target_total
        )
        rows.append(new)

    for deck_hash_value, rank, label, path in (
        (CF11_HASH, 1001, "online_loss_93760760", DECK_DIR / "drag_variant_cf11ddaf7604.csv"),
        (B306_HASH, 1002, "online_loss_93761001", DECK_DIR / "drag_variant_b306db817db4.csv"),
        (GARCHOMP_HASH, 1003, "online_loss_93763591", DECK_DIR / "garchomp_c04e65de725b.csv"),
    ):
        actual = deck_hash(path)
        if actual != deck_hash_value:
            raise ValueError(f"{label}: hash mismatch {actual} != {deck_hash_value}")
        rows.append(
            {
                "rank": rank,
                "label": label,
                "appearances": 0,
                "arena_share": 0.0,
                "deck_hash": deck_hash_value,
                "deck_path": str(path.resolve()),
                "deck_sha256": sha256(path),
                "pool_base_probability": weights[deck_hash_value],
            }
        )

    for source_hash, label, weight in (
        (LEARNER_HASH, "frozen_update90_mirror", weights[LEARNER_HASH]),
        (MARNIE_HASH, "marnie_exact", weights[MARNIE_HASH]),
    ):
        new = dict(by_hash[source_hash])
        new["label"] = label
        new["pool_base_probability"] = weight
        rows.append(new)

    total = sum(float(row["pool_base_probability"]) for row in rows)
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f"probabilities sum to {total}, not 1")
    hashes = [row["deck_hash"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate semantic deck hashes")
    if hashes.count(LEARNER_HASH) != 1:
        raise ValueError("learner deck must occur exactly once")

    payload = {
        "schema_version": "ptcg-recent-day-exact-deck-meta-pool-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_date": "2026-08-13",
        "source_meta_pool": str(SOURCE.resolve()),
        "design": {
            "league_probability": 0.90,
            "current_policy_selfplay_probability": 0.10,
            "marginal_online_loss_dragapult_variants": 0.45,
            "marginal_frozen_update90_mirror": 0.05,
            "marginal_top23_broad_excluding_learner_and_marnie": 0.25,
            "marginal_marnie": 0.075,
            "marginal_garchomp": 0.075,
            "fixed_meta_probability": 0.80,
            "inverse_meta_probability": 0.10,
        },
        "opponents": rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "opponents": len(rows), "probability_sum": total}, indent=2))


if __name__ == "__main__":
    main()
