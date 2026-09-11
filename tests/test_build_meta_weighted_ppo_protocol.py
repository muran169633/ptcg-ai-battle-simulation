from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import build_meta_weighted_ppo_protocol as protocol


def semantic_hash(cards: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(card) for card in sorted(cards)).encode()
    ).hexdigest()


def write_deck(path: Path, cards: list[int]) -> None:
    path.write_text("".join(f"{card}\n" for card in cards))


def test_protocol_maps_learner_to_bc_and_preserves_90_10_split(tmp_path: Path) -> None:
    learner = list(range(60))
    other = list(range(100, 160))
    learner_path = tmp_path / "learner.csv"
    other_path = tmp_path / "other.csv"
    write_deck(learner_path, learner)
    write_deck(other_path, other)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": "ptcg-recent-day-exact-deck-meta-pool-v1",
                "dataset_date": "2026-08-13",
                "selection": {"coverage": 1.0},
                "opponents": [
                    {
                        "rank": 1,
                        "deck_hash": semantic_hash(other),
                        "deck_path": str(other_path),
                        "deck_sha256": "unused",
                        "appearances": 3,
                        "pool_base_probability": 0.75,
                    },
                    {
                        "rank": 2,
                        "deck_hash": semantic_hash(learner),
                        "deck_path": str(learner_path),
                        "deck_sha256": "unused",
                        "appearances": 1,
                        "pool_base_probability": 0.25,
                    },
                ],
            }
        )
    )

    result = protocol.build_protocol(meta_path, learner_path, checkpoint)

    assert result["sampling"]["selfplay_probability"] == 0.1
    assert result["sampling"]["probability_sum"] == 1.0
    assert [row["role"] for row in result["bindings"]] == [
        "extra_recent_meta_deck",
        "base_frozen_same_deck",
    ]
    assert result["bindings"][1]["name"] == "bc"
    flags = result["train_ppo_argument_fragment"]
    assert flags.count("--extra-opponent") == 1
    assert flags.count("--opponent-meta-weight") == 2
