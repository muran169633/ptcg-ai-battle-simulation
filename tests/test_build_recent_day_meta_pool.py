from __future__ import annotations

import json
import zipfile
from pathlib import Path

import orjson

from tools import build_recent_day_meta_pool as pool


def replay(deck_a: list[int], deck_b: list[int], names: list[str]) -> dict:
    return {
        "info": {"TeamNames": names},
        "steps": [
            [
                {"visualize": [{"action": [deck_a, deck_b]}]},
                {},
            ]
        ],
    }


def test_build_pool_uses_seat_frequency_and_normalizes_selected(tmp_path: Path) -> None:
    deck_a = list(range(60))
    deck_b = list(range(100, 160))
    deck_c = list(range(200, 260))
    archive = tmp_path / "episodes-2026-08-13.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("1.json", orjson.dumps(replay(deck_a, deck_b, ["A", "B"])))
        handle.writestr("2.json", orjson.dumps(replay(deck_a, deck_c, ["A", "C"])))
        handle.writestr("3.json", orjson.dumps(replay(deck_a, deck_b, ["D", "B"])))

    result = pool.build_pool(
        archive,
        tmp_path / "output",
        dataset_date="2026-08-13",
        top_n=2,
    )

    assert result["stats"]["seats_with_deck"] == 6
    assert result["stats"]["unique_exact_decks"] == 3
    assert [row["appearances"] for row in result["opponents"]] == [3, 2]
    assert result["selection"]["coverage"] == 5 / 6
    assert sum(
        row["pool_base_probability"] for row in result["opponents"]
    ) == 1.0
    assert result["opponents"][0]["top_teams"] == [
        {"team_name": "A", "appearances": 2},
        {"team_name": "D", "appearances": 1},
    ]
    generated = Path(result["opponents"][0]["deck_path"])
    assert [int(value) for value in generated.read_text().splitlines()] == deck_a
    persisted = json.loads((tmp_path / "output/meta_pool.json").read_text())
    assert persisted["source"]["sha256"] == pool.sha256_file(archive)
