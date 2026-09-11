from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
TOOL = TOOLS / "select_gold_routes.py"
sys.path.insert(0, str(TOOLS))

import select_gold_routes as selector  # noqa: E402


MARNIE = "1" * 64
ALAKAZAM = "2" * 64
ROUTE_2 = "3" * 64
POOL_1 = "4" * 64
POOL_2 = "5" * 64
POOL_3 = "6" * 64
POOL_4 = "7" * 64
ROUTE_3 = "8" * 64
OUTSIDE_TOP_5 = "9" * 64
TOO_FEW_VALID = "a" * 64
ONE_TEAM = "b" * 64


def preregistration() -> dict:
    return {
        "schema_version": "ptcg-gold-push-preregistration-v1",
        "data_protocol": {
            "train_dates": ["2026-08-02"],
            "valid_dates": ["2026-08-07"],
        },
        "route_selection_before_opening_test": {
            "marnie_deck_hash": MARNIE,
            "alakazam_deck_hash": ALAKAZAM,
            "excluded_deck_hashes": [ALAKAZAM],
            "minimum_valid_episode_seats": 100,
            "minimum_train_valid_teams": 2,
            "frequency_metric": "train_plus_valid_episode_seats",
            "route_3_pool_size": 5,
            "wilson_confidence": 0.95,
            "wilson_z": 1.959963984540054,
            "draws_in_wilson_denominator": True,
            "maximum_bc_routes": 3,
            "test_data_must_not_select_hyperparameters": True,
        },
    }


def split(
    episodes: int,
    wins: int,
    teams: dict[str, int],
    dataset_date: str,
) -> dict:
    return {
        "episodes": episodes,
        "wins": wins,
        "losses": episodes - wins,
        "draws": 0,
        "win_rate": wins / episodes if episodes else None,
        "team_count": len(teams),
        "team_counts": teams,
        "date_counts": {dataset_date: episodes} if episodes else {},
    }


def deck(
    deck_hash: str,
    *,
    train_episodes: int,
    valid_episodes: int,
    valid_wins: int,
    teams: tuple[str, str] = ("Train Team", "Valid Team"),
    test_payload: object = "SEALED_TEST_DO_NOT_READ",
) -> dict:
    train_teams = {teams[0]: train_episodes} if train_episodes else {}
    valid_teams = {teams[1]: valid_episodes} if valid_episodes else {}
    return {
        "deck_hash": deck_hash,
        "episodes": "PROFILE_WIDE_TOTAL_MUST_NOT_BE_READ",
        "wins": "PROFILE_WIDE_TOTAL_MUST_NOT_BE_READ",
        "split_stats": {
            "train": split(
                train_episodes, 0, train_teams, "2026-08-02"
            ),
            "valid": split(
                valid_episodes, valid_wins, valid_teams, "2026-08-07"
            ),
            "test": test_payload,
        },
    }


def profile() -> dict:
    decks = [
        # Fixed/excluded routes deliberately have unusable sealed split data;
        # selection needs only their hashes.
        {"deck_hash": MARNIE, "split_stats": "FIXED_ROUTE_NOT_RANKED"},
        {"deck_hash": ALAKAZAM, "split_stats": "EXCLUDED_NOT_RANKED"},
        deck(
            ROUTE_2,
            train_episodes=200,
            valid_episodes=150,
            valid_wins=75,
        ),
        deck(
            POOL_1,
            train_episodes=190,
            valid_episodes=150,
            valid_wins=105,
        ),
        deck(
            POOL_2,
            train_episodes=180,
            valid_episodes=150,
            valid_wins=120,
        ),
        deck(
            POOL_3,
            train_episodes=170,
            valid_episodes=150,
            valid_wins=90,
        ),
        deck(
            POOL_4,
            train_episodes=160,
            valid_episodes=150,
            valid_wins=115,
        ),
        deck(
            ROUTE_3,
            train_episodes=150,
            valid_episodes=150,
            valid_wins=130,
        ),
        # Better Wilson score, but sixth by frequency after route 2.
        deck(
            OUTSIDE_TOP_5,
            train_episodes=140,
            valid_episodes=150,
            valid_wins=145,
        ),
        deck(
            TOO_FEW_VALID,
            train_episodes=1000,
            valid_episodes=99,
            valid_wins=99,
        ),
        deck(
            ONE_TEAM,
            train_episodes=500,
            valid_episodes=150,
            valid_wins=150,
            teams=("Only Team", "Only Team"),
        ),
    ]
    return {
        "schema_version": "ptcg-bc-exact-deck-profile-v1",
        "deduplication_key": ["dataset_date", "episode_id", "seat"],
        "profiled_splits": ["train", "valid"],
        "sealed_splits": ["test"],
        "split_date_counts": {
            "train": {"2026-08-02": 1},
            "valid": {"2026-08-07": 1},
            "test": "SEALED_TEST_DO_NOT_READ",
        },
        "statistics": "PROFILE_WIDE_TOTALS_MUST_NOT_BE_READ",
        "decks": decks,
    }


class SealedSplitStats(dict):
    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("selector accessed sealed test")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "test":
            raise AssertionError("selector accessed sealed test")
        return super().get(key, default)


def test_selects_preregistered_routes_without_accessing_test() -> None:
    source_profile = profile()
    for item in source_profile["decks"]:
        raw_splits = item.get("split_stats")
        if isinstance(raw_splits, dict):
            item["split_stats"] = SealedSplitStats(raw_splits)

    report = selector.select_routes(preregistration(), source_profile)

    assert [item["deck_hash"] for item in report["selected_routes"]] == [
        MARNIE,
        ROUTE_2,
        ROUTE_3,
    ]
    assert report["eligible_candidate_count"] == 7
    assert [item["deck_hash"] for item in report["route_3_pool"]] == [
        POOL_1,
        POOL_2,
        POOL_3,
        POOL_4,
        ROUTE_3,
    ]
    assert report["test_data_firewall"]["profile_test_fields_accessed"] is False
    assert "split_stats.test" not in json.dumps(
        report["selection_fields_used"], sort_keys=True
    )
    assert report["selection_policy"]["frequency_tie_break"] == (
        "deck_hash_ascending"
    )


def test_test_payload_changes_cannot_change_selection() -> None:
    first = profile()
    second = copy.deepcopy(first)
    for index, item in enumerate(second["decks"]):
        raw_splits = item.get("split_stats")
        if isinstance(raw_splits, dict):
            raw_splits["test"] = {
                "episodes": 10**12 + index,
                "wins": index % 2,
                "losses": 0,
                "draws": 0,
                "team_counts": {"Leak Attempt": 10**12},
            }

    selected_first = selector.select_routes(
        preregistration(), first
    )["selected_routes"]
    selected_second = selector.select_routes(
        preregistration(), second
    )["selected_routes"]
    assert selected_first == selected_second


def test_frequency_and_wilson_ties_use_deck_hash_ascending() -> None:
    tied_low = "c" * 64
    tied_high = "d" * 64
    source = {
        "schema_version": "ptcg-bc-exact-deck-profile-v1",
        "deduplication_key": ["dataset_date", "episode_id", "seat"],
        "profiled_splits": ["train", "valid"],
        "sealed_splits": ["test"],
        "split_date_counts": {
            "train": {"2026-08-02": 1},
            "valid": {"2026-08-07": 1},
            "test": "SEALED_TEST_DO_NOT_READ",
        },
        "decks": [
            {"deck_hash": MARNIE},
            {"deck_hash": ALAKAZAM},
            deck(
                tied_high,
                train_episodes=100,
                valid_episodes=100,
                valid_wins=60,
            ),
            deck(
                tied_low,
                train_episodes=100,
                valid_episodes=100,
                valid_wins=60,
            ),
            deck(
                ROUTE_3,
                train_episodes=90,
                valid_episodes=100,
                valid_wins=60,
            ),
        ],
    }
    report = selector.select_routes(preregistration(), source)
    assert report["selected_routes"][1]["deck_hash"] == tied_low
    # tied_high and ROUTE_3 have the same Wilson score; tied_high has higher
    # frequency and is therefore route 3.
    assert report["selected_routes"][2]["deck_hash"] == tied_high


def test_cli_writes_hashes_atomically_and_refuses_overwrite() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        prereg_path = root / "preregistration.json"
        profile_path = root / "profile.json"
        output_path = root / "routes.json"
        prereg_path.write_text(
            json.dumps(preregistration()), encoding="utf-8"
        )
        profile_path.write_text(json.dumps(profile()), encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--preregistration",
                str(prereg_path),
                "--profile",
                str(profile_path),
                "--output",
                str(output_path),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["inputs"]["preregistration"]["sha256"] == hashlib.sha256(
            prereg_path.read_bytes()
        ).hexdigest()
        assert report["inputs"]["exact_deck_profile"][
            "sha256"
        ] == hashlib.sha256(profile_path.read_bytes()).hexdigest()
        assert not list(root.glob(".routes.json.*.tmp"))

        refused = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--preregistration",
                str(prereg_path),
                "--profile",
                str(profile_path),
                "--output",
                str(output_path),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert refused.returncode != 0
        assert "pass --overwrite" in refused.stderr

        output_path.write_text("stale", encoding="utf-8")
        overwritten = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--preregistration",
                str(prereg_path),
                "--profile",
                str(profile_path),
                "--output",
                str(output_path),
                "--overwrite",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert overwritten.returncode == 0, overwritten.stderr
        assert json.loads(output_path.read_text(encoding="utf-8"))[
            "schema_version"
        ] == "ptcg-gold-route-selection-v1"
        assert not list(root.glob(".routes.json.*.tmp"))


def test_rejects_inconsistent_valid_outcomes() -> None:
    source = profile()
    candidate = next(
        item for item in source["decks"] if item["deck_hash"] == ROUTE_2
    )
    candidate["split_stats"]["valid"]["wins"] += 1
    with pytest.raises(ValueError, match="valid outcomes do not sum"):
        selector.select_routes(preregistration(), source)


def test_wilson_denominator_includes_draws() -> None:
    source = deck(
        ROUTE_2,
        train_episodes=100,
        valid_episodes=100,
        valid_wins=60,
    )
    valid = source["split_stats"]["valid"]
    valid["losses"] = 0
    valid["draws"] = 40
    candidate = selector.parse_candidate(
        source,
        selector.parse_policy(preregistration()),
    )
    assert candidate.valid_wilson_lower_bound == pytest.approx(
        selector.wilson_lower_bound(60, 100, 1.959963984540054)
    )


@pytest.mark.parametrize(
    ("split_name", "wrong_date", "message"),
    [
        ("train", "2026-08-01", "train date window"),
        ("valid", "2026-08-06", "valid date window"),
    ],
)
def test_rejects_profile_with_wrong_train_or_valid_window(
    split_name: str,
    wrong_date: str,
    message: str,
) -> None:
    source = profile()
    source["split_date_counts"][split_name] = {wrong_date: 1}
    with pytest.raises(ValueError, match=message):
        selector.select_routes(preregistration(), source)
