from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import prepare_bc_week as prepare  # noqa: E402


def make_episode(episode_id: int, opponent: str) -> dict[str, object]:
    return {
        "id": f"uuid-{episode_id}",
        "info": {
            "EpisodeId": episode_id,
            "TeamNames": ["Allowed Team", opponent],
        },
        "rewards": [1, -1],
        "steps": [
            [
                {
                    "status": "ACTIVE",
                    "observation": {
                        "select": {
                            "context": 0,
                            "type": 1,
                            "minCount": 1,
                            "maxCount": 1,
                            "option": [{"type": 1, "index": 0}],
                        }
                    },
                },
                {"status": "ACTIVE", "observation": {"select": None}},
            ],
            [{"action": [0]}, {"action": []}],
        ],
    }


class PrepareBcWeekTests(unittest.TestCase):
    def test_time_split_defaults_and_configurable_holdouts(self) -> None:
        dates = [f"2026-08-{day:02d}" for day in range(1, 7)]
        self.assertEqual(
            prepare.time_split_policy(dates, 1, 1),
            {
                "mode": "time",
                "train_dates": dates[:4],
                "valid_dates": dates[4:5],
                "test_dates": dates[5:],
                "valid_days": 1,
                "test_days": 1,
            },
        )
        policy = prepare.time_split_policy(dates, 2, 2)
        self.assertEqual(policy["train_dates"], dates[:2])
        self.assertEqual(policy["valid_dates"], dates[2:4])
        self.assertEqual(policy["test_dates"], dates[4:])
        self.assertEqual(
            prepare.split_for_episode(
                "1", dates[2], dates, "time", 7, valid_days=2, test_days=2
            ),
            "valid",
        )
        self.assertEqual(
            prepare.split_for_episode(
                "1", dates[5], dates, "time", 7, valid_days=2, test_days=2
            ),
            "test",
        )
        with self.assertRaisesRegex(ValueError, "at least 5 distinct dates"):
            prepare.time_split_policy(dates[:4], 2, 2)

    def test_parser_defaults_and_repeatable_exclusions(self) -> None:
        args = prepare.build_parser().parse_args(
            [
                "--input",
                "episodes-2026-08-01.json",
                "--exclude-team-name",
                " Team One ",
                "--exclude-team-name",
                "TEAM TWO",
            ]
        )
        self.assertEqual(args.valid_days, 1)
        self.assertEqual(args.test_days, 1)
        self.assertEqual(args.hash_train_fraction, 0.80)
        self.assertEqual(args.hash_valid_fraction, 0.10)
        self.assertEqual(args.exclude_team_name, [" Team One ", "TEAM TWO"])

    def test_daily_hash_split_is_date_aware_and_supports_90_10(self) -> None:
        dates = ["2026-08-13", "2026-08-14"]
        observed_by_date = {
            dataset_date: Counter(
                prepare.split_for_episode(
                    str(episode_id),
                    dataset_date,
                    dates,
                    "daily_hash",
                    20260815,
                    hash_train_fraction=0.90,
                    hash_valid_fraction=0.10,
                )
                for episode_id in range(10_000)
            )
            for dataset_date in dates
        }
        for counts in observed_by_date.values():
            self.assertEqual(counts["test"], 0)
            self.assertGreater(counts["train"], 8_800)
            self.assertGreater(counts["valid"], 800)
        assignments_13 = [
            prepare.split_for_episode(
                str(episode_id),
                dates[0],
                dates,
                "daily_hash",
                20260815,
                hash_train_fraction=0.90,
                hash_valid_fraction=0.10,
            )
            for episode_id in range(100)
        ]
        assignments_14 = [
            prepare.split_for_episode(
                str(episode_id),
                dates[1],
                dates,
                "daily_hash",
                20260815,
                hash_train_fraction=0.90,
                hash_valid_fraction=0.10,
            )
            for episode_id in range(100)
        ]
        self.assertNotEqual(assignments_13, assignments_14)

    def test_hash_split_fraction_validation(self) -> None:
        prepare.validate_hash_split_fractions(0.90, 0.10)
        with self.assertRaisesRegex(ValueError, "must be <= 1"):
            prepare.validate_hash_split_fractions(0.95, 0.10)

    def test_all_train_split_is_explicit(self) -> None:
        self.assertEqual(
            prepare.split_for_episode(
                "episode", "2026-08-14", ["2026-08-14"], "all_train", 7
            ),
            "train",
        )

    def test_exclusion_drops_whole_episode_and_records_hits(self) -> None:
        dates = [f"2026-08-{day:02d}" for day in range(1, 6)]
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            sources: list[prepare.DailySource] = []
            for index, dataset_date in enumerate(dates, 1):
                daily = root / dataset_date
                daily.mkdir()
                (daily / f"{index}01.json").write_text(
                    json.dumps(make_episode(index * 100 + 1, "Safe Opponent")),
                    encoding="utf-8",
                )
                (daily / f"{index}02.json").write_text(
                    json.dumps(make_episode(index * 100 + 2, " EXCLUDED  Team ")),
                    encoding="utf-8",
                )
                (daily / f"{index}03.json").write_text(
                    json.dumps(make_episode(index * 100 + 3, "")),
                    encoding="utf-8",
                )
                sources.append(
                    prepare.DailySource(dataset_date, daily, "directory")
                )

            output = root / "decisions.zip"
            manifest = prepare.build_archive(
                sources=sources,
                output=output,
                team_filter=prepare.TeamFilter(True, set(), {}, []),
                frames_per_shard=10,
                split_mode="time",
                split_seed=7,
                overwrite=False,
                max_episodes=None,
                valid_days=2,
                test_days=1,
                excluded_team_names=["Excluded Team", " excluded   team "],
            )

            self.assertEqual(
                manifest["split_decisions"],
                {"train": 2, "valid": 2, "test": 1},
            )
            self.assertEqual(
                manifest["split_policy"]["valid_dates"], dates[2:4]
            )
            exclusion = manifest["team_exclusions"]
            self.assertEqual(exclusion["display_names"], ["Excluded Team"])
            self.assertEqual(exclusion["normalized_names"], ["excluded team"])
            self.assertEqual(exclusion["episode_hits"], 5)
            self.assertEqual(exclusion["seat_hits"], 5)
            self.assertEqual(exclusion["unresolved_episode_hits"], 5)
            self.assertEqual(exclusion["unresolved_seat_hits"], 5)
            self.assertEqual(
                exclusion["hits_by_normalized_name"], {"excluded team": 5}
            )
            verified = prepare.verify_archive(
                output, excluded_team_names=["EXCLUDED TEAM"]
            )
            self.assertEqual(verified, manifest)

    def test_legacy_iter_episode_decisions_call_still_works(self) -> None:
        rows = list(
            prepare.iter_episode_decisions(
                make_episode(1, "Safe Opponent"),
                "1",
                "2026-08-01",
                prepare.TeamFilter(True, set(), {}, []),
                "train",
                Counter(),
                Counter(),
                Counter(),
            )
        )
        self.assertEqual(len(rows), 1)

    def test_verify_archive_fails_closed_on_excluded_name(self) -> None:
        row = {
            "episode_id": "1",
            "split": "train",
            "team_name": "Allowed Team",
            "opponent_team_name": "Excluded Team",
            "observation": {},
            "action": [],
        }
        manifest = {
            "team_exclusions": {
                "display_names": ["Excluded Team"],
                "normalized_names": ["excluded team"],
                "episode_hits": 0,
                "seat_hits": 0,
                "unresolved_episode_hits": 0,
                "unresolved_seat_hits": 0,
                "hits_by_normalized_name": {"excluded team": 0},
            }
        }
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "bad.zip"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("train/part-00000.jsonl", prepare.json_bytes(row))
                archive.writestr("manifest.json", json.dumps(manifest))
            with self.assertRaisesRegex(
                RuntimeError, "Excluded team name found after build"
            ):
                prepare.verify_archive(
                    path, excluded_team_names=["Excluded Team"]
                )

            row.pop("opponent_team_name")
            missing_opponent = Path(raw_root) / "missing-opponent.zip"
            with zipfile.ZipFile(
                missing_opponent, "w", zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr(
                    "train/part-00000.jsonl", prepare.json_bytes(row)
                )
                archive.writestr("manifest.json", json.dumps(manifest))
            with self.assertRaisesRegex(
                RuntimeError,
                "verification requires non-empty opponent_team_name",
            ):
                prepare.verify_archive(
                    missing_opponent, excluded_team_names=["Excluded Team"]
                )


if __name__ == "__main__":
    unittest.main()
