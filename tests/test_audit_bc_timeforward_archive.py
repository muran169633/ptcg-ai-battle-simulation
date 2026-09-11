from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_bc_timeforward_archive as audit  # noqa: E402
import train_bc_orbit  # noqa: E402


DATES = ("2026-07-25", "2026-07-26", "2026-07-27")
EMPTY_TRAIN_DATE = "2026-07-24"
DECK_HASH = "a" * 64
SCHEMA = "ptcg-bc-visible-decisions-v1"


def make_row(
    split: str,
    episode_id: str,
    dataset_date: str,
    team_name: str,
    *,
    decision_index: int = 0,
) -> dict[str, object]:
    select = {
        "context": 0,
        "type": 1,
        "minCount": 1,
        "maxCount": 1,
        "option": [
            {"type": 1, "area": 0, "index": 0, "value": 1},
            {"type": 1, "area": 0, "index": 1, "value": 2},
        ],
    }
    return {
        "schema_version": SCHEMA,
        "episode_id": episode_id,
        "episode_uuid": f"uuid-{episode_id}",
        "dataset_date": dataset_date,
        "split": split,
        "observation_step_index": decision_index,
        "action_step_index": decision_index + 1,
        "seat": 0,
        "team_name": team_name,
        "deck_hash": DECK_HASH,
        "terminal_reward": 1.0,
        "sample_weight": 1.0,
        "select_context": "0",
        "select_type": 1,
        "min_count": 1,
        "max_count": 1,
        "option_count": 2,
        "action": [decision_index % 2],
        "no_action": False,
        "observation": {"select": select},
    }


def write_archive(
    path: Path,
    *,
    leak: bool = False,
    legacy_filter: bool = False,
    modern_filter: bool = False,
    stale_modern_counts: bool = False,
    source_stats_errors: bool = False,
    row_defects: bool = False,
    trainer_schema_defects: bool = False,
    extra_empty_train_date: bool = False,
    empty_test: bool = False,
    empty_test_member: bool = False,
    wrong_empty_test_manifest: bool = False,
) -> int:
    rows = {
        "train": [
            make_row(
                "train",
                "episode-train",
                DATES[0],
                "Team Alpha",
                decision_index=0,
            ),
            make_row(
                "train",
                "episode-train",
                DATES[0],
                "Team Alpha",
                decision_index=1,
            ),
        ],
        "valid": [
            make_row(
                "valid",
                "episode-train" if leak else "episode-valid",
                DATES[1],
                "Team Beta",
            )
        ],
        "test": [
            make_row("test", "episode-test", DATES[2], "Team Alpha")
        ],
    }
    if empty_test:
        rows["test"] = []
    if row_defects:
        rows["train"][1]["observation_step_index"] = 0
        rows["train"][1]["action_step_index"] = 1
        rows["train"][1]["action"] = [9]
        inconsistent = make_row(
            "train",
            "episode-other",
            DATES[0],
            "Team Beta",
            decision_index=2,
        )
        inconsistent["episode_uuid"] = "uuid-episode-train"
        rows["train"].append(inconsistent)
        rows["valid"][0]["episode_uuid"] = " "
        rows["valid"][0]["seat"] = 2
        rows["valid"][0]["action_step_index"] = 4
    if trainer_schema_defects:
        rows["train"][1]["observation"]["select"]["type"] = "select"
        rows["valid"][0]["observation"]["select"]["option"][0] = "bad-option"
        test_select = rows["test"][0]["observation"]["select"]
        test_select["option"] = [
            {"type": 1, "area": 0, "index": index}
            for index in range(audit.TRAINER_MAX_ACTION_COUNT + 1)
        ]
        test_select["minCount"] = audit.TRAINER_MAX_ACTION_COUNT + 1
        test_select["maxCount"] = audit.TRAINER_MAX_ACTION_COUNT + 1
        rows["test"][0]["action"] = list(
            range(audit.TRAINER_MAX_ACTION_COUNT + 1)
        )
        rows["test"][0]["min_count"] = audit.TRAINER_MAX_ACTION_COUNT + 1
        rows["test"][0]["max_count"] = audit.TRAINER_MAX_ACTION_COUNT + 1
        rows["test"][0]["option_count"] = (
            audit.TRAINER_MAX_ACTION_COUNT + 1
        )

    team_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    split_episodes: dict[str, int] = {}
    for split in audit.SPLITS:
        split_ids = {str(row["episode_id"]) for row in rows[split]}
        split_episodes[split] = len(split_ids)
        for row in rows[split]:
            team_counts[str(row["team_name"])] += 1
            context_counts[str(row["select_context"])] += 1

    # Source scanning happens before row validation/deduplication.
    episodes_scanned = 4 if row_defects else (2 if empty_test else 3)
    active_dates = list(DATES[:2] if empty_test else DATES)
    manifest_dates = (
        [EMPTY_TRAIN_DATE, *active_dates]
        if extra_empty_train_date
        else active_dates
    )
    manifest_train_dates = (
        [EMPTY_TRAIN_DATE, DATES[0]]
        if extra_empty_train_date
        else [DATES[0]]
    )
    manifest = {
        "schema_version": SCHEMA,
        "competition": "pokemon-tcg-ai-battle",
        "dates": manifest_dates,
        "split_policy": {
            "mode": "time",
            "train_dates": manifest_train_dates,
            "valid_dates": [DATES[1]],
            "test_dates": [] if empty_test else [DATES[2]],
        },
        "team_filter": {
            "all_teams": False,
            "global_team_count": 2,
            "dated_team_counts": {},
            "display_names": ["Team Alpha", "Team Beta"],
        },
        "label_alignment": "steps[t-1].observation -> steps[t].action",
        "hidden_information_policy": (
            "Only agent-visible observation is saved; replay visualize is "
            "excluded."
        ),
        "stats": {
            "episodes_scanned": episodes_scanned,
            "decisions": sum(len(value) for value in rows.values()),
            "duplicate_episodes": 1 if source_stats_errors else 0,
            "invalid_json_files": 2 if source_stats_errors else 0,
        },
        "split_decisions": {
            split: len(rows[split]) for split in audit.SPLITS
        },
        "shards": {
            split: (
                0
                if split == "test"
                and empty_test
                and not empty_test_member
                else 1
            )
            for split in audit.SPLITS
        },
        "split_episodes": split_episodes,
        "team_decisions": dict(team_counts),
        "context_decisions": dict(context_counts),
        "sources": [
            {
                "date": dataset_date,
                "kind": "zip",
                "path": f"/frozen/{dataset_date}.zip",
            }
            for dataset_date in manifest_dates
        ],
    }
    if wrong_empty_test_manifest:
        if not empty_test or empty_test_member:
            raise ValueError(
                "wrong_empty_test_manifest requires an empty test without "
                "a test member"
            )
        manifest["split_policy"]["test_dates"] = [DATES[2]]
        manifest["split_decisions"]["test"] = 1
        manifest["shards"]["test"] = 1
    if legacy_filter:
        manifest["filtered_from"] = "/frozen/source.zip"
        manifest["deck_hash_filter"] = DECK_HASH
        manifest["stats"]["decisions"] = 100
        manifest["shards"] = {"train": 9, "valid": 3, "test": 2}
        manifest["team_decisions"] = {"inherited-source-team": 100}
        manifest["context_decisions"] = {"inherited-source-context": 100}
    if modern_filter:
        source_stats = dict(manifest["stats"])
        source_split_decisions = dict(manifest["split_decisions"])
        source_shards = dict(manifest["shards"])
        manifest["filtered_from"] = "/frozen/source.zip"
        manifest["deck_hash_filter"] = DECK_HASH
        manifest["filter_lineage"] = {
            "source_split_decisions": source_split_decisions,
            "source_shards": source_shards,
            "source_stats": source_stats,
        }
        manifest["stats"] = {
            "decisions": sum(len(value) for value in rows.values()),
            "episodes_in_output": len(
                {
                    str(row["episode_id"])
                    for split in audit.SPLITS
                    for row in rows[split]
                }
            ),
            "input_archives": 1,
        }
        if stale_modern_counts:
            manifest["stats"]["decisions"] = 100
            manifest["shards"] = {"train": 9, "valid": 3, "test": 2}
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for split in audit.SPLITS:
            if split == "test" and empty_test and not empty_test_member:
                continue
            payload = b"".join(
                (
                    json.dumps(row, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                for row in rows[split]
            )
            archive.writestr(f"{split}/part-00000.jsonl", payload)
        archive.writestr("manifest.json", json.dumps(manifest))
    return episodes_scanned


def expected_layout(
    *,
    extra_empty_train_date: bool = False,
) -> audit.ExpectedLayout:
    return audit.build_expected_layout(
        train_dates=(
            [EMPTY_TRAIN_DATE, DATES[0]]
            if extra_empty_train_date
            else [DATES[0]]
        ),
        valid_dates=[DATES[1]],
        test_dates=[DATES[2]],
    )


def trainer_consumable_rows(path: Path) -> int:
    count = 0
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".jsonl"):
                continue
            with archive.open(member) as handle:
                for line in handle:
                    row = json.loads(line)
                    try:
                        features = train_bc_orbit.featurize_row(
                            row,
                            hash_size=64,
                            max_state_entities=8,
                        )
                    except (TypeError, ValueError, AttributeError, OverflowError):
                        continue
                    if features is not None:
                        count += 1
    return count


class AuditBcTimeforwardArchiveTests(unittest.TestCase):
    def test_passes_strict_stream_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "bc.zip"
            teams = root / "teams.txt"
            teams.write_text("Team Alpha\nTeam Beta\n", encoding="utf-8")
            write_archive(archive)
            before = archive.read_bytes()

            report = audit.audit_archive(
                archive,
                teams,
                expected_layout(),
                expected_deck_hash=DECK_HASH,
                expected_episodes_scanned=3,
            )

            self.assertTrue(report["pass"], report["issues"])
            self.assertTrue(report["promotion_eligible"])
            self.assertFalse(report["diagnostic_only"])
            self.assertEqual(report["statistics"]["rows_scanned"], 4)
            self.assertEqual(report["statistics"]["invalid_rows"], 0)
            self.assertEqual(
                report["statistics"]["trainable_rows"],
                trainer_consumable_rows(archive),
            )
            self.assertEqual(
                audit.TRAINER_MAX_ACTION_COUNT,
                train_bc_orbit.MAX_ACTION_COUNT,
            )
            self.assertEqual(report["statistics"]["unique_episode_ids"], 3)
            self.assertTrue(report["leakage"]["pass"])
            self.assertEqual(archive.read_bytes(), before)

    def test_cli_reports_cross_split_leakage_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "leaky.zip"
            teams = root / "teams.txt"
            output = root / "audit.json"
            teams.write_text("Team Alpha\nTeam Beta\n", encoding="utf-8")
            write_archive(archive, leak=True)

            exit_code = audit.main(
                [
                    "--archive",
                    str(archive),
                    "--teams-file",
                    str(teams),
                    "--expected-deck-hash",
                    DECK_HASH,
                    "--expected-train-dates",
                    DATES[0],
                    "--expected-valid-dates",
                    DATES[1],
                    "--expected-test-dates",
                    DATES[2],
                    "--expected-episodes-scanned",
                    "3",
                    "--json-output",
                    str(output),
                ]
            )
            report = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(exit_code, 1)
            self.assertFalse(report["pass"])
            self.assertFalse(report["leakage"]["pass"])
            self.assertEqual(
                report["leakage"]["episode_ids_in_multiple_splits"], 1
            )
            self.assertEqual(
                report["leakage"]["episode_uuids_in_multiple_splits"], 1
            )
            self.assertEqual(report["statistics"]["invalid_rows"], 1)
            self.assertIn(
                "episode_id_cross_split_leakage",
                report["issues"]["counts_by_code"],
            )
            self.assertIn(
                "row_episode_identity_inconsistent",
                report["issues"]["counts_by_code"],
            )

    def test_cli_accepts_explicit_empty_test_and_verifies_zero_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "latest-as-valid.zip"
            teams = root / "teams.txt"
            output = root / "audit.json"
            teams.write_text("Team Alpha\nTeam Beta\n", encoding="utf-8")
            write_archive(archive, empty_test=True)

            exit_code = audit.main(
                [
                    "--archive",
                    str(archive),
                    "--teams-file",
                    str(teams),
                    "--expected-deck-hash",
                    DECK_HASH,
                    "--expected-train-dates",
                    DATES[0],
                    "--expected-valid-dates",
                    DATES[1],
                    "--expect-empty-test",
                    "--expected-episodes-scanned",
                    "2",
                    "--json-output",
                    str(output),
                ]
            )
            report = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(exit_code, 0)
            self.assertTrue(report["pass"], report["issues"])
            self.assertEqual(
                report["expected"]["dates_by_split"]["test"], []
            )
            self.assertEqual(
                report["manifest_summary"]["split_policy"]["test_dates"], []
            )
            self.assertEqual(
                report["manifest_summary"]["split_decisions"]["test"], 0
            )
            self.assertEqual(report["manifest_summary"]["shards"]["test"], 0)
            self.assertEqual(
                report["statistics"]["by_split"]["test"]["rows_scanned"], 0
            )
            self.assertEqual(
                report["statistics"]["by_split"]["test"]["shards"], 0
            )

    def test_empty_test_still_rejects_nonzero_manifest_declarations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "bad-latest-as-valid.zip"
            teams = root / "teams.txt"
            teams.write_text("Team Alpha\nTeam Beta\n", encoding="utf-8")
            write_archive(
                archive,
                empty_test=True,
                wrong_empty_test_manifest=True,
            )

            report = audit.audit_archive(
                archive,
                teams,
                audit.build_expected_layout(
                    train_dates=[DATES[0]],
                    valid_dates=[DATES[1]],
                    expect_empty_test=True,
                ),
                expected_deck_hash=DECK_HASH,
                expected_episodes_scanned=2,
            )
            codes = report["issues"]["counts_by_code"]

            self.assertFalse(report["pass"])
            self.assertIn("manifest_split_dates_mismatch", codes)
            self.assertIn("manifest_split_decisions_mismatch", codes)
            self.assertIn("manifest_shards_mismatch", codes)

    def test_empty_test_cli_is_unambiguous_and_legacy_defaults_hold(
        self,
    ) -> None:
        parser = audit.build_parser()
        common = [
            "--archive",
            "archive.zip",
            "--teams-file",
            "teams.txt",
            "--expected-train-dates",
            DATES[0],
            "--expected-valid-dates",
            DATES[1],
        ]
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    *common,
                    "--expected-test-dates",
                    DATES[2],
                    "--expect-empty-test",
                ]
            )
        with self.assertRaisesRegex(
            ValueError,
            "expected test dates cannot be empty",
        ):
            audit.build_expected_layout(
                train_dates=[DATES[0]],
                valid_dates=[DATES[1]],
                test_dates=[],
            )
        with self.assertRaisesRegex(
            ValueError,
            "cannot be combined",
        ):
            audit.build_expected_layout(
                expected_dates=list(DATES) * 4,
                expect_empty_test=True,
            )

    def test_cli_rejects_legacy_filter_manifest_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "legacy-filter.zip"
            teams = root / "teams.txt"
            output = root / "legacy-audit.json"
            teams.write_text("Team Alpha\nTeam Beta\n", encoding="utf-8")
            write_archive(archive, legacy_filter=True)

            exit_code = audit.main(
                [
                    "--archive",
                    str(archive),
                    "--teams-file",
                    str(teams),
                    "--expected-deck-hash",
                    DECK_HASH,
                    "--expected-train-dates",
                    DATES[0],
                    "--expected-valid-dates",
                    DATES[1],
                    "--expected-test-dates",
                    DATES[2],
                    "--expected-episodes-scanned",
                    "3",
                    "--json-output",
                    str(output),
                ]
            )
            report = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(exit_code, 1)
            self.assertFalse(report["pass"])
            self.assertTrue(report["diagnostic_only"])
            self.assertFalse(report["promotion_eligible"])
            self.assertEqual(
                report["filter_lineage"]["mode"],
                "legacy_missing_filter_lineage",
            )
            self.assertIn(
                "manifest_filter_lineage_missing",
                report["issues"]["counts_by_code"],
            )

    def test_modern_filter_uses_source_lineage_and_strict_derived_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teams = root / "teams.txt"
            teams.write_text("Team Alpha\nTeam Beta\n", encoding="utf-8")
            current = root / "modern.zip"
            stale = root / "modern-stale.zip"
            write_archive(current, modern_filter=True)
            write_archive(
                stale,
                modern_filter=True,
                stale_modern_counts=True,
            )

            current_report = audit.audit_archive(
                current,
                teams,
                expected_layout(),
                expected_deck_hash=DECK_HASH,
                expected_episodes_scanned=3,
            )
            stale_report = audit.audit_archive(
                stale,
                teams,
                expected_layout(),
                expected_deck_hash=DECK_HASH,
                expected_episodes_scanned=3,
            )

            self.assertTrue(current_report["pass"], current_report["issues"])
            self.assertTrue(current_report["promotion_eligible"])
            self.assertEqual(
                current_report["filter_lineage"]["mode"],
                "materialized_filter_lineage",
            )
            self.assertEqual(
                current_report["filter_lineage"]["source_stats_location"],
                "filter_lineage.source_stats",
            )
            self.assertFalse(stale_report["pass"])
            self.assertIn(
                "manifest_decisions_mismatch",
                stale_report["issues"]["counts_by_code"],
            )
            self.assertIn(
                "manifest_shards_mismatch",
                stale_report["issues"]["counts_by_code"],
            )

    def test_rejects_source_errors_and_untrainable_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "bad.zip"
            teams = root / "teams.txt"
            teams.write_text("Team Alpha\nTeam Beta\n", encoding="utf-8")
            episodes_scanned = write_archive(
                archive,
                modern_filter=True,
                source_stats_errors=True,
                row_defects=True,
            )

            report = audit.audit_archive(
                archive,
                teams,
                expected_layout(),
                expected_deck_hash=DECK_HASH,
                expected_episodes_scanned=episodes_scanned,
            )
            codes = report["issues"]["counts_by_code"]

            self.assertFalse(report["pass"])
            self.assertGreater(report["statistics"]["invalid_rows"], 0)
            self.assertLess(
                report["statistics"]["trainable_rows"],
                report["statistics"]["rows_scanned"],
            )
            for code in (
                "manifest_duplicate_episodes_nonzero",
                "manifest_invalid_json_files_nonzero",
                "row_action_illegal",
                "row_duplicate_decision_key",
                "row_episode_identity_inconsistent",
                "row_episode_uuid_invalid",
                "row_seat_invalid",
                "row_step_alignment_invalid",
            ):
                self.assertIn(code, codes)

    def test_trainable_rows_match_real_featurizer_on_schema_failures(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "trainer-schema-bad.zip"
            teams = root / "teams.txt"
            teams.write_text("Team Alpha\nTeam Beta\n", encoding="utf-8")
            write_archive(archive, trainer_schema_defects=True)

            report = audit.audit_archive(
                archive,
                teams,
                expected_layout(),
                expected_deck_hash=DECK_HASH,
                expected_episodes_scanned=3,
            )
            codes = report["issues"]["counts_by_code"]

            self.assertFalse(report["pass"])
            self.assertEqual(
                report["statistics"]["trainable_rows"],
                trainer_consumable_rows(archive),
            )
            self.assertEqual(report["statistics"]["trainable_rows"], 1)
            self.assertIn(
                "row_trainer_select_numeric_invalid", codes
            )
            self.assertIn("row_trainer_option_invalid", codes)
            self.assertIn("row_action_too_long", codes)

    def test_raw_requires_exact_date_coverage_filtered_allows_subset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teams = root / "teams.txt"
            teams.write_text("Team Alpha\nTeam Beta\n", encoding="utf-8")
            raw_archive = root / "raw-missing-day.zip"
            filtered_archive = root / "filtered-missing-day.zip"
            write_archive(raw_archive, extra_empty_train_date=True)
            write_archive(
                filtered_archive,
                modern_filter=True,
                extra_empty_train_date=True,
            )
            layout = expected_layout(extra_empty_train_date=True)

            raw_report = audit.audit_archive(
                raw_archive,
                teams,
                layout,
                expected_deck_hash=DECK_HASH,
                expected_episodes_scanned=3,
            )
            filtered_report = audit.audit_archive(
                filtered_archive,
                teams,
                layout,
                expected_deck_hash=DECK_HASH,
                expected_episodes_scanned=3,
            )

            self.assertFalse(raw_report["pass"])
            self.assertIn(
                "split_observed_dates_incomplete",
                raw_report["issues"]["counts_by_code"],
            )
            self.assertTrue(filtered_report["pass"], filtered_report["issues"])
            self.assertTrue(filtered_report["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
