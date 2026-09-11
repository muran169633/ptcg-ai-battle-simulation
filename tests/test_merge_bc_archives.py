from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import merge_bc_archives as merger  # noqa: E402


SCHEMA = "ptcg-bc-visible-decisions-v1"
DECK_HASH = "synthetic-deck"


def _write_archive(
    path: Path,
    dates: tuple[str, ...],
    *,
    team_filter: dict[str, object] | None = None,
) -> None:
    rows = []
    for index, dataset_date in enumerate(dates):
        rows.append(
            {
                "schema_version": SCHEMA,
                "dataset_date": dataset_date,
                "episode_id": f"episode-{dataset_date}",
                "split": "test",
                "deck_hash": DECK_HASH,
                "team_name": f"team-{index}",
                "select_context": str(index),
                "observation": {"date": dataset_date},
                "action": [index],
            }
        )
    manifest = {
        "schema_version": SCHEMA,
        "dates": list(dates),
        "deck_hash_filter": DECK_HASH,
        # Deliberately stale: merge must derive this from emitted rows.
        "split_episodes": {"train": 91, "valid": 92, "test": 93},
        "sources": [{"date": value, "path": f"{value}.zip"} for value in dates],
    }
    if team_filter is not None:
        manifest["team_filter"] = team_filter
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "test/part-00000.jsonl",
            b"".join(merger.orjson.dumps(row) + b"\n" for row in rows),
        )
        archive.writestr("manifest.json", json.dumps(manifest))


def _read_rows(path: Path) -> dict[str, list[dict[str, object]]]:
    rows = {split: [] for split in merger.SPLITS}
    with zipfile.ZipFile(path) as archive:
        for split in merger.SPLITS:
            for member in archive.namelist():
                if member.startswith(f"{split}/") and member.endswith(".jsonl"):
                    rows[split].extend(
                        json.loads(line) for line in archive.read(member).splitlines()
                    )
    return rows


def _args(
    base: Path,
    supplement: Path,
    output: Path,
    *,
    latest_as_valid: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        base=base,
        supplement=[supplement],
        output=output,
        rows_per_shard=2,
        compress=False,
        overwrite=False,
        latest_as_valid=latest_as_valid,
    )


def test_latest_as_valid_uses_latest_date_only_for_valid_and_has_no_test() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        base = root / "base.zip"
        supplement = root / "supplement.zip"
        output = root / "output.zip"
        _write_archive(base, ("2026-07-27", "2026-07-28"))
        _write_archive(supplement, ("2026-07-29",))

        manifest = merger.merge(
            _args(base, supplement, output, latest_as_valid=True)
        )

        rows = _read_rows(output)
        assert {
            row["dataset_date"] for row in rows["train"]
        } == {"2026-07-27", "2026-07-28"}
        assert {
            row["dataset_date"] for row in rows["valid"]
        } == {"2026-07-29"}
        assert rows["test"] == []
        assert all(row["split"] == "train" for row in rows["train"])
        assert all(row["split"] == "valid" for row in rows["valid"])
        assert manifest["split_policy"] == {
            "mode": "time",
            "train_dates": ["2026-07-27", "2026-07-28"],
            "valid_dates": ["2026-07-29"],
            "test_dates": [],
        }
        assert manifest["split_decisions"] == {
            "train": 2,
            "valid": 1,
            "test": 0,
        }
        assert manifest["split_episodes"] == {
            "train": 2,
            "valid": 1,
            "test": 0,
        }
        assert manifest["shards"] == {"train": 1, "valid": 1, "test": 0}
        with zipfile.ZipFile(output) as archive:
            assert not any(name.startswith("test/") for name in archive.namelist())


def test_default_split_remains_train_valid_test() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        base = root / "base.zip"
        supplement = root / "supplement.zip"
        output = root / "output.zip"
        _write_archive(base, ("2026-07-27", "2026-07-28"))
        _write_archive(supplement, ("2026-07-29",))

        args = _args(base, supplement, output, latest_as_valid=False)
        del args.latest_as_valid  # Preserve compatibility with older callers.
        manifest = merger.merge(args)

        rows = _read_rows(output)
        assert [row["dataset_date"] for row in rows["train"]] == ["2026-07-27"]
        assert [row["dataset_date"] for row in rows["valid"]] == ["2026-07-28"]
        assert [row["dataset_date"] for row in rows["test"]] == ["2026-07-29"]
        assert manifest["split_policy"] == {
            "mode": "time",
            "train_dates": ["2026-07-27"],
            "valid_dates": ["2026-07-28"],
            "test_dates": ["2026-07-29"],
        }
        assert manifest["split_episodes"] == {
            "train": 1,
            "valid": 1,
            "test": 1,
        }


def test_team_filter_is_normalized_deduplicated_union() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        base = root / "base.zip"
        supplement = root / "supplement.zip"
        output = root / "output.zip"
        _write_archive(
            base,
            ("2026-07-27", "2026-07-28"),
            team_filter={
                "all_teams": False,
                "global_team_count": 2,
                "dated_team_counts": {"2026-07-27": 2},
                "display_names": [" Team Alpha ", "Ｔｅａｍ　Beta"],
            },
        )
        _write_archive(
            supplement,
            ("2026-07-29",),
            team_filter={
                "all_teams": False,
                "global_team_count": 2,
                "dated_team_counts": {},
                "display_names": ["team alpha", "Team Gamma"],
            },
        )

        manifest = merger.merge(
            _args(base, supplement, output, latest_as_valid=True)
        )

        assert manifest["team_filter"] == {
            "all_teams": False,
            "global_team_count": 3,
            "dated_team_counts": {},
            "display_names": ["Team Alpha", "Team Beta", "Team Gamma"],
        }
        with zipfile.ZipFile(output) as archive:
            installed_manifest = json.loads(archive.read("manifest.json"))
        assert installed_manifest["team_filter"] == manifest["team_filter"]


def test_any_all_teams_input_propagates_unrestricted_scope() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        base = root / "base.zip"
        supplement = root / "supplement.zip"
        output = root / "output.zip"
        _write_archive(
            base,
            ("2026-07-27", "2026-07-28"),
            team_filter={
                "all_teams": False,
                "global_team_count": 1,
                "dated_team_counts": {},
                "display_names": ["Team Alpha"],
            },
        )
        _write_archive(
            supplement,
            ("2026-07-29",),
            team_filter={
                "all_teams": True,
                "global_team_count": 99,
                "dated_team_counts": {"2026-07-29": 99},
                "display_names": ["must-not-be-retained"],
            },
        )

        manifest = merger.merge(
            _args(base, supplement, output, latest_as_valid=True)
        )

        assert manifest["team_filter"] == {
            "all_teams": True,
            "global_team_count": 0,
            "dated_team_counts": {},
            "display_names": [],
        }
