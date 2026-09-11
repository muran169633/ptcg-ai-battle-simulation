from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "filter_bc_archive.py"
TARGET_DECK = "a" * 64
OTHER_DECK = "b" * 64


def _row(
    split: str,
    index: int,
    deck_hash: str,
    terminal_reward: float = 1.0,
) -> dict[str, object]:
    return {
        "schema_version": "ptcg-bc-visible-decisions-v1",
        "split": split,
        "episode_id": f"{split}-{index}",
        "episode_uuid": f"uuid-{split}-{index}",
        "dataset_date": {
            "train": "2026-07-25",
            "valid": "2026-07-26",
            "test": "2026-07-27",
        }[split],
        "team_name": "Team Alpha",
        "deck_hash": deck_hash,
        "terminal_reward": terminal_reward,
        "observation": {"select": {"option": []}},
        "action": [],
    }


def _write_source(path: Path) -> None:
    rows = {
        "train": [
            _row("train", 0, TARGET_DECK),
            _row("train", 1, OTHER_DECK),
            _row("train", 2, TARGET_DECK, -1.0),
        ],
        "valid": [
            _row("valid", 0, TARGET_DECK, -1.0),
            _row("valid", 1, OTHER_DECK),
        ],
        "test": [
            _row("test", 0, OTHER_DECK),
            _row("test", 1, TARGET_DECK, 0.0),
        ],
    }
    manifest = {
        "schema_version": "ptcg-bc-visible-decisions-v1",
        "competition": "pokemon-tcg-ai-battle",
        "stats": {"episodes_scanned": 7, "decisions": 7},
        "split_decisions": {split: len(value) for split, value in rows.items()},
        "shards": {"train": 1, "valid": 1, "test": 1},
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for split, split_rows in rows.items():
            payload = "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in split_rows
            )
            archive.writestr(f"{split}/part-00000.jsonl", payload)
        archive.writestr("manifest.json", json.dumps(manifest))


def test_filter_rewrites_derived_counts_and_preserves_lineage() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source.zip"
        output = root / "filtered.zip"
        _write_source(source)
        source_before = source.read_bytes()

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--input",
                str(source),
                "--output",
                str(output),
                "--deck-hash",
                TARGET_DECK,
                "--rows-per-shard",
                "1",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert source.read_bytes() == source_before
        assert output.is_file()
        assert not output.with_suffix(".zip.partial").exists()
        with zipfile.ZipFile(output) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            row_members = [
                name for name in archive.namelist() if name.endswith(".jsonl")
            ]
        assert manifest["split_decisions"] == {
            "train": 2,
            "valid": 1,
            "test": 1,
        }
        assert manifest["shards"] == {"train": 2, "valid": 1, "test": 1}
        assert manifest["stats"]["decisions"] == 4
        assert manifest["stats"]["episodes_in_output"] == 4
        assert manifest["stats"]["input_archives"] == 1
        assert manifest["split_episodes"] == {
            "train": 2,
            "valid": 1,
            "test": 1,
        }
        assert manifest["team_decisions"] == {"Team Alpha": 4}
        assert manifest["filter_lineage"]["source_split_decisions"] == {
            "train": 3,
            "valid": 2,
            "test": 2,
        }
        assert manifest["filter_lineage"]["source_stats"] == {
            "episodes_scanned": 7,
            "decisions": 7,
        }
        assert len(row_members) == 4


def test_filter_can_select_team_without_deck_hash() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source.zip"
        output = root / "team-filtered.zip"
        _write_source(source)

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--input",
                str(source),
                "--output",
                str(output),
                "--team-name",
                "Team Alpha",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        with zipfile.ZipFile(output) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            output_rows = [
                json.loads(line)
                for name in archive.namelist()
                if name.endswith(".jsonl")
                for line in archive.read(name).decode("utf-8").splitlines()
            ]
        assert manifest["deck_hash_filter"] is None
        assert manifest["team_name_filter"] == "Team Alpha"
        assert manifest["split_decisions"] == {
            "train": 3,
            "valid": 2,
            "test": 2,
        }
        assert {row["deck_hash"] for row in output_rows} == {
            TARGET_DECK,
            OTHER_DECK,
        }


def test_filter_refuses_to_overwrite_existing_output() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source.zip"
        output = root / "filtered.zip"
        _write_source(source)
        output.write_bytes(b"keep")

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--input",
                str(source),
                "--output",
                str(output),
                "--deck-hash",
                TARGET_DECK,
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode != 0
        assert output.read_bytes() == b"keep"
        assert "refusing to overwrite" in completed.stderr


def test_filter_can_keep_winning_demonstrations_only() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source.zip"
        output = root / "wins.zip"
        rows = {
            split: [
                _row(split, 0, TARGET_DECK, 1.0),
                _row(split, 1, TARGET_DECK, -1.0),
                _row(split, 2, TARGET_DECK, 0.0),
            ]
            for split in ("train", "valid", "test")
        }
        manifest = {
            "schema_version": "ptcg-bc-visible-decisions-v1",
            "competition": "pokemon-tcg-ai-battle",
            "stats": {"decisions": 9},
            "split_decisions": {split: 3 for split in rows},
            "shards": {split: 1 for split in rows},
        }
        with zipfile.ZipFile(
            source, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for split, split_rows in rows.items():
                payload = "".join(
                    json.dumps(row, separators=(",", ":")) + "\n"
                    for row in split_rows
                )
                archive.writestr(f"{split}/part-00000.jsonl", payload)
            archive.writestr("manifest.json", json.dumps(manifest))

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--input",
                str(source),
                "--output",
                str(output),
                "--deck-hash",
                TARGET_DECK,
                "--reward-mode",
                "wins",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        with zipfile.ZipFile(output) as archive:
            result = json.loads(archive.read("manifest.json"))
            output_rows = [
                json.loads(line)
                for name in archive.namelist()
                if name.endswith(".jsonl")
                for line in archive.read(name).decode("utf-8").splitlines()
            ]
        assert result["split_decisions"] == {
            "train": 1,
            "valid": 1,
            "test": 1,
        }
        assert result["terminal_reward_filter"] == {
            "mode": "wins",
            "splits": "all",
            "predicate": "terminal_reward > 0",
            "rows_filtered_by_reward": 6,
            "rows_missing_terminal_reward": 0,
            "rows_filtered_by_reward_by_split": {
                "train": 2,
                "valid": 2,
                "test": 2,
            },
            "rows_missing_terminal_reward_by_split": {
                "train": 0,
                "valid": 0,
                "test": 0,
            },
        }
        assert result["stats"]["decisions"] == 3
        assert result["date_decisions"] == {
            "2026-07-25": 1,
            "2026-07-26": 1,
            "2026-07-27": 1,
        }
        assert len(output_rows) == 3
        assert all(row["terminal_reward"] == 1.0 for row in output_rows)


def test_filter_can_limit_reward_filter_to_training_split() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source.zip"
        output = root / "train-wins.zip"
        _write_source(source)

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--input",
                str(source),
                "--output",
                str(output),
                "--deck-hash",
                TARGET_DECK,
                "--reward-mode",
                "wins",
                "--reward-filter-splits",
                "train",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        with zipfile.ZipFile(output) as archive:
            result = json.loads(archive.read("manifest.json"))
        assert result["split_decisions"] == {
            "train": 1,
            "valid": 1,
            "test": 1,
        }
        assert result["terminal_reward_filter"]["splits"] == "train"
        assert result["terminal_reward_filter"][
            "rows_filtered_by_reward_by_split"
        ] == {"train": 1, "valid": 0, "test": 0}


def test_filter_can_select_dataset_dates_and_rewrites_date_manifest() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source.zip"
        output = root / "date-filtered.zip"
        _write_source(source)

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--input",
                str(source),
                "--output",
                str(output),
                "--deck-hash",
                TARGET_DECK,
                "--dataset-date",
                "2026-07-25",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        with zipfile.ZipFile(output) as archive:
            result = json.loads(archive.read("manifest.json"))
            members = archive.namelist()
        assert result["dates"] == ["2026-07-25"]
        assert result["split_policy"] == {
            "mode": "preserved_split_date_filter",
            "train_dates": ["2026-07-25"],
            "valid_dates": [],
            "test_dates": [],
        }
        assert result["split_decisions"] == {
            "train": 2,
            "valid": 0,
            "test": 0,
        }
        assert result["dataset_date_filter"]["requested_dates"] == [
            "2026-07-25"
        ]
        assert result["dataset_date_filter"]["missing_requested_dates"] == []
        assert any(name.startswith("train/") for name in members)
        assert not any(name.startswith("valid/") for name in members)
        assert not any(name.startswith("test/") for name in members)


def test_filter_excludes_entire_episode_when_opponent_alias_matches() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source.zip"
        output = root / "excluded.zip"
        contaminated = _row("train", 10, TARGET_DECK)
        contaminated["episode_id"] = "shared-contaminated"
        contaminated["opponent_team_name"] = "  {{ TEAM_NAME }}  🏆 "
        same_episode = _row("train", 11, TARGET_DECK)
        same_episode["episode_id"] = "shared-contaminated"
        same_episode["opponent_team_name"] = "Clean Opponent"
        clean = _row("train", 12, TARGET_DECK)
        clean["opponent_team_name"] = "Clean Opponent"
        rows = {"train": [contaminated, same_episode, clean]}
        manifest = {
            "schema_version": "ptcg-bc-visible-decisions-v1",
            "competition": "pokemon-tcg-ai-battle",
            "stats": {"decisions": 3},
            "split_decisions": {"train": 3, "valid": 0, "test": 0},
            "shards": {"train": 1, "valid": 0, "test": 0},
        }
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr(
                "train/part-00000.jsonl",
                "".join(json.dumps(row) + "\n" for row in rows["train"]),
            )
            archive.writestr("manifest.json", json.dumps(manifest))

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--input",
                str(source),
                "--output",
                str(output),
                "--deck-hash",
                TARGET_DECK,
                "--exclude-team-name",
                "{{ team_name }} 🏆",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        with zipfile.ZipFile(output) as archive:
            result = json.loads(archive.read("manifest.json"))
            output_rows = [
                json.loads(line)
                for name in archive.namelist()
                if name.endswith(".jsonl")
                for line in archive.read(name).decode("utf-8").splitlines()
            ]
        assert [row["episode_id"] for row in output_rows] == ["train-12"]
        assert result["team_exclusions"]["episode_hits"] == 1
        assert result["team_exclusions"]["rows_filtered"] == 2
        assert result["team_exclusions"]["normalized_names"] == [
            "{{ team_name }} 🏆"
        ]


def test_filter_can_drop_rows_above_trainer_action_limit() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source.zip"
        output = root / "trainer-compatible.zip"
        _write_source(source)
        with zipfile.ZipFile(source, "a") as archive:
            long_row = _row("train", 99, TARGET_DECK)
            long_row["action"] = list(range(17))
            archive.writestr(
                "train/part-00001.jsonl", json.dumps(long_row) + "\n"
            )

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--input",
                str(source),
                "--output",
                str(output),
                "--deck-hash",
                TARGET_DECK,
                "--max-action-count",
                "16",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        with zipfile.ZipFile(output) as archive:
            result = json.loads(archive.read("manifest.json"))
            output_rows = [
                json.loads(line)
                for name in archive.namelist()
                if name.endswith(".jsonl")
                for line in archive.read(name).decode("utf-8").splitlines()
            ]
        assert all(len(row["action"]) <= 16 for row in output_rows)
        assert result["trainer_compatibility_filter"] == {
            "max_action_count": 16,
            "rows_filtered_by_action_count": 1,
            "rows_filtered_by_action_count_by_split": {
                "train": 1,
                "valid": 0,
                "test": 0,
            },
        }
