from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
TOOL = TOOLS / "profile_bc_exact_decks.py"
sys.path.insert(0, str(TOOLS))

import profile_bc_exact_decks as profiler  # noqa: E402


DECK_A = "a" * 64
DECK_B = "b" * 64
SCHEMA = "ptcg-bc-visible-decisions-v1"


def row(
    *,
    split: str,
    dataset_date: str,
    episode_id: str,
    seat: int,
    deck_hash: str,
    team_name: str,
    terminal_reward: float,
    action_step_index: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "split": split,
        "dataset_date": dataset_date,
        "episode_id": episode_id,
        "episode_uuid": f"uuid-{dataset_date}-{episode_id}",
        "seat": seat,
        "deck_hash": deck_hash,
        "team_name": team_name,
        "terminal_reward": terminal_reward,
        "action_step_index": action_step_index,
        "observation": {"select": {"option": []}},
        "action": [],
    }


def write_archive(path: Path, rows_by_member: dict[str, list[dict]]) -> int:
    decision_rows = sum(len(rows) for rows in rows_by_member.values())
    manifest = {
        "schema_version": SCHEMA,
        "competition": "pokemon-tcg-ai-battle",
        "stats": {"decisions": decision_rows},
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for member, rows in rows_by_member.items():
            archive.writestr(
                member,
                "".join(json.dumps(item) + "\n" for item in rows),
            )
    return decision_rows


def fixture_rows() -> dict[str, list[dict]]:
    return {
        "train/part-00000.jsonl": [
            row(
                split="train",
                dataset_date="2026-08-03",
                episode_id="episode-win",
                seat=0,
                deck_hash=DECK_A,
                team_name="Team Alpha",
                terminal_reward=1.0,
                action_step_index=1,
            ),
            row(
                split="train",
                dataset_date="2026-08-03",
                episode_id="episode-win",
                seat=0,
                deck_hash=DECK_A,
                team_name="Team Alpha",
                terminal_reward=1.0,
                action_step_index=2,
            ),
            row(
                split="train",
                dataset_date="2026-08-03",
                episode_id="episode-loss",
                seat=1,
                deck_hash=DECK_A,
                team_name="Team Alpha",
                terminal_reward=-1.0,
                action_step_index=1,
            ),
            row(
                split="train",
                dataset_date="2026-08-03",
                episode_id="small-deck-win",
                seat=1,
                deck_hash=DECK_B,
                team_name="Team Gamma",
                terminal_reward=1.0,
                action_step_index=1,
            ),
            row(
                split="train",
                dataset_date="2026-08-03",
                episode_id="small-deck-win",
                seat=1,
                deck_hash=DECK_B,
                team_name="Team Gamma",
                terminal_reward=1.0,
                action_step_index=2,
            ),
        ],
        "valid/part-00000.jsonl": [
            row(
                split="valid",
                dataset_date="2026-08-04",
                episode_id="episode-draw",
                seat=0,
                deck_hash=DECK_A,
                team_name="Team Beta",
                terminal_reward=0.0,
                action_step_index=1,
            ),
            row(
                split="valid",
                dataset_date="2026-08-04",
                episode_id="episode-draw",
                seat=0,
                deck_hash=DECK_A,
                team_name="Team Beta",
                terminal_reward=0.0,
                action_step_index=2,
            ),
            row(
                split="valid",
                dataset_date="2026-08-04",
                episode_id="episode-draw",
                seat=0,
                deck_hash=DECK_A,
                team_name="Team Beta",
                terminal_reward=0.0,
                action_step_index=3,
            ),
        ],
    }


def test_profiles_episode_outcomes_once_and_decision_rows_exactly() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / "bc.zip"
        total_rows = write_archive(archive, fixture_rows())
        before = archive.read_bytes()

        report = profiler.profile_archive(archive, min_episodes=2)

        assert archive.read_bytes() == before
        assert report["deduplication_key"] == [
            "dataset_date",
            "episode_id",
            "seat",
        ]
        assert report["profiled_splits"] == ["train", "valid", "test"]
        assert report["sealed_splits"] == []
        assert report["split_date_counts"] == {
            "train": {"2026-08-03": 3},
            "valid": {"2026-08-04": 1},
            "test": {},
        }
        assert report["statistics"] == {
            "decision_rows": total_rows,
            "unique_episode_seats": 4,
            "exact_decks": 2,
            "selected_decision_rows": 6,
            "selected_episode_seats": 3,
            "selected_exact_decks": 1,
            "filtered_exact_decks": 1,
        }
        assert report["source"]["manifest_decision_rows_match"] is True
        assert report["source"]["profiled_jsonl_members"] == 2
        assert report["source"]["sealed_jsonl_members"] == 0
        assert len(report["decks"]) == 1
        deck = report["decks"][0]
        assert deck == {
            "deck_hash": DECK_A,
            "episodes": 3,
            "wins": 1,
            "losses": 1,
            "draws": 1,
            "win_rate": 1 / 3,
            "decision_rows": 6,
            "split_stats": {
                "train": {
                    "episodes": 2,
                    "wins": 1,
                    "losses": 1,
                    "draws": 0,
                    "win_rate": 0.5,
                    "team_count": 1,
                    "team_counts": {"Team Alpha": 2},
                    "date_counts": {"2026-08-03": 2},
                },
                "valid": {
                    "episodes": 1,
                    "wins": 0,
                    "losses": 0,
                    "draws": 1,
                    "win_rate": 0.0,
                    "team_count": 1,
                    "team_counts": {"Team Beta": 1},
                    "date_counts": {"2026-08-04": 1},
                },
                "test": {
                    "episodes": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "win_rate": None,
                    "team_count": 0,
                    "team_counts": {},
                    "date_counts": {},
                },
            },
            "seat_counts": {"0": 2, "1": 1},
            "date_counts": {"2026-08-03": 2, "2026-08-04": 1},
            "team_counts": {"Team Alpha": 2, "Team Beta": 1},
        }


def test_cli_atomic_output_refuses_overwrite_then_allows_it() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        archive = root / "bc.zip"
        output = root / "profile.json"
        write_archive(archive, fixture_rows())
        archive_before = archive.read_bytes()
        output.write_text("keep", encoding="utf-8")

        refused = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--input",
                str(archive),
                "--output",
                str(output),
                "--min-episodes",
                "2",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert refused.returncode != 0
        assert "pass --overwrite" in refused.stderr
        assert output.read_text(encoding="utf-8") == "keep"

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--input",
                str(archive),
                "--output",
                str(output),
                "--min-episodes",
                "2",
                "--overwrite",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["statistics"]["selected_exact_decks"] == 1
        assert payload["decks"][0]["deck_hash"] == DECK_A
        assert archive.read_bytes() == archive_before
        assert not list(root.glob(".profile.json.*.tmp"))


def test_rejects_conflicting_identity_without_creating_output() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        archive = root / "conflict.zip"
        output = root / "profile.json"
        first = row(
            split="train",
            dataset_date="2026-08-03",
            episode_id="same",
            seat=0,
            deck_hash=DECK_A,
            team_name="Team Alpha",
            terminal_reward=1.0,
            action_step_index=1,
        )
        conflicting = dict(first)
        conflicting["deck_hash"] = DECK_B
        conflicting["action_step_index"] = 2
        write_archive(
            archive,
            {"train/part-00000.jsonl": [first, conflicting]},
        )

        with pytest.raises(ValueError, match="inconsistent rows for episode key"):
            profiler.run(
                archive,
                output,
                min_episodes=1,
                overwrite=False,
            )
        assert not output.exists()
        assert not list(root.glob(".profile.json.*.tmp"))


def test_rejects_conflicting_split_for_episode_identity() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        archive = root / "split-conflict.zip"
        output = root / "profile.json"
        first = row(
            split="train",
            dataset_date="2026-08-03",
            episode_id="same",
            seat=0,
            deck_hash=DECK_A,
            team_name="Team Alpha",
            terminal_reward=1.0,
            action_step_index=1,
        )
        conflicting = dict(first)
        conflicting["split"] = "valid"
        conflicting["action_step_index"] = 2
        write_archive(
            archive,
            {
                "train/part-00000.jsonl": [first],
                "valid/part-00000.jsonl": [conflicting],
            },
        )

        with pytest.raises(ValueError, match="inconsistent rows for episode key"):
            profiler.run(
                archive,
                output,
                min_episodes=1,
                overwrite=False,
            )
        assert not output.exists()


def test_rejects_nonpositive_minimum() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / "bc.zip"
        write_archive(archive, fixture_rows())
        with pytest.raises(ValueError, match="at least 1"):
            profiler.profile_archive(archive, min_episodes=0)


def test_partial_profile_cli_never_opens_sealed_test_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        archive = root / "bc.zip"
        output = root / "profile.json"
        rows = fixture_rows()
        rows["test/part-00000.jsonl"] = [
            row(
                split="test",
                dataset_date="2026-08-05",
                episode_id="sealed-test",
                seat=1,
                deck_hash=DECK_A,
                team_name="Sealed Team",
                terminal_reward=1.0,
                action_step_index=1,
            )
        ]
        write_archive(archive, rows)

        real_open = zipfile.ZipFile.open

        def guarded_open(self, name, *args, **kwargs):
            filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
            if str(filename).startswith("test/"):
                raise AssertionError("sealed test member was opened")
            return real_open(self, name, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "open", guarded_open)
        return_code = profiler.main(
            [
                "--input",
                str(archive),
                "--output",
                str(output),
                "--profile-splits",
                "train",
                "valid",
            ]
        )

        assert return_code == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["profiled_splits"] == ["train", "valid"]
        assert report["sealed_splits"] == ["test"]
        assert "test" not in report["split_date_counts"]
        assert report["source"]["profiled_jsonl_members"] == 2
        assert report["source"]["sealed_jsonl_members"] == 1
        assert report["source"]["manifest_decision_rows_match"] is None
        assert "Not compared" in report["source"][
            "manifest_decision_rows_match_note"
        ]
        assert report["decks"][0]["split_stats"]["test"] == {
            "profiled": False
        }
