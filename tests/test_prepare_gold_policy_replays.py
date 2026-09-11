from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import orjson
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "prepare_gold_policy_replays.py"


def load_tool():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "prepare_gold_policy_replays_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def deck_hash(deck: list[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(deck))
    return hashlib.sha256(canonical.encode()).hexdigest()


def replay(
    episode_id: int,
    team_names: list[str],
    decks: list[list[int]],
) -> dict:
    select = {
        "context": 7,
        "type": 1,
        "minCount": 1,
        "maxCount": 1,
        "option": [{"id": "left"}, {"id": "right"}],
    }
    setup = {"action": decks}
    return {
        "id": f"uuid-{episode_id}",
        "info": {
            "EpisodeId": episode_id,
            "TeamNames": team_names,
        },
        "rewards": [1, -1],
        "steps": [
            [
                {
                    "status": "ACTIVE",
                    "action": [],
                    "observation": {"select": select, "logs": []},
                    "visualize": [setup],
                },
                {
                    "status": "ACTIVE",
                    "action": [],
                    "observation": {"select": select, "logs": []},
                },
            ],
            [
                {
                    "status": "ACTIVE",
                    "action": [1],
                    "observation": {"select": None},
                },
                {
                    "status": "ACTIVE",
                    "action": [0],
                    "observation": {"select": None},
                },
            ],
        ],
    }


def episode_metadata(episode_id: int) -> dict:
    return {
        "id": episode_id,
        "createTime": "2026-07-26T00:00:00Z",
        "state": "COMPLETED",
        "type": "EPISODE_TYPE_PUBLIC",
        "agents": [
            {
                "submissionId": 111,
                "index": 0,
                "teamName": "Gold A",
            },
            {
                "submissionId": 222,
                "index": 1,
                "teamName": "Gold B",
            },
        ],
    }


def run_pipeline(tmp_path: Path) -> tuple[dict, Path]:
    deck_a = list(range(60))
    deck_b = list(range(100, 160))
    wrong_deck_a = list(range(200, 260))
    active = {
        "schema_version": "ptcg-active-submission-manifest-v1",
        "competition": "pokemon-tcg-ai-battle",
        "policies": [
            {
                "policy_id": "rank001-sub111",
                "submission_id": 111,
                "team_name": "Gold A",
                "deck_hash": deck_hash(deck_a),
                "archetype": "Marnie",
                "rank": 1,
                "score": 1200.0,
                "episodes": [
                    episode_metadata(episode_id)
                    for episode_id in range(101, 106)
                ],
            },
            {
                "policy_id": "rank002-sub222",
                "submission_id": 222,
                "team_name": "Gold B",
                "deck_hash": deck_hash(deck_b),
                "archetype": "Alakazam",
                "rank": 2,
                "score": 1190.0,
                "episodes": [
                    episode_metadata(episode_id)
                    for episode_id in range(101, 106)
                ],
            },
        ],
    }
    active_path = tmp_path / "active.json"
    active_path.write_text(json.dumps(active), encoding="utf-8")
    replay_zip = tmp_path / "episodes-2026-07-26.zip"
    with zipfile.ZipFile(replay_zip, "w") as archive:
        for episode_id in range(101, 106):
            payload = replay(
                episode_id,
                ["Gold A", "Gold B"],
                [
                    wrong_deck_a if episode_id == 105 else deck_a,
                    deck_b,
                ],
            )
            archive.writestr(
                f"{episode_id}.json",
                orjson.dumps(payload),
            )
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--active-manifest",
            str(active_path),
            "--input",
            str(replay_zip),
            "--output-root",
            str(output),
            "--split-seed",
            "17",
            "--holdout-fraction",
            "0.25",
            "--rows-per-shard",
            "2",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert '"policies": 2' in completed.stdout
    return json.loads((output / "manifest.json").read_text()), output


def test_exact_submission_seat_archives_and_contamination_audit(
    tmp_path: Path,
) -> None:
    manifest, output = run_pipeline(tmp_path)
    assert manifest["schema_version"] == "ptcg-gold-policy-clones-v1"
    assert manifest["global_audit"]["duplicate_episode_seat_assignments"] == {}
    assert manifest["global_audit"]["cross_policy_episode_pairs"]
    policies = {item["submission_id"]: item for item in manifest["policies"]}
    assert policies[111]["rank"] == 1
    assert policies[111]["score"] == 1200.0
    assert policies[111]["audit"]["accepted_episodes"] == 4
    assert policies[111]["audit"]["rejected_episodes"] == {
        "replay_deck_hash_mismatch": 1
    }
    assert policies[222]["audit"]["accepted_episodes"] == 5
    assert manifest["global_audit"]["verified_deck_count"] == 2

    for submission_id, expected_seat, expected_deck in (
        (111, 0, list(range(60))),
        (222, 1, list(range(100, 160))),
    ):
        policy = policies[submission_id]
        generated_deck = Path(policy["resolved_deck_path"])
        assert generated_deck.is_file()
        assert [
            int(line)
            for line in generated_deck.read_text().splitlines()
        ] == expected_deck
        assert policy["audit"]["verified_distinct_decklists"] == 1
        archive_path = Path(policy["archive_path"])
        assert archive_path.is_file()
        assert archive_path.parent == output / "policies"
        split_episodes: dict[str, set[str]] = {}
        with zipfile.ZipFile(archive_path) as archive:
            archive_manifest = json.loads(archive.read("manifest.json"))
            assert (
                archive_manifest["hidden_information_policy"]
                .startswith("Replay visualize is used only")
            )
            for split in ("train", "valid"):
                rows = []
                for member in archive.namelist():
                    if member.startswith(f"{split}/") and member.endswith(
                        ".jsonl"
                    ):
                        rows.extend(
                            orjson.loads(line)
                            for line in archive.read(member).splitlines()
                        )
                assert rows
                assert {row["seat"] for row in rows} == {expected_seat}
                assert {
                    row["source_submission_id"] for row in rows
                } == {submission_id}
                assert all("visualize" not in row for row in rows)
                assert all(
                    "visualize" not in row["observation"] for row in rows
                )
                split_episodes[split] = {
                    str(row["episode_id"]) for row in rows
                }
        assert not split_episodes["train"] & split_episodes["valid"]
        assert policy["audit"]["episode_split_overlap"] == 0
        assert policy["audit"]["identity_mismatches"] == 0
        assert policy["audit"]["foreign_seat_decisions"] == 0


def test_duplicate_submission_identity_is_rejected(tmp_path: Path) -> None:
    deck = list(range(60))
    policy = {
        "policy_id": "one",
        "submission_id": 111,
        "team_name": "Gold A",
        "deck_hash": deck_hash(deck),
        "episodes": [episode_metadata(101), episode_metadata(102)],
    }
    active = {"policies": [policy, {**policy, "policy_id": "two"}]}
    active_path = tmp_path / "active.json"
    active_path.write_text(json.dumps(active), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--active-manifest",
            str(active_path),
            "--input",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "assigned to multiple policies" in completed.stderr


@pytest.mark.parametrize("bad_id", ["../escape", "", "space id"])
def test_policy_id_cannot_escape_output_root(
    tmp_path: Path,
    bad_id: str,
) -> None:
    active = {
        "policies": [
            {
                "policy_id": bad_id,
                "submission_id": 111,
                "team_name": "Gold A",
                "deck_hash": deck_hash(list(range(60))),
                "episodes": [episode_metadata(101), episode_metadata(102)],
            }
        ]
    }
    active_path = tmp_path / "active.json"
    active_path.write_text(json.dumps(active), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--active-manifest",
            str(active_path),
            "--input",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "Invalid policy_id" in completed.stderr


def test_chunked_replay_download_does_not_require_content_length(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    payload = orjson.dumps(
        {
            "info": {"EpisodeId": 123},
            "steps": [],
            "rewards": [1, -1],
        }
    )

    class ChunkedResponse:
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, _chunk_size: int):
            yield payload[:7]
            yield b""
            yield payload[7:]

    output = tmp_path / "episode-123-replay.json"
    tool.write_replay_response_atomic(ChunkedResponse(), output, "123")
    assert orjson.loads(output.read_bytes())["info"]["EpisodeId"] == 123
    assert not list(tmp_path.glob("*.partial-*"))


def test_invalid_replay_download_is_not_published(tmp_path: Path) -> None:
    tool = load_tool()

    class InvalidResponse:
        def iter_content(self, _chunk_size: int):
            yield b"not-json"

    output = tmp_path / "episode-456-replay.json"
    with pytest.raises(orjson.JSONDecodeError):
        tool.write_replay_response_atomic(InvalidResponse(), output, "456")
    assert not output.exists()
    assert not list(tmp_path.glob("*.partial-*"))


def test_episode_candidate_bound_selects_newest_deterministically() -> None:
    tool = load_tool()
    indexes = {
        "policy": {
            "101": {"create_time": "2026-08-13T00:00:00Z"},
            "102": {"create_time": "2026-08-14T00:00:00Z"},
            "103": {"create_time": "2026-08-14T00:00:00Z"},
        }
    }

    bounded = tool.bound_episode_indexes_newest(indexes, 2)

    assert list(bounded["policy"]) == ["103", "102"]
    assert list(tool.bound_episode_indexes_newest(indexes, None)["policy"]) == [
        "101",
        "102",
        "103",
    ]
    with pytest.raises(ValueError, match="must be positive"):
        tool.bound_episode_indexes_newest(indexes, 0)
