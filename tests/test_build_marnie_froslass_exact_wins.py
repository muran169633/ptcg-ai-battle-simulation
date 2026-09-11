from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_marnie_froslass_exact_wins as builder


def _deck(start: int) -> list[int]:
    return list(range(start, start + 60))


def _deck_hash(deck: list[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(deck))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=builder.ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    return info


class Fixture:
    def __init__(
        self,
        root: Path,
        *,
        wrong_learner_deck: bool = False,
        wrong_replay_uuid: bool = False,
        wrong_replay_reward: bool = False,
        wrong_replay_seat_identity: bool = False,
    ) -> None:
        self.root = root
        self.source = root / "source.zip"
        self.daily_dir = root / "daily"
        self.daily_dir.mkdir()
        self.dates = ("2026-08-02", "2026-08-03")
        self.learner_deck = _deck(100)
        self.fros_deck = _deck(200)
        self.other_deck = _deck(300)
        self.learner_hash = _deck_hash(self.learner_deck)
        self.fros_hash = _deck_hash(self.fros_deck)
        self.rows = {
            "1001": [
                self._row("2026-08-02", "1001", 0, 1),
                self._row("2026-08-02", "1001", 0, 2),
            ],
            "1002": [self._row("2026-08-02", "1002", 1, 1)],
            "1003": [self._row("2026-08-03", "1003", 0, 1)],
        }
        self.source_lines = tuple(
            builder.canonical_json_bytes(row) + b"\n"
            for episode_id in ("1001", "1002", "1003")
            for row in self.rows[episode_id]
        )
        self._write_source()
        replay_learner_deck = (
            self.other_deck if wrong_learner_deck else self.learner_deck
        )
        replay_1001 = self._replay(
            "1001",
            (
                ("Opponent F", "Learner A")
                if wrong_replay_seat_identity
                else ("Learner A", "Opponent F")
            ),
            (-1, 1) if wrong_replay_reward else (1, -1),
            (replay_learner_deck, self.fros_deck),
        )
        if wrong_replay_uuid:
            replay_1001["id"] = "uuid-not-1001"
        self._write_daily(
            "2026-08-02",
            {
                "1001": replay_1001,
                "1002": self._replay(
                    "1002",
                    ("Opponent O", "Learner B"),
                    (-1, 1),
                    (self.other_deck, self.learner_deck),
                ),
            },
        )
        self._write_daily(
            "2026-08-03",
            {
                "1003": self._replay(
                    "1003",
                    ("Learner C", "Opponent F2"),
                    (1, -1),
                    (self.learner_deck, self.fros_deck),
                )
            },
        )
        self.contract = builder.BuildContract(
            source_sha256=builder.sha256_file(self.source),
            source_manifest_sha256=self.source_manifest_sha256,
            daily_sha256=tuple(
                (
                    date,
                    builder.sha256_file(
                        self.daily_dir
                        / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
                    ),
                )
                for date in self.dates
            ),
            daily_manifest_sha256=tuple(
                (
                    date,
                    self._daily_manifest_sha256(date),
                )
                for date in self.dates
            ),
            daily_missing_replay_ids=tuple((date, ()) for date in self.dates),
            dates=self.dates,
            expected_date_episodes=(
                ("2026-08-02", 1),
                ("2026-08-03", 1),
            ),
            expected_source_train_rows=4,
            expected_source_train_episodes=3,
            expected_selected_rows=3,
            expected_selected_episodes=2,
            learner_deck_hash=self.learner_hash,
            opponent_deck_hash=self.fros_hash,
            source_members=("train/part-00000.jsonl", "manifest.json"),
            source_train_members=("train/part-00000.jsonl",),
            source_logical_path="fixture/source.zip",
            daily_logical_dir="fixture/daily",
            incomplete_terminal_replays=(),
        )

    def _daily_manifest_sha256(self, dataset_date: str) -> str:
        path = (
            self.daily_dir
            / f"pokemon-tcg-ai-battle-episodes-{dataset_date}.zip"
        )
        with zipfile.ZipFile(path) as archive:
            return hashlib.sha256(archive.read("manifest.csv")).hexdigest()

    def _row(
        self,
        dataset_date: str,
        episode_id: str,
        seat: int,
        action_step: int,
    ) -> dict[str, object]:
        learner_name = {
            "1001": "Learner A",
            "1002": "Learner B",
            "1003": "Learner C",
        }[episode_id]
        opponent_name = {
            "1001": "Opponent F",
            "1002": "Opponent O",
            "1003": "Opponent F2",
        }[episode_id]
        return {
            "schema_version": builder.ROW_SCHEMA_VERSION,
            "episode_id": episode_id,
            "episode_uuid": f"uuid-{episode_id}",
            "dataset_date": dataset_date,
            "split": "train",
            "observation_step_index": action_step - 1,
            "action_step_index": action_step,
            "seat": seat,
            "team_name": learner_name,
            "opponent_team_name": opponent_name,
            "deck_hash": self.learner_hash,
            "terminal_reward": 1.0,
            "sample_weight": 1.0,
            "select_context": "0",
            "action": [0],
            "observation": {"select": {"option": [{"type": 1}]}},
        }

    @staticmethod
    def _replay(
        episode_id: str,
        names: tuple[str, str],
        rewards: tuple[int, int],
        decks: tuple[list[int], list[int]],
    ) -> dict[str, object]:
        return {
            "id": f"uuid-{episode_id}",
            "info": {
                "EpisodeId": int(episode_id),
                "TeamNames": list(names),
            },
            "rewards": list(rewards),
            "steps": [
                [
                    {
                        "visualize": [
                            {"action": [list(decks[0]), list(decks[1])]}
                        ]
                    },
                    {},
                ]
            ],
        }

    def _write_source(self) -> None:
        manifest = {
            "schema_version": builder.SOURCE_SCHEMA_VERSION,
            "profile": {"deck_hash": self.learner_hash},
            "deck_hash_filter": self.learner_hash,
            "split_policy": {"train_dates": list(self.dates)},
            "split_decisions": {"train": 4},
            "split_episodes": {"train": 3},
            "terminal_reward_filter": {"mode": "wins", "splits": "train"},
        }
        manifest_payload = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        self.source_manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        with zipfile.ZipFile(
            self.source,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(
                _zip_info("train/part-00000.jsonl"),
                b"".join(self.source_lines),
            )
            archive.writestr(_zip_info("manifest.json"), manifest_payload)

    def _write_daily(
        self,
        dataset_date: str,
        replays: dict[str, dict[str, object]],
    ) -> None:
        path = (
            self.daily_dir
            / f"pokemon-tcg-ai-battle-episodes-{dataset_date}.zip"
        )
        manifest = "episode_id,create_time,avg_score,min_score,sum_score,agent_count,size_bytes\n"
        for episode_id in sorted(replays, key=int):
            manifest += (
                f"{episode_id},{dataset_date}T00:00:00,1,1,2,2,100\n"
            )
        with zipfile.ZipFile(
            path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for episode_id in sorted(replays, key=int):
                archive.writestr(
                    _zip_info(f"{episode_id}.json"),
                    builder.canonical_json_bytes(replays[episode_id]),
                )
            archive.writestr(_zip_info("manifest.csv"), manifest.encode("utf-8"))

    @property
    def selected_raw_payload(self) -> bytes:
        return b"".join(
            builder.canonical_json_bytes(row) + b"\n"
            for episode_id in ("1001", "1003")
            for row in self.rows[episode_id]
        )


def rewrite_daily_archive(
    fixture: Fixture,
    dataset_date: str,
    *,
    append_manifest_id: str | None = None,
    add_numeric_id: str | None = None,
    remove_numeric_id: str | None = None,
    replace_rewards: tuple[str, list[float | None]] | None = None,
) -> None:
    path = (
        fixture.daily_dir
        / f"pokemon-tcg-ai-battle-episodes-{dataset_date}.zip"
    )
    with zipfile.ZipFile(path) as archive:
        members = [(name, archive.read(name)) for name in archive.namelist()]
    rewritten: list[tuple[str, bytes]] = []
    for name, payload in members:
        if remove_numeric_id is not None and name == f"{remove_numeric_id}.json":
            continue
        if name == "manifest.csv" and append_manifest_id is not None:
            payload += (
                f"{append_manifest_id},{dataset_date}T00:00:00,1,1,2,2,100\n"
            ).encode()
        if replace_rewards is not None and name == f"{replace_rewards[0]}.json":
            replay = json.loads(payload)
            replay["rewards"] = replace_rewards[1]
            payload = builder.canonical_json_bytes(replay)
        rewritten.append((name, payload))
    if add_numeric_id is not None:
        rewritten.append((f"{add_numeric_id}.json", b"{}"))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in rewritten:
            archive.writestr(_zip_info(name), payload)


def refreshed_daily_contract(
    fixture: Fixture,
    *,
    missing_by_date: dict[str, tuple[str, ...]] | None = None,
) -> builder.BuildContract:
    missing_by_date = missing_by_date or {}
    return dataclasses.replace(
        fixture.contract,
        daily_sha256=tuple(
            (
                date,
                builder.sha256_file(
                    fixture.daily_dir
                    / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
                ),
            )
            for date in fixture.dates
        ),
        daily_manifest_sha256=tuple(
            (date, fixture._daily_manifest_sha256(date)) for date in fixture.dates
        ),
        daily_missing_replay_ids=tuple(
            (date, missing_by_date.get(date, ())) for date in fixture.dates
        ),
    )


def frozen_incomplete_exception(
    fixture: Fixture,
    dataset_date: str,
    episode_id: str,
    *,
    raw_rewards: tuple[float | None, float | None],
    opponent_deck_hash: str,
) -> builder.IncompleteTerminalReplayContract:
    path = (
        fixture.daily_dir
        / f"pokemon-tcg-ai-battle-episodes-{dataset_date}.zip"
    )
    with zipfile.ZipFile(path) as archive:
        replay_payload = archive.read(f"{episode_id}.json")
    row = fixture.rows[episode_id][0]
    return builder.IncompleteTerminalReplayContract(
        dataset_date=dataset_date,
        episode_id=episode_id,
        episode_uuid=str(row["episode_uuid"]),
        replay_sha256=hashlib.sha256(replay_payload).hexdigest(),
        seat=int(row["seat"]),
        team_name=str(row["team_name"]),
        opponent_team_name=str(row["opponent_team_name"]),
        source_terminal_reward=float(row["terminal_reward"]),
        raw_rewards=raw_rewards,
        learner_deck_hash=fixture.learner_hash,
        opponent_deck_hash=opponent_deck_hash,
    )


class BuildMarnieFroslassExactWinsTests(unittest.TestCase):
    def test_exact_incomplete_terminal_non_target_is_skipped_and_fully_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            rewrite_daily_archive(
                fixture,
                "2026-08-02",
                replace_rewards=("1002", [None, 1.0]),
            )
            exception = frozen_incomplete_exception(
                fixture,
                "2026-08-02",
                "1002",
                raw_rewards=(None, 1.0),
                opponent_deck_hash=_deck_hash(fixture.other_deck),
            )
            contract = dataclasses.replace(
                refreshed_daily_contract(fixture),
                incomplete_terminal_replays=(exception,),
            )
            audit, archive_payload, metadata = builder.audit_and_serialize(
                fixture.source,
                fixture.daily_dir,
                contract,
            )
            self.assertEqual([value.episode_id for value in audit.selected], ["1001", "1003"])
            expected_entry = {
                "dataset_date": exception.dataset_date,
                "episode_id": exception.episode_id,
                "episode_uuid": exception.episode_uuid,
                "replay_sha256": exception.replay_sha256,
                "seat": exception.seat,
                "team_name": exception.team_name,
                "opponent_team_name": exception.opponent_team_name,
                "source_terminal_reward": exception.source_terminal_reward,
                "raw_rewards": list(exception.raw_rewards),
                "learner_deck_hash": exception.learner_deck_hash,
                "opponent_deck_hash": exception.opponent_deck_hash,
                "target_route": False,
            }
            self.assertEqual(
                metadata["plan"]["incomplete_terminal_replays"],
                [expected_entry],
            )
            with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
                manifest = json.loads(archive.read(builder.MANIFEST_MEMBER))
            self.assertEqual(
                manifest["replay_audit"]["incomplete_terminal_replays"],
                [expected_entry],
            )

    def test_incomplete_terminal_drift_target_and_unlisted_null_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def prepared(label: str) -> tuple[Fixture, builder.BuildContract, object]:
                case_root = root / label
                case_root.mkdir()
                fixture = Fixture(case_root)
                rewrite_daily_archive(
                    fixture,
                    "2026-08-02",
                    replace_rewards=("1002", [None, 1.0]),
                )
                exception = frozen_incomplete_exception(
                    fixture,
                    "2026-08-02",
                    "1002",
                    raw_rewards=(None, 1.0),
                    opponent_deck_hash=_deck_hash(fixture.other_deck),
                )
                contract = dataclasses.replace(
                    refreshed_daily_contract(fixture),
                    incomplete_terminal_replays=(exception,),
                )
                return fixture, contract, exception

            for label, mutation, error in (
                (
                    "sha-drift",
                    lambda value: dataclasses.replace(
                        value, replay_sha256="0" * 64
                    ),
                    "incomplete replay SHA drift",
                ),
                (
                    "identity-drift",
                    lambda value: dataclasses.replace(value, team_name="Wrong Learner"),
                    "incomplete source identity drift",
                ),
                (
                    "deck-drift",
                    lambda value: dataclasses.replace(
                        value, opponent_deck_hash="f" * 64
                    ),
                    "incomplete deck-pair drift",
                ),
            ):
                fixture, contract, exception = prepared(label)
                contract = dataclasses.replace(
                    contract,
                    incomplete_terminal_replays=(mutation(exception),),
                )
                with self.subTest(label=label):
                    with self.assertRaisesRegex(RuntimeError, error):
                        builder.build(
                            source=fixture.source,
                            daily_dir=fixture.daily_dir,
                            output=fixture.root / "never" / "archive.zip",
                            contract=contract,
                            execute=False,
                        )

            fixture, contract, _exception = prepared("raw-drift")
            rewrite_daily_archive(
                fixture,
                "2026-08-02",
                replace_rewards=("1002", [None, 0.0]),
            )
            contract = dataclasses.replace(
                refreshed_daily_contract(fixture),
                incomplete_terminal_replays=contract.incomplete_terminal_replays,
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete raw rewards drift"):
                builder.build(
                    source=fixture.source,
                    daily_dir=fixture.daily_dir,
                    output=fixture.root / "never" / "archive.zip",
                    contract=contract,
                    execute=False,
                )

            complete_root = root / "exact-set-drift"
            complete_root.mkdir()
            complete_fixture = Fixture(complete_root)
            stale_exception = builder.IncompleteTerminalReplayContract(
                dataset_date="2026-08-02",
                episode_id="1002",
                episode_uuid="uuid-1002",
                replay_sha256="0" * 64,
                seat=1,
                team_name="Learner B",
                opponent_team_name="Opponent O",
                source_terminal_reward=1.0,
                raw_rewards=(None, 1.0),
                learner_deck_hash=complete_fixture.learner_hash,
                opponent_deck_hash=_deck_hash(complete_fixture.other_deck),
            )
            stale_contract = dataclasses.replace(
                complete_fixture.contract,
                incomplete_terminal_replays=(stale_exception,),
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete-terminal set drift"):
                builder.build(
                    source=complete_fixture.source,
                    daily_dir=complete_fixture.daily_dir,
                    output=complete_fixture.root / "never" / "archive.zip",
                    contract=stale_contract,
                    execute=False,
                )

            for label, episode_id, rewards in (
                ("unlisted-null", "1002", [None, 1.0]),
                ("target-null", "1001", [1.0, None]),
            ):
                case_root = root / label
                case_root.mkdir()
                fixture = Fixture(case_root)
                rewrite_daily_archive(
                    fixture,
                    "2026-08-02",
                    replace_rewards=(episode_id, rewards),
                )
                contract = refreshed_daily_contract(fixture)
                with self.subTest(label=label):
                    with self.assertRaisesRegex(
                        RuntimeError, "unexpected incomplete terminal rewards"
                    ):
                        builder.build(
                            source=fixture.source,
                            daily_dir=fixture.daily_dir,
                            output=case_root / "never" / "archive.zip",
                            contract=contract,
                            execute=False,
                        )

    def test_frozen_daily_manifest_hashes_and_missing_sets_are_exact(self) -> None:
        self.assertEqual(
            dict(builder.DEFAULT_CONTRACT.daily_missing_replay_ids),
            {
                "2026-08-02": ("89488639",),
                "2026-08-03": (
                    "89719318",
                    "89721535",
                    "89733649",
                    "89757221",
                ),
                "2026-08-04": (
                    "89855212",
                    "89919753",
                    "90002731",
                    "90035083",
                    "90041269",
                ),
                "2026-08-05": ("90144162", "90161497", "90197034"),
                "2026-08-06": ("90393929", "90481043"),
            },
        )
        self.assertEqual(
            dict(builder.DEFAULT_CONTRACT.daily_manifest_sha256),
            {
                "2026-08-02": "0d63fea5c93db6458a856c332aeb128b1098bcda44ef35a3a148236f47c389e4",
                "2026-08-03": "ab80af203ef7958503ab5f244bee8adc377eb3b8da71504b7c83e348ffa73991",
                "2026-08-04": "bb190f62f0585dc2a1db2b02752a4d7e6fa6de15a800ed9e769d8daecd8bf9a1",
                "2026-08-05": "b8645318d7b20792dff4f327d5aaeadd98b09fc6064fdcceb5d92897cee7ac11",
                "2026-08-06": "96392a60c27d98e6be7ea45427228ac2a3a9c35b60c4f69f78cf3408018515a3",
            },
        )

    def test_unknown_missing_extra_and_selected_missing_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                (
                    "unknown-missing",
                    {"append_manifest_id": "1999"},
                    {},
                    "exact missing replay set drift",
                ),
                (
                    "extra-member",
                    {"add_numeric_id": "1999"},
                    {},
                    "replay members absent from manifest",
                ),
                (
                    "selected-missing",
                    {"remove_numeric_id": "1001"},
                    {"2026-08-02": ("1001",)},
                    "intersects source-selected episode",
                ),
            )
            for label, mutation, missing_by_date, error in cases:
                case_root = root / label
                case_root.mkdir()
                fixture = Fixture(case_root)
                rewrite_daily_archive(fixture, "2026-08-02", **mutation)
                contract = refreshed_daily_contract(
                    fixture,
                    missing_by_date=missing_by_date,
                )
                with self.subTest(label=label):
                    with self.assertRaisesRegex(RuntimeError, error):
                        builder.build(
                            source=fixture.source,
                            daily_dir=fixture.daily_dir,
                            output=case_root / "never" / "archive.zip",
                            contract=contract,
                            execute=False,
                        )
                    self.assertFalse((case_root / "never").exists())

    def test_dry_run_is_write_free_and_audits_exact_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "new-output" / "archive.zip"
            source_before = fixture.source.read_bytes()
            daily_before = {
                path.name: path.read_bytes()
                for path in fixture.daily_dir.glob("*.zip")
            }

            result = builder.build(
                source=fixture.source,
                daily_dir=fixture.daily_dir,
                output=output,
                contract=fixture.contract,
                execute=False,
            )

            self.assertEqual(result["status"], "dry_run_passed")
            self.assertFalse(result["output_written"])
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())
            self.assertEqual(result["source_train_episodes"], 3)
            self.assertEqual(result["source_train_rows"], 4)
            self.assertEqual(result["selected_episodes"], 2)
            self.assertEqual(result["selected_rows"], 3)
            self.assertEqual(
                result["selected_content_sha256"],
                hashlib.sha256(fixture.selected_raw_payload).hexdigest(),
            )
            self.assertTrue(result["deterministic_rebuild_match"])
            self.assertEqual(fixture.source.read_bytes(), source_before)
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in fixture.daily_dir.glob("*.zip")
                },
                daily_before,
            )

    def test_reviewed_execute_builds_two_byte_identical_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            dry_output = root / "dry" / "archive.zip"
            dry = builder.build(
                source=fixture.source,
                daily_dir=fixture.daily_dir,
                output=dry_output,
                contract=fixture.contract,
                execute=False,
            )
            outputs = [
                root / "first-bundle" / "archive.zip",
                root / "second-bundle" / "archive.zip",
            ]
            results = []
            for output in outputs:
                results.append(
                    builder.build(
                        source=fixture.source,
                        daily_dir=fixture.daily_dir,
                        output=output,
                        contract=fixture.contract,
                        execute=True,
                        expected_plan_sha256=dry["plan_sha256"],
                        expected_output_sha256=dry["archive_sha256"],
                        expected_allowlist_sha256=dry["allowlist_sha256"],
                    )
                )

            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            self.assertEqual(
                (outputs[0].parent / builder.ALLOWLIST_MEMBER).read_bytes(),
                (outputs[1].parent / builder.ALLOWLIST_MEMBER).read_bytes(),
            )
            self.assertEqual(results[0]["archive_sha256"], dry["archive_sha256"])
            with zipfile.ZipFile(outputs[0]) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        builder.TRAIN_MEMBER,
                        builder.ALLOWLIST_MEMBER,
                        builder.MANIFEST_MEMBER,
                    ],
                )
                train_payload = archive.read(builder.TRAIN_MEMBER)
                allowlist_payload = archive.read(builder.ALLOWLIST_MEMBER)
                manifest_payload = archive.read(builder.MANIFEST_MEMBER)
            self.assertEqual(train_payload, fixture.selected_raw_payload)
            allowlist = json.loads(allowlist_payload)
            manifest = json.loads(manifest_payload)
            self.assertEqual(
                allowlist_payload,
                builder.canonical_json_bytes(allowlist) + b"\n",
            )
            self.assertEqual(
                manifest_payload,
                builder.canonical_json_bytes(manifest) + b"\n",
            )
            self.assertEqual(
                hashlib.sha256(allowlist_payload).hexdigest(),
                dry["allowlist_sha256"],
            )
            self.assertEqual(allowlist["stats"]["episodes"], 2)
            self.assertEqual(allowlist["stats"]["rows"], 3)
            self.assertEqual(
                allowlist["schema_version"],
                builder.ALLOWLIST_SCHEMA_VERSION,
            )
            self.assertEqual(allowlist["split"], "train")
            self.assertEqual(allowlist["terminal_reward"], "win")
            self.assertEqual(allowlist["episode_count"], 2)
            self.assertEqual(allowlist["decision_rows"], 3)
            self.assertEqual(
                allowlist["date_episode_counts"],
                {"2026-08-02": 1, "2026-08-03": 1},
            )
            self.assertEqual(
                {row["episode_id"] for row in allowlist["episodes"]},
                {"1001", "1003"},
            )
            self.assertEqual(
                [(row["date"], row["episode_id"], row["decision_rows"])
                 for row in allowlist["episodes"]],
                [
                    ("2026-08-02", "1001", 2),
                    ("2026-08-03", "1003", 1),
                ],
            )
            self.assertEqual(manifest["split_episodes"], {"train": 2})
            self.assertEqual(manifest["split_decisions"], {"train": 3})
            self.assertEqual(
                manifest["data_schema_version"],
                builder.ROW_SCHEMA_VERSION,
            )
            exact_allowlist = manifest["exact_episode_allowlist"]
            self.assertEqual(exact_allowlist["sha256"], dry["allowlist_sha256"])
            self.assertEqual(
                exact_allowlist["canonical_sha256"],
                dry["allowlist_canonical_sha256"],
            )
            self.assertEqual(
                exact_allowlist["sha256"],
                exact_allowlist["canonical_sha256"],
            )
            self.assertTrue(manifest["byte_preservation"]["selected_rows_copied_verbatim"])
            self.assertTrue(manifest["replay_audit"]["both_deck_hashes_verified"])

    def test_execute_requires_reviewed_plan_and_output_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            with self.assertRaisesRegex(RuntimeError, "requires"):
                builder.build(
                    source=fixture.source,
                    daily_dir=fixture.daily_dir,
                    output=root / "never-bundle" / "never.zip",
                    contract=fixture.contract,
                    execute=True,
                )

    def test_wrong_reviewed_plan_or_output_hash_never_publishes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            dry = builder.build(
                source=fixture.source,
                daily_dir=fixture.daily_dir,
                output=root / "dry-bundle" / "archive.zip",
                contract=fixture.contract,
                execute=False,
            )
            cases = (
                ("wrong-plan", "0" * 64, dry["archive_sha256"]),
                ("wrong-output", dry["plan_sha256"], "1" * 64),
            )
            for name, plan_sha, output_sha in cases:
                bundle = root / name
                with self.subTest(name=name):
                    with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                        builder.build(
                            source=fixture.source,
                            daily_dir=fixture.daily_dir,
                            output=bundle / "archive.zip",
                            contract=fixture.contract,
                            execute=True,
                            expected_plan_sha256=plan_sha,
                            expected_output_sha256=output_sha,
                            expected_allowlist_sha256=dry["allowlist_sha256"],
                        )
                    self.assertFalse(bundle.exists())

    def test_count_drift_fails_closed_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "never" / "archive.zip"
            contract = dataclasses.replace(
                fixture.contract,
                expected_selected_rows=4,
            )
            with self.assertRaisesRegex(RuntimeError, "selected row-count drift"):
                builder.build(
                    source=fixture.source,
                    daily_dir=fixture.daily_dir,
                    output=output,
                    contract=contract,
                    execute=False,
                )
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())

    def test_learner_deck_drift_in_official_replay_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root, wrong_learner_deck=True)
            output = root / "never-bundle" / "never.zip"
            with self.assertRaisesRegex(RuntimeError, "learner deck mismatch"):
                builder.build(
                    source=fixture.source,
                    daily_dir=fixture.daily_dir,
                    output=output,
                    contract=fixture.contract,
                    execute=False,
                )
            self.assertFalse(output.exists())

    def test_official_replay_uuid_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root, wrong_replay_uuid=True)
            output = root / "never-bundle" / "never.zip"
            with self.assertRaisesRegex(RuntimeError, "episode UUID mismatch"):
                builder.build(
                    source=fixture.source,
                    daily_dir=fixture.daily_dir,
                    output=output,
                    contract=fixture.contract,
                    execute=False,
                )
            self.assertFalse(output.parent.exists())

    def test_official_replay_reward_and_seat_identity_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, kwargs, error in (
                ("reward", {"wrong_replay_reward": True}, "learner reward mismatch"),
                (
                    "seat",
                    {"wrong_replay_seat_identity": True},
                    "team identity mismatch",
                ),
            ):
                case_root = root / label
                case_root.mkdir()
                fixture = Fixture(case_root, **kwargs)
                output = case_root / "never-bundle" / "never.zip"
                with self.subTest(label=label):
                    with self.assertRaisesRegex(RuntimeError, error):
                        builder.build(
                            source=fixture.source,
                            daily_dir=fixture.daily_dir,
                            output=output,
                            contract=fixture.contract,
                            execute=False,
                        )
                    self.assertFalse(output.parent.exists())

    def test_duplicate_source_zip_member_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(fixture.source, mode="a") as archive:
                    archive.writestr(
                        _zip_info("train/part-00000.jsonl"),
                        b"duplicate\n",
                    )
            contract = dataclasses.replace(
                fixture.contract,
                source_sha256=builder.sha256_file(fixture.source),
            )
            output = root / "never-bundle" / "never.zip"
            with self.assertRaisesRegex(RuntimeError, "duplicate member names"):
                builder.build(
                    source=fixture.source,
                    daily_dir=fixture.daily_dir,
                    output=output,
                    contract=contract,
                    execute=False,
                )
            self.assertFalse(output.parent.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "existing.zip"
            output.write_bytes(b"keep")
            with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
                builder.build(
                    source=fixture.source,
                    daily_dir=fixture.daily_dir,
                    output=output,
                    contract=fixture.contract,
                    execute=False,
                )
            self.assertEqual(output.read_bytes(), b"keep")

    def test_source_symlink_is_rejected_before_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            source_link = root / "source-link.zip"
            source_link.symlink_to(fixture.source)
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                builder.build(
                    source=source_link,
                    daily_dir=fixture.daily_dir,
                    output=root / "never.zip",
                    contract=fixture.contract,
                    execute=False,
                )

    def test_output_symlink_ancestor_is_rejected_before_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            real_parent = root / "real-output"
            real_parent.mkdir()
            linked_parent = root / "linked-output"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                builder.build(
                    source=fixture.source,
                    daily_dir=fixture.daily_dir,
                    output=linked_parent / "archive.zip",
                    contract=fixture.contract,
                    execute=False,
                )
            self.assertFalse((real_parent / "archive.zip").exists())


if __name__ == "__main__":
    unittest.main()
