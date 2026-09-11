from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import build_marnie_froslass_exact_wins as fros_builder  # noqa: E402
import run_gold_push_exact_fros_bc_backup as retired  # noqa: E402
import run_gold_push_s08_antikd_bc_repair as repair  # noqa: E402
from test_build_marnie_exact_anti_kd import AntiFixture  # noqa: E402
from test_build_marnie_froslass_exact_wins import Fixture as FrosFixture  # noqa: E402
from test_build_marnie_general_anchor_exclusion import GeneralFixture  # noqa: E402


class ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Linear(3, 3)
        self.actor_query = torch.nn.Linear(3, 3, bias=False)
        self.actor_key = torch.nn.Linear(3, 3, bias=False)
        self.actor_residual = torch.nn.Sequential(
            torch.nn.Linear(6, 3),
            torch.nn.GELU(),
            torch.nn.Linear(3, 1),
        )
        self.count_head = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.GELU(),
            torch.nn.Linear(3, 2),
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.GELU(),
            torch.nn.Linear(3, 1),
        )


def episode_layout(date_counts: dict[str, int], rows: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    episode_id = 10_000
    for date, count in date_counts.items():
        for _ in range(count):
            records.append(
                {
                    "episode_id": str(episode_id),
                    "date": date,
                    "decision_rows": 1,
                }
            )
            episode_id += 1
    records[0]["decision_rows"] = rows - len(records) + 1
    return records


def common_allowlist(
    *,
    schema: str,
    split: str,
    date_counts: dict[str, int],
    rows: int,
) -> dict[str, object]:
    records = episode_layout(date_counts, rows)
    payload: dict[str, object] = {
        "schema_version": schema,
        "data_schema_version": repair.DATA_SCHEMA_VERSION,
        "learner_deck_hash": repair.MARNIE_DECK_HASH,
        "split": split,
        "episode_count": len(records),
        "decision_rows": rows,
        "date_episode_counts": date_counts,
        "episode_ids": [record["episode_id"] for record in records],
        "episodes": records,
    }
    if schema == fros_builder.ALLOWLIST_SCHEMA_VERSION:
        contract = fros_builder.DEFAULT_CONTRACT
        outer = dict(contract.daily_sha256)
        manifests = dict(contract.daily_manifest_sha256)
        missing = dict(contract.daily_missing_replay_ids)
        payload["daily"] = [
            {
                "dataset_date": date,
                "filename": f"pokemon-tcg-ai-battle-episodes-{date}.zip",
                "sha256": outer[date],
                "manifest_sha256": manifests[date],
                "manifest_episode_ids": 10 + len(missing[date]),
                "replay_members": 10,
                "missing_replay_ids": list(missing[date]),
                "missing_source_episode_overlap": 0,
                "incomplete_terminal_replay_ids": [
                    value.episode_id
                    for value in contract.incomplete_terminal_replays
                    if value.dataset_date == date
                ],
                "source_episodes_checked": date_counts[date],
            }
            for date in contract.dates
        ]
        payload["incomplete_terminal_replays"] = [
            {
                "dataset_date": value.dataset_date,
                "episode_id": value.episode_id,
                "episode_uuid": value.episode_uuid,
                "replay_sha256": value.replay_sha256,
                "seat": value.seat,
                "team_name": value.team_name,
                "opponent_team_name": value.opponent_team_name,
                "source_terminal_reward": value.source_terminal_reward,
                "raw_rewards": list(value.raw_rewards),
                "learner_deck_hash": value.learner_deck_hash,
                "opponent_deck_hash": value.opponent_deck_hash,
                "target_route": False,
            }
            for value in contract.incomplete_terminal_replays
        ]
    return payload


def anti_allowlist() -> dict[str, object]:
    payload = common_allowlist(
        schema=repair.ANTIKD_ALLOWLIST_SCHEMA,
        split="train",
        date_counts=repair.ANTIKD_DATE_COUNTS,
        rows=repair.ANTIKD_ROWS,
    )
    records = payload["episodes"]
    assert isinstance(records, list)
    keys = [(record["date"], record["episode_id"]) for record in records]
    dev = repair.expected_anti_kd_dev_keys(keys)
    for record in records:
        key = (record["date"], record["episode_id"])
        record["derived_split"] = "dev" if key in dev else "train"
        record["split_score_sha256"] = repair.anti_kd_split_score(*key)
        record["team_name"] = repair.anti_kd_builder.PRIMARY_WINNER
        record["opponent_team_name"] = repair.anti_kd_builder.KD_TEAM_NAME
    payload.update(
        {
            "opponent_deck_hash": repair.MARNIE_DECK_HASH,
            "opponent_team_name": "@kdcyberdude",
            "terminal_reward": "win",
            "winner_group_counts": dict(
                repair.anti_kd_builder.DEFAULT_CONTRACT.expected_winner_groups
            ),
            "source_split": "train",
            "partition_role": "train_dev",
            "source": {
                "logical_path": (
                    repair.anti_kd_builder.DEFAULT_CONTRACT.base.source_logical_path
                ),
                "sha256": (
                    repair.anti_kd_builder.DEFAULT_CONTRACT.base.source_sha256
                ),
                "manifest_sha256": (
                    repair.anti_kd_builder.DEFAULT_CONTRACT.base.source_manifest_sha256
                ),
            },
            "split_rule": {
                "algorithm": "sha256_domain_date_episode_lexicographic",
                "domain_separator": repair.ANTIKD_SPLIT_DOMAIN,
                "payload": "domain_separator + NUL + date + NUL + episode_id",
                "order": "ascending_hex_digest_then_date_then_numeric_episode_id",
                "dev_count": repair.ANTIKD_DEV_EPISODES,
            },
            "train_episode_count": repair.ANTIKD_TRAIN_EPISODES,
            "dev_episode_count": repair.ANTIKD_DEV_EPISODES,
        }
    )
    return payload


def behavior_view(
    *,
    set_accuracy: float,
    hybrid_accuracy: float,
    ordered_loss: float,
    count_digest: str = "a" * 64,
    count_correct: int = 10,
) -> dict[str, object]:
    return {
        "rows": 10,
        "metrics": {
            "set_exact_accuracy": set_accuracy,
            "hybrid_order_exact_accuracy": hybrid_accuracy,
            "count_correct": count_correct,
        },
        "losses": {"ordered_bc_loss": ordered_loss},
        "count_prediction_sha256": count_digest,
    }


def write_scan_archive(
    path: Path,
    rows: list[dict[str, object]],
    *,
    split: str = "valid",
) -> repair.ArchiveRecord:
    member = f"{split}/fixture.jsonl"
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, payload)
    return repair.ArchiveRecord(path, "a" * 64, "b" * 64, {}, (member,))


def scan_row(
    *,
    episode_id: str,
    seat: int,
    team_name: str,
    opponent_team_name: str,
    terminal_reward: float,
    action_step_index: int = 2,
) -> dict[str, object]:
    return {
        "dataset_date": "2026-08-07",
        "episode_id": episode_id,
        "seat": seat,
        "team_name": team_name,
        "opponent_team_name": opponent_team_name,
        "terminal_reward": terminal_reward,
        "action_step_index": action_step_index,
    }


class S08AntiKdRepairTests(unittest.TestCase):
    def test_route_constants_are_exact(self) -> None:
        repair.validate_protocol_constants()
        self.assertEqual(repair.S8_PARENT_SHA256, "d7443bda" + repair.S8_PARENT_SHA256[8:])
        self.assertEqual(
            tuple(source for source, _ in repair.STEP_SCHEDULE),
            ("anti_kd", "general", "anti_kd", "fros") * 2,
        )
        self.assertEqual(repair.LEARNING_RATE, 2e-7)
        self.assertEqual(repair.WEIGHT_DECAY, 1e-4)
        self.assertEqual(repair.ADAM_EPS, 1e-5)
        self.assertEqual(repair.MAX_GRAD_NORM, 1.0)
        self.assertEqual(repair.ORDER_CONTEXT_WEIGHT, 8.0)
        self.assertEqual(repair.RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT, 2.0)
        self.assertEqual(repair.CONTRACTIONS, (("E50", 0.5), ("E100", 1.0)))
        self.assertEqual(repair.MAX_FULL_ACTOR6_L2, 2.5e-4)

    def test_old_update0_draft_is_permanently_retired(self) -> None:
        self.assertTrue(retired.RETIRED_NEVER_EXECUTED)
        with self.assertRaisesRegex(RuntimeError, "RETIRED_NEVER_EXECUTED"):
            retired.main([])

    def test_fros_builder_schema_contract_matches_launcher(self) -> None:
        self.assertEqual(repair.FROS_TRAIN_ARCHIVE_SCHEMA, fros_builder.SCHEMA_VERSION)
        self.assertEqual(
            repair.FROS_TRAIN_ALLOWLIST_SCHEMA,
            fros_builder.ALLOWLIST_SCHEMA_VERSION,
        )
        self.assertEqual(repair.DATA_SCHEMA_VERSION, fros_builder.SOURCE_SCHEMA_VERSION)
        source = Path(fros_builder.__file__).read_text(encoding="utf-8")
        self.assertIn('"data_schema_version": ROW_SCHEMA_VERSION', source)

    def test_anti_kd_hash_split_and_allowlist_are_exact(self) -> None:
        payload = anti_allowlist()
        validated, keys = repair.validate_anti_kd_allowlist(payload)
        self.assertIs(validated, payload)
        self.assertEqual(len(keys), 43)
        dev = {
            (record["date"], record["episode_id"])
            for record in payload["episodes"]
            if record["derived_split"] == "dev"
        }
        self.assertEqual(dev, repair.expected_anti_kd_dev_keys(list(keys)))
        self.assertEqual(len(dev), 8)

    def test_real_anti_builder_allowlist_satisfies_launcher_contract(self) -> None:
        """Cross the actual builder serialization boundary, not a hand fixture."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = AntiFixture(root)
            archive = root / "bundle" / "anti.zip"
            allowlist = root / "bundle" / repair.anti_kd_builder.ALLOWLIST_MEMBER
            dry = repair.anti_kd_builder.build(
                source=fixture.source,
                daily_dir=fixture.daily_dir,
                archive_path=archive,
                allowlist_path=allowlist,
                contract=fixture.contract,
                execute=False,
            )
            repair.anti_kd_builder.build(
                source=fixture.source,
                daily_dir=fixture.daily_dir,
                archive_path=archive,
                allowlist_path=allowlist,
                contract=fixture.contract,
                execute=True,
                expected_plan_sha256=dry["plan_sha256"],
                expected_archive_sha256=dry["archive_sha256"],
                expected_allowlist_sha256=dry["allowlist_sha256"],
            )
            raw = allowlist.read_bytes()
            payload = repair.anti_kd_builder.validate_allowlist_document(
                raw,
                fixture.contract,
            )
            validated, keys = repair.validate_anti_kd_allowlist(
                payload,
                contract=fixture.contract,
            )

        self.assertIs(validated, payload)
        self.assertEqual(payload["split"], "train")
        self.assertEqual(payload["source_split"], "train")
        self.assertEqual(payload["partition_role"], "train_dev")
        self.assertEqual(len(keys), fixture.contract.expected_episodes)
        self.assertEqual(
            sum(row["decision_rows"] for row in payload["episodes"]),
            fixture.contract.expected_rows,
        )

    def test_real_fros_builder_archive_satisfies_launcher_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = FrosFixture(root)
            archive = root / "fros-bundle" / "fros.zip"
            dry = fros_builder.build(
                source=fixture.source,
                daily_dir=fixture.daily_dir,
                output=archive,
                contract=fixture.contract,
                execute=False,
            )
            fros_builder.build(
                source=fixture.source,
                daily_dir=fixture.daily_dir,
                output=archive,
                contract=fixture.contract,
                execute=True,
                expected_plan_sha256=dry["plan_sha256"],
                expected_output_sha256=dry["archive_sha256"],
                expected_allowlist_sha256=dry["allowlist_sha256"],
            )
            external = archive.parent / fros_builder.ALLOWLIST_MEMBER
            external_raw = external.read_bytes()
            with zipfile.ZipFile(archive) as built:
                embedded_raw = built.read(fros_builder.ALLOWLIST_MEMBER)
                manifest_raw = built.read(fros_builder.MANIFEST_MEMBER)
            self.assertEqual(external_raw, embedded_raw)
            self.assertTrue(external_raw.endswith(b"\n"))
            self.assertFalse(external_raw.endswith(b"\r\n"))
            self.assertEqual(
                external_raw,
                repair.canonical_line_bytes(json.loads(external_raw)),
            )
            self.assertEqual(
                hashlib.sha256(manifest_raw).hexdigest(),
                dry["manifest_sha256"],
            )
            validator = partial(
                repair.validate_fros_train_allowlist,
                contract=fixture.contract,
            )
            allowlist_record = repair.load_allowlist(
                external,
                expected_raw_sha256=dry["allowlist_sha256"],
                expected_canonical_sha256=dry["allowlist_canonical_sha256"],
                label="synthetic exact-Fros allowlist",
                validator=validator,
            )
            archive_record = repair.load_archive(
                archive,
                expected_archive_sha256=dry["archive_sha256"],
                expected_manifest_sha256=dry["manifest_sha256"],
                expected_schema=repair.FROS_TRAIN_ARCHIVE_SCHEMA,
                label="synthetic exact-Fros archive",
                allowlist=allowlist_record,
            )
            split = repair.scan_archive_split(archive_record, "train")

        self.assertEqual(split.episode_keys, allowlist_record.episode_keys)
        self.assertEqual(split.rows, fixture.contract.expected_selected_rows)
        self.assertEqual(len(split.view_keys), len(split.episode_keys))
        self.assertEqual(
            split.outcome_view_counts,
            {"win": fixture.contract.expected_selected_episodes},
        )

    def test_scan_archive_accepts_reverse_routed_mirror_views(self) -> None:
        rows = [
            scan_row(
                episode_id="90613480",
                seat=0,
                team_name="Alpha",
                opponent_team_name="Beta",
                terminal_reward=1.0,
            ),
            scan_row(
                episode_id="90613480",
                seat=1,
                team_name="Beta",
                opponent_team_name="Alpha",
                terminal_reward=-1.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            record = write_scan_archive(Path(temporary) / "mirror.zip", rows)
            audit = repair.scan_archive_split(record, "valid")
        repair._validate_mirror_views(audit, "fixture")
        self.assertEqual(len(audit.episode_keys), 1)
        self.assertEqual(len(audit.view_keys), 2)
        self.assertEqual(audit.outcome_view_counts, {"loss": 1, "win": 1})
        manifest = repair._split_audit_json(audit)
        self.assertEqual(manifest["view_count_histogram"], {"2": 1})
        self.assertEqual(
            manifest["mirror_episode_keys"],
            [["2026-08-07", "90613480"]],
        )

    def test_scan_archive_rejects_composite_view_drift(self) -> None:
        base = scan_row(
            episode_id="1",
            seat=0,
            team_name="Alpha",
            opponent_team_name="Beta",
            terminal_reward=1.0,
        )
        changed = dict(base, terminal_reward=-1.0, action_step_index=3)
        with tempfile.TemporaryDirectory() as temporary:
            record = write_scan_archive(Path(temporary) / "drift.zip", [base, changed])
            with self.assertRaisesRegex(RuntimeError, "identity/reward drift"):
                repair.scan_archive_split(record, "valid")

    def test_scan_archive_rejects_invalid_view_identity(self) -> None:
        cases = (
            (dict(seat=2), "invalid seat"),
            (dict(team_name=""), "empty team_name"),
            (dict(opponent_team_name=""), "empty opponent_team_name"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                row = scan_row(
                    episode_id="1",
                    seat=0,
                    team_name="Alpha",
                    opponent_team_name="Beta",
                    terminal_reward=1.0,
                )
                row.update(changes)
                with tempfile.TemporaryDirectory() as temporary:
                    record = write_scan_archive(Path(temporary) / "bad.zip", [row])
                    with self.assertRaisesRegex(RuntimeError, message):
                        repair.scan_archive_split(record, "valid")

    def test_mirror_validation_rejects_same_seat_and_three_views(self) -> None:
        cases = (
            [
                scan_row(
                    episode_id="1", seat=0, team_name="Alpha",
                    opponent_team_name="Beta", terminal_reward=1.0,
                ),
                scan_row(
                    episode_id="1", seat=0, team_name="Beta",
                    opponent_team_name="Alpha", terminal_reward=-1.0,
                ),
            ],
            [
                scan_row(
                    episode_id="1", seat=0, team_name="Alpha",
                    opponent_team_name="Beta", terminal_reward=1.0,
                ),
                scan_row(
                    episode_id="1", seat=1, team_name="Beta",
                    opponent_team_name="Alpha", terminal_reward=-1.0,
                ),
                scan_row(
                    episode_id="1", seat=1, team_name="Gamma",
                    opponent_team_name="Alpha", terminal_reward=1.0,
                ),
            ],
        )
        for index, rows in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                record = write_scan_archive(Path(temporary) / "bad-mirror.zip", rows)
                audit = repair.scan_archive_split(record, "valid")
                with self.assertRaisesRegex(RuntimeError, "mirror"):
                    repair._validate_mirror_views(audit, "fixture")

    def test_real_general_behavior_archive_has_frozen_mirror_contract(self) -> None:
        self.assertEqual(
            repair.file_sha256(repair.GENERAL_BEHAVIOR_ARCHIVE),
            repair.GENERAL_BEHAVIOR_ARCHIVE_SHA256,
        )
        with zipfile.ZipFile(repair.GENERAL_BEHAVIOR_ARCHIVE) as archive:
            members = tuple(archive.namelist())
        record = repair.ArchiveRecord(
            repair.GENERAL_BEHAVIOR_ARCHIVE,
            repair.GENERAL_BEHAVIOR_ARCHIVE_SHA256,
            repair.GENERAL_BEHAVIOR_MANIFEST_SHA256,
            {},
            members,
        )
        audit = repair.scan_archive_split(record, "valid")
        repair.validate_general_behavior_split(audit)
        self.assertEqual((audit.rows, len(audit.episode_keys), len(audit.view_keys)), (28800, 295, 299))
        self.assertEqual(audit.outcome_view_counts, {"loss": 138, "win": 161})
        self.assertEqual(
            tuple(key[1] for key in repair._mirror_episode_keys(audit)),
            repair.GENERAL_BEHAVIOR_MIRROR_IDS,
        )

    def test_real_general_builder_archive_satisfies_launcher_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = GeneralFixture(root)
            bundle = root / "general-bundle"
            archive = bundle / "general.zip"
            external_manifest = bundle / repair.general_anchor_builder.MANIFEST_MEMBER
            fros_sha = repair.file_sha256(fixture.fros_allowlist)
            anti_sha = repair.file_sha256(fixture.anti_allowlist)
            dry = repair.general_anchor_builder.build(
                source=fixture.source,
                fros_allowlist=fixture.fros_allowlist,
                anti_allowlist=fixture.anti_allowlist,
                archive_path=archive,
                external_manifest_path=external_manifest,
                expected_fros_allowlist_sha256=fros_sha,
                expected_anti_allowlist_sha256=anti_sha,
                contract=fixture.contract,
                execute=False,
            )
            repair.general_anchor_builder.build(
                source=fixture.source,
                fros_allowlist=fixture.fros_allowlist,
                anti_allowlist=fixture.anti_allowlist,
                archive_path=archive,
                external_manifest_path=external_manifest,
                expected_fros_allowlist_sha256=fros_sha,
                expected_anti_allowlist_sha256=anti_sha,
                contract=fixture.contract,
                execute=True,
                expected_plan_sha256=dry["plan_sha256"],
                expected_archive_sha256=dry["archive_sha256"],
                expected_manifest_sha256=dry["manifest_sha256"],
            )
            validator = partial(
                repair.validate_general_anchor_manifest,
                expected_fros_allowlist_sha256=fros_sha,
                expected_anti_allowlist_sha256=anti_sha,
                contract=fixture.contract,
            )
            record = repair.load_archive(
                archive,
                expected_archive_sha256=dry["archive_sha256"],
                expected_manifest_sha256=dry["manifest_sha256"],
                expected_schema=repair.GENERAL_ANCHOR_ARCHIVE_SCHEMA,
                label="synthetic general anchor",
                manifest_validator=validator,
            )
            repair.validate_general_anchor_archive(record)

            tampered = json.loads(external_manifest.read_bytes())
            tampered["source"]["train_members"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "name/size/SHA"):
                validator(tampered)

        self.assertEqual(record.manifest["source"]["episodes"], 5)
        self.assertEqual(record.manifest["source"]["rows"], 5)

    def test_anti_kd_allowlist_rejects_non_preregistered_dev(self) -> None:
        payload = anti_allowlist()
        records = payload["episodes"]
        dev_record = next(row for row in records if row["derived_split"] == "dev")
        train_record = next(row for row in records if row["derived_split"] == "train")
        dev_record["derived_split"] = "train"
        train_record["derived_split"] = "dev"
        with self.assertRaisesRegex(RuntimeError, "preregistered hashes"):
            repair.validate_anti_kd_allowlist(payload)

    def test_fros_train_and_valid_allowlist_counts_are_exact(self) -> None:
        train = common_allowlist(
            schema=repair.FROS_TRAIN_ALLOWLIST_SCHEMA,
            split="train",
            date_counts=repair.FROS_TRAIN_DATE_COUNTS,
            rows=repair.FROS_TRAIN_ROWS,
        )
        train.update(
            {
                "opponent_deck_hash": repair.FROS_DECK_HASH,
                "terminal_reward": "win",
            }
        )
        _, train_keys = repair.validate_fros_train_allowlist(train)
        self.assertEqual(len(train_keys), 104)

        valid = common_allowlist(
            schema=repair.FROS_VALID_ALLOWLIST_SCHEMA,
            split="valid",
            date_counts=repair.FROS_VALID_DATE_COUNTS,
            rows=repair.FROS_VALID_ROWS,
        )
        valid.update(
            {
                "opponent_deck_hash": repair.FROS_DECK_HASH,
                "terminal_reward": "all",
                "date": "2026-08-07",
                "incomplete_terminal_replays": [],
                "win_episodes": 14,
                "loss_episodes": 16,
            }
        )
        for index, record in enumerate(valid["episodes"]):
            record["outcome"] = "win" if index < 14 else "loss"
        _, valid_keys = repair.validate_fros_valid_allowlist(valid)
        self.assertEqual(len(valid_keys), 30)

    def test_allowlist_binds_raw_and_canonical_hash_separately(self) -> None:
        payload = anti_allowlist()
        canonical = repair.canonical_line_bytes(payload)
        raw = canonical
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "allowlist.json"
            path.write_bytes(raw)
            record = repair.load_allowlist(
                path,
                expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
                expected_canonical_sha256=hashlib.sha256(canonical).hexdigest(),
                label="anti-KD allowlist",
                validator=repair.validate_anti_kd_allowlist,
            )
        self.assertEqual(record.raw, raw)
        self.assertEqual(record.raw_sha256, record.canonical_sha256)

    def test_regular_file_check_rejects_symlink_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            digest = repair.file_sha256(target)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                repair.require_regular_file(link, digest, "test link")

    def test_regular_file_check_rejects_ancestor_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            target = real / "target.json"
            target.write_bytes(b"{}")
            linked_directory = root / "linked-directory"
            linked_directory.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                repair.require_regular_file(
                    linked_directory / target.name,
                    hashlib.sha256(b"{}").hexdigest(),
                    "ancestor-link fixture",
                )

    def test_archive_requires_dedicated_and_data_schema_and_bound_allowlist(self) -> None:
        payload = anti_allowlist()
        allowlist_raw = repair.canonical_line_bytes(payload)
        allowlist = repair.AllowlistRecord(
            path=Path("/tmp/allowlist.json"),
            raw=allowlist_raw,
            raw_sha256=hashlib.sha256(allowlist_raw).hexdigest(),
            canonical_sha256=hashlib.sha256(
                repair.canonical_line_bytes(payload)
            ).hexdigest(),
            payload=payload,
            episode_keys=frozenset(
                (row["date"], row["episode_id"]) for row in payload["episodes"]
            ),
        )
        manifest = {
            "schema_version": repair.ANTIKD_ARCHIVE_SCHEMA,
            "data_schema_version": repair.DATA_SCHEMA_VERSION,
            "exact_episode_allowlist": {
                "member": repair.ALLOWLIST_MEMBER,
                "schema_version": repair.ANTIKD_ALLOWLIST_SCHEMA,
                "sha256": allowlist.raw_sha256,
                "canonical_sha256": allowlist.canonical_sha256,
                "episodes": repair.ANTIKD_EPISODES,
                "decision_rows": repair.ANTIKD_ROWS,
            },
        }
        manifest_raw = repair.canonical_line_bytes(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "archive.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("manifest.json", manifest_raw)
                archive.writestr(repair.ALLOWLIST_MEMBER, allowlist_raw)
            record = repair.load_archive(
                archive_path,
                expected_archive_sha256=repair.file_sha256(archive_path),
                expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                expected_schema=repair.ANTIKD_ARCHIVE_SCHEMA,
                label="anti-KD archive",
                allowlist=allowlist,
            )
        self.assertEqual(record.manifest["data_schema_version"], repair.DATA_SCHEMA_VERSION)

    def test_actor6_contractions_preserve_every_nonactor_tensor(self) -> None:
        model = ToyPolicy()
        parent = repair.training_core.clone_model_state(model)
        trained = {name: tensor.clone() for name, tensor in parent.items()}
        for name in repair.ACTOR6:
            trained[name].add_(1e-6)
        states, audit = repair.build_contractions(parent, trained)
        self.assertEqual(set(states), {"E50", "E100"})
        self.assertLessEqual(audit["E100"]["movement"]["l2"], 2.5e-4)
        for name in parent:
            if name not in repair.ACTOR6:
                self.assertTrue(torch.equal(parent[name], states["E50"][name]), name)
                self.assertTrue(torch.equal(parent[name], states["E100"][name]), name)

    def test_full_correction_movement_cap_is_fail_closed(self) -> None:
        model = ToyPolicy()
        parent = repair.training_core.clone_model_state(model)
        trained = {name: tensor.clone() for name, tensor in parent.items()}
        for name in repair.ACTOR6:
            trained[name].add_(0.1)
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            repair.build_contractions(parent, trained)

    def test_behavior_gate_exact_thresholds(self) -> None:
        baseline = {
            "general": behavior_view(
                set_accuracy=0.8000, hybrid_accuracy=0.7900, ordered_loss=1.0
            ),
            "fros_valid": behavior_view(
                set_accuracy=0.7, hybrid_accuracy=0.7000, ordered_loss=1.2
            ),
            "anti_kd_dev": behavior_view(
                set_accuracy=0.6, hybrid_accuracy=0.6, ordered_loss=1.5
            ),
        }
        candidate = {
            "general": behavior_view(
                set_accuracy=0.7990, hybrid_accuracy=0.7890, ordered_loss=1.0
            ),
            "fros_valid": behavior_view(
                set_accuracy=0.7, hybrid_accuracy=0.7000, ordered_loss=1.2
            ),
            "anti_kd_dev": behavior_view(
                set_accuracy=0.6, hybrid_accuracy=0.6, ordered_loss=1.499
            ),
        }
        self.assertTrue(repair.behavior_gate(baseline, candidate)["pass"])
        candidate["fros_valid"]["metrics"]["hybrid_order_exact_accuracy"] = 0.699
        decision = repair.behavior_gate(baseline, candidate)
        self.assertFalse(decision["pass"])
        self.assertFalse(decision["gates"]["fros_hybrid_non_drop"])

    def test_archive_relationships_require_three_disjoint_training_sources(self) -> None:
        anti = anti_allowlist()
        _, anti_keys = repair.validate_anti_kd_allowlist(anti)
        anti_record = repair.AllowlistRecord(
            Path("anti.json"), b"", "a" * 64, "b" * 64, anti, anti_keys
        )
        fros_payload = common_allowlist(
            schema=repair.FROS_TRAIN_ALLOWLIST_SCHEMA,
            split="train",
            date_counts=repair.FROS_TRAIN_DATE_COUNTS,
            rows=repair.FROS_TRAIN_ROWS,
        )
        fros_payload.update(
            {"opponent_deck_hash": repair.FROS_DECK_HASH, "terminal_reward": "win"}
        )
        _, fros_keys = repair.validate_fros_train_allowlist(fros_payload)
        # Force one cross-source collision: the relationship audit must reject
        # before it could ever trust general-anchor subtraction arithmetic.
        fros_record = repair.AllowlistRecord(
            Path("fros.json"), b"", "c" * 64, "d" * 64, fros_payload,
            frozenset({next(iter(anti_keys)), *list(fros_keys)[1:]}),
        )
        dummy_manifest = {
            "exclusion_sources": {
                "fros": {},
                "anti_kd": {},
            }
        }
        dummy_archive = repair.ArchiveRecord(
            Path("dummy.zip"), "e" * 64, "f" * 64, dummy_manifest, ()
        )
        empty = repair.SplitAudit(
            split="train",
            rows=1,
            episode_keys=frozenset(),
            view_keys=frozenset(),
            date_episode_counts={},
            date_view_counts={},
            outcome_view_counts={},
            view_rewards={},
            view_opponents={},
            view_row_counts={},
            decision_keys_sha256="0" * 64,
        )
        with mock.patch.object(repair, "scan_archive_split", return_value=empty):
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                repair.validate_archive_relationships(
                    anti_allowlist=anti_record,
                    anti_archive=dummy_archive,
                    fros_allowlist=fros_record,
                    fros_archive=dummy_archive,
                    fros_valid_allowlist=fros_record,
                    fros_valid_archive=dummy_archive,
                    general_anchor=dummy_archive,
                    general_behavior=dummy_archive,
                )

    def test_dry_run_bytes_are_repeatable_and_scope_has_no_h2h_or_submit(self) -> None:
        manifest = {
            "schema_version": repair.SCHEMA_VERSION,
            "protocol": {"schedule": list(repair.STEP_SCHEDULE)},
            "scope": {
                "training": True,
                "gameplay_evaluation": False,
                "package": False,
                "upload": False,
                "submission": False,
            },
        }
        first = repair.dry_run_bytes(manifest)
        second = repair.dry_run_bytes(json.loads(json.dumps(manifest)))
        self.assertEqual(first, second)
        envelope = json.loads(first)
        self.assertEqual(envelope["manifest_sha256"], repair.sha256_json(manifest))
        self.assertFalse(manifest["scope"]["gameplay_evaluation"])
        self.assertFalse(manifest["scope"]["submission"])

    def test_execute_mode_requires_reviewed_frozen_manifest(self) -> None:
        for frozen, digest in ((None, None), (Path("x.json"), None), (None, "a" * 64)):
            with self.subTest(frozen=frozen, digest=digest):
                args = SimpleNamespace(
                    execute=True,
                    frozen_manifest=frozen,
                    expected_manifest_sha256=digest,
                )
                with self.assertRaisesRegex(ValueError, "requires"):
                    repair.validate_mode_args(args)
        repair.validate_mode_args(
            SimpleNamespace(
                execute=True,
                frozen_manifest=Path("x.json"),
                expected_manifest_sha256="a" * 64,
            )
        )

    def test_output_directory_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            repair.training_core.assert_output_absent(output)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                repair.training_core.assert_output_absent(output)


if __name__ == "__main__":
    unittest.main()
