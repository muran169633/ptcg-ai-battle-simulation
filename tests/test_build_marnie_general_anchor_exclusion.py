from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_marnie_exact_anti_kd as anti
import build_marnie_froslass_exact_wins as fros
import build_marnie_general_anchor_exclusion as general


class GeneralFixture:
    """Small real source + canonical allowlists for cross-tool consumers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "source.zip"
        self.fros_allowlist = root / "fros_allowlist.json"
        self.anti_allowlist = root / "anti_allowlist.json"
        self.dates = ("2026-08-02", "2026-08-03")
        self.deck_hash = "a" * 64
        self.fros_hash = "b" * 64
        self.rows = {
            episode_id: self._row(episode_id, date, team, opponent)
            for episode_id, date, team, opponent in (
                ("4001", self.dates[0], "A", "Fros"),
                ("4002", self.dates[0], "A", "KD"),
                ("4003", self.dates[1], "B", "KD"),
                ("4004", self.dates[1], "C", "Other"),
                ("4005", self.dates[1], "D", "Other"),
            )
        }
        self._write_source()
        with zipfile.ZipFile(self.source) as source_archive:
            self.source_train_member_audit = tuple(
                (
                    member,
                    len(raw := source_archive.read(member)),
                    hashlib.sha256(raw).hexdigest(),
                )
                for member in (fros.TRAIN_MEMBER,)
            )
        self.source_contract = fros.BuildContract(
            source_sha256=fros.sha256_file(self.source),
            source_manifest_sha256=self.source_manifest_sha,
            daily_sha256=((self.dates[0], "c" * 64), (self.dates[1], "d" * 64)),
            daily_manifest_sha256=(
                (self.dates[0], "e" * 64),
                (self.dates[1], "f" * 64),
            ),
            daily_missing_replay_ids=tuple((date, ()) for date in self.dates),
            dates=self.dates,
            expected_date_episodes=((self.dates[0], 1), (self.dates[1], 0)),
            expected_source_train_rows=5,
            expected_source_train_episodes=5,
            expected_selected_rows=1,
            expected_selected_episodes=1,
            learner_deck_hash=self.deck_hash,
            opponent_deck_hash=self.fros_hash,
            source_members=(fros.TRAIN_MEMBER, fros.MANIFEST_MEMBER),
            source_train_members=(fros.TRAIN_MEMBER,),
            source_logical_path="fixture/source.zip",
            daily_logical_dir="fixture/daily",
            incomplete_terminal_replays=(),
        )
        self.fros_contract = fros.BuildContract(
            **{
                **self.source_contract.__dict__,
                "dates": (self.dates[0],),
                "daily_sha256": ((self.dates[0], "c" * 64),),
                "daily_manifest_sha256": ((self.dates[0], "e" * 64),),
                "daily_missing_replay_ids": ((self.dates[0], ()),),
                "expected_date_episodes": ((self.dates[0], 1),),
            }
        )
        anti_base = fros.BuildContract(
            **{
                **self.source_contract.__dict__,
                "opponent_deck_hash": self.deck_hash,
                "expected_date_episodes": (
                    (self.dates[0], 1),
                    (self.dates[1], 1),
                ),
                "expected_selected_rows": 2,
                "expected_selected_episodes": 2,
            }
        )
        self.anti_contract = anti.AntiKDContract(
            base=anti_base,
            top20_names=("KD", "A", "B", "C", "D"),
            kd_team_name="KD",
            expected_episodes=2,
            expected_rows=2,
            expected_date_episodes=((self.dates[0], 1), (self.dates[1], 1)),
            expected_winner_groups=(
                (anti.PRIMARY_WINNER, 0),
                (anti.SECONDARY_WINNER, 0),
                (anti.TERTIARY_WINNER, 0),
                ("__others__", 2),
            ),
            train_episode_count=1,
            dev_episode_count=1,
            split_domain_separator=anti.SPLIT_DOMAIN_SEPARATOR,
        )
        self._write_fros_allowlist()
        self._write_anti_allowlist()
        self.contract = general.GeneralContract(
            source=self.source_contract,
            fros_allowlist_contract=self.fros_contract,
            anti_allowlist_contract=self.anti_contract,
            expected_excluded_episodes=3,
            expected_excluded_rows=3,
            expected_output_episodes=2,
            expected_output_rows=2,
            source_train_member_audit=self.source_train_member_audit,
        )

    def _row(
        self,
        episode_id: str,
        date: str,
        team: str,
        opponent: str,
    ) -> dict[str, object]:
        return {
            "schema_version": fros.ROW_SCHEMA_VERSION,
            "episode_id": episode_id,
            "episode_uuid": f"uuid-{episode_id}",
            "dataset_date": date,
            "split": "train",
            "observation_step_index": 0,
            "action_step_index": 1,
            "seat": 0,
            "team_name": team,
            "opponent_team_name": opponent,
            "deck_hash": self.deck_hash,
            "terminal_reward": 1.0,
            "sample_weight": 1.0,
            "action": [0],
            "observation": {"select": {"option": [{"type": 1}]}},
        }

    def _write_source(self) -> None:
        manifest = {
            "schema_version": fros.ROW_SCHEMA_VERSION,
            "profile": {"deck_hash": self.deck_hash},
            "deck_hash_filter": self.deck_hash,
            "split_policy": {"train_dates": list(self.dates)},
            "split_decisions": {"train": 5},
            "split_episodes": {"train": 5},
            "terminal_reward_filter": {"mode": "wins", "splits": "train"},
        }
        manifest_payload = json.dumps(manifest, indent=2).encode()
        self.source_manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
        rows = b"".join(
            fros.canonical_json_bytes(self.rows[episode_id]) + b"\n"
            for episode_id in sorted(self.rows, key=int)
        )
        with zipfile.ZipFile(self.source, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(fros.zip_info(fros.TRAIN_MEMBER), rows)
            archive.writestr(fros.zip_info(fros.MANIFEST_MEMBER), manifest_payload)

    def _write_fros_allowlist(self) -> None:
        document = {
            "schema_version": fros.ALLOWLIST_SCHEMA_VERSION,
            "data_schema_version": fros.ROW_SCHEMA_VERSION,
            "learner_deck_hash": self.deck_hash,
            "opponent_deck_hash": self.fros_hash,
            "split": "train",
            "terminal_reward": "win",
            "episode_count": 1,
            "decision_rows": 1,
            "date_episode_counts": {self.dates[0]: 1},
            "daily": [
                {
                    "dataset_date": self.dates[0],
                    "filename": (
                        f"pokemon-tcg-ai-battle-episodes-{self.dates[0]}.zip"
                    ),
                    "sha256": "c" * 64,
                    "manifest_sha256": "e" * 64,
                    "manifest_episode_ids": 1,
                    "replay_members": 1,
                    "missing_replay_ids": [],
                    "missing_source_episode_overlap": 0,
                    "incomplete_terminal_replay_ids": [],
                    "source_episodes_checked": 1,
                }
            ],
            "incomplete_terminal_replays": [],
            "episode_ids": ["4001"],
            "episodes": [
                {
                    "episode_id": "4001",
                    "date": self.dates[0],
                    "decision_rows": 1,
                }
            ],
        }
        self.fros_allowlist.write_bytes(fros.canonical_json_bytes(document) + b"\n")
        fros.validate_allowlist_document(
            self.fros_allowlist.read_bytes(), self.fros_contract
        )

    def _write_anti_allowlist(self) -> None:
        episode_specs = (("4002", self.dates[0], "A"), ("4003", self.dates[1], "B"))
        scores = {
            episode_id: anti.split_score(
                date, episode_id, self.anti_contract.split_domain_separator
            )
            for episode_id, date, _team in episode_specs
        }
        dev_id = min(scores, key=lambda value: (scores[value], value))
        episodes = [
            {
                "episode_id": episode_id,
                "date": date,
                "decision_rows": 1,
                "derived_split": "dev" if episode_id == dev_id else "train",
                "split_score_sha256": scores[episode_id],
                "team_name": team,
                "opponent_team_name": "KD",
            }
            for episode_id, date, team in episode_specs
        ]
        document = {
            "schema_version": anti.ALLOWLIST_SCHEMA_VERSION,
            "data_schema_version": fros.ROW_SCHEMA_VERSION,
            "learner_deck_hash": self.deck_hash,
            "opponent_deck_hash": self.deck_hash,
            "opponent_team_name": "KD",
            "split": "train",
            "source_split": "train",
            "partition_role": "train_dev",
            "terminal_reward": "win",
            "episode_count": 2,
            "decision_rows": 2,
            "date_episode_counts": {self.dates[0]: 1, self.dates[1]: 1},
            "winner_group_counts": {
                anti.PRIMARY_WINNER: 0,
                anti.SECONDARY_WINNER: 0,
                anti.TERTIARY_WINNER: 0,
                "__others__": 2,
            },
            "train_episode_count": 1,
            "dev_episode_count": 1,
            "episode_ids": ["4002", "4003"],
            "source": {
                "logical_path": self.anti_contract.base.source_logical_path,
                "sha256": self.anti_contract.base.source_sha256,
                "manifest_sha256": self.anti_contract.base.source_manifest_sha256,
            },
            "split_rule": {
                "algorithm": anti.SPLIT_ALGORITHM,
                "domain_separator": self.anti_contract.split_domain_separator,
                "payload": anti.SPLIT_PAYLOAD,
                "order": anti.SPLIT_ORDER,
                "dev_count": 1,
            },
            "episodes": episodes,
        }
        self.anti_allowlist.write_bytes(fros.canonical_json_bytes(document) + b"\n")
        anti.validate_allowlist_document(
            self.anti_allowlist.read_bytes(), self.anti_contract
        )


class GeneralAnchorBuilderTests(unittest.TestCase):
    def test_isolated_cli_help_and_minimal_dry_run(self) -> None:
        script = REPO_ROOT / "tools/build_marnie_general_anchor_exclusion.py"
        with tempfile.TemporaryDirectory() as temporary:
            help_result = subprocess.run(
                [sys.executable, "-I", "-B", str(script), "--help"],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("usage:", help_result.stdout)
            self.assertNotIn("ModuleNotFoundError", help_result.stderr)

            code = r'''
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

target = load("build_marnie_general_anchor_exclusion", Path(sys.argv[1]))
fixture_module = load("_isolated_general_fixture", Path(sys.argv[2]))
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    fixture = fixture_module.GeneralFixture(root)
    bundle = root / "never-bundle"
    result = target.build(
        source=fixture.source,
        fros_allowlist=fixture.fros_allowlist,
        anti_allowlist=fixture.anti_allowlist,
        archive_path=bundle / "general.zip",
        external_manifest_path=bundle / target.MANIFEST_MEMBER,
        expected_fros_allowlist_sha256=target.fros.sha256_file(
            fixture.fros_allowlist
        ),
        expected_anti_allowlist_sha256=target.fros.sha256_file(
            fixture.anti_allowlist
        ),
        contract=fixture.contract,
        execute=False,
    )
    assert result["status"] == "dry_run_passed"
    assert not bundle.exists()
    print(json.dumps({"status": result["status"], "output_written": result["output_written"]}))
'''
            dry_result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    code,
                    str(script),
                    str(Path(__file__).resolve()),
                ],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(dry_result.returncode, 0, dry_result.stderr)
            self.assertEqual(
                json.loads(dry_result.stdout),
                {"status": "dry_run_passed", "output_written": False},
            )

    def test_contract_rejects_fros_and_anti_source_lineage_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = GeneralFixture(Path(temporary))
            cases = (
                dataclasses.replace(
                    fixture.contract,
                    fros_allowlist_contract=dataclasses.replace(
                        fixture.fros_contract,
                        source_manifest_sha256="0" * 64,
                    ),
                ),
                dataclasses.replace(
                    fixture.contract,
                    anti_allowlist_contract=dataclasses.replace(
                        fixture.anti_contract,
                        base=dataclasses.replace(
                            fixture.anti_contract.base,
                            source_sha256="1" * 64,
                        ),
                    ),
                ),
            )
            for contract in cases:
                with self.subTest(contract=contract):
                    with self.assertRaisesRegex(
                        RuntimeError, "source lineage differ"
                    ):
                        general.validate_contract(contract)

    def test_preexisting_bundle_directory_is_rejected_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = GeneralFixture(root)
            bundle = root / "existing-bundle"
            bundle.mkdir()
            marker = bundle / "keep.bin"
            marker.write_bytes(b"keep")
            with self.assertRaisesRegex(FileExistsError, "must be absent"):
                general.build(
                    source=fixture.source,
                    fros_allowlist=fixture.fros_allowlist,
                    anti_allowlist=fixture.anti_allowlist,
                    archive_path=bundle / "general.zip",
                    external_manifest_path=bundle / general.MANIFEST_MEMBER,
                    expected_fros_allowlist_sha256=fros.sha256_file(
                        fixture.fros_allowlist
                    ),
                    expected_anti_allowlist_sha256=fros.sha256_file(
                        fixture.anti_allowlist
                    ),
                    contract=fixture.contract,
                    execute=False,
                )
            self.assertEqual(marker.read_bytes(), b"keep")
            self.assertEqual(sorted(path.name for path in bundle.iterdir()), ["keep.bin"])

    def test_real_allowlists_build_exact_remaining_rows_and_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = GeneralFixture(root)
            bundle = root / "bundle"
            archive = bundle / "general.zip"
            external_manifest = bundle / general.MANIFEST_MEMBER
            fros_sha = fros.sha256_file(fixture.fros_allowlist)
            anti_sha = fros.sha256_file(fixture.anti_allowlist)
            dry = general.build(
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
            self.assertFalse(bundle.exists())
            self.assertEqual(dry["output_episodes"], 2)
            self.assertEqual(dry["output_rows"], 2)
            general.build(
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
            with zipfile.ZipFile(archive) as zipped:
                train = zipped.read(general.TRAIN_MEMBER)
                manifest_raw = zipped.read(general.MANIFEST_MEMBER)
            self.assertEqual(manifest_raw, external_manifest.read_bytes())
            manifest = general.validate_manifest_document(
                manifest_raw,
                fixture.contract,
                expected_fros_allowlist_sha256=fros_sha,
                expected_anti_allowlist_sha256=anti_sha,
            )
            self.assertEqual(manifest["source"]["episodes"], 5)
            self.assertEqual(manifest["source"]["rows"], 5)
            self.assertEqual(
                [value["member"] for value in manifest["source"]["train_members"]],
                [fros.TRAIN_MEMBER],
            )
            expected = b"".join(
                fros.canonical_json_bytes(fixture.rows[episode_id]) + b"\n"
                for episode_id in ("4004", "4005")
            )
            self.assertEqual(train, expected)

    def test_manifest_source_and_member_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = GeneralFixture(root)
            fros_sha = fros.sha256_file(fixture.fros_allowlist)
            anti_sha = fros.sha256_file(fixture.anti_allowlist)
            _audit, _archive, manifest_raw, _metadata = general.audit_and_serialize(
                fixture.source,
                fixture.fros_allowlist,
                fixture.anti_allowlist,
                fros_sha,
                anti_sha,
                fixture.contract,
            )
            document = json.loads(manifest_raw)
            for mutate in (
                lambda value: value["source"].__setitem__("sha256", "0" * 64),
                lambda value: value["source"]["train_members"][0].__setitem__(
                    "bytes", value["source"]["train_members"][0]["bytes"] + 1
                ),
                lambda value: value["source"]["train_members"][0].__setitem__(
                    "sha256", "1" * 64
                ),
                lambda value: value["members"][0].__setitem__("sha256", "0" * 64),
            ):
                tampered = json.loads(json.dumps(document))
                mutate(tampered)
                with self.assertRaisesRegex(RuntimeError, "contract drift"):
                    general.validate_manifest_document(
                        tampered,
                        fixture.contract,
                        expected_fros_allowlist_sha256=fros_sha,
                        expected_anti_allowlist_sha256=anti_sha,
                    )


if __name__ == "__main__":
    unittest.main()
