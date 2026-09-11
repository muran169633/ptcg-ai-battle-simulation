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
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_marnie_exact_fros_valid_view as view
import build_marnie_froslass_exact_wins as common


def deck(start: int) -> list[int]:
    return list(range(start, start + 60))


def deck_hash(cards: list[int]) -> str:
    raw = ",".join(str(value) for value in sorted(cards)).encode()
    return hashlib.sha256(raw).hexdigest()


class ViewFixture:
    def __init__(self, root: Path, *, mirror_count: int = 0) -> None:
        self.root = root
        self.source = root / "marnie.zip"
        self.daily = root / "daily.zip"
        self.date = "2026-08-07"
        self.marnie = deck(100)
        self.fros = deck(200)
        self.other = deck(300)
        self.marnie_hash = deck_hash(self.marnie)
        self.fros_hash = deck_hash(self.fros)
        view_specs: list[tuple[str, int, str, str, float, int]] = [
            ("3001", 0, "Learner1", "FrosA", 1.0, 2),
            ("3002", 1, "Learner2", "FrosB", -1.0, 1),
        ]
        self.replays = {
            "3001": self._replay(
                "3001", ("Learner1", "FrosA"), (1, -1), (self.marnie, self.fros)
            ),
            "3002": self._replay(
                "3002", ("FrosB", "Learner2"), (1, -1), (self.fros, self.marnie)
            ),
        }
        top20 = ["Learner1", "Learner2"]
        self.mirror_episode_ids = tuple(
            str(3101 + index) for index in range(mirror_count)
        )
        if mirror_count:
            for index, episode_id in enumerate(self.mirror_episode_ids, 1):
                first = f"Mirror{index}A"
                second = f"Mirror{index}B"
                top20.extend((first, second))
                view_specs.extend(
                    (
                        (episode_id, 0, first, second, 1.0, 1),
                        (episode_id, 1, second, first, -1.0, 1),
                    )
                )
                self.replays[episode_id] = self._replay(
                    episode_id,
                    (first, second),
                    (1, -1),
                    (self.marnie, self.marnie),
                )
        else:
            top20.append("Learner3")
            view_specs.append(("3003", 0, "Learner3", "Other", 1.0, 1))
            self.replays["3003"] = self._replay(
                "3003", ("Learner3", "Other"), (1, -1), (self.marnie, self.other)
            )
        self.top20 = tuple(top20)
        self.rows: dict[str, list[dict[str, object]]] = {}
        view_rows: list[list[dict[str, object]]] = []
        for episode_id, seat, learner, opponent, reward, row_count in view_specs:
            rows = [
                self._row(
                    episode_id,
                    seat,
                    learner,
                    opponent,
                    reward,
                    step,
                )
                for step in range(1, row_count + 1)
            ]
            view_rows.append(rows)
            self.rows.setdefault(episode_id, []).extend(rows)
        self.lines = tuple(
            common.canonical_json_bytes(row) + b"\n"
            for rows in view_rows
            for row in rows
        )
        self._write_source()
        self._write_daily()
        self.contract = view.ValidContract(
            source_sha256=common.sha256_file(self.source),
            source_manifest_sha256=self.manifest_sha,
            daily_sha256=common.sha256_file(self.daily),
            daily_manifest_sha256=self.daily_manifest_sha,
            daily_missing_replay_ids=(),
            dataset_date=self.date,
            learner_deck_hash=self.marnie_hash,
            opponent_deck_hash=self.fros_hash,
            source_members=(view.VALID_MEMBER, view.MANIFEST_MEMBER),
            valid_member=view.VALID_MEMBER,
            source_valid_episodes=len(self.replays),
            source_valid_views=len(view_specs),
            source_mirror_episodes=mirror_count,
            source_mirror_episode_ids=self.mirror_episode_ids,
            source_valid_rows=len(self.lines),
            expected_episodes=2,
            expected_rows=3,
            expected_wins=1,
            expected_losses=1,
            top20_names=self.top20,
            source_logical_path="fixture/marnie.zip",
            daily_logical_path="fixture/daily.zip",
        )

    def _row(
        self,
        episode_id: str,
        seat: int,
        learner: str,
        opponent: str,
        reward: float,
        step: int,
    ) -> dict[str, object]:
        return {
            "schema_version": common.ROW_SCHEMA_VERSION,
            "episode_id": episode_id,
            "episode_uuid": f"uuid-{episode_id}",
            "dataset_date": self.date,
            "split": "valid",
            "observation_step_index": step - 1,
            "action_step_index": step,
            "seat": seat,
            "team_name": learner,
            "opponent_team_name": opponent,
            "deck_hash": self.marnie_hash,
            "terminal_reward": reward,
            "sample_weight": 1.0 if reward > 0 else 0.25,
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
            "info": {"EpisodeId": int(episode_id), "TeamNames": list(names)},
            "rewards": list(rewards),
            "steps": [[{"visualize": [{"action": list(decks)}]}, {}]],
        }

    def _write_source(self) -> None:
        manifest = {
            "schema_version": common.ROW_SCHEMA_VERSION,
            "profile": {"deck_hash": self.marnie_hash},
            "split_policy": {"valid_dates": [self.date]},
            "split_decisions": {"valid": len(self.lines)},
            "split_episodes": {"valid": len(self.replays)},
            "team_filter": {"display_names": list(self.top20)},
        }
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
        self.manifest_sha = hashlib.sha256(payload).hexdigest()
        with zipfile.ZipFile(self.source, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(common.zip_info(view.VALID_MEMBER), b"".join(self.lines))
            archive.writestr(common.zip_info(view.MANIFEST_MEMBER), payload)

    def _write_daily(self) -> None:
        manifest = (
            "episode_id,create_time,avg_score,min_score,sum_score,agent_count,size_bytes\n"
            + "".join(
                f"{episode_id},{self.date}T00:00:00,1,1,2,2,100\n"
                for episode_id in sorted(self.replays, key=int)
            )
        ).encode()
        self.daily_manifest_sha = hashlib.sha256(manifest).hexdigest()
        with zipfile.ZipFile(self.daily, "w", zipfile.ZIP_DEFLATED) as archive:
            for episode_id in sorted(self.replays, key=int):
                archive.writestr(
                    common.zip_info(f"{episode_id}.json"),
                    common.canonical_json_bytes(self.replays[episode_id]),
                )
            archive.writestr(common.zip_info("manifest.csv"), manifest)


class FrosValidViewTests(unittest.TestCase):
    def test_default_aug7_missing_and_mirror_contract_is_exact(self) -> None:
        contract = view.DEFAULT_CONTRACT
        self.assertEqual(
            (
                contract.source_valid_episodes,
                contract.source_valid_views,
                contract.source_mirror_episodes,
            ),
            (295, 299, 4),
        )
        self.assertEqual(
            contract.source_mirror_episode_ids,
            ("90613480", "90626128", "90655502", "90789414"),
        )
        self.assertEqual(
            contract.daily_missing_replay_ids,
            (
                "90658419",
                "90684109",
                "90759581",
                "90765292",
                "90836433",
                "90844697",
            ),
        )
        self.assertEqual(
            view.dataclasses_contract(contract).incomplete_terminal_replays,
            (),
        )
        self.assertEqual(view.plan_document(contract)["incomplete_terminal_replays"], [])
        self.assertEqual(
            view.plan_document(contract)["daily"][
                "incomplete_terminal_replay_ids"
            ],
            [],
        )

    def test_multiple_target_matching_views_and_duplicate_numeric_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mirror = ViewFixture(root, mirror_count=1)
            _manifest, source_views, _member_sha = view.scan_source(
                mirror.source,
                mirror.contract,
            )
            with mock.patch.object(
                view.common,
                "replay_deck_hashes",
                side_effect=(
                    (mirror.marnie_hash, mirror.fros_hash),
                    (mirror.fros_hash, mirror.marnie_hash),
                    (mirror.marnie_hash, mirror.fros_hash),
                    (mirror.fros_hash, mirror.marnie_hash),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "matched multiple learner views"
                ):
                    view.select_replays(
                        mirror.daily,
                        source_views,
                        mirror.contract,
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ViewFixture(root)
            with zipfile.ZipFile(fixture.daily) as archive:
                duplicate_payload = archive.read("3001.json")
            with zipfile.ZipFile(fixture.daily, "a") as archive:
                archive.writestr("./3001.json", duplicate_payload)
            contract = dataclasses.replace(
                fixture.contract,
                daily_sha256=common.sha256_file(fixture.daily),
            )
            _manifest, source_views, _member_sha = view.scan_source(
                fixture.source,
                contract,
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate numeric replay ID"):
                view.select_replays(fixture.daily, source_views, contract)

    def test_four_exact_dual_seat_views_pass_and_fifth_mirror_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            four_root = root / "four"
            four_root.mkdir()
            fixture = ViewFixture(four_root, mirror_count=4)
            source_manifest, source_views, _member_sha = view.scan_source(
                fixture.source,
                fixture.contract,
            )
            self.assertTrue(source_manifest)
            self.assertEqual(len(source_views), 10)
            for episode_id in fixture.mirror_episode_ids:
                episode_views = [
                    value
                    for value in source_views.values()
                    if value.episode_id == episode_id
                ]
                self.assertEqual({value.seat for value in episode_views}, {0, 1})
                self.assertEqual(len({value.team_name for value in episode_views}), 2)
            dry = view.build(
                source=fixture.source,
                daily=fixture.daily,
                archive_path=root / "never" / "valid.zip",
                allowlist_path=root / "never" / view.ALLOWLIST_MEMBER,
                contract=fixture.contract,
                execute=False,
            )
            self.assertEqual(dry["episode_count"], 2)
            self.assertEqual(dry["decision_rows"], 3)
            self.assertFalse((root / "never").exists())

            five_root = root / "five"
            five_root.mkdir()
            fifth = ViewFixture(five_root, mirror_count=5)
            frozen_four = dataclasses.replace(
                fifth.contract,
                source_valid_views=fifth.contract.source_valid_episodes + 4,
                source_mirror_episodes=4,
                source_mirror_episode_ids=fifth.mirror_episode_ids[:4],
            )
            with self.assertRaisesRegex(RuntimeError, "source count drift"):
                view.scan_source(fifth.source, frozen_four)

    def test_isolated_cli_help_and_minimal_dry_run(self) -> None:
        script = REPO_ROOT / "tools/build_marnie_exact_fros_valid_view.py"
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

target = load("build_marnie_exact_fros_valid_view", Path(sys.argv[1]))
fixture_module = load("_isolated_fros_valid_fixture", Path(sys.argv[2]))
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    fixture = fixture_module.ViewFixture(root)
    bundle = root / "never-bundle"
    result = target.build(
        source=fixture.source,
        daily=fixture.daily,
        archive_path=bundle / "valid.zip",
        allowlist_path=bundle / target.ALLOWLIST_MEMBER,
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

    def test_dry_run_and_atomic_reviewed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ViewFixture(root)
            archive = root / "bundle" / "valid.zip"
            allowlist = root / "bundle" / view.ALLOWLIST_MEMBER
            dry = view.build(
                source=fixture.source,
                daily=fixture.daily,
                archive_path=archive,
                allowlist_path=allowlist,
                contract=fixture.contract,
                execute=False,
            )
            self.assertFalse(archive.parent.exists())
            self.assertEqual(dry["episode_count"], 2)
            self.assertEqual(dry["decision_rows"], 3)
            self.assertEqual((dry["win_episodes"], dry["loss_episodes"]), (1, 1))
            view.build(
                source=fixture.source,
                daily=fixture.daily,
                archive_path=archive,
                allowlist_path=allowlist,
                contract=fixture.contract,
                execute=True,
                expected_plan_sha256=dry["plan_sha256"],
                expected_archive_sha256=dry["archive_sha256"],
                expected_allowlist_sha256=dry["allowlist_sha256"],
            )
            with zipfile.ZipFile(archive) as zipped:
                embedded = zipped.read(view.ALLOWLIST_MEMBER)
                manifest = json.loads(zipped.read(view.MANIFEST_MEMBER))
                rows = zipped.read(view.VALID_MEMBER)
            self.assertEqual(embedded, allowlist.read_bytes())
            document = view.validate_allowlist_document(embedded, fixture.contract)
            self.assertEqual(document["date_episode_counts"], {fixture.date: 2})
            self.assertEqual(document["win_episodes"], 1)
            self.assertEqual(document["loss_episodes"], 1)
            self.assertEqual(manifest["data_schema_version"], common.ROW_SCHEMA_VERSION)
            expected_rows = b"".join(
                common.canonical_json_bytes(row) + b"\n"
                for episode_id in ("3001", "3002")
                for row in fixture.rows[episode_id]
            )
            self.assertEqual(rows, expected_rows)
            self.assertEqual(manifest["temporal_isolation"]["aug8_input_count"], 0)

    def test_wrong_reviewed_hash_leaves_bundle_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ViewFixture(root)
            archive = root / "bundle" / "valid.zip"
            allowlist = root / "bundle" / view.ALLOWLIST_MEMBER
            dry = view.build(
                source=fixture.source,
                daily=fixture.daily,
                archive_path=archive,
                allowlist_path=allowlist,
                contract=fixture.contract,
                execute=False,
            )
            with self.assertRaisesRegex(RuntimeError, "gate mismatch"):
                view.build(
                    source=fixture.source,
                    daily=fixture.daily,
                    archive_path=archive,
                    allowlist_path=allowlist,
                    contract=fixture.contract,
                    execute=True,
                    expected_plan_sha256="0" * 64,
                    expected_archive_sha256=dry["archive_sha256"],
                    expected_allowlist_sha256=dry["allowlist_sha256"],
                )
            self.assertFalse(archive.parent.exists())


if __name__ == "__main__":
    unittest.main()
