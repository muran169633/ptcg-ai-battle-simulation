from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
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

import build_marnie_exact_anti_kd as anti
import build_marnie_froslass_exact_wins as common


def deck(start: int) -> list[int]:
    return list(range(start, start + 60))


def deck_hash(cards: list[int]) -> str:
    payload = ",".join(str(value) for value in sorted(cards)).encode()
    return hashlib.sha256(payload).hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    return common.zip_info(name)


class AntiFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "source.zip"
        self.daily_dir = root / "daily"
        self.daily_dir.mkdir()
        self.dates = ("2026-08-02", "2026-08-03")
        self.marnie = deck(100)
        self.other = deck(300)
        self.marnie_hash = deck_hash(self.marnie)
        self.names = ("KD", "Primary", "Secondary", "Other")
        specs = (
            ("2001", "2026-08-02", 0, "Primary", "KD", 2),
            ("2002", "2026-08-02", 1, "Secondary", "KD", 1),
            ("2003", "2026-08-03", 0, "Other", "KD", 1),
            ("2004", "2026-08-03", 0, "Primary", "Other", 1),
            ("2005", "2026-08-03", 0, "Primary", "KD", 1),
        )
        self.rows: dict[str, list[dict[str, object]]] = {}
        for episode_id, date, seat, learner, opponent, row_count in specs:
            self.rows[episode_id] = [
                self._row(
                    episode_id,
                    date,
                    seat,
                    learner,
                    opponent,
                    step,
                )
                for step in range(1, row_count + 1)
            ]
        self.source_lines = tuple(
            common.canonical_json_bytes(row) + b"\n"
            for episode_id, *_rest in specs
            for row in self.rows[episode_id]
        )
        self._write_source()
        self._write_daily(
            "2026-08-02",
            {
                "2001": self._replay(
                    "2001", ("Primary", "KD"), (1, -1), (self.marnie, self.marnie)
                ),
                "2002": self._replay(
                    "2002", ("KD", "Secondary"), (-1, 1), (self.marnie, self.marnie)
                ),
            },
        )
        self._write_daily(
            "2026-08-03",
            {
                "2003": self._replay(
                    "2003", ("Other", "KD"), (1, -1), (self.marnie, self.marnie)
                ),
                "2004": self._replay(
                    "2004", ("Primary", "Other"), (1, -1), (self.marnie, self.marnie)
                ),
                "2005": self._replay(
                    "2005", ("Primary", "KD"), (1, -1), (self.marnie, self.other)
                ),
            },
        )
        base = common.BuildContract(
            source_sha256=common.sha256_file(self.source),
            source_manifest_sha256=self.manifest_sha256,
            daily_sha256=tuple(
                (
                    date,
                    common.sha256_file(
                        self.daily_dir
                        / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
                    ),
                )
                for date in self.dates
            ),
            daily_manifest_sha256=tuple(
                (date, self._daily_manifest_sha256(date)) for date in self.dates
            ),
            daily_missing_replay_ids=tuple((date, ()) for date in self.dates),
            dates=self.dates,
            expected_date_episodes=((self.dates[0], 2), (self.dates[1], 1)),
            expected_source_train_rows=6,
            expected_source_train_episodes=5,
            expected_selected_rows=4,
            expected_selected_episodes=3,
            learner_deck_hash=self.marnie_hash,
            opponent_deck_hash=self.marnie_hash,
            source_members=("train/part-00000.jsonl", "manifest.json"),
            source_train_members=("train/part-00000.jsonl",),
            source_logical_path="fixture/source.zip",
            daily_logical_dir="fixture/daily",
            incomplete_terminal_replays=(),
        )
        self.contract = anti.AntiKDContract(
            base=base,
            top20_names=self.names,
            kd_team_name="KD",
            expected_episodes=3,
            expected_rows=4,
            expected_date_episodes=((self.dates[0], 2), (self.dates[1], 1)),
            expected_winner_groups=(
                (anti.PRIMARY_WINNER, 0),
                (anti.SECONDARY_WINNER, 0),
                (anti.TERTIARY_WINNER, 0),
                ("__others__", 3),
            ),
            train_episode_count=2,
            dev_episode_count=1,
            split_domain_separator=anti.SPLIT_DOMAIN_SEPARATOR,
        )

    def _daily_manifest_sha256(self, date: str) -> str:
        path = self.daily_dir / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
        with zipfile.ZipFile(path) as archive:
            return hashlib.sha256(archive.read("manifest.csv")).hexdigest()

    def _row(
        self,
        episode_id: str,
        date: str,
        seat: int,
        learner: str,
        opponent: str,
        step: int,
    ) -> dict[str, object]:
        return {
            "schema_version": common.ROW_SCHEMA_VERSION,
            "episode_id": episode_id,
            "episode_uuid": f"uuid-{episode_id}",
            "dataset_date": date,
            "split": "train",
            "observation_step_index": step - 1,
            "action_step_index": step,
            "seat": seat,
            "team_name": learner,
            "opponent_team_name": opponent,
            "deck_hash": self.marnie_hash,
            "terminal_reward": 1.0,
            "sample_weight": 1.0,
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
            "steps": [
                [
                    {"visualize": [{"action": [decks[0], decks[1]]}]},
                    {},
                ]
            ],
        }

    def _write_source(self) -> None:
        manifest = {
            "schema_version": common.SOURCE_SCHEMA_VERSION,
            "profile": {"deck_hash": self.marnie_hash},
            "deck_hash_filter": self.marnie_hash,
            "split_policy": {"train_dates": list(self.dates)},
            "split_decisions": {"train": 6},
            "split_episodes": {"train": 5},
            "terminal_reward_filter": {"mode": "wins", "splits": "train"},
            "team_filter": {
                "global_team_count": len(self.names),
                "display_names": list(self.names),
            },
        }
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
        self.manifest_sha256 = hashlib.sha256(payload).hexdigest()
        with zipfile.ZipFile(self.source, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(zip_info("train/part-00000.jsonl"), b"".join(self.source_lines))
            archive.writestr(zip_info("manifest.json"), payload)

    def _write_daily(
        self,
        date: str,
        replays: dict[str, dict[str, object]],
    ) -> None:
        path = self.daily_dir / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
        csv_payload = (
            "episode_id,create_time,avg_score,min_score,sum_score,agent_count,size_bytes\n"
            + "".join(
                f"{episode_id},{date}T00:00:00,1,1,2,2,100\n"
                for episode_id in sorted(replays, key=int)
            )
        ).encode()
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for episode_id in sorted(replays, key=int):
                archive.writestr(
                    zip_info(f"{episode_id}.json"),
                    common.canonical_json_bytes(replays[episode_id]),
                )
            archive.writestr(zip_info("manifest.csv"), csv_payload)


class AntiKDBuilderTests(unittest.TestCase):
    def test_isolated_cli_help_and_minimal_dry_run(self) -> None:
        script = REPO_ROOT / "tools/build_marnie_exact_anti_kd.py"
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

            copied_tools = Path(temporary) / "copied-tools"
            copied_tools.mkdir()
            copied_script = copied_tools / script.name
            shutil.copyfile(script, copied_script)
            (copied_tools / "build_marnie_froslass_exact_wins.py").symlink_to(
                REPO_ROOT / "tools/build_marnie_froslass_exact_wins.py"
            )
            symlink_result = subprocess.run(
                [sys.executable, "-I", "-B", str(copied_script), "--help"],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(symlink_result.returncode, 0)
            self.assertIn("local module path traverses symlink", symlink_result.stderr)

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

target = load("build_marnie_exact_anti_kd", Path(sys.argv[1]))
fixture_module = load("_isolated_anti_fixture", Path(sys.argv[2]))
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    fixture = fixture_module.AntiFixture(root)
    bundle = root / "never-bundle"
    result = target.build(
        source=fixture.source,
        daily_dir=fixture.daily_dir,
        archive_path=bundle / "anti.zip",
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

    def test_dry_run_and_reviewed_execute_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = AntiFixture(root)
            archive = root / "bundle" / "anti.zip"
            allowlist = root / "bundle" / anti.ALLOWLIST_MEMBER
            dry = anti.build(
                source=fixture.source,
                daily_dir=fixture.daily_dir,
                archive_path=archive,
                allowlist_path=allowlist,
                contract=fixture.contract,
                execute=False,
            )
            self.assertFalse(archive.exists())
            self.assertFalse(allowlist.exists())
            self.assertFalse(archive.parent.exists())
            self.assertEqual(dry["episode_count"], 3)
            self.assertEqual(dry["decision_rows"], 4)
            self.assertEqual(dry["split_episodes"], {"train": 2, "dev": 1})
            built = anti.build(
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
            self.assertEqual(archive.read_bytes() and built["status"], "built")
            with zipfile.ZipFile(archive) as zipped:
                embedded = zipped.read(anti.ALLOWLIST_MEMBER)
                manifest = json.loads(zipped.read(anti.MANIFEST_MEMBER))
                train = zipped.read(anti.TRAIN_MEMBER)
                dev = zipped.read(anti.DEV_MEMBER)
            self.assertEqual(embedded, allowlist.read_bytes())
            parsed = json.loads(embedded)
            self.assertEqual(
                embedded,
                common.canonical_json_bytes(parsed) + b"\n",
            )
            self.assertEqual(parsed["schema_version"], anti.ALLOWLIST_SCHEMA_VERSION)
            self.assertEqual(parsed["data_schema_version"], common.ROW_SCHEMA_VERSION)
            self.assertEqual(parsed["opponent_team_name"], "KD")
            self.assertEqual(parsed["train_episode_count"], 2)
            self.assertEqual(parsed["dev_episode_count"], 1)
            self.assertEqual(
                {value["derived_split"] for value in parsed["episodes"]},
                {"train", "dev"},
            )
            self.assertEqual(manifest["schema_version"], anti.ARCHIVE_SCHEMA_VERSION)
            self.assertEqual(manifest["data_schema_version"], common.ROW_SCHEMA_VERSION)
            self.assertEqual(manifest["split_episodes"], {"dev": 1, "train": 2})
            self.assertEqual(len(train.splitlines()) + len(dev.splitlines()), 4)
            self.assertTrue(dev.splitlines())
            self.assertTrue(
                all(json.loads(line)["split"] == "train" for line in dev.splitlines())
            )
            self.assertTrue(
                manifest["byte_preservation"][
                    "source_row_split_field_preserved_as_train"
                ]
            )

    def test_split_assignment_is_order_independent_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AntiFixture(Path(temporary))
            source = common.scan_source(fixture.source, fixture.contract.base)
            selected, _daily, _winners = anti.scan_anti_kd_replays(
                fixture.daily_dir,
                source,
                fixture.contract,
            )
            first = anti.assign_splits(selected, fixture.contract)
            second = anti.assign_splits(tuple(reversed(selected)), fixture.contract)
            self.assertEqual(first, second)
            self.assertEqual(Counter(first.values()), Counter(train=2, dev=1))

    def test_execute_rejects_unreviewed_hashes_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = AntiFixture(root)
            with self.assertRaisesRegex(RuntimeError, "requires reviewed"):
                anti.build(
                    source=fixture.source,
                    daily_dir=fixture.daily_dir,
                    archive_path=root / "bundle" / "anti.zip",
                    allowlist_path=root / "bundle" / "allowlist.json",
                    contract=fixture.contract,
                    execute=True,
                )

    def test_preexisting_bundle_directory_is_rejected_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = AntiFixture(root)
            bundle = root / "bundle"
            bundle.mkdir()
            marker = bundle / "keep.txt"
            marker.write_bytes(b"keep")
            with self.assertRaisesRegex(FileExistsError, "bundle directory"):
                anti.build(
                    source=fixture.source,
                    daily_dir=fixture.daily_dir,
                    archive_path=bundle / "anti.zip",
                    allowlist_path=bundle / anti.ALLOWLIST_MEMBER,
                    contract=fixture.contract,
                    execute=False,
                )
            self.assertEqual(marker.read_bytes(), b"keep")
            self.assertEqual(sorted(path.name for path in bundle.iterdir()), ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
