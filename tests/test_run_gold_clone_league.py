from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "run_gold_clone_league",
    TOOLS_ROOT / "run_gold_clone_league.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def deck_hash(cards: list[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_archive(
    path: Path,
    split_episodes: dict[str, list[str]],
    *,
    policy_deck_hash: str,
    team_name: str = "Gold Team",
) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for split, episodes in split_episodes.items():
            rows: list[bytes] = []
            for episode_id in episodes:
                for decision in range(2):
                    rows.append(
                        json.dumps(
                            {
                                "episode_id": episode_id,
                                "decision": decision,
                                "team_name": team_name,
                                "deck_hash": policy_deck_hash,
                                "action": [0],
                                "observation": {
                                    "current": {
                                        "yourIndex": 0,
                                        "players": [{}, {}],
                                    },
                                    "select": {
                                        "context": 7,
                                        "type": 1,
                                        "minCount": 1,
                                        "maxCount": 1,
                                        "option": [{"id": 7}],
                                    },
                                },
                            },
                            sort_keys=True,
                        ).encode("utf-8")
                        + b"\n"
                    )
                    counts[split] += 1
            archive.writestr(f"{split}/part-00000.jsonl", b"".join(rows))
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema_version": 4,
                    "dates": ["2026-07-26"],
                    "split_decisions": counts,
                }
            ),
        )
    return counts


class GoldCloneRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cards = list(range(1, 61))
        self.hash = deck_hash(self.cards)
        self.deck = self.root / "deck.csv"
        self.deck.write_text(
            "\n".join(str(card) for card in self.cards) + "\n",
            encoding="utf-8",
        )
        self.archive = self.root / "policy.zip"
        self.counts = write_archive(
            self.archive,
            {
                "train": ["tr-1", "tr-2", "tr-3", "tr-4"],
                "valid": ["ho-1", "ho-2"],
            },
            policy_deck_hash=self.hash,
        )
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": runner.INPUT_SCHEMA,
                    "policies": [
                        {
                            "policy_id": "team/submission 42",
                            "submission_id": 42,
                            "team_name": "Gold Team",
                            "deck_hash": self.hash,
                            "archetype": "Marnie",
                            "archive_path": self.archive.name,
                            "train_split": "train",
                            "holdout_split": "valid",
                            "deck_path": self.deck.name,
                            "train": {
                                "episodes": 4,
                                "decisions": self.counts["train"],
                            },
                            "holdout": {
                                "episodes": 2,
                                "decisions": self.counts["valid"],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_manifest_parse_and_episode_isolation(self) -> None:
        _, policies = runner.load_pipeline_manifest(self.manifest)
        self.assertEqual(len(policies), 1)
        policy = policies[0]
        self.assertEqual(policy.train_archive, self.archive.resolve())
        self.assertEqual(policy.holdout_archive, self.archive.resolve())
        self.assertEqual(policy.deck_path, self.deck.resolve())
        train, holdout = runner.inventory_policy(policy)
        self.assertEqual(train.episode_ids, {"tr-1", "tr-2", "tr-3", "tr-4"})
        self.assertEqual(holdout.episode_ids, {"ho-1", "ho-2"})

    def test_leakage_is_rejected(self) -> None:
        leaking = self.root / "leaking.zip"
        write_archive(
            leaking,
            {
                "train": ["shared", "tr-2"],
                "valid": ["shared"],
            },
            policy_deck_hash=self.hash,
        )
        raw = json.loads(self.manifest.read_text(encoding="utf-8"))
        policy_raw = raw["policies"][0]
        policy_raw["archive_path"] = leaking.name
        policy_raw["train"] = {}
        policy_raw["holdout"] = {}
        self.manifest.write_text(json.dumps(raw), encoding="utf-8")
        _, policies = runner.load_pipeline_manifest(self.manifest)
        with self.assertRaisesRegex(ValueError, "episode leakage"):
            runner.inventory_policy(policies[0])

    def test_derived_archive_never_contains_frozen_holdout(self) -> None:
        _, policies = runner.load_pipeline_manifest(self.manifest)
        policy = policies[0]
        train, _ = runner.inventory_policy(policy)
        destination = self.root / "derived.zip"
        manifest = runner.build_derived_train_archive(
            policy,
            destination,
            train,
            seed=7,
            dev_fraction=0.25,
            min_dev_episodes=1,
            source_sha256=runner.file_sha256(self.archive),
            resume=False,
        )
        derived_train = runner.scan_split(destination, "train")
        derived_dev = runner.scan_split(destination, "valid")
        self.assertFalse(derived_train.episode_ids & derived_dev.episode_ids)
        self.assertEqual(
            derived_train.episode_ids | derived_dev.episode_ids,
            train.episode_ids,
        )
        self.assertFalse(
            {"ho-1", "ho-2"}
            & (derived_train.episode_ids | derived_dev.episode_ids)
        )
        self.assertTrue(
            manifest["split_policy"]["frozen_holdout_excluded"]
        )
        self.assertEqual(
            manifest["split_policy"]["row_split_contract"],
            runner.ROW_SPLIT_CONTRACT,
        )
        with zipfile.ZipFile(destination) as archive:
            for member in archive.namelist():
                if not member.endswith(".jsonl"):
                    continue
                member_split = member.split("/", 1)[0]
                with archive.open(member) as handle:
                    for line in handle:
                        row = json.loads(line)
                        self.assertEqual(row["split"], member_split)

        import train_bc_orbit as bc

        valid_dataset = bc.ZipDecisionDataset(
            archive_path=destination,
            split="valid",
            max_rows=None,
            split_seed=7,
            shuffle_seed=11,
            epoch=0,
            hash_size=bc.DEFAULT_HASH_SIZE,
            max_state_entities=bc.DEFAULT_MAX_STATE_ENTITIES,
            use_trajectory_weights=False,
            deck_hashes=(),
            team_names=(),
            split_mode="archive",
        )
        valid_rows = list(valid_dataset)
        self.assertEqual(len(valid_rows), manifest["split_decisions"]["valid"])

    def test_resume_rejects_stale_row_split_archive(self) -> None:
        _, policies = runner.load_pipeline_manifest(self.manifest)
        policy = policies[0]
        train, _ = runner.inventory_policy(policy)
        destination = self.root / "stale-derived.zip"
        source_sha = runner.file_sha256(self.archive)
        stale_signature = runner.derived_signature(
            policy,
            source_sha256=source_sha,
            seed=7,
            dev_fraction=0.25,
            min_dev_episodes=1,
        )
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr(
                "train/part-00000.jsonl",
                json.dumps({"episode_id": "tr-1", "split": "train"}) + "\n",
            )
            archive.writestr(
                "valid/part-00000.jsonl",
                # Reproduce the original bug: member is valid, row says train.
                json.dumps({"episode_id": "tr-2", "split": "train"}) + "\n",
            )
            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "schema_version": runner.DERIVED_SCHEMA,
                        "clone_runner_signature": stale_signature,
                        "split_policy": {
                            "row_split_contract": runner.ROW_SPLIT_CONTRACT,
                        },
                        "split_decisions": {
                            "train": 1,
                            "valid": 1,
                            "test": 0,
                        },
                    }
                ),
            )
        with self.assertRaisesRegex(ValueError, "does not match member prefix"):
            runner.build_derived_train_archive(
                policy,
                destination,
                train,
                seed=7,
                dev_fraction=0.25,
                min_dev_episodes=1,
                source_sha256=source_sha,
                resume=True,
            )

    def test_commands_use_init_checkpoint_and_original_holdout(self) -> None:
        _, policies = runner.load_pipeline_manifest(self.manifest)
        policy = policies[0]
        base = self.root / "base.pt"
        base.write_bytes(b"checkpoint")
        args = argparse.Namespace(
            epochs=2,
            batch_size=8,
            workers=0,
            learning_rate=1e-4,
            weight_decay=1e-4,
            categorical_dim=64,
            model_dim=128,
            layers=4,
            heads=4,
            dropout=0.05,
            seed=11,
            train_target_accuracy=0.6,
            device="cpu",
            max_train_rows=12,
            max_valid_rows=4,
            base_checkpoint=base,
            use_trajectory_weights=False,
            eval_batch_size=16,
            eval_workers=0,
            eval_progress_interval=0,
            max_holdout_rows=None,
        )
        derived = self.root / "derived.zip"
        train_output = self.root / "train"
        train_command = runner.build_train_command(
            args,
            policy,
            derived,
            train_output,
            12,
        )
        self.assertIn("--init-checkpoint", train_command)
        self.assertEqual(
            train_command[train_command.index("--data") + 1],
            str(derived),
        )
        eval_output = self.root / "eval.json"
        eval_command = runner.build_eval_command(
            args,
            policy,
            train_output / "best.pt",
            eval_output,
        )
        self.assertEqual(
            eval_command[eval_command.index("--data") + 1],
            str(self.archive.resolve()),
        )
        self.assertEqual(
            eval_command[eval_command.index("--split") + 1],
            "valid",
        )

    def test_quality_requires_all_three_accuracy_gates(self) -> None:
        result = {
            "metrics": {
                "rows": 400,
                "set_exact_accuracy": 0.72,
                "ordered_exact_accuracy": 0.59,
                "hybrid_order_exact_accuracy": 0.70,
            }
        }
        quality = runner.quality_from_evaluation(
            result,
            min_rows=300,
            min_set=0.65,
            min_ordered=0.60,
            min_hybrid=0.60,
        )
        self.assertFalse(quality["pass"])
        self.assertFalse(
            quality["checks"]["minimum_ordered_exact_accuracy"]
        )
        result["metrics"]["ordered_exact_accuracy"] = 0.61
        quality = runner.quality_from_evaluation(
            result,
            min_rows=300,
            min_set=0.65,
            min_ordered=0.60,
            min_hybrid=0.60,
        )
        self.assertTrue(quality["pass"])

    def test_dry_run_validates_and_writes_nothing(self) -> None:
        output = self.root / "dry-output"
        args = runner.parse_args(
            [
                "--manifest",
                str(self.manifest),
                "--output-dir",
                str(output),
                "--dry-run",
                "--device",
                "cpu",
                "--min-train-episodes",
                "2",
                "--min-train-rows",
                "2",
                "--min-holdout-episodes",
                "1",
                "--min-holdout-rows",
                "2",
            ]
        )
        summary = runner.run(args)
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["results"][0]["status"], "dry_run")
        self.assertTrue(
            summary["plans"][0]["frozen_holdout_excluded_from_training"]
        )
        self.assertFalse(output.exists())

    def test_deck_hash_matches_train_ppo_convention(self) -> None:
        self.assertEqual(runner.compute_deck_hash(self.deck), self.hash)


if __name__ == "__main__":
    unittest.main()
