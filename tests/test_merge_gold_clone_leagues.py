from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "merge_gold_clone_leagues",
    TOOLS_ROOT / "merge_gold_clone_leagues.py",
)
assert SPEC is not None and SPEC.loader is not None
merge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merge
SPEC.loader.exec_module(merge)


def deck_hash(cards: list[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MergeGoldCloneLeaguesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_league(
        self,
        directory_name: str,
        entries: list[tuple[str, bytes, int]],
        *,
        safe: bool = True,
        quality_pass: bool = True,
        excluded: list[dict] | None = None,
    ) -> Path:
        directory = self.root / directory_name
        directory.mkdir()
        opponents: list[dict] = []
        for policy_id, checkpoint_bytes, card_offset in entries:
            checkpoint = directory / f"{policy_id}.pt"
            checkpoint.write_bytes(checkpoint_bytes)
            cards = [card_offset + index for index in range(60)]
            deck = directory / f"{policy_id}-deck.csv"
            deck.write_text(
                "\n".join(str(card) for card in cards) + "\n",
                encoding="utf-8",
            )
            opponents.append(
                {
                    "policy_id": policy_id,
                    "submission_id": card_offset,
                    "team_name": f"team-{policy_id}",
                    "archetype": "test",
                    # Relative paths exercise input-manifest path resolution.
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": merge.file_sha256(checkpoint),
                    "deck": deck.name,
                    "deck_hash": deck_hash(cards),
                    "name": f"{checkpoint.stem}@{deck.stem}",
                    "quality": {
                        "pass": quality_pass,
                        "rows": 100,
                        "checks": {
                            "minimum_rows": quality_pass,
                            "minimum_accuracy": quality_pass,
                        },
                    },
                    "ppo_cli_args": [
                        "--extra-opponent",
                        checkpoint.name,
                        deck.name,
                    ],
                }
            )
        manifest = {
            "schema_version": merge.LEAGUE_SCHEMA,
            "episode_isolation_required": safe,
            "frozen_holdout_excluded_from_training": safe,
            "opponents": opponents,
            "excluded": excluded or [],
            "ppo_cli_args": [],
            "safety": {
                "uses_public_replay_actions_only": safe,
                "uses_open_submission_code": not safe,
                "uploads_or_submissions_performed": not safe,
            },
        }
        path = directory / "league_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def load_inputs(
        self,
        base: Path,
        *recoveries: Path,
    ) -> list[merge.InputLeague]:
        return [
            merge.load_input_league(
                path,
                index,
                "base" if index == 0 else "recovery",
            )
            for index, path in enumerate((base, *recoveries))
        ]

    def test_merges_distinct_entries_and_records_all_sources(self) -> None:
        base = self.make_league(
            "base",
            [("policy-a", b"checkpoint-a", 1)],
            excluded=[{"policy_id": "old-reject", "status": "quality_rejected"}],
        )
        recovery = self.make_league(
            "recovery",
            [("policy-b", b"checkpoint-b", 101)],
        )
        inputs = self.load_inputs(base, recovery)
        result = merge.merge_leagues(
            inputs,
            prefer_later=False,
            checkpoint_dir=None,
        )
        self.assertEqual(
            [entry["policy_id"] for entry in result["opponents"]],
            ["policy-a", "policy-b"],
        )
        self.assertEqual(len(result["merge"]["inputs"]), 2)
        self.assertEqual(
            result["merge"]["inputs"][0]["manifest_sha256"],
            merge.file_sha256(base),
        )
        self.assertEqual(len(result["excluded"]), 1)
        self.assertEqual(result["excluded"][0]["source"]["role"], "base")
        self.assertEqual(result["rejected"], [])
        self.assertEqual(len(result["ppo_cli_args"]), 6)

    def test_duplicate_requires_explicit_prefer_later(self) -> None:
        base = self.make_league(
            "base",
            [("same-policy", b"base-checkpoint", 1)],
        )
        recovery = self.make_league(
            "recovery",
            [("same-policy", b"recovery-checkpoint", 1)],
        )
        with self.assertRaisesRegex(ValueError, "--prefer-later"):
            merge.merge_leagues(
                self.load_inputs(base, recovery),
                prefer_later=False,
                checkpoint_dir=None,
            )

    def test_prefer_later_replaces_and_records_history(self) -> None:
        base = self.make_league(
            "base",
            [("same-policy", b"base-checkpoint", 1)],
        )
        recovery = self.make_league(
            "recovery",
            [("same-policy", b"recovery-checkpoint", 1)],
        )
        result = merge.merge_leagues(
            self.load_inputs(base, recovery),
            prefer_later=True,
            checkpoint_dir=None,
        )
        selected = result["opponents"][0]
        self.assertEqual(
            Path(selected["checkpoint"]).read_bytes(),
            b"recovery-checkpoint",
        )
        selection = result["merge"]["selections"][0]
        self.assertEqual(selection["selected_source"]["role"], "recovery")
        self.assertEqual(
            selection["history"][-1]["action"],
            "replaced_previous",
        )

    def test_rejects_quality_failure_and_carries_excluded(self) -> None:
        base = self.make_league(
            "base",
            [("bad-quality", b"checkpoint", 1)],
            quality_pass=False,
            excluded=[{"policy_id": "already-excluded"}],
        )
        result = merge.merge_leagues(
            self.load_inputs(base),
            prefer_later=False,
            checkpoint_dir=None,
        )
        self.assertEqual(result["opponents"], [])
        self.assertIn(
            "quality_pass_not_true",
            result["rejected"][0]["reasons"],
        )
        self.assertEqual(len(result["excluded"]), 1)

    def test_rejects_unsafe_input_manifest(self) -> None:
        unsafe = self.make_league(
            "unsafe",
            [("unsafe-policy", b"checkpoint", 1)],
            safe=False,
        )
        result = merge.merge_leagues(
            self.load_inputs(unsafe),
            prefer_later=False,
            checkpoint_dir=None,
        )
        self.assertEqual(result["opponents"], [])
        reasons = result["rejected"][0]["reasons"]
        self.assertIn("public_replay_only_not_verified", reasons)
        self.assertIn("episode_isolation_not_verified", reasons)

    def test_rejects_checkpoint_and_deck_hash_mismatches(self) -> None:
        path = self.make_league(
            "bad-hashes",
            [
                ("bad-checkpoint", b"checkpoint-a", 1),
                ("bad-deck", b"checkpoint-b", 101),
            ],
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        value["opponents"][0]["checkpoint_sha256"] = "0" * 64
        value["opponents"][1]["deck_hash"] = "f" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
        result = merge.merge_leagues(
            self.load_inputs(path),
            prefer_later=False,
            checkpoint_dir=None,
        )
        reasons = {
            item["policy_id"]: item["reasons"]
            for item in result["rejected"]
        }
        self.assertIn(
            "checkpoint_sha256_mismatch",
            reasons["bad-checkpoint"],
        )
        self.assertIn("deck_hash_mismatch", reasons["bad-deck"])

    def test_copy_mode_verifies_and_never_overwrites_different_file(self) -> None:
        base = self.make_league(
            "base",
            [("policy-a", b"checkpoint-a", 1)],
        )
        checkpoint_dir = self.root / "merged-checkpoints"
        result = merge.merge_leagues(
            self.load_inputs(base),
            prefer_later=False,
            checkpoint_dir=checkpoint_dir,
        )
        copied = Path(result["opponents"][0]["checkpoint"])
        self.assertEqual(copied.parent, checkpoint_dir.resolve())
        self.assertEqual(copied.read_bytes(), b"checkpoint-a")
        self.assertEqual(
            result["opponents"][0]["merge_provenance"]["materialization"],
            "copied",
        )

        # Identical reruns are allowed and explicitly reported as reuse.
        rerun = merge.merge_leagues(
            self.load_inputs(base),
            prefer_later=False,
            checkpoint_dir=checkpoint_dir,
        )
        self.assertEqual(
            rerun["opponents"][0]["merge_provenance"]["materialization"],
            "reused_identical",
        )
        copied.write_bytes(b"different")
        with self.assertRaisesRegex(FileExistsError, "overwrite different"):
            merge.merge_leagues(
                self.load_inputs(base),
                prefer_later=False,
                checkpoint_dir=checkpoint_dir,
            )

    def test_output_manifest_needs_explicit_overwrite(self) -> None:
        output = self.root / "merged.json"
        value = {"schema_version": merge.LEAGUE_SCHEMA}
        merge.atomic_write_json(output, value, overwrite=False)
        with self.assertRaisesRegex(FileExistsError, "overwrite-output"):
            merge.atomic_write_json(output, value, overwrite=False)
        replacement = {
            "schema_version": merge.LEAGUE_SCHEMA,
            "changed": True,
        }
        merge.atomic_write_json(output, replacement, overwrite=True)
        self.assertTrue(json.loads(output.read_text())["changed"])


if __name__ == "__main__":
    unittest.main()
