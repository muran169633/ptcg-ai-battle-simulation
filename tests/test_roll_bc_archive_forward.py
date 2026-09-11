from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import roll_bc_archive_forward as roll  # noqa: E402


SCHEMA = "ptcg-bc-visible-decisions-v1"
DATES = ("2026-07-16", "2026-07-17", "2026-07-18")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_ids_for_both_splits(
    seed: int,
    threshold: int,
) -> tuple[list[str], list[str]]:
    train: list[str] = []
    valid: list[str] = []
    index = 0
    while len(train) < 3 or len(valid) < 3:
        episode_id = f"fresh-{index:03d}"
        target = roll.assigned_split(episode_id, seed, threshold)
        (train if target == "train" else valid).append(episode_id)
        index += 1
    return train[:3], valid[:3]


def write_source_archive(
    path: Path,
    *,
    mismatch: bool = False,
) -> dict[str, list[dict[str, object]]]:
    seed = 71
    threshold = int(0.8 * roll.UINT64_SPACE)
    hashed_train, hashed_valid = episode_ids_for_both_splits(seed, threshold)
    rows: dict[str, list[dict[str, object]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    episode_layout = {
        "train": [("old-train", 2)],
        "valid": [("old-valid", 3)],
        "test": [
            *[(episode_id, 1) for episode_id in hashed_train],
            *[(episode_id, 2) for episode_id in hashed_valid],
        ],
    }
    for split_index, split in enumerate(roll.SOURCE_SPLITS):
        for episode_id, row_count in episode_layout[split]:
            for row_index in range(row_count):
                rows[split].append(
                    {
                        "schema_version": SCHEMA,
                        "episode_id": episode_id,
                        "episode_uuid": f"uuid-{episode_id}",
                        "dataset_date": DATES[split_index],
                        "split": (
                            "valid"
                            if mismatch and split == "train" and row_index == 0
                            else split
                        ),
                        "team_name": f"team-{split}",
                        "select_context": str(split_index),
                        "action": [row_index],
                        "observation": {"row": row_index},
                    }
                )
    manifest = {
        "schema_version": SCHEMA,
        "competition": "pokemon-tcg-ai-battle",
        "dates": list(DATES),
        "split_policy": {
            "mode": "time",
            "train_dates": [DATES[0]],
            "valid_dates": [DATES[1]],
            "test_dates": [DATES[2]],
        },
        "stats": {"decisions": sum(map(len, rows.values()))},
        "split_decisions": {
            split: len(rows[split]) for split in roll.SOURCE_SPLITS
        },
        "shards": {split: 1 for split in roll.SOURCE_SPLITS},
        "deck_hash_filter": "synthetic-deck",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for split in roll.SOURCE_SPLITS:
            payload = b"".join(
                roll.canonical_row_bytes(row) for row in rows[split]
            )
            archive.writestr(f"{split}/part-00000.jsonl", payload)
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, sort_keys=True).encode("utf-8"),
        )
    return rows


def read_output_rows(path: Path) -> dict[str, list[dict[str, object]]]:
    result = {split: [] for split in roll.OUTPUT_SPLITS}
    with zipfile.ZipFile(path) as archive:
        for split in roll.OUTPUT_SPLITS:
            for member in sorted(
                name
                for name in archive.namelist()
                if name.startswith(f"{split}/") and name.endswith(".jsonl")
            ):
                with archive.open(member) as handle:
                    result[split].extend(json.loads(line) for line in handle)
    return result


class RollBcArchiveForwardTests(unittest.TestCase):
    def test_roll_forward_is_lossless_disjoint_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.zip"
            first_output = root / "first.zip"
            second_output = root / "second.zip"
            source_rows = write_source_archive(source)
            source_before = file_sha256(source)

            first_result = roll.roll_archive(
                source,
                first_output,
                seed=71,
                train_fraction=0.8,
                rows_per_shard=3,
            )
            second_result = roll.roll_archive(
                source,
                second_output,
                seed=71,
                train_fraction=0.8,
                rows_per_shard=3,
            )

            self.assertEqual(source_before, file_sha256(source))
            self.assertEqual(file_sha256(first_output), file_sha256(second_output))
            self.assertEqual(first_result["episode_id_overlap"], 0)
            self.assertEqual(first_result["sha256"], second_result["sha256"])

            output_rows = read_output_rows(first_output)
            output_total = sum(map(len, output_rows.values()))
            source_total = sum(map(len, source_rows.values()))
            self.assertEqual(output_total, source_total)
            self.assertTrue(
                all(row["split"] == "train" for row in output_rows["train"])
            )
            self.assertTrue(
                all(row["split"] == "valid" for row in output_rows["valid"])
            )
            output_ids = {
                split: {str(row["episode_id"]) for row in output_rows[split]}
                for split in roll.OUTPUT_SPLITS
            }
            self.assertFalse(output_ids["train"] & output_ids["valid"])
            self.assertIn("old-train", output_ids["train"])
            self.assertIn("old-valid", output_ids["train"])
            self.assertNotIn("old-train", output_ids["valid"])
            self.assertNotIn("old-valid", output_ids["valid"])

            threshold = int(0.8 * roll.UINT64_SPACE)
            source_test_ids = {
                str(row["episode_id"]) for row in source_rows["test"]
            }
            for episode_id in source_test_ids:
                self.assertIn(
                    episode_id,
                    output_ids[
                        roll.assigned_split(episode_id, 71, threshold)
                    ],
                )

            with zipfile.ZipFile(first_output) as archive:
                self.assertFalse(
                    any(
                        name.startswith("test/")
                        for name in archive.namelist()
                    )
                )
                manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(
                manifest["split_policy"]["mode"],
                "roll_forward_episode_hash",
            )
            self.assertEqual(
                sum(manifest["split_decisions"].values()),
                source_total,
            )
            self.assertTrue(
                manifest["roll_forward"]["validation"]["rows_preserved"]
            )
            self.assertEqual(
                manifest["roll_forward"]["validation"][
                    "output_episode_id_overlap"
                ],
                0,
            )

    def test_input_cannot_be_the_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.zip"
            write_source_archive(source)
            before = file_sha256(source)
            with self.assertRaisesRegex(ValueError, "different"):
                roll.roll_archive(source, source, seed=71)
            self.assertEqual(before, file_sha256(source))

    def test_zero_train_fraction_promotes_old_valid_and_keeps_test_as_valid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.zip"
            output = root / "output.zip"
            source_rows = write_source_archive(source)

            result = roll.roll_archive(
                source,
                output,
                seed=71,
                train_fraction=0.0,
                rows_per_shard=3,
            )

            output_rows = read_output_rows(output)
            train_ids = {
                str(row["episode_id"]) for row in output_rows["train"]
            }
            valid_ids = {
                str(row["episode_id"]) for row in output_rows["valid"]
            }
            source_test_ids = {
                str(row["episode_id"]) for row in source_rows["test"]
            }
            self.assertEqual(train_ids, {"old-train", "old-valid"})
            self.assertEqual(valid_ids, source_test_ids)
            self.assertFalse(train_ids & valid_ids)
            self.assertEqual(result["episode_id_overlap"], 0)
            self.assertEqual(
                result["rows_total"],
                sum(map(len, source_rows.values())),
            )

            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(
                manifest["split_policy"]["train_fraction"],
                0.0,
            )

    def test_invalid_source_does_not_install_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "bad-source.zip"
            output = root / "output.zip"
            write_source_archive(source, mismatch=True)
            with self.assertRaisesRegex(ValueError, "row/member split mismatch"):
                roll.roll_archive(source, output, seed=71)
            self.assertFalse(output.exists())
            self.assertEqual(
                list(root.glob(f".{output.name}.*.partial")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
