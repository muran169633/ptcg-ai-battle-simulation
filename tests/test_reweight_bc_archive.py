from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

import orjson

from tools.reweight_bc_archive import rewrite_archive


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(
    *,
    split: str,
    step: int,
    context: int,
    action: list[int],
    options: list[dict],
    minimum: int = 1,
    maximum: int = 1,
    date: str = "2026-08-06",
) -> dict:
    return {
        "schema_version": "ptcg-bc-visible-decisions-v1",
        "split": split,
        "dataset_date": date,
        "episode_id": f"episode-{split}-{step}",
        "team_name": "AlphaStarmie",
        "seat": step % 2,
        "action_step_index": step,
        "action": action,
        "select_context": str(context),
        "min_count": minimum,
        "max_count": maximum,
        "sample_weight": 0.25,
        "observation": {
            "select": {
                "context": context,
                "minCount": minimum,
                "maxCount": maximum,
                "option": options,
            }
        },
    }


def write_source(path: Path, *, invalid_action: bool = False) -> None:
    primary_action = [5] if invalid_action else [1]
    rows = {
        "train/part-00000.jsonl": [
            # Action values are option-array positions, not option["index"].
            row(
                split="train",
                step=1,
                context=0,
                action=primary_action,
                options=[
                    {"index": 1, "type": 2},
                    {"index": 99, "type": 7},
                ],
            ),
            row(
                split="train",
                step=2,
                context=7,
                action=[0],
                options=[{"type": 4}],
                minimum=0,
                maximum=2,
            ),
            row(
                split="train",
                step=3,
                context=0,
                action=[0],
                options=[{"type": 2}],
            ),
        ],
        "valid/part-00000.jsonl": [
            row(
                split="valid",
                step=4,
                context=0,
                action=[0],
                options=[{"type": 7}],
                date="2026-08-07",
            )
        ],
        "test/part-00000.jsonl": [
            row(
                split="test",
                step=5,
                context=0,
                action=[0],
                options=[{"type": 7}],
                date="2026-08-08",
            )
        ],
    }
    manifest = {
        "schema_version": "ptcg-bc-visible-decisions-v1",
        "split_decisions": {"train": 3, "valid": 1, "test": 1},
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, values in rows.items():
            payload = b"".join(orjson.dumps(value) + b"\n" for value in values)
            archive.writestr(name, payload)
        archive.writestr("manifest.json", orjson.dumps(manifest))


class ReweightArchiveTest(unittest.TestCase):
    def run_rewrite(self, source: Path, output: Path, **overrides):
        kwargs = {
            "input_path": source,
            "output_path": output,
            "expected_input_sha256": sha256(source),
            "target_date": "2026-08-06",
            "primary_context": 0,
            "primary_option_types": frozenset({7, 8, 13}),
            "secondary_context": 7,
            "base_weight": 2 / 3,
            "secondary_weight": 5 / 6,
            "primary_weight": 1.0,
            "expected_train_rows": 3,
            "expected_valid_rows": 1,
            "expected_test_rows": 1,
            "expected_primary_rows": 1,
            "expected_secondary_rows": 1,
        }
        kwargs.update(overrides)
        return rewrite_archive(**kwargs)

    def test_only_train_weight_changes_and_valid_test_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            output = root / "output.zip"
            write_source(source)
            result = self.run_rewrite(source, output)
            self.assertEqual(result["counts"]["train_primary_rows"], 1)
            self.assertEqual(result["counts"]["train_secondary_rows"], 1)
            with zipfile.ZipFile(source) as before, zipfile.ZipFile(output) as after:
                self.assertEqual(
                    before.read("valid/part-00000.jsonl"),
                    after.read("valid/part-00000.jsonl"),
                )
                self.assertEqual(
                    before.read("test/part-00000.jsonl"),
                    after.read("test/part-00000.jsonl"),
                )
                rewritten = [
                    orjson.loads(line)
                    for line in after.read("train/part-00000.jsonl").splitlines()
                ]
                self.assertEqual(
                    [value["sample_weight"] for value in rewritten],
                    [1.0, 5 / 6, 2 / 3],
                )
                manifest = orjson.loads(after.read("manifest.json"))
                self.assertTrue(
                    manifest["sample_reweighting"][
                        "valid_test_member_bytes_preserved"
                    ]
                )

    def test_output_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            first = root / "first.zip"
            second = root / "second.zip"
            write_source(source)
            self.run_rewrite(source, first)
            self.run_rewrite(source, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_invalid_action_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            output = root / "output.zip"
            write_source(source, invalid_action=True)
            with self.assertRaisesRegex(ValueError, "outside the option array"):
                self.run_rewrite(source, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".zip.partial").exists())

    def test_sha_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            output = root / "output.zip"
            write_source(source)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.run_rewrite(
                    source,
                    output,
                    expected_input_sha256="0" * 64,
                )
            self.assertFalse(output.exists())

    def test_out_of_range_weight_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            output = root / "output.zip"
            write_source(source)
            with self.assertRaisesRegex(ValueError, "in \(0, 1\]"):
                self.run_rewrite(source, output, primary_weight=1.5)
            self.assertFalse(output.exists())

    def test_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            output = root / "output.zip"
            write_source(source)
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                self.run_rewrite(source, output)
            self.assertEqual(output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
