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

import build_weighted_bc_mix as mixer  # noqa: E402


DECK_HASH = "d" * 64
OLD_SCHEMA = "synthetic-old-visible-v1"
YANZ_SCHEMA = OLD_SCHEMA
OLD_MANIFEST_SCHEMA = OLD_SCHEMA
YANZ_MANIFEST_SCHEMA = "synthetic-yanz-policy-manifest-v1"


def _row(
    schema: str,
    split: str,
    episode_id: str,
    action_step: int,
    marker: str,
    *,
    dataset_date: str,
) -> dict[str, object]:
    return {
        "schema_version": schema,
        "dataset_date": dataset_date,
        "episode_id": episode_id,
        "episode_uuid": f"uuid-{episode_id}",
        "split": split,
        "deck_hash": DECK_HASH,
        "seat": 0,
        "observation_step_index": action_step - 1,
        "action_step_index": action_step,
        "sample_weight": 0.75,
        "terminal_reward": 1.0,
        "team_name": "source-team",
        "opponent_team_name": "opponent-team",
        "select_context": "0",
        "observation": {"marker": marker, "nested": {"value": action_step}},
        "action": [action_step],
        "source_only_marker": marker,
    }


def _write_source(
    path: Path,
    *,
    role: str,
    schema: str,
    rows: dict[str, list[dict[str, object]]],
) -> None:
    split_rows = {split: len(values) for split, values in rows.items()}
    split_episodes = {
        split: len({str(value["episode_id"]) for value in values})
        for split, values in rows.items()
    }
    manifest: dict[str, object] = {
        "schema_version": schema,
        "competition": "pokemon-tcg-ai-battle",
        "split_decisions": split_rows,
        "split_episodes": split_episodes,
        "shards": {split: 1 for split in rows},
    }
    if role == "old":
        manifest["profile"] = {"deck_hash": DECK_HASH}
    else:
        manifest["policy"] = {"deck_hash": DECK_HASH}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for split, split_values in rows.items():
            payload = b"".join(
                json.dumps(value, separators=(",", ":"), sort_keys=False).encode()
                + b"\n"
                for value in split_values
            )
            archive.writestr(f"{split}/part-00000.jsonl", payload)
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_rows() -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, list[dict[str, object]]],
]:
    old = {
        "train": [
            _row(
                OLD_SCHEMA,
                "train",
                "old-1",
                1,
                "old-train-1",
                dataset_date="2026-08-01",
            ),
            _row(
                OLD_SCHEMA,
                "train",
                "old-1",
                2,
                "old-train-2",
                dataset_date="2026-08-01",
            ),
            _row(
                OLD_SCHEMA,
                "train",
                "old-2",
                1,
                "old-train-3",
                dataset_date="2026-08-02",
            ),
        ],
        "valid": [
            _row(
                OLD_SCHEMA,
                "valid",
                "old-valid",
                1,
                "must-be-excluded-valid",
                dataset_date="2026-08-06",
            )
        ],
        "test": [
            _row(
                OLD_SCHEMA,
                "test",
                "old-test",
                1,
                "must-be-excluded-test",
                dataset_date="2026-08-07",
            )
        ],
    }
    yanz = {
        "train": [
            _row(
                YANZ_SCHEMA,
                "train",
                "yanz-1",
                1,
                "yanz-train-1",
                dataset_date="unknown",
            ),
            _row(
                YANZ_SCHEMA,
                "train",
                "yanz-1",
                2,
                "yanz-train-2",
                dataset_date="unknown",
            ),
            _row(
                YANZ_SCHEMA,
                "train",
                "yanz-2",
                1,
                "yanz-train-3",
                dataset_date="unknown",
            ),
        ],
        "valid": [
            _row(
                YANZ_SCHEMA,
                "valid",
                "yanz-valid",
                1,
                "yanz-valid-1",
                dataset_date="unknown",
            )
        ],
    }
    return old, yanz


def _contract(
    old_archive: Path,
    yanz_archive: Path,
    old_rows: dict[str, list[dict[str, object]]],
    yanz_rows: dict[str, list[dict[str, object]]],
) -> mixer.BuildContract:
    return mixer.BuildContract(
        deck_hash=DECK_HASH,
        old_archive_sha256=_sha256(old_archive),
        yanz_archive_sha256=_sha256(yanz_archive),
        old_manifest_schema=OLD_MANIFEST_SCHEMA,
        yanz_manifest_schema=YANZ_MANIFEST_SCHEMA,
        old_row_schema=OLD_SCHEMA,
        yanz_row_schema=YANZ_SCHEMA,
        old_split_rows={split: len(rows) for split, rows in old_rows.items()},
        yanz_split_rows={split: len(rows) for split, rows in yanz_rows.items()},
        old_split_episodes={
            split: len({str(row["episode_id"]) for row in rows})
            for split, rows in old_rows.items()
        },
        yanz_split_episodes={
            split: len({str(row["episode_id"]) for row in rows})
            for split, rows in yanz_rows.items()
        },
    )


def _read_output_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".jsonl"):
                rows.extend(
                    json.loads(line) for line in archive.read(name).splitlines()
                )
    return rows


class WeightedMixTests(unittest.TestCase):
    def test_happy_path_is_exact_weighted_view_and_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_archive = root / "old.zip"
            yanz_archive = root / "yanz.zip"
            output_a = root / "output-a.zip"
            output_b = root / "output-b.zip"
            old_rows, yanz_rows = _source_rows()
            _write_source(
                old_archive,
                role="old",
                schema=OLD_MANIFEST_SCHEMA,
                rows=old_rows,
            )
            _write_source(
                yanz_archive,
                role="yanz",
                schema=YANZ_MANIFEST_SCHEMA,
                rows=yanz_rows,
            )
            contract = _contract(
                old_archive, yanz_archive, old_rows, yanz_rows
            )

            result_a = mixer.build_weighted_mix(
                old_archive,
                yanz_archive,
                output_a,
                contract=contract,
                rows_per_shard=2,
            )
            result_b = mixer.build_weighted_mix(
                old_archive,
                yanz_archive,
                output_b,
                contract=contract,
                rows_per_shard=2,
            )

            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
            self.assertEqual(result_a["output_sha256"], result_b["output_sha256"])
            self.assertEqual(
                result_a["split_decisions"],
                {"train": 6, "valid": 1, "test": 0},
            )
            self.assertEqual(
                result_a["verification"]["decision_key_duplicate_count"], 0
            )
            self.assertEqual(
                result_a["verification"]["old_yanz_episode_overlap_count"], 0
            )

            output_rows = _read_output_rows(output_a)
            markers = {str(row["source_only_marker"]): row for row in output_rows}
            self.assertNotIn("must-be-excluded-valid", markers)
            self.assertNotIn("must-be-excluded-test", markers)
            self.assertEqual(len(markers), 7)
            for marker, row in markers.items():
                if marker.startswith("old-train"):
                    self.assertEqual(row["split"], "train")
                    self.assertEqual(row["sample_weight"], 0.75 * 0.25)
                    source = next(
                        value
                        for value in old_rows["train"]
                        if value["source_only_marker"] == marker
                    )
                elif marker.startswith("yanz-train"):
                    self.assertEqual(row["split"], "train")
                    self.assertEqual(row["sample_weight"], 0.75)
                    source = next(
                        value
                        for value in yanz_rows["train"]
                        if value["source_only_marker"] == marker
                    )
                else:
                    self.assertEqual(row["split"], "valid")
                    self.assertEqual(row["sample_weight"], 0.75)
                    source = yanz_rows["valid"][0]
                preserved_output = {
                    key: value
                    for key, value in row.items()
                    if key not in ("split", "sample_weight")
                }
                preserved_source = {
                    key: value
                    for key, value in source.items()
                    if key not in ("split", "sample_weight")
                }
                self.assertEqual(preserved_output, preserved_source)

            with zipfile.ZipFile(output_a) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "train/part-00000.jsonl",
                        "train/part-00001.jsonl",
                        "train/part-00002.jsonl",
                        "valid/part-00000.jsonl",
                        "manifest.json",
                    ],
                )
                self.assertIsNone(archive.testzip())
                for info in archive.infolist():
                    self.assertEqual(info.date_time, mixer.ZIP_TIMESTAMP)
                    self.assertEqual(info.create_system, 3)
                    self.assertEqual(info.external_attr, mixer.ZIP_EXTERNAL_ATTR)
                    self.assertEqual(info.extra, b"")
                    self.assertEqual(info.comment, b"")
                manifest = json.loads(archive.read("manifest.json"))
            self.assertFalse(manifest["row_schema_contract"]["mixed_row_schemas"])
            self.assertTrue(
                manifest["source_manifest_schema_contract"][
                    "mixed_source_manifest_schemas"
                ]
            )
            self.assertEqual(
                manifest["row_schema_contract"]["mutated_fields_only"],
                ["split", "sample_weight"],
            )
            self.assertEqual(
                manifest["training_semantics"]["required_train_bc_orbit_flags"],
                [
                    "--use-trajectory-weights",
                    "--trajectory-weight-scope",
                    "policy_only",
                ],
            )
            self.assertEqual(
                manifest["training_semantics"]["value_loss_weighting"],
                {
                    "mode": "uniform",
                    "per_row_weight": 1.0,
                    "archive_sample_weight_ignored": True,
                },
            )
            self.assertIn("value loss uniformly weighted", manifest["purpose"])
            self.assertIn(
                "uniform per-row weight 1.0",
                manifest["training_semantics"]["validation"]["value_loss"],
            )
            self.assertEqual(
                manifest["provenance"]["supersedes"],
                {
                    "artifact": "old025_yanz1.zip",
                    "sha256": (
                        "89067ab5fa2a868e805931c34788dceb7e98d52dc299c7b5de5a63dcfddbb8d2"
                    ),
                    "status": "rejected_metadata_drift",
                    "reason": (
                        "predecessor manifest incorrectly declared "
                        "trajectory_weight_scope=all_losses instead of the "
                        "preregistered policy_only scope"
                    ),
                },
            )
            self.assertEqual(
                manifest["split_policy"]["excluded"],
                ["old_gold8_alakazam.valid", "old_gold8_alakazam.test"],
            )
            self.assertEqual(
                manifest["weight_audit"]["old_gold8_alakazam.train"],
                {
                    "rows": 3,
                    "factor": 0.25,
                    "original_weight_sum": 2.25,
                    "output_weight_sum": 0.5625,
                    "original_weight_min": 0.75,
                    "original_weight_max": 0.75,
                    "output_weight_min": 0.1875,
                    "output_weight_max": 0.1875,
                    "original_weight_histogram_float_hex": {
                        float(0.75).hex(): 3
                    },
                    "output_weight_histogram_float_hex": {
                        float(0.1875).hex(): 3
                    },
                    "output_weight_sha256_in_row_order": mixer._weight_digest(
                        [0.1875, 0.1875, 0.1875]
                    ),
                },
            )

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_archive = root / "old.zip"
            yanz_archive = root / "yanz.zip"
            output = root / "output.zip"
            old_rows, yanz_rows = _source_rows()
            _write_source(
                old_archive,
                role="old",
                schema=OLD_MANIFEST_SCHEMA,
                rows=old_rows,
            )
            _write_source(
                yanz_archive,
                role="yanz",
                schema=YANZ_MANIFEST_SCHEMA,
                rows=yanz_rows,
            )
            contract = _contract(
                old_archive, yanz_archive, old_rows, yanz_rows
            )
            output.write_bytes(b"keep-this")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                mixer.build_weighted_mix(
                    old_archive, yanz_archive, output, contract=contract
                )

            self.assertEqual(output.read_bytes(), b"keep-this")
            self.assertFalse(output.with_suffix(".zip.partial").exists())

    def test_source_sha_mismatch_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_archive = root / "old.zip"
            yanz_archive = root / "yanz.zip"
            output = root / "output.zip"
            old_rows, yanz_rows = _source_rows()
            _write_source(
                old_archive,
                role="old",
                schema=OLD_MANIFEST_SCHEMA,
                rows=old_rows,
            )
            _write_source(
                yanz_archive,
                role="yanz",
                schema=YANZ_MANIFEST_SCHEMA,
                rows=yanz_rows,
            )
            good = _contract(old_archive, yanz_archive, old_rows, yanz_rows)
            bad = mixer.BuildContract(
                **{
                    **good.__dict__,
                    "old_archive_sha256": "0" * 64,
                }
            )

            with self.assertRaisesRegex(mixer.ContractError, "SHA256 mismatch"):
                mixer.build_weighted_mix(
                    old_archive, yanz_archive, output, contract=bad
                )

            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".zip.partial").exists())

    def test_old_yanz_episode_overlap_is_rejected_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_archive = root / "old.zip"
            yanz_archive = root / "yanz.zip"
            output = root / "output.zip"
            old_rows, yanz_rows = _source_rows()
            yanz_rows["train"][0]["episode_id"] = "old-1"
            yanz_rows["train"][0]["episode_uuid"] = "different-uuid"
            _write_source(
                old_archive,
                role="old",
                schema=OLD_MANIFEST_SCHEMA,
                rows=old_rows,
            )
            _write_source(
                yanz_archive,
                role="yanz",
                schema=YANZ_MANIFEST_SCHEMA,
                rows=yanz_rows,
            )
            contract = _contract(
                old_archive, yanz_archive, old_rows, yanz_rows
            )

            with self.assertRaisesRegex(mixer.ContractError, "episode overlap"):
                mixer.build_weighted_mix(
                    old_archive, yanz_archive, output, contract=contract
                )

            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".zip.partial").exists())

    def test_duplicate_decision_key_is_rejected_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_archive = root / "old.zip"
            yanz_archive = root / "yanz.zip"
            output = root / "output.zip"
            old_rows, yanz_rows = _source_rows()
            duplicate = dict(yanz_rows["train"][0])
            duplicate["source_only_marker"] = "duplicate-physical-row"
            yanz_rows["train"].append(duplicate)
            _write_source(
                old_archive,
                role="old",
                schema=OLD_MANIFEST_SCHEMA,
                rows=old_rows,
            )
            _write_source(
                yanz_archive,
                role="yanz",
                schema=YANZ_MANIFEST_SCHEMA,
                rows=yanz_rows,
            )
            contract = _contract(
                old_archive, yanz_archive, old_rows, yanz_rows
            )

            with self.assertRaisesRegex(mixer.ContractError, "duplicate decision key"):
                mixer.build_weighted_mix(
                    old_archive, yanz_archive, output, contract=contract
                )

            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".zip.partial").exists())


if __name__ == "__main__":
    unittest.main()
