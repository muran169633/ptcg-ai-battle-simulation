from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import orjson
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import calibrate_ppo_value_head as calibration  # noqa: E402
import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


class ValueHeadCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_config = {
            "hash_size": 128,
            "categorical_dim": 8,
            "model_dim": 16,
            "layers": 1,
            "heads": 4,
            "dropout": 0.0,
            "max_state_entities": 4,
            "entity_fields": bc.DEFAULT_ENTITY_FIELDS,
            "option_fields": bc.DEFAULT_OPTION_FIELDS,
        }
        self.data = self.root / "canonical.zip"
        self.bc_checkpoint = self.root / "fresh-bc.pt"
        self.ppo_checkpoint = self.root / "ppo.pt"
        self.write_archive(expected_train_rows=4)
        self.write_checkpoints()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def row(index: int) -> dict[str, object]:
        return {
            "split": "train",
            "episode_id": f"episode-{index}",
            "decision_index": index,
            "observation": {
                "yourPlayerIndex": 0,
                "current": {
                    "turnPlayerIndex": index % 2,
                    "turnCount": index + 1,
                    "players": [
                        {
                            "hand": [],
                            "bench": [],
                            "discard": [],
                            "deckCount": 20,
                            "prizeCount": 3,
                        },
                        {
                            "hand": [],
                            "bench": [],
                            "discard": [],
                            "deckCount": 18,
                            "prizeCount": 4,
                        },
                    ],
                },
                "select": {
                    "context": 0,
                    "minCount": 1,
                    "maxCount": 1,
                    "option": [
                        {
                            "type": 1,
                            "index": 0,
                            "playerIndex": 0,
                        },
                        {
                            "type": 2,
                            "index": 1,
                            "playerIndex": 1,
                        },
                    ],
                },
            },
            "action": [index % 2],
            "terminal_reward": float(index % 2),
            "sample_weight": 0.01,
            "deck_hash": calibration.CANONICAL_DECK_HASH,
        }

    def write_archive(self, *, expected_train_rows: int) -> None:
        rows = b"".join(
            orjson.dumps(self.row(index)) + b"\n"
            for index in range(4)
        )
        manifest = {
            "schema_version": "test-v1",
            "split_decisions": {
                "train": expected_train_rows,
                "valid": 1,
            },
            "split_policy": {"mode": "test"},
            "deck_hash_filter": calibration.CANONICAL_DECK_HASH,
        }
        with zipfile.ZipFile(
            self.data,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr("train/part-00000.jsonl", rows)
            # A malformed non-train member proves the calibration loader does
            # not inspect validation rows.
            archive.writestr("valid/part-00000.jsonl", b"not-json\n")
            archive.writestr("manifest.json", orjson.dumps(manifest))

    def write_checkpoints(self) -> None:
        torch.manual_seed(11)
        fresh_model = bc.EntityOptionPolicy(
            hash_size=self.model_config["hash_size"],
            categorical_dim=self.model_config["categorical_dim"],
            model_dim=self.model_config["model_dim"],
            layers=self.model_config["layers"],
            heads=self.model_config["heads"],
            dropout=self.model_config["dropout"],
            max_state_entities=self.model_config["max_state_entities"],
        )
        bc_payload = {
            "feature_version": bc.FEATURE_VERSION,
            "config": {
                **self.model_config,
                "seed": calibration.DEFAULT_SEED,
                "batch_size": calibration.BATCH_SIZE,
                "workers": calibration.WORKERS,
                "split_mode": "archive",
                "use_trajectory_weights": False,
                "max_train_rows": None,
                "expected_train_rows": 4,
                "deck_hashes": (calibration.CANONICAL_DECK_HASH,),
                "team_names": (),
                "train_shuffle_buffer_rows_per_worker": 0,
            },
            "model_state_dict": fresh_model.state_dict(),
            "epoch": calibration.CANONICAL_DONOR_EPOCH,
        }
        torch.save(bc_payload, self.bc_checkpoint)

        ppo_model = ppo.instantiate_model_from_bc(
            bc_payload,
            torch.device("cpu"),
        )
        with torch.no_grad():
            for parameter in ppo_model.value_head.parameters():
                parameter.add_(0.75)
            ppo_model.actor_query.weight.add_(0.01)
        ppo_payload = {
            "feature_version": ppo.PPO_FEATURE_VERSION,
            "bc_feature_version": bc.FEATURE_VERSION,
            "config": {"trainable_scope": "last_block_heads"},
            "model_config": dict(self.model_config),
            "model_state_dict": ppo_model.state_dict(),
            "optimizer_state_dict": {"sentinel": 1},
            "optimizer_parameter_names": {
                "actor": ["actor_query.weight"],
                "value": list(calibration.VALUE_HEAD_NAMES),
            },
            "bc_replay_optimizer_state_dict": {"sentinel": 2},
            "scheduler_state_dict": {"sentinel": 3},
            "update": 448,
            "metrics": {"test": True},
        }
        torch.save(ppo_payload, self.ppo_checkpoint)

    def request(
        self,
        output: Path,
        **overrides: object,
    ) -> calibration.CalibrationRequest:
        values: dict[str, object] = {
            "ppo_checkpoint": self.ppo_checkpoint,
            "ppo_checkpoint_sha256": calibration.file_sha256(
                self.ppo_checkpoint
            ),
            "canonical_bc_checkpoint": self.bc_checkpoint,
            "canonical_bc_checkpoint_sha256": calibration.file_sha256(
                self.bc_checkpoint
            ),
            "data": self.data,
            "data_sha256": calibration.file_sha256(self.data),
            "output": output,
            "seed": calibration.DEFAULT_SEED,
            "device": "cpu",
        }
        values.update(overrides)
        return calibration.CalibrationRequest(**values)

    @contextmanager
    def mini_contract(self):
        with mock.patch.multiple(
            calibration,
            CANONICAL_TRAIN_ROWS=4,
            CANONICAL_TRAIN_BATCHES=1,
            CANONICAL_POSITIVE_TARGETS=2,
            CANONICAL_NEGATIVE_TARGETS=2,
            CANONICAL_TRAIN_MEMBERS=1,
        ):
            yield

    def test_calibrates_only_value_head_and_strips_optimizer_state(self) -> None:
        output = self.root / "calibrated.pt"
        parent = torch.load(
            self.ppo_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        fresh = torch.load(
            self.bc_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        with self.mini_contract():
            result = calibration.run_calibration(self.request(output))

        self.assertEqual(result["rows"], 4)
        self.assertTrue(result["all_non_value_tensors_unchanged"])
        self.assertTrue(result["policy_logits_bitwise_equal"])
        self.assertTrue(result["count_logits_bitwise_equal"])
        self.assertEqual(
            result["optimizer_parameter_names"],
            list(calibration.VALUE_HEAD_NAMES),
        )
        self.assertEqual(
            result["stripped_parent_training_state_keys"],
            [
                "bc_replay_optimizer_state_dict",
                "optimizer_parameter_names",
                "optimizer_state_dict",
                "scheduler_state_dict",
            ],
        )

        calibrated = torch.load(
            output,
            map_location="cpu",
            weights_only=True,
        )
        for key in result["stripped_parent_training_state_keys"]:
            self.assertNotIn(key, calibrated)
        calibrated_state = calibrated["model_state_dict"]
        for name, tensor in parent["model_state_dict"].items():
            if name.startswith("value_head."):
                self.assertFalse(
                    torch.equal(tensor, calibrated_state[name]),
                    name,
                )
            else:
                self.assertTrue(
                    torch.equal(tensor, calibrated_state[name]),
                    name,
                )
        self.assertTrue(
            any(
                not torch.equal(
                    fresh["model_state_dict"][name],
                    calibrated_state[name],
                )
                for name in calibration.VALUE_HEAD_NAMES
            )
        )

        manifest_path = calibration.default_manifest_path(output.resolve())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["data"]["rows_consumed"], 4)
        self.assertEqual(
            manifest["inputs"]["canonical_bc_archive_sha256"],
            calibration.file_sha256(self.data),
        )
        self.assertTrue(manifest["all_non_value_tensors_unchanged"])
        self.assertEqual(
            len(manifest["non_value_tensor_audit"]),
            len(parent["model_state_dict"]) - 4,
        )
        self.assertTrue(
            all(
                item["unchanged"]
                and item["before_sha256"] == item["after_sha256"]
                for item in manifest["non_value_tensor_audit"]
            )
        )
        self.assertTrue(
            manifest["fixed_probe"]["policy_logits_bitwise_equal"]
        )
        self.assertTrue(
            manifest["fixed_probe"]["count_logits_bitwise_equal"]
        )
        self.assertFalse(
            manifest["output"]["optimizer_continuation_state_present"]
        )
        self.assertEqual(
            manifest["canonical_fresh_bc_donor_contract"]["donor_epoch"],
            7,
        )
        self.assertEqual(
            manifest["canonical_exposure_gate"],
            {
                "rows": 4,
                "batches": 1,
                "positive_targets": 2,
                "negative_targets": 2,
                "passed": True,
            },
        )

    def test_rejects_non_frozen_training_contract(self) -> None:
        output = self.root / "bad.pt"
        with self.assertRaisesRegex(ValueError, "exactly 1 epoch"):
            calibration.run_calibration(
                self.request(output, epochs=2)
            )
        with self.assertRaisesRegex(ValueError, "batch size 256"):
            calibration.run_calibration(
                self.request(output, batch_size=128)
            )
        with self.assertRaisesRegex(ValueError, "learning rate"):
            calibration.run_calibration(
                self.request(output, learning_rate=1e-4)
            )
        with self.assertRaisesRegex(ValueError, "requires seed 20260922"):
            calibration.run_calibration(
                self.request(output, seed=123)
            )
        self.assertFalse(output.exists())

    def test_rejects_hash_mismatch_before_training(self) -> None:
        output = self.root / "bad-sha.pt"
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            calibration.run_calibration(
                self.request(output, data_sha256="0" * 64)
            )
        self.assertFalse(output.exists())
        self.assertFalse(
            calibration.default_manifest_path(output.resolve()).exists()
        )

    def test_archive_row_contract_mismatch_publishes_nothing(self) -> None:
        self.write_archive(expected_train_rows=5)
        output = self.root / "row-mismatch.pt"
        with self.mini_contract():
            with self.assertRaisesRegex(ValueError, "train row gate failed"):
                calibration.run_calibration(self.request(output))
        self.assertFalse(output.exists())
        self.assertFalse(
            calibration.default_manifest_path(output.resolve()).exists()
        )

    def test_existing_output_is_never_overwritten(self) -> None:
        output = self.root / "occupied.pt"
        marker = b"user-owned"
        output.write_bytes(marker)
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            calibration.run_calibration(self.request(output))
        self.assertEqual(output.read_bytes(), marker)

    def test_eight_worker_loader_replays_the_same_global_order(self) -> None:
        with zipfile.ZipFile(
            self.data,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for member_index in range(16):
                rows = b"".join(
                    orjson.dumps(
                        self.row(member_index * 2 + row_index)
                    )
                    + b"\n"
                    for row_index in range(2)
                )
                archive.writestr(
                    f"train/part-{member_index:05d}.jsonl",
                    rows,
                )
            archive.writestr(
                "manifest.json",
                orjson.dumps({"split_decisions": {"train": 32}}),
            )

        def replay() -> tuple[int, list[str]]:
            loader = calibration.make_train_loader(
                data=self.data,
                model_config=self.model_config,
                seed=calibration.DEFAULT_SEED,
                epoch=calibration.TRAIN_EPOCH,
                seed_offset=calibration.TRAIN_SHUFFLE_SEED_OFFSET,
                device=torch.device("cpu"),
            )
            rows = 0
            hashes: list[str] = []
            for batch in loader:
                rows += int(batch["win_targets"].numel())
                hashes.append(calibration.batch_sha256(batch))
            return rows, hashes

        first = replay()
        second = replay()
        self.assertEqual(first[0], 32)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
