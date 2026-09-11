from __future__ import annotations

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import interpolate_bc_checkpoints as soup  # noqa: E402
import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


class InterpolateBcCheckpointTests(unittest.TestCase):
    PPO_INTERFACE = {
        "actor": "policy_logits over option_mask",
        "cardinality": "count_logits masked to minCount..maxCount",
        "critic": "sigmoid(value_logits) predicts terminal win probability",
    }

    @staticmethod
    def model(seed: int) -> bc.EntityOptionPolicy:
        torch.manual_seed(seed)
        return bc.EntityOptionPolicy(
            hash_size=128,
            categorical_dim=8,
            model_dim=16,
            layers=2,
            heads=4,
            dropout=0.0,
            max_state_entities=8,
        )

    @classmethod
    def config(cls) -> dict[str, object]:
        return {
            "data": "/not/copied/training.zip",
            "output_dir": "/not/copied/output",
            "epochs": 8,
            "learning_rate": 3e-4,
            "hash_size": 128,
            "categorical_dim": 8,
            "model_dim": 16,
            "layers": 2,
            "heads": 4,
            "dropout": 0.0,
            "max_state_entities": 8,
            "entity_fields": 20,
            "option_fields": 24,
            "deck_hashes": ("deck-sha256",),
        }

    @classmethod
    def checkpoint(
        cls,
        model: bc.EntityOptionPolicy,
        *,
        epoch: object = 7,
    ) -> dict[str, object]:
        return {
            "feature_version": bc.FEATURE_VERSION,
            "config": cls.config(),
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "valid_metrics": {"exact_action_set_accuracy": 0.8},
            "ppo_interface": copy.deepcopy(cls.PPO_INTERFACE),
            "optimizer_state_dict": {"must": "not be copied"},
        }

    @staticmethod
    def save(path: Path, checkpoint: dict[str, object]) -> None:
        torch.save(checkpoint, path)

    def test_success_metadata_runtime_compatibility_and_strict_load(self) -> None:
        model_a = self.model(1)
        model_b = self.model(2)
        alpha = 0.25
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.pt"
            source_b = root / "b.pt"
            output = root / "nested" / "soup.pt"
            self.save(source_a, self.checkpoint(model_a))
            self.save(source_b, self.checkpoint(model_b))

            summary = soup.create_interpolated_checkpoint(
                source_a,
                source_b,
                alpha,
                output,
            )
            checkpoint = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )

            self.assertEqual(summary["output"], str(output.resolve()))
            self.assertEqual(len(summary["output_sha256"]), 64)
            self.assertEqual(checkpoint["feature_version"], bc.FEATURE_VERSION)
            self.assertEqual(checkpoint["epoch"], 7)
            self.assertIsNone(checkpoint["valid_metrics"])
            self.assertEqual(
                checkpoint["ppo_interface"],
                self.PPO_INTERFACE,
            )
            self.assertEqual(
                set(checkpoint["config"]),
                set(soup.RUNTIME_MODEL_CONFIG_KEYS) | {"deck_hashes"},
            )
            self.assertNotIn("data", checkpoint["config"])
            self.assertNotIn("learning_rate", checkpoint["config"])
            self.assertNotIn("optimizer_state_dict", checkpoint)
            self.assertEqual(
                ppo.checkpoint_model_config(checkpoint),
                {
                    key: self.config()[key]
                    for key in soup.RUNTIME_MODEL_CONFIG_KEYS
                },
            )
            self.assertEqual(
                ppo.checkpoint_single_deck_hash(checkpoint),
                "deck-sha256",
            )

            metadata = checkpoint["interpolation"]
            self.assertEqual(metadata["formula"], soup.FORMULA)
            self.assertEqual(metadata["alpha"], alpha)
            self.assertEqual(metadata["checkpoint_a"], str(source_a.resolve()))
            self.assertEqual(metadata["checkpoint_b"], str(source_b.resolve()))
            self.assertEqual(len(metadata["checkpoint_a_sha256"]), 64)
            self.assertEqual(len(metadata["checkpoint_b_sha256"]), 64)
            self.assertEqual(metadata["checkpoint_a_epoch"], 7)
            self.assertEqual(metadata["checkpoint_b_epoch"], 7)
            self.assertFalse(metadata["epoch_mismatch_allowed"])
            self.assertEqual(
                metadata["output_epoch_policy"],
                "max_source_epoch",
            )
            self.assertIsInstance(metadata["created_at_utc"], str)
            self.assertTrue(metadata["optimizer_state_omitted"])
            self.assertFalse(metadata["resume_training"])
            self.assertEqual(
                metadata["config_provenance"]["strategy"],
                "validated_shared_runtime_fields_only",
            )

            loaded_model = self.model(3)
            loaded_model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            for name, tensor in checkpoint["model_state_dict"].items():
                expected = torch.lerp(
                    model_a.state_dict()[name],
                    model_b.state_dict()[name],
                    alpha,
                )
                self.assertTrue(torch.equal(tensor, expected), name)

    def test_formula_uses_a_at_zero_and_b_at_one(self) -> None:
        state_a = {"weight": torch.tensor([2.0, 10.0])}
        state_b = {"weight": torch.tensor([6.0, -2.0])}
        self.assertTrue(
            torch.equal(
                soup.interpolate_state_dict(state_a, state_b, 0.0)["weight"],
                state_a["weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                soup.interpolate_state_dict(state_a, state_b, 1.0)["weight"],
                state_b["weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                soup.interpolate_state_dict(state_a, state_b, 0.25)["weight"],
                torch.tensor([3.0, 7.0]),
            )
        )

    def test_refuses_to_overwrite_before_loading_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing.pt"
            output.write_bytes(b"preserve-me")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                soup.create_interpolated_checkpoint(
                    root / "missing-a.pt",
                    root / "missing-b.pt",
                    0.5,
                    output,
                )
            self.assertEqual(output.read_bytes(), b"preserve-me")

    def test_publish_race_preserves_competing_output(self) -> None:
        model_a = self.model(1)
        model_b = self.model(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.pt"
            source_b = root / "b.pt"
            output = root / "soup.pt"
            self.save(source_a, self.checkpoint(model_a))
            self.save(source_b, self.checkpoint(model_b))
            real_link = soup.os.link

            def competing_link(source, destination):
                Path(destination).write_bytes(b"competitor")
                return real_link(source, destination)

            with (
                mock.patch.object(
                    soup.os,
                    "link",
                    side_effect=competing_link,
                ),
                self.assertRaises(FileExistsError),
            ):
                soup.create_interpolated_checkpoint(
                    source_a,
                    source_b,
                    0.5,
                    output,
                )
            self.assertEqual(output.read_bytes(), b"competitor")

    def run_checkpoint_mismatch(
        self,
        mutate_a: object | None = None,
        mutate_b: object | None = None,
    ) -> None:
        model_a = self.model(1)
        model_b = self.model(2)
        checkpoint_a = self.checkpoint(model_a)
        checkpoint_b = self.checkpoint(model_b)
        if mutate_a is not None:
            mutate_a(checkpoint_a)
        if mutate_b is not None:
            mutate_b(checkpoint_b)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.pt"
            source_b = root / "b.pt"
            self.save(source_a, checkpoint_a)
            self.save(source_b, checkpoint_b)
            soup.create_interpolated_checkpoint(
                source_a,
                source_b,
                0.5,
                root / "output.pt",
            )

    def test_rejects_feature_version_mismatch(self) -> None:
        def mutate(checkpoint: dict[str, object]) -> None:
            checkpoint["feature_version"] = "wrong"

        with self.assertRaisesRegex(ValueError, "feature_version"):
            self.run_checkpoint_mismatch(mutate_b=mutate)

    def test_rejects_runtime_model_config_mismatch(self) -> None:
        def mutate(checkpoint: dict[str, object]) -> None:
            checkpoint["config"]["model_dim"] = 32

        with self.assertRaisesRegex(ValueError, "Runtime model config mismatch"):
            self.run_checkpoint_mismatch(mutate_b=mutate)

    def test_rejects_deck_hash_mismatch_or_non_singleton(self) -> None:
        def mismatch(checkpoint: dict[str, object]) -> None:
            checkpoint["config"]["deck_hashes"] = ("other-deck",)

        with self.assertRaisesRegex(ValueError, "deck hash mismatch"):
            self.run_checkpoint_mismatch(mutate_b=mismatch)

        for invalid in ((), ("a", "b"), ("",)):
            with self.subTest(invalid=invalid):
                def invalid_decks(
                    checkpoint: dict[str, object],
                    value: tuple[str, ...] = invalid,
                ) -> None:
                    checkpoint["config"]["deck_hashes"] = value

                with self.assertRaisesRegex(ValueError, "deck_hashes"):
                    self.run_checkpoint_mismatch(mutate_a=invalid_decks)

    def test_rejects_ppo_interface_mismatch(self) -> None:
        def mutate(checkpoint: dict[str, object]) -> None:
            checkpoint["ppo_interface"] = {"actor": "different"}

        with self.assertRaisesRegex(ValueError, "ppo_interface mismatch"):
            self.run_checkpoint_mismatch(mutate_b=mutate)

    def test_rejects_invalid_or_mismatched_epochs(self) -> None:
        for invalid in (True, 7.0, None):
            with self.subTest(invalid=invalid):
                def mutate(
                    checkpoint: dict[str, object],
                    value: object = invalid,
                ) -> None:
                    checkpoint["epoch"] = value

                with self.assertRaisesRegex(TypeError, "non-bool integers"):
                    self.run_checkpoint_mismatch(mutate_a=mutate)

        def mismatch(checkpoint: dict[str, object]) -> None:
            checkpoint["epoch"] = 8

        with self.assertRaisesRegex(ValueError, "epoch mismatch"):
            self.run_checkpoint_mismatch(mutate_b=mismatch)

    def test_explicitly_allows_epoch_mismatch_and_uses_max_epoch(self) -> None:
        model_a = self.model(1)
        model_b = self.model(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.pt"
            source_b = root / "b.pt"
            output = root / "soup.pt"
            self.save(source_a, self.checkpoint(model_a, epoch=9))
            self.save(source_b, self.checkpoint(model_b, epoch=3))

            summary = soup.create_interpolated_checkpoint(
                source_a,
                source_b,
                0.5,
                output,
                allow_epoch_mismatch=True,
            )
            checkpoint = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )

            self.assertEqual(checkpoint["epoch"], 9)
            self.assertEqual(summary["epoch"], 9)
            self.assertEqual(summary["checkpoint_a_epoch"], 9)
            self.assertEqual(summary["checkpoint_b_epoch"], 3)
            self.assertTrue(summary["epoch_mismatch_allowed"])
            self.assertEqual(
                summary["output_epoch_policy"],
                "max_source_epoch",
            )
            metadata = checkpoint["interpolation"]
            self.assertEqual(metadata["checkpoint_a_epoch"], 9)
            self.assertEqual(metadata["checkpoint_b_epoch"], 3)
            self.assertTrue(metadata["epoch_mismatch_allowed"])
            self.assertEqual(
                metadata["output_epoch_policy"],
                "max_source_epoch",
            )

    def test_cli_passes_allow_epoch_mismatch(self) -> None:
        argv = [
            "interpolate_bc_checkpoints.py",
            "--checkpoint-a",
            "a.pt",
            "--checkpoint-b",
            "b.pt",
            "--alpha",
            "0.25",
            "--output",
            "output.pt",
            "--allow-epoch-mismatch",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                soup,
                "create_interpolated_checkpoint",
                return_value={"output": "output.pt"},
            ) as create,
            mock.patch("builtins.print"),
        ):
            soup.main()

        create.assert_called_once_with(
            Path("a.pt"),
            Path("b.pt"),
            0.25,
            Path("output.pt"),
            allow_epoch_mismatch=True,
        )

    def test_rejects_state_key_shape_dtype_and_layout_mismatches(self) -> None:
        cases = (
            (
                {"a": torch.ones(2)},
                {"b": torch.ones(2)},
                "key mismatch",
            ),
            (
                {"a": torch.ones(2)},
                {"a": torch.ones(3)},
                "shape mismatch",
            ),
            (
                {"a": torch.ones(2, dtype=torch.float32)},
                {"a": torch.ones(2, dtype=torch.float64)},
                "dtype mismatch",
            ),
            (
                {"a": torch.ones((2, 2))},
                {"a": torch.ones((2, 2)).to_sparse()},
                "layout mismatch",
            ),
        )
        for state_a, state_b, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    soup.interpolate_state_dict(state_a, state_b, 0.5)

    def test_nonfloating_state_must_be_equal(self) -> None:
        state_a = {"counter": torch.tensor([1, 2], dtype=torch.int64)}
        state_b = {"counter": torch.tensor([1, 2], dtype=torch.int64)}
        result = soup.interpolate_state_dict(state_a, state_b, 0.5)
        self.assertTrue(torch.equal(result["counter"], state_a["counter"]))
        self.assertIsNot(result["counter"], state_a["counter"])

        with self.assertRaisesRegex(ValueError, "Non-floating tensor"):
            soup.interpolate_state_dict(
                state_a,
                {"counter": torch.tensor([1, 3], dtype=torch.int64)},
                0.5,
            )

    def test_rejects_nonfinite_float_state(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    soup.interpolate_state_dict(
                        {"weight": torch.tensor([invalid])},
                        {"weight": torch.tensor([0.0])},
                        0.5,
                    )
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    soup.interpolate_state_dict(
                        {"weight": torch.tensor([0.0])},
                        {"weight": torch.tensor([invalid])},
                        0.5,
                    )

    def test_rejects_invalid_alpha(self) -> None:
        for invalid in (
            -0.01,
            1.01,
            math.nan,
            math.inf,
            -math.inf,
            True,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and in"):
                    soup.validate_alpha(invalid)


if __name__ == "__main__":
    unittest.main()
