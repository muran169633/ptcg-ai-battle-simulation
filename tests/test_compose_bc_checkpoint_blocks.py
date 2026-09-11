from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import compose_bc_checkpoint_blocks as compose  # noqa: E402
import train_bc_orbit as bc  # noqa: E402


class ComposeBcCheckpointBlocksTests(unittest.TestCase):
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
    def checkpoint(cls, model: bc.EntityOptionPolicy, epoch: int) -> dict[str, object]:
        return {
            "feature_version": bc.FEATURE_VERSION,
            "config": {
                "hash_size": 128,
                "categorical_dim": 8,
                "model_dim": 16,
                "layers": 2,
                "heads": 4,
                "dropout": 0.0,
                "max_state_entities": 8,
                "entity_fields": 20,
                "option_fields": 24,
                "deck_hashes": ["deck-sha256"],
            },
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "epoch": epoch,
            "valid_metrics": {"must": "not survive"},
            "ppo_interface": copy.deepcopy(cls.PPO_INTERFACE),
            "optimizer_state_dict": {"must": "not survive"},
        }

    def test_transformer_actor_is_bitwise_selected_and_complement_is_source(self) -> None:
        source = self.model(1)
        overlay = self.model(2)
        source_state = source.state_dict()
        overlay_state = overlay.state_dict()
        expected_keys = {
            name for name in source_state if compose.selected_by_transformer_actor(name)
        }
        expected_parameters = sum(source_state[name].numel() for name in expected_keys)

        state, audit = compose.compose_transformer_actor_state(
            source_state,
            overlay_state,
            expected_selected_tensors=len(expected_keys),
            expected_selected_parameters=expected_parameters,
        )
        self.assertEqual(set(audit["selected_keys"]), expected_keys)
        self.assertEqual(set(audit["actor_keys"]), compose.ACTOR_KEYS)
        for name, value in state.items():
            expected = overlay_state[name] if name in expected_keys else source_state[name]
            self.assertTrue(torch.equal(value, expected), name)
            self.assertNotEqual(value.data_ptr(), expected.data_ptr(), name)

    def test_checkpoint_is_deployment_only_and_refuses_overwrite(self) -> None:
        source = self.model(3)
        overlay = self.model(4)
        selected = {
            name for name in source.state_dict() if compose.selected_by_transformer_actor(name)
        }
        selected_parameters = sum(source.state_dict()[name].numel() for name in selected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_a = root / "a.pt"
            path_b = root / "b.pt"
            output = root / "nested" / "composed.pt"
            torch.save(self.checkpoint(source, 12), path_a)
            torch.save(self.checkpoint(overlay, 1), path_b)
            result = compose.create_composed_checkpoint(
                path_a,
                path_b,
                output,
                allow_epoch_mismatch=True,
                expected_selected_tensors=len(selected),
                expected_selected_parameters=selected_parameters,
            )
            saved = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(result["epoch"], 12)
            self.assertIsNone(saved["valid_metrics"])
            self.assertNotIn("optimizer_state_dict", saved)
            self.assertFalse(saved["block_composition"]["resume_training"])
            self.assertTrue(saved["block_composition"]["deployment_only"])
            with self.assertRaises(FileExistsError):
                compose.create_composed_checkpoint(
                    path_a,
                    path_b,
                    output,
                    allow_epoch_mismatch=True,
                )

    def test_expected_selected_count_is_a_hard_gate(self) -> None:
        source = self.model(5).state_dict()
        overlay = self.model(6).state_dict()
        with self.assertRaisesRegex(ValueError, "expected_selected_tensors mismatch"):
            compose.compose_transformer_actor_state(
                source,
                overlay,
                expected_selected_tensors=1,
            )


if __name__ == "__main__":
    unittest.main()
