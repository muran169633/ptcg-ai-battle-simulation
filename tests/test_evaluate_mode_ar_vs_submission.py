from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate_mode_ar_vs_submission as adapter  # noqa: E402
import train_bc_mode_ar_v7 as trainer  # noqa: E402
import train_bc_orbit as base  # noqa: E402


def synthetic_batch() -> dict[str, torch.Tensor]:
    batch_size, options, fields, entities = 3, 4, 3, 2
    sequences = torch.full(
        (batch_size, base.MAX_ACTION_COUNT), -1, dtype=torch.long
    )
    return {
        "global_fields": torch.randint(1, 63, (batch_size, fields)),
        "global_field_mask": torch.ones(batch_size, fields, dtype=torch.bool),
        "global_numeric": torch.randn(batch_size, base.GLOBAL_NUMERIC_SIZE),
        "state_fields": torch.randint(1, 63, (batch_size, entities, fields)),
        "state_field_mask": torch.ones(
            batch_size, entities, fields, dtype=torch.bool
        ),
        "state_numeric": torch.randn(
            batch_size, entities, base.ENTITY_NUMERIC_SIZE
        ),
        "state_mask": torch.ones(batch_size, entities, dtype=torch.bool),
        "option_fields": torch.randint(
            1, 63, (batch_size, options, fields)
        ),
        "option_field_mask": torch.ones(
            batch_size, options, fields, dtype=torch.bool
        ),
        "option_numeric": torch.randn(
            batch_size, options, base.OPTION_NUMERIC_SIZE
        ),
        "option_mask": torch.ones(batch_size, options, dtype=torch.bool),
        "targets": torch.zeros(batch_size, options),
        "action_counts": torch.zeros(batch_size, dtype=torch.long),
        "action_sequences": sequences,
        "min_counts": torch.tensor([1, 0, 2]),
        "max_counts": torch.tensor([1, 1, 3]),
        "contexts": torch.tensor([0, 0, trainer.ORDERED_CONTEXT]),
        "sample_weights": torch.ones(batch_size),
        "win_targets": torch.zeros(batch_size),
    }


def tiny_model() -> trainer.ModeAwareARPolicy:
    return trainer.ModeAwareARPolicy(
        hash_size=64,
        categorical_dim=16,
        model_dim=32,
        layers=1,
        heads=4,
        dropout=0.0,
        max_state_entities=2,
        max_options=4,
    )


def checkpoint_config(*, epochs: int = 2) -> dict:
    return {
        "hash_size": 64,
        "categorical_dim": 16,
        "model_dim": 32,
        "layers": 1,
        "heads": 4,
        "dropout": 0.0,
        "max_state_entities": 2,
        "entity_fields": 3,
        "option_fields": 3,
        "epochs": epochs,
        "dates": ("2026-08-12", "2026-08-13"),
        "daily_equal_weighting": True,
    }


class ModeAwareSubmissionAdapterTests(unittest.TestCase):
    def test_load_decode_and_dispatch_restore(self) -> None:
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            torch.save(
                {
                    "feature_version": trainer.FEATURE_VERSION,
                    "config": checkpoint_config(),
                    "model_state_dict": tiny_model().state_dict(),
                    "epoch": 1,
                    "valid_metrics": {"daily_macro_choice_accuracy": 0.7},
                },
                path,
            )
            _, model, config = adapter.load_mode_ar_checkpoint(
                path, torch.device("cpu")
            )
        self.assertEqual(config["model_dim"], 32)
        batch = synthetic_batch()
        marker = adapter.mode_ar_forward(model, batch, torch.device("cpu"))
        self.assertTrue(marker[adapter.MODE_AR_MARKER])
        for action, minimum, maximum in zip(
            marker["actions"], batch["min_counts"], batch["max_counts"]
        ):
            self.assertGreaterEqual(len(action), int(minimum))
            self.assertLessEqual(len(action), int(maximum))
            self.assertEqual(len(action), len(set(action)))

        original_forward = adapter.legacy.model_forward
        original_sample = adapter.legacy.sample_ordered_actions
        with adapter.install_mode_ar_inference_adapter():
            outputs = adapter.legacy.model_forward(
                model, batch, torch.device("cpu")
            )
            actions, log_prob, entropy, value = adapter.legacy.sample_ordered_actions(
                outputs,
                batch,
                deterministic=True,
                canonicalize_order=False,
            )
            self.assertEqual(actions, marker["actions"])
            self.assertEqual(log_prob.shape, (3,))
            self.assertEqual(entropy.shape, (3,))
            self.assertEqual(value.shape, (3,))
        self.assertIs(adapter.legacy.model_forward, original_forward)
        self.assertIs(adapter.legacy.sample_ordered_actions, original_sample)

    def test_completed_training_audit_fails_closed(self) -> None:
        checkpoint = {
            "config": checkpoint_config(),
            "epoch": 2,
            "valid_metrics": {"daily_macro_choice_accuracy": 0.72},
        }
        with tempfile.TemporaryDirectory() as temporary:
            metrics = Path(temporary) / "metrics.jsonl"
            metrics.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "epoch": 1,
                            "valid": {"daily_macro_choice_accuracy": 0.7},
                        },
                        {
                            "epoch": 2,
                            "valid": {"daily_macro_choice_accuracy": 0.72},
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            audit = adapter.audit_completed_training(checkpoint, metrics)
            self.assertEqual(audit["completed_epochs"], 2)
            self.assertEqual(audit["best_epoch"], 2)
            metrics.write_text(
                json.dumps(
                    {
                        "epoch": 1,
                        "valid": {"daily_macro_choice_accuracy": 0.7},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "completed training epochs"):
                adapter.audit_completed_training(checkpoint, metrics)


if __name__ == "__main__":
    unittest.main()
