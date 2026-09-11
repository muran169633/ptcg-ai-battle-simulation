from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import train_bc_mode_ar_v7 as trainer  # noqa: E402
import train_bc_orbit as base  # noqa: E402


def synthetic_batch() -> dict[str, torch.Tensor]:
    batch_size = 4
    options = 4
    fields = 3
    entities = 2
    targets = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    sequences = torch.full(
        (batch_size, base.MAX_ACTION_COUNT), -1, dtype=torch.long
    )
    sequences[0, 0] = 2
    sequences[2, :2] = torch.tensor([2, 0])
    sequences[3, :2] = torch.tensor([2, 0])
    return {
        "global_fields": torch.randint(1, 63, (batch_size, fields)),
        "global_field_mask": torch.ones(batch_size, fields, dtype=torch.bool),
        "global_numeric": torch.randn(batch_size, base.GLOBAL_NUMERIC_SIZE),
        "state_fields": torch.randint(
            1, 63, (batch_size, entities, fields)
        ),
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
        "targets": targets,
        "action_counts": torch.tensor([1, 0, 2, 2]),
        "action_sequences": sequences,
        "min_counts": torch.tensor([1, 0, 2, 2]),
        "max_counts": torch.tensor([1, 1, 3, 3]),
        "contexts": torch.tensor([0, 0, trainer.ORDERED_CONTEXT, 0]),
        "sample_weights": torch.ones(batch_size),
        "win_targets": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "dataset_date_ids": torch.tensor([0, 0, 1, 1]),
    }


class ModeAwareTrainerTests(unittest.TestCase):
    def test_mode_routing_and_sequence_canonicalization(self) -> None:
        batch = synthetic_batch()
        modes = trainer.action_mode_ids(
            batch["contexts"], batch["min_counts"], batch["max_counts"]
        )
        self.assertEqual(
            modes.tolist(),
            [
                trainer.MODE_SINGLE,
                trainer.MODE_OPTIONAL_SINGLE,
                trainer.MODE_ORDERED_SEQUENCE,
                trainer.MODE_UNORDERED_SET,
            ],
        )
        sequences = trainer.canonical_action_sequences(batch, modes)
        self.assertEqual(sequences[2, :2].tolist(), [2, 0])
        self.assertEqual(sequences[3, :2].tolist(), [0, 2])
        trainer.validate_action_targets(batch, modes, sequences)

    def test_unordered_mask_enforces_canonical_reachability(self) -> None:
        batch = synthetic_batch()
        modes = trainer.action_mode_ids(
            batch["contexts"], batch["min_counts"], batch["max_counts"]
        )
        selected = torch.zeros_like(batch["option_mask"])
        previous = torch.full((4,), -1, dtype=torch.long)
        option_mask, stop_mask = trainer.legal_decode_mask(
            batch, modes, selected, 0, previous
        )
        self.assertFalse(bool(option_mask[3, 3]))
        self.assertFalse(bool(stop_mask[3]))
        selected[3, 0] = True
        previous[3] = 0
        option_mask, _ = trainer.legal_decode_mask(
            batch, modes, selected, 1, previous
        )
        self.assertFalse(bool(option_mask[3, 0]))
        self.assertTrue(bool(option_mask[3, 2]))

    def test_model_loss_backward_and_greedy_decode_contract(self) -> None:
        torch.manual_seed(7)
        batch = synthetic_batch()
        model = trainer.ModeAwareARPolicy(
            hash_size=64,
            categorical_dim=16,
            model_dim=32,
            layers=1,
            heads=4,
            dropout=0.0,
            max_state_entities=2,
            max_options=4,
        )
        encoded = model(batch)
        loss, components = trainer.autoregressive_losses(
            model,
            encoded,
            batch,
            count_loss_weight=0.05,
            value_loss_weight=0.05,
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(
            set(components),
            {"autoregressive", "cardinality_aux", "detached_value_aux"},
        )
        loss.backward()
        self.assertIsNotNone(model.decoder.weight_hh.grad)
        decoded = trainer.greedy_decode(model, encoded, batch)
        self.assertTrue(
            bool((decoded["counts"] >= batch["min_counts"]).all())
        )
        self.assertTrue(
            bool((decoded["counts"] <= batch["max_counts"]).all())
        )
        for row in range(batch["targets"].shape[0]):
            values = decoded["sequences"][row]
            values = values[values >= 0]
            self.assertEqual(values.numel(), torch.unique(values).numel())

    def test_ordered_semantic_exact_is_order_sensitive(self) -> None:
        batch = synthetic_batch()
        modes = trainer.action_mode_ids(
            batch["contexts"], batch["min_counts"], batch["max_counts"]
        )
        selected = batch["targets"].bool().clone()
        predicted = torch.full((4, 3), -1, dtype=torch.long)
        predicted[0, 0] = 2
        predicted[2, :2] = torch.tensor([0, 2])
        predicted[3, :2] = torch.tensor([0, 2])
        decoded = {
            "selected": selected,
            "sequences": predicted,
            "counts": batch["action_counts"].clone(),
        }
        exact = trainer.semantic_exact(decoded, batch, modes)
        self.assertEqual(exact.tolist(), [True, True, False, True])

    def test_daily_weights_equalize_total_date_mass(self) -> None:
        counts = {
            "2026-08-13": {"train": 100, "valid": 10},
            "2026-08-14": {"train": 400, "valid": 40},
        }
        weights = trainer.daily_equal_weights(counts)
        self.assertAlmostEqual(100 * weights["2026-08-13"], 250.0)
        self.assertAlmostEqual(400 * weights["2026-08-14"], 250.0)

    def test_choice_metric_excludes_only_forced_outcomes(self) -> None:
        batch = synthetic_batch()
        batch["option_mask"][0, 1:] = False
        modes = trainer.action_mode_ids(
            batch["contexts"], batch["min_counts"], batch["max_counts"]
        )
        choice_rows = trainer.choice_bearing_rows(batch, modes)
        self.assertEqual(choice_rows.tolist(), [False, True, True, True])


if __name__ == "__main__":
    unittest.main()
