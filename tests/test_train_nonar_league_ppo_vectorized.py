from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import train_nonar_league_ppo as trainer  # noqa: E402


def reference_deterministic(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
    modes = trainer.nonar.selection_modes(batch)
    optional = modes == trainer.nonar.MODE_OPTIONAL_SINGLE
    option_logits = outputs["policy_logits"].float()
    count_logits = outputs["count_logits"].float()
    allowed = trainer.legacy.count_allowed_mask(batch)
    masked_count = count_logits.masked_fill(~allowed, -1e9)
    fixed = batch["min_counts"] == batch["max_counts"]
    count_dist = torch.distributions.Categorical(logits=masked_count)
    counts = masked_count.argmax(dim=1)
    counts = torch.where(
        fixed,
        batch["min_counts"].clamp(0, trainer.legacy.MAX_ACTION_COUNT),
        counts,
    )
    counts = torch.minimum(counts, batch["option_mask"].sum(dim=1))
    log_probs = torch.where(
        fixed,
        torch.zeros_like(count_dist.log_prob(counts)),
        count_dist.log_prob(counts),
    )
    entropies = torch.where(
        fixed,
        torch.zeros_like(count_dist.entropy()),
        count_dist.entropy(),
    )
    optional_dist = torch.distributions.Categorical(
        logits=trainer._optional_joint_logits(outputs, batch, 1.0)
    )
    optional_choice = optional_dist.logits.argmax(dim=1)
    skip = option_logits.shape[1]
    counts = torch.where(optional, (optional_choice != skip).long(), counts)
    log_probs = torch.where(
        optional, optional_dist.log_prob(optional_choice), log_probs
    )
    entropies = torch.where(optional, optional_dist.entropy(), entropies)

    actions: list[list[int]] = []
    for row_index in range(option_logits.shape[0]):
        if bool(optional[row_index]):
            choice = int(optional_choice[row_index])
            actions.append([] if choice == skip else [choice])
            continue
        remaining = batch["option_mask"][row_index].clone()
        row_actions: list[int] = []
        for _ in range(int(counts[row_index])):
            logits = option_logits[row_index].masked_fill(~remaining, -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            chosen = int(logits.argmax())
            row_actions.append(chosen)
            log_probs[row_index] += distribution.log_prob(
                torch.tensor(chosen)
            )
            if int(remaining.sum()) > 1:
                entropies[row_index] += distribution.entropy()
            remaining[chosen] = False
        actions.append(row_actions)
    return actions, log_probs, entropies


class VectorizedSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = {
            "min_counts": torch.tensor([1, 0, 2, 1]),
            "max_counts": torch.tensor([1, 1, 3, 3]),
            "contexts": torch.tensor([0, 0, 34, 0]),
            "option_mask": torch.tensor(
                [
                    [True, True, True, False, False],
                    [True, True, True, True, False],
                    [True, True, True, True, True],
                    [False, True, True, True, False],
                ]
            ),
        }
        self.outputs = {
            "policy_logits": torch.tensor(
                [
                    [0.2, 2.0, 0.1, -30.0, -30.0],
                    [0.0, 0.5, 1.0, 1.5, -30.0],
                    [3.0, 1.0, 4.0, 2.0, 0.0],
                    [-30.0, 0.4, 0.9, 0.1, -30.0],
                ]
            ),
            "count_logits": torch.zeros((4, 61)),
            "skip_logits": torch.tensor([0.0, 2.0, 0.0, 0.0]),
            "value_logits": torch.tensor([0.0, 1.0, -1.0, 0.5]),
        }
        self.outputs["count_logits"][2, 3] = 2.0
        self.outputs["count_logits"][3, 2] = 2.0

    def test_deterministic_matches_row_reference(self) -> None:
        expected_actions, expected_log_prob, expected_entropy = (
            reference_deterministic(self.outputs, self.batch)
        )
        actions, log_prob, entropy, values = trainer.sample_nonar_actions(
            self.outputs,
            self.batch,
            deterministic=True,
        )
        self.assertEqual(actions, expected_actions)
        torch.testing.assert_close(log_prob, expected_log_prob)
        torch.testing.assert_close(entropy, expected_entropy)
        torch.testing.assert_close(
            values, torch.sigmoid(self.outputs["value_logits"])
        )

    def test_stochastic_actions_are_legal_and_finite(self) -> None:
        torch.manual_seed(123)
        actions, log_prob, entropy, _ = trainer.sample_nonar_actions(
            self.outputs,
            self.batch,
            deterministic=False,
        )
        for row, selected in enumerate(actions):
            self.assertGreaterEqual(len(selected), int(self.batch["min_counts"][row]))
            self.assertLessEqual(len(selected), int(self.batch["max_counts"][row]))
            self.assertEqual(len(selected), len(set(selected)))
            self.assertTrue(all(bool(self.batch["option_mask"][row, x]) for x in selected))
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue((entropy >= -1e-6).all())


if __name__ == "__main__":
    unittest.main()
