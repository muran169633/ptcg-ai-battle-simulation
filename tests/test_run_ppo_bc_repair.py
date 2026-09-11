from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_ppo_bc_repair as repair  # noqa: E402


class PPOBCRepairTests(unittest.TestCase):
    def test_request_requires_exactly_eight_unique_batches(self) -> None:
        repair.validate_repair_request(8, [58, 70, 60, 62, 59, 61, 14, 2])
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            repair.validate_repair_request(4, [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "length"):
            repair.validate_repair_request(8, [0, 1])
        with self.assertRaisesRegex(ValueError, "without replacement"):
            repair.validate_repair_request(8, [0, 1, 2, 3, 4, 5, 6, 6])

    def test_parent_config_rejects_any_frozen_field_drift(self) -> None:
        for name, expected in repair.EXPECTED_PARENT_CONFIG.items():
            raw = dict(repair.EXPECTED_PARENT_CONFIG)
            raw[name] = (
                expected + 1
                if isinstance(expected, (int, float))
                and not isinstance(expected, bool)
                else f"{expected}-changed"
            )
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    repair.validate_parent_config(raw)

    def test_cache_hash_is_stable_and_order_sensitive(self) -> None:
        first = {
            "contexts": torch.tensor([7, 34], dtype=torch.long),
            "targets": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        second = {
            "targets": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "contexts": torch.tensor([34, 7], dtype=torch.long),
        }
        digest_a, batches_a = repair.replay_cache_manifest([first, second])
        digest_b, batches_b = repair.replay_cache_manifest(
            [copy.deepcopy(first), copy.deepcopy(second)]
        )
        self.assertEqual(digest_a, digest_b)
        self.assertEqual(batches_a, batches_b)
        digest_reversed, _ = repair.replay_cache_manifest([second, first])
        self.assertNotEqual(digest_a, digest_reversed)

    def test_nested_optimizer_hash_ignores_mapping_insertion_order(self) -> None:
        left = {
            "state": {
                0: {
                    "step": torch.tensor(8.0),
                    "exp_avg": torch.tensor([1.0, 2.0]),
                }
            },
            "param_groups": [{"lr": 9e-7, "params": [0]}],
        }
        right = {
            "param_groups": [{"params": [0], "lr": 9e-7}],
            "state": {
                0: {
                    "exp_avg": torch.tensor([1.0, 2.0]),
                    "step": torch.tensor(8.0),
                }
            },
        }
        self.assertEqual(
            repair.nested_sha256(left),
            repair.nested_sha256(right),
        )
        right["state"][0]["step"] = torch.tensor(9.0)
        self.assertNotEqual(
            repair.nested_sha256(left),
            repair.nested_sha256(right),
        )

    def test_optimizer_steps_requires_complete_scalar_steps(self) -> None:
        state = {
            "state": {
                0: {"step": torch.tensor(8.0)},
                1: {"step": torch.tensor(8.0)},
            }
        }
        self.assertEqual(repair.optimizer_steps(state), [8, 8])
        with self.assertRaisesRegex(ValueError, "no AdamW step"):
            repair.optimizer_steps({"state": {0: {}}})
        with self.assertRaisesRegex(ValueError, "not scalar"):
            repair.optimizer_steps(
                {"state": {0: {"step": torch.tensor([8.0, 8.0])}}}
            )

    def test_changed_names_and_finite_checks_fail_closed(self) -> None:
        before = {
            "actor": torch.tensor([1.0]),
            "value": torch.tensor([2.0]),
        }
        after = {
            "actor": torch.tensor([1.5]),
            "value": torch.tensor([2.0]),
        }
        self.assertEqual(
            repair.changed_tensor_names(before, after),
            ["actor"],
        )
        self.assertTrue(repair.finite_nested({"loss": [1.0, torch.ones(2)]}))
        self.assertFalse(
            repair.finite_nested({"loss": torch.tensor(float("nan"))})
        )
        with self.assertRaisesRegex(ValueError, "names changed"):
            repair.changed_tensor_names(before, {"actor": torch.tensor([1.0])})


if __name__ == "__main__":
    unittest.main()
