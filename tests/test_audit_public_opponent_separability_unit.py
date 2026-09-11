from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from audit_public_opponent_separability import (  # noqa: E402
    collection_quality_gates,
    opaque_group_id_pool,
)


class OpaqueGroupIdPoolTests(unittest.TestCase):
    def test_is_reproducible_complete_permutation_without_class_ranges(
        self,
    ) -> None:
        identifiers = opaque_group_id_pool(1024, 20261184)
        self.assertEqual(identifiers, opaque_group_id_pool(1024, 20261184))
        self.assertNotEqual(identifiers, opaque_group_id_pool(1024, 20261185))
        self.assertEqual(set(identifiers), set(range(1024)))
        self.assertNotEqual(identifiers, list(range(1024)))

        # With two 512-row classes committed in an interleaved schedule, the
        # first and second halves of the key pool both span the old class-coded
        # boundary. A key value alone therefore does not expose the label.
        self.assertTrue(any(identifier >= 512 for identifier in identifiers[:512]))
        self.assertTrue(any(identifier < 512 for identifier in identifiers[512:]))

    def test_rejects_empty_pool(self) -> None:
        with self.assertRaises(ValueError):
            opaque_group_id_pool(0, 1)


class CollectionQualityGateTests(unittest.TestCase):
    @staticmethod
    def valid_stats() -> dict[str, int | float]:
        return {
            "replacement_rate": 0.01,
            "start_errors": 0,
            "history_turn_regression_resets": 0,
            "learner_seat_0_trajectories": 256,
            "learner_seat_1_trajectories": 256,
        }

    def test_clean_symmetric_collection_passes(self) -> None:
        gates = collection_quality_gates(
            self.valid_stats(),
            self.valid_stats(),
        )
        self.assertTrue(all(gates.values()))

    def test_start_error_and_turn_regression_are_zero_tolerance(self) -> None:
        stats_a = self.valid_stats()
        stats_b = self.valid_stats()
        stats_a["start_errors"] = 1
        stats_b["history_turn_regression_resets"] = 1
        gates = collection_quality_gates(stats_a, stats_b)
        self.assertFalse(gates["class_0_start_errors_zero"])
        self.assertFalse(
            gates["class_1_history_turn_regression_resets_zero"]
        )


if __name__ == "__main__":
    unittest.main()
