from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import parallel_rollout as parallel  # noqa: E402


class ParallelModeARRolloutTests(unittest.TestCase):
    def test_remote_selfplay_game_trains_both_seats(self) -> None:
        game = parallel._game_record(
            {
                "uid": 1,
                "opponent_index": -1,
                "opponent_name": None,
                "learner_seat": None,
                "trainable_seats": [0, 1],
                "seat_deck_hash": {0: "deck", 1: "deck"},
            }
        )
        self.assertEqual(game.trainable_seats, {0, 1})
        self.assertEqual(game.seat_policy, {0: -1, 1: -1})
        self.assertIsNone(game.opponent_name)

    def test_remote_frozen_game_trains_only_learner_seat(self) -> None:
        game = parallel._game_record(
            {
                "uid": 2,
                "opponent_index": 4,
                "opponent_name": "bc@rank05",
                "learner_seat": 1,
                "trainable_seats": [1],
                "seat_deck_hash": {0: "opponent", 1: "learner"},
            }
        )
        self.assertEqual(game.trainable_seats, {1})
        self.assertEqual(game.seat_policy, {0: 4, 1: -1})
        self.assertEqual(game.opponent_name, "bc@rank05")

    def test_invalid_remote_trainable_seats_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parallel._game_record(
                {
                    "uid": 3,
                    "opponent_index": 0,
                    "opponent_name": "bc",
                    "learner_seat": 0,
                    "trainable_seats": [],
                    "seat_deck_hash": {0: "learner", 1: "opponent"},
                }
            )


if __name__ == "__main__":
    unittest.main()
