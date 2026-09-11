from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "seeded_ptcg_engine.py"
SPEC = importlib.util.spec_from_file_location("seeded_ptcg_engine", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
seeded = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = seeded
SPEC.loader.exec_module(seeded)


def read_deck(path: Path) -> list[int]:
    return [
        int(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def first_legal_action(observation: dict) -> list[int]:
    select = observation["select"]
    minimum = int(select.get("minCount", 0) or 0)
    return list(range(minimum))


def policy_observation(observation: dict) -> dict:
    # State serialization copies a few C++ structs with padding, so two
    # semantically identical search payloads need not be byte-identical. The
    # public JSON observation and engine evolution are the deterministic
    # contract that rollout policies consume here.
    return {
        key: value
        for key, value in observation.items()
        if key != "search_begin_input"
    }


class SeededEngineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = (
            ROOT / "build" / "local_seeded_engine_test" / "libcg_seeded.so"
        )
        subprocess.run(
            [
                "bash",
                str(ROOT / "tools" / "build_local_seeded_engine.sh"),
                str(cls.library),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        cls.deck = read_deck(
            ROOT / "dataset" / "sample_submission" / "sample_submission"
            / "deck.csv"
        )

    def test_same_seed_matches_through_a_complete_battle(self) -> None:
        left = seeded.SeededRawBattle(
            self.library,
            self.deck,
            self.deck,
            seed=0x1234ABCD,
        )
        right = seeded.SeededRawBattle(
            self.library,
            self.deck,
            self.deck,
            seed=0x1234ABCD,
        )
        try:
            for _ in range(512):
                self.assertEqual(
                    policy_observation(left.observation),
                    policy_observation(right.observation),
                )
                if left.result != -1:
                    break
                action = first_legal_action(left.observation)
                left_observation, left_error = left.step(action)
                right_observation, right_error = right.step(action)
                self.assertEqual(left_error, 0)
                self.assertEqual(right_error, 0)
                self.assertEqual(
                    policy_observation(left_observation),
                    policy_observation(right_observation),
                )
            self.assertNotEqual(left.result, -1)
            self.assertEqual(left.result, right.result)
        finally:
            left.close()
            right.close()

    def test_zero_seed_is_deterministic(self) -> None:
        left = seeded.SeededRawBattle(
            self.library,
            self.deck,
            self.deck,
            seed=0,
        )
        right = seeded.SeededRawBattle(
            self.library,
            self.deck,
            self.deck,
            seed=0,
        )
        try:
            self.assertEqual(
                policy_observation(left.observation),
                policy_observation(right.observation),
            )
            action = first_legal_action(left.observation)
            left_observation, left_error = left.step(action)
            right_observation, right_error = right.step(action)
            self.assertEqual(left_error, 0)
            self.assertEqual(right_error, 0)
            self.assertEqual(
                policy_observation(left_observation),
                policy_observation(right_observation),
            )
        finally:
            left.close()
            right.close()

    def test_different_seeds_change_initial_hidden_state(self) -> None:
        left = seeded.SeededRawBattle(
            self.library,
            self.deck,
            self.deck,
            seed=17,
        )
        right = seeded.SeededRawBattle(
            self.library,
            self.deck,
            self.deck,
            seed=18,
        )
        try:
            # The first prompt precedes visible setup. Take the same first-turn
            # choice so the seeded deck permutations become observable.
            left_observation, left_error = left.step([0])
            right_observation, right_error = right.step([0])
            self.assertEqual(left_error, 0)
            self.assertEqual(right_error, 0)
            self.assertNotEqual(
                policy_observation(left_observation),
                policy_observation(right_observation),
            )
        finally:
            left.close()
            right.close()


if __name__ == "__main__":
    unittest.main()
