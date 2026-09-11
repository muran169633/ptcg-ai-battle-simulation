from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import ppo_router  # noqa: E402


EXPECTED_DENSE_FEATURES = (
    "turn",
    "turn_action_count",
    "self_is_first",
    "supporter_played",
    "stadium_played",
    "energy_attached",
    "retreated",
    "stadium_present",
    "self_deck_count",
    "self_hand_count",
    "self_active_count",
    "self_bench_count",
    "self_discard_count",
    "self_prize_count",
    "opponent_deck_count",
    "opponent_hand_count",
    "opponent_active_count",
    "opponent_bench_count",
    "opponent_discard_count",
    "opponent_prize_count",
    "self_active_hp_fraction",
    "self_active_damage_fraction",
    "self_active_energy_count",
    "opponent_active_hp_fraction",
    "opponent_active_damage_fraction",
    "opponent_active_energy_count",
    "select_type",
    "select_context",
    "select_min_count",
    "select_max_count",
    "select_option_count",
    "recent_log_count",
)
EXPECTED_DENSE_BOUNDS = (
    60.0,
    30.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    60.0,
    30.0,
    2.0,
    8.0,
    60.0,
    6.0,
    60.0,
    30.0,
    2.0,
    8.0,
    60.0,
    6.0,
    1.0,
    1.0,
    12.0,
    1.0,
    1.0,
    12.0,
    64.0,
    64.0,
    60.0,
    60.0,
    128.0,
    64.0,
)
EXPECTED_HASH_WHITELIST = {
    "current": (
        "turn",
        "turnActionCount",
        "supporterPlayed",
        "stadiumPlayed",
        "energyAttached",
        "retreated",
    ),
    "player": (
        "benchMax",
        "deckCount",
        "handCount",
        "poisoned",
        "burned",
        "asleep",
        "paralyzed",
        "confused",
    ),
    "card": ("id", "hp", "maxHp", "appearThisTurn"),
    "select": (
        "type",
        "context",
        "minCount",
        "maxCount",
        "remainDamageCounter",
        "remainEnergyCost",
    ),
    "option": (
        "type",
        "number",
        "area",
        "index",
        "toolIndex",
        "energyIndex",
        "count",
        "inPlayArea",
        "inPlayIndex",
        "attackId",
        "cardId",
        "specialConditionType",
    ),
    "log": (
        "type",
        "hasBasicPokemon",
        "cardId",
        "fromArea",
        "toArea",
        "cardIdActive",
        "cardIdBench",
        "cardIdBefore",
        "cardIdAfter",
        "cardIdTarget",
        "attackId",
        "value",
        "putDamageCounter",
        "isRecover",
        "head",
        "result",
        "reason",
    ),
}
EXPECTED_SAMPLE_SHA256 = (
    "0759b0f464a117544d9b8813149703085251adba9b715624e19aacba89c20cf3"
)


def sample_observation() -> dict[str, object]:
    return {
        "search_begin_input": "opaque-engine-state",
        "current": {
            "turn": 5,
            "turnActionCount": 2,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": True,
            "stadiumPlayed": False,
            "energyAttached": True,
            "retreated": False,
            "result": -1,
            "stadium": [{"id": 901, "serial": 9901}],
            "looking": [{"id": 777, "serial": 9777}, None],
            "players": [
                {
                    "active": [
                        {
                            "id": 101,
                            "serial": 1001,
                            "hp": 180,
                            "maxHp": 240,
                            "appearThisTurn": False,
                            "energies": [1, 1],
                            "energyCards": [{"id": 601, "serial": 1601}],
                            "tools": [{"id": 701, "serial": 1701}],
                            "preEvolution": [{"id": 99, "serial": 1099}],
                        }
                    ],
                    "bench": [{"id": 102, "serial": 1002}],
                    "benchMax": 5,
                    "deckCount": 42,
                    "discard": [{"id": 301, "serial": 1301}],
                    "prize": [None, None, None, None],
                    "handCount": 6,
                    "hand": [
                        {"id": 201, "serial": 1201},
                        {"id": 202, "serial": 1202},
                    ],
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
                {
                    "active": [
                        {
                            "id": 401,
                            "serial": 1401,
                            "hp": 120,
                            "maxHp": 180,
                            "appearThisTurn": False,
                            "energies": [2],
                            "energyCards": [{"id": 602, "serial": 1602}],
                            "tools": [],
                            "preEvolution": [],
                        }
                    ],
                    "bench": [
                        {"id": 402, "serial": 1402},
                        {"id": 403, "serial": 1403},
                    ],
                    "benchMax": 5,
                    "deckCount": 39,
                    "discard": [
                        {"id": 501, "serial": 1501},
                        {"id": 502, "serial": 1502},
                    ],
                    "prize": [None, None, None],
                    "handCount": 7,
                    "hand": None,
                    "poisoned": False,
                    "burned": True,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
            ],
        },
        "select": {
            "type": 1,
            "context": 34,
            "minCount": 1,
            "maxCount": 2,
            "remainDamageCounter": 0,
            "remainEnergyCost": 1,
            "option": [
                {
                    "type": 3,
                    "area": 2,
                    "index": 0,
                    "playerIndex": 0,
                    "cardId": 201,
                    "serial": 1201,
                },
                {
                    "type": 8,
                    "area": 3,
                    "index": 0,
                    "playerIndex": 1,
                    "inPlayArea": 4,
                    "inPlayIndex": 0,
                    "serial": 1401,
                },
            ],
            "deck": [{"id": 801, "serial": 1801}],
            "contextCard": {"id": 201, "serial": 1201},
            "effect": {"id": 301, "serial": 1301},
        },
        "logs": [
            {
                "type": 7,
                "playerIndex": 0,
                "cardId": 201,
                "serial": 1201,
                "fromArea": 2,
                "toArea": 3,
            },
            {
                "type": 20,
                "playerIndex": 1,
                "cardIdActive": 401,
                "serialActive": 1401,
                "attackId": 2,
                "value": 60,
            },
        ],
    }


class PublicRouterFeatureTests(unittest.TestCase):
    def test_contract_order_bounds_shape_dtype_and_finite(self) -> None:
        self.assertEqual(
            ppo_router.PUBLIC_ROUTER_FEATURE_VERSION,
            "ptcg-public-router-features-v1",
        )
        self.assertEqual(ppo_router.PUBLIC_ROUTER_FEATURE_DIM, 160)
        self.assertEqual(
            ppo_router.ROUTER_FEATURE_VERSION,
            ppo_router.PUBLIC_ROUTER_FEATURE_VERSION,
        )
        self.assertEqual(
            ppo_router.ROUTER_FEATURE_DIM,
            ppo_router.PUBLIC_ROUTER_FEATURE_DIM,
        )
        self.assertEqual(ppo_router.DENSE_FEATURE_NAMES, EXPECTED_DENSE_FEATURES)
        self.assertEqual(ppo_router.DENSE_FEATURE_BOUNDS, EXPECTED_DENSE_BOUNDS)
        self.assertEqual(
            {
                "current": ppo_router.CURRENT_HASH_FIELDS,
                "player": ppo_router.PLAYER_HASH_FIELDS,
                "card": ppo_router.CARD_HASH_FIELDS,
                "select": ppo_router.SELECT_HASH_FIELDS,
                "option": ppo_router.OPTION_HASH_FIELDS,
                "log": ppo_router.LOG_HASH_FIELDS,
            },
            EXPECTED_HASH_WHITELIST,
        )
        self.assertEqual(len(set(ppo_router.DENSE_FEATURE_NAMES)), 32)
        self.assertNotIn("your_index", ppo_router.DENSE_FEATURE_NAMES)
        self.assertIn("stadium_present", ppo_router.DENSE_FEATURE_NAMES)
        feature = ppo_router.public_router_features(sample_observation())
        self.assertEqual(feature.shape, (160,))
        self.assertEqual(feature.dtype, torch.float32)
        self.assertTrue(torch.isfinite(feature).all().item())
        self.assertEqual(
            hashlib.sha256(feature.numpy().tobytes()).hexdigest(),
            EXPECTED_SAMPLE_SHA256,
        )

    def test_stateless_a_b_a(self) -> None:
        observation_a = sample_observation()
        serialized_before = json.dumps(observation_a, sort_keys=True)
        observation_b = copy.deepcopy(observation_a)
        observation_b["current"]["players"][1]["bench"].append(  # type: ignore[index]
            {"id": 499, "serial": 1499}
        )
        first_a = ppo_router.public_router_features(observation_a)
        feature_b = ppo_router.public_router_features(observation_b)
        second_a = ppo_router.public_router_features(observation_a)
        self.assertTrue(torch.equal(first_a, second_a))
        self.assertFalse(torch.equal(first_a, feature_b))
        self.assertEqual(
            json.dumps(observation_a, sort_keys=True),
            serialized_before,
        )

    def test_json_roundtrip_is_exact(self) -> None:
        observation = sample_observation()
        roundtripped = json.loads(json.dumps(observation))
        self.assertTrue(
            torch.equal(
                ppo_router.public_router_features(observation),
                ppo_router.public_router_features(roundtripped),
            )
        )

    def test_identity_opaque_serial_and_unknown_fields_are_ignored(self) -> None:
        base = sample_observation()
        changed = copy.deepcopy(base)
        changed.update(
            {
                "search_begin_input": "different-opaque-value",
                "opponent_name": "v3",
                "team_name": "secret-team",
                "deck_hash": "identity-bearing-deck-hash",
                "unknown": {"anything": [1, 2, 3]},
            }
        )
        current = changed["current"]  # type: ignore[assignment]
        current["unknownCurrentField"] = 123  # type: ignore[index]
        current["stadium"][0]["serial"] = 1  # type: ignore[index]
        for player in current["players"]:  # type: ignore[index]
            player["opponent_name"] = "ignored"
            for zone in ("active", "bench", "discard"):
                for card in player[zone]:
                    card["serial"] = card.get("serial", 0) + 50000
                    card["unknownCardField"] = "ignored"
            for card in player["hand"] or []:
                card["serial"] += 50000
        select = changed["select"]  # type: ignore[assignment]
        select["unknownSelectField"] = True  # type: ignore[index]
        for option in select["option"]:  # type: ignore[index]
            option["serial"] = 999999
            option["team_name"] = "ignored"
        for event in changed["logs"]:  # type: ignore[index]
            event["serial"] = 999999
            event["serialActive"] = 999999
            event["unknownLogField"] = "ignored"
        self.assertTrue(
            torch.equal(
                ppo_router.public_router_features(base),
                ppo_router.public_router_features(changed),
            )
        )

    def test_zero_player_index_is_preserved_as_self(self) -> None:
        self_event = sample_observation()
        opponent_event = copy.deepcopy(self_event)
        opponent_event["logs"][0]["playerIndex"] = 1  # type: ignore[index]
        self.assertFalse(
            torch.equal(
                ppo_router.public_router_features(self_event),
                ppo_router.public_router_features(opponent_event),
            )
        )

        zero_option = sample_observation()
        missing_option = copy.deepcopy(zero_option)
        del missing_option["select"]["option"][0]["playerIndex"]  # type: ignore[index]
        self.assertFalse(
            torch.equal(
                ppo_router.public_router_features(zero_option),
                ppo_router.public_router_features(missing_option),
            )
        )

    def test_opponent_hand_content_is_ignored_even_if_mistakenly_visible(self) -> None:
        hidden = sample_observation()
        exposed = copy.deepcopy(hidden)
        exposed["current"]["players"][1]["hand"] = [  # type: ignore[index]
            {"id": 999, "serial": 1},
            {"id": 998, "serial": 2, "unknown": "ignored"},
        ]
        self.assertTrue(
            torch.equal(
                ppo_router.public_router_features(hidden),
                ppo_router.public_router_features(exposed),
            )
        )

    def test_nonfinite_values_are_safely_bounded(self) -> None:
        observation = sample_observation()
        observation["current"]["turn"] = float("nan")  # type: ignore[index]
        observation["current"]["turnActionCount"] = float("inf")  # type: ignore[index]
        observation["current"]["players"][0]["active"][0]["hp"] = (  # type: ignore[index]
            float("-inf")
        )
        feature = ppo_router.public_router_features(observation)
        self.assertEqual(feature.shape, (160,))
        self.assertTrue(torch.isfinite(feature).all().item())

    def test_pythonhashseed_does_not_change_features(self) -> None:
        payload = json.dumps(sample_observation(), sort_keys=True)
        code = (
            "import json,sys;"
            "from ppo_router import public_router_features;"
            "x=public_router_features(json.loads(sys.stdin.read()));"
            "sys.stdout.write(x.numpy().tobytes().hex())"
        )
        outputs = []
        for seed in ("1", "777"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            environment["PYTHONPATH"] = str(REPO_ROOT / "tools")
            completed = subprocess.run(
                [sys.executable, "-c", code],
                input=payload,
                text=True,
                capture_output=True,
                check=True,
                cwd=REPO_ROOT,
                env=environment,
            )
            outputs.append(completed.stdout)
        self.assertEqual(outputs[0], outputs[1])


class PublicOpponentHistoryV2Tests(unittest.TestCase):
    @staticmethod
    def observation(
        turn: int,
        logs: list[dict[str, object]],
        your_index: int = 0,
    ) -> dict[str, object]:
        return {
            "current": {
                "turn": turn,
                "yourIndex": your_index,
            },
            "select": {
                "type": 999,
                "option": [{"policy_id": "must-not-enter-history"}],
            },
            "logs": logs,
            "team_name": "must-not-enter-history",
            "submission_id": "must-not-enter-history",
            "policy_id": "must-not-enter-history",
            "deck_hash": "must-not-enter-history",
        }

    def test_contract_and_causal_opponent_only_leakage_barrier(self) -> None:
        self.assertEqual(
            ppo_router.PUBLIC_HISTORY_V2_FEATURE_VERSION,
            "ptcg-public-opponent-history-v2",
        )
        self.assertEqual(ppo_router.PUBLIC_HISTORY_V2_FEATURE_DIM, 136)
        self.assertNotIn(
            "result",
            ppo_router.PUBLIC_HISTORY_V2_EVENT_FIELDS,
        )

        opponent_event = {
            "type": 20,
            "playerIndex": 1,
            "cardIdActive": 401,
            "attackId": 2,
            "value": 60,
            "serial": 1401,
            "team_name": "team-a",
            "submission_id": "submission-a",
            "policy_id": "policy-a",
            "deck_hash": "deck-a",
            "result": 0,
        }
        own_event = {
            "type": 7,
            "playerIndex": 0,
            "cardId": 201,
            "fromArea": 2,
            "toArea": 3,
        }
        base_observation = self.observation(4, [own_event, opponent_event])
        changed_observation = copy.deepcopy(base_observation)
        changed_observation.update(
            {
                "team_name": "team-b",
                "submission_id": "submission-b",
                "policy_id": "policy-b",
                "deck_hash": "deck-b",
            }
        )
        changed_observation["select"] = {  # type: ignore[index]
            "type": 1,
            "option": [{"own_action": "changed"}],
        }
        changed_logs = changed_observation["logs"]  # type: ignore[index]
        changed_logs[0].update(  # type: ignore[index]
            {
                "cardId": 999,
                "attackId": 99,
                "team_name": "own-team-changed",
            }
        )
        changed_logs[1].update(  # type: ignore[index]
            {
                "serial": 999999,
                "team_name": "team-c",
                "submission_id": "submission-c",
                "policy_id": "policy-c",
                "deck_hash": "deck-c",
                "result": 1,
                "unknown": {"future": "must-be-ignored"},
            }
        )
        changed_logs.append(  # type: ignore[union-attr]
            {
                "type": 99,
                "playerIndex": 0,
                "cardId": 12345,
                "attackId": 77,
            }
        )
        changed_logs.append(  # type: ignore[union-attr]
            {
                "type": 88,
                "cardId": 54321,
                "attackId": 66,
            }
        )

        base_state = ppo_router.PublicOpponentHistoryV2(observer_seat=0)
        changed_state = ppo_router.PublicOpponentHistoryV2(observer_seat=0)
        base_feature = base_state.observe_and_features(base_observation)
        changed_feature = changed_state.observe_and_features(
            changed_observation
        )
        self.assertTrue(torch.equal(base_feature, changed_feature))
        self.assertEqual(base_state.event_count, 1)
        self.assertEqual(changed_state.event_count, 1)
        self.assertTrue(torch.isfinite(base_feature).all().item())

        public_change = copy.deepcopy(base_observation)
        public_change["logs"][1]["attackId"] = 3  # type: ignore[index]
        public_change_state = ppo_router.PublicOpponentHistoryV2(
            observer_seat=0
        )
        self.assertFalse(
            torch.equal(
                base_feature,
                public_change_state.observe_and_features(public_change),
            )
        )

    def test_incremental_log_windows_accumulate_once_per_observation(self) -> None:
        event_a = {"type": 1, "playerIndex": 1, "cardId": 101}
        event_b = {"type": 2, "playerIndex": 1, "cardId": 102}
        state = ppo_router.PublicOpponentHistoryV2(observer_seat=0)

        first_observation = self.observation(2, [event_a])
        self.assertEqual(state.observe(first_observation), 1)
        stable = state.features().clone()
        self.assertEqual(state.observe(first_observation), 0)
        self.assertTrue(torch.equal(stable, state.features()))

        # A later engine observation is a new incremental window even when it
        # contains a public payload identical to an earlier event.
        self.assertEqual(state.observe(self.observation(3, [event_a])), 1)
        self.assertEqual(state.observe(self.observation(3, [event_b])), 1)
        self.assertEqual(state.event_count, 3)

    def test_opponent_policy_callback_observations_are_not_visible(self) -> None:
        state = ppo_router.PublicOpponentHistoryV2(observer_seat=0)
        opponent_callback = self.observation(
            3,
            [{"type": 20, "playerIndex": 1, "attackId": 9}],
            your_index=1,
        )
        self.assertEqual(state.observe(opponent_callback), 0)
        self.assertEqual(state.event_count, 0)
        self.assertTrue(torch.equal(state.features(), torch.zeros(136)))

        learner_callback = self.observation(
            4,
            [{"type": 20, "playerIndex": 1, "attackId": 9}],
            your_index=0,
        )
        self.assertEqual(state.observe(learner_callback), 1)
        self.assertEqual(state.event_count, 1)

    def test_explicit_cross_game_reset_clears_process_slot_state(self) -> None:
        state = ppo_router.PublicOpponentHistoryV2(observer_seat=0)
        state.observe(
            self.observation(
                7,
                [{"type": 20, "playerIndex": 1, "attackId": 4}],
            )
        )
        self.assertEqual(state.generation, 0)
        self.assertEqual(state.event_count, 1)
        self.assertFalse(torch.equal(state.features(), torch.zeros(136)))

        state.reset(observer_seat=1, reason="process_slot_reuse")
        self.assertEqual(state.generation, 1)
        self.assertEqual(state.observer_seat, 1)
        self.assertEqual(state.last_reset_reason, "process_slot_reuse")
        self.assertEqual(state.event_count, 0)
        self.assertTrue(torch.equal(state.features(), torch.zeros(136)))

        next_game = self.observation(
            1,
            [{"type": 7, "playerIndex": 0, "cardId": 501}],
            your_index=1,
        )
        reused_feature = state.observe_and_features(next_game)
        fresh_state = ppo_router.PublicOpponentHistoryV2(observer_seat=1)
        fresh_feature = fresh_state.observe_and_features(next_game)
        self.assertTrue(torch.equal(reused_feature, fresh_feature))

    def test_turn_regression_defensively_resets_all_history(self) -> None:
        state = ppo_router.PublicOpponentHistoryV2(observer_seat=0)
        state.observe(
            self.observation(
                9,
                [{"type": 20, "playerIndex": 1, "attackId": 5}],
            )
        )
        regressed = self.observation(
            2,
            [{"type": 7, "playerIndex": 1, "cardId": 777}],
        )
        regressed_feature = state.observe_and_features(regressed)

        fresh_state = ppo_router.PublicOpponentHistoryV2(observer_seat=0)
        fresh_feature = fresh_state.observe_and_features(regressed)
        self.assertTrue(torch.equal(regressed_feature, fresh_feature))
        self.assertEqual(state.generation, 1)
        self.assertEqual(state.turn_regression_resets, 1)
        self.assertEqual(state.last_reset_reason, "turn_regression")
        self.assertEqual(state.event_count, 1)

    def test_observer_seat_requires_explicit_valid_absolute_value(self) -> None:
        for invalid in (-1, 2, True):
            with self.assertRaises(ValueError):
                ppo_router.PublicOpponentHistoryV2(invalid)


if __name__ == "__main__":
    unittest.main()
