from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = load_module(
    "evaluate_ppo_head_to_head_for_test",
    "evaluate_ppo_head_to_head.py",
)
runner = load_module(
    "run_gold_league_h2h_for_test",
    "run_gold_league_h2h.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_eval_result(
    candidate: Path,
    candidate_deck: Path,
    opponent: runner.OpponentSpec,
    *,
    games: int,
    seed: int,
    wins_by_seat: tuple[int, int],
    candidate_canonical_order: bool = False,
    candidate_hybrid_order: bool = False,
    opponent_canonical_order: bool | None = None,
) -> dict:
    if opponent_canonical_order is None:
        opponent_canonical_order = opponent.canonical_order
    per_seat = games // 2
    seat_rows = {}
    total_wins = 0
    total_losses = 0
    for seat, wins in enumerate(wins_by_seat):
        losses = per_seat - wins
        total_wins += wins
        total_losses += losses
        low, high = runner.wilson_interval(wins, per_seat)
        seat_rows[str(seat)] = {
            "valid_games": per_seat,
            "wins": wins,
            "losses": losses,
            "draws": 0,
            "win_rate": wins / per_seat,
            "wilson_95_low": low,
            "wilson_95_high": high,
        }
    low, high = runner.wilson_interval(total_wins, games)
    return {
        "candidate": {
            "path": str(candidate.resolve()),
            "sha256": sha256(candidate),
            "deck": str(candidate_deck.resolve()),
            "canonical_order": candidate_canonical_order,
            "hybrid_order": candidate_hybrid_order,
            "order_mode": evaluator.policy_order_mode(
                canonical_order=candidate_canonical_order,
                hybrid_order=candidate_hybrid_order,
            ),
        },
        "opponent": {
            "path": str(opponent.checkpoint.resolve()),
            "sha256": opponent.checkpoint_sha256,
            "deck": str(opponent.deck.resolve()),
            "canonical_order": opponent_canonical_order,
            "hybrid_order": False,
            "order_mode": (
                "canonical" if opponent_canonical_order else "raw"
            ),
        },
        "engine": {
            "games_requested": games,
            "environments": 4,
            "max_game_decisions": 1000,
            "engine_seed_control": False,
            "python_torch_seed": seed,
        },
        "evaluation": {
            "valid_games": games,
            "wins": total_wins,
            "losses": total_losses,
            "draws": 0,
            "invalid_games": 0,
            "win_rate": total_wins / games,
            "decisive_win_rate": total_wins / games,
            "mean_decisions": 10.0,
            "max_active_games": 4,
            "seconds": 1.0,
            "wilson_95_low": low,
            "wilson_95_high": high,
            "by_candidate_seat": seat_rows,
            "seat_balance": {
                "strict_even_balance_verified": True,
                "aggregate_equals_seat_sum_verified": True,
            },
        },
    }


class EvaluatorSeatAuditTest(unittest.TestCase):
    def test_legacy_totals_are_preserved_and_equal_seat_sum(self) -> None:
        stats = Counter(
            {
                "valid_games": 8,
                "wins": 5,
                "losses": 2,
                "draws": 1,
                "invalid_games": 3,
                "decisions": 80,
                "max_active_games": 4,
                "seat_0_valid_games": 4,
                "seat_0_wins": 3,
                "seat_0_losses": 1,
                "seat_1_valid_games": 4,
                "seat_1_wins": 2,
                "seat_1_losses": 1,
                "seat_1_draws": 1,
                "invalid_reason_battle_step_error": 2,
                "invalid_reason_max_game_decisions": 1,
                "seat_0_invalid_reason_battle_step_error": 1,
                "seat_1_invalid_reason_battle_step_error": 1,
                "seat_1_invalid_reason_max_game_decisions": 1,
                "invalid_error_message::illegal action": 2,
            }
        )
        result = evaluator.summarize_evaluation_stats(stats, 8, 2.5)
        legacy_keys = {
            "valid_games",
            "wins",
            "losses",
            "draws",
            "invalid_games",
            "win_rate",
            "decisive_win_rate",
            "mean_decisions",
            "max_active_games",
            "seconds",
        }
        self.assertTrue(legacy_keys <= set(result))
        self.assertEqual(result["wins"], 5)
        self.assertEqual(result["by_candidate_seat"]["0"]["valid_games"], 4)
        self.assertEqual(result["by_candidate_seat"]["1"]["valid_games"], 4)
        self.assertTrue(
            result["seat_balance"]["strict_even_balance_verified"]
        )
        self.assertTrue(
            result["seat_balance"]["aggregate_equals_seat_sum_verified"]
        )
        self.assertEqual(
            result["invalid_by_reason"],
            {"battle_step_error": 2, "max_game_decisions": 1},
        )
        self.assertEqual(
            result["invalid_by_candidate_seat"]["1"]["invalid_games"],
            2,
        )
        self.assertEqual(
            result["invalid_error_messages"],
            {"illegal action": 2},
        )
        self.assertTrue(result["invalid_audit_complete"])
        conservative = result["conservative_invalid_as_loss"]
        self.assertEqual(conservative["trials"], 11)
        self.assertEqual(conservative["wins"], 5)
        self.assertEqual(conservative["nonwins"], 6)
        self.assertAlmostEqual(conservative["win_rate"], 5 / 11)
        expected_low, expected_high = evaluator.wilson_interval(5, 11)
        self.assertAlmostEqual(conservative["wilson_95_low"], expected_low)
        self.assertAlmostEqual(conservative["wilson_95_high"], expected_high)
        self.assertAlmostEqual(conservative["invalid_rate"], 3 / 11)
        self.assertFalse(conservative["gate_ci_low_above_0_5"])

    def test_inconsistent_seat_totals_are_rejected(self) -> None:
        stats = Counter(
            {
                "valid_games": 8,
                "wins": 5,
                "losses": 3,
                "seat_0_valid_games": 4,
                "seat_0_wins": 3,
                "seat_0_losses": 1,
                "seat_1_valid_games": 4,
                "seat_1_wins": 1,
                "seat_1_losses": 3,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "Aggregate wins"):
            evaluator.summarize_evaluation_stats(stats, 8, 1.0)


class EvaluatorLoopDiagnosticTest(unittest.TestCase):
    @staticmethod
    def observation(*, turn: int = 7) -> dict:
        return {
            "current": {
                "yourIndex": 1,
                "turn": turn,
                "turnActionCount": 3,
                "result": -1,
            },
            "select": {
                "type": 5,
                "context": 34,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"large_private_payload": "do-not-emit" * 10_000}
                ],
            },
            "search_begin_input": "opaque-engine-state" * 10_000,
        }

    def test_default_off_collects_and_emits_nothing(self) -> None:
        collector = evaluator.LoopDiagnosticCollector(0)
        battle = object()
        collector.start_game(battle)
        collector.record_decision(
            battle,
            self.observation(),
            [0],
            decision=1,
            learner_seat=1,
        )
        collector.record_max_decisions(
            battle,
            self.observation(),
            decisions=1,
            learner_seat=1,
        )
        collector.finish_game(battle)
        self.assertIsNone(collector.render())

    def test_trace_is_tail_bounded_and_contains_only_summaries(self) -> None:
        collector = evaluator.LoopDiagnosticCollector(2)
        battle = object()
        collector.start_game(battle)
        oversized_action = list(
            range(evaluator.LOOP_DIAGNOSTIC_MAX_ACTION_ITEMS + 5)
        )
        for decision in (1, 2, 3):
            collector.record_decision(
                battle,
                self.observation(),
                oversized_action,
                decision=decision,
                learner_seat=1,
            )
        collector.record_max_decisions(
            battle,
            self.observation(),
            decisions=3,
            learner_seat=1,
        )
        result = collector.render()
        assert result is not None
        self.assertEqual(result["captured_games"], 1)
        game = result["games"][0]
        self.assertEqual([row["decision"] for row in game["tail"]], [2, 3])
        self.assertEqual(
            len(game["tail"][0]["action"]),
            evaluator.LOOP_DIAGNOSTIC_MAX_ACTION_ITEMS,
        )
        self.assertTrue(game["tail"][0]["action_truncated"])
        self.assertEqual(game["tail"][0]["acting_policy"], "candidate")
        self.assertEqual(game["tail"][0]["state"]["select"]["type"], 5)
        self.assertEqual(game["tail"][0]["state"]["select"]["context"], 34)
        self.assertEqual(
            game["tail_repeat_summary"]["unique_observation_states"],
            1,
        )
        self.assertEqual(
            game["tail_repeat_summary"]["repeated_state_action_entries"],
            1,
        )
        rendered = json.dumps(result)
        self.assertNotIn("do-not-emit", rendered)
        self.assertNotIn("opaque-engine-state", rendered)
        state = game["tail"][0]["state"]
        self.assertEqual(len(state["observation_sha256"]), 64)
        self.assertEqual(len(state["engine_state_sha256"]), 64)
        self.assertGreater(state["observation_json_bytes"], 100_000)
        self.assertGreater(state["engine_state_bytes"], 100_000)

    def test_capture_count_and_input_are_hard_bounded(self) -> None:
        collector = evaluator.LoopDiagnosticCollector(1)
        for index in range(evaluator.LOOP_DIAGNOSTIC_MAX_GAMES + 2):
            battle = object()
            collector.start_game(battle)
            collector.record_decision(
                battle,
                self.observation(turn=index),
                [0],
                decision=1,
                learner_seat=index % 2,
            )
            collector.record_max_decisions(
                battle,
                self.observation(turn=index),
                decisions=1,
                learner_seat=index % 2,
            )
            collector.finish_game(battle)
        result = collector.render()
        assert result is not None
        self.assertEqual(
            result["captured_games"], evaluator.LOOP_DIAGNOSTIC_MAX_GAMES
        )
        self.assertEqual(result["max_decision_events"], 10)
        self.assertEqual(result["dropped_games"], 2)
        for invalid in (-1, evaluator.LOOP_DIAGNOSTIC_MAX_TAIL + 1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "between 0 and"):
                    evaluator.LoopDiagnosticCollector(invalid)

    def test_evaluator_attaches_trace_only_for_decision_limit_when_enabled(
        self,
    ) -> None:
        test_case = self

        class FakeBattle:
            starts = 0

            def __init__(self, _deck0, _deck1) -> None:
                self.attempt = type(self).starts
                type(self).starts += 1
                self.steps = 0
                self.closed = False
                self.observation = test_case.observation(turn=0)

            @property
            def result(self) -> int:
                return int(self.observation["current"]["result"])

            def step(self, _action):
                self.steps += 1
                self.observation = test_case.observation(turn=self.steps)
                if self.attempt > 0:
                    self.observation["current"]["result"] = 1
                return self.observation, 0

            def close(self) -> None:
                self.closed = True

        class DummyModel:
            def eval(self):
                return self

        patches = (
            mock.patch.object(evaluator, "RawBattle", FakeBattle),
            mock.patch.object(
                evaluator,
                "live_feature",
                side_effect=lambda *_args: {"feature": 1},
            ),
            mock.patch.object(
                evaluator,
                "collate_features",
                side_effect=lambda features, *_args: features,
            ),
            mock.patch.object(
                evaluator,
                "model_forward",
                side_effect=lambda *_args: None,
            ),
            mock.patch.object(
                evaluator,
                "sample_ordered_actions",
                side_effect=lambda _outputs, batch, **_kwargs: (
                    [[0] for _ in batch],
                    None,
                    None,
                    None,
                ),
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        enabled = evaluator.evaluate_head_to_head_with_seats(
            DummyModel(),
            DummyModel(),
            [1] * 60,
            {},
            evaluator.torch.device("cpu"),
            games_target=1,
            environments=1,
            max_game_decisions=2,
            loop_diagnostic_tail=2,
        )
        self.assertEqual(enabled["invalid_by_reason"], {"max_game_decisions": 1})
        self.assertEqual(
            [
                row["decision"]
                for row in enabled["loop_diagnostic"]["games"][0]["tail"]
            ],
            [1, 2],
        )

        FakeBattle.starts = 0
        disabled = evaluator.evaluate_head_to_head_with_seats(
            DummyModel(),
            DummyModel(),
            [1] * 60,
            {},
            evaluator.torch.device("cpu"),
            games_target=1,
            environments=1,
            max_game_decisions=2,
        )
        self.assertNotIn("loop_diagnostic", disabled)
        self.assertEqual(
            disabled["invalid_by_reason"], enabled["invalid_by_reason"]
        )


class EvaluatorHybridOrderTest(unittest.TestCase):
    def test_order_mode_identity_is_explicit(self) -> None:
        self.assertEqual(
            evaluator.policy_order_mode(
                canonical_order=False,
                hybrid_order=False,
            ),
            "raw",
        )
        self.assertEqual(
            evaluator.policy_order_mode(
                canonical_order=True,
                hybrid_order=False,
            ),
            "canonical",
        )
        self.assertEqual(
            evaluator.policy_order_mode(
                canonical_order=False,
                hybrid_order=True,
            ),
            "hybrid",
        )

    def test_context34_preserves_order_and_other_contexts_sort(self) -> None:
        action = [2, 0, 1]
        self.assertIs(
            evaluator.apply_hybrid_action_order(
                action,
                select_context=34,
                enabled=True,
            ),
            action,
        )
        self.assertEqual(
            evaluator.apply_hybrid_action_order(
                action,
                select_context=7,
                enabled=True,
            ),
            [0, 1, 2],
        )
        self.assertIs(
            evaluator.apply_hybrid_action_order(
                action,
                select_context=7,
                enabled=False,
            ),
            action,
        )

    def test_candidate_and_opponent_cli_modes_are_independently_exclusive(
        self,
    ) -> None:
        base = ["--candidate", "candidate.pt", "--opponent", "opponent.pt"]
        defaults = evaluator.build_argument_parser().parse_args(base)
        self.assertFalse(defaults.candidate_canonical_order)
        self.assertFalse(defaults.candidate_hybrid_order)
        self.assertFalse(defaults.opponent_canonical_order)
        self.assertFalse(defaults.opponent_hybrid_order)
        independent = evaluator.build_argument_parser().parse_args(
            [
                *base,
                "--candidate-hybrid-order",
                "--opponent-canonical-order",
            ]
        )
        self.assertTrue(independent.candidate_hybrid_order)
        self.assertTrue(independent.opponent_canonical_order)
        for flags in (
            ["--candidate-canonical-order", "--candidate-hybrid-order"],
            ["--opponent-canonical-order", "--opponent-hybrid-order"],
        ):
            with self.subTest(flags=flags):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        evaluator.build_argument_parser().parse_args(
                            [*base, *flags]
                        )

    def test_evaluator_maps_each_raw_context_to_its_action_row(self) -> None:
        class FakeBattle:
            starts = 0
            stepped: list[tuple[int, list[int]]] = []
            acting_opponent = False

            def __init__(self, _deck0, _deck1) -> None:
                self.attempt = type(self).starts
                type(self).starts += 1
                self.context = 34 if self.attempt == 0 else 7
                self.seat = self.attempt
                self.observation = self.make_observation(-1)

            def make_observation(self, result: int) -> dict:
                return {
                    "current": {
                        "yourIndex": (
                            1 - self.seat
                            if type(self).acting_opponent
                            else self.seat
                        ),
                        "turn": 1,
                        "turnActionCount": 0,
                        "result": result,
                    },
                    "select": {
                        "type": 5,
                        "context": self.context,
                        "minCount": 3,
                        "maxCount": 3,
                        "option": [{}, {}, {}],
                    },
                    "search_begin_input": "opaque",
                }

            @property
            def result(self) -> int:
                return int(self.observation["current"]["result"])

            def step(self, action):
                type(self).stepped.append((self.context, list(action)))
                self.observation = self.make_observation(self.seat)
                return self.observation, 0

            def close(self) -> None:
                pass

        class DummyModel:
            def eval(self):
                return self

        def run(
            *,
            candidate_hybrid: bool,
            opponent_hybrid: bool = False,
            acting_opponent: bool = False,
        ) -> list[tuple[int, list[int]]]:
            FakeBattle.starts = 0
            FakeBattle.stepped = []
            FakeBattle.acting_opponent = acting_opponent
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(evaluator, "RawBattle", FakeBattle)
                )
                stack.enter_context(
                    mock.patch.object(
                        evaluator,
                        "live_feature",
                        side_effect=lambda *_args: {"feature": 1},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        evaluator,
                        "collate_features",
                        side_effect=lambda features, *_args: features,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        evaluator,
                        "model_forward",
                        side_effect=lambda *_args: None,
                    )
                )
                sampler = stack.enter_context(
                    mock.patch.object(
                        evaluator,
                        "sample_ordered_actions",
                        side_effect=lambda _outputs, batch, **_kwargs: (
                            [[2, 0, 1] for _ in batch],
                            None,
                            None,
                            None,
                        ),
                    )
                )
                evaluator.evaluate_head_to_head_with_seats(
                    DummyModel(),
                    DummyModel(),
                    [1] * 60,
                    {},
                    evaluator.torch.device("cpu"),
                    games_target=2,
                    environments=2,
                    max_game_decisions=2,
                    opponent_canonical_order=False,
                    current_hybrid_order=candidate_hybrid,
                    opponent_hybrid_order=opponent_hybrid,
                )
                self.assertFalse(
                    sampler.call_args.kwargs["canonicalize_order"]
                )
            return list(FakeBattle.stepped)

        self.assertEqual(
            run(candidate_hybrid=True),
            [(34, [2, 0, 1]), (7, [0, 1, 2])],
        )
        self.assertEqual(
            run(candidate_hybrid=False),
            [(34, [2, 0, 1]), (7, [2, 0, 1])],
        )
        self.assertEqual(
            run(
                candidate_hybrid=False,
                opponent_hybrid=True,
                acting_opponent=True,
            ),
            [(34, [2, 0, 1]), (7, [0, 1, 2])],
        )
        self.assertEqual(
            run(candidate_hybrid=True, acting_opponent=True),
            [(34, [2, 0, 1]), (7, [2, 0, 1])],
        )


class EvaluatorCandidateBCFallbackTest(unittest.TestCase):
    class NamedModel(evaluator.torch.nn.Module):
        def __init__(self, name: str, *, frozen: bool = False) -> None:
            super().__init__()
            self.name = name
            self.weight = evaluator.torch.nn.Parameter(
                evaluator.torch.ones(1),
                requires_grad=not frozen,
            )

    def test_threshold_routes_only_late_candidate_turns_and_sorts(self) -> None:
        class FakeBattle:
            starts = 0
            stepped: list[tuple[int, int, int, list[int]]] = []

            def __init__(self, _deck0, _deck1) -> None:
                self.seat = type(self).starts
                type(self).starts += 1
                self.phase = 0
                self.observation = self.make_observation(result=-1)

            def make_observation(self, *, result: int) -> dict:
                turns = (1, 3, 4)
                acting_seat = (
                    1 - self.seat if self.phase == 1 else self.seat
                )
                return {
                    "current": {
                        "yourIndex": acting_seat,
                        "turn": turns[min(self.phase, 2)],
                        "turnActionCount": self.phase,
                        "result": result,
                    },
                    "select": {
                        "type": 5,
                        "context": 7,
                        "minCount": 3,
                        "maxCount": 3,
                        "option": [{}, {}, {}],
                    },
                    "search_begin_input": "opaque",
                }

            @property
            def result(self) -> int:
                return int(self.observation["current"]["result"])

            def step(self, action):
                current = self.observation["current"]
                type(self).stepped.append(
                    (
                        self.seat,
                        int(current["yourIndex"]),
                        int(current["turn"]),
                        list(action),
                    )
                )
                self.phase += 1
                result = self.seat if self.phase == 3 else -1
                self.observation = self.make_observation(result=result)
                return self.observation, 0

            def close(self) -> None:
                pass

        candidate = self.NamedModel("candidate")
        opponent = self.NamedModel("opponent")
        fallback = self.NamedModel("fallback", frozen=True)

        def run(*, enabled: bool) -> tuple[dict, list[str], list[tuple[str, bool]]]:
            FakeBattle.starts = 0
            FakeBattle.stepped = []
            forward_models: list[str] = []
            sampler_modes: list[tuple[str, bool]] = []

            def forward(model, _batch, _device):
                forward_models.append(model.name)
                return {"model_name": model.name}

            def sample(outputs, batch, **kwargs):
                canonical = bool(kwargs["canonicalize_order"])
                sampler_modes.append((outputs["model_name"], canonical))
                action = [0, 1, 2] if canonical else [2, 0, 1]
                return ([list(action) for _ in batch], None, None, None)

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(evaluator, "RawBattle", FakeBattle)
                )
                stack.enter_context(
                    mock.patch.object(
                        evaluator,
                        "live_feature",
                        side_effect=lambda *_args: {"feature": 1},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        evaluator,
                        "collate_features",
                        side_effect=lambda features, *_args: features,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        evaluator,
                        "model_forward",
                        side_effect=forward,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        evaluator,
                        "sample_ordered_actions",
                        side_effect=sample,
                    )
                )
                result = evaluator.evaluate_head_to_head_with_seats(
                    candidate,
                    opponent,
                    [1] * 60,
                    {},
                    evaluator.torch.device("cpu"),
                    games_target=2,
                    environments=2,
                    max_game_decisions=4,
                    opponent_canonical_order=False,
                    candidate_bc_fallback_model=(fallback if enabled else None),
                    candidate_bc_fallback_after_turn=(3 if enabled else 0),
                )
            return result, forward_models, sampler_modes

        enabled_result, enabled_models, enabled_modes = run(enabled=True)
        self.assertEqual(enabled_models, ["candidate", "opponent", "fallback"])
        self.assertEqual(
            enabled_modes,
            [
                ("candidate", False),
                ("opponent", False),
                ("fallback", True),
            ],
        )
        self.assertEqual(
            FakeBattle.stepped,
            [
                (0, 0, 1, [2, 0, 1]),
                (1, 1, 1, [2, 0, 1]),
                (0, 1, 3, [2, 0, 1]),
                (1, 0, 3, [2, 0, 1]),
                (0, 0, 4, [0, 1, 2]),
                (1, 1, 4, [0, 1, 2]),
            ],
        )
        audit = enabled_result["candidate_bc_fallback"]
        self.assertEqual(audit["decisions"], 2)
        self.assertEqual(audit["by_candidate_seat"]["0"]["decisions"], 1)
        self.assertEqual(audit["by_candidate_seat"]["1"]["decisions"], 1)
        self.assertTrue(audit["seat_sum_verified"])

        disabled_result, disabled_models, disabled_modes = run(enabled=False)
        self.assertEqual(disabled_models, ["candidate", "opponent", "candidate"])
        self.assertEqual(
            disabled_modes,
            [
                ("candidate", False),
                ("opponent", False),
                ("candidate", False),
            ],
        )
        self.assertNotIn("candidate_bc_fallback", disabled_result)
        self.assertEqual(
            [row[3] for row in FakeBattle.stepped[-2:]],
            [[2, 0, 1], [2, 0, 1]],
        )

    def test_cli_and_direct_api_reject_invalid_fallback_contracts(self) -> None:
        base = ["--candidate", "candidate.pt", "--opponent", "opponent.pt"]
        parser = evaluator.build_argument_parser()
        self.assertEqual(parser.parse_args(base).candidate_bc_fallback_after_turn, 0)
        self.assertEqual(
            parser.parse_args(
                [*base, "--candidate-bc-fallback-after-turn", "3"]
            ).candidate_bc_fallback_after_turn,
            3,
        )
        for invalid in ("-1", "not-an-int"):
            with self.subTest(cli_invalid=invalid):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        evaluator.build_argument_parser().parse_args(
                            [
                                *base,
                                "--candidate-bc-fallback-after-turn",
                                invalid,
                            ]
                        )

        candidate = self.NamedModel("candidate")
        opponent = self.NamedModel("opponent")
        frozen = self.NamedModel("fallback", frozen=True)
        trainable = self.NamedModel("fallback")

        def call(**kwargs):
            return evaluator.evaluate_head_to_head_with_seats(
                candidate,
                opponent,
                [1] * 60,
                {},
                evaluator.torch.device("cpu"),
                games_target=1,
                environments=1,
                max_game_decisions=1,
                **kwargs,
            )

        with self.assertRaisesRegex(ValueError, "required"):
            call(candidate_bc_fallback_after_turn=3)
        with self.assertRaisesRegex(ValueError, "must be omitted"):
            call(candidate_bc_fallback_model=frozen)
        with self.assertRaisesRegex(ValueError, "must be frozen"):
            call(
                candidate_bc_fallback_model=trainable,
                candidate_bc_fallback_after_turn=3,
            )
        with self.assertRaisesRegex(ValueError, ">= 0"):
            call(candidate_bc_fallback_after_turn=-1)
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            call(candidate_bc_fallback_after_turn=True)


class GoldLeagueRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidate = self.root / "candidate.pt"
        self.candidate.write_bytes(b"candidate checkpoint")
        self.candidate_deck = self.root / "candidate.csv"
        self.candidate_deck.write_text("1\n", encoding="utf-8")
        self.bc = self.root / "bc.pt"
        self.bc.write_bytes(b"bc checkpoint")
        self.clone = self.root / "clone.pt"
        self.clone.write_bytes(b"clone checkpoint")
        self.clone_deck = self.root / "clone.csv"
        self.clone_deck.write_text("2\n", encoding="utf-8")
        self.deck_hash = "a" * 64
        self.manifest = self.root / "league_manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": runner.LEAGUE_SCHEMA,
                    "opponents": [
                        {
                            "policy_id": "gold/policy 1",
                            "submission_id": 42,
                            "team_name": "Gold Team",
                            "archetype": "Marnie",
                            "checkpoint": self.clone.name,
                            "checkpoint_sha256": sha256(self.clone),
                            "deck": self.clone_deck.name,
                            "deck_hash": self.deck_hash,
                            "canonical_order": True,
                            "quality": {"pass": True},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        _, opponents = runner.load_league_manifest(self.manifest)
        self.opponent = opponents[0]
        self.args = argparse.Namespace(
            league_manifest=self.manifest,
            deployment_contract=None,
            candidate=self.candidate,
            candidate_deck=self.candidate_deck,
            output_dir=self.root / "output",
            bc_checkpoint=self.bc,
            policy_id=[],
            archetype=[],
            policy_regex=None,
            max_policies=None,
            jobs=1,
            resume=False,
            dry_run=True,
            wall_seconds=None,
            screening_games=512,
            confirmation="if-screen-pass",
            confirmation_games=2048,
            environments=4,
            max_game_decisions=1000,
            candidate_raw_order=True,
            candidate_canonical_order=False,
            candidate_hybrid_order=False,
            opponent_canonical_order=False,
            seed=7,
            device="cpu",
            cvar_alpha=0.25,
            screen_min_policy_win_rate=0.45,
            screen_min_policy_wilson_low=0.40,
            screen_min_macro_win_rate=0.50,
            screen_min_cvar_win_rate=0.45,
            promotion_min_policies=1,
            promotion_min_policy_wilson_low=0.50,
            promotion_min_archetype_wilson_low=0.50,
            promotion_min_seat_wilson_low=0.45,
            promotion_min_macro_win_rate=0.55,
            promotion_min_policy_win_rate=0.50,
            promotion_min_cvar_win_rate=0.50,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_command_invokes_existing_evaluator_with_even_games(self) -> None:
        command = runner.build_eval_command(
            self.args,
            self.opponent,
            phase="screening",
            games=512,
            output=self.root / "result.json",
        )
        self.assertEqual(Path(command[1]), runner.EVALUATOR)
        self.assertEqual(command[command.index("--games") + 1], "512")
        self.assertEqual(
            command[command.index("--candidate") + 1],
            str(self.candidate),
        )
        self.assertEqual(
            command[command.index("--opponent") + 1],
            str(self.clone.resolve()),
        )
        self.assertIn("--opponent-canonical-order", command)

    def test_mixed_manifest_drives_per_opponent_commands_and_identity(self) -> None:
        ppo_clone = self.root / "ppo_clone.pt"
        ppo_clone.write_bytes(b"ppo clone checkpoint")
        mixed_manifest = self.root / "mixed_manifest.json"
        raw = json.loads(self.manifest.read_text(encoding="utf-8"))
        raw["opponents"].append(
            {
                "policy_id": "ppo/policy 2",
                "submission_id": None,
                "team_name": "PPO Team",
                "archetype": "Marnie",
                "checkpoint": ppo_clone.name,
                "checkpoint_sha256": sha256(ppo_clone),
                "deck": self.clone_deck.name,
                "deck_hash": self.deck_hash,
                "canonical_order": False,
            }
        )
        mixed_manifest.write_text(json.dumps(raw), encoding="utf-8")
        _, opponents = runner.load_league_manifest(mixed_manifest)

        commands = [
            runner.build_eval_command(
                self.args,
                opponent,
                phase="screening",
                games=512,
                output=self.root / f"result-{index}.json",
            )
            for index, opponent in enumerate(opponents)
        ]
        self.assertIn("--opponent-canonical-order", commands[0])
        self.assertNotIn("--opponent-canonical-order", commands[1])

        for opponent in opponents:
            result = make_eval_result(
                self.candidate,
                self.candidate_deck,
                opponent,
                games=512,
                seed=runner.derived_seed(
                    self.args.seed,
                    "screening",
                    opponent.policy_id,
                ),
                wins_by_seat=(150, 140),
            )
            runner.validate_evaluator_result(
                result,
                args=self.args,
                opponent=opponent,
                phase="screening",
                games=512,
                candidate_sha256=sha256(self.candidate),
            )
            result["opponent"]["canonical_order"] = int(
                opponent.canonical_order
            )
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                runner.validate_evaluator_result(
                    result,
                    args=self.args,
                    opponent=opponent,
                    phase="screening",
                    games=512,
                    candidate_sha256=sha256(self.candidate),
                )
            result["opponent"]["canonical_order"] = (
                not opponent.canonical_order
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                runner.validate_evaluator_result(
                    result,
                    args=self.args,
                    opponent=opponent,
                    phase="screening",
                    games=512,
                    candidate_sha256=sha256(self.candidate),
                )

    def test_manifest_requires_explicit_boolean_canonical_order(self) -> None:
        raw = json.loads(self.manifest.read_text(encoding="utf-8"))
        raw["opponents"][0].pop("canonical_order")
        self.manifest.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            runner.load_league_manifest(self.manifest)

        raw["opponents"][0]["canonical_order"] = 1
        self.manifest.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            runner.load_league_manifest(self.manifest)

    def test_legacy_global_true_flag_rejects_mixed_manifest_conflict(self) -> None:
        self.args.opponent_canonical_order = True
        false_raw = json.loads(self.manifest.read_text(encoding="utf-8"))
        false_raw["opponents"][0]["canonical_order"] = False
        self.manifest.write_text(json.dumps(false_raw), encoding="utf-8")
        _, opponents = runner.load_league_manifest(self.manifest)
        with self.assertRaisesRegex(ValueError, "deprecated all-true"):
            runner.build_eval_command(
                self.args,
                opponents[0],
                phase="screening",
                games=512,
                output=self.root / "result.json",
            )

    def test_dry_run_does_not_create_output_and_discloses_rng_limit(self) -> None:
        result = runner.run(self.args)
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["engine_seed_control"])
        self.assertIn("not reproducible", result["engine_randomness_warning"])
        self.assertEqual(len(result["screening_commands"]), 1)
        self.assertEqual(len(result["confirmation_commands"]), 1)
        self.assertEqual(result["candidate_order_mode"], "raw")
        self.assertFalse(self.args.output_dir.exists())

    def test_hybrid_candidate_mode_is_bound_to_command_and_result(self) -> None:
        self.args.candidate_raw_order = False
        self.args.candidate_hybrid_order = True
        command = runner.build_eval_command(
            self.args,
            self.opponent,
            phase="screening",
            games=512,
            output=self.root / "hybrid-result.json",
        )
        self.assertIn("--candidate-hybrid-order", command)
        self.assertNotIn("--candidate-canonical-order", command)

        result = make_eval_result(
            self.candidate,
            self.candidate_deck,
            self.opponent,
            games=512,
            seed=runner.derived_seed(7, "screening", self.opponent.policy_id),
            wins_by_seat=(150, 140),
            candidate_hybrid_order=True,
        )
        runner.validate_evaluator_result(
            result,
            args=self.args,
            opponent=self.opponent,
            phase="screening",
            games=512,
            candidate_sha256=sha256(self.candidate),
        )
        result["candidate"]["order_mode"] = "raw"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            runner.validate_evaluator_result(
                result,
                args=self.args,
                opponent=self.opponent,
                phase="screening",
                games=512,
                candidate_sha256=sha256(self.candidate),
            )

    def test_deployment_contract_binds_manifest_config_and_output(self) -> None:
        contract = self.root / "deployment.json"
        value = {
            "schema_version": runner.DEPLOYMENT_SCHEMA,
            "candidate": {
                "path": str(self.candidate.resolve()),
                "sha256": sha256(self.candidate),
            },
            "candidate_deck": {
                "path": str(self.candidate_deck.resolve()),
                "file_sha256": sha256(self.candidate_deck),
            },
            "action_order": {
                "mode": "raw",
                "canonical_order": False,
                "hybrid_order": False,
            },
            "panel": {
                "manifest": str(self.manifest.resolve()),
                "manifest_sha256": sha256(self.manifest),
                "output_dir": str(self.args.output_dir.resolve()),
            },
        }
        contract.write_text(json.dumps(value), encoding="utf-8")
        self.args.deployment_contract = contract
        plan = runner.run(self.args)
        self.assertTrue(plan["dry_run"])

        value["action_order"]["mode"] = "hybrid"
        contract.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Deployment contract mismatch"):
            runner.run(self.args)

    def test_candidate_order_mode_must_be_explicit_and_unique(self) -> None:
        self.args.candidate_raw_order = False
        with self.assertRaisesRegex(ValueError, "Exactly one explicit"):
            runner.validate_args(self.args)
        self.args.candidate_hybrid_order = True
        self.args.candidate_canonical_order = True
        with self.assertRaisesRegex(ValueError, "Exactly one explicit"):
            runner.validate_args(self.args)

    def test_evaluator_result_requires_exact_two_seat_accounting(self) -> None:
        seed = runner.derived_seed(7, "screening", self.opponent.policy_id)
        result = make_eval_result(
            self.candidate,
            self.candidate_deck,
            self.opponent,
            games=512,
            seed=seed,
            wins_by_seat=(150, 140),
        )
        runner.validate_evaluator_result(
            result,
            args=self.args,
            opponent=self.opponent,
            phase="screening",
            games=512,
            candidate_sha256=sha256(self.candidate),
        )
        result["evaluation"]["by_candidate_seat"]["1"]["valid_games"] = 255
        with self.assertRaisesRegex(ValueError, "not exactly balanced"):
            runner.validate_evaluator_result(
                result,
                args=self.args,
                opponent=self.opponent,
                phase="screening",
                games=512,
                candidate_sha256=sha256(self.candidate),
            )

    def test_phase_summary_reports_policy_archetype_seat_and_cvar(self) -> None:
        seed = runner.derived_seed(7, "screening", self.opponent.policy_id)
        result = make_eval_result(
            self.candidate,
            self.candidate_deck,
            self.opponent,
            games=512,
            seed=seed,
            wins_by_seat=(160, 144),
        )
        summary = runner.aggregate_phase(
            [self.opponent],
            {self.opponent.policy_id: result},
            phase="screening",
            games_requested_per_policy=512,
            cvar_alpha=0.25,
        )
        self.assertTrue(summary["coverage_complete"])
        self.assertTrue(summary["strict_even_seat_balance_verified"])
        self.assertEqual(summary["policies"][0]["wins"], 304)
        self.assertEqual(summary["archetypes"][0]["wins"], 304)
        self.assertEqual(
            summary["archetypes"][0]["by_candidate_seat"]["1"]["games"],
            256,
        )
        self.assertEqual(
            summary["by_candidate_seat"]["0"]["games"],
            256,
        )
        self.assertEqual(summary["policy_win_rate_cvar"]["count"], 1)
        self.assertAlmostEqual(
            summary["macro_policy_win_rate"],
            304 / 512,
        )

    def test_promotion_gates_are_explicit(self) -> None:
        seed = runner.derived_seed(7, "screening", self.opponent.policy_id)
        result = make_eval_result(
            self.candidate,
            self.candidate_deck,
            self.opponent,
            games=512,
            seed=seed,
            wins_by_seat=(180, 180),
        )
        summary = runner.aggregate_phase(
            [self.opponent],
            {self.opponent.policy_id: result},
            phase="screening",
            games_requested_per_policy=512,
            cvar_alpha=0.25,
        )
        promotion = runner.promotion_gate(
            summary,
            self.args,
            confirmation_required=False,
            confirmation_complete=False,
        )
        expected = {
            "minimum_policy_count",
            "coverage_complete",
            "strict_even_seat_balance",
            "minimum_policy_wilson_95_low",
            "minimum_archetype_wilson_95_low",
            "minimum_policy_seat_wilson_95_low",
            "macro_policy_win_rate",
            "minimum_policy_win_rate",
            "policy_win_rate_cvar",
        }
        self.assertTrue(expected <= set(promotion["gates"]))
        self.assertTrue(promotion["promote"])

    def test_odd_game_counts_are_rejected(self) -> None:
        self.args.screening_games = 511
        with self.assertRaisesRegex(ValueError, "strict seat balance"):
            runner.validate_args(self.args)


if __name__ == "__main__":
    unittest.main()
