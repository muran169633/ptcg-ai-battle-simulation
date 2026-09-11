from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import train_bc_orbit  # noqa: E402
import train_ppo  # noqa: E402


class ResumeLearnerInitializationTests(unittest.TestCase):
    @staticmethod
    def model(seed: int) -> train_bc_orbit.EntityOptionPolicy:
        torch.manual_seed(seed)
        return train_bc_orbit.EntityOptionPolicy(
            hash_size=128,
            categorical_dim=8,
            model_dim=16,
            layers=2,
            heads=4,
            dropout=0.0,
            max_state_entities=8,
        )

    def test_resume_source_preserves_legacy_full_checkpoint_load(self) -> None:
        learner = self.model(1)
        resumed = self.model(2)
        checkpoint = {"model_state_dict": resumed.state_dict()}

        source = train_ppo.apply_resume_learner_weights(
            learner,
            checkpoint,
            weight_source="resume",
            reset_optimizer_on_resume=False,
        )

        self.assertEqual(source, "resume_checkpoint")
        for name, tensor in learner.state_dict().items():
            self.assertTrue(
                torch.equal(tensor, checkpoint["model_state_dict"][name])
            )

    def test_bc_source_keeps_bc_weights_and_requires_optimizer_reset(
        self,
    ) -> None:
        learner = self.model(3)
        resumed = self.model(4)
        before = {
            name: tensor.clone()
            for name, tensor in learner.state_dict().items()
        }
        checkpoint = {"model_state_dict": resumed.state_dict()}

        with self.assertRaisesRegex(ValueError, "requires.*reset-optimizer"):
            train_ppo.apply_resume_learner_weights(
                learner,
                checkpoint,
                weight_source="bc",
                reset_optimizer_on_resume=False,
            )
        source = train_ppo.apply_resume_learner_weights(
            learner,
            checkpoint,
            weight_source="bc",
            reset_optimizer_on_resume=True,
        )

        self.assertEqual(source, "bc_checkpoint")
        for name, tensor in learner.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]))

    def test_bc_source_without_resume_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            train_ppo.validate_resume_learner_weight_selection(
                None,
                "bc",
                reset_optimizer_on_resume=True,
            )

    def test_resume_schema_is_checked_even_when_bc_weights_are_selected(
        self,
    ) -> None:
        learner = self.model(5)
        malformed = dict(self.model(6).state_dict())
        malformed.pop(next(iter(malformed)))

        with self.assertRaisesRegex(ValueError, "incompatible"):
            train_ppo.validate_checkpoint_model_state_schema(
                {"model_state_dict": malformed},
                learner.state_dict(),
                Path("resume.pt"),
            )

    def test_model_state_hash_tracks_effective_weights(self) -> None:
        first = self.model(7)
        second = self.model(7)
        self.assertEqual(
            train_ppo.model_state_sha256(first),
            train_ppo.model_state_sha256(second),
        )
        with torch.no_grad():
            next(second.parameters()).add_(1.0)
        self.assertNotEqual(
            train_ppo.model_state_sha256(first),
            train_ppo.model_state_sha256(second),
        )


class PhaseScheduleTests(unittest.TestCase):
    def test_continuation_progress_uses_explicit_phase_start(self) -> None:
        config = SimpleNamespace(
            updates=450,
            schedule_start_update=441,
        )
        self.assertEqual(train_ppo.phase_schedule_progress(config, 440), 0.0)
        self.assertEqual(train_ppo.phase_schedule_progress(config, 441), 0.0)
        self.assertAlmostEqual(
            train_ppo.phase_schedule_progress(config, 445),
            4.0 / 9.0,
        )
        self.assertEqual(train_ppo.phase_schedule_progress(config, 450), 1.0)
        self.assertEqual(train_ppo.phase_schedule_progress(config, 451), 1.0)

    def test_single_update_phase_starts_at_zero_progress(self) -> None:
        config = SimpleNamespace(
            updates=441,
            schedule_start_update=441,
        )
        self.assertEqual(train_ppo.phase_schedule_progress(config, 441), 0.0)

    def test_kl_schedule_supports_constant_and_disabled_anchor(self) -> None:
        constant = SimpleNamespace(
            updates=450,
            schedule_start_update=441,
            bc_kl_start=0.002,
            bc_kl_end=0.002,
        )
        disabled = SimpleNamespace(
            updates=450,
            schedule_start_update=441,
            bc_kl_start=0.0,
            bc_kl_end=0.0,
        )
        for update in (441, 445, 450):
            self.assertAlmostEqual(
                train_ppo.bc_kl_coefficient(constant, update),
                0.002,
            )
            self.assertEqual(
                train_ppo.bc_kl_coefficient(disabled, update),
                0.0,
            )


class OpponentSamplingTests(unittest.TestCase):
    def test_explicit_zero_weight_is_never_sampled(self) -> None:
        random.seed(7)
        draws = [
            train_ppo.sample_opponent_index(
                2,
                bc_opponent_probability=None,
                sampling_weights=[0.0, 1.0],
            )
            for _ in range(100)
        ]
        self.assertEqual(set(draws), {1})

    def test_explicit_weights_follow_expected_ratio(self) -> None:
        random.seed(11)
        draws = [
            train_ppo.sample_opponent_index(
                2,
                bc_opponent_probability=None,
                sampling_weights=[1.0, 3.0],
            )
            for _ in range(20_000)
        ]
        fraction_second = sum(index == 1 for index in draws) / len(draws)
        self.assertAlmostEqual(fraction_second, 0.75, delta=0.015)

    def test_named_and_history_weights_resolve_separately(self) -> None:
        opponents = [
            SimpleNamespace(name="bc", permanent=True),
            SimpleNamespace(name="v3", permanent=True),
            SimpleNamespace(name="update-0450", permanent=False),
            SimpleNamespace(name="update-0460", permanent=False),
        ]
        weights = train_ppo.resolve_opponent_sampling_weights(
            opponents,
            {"bc": 10.0, "v3": 60.0},
            history_weight=30.0,
        )
        self.assertEqual(weights, [10.0, 60.0, 15.0, 15.0])

    def test_unknown_named_weight_is_rejected(self) -> None:
        opponents = [SimpleNamespace(name="bc", permanent=True)]
        with self.assertRaisesRegex(ValueError, "not match"):
            train_ppo.resolve_opponent_sampling_weights(
                opponents,
                {"missing": 1.0},
                history_weight=0.0,
            )

    def test_recent_meta_sampling_keeps_selfplay_external_weights_conditional(self) -> None:
        controller = train_ppo.OpponentSamplingReweighter(
            opponent_names=["popular", "rare_hard"],
            meta_weights={"popular": 3.0, "rare_hard": 1.0},
            window_games=4,
            inverse_min_factor=0.5,
            inverse_max_factor=2.5,
        )
        opponents = [
            SimpleNamespace(name="popular", permanent=True),
            SimpleNamespace(name="rare_hard", permanent=True),
            SimpleNamespace(name="history", permanent=False),
        ]
        initial = controller.sampling_weights(opponents)
        self.assertEqual(initial, [3.0, 1.0, 0.0])

        controller.observe(
            {
                "popular": ["win"] * 4,
                "rare_hard": ["loss"] * 4,
            }
        )
        audit = controller.audit()
        self.assertGreater(
            audit["rare_hard"]["sampling_probability_given_frozen_opponent"],
            0.25,
        )
        controller.observe(
            {
                "popular": ["loss"] * 5,
                "rare_hard": [],
            }
        )
        self.assertEqual(controller.outcome_window["popular"], [0.0] * 4)

        restored = train_ppo.OpponentSamplingReweighter(
            opponent_names=["popular", "rare_hard"],
            meta_weights={"popular": 3.0, "rare_hard": 1.0},
            window_games=4,
            inverse_min_factor=0.5,
            inverse_max_factor=2.5,
        )
        restored.load_state_dict(controller.state_dict())
        self.assertEqual(restored.state_dict(), controller.state_dict())


class ChampionGateTests(unittest.TestCase):
    def test_rejected_candidate_rolls_back_weights_and_optimizer(self) -> None:
        torch.manual_seed(7)
        champion = torch.nn.Linear(4, 3)
        current = copy.deepcopy(champion)
        optimizer = torch.optim.AdamW(current.parameters(), lr=1e-3)
        loss = current(torch.ones(2, 4)).sum()
        loss.backward()
        optimizer.step()
        self.assertTrue(optimizer.state)
        self.assertTrue(
            any(
                not torch.equal(current.state_dict()[name], tensor)
                for name, tensor in champion.state_dict().items()
            )
        )

        audit = train_ppo.rollback_to_champion(
            current,
            champion,
            optimizer,
            None,
            reset_optimizer=True,
        )

        self.assertTrue(audit["applied"])
        self.assertTrue(audit["optimizer_reset"])
        self.assertFalse(optimizer.state)
        for name, tensor in champion.state_dict().items():
            self.assertTrue(torch.equal(current.state_dict()[name], tensor))

    def test_rejected_candidate_can_preserve_optimizer_moments(self) -> None:
        champion = torch.nn.Linear(2, 1)
        current = copy.deepcopy(champion)
        optimizer = torch.optim.AdamW(current.parameters(), lr=1e-3)
        current(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        state_count = len(optimizer.state)

        audit = train_ppo.rollback_to_champion(
            current,
            champion,
            optimizer,
            None,
            reset_optimizer=False,
        )

        self.assertFalse(audit["optimizer_reset"])
        self.assertEqual(len(optimizer.state), state_count)

    def test_threshold_is_strict(self) -> None:
        evaluation = {"valid_games": 200, "win_rate": 0.58}
        self.assertFalse(
            train_ppo.champion_gate_promotes(evaluation, 0.58)
        )

    def test_candidate_above_threshold_promotes(self) -> None:
        evaluation = {"valid_games": 200, "win_rate": 0.585}
        self.assertTrue(
            train_ppo.champion_gate_promotes(evaluation, 0.58)
        )

    def test_inclusive_threshold_promotes_equality(self) -> None:
        evaluation = {"valid_games": 200, "win_rate": 0.54}
        self.assertTrue(
            train_ppo.champion_gate_promotes(
                evaluation,
                0.54,
                inclusive=True,
            )
        )

    def test_inclusive_threshold_rejects_below_threshold(self) -> None:
        evaluation = {"valid_games": 200, "win_rate": 0.535}
        self.assertFalse(
            train_ppo.champion_gate_promotes(
                evaluation,
                0.54,
                inclusive=True,
            )
        )

    def test_combined_panel_uses_game_counts(self) -> None:
        combined = train_ppo.combine_head_to_head_evaluations(
            [
                {
                    "valid_games": 200,
                    "wins": 108,
                    "losses": 92,
                    "draws": 0,
                    "invalid_games": 0,
                    "mean_decisions": 100.0,
                    "max_active_games": 64,
                    "seconds": 10.0,
                },
                {
                    "valid_games": 200,
                    "wins": 110,
                    "losses": 90,
                    "draws": 0,
                    "invalid_games": 1,
                    "mean_decisions": 200.0,
                    "max_active_games": 32,
                    "seconds": 20.0,
                },
            ]
        )
        self.assertEqual(combined["valid_games"], 400)
        self.assertEqual(combined["wins"], 218)
        self.assertAlmostEqual(combined["win_rate"], 0.545)
        self.assertAlmostEqual(combined["mean_decisions"], 150.0)
        self.assertEqual(combined["invalid_games"], 1)
        self.assertEqual(combined["max_active_games"], 64)
        self.assertEqual(combined["seconds"], 30.0)

    def test_empty_evaluation_never_promotes(self) -> None:
        evaluation = {"valid_games": 0, "win_rate": 1.0}
        self.assertFalse(
            train_ppo.champion_gate_promotes(evaluation, 0.58)
        )


class OpponentQuotaTests(unittest.TestCase):
    @staticmethod
    def adaptive_controller(
        *,
        seed: int = 17,
        refresh_updates: int = 3,
    ) -> train_ppo.OpponentQuotaController:
        return train_ppo.OpponentQuotaController(
            mode="adaptive",
            opponent_names=["bc", "v3", "kang"],
            base_quotas={"bc": 1, "v3": 1, "kang": 1},
            caps={"bc": 9, "v3": 9, "kang": 9},
            initial_audit={
                "bc": {"wins": 0, "losses": 0, "draws": 0},
                "v3": {"wins": 0, "losses": 0, "draws": 0},
                "kang": {"wins": 0, "losses": 0, "draws": 0},
            },
            games_per_update=9,
            refresh_updates=refresh_updates,
            seed=seed,
        )

    @staticmethod
    def outcomes(
        plan: dict[str, int],
        losing_opponent: str = "bc",
    ) -> dict[str, dict[str, int]]:
        return {
            name: {
                "games": games,
                "wins": 0 if name == losing_opponent else games,
                "losses": games if name == losing_opponent else 0,
                "draws": 0,
            }
            for name, games in plan.items()
        }

    def test_largest_remainder_is_stable_and_honors_caps(self) -> None:
        self.assertEqual(
            train_ppo.stable_largest_remainder(
                5,
                [1.0, 1.0, 1.0],
                [5, 5, 5],
            ),
            [2, 2, 1],
        )
        self.assertEqual(
            train_ppo.stable_largest_remainder(
                6,
                [10.0, 1.0, 1.0],
                [2, 6, 6],
            ),
            [2, 2, 2],
        )

    def test_fixed_quota_requires_exact_total(self) -> None:
        with self.assertRaisesRegex(ValueError, "must sum"):
            train_ppo.OpponentQuotaController(
                mode="fixed",
                opponent_names=["bc", "v3"],
                base_quotas={"bc": 3, "v3": 4},
                caps={},
                initial_audit={},
                games_per_update=8,
                refresh_updates=3,
                seed=1,
            )

    def test_schedule_is_deterministic_without_touching_global_rng(self) -> None:
        first = self.adaptive_controller(seed=29)
        second = self.adaptive_controller(seed=29)
        random.seed(12345)
        global_state = random.getstate()
        first_schedule, first_audit = first.begin_update(441)
        self.assertEqual(random.getstate(), global_state)
        second_schedule, second_audit = second.begin_update(441)
        self.assertEqual(first_schedule, second_schedule)
        self.assertEqual(
            first_audit["schedule_seed"],
            second_audit["schedule_seed"],
        )
        self.assertEqual(
            list(first_audit["planned_quotas"]),
            ["bc", "v3", "kang"],
        )

    def test_adaptive_quota_refreshes_every_three_updates(self) -> None:
        controller = self.adaptive_controller()
        first_schedule, first = controller.begin_update(441)
        self.assertEqual(len(first_schedule), 9)
        first_plan = first["planned_quotas"]
        controller.finish_update(
            441,
            first,
            self.outcomes(first_plan),
            {},
        )
        for update in (442, 443):
            _, audit = controller.begin_update(update)
            self.assertFalse(audit["recomputed"])
            self.assertEqual(audit["planned_quotas"], first_plan)
            controller.finish_update(
                update,
                audit,
                self.outcomes(audit["planned_quotas"]),
                {},
            )
        _, refreshed = controller.begin_update(444)
        self.assertTrue(refreshed["recomputed"])
        self.assertGreater(
            refreshed["planned_quotas"]["bc"],
            refreshed["planned_quotas"]["v3"],
        )
        self.assertEqual(sum(refreshed["planned_quotas"].values()), 9)

    def test_recent_meta_inverse_window_multiplies_base_frequency(self) -> None:
        controller = train_ppo.OpponentQuotaController(
            mode="meta_inverse_window",
            opponent_names=["popular", "rare_hard"],
            base_quotas={},
            caps={},
            initial_audit={},
            games_per_update=12,
            refresh_updates=1,
            seed=31,
            meta_weights={"popular": 3.0, "rare_hard": 1.0},
            win_rate_window_games=200,
            inverse_min_factor=0.5,
            inverse_max_factor=2.5,
        )
        _, first = controller.begin_update(1)
        self.assertEqual(first["planned_quotas"], {"popular": 9, "rare_hard": 3})
        controller.finish_update(
            1,
            first,
            {
                "popular": {"games": 9, "wins": 9, "losses": 0, "draws": 0},
                "rare_hard": {"games": 3, "wins": 0, "losses": 3, "draws": 0},
            },
            {},
            outcome_sequences={
                "popular": ["win"] * 9,
                "rare_hard": ["loss"] * 3,
            },
        )

        _, second = controller.begin_update(2)
        self.assertGreater(second["planned_quotas"]["rare_hard"], 3)
        meta = second["allocation_model"]["meta_window"]
        self.assertAlmostEqual(meta["popular"]["meta_weight"], 3.0)
        self.assertLess(
            meta["popular"]["clipped_inverse_factor"],
            meta["rare_hard"]["clipped_inverse_factor"],
        )
        self.assertEqual(sum(second["planned_quotas"].values()), 12)

        restored = train_ppo.OpponentQuotaController(
            mode="meta_inverse_window",
            opponent_names=["popular", "rare_hard"],
            base_quotas={},
            caps={},
            initial_audit={},
            games_per_update=12,
            refresh_updates=1,
            seed=31,
            meta_weights={"popular": 3.0, "rare_hard": 1.0},
            win_rate_window_games=200,
            inverse_min_factor=0.5,
            inverse_max_factor=2.5,
        )
        restored.load_state_dict(controller.state_dict())
        self.assertEqual(restored.state_dict(), controller.state_dict())

    def test_meta_window_rejects_aggregate_sequence_mismatch(self) -> None:
        controller = train_ppo.OpponentQuotaController(
            mode="meta_inverse_window",
            opponent_names=["only"],
            base_quotas={},
            caps={},
            initial_audit={},
            games_per_update=2,
            refresh_updates=1,
            seed=1,
            meta_weights={"only": 1.0},
        )
        _, audit = controller.begin_update(1)
        with self.assertRaisesRegex(RuntimeError, "sequence mismatch"):
            controller.finish_update(
                1,
                audit,
                {
                    "only": {
                        "games": 2,
                        "wins": 1,
                        "losses": 1,
                        "draws": 0,
                    }
                },
                {},
                outcome_sequences={"only": ["win", "win"]},
            )
        self.assertEqual(
            controller.observed["only"],
            {"wins": 0, "losses": 0, "draws": 0},
        )

    def test_invalid_replacement_keeps_same_opponent(self) -> None:
        schedule = train_ppo.ExactQuotaSchedule([0, 1, 2])
        first = schedule.peek()
        self.assertEqual(first, (0, "planned"))
        assert first is not None
        schedule.mark_started(first[1])
        schedule.add_replacement(first[0])
        retry = schedule.peek()
        self.assertEqual(retry, (0, "replacement"))
        assert retry is not None
        schedule.mark_started(retry[1])
        self.assertEqual(schedule.peek(), (1, "planned"))

    def test_even_96_quota_plan_is_exactly_seat_balanced(self) -> None:
        opponent_names = (
            ["bc"] * 32
            + ["v1"] * 4
            + ["v3"] * 48
            + ["v4"] * 4
            + ["dragapult"] * 4
            + ["kangaskhan"] * 4
        )
        random.Random(918).shuffle(opponent_names)
        random.seed(12345)
        global_state = random.getstate()
        seats, audit = train_ppo.balanced_quota_learner_seats(
            opponent_names,
            seed=20261021,
            update=456,
        )
        self.assertEqual(random.getstate(), global_state)
        self.assertEqual(len(seats), 96)
        self.assertEqual(audit["bc"], {"0": 16, "1": 16})
        self.assertEqual(audit["v3"], {"0": 24, "1": 24})
        for name in ("v1", "v4", "dragapult", "kangaskhan"):
            self.assertEqual(audit[name], {"0": 2, "1": 2})
        self.assertEqual(sum(seat == 0 for seat in seats), 48)
        self.assertEqual(sum(seat == 1 for seat in seats), 48)
        repeated, repeated_audit = train_ppo.balanced_quota_learner_seats(
            opponent_names,
            seed=20261021,
            update=456,
        )
        self.assertEqual(repeated, seats)
        self.assertEqual(repeated_audit, audit)

    def test_invalid_replacement_preserves_opponent_and_seat(self) -> None:
        schedule = train_ppo.ExactQuotaSchedule([0, 1], [1, 0])
        first = schedule.peek_claim()
        self.assertEqual(
            first,
            (train_ppo.ExactQuotaClaim(0, 1), "planned"),
        )
        assert first is not None
        schedule.mark_started(first[1])
        schedule.add_replacement(0, 1)
        self.assertEqual(
            schedule.peek_claim(),
            (train_ppo.ExactQuotaClaim(0, 1), "replacement"),
        )

    def test_start_failure_does_not_consume_claim(self) -> None:
        schedule = train_ppo.ExactQuotaSchedule([3], [1])
        before = schedule.peek_claim()
        # BattleStart failure deliberately does not call mark_started().
        after = schedule.peek_claim()
        self.assertEqual(before, after)
        self.assertEqual(schedule.remaining, 1)

    @staticmethod
    def fixed_seat_controller() -> train_ppo.OpponentQuotaController:
        return train_ppo.OpponentQuotaController(
            mode="fixed",
            opponent_names=["bc", "v3"],
            base_quotas={"bc": 4, "v3": 4},
            caps={},
            initial_audit={},
            games_per_update=8,
            refresh_updates=3,
            seed=31,
        )

    @staticmethod
    def balanced_seat_outcomes() -> dict[str, dict[str, int]]:
        return {
            "bc": {
                "games": 4,
                "wins": 2,
                "losses": 2,
                "draws": 0,
                "seat_0_games": 2,
                "seat_0_wins": 1,
                "seat_0_losses": 1,
                "seat_0_draws": 0,
                "seat_1_games": 2,
                "seat_1_wins": 1,
                "seat_1_losses": 1,
                "seat_1_draws": 0,
            },
            "v3": {
                "games": 4,
                "wins": 4,
                "losses": 0,
                "draws": 0,
                "seat_0_games": 2,
                "seat_0_wins": 2,
                "seat_0_losses": 0,
                "seat_0_draws": 0,
                "seat_1_games": 2,
                "seat_1_wins": 2,
                "seat_1_losses": 0,
                "seat_1_draws": 0,
            },
        }

    def test_finish_update_audits_exact_seat_counts(self) -> None:
        controller = self.fixed_seat_controller()
        schedule_names, audit = controller.begin_update(456)
        _, planned_seats = train_ppo.balanced_quota_learner_seats(
            schedule_names,
            seed=31,
            update=456,
        )
        finished = controller.finish_update(
            456,
            audit,
            self.balanced_seat_outcomes(),
            {"bc": 2, "v3": 0},
            planned_seat_quotas=planned_seats,
            invalid_replacements_by_seat={
                "bc": {"0": 1, "1": 1},
                "v3": {"0": 0, "1": 0},
            },
        )
        self.assertTrue(finished["seat_balance_verified"])
        self.assertEqual(finished["planned_max_seat_gap"], 0)
        self.assertEqual(finished["actual_max_seat_gap"], 0)
        self.assertEqual(
            finished["actual_seat_quotas"]["bc"],
            {"0": 2, "1": 2},
        )

    def test_finish_update_rejects_wrong_seats_atomically(self) -> None:
        controller = self.fixed_seat_controller()
        schedule_names, audit = controller.begin_update(456)
        _, planned_seats = train_ppo.balanced_quota_learner_seats(
            schedule_names,
            seed=31,
            update=456,
        )
        outcomes = self.balanced_seat_outcomes()
        outcomes["bc"].update(
            {
                "seat_0_games": 3,
                "seat_0_wins": 2,
                "seat_0_losses": 1,
                "seat_1_games": 1,
                "seat_1_wins": 0,
                "seat_1_losses": 1,
            }
        )
        observed_before = {
            name: dict(row)
            for name, row in controller.observed.items()
        }
        with self.assertRaisesRegex(RuntimeError, "seat quota mismatch"):
            controller.finish_update(
                456,
                audit,
                outcomes,
                {},
                planned_seat_quotas=planned_seats,
                invalid_replacements_by_seat={},
            )
        self.assertEqual(controller.observed, observed_before)

    def test_finish_update_rejects_seat_outcome_aggregate_mismatch(self) -> None:
        controller = self.fixed_seat_controller()
        schedule_names, audit = controller.begin_update(456)
        _, planned_seats = train_ppo.balanced_quota_learner_seats(
            schedule_names,
            seed=31,
            update=456,
        )
        outcomes = self.balanced_seat_outcomes()
        outcomes["bc"]["seat_0_wins"] = 2
        outcomes["bc"]["seat_0_losses"] = 0
        with self.assertRaisesRegex(RuntimeError, "do not aggregate"):
            controller.finish_update(
                456,
                audit,
                outcomes,
                {},
                planned_seat_quotas=planned_seats,
                invalid_replacements_by_seat={},
            )

    def test_finish_update_rejects_replacement_seat_mismatch(self) -> None:
        controller = self.fixed_seat_controller()
        schedule_names, audit = controller.begin_update(456)
        _, planned_seats = train_ppo.balanced_quota_learner_seats(
            schedule_names,
            seed=31,
            update=456,
        )
        with self.assertRaisesRegex(RuntimeError, "replacement seat mismatch"):
            controller.finish_update(
                456,
                audit,
                self.balanced_seat_outcomes(),
                {"bc": 2},
                planned_seat_quotas=planned_seats,
                invalid_replacements_by_seat={
                    "bc": {"0": 1, "1": 0},
                },
            )

    def test_checkpoint_state_round_trip_preserves_posterior_and_plan(self) -> None:
        source = self.adaptive_controller()
        _, audit = source.begin_update(441)
        source.finish_update(
            441,
            audit,
            self.outcomes(audit["planned_quotas"], losing_opponent="kang"),
            {"kang": 2},
        )
        restored = self.adaptive_controller()
        restored.load_state_dict(source.state_dict())
        self.assertEqual(restored.state_dict(), source.state_dict())
        _, next_audit = restored.begin_update(442)
        self.assertFalse(next_audit["recomputed"])

    def test_adaptive_builder_requires_audit_for_every_permanent(self) -> None:
        config = SimpleNamespace(
            opponent_quota_mode="adaptive",
            opponent_base_quotas={"bc": 1},
            opponent_caps={"bc": 4, "v3": 4},
            opponent_audit={
                "bc": {"wins": 1, "losses": 1, "draws": 0},
            },
            games_per_update=6,
            opponent_quota_refresh_updates=3,
            seed=1,
        )
        opponents = [
            SimpleNamespace(name="bc", permanent=True),
            SimpleNamespace(name="v3", permanent=True),
        ]
        with self.assertRaisesRegex(ValueError, "audits must cover"):
            train_ppo.build_opponent_quota_controller(config, opponents)

    def test_legacy_checkpoint_can_start_quota_phase_with_optimizer_reset(
        self,
    ) -> None:
        train_ppo.validate_opponent_quota_resume_transition(
            current_mode="adaptive",
            checkpoint_mode="legacy",
            checkpoint_state=None,
            reset_optimizer=True,
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            train_ppo.validate_opponent_quota_resume_transition(
                current_mode="adaptive",
                checkpoint_mode="legacy",
                checkpoint_state=None,
                reset_optimizer=False,
            )

    def test_exact_checkpoint_without_quota_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks"):
            train_ppo.validate_opponent_quota_resume_transition(
                current_mode="adaptive",
                checkpoint_mode="adaptive",
                checkpoint_state=None,
                reset_optimizer=True,
            )

    def test_slim_exact_checkpoint_can_start_legacy_with_optimizer_reset(
        self,
    ) -> None:
        train_ppo.validate_opponent_quota_resume_transition(
            current_mode="legacy",
            checkpoint_mode="fixed",
            checkpoint_state=None,
            reset_optimizer=True,
        )
        with self.assertRaisesRegex(ValueError, "reset-optimizer"):
            train_ppo.validate_opponent_quota_resume_transition(
                current_mode="legacy",
                checkpoint_mode="fixed",
                checkpoint_state=None,
                reset_optimizer=False,
            )

    def test_exact_mode_change_requires_reset_and_starts_fresh_quota_state(
        self,
    ) -> None:
        source = self.adaptive_controller()
        _, audit = source.begin_update(441)
        source.finish_update(
            441,
            audit,
            self.outcomes(audit["planned_quotas"]),
            {},
        )
        state = source.state_dict()
        with self.assertRaisesRegex(ValueError, "Changing exact"):
            train_ppo.validate_opponent_quota_resume_transition(
                current_mode="fixed",
                checkpoint_mode="adaptive",
                checkpoint_state=state,
                reset_optimizer=True,
                reset_quota_state=False,
            )
        train_ppo.validate_opponent_quota_resume_transition(
            current_mode="fixed",
            checkpoint_mode="adaptive",
            checkpoint_state=state,
            reset_optimizer=True,
            reset_quota_state=True,
        )

        config = SimpleNamespace(
            opponent_quota_mode="fixed",
            opponent_base_quotas={"bc": 4, "v3": 3, "kang": 2},
            opponent_caps={},
            opponent_audit={},
            games_per_update=9,
            opponent_quota_refresh_updates=3,
            seed=23,
        )
        opponents = [
            SimpleNamespace(name="bc", permanent=True),
            SimpleNamespace(name="v3", permanent=True),
            SimpleNamespace(name="kang", permanent=True),
        ]
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            train_ppo.build_opponent_quota_controller(
                config,
                opponents,
                state,
            )
        reset = train_ppo.build_opponent_quota_controller(
            config,
            opponents,
            state,
            reset_resume_state=True,
        )
        self.assertIsNotNone(reset)
        assert reset is not None
        self.assertTrue(reset.resume_state_reset)
        self.assertFalse(reset.resume_state_loaded)
        self.assertEqual(
            reset.observed,
            {
                "bc": {"wins": 0, "losses": 0, "draws": 0},
                "v3": {"wins": 0, "losses": 0, "draws": 0},
                "kang": {"wins": 0, "losses": 0, "draws": 0},
            },
        )


class RolloutBudgetTests(unittest.TestCase):
    def test_replacement_respects_active_games(self) -> None:
        self.assertTrue(train_ppo.needs_rollout_replacement(32, 31, 64))
        self.assertFalse(train_ppo.needs_rollout_replacement(32, 32, 64))
        self.assertFalse(train_ppo.needs_rollout_replacement(64, 0, 64))


class RolloutFailureAuditTests(unittest.TestCase):
    def test_reason_opponent_seat_and_start_error_are_fully_accounted(self) -> None:
        audit = train_ppo.RolloutFailureAudit()
        seat_one_game = SimpleNamespace(
            opponent_name="bc",
            trainable_seats={1},
        )
        selfplay_game = SimpleNamespace(
            opponent_name=None,
            trainable_seats={0, 1},
        )
        audit.record_game(
            seat_one_game,
            "battle_step_error",
            available_rows=40,
            kept_rows=32,
        )
        audit.record_game(
            selfplay_game,
            "invalid_observation",
            available_rows=0,
            kept_rows=0,
        )
        audit.record_start_error("unknown", "unknown")

        rendered = audit.render(
            started_game_attempts=3,
            valid_games=1,
            battle_start_errors=1,
        )

        self.assertEqual(rendered["failed_started_games"], 2)
        self.assertEqual(rendered["hard_failure_attempts"], 3)
        self.assertEqual(rendered["kept_as_loss_attempts"], 1)
        self.assertEqual(rendered["kept_transition_rows"], 32)
        self.assertEqual(
            rendered["by_reason"]["battle_step_error"]["attempts"],
            1,
        )
        self.assertEqual(
            rendered["by_reason"]["battle_start_error"][
                "attempts_without_learner_transitions"
            ],
            1,
        )
        self.assertEqual(rendered["by_opponent"]["bc"]["attempts"], 1)
        self.assertEqual(rendered["by_learner_seat"]["1"]["attempts"], 1)
        self.assertEqual(
            rendered["by_learner_seat"]["selfplay_both"]["attempts"],
            1,
        )

    def test_coverage_mismatch_is_rejected(self) -> None:
        audit = train_ppo.RolloutFailureAudit()
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            audit.render(
                started_game_attempts=2,
                valid_games=1,
                battle_start_errors=0,
            )


class FailedAttemptRolloutWiringTests(unittest.TestCase):
    class FakeModel:
        def eval(self) -> FailedAttemptRolloutWiringTests.FakeModel:
            return self

    class FakeBattle:
        def __init__(self, mode: str, learner_seat: int) -> None:
            self.mode = mode
            self.learner_seat = learner_seat
            self.closed = False
            self.observation = self._valid_observation()

        def _valid_observation(self) -> dict[str, object]:
            return {
                "current": {
                    "yourIndex": self.learner_seat,
                    "result": -1,
                },
                "select": {
                    "option": [{"index": 0}],
                    "minCount": 1,
                    "maxCount": 1,
                },
            }

        @property
        def result(self) -> int:
            current = self.observation.get("current") or {}
            assert isinstance(current, dict)
            return int(current.get("result", -1))

        def step(self, action: list[int]) -> tuple[dict[str, object], int]:
            self.assert_action(action)
            if self.mode == "battle_step_error":
                return self.observation, 7
            if self.mode == "invalid_observation":
                self.observation = {
                    "current": {
                        "yourIndex": self.learner_seat,
                        "result": -1,
                    }
                }
                return self.observation, 0
            if self.mode == "max_game_decisions":
                return self.observation, 0
            current = self.observation["current"]
            assert isinstance(current, dict)
            current["result"] = self.learner_seat
            return self.observation, 0

        @staticmethod
        def assert_action(action: list[int]) -> None:
            if action != [0]:
                raise AssertionError(f"unexpected test action: {action}")

        def close(self) -> None:
            self.closed = True

    class FakeQuotaController:
        def begin_update(self, update: int) -> tuple[list[str], dict[str, object]]:
            return ["bc"], {"planned_quotas": {"bc": 1}, "update": update}

        def finish_update(
            self,
            update: int,
            audit: dict[str, object],
            *_args: object,
            **_kwargs: object,
        ) -> dict[str, object]:
            return {**audit, "finished_update": update}

    def run_rollout(
        self,
        failure_mode: str,
        *,
        enabled: bool,
    ) -> tuple[list[dict[str, object]], dict[str, object], list[tuple[int, int]]]:
        model = self.FakeModel()
        deck = list(range(60))
        deck_hash = train_ppo.compute_deck_hash(deck)
        opponent = train_ppo.FrozenOpponent(
            name="bc",
            model=model,
            deck=deck,
            deck_hash=deck_hash,
            permanent=True,
            canonical_order=True,
        )
        modes = iter([failure_mode, "valid"])
        claims: list[tuple[int, int]] = []

        def make_game(
            _deck: list[int],
            _deck_hash: str,
            uid: int,
            _league_probability: float,
            _opponents: list[train_ppo.FrozenOpponent],
            opponent_index: int | None,
            _bc_probability: float | None,
            _weights: list[float] | None,
            learner_seat: int | None,
        ) -> train_ppo.RunningGame:
            self.assertEqual(opponent_index, 0)
            self.assertIn(learner_seat, (0, 1))
            assert learner_seat is not None
            claims.append((opponent_index, learner_seat))
            mode = next(modes)
            if mode == "battle_start_error":
                raise ValueError("synthetic start failure")
            return train_ppo.RunningGame(
                battle=self.FakeBattle(mode, learner_seat),
                uid=uid,
                seat_policy={learner_seat: -1, 1 - learner_seat: 0},
                seat_deck_hash={0: deck_hash, 1: deck_hash},
                trainable_seats={learner_seat},
                transition_indices={0: [], 1: []},
                opponent_name="bc",
            )

        def collate(
            features: list[dict[str, object]],
            _model_config: dict[str, object],
            _device: torch.device,
        ) -> dict[str, int]:
            return {"rows": len(features)}

        def sample(
            _outputs: dict[str, object],
            batch: dict[str, int],
            **_kwargs: object,
        ) -> tuple[list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor]:
            rows = batch["rows"]
            zeros = torch.zeros(rows)
            return [[0] for _ in range(rows)], zeros, zeros, zeros

        config = SimpleNamespace(
            opponent_weights={},
            history_opponent_weight=0.0,
            opponent_quota_seat_balance=True,
            seed=20260810,
            opponent_sampling="per_game",
            bc_opponent_probability=None,
            environments=1,
            games_per_update=1,
            league_probability=1.0,
            gamma=1.0,
            gae_lambda=1.0,
            policy_temperature=0.8,
            max_game_decisions=2,
            failed_attempt_as_loss=enabled,
            failed_loss_tail_transitions=1,
            truncation_as_loss=False,
        )
        with (
            mock.patch.object(
                train_ppo,
                "make_running_game",
                side_effect=make_game,
            ),
            mock.patch.object(
                train_ppo,
                "live_feature",
                return_value={"synthetic": True},
            ),
            mock.patch.object(train_ppo, "collate_features", side_effect=collate),
            mock.patch.object(train_ppo, "model_forward", return_value={}),
            mock.patch.object(
                train_ppo,
                "sample_ordered_actions",
                side_effect=sample,
            ),
        ):
            rollout, metrics = train_ppo.collect_rollout(
                model,
                [opponent],
                deck,
                {},
                config,
                torch.device("cpu"),
                update=1,
                opponent_quota_controller=self.FakeQuotaController(),
            )
        return rollout, metrics, claims

    def test_all_failure_paths_keep_tail_and_preserve_exact_claim(self) -> None:
        for failure_mode, reason in (
            ("invalid_observation", "invalid_observation"),
            ("battle_step_error", "battle_step_error"),
            ("max_game_decisions", "max_game_decisions"),
            ("battle_start_error", "battle_start_error"),
        ):
            with self.subTest(failure_mode=failure_mode):
                rollout, metrics, claims = self.run_rollout(
                    failure_mode,
                    enabled=True,
                )
                failure = metrics["failure_audit"]
                self.assertEqual(metrics["valid_games"], 1)
                self.assertGreaterEqual(
                    metrics["mean_decisions_per_started_attempt"],
                    0.0,
                )
                self.assertEqual(failure["hard_failure_attempts"], 1)
                self.assertEqual(failure["by_reason"][reason]["attempts"], 1)
                self.assertEqual(len(set(claims)), 1)
                learner_seat = str(claims[0][1])
                self.assertEqual(failure["by_opponent"]["bc"]["attempts"], 1)
                self.assertEqual(
                    failure["by_learner_seat"][learner_seat]["attempts"],
                    1,
                )
                expected_kept = 0 if reason == "battle_start_error" else 1
                self.assertEqual(failure["kept_as_loss_attempts"], expected_kept)
                self.assertEqual(failure["kept_transition_rows"], expected_kept)
                failed_rows = [
                    row for row in rollout if row.get("failed_attempt")
                ]
                self.assertEqual(len(failed_rows), expected_kept)
                if failed_rows:
                    self.assertEqual(failed_rows[0]["failure_reason"], reason)
                    self.assertEqual(failed_rows[0]["outcome_target"], 0.0)

    def test_default_off_still_discards_failed_trajectory(self) -> None:
        rollout, metrics, _ = self.run_rollout(
            "max_game_decisions",
            enabled=False,
        )

        self.assertEqual(len(rollout), 1)
        self.assertFalse(any(row.get("failed_attempt") for row in rollout))
        failure = metrics["failure_audit"]
        self.assertEqual(failure["kept_as_loss_attempts"], 0)
        self.assertEqual(failure["discarded_attempts"], 1)


class TerminalOutcomeGaeTests(unittest.TestCase):
    @staticmethod
    def transitions() -> list[dict[str, object]]:
        return [
            {"old_value": value, "keep": True}
            for value in (0.2, 0.5, 0.8)
        ]

    def test_lambda_one_propagates_terminal_outcome_to_every_transition(
        self,
    ) -> None:
        for result, outcome in ((0, 1.0), (1, 0.0), (2, 0.0)):
            with self.subTest(result=result):
                transitions = self.transitions()
                game = SimpleNamespace(transition_indices={0: [0, 1, 2]})

                train_ppo.finalize_episode(
                    game,
                    transitions,
                    result=result,
                    gamma=1.0,
                    gae_lambda=1.0,
                    valid=True,
                )

                self.assertEqual(
                    [row["outcome_target"] for row in transitions],
                    [outcome, outcome, outcome],
                )
                self.assertEqual(
                    [row["return"] for row in transitions],
                    [outcome, outcome, outcome],
                )
                for row in transitions:
                    self.assertAlmostEqual(
                        float(row["advantage"]),
                        outcome - float(row["old_value"]),
                        places=7,
                    )
                self.assertEqual(
                    [row["terminal_reward"] for row in transitions],
                    [0.0, 0.0, outcome],
                )

    def test_invalid_episode_is_excluded_without_bootstrap_targets(self) -> None:
        transitions = self.transitions()
        game = SimpleNamespace(transition_indices={0: [0, 1, 2]})

        train_ppo.finalize_episode(
            game,
            transitions,
            result=0,
            gamma=1.0,
            gae_lambda=1.0,
            valid=False,
        )

        self.assertTrue(all(row["keep"] is False for row in transitions))
        self.assertTrue(all("advantage" not in row for row in transitions))

    def test_truncation_as_loss_keeps_zero_return_targets(self) -> None:
        transitions = self.transitions()
        game = SimpleNamespace(transition_indices={0: [0, 1, 2]})

        kept = train_ppo.finalize_truncated_episode(
            game,
            transitions,
            gamma=1.0,
            gae_lambda=1.0,
            as_loss=True,
        )

        self.assertEqual(kept, 3)
        self.assertTrue(all(row["keep"] is True for row in transitions))
        self.assertEqual(
            [row["outcome_target"] for row in transitions],
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(
            [row["return"] for row in transitions],
            [0.0, 0.0, 0.0],
        )
        for row in transitions:
            self.assertAlmostEqual(
                float(row["advantage"]),
                -float(row["old_value"]),
                places=7,
            )

    def test_legacy_truncation_still_excludes_episode(self) -> None:
        transitions = self.transitions()
        game = SimpleNamespace(transition_indices={0: [0, 1, 2]})

        kept = train_ppo.finalize_truncated_episode(
            game,
            transitions,
            gamma=1.0,
            gae_lambda=1.0,
            as_loss=False,
        )

        self.assertEqual(kept, 0)
        self.assertTrue(all(row["keep"] is False for row in transitions))
        self.assertTrue(all("advantage" not in row for row in transitions))

    def test_failed_episode_keeps_only_one_global_bounded_tail(self) -> None:
        transitions = [
            {"old_value": index / 10.0, "keep": True, "game_uid": 99}
            for index in range(8)
        ]
        game = SimpleNamespace(
            transition_indices={
                0: [0, 2, 4, 6],
                1: [1, 3, 5, 7],
            }
        )

        kept = train_ppo.finalize_failed_episode_tail(
            game,
            transitions,
            gamma=1.0,
            gae_lambda=1.0,
            tail_transitions=3,
            failure_reason="max_game_decisions",
        )

        self.assertEqual(kept, 3)
        self.assertEqual(
            [bool(row["keep"]) for row in transitions],
            [False, False, False, False, False, True, True, True],
        )
        for row in transitions[:5]:
            self.assertNotIn("outcome_target", row)
            self.assertNotIn("failed_attempt", row)
        for row in transitions[5:]:
            self.assertEqual(row["outcome_target"], 0.0)
            self.assertEqual(row["return"], 0.0)
            self.assertEqual(row["terminal_reward"], 0.0)
            self.assertAlmostEqual(
                float(row["advantage"]),
                -float(row["old_value"]),
                places=7,
            )
            self.assertTrue(row["failed_attempt"])
            self.assertEqual(row["failure_reason"], "max_game_decisions")

    def test_failed_episode_without_learner_rows_creates_no_sample(self) -> None:
        transitions: list[dict[str, object]] = []
        game = SimpleNamespace(transition_indices={0: [], 1: []})

        kept = train_ppo.finalize_failed_episode_tail(
            game,
            transitions,
            gamma=1.0,
            gae_lambda=1.0,
            tail_transitions=32,
            failure_reason="invalid_observation",
        )

        self.assertEqual(kept, 0)
        self.assertEqual(transitions, [])

    def test_failure_loss_configuration_is_explicit_and_unambiguous(self) -> None:
        train_ppo.validate_failed_attempt_loss_configuration(False, 32, False)
        train_ppo.validate_failed_attempt_loss_configuration(True, 32, False)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            train_ppo.validate_failed_attempt_loss_configuration(True, 0, False)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            train_ppo.validate_failed_attempt_loss_configuration(True, 32, True)

    def test_failure_loss_config_defaults_preserve_legacy_behavior(self) -> None:
        fields = train_ppo.PPOConfig.__dataclass_fields__
        self.assertIs(fields["failed_attempt_as_loss"].default, False)
        self.assertEqual(fields["failed_loss_tail_transitions"].default, 32)
        legacy = SimpleNamespace(
            truncation_as_loss=False,
            failed_attempt_as_loss=False,
            failed_loss_tail_transitions=32,
        )
        self.assertEqual(
            train_ppo.reward_manifest(legacy),
            {
                "win": 1.0,
                "loss": 0.0,
                "draw": 0.0,
                "truncation_handling": "exclude_and_replace",
            },
        )
        enabled = SimpleNamespace(
            truncation_as_loss=False,
            failed_attempt_as_loss=True,
            failed_loss_tail_transitions=32,
        )
        manifest = train_ppo.reward_manifest(enabled)
        self.assertEqual(
            manifest["truncation_handling"],
            "tail_terminal_outcome_0_and_replace",
        )
        self.assertEqual(
            manifest["failed_attempt_handling"]["tail_transitions"],
            32,
        )

    def test_lambda_decay_has_no_reverse_or_off_by_one(self) -> None:
        transitions = [
            {"old_value": 0.0, "keep": True}
            for _ in range(4)
        ]
        game = SimpleNamespace(transition_indices={0: [0, 1, 2, 3]})

        train_ppo.finalize_episode(
            game,
            transitions,
            result=0,
            gamma=1.0,
            gae_lambda=0.97,
            valid=True,
        )

        expected = [0.97**3, 0.97**2, 0.97, 1.0]
        for row, advantage in zip(transitions, expected, strict=True):
            self.assertAlmostEqual(
                float(row["advantage"]),
                advantage,
                places=7,
            )


class KLEarlyStopAggregationTests(unittest.TestCase):
    def test_minibatch_kl_is_weighted_by_rows_including_remainder(self) -> None:
        actual = train_ppo.row_weighted_minibatch_mean(
            [(0.0, 512), (1.0, 1)]
        )
        self.assertAlmostEqual(actual, 1.0 / 513.0, places=12)
        self.assertNotAlmostEqual(actual, 0.5, places=3)

    def test_row_weighted_mean_rejects_empty_or_zero_row_batches(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            train_ppo.row_weighted_minibatch_mean([])
        with self.assertRaisesRegex(ValueError, "positive"):
            train_ppo.row_weighted_minibatch_mean([(0.2, 0)])


class AdvantageNormalizationTests(unittest.TestCase):
    def test_global_matches_legacy_sample_standardization(self) -> None:
        transitions = [
            {"advantage": value}
            for value in (1.0, 2.0, 4.0, 8.0)
        ]
        actual, audit = train_ppo.normalize_rollout_advantages(
            transitions,
            "global",
        )
        raw = torch.tensor([1.0, 2.0, 4.0, 8.0])
        expected = (raw - raw.mean()) / raw.std()
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(audit["mode"], "global")

    def test_per_opponent_standardizes_each_frozen_group(self) -> None:
        transitions = [
            {"advantage": 1.0, "opponent_name": "bc"},
            {"advantage": 3.0, "opponent_name": "bc"},
            {"advantage": 100.0, "opponent_name": "v3"},
            {"advantage": 140.0, "opponent_name": "v3"},
        ]
        normalized, audit = train_ppo.normalize_rollout_advantages(
            transitions,
            "per_opponent",
        )
        for indices in ([0, 1], [2, 3]):
            values = normalized[indices]
            self.assertAlmostEqual(float(values.mean()), 0.0, places=6)
            self.assertAlmostEqual(float(values.std()), 1.0, places=6)
        self.assertEqual(audit["group_count"], 2)
        self.assertEqual(
            sum(row["transitions"] for row in audit["by_group"].values()),
            4,
        )

    def test_degenerate_and_selfplay_groups_are_finite_zero(self) -> None:
        transitions = [
            {"advantage": 2.0, "opponent_name": None},
            {"advantage": 9.0, "opponent_name": "bc"},
            {"advantage": 9.0, "opponent_name": "bc"},
        ]
        normalized, audit = train_ppo.normalize_rollout_advantages(
            transitions,
            "per_opponent",
        )
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertEqual(normalized.tolist(), [0.0, 0.0, 0.0])
        self.assertIn(
            train_ppo.SELFPLAY_OPPONENT_GROUP,
            audit["degenerate_groups"],
        )

    def test_per_opponent_requires_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "opponent_name"):
            train_ppo.normalize_rollout_advantages(
                [{"advantage": 1.0}],
                "per_opponent",
            )


class QuotaGroupActorReductionTests(unittest.TestCase):
    @staticmethod
    def transitions() -> list[dict[str, object]]:
        return [
            *({"opponent_name": "short"} for _ in range(2)),
            *({"opponent_name": "long"} for _ in range(6)),
        ]

    def test_multipliers_equal_quota_weighted_group_mean(self) -> None:
        transitions = self.transitions()
        multipliers, audit = train_ppo.quota_group_actor_multipliers(
            transitions,
            {"short": 1, "long": 1},
        )
        values = torch.tensor([2.0, 4.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        reduced = float((values * multipliers).mean())
        expected = float(
            0.5 * values[:2].mean() + 0.5 * values[2:].mean()
        )
        self.assertAlmostEqual(reduced, expected, places=6)
        self.assertAlmostEqual(float(multipliers.mean()), 1.0, places=6)
        self.assertAlmostEqual(
            float(audit["groups"]["short"]["row_multiplier"]),
            2.0,
        )
        self.assertAlmostEqual(
            float(audit["groups"]["long"]["row_multiplier"]),
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            float(audit["max_relative_weight_deviation"]),
            0.5,
        )

    def test_matching_row_and_quota_shares_are_identity(self) -> None:
        multipliers, audit = train_ppo.quota_group_actor_multipliers(
            self.transitions(),
            {"short": 1, "long": 3},
        )
        torch.testing.assert_close(multipliers, torch.ones(8))
        self.assertAlmostEqual(
            float(audit["max_relative_weight_deviation"]),
            0.0,
        )

    def test_missing_or_unexpected_rollout_group_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            train_ppo.quota_group_actor_multipliers(
                self.transitions(),
                {"short": 1, "other": 1},
            )

    def test_empty_or_nonpositive_quota_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive quotas"):
            train_ppo.quota_group_actor_multipliers(
                self.transitions(),
                {"short": 0, "long": 0},
            )

    def test_actor_loss_reduction_does_not_renormalize_minibatch(self) -> None:
        surrogate = torch.tensor([1.0, 3.0, -2.0])
        multipliers = torch.tensor([2.0, 0.5, 0.5])
        transition_mean = train_ppo.standard_actor_policy_loss(
            surrogate,
            multipliers,
            "transition_mean",
        )
        quota_group = train_ppo.standard_actor_policy_loss(
            surrogate,
            multipliers,
            "quota_group_mean",
        )
        episode_mean = train_ppo.standard_actor_policy_loss(
            surrogate,
            multipliers,
            "episode_mean",
        )
        torch.testing.assert_close(transition_mean, -surrogate.mean())
        torch.testing.assert_close(
            quota_group,
            -(surrogate * multipliers).mean(),
        )
        torch.testing.assert_close(episode_mean, quota_group)

    def test_audit_only_always_suppresses_initial_evaluation(self) -> None:
        self.assertTrue(
            train_ppo.should_run_initial_evaluation(False, False)
        )
        self.assertFalse(
            train_ppo.should_run_initial_evaluation(True, False)
        )
        self.assertFalse(
            train_ppo.should_run_initial_evaluation(False, True)
        )
        self.assertFalse(
            train_ppo.should_run_initial_evaluation(True, True)
        )
        self.assertFalse(
            train_ppo.should_run_initial_evaluation(False, False, True)
        )


class EpisodeMeanActorReductionTests(unittest.TestCase):
    @staticmethod
    def transitions() -> list[dict[str, object]]:
        return [
            *({"game_uid": 101} for _ in range(2)),
            *({"game_uid": 202} for _ in range(6)),
        ]

    def test_multipliers_equal_mean_of_per_game_means(self) -> None:
        transitions = self.transitions()
        multipliers, audit = train_ppo.episode_mean_actor_multipliers(
            transitions,
            expected_episode_count=2,
        )
        values = torch.tensor(
            [2.0, 4.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        )
        reduced = float((values * multipliers).mean())
        expected = float(
            0.5 * values[:2].mean() + 0.5 * values[2:].mean()
        )
        self.assertAlmostEqual(reduced, expected, places=6)
        self.assertAlmostEqual(float(multipliers.mean()), 1.0, places=6)
        self.assertEqual(audit["episode_count"], 2)
        self.assertEqual(audit["expected_episode_count"], 2)
        self.assertEqual(audit["covered_transition_rows"], len(transitions))
        self.assertEqual(audit["missing_game_uid_rows"], 0)
        self.assertTrue(audit["all_transition_rows_accounted_for"])
        self.assertTrue(audit["all_games_equal_weight"])
        self.assertAlmostEqual(
            float(audit["episodes"]["101"]["episode_total_weight"]),
            0.5,
        )
        self.assertAlmostEqual(
            float(audit["episodes"]["202"]["episode_total_weight"]),
            0.5,
        )
        self.assertAlmostEqual(
            float(audit["max_episode_weight_error"]),
            0.0,
        )

    def test_missing_uid_or_missing_episode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "game_uid on every"):
            train_ppo.episode_mean_actor_multipliers(
                [{"game_uid": 1}, {}],
            )
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            train_ppo.episode_mean_actor_multipliers(
                self.transitions(),
                expected_episode_count=3,
            )

    def test_non_integer_uid_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer game_uid"):
            train_ppo.episode_mean_actor_multipliers(
                [{"game_uid": "one"}],
            )

    def test_kept_episode_count_uses_actual_failure_and_valid_uids(self) -> None:
        transitions = [
            {"game_uid": 101},
            {"game_uid": 101},
            {"game_uid": 202},
            {"game_uid": 303, "failed_attempt": True},
        ]
        self.assertEqual(train_ppo.kept_episode_count(transitions), 3)


class ActorReductionGradientAuditTests(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[
        train_bc_orbit.EntityOptionPolicy,
        list[dict[str, object]],
        dict[str, int],
        list[torch.nn.Parameter],
    ]:
        torch.manual_seed(707)
        model = train_bc_orbit.EntityOptionPolicy(
            hash_size=64,
            categorical_dim=8,
            model_dim=16,
            layers=2,
            heads=4,
            dropout=0.0,
            max_state_entities=8,
        )
        model.count_head[-1] = torch.nn.Linear(
            16,
            train_ppo.MAX_ACTION_COUNT + 1,
        )
        actor_parameters, _, _ = train_ppo.configure_trainable_scope(
            model,
            "full",
        )
        model_config = {
            "hash_size": 64,
            "max_state_entities": 8,
            "entity_fields": 12,
            "option_fields": 12,
        }
        features = []
        for action in (0, 1, 0, 1, 1, 0, 1, 0):
            feature = train_bc_orbit.featurize_row(
                {
                    "observation": {
                        "select": {
                            "option": [
                                {"index": 0},
                                {"index": 1},
                            ],
                            "minCount": 1,
                            "maxCount": 1,
                            "context": 7,
                        }
                    },
                    "action": [action],
                },
                hash_size=64,
                max_state_entities=8,
            )
            assert feature is not None
            features.append(feature)
        batch = train_ppo.collate_features_cpu(features, model_config)
        action_sequences = torch.full(
            (len(features), train_ppo.MAX_ACTION_COUNT),
            -1,
            dtype=torch.long,
        )
        actions = (0, 1, 0, 1, 1, 0, 1, 0)
        action_sequences[:, 0] = torch.tensor(actions)
        with torch.no_grad():
            outputs = train_ppo.model_forward(
                model,
                batch,
                torch.device("cpu"),
            )
            old_log_probs, _ = (
                train_ppo.ordered_action_log_prob_entropy(
                    outputs,
                    batch,
                    action_sequences,
                    torch.ones(len(features), dtype=torch.long),
                    temperature=0.8,
                )
            )
        advantages = (-1.5, 2.0, -0.7, 1.4, -1.1, 0.5, 1.8, -0.2)
        transitions: list[dict[str, object]] = []
        for index, (feature, action, advantage, old_log_prob) in enumerate(
            zip(
                features,
                actions,
                advantages,
                old_log_probs.tolist(),
                strict=True,
            )
        ):
            transitions.append(
                {
                    "feature": feature,
                    "action": [action],
                    "action_count": 1,
                    "advantage": advantage,
                    "old_log_prob": old_log_prob,
                    "opponent_name": "short" if index < 2 else "long",
                    "game_uid": 100 if index < 2 else 200,
                }
            )
        return model, transitions, model_config, actor_parameters

    def test_chunked_audit_matches_direct_full_batch_and_has_no_side_effect(
        self,
    ) -> None:
        model, transitions, model_config, actor_parameters = self.fixture()
        state_before = train_ppo.model_state_sha256(model)
        chunked_config = SimpleNamespace(
            ppo_objective="standard",
            advantage_normalization="global",
            actor_reduction="quota_group_mean",
            opponent_base_quotas={"short": 1, "long": 1},
            minibatch_size=3,
            policy_temperature=0.8,
            clip_ratio=0.15,
        )
        direct_config = SimpleNamespace(
            **{
                **vars(chunked_config),
                "minibatch_size": len(transitions),
            }
        )
        chunked = train_ppo.audit_actor_reduction_gradients(
            model,
            transitions,
            model_config,
            chunked_config,
            torch.device("cpu"),
            actor_parameters,
        )
        direct = train_ppo.audit_actor_reduction_gradients(
            model,
            transitions,
            model_config,
            direct_config,
            torch.device("cpu"),
            actor_parameters,
        )
        for key in (
            "cosine",
            "standard_norm",
            "quota_group_norm",
            "quota_to_standard_norm_ratio",
        ):
            self.assertAlmostEqual(
                float(chunked["gradient"][key]),
                float(direct["gradient"][key]),
                places=5,
            )
        self.assertEqual(chunked["optimizer_steps"], 0)
        self.assertEqual(
            chunked["model_state_sha256_before"],
            state_before,
        )
        self.assertEqual(
            chunked["model_state_sha256_after"],
            state_before,
        )
        self.assertTrue(
            chunked["pre_registered_gate"]["row_weight_triggered"]
        )
        self.assertTrue(
            chunked["pre_registered_gate"]["training_authorized"]
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )

    def test_episode_mean_audit_is_chunk_invariant_and_checks_coverage(
        self,
    ) -> None:
        model, transitions, model_config, actor_parameters = self.fixture()
        for transition in transitions[-2:]:
            transition["game_uid"] = 300
        state_before = train_ppo.model_state_sha256(model)
        chunked_config = SimpleNamespace(
            ppo_objective="standard",
            advantage_normalization="global",
            actor_reduction="episode_mean",
            opponent_base_quotas={},
            games_per_update=2,
            minibatch_size=3,
            policy_temperature=0.8,
            clip_ratio=0.15,
        )
        direct_config = SimpleNamespace(
            **{
                **vars(chunked_config),
                "minibatch_size": len(transitions),
            }
        )
        chunked = train_ppo.audit_actor_reduction_gradients(
            model,
            transitions,
            model_config,
            chunked_config,
            torch.device("cpu"),
            actor_parameters,
        )
        direct = train_ppo.audit_actor_reduction_gradients(
            model,
            transitions,
            model_config,
            direct_config,
            torch.device("cpu"),
            actor_parameters,
        )
        for key in (
            "cosine",
            "standard_norm",
            "candidate_norm",
            "candidate_to_standard_norm_ratio",
            "episode_mean_norm",
            "episode_to_standard_norm_ratio",
        ):
            self.assertAlmostEqual(
                float(chunked["gradient"][key]),
                float(direct["gradient"][key]),
                places=5,
            )
        self.assertEqual(chunked["candidate_reduction"], "episode_mean")
        self.assertEqual(
            chunked["row_weight_audit"]["episode_count"],
            3,
        )
        self.assertEqual(
            chunked["row_weight_audit"]["expected_episode_count"],
            3,
        )
        self.assertTrue(
            chunked["row_weight_audit"]["all_games_equal_weight"]
        )
        self.assertTrue(
            chunked["row_weight_audit"][
                "all_transition_rows_accounted_for"
            ]
        )
        self.assertEqual(chunked["optimizer_steps"], 0)
        self.assertEqual(
            chunked["model_state_sha256_before"],
            state_before,
        )
        self.assertEqual(
            chunked["model_state_sha256_after"],
            state_before,
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )

    def test_standard_update_accepts_per_opponent_and_records_audit(
        self,
    ) -> None:
        model, transitions, model_config, _ = self.fixture()
        for index, transition in enumerate(transitions):
            transition["outcome_target"] = float(index % 2)
        reference_model = copy.deepcopy(model)
        actor_parameters, value_parameters, _ = (
            train_ppo.configure_trainable_scope(model, "full")
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": actor_parameters, "lr": 1e-4},
                {"params": value_parameters, "lr": 1e-4},
            ],
            eps=1e-5,
            weight_decay=0.0,
        )
        config = SimpleNamespace(
            ppo_objective="standard",
            advantage_normalization="per_opponent",
            actor_reduction="transition_mean",
            learning_rate_schedule="constant",
            learning_rate=1e-4,
            value_learning_rate=1e-4,
            updates=1,
            schedule_start_update=1,
            ppo_epochs=1,
            minibatch_size=len(transitions),
            policy_temperature=0.8,
            clip_ratio=0.15,
            value_coefficient=0.0,
            value_trunk_gradient_scale=0.0,
            entropy_coefficient=0.0,
            max_grad_norm=1.0,
            bc_kl_start=0.0,
            bc_kl_end=0.0,
            target_kl=1.0,
        )

        observed_scales: list[
            tuple[train_bc_orbit.EntityOptionPolicy, float]
        ] = []
        original_model_forward = train_ppo.model_forward

        def recording_model_forward(
            forwarded_model: train_bc_orbit.EntityOptionPolicy,
            batch: dict[str, torch.Tensor],
            device: torch.device,
            *,
            value_trunk_gradient_scale: float = 1.0,
        ) -> dict[str, torch.Tensor]:
            observed_scales.append(
                (forwarded_model, value_trunk_gradient_scale)
            )
            return original_model_forward(
                forwarded_model,
                batch,
                device,
                value_trunk_gradient_scale=value_trunk_gradient_scale,
            )

        with mock.patch.object(
            train_ppo,
            "model_forward",
            side_effect=recording_model_forward,
        ):
            result = train_ppo.ppo_update(
                model,
                reference_model,
                optimizer,
                transitions,
                model_config,
                config,
                torch.device("cpu"),
                update=1,
            )

        audit = result["advantage_normalization"]
        self.assertEqual(audit["mode"], "per_opponent")
        self.assertEqual(audit["group_count"], 2)
        self.assertEqual(audit["degenerate_groups"], [])
        self.assertEqual(
            sum(row["transitions"] for row in audit["by_group"].values()),
            len(transitions),
        )
        for row in audit["by_group"].values():
            self.assertAlmostEqual(row["normalized_mean"], 0.0, places=6)
            self.assertAlmostEqual(row["normalized_std"], 1.0, places=6)
        self.assertEqual(
            result["value_trunk_gradient"],
            train_ppo.value_trunk_gradient_audit(0.0),
        )
        self.assertIn((model, 0.0), observed_scales)
        self.assertIn((reference_model, 1.0), observed_scales)

    def test_standard_update_records_episode_mean_invariants(self) -> None:
        model, transitions, model_config, _ = self.fixture()
        for transition in transitions[-2:]:
            transition["game_uid"] = 300
        for index, transition in enumerate(transitions):
            transition["outcome_target"] = float(index % 2)
        reference_model = copy.deepcopy(model)
        actor_parameters, value_parameters, _ = (
            train_ppo.configure_trainable_scope(model, "full")
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": actor_parameters, "lr": 1e-4},
                {"params": value_parameters, "lr": 1e-4},
            ],
            eps=1e-5,
            weight_decay=0.0,
        )
        config = SimpleNamespace(
            ppo_objective="standard",
            advantage_normalization="global",
            actor_reduction="episode_mean",
            opponent_base_quotas={},
            games_per_update=2,
            learning_rate_schedule="constant",
            learning_rate=1e-4,
            value_learning_rate=1e-4,
            updates=1,
            schedule_start_update=1,
            ppo_epochs=1,
            minibatch_size=len(transitions),
            policy_temperature=0.8,
            clip_ratio=0.15,
            value_coefficient=0.0,
            value_trunk_gradient_scale=0.0,
            entropy_coefficient=0.0,
            max_grad_norm=1.0,
            bc_kl_start=0.0,
            bc_kl_end=0.0,
            target_kl=1.0,
        )
        result = train_ppo.ppo_update(
            model,
            reference_model,
            optimizer,
            transitions,
            model_config,
            config,
            torch.device("cpu"),
            update=1,
        )
        audit = result["actor_reduction"]
        self.assertEqual(audit["mode"], "episode_mean")
        self.assertEqual(audit["episode_count"], 3)
        self.assertEqual(audit["expected_episode_count"], 3)
        self.assertTrue(audit["all_games_equal_weight"])
        self.assertTrue(audit["all_transition_rows_accounted_for"])
        self.assertAlmostEqual(audit["mean_row_multiplier"], 1.0, places=6)


class ValueTrunkGradientScaleTests(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[
        train_bc_orbit.EntityOptionPolicy,
        dict[str, torch.Tensor],
    ]:
        model, transitions, model_config, _ = (
            ActorReductionGradientAuditTests.fixture()
        )
        batch = train_ppo.collate_features_cpu(
            [transition["feature"] for transition in transitions],
            model_config,
        )
        return model, batch

    def test_default_and_explicit_one_are_elementwise_identical(self) -> None:
        field = train_ppo.PPOConfig.__dataclass_fields__[
            "value_trunk_gradient_scale"
        ]
        self.assertEqual(field.default, 1.0)
        model, batch = self.fixture()
        default_outputs = train_ppo.model_forward(
            model,
            batch,
            torch.device("cpu"),
        )
        explicit_outputs = train_ppo.model_forward(
            model,
            batch,
            torch.device("cpu"),
            value_trunk_gradient_scale=1.0,
        )
        for name in ("policy_logits", "count_logits", "value_logits"):
            self.assertTrue(
                torch.equal(default_outputs[name], explicit_outputs[name]),
                name,
            )

    def test_forward_values_are_identical_at_every_scale(self) -> None:
        model, batch = self.fixture()
        baseline = train_ppo.model_forward(
            model,
            batch,
            torch.device("cpu"),
        )
        for scale in (0.0, 0.5):
            outputs = train_ppo.model_forward(
                model,
                batch,
                torch.device("cpu"),
                value_trunk_gradient_scale=scale,
            )
            for name in ("policy_logits", "count_logits", "value_logits"):
                self.assertTrue(
                    torch.equal(outputs[name], baseline[name]),
                    f"{scale}/{name}",
                )

    def test_only_value_gradient_into_shared_trunk_is_scaled(self) -> None:
        model, batch = self.fixture()

        def gradients(
            scale: float,
        ) -> tuple[torch.Tensor | None, torch.Tensor]:
            model.zero_grad(set_to_none=True)
            outputs = train_ppo.model_forward(
                model,
                batch,
                torch.device("cpu"),
                value_trunk_gradient_scale=scale,
            )
            outputs["value_logits"].square().mean().backward()
            shared = model.transformer.layers[-1].linear1.weight.grad
            head = model.value_head[0].weight.grad
            self.assertIsNotNone(head)
            return (
                None if shared is None else shared.detach().clone(),
                head.detach().clone(),
            )

        shared_one, head_one = gradients(1.0)
        shared_half, head_half = gradients(0.5)
        shared_zero, head_zero = gradients(0.0)
        zero_gradient_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

        self.assertIsNotNone(shared_one)
        self.assertIsNotNone(shared_half)
        self.assertGreater(float(shared_one.norm()), 0.0)
        torch.testing.assert_close(shared_half, shared_one * 0.5)
        self.assertIsNone(shared_zero)
        self.assertTrue(torch.equal(head_half, head_one))
        self.assertTrue(torch.equal(head_zero, head_one))
        self.assertTrue(zero_gradient_names)
        self.assertTrue(
            all(name.startswith("value_head.") for name in zero_gradient_names)
        )

    def test_actor_and_count_gradients_do_not_depend_on_value_scale(self) -> None:
        model, batch = self.fixture()

        def gradients(
            output_name: str,
            scale: float,
        ) -> dict[str, torch.Tensor]:
            model.zero_grad(set_to_none=True)
            outputs = train_ppo.model_forward(
                model,
                batch,
                torch.device("cpu"),
                value_trunk_gradient_scale=scale,
            )
            outputs[output_name].square().mean().backward()
            return {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }

        for output_name in ("policy_logits", "count_logits"):
            baseline = gradients(output_name, 1.0)
            detached = gradients(output_name, 0.0)
            self.assertEqual(detached.keys(), baseline.keys(), output_name)
            for name in baseline:
                self.assertTrue(
                    torch.equal(detached[name], baseline[name]),
                    f"{output_name}/{name}",
                )

    def test_validator_and_manifest_cover_boundaries_and_invalids(self) -> None:
        for scale in (0.0, 0.5, 1.0):
            with self.subTest(valid=scale):
                self.assertEqual(
                    train_ppo.validate_value_trunk_gradient_scale(scale),
                    scale,
                )
                audit = train_ppo.value_trunk_gradient_audit(scale)
                self.assertEqual(audit["scale"], scale)
                self.assertTrue(audit["value_head_receives_full_gradient"])
                self.assertTrue(audit["forward_values_unchanged"])
        for scale in (
            -0.01,
            1.01,
            float("nan"),
            float("inf"),
            -float("inf"),
        ):
            with self.subTest(invalid=scale):
                with self.assertRaisesRegex(ValueError, "finite.*\\[0, 1\\]"):
                    train_ppo.validate_value_trunk_gradient_scale(scale)


class PolicyTemperatureTests(unittest.TestCase):
    @staticmethod
    def batch() -> dict[str, torch.Tensor]:
        return {
            "option_mask": torch.tensor([[True, True, True]]),
            "min_counts": torch.tensor([1]),
            "max_counts": torch.tensor([2]),
        }

    @staticmethod
    def outputs() -> dict[str, torch.Tensor]:
        count_logits = torch.full(
            (1, train_ppo.MAX_ACTION_COUNT + 1),
            -10.0,
        )
        count_logits[0, 1] = 0.0
        count_logits[0, 2] = 1.0
        return {
            "policy_logits": torch.tensor([[0.0, 2.0, 1.0]]),
            "count_logits": count_logits,
            "value_logits": torch.zeros(1),
        }

    def test_invalid_temperatures_are_rejected(self) -> None:
        for temperature in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    train_ppo.validate_policy_temperature(temperature)

    def test_deterministic_action_is_temperature_invariant(self) -> None:
        hot, _, hot_entropy, _ = train_ppo.sample_ordered_actions(
            self.outputs(),
            self.batch(),
            deterministic=True,
            temperature=1.0,
        )
        cool, _, cool_entropy, _ = train_ppo.sample_ordered_actions(
            self.outputs(),
            self.batch(),
            deterministic=True,
            temperature=0.5,
        )
        self.assertEqual(hot, [[1, 2]])
        self.assertEqual(cool, hot)
        self.assertLess(float(cool_entropy), float(hot_entropy))

    def test_sample_and_update_log_prob_use_same_temperature(self) -> None:
        outputs = self.outputs()
        batch = self.batch()
        actions, sampled_log_prob, _, _ = (
            train_ppo.sample_ordered_actions(
                outputs,
                batch,
                deterministic=True,
                temperature=0.5,
            )
        )
        sequence = torch.full(
            (1, train_ppo.MAX_ACTION_COUNT),
            -1,
            dtype=torch.long,
        )
        sequence[0, : len(actions[0])] = torch.tensor(actions[0])
        recomputed, _ = train_ppo.ordered_action_log_prob_entropy(
            outputs,
            batch,
            sequence,
            torch.tensor([len(actions[0])]),
            temperature=0.5,
        )
        torch.testing.assert_close(recomputed, sampled_log_prob)


class PCGradTests(unittest.TestCase):
    def test_conflicting_gradients_are_symmetrically_projected(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        primary_vector = torch.tensor([1.0, 0.0])
        guard_vector = torch.tensor([-1.0, 1.0])
        primary_loss = (parameter * primary_vector).sum()
        guard_loss = (parameter * guard_vector).sum()
        adjustments, audit = train_ppo.symmetric_pcgrad_adjustment(
            primary_loss,
            guard_loss,
            [parameter],
        )
        (primary_loss + guard_loss).backward()
        assert parameter.grad is not None
        assert adjustments[0] is not None
        parameter.grad.add_(adjustments[0])
        self.assertTrue(audit["conflict"])
        self.assertGreaterEqual(
            float(parameter.grad @ primary_vector),
            -1e-7,
        )
        self.assertGreaterEqual(
            float(parameter.grad @ guard_vector),
            -1e-7,
        )
        self.assertGreaterEqual(
            float(audit["projected_core_dot_primary"]),
            -1e-7,
        )
        self.assertGreaterEqual(
            float(audit["projected_core_dot_guard"]),
            -1e-7,
        )

    def test_aligned_gradients_are_not_changed(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        primary_loss = (parameter * torch.tensor([1.0, 0.0])).sum()
        guard_loss = (parameter * torch.tensor([1.0, 1.0])).sum()
        adjustments, audit = train_ppo.symmetric_pcgrad_adjustment(
            primary_loss,
            guard_loss,
            [parameter],
        )
        self.assertFalse(audit["conflict"])
        assert adjustments[0] is not None
        torch.testing.assert_close(
            adjustments[0],
            torch.zeros_like(adjustments[0]),
        )

    def test_guard_priority_preserves_full_guard_direction(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        primary_vector = torch.tensor([1.0, 0.0])
        guard_vector = torch.tensor([-1.0, 1.0])
        primary_loss = (parameter * primary_vector).sum()
        guard_loss = (parameter * guard_vector).sum()
        adjustments, audit = (
            train_ppo.guard_priority_pcgrad_adjustment(
                primary_loss,
                guard_loss,
                [parameter],
            )
        )
        (primary_loss + guard_loss).backward()
        assert parameter.grad is not None
        assert adjustments[0] is not None
        parameter.grad.add_(adjustments[0])
        self.assertTrue(audit["conflict"])
        self.assertGreaterEqual(
            float(parameter.grad @ guard_vector),
            float(guard_vector @ guard_vector) - 1e-7,
        )
        self.assertGreaterEqual(
            float(audit["projected_core_dot_guard"]),
            float(guard_vector @ guard_vector) - 1e-7,
        )
        self.assertEqual(
            float(audit["guard_projection_coefficient"]),
            0.0,
        )


class ActorPriorityValuePCGradTests(unittest.TestCase):
    def test_conflict_projects_only_shared_critic_gradient(self) -> None:
        shared = torch.nn.Parameter(torch.zeros(2))
        actor_only = torch.nn.Parameter(torch.zeros(()))
        value_head = torch.nn.Parameter(torch.zeros(()))
        actor_vector = torch.tensor([1.0, 0.0])
        value_vector = torch.tensor([-1.0, 1.0])
        actor_loss = (
            (shared * actor_vector).sum() + 2.0 * actor_only
        )
        value_loss = (
            (shared * value_vector).sum() + 3.0 * value_head
        )

        adjustments, audit = (
            train_ppo.actor_priority_value_pcgrad_adjustment(
                actor_loss,
                value_loss,
                [shared, actor_only],
            )
        )
        (actor_loss + value_loss).backward()
        assert shared.grad is not None
        assert actor_only.grad is not None
        assert value_head.grad is not None
        scalar_actor_only_gradient = actor_only.grad.clone()
        scalar_value_head_gradient = value_head.grad.clone()
        assert adjustments[0] is not None
        shared.grad.add_(adjustments[0])

        self.assertTrue(audit["conflict"])
        self.assertEqual(audit["shared_tensor_count"], 1)
        self.assertEqual(audit["actor_only_tensor_count"], 1)
        self.assertIsNone(adjustments[1])
        torch.testing.assert_close(
            shared.grad,
            torch.tensor([1.0, 1.0]),
        )
        torch.testing.assert_close(
            actor_only.grad,
            scalar_actor_only_gradient,
        )
        torch.testing.assert_close(
            value_head.grad,
            scalar_value_head_gradient,
        )
        self.assertGreaterEqual(
            float(audit["projected_value_dot_actor"]),
            -1e-7,
        )
        self.assertGreaterEqual(
            float(audit["final_dot_actor_margin"]),
            -1e-7,
        )

    def test_nonconflict_orthogonal_and_zero_actor_are_noops(self) -> None:
        cases = (
            (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])),
            (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
            (torch.tensor([0.0, 0.0]), torch.tensor([-1.0, 1.0])),
        )
        for actor_vector, value_vector in cases:
            with self.subTest(
                actor=actor_vector.tolist(),
                value=value_vector.tolist(),
            ):
                parameter = torch.nn.Parameter(torch.zeros(2))
                actor_loss = (parameter * actor_vector).sum()
                value_loss = (parameter * value_vector).sum()
                adjustments, audit = (
                    train_ppo.actor_priority_value_pcgrad_adjustment(
                        actor_loss,
                        value_loss,
                        [parameter],
                    )
                )
                self.assertFalse(audit["conflict"])
                self.assertTrue(audit["finite"])
                assert adjustments[0] is not None
                torch.testing.assert_close(
                    adjustments[0],
                    torch.zeros_like(adjustments[0]),
                )

    def test_nonconflict_candidate_matches_scalar_gradients_and_losses(
        self,
    ) -> None:
        scalar_parameter = torch.nn.Parameter(torch.zeros(2))
        candidate_parameter = torch.nn.Parameter(torch.zeros(2))
        actor_vector = torch.tensor([1.0, 0.0])
        value_vector = torch.tensor([1.0, 1.0])

        scalar_policy = (scalar_parameter * actor_vector).sum()
        scalar_kl = 0.12 * scalar_parameter[0]
        scalar_entropy = 0.03 * scalar_parameter[1]
        scalar_value = (scalar_parameter * value_vector).sum()
        scalar_actor_side = scalar_policy + scalar_kl - scalar_entropy
        scalar_value_side = 0.25 * scalar_value
        scalar_loss = scalar_actor_side + scalar_value_side
        scalar_loss.backward()

        candidate_policy = (candidate_parameter * actor_vector).sum()
        candidate_kl = 0.12 * candidate_parameter[0]
        candidate_entropy = 0.03 * candidate_parameter[1]
        candidate_value = (candidate_parameter * value_vector).sum()
        candidate_actor_side = (
            candidate_policy + candidate_kl - candidate_entropy
        )
        candidate_value_side = 0.25 * candidate_value
        candidate_loss = candidate_actor_side + candidate_value_side
        adjustments, audit = (
            train_ppo.actor_priority_value_pcgrad_adjustment(
                candidate_actor_side,
                candidate_value_side,
                [candidate_parameter],
            )
        )
        candidate_loss.backward()
        assert candidate_parameter.grad is not None
        assert scalar_parameter.grad is not None
        assert adjustments[0] is not None
        candidate_parameter.grad.add_(adjustments[0])

        self.assertFalse(audit["conflict"])
        torch.testing.assert_close(candidate_policy, scalar_policy)
        torch.testing.assert_close(candidate_kl, scalar_kl)
        torch.testing.assert_close(candidate_entropy, scalar_entropy)
        torch.testing.assert_close(candidate_value, scalar_value)
        torch.testing.assert_close(candidate_loss, scalar_loss)
        torch.testing.assert_close(
            candidate_parameter.grad,
            scalar_parameter.grad,
        )

    def test_validation_rejects_nonstandard_detached_or_head_only(self) -> None:
        valid = train_ppo.validate_actor_value_gradient_configuration(
            mode="actor_priority_value_pcgrad",
            ppo_objective="standard",
            trainable_scope="last_block_heads",
            value_trunk_gradient_scale=1.0,
        )
        self.assertEqual(valid, "actor_priority_value_pcgrad")
        self.assertEqual(
            train_ppo.validate_actor_value_gradient_configuration(
                mode="actor_priority_value_pcgrad",
                ppo_objective="standard",
                trainable_scope="last_two_blocks_heads",
                value_trunk_gradient_scale=1.0,
            ),
            "actor_priority_value_pcgrad",
        )
        invalids = (
            {
                "ppo_objective": "constrained",
                "trainable_scope": "last_block_heads",
                "value_trunk_gradient_scale": 1.0,
            },
            {
                "ppo_objective": "standard",
                "trainable_scope": "last_block_heads",
                "value_trunk_gradient_scale": 0.0,
            },
            {
                "ppo_objective": "standard",
                "trainable_scope": "heads",
                "value_trunk_gradient_scale": 1.0,
            },
        )
        for case in invalids:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    train_ppo.validate_actor_value_gradient_configuration(
                        mode="actor_priority_value_pcgrad",
                        **case,
                    )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            train_ppo.validate_actor_value_gradient_configuration(
                mode="actor_priority_value_pcgrad",
                ppo_objective="standard",
                trainable_scope="full",
                value_trunk_gradient_scale=1.0,
                audit_only=True,
                actor_reduction_audit_only=True,
            )

    def test_resume_mode_change_requires_reset_and_legacy_is_scalar(
        self,
    ) -> None:
        model = train_bc_orbit.EntityOptionPolicy(
            hash_size=128,
            categorical_dim=8,
            model_dim=16,
            layers=2,
            heads=4,
            dropout=0.0,
            max_state_entities=8,
        )
        _, _, manifest = train_ppo.configure_trainable_scope(
            model,
            "last_block_heads",
        )
        legacy = {
            "config": {
                "trainable_scope": "last_block_heads",
                "advantage_normalization": "global",
                "ppo_objective": "standard",
            }
        }
        train_ppo.validate_optimizer_resume_compatibility(
            legacy,
            "last_block_heads",
            manifest,
            requested_actor_value_gradient_mode="scalar",
        )
        with self.assertRaisesRegex(
            ValueError,
            "actor/value gradient mode mismatch",
        ):
            train_ppo.validate_optimizer_resume_compatibility(
                legacy,
                "last_block_heads",
                manifest,
                requested_actor_value_gradient_mode=(
                    "actor_priority_value_pcgrad"
                ),
            )

    def test_no_update_audit_preserves_all_state_and_observes_conflict(
        self,
    ) -> None:
        class TinyPolicy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shared = torch.nn.Parameter(torch.zeros(()))
                self.actor_only = torch.nn.Parameter(torch.zeros(()))
                self.value_head = torch.nn.Parameter(torch.zeros(()))

            def forward(
                self,
                batch: dict[str, torch.Tensor],
            ) -> dict[str, torch.Tensor]:
                x = batch["x"].float().flatten()
                return {
                    "actor_signal": self.shared + self.actor_only,
                    "value_logits": (
                        (self.shared + self.value_head) * x
                    ),
                }

        learner = TinyPolicy()
        reference = copy.deepcopy(learner)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        actor_parameters = [learner.shared, learner.actor_only]
        value_parameters = [learner.value_head]
        optimizer = torch.optim.AdamW(
            [
                {"params": actor_parameters, "lr": 1e-3},
                {"params": value_parameters, "lr": 1e-3},
            ]
        )
        transitions = [
            {
                "feature": {"x": 1.0},
                "advantage": -1.0,
                "outcome_target": 0.0,
                "old_log_prob": 0.0,
                "action_count": 1,
                "action": [0],
                "opponent_name": "gold",
            },
            {
                "feature": {"x": 2.0},
                "advantage": 1.0,
                "outcome_target": 0.0,
                "old_log_prob": 0.0,
                "action_count": 1,
                "action": [0],
                "opponent_name": "gold",
            },
        ]
        config = SimpleNamespace(
            actor_value_gradient_mode="actor_priority_value_pcgrad",
            actor_value_gradient_audit_only=True,
            actor_reduction_audit_only=False,
            ppo_objective="standard",
            trainable_scope="last_block_heads",
            value_trunk_gradient_scale=1.0,
            advantage_normalization="global",
            actor_reduction="transition_mean",
            opponent_base_quotas={},
            minibatch_size=2,
            policy_temperature=1.0,
            clip_ratio=0.15,
            value_coefficient=0.5,
            entropy_coefficient=0.001,
            updates=1,
            schedule_start_update=1,
            bc_kl_start=0.012,
            bc_kl_end=0.012,
        )

        def fake_collate(
            features: list[dict[str, float]],
            _model_config: dict[str, object],
        ) -> dict[str, torch.Tensor]:
            return {
                "x": torch.tensor(
                    [[row["x"]] for row in features],
                    dtype=torch.float32,
                )
            }

        def fake_ordered(
            outputs: dict[str, torch.Tensor],
            batch: dict[str, torch.Tensor],
            *_args: object,
            **_kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rows = batch["x"].shape[0]
            log_probs = (
                outputs["actor_signal"] * batch["x"].float().flatten()
            )
            entropy = outputs["actor_signal"].expand(rows) * 0.0
            return log_probs, entropy

        def fake_reference_kl(
            outputs: dict[str, torch.Tensor],
            _reference: dict[str, torch.Tensor],
            batch: dict[str, torch.Tensor],
            *_args: object,
            **_kwargs: object,
        ) -> torch.Tensor:
            return (
                outputs["actor_signal"].expand(batch["x"].shape[0]) * 0.0
            )

        with (
            mock.patch.object(
                train_ppo,
                "collate_features_cpu",
                side_effect=fake_collate,
            ),
            mock.patch.object(
                train_ppo,
                "model_forward",
                side_effect=lambda model, batch, _device, **_kwargs: model(
                    batch
                ),
            ),
            mock.patch.object(
                train_ppo,
                "ordered_action_log_prob_entropy",
                side_effect=fake_ordered,
            ),
            mock.patch.object(
                train_ppo,
                "reference_policy_kl",
                side_effect=fake_reference_kl,
            ),
        ):
            audit = train_ppo.audit_actor_value_gradients(
                learner,
                reference,
                optimizer,
                transitions,
                {},
                config,
                torch.device("cpu"),
                1,
                actor_parameters,
                value_parameters,
            )

        self.assertEqual(audit["status"], "completed_no_update")
        self.assertEqual(audit["optimizer_steps"], 0)
        self.assertEqual(audit["bc_replay_steps"], 0)
        self.assertGreater(audit["gradient"]["conflict_batches"], 0)
        self.assertTrue(audit["pre_registered_gate"]["training_authorized"])
        self.assertEqual(
            audit["state_integrity"]["learner_model_sha256_before"],
            audit["state_integrity"]["learner_model_sha256_after"],
        )
        self.assertEqual(
            audit["state_integrity"]["reference_model_sha256_before"],
            audit["state_integrity"]["reference_model_sha256_after"],
        )
        self.assertEqual(
            audit["state_integrity"]["optimizer_state_sha256_before"],
            audit["state_integrity"]["optimizer_state_sha256_after"],
        )
        self.assertTrue(
            audit["state_integrity"]["parameter_gradients_remain_none"]
        )
        self.assertGreater(
            audit["parameter_scope"]["shared_parameter_count"],
            0,
        )
        self.assertGreaterEqual(
            audit["gradient"]["min_projected_value_dot_actor"],
            -audit["gradient"]["max_numerical_tolerance"],
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in learner.parameters())
        )


class ConstrainedObjectiveStateTests(unittest.TestCase):
    def test_dual_updates_and_round_trips(self) -> None:
        state = train_ppo.ConstrainedObjectiveState(dual_value=1.0)
        audit = state.finish_update(
            guard_surrogate=-0.012,
            update=456,
            floor=-0.002,
            dual_learning_rate=0.05,
            dual_max=10.0,
        )
        self.assertAlmostEqual(audit["dual_after"], 1.0005)
        restored = train_ppo.ConstrainedObjectiveState.from_state_dict(
            state.state_dict(),
            dual_max=10.0,
        )
        self.assertEqual(restored, state)

    def test_pre_update_and_v1_resume_alignment(self) -> None:
        pre_update = train_ppo.ConstrainedObjectiveState(dual_value=1.0)
        self.assertEqual(
            train_ppo.align_constrained_objective_resume_state(
                pre_update,
                checkpoint_update=455,
                serialized_version=2,
            ),
            "pre_update",
        )
        migrated = train_ppo.ConstrainedObjectiveState(
            dual_value=1.2,
            completed_updates=3,
            last_guard_surrogate=-0.001,
        )
        self.assertEqual(
            train_ppo.align_constrained_objective_resume_state(
                migrated,
                checkpoint_update=458,
                serialized_version=1,
            ),
            "version1_migrated",
        )
        self.assertEqual(migrated.last_update, 458)

    def test_misaligned_v2_objective_state_is_rejected(self) -> None:
        state = train_ppo.ConstrainedObjectiveState(
            dual_value=1.0,
            completed_updates=1,
            last_guard_surrogate=0.0,
            last_update=456,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            train_ppo.align_constrained_objective_resume_state(
                state,
                checkpoint_update=457,
                serialized_version=2,
            )

    def test_normalization_change_requires_optimizer_reset(self) -> None:
        model = train_bc_orbit.EntityOptionPolicy(
            hash_size=128,
            categorical_dim=8,
            model_dim=16,
            layers=2,
            heads=4,
            dropout=0.0,
            max_state_entities=8,
        )
        _, _, manifest = train_ppo.configure_trainable_scope(model, "full")
        checkpoint = {
            "config": {
                "trainable_scope": "full",
                "advantage_normalization": "global",
                "ppo_objective": "standard",
            }
        }
        with self.assertRaisesRegex(ValueError, "normalization mismatch"):
            train_ppo.validate_optimizer_resume_compatibility(
                checkpoint,
                "full",
                manifest,
                requested_advantage_normalization="per_opponent",
                requested_ppo_objective="standard",
            )

    def test_temperature_change_requires_optimizer_reset(self) -> None:
        model = train_bc_orbit.EntityOptionPolicy(
            hash_size=128,
            categorical_dim=8,
            model_dim=16,
            layers=2,
            heads=4,
            dropout=0.0,
            max_state_entities=8,
        )
        _, _, manifest = train_ppo.configure_trainable_scope(model, "full")
        checkpoint = {
            "config": {
                "trainable_scope": "full",
                "advantage_normalization": "global",
                "ppo_objective": "standard",
                "policy_temperature": 1.0,
            }
        }
        with self.assertRaisesRegex(ValueError, "temperature mismatch"):
            train_ppo.validate_optimizer_resume_compatibility(
                checkpoint,
                "full",
                manifest,
                requested_policy_temperature=0.5,
            )

    def test_value_trunk_scale_change_requires_optimizer_reset(self) -> None:
        model = train_bc_orbit.EntityOptionPolicy(
            hash_size=128,
            categorical_dim=8,
            model_dim=16,
            layers=2,
            heads=4,
            dropout=0.0,
            max_state_entities=8,
        )
        _, _, manifest = train_ppo.configure_trainable_scope(model, "full")
        checkpoint = {
            "config": {
                "trainable_scope": "full",
                "advantage_normalization": "global",
                "ppo_objective": "standard",
                "value_trunk_gradient_scale": 1.0,
            }
        }
        with self.assertRaisesRegex(
            ValueError,
            "value-trunk gradient scale mismatch",
        ):
            train_ppo.validate_optimizer_resume_compatibility(
                checkpoint,
                "full",
                manifest,
                requested_value_trunk_gradient_scale=0.0,
            )

        legacy_checkpoint = {
            "config": {
                "trainable_scope": "full",
                "advantage_normalization": "global",
                "ppo_objective": "standard",
            }
        }
        train_ppo.validate_optimizer_resume_compatibility(
            legacy_checkpoint,
            "full",
            manifest,
            requested_value_trunk_gradient_scale=1.0,
        )


class ContextStratifiedReplayTests(unittest.TestCase):
    @staticmethod
    def cached_batch(
        contexts: list[int],
        marker_offset: int,
        option_width: int,
    ) -> dict[str, torch.Tensor]:
        rows = len(contexts)
        action_sequences = torch.full((rows, 4), -1, dtype=torch.long)
        action_sequences[:, 0] = 0
        return {
            "contexts": torch.tensor(contexts, dtype=torch.long),
            "markers": torch.arange(
                marker_offset,
                marker_offset + rows,
                dtype=torch.long,
            ),
            "option_mask": torch.ones(
                (rows, option_width),
                dtype=torch.bool,
            ),
            "targets": torch.ones((rows, option_width)),
            "action_sequences": action_sequences,
        }

    def test_zero_quota_preserves_legacy_batches_and_rng(self) -> None:
        batches = [self.cached_batch([34, 1, 1, 1], 10, 2)]
        random.seed(91)
        state = random.getstate()

        result = train_ppo.stratify_bc_replay_batches(
            batches,
            context34_rows_per_batch=0,
            seed=17,
        )

        self.assertIs(result, batches)
        self.assertEqual(random.getstate(), state)

    def test_exact_quota_keeps_batch_size_and_pads_cached_rows(self) -> None:
        batches = [
            self.cached_batch([34, 1, 1, 1], 10, 2),
            self.cached_batch([1, 34, 1, 1], 20, 3),
        ]
        random.seed(92)
        state = random.getstate()

        result = train_ppo.stratify_bc_replay_batches(
            batches,
            context34_rows_per_batch=2,
            seed=17,
        )

        self.assertEqual(random.getstate(), state)
        self.assertEqual(len(result), 2)
        for batch in result:
            self.assertEqual(batch["contexts"].shape[0], 4)
            self.assertEqual(
                int((batch["contexts"] == 34).sum()),
                2,
            )
            self.assertEqual(batch["option_mask"].shape[0], 4)
            self.assertTrue(
                (batch["action_sequences"][:, 1:] == -1).all()
            )
        context_markers = sorted(
            int(marker)
            for batch in result
            for marker, context in zip(
                batch["markers"],
                batch["contexts"],
            )
            if int(context) == 34
        )
        self.assertEqual(context_markers, [10, 10, 21, 21])

    def test_positive_quota_requires_both_context_pools(self) -> None:
        without_context = [
            self.cached_batch([1, 1, 1, 1], 10, 2)
        ]
        without_ordinary = [
            self.cached_batch([34, 34, 34, 34], 20, 2)
        ]
        with self.assertRaisesRegex(RuntimeError, "no context-34"):
            train_ppo.stratify_bc_replay_batches(
                without_context,
                context34_rows_per_batch=2,
                seed=17,
            )
        with self.assertRaisesRegex(RuntimeError, "no non-context-34"):
            train_ppo.stratify_bc_replay_batches(
                without_ordinary,
                context34_rows_per_batch=2,
                seed=17,
            )


class OrderedReplayTests(unittest.TestCase):
    @staticmethod
    def batch(sequence: list[int], context: int = 34) -> dict[str, torch.Tensor]:
        padded_sequence = torch.full(
            (1, train_bc_orbit.MAX_ACTION_COUNT),
            -1,
            dtype=torch.long,
        )
        padded_sequence[0, : len(sequence)] = torch.tensor(sequence)
        return {
            "option_mask": torch.tensor([[True, True]]),
            "targets": torch.tensor([[1.0, 1.0]]),
            "sample_weights": torch.ones(1),
            "action_counts": torch.tensor([2]),
            "action_sequences": padded_sequence,
            "min_counts": torch.tensor([2]),
            "max_counts": torch.tensor([2]),
            "contexts": torch.tensor([context]),
        }

    @staticmethod
    def outputs() -> dict[str, torch.Tensor]:
        return {
            "policy_logits": torch.tensor([[0.0, 2.0]], requires_grad=True),
            "count_logits": torch.zeros(
                (1, train_ppo.MAX_ACTION_COUNT + 1),
                requires_grad=True,
            ),
        }

    @staticmethod
    def weighted_batch() -> dict[str, torch.Tensor]:
        action_sequences = torch.full(
            (4, train_bc_orbit.MAX_ACTION_COUNT),
            -1,
            dtype=torch.long,
        )
        action_sequences[0, :2] = torch.tensor([0, 1])
        action_sequences[1, :2] = torch.tensor([1, 0])
        action_sequences[2, :2] = torch.tensor([1, 0])
        action_sequences[3, 0] = 0
        return {
            "option_mask": torch.ones((4, 2), dtype=torch.bool),
            "targets": torch.tensor(
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
            "sample_weights": torch.tensor([2.0, 3.0, 1.0, 4.0]),
            "action_counts": torch.tensor([2, 2, 2, 1]),
            "action_sequences": action_sequences,
            "min_counts": torch.tensor([2, 1, 2, 1]),
            "max_counts": torch.tensor([2, 2, 2, 1]),
            "contexts": torch.tensor([7, 22, 34, 8]),
        }

    @staticmethod
    def weighted_outputs() -> dict[str, torch.Tensor]:
        return {
            "policy_logits": torch.tensor(
                [[0.0, 2.0]] * 4,
                requires_grad=True,
            ),
            "count_logits": torch.zeros(
                (4, train_ppo.MAX_ACTION_COUNT + 1),
                requires_grad=True,
            ),
        }

    def test_set_loss_is_order_invariant(self) -> None:
        first, _ = train_ppo.bc_expert_actor_loss(
            self.outputs(),
            self.batch([1, 0]),
            loss_mode="set",
        )
        second, _ = train_ppo.bc_expert_actor_loss(
            self.outputs(),
            self.batch([0, 1]),
            loss_mode="set",
        )
        self.assertAlmostEqual(
            float(first.detach()),
            float(second.detach()),
            places=6,
        )

    def test_ordered_pl_loss_preserves_sequence(self) -> None:
        first, first_parts = train_ppo.bc_expert_actor_loss(
            self.outputs(),
            self.batch([1, 0]),
            loss_mode="ordered",
        )
        second, second_parts = train_ppo.bc_expert_actor_loss(
            self.outputs(),
            self.batch([0, 1]),
            loss_mode="ordered",
        )
        self.assertAlmostEqual(
            float(first_parts["selection_loss"].detach()),
            0.126928,
            places=5,
        )
        self.assertAlmostEqual(
            float(second_parts["selection_loss"].detach()),
            2.126928,
            places=5,
        )
        self.assertLess(float(first.detach()), float(second.detach()))

    def test_hybrid_only_uses_order_for_context_34(self) -> None:
        context_22_first, _ = train_ppo.bc_expert_actor_loss(
            self.outputs(),
            self.batch([1, 0], context=22),
            loss_mode="hybrid_ordered",
        )
        context_22_second, _ = train_ppo.bc_expert_actor_loss(
            self.outputs(),
            self.batch([0, 1], context=22),
            loss_mode="hybrid_ordered",
        )
        context_34_first, _ = train_ppo.bc_expert_actor_loss(
            self.outputs(),
            self.batch([1, 0], context=34),
            loss_mode="hybrid_ordered",
        )
        context_34_second, _ = train_ppo.bc_expert_actor_loss(
            self.outputs(),
            self.batch([0, 1], context=34),
            loss_mode="hybrid_ordered",
        )
        self.assertAlmostEqual(
            float(context_22_first.detach()),
            float(context_22_second.detach()),
            places=6,
        )
        self.assertLess(
            float(context_34_first.detach()),
            float(context_34_second.detach()),
        )

    def test_fixed_multi_action_weight_is_selection_only(self) -> None:
        baseline_outputs = self.weighted_outputs()
        baseline, baseline_parts = train_ppo.bc_expert_actor_loss(
            baseline_outputs,
            self.weighted_batch(),
            loss_mode="ordered",
            order_context_weight=8.0,
            non_context34_fixed_multi_action_order_weight=1.0,
        )
        baseline.backward()
        baseline_policy_grad = (
            baseline_outputs["policy_logits"].grad.detach().clone()
        )
        baseline_count_grad = (
            baseline_outputs["count_logits"].grad.detach().clone()
        )

        legacy_outputs = self.weighted_outputs()
        legacy, _ = train_ppo.bc_expert_actor_loss(
            legacy_outputs,
            self.weighted_batch(),
            loss_mode="ordered",
            order_context_weight=8.0,
        )
        legacy.backward()
        self.assertTrue(torch.equal(baseline.detach(), legacy.detach()))
        self.assertTrue(
            torch.equal(
                baseline_policy_grad,
                legacy_outputs["policy_logits"].grad,
            )
        )
        self.assertTrue(
            torch.equal(
                baseline_count_grad,
                legacy_outputs["count_logits"].grad,
            )
        )

        weighted_outputs = self.weighted_outputs()
        weighted, weighted_parts = train_ppo.bc_expert_actor_loss(
            weighted_outputs,
            self.weighted_batch(),
            loss_mode="ordered",
            order_context_weight=8.0,
            non_context34_fixed_multi_action_order_weight=2.0,
        )
        weighted.backward()

        high = 2.1269280110429727
        low = 0.1269280110429726
        expected_baseline = (
            2.0 * high + 3.0 * low + 8.0 * low + 4.0 * high
        ) / 17.0
        expected_weighted = (
            4.0 * high + 3.0 * low + 8.0 * low + 4.0 * high
        ) / 19.0
        self.assertAlmostEqual(
            float(baseline_parts["selection_loss"].detach()),
            expected_baseline,
            places=6,
        )
        self.assertAlmostEqual(
            float(weighted_parts["selection_loss"].detach()),
            expected_weighted,
            places=6,
        )
        self.assertEqual(
            float(weighted_parts["selection_base_weight_sum"].detach()),
            10.0,
        )
        self.assertEqual(
            float(weighted_parts["selection_effective_weight_sum"].detach()),
            19.0,
        )
        self.assertEqual(
            float(
                weighted_parts[
                    "non_context34_fixed_multi_action_base_weight_sum"
                ].detach()
            ),
            2.0,
        )
        self.assertEqual(
            float(
                weighted_parts[
                    "non_context34_fixed_multi_action_effective_weight_sum"
                ].detach()
            ),
            4.0,
        )
        self.assertAlmostEqual(
            float(
                weighted_parts[
                    "non_context34_fixed_multi_action_effective_weight_share"
                ].detach()
            ),
            4.0 / 19.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(baseline_parts["count_loss"].detach()),
            float(weighted_parts["count_loss"].detach()),
            places=7,
        )
        self.assertAlmostEqual(
            float(baseline_parts["context_34_ordered_loss"].detach()),
            float(weighted_parts["context_34_ordered_loss"].detach()),
            places=7,
        )
        self.assertTrue(
            torch.equal(
                baseline_count_grad,
                weighted_outputs["count_logits"].grad,
            )
        )
        self.assertFalse(
            torch.equal(
                baseline_policy_grad,
                weighted_outputs["policy_logits"].grad,
            )
        )

    def test_fixed_multi_action_weight_rejects_invalid_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires ordered replay loss"):
            train_ppo.bc_expert_actor_loss(
                self.outputs(),
                self.batch([1, 0], context=7),
                loss_mode="hybrid_ordered",
                non_context34_fixed_multi_action_order_weight=2.0,
            )
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite and positive",
                ):
                    train_ppo.bc_expert_actor_loss(
                        self.outputs(),
                        self.batch([1, 0], context=7),
                        loss_mode="ordered",
                        non_context34_fixed_multi_action_order_weight=invalid,
                    )

    def test_duplicate_sequence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            train_ppo.bc_expert_actor_loss(
                self.outputs(),
                self.batch([1, 1]),
                loss_mode="ordered",
            )

    def test_featurizer_rejects_duplicate_actions(self) -> None:
        row = {
            "observation": {"select": {"option": [{}, {}]}},
            "action": [1, 1],
        }
        self.assertIsNone(
            train_bc_orbit.featurize_row(
                row,
                hash_size=64,
                max_state_entities=8,
            )
        )

    def test_featurizer_and_collate_preserve_action_order(self) -> None:
        row = {
            "observation": {
                "select": {
                    "option": [{}, {}],
                    "minCount": 2,
                    "maxCount": 2,
                    "context": 34,
                }
            },
            "action": [1, 0],
        }
        feature = train_bc_orbit.featurize_row(
            row,
            hash_size=64,
            max_state_entities=8,
        )
        self.assertIsNotNone(feature)
        assert feature is not None
        self.assertEqual(feature["action_sequence"], [1, 0])
        self.assertEqual(feature["targets"], [1.0, 1.0])
        self.assertEqual(feature["action_count"], 2)
        batch = train_bc_orbit.collate_decisions(
            [feature],
            max_state_entities=8,
            entity_fields=12,
            option_fields=12,
        )
        self.assertEqual(
            batch["action_sequences"][0, :4].tolist(),
            [1, 0, -1, -1],
        )


class TrainableScopeTests(unittest.TestCase):
    @staticmethod
    def model(layers: int = 2) -> train_bc_orbit.EntityOptionPolicy:
        return train_bc_orbit.EntityOptionPolicy(
            hash_size=128,
            categorical_dim=8,
            model_dim=16,
            layers=layers,
            heads=4,
            dropout=0.0,
            max_state_entities=8,
        )

    def test_last_block_heads_freezes_earlier_trunk(self) -> None:
        model = self.model()
        actor, value, manifest = train_ppo.configure_trainable_scope(
            model,
            "last_block_heads",
        )
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(any(name.startswith("transformer.layers.1.") for name in trainable_names))
        self.assertFalse(any(name.startswith("transformer.layers.0.") for name in trainable_names))
        self.assertTrue(any(name.startswith("transformer.norm.") for name in trainable_names))
        self.assertTrue(any(name.startswith("actor_query.") for name in trainable_names))
        self.assertTrue(any(name.startswith("value_head.") for name in trainable_names))
        self.assertEqual(len(actor), manifest["actor_tensor_count"])
        self.assertEqual(len(value), manifest["value_tensor_count"])

    def test_last_two_blocks_heads_freezes_all_earlier_trunk(self) -> None:
        model = self.model(layers=4)
        _, _, old_manifest = train_ppo.configure_trainable_scope(
            model,
            "last_block_heads",
        )
        self.assertEqual(old_manifest["trainable_parameter_count"], 5219)
        self.assertEqual(old_manifest["actor_parameter_count"], 4930)
        self.assertEqual(old_manifest["value_parameter_count"], 289)
        self.assertEqual(old_manifest["actor_tensor_count"], 24)
        self.assertEqual(old_manifest["value_tensor_count"], 4)
        actor, value, manifest = train_ppo.configure_trainable_scope(
            model,
            "last_two_blocks_heads",
        )
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            any(
                name.startswith("transformer.layers.3.")
                for name in trainable_names
            )
        )
        self.assertTrue(
            any(
                name.startswith("transformer.layers.2.")
                for name in trainable_names
            )
        )
        self.assertFalse(
            any(
                name.startswith(
                    ("transformer.layers.0.", "transformer.layers.1.")
                )
                for name in trainable_names
            )
        )
        self.assertFalse(
            any(
                name.startswith(
                    (
                        "categorical_embedding.",
                        "numeric_projection.",
                        "state_position.",
                        "option_position.",
                        "kind_embedding.",
                        "state_input.",
                        "option_input.",
                    )
                )
                for name in trainable_names
            )
        )
        self.assertTrue(
            any(
                name.startswith("transformer.norm.")
                for name in trainable_names
            )
        )
        self.assertTrue(
            any(name.startswith("actor_query.") for name in trainable_names)
        )
        self.assertTrue(
            any(name.startswith("value_head.") for name in trainable_names)
        )
        self.assertEqual(manifest["scope"], "last_two_blocks_heads")
        self.assertEqual(manifest["trainable_parameter_count"], 8499)
        self.assertEqual(manifest["actor_parameter_count"], 8210)
        self.assertEqual(manifest["value_parameter_count"], 289)
        self.assertEqual(manifest["actor_tensor_count"], 36)
        self.assertEqual(manifest["value_tensor_count"], 4)
        self.assertEqual(len(actor), manifest["actor_tensor_count"])
        self.assertEqual(len(value), manifest["value_tensor_count"])

    def test_last_two_blocks_heads_requires_two_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires at least 2"):
            train_ppo.configure_trainable_scope(
                self.model(layers=1),
                "last_two_blocks_heads",
            )

    def test_scope_change_requires_optimizer_reset(self) -> None:
        model = self.model()
        _, _, manifest = train_ppo.configure_trainable_scope(
            model,
            "last_block_heads",
        )
        checkpoint = {"config": {"trainable_scope": "full"}}
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            train_ppo.validate_optimizer_resume_compatibility(
                checkpoint,
                "last_block_heads",
                manifest,
            )
        self.assertEqual(
            train_ppo.checkpoint_trainable_scope(
                {
                    "config": {
                        "trainable_scope": "last_two_blocks_heads"
                    }
                }
            ),
            "last_two_blocks_heads",
        )

    def test_last_block_to_last_two_blocks_requires_optimizer_reset(
        self,
    ) -> None:
        model = self.model(layers=3)
        _, _, manifest = train_ppo.configure_trainable_scope(
            model,
            "last_two_blocks_heads",
        )
        checkpoint = {
            "config": {"trainable_scope": "last_block_heads"}
        }
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            train_ppo.validate_optimizer_resume_compatibility(
                checkpoint,
                "last_two_blocks_heads",
                manifest,
            )


if __name__ == "__main__":
    unittest.main()
