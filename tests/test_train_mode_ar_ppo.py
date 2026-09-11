from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import train_bc_mode_ar_v7 as bc  # noqa: E402
import train_bc_orbit as base  # noqa: E402
import train_mode_ar_ppo as ppo  # noqa: E402


def synthetic_batch() -> dict[str, torch.Tensor]:
    batch_size, options, fields, entities = 4, 4, 3, 2
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
        "action_sequences": torch.full(
            (batch_size, base.MAX_ACTION_COUNT), -1, dtype=torch.long
        ),
        "min_counts": torch.tensor([1, 0, 2, 2]),
        "max_counts": torch.tensor([1, 1, 3, 3]),
        "contexts": torch.tensor([0, 0, bc.ORDERED_CONTEXT, 0]),
        "sample_weights": torch.ones(batch_size),
        "win_targets": torch.zeros(batch_size),
    }


def tiny_model() -> bc.ModeAwareARPolicy:
    model = bc.ModeAwareARPolicy(
        hash_size=64,
        categorical_dim=16,
        model_dim=32,
        layers=1,
        heads=4,
        dropout=0.0,
        max_state_entities=2,
        max_options=4,
    )
    model.eval()
    return model


class ModeAwarePPOTests(unittest.TestCase):
    def test_even_game_shards_preserve_total_and_seat_balance(self) -> None:
        shards = ppo.split_even_game_shards(1000, 8)
        self.assertEqual(shards, [126, 126, 126, 126, 124, 124, 124, 124])
        self.assertEqual(sum(shards), 1000)
        self.assertTrue(all(value > 0 and value % 2 == 0 for value in shards))

    def test_champion_gate_uses_target_and_reports_margin_separately(self) -> None:
        self.assertTrue(ppo.passes_champion_gate(0.65, 0.60, 0.05))
        self.assertTrue(ppo.passes_champion_gate(0.60, 0.60, 0.05))
        self.assertFalse(ppo.passes_champion_gate(0.599, 0.60, 0.05))
        self.assertEqual(
            ppo.synchronize_ppo_epoch_kl(0.00125, torch.device("cpu")),
            0.00125,
        )

    def test_failed_gate_restores_champion_and_clears_optimizer_state(self) -> None:
        torch.manual_seed(30)
        champion = tiny_model()
        current = copy.deepcopy(champion)
        optimizer = torch.optim.AdamW(current.parameters(), lr=1e-3)
        loss = sum(parameter.square().mean() for parameter in current.parameters())
        loss.backward()
        optimizer.step()
        self.assertGreater(len(optimizer.state), 0)
        self.assertTrue(
            any(
                not torch.equal(current_value, champion.state_dict()[name])
                for name, current_value in current.state_dict().items()
            )
        )

        audit = ppo.restore_champion_after_failed_gate(
            current,
            champion,
            optimizer,
            champion_update=40,
            reset_optimizer=True,
        )

        self.assertEqual(audit["restored_champion_update"], 40)
        self.assertTrue(audit["optimizer_reset"])
        self.assertGreater(audit["optimizer_state_entries_before"], 0)
        self.assertEqual(audit["optimizer_state_entries_after"], 0)
        self.assertEqual(len(optimizer.state), 0)
        for name, current_value in current.state_dict().items():
            self.assertTrue(
                torch.equal(current_value, champion.state_dict()[name]),
                name,
            )

    def test_mixed_opponent_sampling_is_70_20_conditional_mixture(self) -> None:
        sampler = ppo.MixedOpponentSamplingReweighter(
            opponent_names=["common", "hard"],
            meta_weights={"common": 0.8, "hard": 0.2},
            window_games=200,
            inverse_min_factor=0.5,
            inverse_max_factor=2.5,
            fixed_meta_probability=0.7,
            inverse_meta_probability=0.2,
        )
        sampler.observe(
            {
                "common": ["win"] * 20,
                "hard": ["loss"] * 20,
            }
        )
        audit = sampler.audit()
        fixed_hard = audit["hard"]["fixed_meta_probability"]
        inverse_hard = audit["hard"]["inverse_meta_probability"]
        expected_hard = (0.7 * fixed_hard + 0.2 * inverse_hard) / 0.9
        self.assertAlmostEqual(
            audit["hard"]["sampling_probability_given_frozen_opponent"],
            expected_hard,
        )
        self.assertAlmostEqual(
            sum(
                row["marginal_frozen_opponent_probability"]
                for row in audit.values()
            ),
            0.9,
        )
        self.assertGreater(inverse_hard, fixed_hard)

    def test_mixed_sampler_accepts_legacy_window_state(self) -> None:
        legacy_sampler = ppo.legacy.OpponentSamplingReweighter(
            opponent_names=["a", "b"],
            meta_weights={"a": 0.6, "b": 0.4},
            window_games=4,
            inverse_min_factor=0.5,
            inverse_max_factor=2.5,
        )
        legacy_sampler.observe({"a": ["win", "loss"], "b": ["loss"]})
        sampler = ppo.MixedOpponentSamplingReweighter(
            opponent_names=["a", "b"],
            meta_weights={"a": 0.6, "b": 0.4},
            window_games=4,
            inverse_min_factor=0.5,
            inverse_max_factor=2.5,
            fixed_meta_probability=0.7,
            inverse_meta_probability=0.2,
        )
        sampler.load_state_dict(legacy_sampler.state_dict())
        self.assertEqual(
            sampler.outcome_window,
            legacy_sampler.outcome_window,
        )

    def test_sample_log_probability_matches_teacher_forcing(self) -> None:
        torch.manual_seed(31)
        model = tiny_model()
        reference = copy.deepcopy(model)
        batch = synthetic_batch()
        encoded = model(batch)
        actions, sampled_logp, entropy, values = ppo.sample_mode_ar_actions(
            model,
            encoded,
            batch,
            deterministic=False,
            temperature=0.8,
        )
        counts = torch.tensor([len(action) for action in actions])
        sequences = torch.full(
            (len(actions), base.MAX_ACTION_COUNT), -1, dtype=torch.long
        )
        for row, action in enumerate(actions):
            sequences[row, : len(action)] = torch.tensor(action)
        logp, replay_entropy, value_raw, anchor_kl = ppo.action_path_statistics(
            model,
            batch,
            sequences,
            counts,
            temperature=0.8,
            reference_model=reference,
        )
        self.assertTrue(torch.allclose(logp, sampled_logp, atol=1e-5))
        self.assertTrue(bool((entropy >= 0).all()))
        self.assertTrue(bool((replay_entropy >= 0).all()))
        self.assertTrue(bool((values >= 0).all() and (values <= 1).all()))
        self.assertEqual(value_raw.shape, (4,))
        self.assertTrue(torch.allclose(anchor_kl, torch.zeros(4), atol=1e-6))
        for action, minimum, maximum in zip(
            actions, batch["min_counts"], batch["max_counts"]
        ):
            self.assertGreaterEqual(len(action), int(minimum))
            self.assertLessEqual(len(action), int(maximum))
            self.assertEqual(len(action), len(set(action)))

    def test_actor_and_detached_value_head_receive_expected_gradients(self) -> None:
        torch.manual_seed(32)
        model = tiny_model()
        reference = copy.deepcopy(model)
        batch = synthetic_batch()
        encoded = model(batch)
        actions, _, _, _ = ppo.sample_mode_ar_actions(
            model,
            encoded,
            batch,
            deterministic=True,
            temperature=0.8,
        )
        counts = torch.tensor([len(action) for action in actions])
        sequences = torch.full(
            (len(actions), base.MAX_ACTION_COUNT), -1, dtype=torch.long
        )
        for row, action in enumerate(actions):
            sequences[row, : len(action)] = torch.tensor(action)
        logp, _, value_raw, anchor_kl = ppo.action_path_statistics(
            model,
            batch,
            sequences,
            counts,
            temperature=0.8,
            reference_model=reference,
        )
        value_loss = F.mse_loss(value_raw, torch.zeros_like(value_raw))
        value_loss.backward(retain_graph=True)
        self.assertIsNotNone(model.value_head[-1].weight.grad)
        self.assertIsNone(model.global_encoder[0].weight.grad)
        model.zero_grad(set_to_none=True)
        actor_loss = -logp.mean() + 0.04 * anchor_kl.mean()
        actor_loss.backward()
        self.assertIsNotNone(model.decoder.weight_hh.grad)

    def test_trainable_scope_and_rollout_adapter_restore(self) -> None:
        model = tiny_model()
        actor, value, audit = ppo.configure_trainable_parameters(model)
        self.assertGreater(sum(parameter.numel() for parameter in actor), 0)
        self.assertGreater(sum(parameter.numel() for parameter in value), 0)
        self.assertEqual(audit["actor_tensors"], len(actor))
        original_forward = ppo.legacy.model_forward
        original_sample = ppo.legacy.sample_ordered_actions
        batch = synthetic_batch()
        with ppo.install_rollout_adapter():
            outputs = ppo.legacy.model_forward(
                model, batch, torch.device("cpu")
            )
            actions, logp, entropy, values = ppo.legacy.sample_ordered_actions(
                outputs,
                batch,
                deterministic=False,
                temperature=0.8,
            )
            self.assertEqual(len(actions), 4)
            self.assertEqual(logp.shape, (4,))
            self.assertEqual(entropy.shape, (4,))
            self.assertEqual(values.shape, (4,))
        self.assertIs(ppo.legacy.model_forward, original_forward)
        self.assertIs(ppo.legacy.sample_ordered_actions, original_sample)


if __name__ == "__main__":
    unittest.main()
