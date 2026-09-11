#!/usr/bin/env python3
"""Terminal-reward PPO for the mode-aware autoregressive PTCG policy.

This keeps the mature official-engine rollout implementation from
``train_ppo.py`` but supplies action-sequence sampling and PPO likelihoods
that match ``ModeAwareARPolicy``.  It deliberately does not reinterpret the
pointer model as the legacy independent option-logit policy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import hashlib
import json
import math
import multiprocessing
import os
import random
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluate_mode_ar_vs_submission as mode_eval
import evaluate_ppo_head_to_head as h2h
import train_bc_mode_ar_v7 as bc
import train_ppo as legacy


FEATURE_VERSION = "ptcg-mode-ar-ppo-terminal01-v1"
INFERENCE_MARKER = "__ptcg_mode_ar_ppo_inference_v1__"


@dataclass(frozen=True)
class Route:
    name: str
    learner_deck: Path


@dataclass(frozen=True)
class TrainConfig:
    bc_checkpoint: str
    meta_pool: str
    route: str
    learner_deck: str
    output_dir: str
    updates: int
    environments: int
    games_per_update: int
    ppo_epochs: int
    minibatch_size: int
    actor_learning_rate: float
    value_learning_rate: float
    weight_decay: float
    gamma: float
    gae_lambda: float
    policy_temperature: float
    clip_ratio: float
    value_coefficient: float
    entropy_coefficient: float
    anchor_kl_coefficient: float
    target_kl: float
    max_grad_norm: float
    league_probability: float
    fixed_meta_probability: float
    inverse_meta_probability: float
    opponent_window_games: int
    opponent_inverse_min_factor: float
    opponent_inverse_max_factor: float
    initialize_opponent_window_from_anchor: bool
    champion_gate_interval: int
    champion_gate_games: int
    champion_gate_min_win_rate: float
    champion_gate_error_margin: float
    bc_eval_games: int
    evaluation_workers: int
    checkpoint_interval: int
    candidate_snapshot_interval: int
    rollback_on_gate_failure: bool
    reset_optimizer_on_gate_rollback: bool
    max_game_decisions: int
    seed: int
    device: str


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    embedded = checkpoint.get("model_config")
    if isinstance(embedded, Mapping):
        source = dict(embedded)
        required = (
            "hash_size",
            "categorical_dim",
            "model_dim",
            "layers",
            "heads",
            "dropout",
            "max_state_entities",
            "entity_fields",
            "option_fields",
        )
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"embedded model_config is missing {missing}")
        return {
            key: (
                float(source[key])
                if key == "dropout"
                else int(source[key])
            )
            for key in required
        }
    return mode_eval.mode_ar_model_config(dict(checkpoint))


def instantiate_model(
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> bc.ModeAwareARPolicy:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint is missing model_state_dict")
    option_positions = state.get("option_position.weight")
    if not isinstance(option_positions, torch.Tensor):
        raise ValueError("checkpoint is missing option position weights")
    config = model_config(checkpoint)
    model = bc.ModeAwareARPolicy(
        hash_size=config["hash_size"],
        categorical_dim=config["categorical_dim"],
        model_dim=config["model_dim"],
        layers=config["layers"],
        heads=config["heads"],
        dropout=config["dropout"],
        max_state_entities=config["max_state_entities"],
        max_options=int(option_positions.shape[0]),
    )
    model.load_state_dict(dict(state), strict=True)
    model.to(device)
    return model


def load_anchor(
    path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], bc.ModeAwareARPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("BC anchor is not a checkpoint mapping")
    feature_version = checkpoint.get("feature_version")
    accepted_versions = {bc.FEATURE_VERSION, FEATURE_VERSION}
    if feature_version not in accepted_versions:
        raise ValueError(
            "Expected a mode-aware BC/PPO checkpoint, got "
            f"{feature_version!r}"
        )
    if (
        feature_version == FEATURE_VERSION
        and checkpoint.get("bc_feature_version") != bc.FEATURE_VERSION
    ):
        raise ValueError("PPO continuation checkpoint has incompatible BC features")
    model = instantiate_model(checkpoint, device)
    model.eval()
    return checkpoint, model, model_config(checkpoint)


def value_probability(value_prediction: torch.Tensor) -> torch.Tensor:
    """Map the BC head's signed outcome regression to rollout V(s) in [0, 1]."""

    return ((value_prediction.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def passes_champion_gate(
    win_rate: float,
    minimum_win_rate: float,
    error_margin: float,
) -> bool:
    """Gate on the point estimate while validating its reported error band."""

    values = (win_rate, minimum_win_rate, error_margin)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Champion gate values must be finite")
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("Champion gate win rate must be in [0, 1]")
    if not 0.0 <= minimum_win_rate <= 1.0:
        raise ValueError("Champion gate minimum must be in [0, 1]")
    if not 0.0 <= error_margin <= 1.0:
        raise ValueError("Champion gate error margin must be in [0, 1]")
    return win_rate >= minimum_win_rate


def synchronize_ppo_epoch_kl(
    mean_approx_kl: float,
    device: torch.device,
) -> float:
    """Hook used by the DDP launcher to make early stopping rank-consistent."""

    del device
    return float(mean_approx_kl)


class MixedOpponentSamplingReweighter(legacy.OpponentSamplingReweighter):
    """Mix frozen recent-meta sampling with inverse-win hard mining.

    ``fixed_meta_probability`` and ``inverse_meta_probability`` are marginal
    probabilities over all games.  ``collect_rollout`` separately applies
    ``league_probability``; consequently this class returns their normalized
    mixture conditional on choosing a frozen opponent.
    """

    def __init__(
        self,
        *,
        opponent_names: list[str],
        meta_weights: dict[str, float],
        window_games: int,
        inverse_min_factor: float,
        inverse_max_factor: float,
        fixed_meta_probability: float,
        inverse_meta_probability: float,
    ) -> None:
        super().__init__(
            opponent_names=opponent_names,
            meta_weights=meta_weights,
            window_games=window_games,
            inverse_min_factor=inverse_min_factor,
            inverse_max_factor=inverse_max_factor,
        )
        self.fixed_meta_probability = float(fixed_meta_probability)
        self.inverse_meta_probability = float(inverse_meta_probability)
        if (
            not math.isfinite(self.fixed_meta_probability)
            or not math.isfinite(self.inverse_meta_probability)
            or self.fixed_meta_probability < 0.0
            or self.inverse_meta_probability < 0.0
            or self.fixed_meta_probability + self.inverse_meta_probability
            <= 0.0
        ):
            raise ValueError("Invalid fixed/inverse opponent mixture")

    def audit(self) -> dict[str, dict[str, float | int]]:
        rows = super().audit()
        meta_total = sum(float(row["meta_weight"]) for row in rows.values())
        inverse_total = sum(
            float(row["combined_weight"]) for row in rows.values()
        )
        frozen_probability = (
            self.fixed_meta_probability + self.inverse_meta_probability
        )
        for row in rows.values():
            fixed_probability = float(row["meta_weight"]) / meta_total
            inverse_probability = (
                float(row["combined_weight"]) / inverse_total
            )
            conditional_probability = (
                self.fixed_meta_probability * fixed_probability
                + self.inverse_meta_probability * inverse_probability
            ) / frozen_probability
            row.update(
                {
                    "fixed_meta_probability": fixed_probability,
                    "inverse_meta_probability": inverse_probability,
                    "sampling_probability_given_frozen_opponent": (
                        conditional_probability
                    ),
                    "marginal_fixed_stream_probability": (
                        self.fixed_meta_probability * fixed_probability
                    ),
                    "marginal_inverse_stream_probability": (
                        self.inverse_meta_probability * inverse_probability
                    ),
                    "marginal_frozen_opponent_probability": (
                        frozen_probability * conditional_probability
                    ),
                }
            )
        return rows

    def sampling_weights(
        self,
        opponents: list[legacy.FrozenOpponent],
    ) -> list[float]:
        audit = self.audit()
        weights = [
            (
                float(
                    audit[opponent.name][
                        "sampling_probability_given_frozen_opponent"
                    ]
                )
                if opponent.permanent and opponent.name in audit
                else 0.0
            )
            for opponent in opponents
        ]
        if sum(weights) <= 0.0:
            raise RuntimeError("Mixed opponent weights sum to zero")
        return weights

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "version": 2,
                "fixed_meta_probability": self.fixed_meta_probability,
                "inverse_meta_probability": self.inverse_meta_probability,
            }
        )
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_fixed = state.get("fixed_meta_probability")
        saved_inverse = state.get("inverse_meta_probability")
        if saved_fixed is not None and not math.isclose(
            float(saved_fixed), self.fixed_meta_probability, abs_tol=1e-12
        ):
            raise ValueError("Fixed-meta sampling probability mismatch")
        if saved_inverse is not None and not math.isclose(
            float(saved_inverse), self.inverse_meta_probability, abs_tol=1e-12
        ):
            raise ValueError("Inverse-meta sampling probability mismatch")
        super().load_state_dict(state)


def _advance_decoder(
    model: bc.ModeAwareARPolicy,
    option_vectors: torch.Tensor,
    previous: torch.Tensor,
    hidden: torch.Tensor,
    chosen: torch.Tensor,
    chose_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gathered = option_vectors.gather(
        1,
        chosen.view(chosen.shape[0], 1, 1).expand(-1, 1, model.model_dim),
    ).squeeze(1)
    previous = torch.where(chose_action.unsqueeze(1), gathered, previous)
    return previous, hidden


@torch.no_grad()
def sample_mode_ar_actions(
    model: bc.ModeAwareARPolicy,
    encoded: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    deterministic: bool,
    temperature: float,
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    temperature = legacy.validate_policy_temperature(temperature)
    modes = encoded["modes"]
    batch_size, option_width = batch["targets"].shape
    option_count = batch["option_mask"].sum(1)
    effective_max = torch.minimum(batch["max_counts"], option_count)
    maximum_steps = max(int(effective_max.max().item()), 1)
    selected = torch.zeros_like(batch["option_mask"])
    previous_indices = torch.full(
        (batch_size,), -1, dtype=torch.long, device=option_count.device
    )
    previous = model.bos.unsqueeze(0).expand(batch_size, -1)
    hidden = encoded["select_summary"]
    finished = effective_max == 0
    counts = torch.zeros(batch_size, dtype=torch.long, device=option_count.device)
    log_probs = torch.zeros(batch_size, device=option_count.device)
    entropies = torch.zeros(batch_size, device=option_count.device)
    sequences = torch.full(
        (batch_size, maximum_steps),
        -1,
        dtype=torch.long,
        device=option_count.device,
    )
    for step in range(maximum_steps + 1):
        if bool(finished.all()):
            break
        hidden, option_logits, stop_logits = model.decoder_step(
            previous,
            hidden,
            encoded["option_vectors"],
            encoded["select_summary"],
        )
        legal_options, stop_legal = bc.legal_decode_mask(
            batch, modes, selected, step, previous_indices
        )
        legal_options &= ~finished.unsqueeze(1)
        stop_legal &= ~finished
        logits = torch.cat((option_logits, stop_logits.unsqueeze(1)), dim=1)
        legal = torch.cat((legal_options, stop_legal.unsqueeze(1)), dim=1)
        legal[finished, option_width] = True
        scaled = (logits.float() / temperature).masked_fill(~legal, -1e9)
        distribution = torch.distributions.Categorical(logits=scaled)
        choice = scaled.argmax(1) if deterministic else distribution.sample()
        active = ~finished
        log_probs += distribution.log_prob(choice) * active
        entropies += distribution.entropy() * active
        chose_stop = choice == option_width
        chose_action = active & ~chose_stop
        safe_choice = choice.clamp_max(option_width - 1)
        if step < maximum_steps:
            sequences[:, step] = torch.where(
                chose_action,
                safe_choice,
                torch.full_like(safe_choice, -1),
            )
        selected |= F.one_hot(
            safe_choice, num_classes=option_width
        ).bool() & chose_action.unsqueeze(1)
        previous, _ = _advance_decoder(
            model,
            encoded["option_vectors"],
            previous,
            hidden,
            safe_choice,
            chose_action,
        )
        previous_indices = torch.where(
            chose_action, safe_choice, previous_indices
        )
        counts += chose_action.long()
        finished |= chose_stop | (counts >= effective_max)
    if not bool(finished.all()):
        raise RuntimeError("Mode-aware sampler failed to terminate")
    actions = [
        [int(value) for value in row.tolist() if int(value) >= 0]
        for row in sequences.cpu()
    ]
    if any(len(action) != int(count) for action, count in zip(actions, counts.cpu())):
        raise RuntimeError("Sampled action sequence/count mismatch")
    return (
        actions,
        log_probs,
        entropies,
        value_probability(encoded["value_predictions"]),
    )


def action_path_statistics(
    model: bc.ModeAwareARPolicy,
    batch: dict[str, torch.Tensor],
    action_sequences: torch.Tensor,
    action_counts: torch.Tensor,
    *,
    temperature: float,
    reference_model: bc.ModeAwareARPolicy | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher-force an action path and return logp, entropy, value, anchor KL."""

    temperature = legacy.validate_policy_temperature(temperature)
    encoded = model(batch)
    reference = None
    if reference_model is not None:
        with torch.no_grad():
            reference = reference_model(batch)
    modes = encoded["modes"]
    batch_size, option_width = batch["targets"].shape
    option_count = batch["option_mask"].sum(1)
    effective_max = torch.minimum(batch["max_counts"], option_count)
    needs_stop = (modes != bc.MODE_SINGLE) & (action_counts < effective_max)
    learned_steps = action_counts + needs_stop.long()
    maximum_steps = max(int(learned_steps.max().item()), 1)

    previous = model.bos.unsqueeze(0).expand(batch_size, -1)
    hidden = encoded["select_summary"]
    reference_previous = (
        reference_model.bos.unsqueeze(0).expand(batch_size, -1)
        if reference_model is not None
        else None
    )
    reference_hidden = reference["select_summary"] if reference is not None else None
    selected = torch.zeros_like(batch["option_mask"])
    previous_indices = torch.full(
        (batch_size,), -1, dtype=torch.long, device=option_count.device
    )
    total_log_prob = torch.zeros(batch_size, device=option_count.device)
    total_entropy = torch.zeros(batch_size, device=option_count.device)
    total_anchor_kl = torch.zeros(batch_size, device=option_count.device)
    for step in range(maximum_steps):
        hidden, option_logits, stop_logits = model.decoder_step(
            previous,
            hidden,
            encoded["option_vectors"],
            encoded["select_summary"],
        )
        legal_options, stop_legal = bc.legal_decode_mask(
            batch, modes, selected, step, previous_indices
        )
        legal = torch.cat((legal_options, stop_legal.unsqueeze(1)), dim=1)
        logits = torch.cat((option_logits, stop_logits.unsqueeze(1)), dim=1)
        current_log_distribution = F.log_softmax(
            (logits.float() / temperature).masked_fill(~legal, -1e9), dim=1
        )
        current_distribution = current_log_distribution.exp()
        active = step < learned_steps
        action_step = step < action_counts
        target = torch.where(
            action_step,
            action_sequences[:, step].clamp_min(0),
            torch.full_like(action_counts, option_width),
        )
        chosen_legal = legal.gather(1, target.unsqueeze(1)).squeeze(1)
        if not bool(chosen_legal[active].all()):
            raise ValueError("rollout action violates mode-aware decode constraints")
        total_log_prob += (
            current_log_distribution.gather(1, target.unsqueeze(1)).squeeze(1)
            * active
        )
        total_entropy += (
            -(current_distribution * current_log_distribution).sum(1) * active
        )

        if reference is not None and reference_model is not None:
            assert reference_previous is not None and reference_hidden is not None
            with torch.no_grad():
                (
                    reference_hidden,
                    reference_option_logits,
                    reference_stop_logits,
                ) = reference_model.decoder_step(
                    reference_previous,
                    reference_hidden,
                    reference["option_vectors"],
                    reference["select_summary"],
                )
                reference_logits = torch.cat(
                    (
                        reference_option_logits,
                        reference_stop_logits.unsqueeze(1),
                    ),
                    dim=1,
                )
                reference_log_distribution = F.log_softmax(
                    (reference_logits.float() / temperature).masked_fill(
                        ~legal, -1e9
                    ),
                    dim=1,
                )
            total_anchor_kl += (
                (
                    current_distribution
                    * (current_log_distribution - reference_log_distribution)
                ).sum(1)
                * active
            )

        chose_action = active & action_step
        safe_target = target.clamp_max(option_width - 1)
        selected |= F.one_hot(
            safe_target, num_classes=option_width
        ).bool() & chose_action.unsqueeze(1)
        previous, _ = _advance_decoder(
            model,
            encoded["option_vectors"],
            previous,
            hidden,
            safe_target,
            chose_action,
        )
        if reference is not None and reference_model is not None:
            assert reference_previous is not None
            reference_previous, _ = _advance_decoder(
                reference_model,
                reference["option_vectors"],
                reference_previous,
                reference_hidden,
                safe_target,
                chose_action,
            )
        previous_indices = torch.where(
            chose_action, safe_target, previous_indices
        )
    normalization = learned_steps.float().clamp_min(1.0)
    return (
        total_log_prob,
        total_entropy / normalization,
        encoded["value_predictions"].float(),
        total_anchor_kl / normalization,
    )


@contextlib.contextmanager
def install_rollout_adapter() -> Iterator[None]:
    original_forward = legacy.model_forward
    original_sample = legacy.sample_ordered_actions

    def forward_dispatch(
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if isinstance(model, bc.ModeAwareARPolicy):
            if args or kwargs:
                raise TypeError("unexpected mode-aware rollout forward arguments")
            with torch.amp.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                encoded = model(batch)
            return {INFERENCE_MARKER: True, "model": model, "encoded": encoded}
        return original_forward(model, batch, device, *args, **kwargs)

    def sample_dispatch(
        outputs: Any,
        batch: dict[str, torch.Tensor],
        deterministic: bool,
        canonicalize_order: bool = False,
        temperature: float = 1.0,
    ) -> Any:
        if isinstance(outputs, dict) and outputs.get(INFERENCE_MARKER) is True:
            if canonicalize_order:
                raise ValueError("Mode-aware policy must not be post-sorted")
            return sample_mode_ar_actions(
                outputs["model"],
                outputs["encoded"],
                batch,
                deterministic=deterministic,
                temperature=temperature,
            )
        return original_sample(
            outputs,
            batch,
            deterministic=deterministic,
            canonicalize_order=canonicalize_order,
            temperature=temperature,
        )

    legacy.model_forward = forward_dispatch
    legacy.sample_ordered_actions = sample_dispatch
    try:
        yield
    finally:
        legacy.model_forward = original_forward
        legacy.sample_ordered_actions = original_sample


def configure_trainable_parameters(
    model: bc.ModeAwareARPolicy,
) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[str, Any]]:
    model.requires_grad_(False)
    last_layer = model.layers - 1
    actor_prefixes = (
        "decoder.",
        "query.",
        "option_key.",
        "pointer_residual.",
        "stop_head.",
        "select_encoder.",
        "option_cross_attention.",
        "option_cross_norm.",
        "option_ffn.",
        "option_ffn_norm.",
        f"board_transformer.layers.{last_layer}.",
    )
    actor: list[nn.Parameter] = []
    value: list[nn.Parameter] = []
    actor_names: list[str] = []
    value_names: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith("value_head."):
            parameter.requires_grad_(True)
            value.append(parameter)
            value_names.append(name)
        elif name.startswith(actor_prefixes):
            parameter.requires_grad_(True)
            actor.append(parameter)
            actor_names.append(name)
    if not actor or not value:
        raise RuntimeError("failed to select actor/value PPO parameters")
    return actor, value, {
        "actor_tensors": len(actor),
        "actor_parameters": sum(p.numel() for p in actor),
        "value_tensors": len(value),
        "value_parameters": sum(p.numel() for p in value),
        "actor_names": actor_names,
        "value_names": value_names,
    }


def ppo_update(
    model: bc.ModeAwareARPolicy,
    reference_model: bc.ModeAwareARPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    config: TrainConfig,
    device: torch.device,
    update: int,
) -> dict[str, Any]:
    model.eval()
    reference_model.eval()
    advantages, advantage_audit = legacy.normalize_rollout_advantages(
        transitions, "per_opponent"
    )
    actor_multipliers, actor_audit = legacy.actor_reduction_multipliers(
        transitions,
        "episode_mean",
        expected_episode_count=legacy.kept_episode_count(transitions),
    )
    old_log_probs = torch.tensor(
        [float(row["old_log_prob"]) for row in transitions], dtype=torch.float32
    )
    outcomes = torch.tensor(
        [float(row["outcome_target"]) for row in transitions], dtype=torch.float32
    )
    action_counts = torch.tensor(
        [int(row["action_count"]) for row in transitions], dtype=torch.long
    )
    action_sequences = torch.full(
        (len(transitions), legacy.MAX_ACTION_COUNT), -1, dtype=torch.long
    )
    for row_index, transition in enumerate(transitions):
        action = transition["action"]
        action_sequences[row_index, : len(action)] = torch.tensor(action)
    rollout_batch = legacy.collate_features_cpu(
        [row["feature"] for row in transitions], model_cfg
    )

    progress = (update - 1) / max(config.updates - 1, 1)
    lr_factor = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))
    optimizer.param_groups[0]["lr"] = config.actor_learning_rate * lr_factor
    optimizer.param_groups[1]["lr"] = config.value_learning_rate * lr_factor
    stats: dict[str, float] = {
        key: 0.0
        for key in (
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "anchor_kl",
            "approx_kl",
            "clip_fraction",
            "grad_norm",
        )
    }
    rows_seen = 0
    optimizer_steps = 0
    early_stop = False
    started = time.time()
    trainable = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    for epoch in range(config.ppo_epochs):
        permutation = torch.randperm(len(transitions))
        epoch_kl_sum = 0.0
        epoch_rows = 0
        for start in range(0, len(transitions), config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            batch = {
                key: value.index_select(0, indices).to(device, non_blocking=True)
                for key, value in rollout_batch.items()
            }
            actions = action_sequences.index_select(0, indices).to(device)
            counts = action_counts.index_select(0, indices).to(device)
            old_logp = old_log_probs.index_select(0, indices).to(device)
            mb_advantage = advantages.index_select(0, indices).to(device)
            mb_multiplier = actor_multipliers.index_select(0, indices).to(device)
            signed_outcome = (
                outcomes.index_select(0, indices).to(device) * 2.0 - 1.0
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logp, entropy, value_raw, anchor_kl = action_path_statistics(
                    model,
                    batch,
                    actions,
                    counts,
                    temperature=config.policy_temperature,
                    reference_model=reference_model,
                )
                log_ratio = logp.float() - old_logp
                ratio = log_ratio.exp()
                unclipped = ratio * mb_advantage
                clipped = ratio.clamp(
                    1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                ) * mb_advantage
                surrogate = torch.minimum(unclipped, clipped)
                policy_loss = -(surrogate * mb_multiplier).mean()
                value_loss = F.mse_loss(value_raw.float(), signed_outcome)
                entropy_mean = entropy.float().mean()
                anchor_kl_mean = anchor_kl.float().mean()
                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    + config.anchor_kl_coefficient * anchor_kl_mean
                    - config.entropy_coefficient * entropy_mean
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite Mode-aware PPO loss")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_ratio
                ).float().mean()
            rows = int(indices.numel())
            values = {
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_mean,
                "anchor_kl": anchor_kl_mean,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "grad_norm": grad_norm,
            }
            for key, value in values.items():
                stats[key] += float(value.detach()) * rows
            rows_seen += rows
            epoch_kl_sum += float(approx_kl.detach()) * rows
            epoch_rows += rows
            optimizer_steps += 1
        epoch_mean_kl = synchronize_ppo_epoch_kl(
            epoch_kl_sum / max(epoch_rows, 1),
            device,
        )
        if epoch_rows and epoch_mean_kl > config.target_kl * 1.5:
            early_stop = True
            break
    return {
        **{key: value / max(rows_seen, 1) for key, value in stats.items()},
        "rows": rows_seen,
        "transitions": len(transitions),
        "optimizer_steps": optimizer_steps,
        "epochs_completed": epoch + 1,
        "early_stop": early_stop,
        "learning_rate_factor": lr_factor,
        "actor_learning_rate": optimizer.param_groups[0]["lr"],
        "value_learning_rate": optimizer.param_groups[1]["lr"],
        "advantage_normalization": advantage_audit,
        "actor_reduction": actor_audit,
        "outcome_mean": float(outcomes.mean()),
        "seconds": time.time() - started,
    }


def build_meta_opponents(
    meta_pool_path: Path,
    learner_deck_path: Path,
    reference_model: bc.ModeAwareARPolicy,
) -> tuple[list[legacy.FrozenOpponent], dict[str, float], dict[str, Any]]:
    meta = json.loads(meta_pool_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != "ptcg-recent-day-exact-deck-meta-pool-v1":
        raise ValueError("unexpected meta-pool schema")
    learner_hash = legacy.compute_deck_hash(legacy.read_deck(learner_deck_path))
    opponents: list[legacy.FrozenOpponent] = []
    weights: dict[str, float] = {}
    bindings: list[dict[str, Any]] = []
    learner_matches = 0
    for row in meta.get("opponents", []):
        deck_path = Path(str(row["deck_path"])).resolve()
        deck = legacy.read_deck(deck_path)
        deck_hash = legacy.compute_deck_hash(deck)
        if deck_hash != str(row["deck_hash"]):
            raise ValueError(f"meta-pool deck hash mismatch: {deck_path}")
        if deck_hash == learner_hash:
            name = "bc"
            learner_matches += 1
        else:
            name = f"bc@{deck_path.stem}"
        weight = float(row["pool_base_probability"])
        opponents.append(
            legacy.FrozenOpponent(
                name=name,
                model=reference_model,
                deck=deck,
                deck_hash=deck_hash,
                permanent=True,
                canonical_order=False,
            )
        )
        weights[name] = weight
        bindings.append(
            {
                "name": name,
                "rank": int(row["rank"]),
                "deck_path": str(deck_path),
                "deck_hash": deck_hash,
                "meta_probability": weight,
            }
        )
    if learner_matches != 1:
        raise ValueError(
            f"learner deck must occur once in meta pool, got {learner_matches}"
        )
    if len(opponents) != len(weights) or not opponents:
        raise ValueError("meta-pool opponent names are not unique")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("meta-pool weights do not sum to one")
    return opponents, weights, {
        "path": str(meta_pool_path.resolve()),
        "sha256": sha256_file(meta_pool_path),
        "dataset_date": meta.get("dataset_date"),
        "bindings": bindings,
    }


def evaluate_models(
    candidate: bc.ModeAwareARPolicy,
    opponent: bc.ModeAwareARPolicy,
    deck: list[int],
    config: dict[str, Any],
    device: torch.device,
    games: int,
    environments: int,
    max_game_decisions: int,
    workers: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    if workers <= 1:
        with mode_eval.install_mode_ar_inference_adapter():
            result = h2h.evaluate_head_to_head_with_seats(
                candidate,
                opponent,
                deck,
                config,
                device,
                games,
                min(environments, games),
                max_game_decisions,
                current_canonical_order=False,
                opponent_canonical_order=False,
                opponent_deck=deck,
                current_hybrid_order=False,
                opponent_hybrid_order=False,
                opponent_model_config=config,
            )
        result["parallel_cpu"] = {
            "enabled": False,
            "workers": 1,
            "seed": int(seed),
        }
    else:
        result = evaluate_models_parallel_cpu(
            candidate,
            opponent,
            deck,
            config,
            games,
            environments,
            max_game_decisions,
            workers,
            seed,
        )
    if int(result["invalid_games"]) != 0:
        raise RuntimeError("Mode-aware PPO gate requires zero invalid games")
    low, high = h2h.wilson_interval(int(result["wins"]), int(result["valid_games"]))
    result["wilson_95_low"] = low
    result["wilson_95_high"] = high
    return result


def split_even_game_shards(games: int, workers: int) -> list[int]:
    """Split an even game target into positive even, seat-balanced shards."""

    if games < 2 or games % 2:
        raise ValueError("Parallel evaluation games must be positive and even")
    if workers < 1:
        raise ValueError("Parallel evaluation workers must be positive")
    resolved_workers = min(workers, games // 2)
    pairs, remainder = divmod(games // 2, resolved_workers)
    return [2 * (pairs + int(index < remainder)) for index in range(resolved_workers)]


def _parallel_evaluation_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run one exact-seat evaluation shard in an isolated CPU process."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    worker_seed = int(payload["seed"])
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    device = torch.device("cpu")
    candidate_checkpoint = torch.load(
        str(payload["candidate_path"]), map_location="cpu", weights_only=False
    )
    opponent_checkpoint = torch.load(
        str(payload["opponent_path"]), map_location="cpu", weights_only=False
    )
    candidate = instantiate_model(candidate_checkpoint, device)
    opponent = instantiate_model(opponent_checkpoint, device)
    candidate.requires_grad_(False)
    opponent.requires_grad_(False)
    model_cfg = dict(payload["model_config"])
    worker_games = int(payload["games"])
    with mode_eval.install_mode_ar_inference_adapter():
        result = h2h.evaluate_head_to_head_with_seats(
            candidate,
            opponent,
            list(payload["deck"]),
            model_cfg,
            device,
            worker_games,
            min(int(payload["environments"]), worker_games),
            int(payload["max_game_decisions"]),
            current_canonical_order=False,
            opponent_canonical_order=False,
            opponent_deck=list(payload["deck"]),
            current_hybrid_order=False,
            opponent_hybrid_order=False,
            opponent_model_config=model_cfg,
        )
    result["worker_seed"] = worker_seed
    return result


def merge_parallel_evaluations(
    results: list[dict[str, Any]],
    games: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not results:
        raise ValueError("Parallel evaluation returned no worker results")
    stats: Counter[str] = Counter()
    for result in results:
        valid_games = int(result["valid_games"])
        stats["valid_games"] += valid_games
        for key in ("wins", "losses", "draws", "invalid_games"):
            stats[key] += int(result[key])
        stats["decisions"] += round(float(result["mean_decisions"]) * valid_games)
        stats["max_active_games"] += int(result["max_active_games"])
        for seat in ("0", "1"):
            row = result["by_candidate_seat"][seat]
            prefix = f"seat_{seat}_"
            for key in ("valid_games", "wins", "losses", "draws"):
                stats[prefix + key] += int(row[key])
            invalid = result["invalid_by_candidate_seat"][seat]
            for reason, count in invalid["by_reason"].items():
                stats[prefix + "invalid_reason_" + reason] += int(count)
        for reason, count in result["invalid_by_reason"].items():
            stats["invalid_reason_" + reason] += int(count)
        for message, count in result["invalid_error_messages"].items():
            stats["invalid_error_message::" + message] += int(count)
    merged = h2h.summarize_evaluation_stats(stats, games, elapsed_seconds)
    merged["parallel_cpu"] = {
        "enabled": True,
        "workers": len(results),
        "torch_threads_per_worker": 1,
        "shard_games": [int(result["valid_games"]) for result in results],
        "worker_seconds": [float(result["seconds"]) for result in results],
        "worker_seeds": [int(result["worker_seed"]) for result in results],
        "aggregate_wall_seconds": elapsed_seconds,
    }
    return merged


def evaluate_models_parallel_cpu(
    candidate: bc.ModeAwareARPolicy,
    opponent: bc.ModeAwareARPolicy,
    deck: list[int],
    config: dict[str, Any],
    games: int,
    environments: int,
    max_game_decisions: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    shards = split_even_game_shards(games, workers)
    worker_environments = max(1, environments // len(shards))
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="ptcg-mode-ar-eval-") as raw_tmp:
        tmp = Path(raw_tmp)
        candidate_path = tmp / "candidate.pt"
        opponent_path = tmp / "opponent.pt"
        atomic_torch_save(
            {
                "model_config": config,
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in candidate.state_dict().items()
                },
            },
            candidate_path,
        )
        atomic_torch_save(
            {
                "model_config": config,
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in opponent.state_dict().items()
                },
            },
            opponent_path,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        payloads = [
            {
                "candidate_path": str(candidate_path),
                "opponent_path": str(opponent_path),
                "model_config": config,
                "deck": list(deck),
                "games": shard_games,
                "environments": worker_environments,
                "max_game_decisions": max_game_decisions,
                "seed": int(seed) + 104729 * index,
            }
            for index, shard_games in enumerate(shards)
        ]
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(shards),
            mp_context=context,
        ) as executor:
            results = list(executor.map(_parallel_evaluation_worker, payloads))
    return merge_parallel_evaluations(results, games, time.time() - started)


def checkpoint_payload(
    *,
    model: bc.ModeAwareARPolicy,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    model_cfg: dict[str, Any],
    update: int,
    metrics: dict[str, Any],
    anchor_path: Path,
    anchor_sha256: str,
    deck_hash: str,
    reweighter: legacy.OpponentSamplingReweighter,
    champion_update: int,
) -> dict[str, Any]:
    return {
        "feature_version": FEATURE_VERSION,
        "bc_feature_version": bc.FEATURE_VERSION,
        "config": asdict(config),
        "model_config": model_cfg,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update": update,
        "metrics": metrics,
        "bc_anchor": {
            "path": str(anchor_path.resolve()),
            "sha256": anchor_sha256,
        },
        "learner_deck_hash": deck_hash,
        "opponent_sampling_state": reweighter.state_dict(),
        "champion_update": champion_update,
        "reward": {"win": 1.0, "loss": 0.0, "draw": 0.0},
        "action_distribution": "mode-aware autoregressive pointer with STOP",
    }


def restore_champion_after_failed_gate(
    current_model: bc.ModeAwareARPolicy,
    champion_model: bc.ModeAwareARPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    champion_update: int,
    reset_optimizer: bool,
) -> dict[str, Any]:
    """Restore the deployed champion after a rejected PPO candidate.

    The rejected weights are expected to have been saved separately before
    this function is called.  Clearing Adam moments is important because they
    were accumulated around the rejected candidate and would otherwise push
    the restored champion back toward that candidate on the next update.
    """

    optimizer_state_entries_before = len(optimizer.state)
    current_model.load_state_dict(champion_model.state_dict(), strict=True)
    current_model.eval()
    if reset_optimizer:
        optimizer.state.clear()
    return {
        "applied": True,
        "restored_champion_update": int(champion_update),
        "optimizer_reset": bool(reset_optimizer),
        "optimizer_state_entries_before": optimizer_state_entries_before,
        "optimizer_state_entries_after": len(optimizer.state),
    }


def train(config: TrainConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(config.device)
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "champions").mkdir()
    (output_dir / "candidates").mkdir()

    anchor_path = Path(config.bc_checkpoint).resolve()
    anchor_checkpoint, current_model, model_cfg = load_anchor(anchor_path, device)
    reference_model = copy.deepcopy(current_model).to(device)
    reference_model.requires_grad_(False)
    reference_model.eval()
    champion_model = copy.deepcopy(reference_model).to(device)
    champion_update = 0
    actor, value, trainable_audit = configure_trainable_parameters(current_model)
    optimizer = torch.optim.AdamW(
        [
            {"params": actor, "lr": config.actor_learning_rate},
            {"params": value, "lr": config.value_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    learner_deck_path = Path(config.learner_deck).resolve()
    learner_deck = legacy.read_deck(learner_deck_path)
    learner_deck_hash = legacy.compute_deck_hash(learner_deck)
    opponents, meta_weights, meta_audit = build_meta_opponents(
        Path(config.meta_pool), learner_deck_path, reference_model
    )
    reweighter = MixedOpponentSamplingReweighter(
        opponent_names=[opponent.name for opponent in opponents],
        meta_weights=meta_weights,
        window_games=config.opponent_window_games,
        inverse_min_factor=config.opponent_inverse_min_factor,
        inverse_max_factor=config.opponent_inverse_max_factor,
        fixed_meta_probability=config.fixed_meta_probability,
        inverse_meta_probability=config.inverse_meta_probability,
    )
    anchor_sampling_state = anchor_checkpoint.get("opponent_sampling_state")
    opponent_window_initialized_from_anchor = False
    if config.initialize_opponent_window_from_anchor:
        if not isinstance(anchor_sampling_state, Mapping):
            raise ValueError(
                "Requested opponent-window continuation but the anchor lacks "
                "opponent_sampling_state"
            )
        reweighter.load_state_dict(anchor_sampling_state)
        opponent_window_initialized_from_anchor = True
    rollout_config = SimpleNamespace(
        bc_opponent_probability=None,
        environments=config.environments,
        failed_attempt_as_loss=False,
        failed_loss_tail_transitions=32,
        gae_lambda=config.gae_lambda,
        games_per_update=config.games_per_update,
        gamma=config.gamma,
        history_opponent_weight=0.0,
        league_probability=config.league_probability,
        max_game_decisions=config.max_game_decisions,
        opponent_quota_seat_balance=False,
        opponent_sampling="per_game",
        opponent_weights={name: 1.0 for name in meta_weights},
        policy_temperature=config.policy_temperature,
        seed=config.seed,
        truncation_as_loss=False,
    )
    anchor_sha = sha256_file(anchor_path)
    run_config = {
        "schema_version": "ptcg-mode-ar-ppo-run-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "model_config": model_cfg,
        "anchor_epoch": anchor_checkpoint.get("epoch"),
        "anchor_sha256": anchor_sha,
        "learner_deck_hash": learner_deck_hash,
        "meta_pool": meta_audit,
        "sampling_protocol": {
            "fixed_recent_meta_probability": config.fixed_meta_probability,
            "inverse_recent_meta_probability": config.inverse_meta_probability,
            "selfplay_probability": 1.0 - config.league_probability,
            "frozen_opponent_probability": config.league_probability,
            "opponent_window_initialized_from_anchor": (
                opponent_window_initialized_from_anchor
            ),
        },
        "trainable": trainable_audit,
        "checkpoint_contract": {
            "interval_updates": config.checkpoint_interval,
            "candidate_snapshot_interval": config.candidate_snapshot_interval,
            "champion_gate_interval": config.champion_gate_interval,
            "champion_gate_games": config.champion_gate_games,
            "champion_gate_min_win_rate_inclusive": config.champion_gate_min_win_rate,
            "champion_gate_error_margin": config.champion_gate_error_margin,
            "champion_gate_required_point_estimate": (
                config.champion_gate_min_win_rate
            ),
            "champion_gate_error_interpretation": "point_estimate_plus_or_minus",
            "fixed_bc_eval_games": config.bc_eval_games,
            "evaluation_workers": config.evaluation_workers,
            "evaluation_device": (
                "cpu_multiprocess" if config.evaluation_workers > 1 else config.device
            ),
            "rollback_on_gate_failure": config.rollback_on_gate_failure,
            "reset_optimizer_on_gate_rollback": (
                config.reset_optimizer_on_gate_rollback
            ),
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    initial_metrics = {"status": "initial_bc_champion"}
    initial_payload = checkpoint_payload(
        model=current_model,
        optimizer=optimizer,
        config=config,
        model_cfg=model_cfg,
        update=0,
        metrics=initial_metrics,
        anchor_path=anchor_path,
        anchor_sha256=anchor_sha,
        deck_hash=learner_deck_hash,
        reweighter=reweighter,
        champion_update=0,
    )
    atomic_torch_save(initial_payload, output_dir / "best.pt")
    atomic_torch_save(initial_payload, output_dir / "champions/update-0000.pt")

    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for update in range(1, config.updates + 1):
            with install_rollout_adapter():
                transitions, rollout = legacy.collect_rollout(
                    current_model,
                    opponents,
                    learner_deck,
                    model_cfg,
                    rollout_config,
                    device,
                    update,
                    opponent_sampling_reweighter=reweighter,
                )
            optimization = ppo_update(
                current_model,
                reference_model,
                optimizer,
                transitions,
                model_cfg,
                config,
                device,
                update,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            champion_gate = None
            bc_evaluation = None
            promoted = False
            if update % config.champion_gate_interval == 0 or update == config.updates:
                champion_gate = evaluate_models(
                    current_model,
                    champion_model,
                    learner_deck,
                    model_cfg,
                    device,
                    config.champion_gate_games,
                    config.environments,
                    config.max_game_decisions,
                    config.evaluation_workers,
                    config.seed + update * 10_000 + 101,
                )
                champion_gate.update(
                    {
                        "candidate_update": update,
                        "champion_update_before": champion_update,
                        "minimum_win_rate": config.champion_gate_min_win_rate,
                        "fixed_error_margin": config.champion_gate_error_margin,
                        "error_band_low": max(
                            0.0,
                            float(champion_gate["win_rate"])
                            - config.champion_gate_error_margin,
                        ),
                        "error_band_high": min(
                            1.0,
                            float(champion_gate["win_rate"])
                            + config.champion_gate_error_margin,
                        ),
                        "required_point_estimate": config.champion_gate_min_win_rate,
                        "promotion_rule": "point_estimate_at_least_target",
                        "inclusive_threshold": True,
                    }
                )
                promoted = passes_champion_gate(
                    float(champion_gate["win_rate"]),
                    config.champion_gate_min_win_rate,
                    config.champion_gate_error_margin,
                )
                if promoted:
                    champion_model = copy.deepcopy(current_model).to(device)
                    champion_model.requires_grad_(False)
                    champion_model.eval()
                    champion_update = update
                champion_gate["promoted"] = promoted
                champion_gate["champion_update_after"] = champion_update
                bc_evaluation = evaluate_models(
                    current_model,
                    reference_model,
                    learner_deck,
                    model_cfg,
                    device,
                    config.bc_eval_games,
                    config.environments,
                    config.max_game_decisions,
                    config.evaluation_workers,
                    config.seed + update * 10_000 + 202,
                )
            result = {
                "update": update,
                "rollout": rollout,
                "optimization": optimization,
                "champion_gate": champion_gate,
                "evaluation_vs_initial_bc": bc_evaluation,
                "champion_update": champion_update,
            }
            snapshot_due = (
                config.candidate_snapshot_interval > 0
                and update % config.candidate_snapshot_interval == 0
            )
            if champion_gate is not None or snapshot_due:
                candidate_path = (
                    output_dir / f"candidates/update-{update:04d}.pt"
                )
                result["candidate_snapshot"] = {
                    "path": str(candidate_path.resolve()),
                    "saved_before_gate_rollback": True,
                    "trigger": (
                        "champion_gate"
                        if champion_gate is not None
                        else "periodic"
                    ),
                }
                candidate_payload = checkpoint_payload(
                    model=current_model,
                    optimizer=optimizer,
                    config=config,
                    model_cfg=model_cfg,
                    update=update,
                    metrics=result,
                    anchor_path=anchor_path,
                    anchor_sha256=anchor_sha,
                    deck_hash=learner_deck_hash,
                    reweighter=reweighter,
                    champion_update=champion_update,
                )
                candidate_payload["snapshot_kind"] = "pre_gate_candidate"
                atomic_torch_save(candidate_payload, candidate_path)
            if (
                champion_gate is not None
                and not promoted
                and config.rollback_on_gate_failure
            ):
                champion_gate["rollback"] = restore_champion_after_failed_gate(
                    current_model,
                    champion_model,
                    optimizer,
                    champion_update=champion_update,
                    reset_optimizer=config.reset_optimizer_on_gate_rollback,
                )
            metrics_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            metrics_file.flush()
            payload = checkpoint_payload(
                model=current_model,
                optimizer=optimizer,
                config=config,
                model_cfg=model_cfg,
                update=update,
                metrics=result,
                anchor_path=anchor_path,
                anchor_sha256=anchor_sha,
                deck_hash=learner_deck_hash,
                reweighter=reweighter,
                champion_update=champion_update,
            )
            if promoted:
                atomic_torch_save(payload, output_dir / "best.pt")
                atomic_torch_save(
                    payload, output_dir / f"champions/update-{update:04d}.pt"
                )
            if update % config.checkpoint_interval == 0 or update == config.updates:
                atomic_torch_save(
                    payload, output_dir / f"checkpoints/update-{update:04d}.pt"
                )
            log(
                f"route={config.route} update={update}/{config.updates} "
                f"games={rollout['valid_games']} transitions={rollout['transitions_kept']} "
                f"sps={rollout['decisions_per_second']:.0f} "
                f"policy_loss={optimization['policy_loss']:.5f} "
                f"approx_kl={optimization['approx_kl']:.6f} "
                f"anchor_kl={optimization['anchor_kl']:.6f}"
                + (
                    f" gate={champion_gate['win_rate']:.3f} promoted={promoted} "
                    f"vs_bc={bc_evaluation['win_rate']:.3f}"
                    if champion_gate is not None and bc_evaluation is not None
                    else ""
                )
            )
    final_payload = checkpoint_payload(
        model=current_model,
        optimizer=optimizer,
        config=config,
        model_cfg=model_cfg,
        update=config.updates,
        metrics=result,
        anchor_path=anchor_path,
        anchor_sha256=anchor_sha,
        deck_hash=learner_deck_hash,
        reweighter=reweighter,
        champion_update=champion_update,
    )
    atomic_torch_save(final_payload, output_dir / "last.pt")
    summary = {
        "schema_version": "ptcg-mode-ar-ppo-summary-v1",
        "completed_updates": config.updates,
        "champion_update": champion_update,
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--meta-pool", type=Path, required=True)
    parser.add_argument("--route", choices=("rmy", "dragapult"), required=True)
    parser.add_argument("--learner-deck", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--games-per-update", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-6)
    parser.add_argument("--value-learning-rate", type=float, default=1.5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--policy-temperature", type=float, default=0.8)
    parser.add_argument("--clip-ratio", type=float, default=0.10)
    parser.add_argument("--value-coefficient", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0005)
    parser.add_argument("--anchor-kl-coefficient", type=float, default=0.04)
    parser.add_argument("--target-kl", type=float, default=0.0005)
    parser.add_argument("--max-grad-norm", type=float, default=0.25)
    parser.add_argument("--league-probability", type=float, default=0.90)
    parser.add_argument("--fixed-meta-probability", type=float, default=0.0)
    parser.add_argument("--inverse-meta-probability", type=float, default=0.90)
    parser.add_argument("--opponent-window-games", type=int, default=200)
    parser.add_argument("--opponent-inverse-min-factor", type=float, default=0.5)
    parser.add_argument("--opponent-inverse-max-factor", type=float, default=2.5)
    parser.add_argument(
        "--initialize-opponent-window-from-anchor",
        action="store_true",
    )
    parser.add_argument("--champion-gate-interval", type=int, default=10)
    parser.add_argument("--champion-gate-games", type=int, default=200)
    parser.add_argument("--champion-gate-min-win-rate", type=float, default=0.54)
    parser.add_argument("--champion-gate-error-margin", type=float, default=0.0)
    parser.add_argument("--bc-eval-games", type=int, default=200)
    parser.add_argument("--evaluation-workers", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--candidate-snapshot-interval", type=int, default=0)
    parser.add_argument("--rollback-on-gate-failure", action="store_true")
    parser.add_argument(
        "--reset-optimizer-on-gate-rollback",
        action="store_true",
    )
    parser.add_argument("--max-game-decisions", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.bc_checkpoint, args.meta_pool, args.learner_deck):
        if not path.is_file():
            raise FileNotFoundError(path)
    positive = (
        "updates",
        "environments",
        "games_per_update",
        "ppo_epochs",
        "minibatch_size",
        "champion_gate_interval",
        "champion_gate_games",
        "bc_eval_games",
        "evaluation_workers",
        "checkpoint_interval",
        "max_game_decisions",
    )
    if any(int(getattr(args, name)) < 1 for name in positive):
        raise ValueError("all count and interval arguments must be positive")
    if args.candidate_snapshot_interval < 0:
        raise ValueError("candidate snapshot interval must be non-negative")
    if (
        args.reset_optimizer_on_gate_rollback
        and not args.rollback_on_gate_failure
    ):
        raise ValueError(
            "optimizer reset on gate rollback requires rollback on gate failure"
        )
    if args.champion_gate_games % 2 or args.bc_eval_games % 2:
        raise ValueError("H2H game counts must be even for exact seat balance")
    if not 0.0 < args.league_probability < 1.0:
        raise ValueError("league probability must be in (0, 1)")
    if args.fixed_meta_probability < 0.0:
        raise ValueError("fixed-meta probability must be non-negative")
    if args.inverse_meta_probability < 0.0:
        raise ValueError("inverse-meta probability must be non-negative")
    if not math.isclose(
        args.fixed_meta_probability + args.inverse_meta_probability,
        args.league_probability,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "fixed-meta + inverse-meta probabilities must equal league "
            "probability"
        )
    if not 0.0 <= args.champion_gate_min_win_rate <= 1.0:
        raise ValueError("champion gate win rate must be in [0, 1]")
    if not 0.0 <= args.champion_gate_error_margin <= 1.0:
        raise ValueError("champion gate error margin must be in [0, 1]")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    config = TrainConfig(
        **{
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
    )
    train(config)


if __name__ == "__main__":
    main()
