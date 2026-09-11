#!/usr/bin/env python3
"""Run the mature terminal-reward PPO harness with the non-AR V7 policy.

The base PPO implementation predates the mode-routed optional-single head.
This adapter keeps the official-engine rollout, GAE, PPO, league snapshots,
recent-meta inverse weighting, and checkpointing, while replacing only the
model/feature contract and action-distribution math.  No recurrent module,
STOP decoder, or autoregressive neural forward pass is introduced.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

import bc_nonar_v7 as nonar
import train_ppo as legacy


PPO_FEATURE_VERSION = "ptcg-nonar-v7-ppo-terminal01-v1"
_BASE_FROZEN_COPY = legacy.frozen_copy
_BASE_REWEIGHTER = legacy.OpponentSamplingReweighter
_SHARED_FROZEN_MODELS: dict[str, torch.nn.Module] = {}


class League801010Reweighter(_BASE_REWEIGHTER):
    """Dynamic 80/10/10 league sampler with a diverse promoted history.

    Permanent ``best@deck`` bindings form the recent-meta stream and receive
    clipped inverse rolling-win adjustment. ``bc`` is the fixed mirror stream.
    The bootstrap ``last@deck`` anchor and every promoted non-permanent
    snapshot share the historical ten percent uniformly.
    """

    def audit(self) -> dict[str, dict[str, float | int | str]]:
        rows = super().audit()
        for name, row in rows.items():
            if name == "bc":
                row["league_stream"] = "fixed_bc_mirror"
                row["target_stream_mass"] = 0.10
            elif name.startswith("last@"):
                row["league_stream"] = "history_bootstrap"
                row["target_stream_mass"] = 0.10
            else:
                row["league_stream"] = "recent_meta_inverse_window"
                row["target_stream_mass"] = 0.80
        return rows

    def sampling_weights(
        self,
        opponents: list[legacy.FrozenOpponent],
    ) -> list[float]:
        audit = self.audit()
        mirror_indices = [
            index for index, opponent in enumerate(opponents)
            if opponent.permanent and opponent.name == "bc"
        ]
        history_indices = [
            index for index, opponent in enumerate(opponents)
            if (
                (opponent.permanent and opponent.name.startswith("last@"))
                or not opponent.permanent
            )
        ]
        meta_indices = [
            index for index, opponent in enumerate(opponents)
            if (
                opponent.permanent
                and opponent.name != "bc"
                and not opponent.name.startswith("last@")
            )
        ]
        if len(mirror_indices) != 1 or not history_indices or not meta_indices:
            raise RuntimeError(
                "80/10/10 league requires one bc mirror, at least one "
                "history policy, and at least one recent-meta opponent"
            )
        weights = [0.0] * len(opponents)
        weights[mirror_indices[0]] = 0.10
        history_weight = 0.10 / len(history_indices)
        for index in history_indices:
            weights[index] = history_weight
        meta_total = sum(
            float(audit[opponents[index].name]["combined_weight"])
            for index in meta_indices
        )
        if meta_total <= 0.0:
            raise RuntimeError("Recent-meta inverse weights sum to zero")
        for index in meta_indices:
            weights[index] = 0.80 * (
                float(audit[opponents[index].name]["combined_weight"])
                / meta_total
            )
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise RuntimeError("80/10/10 league weights do not sum to one")
        return weights

    def observe(self, outcome_sequences: Any) -> None:
        # Base rolling windows apply only to registered permanent bindings.
        # Promoted-history outcomes do not change their uniform 10% share.
        super().observe(
            {
                name: values
                for name, values in outcome_sequences.items()
                if name in self.opponent_names
            }
        )


def _optional_joint_logits(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    option_logits = outputs["policy_logits"].float() / temperature
    option_logits = option_logits.masked_fill(~batch["option_mask"], -1e9)
    skip_logits = outputs["skip_logits"].float().unsqueeze(1) / temperature
    return torch.cat((option_logits, skip_logits), dim=1)


@torch.no_grad()
def sample_nonar_actions(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    deterministic: bool,
    canonicalize_order: bool = False,
    temperature: float = 1.0,
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the static mode-routed policy with a joint optional SKIP."""

    del canonicalize_order  # Static ordered scores already define legal order.
    temperature = legacy.validate_policy_temperature(temperature)
    modes = nonar.selection_modes(batch)
    optional = modes == nonar.MODE_OPTIONAL_SINGLE
    option_logits = outputs["policy_logits"].float() / temperature
    count_logits = outputs["count_logits"].float() / temperature
    allowed_counts = legacy.count_allowed_mask(batch)
    if not bool(allowed_counts.any(dim=1).all()):
        raise ValueError("No allowed cardinality for at least one observation")
    masked_count_logits = count_logits.masked_fill(~allowed_counts, -1e9)
    fixed = batch["min_counts"] == batch["max_counts"]
    count_distribution = torch.distributions.Categorical(
        logits=masked_count_logits
    )
    counts = (
        masked_count_logits.argmax(dim=1)
        if deterministic
        else count_distribution.sample()
    )
    counts = torch.where(
        fixed,
        batch["min_counts"].clamp(0, legacy.MAX_ACTION_COUNT),
        counts,
    )
    counts = torch.minimum(counts, batch["option_mask"].sum(dim=1))
    count_log_prob = torch.where(
        fixed,
        torch.zeros_like(count_distribution.log_prob(counts)),
        count_distribution.log_prob(counts),
    )
    count_entropy = torch.where(
        fixed,
        torch.zeros_like(count_distribution.entropy()),
        count_distribution.entropy(),
    )

    optional_distribution = torch.distributions.Categorical(
        logits=_optional_joint_logits(outputs, batch, temperature)
    )
    optional_choice = (
        optional_distribution.logits.argmax(dim=1)
        if deterministic
        else optional_distribution.sample()
    )
    skip_index = option_logits.shape[1]
    counts = torch.where(
        optional,
        (optional_choice != skip_index).long(),
        counts,
    )
    log_probs = torch.where(
        optional,
        optional_distribution.log_prob(optional_choice),
        count_log_prob,
    )
    entropies = torch.where(
        optional,
        optional_distribution.entropy(),
        count_entropy,
    )

    # Select every row at a given sequence position in one CUDA operation.
    # The former row-by-row Categorical loop converted CUDA scalars to Python
    # for every choice, serializing the rollout on device synchronizations.
    batch_size, option_count = option_logits.shape
    action_matrix = torch.full(
        (batch_size, legacy.MAX_ACTION_COUNT),
        -1,
        dtype=torch.long,
        device=option_logits.device,
    )
    remaining = batch["option_mask"].clone()
    selection_log_prob = torch.zeros_like(log_probs)
    selection_entropy = torch.zeros_like(entropies)
    maximum_count = int(counts.max().detach().cpu())
    for step in range(maximum_count):
        active = (~optional) & (counts > step)
        logits = option_logits.masked_fill(~remaining, -1e9)
        distribution = torch.distributions.Categorical(logits=logits)
        chosen = (
            logits.argmax(dim=1)
            if deterministic
            else distribution.sample()
        )
        safe_chosen = torch.where(active, chosen, torch.zeros_like(chosen))
        action_matrix[:, step] = torch.where(
            active,
            chosen,
            action_matrix[:, step],
        )
        selection_log_prob = selection_log_prob + torch.where(
            active,
            distribution.log_prob(safe_chosen),
            torch.zeros_like(selection_log_prob),
        )
        selection_entropy = selection_entropy + torch.where(
            active & (remaining.sum(dim=1) > 1),
            distribution.entropy(),
            torch.zeros_like(selection_entropy),
        )
        previous_remaining = remaining.gather(1, safe_chosen.unsqueeze(1))
        remaining.scatter_(
            1,
            safe_chosen.unsqueeze(1),
            previous_remaining & ~active.unsqueeze(1),
        )

    optional_selected = optional & (optional_choice != skip_index)
    action_matrix[:, 0] = torch.where(
        optional_selected,
        optional_choice,
        action_matrix[:, 0],
    )
    log_probs = log_probs + selection_log_prob
    entropies = entropies + selection_entropy

    # A single device-to-host transfer replaces O(batch * action_count)
    # scalar synchronizations.  Python lists are still required by libcg.Select.
    action_rows = action_matrix.detach().cpu().tolist()
    count_rows = counts.detach().cpu().tolist()
    actions = [row[: int(count)] for row, count in zip(action_rows, count_rows)]
    return actions, log_probs, entropies, torch.sigmoid(
        outputs["value_logits"].float()
    )


def nonar_action_log_prob_entropy(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    action_sequences: torch.Tensor,
    action_counts: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_prob, entropy = legacy._NONAR_BASE_ACTION_STATS(
        outputs,
        batch,
        action_sequences,
        action_counts,
        temperature=temperature,
    )
    modes = nonar.selection_modes(batch)
    optional = modes == nonar.MODE_OPTIONAL_SINGLE
    if optional.any():
        joint_logits = _optional_joint_logits(outputs, batch, temperature)
        joint_log = F.log_softmax(joint_logits, dim=1)
        joint_prob = joint_log.exp()
        skip_index = outputs["policy_logits"].shape[1]
        chosen_option = action_sequences[:, 0].clamp_min(0)
        target = torch.where(
            action_counts == 0,
            torch.full_like(action_counts, skip_index),
            chosen_option,
        )
        optional_log_prob = joint_log.gather(1, target.unsqueeze(1)).squeeze(1)
        optional_entropy = -(joint_prob * joint_log).sum(dim=1)
        log_prob = torch.where(optional, optional_log_prob, log_prob)
        entropy = torch.where(optional, optional_entropy, entropy)
    return log_prob, entropy


def nonar_reference_policy_kl(
    outputs: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    action_sequences: torch.Tensor,
    action_counts: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    kl = legacy._NONAR_BASE_REFERENCE_KL(
        outputs,
        reference,
        batch,
        action_sequences,
        action_counts,
        temperature=temperature,
    )
    optional = nonar.selection_modes(batch) == nonar.MODE_OPTIONAL_SINGLE
    if optional.any():
        current_log = F.log_softmax(
            _optional_joint_logits(outputs, batch, temperature), dim=1
        )
        reference_log = F.log_softmax(
            _optional_joint_logits(reference, batch, temperature), dim=1
        )
        reference_prob = reference_log.exp()
        optional_kl = (
            reference_prob * (reference_log - current_log)
        ).sum(dim=1)
        kl = torch.where(optional, optional_kl, kl)
    return kl


def install_nonar_adapter() -> None:
    """Patch the generic harness before it parses or loads a checkpoint."""

    legacy.EntityOptionPolicy = nonar.EntityOptionPolicy
    legacy.featurize_row = nonar.featurize_row
    legacy.BC_FEATURE_VERSION = nonar.FEATURE_VERSION
    legacy.PPO_FEATURE_VERSION = PPO_FEATURE_VERSION
    # Preserve the generic static Plackett-Luce implementations so the two
    # wrappers above can specialize optional-single rows without recursion.
    if not hasattr(legacy, "_NONAR_BASE_ACTION_STATS"):
        legacy._NONAR_BASE_ACTION_STATS = legacy.ordered_action_log_prob_entropy
    if not hasattr(legacy, "_NONAR_BASE_REFERENCE_KL"):
        legacy._NONAR_BASE_REFERENCE_KL = legacy.reference_policy_kl
    legacy.sample_ordered_actions = sample_nonar_actions
    legacy.ordered_action_log_prob_entropy = nonar_action_log_prob_entropy
    legacy.reference_policy_kl = nonar_reference_policy_kl
    legacy.OpponentSamplingReweighter = League801010Reweighter

    def shared_frozen_copy(
        model: torch.nn.Module,
        name: str,
        device: torch.device,
        deck: list[int],
        deck_hash: str,
        permanent: bool,
        canonical_order: bool,
    ) -> legacy.FrozenOpponent:
        # The recent-meta deck bindings all use the same immutable BC policy.
        # Share exact-equal frozen states, while keeping genuinely different
        # historical policies independent.
        if permanent:
            state_hash = legacy.model_state_sha256(model)
            frozen_model = _SHARED_FROZEN_MODELS.get(state_hash)
            if frozen_model is None:
                frozen = _BASE_FROZEN_COPY(
                    model,
                    name,
                    device,
                    deck,
                    deck_hash,
                    permanent,
                    canonical_order,
                )
                _SHARED_FROZEN_MODELS[state_hash] = frozen.model
                return frozen
            return legacy.FrozenOpponent(
                name=name,
                model=frozen_model,
                deck=list(deck),
                deck_hash=deck_hash,
                permanent=True,
                canonical_order=canonical_order,
            )
        return _BASE_FROZEN_COPY(
            model,
            name,
            device,
            deck,
            deck_hash,
            permanent,
            canonical_order,
        )

    legacy.frozen_copy = shared_frozen_copy


def main() -> None:
    install_nonar_adapter()
    legacy.main()


if __name__ == "__main__":
    main()
