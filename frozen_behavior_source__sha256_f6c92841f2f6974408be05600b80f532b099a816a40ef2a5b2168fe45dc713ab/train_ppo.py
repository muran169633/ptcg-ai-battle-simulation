#!/usr/bin/env python3
"""Self-play PPO for the PTCG entity Transformer policy.

The only environment reward is terminal:

    winner seat -> 1.0
    loser/draw  -> 0.0

Actions are modeled as an allowed cardinality followed by an ordered
Plackett-Luce sample without replacement over the dynamic legal options.  The
order is retained because a small number of PTCG contexts (for example skill
ordering) are order-sensitive.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import random
import sys
import time
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "dataset" / "sample_submission" / "sample_submission"
if str(SAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from cg.sim import lib  # noqa: E402
from train_bc_orbit import (  # noqa: E402
    FEATURE_VERSION as BC_FEATURE_VERSION,
    MAX_ACTION_COUNT as BC_MAX_ACTION_COUNT,
    EntityOptionPolicy,
    ZipDecisionDataset,
    collate_decisions,
    featurize_row,
)


PPO_FEATURE_VERSION = "ptcg-selfplay-ppo-terminal01-v1"
# The engine accepts up to 60 unique selected indices.  BC V5 was trained with
# a 0..16 count head, so PPO expands that final layer while preserving its
# learned rows.
MAX_ACTION_COUNT = 60
SKILL_ORDER_CONTEXT = 34
TRAINABLE_SCOPES = (
    "full",
    "heads",
    "last_block_heads",
    "last_two_blocks_heads",
)
RESUME_LEARNER_WEIGHT_SOURCES = ("resume", "bc")
ADVANTAGE_NORMALIZATION_MODES = ("global", "per_opponent")
PPO_OBJECTIVES = ("standard", "constrained")
ACTOR_REDUCTIONS = (
    "transition_mean",
    "quota_group_mean",
    "episode_mean",
)
CONSTRAINED_GRADIENT_MODES = ("scalar", "pcgrad", "guard_pcgrad")
ACTOR_VALUE_GRADIENT_MODES = (
    "scalar",
    "actor_priority_value_pcgrad",
)
SELFPLAY_OPPONENT_GROUP = "__selfplay__"
CONSTRAINED_RESUME_FIELDS = (
    "policy_temperature",
    "constrained_gradient_mode",
    "opponent_quota_seat_balance",
    "primary_opponent_name",
    "guard_opponent_name",
    "primary_policy_weight",
    "guard_policy_weight",
    "auxiliary_policy_weight",
    "guard_surrogate_floor",
    "constraint_dual_lr",
    "constraint_dual_max",
    "opponent_loss_weights",
)


def log(message: str) -> None:
    print(message, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_deck_hash(deck: list[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(deck))
    return hashlib.sha256(canonical.encode()).hexdigest()


def read_deck(path: Path) -> list[int]:
    values = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != 60:
        raise ValueError(f"{path} contains {len(values)} cards; expected 60")
    return values


class RawBattle:
    """One official engine battle pointer without cg.game's global wrapper."""

    def __init__(self, deck0: list[int], deck1: list[int]) -> None:
        cards = deck0 + deck1
        arg = (ctypes.c_int * len(cards))(*cards)
        start = lib.BattleStart(arg)
        self.ptr = start.battlePtr
        self.closed = False
        if not self.ptr:
            raise ValueError(
                f"BattleStart failed: player={start.errorPlayer} "
                f"type={start.errorType}"
            )
        self.observation = self._get_observation()

    def _get_observation(self) -> dict[str, Any]:
        serial = lib.GetBattleData(self.ptr)
        observation = json.loads(serial.json.decode())
        observation["search_begin_input"] = ctypes.string_at(
            serial.data,
            serial.count,
        ).decode("ascii")
        return observation

    @property
    def result(self) -> int:
        current = self.observation.get("current") or {}
        return int(current.get("result", -1))

    def step(self, action: list[int]) -> tuple[dict[str, Any], int]:
        arg = (ctypes.c_int * len(action))(*action)
        error = int(lib.Select(self.ptr, arg, len(action)))
        if error:
            return self.observation, error
        self.observation = self._get_observation()
        return self.observation, 0

    def close(self) -> None:
        if not self.closed and self.ptr:
            lib.BattleFinish(self.ptr)
            self.closed = True
            self.ptr = None


@dataclass
class PPOConfig:
    bc_checkpoint: str
    kl_reference_checkpoint: str | None
    deck: str
    extra_opponents: list[dict[str, str]]
    output_dir: str
    updates: int
    environments: int
    games_per_update: int
    ppo_epochs: int
    minibatch_size: int
    learning_rate: float
    value_learning_rate: float
    weight_decay: float
    gamma: float
    gae_lambda: float
    advantage_normalization: str
    clip_ratio: float
    value_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float
    policy_temperature: float
    trainable_scope: str
    learning_rate_schedule: str
    schedule_start_update: int
    bc_kl_start: float
    bc_kl_end: float
    target_kl: float
    league_probability: float
    opponent_sampling: str
    bc_opponent_probability: float | None
    opponent_weights: dict[str, float]
    history_opponent_weight: float
    opponent_quota_mode: str
    opponent_base_quotas: dict[str, int]
    opponent_caps: dict[str, int]
    opponent_audit: dict[str, dict[str, int]]
    opponent_quota_refresh_updates: int
    opponent_quota_seat_balance: bool
    ppo_objective: str
    actor_reduction: str
    actor_reduction_audit_only: bool
    constrained_gradient_mode: str
    primary_opponent_name: str | None
    guard_opponent_name: str | None
    primary_policy_weight: float
    guard_policy_weight: float
    auxiliary_policy_weight: float
    guard_surrogate_floor: float
    constraint_dual_initial: float
    constraint_dual_lr: float
    constraint_dual_max: float
    opponent_loss_weights: dict[str, float]
    snapshot_interval: int
    max_pool_size: int
    bc_replay_data: str | None
    bc_replay_split: str
    bc_replay_batches: int
    bc_replay_batch_size: int
    bc_replay_workers: int
    bc_replay_steps: int
    bc_replay_lr_scale: float
    bc_replay_loss: str
    bc_replay_order_context_weight: float
    bc_replay_context34_rows_per_batch: int
    eval_interval: int
    eval_games: int
    eval_all_permanent_opponents: bool
    selection_aggregation: str
    checkpoint_interval: int
    max_game_decisions: int
    seed: int
    device: str
    resume_checkpoint: str | None
    resume_learner_weights: str
    reset_optimizer_on_resume: bool
    reset_opponent_quota_on_resume: bool
    value_trunk_gradient_scale: float = 1.0
    actor_value_gradient_mode: str = "scalar"
    actor_value_gradient_audit_only: bool = False
    bc_replay_non_context34_fixed_multi_action_order_weight: float = 1.0


def constrained_resume_config(config: PPOConfig) -> dict[str, Any]:
    return {
        field: copy.deepcopy(getattr(config, field))
        for field in CONSTRAINED_RESUME_FIELDS
    }


@dataclass
class FrozenOpponent:
    name: str
    model: EntityOptionPolicy
    deck: list[int]
    deck_hash: str
    permanent: bool
    canonical_order: bool


@dataclass
class RunningGame:
    battle: RawBattle
    uid: int
    seat_policy: dict[int, int]
    seat_deck_hash: dict[int, str]
    trainable_seats: set[int]
    transition_indices: dict[int, list[int]]
    opponent_name: str | None
    decisions: int = 0


@dataclass
class ConstrainedObjectiveState:
    dual_value: float
    completed_updates: int = 0
    last_guard_surrogate: float | None = None
    last_update: int | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "dual_value": float(self.dual_value),
            "completed_updates": int(self.completed_updates),
            "last_guard_surrogate": (
                float(self.last_guard_surrogate)
                if self.last_guard_surrogate is not None
                else None
            ),
            "last_update": self.last_update,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        dual_max: float,
    ) -> ConstrainedObjectiveState:
        version = int(state.get("version", -1))
        if version not in {1, 2}:
            raise ValueError("Unsupported constrained-objective state version")
        dual_value = float(state.get("dual_value", float("nan")))
        completed_updates = int(state.get("completed_updates", -1))
        raw_last = state.get("last_guard_surrogate")
        last_guard_surrogate = (
            None if raw_last is None else float(raw_last)
        )
        raw_last_update = state.get("last_update")
        last_update = (
            None if raw_last_update is None else int(raw_last_update)
        )
        if (
            not math.isfinite(dual_value)
            or not 0.0 <= dual_value <= dual_max
            or completed_updates < 0
            or (last_update is not None and last_update < 1)
            or (
                last_guard_surrogate is not None
                and not math.isfinite(last_guard_surrogate)
            )
        ):
            raise ValueError("Malformed constrained-objective state")
        return cls(
            dual_value=dual_value,
            completed_updates=completed_updates,
            last_guard_surrogate=last_guard_surrogate,
            last_update=last_update,
        )

    def finish_update(
        self,
        guard_surrogate: float,
        *,
        update: int,
        floor: float,
        dual_learning_rate: float,
        dual_max: float,
    ) -> dict[str, float | int]:
        if not math.isfinite(guard_surrogate):
            raise ValueError("Guard surrogate must be finite")
        if update < 1 or (
            self.last_update is not None and update <= self.last_update
        ):
            raise ValueError(
                "Constrained-objective updates must increase monotonically"
            )
        dual_before = self.dual_value
        violation = floor - guard_surrogate
        self.dual_value = min(
            max(
                self.dual_value + dual_learning_rate * violation,
                0.0,
            ),
            dual_max,
        )
        self.completed_updates += 1
        self.last_guard_surrogate = guard_surrogate
        self.last_update = update
        return {
            "guard_surrogate": guard_surrogate,
            "guard_surrogate_floor": floor,
            "constraint_violation": max(violation, 0.0),
            "signed_constraint_gap": guard_surrogate - floor,
            "dual_before": dual_before,
            "dual_after": self.dual_value,
            "completed_updates": self.completed_updates,
            "last_update": update,
        }


def align_constrained_objective_resume_state(
    state: ConstrainedObjectiveState,
    *,
    checkpoint_update: int,
    serialized_version: int,
) -> str:
    """Validate update alignment, including pre-update and v1 checkpoints."""
    if state.last_update == checkpoint_update:
        return "matched"
    if state.last_update is None and state.completed_updates == 0:
        return "pre_update"
    if state.last_update is None and serialized_version == 1:
        state.last_update = checkpoint_update
        return "version1_migrated"
    raise ValueError(
        "Constrained objective state update does not match "
        "the resume checkpoint update"
    )


def transition_opponent_group(transition: dict[str, Any]) -> str:
    if "opponent_name" not in transition:
        raise ValueError(
            "Per-opponent PPO requires opponent_name on every transition"
        )
    opponent_name = transition["opponent_name"]
    return (
        SELFPLAY_OPPONENT_GROUP
        if opponent_name is None
        else str(opponent_name)
    )


def tensor_sample_stats(values: torch.Tensor) -> dict[str, float | int]:
    count = int(values.numel())
    if count == 0:
        raise ValueError("Cannot summarize an empty tensor")
    std = (
        float(values.std(unbiased=True))
        if count > 1
        else 0.0
    )
    return {
        "count": count,
        "mean": float(values.mean()),
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def normalize_rollout_advantages(
    transitions: list[dict[str, Any]],
    mode: str,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalize once over the frozen rollout, globally or by opponent."""
    if mode not in ADVANTAGE_NORMALIZATION_MODES:
        raise ValueError(f"Unsupported advantage normalization mode: {mode!r}")
    if not transitions:
        raise ValueError("Cannot normalize an empty rollout")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Advantage normalization epsilon must be positive")
    raw = torch.tensor(
        [float(transition["advantage"]) for transition in transitions],
        dtype=torch.float32,
    )
    if not torch.isfinite(raw).all():
        raise ValueError("Rollout advantages must be finite")

    global_raw = tensor_sample_stats(raw)
    normalized = torch.empty_like(raw)
    by_group: dict[str, dict[str, Any]] = {}
    degenerate_groups: list[str] = []
    if mode == "global":
        std = raw.std(unbiased=True) if raw.numel() > 1 else raw.new_tensor(0.0)
        degenerate = raw.numel() <= 1 or float(std) < epsilon
        normalized.copy_(
            torch.zeros_like(raw)
            if degenerate
            else (raw - raw.mean()) / std
        )
        if degenerate:
            degenerate_groups.append("__global__")
    else:
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index, transition in enumerate(transitions):
            grouped_indices[transition_opponent_group(transition)].append(index)
        for group_name in sorted(grouped_indices):
            indices = torch.tensor(grouped_indices[group_name], dtype=torch.long)
            group_raw = raw.index_select(0, indices)
            group_std = (
                group_raw.std(unbiased=True)
                if group_raw.numel() > 1
                else group_raw.new_tensor(0.0)
            )
            degenerate = (
                group_raw.numel() <= 1
                or not math.isfinite(float(group_std))
                or float(group_std) < epsilon
            )
            group_normalized = (
                torch.zeros_like(group_raw)
                if degenerate
                else (group_raw - group_raw.mean()) / group_std
            )
            normalized.index_copy_(0, indices, group_normalized)
            if degenerate:
                degenerate_groups.append(group_name)
            by_group[group_name] = {
                "transitions": int(group_raw.numel()),
                "raw_mean": float(group_raw.mean()),
                "raw_std": (
                    float(group_std)
                    if math.isfinite(float(group_std))
                    else 0.0
                ),
                "normalized_mean": float(group_normalized.mean()),
                "normalized_std": (
                    float(group_normalized.std(unbiased=True))
                    if group_normalized.numel() > 1
                    else 0.0
                ),
                "degenerate": degenerate,
            }
        if sum(row["transitions"] for row in by_group.values()) != len(
            transitions
        ):
            raise RuntimeError("Opponent advantage groups lost transitions")

    audit = {
        "mode": mode,
        "group_field": (
            None if mode == "global" else "opponent_name"
        ),
        "selfplay_sentinel": SELFPLAY_OPPONENT_GROUP,
        "epsilon": epsilon,
        "group_count": 1 if mode == "global" else len(by_group),
        "degenerate_groups": degenerate_groups,
        "global_raw": global_raw,
        "global_normalized": tensor_sample_stats(normalized),
        "by_group": by_group,
    }
    return normalized, audit


def opponent_group_means(
    values: torch.Tensor,
    group_names: list[str],
) -> dict[str, torch.Tensor]:
    if values.ndim != 1 or values.shape[0] != len(group_names):
        raise ValueError("Opponent group reduction shape mismatch")
    grouped_rows: dict[str, list[int]] = defaultdict(list)
    for row_index, group_name in enumerate(group_names):
        grouped_rows[group_name].append(row_index)
    return {
        group_name: values[
            torch.tensor(row_indices, device=values.device, dtype=torch.long)
        ].mean()
        for group_name, row_indices in grouped_rows.items()
    }


def weighted_opponent_group_mean(
    values: torch.Tensor,
    group_names: list[str],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    means = opponent_group_means(values, group_names)
    unknown = set(means) - set(weights)
    if unknown:
        raise ValueError(
            "Missing opponent loss weights for "
            + ", ".join(sorted(unknown))
        )
    present_weight = sum(weights[name] for name in means)
    if not math.isfinite(present_weight) or present_weight <= 0.0:
        raise ValueError("Present opponent loss weights must sum positive")
    weighted = sum(
        means[name] * weights[name]
        for name in means
    ) / present_weight
    return weighted, means


def quota_group_actor_multipliers(
    transitions: list[dict[str, Any]],
    opponent_quotas: dict[str, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert game quotas into per-transition actor-loss multipliers.

    A transition-mean objective weights opponents in proportion to episode
    length.  Multiplying every row from opponent ``i`` by

        (quota_i / sum(quota)) / (rows_i / sum(rows))

    makes the full-rollout reduction equal to a quota-weighted mean of the
    within-opponent means while leaving all non-actor losses untouched.
    """
    if not transitions:
        raise ValueError("Cannot reduce an empty rollout")
    positive_quotas = {
        str(name): int(quota)
        for name, quota in opponent_quotas.items()
        if int(quota) > 0
    }
    if not positive_quotas:
        raise ValueError("Quota-group actor reduction requires positive quotas")
    if any(int(quota) != quota or int(quota) < 0 for quota in opponent_quotas.values()):
        raise ValueError("Opponent quotas must be non-negative integers")

    group_names = [
        transition_opponent_group(transition)
        for transition in transitions
    ]
    row_counts = Counter(group_names)
    expected_groups = set(positive_quotas)
    observed_groups = set(row_counts)
    if observed_groups != expected_groups:
        missing = sorted(expected_groups - observed_groups)
        unexpected = sorted(observed_groups - expected_groups)
        raise RuntimeError(
            "Quota-group actor rollout coverage mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )

    total_rows = len(transitions)
    total_quota = sum(positive_quotas.values())
    multiplier_by_group: dict[str, float] = {}
    group_audit: dict[str, dict[str, float | int]] = {}
    max_relative_weight_deviation = 0.0
    for group_name in sorted(expected_groups):
        rows = int(row_counts[group_name])
        row_weight = rows / total_rows
        quota_weight = positive_quotas[group_name] / total_quota
        multiplier = quota_weight / row_weight
        relative_weight_deviation = (
            abs(row_weight - quota_weight) / quota_weight
        )
        max_relative_weight_deviation = max(
            max_relative_weight_deviation,
            relative_weight_deviation,
        )
        multiplier_by_group[group_name] = multiplier
        group_audit[group_name] = {
            "quota_games": positive_quotas[group_name],
            "transition_rows": rows,
            "quota_weight": quota_weight,
            "row_weight": row_weight,
            "row_multiplier": multiplier,
            "relative_weight_deviation": relative_weight_deviation,
        }

    multipliers = torch.tensor(
        [multiplier_by_group[group_name] for group_name in group_names],
        dtype=torch.float32,
    )
    mean_multiplier = float(multipliers.mean())
    if not torch.isfinite(multipliers).all() or not math.isclose(
        mean_multiplier,
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "Quota-group actor multipliers do not preserve total loss scale"
        )
    return multipliers, {
        "mode": "quota_group_mean",
        "group_field": "opponent_name",
        "groups": group_audit,
        "group_count": len(group_audit),
        "total_quota_games": total_quota,
        "total_transition_rows": total_rows,
        "mean_row_multiplier": mean_multiplier,
        "max_relative_weight_deviation": max_relative_weight_deviation,
        "full_rollout_weight_error": abs(mean_multiplier - 1.0),
    }


def episode_mean_actor_multipliers(
    transitions: list[dict[str, Any]],
    expected_episode_count: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert frozen game membership into equal-episode actor weights.

    For ``N`` rollout transitions, ``G`` games, and ``n_g`` transitions in
    game ``g``, every row from that game receives multiplier

        N / (G * n_g)

    Thus the transition mean of the weighted clipped surrogate is exactly the
    mean of the per-game surrogate means.  Minibatch SGD over uniformly
    shuffled transition rows is an unbiased estimator of that frozen-rollout
    objective.  Only the actor surrogate consumes these multipliers.
    """
    if not transitions:
        raise ValueError("Cannot reduce an empty rollout")
    if expected_episode_count is not None and expected_episode_count <= 0:
        raise ValueError("Expected episode count must be positive")

    game_uids: list[int] = []
    missing_rows: list[int] = []
    for row_index, transition in enumerate(transitions):
        if "game_uid" not in transition or transition["game_uid"] is None:
            missing_rows.append(row_index)
            continue
        game_uid = transition["game_uid"]
        if isinstance(game_uid, bool) or not isinstance(game_uid, int):
            raise ValueError(
                "Episode-mean actor reduction requires integer game_uid "
                f"values; row {row_index} has {game_uid!r}"
            )
        game_uids.append(game_uid)
    if missing_rows:
        raise ValueError(
            "Episode-mean actor reduction requires game_uid on every "
            f"transition; missing rows={missing_rows[:16]}"
        )

    row_counts = Counter(game_uids)
    episode_count = len(row_counts)
    if (
        expected_episode_count is not None
        and episode_count != expected_episode_count
    ):
        raise RuntimeError(
            "Episode-mean actor rollout coverage mismatch: "
            f"observed_games={episode_count} "
            f"expected_games={expected_episode_count}"
        )

    total_rows = len(transitions)
    expected_episode_weight = 1.0 / episode_count
    multiplier_by_uid = {
        game_uid: total_rows / (episode_count * rows)
        for game_uid, rows in row_counts.items()
    }
    multipliers = torch.tensor(
        [multiplier_by_uid[game_uid] for game_uid in game_uids],
        dtype=torch.float32,
    )
    mean_multiplier = float(multipliers.mean())
    episode_audit: dict[str, dict[str, float | int]] = {}
    max_episode_weight_error = 0.0
    max_relative_row_weight_deviation = 0.0
    min_episode_total_weight = float("inf")
    max_episode_total_weight = 0.0
    for game_uid in sorted(row_counts):
        rows = int(row_counts[game_uid])
        raw_row_weight = rows / total_rows
        row_multiplier = multiplier_by_uid[game_uid]
        episode_total_weight = (
            raw_row_weight * row_multiplier
        )
        episode_weight_error = abs(
            episode_total_weight - expected_episode_weight
        )
        relative_row_weight_deviation = abs(
            raw_row_weight - expected_episode_weight
        ) / expected_episode_weight
        max_episode_weight_error = max(
            max_episode_weight_error,
            episode_weight_error,
        )
        max_relative_row_weight_deviation = max(
            max_relative_row_weight_deviation,
            relative_row_weight_deviation,
        )
        min_episode_total_weight = min(
            min_episode_total_weight,
            episode_total_weight,
        )
        max_episode_total_weight = max(
            max_episode_total_weight,
            episode_total_weight,
        )
        episode_audit[str(game_uid)] = {
            "transition_rows": rows,
            "raw_row_weight": raw_row_weight,
            "row_multiplier": row_multiplier,
            "episode_total_weight": episode_total_weight,
            "expected_episode_weight": expected_episode_weight,
            "episode_weight_error": episode_weight_error,
            "relative_row_weight_deviation": (
                relative_row_weight_deviation
            ),
        }

    full_rollout_weight_error = abs(mean_multiplier - 1.0)
    tolerance = 1e-6
    all_games_equal_weight = (
        max_episode_weight_error <= tolerance
        and max_episode_total_weight - min_episode_total_weight <= tolerance
    )
    if (
        not torch.isfinite(multipliers).all()
        or full_rollout_weight_error > tolerance
        or not all_games_equal_weight
    ):
        raise RuntimeError(
            "Episode-mean actor multipliers failed equal-game invariants"
        )
    return multipliers, {
        "mode": "episode_mean",
        "group_field": "game_uid",
        "episodes": episode_audit,
        "episode_count": episode_count,
        "expected_episode_count": expected_episode_count,
        "total_transition_rows": total_rows,
        "covered_transition_rows": sum(row_counts.values()),
        "missing_game_uid_rows": 0,
        "all_transition_rows_accounted_for": (
            sum(row_counts.values()) == total_rows
        ),
        "mean_row_multiplier": mean_multiplier,
        "full_rollout_weight_error": full_rollout_weight_error,
        "expected_episode_weight": expected_episode_weight,
        "min_episode_total_weight": min_episode_total_weight,
        "max_episode_total_weight": max_episode_total_weight,
        "max_episode_weight_error": max_episode_weight_error,
        "all_games_equal_weight": all_games_equal_weight,
        "max_relative_weight_deviation": (
            max_relative_row_weight_deviation
        ),
    }


def actor_reduction_multipliers(
    transitions: list[dict[str, Any]],
    reduction: str,
    *,
    opponent_quotas: dict[str, int] | None = None,
    expected_episode_count: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Resolve frozen-rollout actor weights without changing other losses."""
    if reduction == "transition_mean":
        if not transitions:
            raise ValueError("Cannot reduce an empty rollout")
        return torch.ones(len(transitions), dtype=torch.float32), {
            "mode": "transition_mean",
            "group_field": None,
            "group_count": 1,
            "total_transition_rows": len(transitions),
            "mean_row_multiplier": 1.0,
            "full_rollout_weight_error": 0.0,
        }
    if reduction == "quota_group_mean":
        return quota_group_actor_multipliers(
            transitions,
            opponent_quotas or {},
        )
    if reduction == "episode_mean":
        return episode_mean_actor_multipliers(
            transitions,
            expected_episode_count,
        )
    raise ValueError(f"Unsupported actor reduction: {reduction!r}")


def standard_actor_policy_loss(
    surrogate: torch.Tensor,
    row_multipliers: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    if surrogate.ndim != 1 or row_multipliers.shape != surrogate.shape:
        raise ValueError("Actor surrogate reduction shape mismatch")
    if reduction == "transition_mean":
        return -surrogate.mean()
    if reduction in {"quota_group_mean", "episode_mean"}:
        if not torch.isfinite(row_multipliers).all():
            raise ValueError("Actor row multipliers must be finite")
        return -(surrogate * row_multipliers).mean()
    raise ValueError(f"Unsupported actor reduction: {reduction!r}")


def should_run_initial_evaluation(
    skip_initial_eval: bool,
    actor_reduction_audit_only: bool,
    actor_value_gradient_audit_only: bool = False,
) -> bool:
    """Keep audit-only runs strictly free of evaluation checkpoints."""
    return (
        not skip_initial_eval
        and not actor_reduction_audit_only
        and not actor_value_gradient_audit_only
    )


def stable_largest_remainder(
    total: int,
    weights: list[float],
    capacities: list[int],
) -> list[int]:
    """Allocate integer slots by Hamilton rounding with stable cap handling."""
    if total < 0:
        raise ValueError("Largest-remainder total must be non-negative")
    if len(weights) != len(capacities):
        raise ValueError("Largest-remainder weights/capacities length mismatch")
    if any(
        not math.isfinite(weight) or weight < 0.0
        for weight in weights
    ):
        raise ValueError("Largest-remainder weights must be finite and non-negative")
    if any(capacity < 0 for capacity in capacities):
        raise ValueError("Largest-remainder capacities must be non-negative")
    if sum(capacities) < total:
        raise ValueError("Largest-remainder capacities cannot satisfy total")
    if total == 0:
        return [0] * len(weights)

    allocation = [0] * len(weights)
    active = {
        index
        for index, capacity in enumerate(capacities)
        if capacity > 0
    }
    remaining = total
    while remaining:
        if not active:
            raise ValueError("Largest-remainder allocation exhausted capacity")
        active_weight = sum(weights[index] for index in active)
        effective_weights = (
            weights
            if active_weight > 0.0
            else [
                1.0 if index in active else 0.0
                for index in range(len(weights))
            ]
        )
        active_weight = sum(effective_weights[index] for index in active)
        raw = {
            index: remaining * effective_weights[index] / active_weight
            for index in active
        }
        newly_capped = [
            index
            for index in active
            if raw[index] >= capacities[index] - allocation[index]
        ]
        if newly_capped:
            for index in sorted(newly_capped):
                granted = capacities[index] - allocation[index]
                allocation[index] += granted
                remaining -= granted
                active.remove(index)
            continue

        floors = {
            index: min(
                capacities[index] - allocation[index],
                int(math.floor(raw[index])),
            )
            for index in active
        }
        for index, granted in floors.items():
            allocation[index] += granted
            remaining -= granted
        if not remaining:
            break
        ranked = sorted(
            active,
            key=lambda index: (-(raw[index] - math.floor(raw[index])), index),
        )
        for index in ranked:
            if remaining == 0:
                break
            if allocation[index] >= capacities[index]:
                continue
            allocation[index] += 1
            remaining -= 1
        if remaining:
            active = {
                index
                for index in active
                if allocation[index] < capacities[index]
            }
    return allocation


def beta_failure_posterior(
    audit: dict[str, int],
) -> dict[str, float]:
    """Beta(2,2) posterior for loss-or-half-draw failure probability."""
    wins = int(audit.get("wins", 0))
    losses = int(audit.get("losses", 0))
    draws = int(audit.get("draws", 0))
    if min(wins, losses, draws) < 0:
        raise ValueError("Opponent audit counts must be non-negative")
    alpha = 2.0 + losses + 0.5 * draws
    beta = 2.0 + wins + 0.5 * draws
    total = alpha + beta
    mean = alpha / total
    std = math.sqrt(alpha * beta / (total * total * (total + 1.0)))
    return {
        "alpha": alpha,
        "beta": beta,
        "mean": mean,
        "std": std,
        "failure_ucb": mean + std,
    }


def opponent_quota_schedule_seed(seed: int, update: int) -> int:
    payload = f"ptcg-opponent-quota-v1:{seed}:{update}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class ExactQuotaClaim:
    opponent_index: int
    learner_seat: int | None = None


class ExactQuotaSchedule:
    """Tracks planned opponent/seat claims and exact invalid replacements."""

    def __init__(
        self,
        opponent_indices: list[int],
        learner_seats: list[int | None] | None = None,
    ) -> None:
        if learner_seats is None:
            learner_seats = [None] * len(opponent_indices)
        if len(learner_seats) != len(opponent_indices):
            raise ValueError("Exact-quota opponent/seat schedule length mismatch")
        if any(seat not in {None, 0, 1} for seat in learner_seats):
            raise ValueError("Exact-quota learner seats must be 0, 1, or None")
        self._planned = [
            ExactQuotaClaim(int(opponent_index), learner_seat)
            for opponent_index, learner_seat in zip(
                opponent_indices,
                learner_seats,
            )
        ]
        self._planned_position = 0
        self._replacements: deque[ExactQuotaClaim] = deque()

    def peek(self) -> tuple[int, str] | None:
        claim = self.peek_claim()
        if claim is None:
            return None
        return claim[0].opponent_index, claim[1]

    def peek_claim(self) -> tuple[ExactQuotaClaim, str] | None:
        if self._replacements:
            return self._replacements[0], "replacement"
        if self._planned_position < len(self._planned):
            return self._planned[self._planned_position], "planned"
        return None

    def mark_started(self, source: str) -> None:
        if source == "replacement":
            self._replacements.popleft()
        elif source == "planned":
            self._planned_position += 1
        else:
            raise ValueError(f"Unknown exact-quota source: {source!r}")

    def add_replacement(
        self,
        opponent_index: int,
        learner_seat: int | None = None,
    ) -> None:
        if learner_seat not in {None, 0, 1}:
            raise ValueError("Replacement learner seat must be 0, 1, or None")
        self._replacements.append(
            ExactQuotaClaim(int(opponent_index), learner_seat)
        )

    @property
    def remaining(self) -> int:
        return (
            len(self._planned) - self._planned_position
            + len(self._replacements)
        )


def balanced_quota_learner_seats(
    opponent_names: list[str],
    seed: int,
    update: int,
) -> tuple[list[int], dict[str, dict[str, int]]]:
    """Assign deterministic near-equal learner seats within each opponent."""
    totals = Counter(opponent_names)
    starts = {
        name: int.from_bytes(
            hashlib.sha256(
                f"ptcg-seat-balance-v1:{seed}:{update}:{name}".encode()
            ).digest()[:8],
            "big",
        )
        % 2
        for name in totals
    }
    seen: Counter[str] = Counter()
    seats: list[int] = []
    planned: dict[str, dict[str, int]] = {
        name: {"0": 0, "1": 0}
        for name in totals
    }
    for name in opponent_names:
        seat = (starts[name] + seen[name]) % 2
        seen[name] += 1
        seats.append(seat)
        planned[name][str(seat)] += 1
    if any(
        abs(counts["0"] - counts["1"]) > 1
        for counts in planned.values()
    ):
        raise RuntimeError("Balanced quota learner-seat assignment failed")
    return seats, planned


class OpponentQuotaController:
    """Exact permanent-opponent quotas with optional uncertainty adaptation."""

    def __init__(
        self,
        *,
        mode: str,
        opponent_names: list[str],
        base_quotas: dict[str, int],
        caps: dict[str, int],
        initial_audit: dict[str, dict[str, int]],
        games_per_update: int,
        refresh_updates: int,
        seed: int,
    ) -> None:
        if mode not in {"fixed", "adaptive"}:
            raise ValueError(f"Unsupported opponent quota mode: {mode!r}")
        if len(opponent_names) != len(set(opponent_names)):
            raise ValueError("Opponent quota names must be unique")
        if refresh_updates < 1:
            raise ValueError("Opponent quota refresh interval must be positive")
        self.mode = mode
        self.opponent_names = list(opponent_names)
        self.base_quotas = {
            name: int(base_quotas.get(name, 0))
            for name in self.opponent_names
        }
        self.caps = {
            name: int(caps.get(name, self.base_quotas[name]))
            for name in self.opponent_names
        }
        self.initial_audit = {
            name: {
                key: int(initial_audit.get(name, {}).get(key, 0))
                for key in ("wins", "losses", "draws")
            }
            for name in self.opponent_names
        }
        self.observed = {
            name: {"wins": 0, "losses": 0, "draws": 0}
            for name in self.opponent_names
        }
        self.games_per_update = int(games_per_update)
        self.refresh_updates = int(refresh_updates)
        self.seed = int(seed)
        self.current_plan: dict[str, int] | None = None
        self.current_dynamic_quotas: dict[str, int] | None = None
        self.current_allocation_posterior: (
            dict[str, dict[str, float]] | None
        ) = None
        self.current_allocation_audit: dict[str, dict[str, int]] | None = None
        self.last_refresh_update: int | None = None
        self.resume_state_loaded = False
        self.resume_state_reset = False
        self._validate()

    def _validate(self) -> None:
        if self.games_per_update < 1:
            raise ValueError("Opponent quota games_per_update must be positive")
        if any(value < 0 for value in self.base_quotas.values()):
            raise ValueError("Opponent base quotas must be non-negative")
        if any(value < 0 for value in self.caps.values()):
            raise ValueError("Opponent caps must be non-negative")
        if any(
            self.caps[name] < self.base_quotas[name]
            for name in self.opponent_names
        ):
            raise ValueError("Every opponent cap must be >= its base quota")
        for audit in self.initial_audit.values():
            beta_failure_posterior(audit)
        base_total = sum(self.base_quotas.values())
        if self.mode == "fixed":
            if base_total != self.games_per_update:
                raise ValueError(
                    "Fixed opponent base quotas must sum to games_per_update"
                )
        else:
            if base_total >= self.games_per_update:
                raise ValueError(
                    "Adaptive opponent base quotas must sum below "
                    "games_per_update"
                )
            if sum(self.caps.values()) < self.games_per_update:
                raise ValueError(
                    "Adaptive opponent caps cannot cover games_per_update"
                )

    def combined_audit(self, name: str) -> dict[str, int]:
        return {
            key: self.initial_audit[name][key] + self.observed[name][key]
            for key in ("wins", "losses", "draws")
        }

    def posterior(self) -> dict[str, dict[str, float]]:
        return {
            name: beta_failure_posterior(self.combined_audit(name))
            for name in self.opponent_names
        }

    def begin_update(
        self,
        update: int,
    ) -> tuple[list[str], dict[str, Any]]:
        recomputed = (
            self.current_plan is None
            or self.last_refresh_update is None
            or update - self.last_refresh_update >= self.refresh_updates
        )
        posterior = self.posterior()
        dynamic_slots = self.games_per_update - sum(self.base_quotas.values())
        if recomputed:
            if self.mode == "fixed":
                dynamic = {name: 0 for name in self.opponent_names}
            else:
                dynamic_values = stable_largest_remainder(
                    dynamic_slots,
                    [
                        posterior[name]["failure_ucb"]
                        for name in self.opponent_names
                    ],
                    [
                        self.caps[name] - self.base_quotas[name]
                        for name in self.opponent_names
                    ],
                )
                dynamic = dict(zip(self.opponent_names, dynamic_values))
            self.current_dynamic_quotas = dynamic
            self.current_plan = {
                name: self.base_quotas[name] + dynamic[name]
                for name in self.opponent_names
            }
            self.current_allocation_posterior = copy.deepcopy(posterior)
            self.current_allocation_audit = {
                name: self.combined_audit(name)
                for name in self.opponent_names
            }
            self.last_refresh_update = update
        assert self.current_plan is not None
        assert self.current_dynamic_quotas is not None
        assert self.current_allocation_posterior is not None
        assert self.current_allocation_audit is not None
        schedule = [
            name
            for name in self.opponent_names
            for _ in range(self.current_plan[name])
        ]
        schedule_seed = opponent_quota_schedule_seed(self.seed, update)
        random.Random(schedule_seed).shuffle(schedule)
        audit = {
            "mode": self.mode,
            "opponent_order": list(self.opponent_names),
            "refresh_updates": self.refresh_updates,
            "refresh_update": self.last_refresh_update,
            "recomputed": recomputed,
            "base_quotas": dict(self.base_quotas),
            "caps": dict(self.caps),
            "dynamic_slots": dynamic_slots,
            "dynamic_quotas": dict(self.current_dynamic_quotas),
            "planned_quotas": dict(self.current_plan),
            "posterior_model": {
                "prior": "Beta(2,2)",
                "failure": "loss + 0.5 * draw",
                "score": "posterior mean + posterior std",
            },
            "allocation_posterior": copy.deepcopy(
                self.current_allocation_posterior
            ),
            "allocation_combined_audit": copy.deepcopy(
                self.current_allocation_audit
            ),
            "posterior_before": posterior,
            "combined_audit_before": {
                name: self.combined_audit(name)
                for name in self.opponent_names
            },
            "schedule_seed": schedule_seed,
        }
        return schedule, audit

    def finish_update(
        self,
        update: int,
        audit: dict[str, Any],
        outcomes: dict[str, dict[str, int]],
        invalid_replacements: dict[str, int],
        planned_seat_quotas: dict[str, dict[str, int]] | None = None,
        invalid_replacements_by_seat: (
            dict[str, dict[str, int]] | None
        ) = None,
    ) -> dict[str, Any]:
        if sum(
            int(outcomes.get(name, {}).get("games", 0))
            for name in self.opponent_names
        ) != self.games_per_update:
            raise RuntimeError(
                f"Update {update} exact-quota rollout did not complete "
                f"{self.games_per_update} permanent-opponent games"
            )
        actual_quotas: dict[str, int] = {}
        actual_outcomes: dict[str, dict[str, int]] = {}
        for name in self.opponent_names:
            row = outcomes.get(name, {})
            actual = int(row.get("games", 0))
            actual_quotas[name] = actual
            planned = int(audit["planned_quotas"][name])
            if actual != planned:
                raise RuntimeError(
                    f"Update {update} opponent quota mismatch for {name}: "
                    f"planned={planned} actual={actual}"
                )
            actual_outcomes[name] = {
                key: int(row.get(key, 0))
                for key in ("wins", "losses", "draws")
            }
            if sum(actual_outcomes[name].values()) != actual:
                raise RuntimeError(
                    f"Update {update} aggregate outcome mismatch for {name}: "
                    f"games={actual} outcomes={actual_outcomes[name]}"
                )

        normalized_invalid_replacements = {
            name: int(invalid_replacements.get(name, 0))
            for name in self.opponent_names
        }
        normalized_planned_seats: dict[str, dict[str, int]] | None = None
        actual_seat_quotas: dict[str, dict[str, int]] | None = None
        actual_outcomes_by_seat: (
            dict[str, dict[str, dict[str, int]]] | None
        ) = None
        normalized_invalid_replacements_by_seat: (
            dict[str, dict[str, int]] | None
        ) = None
        if planned_seat_quotas is not None:
            normalized_planned_seats = {}
            actual_seat_quotas = {}
            actual_outcomes_by_seat = {}
            normalized_invalid_replacements_by_seat = {}
            for name in self.opponent_names:
                planned_by_seat = {
                    seat: int(
                        planned_seat_quotas.get(name, {}).get(seat, 0)
                    )
                    for seat in ("0", "1")
                }
                if (
                    min(planned_by_seat.values()) < 0
                    or sum(planned_by_seat.values())
                    != int(audit["planned_quotas"][name])
                    or abs(
                        planned_by_seat["0"] - planned_by_seat["1"]
                    ) > 1
                ):
                    raise RuntimeError(
                        f"Update {update} invalid planned learner seats for "
                        f"{name}: {planned_by_seat}"
                    )
                normalized_planned_seats[name] = planned_by_seat

                row = outcomes.get(name, {})
                seat_outcomes: dict[str, dict[str, int]] = {}
                seat_games: dict[str, int] = {}
                for seat in ("0", "1"):
                    prefix = f"seat_{seat}_"
                    games = int(row.get(prefix + "games", 0))
                    seat_row = {
                        key: int(row.get(prefix + key, 0))
                        for key in ("wins", "losses", "draws")
                    }
                    if min(games, *seat_row.values()) < 0:
                        raise RuntimeError(
                            f"Update {update} negative learner-seat outcome "
                            f"for {name} seat={seat}"
                        )
                    if sum(seat_row.values()) != games:
                        raise RuntimeError(
                            f"Update {update} learner-seat outcome mismatch "
                            f"for {name} seat={seat}: games={games} "
                            f"outcomes={seat_row}"
                        )
                    seat_games[seat] = games
                    seat_outcomes[seat] = {
                        "games": games,
                        **seat_row,
                    }
                if seat_games != planned_by_seat:
                    raise RuntimeError(
                        f"Update {update} learner-seat quota mismatch for "
                        f"{name}: planned={planned_by_seat} "
                        f"actual={seat_games}"
                    )
                if sum(seat_games.values()) != actual_quotas[name]:
                    raise RuntimeError(
                        f"Update {update} learner-seat games do not aggregate "
                        f"for {name}"
                    )
                for key in ("wins", "losses", "draws"):
                    if (
                        sum(seat_outcomes[seat][key] for seat in ("0", "1"))
                        != actual_outcomes[name][key]
                    ):
                        raise RuntimeError(
                            f"Update {update} learner-seat {key} do not "
                            f"aggregate for {name}"
                        )
                actual_seat_quotas[name] = seat_games
                actual_outcomes_by_seat[name] = seat_outcomes

                replacements_by_seat = {
                    seat: int(
                        (invalid_replacements_by_seat or {})
                        .get(name, {})
                        .get(seat, 0)
                    )
                    for seat in ("0", "1")
                }
                if min(replacements_by_seat.values()) < 0:
                    raise RuntimeError(
                        f"Update {update} negative invalid replacements for "
                        f"{name}: {replacements_by_seat}"
                    )
                if (
                    sum(replacements_by_seat.values())
                    != normalized_invalid_replacements[name]
                ):
                    raise RuntimeError(
                        f"Update {update} invalid replacement seat mismatch "
                        f"for {name}: aggregate="
                        f"{normalized_invalid_replacements[name]} "
                        f"by_seat={replacements_by_seat}"
                    )
                normalized_invalid_replacements_by_seat[name] = (
                    replacements_by_seat
                )

        # Mutate posterior state only after every aggregate and seat-level
        # invariant has passed, so a rejected update cannot be half-applied.
        for name in self.opponent_names:
            for key in ("wins", "losses", "draws"):
                self.observed[name][key] += actual_outcomes[name][key]
        audit["actual_quotas"] = actual_quotas
        audit["actual_outcomes"] = actual_outcomes
        audit["invalid_replacements"] = normalized_invalid_replacements
        audit["seat_assignment_mode"] = (
            "per_opponent_balanced_v1"
            if normalized_planned_seats is not None
            else "random"
        )
        audit["seat_balance_verified"] = normalized_planned_seats is not None
        if normalized_planned_seats is not None:
            assert actual_seat_quotas is not None
            assert actual_outcomes_by_seat is not None
            assert normalized_invalid_replacements_by_seat is not None
            audit["planned_seat_quotas"] = normalized_planned_seats
            audit["actual_seat_quotas"] = actual_seat_quotas
            audit["actual_outcomes_by_seat"] = actual_outcomes_by_seat
            audit["invalid_replacements_by_seat"] = (
                normalized_invalid_replacements_by_seat
            )
            audit["planned_max_seat_gap"] = max(
                abs(row["0"] - row["1"])
                for row in normalized_planned_seats.values()
            )
            audit["actual_max_seat_gap"] = max(
                abs(row["0"] - row["1"])
                for row in actual_seat_quotas.values()
            )
        audit["combined_audit_after"] = {
            name: self.combined_audit(name)
            for name in self.opponent_names
        }
        audit["posterior_after"] = self.posterior()
        return audit

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "mode": self.mode,
            "opponent_names": list(self.opponent_names),
            "base_quotas": dict(self.base_quotas),
            "caps": dict(self.caps),
            "initial_audit": copy.deepcopy(self.initial_audit),
            "observed": copy.deepcopy(self.observed),
            "games_per_update": self.games_per_update,
            "refresh_updates": self.refresh_updates,
            "seed": self.seed,
            "current_plan": copy.deepcopy(self.current_plan),
            "current_dynamic_quotas": copy.deepcopy(
                self.current_dynamic_quotas
            ),
            "current_allocation_posterior": copy.deepcopy(
                self.current_allocation_posterior
            ),
            "current_allocation_audit": copy.deepcopy(
                self.current_allocation_audit
            ),
            "last_refresh_update": self.last_refresh_update,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        identity = {
            "mode": self.mode,
            "opponent_names": self.opponent_names,
            "base_quotas": self.base_quotas,
            "caps": self.caps,
            "initial_audit": self.initial_audit,
            "games_per_update": self.games_per_update,
            "refresh_updates": self.refresh_updates,
            "seed": self.seed,
        }
        saved_identity = {
            key: state.get(key)
            for key in identity
        }
        if saved_identity != identity:
            raise ValueError(
                "Resume checkpoint opponent quota configuration mismatch"
            )
        observed = state.get("observed")
        if not isinstance(observed, dict):
            raise ValueError("Resume checkpoint lacks opponent quota observations")
        self.observed = {
            name: {
                key: int(observed.get(name, {}).get(key, 0))
                for key in ("wins", "losses", "draws")
            }
            for name in self.opponent_names
        }
        for audit in self.observed.values():
            beta_failure_posterior(audit)
        current_plan = state.get("current_plan")
        current_dynamic = state.get("current_dynamic_quotas")
        current_allocation_posterior = state.get(
            "current_allocation_posterior"
        )
        current_allocation_audit = state.get("current_allocation_audit")
        self.current_plan = (
            {name: int(current_plan[name]) for name in self.opponent_names}
            if isinstance(current_plan, dict)
            else None
        )
        self.current_dynamic_quotas = (
            {name: int(current_dynamic[name]) for name in self.opponent_names}
            if isinstance(current_dynamic, dict)
            else None
        )
        self.current_allocation_posterior = (
            {
                name: {
                    key: float(current_allocation_posterior[name][key])
                    for key in (
                        "alpha",
                        "beta",
                        "mean",
                        "std",
                        "failure_ucb",
                    )
                }
                for name in self.opponent_names
            }
            if isinstance(current_allocation_posterior, dict)
            else None
        )
        self.current_allocation_audit = (
            {
                name: {
                    key: int(current_allocation_audit[name][key])
                    for key in ("wins", "losses", "draws")
                }
                for name in self.opponent_names
            }
            if isinstance(current_allocation_audit, dict)
            else None
        )
        last_refresh = state.get("last_refresh_update")
        self.last_refresh_update = (
            int(last_refresh) if last_refresh is not None else None
        )
        self.resume_state_loaded = True


def instantiate_model_from_bc(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> EntityOptionPolicy:
    config = checkpoint["config"]
    model = EntityOptionPolicy(
        hash_size=int(config["hash_size"]),
        categorical_dim=int(config["categorical_dim"]),
        model_dim=int(config["model_dim"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        max_state_entities=int(config["max_state_entities"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    old_count_layer = model.count_head[-1]
    if not isinstance(old_count_layer, nn.Linear):
        raise TypeError("Unexpected count head")
    if old_count_layer.out_features != BC_MAX_ACTION_COUNT + 1:
        raise ValueError(
            f"Unexpected BC count classes: {old_count_layer.out_features}"
        )
    expanded_count_layer = nn.Linear(
        old_count_layer.in_features,
        MAX_ACTION_COUNT + 1,
    )
    nn.init.zeros_(expanded_count_layer.weight)
    nn.init.constant_(expanded_count_layer.bias, -10.0)
    with torch.no_grad():
        expanded_count_layer.weight[: old_count_layer.out_features].copy_(
            old_count_layer.weight
        )
        expanded_count_layer.bias[: old_count_layer.out_features].copy_(
            old_count_layer.bias
        )
    model.count_head[-1] = expanded_count_layer
    model.to(device)
    model.eval()
    return model


def checkpoint_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    embedded_model_config = checkpoint.get("model_config")
    config = (
        embedded_model_config
        if isinstance(embedded_model_config, dict)
        else checkpoint["config"]
    )
    return {
        "hash_size": int(config["hash_size"]),
        "categorical_dim": int(config["categorical_dim"]),
        "model_dim": int(config["model_dim"]),
        "layers": int(config["layers"]),
        "heads": int(config["heads"]),
        "dropout": float(config["dropout"]),
        "max_state_entities": int(config["max_state_entities"]),
        "entity_fields": int(config["entity_fields"]),
        "option_fields": int(config["option_fields"]),
    }


def validate_checkpoint_model_state_schema(
    checkpoint: dict[str, Any],
    expected_state: dict[str, torch.Tensor],
    source: Path,
) -> None:
    """Fail clearly when checkpoint tensors cannot represent this learner."""
    raw_state = checkpoint.get("model_state_dict")
    if not isinstance(raw_state, dict):
        raise ValueError(
            f"Resume checkpoint {source} has no model_state_dict mapping"
        )
    expected_keys = set(expected_state)
    actual_keys = set(raw_state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    shape_mismatches: dict[str, dict[str, list[int]]] = {}
    for name in sorted(expected_keys & actual_keys):
        tensor = raw_state[name]
        if not isinstance(tensor, torch.Tensor):
            shape_mismatches[name] = {
                "expected": list(expected_state[name].shape),
                "actual": [],
            }
            continue
        if tensor.shape != expected_state[name].shape:
            shape_mismatches[name] = {
                "expected": list(expected_state[name].shape),
                "actual": list(tensor.shape),
            }
    if missing or unexpected or shape_mismatches:
        details = {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatches": shape_mismatches,
        }
        raise ValueError(
            f"Resume checkpoint {source} model_state_dict schema is "
            "incompatible with the BC-initialized learner: "
            + json.dumps(details, sort_keys=True)
        )


def model_state_sha256(model: nn.Module) -> str:
    """Hash names, tensor metadata, and bytes of the effective model state."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor model state entry: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def structured_state_sha256(value: Any) -> str:
    """Deterministically hash nested optimizer-style state."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
            digest.update(b"\0")
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            for child in item:
                update(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(repr(item).encode("utf-8") + b"\0")
        else:
            raise TypeError(
                f"Unsupported structured-state value: {type(item).__name__}"
            )

    update(value)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def instantiate_model_from_checkpoint(
    checkpoint: dict[str, Any],
    bc_checkpoint: dict[str, Any],
    device: torch.device,
) -> EntityOptionPolicy:
    """Load either a BC actor or a PPO actor with the same architecture."""
    feature_version = checkpoint.get("feature_version")
    if feature_version == BC_FEATURE_VERSION:
        return instantiate_model_from_bc(checkpoint, device)
    if feature_version == PPO_FEATURE_VERSION:
        model = instantiate_model_from_bc(bc_checkpoint, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
    raise ValueError(
        "Unsupported opponent feature version: "
        f"{feature_version!r}; expected {BC_FEATURE_VERSION!r} or "
        f"{PPO_FEATURE_VERSION!r}"
    )


def validate_resume_learner_weight_selection(
    resume_checkpoint: Path | None,
    weight_source: str,
    reset_optimizer_on_resume: bool,
) -> None:
    """Validate the explicit learner-weight source for a resumed run.

    Replacing a resumed learner with BC weights invalidates every optimizer
    moment saved for the resumed PPO weights.  Require an explicit optimizer
    reset instead of silently pairing those incompatible states.
    """
    if weight_source not in RESUME_LEARNER_WEIGHT_SOURCES:
        raise ValueError(
            f"Unsupported resume learner weight source: {weight_source!r}"
        )
    if resume_checkpoint is None:
        if weight_source != "resume":
            raise ValueError(
                "--resume-learner-weights bc requires --resume"
            )
        return
    if weight_source == "bc" and not reset_optimizer_on_resume:
        raise ValueError(
            "--resume-learner-weights bc requires "
            "--reset-optimizer-on-resume"
        )


def apply_resume_learner_weights(
    model: EntityOptionPolicy,
    resume_checkpoint: dict[str, Any],
    weight_source: str,
    reset_optimizer_on_resume: bool,
) -> str:
    """Apply the selected full-model weights and return their source label.

    ``model`` is already initialized from the expanded BC checkpoint.  The
    legacy ``resume`` mode replaces it with the full saved PPO state; ``bc``
    deliberately leaves the full BC initialization intact.
    """
    validate_resume_learner_weight_selection(
        Path("<loaded-resume-checkpoint>"),
        weight_source,
        reset_optimizer_on_resume,
    )
    if weight_source == "resume":
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        return "resume_checkpoint"
    return "bc_checkpoint"


def checkpoint_trainable_scope(checkpoint: dict[str, Any]) -> str:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        return "full"
    scope = str(config.get("trainable_scope", "full"))
    if scope not in TRAINABLE_SCOPES:
        raise ValueError(
            f"Checkpoint has unsupported trainable_scope {scope!r}"
        )
    return scope


def configure_trainable_scope(
    model: EntityOptionPolicy,
    scope: str,
) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[str, Any]]:
    """Freeze the model to a named PPO scope and return optimizer groups."""
    if scope not in TRAINABLE_SCOPES:
        raise ValueError(f"Unsupported trainable scope: {scope!r}")
    model.requires_grad_(False)
    if scope == "full":
        model.requires_grad_(True)
    else:
        actor_modules: list[nn.Module] = [
            model.actor_query,
            model.actor_key,
            model.actor_residual,
            model.count_head,
        ]
        if scope in {"last_block_heads", "last_two_blocks_heads"}:
            required_layers = 2 if scope == "last_two_blocks_heads" else 1
            if len(model.transformer.layers) < required_layers:
                raise ValueError(
                    f"{scope} requires at least {required_layers} "
                    "transformer layer(s)"
                )
            actor_modules.extend(
                model.transformer.layers[-required_layers:]
            )
            if model.transformer.norm is not None:
                actor_modules.append(model.transformer.norm)
        for module in actor_modules:
            module.requires_grad_(True)
        model.value_head.requires_grad_(True)

    value_parameter_ids = {
        id(parameter)
        for parameter in model.value_head.parameters()
        if parameter.requires_grad
    }
    actor_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in value_parameter_ids
    ]
    value_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) in value_parameter_ids
    ]
    actor_parameters = [parameter for _, parameter in actor_named]
    value_parameters = [parameter for _, parameter in value_named]
    if not actor_parameters or not value_parameters:
        raise RuntimeError(
            f"Trainable scope {scope!r} produced an empty optimizer group"
        )
    actor_ids = {id(parameter) for parameter in actor_parameters}
    value_ids = {id(parameter) for parameter in value_parameters}
    if actor_ids & value_ids:
        raise RuntimeError("Actor/value optimizer groups overlap")
    all_trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if actor_ids | value_ids != all_trainable_ids:
        raise RuntimeError("Actor/value groups do not cover trainable parameters")

    total_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    actor_parameter_count = sum(
        parameter.numel() for parameter in actor_parameters
    )
    value_parameter_count = sum(
        parameter.numel() for parameter in value_parameters
    )
    trainable_parameter_count = (
        actor_parameter_count + value_parameter_count
    )
    manifest = {
        "scope": scope,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_ratio": (
            trainable_parameter_count / max(total_parameter_count, 1)
        ),
        "actor_parameter_count": actor_parameter_count,
        "value_parameter_count": value_parameter_count,
        "actor_tensor_count": len(actor_parameters),
        "value_tensor_count": len(value_parameters),
        "actor_parameter_names": [name for name, _ in actor_named],
        "value_parameter_names": [name for name, _ in value_named],
    }
    return actor_parameters, value_parameters, manifest


def validate_optimizer_resume_compatibility(
    checkpoint: dict[str, Any],
    requested_scope: str,
    parameter_manifest: dict[str, Any],
    requested_advantage_normalization: str = "global",
    requested_ppo_objective: str = "standard",
    requested_objective_config: dict[str, Any] | None = None,
    requested_policy_temperature: float = 1.0,
    requested_actor_reduction: str = "transition_mean",
    requested_value_trunk_gradient_scale: float = 1.0,
    requested_actor_value_gradient_mode: str = "scalar",
) -> None:
    saved_scope = checkpoint_trainable_scope(checkpoint)
    if saved_scope != requested_scope:
        raise ValueError(
            "Resume checkpoint trainable scope mismatch: "
            f"checkpoint={saved_scope!r} requested={requested_scope!r}; "
            "use --reset-optimizer-on-resume"
        )
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        saved_config = {}
    saved_advantage_normalization = str(
        saved_config.get("advantage_normalization", "global")
    )
    if saved_advantage_normalization != requested_advantage_normalization:
        raise ValueError(
            "Resume checkpoint advantage normalization mismatch: "
            f"checkpoint={saved_advantage_normalization!r} "
            f"requested={requested_advantage_normalization!r}; "
            "use --reset-optimizer-on-resume"
        )
    saved_objective = str(saved_config.get("ppo_objective", "standard"))
    if saved_objective != requested_ppo_objective:
        raise ValueError(
            "Resume checkpoint PPO objective mismatch: "
            f"checkpoint={saved_objective!r} "
            f"requested={requested_ppo_objective!r}; "
            "use --reset-optimizer-on-resume"
        )
    saved_actor_reduction = str(
        saved_config.get("actor_reduction", "transition_mean")
    )
    if saved_actor_reduction != requested_actor_reduction:
        raise ValueError(
            "Resume checkpoint actor reduction mismatch: "
            f"checkpoint={saved_actor_reduction!r} "
            f"requested={requested_actor_reduction!r}; "
            "use --reset-optimizer-on-resume"
        )
    saved_policy_temperature = float(
        saved_config.get("policy_temperature", 1.0)
    )
    if saved_policy_temperature != requested_policy_temperature:
        raise ValueError(
            "Resume checkpoint policy temperature mismatch: "
            f"checkpoint={saved_policy_temperature!r} "
            f"requested={requested_policy_temperature!r}; "
            "use --reset-optimizer-on-resume"
        )
    saved_value_trunk_gradient_scale = float(
        saved_config.get("value_trunk_gradient_scale", 1.0)
    )
    if (
        saved_value_trunk_gradient_scale
        != requested_value_trunk_gradient_scale
    ):
        raise ValueError(
            "Resume checkpoint value-trunk gradient scale mismatch: "
            f"checkpoint={saved_value_trunk_gradient_scale!r} "
            f"requested={requested_value_trunk_gradient_scale!r}; "
            "use --reset-optimizer-on-resume"
        )
    saved_actor_value_gradient_mode = str(
        saved_config.get("actor_value_gradient_mode", "scalar")
    )
    if saved_actor_value_gradient_mode != requested_actor_value_gradient_mode:
        raise ValueError(
            "Resume checkpoint actor/value gradient mode mismatch: "
            f"checkpoint={saved_actor_value_gradient_mode!r} "
            f"requested={requested_actor_value_gradient_mode!r}; "
            "use --reset-optimizer-on-resume"
        )
    if (
        requested_ppo_objective == "constrained"
        and requested_objective_config is not None
    ):
        mismatches = {
            key: {
                "checkpoint": saved_config.get(key),
                "requested": requested_value,
            }
            for key, requested_value in requested_objective_config.items()
            if saved_config.get(key) != requested_value
        }
        if mismatches:
            raise ValueError(
                "Resume checkpoint constrained objective configuration "
                "mismatch; use --reset-optimizer-on-resume: "
                + json.dumps(mismatches, sort_keys=True)
            )
    saved_manifest = checkpoint.get("optimizer_parameter_names")
    if isinstance(saved_manifest, dict):
        for group in ("actor", "value"):
            expected_names = parameter_manifest[f"{group}_parameter_names"]
            saved_names = saved_manifest.get(group)
            if saved_names != expected_names:
                raise ValueError(
                    f"Resume optimizer {group} parameter manifest mismatch; "
                    "use --reset-optimizer-on-resume"
                )
        return
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        return
    saved_groups = optimizer_state.get("param_groups")
    expected_sizes = [
        parameter_manifest["actor_tensor_count"],
        parameter_manifest["value_tensor_count"],
    ]
    if (
        not isinstance(saved_groups, list)
        or len(saved_groups) != len(expected_sizes)
        or [
            len(group.get("params", []))
            for group in saved_groups
            if isinstance(group, dict)
        ]
        != expected_sizes
    ):
        raise ValueError(
            "Resume optimizer parameter-group shape mismatch; "
            "use --reset-optimizer-on-resume"
        )


def checkpoint_single_deck_hash(
    checkpoint: dict[str, Any],
) -> str | None:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        return None
    deck_hashes = config.get("deck_hashes")
    if isinstance(deck_hashes, (list, tuple)) and len(deck_hashes) == 1:
        return str(deck_hashes[0])
    embedded_hash = checkpoint.get("learner_deck_hash")
    if isinstance(embedded_hash, str):
        return embedded_hash
    deck_path_value = config.get("deck")
    if isinstance(deck_path_value, str):
        deck_path = Path(deck_path_value)
        if deck_path.is_file():
            return compute_deck_hash(read_deck(deck_path))
    return None


def validate_checkpoint_deck(
    checkpoint: dict[str, Any],
    deck_hash: str,
    checkpoint_path: Path,
    deck_path: Path,
) -> None:
    expected_hash = checkpoint_single_deck_hash(checkpoint)
    if expected_hash is not None and deck_hash != expected_hash:
        raise ValueError(
            f"Deck hash mismatch for {checkpoint_path}: {deck_path} has "
            f"{deck_hash}, checkpoint config.deck_hashes expects "
            f"{expected_hash}"
        )


def live_feature(
    observation: dict[str, Any],
    model_config: dict[str, Any],
    deck_hash: str,
) -> dict[str, Any] | None:
    current = observation.get("current") or {}
    row = {
        "observation": observation,
        "seat": int(current.get("yourIndex", 0) or 0),
        "deck_hash": deck_hash,
        "team_name": "",
        "action": [],
        "terminal_reward": 0.0,
        "sample_weight": 1.0,
    }
    return featurize_row(
        row,
        hash_size=model_config["hash_size"],
        max_state_entities=model_config["max_state_entities"],
    )


def collate_features(
    features: list[dict[str, Any]],
    model_config: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = collate_decisions(
        features,
        max_state_entities=model_config["max_state_entities"],
        entity_fields=model_config["entity_fields"],
        option_fields=model_config["option_fields"],
    )
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def collate_features_cpu(
    features: list[dict[str, Any]],
    model_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Collate a rollout once; PPO epochs then index this tensor cache."""
    return collate_decisions(
        features,
        max_state_entities=model_config["max_state_entities"],
        entity_fields=model_config["entity_fields"],
        option_fields=model_config["option_fields"],
    )


def count_allowed_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    values = torch.arange(
        MAX_ACTION_COUNT + 1,
        device=batch["min_counts"].device,
    ).unsqueeze(0)
    option_counts = batch["option_mask"].sum(dim=1).unsqueeze(1)
    minimum = batch["min_counts"].clamp(0, MAX_ACTION_COUNT).unsqueeze(1)
    maximum = torch.minimum(
        batch["max_counts"].clamp(0, MAX_ACTION_COUNT).unsqueeze(1),
        option_counts,
    )
    return (values >= minimum) & (values <= maximum)


def validate_policy_temperature(temperature: float) -> float:
    resolved = float(temperature)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("Policy temperature must be finite and positive")
    return resolved


def validate_value_trunk_gradient_scale(scale: float) -> float:
    resolved = float(scale)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(
            "Value-trunk gradient scale must be finite and in [0, 1]"
        )
    return resolved


def validate_actor_value_gradient_configuration(
    *,
    mode: str,
    ppo_objective: str,
    trainable_scope: str,
    value_trunk_gradient_scale: float,
    audit_only: bool = False,
    actor_reduction_audit_only: bool = False,
) -> str:
    resolved = str(mode)
    if resolved not in ACTOR_VALUE_GRADIENT_MODES:
        raise ValueError(
            f"Unsupported actor/value gradient mode: {resolved!r}"
        )
    if audit_only and actor_reduction_audit_only:
        raise ValueError(
            "Actor/value gradient audit is mutually exclusive with "
            "actor-reduction audit"
        )
    if audit_only and resolved != "actor_priority_value_pcgrad":
        raise ValueError(
            "Actor/value gradient audit requires "
            "actor_priority_value_pcgrad mode"
        )
    if resolved == "actor_priority_value_pcgrad":
        if ppo_objective != "standard":
            raise ValueError(
                "actor_priority_value_pcgrad requires standard PPO"
            )
        if value_trunk_gradient_scale != 1.0:
            raise ValueError(
                "actor_priority_value_pcgrad requires "
                "value_trunk_gradient_scale=1.0"
            )
        if trainable_scope not in {
            "full",
            "last_block_heads",
            "last_two_blocks_heads",
        }:
            raise ValueError(
                "actor_priority_value_pcgrad requires a shared trainable "
                "trunk scope (full, last_block_heads, or "
                "last_two_blocks_heads)"
            )
    return resolved


def actor_value_gradient_manifest(
    mode: str,
    *,
    audit_only: bool = False,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "audit_only": bool(audit_only),
        "actor_side": (
            "policy_loss + bc_kl * anchor_kl - entropy_coefficient * entropy"
        ),
        "value_side": "value_coefficient * value_loss",
        "projection": (
            "conflicting_value_gradient_off_actor_gradient"
            if mode == "actor_priority_value_pcgrad"
            else None
        ),
        "projection_scope": (
            "actor_optimizer_parameters_with_both_actor_and_value_gradients"
            if mode == "actor_priority_value_pcgrad"
            else None
        ),
        "value_head": "untouched",
        "actor_only_heads": "untouched",
        "bc_replay": "untouched",
        "guarantee_scope": "pre_clip_standard_ppo_minibatch_only",
        "optimizer_note": (
            "AdamW clipping and weight decay execute after gradient surgery"
        ),
    }


def value_trunk_gradient_audit(scale: float) -> dict[str, Any]:
    resolved = validate_value_trunk_gradient_scale(scale)
    return {
        "scale": resolved,
        "applies_to": "standard_ppo_value_loss_only",
        "value_head_receives_full_gradient": True,
        "forward_values_unchanged": True,
        "shared_trunk_scope": (
            "embedding_encoders_positions_transformer_before_global_encoded"
        ),
    }


@torch.no_grad()
def sample_ordered_actions(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    deterministic: bool,
    canonicalize_order: bool = False,
    temperature: float = 1.0,
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample cardinality and an ordered option sequence without replacement."""
    temperature = validate_policy_temperature(temperature)
    policy_logits = outputs["policy_logits"].float() / temperature
    count_logits = outputs["count_logits"].float() / temperature
    allowed_counts = count_allowed_mask(batch)
    if not allowed_counts.any(dim=1).all():
        raise ValueError("No allowed cardinality for at least one observation")
    masked_count_logits = count_logits.masked_fill(~allowed_counts, -1e9)
    fixed = batch["min_counts"] == batch["max_counts"]
    if deterministic:
        counts = masked_count_logits.argmax(dim=1)
    else:
        counts = torch.distributions.Categorical(
            logits=masked_count_logits
        ).sample()
    counts = torch.where(
        fixed,
        batch["min_counts"].clamp(0, MAX_ACTION_COUNT),
        counts,
    )
    counts = torch.minimum(counts, batch["option_mask"].sum(dim=1))

    count_log_probs_all = F.log_softmax(masked_count_logits, dim=1)
    count_probs = count_log_probs_all.exp()
    count_log_prob = count_log_probs_all.gather(
        1,
        counts.unsqueeze(1),
    ).squeeze(1)
    count_log_prob = torch.where(fixed, torch.zeros_like(count_log_prob), count_log_prob)
    count_entropy = -(count_probs * count_log_probs_all).sum(dim=1)
    count_entropy = torch.where(fixed, torch.zeros_like(count_entropy), count_entropy)

    actions: list[list[int]] = []
    selection_log_probs = torch.zeros_like(count_log_prob)
    selection_entropies = torch.zeros_like(count_log_prob)
    for row_index in range(policy_logits.shape[0]):
        remaining = batch["option_mask"][row_index].clone()
        row_actions: list[int] = []
        row_log_prob = torch.zeros((), device=policy_logits.device)
        row_entropy = torch.zeros((), device=policy_logits.device)
        for _ in range(int(counts[row_index])):
            logits = policy_logits[row_index].masked_fill(~remaining, -1e9)
            log_probs = F.log_softmax(logits, dim=0)
            probs = log_probs.exp()
            if deterministic:
                chosen = int(logits.argmax())
            else:
                chosen = int(torch.multinomial(probs, 1))
            row_actions.append(chosen)
            row_log_prob = row_log_prob + log_probs[chosen]
            row_entropy = row_entropy - (probs * log_probs).sum()
            remaining[chosen] = False
        if canonicalize_order:
            # Neural BC was trained on action sets, while its expert replays
            # use ascending indices.  Keep that compatibility mode only for
            # the frozen BC reference.  PPO policies retain the Plackett-Luce
            # order because SKILL_ORDER decisions are order-sensitive.
            row_actions.sort()
        actions.append(row_actions)
        selection_log_probs[row_index] = row_log_prob
        selection_entropies[row_index] = row_entropy
    return (
        actions,
        count_log_prob + selection_log_probs,
        count_entropy + selection_entropies,
        torch.sigmoid(outputs["value_logits"].float()),
    )


def ordered_action_log_prob_entropy(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    action_sequences: torch.Tensor,
    action_counts: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    temperature = validate_policy_temperature(temperature)
    policy_logits = outputs["policy_logits"].float() / temperature
    count_logits = outputs["count_logits"].float() / temperature
    allowed_counts = count_allowed_mask(batch)
    masked_count_logits = count_logits.masked_fill(~allowed_counts, -1e9)
    fixed = batch["min_counts"] == batch["max_counts"]
    count_log_probs = F.log_softmax(masked_count_logits, dim=1)
    count_probs = count_log_probs.exp()
    count_log_prob = count_log_probs.gather(
        1,
        action_counts.unsqueeze(1),
    ).squeeze(1)
    count_log_prob = torch.where(fixed, torch.zeros_like(count_log_prob), count_log_prob)
    count_entropy = -(count_probs * count_log_probs).sum(dim=1)
    count_entropy = torch.where(fixed, torch.zeros_like(count_entropy), count_entropy)

    selected = torch.zeros_like(batch["option_mask"])
    selection_log_prob = torch.zeros_like(count_log_prob)
    selection_entropy = torch.zeros_like(count_log_prob)
    entropy_units = (~fixed).float()
    for step in range(MAX_ACTION_COUNT):
        active = step < action_counts
        if not active.any():
            break
        allowed = batch["option_mask"] & ~selected
        non_forced = allowed.sum(dim=1) > 1
        logits = policy_logits.masked_fill(~allowed, -1e9)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        chosen = action_sequences[:, step].clamp_min(0)
        chosen_log_prob = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs).sum(dim=1)
        selection_log_prob = selection_log_prob + torch.where(
            active,
            chosen_log_prob,
            torch.zeros_like(chosen_log_prob),
        )
        selection_entropy = selection_entropy + torch.where(
            active & non_forced,
            entropy,
            torch.zeros_like(entropy),
        )
        entropy_units = entropy_units + (active & non_forced).float()
        selected.scatter_(1, chosen.unsqueeze(1), active.unsqueeze(1))
    normalized_entropy = (
        count_entropy + selection_entropy
    ) / entropy_units.clamp_min(1.0)
    return count_log_prob + selection_log_prob, normalized_entropy


def reference_policy_kl(
    outputs: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    action_sequences: torch.Tensor,
    action_counts: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """BC anchor along count and each sampled without-replacement prefix."""
    temperature = validate_policy_temperature(temperature)
    allowed_counts = count_allowed_mask(batch)
    current_count_log = F.log_softmax(
        (outputs["count_logits"].float() / temperature).masked_fill(
            ~allowed_counts,
            -1e9,
        ),
        dim=1,
    )
    reference_count_log = F.log_softmax(
        (reference["count_logits"].float() / temperature).masked_fill(
            ~allowed_counts,
            -1e9,
        ),
        dim=1,
    )
    reference_count_prob = reference_count_log.exp()
    count_kl = (
        reference_count_prob * (reference_count_log - current_count_log)
    ).sum(dim=1)
    fixed = batch["min_counts"] == batch["max_counts"]
    count_kl = torch.where(fixed, torch.zeros_like(count_kl), count_kl)

    total_kl = count_kl
    choice_units = (~fixed).float()
    selected = torch.zeros_like(batch["option_mask"])
    current_logits = outputs["policy_logits"].float() / temperature
    reference_logits = reference["policy_logits"].float() / temperature
    for step in range(MAX_ACTION_COUNT):
        active = step < action_counts
        if not active.any():
            break
        allowed = batch["option_mask"] & ~selected
        non_forced = allowed.sum(dim=1) > 1
        current_log = F.log_softmax(
            current_logits.masked_fill(~allowed, -1e9),
            dim=1,
        )
        reference_log = F.log_softmax(
            reference_logits.masked_fill(~allowed, -1e9),
            dim=1,
        )
        reference_prob = reference_log.exp()
        prefix_kl = (
            reference_prob * (reference_log - current_log)
        ).sum(dim=1)
        include = active & non_forced
        total_kl = total_kl + torch.where(
            include,
            prefix_kl,
            torch.zeros_like(prefix_kl),
        )
        choice_units = choice_units + include.float()
        chosen = action_sequences[:, step].clamp_min(0)
        selected.scatter_(1, chosen.unsqueeze(1), active.unsqueeze(1))
    return total_kl / choice_units.clamp_min(1.0)


def model_forward(
    model: EntityOptionPolicy,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    value_trunk_gradient_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        if value_trunk_gradient_scale == 1.0:
            return model(batch)
        return model(
            batch,
            value_trunk_gradient_scale=value_trunk_gradient_scale,
        )


def gradient_l2_norm(parameters: list[nn.Parameter]) -> torch.Tensor:
    squared_norms = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared_norms:
        if parameters:
            return parameters[0].detach().new_zeros((), dtype=torch.float32)
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(squared_norms).sum().sqrt()


def collate_cached_replay_rows(
    row_references: list[tuple[dict[str, torch.Tensor], int]],
) -> dict[str, torch.Tensor]:
    """Copy cached, already-collated rows into one correctly padded batch."""
    if not row_references:
        raise ValueError("Cannot collate an empty cached replay row list")
    first_batch = row_references[0][0]
    keys = tuple(first_batch)
    output: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [batch[key] for batch, _ in row_references]
        reference = tensors[0]
        if reference.ndim < 1:
            raise ValueError(f"Replay tensor {key!r} has no batch dimension")
        if any(
            tensor.ndim != reference.ndim
            or tensor.dtype != reference.dtype
            or tensor.device != reference.device
            for tensor in tensors
        ):
            raise ValueError(f"Incompatible cached replay tensor {key!r}")
        tail_shape = tuple(
            max(tensor.shape[dimension] for tensor in tensors)
            for dimension in range(1, reference.ndim)
        )
        fill_value = -1 if key == "action_sequences" else 0
        combined = torch.full(
            (len(row_references), *tail_shape),
            fill_value,
            dtype=reference.dtype,
            device=reference.device,
        )
        for output_index, ((_, row_index), tensor) in enumerate(
            zip(row_references, tensors)
        ):
            if row_index < 0 or row_index >= tensor.shape[0]:
                raise IndexError(
                    f"Replay row {row_index} is outside tensor {key!r}"
                )
            source = tensor[row_index]
            destination = (output_index,) + tuple(
                slice(0, size) for size in source.shape
            )
            combined[destination] = source
        output[key] = combined
    if any(tuple(batch) != keys for batch, _ in row_references):
        raise ValueError("Cached replay batches have inconsistent tensor keys")
    return output


def stratify_bc_replay_batches(
    batches: list[dict[str, torch.Tensor]],
    context34_rows_per_batch: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    """Rebatch a fixed cache with an exact context-34 row quota."""
    if context34_rows_per_batch == 0:
        return batches
    if not batches:
        raise ValueError("Context-stratified BC replay requires cached batches")
    batch_sizes = {
        int(batch["contexts"].shape[0])
        for batch in batches
    }
    if len(batch_sizes) != 1:
        raise ValueError("Cached replay batches have inconsistent row counts")
    batch_size = batch_sizes.pop()
    if not 0 < context34_rows_per_batch < batch_size:
        raise ValueError(
            "Context-34 replay rows per batch must be in "
            f"[1, {batch_size - 1}]"
        )

    context_rows: list[tuple[dict[str, torch.Tensor], int]] = []
    ordinary_rows: list[tuple[dict[str, torch.Tensor], int]] = []
    for batch in batches:
        contexts = batch["contexts"]
        for row_index in range(batch_size):
            destination = (
                context_rows
                if int(contexts[row_index]) == SKILL_ORDER_CONTEXT
                else ordinary_rows
            )
            destination.append((batch, row_index))
    if not context_rows:
        raise RuntimeError(
            "BC replay context quota requested, but the cache has no "
            "context-34 rows"
        )
    if not ordinary_rows:
        raise RuntimeError(
            "BC replay context quota requested, but the cache has no "
            "non-context-34 rows"
        )

    rng = random.Random(seed)
    rng.shuffle(context_rows)
    rng.shuffle(ordinary_rows)
    context_cursor = 0
    ordinary_cursor = 0
    stratified: list[dict[str, torch.Tensor]] = []
    ordinary_rows_per_batch = batch_size - context34_rows_per_batch
    for _ in batches:
        selected: list[tuple[dict[str, torch.Tensor], int]] = []
        for _ in range(context34_rows_per_batch):
            selected.append(context_rows[context_cursor % len(context_rows)])
            context_cursor += 1
        for _ in range(ordinary_rows_per_batch):
            selected.append(ordinary_rows[ordinary_cursor % len(ordinary_rows)])
            ordinary_cursor += 1
        rng.shuffle(selected)
        stratified.append(collate_cached_replay_rows(selected))

    actual_context_rows = [
        int((batch["contexts"] == SKILL_ORDER_CONTEXT).sum())
        for batch in stratified
    ]
    if any(
        rows != context34_rows_per_batch
        for rows in actual_context_rows
    ):
        raise RuntimeError("Failed to enforce the BC replay context-34 quota")
    log(
        "bc_replay_context34_stratification "
        f"rows_per_batch={context34_rows_per_batch} "
        f"source_context_rows={len(context_rows)} "
        f"source_ordinary_rows={len(ordinary_rows)} "
        f"output_context_rows={sum(actual_context_rows)} "
        f"batches={len(stratified)}"
    )
    return stratified


def build_bc_replay_batches(
    config: PPOConfig,
    model_config: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    if (
        not config.bc_replay_data
        or config.bc_replay_batches <= 0
        or config.bc_replay_steps <= 0
    ):
        return []
    requested_rows = config.bc_replay_batches * config.bc_replay_batch_size
    with zipfile.ZipFile(config.bc_replay_data) as archive:
        replay_shards = [
            name
            for name in archive.namelist()
            if name.startswith(f"{config.bc_replay_split}/")
            and name.endswith(".jsonl")
        ]
    if not replay_shards:
        raise RuntimeError(
            "BC replay archive contains no "
            f"{config.bc_replay_split!r} shards"
        )
    worker_limit = min(
        config.bc_replay_workers,
        len(replay_shards),
        config.bc_replay_batches,
    )
    worker_count = next(
        candidate
        for candidate in range(worker_limit, 0, -1)
        if config.bc_replay_batches % candidate == 0
    )
    dataset = ZipDecisionDataset(
        archive_path=Path(config.bc_replay_data),
        split=config.bc_replay_split,
        max_rows=requested_rows,
        split_seed=config.seed,
        shuffle_seed=config.seed + 41,
        epoch=0,
        hash_size=int(model_config["hash_size"]),
        max_state_entities=int(model_config["max_state_entities"]),
        use_trajectory_weights=False,
        deck_hashes=(),
        team_names=(),
        split_mode="archive",
    )
    loader = DataLoader(
        dataset,
        batch_size=config.bc_replay_batch_size,
        num_workers=worker_count,
        collate_fn=partial(
            collate_decisions,
            max_state_entities=int(model_config["max_state_entities"]),
            entity_fields=int(model_config["entity_fields"]),
            option_fields=int(model_config["option_fields"]),
        ),
    )
    batches: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        if int(batch["action_counts"].shape[0]) < config.bc_replay_batch_size:
            continue
        batches.append(batch)
        if len(batches) >= config.bc_replay_batches:
            break
    if not batches:
        raise RuntimeError("BC replay data produced no complete batches")
    batches = stratify_bc_replay_batches(
        batches,
        config.bc_replay_context34_rows_per_batch,
        seed=config.seed + 73,
    )
    log(
        f"bc_replay_cache rows={sum(int(batch['action_counts'].shape[0]) for batch in batches)} "
        f"batches={len(batches)} split={config.bc_replay_split} "
        f"shards={len(replay_shards)} "
        f"workers={worker_count}"
    )
    return batches


def bc_expert_actor_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_mode: str = "set",
    order_context_weight: float = 1.0,
    non_context34_fixed_multi_action_order_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_mode not in {"set", "ordered", "hybrid_ordered"}:
        raise ValueError(f"Unsupported BC replay loss: {loss_mode!r}")
    if (
        not math.isfinite(non_context34_fixed_multi_action_order_weight)
        or non_context34_fixed_multi_action_order_weight <= 0.0
    ):
        raise ValueError(
            "non_context34_fixed_multi_action_order_weight must be finite "
            "and positive"
        )
    if (
        loss_mode != "ordered"
        and non_context34_fixed_multi_action_order_weight != 1.0
    ):
        raise ValueError(
            "non_context34_fixed_multi_action_order_weight requires ordered "
            "replay loss"
        )
    logits = outputs["policy_logits"].float()
    mask = batch["option_mask"]
    targets = batch["targets"].float()
    weights = batch["sample_weights"].float()
    action_counts = batch["action_counts"]
    contexts = batch["contexts"]

    log_probs = F.log_softmax(logits, dim=-1)
    normalized_targets = targets / action_counts.clamp_min(1).unsqueeze(1)
    pointer_per_row = -(normalized_targets * log_probs).sum(dim=1)
    pointer_active = action_counts > 0
    flexible = batch["min_counts"] != batch["max_counts"]

    def weighted_mean(
        per_row: torch.Tensor,
        row_weights: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        if active.any():
            return (
                per_row[active] * row_weights[active]
            ).sum() / row_weights[active].sum().clamp_min(1.0)
        return logits.sum() * 0.0

    ordered_per_row = torch.zeros_like(pointer_per_row)
    if loss_mode != "set":
        sequences = batch.get("action_sequences")
        if sequences is None:
            raise ValueError(
                f"BC replay loss {loss_mode!r} requires action_sequences"
            )
        selected = torch.zeros_like(mask)
        for step in range(sequences.shape[1]):
            active = step < action_counts
            if not active.any():
                break
            raw_chosen = sequences[:, step]
            active_chosen = raw_chosen[active]
            if (
                (active_chosen < 0).any()
                or (active_chosen >= logits.shape[1]).any()
            ):
                raise ValueError(
                    "BC replay contains a missing or out-of-range ordered action"
                )
            active_rows = active.nonzero(as_tuple=False).squeeze(1)
            if not (
                mask[active_rows, active_chosen]
                & ~selected[active_rows, active_chosen]
            ).all():
                raise ValueError(
                    "BC replay contains an illegal or duplicate ordered action"
                )
            allowed = mask & ~selected
            step_log_probs = F.log_softmax(
                logits.masked_fill(~allowed, -1e9),
                dim=1,
            )
            chosen = raw_chosen.clamp(0, logits.shape[1] - 1)
            chosen_log_prob = step_log_probs.gather(
                1,
                chosen.unsqueeze(1),
            ).squeeze(1)
            ordered_per_row = ordered_per_row - torch.where(
                active,
                chosen_log_prob,
                torch.zeros_like(chosen_log_prob),
            )
            selected.scatter_(1, chosen.unsqueeze(1), active.unsqueeze(1))

    effective_weights = weights * torch.where(
        contexts == SKILL_ORDER_CONTEXT,
        torch.full_like(weights, order_context_weight),
        torch.ones_like(weights),
    )
    context_34 = contexts == SKILL_ORDER_CONTEXT
    non_context34_fixed_multi_action = (
        (~context_34)
        & ~flexible
        & (action_counts > 1)
    )
    ordered_selection_weights = effective_weights * torch.where(
        non_context34_fixed_multi_action,
        torch.full_like(
            weights,
            non_context34_fixed_multi_action_order_weight,
        ),
        torch.ones_like(weights),
    )

    if loss_mode == "set":
        selection_per_row = pointer_per_row
        selection_weights = weights
    elif loss_mode == "ordered":
        selection_per_row = ordered_per_row
        selection_weights = ordered_selection_weights
    else:
        selection_per_row = torch.where(
            context_34,
            ordered_per_row,
            pointer_per_row,
        )
        selection_weights = effective_weights
    selection_loss = weighted_mean(
        selection_per_row,
        selection_weights,
        pointer_active,
    )
    selection_base_weight_sum = weights[pointer_active].sum()
    selection_effective_weight_sum = selection_weights[pointer_active].sum()
    fixed_multi_base_weight_sum = weights[
        pointer_active & non_context34_fixed_multi_action
    ].sum()
    fixed_multi_effective_weight_sum = selection_weights[
        pointer_active & non_context34_fixed_multi_action
    ].sum()
    fixed_multi_effective_weight_share = (
        fixed_multi_effective_weight_sum
        / selection_effective_weight_sum.clamp_min(1.0)
    )

    if loss_mode in {"set", "hybrid_ordered"}:
        bce_raw = F.binary_cross_entropy_with_logits(
            logits.masked_fill(~mask, 0.0),
            targets,
            reduction="none",
        )
        bce_per_row = (
            (bce_raw * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        )
        bce_weights = weights if loss_mode == "set" else effective_weights
        set_bce_loss = weighted_mean(
            bce_per_row,
            bce_weights,
            torch.ones_like(pointer_active),
        )
    else:
        set_bce_loss = logits.sum() * 0.0

    if loss_mode == "ordered":
        allowed_counts = count_allowed_mask(batch)
        target_counts_allowed = allowed_counts.gather(
            1,
            action_counts.unsqueeze(1),
        ).squeeze(1)
        if not target_counts_allowed.all():
            raise ValueError(
                "BC replay action count is outside the observation bounds"
            )
        masked_count_logits = outputs["count_logits"].float().masked_fill(
            ~allowed_counts,
            -1e9,
        )
        count_per_row = F.cross_entropy(
            masked_count_logits,
            action_counts.clamp_max(MAX_ACTION_COUNT),
            reduction="none",
        )
        count_weights = effective_weights
    else:
        count_per_row = F.cross_entropy(
            outputs["count_logits"].float(),
            action_counts.clamp_max(MAX_ACTION_COUNT),
            reduction="none",
        )
        count_weights = weights if loss_mode == "set" else effective_weights
    count_loss = weighted_mean(count_per_row, count_weights, flexible)

    ordered_active = pointer_active & context_34
    context_34_ordered_loss = weighted_mean(
        ordered_per_row,
        weights,
        ordered_active,
    )
    non_context34_fixed_multi_action_ordered_loss = weighted_mean(
        ordered_per_row,
        weights,
        pointer_active & non_context34_fixed_multi_action,
    )
    context_7_fixed_multiaction = (
        (contexts == 7)
        & ~flexible
        & (action_counts > 1)
    )
    context_7_fixed_ordered_loss = weighted_mean(
        ordered_per_row,
        weights,
        pointer_active & context_7_fixed_multiaction,
    )
    auxiliary_bce = (
        0.25 * set_bce_loss
        if loss_mode in {"set", "hybrid_ordered"}
        else logits.sum() * 0.0
    )
    total = selection_loss + auxiliary_bce + count_loss
    return total, {
        "pointer_loss": selection_loss,
        "selection_loss": selection_loss,
        "ordered_selection_loss": weighted_mean(
            ordered_per_row,
            weights,
            pointer_active,
        ),
        "context_34_ordered_loss": context_34_ordered_loss,
        "non_context34_fixed_multi_action_ordered_loss": (
            non_context34_fixed_multi_action_ordered_loss
        ),
        "context_7_fixed_ordered_loss": context_7_fixed_ordered_loss,
        "selection_base_weight_sum": selection_base_weight_sum,
        "selection_effective_weight_sum": selection_effective_weight_sum,
        "non_context34_fixed_multi_action_base_weight_sum": (
            fixed_multi_base_weight_sum
        ),
        "non_context34_fixed_multi_action_effective_weight_sum": (
            fixed_multi_effective_weight_sum
        ),
        "non_context34_fixed_multi_action_effective_weight_share": (
            fixed_multi_effective_weight_share
        ),
        "set_bce_loss": set_bce_loss,
        "count_loss": count_loss,
    }


def bc_replay_update(
    model: EntityOptionPolicy,
    replay_optimizer: torch.optim.Optimizer | None,
    replay_batches: list[dict[str, torch.Tensor]],
    config: PPOConfig,
    device: torch.device,
    actor_learning_rate: float,
) -> dict[str, Any] | None:
    if not replay_batches or config.bc_replay_steps <= 0:
        return None
    if replay_optimizer is None:
        raise RuntimeError("BC replay is enabled without a replay optimizer")
    model.eval()
    started = time.time()
    replay_learning_rate = actor_learning_rate * config.bc_replay_lr_scale
    replay_optimizer.param_groups[0]["lr"] = replay_learning_rate
    replay_parameters = [
        parameter
        for group in replay_optimizer.param_groups
        for parameter in group["params"]
    ]
    stats: Counter[str] = Counter()
    selected_batch_indices: list[int] = []
    fixed_multi_hit_steps = 0
    for _ in range(config.bc_replay_steps):
        cpu_batch = random.choice(replay_batches)
        selected_batch_indices.append(
            next(
                index
                for index, candidate in enumerate(replay_batches)
                if candidate is cpu_batch
            )
        )
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in cpu_batch.items()
        }
        # The PPO optimizer also owns the value head, so clear the whole model
        # before an actor-only replay step to avoid clipping stale value grads.
        model.zero_grad(set_to_none=True)
        outputs = model_forward(model, batch, device)
        loss, parts = bc_expert_actor_loss(
            outputs,
            batch,
            loss_mode=config.bc_replay_loss,
            order_context_weight=config.bc_replay_order_context_weight,
            non_context34_fixed_multi_action_order_weight=(
                config.bc_replay_non_context34_fixed_multi_action_order_weight
            ),
        )
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            replay_parameters,
            config.max_grad_norm,
        )
        replay_optimizer.step()
        rows = int(batch["action_counts"].shape[0])
        stats["rows"] += rows
        stats["loss"] += float(loss.detach()) * rows
        for key, value in parts.items():
            stats[key] += float(value.detach()) * rows
        stats["grad_norm"] += float(grad_norm.detach()) * rows
        stats["context_34_rows"] += int(
            (batch["contexts"] == SKILL_ORDER_CONTEXT).sum()
        )
        stats["non_context34_fixed_multi_action_rows"] += int(
            (
                (batch["contexts"] != SKILL_ORDER_CONTEXT)
                & (batch["min_counts"] == batch["max_counts"])
                & (batch["action_counts"] > 1)
            ).sum()
        )
        fixed_multi_mask = (
            (batch["contexts"] != SKILL_ORDER_CONTEXT)
            & (batch["min_counts"] == batch["max_counts"])
            & (batch["action_counts"] > 1)
        )
        fixed_multi_rows = int(fixed_multi_mask.sum())
        fixed_multi_hit_steps += int(fixed_multi_rows > 0)
        stats["non_context34_fixed_multi_action_units"] += int(
            batch["action_counts"][fixed_multi_mask].sum()
        )
        stats["context_7_fixed_multiaction_rows"] += int(
            (
                (batch["contexts"] == 7)
                & (batch["min_counts"] == batch["max_counts"])
                & (batch["action_counts"] > 1)
            ).sum()
        )
    rows = int(stats["rows"])
    count_keys = {
        "rows",
        "context_34_rows",
        "non_context34_fixed_multi_action_rows",
        "non_context34_fixed_multi_action_units",
        "context_7_fixed_multiaction_rows",
    }
    return {
        key: value / max(rows, 1)
        for key, value in stats.items()
        if key not in count_keys
    } | {
        "rows": rows,
        "steps": config.bc_replay_steps,
        "lr_scale": config.bc_replay_lr_scale,
        "learning_rate": replay_learning_rate,
        "loss_mode": config.bc_replay_loss,
        "order_context_weight": config.bc_replay_order_context_weight,
        "non_context34_fixed_multi_action_order_weight": (
            config.bc_replay_non_context34_fixed_multi_action_order_weight
        ),
        "context34_rows_per_batch": (
            config.bc_replay_context34_rows_per_batch
        ),
        "context_34_rows": int(stats["context_34_rows"]),
        "non_context34_fixed_multi_action_rows": int(
            stats["non_context34_fixed_multi_action_rows"]
        ),
        "non_context34_fixed_multi_action_units": int(
            stats["non_context34_fixed_multi_action_units"]
        ),
        "non_context34_fixed_multi_action_hit_steps": fixed_multi_hit_steps,
        "selected_batch_indices": selected_batch_indices,
        "count_uses_new_order_weight": False,
        "context_7_fixed_multiaction_rows": int(
            stats["context_7_fixed_multiaction_rows"]
        ),
        "seconds": time.time() - started,
    }


def sample_opponent_index(
    opponent_count: int,
    bc_opponent_probability: float | None,
    sampling_weights: list[float] | None = None,
) -> int:
    if opponent_count < 1:
        raise ValueError("At least one frozen opponent is required")
    if sampling_weights is not None:
        if len(sampling_weights) != opponent_count:
            raise ValueError(
                "Opponent sampling weight count does not match opponent count"
            )
        if any(weight < 0.0 for weight in sampling_weights):
            raise ValueError("Opponent sampling weights must be non-negative")
        if sum(sampling_weights) <= 0.0:
            raise ValueError("At least one opponent sampling weight must be positive")
        return random.choices(
            range(opponent_count),
            weights=sampling_weights,
            k=1,
        )[0]
    if opponent_count == 1:
        return 0
    if (
        bc_opponent_probability is not None
        and random.random() < bc_opponent_probability
    ):
        return 0
    # Index 0 is the permanent BC opponent. When an explicit BC probability is
    # configured, sample the remaining mass uniformly over extra opponents and
    # historical snapshots.
    if bc_opponent_probability is not None:
        return random.randrange(1, opponent_count)
    return random.randrange(opponent_count)


def resolve_opponent_sampling_weights(
    opponents: list[FrozenOpponent],
    configured_weights: dict[str, float],
    history_weight: float,
) -> list[float] | None:
    """Resolve named permanent weights and a total historical-snapshot weight."""
    if not configured_weights and history_weight <= 0.0:
        return None
    opponent_names = {opponent.name for opponent in opponents}
    unknown = sorted(set(configured_weights) - opponent_names)
    if unknown:
        raise ValueError(
            "Configured opponent weights do not match the loaded pool: "
            + ", ".join(unknown)
        )
    history_indices = [
        index
        for index, opponent in enumerate(opponents)
        if not opponent.permanent
    ]
    per_history_weight = (
        history_weight / len(history_indices)
        if history_indices
        else 0.0
    )
    weights = [
        (
            configured_weights.get(opponent.name, 0.0)
            if opponent.permanent
            else per_history_weight
        )
        for opponent in opponents
    ]
    if sum(weights) <= 0.0:
        raise ValueError(
            "Explicit opponent weighting assigned zero mass to the loaded pool"
        )
    return weights


def build_opponent_quota_controller(
    config: PPOConfig,
    opponents: list[FrozenOpponent],
    resume_state: dict[str, Any] | None = None,
    reset_resume_state: bool = False,
) -> OpponentQuotaController | None:
    if config.opponent_quota_mode == "legacy":
        return None
    permanent_names = [
        opponent.name
        for opponent in opponents
        if opponent.permanent
    ]
    permanent_name_set = set(permanent_names)
    configured_name_sets = {
        "base quota": set(config.opponent_base_quotas),
        "cap": set(config.opponent_caps),
        "audit": set(config.opponent_audit),
    }
    for label, configured_names in configured_name_sets.items():
        unknown = sorted(configured_names - permanent_name_set)
        if unknown:
            raise ValueError(
                f"Opponent {label} names do not match the loaded permanent "
                f"pool: {', '.join(unknown)}"
            )
    if config.opponent_quota_mode == "adaptive":
        missing_caps = [
            name for name in permanent_names
            if name not in config.opponent_caps
        ]
        missing_audits = [
            name for name in permanent_names
            if name not in config.opponent_audit
        ]
        if missing_caps:
            raise ValueError(
                "Adaptive opponent caps must cover every permanent opponent: "
                + ", ".join(missing_caps)
            )
        if missing_audits:
            raise ValueError(
                "Adaptive opponent audits must cover every permanent opponent: "
                + ", ".join(missing_audits)
            )
    controller = OpponentQuotaController(
        mode=config.opponent_quota_mode,
        opponent_names=permanent_names,
        base_quotas=config.opponent_base_quotas,
        caps=config.opponent_caps,
        initial_audit=config.opponent_audit,
        games_per_update=config.games_per_update,
        refresh_updates=config.opponent_quota_refresh_updates,
        seed=config.seed,
    )
    if resume_state is not None:
        if reset_resume_state:
            validate_saved_opponent_quota_state(resume_state)
            controller.resume_state_reset = True
        else:
            controller.load_state_dict(resume_state)
    return controller


def validate_saved_opponent_quota_state(state: dict[str, Any]) -> None:
    """Validate an exact-quota state before deliberately discarding it."""
    opponent_names = state.get("opponent_names")
    if not isinstance(opponent_names, list):
        raise ValueError("Resume checkpoint opponent quota state lacks names")
    controller = OpponentQuotaController(
        mode=str(state.get("mode")),
        opponent_names=[str(name) for name in opponent_names],
        base_quotas=dict(state.get("base_quotas") or {}),
        caps=dict(state.get("caps") or {}),
        initial_audit=dict(state.get("initial_audit") or {}),
        games_per_update=int(state.get("games_per_update", 0)),
        refresh_updates=int(state.get("refresh_updates", 0)),
        seed=int(state.get("seed", 0)),
    )
    controller.load_state_dict(state)


def validate_opponent_quota_resume_transition(
    *,
    current_mode: str,
    checkpoint_mode: str,
    checkpoint_state: dict[str, Any] | None,
    reset_optimizer: bool,
    reset_quota_state: bool = False,
) -> None:
    if checkpoint_mode in {"fixed", "adaptive"} and checkpoint_state is None:
        raise ValueError(
            "Exact-quota resume checkpoint lacks opponent quota state"
        )
    if current_mode == "legacy" and checkpoint_state is not None:
        raise ValueError(
            "Cannot resume an exact-quota checkpoint in legacy quota mode"
        )
    if reset_quota_state and not reset_optimizer:
        raise ValueError(
            "--reset-opponent-quota-on-resume requires "
            "--reset-optimizer-on-resume"
        )
    if (
        current_mode != "legacy"
        and checkpoint_mode == "legacy"
        and not reset_optimizer
    ):
        raise ValueError(
            "Starting exact opponent quotas from a legacy checkpoint requires "
            "--reset-optimizer-on-resume"
        )
    if (
        current_mode in {"fixed", "adaptive"}
        and checkpoint_mode in {"fixed", "adaptive"}
        and current_mode != checkpoint_mode
        and not reset_quota_state
    ):
        raise ValueError(
            "Changing exact opponent quota mode on resume requires "
            "--reset-opponent-quota-on-resume"
        )


def make_running_game(
    deck: list[int],
    deck_hash: str,
    uid: int,
    league_probability: float,
    opponents: list[FrozenOpponent],
    league_opponent_index: int | None = None,
    bc_opponent_probability: float | None = None,
    opponent_sampling_weights: list[float] | None = None,
    league_current_seat: int | None = None,
) -> RunningGame:
    if opponents and random.random() < league_probability:
        if league_current_seat not in {None, 0, 1}:
            raise ValueError("League learner seat must be 0, 1, or None")
        current_seat = (
            random.randrange(2)
            if league_current_seat is None
            else league_current_seat
        )
        opponent_index = (
            league_opponent_index
            if league_opponent_index is not None
            else sample_opponent_index(
                len(opponents),
                bc_opponent_probability,
                opponent_sampling_weights,
            )
        )
        opponent = opponents[opponent_index]
        seat_policy = {
            current_seat: -1,
            1 - current_seat: opponent_index,
        }
        seat_decks = {
            current_seat: deck,
            1 - current_seat: opponent.deck,
        }
        seat_deck_hash = {
            current_seat: deck_hash,
            1 - current_seat: opponent.deck_hash,
        }
        trainable_seats = {current_seat}
        opponent_name = opponent.name
    else:
        if league_current_seat is not None:
            raise ValueError(
                "League learner seat override requires a frozen opponent game"
            )
        seat_policy = {0: -1, 1: -1}
        seat_decks = {0: deck, 1: deck}
        seat_deck_hash = {0: deck_hash, 1: deck_hash}
        trainable_seats = {0, 1}
        opponent_name = None
    return RunningGame(
        battle=RawBattle(seat_decks[0], seat_decks[1]),
        uid=uid,
        seat_policy=seat_policy,
        seat_deck_hash=seat_deck_hash,
        trainable_seats=trainable_seats,
        transition_indices={0: [], 1: []},
        opponent_name=opponent_name,
    )


def needs_rollout_replacement(
    valid_completed: int,
    active_games: int,
    games_target: int,
) -> bool:
    """Whether one more game can start without overshooting the valid target."""
    return valid_completed + active_games < games_target


def running_game_opponent_index(game: RunningGame) -> int:
    opponent_indices = [
        policy_index
        for policy_index in game.seat_policy.values()
        if policy_index >= 0
    ]
    if len(opponent_indices) != 1:
        raise RuntimeError(
            f"Expected one frozen opponent, got {opponent_indices}"
        )
    return opponent_indices[0]


def finalize_episode(
    game: RunningGame,
    transitions: list[dict[str, Any]],
    result: int,
    gamma: float,
    gae_lambda: float,
    valid: bool,
) -> None:
    for seat, indices in game.transition_indices.items():
        if not valid:
            for index in indices:
                transitions[index]["keep"] = False
            continue
        gae = 0.0
        next_value = 0.0
        outcome = 1.0 if result == seat else 0.0
        for position, index in enumerate(reversed(indices)):
            transition = transitions[index]
            terminal_step = position == 0
            reward = 1.0 if terminal_step and result == seat else 0.0
            value = float(transition["old_value"])
            delta = reward + gamma * next_value - value
            gae = delta + gamma * gae_lambda * gae
            transition["advantage"] = gae
            transition["return"] = gae + value
            transition["terminal_reward"] = reward if terminal_step else 0.0
            transition["outcome_target"] = outcome
            next_value = value


def collect_rollout(
    current_model: EntityOptionPolicy,
    opponents: list[FrozenOpponent],
    deck: list[int],
    model_config: dict[str, Any],
    config: PPOConfig,
    device: torch.device,
    update: int,
    opponent_quota_controller: OpponentQuotaController | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_model.eval()
    for opponent in opponents:
        opponent.model.eval()
    learner_deck_hash = compute_deck_hash(deck)
    transitions: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    league_by_opponent: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.time()
    uid_counter = update * 1_000_000
    opponent_sampling_weights = resolve_opponent_sampling_weights(
        opponents,
        config.opponent_weights,
        config.history_opponent_weight,
    )
    quota_audit: dict[str, Any] | None = None
    exact_quota_schedule: ExactQuotaSchedule | None = None
    quota_invalid_replacements: Counter[str] = Counter()
    quota_invalid_replacements_by_seat: dict[str, Counter[str]] = defaultdict(
        Counter
    )
    quota_start_errors_by_seat: dict[str, Counter[str]] = defaultdict(Counter)
    planned_seat_quotas: dict[str, dict[str, int]] | None = None
    if opponent_quota_controller is not None:
        schedule_names, quota_audit = (
            opponent_quota_controller.begin_update(update)
        )
        opponent_index_by_name = {
            opponent.name: index
            for index, opponent in enumerate(opponents)
        }
        try:
            schedule_indices = [
                opponent_index_by_name[name]
                for name in schedule_names
            ]
        except KeyError as error:
            raise RuntimeError(
                f"Exact-quota opponent left the loaded pool: {error.args[0]}"
            ) from error
        if config.opponent_quota_seat_balance:
            schedule_seats, planned_seat_quotas = (
                balanced_quota_learner_seats(
                    schedule_names,
                    config.seed,
                    update,
                )
            )
        else:
            schedule_seats = [None] * len(schedule_indices)
        exact_quota_schedule = ExactQuotaSchedule(
            schedule_indices,
            schedule_seats,
        )
    rollout_opponent_index = (
        sample_opponent_index(
            len(opponents),
            config.bc_opponent_probability,
            opponent_sampling_weights,
        )
        if (
            exact_quota_schedule is None
            and opponents
            and config.opponent_sampling == "per_update"
        )
        else None
    )
    games: list[RunningGame] = []
    initial_game_count = min(config.environments, config.games_per_update)
    while len(games) < initial_game_count:
        quota_claim = (
            exact_quota_schedule.peek_claim()
            if exact_quota_schedule is not None
            else None
        )
        if exact_quota_schedule is not None and quota_claim is None:
            break
        requested_opponent_index = (
            quota_claim[0].opponent_index
            if quota_claim is not None
            else rollout_opponent_index
        )
        requested_learner_seat = (
            quota_claim[0].learner_seat
            if quota_claim is not None
            else None
        )
        try:
            games.append(
                make_running_game(
                    deck,
                    learner_deck_hash,
                    uid_counter,
                    config.league_probability,
                    opponents,
                    requested_opponent_index,
                    config.bc_opponent_probability,
                    opponent_sampling_weights,
                    requested_learner_seat,
                )
            )
            if quota_claim is not None:
                exact_quota_schedule.mark_started(quota_claim[1])
            uid_counter += 1
        except ValueError:
            stats["start_errors"] += 1
            if (
                quota_claim is not None
                and requested_learner_seat is not None
            ):
                quota_start_errors_by_seat[
                    opponents[requested_opponent_index].name
                ][str(requested_learner_seat)] += 1
            if stats["start_errors"] > config.games_per_update * 5:
                for active_game in games:
                    active_game.battle.close()
                raise RuntimeError(
                    "Repeated BattleStart failures prevented rollout start"
                )
    stats["max_active_games"] = len(games)

    valid_completed = 0
    attempts = 0
    while games:
        groups: dict[int, list[tuple[RunningGame, dict[str, Any]]]] = defaultdict(list)
        invalid_games: list[RunningGame] = []
        for game in games:
            observation = game.battle.observation
            if game.battle.result != -1:
                continue
            select = observation.get("select")
            current = observation.get("current") or {}
            if not isinstance(select, dict):
                invalid_games.append(game)
                continue
            options = select.get("option")
            minimum = int(select.get("minCount", 0) or 0)
            maximum = int(select.get("maxCount", 0) or 0)
            if (
                not isinstance(options, list)
                or not options
                or minimum < 0
                or maximum < minimum
                or maximum > len(options)
                or maximum > MAX_ACTION_COUNT
            ):
                invalid_games.append(game)
                continue
            seat = int(current.get("yourIndex", 0) or 0)
            feature = live_feature(
                observation,
                model_config,
                game.seat_deck_hash[seat],
            )
            if feature is None:
                invalid_games.append(game)
                continue
            policy_index = game.seat_policy[seat]
            groups[policy_index].append((game, feature))

        for game in invalid_games:
            stats["invalid_observation_games"] += 1
            if exact_quota_schedule is not None:
                opponent_index = running_game_opponent_index(game)
                learner_seat = next(iter(game.trainable_seats))
                exact_quota_schedule.add_replacement(
                    opponent_index,
                    (
                        learner_seat
                        if config.opponent_quota_seat_balance
                        else None
                    ),
                )
                quota_invalid_replacements[
                    opponents[opponent_index].name
                ] += 1
                if config.opponent_quota_seat_balance:
                    quota_invalid_replacements_by_seat[
                        opponents[opponent_index].name
                    ][str(learner_seat)] += 1
            finalize_episode(
                game,
                transitions,
                result=2,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                valid=False,
            )
            game.battle.close()
            games.remove(game)
            attempts += 1
            if (
                exact_quota_schedule is None
                and needs_rollout_replacement(
                valid_completed,
                len(games),
                config.games_per_update,
                )
            ):
                try:
                    games.append(
                        make_running_game(
                            deck,
                            learner_deck_hash,
                            uid_counter,
                            config.league_probability,
                            opponents,
                            rollout_opponent_index,
                            config.bc_opponent_probability,
                            opponent_sampling_weights,
                            None,
                        )
                    )
                    uid_counter += 1
                except ValueError:
                    stats["start_errors"] += 1

        actions_by_uid: dict[int, tuple[list[int], dict[str, Any]]] = {}
        for policy_index, items in groups.items():
            features = [feature for _, feature in items]
            batch = collate_features(features, model_config, device)
            model = (
                current_model
                if policy_index == -1
                else opponents[policy_index].model
            )
            with torch.no_grad():
                outputs = model_forward(model, batch, device)
                actions, log_probs, entropies, values = sample_ordered_actions(
                    outputs,
                    batch,
                    deterministic=policy_index != -1,
                    canonicalize_order=(
                        policy_index != -1
                        and opponents[policy_index].canonical_order
                    ),
                    temperature=(
                        config.policy_temperature
                        if policy_index == -1
                        else 1.0
                    ),
                )
            for item_index, (game, feature) in enumerate(items):
                action = actions[item_index]
                current = game.battle.observation.get("current") or {}
                seat = int(current.get("yourIndex", 0) or 0)
                if policy_index == -1 and seat in game.trainable_seats:
                    transition_index = len(transitions)
                    transitions.append(
                        {
                            "feature": feature,
                            "action": action,
                            "action_count": len(action),
                            "old_log_prob": float(log_probs[item_index]),
                            "old_value": float(values[item_index]),
                            "old_entropy": float(entropies[item_index]),
                            "game_uid": game.uid,
                            "seat": seat,
                            "opponent_name": game.opponent_name,
                            "keep": True,
                        }
                    )
                    game.transition_indices[seat].append(transition_index)
                actions_by_uid[game.uid] = (action, feature)

        finished: list[RunningGame] = []
        for game in list(games):
            if game in invalid_games or game.uid not in actions_by_uid:
                continue
            action, _ = actions_by_uid[game.uid]
            _, error = game.battle.step(action)
            game.decisions += 1
            stats["engine_decisions"] += 1
            if error:
                stats[f"select_error_{error}"] += 1
                if exact_quota_schedule is not None:
                    opponent_index = running_game_opponent_index(game)
                    learner_seat = next(iter(game.trainable_seats))
                    exact_quota_schedule.add_replacement(
                        opponent_index,
                        (
                            learner_seat
                            if config.opponent_quota_seat_balance
                            else None
                        ),
                    )
                    quota_invalid_replacements[
                        opponents[opponent_index].name
                    ] += 1
                    if config.opponent_quota_seat_balance:
                        quota_invalid_replacements_by_seat[
                            opponents[opponent_index].name
                        ][str(learner_seat)] += 1
                finalize_episode(
                    game,
                    transitions,
                    result=2,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    valid=False,
                )
                finished.append(game)
                continue
            result = game.battle.result
            if result != -1:
                valid_completed += 1
                stats["valid_games"] += 1
                stats[f"result_{result}"] += 1
                if len(game.trainable_seats) == 1:
                    current_seat = next(iter(game.trainable_seats))
                    opponent_index = running_game_opponent_index(game)
                    opponent_name = opponents[opponent_index].name
                    stats["league_games"] += 1
                    league_by_opponent[opponent_name]["games"] += 1
                    seat_prefix = f"seat_{current_seat}_"
                    league_by_opponent[opponent_name][
                        seat_prefix + "games"
                    ] += 1
                    if result == current_seat:
                        stats["league_current_wins"] += 1
                        league_by_opponent[opponent_name]["wins"] += 1
                        league_by_opponent[opponent_name][
                            seat_prefix + "wins"
                        ] += 1
                    elif result == 2:
                        stats["league_draws"] += 1
                        league_by_opponent[opponent_name]["draws"] += 1
                        league_by_opponent[opponent_name][
                            seat_prefix + "draws"
                        ] += 1
                    else:
                        stats["league_current_losses"] += 1
                        league_by_opponent[opponent_name]["losses"] += 1
                        league_by_opponent[opponent_name][
                            seat_prefix + "losses"
                        ] += 1
                else:
                    stats["selfplay_games"] += 1
                finalize_episode(
                    game,
                    transitions,
                    result=result,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    valid=True,
                )
                finished.append(game)
                continue
            if game.decisions >= config.max_game_decisions:
                stats["truncated_games"] += 1
                if exact_quota_schedule is not None:
                    opponent_index = running_game_opponent_index(game)
                    learner_seat = next(iter(game.trainable_seats))
                    exact_quota_schedule.add_replacement(
                        opponent_index,
                        (
                            learner_seat
                            if config.opponent_quota_seat_balance
                            else None
                        ),
                    )
                    quota_invalid_replacements[
                        opponents[opponent_index].name
                    ] += 1
                    if config.opponent_quota_seat_balance:
                        quota_invalid_replacements_by_seat[
                            opponents[opponent_index].name
                        ][str(learner_seat)] += 1
                finalize_episode(
                    game,
                    transitions,
                    result=2,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    valid=False,
                )
                finished.append(game)

        for game in finished:
            game.battle.close()
            if game in games:
                games.remove(game)
            attempts += 1
            if (
                exact_quota_schedule is None
                and needs_rollout_replacement(
                valid_completed,
                len(games),
                config.games_per_update,
                )
            ):
                try:
                    games.append(
                        make_running_game(
                            deck,
                            learner_deck_hash,
                            uid_counter,
                            config.league_probability,
                            opponents,
                            rollout_opponent_index,
                            config.bc_opponent_probability,
                            opponent_sampling_weights,
                            None,
                        )
                    )
                    uid_counter += 1
                except ValueError:
                    stats["start_errors"] += 1

        while len(games) < config.environments:
            if exact_quota_schedule is not None:
                quota_claim = exact_quota_schedule.peek_claim()
                if quota_claim is None:
                    break
                requested_opponent_index = quota_claim[0].opponent_index
                requested_learner_seat = quota_claim[0].learner_seat
            else:
                if not needs_rollout_replacement(
                    valid_completed,
                    len(games),
                    config.games_per_update,
                ):
                    break
                quota_claim = None
                requested_opponent_index = rollout_opponent_index
                requested_learner_seat = None
            try:
                games.append(
                    make_running_game(
                        deck,
                        learner_deck_hash,
                        uid_counter,
                        config.league_probability,
                        opponents,
                        requested_opponent_index,
                        config.bc_opponent_probability,
                        opponent_sampling_weights,
                        requested_learner_seat,
                    )
                )
                if quota_claim is not None:
                    exact_quota_schedule.mark_started(quota_claim[1])
                uid_counter += 1
                stats["max_active_games"] = max(
                    stats["max_active_games"],
                    len(games),
                )
            except ValueError:
                stats["start_errors"] += 1
                if (
                    quota_claim is not None
                    and requested_learner_seat is not None
                ):
                    quota_start_errors_by_seat[
                        opponents[requested_opponent_index].name
                    ][str(requested_learner_seat)] += 1
                if stats["start_errors"] > config.games_per_update * 5:
                    for active_game in games:
                        active_game.battle.close()
                    raise RuntimeError(
                        "Repeated BattleStart failures prevented rollout refill"
                    )

        failed_attempts = attempts - valid_completed
        if (
            failed_attempts + stats["start_errors"]
            > config.games_per_update * 5
        ):
            for game in games:
                game.battle.close()
            raise RuntimeError(
                "Rollout exceeded the failed-game budget: "
                f"valid={valid_completed} failed={failed_attempts} "
                f"start_errors={stats['start_errors']}"
            )

    if valid_completed != config.games_per_update:
        raise RuntimeError(
            f"Rollout completed {valid_completed} valid games; expected "
            f"{config.games_per_update}"
        )
    kept = [transition for transition in transitions if transition.get("keep")]
    if not kept:
        raise RuntimeError("Rollout produced no valid trainable transitions")
    stats["transitions_total"] = len(transitions)
    stats["transitions_kept"] = len(kept)
    stats["seconds"] = time.time() - started
    stats["decisions_per_second"] = stats["engine_decisions"] / max(
        stats["seconds"],
        1e-6,
    )
    stats["mean_episode_decisions"] = stats["engine_decisions"] / max(
        stats["valid_games"],
        1,
    )
    stats["mean_old_value"] = sum(float(t["old_value"]) for t in kept) / len(kept)
    stats["mean_terminal_return"] = sum(float(t["return"]) for t in kept) / len(kept)
    result_stats = dict(stats)
    result_stats["league_by_opponent"] = {
        name: dict(opponent_stats)
        for name, opponent_stats in sorted(league_by_opponent.items())
    }
    result_stats["league_by_opponent_seat"] = {
        name: {
            seat: {
                key: int(opponent_stats.get(f"seat_{seat}_{key}", 0))
                for key in ("games", "wins", "losses", "draws")
            }
            for seat in ("0", "1")
        }
        for name, opponent_stats in sorted(league_by_opponent.items())
    }
    transitions_by_opponent_seat: dict[str, Counter[str]] = defaultdict(Counter)
    for transition in kept:
        opponent_name = transition.get("opponent_name")
        group_name = (
            SELFPLAY_OPPONENT_GROUP
            if opponent_name is None
            else str(opponent_name)
        )
        transitions_by_opponent_seat[group_name][
            str(int(transition["seat"]))
        ] += 1
    result_stats["transitions_by_opponent_seat"] = {
        name: {
            seat: int(counts.get(seat, 0))
            for seat in ("0", "1")
        }
        for name, counts in sorted(transitions_by_opponent_seat.items())
    }
    if opponent_quota_controller is not None:
        if exact_quota_schedule is None or quota_audit is None:
            raise RuntimeError("Exact-quota rollout state was not initialized")
        if exact_quota_schedule.remaining:
            raise RuntimeError(
                "Exact-quota rollout ended with "
                f"{exact_quota_schedule.remaining} unstarted games"
            )
        start_errors_by_seat: dict[str, dict[str, int]] | None = None
        if planned_seat_quotas is not None:
            start_errors_by_seat = {
                name: {
                    seat: int(
                        quota_start_errors_by_seat.get(name, {}).get(seat, 0)
                    )
                    for seat in ("0", "1")
                }
                for name in quota_audit["planned_quotas"]
            }
            if (
                sum(
                    sum(row.values())
                    for row in start_errors_by_seat.values()
                )
                != int(stats["start_errors"])
            ):
                raise RuntimeError(
                    "Exact-quota learner-seat start errors do not match "
                    "the rollout aggregate"
                )
        quota_audit = opponent_quota_controller.finish_update(
            update,
            quota_audit,
            result_stats["league_by_opponent"],
            dict(quota_invalid_replacements),
            planned_seat_quotas=planned_seat_quotas,
            invalid_replacements_by_seat={
                name: dict(counts)
                for name, counts in quota_invalid_replacements_by_seat.items()
            },
        )
        if start_errors_by_seat is not None:
            quota_audit["start_errors_by_seat"] = start_errors_by_seat
        result_stats["opponent_quota"] = quota_audit
        log(
            "opponent_quota_update="
            + json.dumps(quota_audit, ensure_ascii=False)
        )
    if opponent_sampling_weights is not None:
        total_weight = sum(opponent_sampling_weights)
        result_stats["opponent_sampling_probabilities"] = {
            opponent.name: weight / total_weight
            for opponent, weight in zip(
                opponents,
                opponent_sampling_weights,
            )
            if weight > 0.0
        }
    return kept, result_stats


def phase_schedule_progress(config: PPOConfig, update: int) -> float:
    """Return clamped progress within the explicitly configured train phase."""
    if config.updates <= config.schedule_start_update:
        return 0.0
    return min(
        max(
            (update - config.schedule_start_update)
            / (config.updates - config.schedule_start_update),
            0.0,
        ),
        1.0,
    )


def bc_kl_coefficient(config: PPOConfig, update: int) -> float:
    if config.bc_kl_start == 0.0 and config.bc_kl_end == 0.0:
        return 0.0
    if config.updates <= config.schedule_start_update:
        return config.bc_kl_start
    progress = phase_schedule_progress(config, update)
    return config.bc_kl_start * (
        config.bc_kl_end / config.bc_kl_start
    ) ** progress


def gradient_cosine(
    first_objective: torch.Tensor,
    second_objective: torch.Tensor,
    parameters: list[nn.Parameter],
) -> dict[str, float]:
    first_gradients = torch.autograd.grad(
        first_objective,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    second_gradients = torch.autograd.grad(
        second_objective,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    dot = first_objective.detach().new_zeros((), dtype=torch.float32)
    first_squared = dot.clone()
    second_squared = dot.clone()
    for first, second in zip(first_gradients, second_gradients):
        if first is None or second is None:
            continue
        first_float = first.detach().float()
        second_float = second.detach().float()
        dot = dot + (first_float * second_float).sum()
        first_squared = first_squared + first_float.square().sum()
        second_squared = second_squared + second_float.square().sum()
    first_norm = first_squared.sqrt()
    second_norm = second_squared.sqrt()
    denominator = first_norm * second_norm
    cosine = (
        dot / denominator
        if float(denominator) > 0.0
        else dot.new_tensor(0.0)
    )
    return {
        "cosine": float(cosine),
        "primary_norm": float(first_norm),
        "guard_norm": float(second_norm),
    }


def symmetric_pcgrad_adjustment(
    primary_loss: torch.Tensor,
    guard_loss: torch.Tensor,
    parameters: list[nn.Parameter],
) -> tuple[list[torch.Tensor | None], dict[str, float | bool]]:
    """Return the symmetric PCGrad delta for two conflicting losses."""
    primary_gradients = torch.autograd.grad(
        primary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    guard_gradients = torch.autograd.grad(
        guard_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    origin = primary_loss.detach().new_zeros((), dtype=torch.float32)
    dot = origin.clone()
    primary_squared = origin.clone()
    guard_squared = origin.clone()
    for primary, guard in zip(primary_gradients, guard_gradients):
        if primary is not None:
            primary_float = primary.detach().float()
            primary_squared = (
                primary_squared + primary_float.square().sum()
            )
        if guard is not None:
            guard_float = guard.detach().float()
            guard_squared = guard_squared + guard_float.square().sum()
        if primary is not None and guard is not None:
            dot = dot + (
                primary.detach().float() * guard.detach().float()
            ).sum()

    primary_norm = primary_squared.sqrt()
    guard_norm = guard_squared.sqrt()
    denominator = primary_norm * guard_norm
    cosine = (
        dot / denominator
        if float(denominator) > 0.0
        else origin.clone()
    )
    conflict = (
        float(dot) < 0.0
        and float(primary_squared) > 0.0
        and float(guard_squared) > 0.0
    )
    primary_projection = (
        float(-dot / guard_squared) if conflict else 0.0
    )
    guard_projection = (
        float(-dot / primary_squared) if conflict else 0.0
    )

    adjustments: list[torch.Tensor | None] = []
    projected_primary_dot = origin.clone()
    projected_guard_dot = origin.clone()
    for primary, guard in zip(primary_gradients, guard_gradients):
        if primary is None and guard is None:
            adjustments.append(None)
            continue
        template = primary if primary is not None else guard
        assert template is not None
        primary_float = (
            primary.detach().float()
            if primary is not None
            else torch.zeros_like(template, dtype=torch.float32)
        )
        guard_float = (
            guard.detach().float()
            if guard is not None
            else torch.zeros_like(template, dtype=torch.float32)
        )
        projected_primary = (
            primary_float + primary_projection * guard_float
        )
        projected_guard = (
            guard_float + guard_projection * primary_float
        )
        combined = projected_primary + projected_guard
        projected_primary_dot = (
            projected_primary_dot + (combined * primary_float).sum()
        )
        projected_guard_dot = (
            projected_guard_dot + (combined * guard_float).sum()
        )
        adjustment = (
            primary_projection * guard_float
            + guard_projection * primary_float
        )
        adjustments.append(adjustment.to(dtype=template.dtype))

    return adjustments, {
        "conflict": conflict,
        "pre_dot": float(dot),
        "pre_cosine": float(cosine),
        "primary_norm": float(primary_norm),
        "guard_norm": float(guard_norm),
        "primary_projection_coefficient": primary_projection,
        "guard_projection_coefficient": guard_projection,
        "projected_core_dot_primary": float(projected_primary_dot),
        "projected_core_dot_guard": float(projected_guard_dot),
    }


def guard_priority_pcgrad_adjustment(
    primary_loss: torch.Tensor,
    guard_loss: torch.Tensor,
    parameters: list[nn.Parameter],
) -> tuple[list[torch.Tensor | None], dict[str, float | bool]]:
    """Project only a conflicting primary gradient off the guard gradient."""
    primary_gradients = torch.autograd.grad(
        primary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    guard_gradients = torch.autograd.grad(
        guard_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    origin = primary_loss.detach().new_zeros((), dtype=torch.float32)
    dot = origin.clone()
    primary_squared = origin.clone()
    guard_squared = origin.clone()
    for primary, guard in zip(primary_gradients, guard_gradients):
        if primary is not None:
            primary_squared = (
                primary_squared + primary.detach().float().square().sum()
            )
        if guard is not None:
            guard_squared = (
                guard_squared + guard.detach().float().square().sum()
            )
        if primary is not None and guard is not None:
            dot = dot + (
                primary.detach().float() * guard.detach().float()
            ).sum()

    primary_norm = primary_squared.sqrt()
    guard_norm = guard_squared.sqrt()
    denominator = primary_norm * guard_norm
    cosine = (
        dot / denominator
        if float(denominator) > 0.0
        else origin.clone()
    )
    conflict = float(dot) < 0.0 and float(guard_squared) > 0.0
    primary_projection = (
        float(-dot / guard_squared) if conflict else 0.0
    )

    adjustments: list[torch.Tensor | None] = []
    projected_primary_dot = origin.clone()
    projected_guard_dot = origin.clone()
    for primary, guard in zip(primary_gradients, guard_gradients):
        if primary is None and guard is None:
            adjustments.append(None)
            continue
        template = primary if primary is not None else guard
        assert template is not None
        primary_float = (
            primary.detach().float()
            if primary is not None
            else torch.zeros_like(template, dtype=torch.float32)
        )
        guard_float = (
            guard.detach().float()
            if guard is not None
            else torch.zeros_like(template, dtype=torch.float32)
        )
        projected_primary = (
            primary_float + primary_projection * guard_float
        )
        combined = projected_primary + guard_float
        projected_primary_dot = (
            projected_primary_dot + (combined * primary_float).sum()
        )
        projected_guard_dot = (
            projected_guard_dot + (combined * guard_float).sum()
        )
        adjustment = primary_projection * guard_float
        adjustments.append(adjustment.to(dtype=template.dtype))

    return adjustments, {
        "conflict": conflict,
        "pre_dot": float(dot),
        "pre_cosine": float(cosine),
        "primary_norm": float(primary_norm),
        "guard_norm": float(guard_norm),
        "primary_projection_coefficient": primary_projection,
        "guard_projection_coefficient": 0.0,
        "projected_core_dot_primary": float(projected_primary_dot),
        "projected_core_dot_guard": float(projected_guard_dot),
    }


def actor_priority_value_pcgrad_adjustment(
    actor_side_loss: torch.Tensor,
    value_side_loss: torch.Tensor,
    actor_parameters: list[nn.Parameter],
) -> tuple[list[torch.Tensor | None], dict[str, float | int | bool]]:
    """Project a conflicting critic gradient off the actor gradient.

    Only parameters receiving both gradients participate.  Actor-only heads
    are therefore untouched, while the value head is kept out of surgery and
    receives its complete scalar value-loss gradient.
    """
    actor_gradients = torch.autograd.grad(
        actor_side_loss,
        actor_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    value_gradients = torch.autograd.grad(
        value_side_loss,
        actor_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    origin = actor_side_loss.detach().new_zeros((), dtype=torch.float32)
    dot = origin.clone()
    actor_squared = origin.clone()
    value_squared = origin.clone()
    shared_tensor_count = 0
    shared_parameter_count = 0
    actor_only_tensor_count = 0
    actor_only_parameter_count = 0
    for parameter, actor_gradient, value_gradient in zip(
        actor_parameters,
        actor_gradients,
        value_gradients,
    ):
        if actor_gradient is not None and value_gradient is not None:
            actor_float = actor_gradient.detach().float()
            value_float = value_gradient.detach().float()
            dot = dot + (actor_float * value_float).sum()
            actor_squared = actor_squared + actor_float.square().sum()
            value_squared = value_squared + value_float.square().sum()
            shared_tensor_count += 1
            shared_parameter_count += parameter.numel()
        elif actor_gradient is not None:
            actor_only_tensor_count += 1
            actor_only_parameter_count += parameter.numel()

    actor_norm = actor_squared.sqrt()
    value_norm = value_squared.sqrt()
    norm_product = actor_norm * value_norm
    cosine = (
        dot / norm_product
        if float(norm_product) > 0.0
        else origin.clone()
    )
    conflict = float(dot) < 0.0 and float(actor_squared) > 0.0
    projection_coefficient = (
        float(-dot / actor_squared) if conflict else 0.0
    )

    adjustments: list[torch.Tensor | None] = []
    for actor_gradient, value_gradient in zip(
        actor_gradients,
        value_gradients,
    ):
        if actor_gradient is None or value_gradient is None:
            adjustments.append(None)
            continue
        adjustments.append(
            (projection_coefficient * actor_gradient.detach()).to(
                dtype=actor_gradient.dtype,
            )
        )

    projected_value_dot_actor = (
        dot + projection_coefficient * actor_squared
    )
    final_dot_actor_margin = projected_value_dot_actor
    adjustment_norm = abs(projection_coefficient) * float(actor_norm)
    finite_values = (
        dot,
        cosine,
        actor_norm,
        value_norm,
        projected_value_dot_actor,
        final_dot_actor_margin,
    )
    return adjustments, {
        "conflict": conflict,
        "shared_tensor_count": shared_tensor_count,
        "shared_parameter_count": shared_parameter_count,
        "actor_only_tensor_count": actor_only_tensor_count,
        "actor_only_parameter_count": actor_only_parameter_count,
        "pre_dot": float(dot),
        "pre_cosine": float(cosine),
        "actor_norm": float(actor_norm),
        "value_norm": float(value_norm),
        "projection_coefficient": projection_coefficient,
        "adjustment_norm": adjustment_norm,
        "projected_value_dot_actor": float(projected_value_dot_actor),
        "final_dot_actor_margin": float(final_dot_actor_margin),
        "finite": all(math.isfinite(float(value)) for value in finite_values)
        and math.isfinite(projection_coefficient)
        and math.isfinite(adjustment_norm),
    }


@torch.no_grad()
def evaluate_constrained_policy_shift(
    model: EntityOptionPolicy,
    rollout_batch: dict[str, torch.Tensor],
    action_sequences: torch.Tensor,
    action_counts: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    opponent_groups: list[str],
    config: PPOConfig,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    """Re-evaluate the final actor against the frozen rollout after PPO."""
    model.eval()
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for start in range(0, len(opponent_groups), config.minibatch_size):
        indices = torch.arange(
            start,
            min(start + config.minibatch_size, len(opponent_groups)),
            dtype=torch.long,
        )
        index_list = indices.tolist()
        batch = {
            key: value.index_select(0, indices).to(
                device,
                non_blocking=True,
            )
            for key, value in rollout_batch.items()
        }
        mb_actions = action_sequences[indices].to(device)
        mb_counts = action_counts[indices].to(device)
        mb_old_log_probs = old_log_probs[indices].to(device)
        mb_advantages = advantages[indices].to(device)
        mb_groups = [opponent_groups[index] for index in index_list]
        outputs = model_forward(model, batch, device)
        new_log_probs, _ = ordered_action_log_prob_entropy(
            outputs,
            batch,
            mb_actions,
            mb_counts,
            temperature=config.policy_temperature,
        )
        log_ratio = new_log_probs - mb_old_log_probs
        ratio = log_ratio.exp()
        surrogate = torch.minimum(
            ratio * mb_advantages,
            ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            )
            * mb_advantages,
        )
        approx_kl = (ratio - 1.0) - log_ratio
        clip_fraction = (
            (ratio - 1.0).abs() > config.clip_ratio
        ).float()
        surrogate_means = opponent_group_means(surrogate, mb_groups)
        approx_kl_means = opponent_group_means(approx_kl, mb_groups)
        clip_means = opponent_group_means(clip_fraction, mb_groups)
        row_counts = Counter(mb_groups)
        for group_name, rows in row_counts.items():
            totals[group_name]["rows"] += rows
            totals[group_name]["surrogate"] += (
                float(surrogate_means[group_name]) * rows
            )
            totals[group_name]["approx_kl"] += (
                float(approx_kl_means[group_name]) * rows
            )
            totals[group_name]["clip_fraction"] += (
                float(clip_means[group_name]) * rows
            )
    return {
        group_name: {
            "rows": int(group_totals["rows"]),
            **{
                key: float(group_totals[key])
                / max(int(group_totals["rows"]), 1)
                for key in ("surrogate", "approx_kl", "clip_fraction")
            },
        }
        for group_name, group_totals in sorted(totals.items())
    }


def ppo_update_constrained(
    model: EntityOptionPolicy,
    reference_model: EntityOptionPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Any]],
    model_config: dict[str, Any],
    config: PPOConfig,
    device: torch.device,
    update: int,
    objective_state: ConstrainedObjectiveState,
) -> dict[str, Any]:
    """Opponent-balanced PPO with a primal-dual guard-opponent constraint."""
    model.eval()
    reference_model.eval()
    schedule_progress = phase_schedule_progress(config, update)
    if config.learning_rate_schedule == "constant":
        learning_rate_factor = 1.0
    elif config.learning_rate_schedule == "cosine":
        learning_rate_factor = 0.10 + 0.90 * 0.5 * (
            1.0 + math.cos(math.pi * schedule_progress)
        )
    else:
        raise ValueError(
            f"Unsupported learning-rate schedule: "
            f"{config.learning_rate_schedule!r}"
        )
    optimizer.param_groups[0]["lr"] = (
        config.learning_rate * learning_rate_factor
    )
    optimizer.param_groups[1]["lr"] = (
        config.value_learning_rate * learning_rate_factor
    )

    advantages, advantage_audit = normalize_rollout_advantages(
        transitions,
        config.advantage_normalization,
    )
    opponent_groups = [
        transition_opponent_group(transition)
        for transition in transitions
    ]
    rollout_groups = set(opponent_groups)
    primary_name = config.primary_opponent_name
    guard_name = config.guard_opponent_name
    if primary_name is None or guard_name is None:
        raise ValueError("Constrained PPO requires primary and guard opponents")
    missing_core = {primary_name, guard_name} - rollout_groups
    if missing_core:
        raise RuntimeError(
            "Constrained rollout is missing core opponents: "
            + ", ".join(sorted(missing_core))
        )
    missing_weights = rollout_groups - set(config.opponent_loss_weights)
    if missing_weights:
        raise ValueError(
            "Constrained rollout lacks group loss weights for "
            + ", ".join(sorted(missing_weights))
        )

    outcomes = torch.tensor(
        [float(t["outcome_target"]) for t in transitions],
        dtype=torch.float32,
    )
    old_log_probs = torch.tensor(
        [float(t["old_log_prob"]) for t in transitions],
        dtype=torch.float32,
    )
    action_counts = torch.tensor(
        [int(t["action_count"]) for t in transitions],
        dtype=torch.long,
    )
    action_sequences = torch.full(
        (len(transitions), MAX_ACTION_COUNT),
        -1,
        dtype=torch.long,
    )
    for row_index, transition in enumerate(transitions):
        action = transition["action"]
        action_sequences[row_index, : len(action)] = torch.tensor(action)
    rollout_batch = collate_features_cpu(
        [transition["feature"] for transition in transitions],
        model_config,
    )

    stats: Counter[str] = Counter()
    group_stats: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.time()
    kl_coefficient = bc_kl_coefficient(config, update)
    actor_parameters = list(optimizer.param_groups[0]["params"])
    value_parameters = list(optimizer.param_groups[1]["params"])
    trainable_parameters = actor_parameters + value_parameters
    optimizer_steps = 0
    early_stop = False
    early_stop_reason: str | None = None
    last_epoch_core_approx_kl: dict[str, float] = {}
    gradient_alignment: dict[str, float] | None = None
    pcgrad_stats: Counter[str] = Counter()
    pcgrad_min_primary_dot: float | None = None
    pcgrad_min_guard_dot: float | None = None
    dual_during_update = objective_state.dual_value

    for epoch in range(config.ppo_epochs):
        permutation = torch.randperm(len(transitions))
        epoch_approx_kl: list[float] = []
        epoch_group_approx_kl: Counter[str] = Counter()
        epoch_group_rows: Counter[str] = Counter()
        for start in range(0, len(transitions), config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            index_list = indices.tolist()
            batch = {
                key: value.index_select(0, indices).to(
                    device,
                    non_blocking=True,
                )
                for key, value in rollout_batch.items()
            }
            mb_actions = action_sequences[indices].to(device)
            mb_counts = action_counts[indices].to(device)
            mb_old_log_probs = old_log_probs[indices].to(device)
            mb_advantages = advantages[indices].to(device)
            mb_outcomes = outcomes[indices].to(device)
            mb_groups = [opponent_groups[index] for index in index_list]

            optimizer.zero_grad(set_to_none=True)
            outputs = model_forward(model, batch, device)
            new_log_probs, entropy = ordered_action_log_prob_entropy(
                outputs,
                batch,
                mb_actions,
                mb_counts,
                temperature=config.policy_temperature,
            )
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = log_ratio.exp()
            unclipped = ratio * mb_advantages
            clipped = ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * mb_advantages
            surrogate = torch.minimum(unclipped, clipped)
            surrogate_by_group = opponent_group_means(
                surrogate,
                mb_groups,
            )

            base_policy_objective = surrogate.new_zeros(())
            if primary_name in surrogate_by_group:
                base_policy_objective = (
                    base_policy_objective
                    + config.primary_policy_weight
                    * surrogate_by_group[primary_name]
                )
            if guard_name in surrogate_by_group:
                base_policy_objective = (
                    base_policy_objective
                    + config.guard_policy_weight
                    * surrogate_by_group[guard_name]
                )
            auxiliary_surrogates = [
                value
                for name, value in surrogate_by_group.items()
                if name not in {primary_name, guard_name}
            ]
            if auxiliary_surrogates:
                base_policy_objective = (
                    base_policy_objective
                    + config.auxiliary_policy_weight
                    * torch.stack(auxiliary_surrogates).mean()
                )
            guard_surrogate = surrogate_by_group.get(guard_name)
            constraint_term = (
                dual_during_update
                * (guard_surrogate - config.guard_surrogate_floor)
                if guard_surrogate is not None
                else surrogate.new_zeros(())
            )
            policy_objective = base_policy_objective + constraint_term
            policy_loss = -policy_objective

            value_loss_per_row = F.binary_cross_entropy_with_logits(
                outputs["value_logits"].float(),
                mb_outcomes,
                reduction="none",
            )
            value_loss, value_loss_by_group = weighted_opponent_group_mean(
                value_loss_per_row,
                mb_groups,
                config.opponent_loss_weights,
            )

            with torch.no_grad():
                reference_outputs = model_forward(reference_model, batch, device)
            anchor_kl_per_row = reference_policy_kl(
                outputs,
                reference_outputs,
                batch,
                mb_actions,
                mb_counts,
                temperature=config.policy_temperature,
            )
            anchor_kl, anchor_kl_by_group = weighted_opponent_group_mean(
                anchor_kl_per_row,
                mb_groups,
                config.opponent_loss_weights,
            )
            entropy_mean, entropy_by_group = weighted_opponent_group_mean(
                entropy,
                mb_groups,
                config.opponent_loss_weights,
            )
            approx_kl_per_row = (ratio - 1.0) - log_ratio
            approx_kl, approx_kl_by_group = weighted_opponent_group_mean(
                approx_kl_per_row,
                mb_groups,
                config.opponent_loss_weights,
            )
            clip_fraction_per_row = (
                (ratio - 1.0).abs() > config.clip_ratio
            ).float()
            clip_fraction, clip_fraction_by_group = (
                weighted_opponent_group_mean(
                    clip_fraction_per_row,
                    mb_groups,
                    config.opponent_loss_weights,
                )
            )
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                + kl_coefficient * anchor_kl
                - config.entropy_coefficient * entropy_mean
            )

            pcgrad_adjustments: list[torch.Tensor | None] | None = None
            pcgrad_audit: dict[str, float | bool] | None = None
            if config.constrained_gradient_mode in {
                "pcgrad",
                "guard_pcgrad",
            }:
                if (
                    primary_name not in surrogate_by_group
                    or guard_name not in surrogate_by_group
                ):
                    raise RuntimeError(
                        "PCGrad minibatch is missing a core opponent"
                    )
                primary_task_loss = (
                    -config.primary_policy_weight
                    * surrogate_by_group[primary_name]
                )
                guard_task_loss = (
                    -config.guard_policy_weight
                    * surrogate_by_group[guard_name]
                )
                projection_function = (
                    symmetric_pcgrad_adjustment
                    if config.constrained_gradient_mode == "pcgrad"
                    else guard_priority_pcgrad_adjustment
                )
                pcgrad_adjustments, pcgrad_audit = (
                    projection_function(
                        primary_task_loss,
                        guard_task_loss,
                        actor_parameters,
                    )
                )
            if (
                gradient_alignment is None
                and primary_name in surrogate_by_group
                and guard_name in surrogate_by_group
            ):
                gradient_alignment = gradient_cosine(
                    surrogate_by_group[primary_name],
                    surrogate_by_group[guard_name],
                    actor_parameters,
                )
            loss.backward()
            if pcgrad_adjustments is not None and pcgrad_audit is not None:
                for parameter, adjustment in zip(
                    actor_parameters,
                    pcgrad_adjustments,
                ):
                    if adjustment is None:
                        continue
                    if parameter.grad is None:
                        parameter.grad = adjustment.clone()
                    else:
                        parameter.grad.add_(
                            adjustment.to(
                                device=parameter.grad.device,
                                dtype=parameter.grad.dtype,
                            )
                        )
                pcgrad_stats["batches"] += 1
                pcgrad_stats["conflict_batches"] += int(
                    pcgrad_audit["conflict"]
                )
                for key in (
                    "pre_dot",
                    "pre_cosine",
                    "primary_norm",
                    "guard_norm",
                    "primary_projection_coefficient",
                    "guard_projection_coefficient",
                ):
                    pcgrad_stats[key] += float(pcgrad_audit[key])
                projected_primary_dot = float(
                    pcgrad_audit["projected_core_dot_primary"]
                )
                projected_guard_dot = float(
                    pcgrad_audit["projected_core_dot_guard"]
                )
                pcgrad_min_primary_dot = (
                    projected_primary_dot
                    if pcgrad_min_primary_dot is None
                    else min(pcgrad_min_primary_dot, projected_primary_dot)
                )
                pcgrad_min_guard_dot = (
                    projected_guard_dot
                    if pcgrad_min_guard_dot is None
                    else min(pcgrad_min_guard_dot, projected_guard_dot)
                )
            actor_grad_norm = gradient_l2_norm(actor_parameters)
            value_grad_norm = gradient_l2_norm(value_parameters)
            grad_norm = nn.utils.clip_grad_norm_(
                trainable_parameters,
                config.max_grad_norm,
            )
            optimizer.step()
            optimizer_steps += 1

            batch_rows = len(index_list)
            stats["rows"] += batch_rows
            stats["loss"] += float(loss.detach()) * batch_rows
            stats["policy_loss"] += float(policy_loss.detach()) * batch_rows
            stats["policy_objective"] += (
                float(policy_objective.detach()) * batch_rows
            )
            stats["base_policy_objective"] += (
                float(base_policy_objective.detach()) * batch_rows
            )
            stats["constraint_term"] += (
                float(constraint_term.detach()) * batch_rows
            )
            stats["value_loss"] += float(value_loss.detach()) * batch_rows
            stats["entropy"] += float(entropy_mean.detach()) * batch_rows
            stats["bc_anchor_kl"] += float(anchor_kl.detach()) * batch_rows
            stats["approx_kl"] += float(approx_kl.detach()) * batch_rows
            stats["clip_fraction"] += float(clip_fraction.detach()) * batch_rows
            stats["grad_norm"] += float(grad_norm.detach()) * batch_rows
            stats["actor_grad_norm"] += (
                float(actor_grad_norm.detach()) * batch_rows
            )
            stats["value_grad_norm"] += (
                float(value_grad_norm.detach()) * batch_rows
            )

            group_row_counts = Counter(mb_groups)
            for group_name, group_rows in group_row_counts.items():
                group_stats[group_name]["rows"] += group_rows
                group_stats[group_name]["surrogate"] += (
                    float(surrogate_by_group[group_name].detach()) * group_rows
                )
                group_stats[group_name]["value_loss"] += (
                    float(value_loss_by_group[group_name].detach()) * group_rows
                )
                group_stats[group_name]["entropy"] += (
                    float(entropy_by_group[group_name].detach()) * group_rows
                )
                group_stats[group_name]["anchor_kl"] += (
                    float(anchor_kl_by_group[group_name].detach()) * group_rows
                )
                group_stats[group_name]["approx_kl"] += (
                    float(approx_kl_by_group[group_name].detach()) * group_rows
                )
                group_stats[group_name]["clip_fraction"] += (
                    float(
                        clip_fraction_by_group[group_name].detach()
                    )
                    * group_rows
                )
                epoch_group_approx_kl[group_name] += (
                    float(approx_kl_by_group[group_name].detach())
                    * group_rows
                )
                epoch_group_rows[group_name] += group_rows
            epoch_approx_kl.append(float(approx_kl.detach()))

        last_epoch_core_approx_kl = {
            name: epoch_group_approx_kl[name]
            / max(epoch_group_rows[name], 1)
            for name in (primary_name, guard_name)
        }
        aggregate_kl_exceeded = (
            epoch_approx_kl
            and sum(epoch_approx_kl) / len(epoch_approx_kl)
            > config.target_kl * 1.5
        )
        core_kl_exceeded = any(
            value > config.target_kl * 1.5
            for value in last_epoch_core_approx_kl.values()
        )
        if aggregate_kl_exceeded or core_kl_exceeded:
            early_stop = True
            early_stop_reason = (
                "core_opponent_approx_kl"
                if core_kl_exceeded
                else "weighted_approx_kl"
            )
            break

    final_policy_shift = evaluate_constrained_policy_shift(
        model,
        rollout_batch,
        action_sequences,
        action_counts,
        old_log_probs,
        advantages,
        opponent_groups,
        config,
        device,
    )
    missing_final_groups = rollout_groups - set(final_policy_shift)
    if missing_final_groups:
        raise RuntimeError(
            "Final constrained-policy audit lost opponent groups: "
            + ", ".join(sorted(missing_final_groups))
        )
    final_core_approx_kl = {
        name: float(final_policy_shift[name]["approx_kl"])
        for name in (primary_name, guard_name)
    }
    unsafe_core_kl = {
        name: value
        for name, value in final_core_approx_kl.items()
        if value > config.target_kl * 1.5
    }
    if unsafe_core_kl:
        raise RuntimeError(
            "Final constrained policy exceeded core opponent KL safety: "
            + json.dumps(unsafe_core_kl, sort_keys=True)
        )
    mean_guard_surrogate = float(
        final_policy_shift[guard_name]["surrogate"]
    )
    constraint_audit = objective_state.finish_update(
        mean_guard_surrogate,
        update=update,
        floor=config.guard_surrogate_floor,
        dual_learning_rate=config.constraint_dual_lr,
        dual_max=config.constraint_dual_max,
    )
    rows = stats["rows"]
    result = {
        key: value / max(rows, 1)
        for key, value in stats.items()
        if key != "rows"
    }
    result.update(
        {
            "rows": rows,
            "transitions": len(transitions),
            "optimizer_steps": optimizer_steps,
            "epochs_completed": epoch + 1,
            "early_stop": early_stop,
            "early_stop_reason": early_stop_reason,
            "bc_kl_coefficient": kl_coefficient,
            "schedule_progress": schedule_progress,
            "phase_step": update - config.schedule_start_update + 1,
            "learning_rate_factor": learning_rate_factor,
            "actor_learning_rate": optimizer.param_groups[0]["lr"],
            "value_learning_rate": optimizer.param_groups[1]["lr"],
            "advantage_mean_raw": float(
                torch.tensor(
                    [float(t["advantage"]) for t in transitions]
                ).mean()
            ),
            "advantage_std_raw": float(
                torch.tensor(
                    [float(t["advantage"]) for t in transitions]
                ).std()
            ),
            "advantage_normalization": advantage_audit,
            "outcome_mean": float(outcomes.mean()),
            "objective": {
                "mode": "constrained",
                "primary_opponent": primary_name,
                "guard_opponent": guard_name,
                "primary_policy_weight": config.primary_policy_weight,
                "guard_policy_weight": config.guard_policy_weight,
                "auxiliary_policy_weight": config.auxiliary_policy_weight,
                "opponent_loss_weights": dict(
                    config.opponent_loss_weights
                ),
                **constraint_audit,
                "gradient_alignment": gradient_alignment,
                "gradient_surgery": (
                    {
                        "mode": config.constrained_gradient_mode,
                        "batches": int(pcgrad_stats["batches"]),
                        "conflict_batches": int(
                            pcgrad_stats["conflict_batches"]
                        ),
                        "conflict_fraction": (
                            pcgrad_stats["conflict_batches"]
                            / max(pcgrad_stats["batches"], 1)
                        ),
                        **{
                            f"mean_{key}": (
                                float(pcgrad_stats[key])
                                / max(pcgrad_stats["batches"], 1)
                            )
                            for key in (
                                "pre_dot",
                                "pre_cosine",
                                "primary_norm",
                                "guard_norm",
                                "primary_projection_coefficient",
                                "guard_projection_coefficient",
                            )
                        },
                        "min_projected_core_dot_primary": (
                            pcgrad_min_primary_dot
                        ),
                        "min_projected_core_dot_guard": (
                            pcgrad_min_guard_dot
                        ),
                    }
                    if config.constrained_gradient_mode
                    in {"pcgrad", "guard_pcgrad"}
                    else {"mode": "scalar"}
                ),
                "last_epoch_core_approx_kl": last_epoch_core_approx_kl,
                "final_core_approx_kl": final_core_approx_kl,
            },
            "final_policy_shift_by_opponent": final_policy_shift,
            "by_opponent": {
                group_name: {
                    key: (
                        float(value) / max(int(group_metrics["rows"]), 1)
                    )
                    for key, value in group_metrics.items()
                    if key != "rows"
                }
                | {"rows": int(group_metrics["rows"])}
                for group_name, group_metrics in sorted(group_stats.items())
            },
            "seconds": time.time() - started,
        }
    )
    return result


def audit_actor_reduction_gradients(
    model: EntityOptionPolicy,
    transitions: list[dict[str, Any]],
    model_config: dict[str, Any],
    config: PPOConfig,
    device: torch.device,
    actor_parameters: list[nn.Parameter],
) -> dict[str, Any]:
    """Compare transition-mean and candidate actor gradients without an update."""
    if config.ppo_objective != "standard":
        raise ValueError("Actor-reduction audit requires standard PPO")
    if config.advantage_normalization != "global":
        raise ValueError("Actor-reduction audit requires global advantages")
    if not actor_parameters:
        raise ValueError("Actor-reduction audit requires actor parameters")
    candidate_reduction = getattr(
        config,
        "actor_reduction",
        "quota_group_mean",
    )
    if candidate_reduction == "transition_mean":
        raise ValueError(
            "Actor-reduction audit requires a non-baseline candidate"
        )

    model.eval()
    state_before = model_state_sha256(model)
    raw_advantages = torch.tensor(
        [float(transition["advantage"]) for transition in transitions],
        dtype=torch.float32,
    )
    advantages = (
        raw_advantages - raw_advantages.mean()
    ) / raw_advantages.std().clamp_min(1e-6)
    old_log_probs = torch.tensor(
        [float(transition["old_log_prob"]) for transition in transitions],
        dtype=torch.float32,
    )
    action_counts = torch.tensor(
        [int(transition["action_count"]) for transition in transitions],
        dtype=torch.long,
    )
    action_sequences = torch.full(
        (len(transitions), MAX_ACTION_COUNT),
        -1,
        dtype=torch.long,
    )
    for row_index, transition in enumerate(transitions):
        action = transition["action"]
        action_sequences[row_index, : len(action)] = torch.tensor(action)
    rollout_batch = collate_features_cpu(
        [transition["feature"] for transition in transitions],
        model_config,
    )
    candidate_multipliers, weight_audit = actor_reduction_multipliers(
        transitions,
        candidate_reduction,
        opponent_quotas=getattr(config, "opponent_base_quotas", {}),
        expected_episode_count=getattr(
            config,
            "games_per_update",
            None,
        ),
    )

    standard_gradients = [
        torch.zeros_like(parameter, dtype=torch.float32)
        for parameter in actor_parameters
    ]
    candidate_gradients = [
        torch.zeros_like(parameter, dtype=torch.float32)
        for parameter in actor_parameters
    ]
    ratio_abs_max = 0.0
    ratio_abs_sum = 0.0
    ratio_rows = 0
    minibatches = 0
    total_rows = len(transitions)
    for start in range(0, total_rows, config.minibatch_size):
        indices = torch.arange(
            start,
            min(start + config.minibatch_size, total_rows),
            dtype=torch.long,
        )
        batch = {
            key: value.index_select(0, indices).to(
                device,
                non_blocking=True,
            )
            for key, value in rollout_batch.items()
        }
        mb_actions = action_sequences[indices].to(device)
        mb_counts = action_counts[indices].to(device)
        mb_old_log_probs = old_log_probs[indices].to(device)
        mb_advantages = advantages[indices].to(device)
        mb_multipliers = candidate_multipliers[indices].to(device)

        outputs = model_forward(model, batch, device)
        new_log_probs, _ = ordered_action_log_prob_entropy(
            outputs,
            batch,
            mb_actions,
            mb_counts,
            temperature=config.policy_temperature,
        )
        log_ratio = new_log_probs - mb_old_log_probs
        ratio = log_ratio.exp()
        surrogate = torch.minimum(
            ratio * mb_advantages,
            ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            )
            * mb_advantages,
        )
        # Divide each chunk by the full rollout size so the accumulated
        # gradients are the exact full-rollout objectives, independent of the
        # final chunk size.
        standard_loss = -surrogate.sum() / total_rows
        candidate_loss = (
            -(surrogate * mb_multipliers).sum() / total_rows
        )
        standard_chunk = torch.autograd.grad(
            standard_loss,
            actor_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        candidate_chunk = torch.autograd.grad(
            candidate_loss,
            actor_parameters,
            allow_unused=True,
        )
        for destination, gradient in zip(
            standard_gradients,
            standard_chunk,
        ):
            if gradient is not None:
                destination.add_(gradient.detach().float())
        for destination, gradient in zip(
            candidate_gradients,
            candidate_chunk,
        ):
            if gradient is not None:
                destination.add_(gradient.detach().float())
        ratio_abs = log_ratio.detach().abs()
        ratio_abs_max = max(ratio_abs_max, float(ratio_abs.max()))
        ratio_abs_sum += float(ratio_abs.sum())
        ratio_rows += int(ratio_abs.numel())
        minibatches += 1

    origin = standard_gradients[0].new_zeros(())
    dot = origin.clone()
    standard_squared = origin.clone()
    candidate_squared = origin.clone()
    for standard_gradient, candidate_gradient in zip(
        standard_gradients,
        candidate_gradients,
    ):
        dot = dot + (standard_gradient * candidate_gradient).sum()
        standard_squared = (
            standard_squared + standard_gradient.square().sum()
        )
        candidate_squared = (
            candidate_squared + candidate_gradient.square().sum()
        )
    standard_norm = standard_squared.sqrt()
    candidate_norm = candidate_squared.sqrt()
    denominator = standard_norm * candidate_norm
    cosine = (
        dot / denominator
        if float(denominator) > 0.0
        else origin.new_tensor(0.0)
    )
    norm_ratio = (
        float(candidate_norm / standard_norm)
        if float(standard_norm) > 0.0
        else float("nan")
    )
    state_after = model_state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("Actor-reduction audit mutated model weights")
    cosine_value = float(cosine)
    weight_trigger = (
        float(weight_audit["max_relative_weight_deviation"]) >= 0.10
    )
    cosine_trigger = cosine_value < 0.995
    gradient_metrics = {
        "cosine": cosine_value,
        "standard_norm": float(standard_norm),
        "candidate_norm": float(candidate_norm),
        "candidate_to_standard_norm_ratio": norm_ratio,
    }
    if candidate_reduction == "quota_group_mean":
        gradient_metrics.update(
            {
                "quota_group_norm": float(candidate_norm),
                "quota_to_standard_norm_ratio": norm_ratio,
            }
        )
    elif candidate_reduction == "episode_mean":
        gradient_metrics.update(
            {
                "episode_mean_norm": float(candidate_norm),
                "episode_to_standard_norm_ratio": norm_ratio,
            }
        )
    return {
        "schema_version": "ptcg-ppo-actor-reduction-audit-v1",
        "status": "completed_no_update",
        "standard_reduction": "transition_mean",
        "candidate_reduction": candidate_reduction,
        "transitions": total_rows,
        "minibatches": minibatches,
        "actor_parameter_tensors": len(actor_parameters),
        "actor_parameters": sum(
            parameter.numel() for parameter in actor_parameters
        ),
        "row_weight_audit": weight_audit,
        "gradient": gradient_metrics,
        "behavior_policy_alignment": {
            "mean_abs_log_ratio": ratio_abs_sum / max(ratio_rows, 1),
            "max_abs_log_ratio": ratio_abs_max,
        },
        "pre_registered_gate": {
            "cosine_threshold": 0.995,
            "relative_row_weight_deviation_threshold": 0.10,
            "cosine_triggered": cosine_trigger,
            "row_weight_triggered": weight_trigger,
            "training_authorized": cosine_trigger or weight_trigger,
        },
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "optimizer_steps": 0,
    }


def audit_actor_value_gradients(
    model: EntityOptionPolicy,
    reference_model: EntityOptionPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Any]],
    model_config: dict[str, Any],
    config: PPOConfig,
    device: torch.device,
    update: int,
    actor_parameters: list[nn.Parameter],
    value_parameters: list[nn.Parameter],
    replay_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Audit actor-priority critic projection without any parameter update."""
    mode = validate_actor_value_gradient_configuration(
        mode=getattr(
            config,
            "actor_value_gradient_mode",
            "scalar",
        ),
        ppo_objective=config.ppo_objective,
        trainable_scope=config.trainable_scope,
        value_trunk_gradient_scale=getattr(
            config,
            "value_trunk_gradient_scale",
            1.0,
        ),
        audit_only=True,
        actor_reduction_audit_only=config.actor_reduction_audit_only,
    )
    if not actor_parameters:
        raise ValueError("Actor/value gradient audit requires actor parameters")
    if any(
        parameter.grad is not None
        for parameter in actor_parameters + value_parameters
    ):
        raise RuntimeError(
            "Actor/value gradient audit requires pristine parameter gradients"
        )

    model.eval()
    reference_model.eval()
    learner_hash_before = model_state_sha256(model)
    reference_hash_before = model_state_sha256(reference_model)
    optimizer_hash_before = structured_state_sha256(optimizer.state_dict())
    replay_optimizer_hash_before = (
        structured_state_sha256(replay_optimizer.state_dict())
        if replay_optimizer is not None
        else None
    )
    advantages, advantage_audit = normalize_rollout_advantages(
        transitions,
        config.advantage_normalization,
    )
    actor_multipliers, actor_reduction_audit = (
        actor_reduction_multipliers(
            transitions,
            config.actor_reduction,
            opponent_quotas=getattr(config, "opponent_base_quotas", {}),
            expected_episode_count=getattr(
                config,
                "games_per_update",
                None,
            ),
        )
    )

    outcomes = torch.tensor(
        [float(transition["outcome_target"]) for transition in transitions],
        dtype=torch.float32,
    )
    old_log_probs = torch.tensor(
        [float(transition["old_log_prob"]) for transition in transitions],
        dtype=torch.float32,
    )
    action_counts = torch.tensor(
        [int(transition["action_count"]) for transition in transitions],
        dtype=torch.long,
    )
    action_sequences = torch.full(
        (len(transitions), MAX_ACTION_COUNT),
        -1,
        dtype=torch.long,
    )
    for row_index, transition in enumerate(transitions):
        action = transition["action"]
        action_sequences[row_index, : len(action)] = torch.tensor(action)
    rollout_batch = collate_features_cpu(
        [transition["feature"] for transition in transitions],
        model_config,
    )

    kl_coefficient = bc_kl_coefficient(config, update)
    audit_totals: Counter[str] = Counter()
    conflict_batches = 0
    minibatches = 0
    min_pre_dot: float | None = None
    min_pre_cosine: float | None = None
    min_actor_norm: float | None = None
    min_value_norm: float | None = None
    min_projected_dot: float | None = None
    min_final_margin: float | None = None
    max_tolerance = 0.0
    invariant_violations: list[dict[str, Any]] = []
    shared_tensor_count = 0
    shared_parameter_count = 0
    actor_only_tensor_count = 0
    actor_only_parameter_count = 0
    log_ratio_abs_sum = 0.0
    log_ratio_abs_max = 0.0
    log_ratio_rows = 0

    for start in range(0, len(transitions), config.minibatch_size):
        indices = torch.arange(
            start,
            min(start + config.minibatch_size, len(transitions)),
            dtype=torch.long,
        )
        batch = {
            key: value.index_select(0, indices).to(
                device,
                non_blocking=True,
            )
            for key, value in rollout_batch.items()
        }
        mb_actions = action_sequences[indices].to(device)
        mb_counts = action_counts[indices].to(device)
        mb_old_log_probs = old_log_probs[indices].to(device)
        mb_advantages = advantages[indices].to(device)
        mb_actor_multipliers = actor_multipliers[indices].to(device)
        mb_outcomes = outcomes[indices].to(device)

        outputs = model_forward(model, batch, device)
        new_log_probs, entropy = ordered_action_log_prob_entropy(
            outputs,
            batch,
            mb_actions,
            mb_counts,
            temperature=config.policy_temperature,
        )
        log_ratio = new_log_probs - mb_old_log_probs
        ratio = log_ratio.exp()
        surrogate = torch.minimum(
            ratio * mb_advantages,
            ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            )
            * mb_advantages,
        )
        policy_loss = standard_actor_policy_loss(
            surrogate,
            mb_actor_multipliers,
            config.actor_reduction,
        )
        value_loss = F.binary_cross_entropy_with_logits(
            outputs["value_logits"].float(),
            mb_outcomes,
        )
        with torch.no_grad():
            reference_outputs = model_forward(
                reference_model,
                batch,
                device,
            )
        anchor_kl = reference_policy_kl(
            outputs,
            reference_outputs,
            batch,
            mb_actions,
            mb_counts,
            temperature=config.policy_temperature,
        ).mean()
        entropy_mean = entropy.mean()
        actor_side_loss = (
            policy_loss
            + kl_coefficient * anchor_kl
            - config.entropy_coefficient * entropy_mean
        )
        value_side_loss = config.value_coefficient * value_loss
        _, gradient_audit = actor_priority_value_pcgrad_adjustment(
            actor_side_loss,
            value_side_loss,
            actor_parameters,
        )

        shared_tensor_count = int(
            gradient_audit["shared_tensor_count"]
        )
        shared_parameter_count = int(
            gradient_audit["shared_parameter_count"]
        )
        actor_only_tensor_count = int(
            gradient_audit["actor_only_tensor_count"]
        )
        actor_only_parameter_count = int(
            gradient_audit["actor_only_parameter_count"]
        )
        conflict_batches += int(gradient_audit["conflict"])
        for key in (
            "pre_dot",
            "pre_cosine",
            "actor_norm",
            "value_norm",
            "projection_coefficient",
            "adjustment_norm",
        ):
            audit_totals[key] += float(gradient_audit[key])
        pre_dot = float(gradient_audit["pre_dot"])
        pre_cosine = float(gradient_audit["pre_cosine"])
        actor_norm = float(gradient_audit["actor_norm"])
        value_norm = float(gradient_audit["value_norm"])
        projected_dot = float(
            gradient_audit["projected_value_dot_actor"]
        )
        final_margin = float(
            gradient_audit["final_dot_actor_margin"]
        )
        tolerance = 1e-6 * max(
            abs(pre_dot),
            float(gradient_audit["actor_norm"])
            * float(gradient_audit["value_norm"]),
            1e-6,
        )
        max_tolerance = max(max_tolerance, tolerance)
        violations: list[str] = []
        if not bool(gradient_audit["finite"]):
            violations.append("non_finite")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (
                actor_side_loss.detach(),
                value_side_loss.detach(),
                log_ratio.detach(),
            )
        ):
            violations.append("non_finite_loss_or_log_ratio")
        if shared_parameter_count <= 0:
            violations.append("no_shared_parameters")
        if bool(gradient_audit["conflict"]) and projected_dot < -tolerance:
            violations.append("projected_value_dot_actor_negative")
        if final_margin < -tolerance:
            violations.append("final_dot_actor_margin_negative")
        if violations:
            invariant_violations.append(
                {
                    "minibatch": minibatches,
                    "violations": violations,
                    "tolerance": tolerance,
                    "gradient": gradient_audit,
                }
            )
        min_pre_dot = (
            pre_dot if min_pre_dot is None else min(min_pre_dot, pre_dot)
        )
        min_pre_cosine = (
            pre_cosine
            if min_pre_cosine is None
            else min(min_pre_cosine, pre_cosine)
        )
        min_actor_norm = (
            actor_norm
            if min_actor_norm is None
            else min(min_actor_norm, actor_norm)
        )
        min_value_norm = (
            value_norm
            if min_value_norm is None
            else min(min_value_norm, value_norm)
        )
        min_projected_dot = (
            projected_dot
            if min_projected_dot is None
            else min(min_projected_dot, projected_dot)
        )
        min_final_margin = (
            final_margin
            if min_final_margin is None
            else min(min_final_margin, final_margin)
        )
        log_ratio_abs = log_ratio.detach().abs()
        log_ratio_abs_sum += float(log_ratio_abs.sum())
        log_ratio_abs_max = max(
            log_ratio_abs_max,
            float(log_ratio_abs.max()),
        )
        log_ratio_rows += int(log_ratio_abs.numel())
        minibatches += 1

    learner_hash_after = model_state_sha256(model)
    reference_hash_after = model_state_sha256(reference_model)
    optimizer_hash_after = structured_state_sha256(optimizer.state_dict())
    replay_optimizer_hash_after = (
        structured_state_sha256(replay_optimizer.state_dict())
        if replay_optimizer is not None
        else None
    )
    gradients_remain_none = all(
        parameter.grad is None
        for parameter in actor_parameters + value_parameters
    )
    if learner_hash_after != learner_hash_before:
        invariant_violations.append(
            {"violations": ["learner_model_mutated"]}
        )
    if reference_hash_after != reference_hash_before:
        invariant_violations.append(
            {"violations": ["reference_model_mutated"]}
        )
    if optimizer_hash_after != optimizer_hash_before:
        invariant_violations.append(
            {"violations": ["optimizer_state_mutated"]}
        )
    if replay_optimizer_hash_after != replay_optimizer_hash_before:
        invariant_violations.append(
            {"violations": ["replay_optimizer_state_mutated"]}
        )
    if not gradients_remain_none:
        invariant_violations.append(
            {"violations": ["parameter_grad_buffers_mutated"]}
        )

    effect_status = (
        "conflict_observed"
        if conflict_batches > 0
        else "no_effect_observed"
    )
    return {
        "schema_version": "ptcg-ppo-actor-value-gradient-audit-v1",
        "status": "completed_no_update",
        "effect_status": effect_status,
        "mode": mode,
        "surgery": actor_value_gradient_manifest(
            mode,
            audit_only=True,
        ),
        "rollout_integrity": {
            "transitions": len(transitions),
            "minibatches": minibatches,
            "minibatch_order": "sequential_frozen_rollout",
            "advantage_normalization": advantage_audit,
            "actor_reduction": actor_reduction_audit,
            "bc_kl_coefficient": kl_coefficient,
            "behavior_policy_alignment": {
                "mean_abs_log_ratio": (
                    log_ratio_abs_sum / max(log_ratio_rows, 1)
                ),
                "max_abs_log_ratio": log_ratio_abs_max,
            },
        },
        "parameter_scope": {
            "shared_tensor_count": shared_tensor_count,
            "shared_parameter_count": shared_parameter_count,
            "actor_only_tensor_count": actor_only_tensor_count,
            "actor_only_parameter_count": actor_only_parameter_count,
            "value_head_tensor_count": len(value_parameters),
            "value_head_parameter_count": sum(
                parameter.numel() for parameter in value_parameters
            ),
        },
        "gradient": {
            "batches": minibatches,
            "conflict_batches": conflict_batches,
            "conflict_fraction": conflict_batches / max(minibatches, 1),
            **{
                f"mean_{key}": (
                    float(audit_totals[key]) / max(minibatches, 1)
                )
                for key in (
                    "pre_dot",
                    "pre_cosine",
                    "actor_norm",
                    "value_norm",
                    "projection_coefficient",
                    "adjustment_norm",
                )
            },
            "min_pre_dot": min_pre_dot,
            "min_pre_cosine": min_pre_cosine,
            "min_actor_norm": min_actor_norm,
            "min_value_norm": min_value_norm,
            "min_projected_value_dot_actor": min_projected_dot,
            "min_final_dot_actor_margin": min_final_margin,
            "max_numerical_tolerance": max_tolerance,
            "finite_and_invariant_violation_count": len(
                invariant_violations
            ),
            "invariant_violations": invariant_violations,
        },
        "state_integrity": {
            "learner_model_sha256_before": learner_hash_before,
            "learner_model_sha256_after": learner_hash_after,
            "reference_model_sha256_before": reference_hash_before,
            "reference_model_sha256_after": reference_hash_after,
            "optimizer_state_sha256_before": optimizer_hash_before,
            "optimizer_state_sha256_after": optimizer_hash_after,
            "replay_optimizer": (
                "present" if replay_optimizer is not None else "absent"
            ),
            "replay_optimizer_state_sha256_before": (
                replay_optimizer_hash_before
            ),
            "replay_optimizer_state_sha256_after": (
                replay_optimizer_hash_after
            ),
            "parameter_gradients_remain_none": gradients_remain_none,
        },
        "optimizer_steps": 0,
        "bc_replay_steps": 0,
        "evaluation_calls": 0,
        "checkpoint_saves": 0,
        "pre_registered_gate": {
            "shared_parameters_present": shared_parameter_count > 0,
            "all_values_finite_and_invariants_hold": not invariant_violations,
            "conflict_observed": conflict_batches > 0,
            "training_authorized": (
                shared_parameter_count > 0
                and conflict_batches > 0
                and not invariant_violations
            ),
        },
    }


def ppo_update(
    model: EntityOptionPolicy,
    reference_model: EntityOptionPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Any]],
    model_config: dict[str, Any],
    config: PPOConfig,
    device: torch.device,
    update: int,
    objective_state: ConstrainedObjectiveState | None = None,
) -> dict[str, Any]:
    value_trunk_gradient_scale = validate_value_trunk_gradient_scale(
        getattr(config, "value_trunk_gradient_scale", 1.0)
    )
    actor_value_gradient_mode = (
        validate_actor_value_gradient_configuration(
            mode=getattr(config, "actor_value_gradient_mode", "scalar"),
            ppo_objective=config.ppo_objective,
            trainable_scope=getattr(
                config,
                "trainable_scope",
                "last_block_heads",
            ),
            value_trunk_gradient_scale=value_trunk_gradient_scale,
            audit_only=False,
        )
    )
    if config.ppo_objective == "constrained":
        if value_trunk_gradient_scale != 1.0:
            raise ValueError(
                "Value-trunk gradient scaling currently requires standard PPO"
            )
        if objective_state is None:
            raise ValueError("Constrained PPO requires objective state")
        return ppo_update_constrained(
            model,
            reference_model,
            optimizer,
            transitions,
            model_config,
            config,
            device,
            update,
            objective_state,
        )
    if config.ppo_objective != "standard":
        raise ValueError(f"Unsupported PPO objective: {config.ppo_objective!r}")
    # PPO uses dropout-free forwards so stored and current action distributions match.
    model.eval()
    reference_model.eval()
    schedule_progress = phase_schedule_progress(config, update)
    if config.learning_rate_schedule == "constant":
        learning_rate_factor = 1.0
    elif config.learning_rate_schedule == "cosine":
        learning_rate_factor = 0.10 + 0.90 * 0.5 * (
            1.0 + math.cos(math.pi * schedule_progress)
        )
    else:
        raise ValueError(
            f"Unsupported learning-rate schedule: "
            f"{config.learning_rate_schedule!r}"
        )
    optimizer.param_groups[0]["lr"] = (
        config.learning_rate * learning_rate_factor
    )
    optimizer.param_groups[1]["lr"] = (
        config.value_learning_rate * learning_rate_factor
    )
    advantages, advantage_audit = normalize_rollout_advantages(
        transitions,
        config.advantage_normalization,
    )
    opponent_groups = [
        transition_opponent_group(transition)
        for transition in transitions
    ]
    actor_multipliers, actor_reduction_audit = (
        actor_reduction_multipliers(
            transitions,
            config.actor_reduction,
            opponent_quotas=getattr(config, "opponent_base_quotas", {}),
            expected_episode_count=getattr(
                config,
                "games_per_update",
                None,
            ),
        )
    )
    outcomes = torch.tensor(
        [float(t["outcome_target"]) for t in transitions],
        dtype=torch.float32,
    )
    old_log_probs = torch.tensor(
        [float(t["old_log_prob"]) for t in transitions],
        dtype=torch.float32,
    )
    action_counts = torch.tensor(
        [int(t["action_count"]) for t in transitions],
        dtype=torch.long,
    )
    action_sequences = torch.full(
        (len(transitions), MAX_ACTION_COUNT),
        -1,
        dtype=torch.long,
    )
    for row_index, transition in enumerate(transitions):
        action = transition["action"]
        action_sequences[row_index, : len(action)] = torch.tensor(action)
    rollout_batch = collate_features_cpu(
        [transition["feature"] for transition in transitions],
        model_config,
    )

    stats: Counter[str] = Counter()
    started = time.time()
    kl_coefficient = bc_kl_coefficient(config, update)
    actor_parameters = list(optimizer.param_groups[0]["params"])
    value_parameters = list(optimizer.param_groups[1]["params"])
    trainable_parameters = actor_parameters + value_parameters
    optimizer_steps = 0
    early_stop = False
    actor_value_pcgrad_stats: Counter[str] = Counter()
    min_projected_value_dot_actor: float | None = None
    min_final_dot_actor_margin: float | None = None
    for epoch in range(config.ppo_epochs):
        permutation = torch.randperm(len(transitions))
        epoch_approx_kl = []
        for start in range(0, len(transitions), config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            index_list = indices.tolist()
            batch = {
                key: value.index_select(0, indices).to(
                    device,
                    non_blocking=True,
                )
                for key, value in rollout_batch.items()
            }
            mb_actions = action_sequences[indices].to(device)
            mb_counts = action_counts[indices].to(device)
            mb_old_log_probs = old_log_probs[indices].to(device)
            mb_advantages = advantages[indices].to(device)
            mb_actor_multipliers = actor_multipliers[indices].to(device)
            mb_outcomes = outcomes[indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model_forward(
                model,
                batch,
                device,
                value_trunk_gradient_scale=value_trunk_gradient_scale,
            )
            new_log_probs, entropy = ordered_action_log_prob_entropy(
                outputs,
                batch,
                mb_actions,
                mb_counts,
                temperature=config.policy_temperature,
            )
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = log_ratio.exp()
            unclipped = ratio * mb_advantages
            clipped = ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * mb_advantages
            surrogate = torch.minimum(unclipped, clipped)
            policy_loss = standard_actor_policy_loss(
                surrogate,
                mb_actor_multipliers,
                config.actor_reduction,
            )

            value_loss = F.binary_cross_entropy_with_logits(
                outputs["value_logits"].float(),
                mb_outcomes,
            )

            with torch.no_grad():
                reference_outputs = model_forward(reference_model, batch, device)
            anchor_kl = reference_policy_kl(
                outputs,
                reference_outputs,
                batch,
                mb_actions,
                mb_counts,
                temperature=config.policy_temperature,
            ).mean()
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                + kl_coefficient * anchor_kl
                - config.entropy_coefficient * entropy_mean
            )
            actor_value_adjustments: list[torch.Tensor | None] | None = None
            actor_value_audit: dict[str, float | int | bool] | None = None
            if actor_value_gradient_mode == "actor_priority_value_pcgrad":
                actor_side_loss = (
                    policy_loss
                    + kl_coefficient * anchor_kl
                    - config.entropy_coefficient * entropy_mean
                )
                value_side_loss = config.value_coefficient * value_loss
                actor_value_adjustments, actor_value_audit = (
                    actor_priority_value_pcgrad_adjustment(
                        actor_side_loss,
                        value_side_loss,
                        actor_parameters,
                    )
                )
                if not actor_value_audit["finite"]:
                    raise FloatingPointError(
                        "Non-finite actor/value PCGrad audit"
                    )
                if int(actor_value_audit["shared_parameter_count"]) <= 0:
                    raise RuntimeError(
                        "actor_priority_value_pcgrad found no shared "
                        "actor/value parameters"
                    )
            loss.backward()
            if (
                actor_value_adjustments is not None
                and actor_value_audit is not None
            ):
                for parameter, adjustment in zip(
                    actor_parameters,
                    actor_value_adjustments,
                ):
                    if adjustment is None:
                        continue
                    if parameter.grad is None:
                        parameter.grad = adjustment.clone()
                    else:
                        parameter.grad.add_(
                            adjustment.to(
                                device=parameter.grad.device,
                                dtype=parameter.grad.dtype,
                            )
                        )
                actor_value_pcgrad_stats["batches"] += 1
                actor_value_pcgrad_stats["conflict_batches"] += int(
                    actor_value_audit["conflict"]
                )
                for key in (
                    "pre_dot",
                    "pre_cosine",
                    "actor_norm",
                    "value_norm",
                    "projection_coefficient",
                    "adjustment_norm",
                ):
                    actor_value_pcgrad_stats[key] += float(
                        actor_value_audit[key]
                    )
                actor_value_pcgrad_stats["shared_tensor_count"] = int(
                    actor_value_audit["shared_tensor_count"]
                )
                actor_value_pcgrad_stats["shared_parameter_count"] = int(
                    actor_value_audit["shared_parameter_count"]
                )
                actor_value_pcgrad_stats["actor_only_tensor_count"] = int(
                    actor_value_audit["actor_only_tensor_count"]
                )
                actor_value_pcgrad_stats["actor_only_parameter_count"] = int(
                    actor_value_audit["actor_only_parameter_count"]
                )
                projected_dot = float(
                    actor_value_audit["projected_value_dot_actor"]
                )
                final_margin = float(
                    actor_value_audit["final_dot_actor_margin"]
                )
                min_projected_value_dot_actor = (
                    projected_dot
                    if min_projected_value_dot_actor is None
                    else min(
                        min_projected_value_dot_actor,
                        projected_dot,
                    )
                )
                min_final_dot_actor_margin = (
                    final_margin
                    if min_final_dot_actor_margin is None
                    else min(min_final_dot_actor_margin, final_margin)
                )
            actor_grad_norm = gradient_l2_norm(actor_parameters)
            value_grad_norm = gradient_l2_norm(value_parameters)
            grad_norm = nn.utils.clip_grad_norm_(
                trainable_parameters,
                config.max_grad_norm,
            )
            optimizer.step()
            optimizer_steps += 1

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_ratio
                ).float().mean()
            batch_rows = len(index_list)
            stats["rows"] += batch_rows
            stats["loss"] += float(loss.detach()) * batch_rows
            stats["policy_loss"] += float(policy_loss.detach()) * batch_rows
            stats["value_loss"] += float(value_loss.detach()) * batch_rows
            stats["entropy"] += float(entropy_mean.detach()) * batch_rows
            stats["bc_anchor_kl"] += float(anchor_kl.detach()) * batch_rows
            stats["approx_kl"] += float(approx_kl.detach()) * batch_rows
            stats["clip_fraction"] += float(clip_fraction.detach()) * batch_rows
            stats["grad_norm"] += float(grad_norm.detach()) * batch_rows
            stats["actor_grad_norm"] += (
                float(actor_grad_norm.detach()) * batch_rows
            )
            stats["value_grad_norm"] += (
                float(value_grad_norm.detach()) * batch_rows
            )
            epoch_approx_kl.append(float(approx_kl.detach()))

        if epoch_approx_kl and sum(epoch_approx_kl) / len(epoch_approx_kl) > (
            config.target_kl * 1.5
        ):
            early_stop = True
            break

    rows = stats["rows"]
    result = {
        key: value / max(rows, 1)
        for key, value in stats.items()
        if key != "rows"
    }
    result.update(
        {
            "rows": rows,
            "transitions": len(transitions),
            "optimizer_steps": optimizer_steps,
            "epochs_completed": epoch + 1,
            "early_stop": early_stop,
            "bc_kl_coefficient": kl_coefficient,
            "schedule_progress": schedule_progress,
            "phase_step": update - config.schedule_start_update + 1,
            "learning_rate_factor": learning_rate_factor,
            "actor_learning_rate": optimizer.param_groups[0]["lr"],
            "value_learning_rate": optimizer.param_groups[1]["lr"],
            "advantage_mean_raw": float(
                torch.tensor([float(t["advantage"]) for t in transitions]).mean()
            ),
            "advantage_std_raw": float(
                torch.tensor([float(t["advantage"]) for t in transitions]).std()
            ),
            "advantage_normalization": advantage_audit,
            "outcome_mean": float(outcomes.mean()),
            "actor_reduction": actor_reduction_audit,
            "value_trunk_gradient": value_trunk_gradient_audit(
                value_trunk_gradient_scale
            ),
            "actor_value_gradient": (
                {
                    **actor_value_gradient_manifest(
                        actor_value_gradient_mode
                    ),
                    "batches": int(actor_value_pcgrad_stats["batches"]),
                    "conflict_batches": int(
                        actor_value_pcgrad_stats["conflict_batches"]
                    ),
                    "conflict_fraction": (
                        actor_value_pcgrad_stats["conflict_batches"]
                        / max(actor_value_pcgrad_stats["batches"], 1)
                    ),
                    **{
                        f"mean_{key}": (
                            float(actor_value_pcgrad_stats[key])
                            / max(actor_value_pcgrad_stats["batches"], 1)
                        )
                        for key in (
                            "pre_dot",
                            "pre_cosine",
                            "actor_norm",
                            "value_norm",
                            "projection_coefficient",
                            "adjustment_norm",
                        )
                    },
                    "shared_tensor_count": int(
                        actor_value_pcgrad_stats["shared_tensor_count"]
                    ),
                    "shared_parameter_count": int(
                        actor_value_pcgrad_stats["shared_parameter_count"]
                    ),
                    "actor_only_tensor_count": int(
                        actor_value_pcgrad_stats["actor_only_tensor_count"]
                    ),
                    "actor_only_parameter_count": int(
                        actor_value_pcgrad_stats[
                            "actor_only_parameter_count"
                        ]
                    ),
                    "min_projected_value_dot_actor": (
                        min_projected_value_dot_actor
                    ),
                    "min_final_dot_actor_margin": (
                        min_final_dot_actor_margin
                    ),
                }
                if actor_value_gradient_mode
                == "actor_priority_value_pcgrad"
                else actor_value_gradient_manifest("scalar")
            ),
            "seconds": time.time() - started,
        }
    )
    if config.actor_reduction == "quota_group_mean":
        final_policy_shift = evaluate_constrained_policy_shift(
            model,
            rollout_batch,
            action_sequences,
            action_counts,
            old_log_probs,
            advantages,
            opponent_groups,
            config,
            device,
        )
        result["final_policy_shift_by_opponent"] = final_policy_shift
        result["max_final_opponent_approx_kl"] = max(
            float(row["approx_kl"])
            for row in final_policy_shift.values()
        )
    return result


@torch.no_grad()
def evaluate_head_to_head(
    current_model: EntityOptionPolicy,
    opponent_model: EntityOptionPolicy,
    deck: list[int],
    model_config: dict[str, Any],
    device: torch.device,
    games_target: int,
    environments: int,
    max_game_decisions: int,
    current_canonical_order: bool = False,
    opponent_canonical_order: bool = True,
    opponent_deck: list[int] | None = None,
) -> dict[str, Any]:
    current_model.eval()
    opponent_model.eval()
    learner_deck_hash = compute_deck_hash(deck)
    resolved_opponent_deck = deck if opponent_deck is None else opponent_deck
    opponent_deck_hash = compute_deck_hash(resolved_opponent_deck)
    started = time.time()
    stats: Counter[str] = Counter()
    games: list[tuple[RawBattle, int, int]] = []

    # Randomize which member of each balanced seat pair contains the learner
    # without consuming the global training RNG. This preserves a 50/50 seat
    # allocation for even-sized evaluations and is shared across opponents.
    seat_rng = random.Random(0)
    seat_pair: list[int] = []

    def next_learner_seat() -> int:
        nonlocal seat_pair
        if not seat_pair:
            seat_pair = [0, 1]
            seat_rng.shuffle(seat_pair)
        return seat_pair.pop()

    def start_evaluation_game() -> tuple[RawBattle, int, int]:
        learner_seat = next_learner_seat()
        seat_decks = (
            (deck, resolved_opponent_deck)
            if learner_seat == 0
            else (resolved_opponent_deck, deck)
        )
        return RawBattle(seat_decks[0], seat_decks[1]), learner_seat, 0

    for _ in range(min(environments, games_target)):
        games.append(start_evaluation_game())
    stats["max_active_games"] = len(games)
    valid = 0
    while games:
        groups: dict[int, list[tuple[int, RawBattle, int, dict[str, Any]]]] = defaultdict(list)
        invalid_indices: list[int] = []
        for game_index, (battle, current_seat, decisions) in enumerate(games):
            observation = battle.observation
            if battle.result != -1:
                continue
            select = observation.get("select")
            current = observation.get("current") or {}
            if not isinstance(select, dict):
                invalid_indices.append(game_index)
                continue
            options = select.get("option")
            minimum = int(select.get("minCount", 0) or 0)
            maximum = int(select.get("maxCount", 0) or 0)
            if (
                not isinstance(options, list)
                or not options
                or minimum < 0
                or maximum < minimum
                or maximum > len(options)
                or maximum > MAX_ACTION_COUNT
            ):
                invalid_indices.append(game_index)
                continue
            seat = int(current.get("yourIndex", 0) or 0)
            feature = live_feature(
                observation,
                model_config,
                (
                    learner_deck_hash
                    if seat == current_seat
                    else opponent_deck_hash
                ),
            )
            if feature is None:
                invalid_indices.append(game_index)
                continue
            policy_index = -1 if seat == current_seat else 0
            groups[policy_index].append(
                (game_index, battle, current_seat, feature)
            )

        actions_by_index: dict[int, list[int]] = {}
        for policy_index, items in groups.items():
            batch = collate_features(
                [item[3] for item in items],
                model_config,
                device,
            )
            model = current_model if policy_index == -1 else opponent_model
            outputs = model_forward(model, batch, device)
            actions, _, _, _ = sample_ordered_actions(
                outputs,
                batch,
                deterministic=True,
                canonicalize_order=(
                    current_canonical_order
                    if policy_index == -1
                    else opponent_canonical_order
                ),
            )
            for item, action in zip(items, actions):
                actions_by_index[item[0]] = action

        finished_indices = set(invalid_indices)
        stats["invalid_games"] += len(invalid_indices)
        for game_index, (battle, current_seat, decisions) in enumerate(games):
            if game_index in finished_indices or game_index not in actions_by_index:
                continue
            _, error = battle.step(actions_by_index[game_index])
            decisions += 1
            games[game_index] = (battle, current_seat, decisions)
            stats["decisions"] += 1
            if error:
                stats["invalid_games"] += 1
                finished_indices.add(game_index)
                continue
            result = battle.result
            if result != -1:
                valid += 1
                stats["valid_games"] += 1
                if result == current_seat:
                    stats["wins"] += 1
                elif result == 2:
                    stats["draws"] += 1
                else:
                    stats["losses"] += 1
                finished_indices.add(game_index)
                continue
            if decisions >= max_game_decisions:
                stats["invalid_games"] += 1
                finished_indices.add(game_index)

        for game_index in sorted(finished_indices, reverse=True):
            battle, _, _ = games[game_index]
            battle.close()
            games.pop(game_index)
        desired_active_games = min(
            environments,
            games_target - valid,
        )
        while len(games) < desired_active_games:
            games.append(start_evaluation_game())
        stats["max_active_games"] = max(
            stats["max_active_games"],
            len(games),
        )
        if stats["invalid_games"] > games_target * 4:
            for battle, _, _ in games:
                battle.close()
            raise RuntimeError(
                "Evaluation exceeded the invalid-game budget: "
                f"valid={valid} invalid={stats['invalid_games']}"
            )

    valid_games = stats["valid_games"]
    decisive = stats["wins"] + stats["losses"]
    return {
        "valid_games": valid_games,
        "wins": stats["wins"],
        "losses": stats["losses"],
        "draws": stats["draws"],
        "invalid_games": stats["invalid_games"],
        "win_rate": stats["wins"] / max(valid_games, 1),
        "decisive_win_rate": stats["wins"] / max(decisive, 1),
        "mean_decisions": stats["decisions"] / max(valid_games, 1),
        "max_active_games": stats["max_active_games"],
        "seconds": time.time() - started,
    }


def evaluate_permanent_opponents(
    current_model: EntityOptionPolicy,
    opponents: list[FrozenOpponent],
    deck: list[int],
    model_config: dict[str, Any],
    device: torch.device,
    games_target: int,
    environments: int,
    max_game_decisions: int,
    selection_aggregation: str = "mean",
) -> tuple[dict[str, dict[str, Any]], float]:
    permanent_opponents = [
        opponent for opponent in opponents if opponent.permanent
    ]
    if not permanent_opponents:
        raise ValueError(
            "Multi-opponent evaluation requires at least one permanent opponent"
        )
    evaluation_by_opponent: dict[str, dict[str, Any]] = {}
    for opponent in permanent_opponents:
        evaluation = evaluate_head_to_head(
            current_model,
            opponent.model,
            deck,
            model_config,
            device,
            games_target,
            environments,
            max_game_decisions,
            # The candidate follows the PPO submission runtime's ordered
            # Plackett-Luce greedy action. BC opponents retain canonical set
            # order, while PPO incumbents retain their learned action order.
            current_canonical_order=False,
            opponent_canonical_order=opponent.canonical_order,
            opponent_deck=opponent.deck,
        )
        evaluation_by_opponent[opponent.name] = {
            **evaluation,
            "opponent_deck_hash": opponent.deck_hash,
        }
    win_rates = [
        evaluation["win_rate"]
        for evaluation in evaluation_by_opponent.values()
    ]
    if selection_aggregation == "mean":
        selection_score = sum(win_rates) / len(win_rates)
    elif selection_aggregation == "min":
        selection_score = min(win_rates)
    else:
        raise ValueError(
            f"Unsupported selection aggregation: {selection_aggregation}"
        )
    return evaluation_by_opponent, selection_score


def frozen_copy(
    model: EntityOptionPolicy,
    name: str,
    device: torch.device,
    deck: list[int],
    deck_hash: str,
    permanent: bool,
    canonical_order: bool,
) -> FrozenOpponent:
    frozen = copy.deepcopy(model).to(device)
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return FrozenOpponent(
        name=name,
        model=frozen,
        deck=list(deck),
        deck_hash=deck_hash,
        permanent=permanent,
        canonical_order=canonical_order,
    )


def trim_opponent_pool(
    opponents: list[FrozenOpponent],
    max_pool_size: int,
) -> None:
    while len(opponents) > max_pool_size:
        removable_index = next(
            (
                index
                for index, opponent in enumerate(opponents)
                if not opponent.permanent
            ),
            None,
        )
        if removable_index is None:
            # Permanent anchors may intentionally exceed max_pool_size.
            return
        opponents.pop(removable_index)


def save_ppo_checkpoint(
    path: Path,
    model: EntityOptionPolicy,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    model_config: dict[str, Any],
    update: int,
    metrics: dict[str, Any],
    replay_optimizer: torch.optim.Optimizer | None = None,
    opponent_quota_controller: OpponentQuotaController | None = None,
    objective_state: ConstrainedObjectiveState | None = None,
) -> None:
    parameter_names_by_id = {
        id(parameter): name
        for name, parameter in model.named_parameters()
    }
    optimizer_parameter_names: dict[str, list[str]] = {}
    for group_name, group in zip(
        ("actor", "value"),
        optimizer.param_groups,
    ):
        names = [
            parameter_names_by_id[id(parameter)]
            for parameter in group["params"]
            if id(parameter) in parameter_names_by_id
        ]
        if len(names) != len(group["params"]):
            raise RuntimeError(
                f"Could not name every {group_name} optimizer parameter"
            )
        optimizer_parameter_names[group_name] = names
    payload = {
        "feature_version": PPO_FEATURE_VERSION,
        "bc_feature_version": BC_FEATURE_VERSION,
        "config": asdict(config),
        "model_config": model_config,
        "learner_deck_hash": compute_deck_hash(
            read_deck(Path(config.deck))
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_parameter_names": optimizer_parameter_names,
        "bc_replay_optimizer_state_dict": (
            replay_optimizer.state_dict()
            if replay_optimizer is not None
            else None
        ),
        "update": update,
        "metrics": metrics,
        "opponent_quota_state": (
            opponent_quota_controller.state_dict()
            if opponent_quota_controller is not None
            else None
        ),
        "reward": {"win": 1.0, "loss": 0.0, "draw": 0.0},
        "value_trunk_gradient": value_trunk_gradient_audit(
            config.value_trunk_gradient_scale
        ),
        "actor_value_gradient": actor_value_gradient_manifest(
            config.actor_value_gradient_mode,
            audit_only=config.actor_value_gradient_audit_only,
        ),
        "action_distribution": (
            "masked cardinality categorical + ordered Plackett-Luce "
            "without replacement"
        ),
    }
    if objective_state is not None:
        payload["ppo_objective_state"] = objective_state.state_dict()
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=Path("artifacts/bc_marnie_luca_orbit_v5/best.pt"),
    )
    parser.add_argument(
        "--kl-reference-checkpoint",
        type=Path,
        help=(
            "Optional compatible BC/PPO checkpoint used only as the policy-KL "
            "reference. By default --bc-checkpoint is also the KL reference."
        ),
    )
    parser.add_argument(
        "--deck",
        type=Path,
        default=Path("data/decks/marnie_grimmsnarl_froslass_luca.csv"),
    )
    parser.add_argument(
        "--extra-opponent",
        action="append",
        nargs=2,
        type=Path,
        metavar=("CHECKPOINT", "DECK"),
        default=[],
        help=(
            "Add a permanent frozen opponent from a compatible BC or PPO "
            "checkpoint and its 60-card deck; repeat for multiple opponents."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ppo_marnie_v1"))
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--environments", type=int, default=32)
    parser.add_argument("--games-per-update", type=int, default=96)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--value-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--advantage-normalization",
        choices=ADVANTAGE_NORMALIZATION_MODES,
        default="global",
        help=(
            "Normalize advantages over the full rollout globally (legacy) "
            "or independently within each frozen-opponent group."
        ),
    )
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument(
        "--value-trunk-gradient-scale",
        type=float,
        default=1.0,
        help=(
            "Scale only the standard-PPO value-loss gradient entering the "
            "shared trunk. The value head keeps its full gradient and forward "
            "values are unchanged; 1.0 preserves prior behavior and 0.0 "
            "trains the value head on detached shared features."
        ),
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature used by the learner rollout distribution, PPO "
            "likelihood ratio, entropy, and reference KL. Values below one "
            "make training behavior closer to greedy deployment."
        ),
    )
    parser.add_argument(
        "--trainable-scope",
        choices=TRAINABLE_SCOPES,
        default="full",
        help=(
            "PPO parameters to update: the full actor-critic, actor/value "
            "heads only, or the final one/two Transformer blocks plus heads."
        ),
    )
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("cosine", "constant"),
        default="cosine",
        help="Actor/value learning-rate schedule within the configured phase.",
    )
    parser.add_argument(
        "--schedule-start-update",
        type=int,
        default=1,
        help=(
            "Global update number treated as progress zero for LR and KL "
            "schedules. Set this to resume_update+1 for a fresh continuation "
            "phase."
        ),
    )
    parser.add_argument("--bc-kl-start", type=float, default=0.05)
    parser.add_argument("--bc-kl-end", type=float, default=0.005)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--league-probability", type=float, default=0.20)
    parser.add_argument(
        "--opponent-sampling",
        choices=("per_update", "per_game"),
        default="per_update",
        help="Sample one frozen opponent per update (v1) or independently per game (v2).",
    )
    parser.add_argument(
        "--bc-opponent-probability",
        type=float,
        help=(
            "Probability of choosing the base BC opponent from the frozen "
            "pool; omit for uniform opponent sampling."
        ),
    )
    parser.add_argument(
        "--opponent-weight",
        action="append",
        nargs=2,
        metavar=("NAME", "WEIGHT"),
        default=[],
        help=(
            "Assign an explicit non-negative sampling weight to a loaded "
            "permanent opponent name; repeat for multiple opponents. This is "
            "mutually exclusive with --bc-opponent-probability."
        ),
    )
    parser.add_argument(
        "--history-opponent-weight",
        type=float,
        default=0.0,
        help=(
            "Total explicit sampling weight shared equally by non-permanent "
            "historical snapshots."
        ),
    )
    parser.add_argument(
        "--opponent-quota-mode",
        choices=("legacy", "fixed", "adaptive"),
        default="legacy",
        help=(
            "Use legacy random sampling (default), a fixed exact per-update "
            "quota, or an uncertainty-adaptive exact quota."
        ),
    )
    parser.add_argument(
        "--opponent-base-quota",
        action="append",
        nargs=2,
        metavar=("NAME", "GAMES"),
        default=[],
        help=(
            "Minimum valid games per update for a permanent opponent. Repeat "
            "per opponent. In fixed mode these quotas must sum to "
            "--games-per-update."
        ),
    )
    parser.add_argument(
        "--opponent-cap",
        action="append",
        nargs=2,
        metavar=("NAME", "GAMES"),
        default=[],
        help=(
            "Maximum valid games per update for an adaptive-quota permanent "
            "opponent; must cover every permanent opponent."
        ),
    )
    parser.add_argument(
        "--opponent-audit",
        action="append",
        nargs=4,
        metavar=("NAME", "WINS", "LOSSES", "DRAWS"),
        default=[],
        help=(
            "Shared audit W/L/D seed from the learner perspective for one "
            "adaptive-quota permanent opponent; repeat for all permanent "
            "opponents."
        ),
    )
    parser.add_argument(
        "--opponent-quota-refresh-updates",
        type=int,
        default=3,
        help="Recompute adaptive dynamic quotas after this many updates.",
    )
    parser.add_argument(
        "--opponent-quota-seat-balance",
        action="store_true",
        help=(
            "Assign deterministic near-equal learner seats within every exact "
            "opponent quota and preserve the same seat for invalid replacements."
        ),
    )
    parser.add_argument(
        "--ppo-objective",
        choices=PPO_OBJECTIVES,
        default="standard",
        help=(
            "Use legacy transition-mean PPO or the opponent-grouped "
            "primal-dual constrained objective."
        ),
    )
    parser.add_argument(
        "--actor-reduction",
        choices=ACTOR_REDUCTIONS,
        default="transition_mean",
        help=(
            "Reduce the standard PPO actor surrogate over transitions, fixed "
            "opponent quota groups, or equally weighted frozen game_uid "
            "episodes. Other losses remain transition means."
        ),
    )
    parser.add_argument(
        "--actor-reduction-audit-only",
        action="store_true",
        help=(
            "Collect exactly one frozen rollout, compare transition-mean and "
            "the configured non-baseline actor gradients, write "
            "actor_reduction_audit.json, "
            "and exit without an optimizer step or checkpoint."
        ),
    )
    parser.add_argument(
        "--actor-value-gradient-mode",
        choices=ACTOR_VALUE_GRADIENT_MODES,
        default="scalar",
        help=(
            "Keep the standard scalar actor/value gradient, or project only "
            "a conflicting critic gradient off the actor gradient on shared "
            "trainable parameters."
        ),
    )
    parser.add_argument(
        "--actor-value-gradient-audit-only",
        action="store_true",
        help=(
            "Collect one frozen rollout, audit actor-priority value PCGrad "
            "over sequential minibatches, write "
            "actor_value_gradient_audit.json, and exit without backward, "
            "optimizer/replay steps, evaluation, or checkpoints."
        ),
    )
    parser.add_argument(
        "--constrained-gradient-mode",
        choices=CONSTRAINED_GRADIENT_MODES,
        default="scalar",
        help=(
            "Combine core opponent policy losses as a scalar, or apply "
            "symmetric or guard-priority PCGrad before the optimizer step."
        ),
    )
    parser.add_argument("--primary-opponent-name")
    parser.add_argument("--guard-opponent-name")
    parser.add_argument("--primary-policy-weight", type=float, default=1.0)
    parser.add_argument("--guard-policy-weight", type=float, default=0.25)
    parser.add_argument("--auxiliary-policy-weight", type=float, default=0.10)
    parser.add_argument("--guard-surrogate-floor", type=float, default=-0.002)
    parser.add_argument("--constraint-dual-initial", type=float, default=1.0)
    parser.add_argument("--constraint-dual-lr", type=float, default=0.05)
    parser.add_argument("--constraint-dual-max", type=float, default=10.0)
    parser.add_argument(
        "--opponent-loss-weight",
        action="append",
        nargs=2,
        metavar=("NAME", "WEIGHT"),
        default=[],
        help=(
            "Fixed group weight for constrained value, entropy, anchor-KL, "
            "and approximate-KL reductions; repeat for every opponent."
        ),
    )
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--max-pool-size", type=int, default=8)
    parser.add_argument(
        "--bc-replay-data",
        type=Path,
        help="Filtered BC decision ZIP used for auxiliary expert replay.",
    )
    parser.add_argument(
        "--bc-replay-split",
        choices=("train", "valid", "test"),
        default="train",
        help=(
            "Archive split used for auxiliary expert replay. Keep the default "
            "train split unless a later dated validation split is deliberately "
            "promoted into final training while a separate test split remains "
            "untouched."
        ),
    )
    parser.add_argument("--bc-replay-batches", type=int, default=0)
    parser.add_argument("--bc-replay-batch-size", type=int, default=256)
    parser.add_argument("--bc-replay-workers", type=int, default=8)
    parser.add_argument("--bc-replay-steps", type=int, default=0)
    parser.add_argument("--bc-replay-lr-scale", type=float, default=0.25)
    parser.add_argument(
        "--bc-replay-loss",
        choices=("set", "ordered", "hybrid_ordered"),
        default="set",
        help=(
            "Legacy unordered set loss, full ordered Plackett-Luce sequence "
            "loss, or ordered sequence loss only for context 34."
        ),
    )
    parser.add_argument(
        "--bc-replay-order-context-weight",
        type=float,
        default=1.0,
        help=(
            "Relative replay row weight for context 34. Keep 1.0 unless an "
            "explicit context-focused ablation is intended."
        ),
    )
    parser.add_argument(
        "--bc-replay-non-context34-fixed-multi-action-order-weight",
        type=float,
        default=1.0,
        help=(
            "Relative ordered-selection row weight for fixed-count, "
            "non-context-34 rows with more than one expert action. This does "
            "not alter count loss or the context-34 row weight."
        ),
    )
    parser.add_argument(
        "--bc-replay-context34-rows-per-batch",
        type=int,
        default=0,
        help=(
            "Exact number of context-34 rows in every cached replay batch. "
            "Zero preserves the legacy unstratified replay batches."
        ),
    )
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=128)
    parser.add_argument(
        "--eval-all-permanent-opponents",
        action="store_true",
        help=(
            "Evaluate every permanent frozen opponent and select checkpoints "
            "using --selection-aggregation. By default only the base BC mirror "
            "is evaluated."
        ),
    )
    parser.add_argument(
        "--selection-aggregation",
        choices=("mean", "min"),
        default="mean",
        help=(
            "Aggregate permanent-opponent evaluation win rates for checkpoint "
            "selection. Use min to require progress against the hardest anchor."
        ),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--max-game-decisions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--resume-learner-weights",
        choices=RESUME_LEARNER_WEIGHT_SOURCES,
        default="resume",
        help=(
            "Full-model learner weights to use with --resume. 'resume' keeps "
            "the legacy behavior and restores the PPO checkpoint weights; "
            "'bc' keeps the full --bc-checkpoint initialization while still "
            "using resume metadata such as the update number. The 'bc' mode "
            "requires --reset-optimizer-on-resume."
        ),
    )
    parser.add_argument(
        "--reset-optimizer-on-resume",
        action="store_true",
        help=(
            "Initialize fresh PPO/replay optimizer state for a changed "
            "continuation objective or --resume-learner-weights bc."
        ),
    )
    parser.add_argument(
        "--reset-opponent-quota-on-resume",
        action="store_true",
        help=(
            "Validate but discard an exact-quota checkpoint's sampler state "
            "and initialize the quota controller from the current CLI. "
            "Requires --resume and --reset-optimizer-on-resume."
        ),
    )
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.reset_optimizer_on_resume and args.resume is None:
        raise ValueError("--reset-optimizer-on-resume requires --resume")
    validate_resume_learner_weight_selection(
        args.resume,
        args.resume_learner_weights,
        args.reset_optimizer_on_resume,
    )
    if args.reset_opponent_quota_on_resume and args.resume is None:
        raise ValueError("--reset-opponent-quota-on-resume requires --resume")
    if (
        args.reset_opponent_quota_on_resume
        and not args.reset_optimizer_on_resume
    ):
        raise ValueError(
            "--reset-opponent-quota-on-resume requires "
            "--reset-optimizer-on-resume"
        )
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(args.bc_checkpoint)
    if (
        args.kl_reference_checkpoint is not None
        and not args.kl_reference_checkpoint.is_file()
    ):
        raise FileNotFoundError(args.kl_reference_checkpoint)
    if not args.deck.is_file():
        raise FileNotFoundError(args.deck)
    extra_opponent_specs: list[dict[str, str]] = []
    extra_opponent_names: set[str] = set()
    for checkpoint_path, deck_path in args.extra_opponent:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if not deck_path.is_file():
            raise FileNotFoundError(deck_path)
        name = f"{checkpoint_path.stem}@{deck_path.stem}"
        if name in extra_opponent_names:
            raise ValueError(f"Duplicate --extra-opponent name: {name}")
        extra_opponent_names.add(name)
        extra_opponent_specs.append(
            {
                "name": name,
                "checkpoint": str(checkpoint_path.resolve()),
                "deck": str(deck_path.resolve()),
            }
        )
    opponent_weights: dict[str, float] = {}
    for name, raw_weight in args.opponent_weight:
        if name in opponent_weights:
            raise ValueError(f"Duplicate --opponent-weight name: {name}")
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(
                f"Invalid --opponent-weight value for {name!r}: {raw_weight!r}"
            ) from error
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"--opponent-weight for {name!r} must be finite and non-negative"
            )
        opponent_weights[name] = weight
    opponent_loss_weights: dict[str, float] = {}
    for name, raw_weight in args.opponent_loss_weight:
        if name in opponent_loss_weights:
            raise ValueError(f"Duplicate --opponent-loss-weight name: {name}")
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(
                f"Invalid --opponent-loss-weight value for {name!r}: "
                f"{raw_weight!r}"
            ) from error
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"--opponent-loss-weight for {name!r} must be finite and "
                "non-negative"
            )
        opponent_loss_weights[name] = weight
    opponent_base_quotas: dict[str, int] = {}
    for name, raw_games in args.opponent_base_quota:
        if name in opponent_base_quotas:
            raise ValueError(f"Duplicate --opponent-base-quota name: {name}")
        try:
            games = int(raw_games)
        except ValueError as error:
            raise ValueError(
                f"Invalid --opponent-base-quota for {name!r}: {raw_games!r}"
            ) from error
        if games < 0:
            raise ValueError("--opponent-base-quota must be non-negative")
        opponent_base_quotas[name] = games
    opponent_caps: dict[str, int] = {}
    for name, raw_games in args.opponent_cap:
        if name in opponent_caps:
            raise ValueError(f"Duplicate --opponent-cap name: {name}")
        try:
            games = int(raw_games)
        except ValueError as error:
            raise ValueError(
                f"Invalid --opponent-cap for {name!r}: {raw_games!r}"
            ) from error
        if games < 0:
            raise ValueError("--opponent-cap must be non-negative")
        opponent_caps[name] = games
    opponent_audit: dict[str, dict[str, int]] = {}
    for name, raw_wins, raw_losses, raw_draws in args.opponent_audit:
        if name in opponent_audit:
            raise ValueError(f"Duplicate --opponent-audit name: {name}")
        try:
            wins, losses, draws = (
                int(raw_wins),
                int(raw_losses),
                int(raw_draws),
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid --opponent-audit counts for {name!r}"
            ) from error
        if min(wins, losses, draws) < 0:
            raise ValueError("--opponent-audit counts must be non-negative")
        opponent_audit[name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }
    if args.updates < 1 or args.environments < 1 or args.games_per_update < 1:
        raise ValueError("updates/environments/games-per-update must be positive")
    if args.schedule_start_update < 1 or args.schedule_start_update > args.updates:
        raise ValueError(
            "--schedule-start-update must be between 1 and --updates"
        )
    if args.ppo_epochs < 1 or args.minibatch_size < 1:
        raise ValueError("--ppo-epochs/--minibatch-size must be positive")
    if not math.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not math.isfinite(args.gae_lambda) or not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("--gae-lambda must be in [0, 1]")
    validate_value_trunk_gradient_scale(args.value_trunk_gradient_scale)
    validate_actor_value_gradient_configuration(
        mode=args.actor_value_gradient_mode,
        ppo_objective=args.ppo_objective,
        trainable_scope=args.trainable_scope,
        value_trunk_gradient_scale=args.value_trunk_gradient_scale,
        audit_only=args.actor_value_gradient_audit_only,
        actor_reduction_audit_only=args.actor_reduction_audit_only,
    )
    optimizer_scalars = (
        args.learning_rate,
        args.value_learning_rate,
        args.clip_ratio,
        args.value_coefficient,
        args.entropy_coefficient,
        args.max_grad_norm,
        args.policy_temperature,
        args.target_kl,
    )
    if (
        any(not math.isfinite(value) for value in optimizer_scalars)
        or args.learning_rate <= 0.0
        or args.value_learning_rate <= 0.0
        or args.clip_ratio <= 0.0
        or args.value_coefficient < 0.0
        or args.entropy_coefficient < 0.0
        or args.max_grad_norm <= 0.0
        or args.policy_temperature <= 0.0
        or args.target_kl <= 0.0
    ):
        raise ValueError("Invalid PPO optimizer/loss configuration")
    if (
        not math.isfinite(args.bc_kl_start)
        or not math.isfinite(args.bc_kl_end)
        or (args.bc_kl_start == 0.0) != (args.bc_kl_end == 0.0)
        or args.bc_kl_start < 0.0
        or args.bc_kl_end < 0.0
    ):
        raise ValueError(
            "--bc-kl-start/--bc-kl-end must both be positive or both be zero"
        )
    if args.max_pool_size < 1:
        raise ValueError("--max-pool-size must be positive")
    if args.opponent_quota_refresh_updates < 1:
        raise ValueError("--opponent-quota-refresh-updates must be positive")
    if args.snapshot_interval < 1 or args.eval_interval < 1:
        raise ValueError("--snapshot-interval/--eval-interval must be positive")
    if args.eval_games < 1 or args.checkpoint_interval < 1:
        raise ValueError("--eval-games/--checkpoint-interval must be positive")
    if args.max_game_decisions < 1:
        raise ValueError("--max-game-decisions must be positive")
    if not 0.0 <= args.league_probability <= 1.0:
        raise ValueError("--league-probability must be in [0, 1]")
    if (
        args.bc_opponent_probability is not None
        and not 0.0 <= args.bc_opponent_probability <= 1.0
    ):
        raise ValueError("--bc-opponent-probability must be in [0, 1]")
    if not math.isfinite(args.history_opponent_weight) or (
        args.history_opponent_weight < 0.0
    ):
        raise ValueError("--history-opponent-weight must be finite and non-negative")
    if opponent_weights and args.bc_opponent_probability is not None:
        raise ValueError(
            "--opponent-weight is mutually exclusive with "
            "--bc-opponent-probability"
        )
    if args.history_opponent_weight > 0.0 and not opponent_weights:
        raise ValueError(
            "--history-opponent-weight requires at least one --opponent-weight"
        )
    quota_arguments_present = bool(
        opponent_base_quotas or opponent_caps or opponent_audit
    )
    if args.opponent_quota_mode == "legacy" and quota_arguments_present:
        raise ValueError(
            "Opponent quota arguments require --opponent-quota-mode "
            "fixed or adaptive"
        )
    if args.opponent_quota_mode != "legacy":
        if args.league_probability != 1.0:
            raise ValueError(
                "Exact opponent quotas require --league-probability 1.0"
            )
        if opponent_weights or args.bc_opponent_probability is not None:
            raise ValueError(
                "Exact opponent quotas are mutually exclusive with random "
                "opponent weights/probabilities"
            )
        if args.history_opponent_weight > 0.0:
            raise ValueError(
                "Exact opponent quotas are mutually exclusive with "
                "--history-opponent-weight"
            )
        base_total = sum(opponent_base_quotas.values())
        if args.opponent_quota_mode == "fixed":
            if base_total != args.games_per_update:
                raise ValueError(
                    "Fixed --opponent-base-quota values must sum to "
                    "--games-per-update"
                )
            if opponent_caps or opponent_audit:
                raise ValueError(
                    "Fixed opponent quotas do not use --opponent-cap or "
                    "--opponent-audit"
                )
        elif base_total >= args.games_per_update:
            raise ValueError(
                "Adaptive --opponent-base-quota values must sum below "
                "--games-per-update"
            )
    if (
        args.opponent_quota_seat_balance
        and args.opponent_quota_mode == "legacy"
    ):
        raise ValueError(
            "--opponent-quota-seat-balance requires fixed or adaptive "
            "exact opponent quotas"
        )
    permanent_opponent_names = {
        "bc",
        *(spec["name"] for spec in extra_opponent_specs),
    }
    if (
        args.actor_reduction in {"quota_group_mean", "episode_mean"}
        and args.ppo_objective != "standard"
    ):
        raise ValueError(
            "Weighted actor reductions require --ppo-objective standard"
        )
    if args.actor_reduction == "quota_group_mean":
        if args.advantage_normalization != "global":
            raise ValueError(
                "Quota-group actor reduction requires global advantages"
            )
        if args.opponent_quota_mode != "fixed":
            raise ValueError(
                "Quota-group actor reduction requires fixed opponent quotas"
            )
        if not args.opponent_quota_seat_balance:
            raise ValueError(
                "Quota-group actor reduction requires quota seat balancing"
            )
        if (
            set(opponent_base_quotas) != permanent_opponent_names
            or any(quota <= 0 for quota in opponent_base_quotas.values())
        ):
            raise ValueError(
                "Quota-group actor reduction requires a positive fixed quota "
                "for every permanent opponent"
            )
    if (
        args.actor_reduction_audit_only
        and args.actor_reduction == "transition_mean"
    ):
        raise ValueError(
            "--actor-reduction-audit-only requires "
            "--actor-reduction quota_group_mean or episode_mean"
        )
    if (
        args.actor_reduction_audit_only
        and args.advantage_normalization != "global"
    ):
        raise ValueError(
            "--actor-reduction-audit-only requires global advantages"
        )
    objective_scalars = (
        args.primary_policy_weight,
        args.guard_policy_weight,
        args.auxiliary_policy_weight,
        args.guard_surrogate_floor,
        args.constraint_dual_initial,
        args.constraint_dual_lr,
        args.constraint_dual_max,
    )
    if any(not math.isfinite(value) for value in objective_scalars):
        raise ValueError("Constrained PPO objective values must be finite")
    if (
        args.primary_policy_weight <= 0.0
        or args.guard_policy_weight < 0.0
        or args.auxiliary_policy_weight < 0.0
        or args.constraint_dual_lr < 0.0
        or args.constraint_dual_max <= 0.0
        or not (
            0.0
            <= args.constraint_dual_initial
            <= args.constraint_dual_max
        )
    ):
        raise ValueError("Invalid constrained PPO objective configuration")
    if args.ppo_objective == "standard":
        if (
            args.primary_opponent_name is not None
            or args.guard_opponent_name is not None
            or opponent_loss_weights
            or args.constrained_gradient_mode != "scalar"
        ):
            raise ValueError(
                "Opponent objective names/weights require "
                "--ppo-objective constrained"
            )
    else:
        if args.value_trunk_gradient_scale != 1.0:
            raise ValueError(
                "--value-trunk-gradient-scale currently requires "
                "--ppo-objective standard"
            )
        if args.opponent_quota_mode != "fixed":
            raise ValueError(
                "Constrained PPO requires --opponent-quota-mode fixed"
            )
        if not args.opponent_quota_seat_balance:
            raise ValueError(
                "Constrained PPO requires --opponent-quota-seat-balance"
            )
        if args.advantage_normalization != "per_opponent":
            raise ValueError(
                "Constrained PPO requires per-opponent advantage normalization"
            )
        if (
            not args.primary_opponent_name
            or not args.guard_opponent_name
            or args.primary_opponent_name == args.guard_opponent_name
        ):
            raise ValueError(
                "Constrained PPO requires distinct primary and guard opponents"
            )
        core_names = {
            args.primary_opponent_name,
            args.guard_opponent_name,
        }
        if not core_names <= permanent_opponent_names:
            raise ValueError(
                "Constrained PPO primary/guard names must match loaded "
                "permanent opponents"
            )
        if set(opponent_loss_weights) != permanent_opponent_names:
            raise ValueError(
                "Constrained --opponent-loss-weight names must exactly cover "
                "all permanent opponents"
            )
        nonpositive_loss_weights = {
            name
            for name, weight in opponent_loss_weights.items()
            if weight <= 0.0
        }
        if nonpositive_loss_weights:
            raise ValueError(
                "Constrained opponent loss weights must all be positive: "
                + ", ".join(sorted(nonpositive_loss_weights))
            )
        if set(opponent_base_quotas) != permanent_names:
            raise ValueError(
                "Constrained opponent quotas must exactly cover all "
                "permanent opponents"
            )
        missing_rollout_groups = {
            name
            for name in permanent_names
            if opponent_base_quotas[name] <= 0
        }
        if missing_rollout_groups:
            raise ValueError(
                "Constrained opponent quotas must be positive for "
                + ", ".join(sorted(missing_rollout_groups))
            )
        if (
            args.bc_replay_data is not None
            or args.bc_replay_batches != 0
            or args.bc_replay_steps != 0
        ):
            raise ValueError(
                "Constrained PPO forbids BC replay outside the guarded "
                "objective"
            )
        if args.constrained_gradient_mode != "scalar":
            if args.trainable_scope != "heads":
                raise ValueError(
                    "Constrained gradient projection requires "
                    "--trainable-scope heads"
                )
            if permanent_names != core_names:
                raise ValueError(
                    "Constrained gradient projection currently requires "
                    "exactly the primary and guard opponents"
                )
            if args.auxiliary_policy_weight != 0.0:
                raise ValueError(
                    "Constrained gradient projection requires zero "
                    "auxiliary policy weight"
                )
            if (
                args.guard_policy_weight <= 0.0
                or args.constraint_dual_initial != 0.0
                or args.constraint_dual_lr != 0.0
                ):
                raise ValueError(
                    "Constrained gradient projection requires a positive "
                    "guard policy weight and disables the primal-dual term"
                )
            if args.entropy_coefficient != 0.0 or args.weight_decay != 0.0:
                raise ValueError(
                    "Constrained gradient projection requires zero entropy "
                    "coefficient and weight decay"
                )
    if args.bc_replay_data and not args.bc_replay_data.is_file():
        raise FileNotFoundError(args.bc_replay_data)
    if args.bc_replay_steps > 0 and (
        not args.bc_replay_data or args.bc_replay_batches < 1
    ):
        raise ValueError(
            "--bc-replay-steps requires --bc-replay-data and "
            "--bc-replay-batches >= 1"
        )
    if (
        args.bc_replay_context34_rows_per_batch > 0
        and args.bc_replay_steps <= 0
    ):
        raise ValueError(
            "--bc-replay-context34-rows-per-batch requires enabled BC replay"
        )
    if (
        args.bc_replay_batches < 0
        or args.bc_replay_batch_size < 1
        or args.bc_replay_workers < 1
        or args.bc_replay_steps < 0
        or args.bc_replay_context34_rows_per_batch < 0
        or (
            args.bc_replay_context34_rows_per_batch
            >= args.bc_replay_batch_size
        )
        or not 0.0 < args.bc_replay_lr_scale <= 1.0
        or not math.isfinite(args.bc_replay_order_context_weight)
        or args.bc_replay_order_context_weight <= 0.0
        or not math.isfinite(
            args.bc_replay_non_context34_fixed_multi_action_order_weight
        )
        or args.bc_replay_non_context34_fixed_multi_action_order_weight <= 0.0
    ):
        raise ValueError("Invalid BC replay configuration")
    if (
        args.bc_replay_non_context34_fixed_multi_action_order_weight != 1.0
        and args.bc_replay_loss != "ordered"
    ):
        raise ValueError(
            "--bc-replay-non-context34-fixed-multi-action-order-weight "
            "requires "
            "--bc-replay-loss ordered"
        )

    config = PPOConfig(
        bc_checkpoint=str(args.bc_checkpoint.resolve()),
        kl_reference_checkpoint=(
            str(args.kl_reference_checkpoint.resolve())
            if args.kl_reference_checkpoint is not None
            else None
        ),
        deck=str(args.deck.resolve()),
        extra_opponents=extra_opponent_specs,
        output_dir=str(args.output_dir.resolve()),
        updates=args.updates,
        environments=args.environments,
        games_per_update=args.games_per_update,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        value_learning_rate=args.value_learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        advantage_normalization=args.advantage_normalization,
        clip_ratio=args.clip_ratio,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        policy_temperature=args.policy_temperature,
        trainable_scope=args.trainable_scope,
        learning_rate_schedule=args.learning_rate_schedule,
        schedule_start_update=args.schedule_start_update,
        bc_kl_start=args.bc_kl_start,
        bc_kl_end=args.bc_kl_end,
        target_kl=args.target_kl,
        league_probability=args.league_probability,
        opponent_sampling=args.opponent_sampling,
        bc_opponent_probability=args.bc_opponent_probability,
        opponent_weights=opponent_weights,
        history_opponent_weight=args.history_opponent_weight,
        opponent_quota_mode=args.opponent_quota_mode,
        opponent_base_quotas=opponent_base_quotas,
        opponent_caps=opponent_caps,
        opponent_audit=opponent_audit,
        opponent_quota_refresh_updates=(
            args.opponent_quota_refresh_updates
        ),
        opponent_quota_seat_balance=args.opponent_quota_seat_balance,
        ppo_objective=args.ppo_objective,
        actor_reduction=args.actor_reduction,
        actor_reduction_audit_only=args.actor_reduction_audit_only,
        constrained_gradient_mode=args.constrained_gradient_mode,
        primary_opponent_name=args.primary_opponent_name,
        guard_opponent_name=args.guard_opponent_name,
        primary_policy_weight=args.primary_policy_weight,
        guard_policy_weight=args.guard_policy_weight,
        auxiliary_policy_weight=args.auxiliary_policy_weight,
        guard_surrogate_floor=args.guard_surrogate_floor,
        constraint_dual_initial=args.constraint_dual_initial,
        constraint_dual_lr=args.constraint_dual_lr,
        constraint_dual_max=args.constraint_dual_max,
        opponent_loss_weights=opponent_loss_weights,
        snapshot_interval=args.snapshot_interval,
        max_pool_size=args.max_pool_size,
        bc_replay_data=(
            str(args.bc_replay_data.resolve())
            if args.bc_replay_data is not None
            else None
        ),
        bc_replay_split=args.bc_replay_split,
        bc_replay_batches=args.bc_replay_batches,
        bc_replay_batch_size=args.bc_replay_batch_size,
        bc_replay_workers=args.bc_replay_workers,
        bc_replay_steps=args.bc_replay_steps,
        bc_replay_lr_scale=args.bc_replay_lr_scale,
        bc_replay_loss=args.bc_replay_loss,
        bc_replay_order_context_weight=args.bc_replay_order_context_weight,
        bc_replay_context34_rows_per_batch=(
            args.bc_replay_context34_rows_per_batch
        ),
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        eval_all_permanent_opponents=args.eval_all_permanent_opponents,
        selection_aggregation=args.selection_aggregation,
        checkpoint_interval=args.checkpoint_interval,
        max_game_decisions=args.max_game_decisions,
        seed=args.seed,
        device=args.device,
        resume_checkpoint=(
            str(args.resume.resolve()) if args.resume is not None else None
        ),
        resume_learner_weights=args.resume_learner_weights,
        reset_optimizer_on_resume=args.reset_optimizer_on_resume,
        reset_opponent_quota_on_resume=(
            args.reset_opponent_quota_on_resume
        ),
        value_trunk_gradient_scale=args.value_trunk_gradient_scale,
        actor_value_gradient_mode=args.actor_value_gradient_mode,
        actor_value_gradient_audit_only=(
            args.actor_value_gradient_audit_only
        ),
        bc_replay_non_context34_fixed_multi_action_order_weight=(
            args.bc_replay_non_context34_fixed_multi_action_order_weight
        ),
    )
    seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "league").mkdir(exist_ok=True)
    deck = read_deck(Path(config.deck))
    learner_deck_hash = compute_deck_hash(deck)

    bc_checkpoint = torch.load(
        config.bc_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if bc_checkpoint.get("feature_version") != BC_FEATURE_VERSION:
        raise ValueError(
            f"BC feature version mismatch: {bc_checkpoint.get('feature_version')}"
        )
    validate_checkpoint_deck(
        bc_checkpoint,
        learner_deck_hash,
        Path(config.bc_checkpoint),
        Path(config.deck),
    )
    model_config = checkpoint_model_config(bc_checkpoint)
    current_model = instantiate_model_from_bc(bc_checkpoint, device)
    bc_opponent_model = instantiate_model_from_bc(bc_checkpoint, device)
    if config.kl_reference_checkpoint is not None:
        reference_path = Path(config.kl_reference_checkpoint)
        reference_checkpoint = torch.load(
            reference_path,
            map_location="cpu",
            weights_only=False,
        )
        reference_feature_version = reference_checkpoint.get("feature_version")
        if reference_feature_version not in {
            BC_FEATURE_VERSION,
            PPO_FEATURE_VERSION,
        }:
            raise ValueError(
                f"KL reference {reference_path} feature version mismatch: "
                f"{reference_feature_version!r}"
            )
        reference_model_config = checkpoint_model_config(reference_checkpoint)
        if reference_model_config != model_config:
            raise ValueError(
                f"KL reference {reference_path} model config does not match "
                "the BC initialization architecture"
            )
        validate_checkpoint_deck(
            reference_checkpoint,
            learner_deck_hash,
            reference_path,
            Path(config.deck),
        )
        reference_model = instantiate_model_from_checkpoint(
            reference_checkpoint,
            bc_checkpoint,
            device,
        )
    else:
        reference_path = Path(config.bc_checkpoint)
        reference_checkpoint = bc_checkpoint
        reference_feature_version = BC_FEATURE_VERSION
        reference_model = instantiate_model_from_bc(bc_checkpoint, device)
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    (
        actor_parameters,
        value_parameters,
        trainable_parameter_manifest,
    ) = configure_trainable_scope(current_model, config.trainable_scope)
    optimizer = torch.optim.AdamW(
        [
            {"params": actor_parameters, "lr": config.learning_rate},
            {"params": value_parameters, "lr": config.value_learning_rate},
        ],
        eps=1e-5,
        weight_decay=config.weight_decay,
    )
    replay_optimizer = (
        torch.optim.AdamW(
            actor_parameters,
            lr=config.learning_rate * config.bc_replay_lr_scale,
            eps=1e-5,
            weight_decay=config.weight_decay,
        )
        if config.bc_replay_steps > 0
        else None
    )
    learner_weight_source = "bc_checkpoint"
    learner_weight_path = Path(config.bc_checkpoint)
    optimizer_state_loaded = False
    replay_optimizer_state_loaded = False
    optimizer_reset_reason: str | None = "fresh_run"
    start_update = 1
    resume_update: int | None = None
    resume_quota_state: dict[str, Any] | None = None
    resume_checkpoint_quota_mode = "legacy"
    resume: dict[str, Any] | None = None
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("feature_version") != PPO_FEATURE_VERSION:
            raise ValueError(
                f"Resume checkpoint {args.resume} is not a PPO checkpoint: "
                f"{resume.get('feature_version')!r}"
            )
        try:
            resume_model_config = checkpoint_model_config(resume)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Resume checkpoint {args.resume} has invalid model schema "
                "metadata for the BC-initialized learner"
            ) from error
        if resume_model_config != model_config:
            raise ValueError(
                f"Resume checkpoint {args.resume} model config does not match "
                "the BC initialization architecture"
            )
        validate_checkpoint_model_state_schema(
            resume,
            current_model.state_dict(),
            args.resume,
        )
        validate_checkpoint_deck(
            resume,
            learner_deck_hash,
            args.resume,
            Path(config.deck),
        )
        if int(resume["update"]) >= config.updates:
            raise ValueError(
                f"Resume update {resume['update']} must be lower than "
                f"--updates {config.updates}"
            )
        learner_weight_source = apply_resume_learner_weights(
            current_model,
            resume,
            config.resume_learner_weights,
            config.reset_optimizer_on_resume,
        )
        learner_weight_path = (
            Path(config.resume_checkpoint)
            if learner_weight_source == "resume_checkpoint"
            else Path(config.bc_checkpoint)
        )
        if (
            "optimizer_state_dict" in resume
            and not config.reset_optimizer_on_resume
        ):
            validate_optimizer_resume_compatibility(
                resume,
                config.trainable_scope,
                trainable_parameter_manifest,
                config.advantage_normalization,
                config.ppo_objective,
                (
                    constrained_resume_config(config)
                    if config.ppo_objective == "constrained"
                    else None
                ),
                config.policy_temperature,
                config.actor_reduction,
                config.value_trunk_gradient_scale,
                config.actor_value_gradient_mode,
            )
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            optimizer_state_loaded = True
        replay_state = resume.get("bc_replay_optimizer_state_dict")
        if (
            replay_optimizer is not None
            and replay_state is not None
            and not config.reset_optimizer_on_resume
        ):
            replay_optimizer.load_state_dict(replay_state)
            replay_optimizer_state_loaded = True
        resume_update = int(resume["update"])
        resume_config = resume.get("config")
        if isinstance(resume_config, dict):
            resume_checkpoint_quota_mode = str(
                resume_config.get("opponent_quota_mode", "legacy")
            )
        saved_quota_state = resume.get("opponent_quota_state")
        if saved_quota_state is not None:
            if not isinstance(saved_quota_state, dict):
                raise ValueError(
                    "Resume checkpoint opponent quota state is malformed"
                )
            resume_quota_state = saved_quota_state
        start_update = resume_update + 1
        if config.reset_optimizer_on_resume:
            optimizer_reset_reason = (
                "learner_weights_replaced_with_bc"
                if config.resume_learner_weights == "bc"
                else "explicit_reset_on_resume"
            )
            log(
                f"resume_optimizer_reset=true checkpoint_update={resume['update']}"
            )
        elif optimizer_state_loaded:
            optimizer_reset_reason = None
        else:
            optimizer_reset_reason = "resume_checkpoint_missing_optimizer_state"
        log(
            "learner_weight_selection="
            + json.dumps(
                {
                    "source": learner_weight_source,
                    "checkpoint": str(learner_weight_path.resolve()),
                    "resume_checkpoint": config.resume_checkpoint,
                    "resume_update": resume_update,
                    "optimizer_state_loaded": optimizer_state_loaded,
                    "optimizer_reset_reason": optimizer_reset_reason,
                },
                sort_keys=True,
            )
        )
    objective_state: ConstrainedObjectiveState | None = None
    objective_state_source = (
        "disabled" if config.ppo_objective != "constrained" else "fresh"
    )
    if config.ppo_objective == "constrained":
        if resume is not None and not config.reset_optimizer_on_resume:
            saved_objective_state = resume.get("ppo_objective_state")
            if not isinstance(saved_objective_state, dict):
                raise ValueError(
                    "Constrained resume checkpoint lacks PPO objective state"
                )
            objective_state = ConstrainedObjectiveState.from_state_dict(
                saved_objective_state,
                dual_max=config.constraint_dual_max,
            )
            objective_state_source = "resume_checkpoint"
            alignment = align_constrained_objective_resume_state(
                objective_state,
                checkpoint_update=int(resume_update),
                serialized_version=int(
                    saved_objective_state.get("version", -1)
                ),
            )
            if alignment != "matched":
                log(f"ppo_objective_resume_alignment={alignment}")
        else:
            objective_state = ConstrainedObjectiveState(
                dual_value=config.constraint_dual_initial,
            )
            objective_state_source = (
                "fresh_after_resume_reset"
                if resume is not None
                else "fresh"
            )
            if resume is not None:
                log(
                    "ppo_objective_state_reset=true "
                    "reason=optimizer_reset_on_resume"
                )
    if config.schedule_start_update > start_update:
        raise ValueError(
            f"--schedule-start-update {config.schedule_start_update} cannot be "
            f"later than the first trained update {start_update}"
        )

    opponents = [
        frozen_copy(
            bc_opponent_model,
            "bc",
            device,
            deck,
            learner_deck_hash,
            permanent=True,
            canonical_order=True,
        )
    ]
    extra_run_info: list[dict[str, Any]] = []
    for spec in config.extra_opponents:
        checkpoint_path = Path(spec["checkpoint"])
        deck_path = Path(spec["deck"])
        extra_checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        extra_feature_version = extra_checkpoint.get("feature_version")
        if extra_feature_version not in {
            BC_FEATURE_VERSION,
            PPO_FEATURE_VERSION,
        }:
            raise ValueError(
                f"Extra opponent {checkpoint_path} feature version mismatch: "
                f"{extra_feature_version!r} is not a supported BC/PPO version"
            )
        extra_model_config = checkpoint_model_config(extra_checkpoint)
        if extra_model_config != model_config:
            mismatches = {
                key: {
                    "learner": model_config.get(key),
                    "extra": extra_model_config.get(key),
                }
                for key in sorted(model_config.keys() | extra_model_config.keys())
                if model_config.get(key) != extra_model_config.get(key)
            }
            raise ValueError(
                f"Extra opponent {checkpoint_path} model_config mismatch: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        extra_deck = read_deck(deck_path)
        extra_deck_hash = compute_deck_hash(extra_deck)
        validate_checkpoint_deck(
            extra_checkpoint,
            extra_deck_hash,
            checkpoint_path,
            deck_path,
        )
        extra_model = instantiate_model_from_checkpoint(
            extra_checkpoint,
            bc_checkpoint,
            device,
        )
        opponents.append(
            frozen_copy(
                extra_model,
                spec["name"],
                device,
                extra_deck,
                extra_deck_hash,
                permanent=True,
                canonical_order=(
                    extra_feature_version == BC_FEATURE_VERSION
                ),
            )
        )
        extra_run_info.append(
            {
                **spec,
                "deck_hash": extra_deck_hash,
                "permanent": True,
                "canonical_order": (
                    extra_feature_version == BC_FEATURE_VERSION
                ),
                "feature_version": extra_feature_version,
            }
        )
    if args.resume:
        if resume_update is None:
            raise RuntimeError("Resume update was not initialized")
        league_paths = [
            path
            for path in sorted((output_dir / "league").glob("update-*.pt"))
            if int(path.stem.rsplit("-", 1)[-1]) <= resume_update
        ]
        snapshot_capacity = max(config.max_pool_size - len(opponents), 0)
        selected_league_paths = (
            league_paths[-snapshot_capacity:]
            if snapshot_capacity
            else []
        )
        for league_path in selected_league_paths:
            league_checkpoint = torch.load(
                league_path,
                map_location=device,
                weights_only=False,
            )
            league_model = instantiate_model_from_bc(bc_checkpoint, device)
            league_model.load_state_dict(league_checkpoint["model_state_dict"])
            opponents.append(
                frozen_copy(
                    league_model,
                    league_path.stem,
                    device,
                    deck,
                    learner_deck_hash,
                    permanent=False,
                    canonical_order=False,
                )
            )
    if args.resume:
        validate_opponent_quota_resume_transition(
            current_mode=config.opponent_quota_mode,
            checkpoint_mode=resume_checkpoint_quota_mode,
            checkpoint_state=resume_quota_state,
            reset_optimizer=config.reset_optimizer_on_resume,
            reset_quota_state=config.reset_opponent_quota_on_resume,
        )
    opponent_quota_controller = build_opponent_quota_controller(
        config,
        opponents,
        resume_quota_state,
        reset_resume_state=config.reset_opponent_quota_on_resume,
    )
    initial_sampling_weights = resolve_opponent_sampling_weights(
        opponents,
        config.opponent_weights,
        config.history_opponent_weight,
    )
    if initial_sampling_weights is not None:
        total_sampling_weight = sum(initial_sampling_weights)
        log(
            "opponent_sampling_probabilities="
            + json.dumps(
                {
                    opponent.name: weight / total_sampling_weight
                    for opponent, weight in zip(
                        opponents,
                        initial_sampling_weights,
                    )
                    if weight > 0.0
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    replay_batches = build_bc_replay_batches(config, model_config)
    parameter_count = sum(parameter.numel() for parameter in current_model.parameters())
    learner_initialization = {
        "selection": (
            "fresh_bc"
            if resume is None
            else config.resume_learner_weights
        ),
        "weight_source": learner_weight_source,
        "checkpoint": str(learner_weight_path.resolve()),
        "checkpoint_sha256": file_sha256(learner_weight_path),
        "initialized_model_state_sha256": model_state_sha256(current_model),
        "weights_loaded": "full_model",
        "optimizer_state_source": (
            "resume_checkpoint" if optimizer_state_loaded else "fresh"
        ),
        "replay_optimizer_state_source": (
            "disabled"
            if replay_optimizer is None
            else (
                "resume_checkpoint"
                if replay_optimizer_state_loaded
                else "fresh"
            )
        ),
        "optimizer_reset_reason": optimizer_reset_reason,
    }
    resume_metadata = {
        "source": (
            "resume_checkpoint" if resume is not None else None
        ),
        "checkpoint": config.resume_checkpoint,
        "checkpoint_sha256": (
            file_sha256(Path(config.resume_checkpoint))
            if config.resume_checkpoint is not None
            else None
        ),
        "checkpoint_update": resume_update,
        "first_trained_update": start_update,
        "opponent_quota_state": (
            "resume_checkpoint"
            if resume_quota_state is not None
            else "absent"
        ),
        "objective_state": objective_state_source,
        "league_snapshot_directory": (
            str((output_dir / "league").resolve())
            if resume is not None
            else None
        ),
    }
    reference_manifest = {
        "selection": (
            "explicit_kl_reference_checkpoint"
            if config.kl_reference_checkpoint is not None
            else "bc_checkpoint_default"
        ),
        "checkpoint": str(reference_path.resolve()),
        "checkpoint_sha256": file_sha256(reference_path),
        "feature_version": reference_feature_version,
        "model_state_sha256": model_state_sha256(reference_model),
    }
    log(
        "learner_initialization="
        + json.dumps(learner_initialization, sort_keys=True)
    )
    log(
        "kl_reference_initialization="
        + json.dumps(reference_manifest, sort_keys=True)
    )
    run_info = {
        "feature_version": PPO_FEATURE_VERSION,
        "config": asdict(config),
        "learner_initialization": learner_initialization,
        "resume_metadata": resume_metadata,
        "learner_deck_hash": learner_deck_hash,
        "extra_opponents": extra_run_info,
        "model_config": model_config,
        "parameter_count": parameter_count,
        "trainable_parameters": trainable_parameter_manifest,
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "reward": {"win": 1.0, "loss": 0.0, "draw": 0.0},
        "engine_seed_control": False,
        "kl_reference": reference_manifest,
        "opponent_quota_initial_state": (
            opponent_quota_controller.state_dict()
            if opponent_quota_controller is not None
            else None
        ),
        "opponent_quota_resume": {
            "checkpoint_mode": resume_checkpoint_quota_mode,
            "checkpoint_state_present": resume_quota_state is not None,
            "state_loaded": (
                opponent_quota_controller.resume_state_loaded
                if opponent_quota_controller is not None
                else False
            ),
            "state_reset_for_new_configuration": (
                opponent_quota_controller.resume_state_reset
                if opponent_quota_controller is not None
                else False
            ),
        },
        "ppo_objective_initial_state": (
            objective_state.state_dict()
            if objective_state is not None
            else None
        ),
        "advantage_normalization_manifest": {
            "mode": config.advantage_normalization,
            "scope": "frozen_full_rollout",
            "group_field": (
                "opponent_name"
                if config.advantage_normalization == "per_opponent"
                else None
            ),
            "selfplay_sentinel": SELFPLAY_OPPONENT_GROUP,
            "std_correction": "sample",
            "epsilon": 1e-6,
        },
        "actor_reduction_manifest": {
            "mode": config.actor_reduction,
            "scope": "actor_surrogate_only",
            "group_field": (
                "opponent_name"
                if config.actor_reduction == "quota_group_mean"
                else (
                    "game_uid"
                    if config.actor_reduction == "episode_mean"
                    else None
                )
            ),
            "weight_source": (
                "fixed_games_per_update"
                if config.actor_reduction == "quota_group_mean"
                else (
                    "frozen_rollout_game_uid"
                    if config.actor_reduction == "episode_mean"
                    else None
                )
            ),
            "quota_source": (
                "fixed_games_per_update"
                if config.actor_reduction == "quota_group_mean"
                else None
            ),
            "full_rollout_objective": (
                "mean(per_game(mean(clipped_surrogate)))"
                if config.actor_reduction == "episode_mean"
                else None
            ),
            "expected_episode_count": (
                config.games_per_update
                if config.actor_reduction == "episode_mean"
                else None
            ),
            "audit_only": config.actor_reduction_audit_only,
            "unchanged_reductions": [
                "value",
                "entropy",
                "bc_anchor_kl",
                "approx_kl",
                "clip_fraction",
            ],
        },
        "value_trunk_gradient_manifest": value_trunk_gradient_audit(
            config.value_trunk_gradient_scale
        ),
        "actor_value_gradient_manifest": actor_value_gradient_manifest(
            config.actor_value_gradient_mode,
            audit_only=config.actor_value_gradient_audit_only,
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(
        f"device={device} parameters={parameter_count:,} "
        f"trainable={trainable_parameter_manifest['trainable_parameter_count']:,} "
        f"trainable_scope={config.trainable_scope} "
        f"updates={config.updates} envs={config.environments} "
        f"games_per_update={config.games_per_update} "
        f"policy_temperature={config.policy_temperature} "
        f"ppo_objective={config.ppo_objective} "
        f"actor_reduction={config.actor_reduction} "
        f"actor_reduction_audit_only={config.actor_reduction_audit_only} "
        f"actor_value_gradient_mode={config.actor_value_gradient_mode} "
        "actor_value_gradient_audit_only="
        f"{config.actor_value_gradient_audit_only} "
        f"constrained_gradient_mode={config.constrained_gradient_mode} "
        f"advantage_normalization={config.advantage_normalization} "
        "value_trunk_gradient_scale="
        f"{config.value_trunk_gradient_scale} "
        f"opponent_sampling={config.opponent_sampling} "
        f"opponent_quota_mode={config.opponent_quota_mode} "
        f"extra_opponents={len(extra_run_info)} "
        f"bc_replay_batches={len(replay_batches)} "
        "bc_replay_non_context34_fixed_multi_action_order_weight="
        f"{config.bc_replay_non_context34_fixed_multi_action_order_weight} "
        f"lr_schedule={config.learning_rate_schedule} "
        f"schedule_start_update={config.schedule_start_update} "
        f"kl_reference={reference_path}"
    )
    if opponent_quota_controller is not None:
        if opponent_quota_controller.resume_state_reset:
            log(
                "opponent_quota_resume_state_reset=true "
                "reason=new_exact_quota_configuration"
            )
        log(
            "opponent_quota_initial_state="
            + json.dumps(
                opponent_quota_controller.state_dict(),
                ensure_ascii=False,
            )
        )
    permanent_count = sum(opponent.permanent for opponent in opponents)
    if permanent_count > config.max_pool_size:
        log(
            f"warning: permanent_opponents={permanent_count} exceeds "
            f"max_pool_size={config.max_pool_size}; permanent opponents are "
            "retained and snapshots will be pruned"
        )
    if config.league_probability < 1.0:
        log(
            "warning: dual-current self-play is enabled; for single-learner v2 "
            "set --league-probability 1.0"
        )

    best_selection_score = -1.0
    best_eval: dict[str, Any] | None = None
    best_evaluation_by_opponent: dict[str, dict[str, Any]] | None = None
    initial_eval = None
    initial_evaluation_by_opponent = None
    initial_selection_score = None
    existing_best_path = output_dir / "best.pt"
    if args.resume and existing_best_path.is_file():
        existing_best = torch.load(
            existing_best_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            resume_update is not None
            and int(existing_best.get("update", -1)) > resume_update
        ):
            log(
                f"ignoring_future_best_checkpoint="
                f"{existing_best.get('update')} resume_update={resume_update}"
            )
            existing_best = {}
        existing_metrics = existing_best.get("metrics") or {}
        existing_evaluation = (
            existing_metrics.get("evaluation")
            or existing_metrics.get("evaluation_vs_bc")
        )
        existing_evaluation_by_opponent = existing_metrics.get(
            "evaluation_by_opponent"
        )
        existing_selection_score = existing_metrics.get("selection_score")
        if (
            config.eval_all_permanent_opponents
            and isinstance(existing_evaluation_by_opponent, dict)
            and isinstance(existing_selection_score, (int, float))
        ):
            best_evaluation_by_opponent = existing_evaluation_by_opponent
            best_selection_score = float(existing_selection_score)
            if isinstance(existing_evaluation, dict):
                best_eval = existing_evaluation
        elif (
            not config.eval_all_permanent_opponents
            and isinstance(existing_evaluation, dict)
        ):
            best_eval = existing_evaluation
            best_selection_score = float(
                existing_evaluation.get("win_rate", -1.0)
            )
    # Audit-only must be a strict zero-update, zero-checkpoint path even when
    # the caller does not remember the normal training-only skip flag.
    if should_run_initial_evaluation(
        args.skip_initial_eval,
        config.actor_reduction_audit_only,
        config.actor_value_gradient_audit_only,
    ):
        if config.eval_all_permanent_opponents:
            (
                initial_evaluation_by_opponent,
                initial_selection_score,
            ) = evaluate_permanent_opponents(
                current_model,
                opponents,
                deck,
                model_config,
                device,
                config.eval_games,
                min(config.environments, config.eval_games),
                config.max_game_decisions,
                config.selection_aggregation,
            )
            initial_eval = initial_evaluation_by_opponent["bc"]
            initial_rates = {
                name: evaluation["win_rate"]
                for name, evaluation in initial_evaluation_by_opponent.items()
            }
            log(
                f"initial_eval_{config.selection_aggregation}="
                f"{initial_selection_score:.6f} "
                f"by_opponent="
                f"{json.dumps(initial_rates, ensure_ascii=False)}"
            )
        else:
            initial_eval = evaluate_head_to_head(
                current_model,
                bc_opponent_model,
                deck,
                model_config,
                device,
                config.eval_games,
                min(config.environments, config.eval_games),
                config.max_game_decisions,
                opponent_canonical_order=True,
            )
            initial_selection_score = initial_eval["win_rate"]
            log(f"initial_eval={json.dumps(initial_eval, ensure_ascii=False)}")
        if initial_selection_score > best_selection_score:
            best_selection_score = initial_selection_score
            best_eval = initial_eval
            best_evaluation_by_opponent = initial_evaluation_by_opponent
            initial_checkpoint_metrics: dict[str, Any] = {
                "evaluation": initial_eval,
            }
            if config.eval_all_permanent_opponents:
                initial_checkpoint_metrics.update(
                    {
                        "evaluation_by_opponent": (
                            initial_evaluation_by_opponent
                        ),
                        "selection_score": initial_selection_score,
                    }
                )
            save_ppo_checkpoint(
                output_dir / "best.pt",
                current_model,
                optimizer,
                config,
                model_config,
                start_update - 1,
                initial_checkpoint_metrics,
                replay_optimizer=replay_optimizer,
                opponent_quota_controller=opponent_quota_controller,
                objective_state=objective_state,
            )

    if config.actor_value_gradient_audit_only:
        preexisting_checkpoints = sorted(output_dir.rglob("*.pt"))
        if preexisting_checkpoints:
            raise RuntimeError(
                "Actor/value gradient audit output directory already "
                "contains checkpoints: "
                + ", ".join(str(path) for path in preexisting_checkpoints)
            )
        rollout, rollout_metrics = collect_rollout(
            current_model,
            opponents,
            deck,
            model_config,
            config,
            device,
            start_update,
            opponent_quota_controller,
        )
        gradient_audit = audit_actor_value_gradients(
            current_model,
            reference_model,
            optimizer,
            rollout,
            model_config,
            config,
            device,
            start_update,
            actor_parameters,
            value_parameters,
            replay_optimizer,
        )
        gradient_audit.update(
            {
                "seed": config.seed,
                "update": start_update,
                "checkpoint_source": learner_initialization,
                "reference_source": reference_manifest,
                "rollout": rollout_metrics,
            }
        )
        audit_path = output_dir / "actor_value_gradient_audit.json"
        audit_path.write_text(
            json.dumps(
                gradient_audit,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        emitted_checkpoints = sorted(output_dir.rglob("*.pt"))
        if emitted_checkpoints:
            raise RuntimeError(
                "Actor/value gradient audit unexpectedly emitted a "
                "checkpoint: "
                + ", ".join(str(path) for path in emitted_checkpoints)
            )
        log(
            "actor_value_gradient_audit="
            + json.dumps(
                {
                    "path": str(audit_path),
                    "effect_status": gradient_audit["effect_status"],
                    "gradient": gradient_audit["gradient"],
                    "gate": gradient_audit["pre_registered_gate"],
                },
                ensure_ascii=False,
            )
        )
        return

    metrics_path = output_dir / "metrics.jsonl"
    metrics_mode = "a" if args.resume and metrics_path.exists() else "w"
    last_result: dict[str, Any] | None = None
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for update in range(start_update, config.updates + 1):
            rollout, rollout_metrics = collect_rollout(
                current_model,
                opponents,
                deck,
                model_config,
                config,
                device,
                update,
                opponent_quota_controller,
            )
            if config.actor_reduction_audit_only:
                actor_reduction_audit = audit_actor_reduction_gradients(
                    current_model,
                    rollout,
                    model_config,
                    config,
                    device,
                    actor_parameters,
                )
                actor_reduction_audit.update(
                    {
                        "seed": config.seed,
                        "update": update,
                        "checkpoint_source": learner_initialization,
                        "rollout": rollout_metrics,
                    }
                )
                audit_path = output_dir / "actor_reduction_audit.json"
                audit_path.write_text(
                    json.dumps(
                        actor_reduction_audit,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                metrics_file.write(
                    json.dumps(
                        {
                            "update": update,
                            "rollout": rollout_metrics,
                            "actor_reduction_audit": actor_reduction_audit,
                            "optimization": None,
                            "bc_replay": None,
                            "evaluation_vs_bc": None,
                            "league_pool": [
                                opponent.name for opponent in opponents
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                metrics_file.flush()
                log(
                    "actor_reduction_audit="
                    + json.dumps(
                        {
                            "path": str(audit_path),
                            "gradient": actor_reduction_audit["gradient"],
                            "gate": actor_reduction_audit[
                                "pre_registered_gate"
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
                return
            update_metrics = ppo_update(
                current_model,
                reference_model,
                optimizer,
                rollout,
                model_config,
                config,
                device,
                update,
                objective_state,
            )
            replay_metrics = bc_replay_update(
                current_model,
                replay_optimizer,
                replay_batches,
                config,
                device,
                actor_learning_rate=optimizer.param_groups[0]["lr"],
            )
            evaluation = None
            evaluation_by_opponent = None
            evaluation_selection_score = None
            if update % config.eval_interval == 0 or update == config.updates:
                if config.eval_all_permanent_opponents:
                    (
                        evaluation_by_opponent,
                        evaluation_selection_score,
                    ) = evaluate_permanent_opponents(
                        current_model,
                        opponents,
                        deck,
                        model_config,
                        device,
                        config.eval_games,
                        min(config.environments, config.eval_games),
                        config.max_game_decisions,
                        config.selection_aggregation,
                    )
                    evaluation = evaluation_by_opponent["bc"]
                else:
                    evaluation = evaluate_head_to_head(
                        current_model,
                        bc_opponent_model,
                        deck,
                        model_config,
                        device,
                        config.eval_games,
                        min(config.environments, config.eval_games),
                        config.max_game_decisions,
                        opponent_canonical_order=True,
                    )
                    evaluation_selection_score = evaluation["win_rate"]
            result = {
                "update": update,
                "rollout": rollout_metrics,
                "optimization": update_metrics,
                "bc_replay": replay_metrics,
                "evaluation_vs_bc": evaluation,
                "league_pool": [opponent.name for opponent in opponents],
            }
            if config.eval_all_permanent_opponents:
                result.update(
                    {
                        "evaluation_by_opponent": evaluation_by_opponent,
                        "selection_score": evaluation_selection_score,
                    }
                )
            metrics_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            metrics_file.flush()
            last_result = result
            message = (
                f"update={update}/{config.updates} "
                f"games={rollout_metrics['valid_games']} "
                f"transitions={rollout_metrics['transitions_kept']} "
                f"rollout_rate={rollout_metrics['decisions_per_second']:.0f}/s "
                f"policy_loss={update_metrics['policy_loss']:.5f} "
                f"value_loss={update_metrics['value_loss']:.5f} "
                f"approx_kl={update_metrics['approx_kl']:.5f} "
                f"bc_kl={update_metrics['bc_anchor_kl']:.5f}"
            )
            if replay_metrics:
                message += (
                    f" replay_loss={replay_metrics['loss']:.5f} "
                    f"replay_rows={replay_metrics['rows']}"
                )
            if evaluation:
                if config.eval_all_permanent_opponents:
                    opponent_rates = ",".join(
                        f"{name}:{opponent_eval['win_rate']:.3f}"
                        for name, opponent_eval in (
                            evaluation_by_opponent or {}
                        ).items()
                    )
                    message += (
                        f" eval_{config.selection_aggregation}="
                        f"{evaluation_selection_score:.4f} "
                        f"eval_by={opponent_rates}"
                    )
                else:
                    message += (
                        f" eval_win={evaluation['win_rate']:.4f} "
                        f"W/L/D={evaluation['wins']}/"
                        f"{evaluation['losses']}/{evaluation['draws']}"
                    )
            log(message)

            if (
                evaluation
                and evaluation_selection_score is not None
                and evaluation_selection_score > best_selection_score
            ):
                best_selection_score = evaluation_selection_score
                best_eval = evaluation
                best_evaluation_by_opponent = evaluation_by_opponent
                save_ppo_checkpoint(
                    output_dir / "best.pt",
                    current_model,
                    optimizer,
                    config,
                    model_config,
                    update,
                    result,
                    replay_optimizer=replay_optimizer,
                    opponent_quota_controller=opponent_quota_controller,
                    objective_state=objective_state,
                )
            if (
                update % config.checkpoint_interval == 0
                or update == config.updates
            ):
                save_ppo_checkpoint(
                    output_dir / "checkpoints" / f"update-{update:04d}.pt",
                    current_model,
                    optimizer,
                    config,
                    model_config,
                    update,
                    result,
                    replay_optimizer=replay_optimizer,
                    opponent_quota_controller=opponent_quota_controller,
                    objective_state=objective_state,
                )
            if update % config.snapshot_interval == 0:
                opponent = frozen_copy(
                    current_model,
                    f"update-{update:04d}",
                    device,
                    deck,
                    learner_deck_hash,
                    permanent=False,
                    canonical_order=False,
                )
                opponents.append(opponent)
                trim_opponent_pool(opponents, config.max_pool_size)
                torch.save(
                    {
                        "feature_version": PPO_FEATURE_VERSION,
                        "model_config": model_config,
                        "model_state_dict": current_model.state_dict(),
                        "update": update,
                    },
                    output_dir / "league" / f"update-{update:04d}.pt",
                )

    save_ppo_checkpoint(
        output_dir / "last.pt",
        current_model,
        optimizer,
        config,
        model_config,
        config.updates,
        last_result or {},
        replay_optimizer=replay_optimizer,
        opponent_quota_controller=opponent_quota_controller,
        objective_state=objective_state,
    )
    summary = {
        "feature_version": PPO_FEATURE_VERSION,
        "reward": {"win": 1.0, "loss": 0.0, "draw": 0.0},
        "value_trunk_gradient": value_trunk_gradient_audit(
            config.value_trunk_gradient_scale
        ),
        "updates_completed": config.updates,
        "initial_evaluation_vs_bc": initial_eval,
        "best_evaluation_vs_bc": best_eval,
        "best_win_rate_vs_bc": (
            float(best_eval.get("win_rate", -1.0))
            if isinstance(best_eval, dict)
            else -1.0
        ),
        "best_selection_score": best_selection_score,
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "metrics": str(metrics_path),
        "parameter_count": parameter_count,
        "trainable_parameters": trainable_parameter_manifest,
        "opponent_quota_final_state": (
            opponent_quota_controller.state_dict()
            if opponent_quota_controller is not None
            else None
        ),
        "ppo_objective_final_state": (
            objective_state.state_dict()
            if objective_state is not None
            else None
        ),
    }
    if config.eval_all_permanent_opponents:
        summary.update(
            {
                "initial_evaluation_by_opponent": (
                    initial_evaluation_by_opponent
                ),
                "initial_selection_score": initial_selection_score,
                "best_evaluation_by_opponent": (
                    best_evaluation_by_opponent
                ),
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
