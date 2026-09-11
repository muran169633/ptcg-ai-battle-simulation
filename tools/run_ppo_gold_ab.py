#!/usr/bin/env python3
"""Preregister and run an equal-budget Marnie-control versus gold-league PPO A/B.

This is deliberately a local training orchestrator.  It can invoke only
``tools/train_ppo.py``; it has no upload, packaging, Kaggle, or submission
path.  The two branches use paired seeds and identical optimization/rollout
budgets.  They differ only in their frozen-opponent leagues:

* A: the base BC mirror plus either incumbent v3 or quality-passing Marnie
  policy clones.
* B: the base BC mirror plus every quality-passing policy clone, allocated in
  even game pairs hierarchically by archetype and then policy.

Every non-zero opponent quota is even and ``train_ppo.py`` is called with
``--opponent-quota-seat-balance``.  Consequently each frozen opponent receives
an exact 50/50 learner-seat quota on every update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train_ppo.py"
LEAGUE_SCHEMA = "ptcg-ppo-opponent-league-v1"
PLAN_SCHEMA = "ptcg-gold-ppo-ab-plan-v1"
PREREGISTRATION_SCHEMA = "ptcg-gold-ppo-ab-preregistration-v1"
SUMMARY_SCHEMA = "ptcg-gold-ppo-ab-run-v1"

DEFAULT_BC_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "bc_marnie_train24_day25hash80_orbit_v3"
    / "best.pt"
)
DEFAULT_RESUME_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "ppo_marnie_v3_incumbent600"
    / "checkpoints"
    / "update-0440.pt"
)
DEFAULT_LEARNER_DECK = (
    REPO_ROOT / "data" / "decks" / "marnie_grimmsnarl_froslass_luca.csv"
)
DEFAULT_ACTOR_LEARNING_RATE = 6e-6
DEFAULT_GAE_LAMBDA = 0.97
DEFAULT_VALUE_LEARNING_RATE = 1.5e-5
DEFAULT_VALUE_COEFFICIENT = 0.5
DEFAULT_ENTROPY_COEFFICIENT = 0.001
DEFAULT_POLICY_TEMPERATURE = 0.8
DEFAULT_VALUE_TRUNK_GRADIENT_SCALE = 1.0
ACTOR_VALUE_GRADIENT_MODES = (
    "scalar",
    "actor_priority_value_pcgrad",
)
DEFAULT_ACTOR_VALUE_GRADIENT_MODE = "scalar"
DEFAULT_BC_KL_COEFFICIENT = 0.004
RUNNER_TRAINABLE_SCOPES = (
    "last_block_heads",
    "last_two_blocks_heads",
)
DEFAULT_TRAINABLE_SCOPE = "last_block_heads"
ACTOR_REDUCTIONS = (
    "transition_mean",
    "quota_group_mean",
    "episode_mean",
)
DEFAULT_ACTOR_REDUCTION = "transition_mean"
ADVANTAGE_NORMALIZATIONS = ("global", "per_opponent")
DEFAULT_ADVANTAGE_NORMALIZATION = "global"
BC_REPLAY_SPLITS = ("train", "valid", "test")
BC_REPLAY_LOSSES = ("set", "ordered", "hybrid_ordered")
DEFAULT_BC_REPLAY_BATCHES = 0
DEFAULT_BC_REPLAY_BATCH_SIZE = 256
DEFAULT_BC_REPLAY_WORKERS = 8
DEFAULT_BC_REPLAY_STEPS = 0
DEFAULT_BC_REPLAY_LR_SCALE = 0.25
DEFAULT_BC_REPLAY_LOSS = "set"
DEFAULT_BC_REPLAY_ORDER_CONTEXT_WEIGHT = 1.0
DEFAULT_BC_REPLAY_NON_CONTEXT34_FIXED_MULTI_ACTION_ORDER_WEIGHT = 1.0
DEFAULT_BC_REPLAY_CONTEXT34_ROWS_PER_BATCH = 0
RESUME_LEARNER_WEIGHT_SOURCES = ("bc", "resume")
DEFAULT_RESUME_LEARNER_WEIGHTS = "bc"


@dataclass(frozen=True)
class BudgetProfile:
    name: str
    additional_updates: int
    games_per_update: int
    ppo_epochs: int
    environments: int
    minibatch_size: int
    eval_games: int
    bc_anchor_games: int
    seeds: tuple[int, ...]


PROFILES: dict[str, BudgetProfile] = {
    # 21 gold policies need at least 42 paired-seat rollout games.  After the
    # fixed eight-game BC anchor, 56 games remain, so the complete Top-21
    # policy set still fits in this deliberately small two-update pilot.
    "pilot": BudgetProfile(
        name="pilot",
        additional_updates=2,
        games_per_update=64,
        ppo_epochs=2,
        environments=16,
        minibatch_size=512,
        eval_games=32,
        bc_anchor_games=8,
        seeds=(20260727,),
    ),
    # Single-variable duration probe: identical to ``pilot`` except that the
    # learner receives eight updates.  This tests whether the two-update
    # pilot was simply too weak before changing opponent quotas or optimizer
    # hyperparameters.
    "duration8": BudgetProfile(
        name="duration8",
        additional_updates=8,
        games_per_update=64,
        ppo_epochs=2,
        environments=16,
        minibatch_size=512,
        eval_games=64,
        bc_anchor_games=8,
        seeds=(20260727,),
    ),
    # Current-stack duration probe: identical to ``duration8`` except for
    # four additional fresh on-policy updates.  Keep this as a named profile
    # rather than exposing an arbitrary horizon override so the experiment
    # remains preregistered and attributable.
    "duration12": BudgetProfile(
        name="duration12",
        additional_updates=12,
        games_per_update=64,
        ppo_epochs=2,
        environments=16,
        minibatch_size=512,
        eval_games=64,
        bc_anchor_games=8,
        seeds=(20260727,),
    ),
    # Three paired seeds are preregistered for the formal comparison.  The
    # orchestration does not launch anything unless --execute is explicit.
    "formal": BudgetProfile(
        name="formal",
        additional_updates=20,
        games_per_update=128,
        ppo_epochs=4,
        environments=16,
        minibatch_size=1024,
        eval_games=64,
        bc_anchor_games=16,
        seeds=(20260727, 20260728, 20260729),
    ),
}


@dataclass(frozen=True)
class Opponent:
    name: str
    checkpoint: Path | None
    deck: Path
    archetype: str
    policy_id: str
    submission_id: int | None
    quality_pass: bool
    checkpoint_sha256: str | None = None
    deck_hash: str | None = None

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["checkpoint"] = (
            str(self.checkpoint) if self.checkpoint is not None else None
        )
        value["deck"] = str(self.deck)
        return value


@dataclass(frozen=True)
class BCReplaySettings:
    data: Path | None
    data_sha256: str | None
    split: str
    batches: int
    batch_size: int
    workers: int
    steps: int
    lr_scale: float
    loss: str
    order_context_weight: float
    non_context34_fixed_multi_action_order_weight: float
    context34_rows_per_batch: int

    @property
    def enabled(self) -> bool:
        return self.data is not None

    def public_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "data": str(self.data) if self.data is not None else None,
            "data_sha256": self.data_sha256,
            "split": self.split,
            "batches": self.batches,
            "batch_size": self.batch_size,
            "workers": self.workers,
            "steps": self.steps,
            "lr_scale": self.lr_scale,
            "loss": self.loss,
            "order_context_weight": self.order_context_weight,
            "non_context34_fixed_multi_action_order_weight": (
                self.non_context34_fixed_multi_action_order_weight
            ),
            "context34_rows_per_batch": self.context34_rows_per_batch,
        }

    def command_args(self) -> list[str]:
        if not self.enabled:
            return []
        assert self.data is not None
        return [
            "--bc-replay-data",
            str(self.data),
            "--bc-replay-split",
            self.split,
            "--bc-replay-batches",
            str(self.batches),
            "--bc-replay-batch-size",
            str(self.batch_size),
            "--bc-replay-workers",
            str(self.workers),
            "--bc-replay-steps",
            str(self.steps),
            "--bc-replay-lr-scale",
            str(self.lr_scale),
            "--bc-replay-loss",
            self.loss,
            "--bc-replay-order-context-weight",
            str(self.order_context_weight),
            "--bc-replay-non-context34-fixed-multi-action-order-weight",
            str(self.non_context34_fixed_multi_action_order_weight),
            "--bc-replay-context34-rows-per-batch",
            str(self.context34_rows_per_batch),
        ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_actor_learning_rate(value: Any) -> float:
    try:
        learning_rate = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Actor learning rate must be a finite positive number") from error
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("Actor learning rate must be a finite positive number")
    return learning_rate


def actor_learning_rate_arg(value: str) -> float:
    try:
        return validate_actor_learning_rate(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def validate_gae_lambda(value: Any) -> float:
    try:
        gae_lambda = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "GAE lambda must be finite and in [0, 1]"
        ) from error
    if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("GAE lambda must be finite and in [0, 1]")
    return gae_lambda


def gae_lambda_arg(value: str) -> float:
    try:
        return validate_gae_lambda(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def validate_actor_reduction(value: Any) -> str:
    reduction = str(value)
    if reduction not in ACTOR_REDUCTIONS:
        raise ValueError(
            "Actor reduction must be one of: "
            + ", ".join(ACTOR_REDUCTIONS)
        )
    return reduction


def validate_advantage_normalization(value: Any) -> str:
    normalization = str(value)
    if normalization not in ADVANTAGE_NORMALIZATIONS:
        raise ValueError(
            "Advantage normalization must be one of: "
            + ", ".join(ADVANTAGE_NORMALIZATIONS)
        )
    return normalization


def validate_value_learning_rate(value: Any) -> float:
    try:
        learning_rate = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Value learning rate must be a finite positive number") from error
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("Value learning rate must be a finite positive number")
    return learning_rate


def value_learning_rate_arg(value: str) -> float:
    try:
        return validate_value_learning_rate(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def validate_value_coefficient(value: Any) -> float:
    try:
        coefficient = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Value coefficient must be a finite positive number"
        ) from error
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("Value coefficient must be a finite positive number")
    return coefficient


def value_coefficient_arg(value: str) -> float:
    try:
        return validate_value_coefficient(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def validate_entropy_coefficient(value: Any) -> float:
    try:
        coefficient = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Entropy coefficient must be a finite non-negative number"
        ) from error
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError(
            "Entropy coefficient must be a finite non-negative number"
        )
    return coefficient


def entropy_coefficient_arg(value: str) -> float:
    try:
        return validate_entropy_coefficient(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def validate_policy_temperature(value: Any) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Policy temperature must be a finite positive number"
        ) from error
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Policy temperature must be a finite positive number")
    return temperature


def policy_temperature_arg(value: str) -> float:
    try:
        return validate_policy_temperature(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def validate_value_trunk_gradient_scale(value: Any) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Value-trunk gradient scale must be finite and in [0, 1]"
        ) from error
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError(
            "Value-trunk gradient scale must be finite and in [0, 1]"
        )
    return scale


def value_trunk_gradient_scale_arg(value: str) -> float:
    try:
        return validate_value_trunk_gradient_scale(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def validate_actor_value_gradient_mode(value: Any) -> str:
    mode = str(value)
    if mode not in ACTOR_VALUE_GRADIENT_MODES:
        raise ValueError(f"Unsupported actor/value gradient mode: {mode!r}")
    return mode


def validate_bc_kl_coefficient(value: Any) -> float:
    try:
        coefficient = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "BC KL coefficient must be a finite positive number"
        ) from error
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("BC KL coefficient must be a finite positive number")
    return coefficient


def bc_kl_coefficient_arg(value: str) -> float:
    try:
        return validate_bc_kl_coefficient(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def nonnegative_int_arg(value: str) -> int:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return integer


def positive_int_arg(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return integer


def validate_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_resume_learner_weights(value: Any) -> str:
    source = str(value)
    if source not in RESUME_LEARNER_WEIGHT_SOURCES:
        raise ValueError(
            f"Unsupported resume learner weight source: {source!r}"
        )
    return source


def positive_even_int_arg(value: str) -> int:
    integer = positive_int_arg(value)
    if integer % 2:
        raise argparse.ArgumentTypeError("value must be an even integer")
    return integer


def replay_lr_scale_arg(value: str) -> float:
    scale = float(value)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in (0, 1]")
    return scale


def positive_float_arg(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def make_bc_replay_settings(
    *,
    data: Path | None = None,
    split: str = "train",
    batches: int = DEFAULT_BC_REPLAY_BATCHES,
    batch_size: int = DEFAULT_BC_REPLAY_BATCH_SIZE,
    workers: int = DEFAULT_BC_REPLAY_WORKERS,
    steps: int = DEFAULT_BC_REPLAY_STEPS,
    lr_scale: float = DEFAULT_BC_REPLAY_LR_SCALE,
    loss: str = DEFAULT_BC_REPLAY_LOSS,
    order_context_weight: float = DEFAULT_BC_REPLAY_ORDER_CONTEXT_WEIGHT,
    non_context34_fixed_multi_action_order_weight: float = (
        DEFAULT_BC_REPLAY_NON_CONTEXT34_FIXED_MULTI_ACTION_ORDER_WEIGHT
    ),
    context34_rows_per_batch: int = (
        DEFAULT_BC_REPLAY_CONTEXT34_ROWS_PER_BATCH
    ),
) -> BCReplaySettings:
    integer_values = {
        "batches": (batches, 0),
        "batch_size": (batch_size, 1),
        "workers": (workers, 1),
        "steps": (steps, 0),
        "context34_rows_per_batch": (context34_rows_per_batch, 0),
    }
    for label, (value, minimum) in integer_values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            qualifier = "positive" if minimum == 1 else "non-negative"
            raise ValueError(f"BC replay {label} must be a {qualifier} integer")
    try:
        lr_scale = float(lr_scale)
        order_context_weight = float(order_context_weight)
        non_context34_fixed_multi_action_order_weight = float(
            non_context34_fixed_multi_action_order_weight
        )
    except (TypeError, ValueError) as error:
        raise ValueError("BC replay floating-point settings are invalid") from error
    if not math.isfinite(lr_scale) or not 0.0 < lr_scale <= 1.0:
        raise ValueError("BC replay lr_scale must be finite and in (0, 1]")
    if (
        not math.isfinite(order_context_weight)
        or order_context_weight <= 0.0
    ):
        raise ValueError(
            "BC replay order_context_weight must be finite and positive"
        )
    if (
        not math.isfinite(
            non_context34_fixed_multi_action_order_weight
        )
        or non_context34_fixed_multi_action_order_weight <= 0.0
    ):
        raise ValueError(
            "BC replay non_context34_fixed_multi_action_order_weight must "
            "be finite and positive"
        )
    if split not in BC_REPLAY_SPLITS:
        raise ValueError(f"Unsupported BC replay split: {split!r}")
    if loss not in BC_REPLAY_LOSSES:
        raise ValueError(f"Unsupported BC replay loss: {loss!r}")

    enablement = (data is not None, batches > 0, steps > 0)
    if any(enablement) and not all(enablement):
        raise ValueError(
            "BC replay requires data, batches > 0, and steps > 0 together"
        )
    enabled = all(enablement)
    nondefault_tuning = (
        split != "train"
        or batch_size != DEFAULT_BC_REPLAY_BATCH_SIZE
        or workers != DEFAULT_BC_REPLAY_WORKERS
        or lr_scale != DEFAULT_BC_REPLAY_LR_SCALE
        or loss != DEFAULT_BC_REPLAY_LOSS
        or order_context_weight != DEFAULT_BC_REPLAY_ORDER_CONTEXT_WEIGHT
        or non_context34_fixed_multi_action_order_weight
        != DEFAULT_BC_REPLAY_NON_CONTEXT34_FIXED_MULTI_ACTION_ORDER_WEIGHT
        or context34_rows_per_batch
        != DEFAULT_BC_REPLAY_CONTEXT34_ROWS_PER_BATCH
    )
    if not enabled and nondefault_tuning:
        raise ValueError(
            "BC replay tuning parameters require data, batches > 0, and "
            "steps > 0"
        )
    if enabled and context34_rows_per_batch >= batch_size:
        raise ValueError(
            "BC replay context34_rows_per_batch must be strictly less than "
            "batch_size when replay is enabled"
        )
    if (
        enabled
        and non_context34_fixed_multi_action_order_weight != 1.0
        and loss != "ordered"
    ):
        raise ValueError(
            "BC replay non_context34_fixed_multi_action_order_weight "
            "requires ordered loss"
        )

    resolved_data: Path | None = None
    data_sha256: str | None = None
    if enabled:
        assert data is not None
        resolved_data = data.expanduser().resolve()
        if not resolved_data.is_file():
            raise FileNotFoundError(f"BC replay data: {resolved_data}")
        data_sha256 = file_sha256(resolved_data)
    return BCReplaySettings(
        data=resolved_data,
        data_sha256=data_sha256,
        split=split,
        batches=batches,
        batch_size=batch_size,
        workers=workers,
        steps=steps,
        lr_scale=lr_scale,
        loss=loss,
        order_context_weight=order_context_weight,
        non_context34_fixed_multi_action_order_weight=(
            non_context34_fixed_multi_action_order_weight
        ),
        context34_rows_per_batch=context34_rows_per_batch,
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def resolve_manifest_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def read_deck(path: Path) -> list[int]:
    try:
        cards = [
            int(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except ValueError as error:
        raise ValueError(f"{path}: deck must contain one integer per line") from error
    if len(cards) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(cards)}")
    return cards


def compute_deck_hash(path: Path) -> str:
    canonical = ",".join(str(card) for card in sorted(read_deck(path)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def expected_extra_opponent_name(checkpoint: Path, deck: Path) -> str:
    """Mirror the permanent-opponent naming rule in ``train_ppo.py``."""
    return f"{checkpoint.stem}@{deck.stem}"


def parse_quality_pass(raw: dict[str, Any]) -> bool:
    quality = raw.get("quality")
    return bool(isinstance(quality, dict) and quality.get("pass") is True)


def load_quality_passing_league(
    manifest_path: Path,
) -> tuple[list[Opponent], list[dict[str, Any]], dict[str, Any]]:
    """Load only quality-passing, PPO-usable entries from a clone manifest."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != LEAGUE_SCHEMA:
        raise ValueError(
            f"{manifest_path}: expected schema {LEAGUE_SCHEMA!r}, "
            f"found {manifest.get('schema_version')!r}"
        )
    safety = manifest.get("safety")
    if not isinstance(safety, dict):
        raise ValueError(f"{manifest_path}: missing safety declaration")
    if safety.get("uploads_or_submissions_performed") is not False:
        raise ValueError(
            f"{manifest_path}: league must declare no uploads/submissions"
        )
    if safety.get("uses_open_submission_code") is not False:
        raise ValueError(
            f"{manifest_path}: open-submission code is not allowed"
        )
    if safety.get("uses_public_replay_actions_only") is not True:
        raise ValueError(
            f"{manifest_path}: expected public-replay-actions-only provenance"
        )
    raw_opponents = manifest.get("opponents")
    if not isinstance(raw_opponents, list):
        raise ValueError(f"{manifest_path}: opponents must be a list")

    included: list[Opponent] = []
    excluded: list[dict[str, Any]] = []
    declared_excluded = manifest.get("excluded", [])
    if not isinstance(declared_excluded, list):
        raise ValueError(f"{manifest_path}: excluded must be a list")
    for index, raw in enumerate(declared_excluded):
        if not isinstance(raw, dict):
            raise ValueError(f"excluded[{index}] must be an object")
        entry = raw.get("entry", raw)
        if not isinstance(entry, dict):
            raise ValueError(f"excluded[{index}].entry must be an object")
        excluded.append(
            {
                "policy_id": str(entry.get("policy_id", "")).strip(),
                "archetype": str(entry.get("archetype", "")).strip(),
                "reason": (
                    entry.get("ppo_ineligible_reason")
                    or entry.get("status")
                    or "league_manifest_excluded"
                ),
                "quality": entry.get("quality"),
            }
        )
    names: set[str] = set()
    policy_ids: set[str] = set()
    base = manifest_path.parent
    for index, raw in enumerate(raw_opponents):
        if not isinstance(raw, dict):
            raise ValueError(f"opponents[{index}] must be an object")
        policy_id = str(raw.get("policy_id", "")).strip()
        archetype = str(raw.get("archetype", "")).strip()
        if not policy_id or not archetype:
            raise ValueError(
                f"opponents[{index}] requires policy_id and archetype"
            )
        if policy_id in policy_ids:
            raise ValueError(f"Duplicate league policy_id {policy_id!r}")
        policy_ids.add(policy_id)
        if not parse_quality_pass(raw):
            excluded.append(
                {
                    "policy_id": policy_id,
                    "archetype": archetype,
                    "reason": "quality_pass_is_not_true",
                }
            )
            continue
        checkpoint = resolve_manifest_path(
            raw.get("checkpoint"),
            base,
            f"opponents[{index}].checkpoint",
        )
        deck = resolve_manifest_path(
            raw.get("deck"),
            base,
            f"opponents[{index}].deck",
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not deck.is_file():
            raise FileNotFoundError(deck)
        expected_name = expected_extra_opponent_name(checkpoint, deck)
        declared_name = str(raw.get("name", ""))
        if declared_name != expected_name:
            raise ValueError(
                f"{policy_id}: declared opponent name {declared_name!r} "
                f"does not match train_ppo name {expected_name!r}"
            )
        if expected_name == "bc" or expected_name in names:
            raise ValueError(f"Duplicate/reserved opponent name {expected_name!r}")
        names.add(expected_name)

        declared_checkpoint_sha = raw.get("checkpoint_sha256")
        actual_checkpoint_sha = file_sha256(checkpoint)
        if (
            declared_checkpoint_sha is not None
            and declared_checkpoint_sha != actual_checkpoint_sha
        ):
            raise ValueError(f"{policy_id}: checkpoint SHA-256 mismatch")
        declared_deck_hash = raw.get("deck_hash")
        actual_deck_hash = compute_deck_hash(deck)
        if declared_deck_hash != actual_deck_hash:
            raise ValueError(f"{policy_id}: deck hash mismatch")
        submission_id_raw = raw.get("submission_id")
        submission_id = (
            int(submission_id_raw) if submission_id_raw is not None else None
        )
        included.append(
            Opponent(
                name=expected_name,
                checkpoint=checkpoint,
                deck=deck,
                archetype=archetype,
                policy_id=policy_id,
                submission_id=submission_id,
                quality_pass=True,
                checkpoint_sha256=actual_checkpoint_sha,
                deck_hash=actual_deck_hash,
            )
        )
    if not included:
        raise ValueError("No quality-passing gold clone is PPO-eligible")
    included.sort(key=lambda item: (item.archetype.casefold(), item.policy_id))
    included_policy_ids = {item.policy_id for item in included}
    excluded = [
        item
        for item in excluded
        if item["policy_id"] not in included_policy_ids
    ]
    excluded_by_policy: dict[str, dict[str, Any]] = {}
    anonymous_excluded: list[dict[str, Any]] = []
    for item in excluded:
        policy_id = item["policy_id"]
        if policy_id:
            # Merged manifests retain rejection history. The last occurrence
            # is the newest recovery attempt and is the relevant final audit.
            excluded_by_policy[policy_id] = item
        else:
            anonymous_excluded.append(item)
    excluded = anonymous_excluded + list(excluded_by_policy.values())
    return included, excluded, manifest


def select_gold_policy_allowlist(
    opponents: Sequence[Opponent],
    policy_ids: Sequence[str] | None,
) -> tuple[list[Opponent], list[dict[str, Any]]]:
    """Apply an explicit, auditable policy allowlist after quality gating."""
    if policy_ids is None:
        return list(opponents), []
    normalized = [str(policy_id).strip() for policy_id in policy_ids]
    if not normalized or any(not policy_id for policy_id in normalized):
        raise ValueError("Gold policy allowlist must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Gold policy allowlist contains duplicate policy IDs")
    available = {opponent.policy_id: opponent for opponent in opponents}
    unknown = sorted(set(normalized) - set(available))
    if unknown:
        raise ValueError(
            "Gold policy allowlist contains unavailable or quality-rejected "
            f"policy IDs: {unknown}"
        )
    selected_ids = set(normalized)
    selected = [
        opponent
        for opponent in opponents
        if opponent.policy_id in selected_ids
    ]
    excluded = [
        {
            "policy_id": opponent.policy_id,
            "archetype": opponent.archetype,
            "reason": "protocol_policy_allowlist_excluded",
            "quality": {"pass": True},
        }
        for opponent in opponents
        if opponent.policy_id not in selected_ids
    ]
    return selected, excluded


def validate_even_quota_total(total_games: int, opponents: int) -> None:
    if total_games <= 0 or total_games % 2:
        raise ValueError("Seat-balanced game budget must be positive and even")
    if opponents <= 0:
        raise ValueError("At least one opponent is required")
    if total_games < 2 * opponents:
        raise ValueError(
            f"{total_games} games cannot cover {opponents} opponents with "
            "both learner seats; need at least "
            f"{2 * opponents}"
        )


def allocate_even_policy_quotas(
    total_games: int,
    opponents: Sequence[Opponent],
) -> dict[str, int]:
    """Allocate exact paired-seat quotas, balancing archetypes before policies.

    Each policy first receives one two-game pair.  Remaining pairs go to the
    archetype with the smallest current aggregate quota (stable lexical ties),
    then to that archetype's least-exposed policy.  This prevents a populous
    archetype from consuming all of B while retaining exact policy coverage.
    """
    validate_even_quota_total(total_games, len(opponents))
    grouped: dict[str, list[Opponent]] = {}
    for opponent in opponents:
        grouped.setdefault(opponent.archetype, []).append(opponent)
    for policies in grouped.values():
        policies.sort(key=lambda item: item.policy_id)

    pairs = {opponent.name: 1 for opponent in opponents}
    remaining = total_games // 2 - len(opponents)
    archetypes = sorted(grouped, key=str.casefold)
    while remaining:
        archetype = min(
            archetypes,
            key=lambda key: (
                sum(pairs[item.name] for item in grouped[key]),
                key.casefold(),
            ),
        )
        policy = min(
            grouped[archetype],
            key=lambda item: (pairs[item.name], item.policy_id),
        )
        pairs[policy.name] += 1
        remaining -= 1
    return {name: count * 2 for name, count in sorted(pairs.items())}


def allocate_even_flat_quotas(
    total_games: int,
    opponents: Sequence[Opponent],
) -> dict[str, int]:
    """Allocate exact paired-seat quotas evenly across a control league."""
    validate_even_quota_total(total_games, len(opponents))
    ordered = sorted(opponents, key=lambda item: item.name)
    pairs = {opponent.name: 1 for opponent in ordered}
    remaining = total_games // 2 - len(ordered)
    cursor = 0
    while remaining:
        pairs[ordered[cursor % len(ordered)].name] += 1
        cursor += 1
        remaining -= 1
    return {name: count * 2 for name, count in sorted(pairs.items())}


def allocate_explicit_policy_quotas(
    total_games: int,
    opponents: Sequence[Opponent],
    policy_quotas: dict[str, int],
) -> dict[str, int]:
    """Resolve audited per-policy game quotas to train_ppo opponent names."""
    expected_policy_ids = {opponent.policy_id for opponent in opponents}
    provided_policy_ids = set(policy_quotas)
    if provided_policy_ids != expected_policy_ids:
        missing = sorted(expected_policy_ids - provided_policy_ids)
        extra = sorted(provided_policy_ids - expected_policy_ids)
        raise ValueError(
            "Explicit gold policy quotas must cover every eligible policy "
            f"exactly; missing={missing}, extra={extra}"
        )
    for policy_id, games in policy_quotas.items():
        if games <= 0 or games % 2:
            raise ValueError(
                f"{policy_id}: explicit quota must be positive and even"
            )
    if sum(policy_quotas.values()) != total_games:
        raise ValueError(
            "Explicit gold policy quotas sum to "
            f"{sum(policy_quotas.values())}, expected {total_games}"
        )
    return {
        opponent.name: policy_quotas[opponent.policy_id]
        for opponent in sorted(opponents, key=lambda item: item.name)
    }


def quota_seat_audit(quotas: dict[str, int]) -> dict[str, dict[str, int]]:
    audit: dict[str, dict[str, int]] = {}
    for name, games in sorted(quotas.items()):
        if games <= 0 or games % 2:
            raise ValueError(f"{name}: quota must be positive and even")
        audit[name] = {
            "games": games,
            "learner_seat_0": games // 2,
            "learner_seat_1": games // 2,
        }
    return audit


def make_bc_opponent(deck: Path) -> Opponent:
    return Opponent(
        name="bc",
        checkpoint=None,
        deck=deck,
        archetype="Marnie BC anchor",
        policy_id="base_bc_anchor",
        submission_id=None,
        quality_pass=True,
        deck_hash=compute_deck_hash(deck),
    )


def make_v3_opponent(checkpoint: Path, deck: Path) -> Opponent:
    return Opponent(
        name=expected_extra_opponent_name(checkpoint, deck),
        checkpoint=checkpoint,
        deck=deck,
        archetype="Marnie incumbent v3",
        policy_id="incumbent_v3",
        submission_id=None,
        quality_pass=True,
        checkpoint_sha256=file_sha256(checkpoint),
        deck_hash=compute_deck_hash(deck),
    )


def read_resume_update(checkpoint: Path) -> tuple[int, str]:
    """Read only metadata needed to continue global update numbering."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - training also needs torch.
        raise RuntimeError("PyTorch is required to inspect the resume checkpoint") from error
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "update" not in payload:
        raise ValueError(f"{checkpoint}: resume checkpoint has no update metadata")
    update = int(payload["update"])
    if update < 0:
        raise ValueError(f"{checkpoint}: invalid negative update")
    feature_version = str(payload.get("feature_version", ""))
    if not feature_version.startswith("ptcg-selfplay-ppo-"):
        raise ValueError(f"{checkpoint}: not a PPO resume checkpoint")
    return update, feature_version


def command_for_run(
    *,
    output_dir: Path,
    seed: int,
    end_update: int,
    schedule_start_update: int,
    profile: BudgetProfile,
    bc_checkpoint: Path,
    kl_reference_checkpoint: Path,
    learner_deck: Path,
    resume_checkpoint: Path,
    opponents: Sequence[Opponent],
    quotas: dict[str, int],
    device: str,
    resume_learner_weights: str = DEFAULT_RESUME_LEARNER_WEIGHTS,
    minibatch_size: int | None = None,
    trainable_scope: str = DEFAULT_TRAINABLE_SCOPE,
    actor_learning_rate: float = DEFAULT_ACTOR_LEARNING_RATE,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
    actor_reduction: str = DEFAULT_ACTOR_REDUCTION,
    advantage_normalization: str = DEFAULT_ADVANTAGE_NORMALIZATION,
    value_learning_rate: float = DEFAULT_VALUE_LEARNING_RATE,
    value_coefficient: float = DEFAULT_VALUE_COEFFICIENT,
    entropy_coefficient: float = DEFAULT_ENTROPY_COEFFICIENT,
    policy_temperature: float = DEFAULT_POLICY_TEMPERATURE,
    value_trunk_gradient_scale: float = (
        DEFAULT_VALUE_TRUNK_GRADIENT_SCALE
    ),
    actor_value_gradient_mode: str = DEFAULT_ACTOR_VALUE_GRADIENT_MODE,
    bc_kl_coefficient: float = DEFAULT_BC_KL_COEFFICIENT,
    bc_replay: BCReplaySettings | None = None,
) -> list[str]:
    resume_learner_weights = validate_resume_learner_weights(
        resume_learner_weights
    )
    effective_minibatch_size = (
        profile.minibatch_size
        if minibatch_size is None
        else validate_positive_int(minibatch_size, "Minibatch size")
    )
    if trainable_scope not in RUNNER_TRAINABLE_SCOPES:
        raise ValueError(
            f"Unsupported trainable scope: {trainable_scope!r}"
        )
    actor_learning_rate = validate_actor_learning_rate(actor_learning_rate)
    gae_lambda = validate_gae_lambda(gae_lambda)
    actor_reduction = validate_actor_reduction(actor_reduction)
    advantage_normalization = validate_advantage_normalization(
        advantage_normalization
    )
    if (
        actor_reduction == "quota_group_mean"
        and advantage_normalization != "global"
    ):
        raise ValueError(
            "Quota-group actor reduction requires global advantage "
            "normalization"
        )
    value_learning_rate = validate_value_learning_rate(value_learning_rate)
    value_coefficient = validate_value_coefficient(value_coefficient)
    entropy_coefficient = validate_entropy_coefficient(entropy_coefficient)
    policy_temperature = validate_policy_temperature(policy_temperature)
    value_trunk_gradient_scale = validate_value_trunk_gradient_scale(
        value_trunk_gradient_scale
    )
    actor_value_gradient_mode = validate_actor_value_gradient_mode(
        actor_value_gradient_mode
    )
    if (
        actor_value_gradient_mode == "actor_priority_value_pcgrad"
        and value_trunk_gradient_scale != 1.0
    ):
        raise ValueError(
            "actor_priority_value_pcgrad requires "
            "value_trunk_gradient_scale=1.0"
        )
    bc_kl_coefficient = validate_bc_kl_coefficient(bc_kl_coefficient)
    if bc_replay is None:
        bc_replay = make_bc_replay_settings()
    expected_names = {"bc", *(item.name for item in opponents)}
    if set(quotas) != expected_names:
        raise ValueError(
            "Quota names must exactly cover the BC anchor and every extra opponent"
        )
    quota_seat_audit(quotas)
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--bc-checkpoint",
        str(bc_checkpoint),
        "--kl-reference-checkpoint",
        str(kl_reference_checkpoint),
        "--deck",
        str(learner_deck),
    ]
    for opponent in opponents:
        if opponent.checkpoint is None:
            raise ValueError("Only the implicit BC anchor may omit a checkpoint")
        command.extend(
            (
                "--extra-opponent",
                str(opponent.checkpoint),
                str(opponent.deck),
            )
        )
    command.extend(
        (
            "--output-dir",
            str(output_dir),
            "--updates",
            str(end_update),
            "--environments",
            str(profile.environments),
            "--games-per-update",
            str(profile.games_per_update),
            "--ppo-epochs",
            str(profile.ppo_epochs),
            "--minibatch-size",
            str(effective_minibatch_size),
            "--learning-rate",
            str(actor_learning_rate),
            "--value-learning-rate",
            str(value_learning_rate),
            "--weight-decay",
            "1e-4",
            "--learning-rate-schedule",
            "constant",
            "--schedule-start-update",
            str(schedule_start_update),
            "--gamma",
            "1.0",
            "--gae-lambda",
            str(gae_lambda),
            "--advantage-normalization",
            advantage_normalization,
            "--clip-ratio",
            "0.15",
            "--value-coefficient",
            str(value_coefficient),
            "--value-trunk-gradient-scale",
            str(value_trunk_gradient_scale),
            "--actor-value-gradient-mode",
            actor_value_gradient_mode,
            "--entropy-coefficient",
            str(entropy_coefficient),
            "--max-grad-norm",
            "0.5",
            "--policy-temperature",
            str(policy_temperature),
            "--trainable-scope",
            trainable_scope,
            "--bc-kl-start",
            str(bc_kl_coefficient),
            "--bc-kl-end",
            str(bc_kl_coefficient),
            "--target-kl",
            "0.006",
            "--league-probability",
            "1.0",
            "--opponent-sampling",
            "per_game",
            "--opponent-quota-mode",
            "fixed",
            "--opponent-quota-seat-balance",
            "--ppo-objective",
            "standard",
            "--actor-reduction",
            actor_reduction,
            "--snapshot-interval",
            "1000000",
            "--max-pool-size",
            str(max(len(expected_names), 2)),
            "--eval-interval",
            str(end_update),
            "--eval-games",
            str(profile.eval_games),
            "--selection-aggregation",
            "mean",
            "--checkpoint-interval",
            str(end_update),
            "--max-game-decisions",
            "1000",
            "--seed",
            str(seed),
            "--resume",
            str(resume_checkpoint),
            "--resume-learner-weights",
            resume_learner_weights,
            "--reset-optimizer-on-resume",
            "--reset-opponent-quota-on-resume",
            "--skip-initial-eval",
            "--device",
            device,
        )
    )
    command.extend(bc_replay.command_args())
    for name, games in sorted(quotas.items()):
        command.extend(("--opponent-base-quota", name, str(games)))
    return command


def budget_dict(profile: BudgetProfile, seed_count: int) -> dict[str, Any]:
    per_seed_rollout_games = (
        profile.additional_updates * profile.games_per_update
    )
    return {
        "additional_updates_per_seed": profile.additional_updates,
        "games_per_update": profile.games_per_update,
        "rollout_games_per_seed": per_seed_rollout_games,
        "rollout_games_per_branch": per_seed_rollout_games * seed_count,
        "ppo_epochs": profile.ppo_epochs,
        "minibatch_size": profile.minibatch_size,
        "optimizer_pass_protocol": (
            f"{profile.ppo_epochs} PPO epochs per collected rollout"
        ),
        "seed_count": seed_count,
        "terminal_bc_mirror_eval_games_per_seed": profile.eval_games,
        "terminal_bc_mirror_eval_games_per_branch": (
            profile.eval_games * seed_count
        ),
    }


def build_plan(
    *,
    league_manifest: Path,
    output_root: Path,
    profile: BudgetProfile,
    seeds: Sequence[int],
    control_mode: str,
    bc_checkpoint: Path,
    kl_reference_checkpoint: Path,
    learner_deck: Path,
    resume_checkpoint: Path,
    resume_update: int,
    resume_feature_version: str,
    gold_opponents: Sequence[Opponent],
    quality_excluded: Sequence[dict[str, Any]],
    device: str,
    resume_learner_weights: str = DEFAULT_RESUME_LEARNER_WEIGHTS,
    minibatch_size: int | None = None,
    trainable_scope: str = DEFAULT_TRAINABLE_SCOPE,
    gold_policy_quotas: dict[str, int] | None = None,
    actor_learning_rate: float = DEFAULT_ACTOR_LEARNING_RATE,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
    actor_reduction: str = DEFAULT_ACTOR_REDUCTION,
    advantage_normalization: str = DEFAULT_ADVANTAGE_NORMALIZATION,
    value_learning_rate: float = DEFAULT_VALUE_LEARNING_RATE,
    value_coefficient: float = DEFAULT_VALUE_COEFFICIENT,
    entropy_coefficient: float = DEFAULT_ENTROPY_COEFFICIENT,
    policy_temperature: float = DEFAULT_POLICY_TEMPERATURE,
    value_trunk_gradient_scale: float = (
        DEFAULT_VALUE_TRUNK_GRADIENT_SCALE
    ),
    actor_value_gradient_mode: str = DEFAULT_ACTOR_VALUE_GRADIENT_MODE,
    bc_kl_coefficient: float = DEFAULT_BC_KL_COEFFICIENT,
    bc_replay_data: Path | None = None,
    bc_replay_split: str = "train",
    bc_replay_batches: int = DEFAULT_BC_REPLAY_BATCHES,
    bc_replay_batch_size: int = DEFAULT_BC_REPLAY_BATCH_SIZE,
    bc_replay_workers: int = DEFAULT_BC_REPLAY_WORKERS,
    bc_replay_steps: int = DEFAULT_BC_REPLAY_STEPS,
    bc_replay_lr_scale: float = DEFAULT_BC_REPLAY_LR_SCALE,
    bc_replay_loss: str = DEFAULT_BC_REPLAY_LOSS,
    bc_replay_order_context_weight: float = (
        DEFAULT_BC_REPLAY_ORDER_CONTEXT_WEIGHT
    ),
    bc_replay_non_context34_fixed_multi_action_order_weight: float = (
        DEFAULT_BC_REPLAY_NON_CONTEXT34_FIXED_MULTI_ACTION_ORDER_WEIGHT
    ),
    bc_replay_context34_rows_per_batch: int = (
        DEFAULT_BC_REPLAY_CONTEXT34_ROWS_PER_BATCH
    ),
) -> dict[str, Any]:
    resume_learner_weights = validate_resume_learner_weights(
        resume_learner_weights
    )
    if minibatch_size is not None:
        profile = replace(
            profile,
            minibatch_size=validate_positive_int(
                minibatch_size,
                "Minibatch size",
            ),
        )
    if trainable_scope not in RUNNER_TRAINABLE_SCOPES:
        raise ValueError(
            f"Unsupported trainable scope: {trainable_scope!r}"
        )
    actor_learning_rate = validate_actor_learning_rate(actor_learning_rate)
    gae_lambda = validate_gae_lambda(gae_lambda)
    actor_reduction = validate_actor_reduction(actor_reduction)
    advantage_normalization = validate_advantage_normalization(
        advantage_normalization
    )
    value_learning_rate = validate_value_learning_rate(value_learning_rate)
    value_coefficient = validate_value_coefficient(value_coefficient)
    entropy_coefficient = validate_entropy_coefficient(entropy_coefficient)
    policy_temperature = validate_policy_temperature(policy_temperature)
    value_trunk_gradient_scale = validate_value_trunk_gradient_scale(
        value_trunk_gradient_scale
    )
    actor_value_gradient_mode = validate_actor_value_gradient_mode(
        actor_value_gradient_mode
    )
    if (
        actor_value_gradient_mode == "actor_priority_value_pcgrad"
        and value_trunk_gradient_scale != 1.0
    ):
        raise ValueError(
            "actor_priority_value_pcgrad requires "
            "value_trunk_gradient_scale=1.0"
        )
    bc_kl_coefficient = validate_bc_kl_coefficient(bc_kl_coefficient)
    bc_replay = make_bc_replay_settings(
        data=bc_replay_data,
        split=bc_replay_split,
        batches=bc_replay_batches,
        batch_size=bc_replay_batch_size,
        workers=bc_replay_workers,
        steps=bc_replay_steps,
        lr_scale=bc_replay_lr_scale,
        loss=bc_replay_loss,
        order_context_weight=bc_replay_order_context_weight,
        non_context34_fixed_multi_action_order_weight=(
            bc_replay_non_context34_fixed_multi_action_order_weight
        ),
        context34_rows_per_batch=bc_replay_context34_rows_per_batch,
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be non-empty and unique")
    if profile.bc_anchor_games <= 0 or profile.bc_anchor_games % 2:
        raise ValueError("BC anchor quota must be positive and even")
    if profile.bc_anchor_games >= profile.games_per_update:
        raise ValueError("BC anchor quota must be below games per update")

    bc_anchor = make_bc_opponent(learner_deck)
    if control_mode == "bc_v3":
        control_extras = [make_v3_opponent(resume_checkpoint, learner_deck)]
    elif control_mode == "marnie_league":
        control_extras = [
            opponent
            for opponent in gold_opponents
            if "marnie" in opponent.archetype.casefold()
        ]
        if not control_extras:
            raise ValueError("No quality-passing Marnie clone for control A")
    else:
        raise ValueError(f"Unsupported control mode {control_mode!r}")

    remaining_games = profile.games_per_update - profile.bc_anchor_games
    control_extra_quotas = allocate_even_flat_quotas(
        remaining_games,
        control_extras,
    )
    control_quotas = {
        "bc": profile.bc_anchor_games,
        **control_extra_quotas,
    }
    if gold_policy_quotas is None:
        gold_extra_quotas = allocate_even_policy_quotas(
            remaining_games,
            gold_opponents,
        )
        gold_quota_mode = "archetype_balanced"
    else:
        gold_extra_quotas = allocate_explicit_policy_quotas(
            remaining_games,
            gold_opponents,
            gold_policy_quotas,
        )
        gold_quota_mode = "explicit_policy"
    gold_quotas = {"bc": profile.bc_anchor_games, **gold_extra_quotas}
    control_seats = quota_seat_audit(control_quotas)
    gold_seats = quota_seat_audit(gold_quotas)
    if sum(control_quotas.values()) != profile.games_per_update:
        raise RuntimeError("A quota sum does not equal games per update")
    if sum(gold_quotas.values()) != profile.games_per_update:
        raise RuntimeError("B quota sum does not equal games per update")

    end_update = resume_update + profile.additional_updates
    common_budget = budget_dict(profile, len(seeds))
    branches: dict[str, dict[str, Any]] = {}
    branch_specs = (
        ("A_marnie_control", control_extras, control_quotas, control_seats),
        ("B_gold_league", list(gold_opponents), gold_quotas, gold_seats),
    )
    for branch_name, extras, quotas, seats in branch_specs:
        runs: list[dict[str, Any]] = []
        for seed in seeds:
            run_output = (
                output_root / branch_name / f"seed-{int(seed)}"
            ).resolve()
            command = command_for_run(
                output_dir=run_output,
                seed=int(seed),
                end_update=end_update,
                schedule_start_update=resume_update + 1,
                profile=profile,
                bc_checkpoint=bc_checkpoint,
                kl_reference_checkpoint=kl_reference_checkpoint,
                learner_deck=learner_deck,
                resume_checkpoint=resume_checkpoint,
                opponents=extras,
                quotas=quotas,
                device=device,
                resume_learner_weights=resume_learner_weights,
                minibatch_size=profile.minibatch_size,
                trainable_scope=trainable_scope,
                actor_learning_rate=actor_learning_rate,
                gae_lambda=gae_lambda,
                actor_reduction=actor_reduction,
                advantage_normalization=advantage_normalization,
                value_learning_rate=value_learning_rate,
                value_coefficient=value_coefficient,
                entropy_coefficient=entropy_coefficient,
                policy_temperature=policy_temperature,
                value_trunk_gradient_scale=value_trunk_gradient_scale,
                actor_value_gradient_mode=actor_value_gradient_mode,
                bc_kl_coefficient=bc_kl_coefficient,
                bc_replay=bc_replay,
            )
            runs.append(
                {
                    "seed": int(seed),
                    "output_dir": str(run_output),
                    "command": command,
                }
            )
        branches[branch_name] = {
            "budget": common_budget,
            "opponents": [bc_anchor.public_dict()]
            + [opponent.public_dict() for opponent in extras],
            "opponent_quotas_per_update": quotas,
            "learner_seat_quotas_per_update": seats,
            "archetypes": sorted(
                {opponent.archetype for opponent in extras},
                key=str.casefold,
            ),
            "policy_ids": [opponent.policy_id for opponent in extras],
            "runs": runs,
        }

    eligible_archetypes = {
        opponent.archetype for opponent in gold_opponents
    }
    b_archetypes = set(branches["B_gold_league"]["archetypes"])
    eligible_policy_ids = {opponent.policy_id for opponent in gold_opponents}
    b_policy_ids = set(branches["B_gold_league"]["policy_ids"])
    a_budget = branches["A_marnie_control"]["budget"]
    b_budget = branches["B_gold_league"]["budget"]
    learner_source_invariants = (
        {
            "learner_initialization_is_full_fresh_bc": True,
            "resume_is_update_metadata_only": True,
        }
        if resume_learner_weights == "bc"
        else {
            "learner_initialization_is_full_incumbent_ppo": True,
            "resume_provides_update_metadata_and_learner_weights": True,
        }
    )
    invariants = {
        "branches_have_identical_budget": a_budget == b_budget,
        "branches_have_identical_seed_protocol": (
            [run["seed"] for run in branches["A_marnie_control"]["runs"]]
            == [run["seed"] for run in branches["B_gold_league"]["runs"]]
        ),
        "b_covers_every_quality_passing_archetype": (
            b_archetypes == eligible_archetypes
        ),
        "b_covers_every_quality_passing_policy": (
            b_policy_ids == eligible_policy_ids
        ),
        "all_quotas_are_exactly_two_seat_balanced": all(
            seat["learner_seat_0"] == seat["learner_seat_1"]
            and seat["games"]
            == seat["learner_seat_0"] + seat["learner_seat_1"]
            for branch in branches.values()
            for seat in branch["learner_seat_quotas_per_update"].values()
        ),
        "clone_quality_gate_enforced": all(
            opponent.quality_pass for opponent in gold_opponents
        ),
        **learner_source_invariants,
        "resume_learner_weights_matches_every_command": all(
            run["command"].count("--resume-learner-weights") == 1
            and run["command"][
                run["command"].index("--resume-learner-weights") + 1
            ]
            == resume_learner_weights
            for branch in branches.values()
            for run in branch["runs"]
        ),
        "minibatch_size_matches_every_command": all(
            run["command"].count("--minibatch-size") == 1
            and run["command"][
                run["command"].index("--minibatch-size") + 1
            ]
            == str(profile.minibatch_size)
            for branch in branches.values()
            for run in branch["runs"]
        ),
        "objective_is_standard_multi_opponent_ppo": True,
        "trainable_scope_matches_every_command": all(
            run["command"].count("--trainable-scope") == 1
            and run["command"][
                run["command"].index("--trainable-scope") + 1
            ]
            == trainable_scope
            for branch in branches.values()
            for run in branch["runs"]
        ),
        "no_external_publish_package_or_submit_action": True,
    }
    if not all(invariants.values()):
        failed = [name for name, passed in invariants.items() if not passed]
        raise RuntimeError(f"A/B preregistration invariant failed: {failed}")

    execution_order: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        order = (
            ["A_marnie_control", "B_gold_league"]
            if index % 2 == 0
            else ["B_gold_league", "A_marnie_control"]
        )
        execution_order.append({"seed": int(seed), "branches": order})

    return {
        "schema_version": PLAN_SCHEMA,
        "profile": profile.name,
        "control_mode": control_mode,
        "output_root": str(output_root.resolve()),
        "league_manifest": str(league_manifest.resolve()),
        "league_manifest_sha256": file_sha256(league_manifest),
        "inputs": {
            "train_script": str(TRAIN_SCRIPT),
            "train_script_sha256": file_sha256(TRAIN_SCRIPT),
            "python_executable": str(Path(sys.executable).resolve()),
            "bc_checkpoint": str(bc_checkpoint),
            "bc_checkpoint_sha256": file_sha256(bc_checkpoint),
            "learner_deck": str(learner_deck),
            "learner_deck_hash": compute_deck_hash(learner_deck),
            "resume_checkpoint": str(resume_checkpoint),
            "resume_checkpoint_sha256": file_sha256(resume_checkpoint),
            "resume_checkpoint_update": resume_update,
            "resume_checkpoint_feature_version": resume_feature_version,
            "resume_usage": (
                "global update metadata only"
                if resume_learner_weights == "bc"
                else "global update metadata and full learner weights"
            ),
            "learner_weight_source": (
                "full bc checkpoint"
                if resume_learner_weights == "bc"
                else "full incumbent PPO resume checkpoint"
            ),
            "learner_weight_checkpoint": str(
                bc_checkpoint
                if resume_learner_weights == "bc"
                else resume_checkpoint
            ),
            "learner_weight_checkpoint_sha256": file_sha256(
                bc_checkpoint
                if resume_learner_weights == "bc"
                else resume_checkpoint
            ),
            "kl_reference_checkpoint": str(kl_reference_checkpoint),
            "kl_reference_checkpoint_sha256": file_sha256(
                kl_reference_checkpoint
            ),
            "kl_reference_explicit": True,
        },
        "protocol": {
            "paired_seeds": [int(seed) for seed in seeds],
            "execution_order": execution_order,
            "common_budget": common_budget,
            "minibatch_size": profile.minibatch_size,
            "objective": "standard",
            "actor_reduction": actor_reduction,
            "advantage_normalization": advantage_normalization,
            "trainable_scope": trainable_scope,
            "actor_learning_rate": actor_learning_rate,
            "gae_lambda": gae_lambda,
            "value_learning_rate": value_learning_rate,
            "value_coefficient": value_coefficient,
            "entropy_coefficient": entropy_coefficient,
            "policy_temperature": policy_temperature,
            "value_trunk_gradient_scale": value_trunk_gradient_scale,
            "actor_value_gradient_mode": actor_value_gradient_mode,
            "bc_kl_coefficient": bc_kl_coefficient,
            "bc_replay": bc_replay.public_dict(),
            "fixed_exact_opponent_quotas": True,
            "gold_quota_mode": gold_quota_mode,
            "explicit_gold_policy_quotas": (
                dict(sorted(gold_policy_quotas.items()))
                if gold_policy_quotas is not None
                else None
            ),
            "strict_per_opponent_seat_balance": True,
            "resume_learner_weights": resume_learner_weights,
            "reset_optimizer_on_resume": True,
            "reset_opponent_quota_on_resume": True,
            "terminal_evaluation": (
                "same-size base-BC mirror only; gold PK is a separate "
                "frozen evaluation stage"
            ),
        },
        "quality_gate": {
            "included_policy_ids": sorted(eligible_policy_ids),
            "excluded": list(quality_excluded),
        },
        "branches": branches,
        "invariants": invariants,
        "safety": {
            "local_training_only": True,
            "allowed_child_program": str(TRAIN_SCRIPT),
            "network_calls": False,
            "uploads": False,
            "submission": False,
            "packaging": False,
            "uses_open_submission_code": False,
            "uses_quality_passing_public_replay_clones_only": True,
        },
    }


def preregistration_envelope(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PREREGISTRATION_SCHEMA,
        "created_at": utc_now(),
        "plan_sha256": canonical_json_sha256(plan),
        "plan": plan,
        "status": "preregistered_not_started",
    }


def validate_existing_preregistration(
    path: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    registration = read_json_object(path)
    if registration.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise ValueError(f"{path}: wrong preregistration schema")
    expected_hash = canonical_json_sha256(plan)
    if registration.get("plan_sha256") != expected_hash:
        raise ValueError(
            f"{path}: preregistered plan differs from the requested plan"
        )
    if registration.get("plan") != plan:
        raise ValueError(f"{path}: preregistration payload/hash mismatch")
    return registration


def execute_plan(
    plan: dict[str, Any],
    output_root: Path,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    """Execute only the preregistered local train_ppo child commands."""
    resolved_root = output_root.resolve()
    commands_by_key: dict[tuple[str, int], tuple[list[str], Path]] = {}
    seen_outputs: set[Path] = set()
    for branch_name, branch in plan["branches"].items():
        for run in branch["runs"]:
            seed = int(run["seed"])
            command = [str(value) for value in run["command"]]
            run_output = Path(run["output_dir"]).resolve()
            if (
                len(command) < 2
                or Path(command[0]).resolve() != Path(sys.executable).resolve()
                or Path(command[1]).resolve() != TRAIN_SCRIPT.resolve()
            ):
                raise RuntimeError("Refusing non-train_ppo child command")
            try:
                run_output.relative_to(resolved_root)
            except ValueError as error:
                raise RuntimeError(
                    f"Run output escapes the A/B root: {run_output}"
                ) from error
            if run_output in seen_outputs:
                raise RuntimeError(f"Duplicate run output: {run_output}")
            seen_outputs.add(run_output)
            try:
                output_flag_index = command.index("--output-dir")
                command_output = Path(command[output_flag_index + 1]).resolve()
            except (ValueError, IndexError) as error:
                raise RuntimeError("Child command lacks --output-dir") from error
            if command_output != run_output:
                raise RuntimeError(
                    "Child --output-dir differs from preregistered run output"
                )
            commands_by_key[(branch_name, seed)] = (command, run_output)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    atomic_write_json(
        output_root / "preregistration.json",
        preregistration,
    )
    started_at = utc_now()
    results: list[dict[str, Any]] = []
    branches = plan["branches"]
    for pair in plan["protocol"]["execution_order"]:
        seed = int(pair["seed"])
        for branch_name in pair["branches"]:
            command, run_output = commands_by_key[(branch_name, seed)]
            if run_output.exists():
                raise FileExistsError(run_output)
            log_path = output_root / f"{branch_name}_seed-{seed}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result = {
                "branch": branch_name,
                "seed": seed,
                "return_code": completed.returncode,
                "log": str(log_path),
                "output_dir": str(run_output),
            }
            results.append(result)
            atomic_write_json(
                output_root / "run_summary.json",
                {
                    "schema_version": SUMMARY_SCHEMA,
                    "started_at": started_at,
                    "completed_at": None,
                    "plan_sha256": preregistration["plan_sha256"],
                    "status": (
                        "running"
                        if completed.returncode == 0
                        else "failed"
                    ),
                    "results": results,
                    "safety": plan["safety"],
                },
            )
            if completed.returncode != 0:
                return read_json_object(output_root / "run_summary.json")
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "started_at": started_at,
        "completed_at": utc_now(),
        "plan_sha256": preregistration["plan_sha256"],
        "status": "completed",
        "results": results,
        "safety": plan["safety"],
    }
    atomic_write_json(output_root / "run_summary.json", summary)
    return summary


def validate_local_input(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}: {resolved}")
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preregister or run a local equal-budget PPO A/B. The default is "
            "a no-write dry run; training requires explicit --execute."
        )
    )
    parser.add_argument("--league-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="pilot")
    parser.add_argument(
        "--bc-anchor-games",
        type=positive_even_int_arg,
        help=(
            "Override the selected profile's even per-update BC opponent "
            "quota. The remaining games are allocated to the selected gold "
            "policies; by default the profile value is unchanged."
        ),
    )
    parser.add_argument(
        "--ppo-epochs",
        type=positive_int_arg,
        help=(
            "Override the selected profile's PPO epochs per collected "
            "rollout. By default the profile value is unchanged."
        ),
    )
    parser.add_argument(
        "--minibatch-size",
        type=positive_int_arg,
        help=(
            "Override the selected profile's optimizer minibatch size. "
            "By default the profile value is unchanged."
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=("bc_v3", "marnie_league"),
        default="bc_v3",
    )
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=DEFAULT_RESUME_CHECKPOINT,
    )
    parser.add_argument(
        "--resume-learner-weights",
        choices=RESUME_LEARNER_WEIGHT_SOURCES,
        default=DEFAULT_RESUME_LEARNER_WEIGHTS,
        help=(
            "Choose whether learner weights come from the full BC checkpoint "
            "or the PPO resume checkpoint; optimizer and opponent quotas are "
            "reset in both cases (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--kl-reference-checkpoint",
        type=Path,
        help=(
            "Explicit PPO KL anchor. If omitted, the fresh BC checkpoint is "
            "still passed explicitly to train_ppo as the KL reference."
        ),
    )
    parser.add_argument("--learner-deck", type=Path, default=DEFAULT_LEARNER_DECK)
    parser.add_argument(
        "--trainable-scope",
        choices=RUNNER_TRAINABLE_SCOPES,
        default=DEFAULT_TRAINABLE_SCOPE,
        help=(
            "Named trainable parameter scope passed to train_ppo.py "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--actor-learning-rate",
        type=actor_learning_rate_arg,
        default=DEFAULT_ACTOR_LEARNING_RATE,
        help=(
            "Actor optimizer learning rate passed to train_ppo.py as "
            "--learning-rate (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--actor-reduction",
        choices=ACTOR_REDUCTIONS,
        default=DEFAULT_ACTOR_REDUCTION,
        help=(
            "Actor surrogate reduction passed to train_ppo.py. "
            "quota_group_mean removes transition-count bias while preserving "
            "the frozen per-game opponent quotas; episode_mean gives every "
            "frozen game_uid equal actor weight."
        ),
    )
    parser.add_argument(
        "--advantage-normalization",
        choices=ADVANTAGE_NORMALIZATIONS,
        default=DEFAULT_ADVANTAGE_NORMALIZATION,
        help=(
            "Frozen-rollout advantage normalization passed to train_ppo.py. "
            "per_opponent standardizes independently within each frozen "
            "opponent group (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--gae-lambda",
        type=gae_lambda_arg,
        default=DEFAULT_GAE_LAMBDA,
        help=(
            "GAE lambda passed to train_ppo.py as --gae-lambda "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--value-learning-rate",
        type=value_learning_rate_arg,
        default=DEFAULT_VALUE_LEARNING_RATE,
        help=(
            "Value optimizer learning rate passed to train_ppo.py as "
            "--value-learning-rate (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--value-coefficient",
        type=value_coefficient_arg,
        default=DEFAULT_VALUE_COEFFICIENT,
        help=(
            "Value-loss coefficient passed to train_ppo.py as "
            "--value-coefficient (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--entropy-coefficient",
        type=entropy_coefficient_arg,
        default=DEFAULT_ENTROPY_COEFFICIENT,
        help=(
            "Entropy-bonus coefficient passed to train_ppo.py as "
            "--entropy-coefficient; zero disables the bonus "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--policy-temperature",
        type=policy_temperature_arg,
        default=DEFAULT_POLICY_TEMPERATURE,
        help=(
            "Rollout action-sampling temperature passed to train_ppo.py as "
            "--policy-temperature (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--bc-kl-coefficient",
        type=bc_kl_coefficient_arg,
        default=DEFAULT_BC_KL_COEFFICIENT,
        help=(
            "Constant frozen-BC KL loss coefficient passed to train_ppo.py "
            "as both --bc-kl-start and --bc-kl-end "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--value-trunk-gradient-scale",
        type=value_trunk_gradient_scale_arg,
        default=DEFAULT_VALUE_TRUNK_GRADIENT_SCALE,
        help=(
            "Scale only the standard-PPO value-loss gradient entering the "
            "shared trunk; 1.0 preserves prior behavior and 0.0 leaves only "
            "the value head trainable from value loss (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--actor-value-gradient-mode",
        choices=ACTOR_VALUE_GRADIENT_MODES,
        default=DEFAULT_ACTOR_VALUE_GRADIENT_MODE,
        help=(
            "Standard scalar actor/value gradients, or actor-priority "
            "projection of only conflicting critic gradients on shared "
            "parameters (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--bc-replay-data",
        type=Path,
        help=(
            "Filtered BC decision ZIP for auxiliary expert replay. Replay is "
            "disabled unless this, --bc-replay-batches, and "
            "--bc-replay-steps are all supplied/enabled."
        ),
    )
    parser.add_argument(
        "--bc-replay-split",
        choices=BC_REPLAY_SPLITS,
        default="train",
    )
    parser.add_argument(
        "--bc-replay-batches",
        type=nonnegative_int_arg,
        default=DEFAULT_BC_REPLAY_BATCHES,
    )
    parser.add_argument(
        "--bc-replay-batch-size",
        type=positive_int_arg,
        default=DEFAULT_BC_REPLAY_BATCH_SIZE,
    )
    parser.add_argument(
        "--bc-replay-workers",
        type=positive_int_arg,
        default=DEFAULT_BC_REPLAY_WORKERS,
    )
    parser.add_argument(
        "--bc-replay-steps",
        type=nonnegative_int_arg,
        default=DEFAULT_BC_REPLAY_STEPS,
    )
    parser.add_argument(
        "--bc-replay-lr-scale",
        type=replay_lr_scale_arg,
        default=DEFAULT_BC_REPLAY_LR_SCALE,
    )
    parser.add_argument(
        "--bc-replay-loss",
        choices=BC_REPLAY_LOSSES,
        default=DEFAULT_BC_REPLAY_LOSS,
    )
    parser.add_argument(
        "--bc-replay-order-context-weight",
        type=positive_float_arg,
        default=DEFAULT_BC_REPLAY_ORDER_CONTEXT_WEIGHT,
    )
    parser.add_argument(
        "--bc-replay-non-context34-fixed-multi-action-order-weight",
        type=positive_float_arg,
        default=(
            DEFAULT_BC_REPLAY_NON_CONTEXT34_FIXED_MULTI_ACTION_ORDER_WEIGHT
        ),
    )
    parser.add_argument(
        "--bc-replay-context34-rows-per-batch",
        type=nonnegative_int_arg,
        default=DEFAULT_BC_REPLAY_CONTEXT34_ROWS_PER_BATCH,
    )
    parser.add_argument(
        "--gold-policy-quota",
        action="append",
        nargs=2,
        metavar=("POLICY_ID", "GAMES"),
        help=(
            "Explicit even per-update quota for one eligible gold policy. "
            "When used, repeat it for every eligible policy; quotas must sum "
            "to games-per-update minus the BC anchor."
        ),
    )
    parser.add_argument(
        "--gold-policy-id",
        action="append",
        help=(
            "Explicit quality-passing Gold policy allowlist; repeat once per "
            "selected policy. Unselected quality-passing policies are "
            "recorded as protocol exclusions."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Override the profile seed set; repeat for paired replicates.",
    )
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the complete plan without writing or training.",
    )
    mode.add_argument(
        "--preregister-only",
        action="store_true",
        help="Write an immutable sidecar preregistration but do not train.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run both local branches after writing/verifying preregistration.",
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        help=(
            "Sidecar preregistration path. Defaults to "
            "<output-root>.preregistration.json."
        ),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    league_manifest = validate_local_input(
        args.league_manifest,
        "league manifest",
    )
    bc_checkpoint = validate_local_input(args.bc_checkpoint, "BC checkpoint")
    resume_checkpoint = validate_local_input(
        args.resume_checkpoint,
        "resume checkpoint",
    )
    learner_deck = validate_local_input(args.learner_deck, "learner deck")
    kl_reference_checkpoint = validate_local_input(
        args.kl_reference_checkpoint or bc_checkpoint,
        "KL reference checkpoint",
    )
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(TRAIN_SCRIPT)

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing A/B root: {output_root}"
        )
    preregistration_path = (
        args.preregistration.expanduser().resolve()
        if args.preregistration is not None
        else output_root.with_name(
            f"{output_root.name}.preregistration.json"
        )
    )
    if preregistration_path == output_root:
        raise ValueError("Preregistration path cannot equal output root")

    gold_opponents, quality_excluded, _ = load_quality_passing_league(
        league_manifest
    )
    gold_opponents, protocol_excluded = select_gold_policy_allowlist(
        gold_opponents,
        args.gold_policy_id,
    )
    quality_excluded.extend(protocol_excluded)
    gold_policy_quotas: dict[str, int] | None = None
    if args.gold_policy_quota:
        gold_policy_quotas = {}
        for policy_id_raw, games_raw in args.gold_policy_quota:
            policy_id = str(policy_id_raw).strip()
            if not policy_id:
                raise ValueError("Gold policy quota requires a policy_id")
            if policy_id in gold_policy_quotas:
                raise ValueError(f"Duplicate gold policy quota: {policy_id}")
            try:
                games = int(games_raw)
            except ValueError as error:
                raise ValueError(
                    f"{policy_id}: quota games must be an integer"
                ) from error
            gold_policy_quotas[policy_id] = games
    resume_update, resume_feature_version = read_resume_update(
        resume_checkpoint
    )
    profile = PROFILES[args.profile]
    if args.bc_anchor_games is not None:
        profile = replace(profile, bc_anchor_games=args.bc_anchor_games)
    if args.ppo_epochs is not None:
        profile = replace(profile, ppo_epochs=args.ppo_epochs)
    seeds = tuple(args.seed) if args.seed else profile.seeds
    plan = build_plan(
        league_manifest=league_manifest,
        output_root=output_root,
        profile=profile,
        seeds=seeds,
        control_mode=args.control_mode,
        bc_checkpoint=bc_checkpoint,
        kl_reference_checkpoint=kl_reference_checkpoint,
        learner_deck=learner_deck,
        resume_checkpoint=resume_checkpoint,
        resume_update=resume_update,
        resume_feature_version=resume_feature_version,
        gold_opponents=gold_opponents,
        quality_excluded=quality_excluded,
        device=args.device,
        resume_learner_weights=args.resume_learner_weights,
        minibatch_size=args.minibatch_size,
        trainable_scope=args.trainable_scope,
        gold_policy_quotas=gold_policy_quotas,
        actor_learning_rate=args.actor_learning_rate,
        gae_lambda=args.gae_lambda,
        actor_reduction=args.actor_reduction,
        advantage_normalization=args.advantage_normalization,
        value_learning_rate=args.value_learning_rate,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        policy_temperature=args.policy_temperature,
        value_trunk_gradient_scale=args.value_trunk_gradient_scale,
        actor_value_gradient_mode=args.actor_value_gradient_mode,
        bc_kl_coefficient=args.bc_kl_coefficient,
        bc_replay_data=args.bc_replay_data,
        bc_replay_split=args.bc_replay_split,
        bc_replay_batches=args.bc_replay_batches,
        bc_replay_batch_size=args.bc_replay_batch_size,
        bc_replay_workers=args.bc_replay_workers,
        bc_replay_steps=args.bc_replay_steps,
        bc_replay_lr_scale=args.bc_replay_lr_scale,
        bc_replay_loss=args.bc_replay_loss,
        bc_replay_order_context_weight=(
            args.bc_replay_order_context_weight
        ),
        bc_replay_non_context34_fixed_multi_action_order_weight=(
            args.bc_replay_non_context34_fixed_multi_action_order_weight
        ),
        bc_replay_context34_rows_per_batch=(
            args.bc_replay_context34_rows_per_batch
        ),
    )

    # No mode is intentionally equivalent to --dry-run.  Training is never an
    # implicit side effect of supplying paths.
    if args.dry_run or not (args.preregister_only or args.execute):
        envelope = preregistration_envelope(plan)
        print(json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True))
        return envelope

    if args.preregister_only:
        if preregistration_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite preregistration: {preregistration_path}"
            )
        envelope = preregistration_envelope(plan)
        atomic_write_json(preregistration_path, envelope)
        print(str(preregistration_path))
        return envelope

    if preregistration_path.exists():
        envelope = validate_existing_preregistration(
            preregistration_path,
            plan,
        )
    else:
        envelope = preregistration_envelope(plan)
        atomic_write_json(preregistration_path, envelope)
    return execute_plan(plan, output_root, envelope)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
