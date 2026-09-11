#!/usr/bin/env python3
"""Evaluate two BC/PPO checkpoints with the official PTCG engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from train_ppo import (
    BC_FEATURE_SOURCE_PATH,
    BC_FEATURE_SOURCE_SHA256,
    BC_FEATURE_VERSION,
    EntityOptionPolicy,
    MAX_ACTION_COUNT,
    PPO_FEATURE_VERSION,
    RawBattle,
    checkpoint_model_config,
    collate_features,
    compute_deck_hash,
    instantiate_model_from_checkpoint,
    live_feature,
    model_forward,
    read_deck,
    sample_ordered_actions,
    validate_checkpoint_deck,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


LOOP_DIAGNOSTIC_MAX_TAIL = 32
LOOP_DIAGNOSTIC_MAX_GAMES = 8
LOOP_DIAGNOSTIC_MAX_ACTION_ITEMS = MAX_ACTION_COUNT
LOOP_DIAGNOSTIC_SCHEMA = "h2h_max_decision_loop_diagnostic_v1"
HYBRID_ORDER_PRESERVE_CONTEXT = 34
POLICY_CANDIDATE = -1
POLICY_OPPONENT = 0
POLICY_CANDIDATE_BC_FALLBACK = -2


def apply_hybrid_action_order(
    action: list[int],
    *,
    select_context: int,
    enabled: bool,
) -> list[int]:
    """Sort set-like actions while preserving SKILL_ORDER context 34."""
    if not enabled or select_context == HYBRID_ORDER_PRESERVE_CONTEXT:
        return action
    return sorted(action)


def validate_policy_order_mode(
    *,
    label: str,
    canonical_order: bool,
    hybrid_order: bool,
) -> None:
    if canonical_order and hybrid_order:
        raise ValueError(
            f"{label} canonical order and hybrid order are mutually exclusive"
        )


def policy_order_mode(*, canonical_order: bool, hybrid_order: bool) -> str:
    validate_policy_order_mode(
        label="policy",
        canonical_order=canonical_order,
        hybrid_order=hybrid_order,
    )
    if canonical_order:
        return "canonical"
    if hybrid_order:
        return "hybrid"
    return "raw"


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected an integer >= 0")
    return parsed


def validate_candidate_bc_fallback(
    *,
    after_turn: int,
    model: torch.nn.Module | None,
) -> None:
    if not isinstance(after_turn, int) or isinstance(after_turn, bool):
        raise TypeError("candidate BC fallback turn must be an integer")
    if after_turn < 0:
        raise ValueError("candidate BC fallback turn must be >= 0")
    if after_turn == 0 and model is not None:
        raise ValueError(
            "candidate BC fallback model must be omitted when fallback is disabled"
        )
    if after_turn > 0 and model is None:
        raise ValueError(
            "candidate BC fallback model is required when fallback is enabled"
        )
    if model is not None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("candidate BC fallback model must be a torch module")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("candidate BC fallback model must be frozen")


def _diagnostic_int(value: object) -> int | None:
    """Return a small JSON-safe integer without propagating malformed input."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None


def summarize_loop_state(observation: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint an engine state without returning its potentially huge body."""
    public_state = {
        key: value
        for key, value in observation.items()
        if key != "search_begin_input"
    }
    rendered = json.dumps(
        public_state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    opaque = observation.get("search_begin_input")
    opaque_bytes = (
        opaque.encode("utf-8", errors="replace")
        if isinstance(opaque, str)
        else b""
    )
    current = observation.get("current")
    if not isinstance(current, dict):
        current = {}
    select = observation.get("select")
    if not isinstance(select, dict):
        select = {}
    options = select.get("option")
    return {
        "observation_sha256": hashlib.sha256(rendered).hexdigest(),
        "observation_json_bytes": len(rendered),
        "engine_state_sha256": (
            hashlib.sha256(opaque_bytes).hexdigest() if opaque_bytes else None
        ),
        "engine_state_bytes": len(opaque_bytes),
        "current": {
            "your_index": _diagnostic_int(current.get("yourIndex")),
            "turn": _diagnostic_int(current.get("turn")),
            "turn_action_count": _diagnostic_int(
                current.get("turnActionCount")
            ),
            "result": _diagnostic_int(current.get("result")),
        },
        "select": {
            "type": _diagnostic_int(select.get("type")),
            "context": _diagnostic_int(select.get("context")),
            "min_count": _diagnostic_int(select.get("minCount")),
            "max_count": _diagnostic_int(select.get("maxCount")),
            "option_count": len(options) if isinstance(options, list) else None,
        },
    }


def summarize_loop_decision(
    observation: dict[str, Any],
    action: list[int],
    *,
    decision: int,
    learner_seat: int,
) -> dict[str, Any]:
    """Create one bounded pre-step record for the diagnostic tail."""
    bounded_action = [
        _diagnostic_int(value)
        for value in action[:LOOP_DIAGNOSTIC_MAX_ACTION_ITEMS]
    ]
    current = observation.get("current")
    if not isinstance(current, dict):
        current = {}
    acting_seat = _diagnostic_int(current.get("yourIndex"))
    return {
        "decision": int(decision),
        "learner_seat": int(learner_seat),
        "acting_seat": acting_seat,
        "acting_policy": (
            None
            if acting_seat is None
            else ("candidate" if acting_seat == learner_seat else "opponent")
        ),
        "action_count": len(action),
        "action": bounded_action,
        "action_truncated": len(action) > len(bounded_action),
        "state": summarize_loop_state(observation),
    }


class LoopDiagnosticCollector:
    """Bounded, opt-in recorder for games reaching the decision limit."""

    def __init__(self, tail_decisions: int) -> None:
        if not isinstance(tail_decisions, int) or isinstance(
            tail_decisions, bool
        ):
            raise TypeError("loop diagnostic tail must be an integer")
        if not 0 <= tail_decisions <= LOOP_DIAGNOSTIC_MAX_TAIL:
            raise ValueError(
                "loop diagnostic tail must be between 0 and "
                f"{LOOP_DIAGNOSTIC_MAX_TAIL}"
            )
        self.tail_decisions = tail_decisions
        self._tails: dict[int, deque[dict[str, Any]]] = {}
        self._max_decision_events = 0
        self._games: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.tail_decisions > 0

    def start_game(self, battle: object) -> None:
        if self.enabled:
            self._tails[id(battle)] = deque(maxlen=self.tail_decisions)

    def record_decision(
        self,
        battle: object,
        observation: dict[str, Any],
        action: list[int],
        *,
        decision: int,
        learner_seat: int,
    ) -> None:
        if not self.enabled or len(self._games) >= LOOP_DIAGNOSTIC_MAX_GAMES:
            return
        tail = self._tails.get(id(battle))
        if tail is None:
            return
        tail.append(
            summarize_loop_decision(
                observation,
                action,
                decision=decision,
                learner_seat=learner_seat,
            )
        )

    def record_max_decisions(
        self,
        battle: object,
        observation: dict[str, Any],
        *,
        decisions: int,
        learner_seat: int,
    ) -> None:
        if not self.enabled:
            return
        self._max_decision_events += 1
        if len(self._games) >= LOOP_DIAGNOSTIC_MAX_GAMES:
            return
        tail = list(self._tails.get(id(battle), ()))
        state_hashes = [row["state"]["observation_sha256"] for row in tail]
        state_actions = [
            (row["state"]["observation_sha256"], tuple(row["action"]))
            for row in tail
        ]
        self._games.append(
            {
                "learner_seat": int(learner_seat),
                "decisions": int(decisions),
                "tail": tail,
                "terminal_state": summarize_loop_state(observation),
                "tail_repeat_summary": {
                    "entries": len(tail),
                    "unique_observation_states": len(set(state_hashes)),
                    "repeated_observation_entries": (
                        len(state_hashes) - len(set(state_hashes))
                    ),
                    "adjacent_same_observation_state": sum(
                        left == right
                        for left, right in zip(state_hashes, state_hashes[1:])
                    ),
                    "unique_state_action_pairs": len(set(state_actions)),
                    "repeated_state_action_entries": (
                        len(state_actions) - len(set(state_actions))
                    ),
                },
            }
        )

    def finish_game(self, battle: object) -> None:
        self._tails.pop(id(battle), None)

    def render(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "schema_version": LOOP_DIAGNOSTIC_SCHEMA,
            "tail_decisions": self.tail_decisions,
            "max_captured_games": LOOP_DIAGNOSTIC_MAX_GAMES,
            "max_decision_events": self._max_decision_events,
            "captured_games": len(self._games),
            "dropped_games": max(
                0,
                self._max_decision_events - len(self._games),
            ),
            "games": self._games,
        }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - margin, center + margin


def summarize_evaluation_stats(
    stats: Counter[str],
    games_target: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Render the legacy totals plus an auditable candidate-seat breakdown."""
    by_seat: dict[str, dict[str, Any]] = {}
    for seat in (0, 1):
        prefix = f"seat_{seat}_"
        wins = int(stats[prefix + "wins"])
        losses = int(stats[prefix + "losses"])
        draws = int(stats[prefix + "draws"])
        games = int(stats[prefix + "valid_games"])
        if games != wins + losses + draws:
            raise RuntimeError(
                f"Candidate seat {seat} outcome total does not match games"
            )
        low, high = wilson_interval(wins, games)
        by_seat[str(seat)] = {
            "valid_games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / max(games, 1),
            "wilson_95_low": low,
            "wilson_95_high": high,
        }

    valid_games = int(stats["valid_games"])
    wins = int(stats["wins"])
    losses = int(stats["losses"])
    draws = int(stats["draws"])
    for key, total in (
        ("valid_games", valid_games),
        ("wins", wins),
        ("losses", losses),
        ("draws", draws),
    ):
        if sum(int(row[key]) for row in by_seat.values()) != total:
            raise RuntimeError(
                f"Aggregate {key} does not equal the two candidate seats"
            )
    decisive = wins + losses
    requested_targets = {
        "0": games_target // 2,
        "1": games_target - games_target // 2,
    }
    actual_games = {
        seat: int(row["valid_games"]) for seat, row in by_seat.items()
    }
    even_request = games_target % 2 == 0
    seat_balance = {
        "assignment_mode": "exact_valid_game_quota_v1",
        "requested_games_by_candidate_seat": requested_targets,
        "actual_valid_games_by_candidate_seat": actual_games,
        "requested_games_even": even_request,
        "strict_even_balance_required": even_request,
        "strict_even_balance_verified": (
            even_request
            and actual_games == requested_targets
            and actual_games["0"] == actual_games["1"]
        ),
        "actual_valid_game_gap": abs(actual_games["0"] - actual_games["1"]),
        "aggregate_equals_seat_sum_verified": True,
    }
    invalid_by_reason = {
        key.removeprefix("invalid_reason_"): int(value)
        for key, value in sorted(stats.items())
        if key.startswith("invalid_reason_") and value
    }
    invalid_by_candidate_seat: dict[str, dict[str, Any]] = {}
    for seat in (0, 1):
        prefix = f"seat_{seat}_invalid_reason_"
        reasons = {
            key.removeprefix(prefix): int(value)
            for key, value in sorted(stats.items())
            if key.startswith(prefix) and value
        }
        invalid_by_candidate_seat[str(seat)] = {
            "invalid_games": sum(reasons.values()),
            "by_reason": reasons,
        }
    audited_invalid_games = sum(
        int(row["invalid_games"])
        for row in invalid_by_candidate_seat.values()
    )
    invalid_error_messages = {
        key.removeprefix("invalid_error_message::"): int(value)
        for key, value in sorted(stats.items())
        if key.startswith("invalid_error_message::") and value
    }
    attempted_games = valid_games + int(stats["invalid_games"])
    conservative_low, conservative_high = wilson_interval(
        wins,
        attempted_games,
    )
    return {
        "valid_games": valid_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "invalid_games": int(stats["invalid_games"]),
        "win_rate": wins / max(valid_games, 1),
        "decisive_win_rate": wins / max(decisive, 1),
        "mean_decisions": stats["decisions"] / max(valid_games, 1),
        "max_active_games": int(stats["max_active_games"]),
        "seconds": elapsed_seconds,
        "by_candidate_seat": by_seat,
        "seat_balance": seat_balance,
        "invalid_by_reason": invalid_by_reason,
        "invalid_by_candidate_seat": invalid_by_candidate_seat,
        "invalid_error_messages": invalid_error_messages,
        "invalid_audit_complete": (
            audited_invalid_games == int(stats["invalid_games"])
        ),
        "conservative_invalid_as_loss": {
            "trials": attempted_games,
            "wins": wins,
            "nonwins": attempted_games - wins,
            "win_rate": wins / max(attempted_games, 1),
            "wilson_95_low": conservative_low,
            "wilson_95_high": conservative_high,
            "gate_ci_low_above_0_5": conservative_low > 0.5,
            "invalid_rate": int(stats["invalid_games"])
            / max(attempted_games, 1),
            "interpretation": (
                "every invalid game is counted as a candidate loss"
            ),
        },
    }


@torch.no_grad()
def evaluate_head_to_head_with_seats(
    current_model: torch.nn.Module,
    opponent_model: torch.nn.Module,
    deck: list[int],
    model_config: dict[str, Any],
    device: torch.device,
    games_target: int,
    environments: int,
    max_game_decisions: int,
    current_canonical_order: bool = False,
    opponent_canonical_order: bool = True,
    opponent_deck: list[int] | None = None,
    loop_diagnostic_tail: int = 0,
    current_hybrid_order: bool = False,
    opponent_hybrid_order: bool = False,
    candidate_bc_fallback_model: torch.nn.Module | None = None,
    candidate_bc_fallback_after_turn: int = 0,
    opponent_model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate with exact valid-game quotas for each candidate seat.

    The training helper balances attempted games in pairs. Invalid games can
    therefore leave its valid-game totals slightly imbalanced. This local
    evaluator retries the same seat after an invalid game until each requested
    valid-game quota is met. No training code is changed.
    """
    validate_policy_order_mode(
        label="candidate",
        canonical_order=current_canonical_order,
        hybrid_order=current_hybrid_order,
    )
    validate_policy_order_mode(
        label="opponent",
        canonical_order=opponent_canonical_order,
        hybrid_order=opponent_hybrid_order,
    )
    validate_candidate_bc_fallback(
        after_turn=candidate_bc_fallback_after_turn,
        model=candidate_bc_fallback_model,
    )
    current_model.eval()
    opponent_model.eval()
    resolved_opponent_model_config = (
        model_config
        if opponent_model_config is None
        else opponent_model_config
    )
    if candidate_bc_fallback_model is not None:
        candidate_bc_fallback_model.eval()
    learner_deck_hash = compute_deck_hash(deck)
    resolved_opponent_deck = deck if opponent_deck is None else opponent_deck
    opponent_deck_hash = compute_deck_hash(resolved_opponent_deck)
    started = time.time()
    stats: Counter[str] = Counter()
    loop_diagnostic = LoopDiagnosticCollector(loop_diagnostic_tail)
    games: list[tuple[RawBattle, int, int]] = []
    seat_targets = {
        0: games_target // 2,
        1: games_target - games_target // 2,
    }
    seat_started: Counter[int] = Counter()
    seat_valid: Counter[int] = Counter()
    next_tie_seat = 0

    def record_invalid(
        reason: str,
        learner_seat: int,
        error_message: object | None = None,
    ) -> None:
        stats["invalid_games"] += 1
        stats[f"invalid_reason_{reason}"] += 1
        stats[f"seat_{learner_seat}_invalid_reason_{reason}"] += 1
        if error_message is not None:
            rendered = str(error_message).replace("\n", " ")[:200]
            stats[f"invalid_error_message::{rendered}"] += 1

    def next_learner_seat() -> int | None:
        nonlocal next_tie_seat
        active_by_seat: Counter[int] = Counter(
            current_seat for _, current_seat, _ in games
        )
        capacity = {
            seat: seat_targets[seat] - seat_valid[seat] - active_by_seat[seat]
            for seat in (0, 1)
        }
        candidates = [seat for seat in (0, 1) if capacity[seat] > 0]
        if not candidates:
            return None
        maximum = max(capacity[seat] for seat in candidates)
        tied = [seat for seat in candidates if capacity[seat] == maximum]
        if len(tied) == 1:
            chosen = tied[0]
        else:
            chosen = next_tie_seat if next_tie_seat in tied else tied[0]
            next_tie_seat = 1 - chosen
        return chosen

    def start_evaluation_game(learner_seat: int) -> tuple[RawBattle, int, int]:
        seat_decks = (
            (deck, resolved_opponent_deck)
            if learner_seat == 0
            else (resolved_opponent_deck, deck)
        )
        seat_started[learner_seat] += 1
        battle = RawBattle(seat_decks[0], seat_decks[1])
        loop_diagnostic.start_game(battle)
        return battle, learner_seat, 0

    def refill_games() -> None:
        while len(games) < min(environments, games_target - sum(seat_valid.values())):
            learner_seat = next_learner_seat()
            if learner_seat is None:
                break
            games.append(start_evaluation_game(learner_seat))
        stats["max_active_games"] = max(
            stats["max_active_games"],
            len(games),
        )

    refill_games()
    while games:
        groups: dict[
            int,
            list[tuple[int, RawBattle, int, dict[str, Any], int]],
        ] = defaultdict(list)
        invalid_indices: dict[int, tuple[str, object | None]] = {}
        for game_index, (battle, current_seat, decisions) in enumerate(games):
            observation = battle.observation
            if battle.result != -1:
                continue
            select = observation.get("select")
            current = observation.get("current") or {}
            if not isinstance(select, dict):
                invalid_indices[game_index] = ("missing_select", None)
                continue
            options = select.get("option")
            minimum = int(select.get("minCount", 0) or 0)
            maximum = int(select.get("maxCount", 0) or 0)
            if not isinstance(options, list):
                invalid_indices[game_index] = ("missing_options", None)
                continue
            if not options:
                invalid_indices[game_index] = ("empty_options", None)
                continue
            if (
                minimum < 0
                or maximum < minimum
                or maximum > len(options)
            ):
                invalid_indices[game_index] = (
                    "invalid_select_count_range",
                    f"min={minimum} max={maximum} options={len(options)}",
                )
                continue
            if maximum > MAX_ACTION_COUNT:
                invalid_indices[game_index] = (
                    "max_action_count_exceeded",
                    f"max={maximum} limit={MAX_ACTION_COUNT}",
                )
                continue
            select_context = int(select.get("context", 0) or 0)
            seat = int(current.get("yourIndex", 0) or 0)
            candidate_acting = seat == current_seat
            acting_model_config = (
                model_config
                if candidate_acting
                else resolved_opponent_model_config
            )
            feature = live_feature(
                observation,
                acting_model_config,
                (
                    learner_deck_hash
                    if seat == current_seat
                    else opponent_deck_hash
                ),
            )
            if feature is None:
                invalid_indices[game_index] = ("feature_none", None)
                continue
            use_candidate_bc_fallback = (
                candidate_acting
                and candidate_bc_fallback_after_turn > 0
                and int(current.get("turn", 0) or 0)
                >= candidate_bc_fallback_after_turn
            )
            if use_candidate_bc_fallback:
                policy_index = POLICY_CANDIDATE_BC_FALLBACK
            elif candidate_acting:
                policy_index = POLICY_CANDIDATE
            else:
                policy_index = POLICY_OPPONENT
            groups[policy_index].append(
                (game_index, battle, current_seat, feature, select_context)
            )

        actions_by_index: dict[int, list[int]] = {}
        candidate_bc_fallback_indices: set[int] = set()
        for policy_index, items in groups.items():
            policy_model_config = (
                resolved_opponent_model_config
                if policy_index == POLICY_OPPONENT
                else model_config
            )
            batch = collate_features(
                [item[3] for item in items],
                policy_model_config,
                device,
            )
            if policy_index == POLICY_CANDIDATE_BC_FALLBACK:
                if candidate_bc_fallback_model is None:
                    raise RuntimeError(
                        "candidate BC fallback route has no fallback model"
                    )
                model = candidate_bc_fallback_model
                canonical_order = True
                hybrid_order = False
            elif policy_index == POLICY_CANDIDATE:
                model = current_model
                canonical_order = current_canonical_order
                hybrid_order = current_hybrid_order
            else:
                model = opponent_model
                canonical_order = opponent_canonical_order
                hybrid_order = opponent_hybrid_order
            outputs = model_forward(model, batch, device)
            actions, _, _, _ = sample_ordered_actions(
                outputs,
                batch,
                deterministic=True,
                canonicalize_order=canonical_order,
            )
            for item, action in zip(items, actions):
                actions_by_index[item[0]] = apply_hybrid_action_order(
                    action,
                    select_context=item[4],
                    enabled=hybrid_order,
                )
                if policy_index == POLICY_CANDIDATE_BC_FALLBACK:
                    candidate_bc_fallback_indices.add(item[0])

        finished_indices = set(invalid_indices)
        for game_index, (reason, error_message) in invalid_indices.items():
            _, current_seat, _ = games[game_index]
            record_invalid(reason, current_seat, error_message)
        for game_index, (battle, current_seat, decisions) in enumerate(games):
            if game_index in finished_indices:
                continue
            if game_index not in actions_by_index:
                record_invalid("missing_model_action", current_seat)
                finished_indices.add(game_index)
                continue
            action = actions_by_index[game_index]
            if game_index in candidate_bc_fallback_indices:
                stats["candidate_bc_fallback_decisions"] += 1
                stats[
                    f"seat_{current_seat}_candidate_bc_fallback_decisions"
                ] += 1
            loop_diagnostic.record_decision(
                battle,
                battle.observation,
                action,
                decision=decisions + 1,
                learner_seat=current_seat,
            )
            _, error = battle.step(action)
            decisions += 1
            games[game_index] = (battle, current_seat, decisions)
            stats["decisions"] += 1
            if error:
                record_invalid("battle_step_error", current_seat, error)
                finished_indices.add(game_index)
                continue
            result = battle.result
            if result != -1:
                stats["valid_games"] += 1
                stats[f"seat_{current_seat}_valid_games"] += 1
                seat_valid[current_seat] += 1
                if result == current_seat:
                    outcome = "wins"
                elif result == 2:
                    outcome = "draws"
                else:
                    outcome = "losses"
                stats[outcome] += 1
                stats[f"seat_{current_seat}_{outcome}"] += 1
                finished_indices.add(game_index)
                continue
            if decisions >= max_game_decisions:
                loop_diagnostic.record_max_decisions(
                    battle,
                    battle.observation,
                    decisions=decisions,
                    learner_seat=current_seat,
                )
                record_invalid("max_game_decisions", current_seat)
                finished_indices.add(game_index)

        for game_index in sorted(finished_indices, reverse=True):
            battle, _, _ = games[game_index]
            loop_diagnostic.finish_game(battle)
            battle.close()
            games.pop(game_index)
        refill_games()
        if stats["invalid_games"] > games_target * 4:
            for battle, _, _ in games:
                battle.close()
            raise RuntimeError(
                "Evaluation exceeded the invalid-game budget: "
                f"valid={stats['valid_games']} "
                f"invalid={stats['invalid_games']}"
            )

    if seat_valid != Counter(seat_targets):
        raise RuntimeError(
            "Evaluation did not meet exact candidate-seat valid-game quotas: "
            f"target={seat_targets} actual={dict(seat_valid)} "
            f"started={dict(seat_started)}"
        )
    result = summarize_evaluation_stats(
        stats,
        games_target,
        time.time() - started,
    )
    rendered_loop_diagnostic = loop_diagnostic.render()
    if rendered_loop_diagnostic is not None:
        result["loop_diagnostic"] = rendered_loop_diagnostic
    if candidate_bc_fallback_after_turn > 0:
        fallback_by_seat = {
            str(seat): int(
                stats[f"seat_{seat}_candidate_bc_fallback_decisions"]
            )
            for seat in (0, 1)
        }
        fallback_decisions = int(stats["candidate_bc_fallback_decisions"])
        if sum(fallback_by_seat.values()) != fallback_decisions:
            raise RuntimeError(
                "Candidate BC fallback seat totals do not match decisions"
            )
        result["candidate_bc_fallback"] = {
            "enabled": True,
            "after_turn": candidate_bc_fallback_after_turn,
            "canonical_order": True,
            "decisions": fallback_decisions,
            "by_candidate_seat": {
                seat: {"decisions": decisions}
                for seat, decisions in fallback_by_seat.items()
            },
            "seat_sum_verified": True,
        }
    return result


def load_checkpoint(
    path: Path,
    deck_hash: str,
    deck_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    feature_version = checkpoint.get("feature_version")
    if feature_version not in {BC_FEATURE_VERSION, PPO_FEATURE_VERSION}:
        raise ValueError(
            f"{path} has unsupported feature version {feature_version!r}"
        )
    model_config = checkpoint_model_config(checkpoint)
    validate_checkpoint_deck(checkpoint, deck_hash, path, deck_path)
    if feature_version == BC_FEATURE_VERSION:
        model = instantiate_model_from_checkpoint(
            checkpoint,
            checkpoint,
            device,
        )
    else:
        model = EntityOptionPolicy(
            hash_size=int(model_config["hash_size"]),
            categorical_dim=int(model_config["categorical_dim"]),
            model_dim=int(model_config["model_dim"]),
            layers=int(model_config["layers"]),
            heads=int(model_config["heads"]),
            dropout=float(model_config["dropout"]),
            max_state_entities=int(model_config["max_state_entities"]),
        )
        old_count_layer = model.count_head[-1]
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"{path} is missing model_state_dict")
        count_weight = state_dict.get("count_head.2.weight")
        if not isinstance(count_weight, torch.Tensor) or count_weight.ndim != 2:
            raise ValueError(f"{path} has an invalid count head")
        count_classes = int(count_weight.shape[0])
        if old_count_layer.out_features != count_classes:
            model.count_head[-1] = torch.nn.Linear(
                old_count_layer.in_features,
                count_classes,
            )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    return checkpoint, model, model_config


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=Path("artifacts/bc_marnie_luca_orbit_v5/best.pt"),
    )
    parser.add_argument(
        "--candidate-deck",
        type=Path,
        default=Path("data/decks/marnie_grimmsnarl_froslass_luca.csv"),
    )
    parser.add_argument(
        "--opponent-deck",
        type=Path,
        default=Path("data/decks/marnie_grimmsnarl_froslass_luca.csv"),
    )
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--environments", type=int, default=32)
    parser.add_argument("--max-game-decisions", type=int, default=1000)
    parser.add_argument(
        "--loop-diagnostic-tail",
        type=int,
        default=0,
        help=(
            "Opt-in bounded trace length for games that hit "
            "--max-game-decisions (0 disables; max 32)"
        ),
    )
    parser.add_argument(
        "--candidate-bc-fallback-after-turn",
        type=nonnegative_int,
        default=0,
        metavar="N",
        help=(
            "Use the frozen BC anchor with canonical action order for "
            "candidate decisions at current.turn >= N (0 disables)"
        ),
    )
    candidate_order = parser.add_mutually_exclusive_group()
    candidate_order.add_argument(
        "--candidate-canonical-order",
        action="store_true",
    )
    candidate_order.add_argument(
        "--candidate-hybrid-order",
        action="store_true",
        help=(
            "Preserve Plackett-Luce order only for select.context=34; "
            "sort actions for every other candidate context"
        ),
    )
    opponent_order = parser.add_mutually_exclusive_group()
    opponent_order.add_argument(
        "--opponent-canonical-order",
        action="store_true",
    )
    opponent_order.add_argument(
        "--opponent-hybrid-order",
        action="store_true",
        help=(
            "Preserve Plackett-Luce order only for select.context=34; "
            "sort actions for every other opponent context"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    for path in (
        args.candidate,
        args.opponent,
        args.bc_checkpoint,
        args.candidate_deck,
        args.opponent_deck,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.games < 1
        or args.environments < 1
        or args.max_game_decisions < 1
    ):
        raise ValueError(
            "--games, --environments, and --max-game-decisions must be positive"
        )
    LoopDiagnosticCollector(args.loop_diagnostic_tail)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    candidate_deck = read_deck(args.candidate_deck)
    opponent_deck = read_deck(args.opponent_deck)
    candidate_deck_hash = compute_deck_hash(candidate_deck)
    opponent_deck_hash = compute_deck_hash(opponent_deck)
    bc_checkpoint = torch.load(
        args.bc_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if bc_checkpoint.get("feature_version") != BC_FEATURE_VERSION:
        raise ValueError(f"{args.bc_checkpoint} is not the expected BC anchor")
    bc_model_config = checkpoint_model_config(bc_checkpoint)

    candidate_checkpoint, candidate_model, candidate_model_config = load_checkpoint(
        args.candidate,
        candidate_deck_hash,
        args.candidate_deck,
        device,
    )
    opponent_checkpoint, opponent_model, opponent_model_config = load_checkpoint(
        args.opponent,
        opponent_deck_hash,
        args.opponent_deck,
        device,
    )
    candidate_bc_fallback_model: torch.nn.Module | None = None
    if args.candidate_bc_fallback_after_turn > 0:
        if bc_model_config != candidate_model_config:
            raise ValueError(
                "Candidate BC fallback model config does not match the "
                "candidate model config"
            )
        validate_checkpoint_deck(
            bc_checkpoint,
            candidate_deck_hash,
            args.bc_checkpoint,
            args.candidate_deck,
        )
        candidate_bc_fallback_model = instantiate_model_from_checkpoint(
            bc_checkpoint,
            bc_checkpoint,
            device,
        )
        candidate_bc_fallback_model.requires_grad_(False)
        candidate_bc_fallback_model.eval()

    started_at = datetime.now(timezone.utc)
    started = time.time()
    evaluation = evaluate_head_to_head_with_seats(
        candidate_model,
        opponent_model,
        candidate_deck,
        candidate_model_config,
        device,
        args.games,
        min(args.environments, args.games),
        args.max_game_decisions,
        current_canonical_order=args.candidate_canonical_order,
        opponent_canonical_order=args.opponent_canonical_order,
        opponent_deck=opponent_deck,
        loop_diagnostic_tail=args.loop_diagnostic_tail,
        current_hybrid_order=args.candidate_hybrid_order,
        opponent_hybrid_order=args.opponent_hybrid_order,
        candidate_bc_fallback_model=candidate_bc_fallback_model,
        candidate_bc_fallback_after_turn=(
            args.candidate_bc_fallback_after_turn
        ),
        opponent_model_config=opponent_model_config,
    )
    elapsed = time.time() - started
    ci_low, ci_high = wilson_interval(
        int(evaluation["wins"]),
        int(evaluation["valid_games"]),
    )
    result = {
        "bc_feature_source": {
            "path": str(BC_FEATURE_SOURCE_PATH),
            "sha256": BC_FEATURE_SOURCE_SHA256,
            "feature_version": BC_FEATURE_VERSION,
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": sha256_file(args.candidate),
            "feature_version": candidate_checkpoint.get("feature_version"),
            "update": candidate_checkpoint.get("update"),
            "deck": str(args.candidate_deck.resolve()),
            "deck_hash": candidate_deck_hash,
            "canonical_order": args.candidate_canonical_order,
            "hybrid_order": args.candidate_hybrid_order,
            "order_mode": policy_order_mode(
                canonical_order=args.candidate_canonical_order,
                hybrid_order=args.candidate_hybrid_order,
            ),
            "model_config": candidate_model_config,
        },
        "opponent": {
            "path": str(args.opponent.resolve()),
            "sha256": sha256_file(args.opponent),
            "feature_version": opponent_checkpoint.get("feature_version"),
            "update": opponent_checkpoint.get("update"),
            "deck": str(args.opponent_deck.resolve()),
            "deck_hash": opponent_deck_hash,
            "canonical_order": args.opponent_canonical_order,
            "hybrid_order": args.opponent_hybrid_order,
            "order_mode": policy_order_mode(
                canonical_order=args.opponent_canonical_order,
                hybrid_order=args.opponent_hybrid_order,
            ),
            "model_config": opponent_model_config,
        },
        "engine": {
            "games_requested": args.games,
            "environments": args.environments,
            "max_game_decisions": args.max_game_decisions,
            "engine_seed_control": False,
            "engine_seed_warning": (
                "Python/Torch seeds do not control the official battle "
                "engine RNG; repeated runs are independent, not reproducible "
                "or paired engine-seed trials."
            ),
            "python_torch_seed": args.seed,
            "device": str(device),
        },
        "evaluation": {
            **evaluation,
            "wilson_95_low": ci_low,
            "wilson_95_high": ci_high,
            "gate_ci_low_above_0_5": ci_low > 0.5,
        },
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
    }
    if args.loop_diagnostic_tail:
        result["engine"]["loop_diagnostic_tail"] = args.loop_diagnostic_tail
    if args.candidate_bc_fallback_after_turn > 0:
        result["candidate"]["bc_fallback"] = {
            "checkpoint": str(args.bc_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.bc_checkpoint),
            "feature_version": bc_checkpoint.get("feature_version"),
            "after_turn": args.candidate_bc_fallback_after_turn,
            "canonical_order": True,
            "frozen": True,
        }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
