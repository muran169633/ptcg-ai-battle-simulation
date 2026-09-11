#!/usr/bin/env python3
"""Pre-registered public-observation separability audit for a PPO router.

The engine supplies the frozen probe with ordinary agent observations. At the
first learner selection after the opponent's first, second, and third normal
turns, this tool captures either the legacy stateless router feature or the
explicitly selected causal public-opponent-history-v2 feature. The opponent
label is stored separately and is never passed to either feature call.

Only the second-opponent-turn snapshot is used for the go/no-go decision. The
first and third snapshots are diagnostics and cannot rescue a failed main gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ppo_router import (
    PUBLIC_HISTORY_V2_FEATURE_DIM,
    PUBLIC_HISTORY_V2_FEATURE_VERSION,
    PUBLIC_ROUTER_FEATURE_DIM,
    PUBLIC_ROUTER_FEATURE_VERSION,
    PublicOpponentHistoryV2,
    public_router_features,
)
from train_ppo import (
    BC_FEATURE_VERSION,
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


PREFIXES = (1, 2, 3)
PRIMARY_PREFIX = 2
ROUTER_FEATURE_MODES = ("stateless-v1", "public-history-v2")


def opaque_group_id_pool(total_groups: int, seed: int) -> list[int]:
    """Return reproducible join keys with no class-coded numeric ranges."""

    if total_groups < 1:
        raise ValueError("total_groups must be positive")
    identifiers = list(range(total_groups))
    random.Random(seed).shuffle(identifiers)
    return identifiers


def collection_quality_gates(
    stats_a: dict[str, Any],
    stats_b: dict[str, Any],
) -> dict[str, bool]:
    """Evaluate pre-registered collection integrity symmetrically."""

    rate_a = float(stats_a["replacement_rate"])
    rate_b = float(stats_b["replacement_rate"])
    return {
        "class_0_replacement_rate": rate_a <= 0.02,
        "class_1_replacement_rate": rate_b <= 0.02,
        "replacement_rate_gap": abs(rate_a - rate_b) <= 0.02,
        "class_0_start_errors_zero": int(stats_a.get("start_errors", 0)) == 0,
        "class_1_start_errors_zero": int(stats_b.get("start_errors", 0)) == 0,
        "class_0_history_turn_regression_resets_zero": (
            int(stats_a.get("history_turn_regression_resets", 0)) == 0
        ),
        "class_1_history_turn_regression_resets_zero": (
            int(stats_b.get("history_turn_regression_resets", 0)) == 0
        ),
        "class_0_seat_balance": (
            abs(
                int(stats_a["learner_seat_0_trajectories"])
                - int(stats_a["learner_seat_1_trajectories"])
            )
            <= 1
        ),
        "class_1_seat_balance": (
            abs(
                int(stats_b["learner_seat_0_trajectories"])
                - int(stats_b["learner_seat_1_trajectories"])
            )
            <= 1
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(
    path: Path,
    bc_checkpoint: dict[str, Any],
    model_config: dict[str, Any],
    deck_hash: str,
    deck_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_version") not in {
        BC_FEATURE_VERSION,
        PPO_FEATURE_VERSION,
    }:
        raise ValueError(f"Unsupported feature version in {path}")
    if checkpoint_model_config(checkpoint) != model_config:
        raise ValueError(f"Model config mismatch in {path}")
    validate_checkpoint_deck(checkpoint, deck_hash, path, deck_path)
    model = instantiate_model_from_checkpoint(
        checkpoint,
        bc_checkpoint,
        device,
    )
    model.eval()
    return checkpoint, model


@dataclass
class Snapshot:
    embedding: np.ndarray
    prefix: int
    turn: int
    learner_seat: int
    learner_first: int


@dataclass
class PrefixGame:
    battle: RawBattle
    learner_seat: int
    attempt_uid: int
    class_label: int
    public_history: PublicOpponentHistoryV2 | None = None
    decisions: int = 0
    captured_prefixes: set[int] = field(default_factory=set)
    snapshots: list[Snapshot] = field(default_factory=list)


def valid_selection(observation: dict[str, Any]) -> bool:
    select = observation.get("select")
    if not isinstance(select, dict):
        return False
    options = select.get("option")
    minimum = int(select.get("minCount", 0) or 0)
    maximum = int(select.get("maxCount", 0) or 0)
    return (
        isinstance(options, list)
        and bool(options)
        and minimum >= 0
        and maximum >= minimum
        and maximum <= len(options)
        and maximum <= MAX_ACTION_COUNT
    )


def target_prefix(
    observation: dict[str, Any],
    learner_seat: int,
) -> int | None:
    """Return k when this is the first learner turn after opponent turn k."""
    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", 0) or 0)
    if your_index != learner_seat:
        return None
    turn = int(current.get("turn", 0) or 0)
    first_player = int(current.get("firstPlayer", -1))
    learner_first = first_player == learner_seat
    for prefix in PREFIXES:
        expected_turn = 2 * prefix + 1 if learner_first else 2 * prefix
        if turn == expected_turn:
            return prefix
    return None


def collect_interleaved_prefix_trajectories(
    *,
    probe_model: torch.nn.Module,
    opponent_models: tuple[torch.nn.Module, torch.nn.Module],
    opponent_canonical_orders: tuple[bool, bool],
    class_names: tuple[str, str],
    deck: list[int],
    deck_hash: str,
    model_config: dict[str, Any],
    device: torch.device,
    trajectories_per_class: int,
    environments: int,
    max_game_decisions: int,
    schedule_seed: int,
    router_feature_mode: str = "stateless-v1",
) -> tuple[list[Snapshot], np.ndarray, dict[int, int], dict[str, Any]]:
    """Collect both labels with interleaved class/seat claims.

    ``class_label`` selects only the frozen opponent and offline label. Neither
    router feature receives the class label. V2 receives only observations
    actually delivered to the frozen probe, plus the fixed learner seat
    assigned before ``BattleStart``. Opponent-policy callback observations are
    never accumulated because they are unavailable to a submitted learner.
    """
    if router_feature_mode not in ROUTER_FEATURE_MODES:
        raise ValueError(
            f"Unsupported router_feature_mode {router_feature_mode!r}"
        )
    probe_model.eval()
    for opponent_model in opponent_models:
        opponent_model.eval()
    stats_by_label = (Counter(), Counter())
    snapshots: list[Snapshot] = []
    groups: list[int] = []
    group_labels: dict[int, int] = {}
    active: list[PrefixGame] = []
    schedule_rng = random.Random(schedule_seed)
    # These identifiers are join keys only. Their numeric ranges cannot reveal
    # a class label, unlike the historical ``label * N + index`` convention.
    group_id_pool = opaque_group_id_pool(
        2 * trajectories_per_class,
        schedule_seed + 2,
    )
    next_group_id = 0
    claims: list[tuple[int, int]] = []
    for _pair_index in range(trajectories_per_class // 2):
        seats_by_label: list[list[int]] = []
        for _label in (0, 1):
            seats = [0, 1]
            schedule_rng.shuffle(seats)
            seats_by_label.append(seats)
        for pair_position in (0, 1):
            claims.append((0, seats_by_label[0][pair_position]))
            claims.append((1, seats_by_label[1][pair_position]))
    if trajectories_per_class % 2:
        labels = [0, 1]
        schedule_rng.shuffle(labels)
        claims.extend((label, schedule_rng.randrange(2)) for label in labels)
    next_attempt_uid = 0
    committed_by_label = [0, 0]
    committed_attempt_uids: set[int] = set()

    def start_game(class_label: int, learner_seat: int) -> PrefixGame:
        nonlocal next_attempt_uid
        attempt_uid = next_attempt_uid
        next_attempt_uid += 1
        # Count the claim before BattleStart so engine start failures are part
        # of the replacement-rate denominator and cannot evade that gate.
        stats_by_label[class_label]["attempted_trajectories"] += 1
        game = PrefixGame(
            battle=RawBattle(deck, deck),
            learner_seat=learner_seat,
            attempt_uid=attempt_uid,
            class_label=class_label,
            # One state object per BattleStart. It is discarded with this
            # PrefixGame at BattleFinish, so environment-slot/process reuse
            # cannot carry history into the replacement game.
            public_history=(
                PublicOpponentHistoryV2(learner_seat)
                if router_feature_mode == "public-history-v2"
                else None
            ),
        )
        return game

    while len(active) < min(environments, 2 * trajectories_per_class):
        if not claims:
            break
        class_label, learner_seat = claims.pop(0)
        try:
            active.append(start_game(class_label, learner_seat))
        except ValueError:
            stats_by_label[class_label]["start_errors"] += 1
            claims.append((class_label, learner_seat))
            if sum(
                stats["start_errors"]
                for stats in stats_by_label
            ) > 2 * trajectories_per_class * 5:
                raise RuntimeError("Repeated BattleStart failures")

    while active:
        role_items: dict[
            str,
            list[tuple[int, PrefixGame, dict[str, Any]]],
        ] = defaultdict(list)
        invalid_indices: set[int] = set()
        for game_index, game in enumerate(active):
            stats = stats_by_label[game.class_label]
            observation = game.battle.observation
            if game.battle.result != -1:
                stats["early_terminal_trajectories"] += 1
                invalid_indices.add(game_index)
                continue
            if not valid_selection(observation):
                stats["invalid_observation_trajectories"] += 1
                invalid_indices.add(game_index)
                continue
            feature = live_feature(observation, model_config, deck_hash)
            if feature is None:
                stats["invalid_feature_trajectories"] += 1
                invalid_indices.add(game_index)
                continue
            current = observation.get("current") or {}
            seat = int(current.get("yourIndex", 0) or 0)
            role = (
                "probe"
                if seat == game.learner_seat
                else f"opponent_{game.class_label}"
            )
            if role == "probe" and game.public_history is not None:
                game.public_history.observe(observation)
            role_items[role].append((game_index, game, feature))
            if role == "probe":
                prefix = target_prefix(observation, game.learner_seat)
                if (
                    prefix is not None
                    and prefix not in game.captured_prefixes
                ):
                    if router_feature_mode == "stateless-v1":
                        router_x = public_router_features(observation)
                        expected_dim = PUBLIC_ROUTER_FEATURE_DIM
                    elif router_feature_mode == "public-history-v2":
                        if game.public_history is None:
                            raise AssertionError("Missing v2 history state")
                        router_x = game.public_history.features()
                        expected_dim = PUBLIC_HISTORY_V2_FEATURE_DIM
                    else:
                        raise ValueError(
                            "Unsupported router_feature_mode "
                            f"{router_feature_mode!r}"
                        )
                    if tuple(router_x.shape) != (expected_dim,):
                        raise RuntimeError("Router feature shape mismatch")
                    game.captured_prefixes.add(prefix)
                    game.snapshots.append(
                        Snapshot(
                            embedding=router_x.detach().cpu().numpy(),
                            prefix=prefix,
                            turn=int(current.get("turn", 0) or 0),
                            learner_seat=game.learner_seat,
                            learner_first=int(
                                int(current.get("firstPlayer", -1))
                                == game.learner_seat
                            ),
                        )
                    )

        actions: dict[int, list[int]] = {}
        complete_indices: set[int] = set()
        with torch.no_grad():
            for role in ("probe", "opponent_0", "opponent_1"):
                items = role_items.get(role, [])
                if not items:
                    continue
                batch = collate_features(
                    [item[2] for item in items],
                    model_config,
                    device,
                )
                model = (
                    probe_model
                    if role == "probe"
                    else opponent_models[int(role[-1])]
                )
                outputs = model_forward(model, batch, device)
                chosen, _, _, _ = sample_ordered_actions(
                    outputs,
                    batch,
                    deterministic=True,
                    canonicalize_order=(
                        False
                        if role == "probe"
                        else opponent_canonical_orders[int(role[-1])]
                    ),
                )
                for item_index, (game_index, game, _feature) in enumerate(items):
                    actions[game_index] = chosen[item_index]
                    if role != "probe":
                        continue
                    if PREFIXES[-1] in game.captured_prefixes:
                        complete_indices.add(game_index)

        finished_indices = set(invalid_indices)
        for game_index in sorted(complete_indices):
            if game_index in finished_indices:
                continue
            game = active[game_index]
            stats = stats_by_label[game.class_label]
            observed = sorted(snapshot.prefix for snapshot in game.snapshots)
            if observed != list(PREFIXES):
                stats["incomplete_snapshot_trajectories"] += 1
                finished_indices.add(game_index)
                continue
            class_label = game.class_label
            group = group_id_pool[next_group_id]
            next_group_id += 1
            snapshots.extend(game.snapshots)
            groups.extend([group] * len(game.snapshots))
            group_labels[group] = class_label
            committed_attempt_uids.add(game.attempt_uid)
            committed_by_label[class_label] += 1
            stats["committed_trajectories"] += 1
            stats[f"learner_seat_{game.learner_seat}_trajectories"] += 1
            stats[
                "learner_first_trajectories"
                if game.snapshots[0].learner_first
                else "learner_second_trajectories"
            ] += 1
            finished_indices.add(game_index)

        for game_index, game in enumerate(active):
            if game_index in finished_indices:
                continue
            stats = stats_by_label[game.class_label]
            action = actions.get(game_index)
            if action is None:
                stats["missing_action_trajectories"] += 1
                finished_indices.add(game_index)
                continue
            _, error = game.battle.step(action)
            game.decisions += 1
            stats["engine_decisions"] += 1
            if error:
                stats[f"select_error_{error}_trajectories"] += 1
                finished_indices.add(game_index)
                continue
            if game.decisions >= max_game_decisions:
                stats["max_decision_trajectories"] += 1
                finished_indices.add(game_index)

        for game_index in sorted(finished_indices, reverse=True):
            game = active[game_index]
            if game.public_history is not None:
                stats_by_label[game.class_label][
                    "history_turn_regression_resets"
                ] += game.public_history.turn_regression_resets
            if game.attempt_uid not in committed_attempt_uids:
                claims.append((game.class_label, game.learner_seat))
            active[game_index].battle.close()
            active.pop(game_index)

        while len(active) < environments and claims:
            class_label, learner_seat = claims.pop(0)
            try:
                active.append(start_game(class_label, learner_seat))
            except ValueError:
                stats_by_label[class_label]["start_errors"] += 1
                claims.append((class_label, learner_seat))
                if sum(
                    stats["start_errors"]
                    for stats in stats_by_label
                ) > 2 * trajectories_per_class * 5:
                    for game in active:
                        game.battle.close()
                    raise RuntimeError("Repeated BattleStart failures")

    rendered_stats: dict[str, Any] = {}
    for label, class_name in enumerate(class_names):
        stats = stats_by_label[label]
        if committed_by_label[label] != trajectories_per_class:
            raise RuntimeError(
                f"{class_name}: collected "
                f"{committed_by_label[label]}/{trajectories_per_class}"
            )
        if abs(
            stats["learner_seat_0_trajectories"]
            - stats["learner_seat_1_trajectories"]
        ) > 1:
            raise RuntimeError(
                f"{class_name}: valid trajectories are seat-skewed"
            )
        stats["replacement_trajectories"] = (
            stats["attempted_trajectories"]
            - stats["committed_trajectories"]
        )
        stats["replacement_rate"] = (
            stats["replacement_trajectories"]
            / max(stats["attempted_trajectories"], 1)
        )
        rendered_stats[class_name] = dict(stats)
    if len(snapshots) != 2 * trajectories_per_class * len(PREFIXES):
        raise RuntimeError("Interleaved snapshot count mismatch")
    if next_group_id != 2 * trajectories_per_class:
        raise RuntimeError("Opaque group identifier allocation mismatch")
    return (
        snapshots,
        np.asarray(groups, dtype=np.int32),
        group_labels,
        rendered_stats,
    )


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return float(center - margin), float(center + margin)


def fixed_game_split(
    groups: np.ndarray,
    group_labels: dict[int, int],
    group_seats: dict[int, int],
    train_per_class: int,
    seed: int,
) -> tuple[set[int], set[int]]:
    if train_per_class % 2:
        raise ValueError("Seat-stratified train_per_class must be even")
    rng = np.random.default_rng(seed)
    train: set[int] = set()
    test: set[int] = set()
    for label in (0, 1):
        for seat in (0, 1):
            stratum_groups = np.asarray(
                sorted(
                    group
                    for group, group_label in group_labels.items()
                    if group_label == label and group_seats[group] == seat
                ),
                dtype=np.int32,
            )
            rng.shuffle(stratum_groups)
            stratum_train = train_per_class // 2
            train.update(
                int(group) for group in stratum_groups[:stratum_train]
            )
            test.update(
                int(group) for group in stratum_groups[stratum_train:]
            )
    if train & test:
        raise AssertionError("Train/test game overlap")
    if train | test != set(int(group) for group in np.unique(groups)):
        raise AssertionError("Train/test game coverage mismatch")
    return train, test


def prefix_indices(
    prefixes: np.ndarray,
    groups: np.ndarray,
    selected_groups: set[int],
    prefix: int,
) -> np.ndarray:
    group_mask = np.fromiter(
        (int(group) in selected_groups for group in groups),
        dtype=np.bool_,
        count=len(groups),
    )
    return np.flatnonzero(group_mask & (prefixes == prefix))


def diagnostic_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    learner_first: np.ndarray,
) -> dict[str, Any]:
    correct = predictions == labels
    metrics: dict[str, Any] = {
        "rows": int(labels.size),
        "correct": int(correct.sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "recall_class_0": float(
            recall_score(labels, predictions, pos_label=0)
        ),
        "recall_class_1": float(
            recall_score(labels, predictions, pos_label=1)
        ),
        "by_learner_order": {},
    }
    for first_value, name in ((1, "learner_first"), (0, "learner_second")):
        mask = learner_first == first_value
        metrics["by_learner_order"][name] = {
            "rows": int(mask.sum()),
            "correct": int((predictions[mask] == labels[mask]).sum()),
            "accuracy": float(
                accuracy_score(labels[mask], predictions[mask])
            ),
        }
    low, high = wilson_interval(metrics["correct"], metrics["rows"])
    metrics["wilson_95_low"] = low
    metrics["wilson_95_high"] = high
    return metrics


def permutation_p_value(
    labels: np.ndarray,
    predictions: np.ndarray,
    permutations: int,
    seed: int,
) -> float:
    observed = int((labels == predictions).sum())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        extreme += int(int((permuted == predictions).sum()) >= observed)
    return (extreme + 1) / (permutations + 1)


def analyze_fixed_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    prefixes: np.ndarray,
    learner_first: np.ndarray,
    group_labels: dict[int, int],
    group_seats: dict[int, int],
    train_per_class: int,
    split_seed: int,
    permutation_samples: int,
) -> dict[str, Any]:
    train_groups, test_groups = fixed_game_split(
        groups,
        group_labels,
        group_seats,
        train_per_class,
        split_seed,
    )
    train_indices = prefix_indices(
        prefixes,
        groups,
        train_groups,
        PRIMARY_PREFIX,
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight=None,
            max_iter=2000,
            solver="lbfgs",
            random_state=split_seed,
        ),
    )
    classifier.fit(embeddings[train_indices], labels[train_indices])

    diagnostics: dict[str, Any] = {}
    primary_predictions: np.ndarray | None = None
    primary_labels: np.ndarray | None = None
    for prefix in PREFIXES:
        test_indices = prefix_indices(
            prefixes,
            groups,
            test_groups,
            prefix,
        )
        predictions = classifier.predict(embeddings[test_indices])
        diagnostics[f"opponent_turn_prefix_{prefix}"] = diagnostic_metrics(
            labels[test_indices],
            predictions,
            learner_first[test_indices],
        )
        if prefix == PRIMARY_PREFIX:
            primary_predictions = predictions
            primary_labels = labels[test_indices]

    if primary_predictions is None or primary_labels is None:
        raise AssertionError("Primary prefix was not evaluated")
    primary = diagnostics[f"opponent_turn_prefix_{PRIMARY_PREFIX}"]
    p_value = permutation_p_value(
        primary_labels,
        primary_predictions,
        permutations=permutation_samples,
        seed=split_seed + 1,
    )
    primary["test_label_permutation_p_value"] = p_value
    required_correct = (
        333
        if primary["rows"] == 512
        else math.ceil((333 / 512) * primary["rows"])
    )
    gates = {
        "test_rows_exact": primary["rows"] == 2 * (
            len(test_groups) // 2
        ),
        "correct": primary["correct"] >= required_correct,
        "wilson_low": primary["wilson_95_low"] > 0.60,
        "class_0_recall": primary["recall_class_0"] >= 0.60,
        "class_1_recall": primary["recall_class_1"] >= 0.60,
        "learner_first_accuracy": (
            primary["by_learner_order"]["learner_first"]["accuracy"] >= 0.60
        ),
        "learner_second_accuracy": (
            primary["by_learner_order"]["learner_second"]["accuracy"] >= 0.60
        ),
        "permutation": p_value < 0.01,
    }
    return {
        "classifier": {
            "type": "standardized_logistic_regression",
            "regularization_C": 1.0,
            "solver": "lbfgs",
            "threshold": 0.5,
            "primary_prefix": PRIMARY_PREFIX,
        },
        "split": {
            "unit": "complete_early_trajectory",
            "train_games": len(train_groups),
            "test_games": len(test_groups),
            "train_test_group_overlap": len(train_groups & test_groups),
            "train_per_class": train_per_class,
            "test_per_class": len(test_groups) // 2,
            "seat_stratified": True,
            "seed": split_seed,
        },
        "diagnostics": diagnostics,
        "primary_gates": gates,
        "primary_required_correct": required_correct,
        "primary_passed_before_collection_gates": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--opponent-a", type=Path, required=True)
    parser.add_argument("--opponent-b", type=Path, required=True)
    parser.add_argument("--opponent-a-name", default="fresh_bc")
    parser.add_argument("--opponent-b-name", default="v3_u440")
    parser.add_argument("--opponent-a-canonical-order", action="store_true")
    parser.add_argument("--opponent-b-canonical-order", action="store_true")
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--games-per-class", type=int, default=512)
    parser.add_argument("--train-per-class", type=int, default=256)
    parser.add_argument("--environments", type=int, default=16)
    parser.add_argument("--max-game-decisions", type=int, default=1000)
    parser.add_argument("--permutation-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261180)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--router-feature-mode",
        choices=ROUTER_FEATURE_MODES,
        default="stateless-v1",
        help=(
            "Keep stateless-v1 for historical reproducibility; select "
            "public-history-v2 explicitly for the causal opponent-history "
            "experiment."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.probe,
        args.opponent_a,
        args.opponent_b,
        args.bc_checkpoint,
        args.deck,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (
        args.games_per_class < 2
        or args.train_per_class < 1
        or args.train_per_class >= args.games_per_class
        or args.environments < 1
        or args.max_game_decisions < 1
        or args.permutation_samples < 1
    ):
        raise ValueError("Invalid collection or analysis settings")
    test_per_class = args.games_per_class - args.train_per_class
    if args.games_per_class == 512 and test_per_class != 256:
        raise ValueError("Formal 512-game audit requires 256 test games/class")

    args.output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    deck = read_deck(args.deck)
    deck_hash = compute_deck_hash(deck)
    bc_checkpoint = torch.load(
        args.bc_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if bc_checkpoint.get("feature_version") != BC_FEATURE_VERSION:
        raise ValueError("Unexpected BC anchor feature version")
    model_config = checkpoint_model_config(bc_checkpoint)
    probe_checkpoint, probe_model = load_model(
        args.probe,
        bc_checkpoint,
        model_config,
        deck_hash,
        args.deck,
        device,
    )
    opponent_a_checkpoint, opponent_a_model = load_model(
        args.opponent_a,
        bc_checkpoint,
        model_config,
        deck_hash,
        args.deck,
        device,
    )
    opponent_b_checkpoint, opponent_b_model = load_model(
        args.opponent_b,
        bc_checkpoint,
        model_config,
        deck_hash,
        args.deck,
        device,
    )
    if args.router_feature_mode == "public-history-v2":
        router_feature_version = PUBLIC_HISTORY_V2_FEATURE_VERSION
        router_feature_dim = PUBLIC_HISTORY_V2_FEATURE_DIM
        feature_description = (
            "Causal cumulative public opponent log events from observations "
            "actually delivered to the fixed learner agent only; opponent "
            "callback observations are unavailable and excluded; positive "
            "scalar whitelist; no own or unattributed events, identity "
            "metadata, model row, policy, submission, team, deck hash, serial, "
            "terminal absolute result, or offline label."
        )
        state_lifecycle: dict[str, Any] | None = {
            "scope": "one state object per RawBattle",
            "battle_start": "construct empty state with fixed learner seat",
            "battle_finish": "discard state before environment slot reuse",
            "explicit_process_reuse": (
                "construct a new state or call reset before the next game"
            ),
            "turn_regression": "automatic defensive full reset",
            "same_turn_new_game": (
                "requires explicit reset; automatic detection is impossible "
                "without accepting identity metadata"
            ),
        }
    else:
        router_feature_version = PUBLIC_ROUTER_FEATURE_VERSION
        router_feature_dim = PUBLIC_ROUTER_FEATURE_DIM
        feature_description = (
            "Exact stateless public_router_features(observation) used by the "
            "historical v18 audit; no model row, deck hash, label, or history."
        )
        state_lifecycle = None

    preregistration = {
        "schema_version": 1,
        "created_at_utc": started_at.isoformat(),
        "purpose": (
            "Go/no-go audit for a router using only the frozen probe's legal "
            "public agent observation representation."
        ),
        "probe": {
            "path": str(args.probe.resolve()),
            "sha256": sha256_file(args.probe),
            "feature_version": probe_checkpoint.get("feature_version"),
            "deterministic": True,
            "canonical_order": False,
        },
        "classes": [
            {
                "label": 0,
                "name": args.opponent_a_name,
                "path": str(args.opponent_a.resolve()),
                "sha256": sha256_file(args.opponent_a),
                "feature_version": opponent_a_checkpoint.get("feature_version"),
                "canonical_order": args.opponent_a_canonical_order,
            },
            {
                "label": 1,
                "name": args.opponent_b_name,
                "path": str(args.opponent_b.resolve()),
                "sha256": sha256_file(args.opponent_b),
                "feature_version": opponent_b_checkpoint.get("feature_version"),
                "canonical_order": args.opponent_b_canonical_order,
            },
        ],
        "deck": str(args.deck.resolve()),
        "deck_hash": deck_hash,
        "games_per_class": args.games_per_class,
        "train_per_class": args.train_per_class,
        "test_per_class": test_per_class,
        "snapshot_prefixes": list(PREFIXES),
        "primary_prefix": PRIMARY_PREFIX,
        "router_feature_mode": args.router_feature_mode,
        "router_feature_version": router_feature_version,
        "router_feature_dim": router_feature_dim,
        "feature": feature_description,
        "state_lifecycle": state_lifecycle,
        "classifier": (
            "StandardScaler + LogisticRegression(C=1,L2,lbfgs,threshold=0.5)"
        ),
        "main_gates": {
            "test_rows": 2 * test_per_class,
            "correct_min": 333 if test_per_class == 256 else None,
            "wilson_95_low_strictly_above": 0.60,
            "per_class_recall_min": 0.60,
            "per_learner_order_accuracy_min": 0.60,
            "test_label_permutation_p_strictly_below": 0.01,
            "replacement_rate_each_max": 0.02,
            "replacement_rate_class_gap_max": 0.02,
            "start_errors_each_exact": 0,
            "history_turn_regression_resets_each_exact": 0,
        },
        "label_isolation": (
            "Labels are written to labels.json separately from the label-free "
            "embedding tensor file. Seeded opaque group identifiers have no "
            "class-coded numeric ranges and are joined only for offline "
            "fitting."
        ),
        "engine_seed_control": False,
        "python_torch_seed": args.seed,
        "submission_or_packaging_authorized": False,
    }
    (args.output_dir / "preregistration.json").write_text(
        json.dumps(preregistration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    snapshots, groups, group_labels, collection_stats = (
        collect_interleaved_prefix_trajectories(
            probe_model=probe_model,
            opponent_models=(opponent_a_model, opponent_b_model),
            opponent_canonical_orders=(
                args.opponent_a_canonical_order,
                args.opponent_b_canonical_order,
            ),
            class_names=(args.opponent_a_name, args.opponent_b_name),
            deck=deck,
            deck_hash=deck_hash,
            model_config=model_config,
            device=device,
            trajectories_per_class=args.games_per_class,
            environments=min(args.environments, args.games_per_class),
            max_game_decisions=args.max_game_decisions,
            schedule_seed=args.seed + 1,
            router_feature_mode=args.router_feature_mode,
        )
    )
    embeddings = np.stack(
        [snapshot.embedding for snapshot in snapshots]
    ).astype(np.float32, copy=False)
    prefixes = np.asarray(
        [snapshot.prefix for snapshot in snapshots],
        dtype=np.int8,
    )
    turns = np.asarray(
        [snapshot.turn for snapshot in snapshots],
        dtype=np.int16,
    )
    learner_seats = np.asarray(
        [snapshot.learner_seat for snapshot in snapshots],
        dtype=np.int8,
    )
    learner_first = np.asarray(
        [snapshot.learner_first for snapshot in snapshots],
        dtype=np.int8,
    )
    group_seats = {
        int(group): int(learner_seats[index])
        for index, group in enumerate(groups)
    }
    labels = np.asarray(
        [group_labels[int(group)] for group in groups],
        dtype=np.int64,
    )

    np.savez_compressed(
        args.output_dir / "public_embeddings.npz",
        embeddings=embeddings,
        groups=groups,
        prefixes=prefixes,
        turns=turns,
        learner_seats=learner_seats,
        learner_first=learner_first,
    )
    (args.output_dir / "labels.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "labels": {
                    str(group): {
                        "label": label,
                        "class_name": (
                            args.opponent_a_name
                            if label == 0
                            else args.opponent_b_name
                        ),
                    }
                    for group, label in sorted(group_labels.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    analysis = analyze_fixed_split(
        embeddings,
        labels,
        groups,
        prefixes,
        learner_first,
        group_labels,
        group_seats,
        train_per_class=args.train_per_class,
        split_seed=args.seed + 3,
        permutation_samples=args.permutation_samples,
    )
    stats_a = collection_stats[args.opponent_a_name]
    stats_b = collection_stats[args.opponent_b_name]
    collection_gates = collection_quality_gates(stats_a, stats_b)
    passed = (
        analysis["primary_passed_before_collection_gates"]
        and all(collection_gates.values())
    )
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": preregistration,
        "collection": {
            **collection_stats,
            "embedding_rows": int(len(embeddings)),
            "embedding_dim": int(embeddings.shape[1]),
            "unique_trajectories": int(np.unique(groups).size),
            "collection_gates": collection_gates,
        },
        "analysis": analysis,
        "passed": passed,
        "status": (
            "public_router_feasibility_passed"
            if passed
            else "public_router_feasibility_failed"
        ),
        "elapsed_seconds": time.time() - started,
        "submission_or_packaging_performed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
