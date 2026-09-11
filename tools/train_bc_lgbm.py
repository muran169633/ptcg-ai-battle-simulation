#!/usr/bin/env python3
"""Train a LightGBM candidate-ranking BC teacher for one deck/expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import lightgbm as lgb
import numpy as np
import orjson

from train_bc import card_id, resolve_card


FEATURE_NAMES = [
    "context",
    "select_type",
    "min_count",
    "max_count",
    "option_count",
    "turn",
    "turn_action_count",
    "first_relative",
    "supporter_played",
    "stadium_played",
    "energy_attached",
    "retreated",
    "self_deck_count",
    "self_hand_count",
    "self_active_count",
    "self_bench_count",
    "self_discard_count",
    "self_prize_count",
    "opp_deck_count",
    "opp_hand_count",
    "opp_active_count",
    "opp_bench_count",
    "opp_discard_count",
    "opp_prize_count",
    "self_active_id",
    "self_active_hp",
    "self_active_max_hp",
    "self_active_energy_count",
    "opp_active_id",
    "opp_active_hp",
    "opp_active_max_hp",
    "opp_active_energy_count",
    *[f"self_bench_id_{i}" for i in range(5)],
    *[f"opp_bench_id_{i}" for i in range(5)],
    *[f"hand_id_{i}" for i in range(12)],
    "option_position",
    "option_type",
    "option_area",
    "option_index",
    "option_player",
    "option_inplay_area",
    "option_inplay_index",
    "option_attack_id",
    "option_number",
    "option_energy_index",
    "option_energy_count",
    "source_card_id",
    "target_card_id",
    "attached_card_id",
    "source_hp",
    "source_max_hp",
    "source_energy_count",
    "source_hand_copies",
    "source_discard_copies",
    "context_option_cross",
    "context_source_cross",
    "type_source_cross",
    "recent_log_type_0",
    "recent_log_card_0",
    "recent_log_type_1",
    "recent_log_card_1",
    "recent_log_type_2",
    "recent_log_card_2",
]

CATEGORICAL_NAMES = {
    "context",
    "select_type",
    "first_relative",
    "supporter_played",
    "stadium_played",
    "energy_attached",
    "retreated",
    "self_active_id",
    "opp_active_id",
    *[f"self_bench_id_{i}" for i in range(5)],
    *[f"opp_bench_id_{i}" for i in range(5)],
    *[f"hand_id_{i}" for i in range(12)],
    "option_type",
    "option_area",
    "option_player",
    "option_inplay_area",
    "option_attack_id",
    "source_card_id",
    "target_card_id",
    "attached_card_id",
    "context_option_cross",
    "context_source_cross",
    "type_source_cross",
    "recent_log_type_0",
    "recent_log_card_0",
    "recent_log_type_1",
    "recent_log_card_1",
    "recent_log_type_2",
    "recent_log_card_2",
}
CATEGORICAL_INDICES = [
    index for index, name in enumerate(FEATURE_NAMES) if name in CATEGORICAL_NAMES
]


def episode_split(episode_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "valid"
    return "test"


def iter_filtered_rows(
    archive_path: Path,
    deck_hash: str,
    team_name: str,
    seed: int,
    split_mode: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name for name in archive.namelist() if name.endswith(".jsonl")
        ]
        for member in members:
            with archive.open(member) as handle:
                for line in handle:
                    row = orjson.loads(line)
                    if row.get("deck_hash") != deck_hash:
                        continue
                    if team_name and row.get("team_name") != team_name:
                        continue
                    split = (
                        str(row["split"])
                        if split_mode == "archive"
                        else episode_split(str(row["episode_id"]), seed)
                    )
                    yield split, row


def safe_card(zone: list[Any], index: int) -> dict[str, Any]:
    if 0 <= index < len(zone) and isinstance(zone[index], dict):
        return zone[index]
    return {}


def categorical(value: Any) -> int:
    try:
        return max(0, int(value) + 1)
    except (TypeError, ValueError):
        return 0


def option_source_target(
    row: dict[str, Any],
    option: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observation = row["observation"]
    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", row.get("seat", 0)) or 0)
    option_type = int(option.get("type", -1) or 0)
    area = int(option.get("area", 0) or 0)
    index = int(option.get("index", -1) if option.get("index") is not None else -1)
    player_index = int(option.get("playerIndex", your_index) or 0)
    if option_type == 7:
        area, player_index = 2, your_index
    source: Any = None
    target: Any = None
    attached: Any = None
    if option_type in (3, 4, 5, 6, 8, 9, 10, 11) and index >= 0:
        source = resolve_card(observation, area, index, player_index)
    if option_type in (8, 9):
        target = resolve_card(
            observation,
            int(option.get("inPlayArea", 0) or 0),
            int(option.get("inPlayIndex", -1) or 0),
            your_index,
        )
    if option_type in (4, 5, 6) and isinstance(source, dict):
        key = "tools" if option_type == 4 else "energyCards"
        index_key = "toolIndex" if option_type == 4 else "energyIndex"
        attached_index = int(option.get(index_key, -1) or 0)
        values = source.get(key) or []
        if 0 <= attached_index < len(values):
            attached = values[attached_index]
    return (
        source if isinstance(source, dict) else {},
        target if isinstance(target, dict) else {},
        attached if isinstance(attached, dict) else {},
    )


def row_option_features(
    row: dict[str, Any],
    option: dict[str, Any],
    option_position: int,
) -> list[float]:
    observation = row["observation"]
    select = observation["select"]
    current = observation["current"]
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", row.get("seat", 0)) or 0)
    opponent_index = 1 - your_index
    self_player = players[your_index] if your_index < len(players) else {}
    opponent = players[opponent_index] if opponent_index < len(players) else {}
    self_active = safe_card(self_player.get("active") or [], 0)
    opp_active = safe_card(opponent.get("active") or [], 0)
    self_bench = self_player.get("bench") or []
    opp_bench = opponent.get("bench") or []
    hand = self_player.get("hand") or []
    discard = self_player.get("discard") or []
    hand_ids = sorted(card_id(card) for card in hand if card_id(card))
    source, target, attached = option_source_target(row, option)
    source_id = card_id(source)
    option_type = int(option.get("type", -1) or 0)
    context = int(select.get("context", row.get("select_context", 0)) or 0)
    logs = observation.get("logs") or []
    recent: list[int] = []
    for event in reversed(logs[-3:]):
        event = event if isinstance(event, dict) else {}
        recent.extend(
            (
                categorical(event.get("type", -1)),
                categorical(event.get("cardId", -1)),
            )
        )
    recent.extend([0] * (6 - len(recent)))

    values: list[float] = [
        categorical(context),
        categorical(select.get("type", row.get("select_type", 0))),
        int(select.get("minCount", 0) or 0),
        int(select.get("maxCount", 0) or 0),
        len(select.get("option") or []),
        int(current.get("turn", 0) or 0),
        int(current.get("turnActionCount", 0) or 0),
        categorical(int(current.get("firstPlayer", 0) == your_index)),
        categorical(int(bool(current.get("supporterPlayed")))),
        categorical(int(bool(current.get("stadiumPlayed")))),
        categorical(int(bool(current.get("energyAttached")))),
        categorical(int(bool(current.get("retreated")))),
        int(self_player.get("deckCount", 0) or 0),
        int(self_player.get("handCount", 0) or 0),
        len(self_player.get("active") or []),
        len(self_bench),
        len(discard),
        len(self_player.get("prize") or []),
        int(opponent.get("deckCount", 0) or 0),
        int(opponent.get("handCount", 0) or 0),
        len(opponent.get("active") or []),
        len(opp_bench),
        len(opponent.get("discard") or []),
        len(opponent.get("prize") or []),
        categorical(card_id(self_active)),
        int(self_active.get("hp", 0) or 0),
        int(self_active.get("maxHp", 0) or 0),
        len(self_active.get("energies") or []),
        categorical(card_id(opp_active)),
        int(opp_active.get("hp", 0) or 0),
        int(opp_active.get("maxHp", 0) or 0),
        len(opp_active.get("energies") or []),
        *[
            categorical(card_id(safe_card(self_bench, index)))
            for index in range(5)
        ],
        *[
            categorical(card_id(safe_card(opp_bench, index)))
            for index in range(5)
        ],
        *[
            categorical(hand_ids[index] if index < len(hand_ids) else -1)
            for index in range(12)
        ],
        option_position,
        categorical(option_type),
        categorical(option.get("area", -1)),
        int(option.get("index", -1) or 0),
        categorical(option.get("playerIndex", -1)),
        categorical(option.get("inPlayArea", -1)),
        int(option.get("inPlayIndex", -1) or 0),
        categorical(option.get("attackId", -1)),
        int(option.get("number", 0) or 0),
        int(option.get("energyIndex", -1) or 0),
        int(option.get("count", 0) or 0),
        categorical(source_id),
        categorical(card_id(target)),
        categorical(card_id(attached)),
        int(source.get("hp", 0) or 0),
        int(source.get("maxHp", 0) or 0),
        len(source.get("energies") or []),
        sum(card_id(card) == source_id for card in hand) if source_id else 0,
        sum(card_id(card) == source_id for card in discard) if source_id else 0,
        categorical(context * 32 + option_type),
        categorical(context * 2048 + source_id),
        categorical(option_type * 2048 + source_id),
        *recent,
    ]
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError(f"feature length {len(values)} != {len(FEATURE_NAMES)}")
    return values


def count_rows(
    data: Path,
    deck_hash: str,
    team_name: str,
    seed: int,
    split_mode: str,
) -> dict[str, Counter[str]]:
    counts = {"train": Counter(), "valid": Counter(), "test": Counter()}
    started = time.time()
    for index, (split, row) in enumerate(
        iter_filtered_rows(data, deck_hash, team_name, seed, split_mode),
        1,
    ):
        counts[split]["decisions"] += 1
        counts[split]["candidates"] += int(row["option_count"])
        if index % 50_000 == 0:
            print(f"count decisions={index} elapsed={time.time()-started:.1f}s", flush=True)
    return counts


def allocate_and_fill(
    data: Path,
    deck_hash: str,
    team_name: str,
    seed: int,
    counts: dict[str, Counter[str]],
    split_mode: str,
) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "valid", "test"):
        decisions = counts[split]["decisions"]
        candidates = counts[split]["candidates"]
        arrays[split] = {
            "x": np.empty((candidates, len(FEATURE_NAMES)), dtype=np.float32),
            "y": np.empty(candidates, dtype=np.uint8),
            "weight": np.empty(candidates, dtype=np.float32),
            "offsets": np.empty(decisions + 1, dtype=np.int64),
            "min": np.empty(decisions, dtype=np.uint8),
            "max": np.empty(decisions, dtype=np.uint8),
        }
        arrays[split]["offsets"][0] = 0
    decision_at = Counter()
    candidate_at = Counter()
    started = time.time()
    for processed, (split, row) in enumerate(
        iter_filtered_rows(data, deck_hash, team_name, seed, split_mode),
        1,
    ):
        target = set(int(value) for value in row["action"])
        options = row["observation"]["select"]["option"]
        decision_index = decision_at[split]
        start = candidate_at[split]
        for option_index, option in enumerate(options):
            position = candidate_at[split]
            arrays[split]["x"][position] = row_option_features(
                row,
                option,
                option_index,
            )
            arrays[split]["y"][position] = option_index in target
            arrays[split]["weight"][position] = 1.0 / len(options)
            candidate_at[split] += 1
        arrays[split]["offsets"][decision_index + 1] = candidate_at[split]
        arrays[split]["min"][decision_index] = row["min_count"]
        arrays[split]["max"][decision_index] = row["max_count"]
        decision_at[split] += 1
        if processed % 50_000 == 0:
            print(f"features decisions={processed} elapsed={time.time()-started:.1f}s", flush=True)
    return arrays


def exact_metrics(
    predictions: np.ndarray,
    arrays: dict[str, np.ndarray],
    thresholds: list[float],
) -> dict[str, Any]:
    offsets = arrays["offsets"]
    labels = arrays["y"]
    mins = arrays["min"]
    maxs = arrays["max"]
    results: dict[str, Any] = {}
    oracle_correct = 0
    for threshold in thresholds:
        correct = fixed_correct = flexible_correct = 0
        fixed_count = flexible_count = 0
        for row_index in range(len(mins)):
            begin, end = offsets[row_index : row_index + 2]
            scores = predictions[begin:end]
            truth = np.flatnonzero(labels[begin:end])
            selected_count = int((scores >= threshold).sum())
            selected_count = max(int(mins[row_index]), selected_count)
            selected_count = min(int(maxs[row_index]), selected_count, len(scores))
            predicted = np.sort(
                np.argsort(-scores, kind="stable")[:selected_count]
            )
            exact = np.array_equal(predicted, truth)
            correct += exact
            fixed = mins[row_index] == maxs[row_index]
            fixed_count += fixed
            flexible_count += not fixed
            fixed_correct += exact and fixed
            flexible_correct += exact and not fixed
            if threshold == thresholds[0]:
                oracle = np.sort(
                    np.argsort(-scores, kind="stable")[: len(truth)]
                )
                oracle_correct += np.array_equal(oracle, truth)
        results[str(threshold)] = {
            "exact": correct / len(mins),
            "fixed": fixed_correct / max(fixed_count, 1),
            "flexible": flexible_correct / max(flexible_count, 1),
        }
    best_threshold = max(thresholds, key=lambda value: results[str(value)]["exact"])
    return {
        "rows": len(mins),
        "best_threshold": best_threshold,
        "exact_action_set_accuracy": results[str(best_threshold)]["exact"],
        "fixed_cardinality_accuracy": results[str(best_threshold)]["fixed"],
        "flexible_cardinality_accuracy": results[str(best_threshold)]["flexible"],
        "oracle_cardinality_accuracy": oracle_correct / len(mins),
        "thresholds": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/bc_recent7_top20.zip"))
    parser.add_argument("--deck-hash", required=True)
    parser.add_argument("--team-name", default="")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bc_lgbm"))
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument(
        "--final-train-valid",
        action="store_true",
        help="Retrain on hash train+valid and report only untouched hash-test.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.05)
    parser.add_argument(
        "--split-mode",
        choices=("episode_hash", "archive"),
        default="episode_hash",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = count_rows(
        args.data,
        args.deck_hash,
        args.team_name,
        args.seed,
        args.split_mode,
    )
    print("counts", {key: dict(value) for key, value in counts.items()}, flush=True)
    arrays = allocate_and_fill(
        args.data,
        args.deck_hash,
        args.team_name,
        args.seed,
        counts,
        args.split_mode,
    )
    train_x = arrays["train"]["x"]
    train_y = arrays["train"]["y"]
    train_weight = arrays["train"]["weight"]
    if args.final_train_valid:
        train_x = np.concatenate((train_x, arrays["valid"]["x"]), axis=0)
        train_y = np.concatenate((train_y, arrays["valid"]["y"]), axis=0)
        train_weight = np.concatenate(
            (train_weight, arrays["valid"]["weight"]),
            axis=0,
        )
    train = lgb.Dataset(
        train_x,
        label=train_y,
        weight=train_weight,
        feature_name=FEATURE_NAMES,
        categorical_feature=CATEGORICAL_INDICES,
        free_raw_data=False,
    )
    valid = lgb.Dataset(
        arrays["valid"]["x"],
        label=arrays["valid"]["y"],
        weight=arrays["valid"]["weight"],
        feature_name=FEATURE_NAMES,
        categorical_feature=CATEGORICAL_INDICES,
        reference=train,
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "max_depth": 14,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 0.5,
        "num_threads": min(os.cpu_count() or 1, 32),
        "verbosity": -1,
        "seed": args.seed,
    }
    if args.final_train_valid:
        model = lgb.train(
            params,
            train,
            num_boost_round=args.rounds,
            callbacks=[lgb.log_evaluation(50)],
        )
        evaluation_split = "test"
        thresholds = [args.fixed_threshold]
    else:
        model = lgb.train(
            params,
            train,
            num_boost_round=args.rounds,
            valid_sets=[valid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(50)],
        )
        evaluation_split = "valid"
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65]
    prediction_iteration = args.rounds if args.final_train_valid else model.best_iteration
    predictions = model.predict(
        arrays[evaluation_split]["x"],
        num_iteration=prediction_iteration,
    )
    metrics = exact_metrics(
        predictions,
        arrays[evaluation_split],
        thresholds,
    )
    model.save_model(str(args.output_dir / "teacher.txt"))
    summary = {
        "feature_version": "ptcg-bc-lgbm-visible-v1",
        "deck_hash": args.deck_hash,
        "team_name": args.team_name,
        "split_mode": (
            "archive_time_dates"
            if args.split_mode == "archive"
            else "episode_hash_80_10_10"
        ),
        "seed": args.seed,
        "counts": {key: dict(value) for key, value in counts.items()},
        "best_iteration": prediction_iteration,
        "final_train_valid": args.final_train_valid,
        "evaluation_split": evaluation_split,
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
