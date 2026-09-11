#!/usr/bin/env python3
"""Matched official-engine H2H for non-AR V7 versus mode-aware AR policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

import bc_nonar_v7 as nonar
import train_bc_mode_ar_v7 as mode_bc
import train_bc_orbit as base
import train_mode_ar_ppo as mode_ppo
import train_nonar_league_ppo as nonar_ppo
import train_ppo as engine


NONAR_FEATURE_VERSION = "ptcg-nonar-v7-ppo-terminal01-v1"
MODE_AR_FEATURE_VERSION = "ptcg-mode-ar-ppo-terminal01-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    raw = checkpoint.get("model_config")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint is missing model_config")
    return dict(raw)


def load_nonar(
    path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], nonar.EntityOptionPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_version") != NONAR_FEATURE_VERSION:
        raise ValueError(f"{path} is not a non-AR V7 PPO checkpoint")
    config = checkpoint_model_config(checkpoint)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{path} is missing model_state_dict")
    option_positions = state.get("option_position.weight")
    if not isinstance(option_positions, torch.Tensor):
        raise ValueError(f"{path} is missing option positions")
    model = nonar.EntityOptionPolicy(
        hash_size=int(config["hash_size"]),
        categorical_dim=int(config["categorical_dim"]),
        model_dim=int(config["model_dim"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        max_state_entities=int(config["max_state_entities"]),
        max_options=int(option_positions.shape[0]),
    )
    count_weight = state.get("count_head.2.weight")
    if not isinstance(count_weight, torch.Tensor) or count_weight.ndim != 2:
        raise ValueError(f"{path} has an invalid count head")
    count_classes = int(count_weight.shape[0])
    old_count_layer = model.count_head[-1]
    if old_count_layer.out_features != count_classes:
        model.count_head[-1] = torch.nn.Linear(
            old_count_layer.in_features,
            count_classes,
        )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).to(device).eval()
    return checkpoint, model, config


def load_mode_ar(
    path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], mode_bc.ModeAwareARPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_version") != MODE_AR_FEATURE_VERSION:
        raise ValueError(f"{path} is not a mode-aware AR PPO checkpoint")
    model = mode_ppo.instantiate_model(checkpoint, device)
    model.requires_grad_(False).eval()
    return checkpoint, model, mode_ppo.model_config(checkpoint)


def live_feature_with(
    featurizer: Callable[[dict[str, Any], int, int], dict[str, Any] | None],
    observation: dict[str, Any],
    config: dict[str, Any],
    deck_hash: str,
    persistent_logs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    current = observation.get("current") or {}
    return featurizer(
        {
            "observation": observation,
            "seat": int(current.get("yourIndex", 0) or 0),
            "deck_hash": deck_hash,
            "team_name": "",
            "action": [],
            "terminal_reward": 0.0,
            "sample_weight": 1.0,
            "_persistent_logs": list(persistent_logs),
        },
        int(config["hash_size"]),
        int(config["max_state_entities"]),
    )


@torch.no_grad()
def nonar_actions(
    model: nonar.EntityOptionPolicy,
    features: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> list[list[int]]:
    batch = engine.collate_features(features, config, device)
    with torch.amp.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        outputs = model(batch)
    actions, _, _, _ = nonar_ppo.sample_nonar_actions(
        outputs,
        batch,
        deterministic=True,
        temperature=1.0,
    )
    return actions


@torch.no_grad()
def mode_ar_actions(
    model: mode_bc.ModeAwareARPolicy,
    features: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> list[list[int]]:
    batch = engine.collate_features(features, config, device)
    with torch.amp.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        encoded = model(batch)
        actions, _, _, _ = mode_ppo.sample_mode_ar_actions(
            model,
            encoded,
            batch,
            deterministic=True,
            temperature=1.0,
        )
    return actions


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> list[float]:
    if games <= 0:
        return [0.0, 1.0]
    probability = wins / games
    denominator = 1.0 + z * z / games
    center = (probability + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / games
        + z * z / (4.0 * games * games)
    ) / denominator
    return [center - radius, center + radius]


@torch.no_grad()
def evaluate(
    candidate: nonar.EntityOptionPolicy,
    candidate_config: dict[str, Any],
    opponent: mode_bc.ModeAwareARPolicy,
    opponent_config: dict[str, Any],
    deck: list[int],
    device: torch.device,
    games_target: int,
    environments: int,
    max_game_decisions: int,
    seed: int,
) -> dict[str, Any]:
    deck_hash = engine.compute_deck_hash(deck)
    seat_rng = random.Random(seed)
    seat_pair: list[int] = []
    stats: Counter[str] = Counter()
    started = time.time()

    def next_candidate_seat() -> int:
        nonlocal seat_pair
        if not seat_pair:
            seat_pair = [0, 1]
            seat_rng.shuffle(seat_pair)
        return seat_pair.pop()

    def start_game() -> tuple[
        engine.RawBattle,
        int,
        int,
        dict[int, list[dict[str, Any]]],
    ]:
        return engine.RawBattle(deck, deck), next_candidate_seat(), 0, {0: [], 1: []}

    games = [start_game() for _ in range(min(environments, games_target))]
    stats["max_active_games"] = len(games)
    valid = 0
    while games:
        groups: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        invalid: set[int] = set()
        for index, (battle, candidate_seat, decisions, public_logs) in enumerate(games):
            observation = battle.observation
            if battle.result != -1:
                continue
            select = observation.get("select")
            current = observation.get("current") or {}
            options = select.get("option") if isinstance(select, dict) else None
            minimum = int((select or {}).get("minCount", 0) or 0)
            maximum = int((select or {}).get("maxCount", 0) or 0)
            if (
                not isinstance(options, list)
                or not options
                or minimum < 0
                or maximum < minimum
                or maximum > len(options)
                or maximum > engine.MAX_ACTION_COUNT
            ):
                invalid.add(index)
                continue
            seat = int(current.get("yourIndex", 0) or 0)
            logs = observation.get("logs") or []
            if isinstance(logs, list):
                public_logs[seat].extend(event for event in logs if isinstance(event, dict))
                public_logs[seat] = public_logs[seat][-64:]
            policy = -1 if seat == candidate_seat else 0
            config = candidate_config if policy == -1 else opponent_config
            featurizer = nonar.featurize_row if policy == -1 else base.featurize_row
            feature = live_feature_with(
                featurizer,
                observation,
                config,
                deck_hash,
                public_logs[seat],
            )
            if feature is None:
                invalid.add(index)
                continue
            groups[policy].append((index, feature))

        selected_actions: dict[int, list[int]] = {}
        if groups.get(-1):
            actions = nonar_actions(
                candidate,
                [feature for _, feature in groups[-1]],
                candidate_config,
                device,
            )
            selected_actions.update(
                (item[0], action) for item, action in zip(groups[-1], actions)
            )
        if groups.get(0):
            actions = mode_ar_actions(
                opponent,
                [feature for _, feature in groups[0]],
                opponent_config,
                device,
            )
            selected_actions.update(
                (item[0], action) for item, action in zip(groups[0], actions)
            )

        finished = set(invalid)
        stats["invalid_games"] += len(invalid)
        for index, (battle, candidate_seat, decisions, public_logs) in enumerate(games):
            if index in finished or index not in selected_actions:
                continue
            _, error = battle.step(selected_actions[index])
            decisions += 1
            games[index] = (battle, candidate_seat, decisions, public_logs)
            stats["decisions"] += 1
            if error:
                stats["invalid_games"] += 1
                finished.add(index)
                continue
            result = battle.result
            if result != -1:
                valid += 1
                stats["valid_games"] += 1
                if result == candidate_seat:
                    stats["wins"] += 1
                elif result == 2:
                    stats["draws"] += 1
                else:
                    stats["losses"] += 1
                finished.add(index)
            elif decisions >= max_game_decisions:
                stats["invalid_games"] += 1
                finished.add(index)

        for index in sorted(finished, reverse=True):
            games[index][0].close()
            games.pop(index)
        while len(games) < min(environments, games_target - valid):
            games.append(start_game())
        stats["max_active_games"] = max(stats["max_active_games"], len(games))
        if stats["invalid_games"] > games_target * 4:
            for battle, _, _, _ in games:
                battle.close()
            raise RuntimeError("mixed H2H exceeded invalid-game budget")

    decisive = stats["wins"] + stats["losses"]
    return {
        "valid_games": stats["valid_games"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "draws": stats["draws"],
        "invalid_games": stats["invalid_games"],
        "win_rate": stats["wins"] / max(stats["valid_games"], 1),
        "decisive_win_rate": stats["wins"] / max(decisive, 1),
        "wilson_95": wilson_interval(stats["wins"], stats["valid_games"]),
        "mean_decisions": stats["decisions"] / max(stats["valid_games"], 1),
        "max_active_games": stats["max_active_games"],
        "seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--environments", type=int, default=128)
    parser.add_argument("--max-game-decisions", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2026081631)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.candidate, args.opponent, args.deck):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.games < 1 or args.environments < 1:
        raise ValueError("games and environments must be positive")

    device = torch.device(args.device)
    candidate_checkpoint, candidate, candidate_config = load_nonar(
        args.candidate, device
    )
    opponent_checkpoint, opponent, opponent_config = load_mode_ar(
        args.opponent, device
    )
    if candidate_config != opponent_config:
        raise ValueError("candidate and opponent tensor schemas differ")
    deck = engine.read_deck(args.deck)
    deck_hash = engine.compute_deck_hash(deck)
    for label, checkpoint in (
        ("candidate", candidate_checkpoint),
        ("opponent", opponent_checkpoint),
    ):
        expected = engine.checkpoint_single_deck_hash(checkpoint)
        if expected is not None and expected != deck_hash:
            raise ValueError(f"{label} checkpoint deck hash mismatch")

    result = {
        "schema_version": "ptcg-mixed-nonar-v7-vs-mode-ar-h2h-v1",
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": sha256_file(args.candidate),
            "feature_version": candidate_checkpoint.get("feature_version"),
            "update": candidate_checkpoint.get("update"),
        },
        "opponent": {
            "path": str(args.opponent.resolve()),
            "sha256": sha256_file(args.opponent),
            "feature_version": opponent_checkpoint.get("feature_version"),
            "update": opponent_checkpoint.get("update"),
        },
        "deck": {
            "path": str(args.deck.resolve()),
            "sha256": sha256_file(args.deck),
            "semantic_hash": deck_hash,
        },
        "protocol": {
            "games": args.games,
            "environments": min(args.environments, args.games),
            "balanced_seats": True,
            "candidate_action_order": "nonar_static_ordered_scores",
            "opponent_action_order": "mode_ar_deployed_pointer_order",
            "seed": args.seed,
            "engine_seed_control": False,
        },
        "result": evaluate(
            candidate,
            candidate_config,
            opponent,
            opponent_config,
            deck,
            device,
            args.games,
            min(args.environments, args.games),
            args.max_game_decisions,
            args.seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
