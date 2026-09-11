#!/usr/bin/env python3
"""Seat-balanced CPU H2H for mode-aware checkpoints over multiple decks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import evaluate_mode_ar_vs_submission as mode_eval
import evaluate_ppo_head_to_head as h2h
import train_mode_ar_ppo as mode_ppo


def sha256(path: Path) -> str:
    return mode_ppo.sha256_file(path)


def parse_opponent(raw: str) -> tuple[str, Path, int]:
    try:
        label, path, games = raw.split("=", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LABEL=DECK_PATH=GAMES") from exc
    value = int(games)
    if not label or value < 2 or value % 2:
        raise argparse.ArgumentTypeError("label required and games must be positive/even")
    return label, Path(path), value


def boss_guard_positions(observation: dict[str, Any]) -> set[int]:
    select = observation.get("select")
    current = observation.get("current")
    if not isinstance(select, dict) or not isinstance(current, dict):
        return set()
    if int(select.get("context", -1) or 0) != 0 or int(select.get("type", -1) or 0) != 0:
        return set()
    options = select.get("option")
    players = current.get("players")
    seat = int(current.get("yourIndex", 0) or 0)
    if not isinstance(options, list) or not isinstance(players, list) or not 0 <= seat < len(players):
        return set()
    hand = players[seat].get("hand") if isinstance(players[seat], dict) else None
    if not isinstance(hand, list):
        return set()
    blocked = set()
    for position, option in enumerate(options):
        if not isinstance(option, dict) or int(option.get("type", -1) or -1) != 7:
            continue
        hand_index = option.get("index")
        if isinstance(hand_index, int) and 0 <= hand_index < len(hand):
            card = hand[hand_index]
            if isinstance(card, dict) and int(card.get("id", -1) or -1) == 1182:
                blocked.add(position)
    attacks = [option for option in options if isinstance(option, dict) and int(option.get("type", -1) or -1) == 13]
    effective = bool(attacks) and any(int(option.get("attackId", -1) or -1) != 323 for option in attacks)
    minimum = int(select.get("minCount", 0) or 0)
    return blocked if not effective and len(options) - len(blocked) >= minimum else set()


def split_even(total: int, shards: int) -> list[int]:
    shards = max(1, min(shards, total // 2))
    pairs, remainder = divmod(total // 2, shards)
    return [2 * (pairs + int(index < remainder)) for index in range(shards)]


def worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed = int(payload["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    _, candidate, config = mode_ppo.load_anchor(Path(payload["candidate"]), device)
    _, opponent, opponent_config = mode_ppo.load_anchor(Path(payload["opponent"]), device)
    candidate.requires_grad_(False)
    opponent.requires_grad_(False)
    learner_deck = h2h.read_deck(Path(payload["learner_deck"]))
    opponent_deck = h2h.read_deck(Path(payload["opponent_deck"]))
    original_live = h2h.live_feature
    original_collate = h2h.collate_features

    def guarded_live(*args: Any, **kwargs: Any) -> Any:
        feature = original_live(*args, **kwargs)
        if feature is not None:
            feature["_boss_guard_positions"] = sorted(boss_guard_positions(args[0]))
        return feature

    def guarded_collate(features: list[dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
        batch = original_collate(features, *args, **kwargs)
        mask = torch.zeros_like(batch["option_mask"])
        for row, feature in enumerate(features):
            positions = feature.get("_boss_guard_positions", [])
            if positions:
                mask[row, positions] = True
        batch["_boss_guard_mask"] = mask
        return batch

    with mode_eval.install_mode_ar_inference_adapter():
        adapter_forward = h2h.model_forward

        def guarded_forward(model: torch.nn.Module, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            if bool(payload.get("candidate_boss_guard")) and model is candidate:
                blocked = batch.get("_boss_guard_mask")
                if isinstance(blocked, torch.Tensor) and bool(blocked.any()):
                    batch = dict(batch)
                    batch["option_mask"] = batch["option_mask"].clone()
                    batch["option_mask"][blocked] = False
            return adapter_forward(model, batch, *args, **kwargs)

        h2h.live_feature = guarded_live
        h2h.collate_features = guarded_collate
        h2h.model_forward = guarded_forward
        try:
            result = h2h.evaluate_head_to_head_with_seats(
                candidate,
                opponent,
                learner_deck,
                config,
                device,
                int(payload["games"]),
                min(int(payload["environments"]), int(payload["games"])),
                int(payload["max_game_decisions"]),
                current_canonical_order=False,
                opponent_canonical_order=False,
                opponent_deck=opponent_deck,
                current_hybrid_order=False,
                opponent_hybrid_order=False,
                opponent_model_config=opponent_config,
            )
        finally:
            h2h.live_feature = original_live
            h2h.collate_features = original_collate
            h2h.model_forward = adapter_forward
    return {"label": payload["label"], "seed": seed, "result": result}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(int(row["result"]["wins"]) for row in rows)
    losses = sum(int(row["result"]["losses"]) for row in rows)
    draws = sum(int(row["result"]["draws"]) for row in rows)
    invalid = sum(int(row["result"]["invalid_games"]) for row in rows)
    valid = wins + losses + draws
    low, high = h2h.wilson_interval(wins, valid)
    seats: dict[str, Any] = {}
    for seat in ("0", "1"):
        seat_rows = [row["result"]["by_candidate_seat"][seat] for row in rows]
        seat_wins = sum(int(row["wins"]) for row in seat_rows)
        seat_valid = sum(int(row["valid_games"]) for row in seat_rows)
        seat_low, seat_high = h2h.wilson_interval(seat_wins, seat_valid)
        seats[seat] = {
            "valid_games": seat_valid,
            "wins": seat_wins,
            "win_rate": seat_wins / seat_valid,
            "wilson_95": [seat_low, seat_high],
        }
    return {
        "valid_games": valid,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "invalid_games": invalid,
        "win_rate": wins / valid,
        "wilson_95": [low, high],
        "by_candidate_seat": seats,
        "shards": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--learner-deck", type=Path, required=True)
    parser.add_argument(
        "--opponent-deck",
        dest="opponent_specs",
        action="append",
        type=parse_opponent,
        required=True,
        metavar="LABEL=DECK_PATH=GAMES",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--games-per-shard", type=int, default=100)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--max-game-decisions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--candidate-boss-guard", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.candidate, args.reference, args.learner_deck):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.workers < 1 or args.games_per_shard < 2 or args.games_per_shard % 2:
        raise ValueError("workers positive; games-per-shard positive and even")
    payloads: list[dict[str, Any]] = []
    deck_audit: dict[str, Any] = {}
    task_index = 0
    for label, deck_path, games in args.opponent_specs:
        if not deck_path.is_file():
            raise FileNotFoundError(deck_path)
        if label in deck_audit:
            raise ValueError(f"duplicate opponent label: {label}")
        shards = split_even(games, (games + args.games_per_shard - 1) // args.games_per_shard)
        deck_audit[label] = {
            "path": str(deck_path.resolve()),
            "sha256": sha256(deck_path),
            "semantic_hash": h2h.compute_deck_hash(h2h.read_deck(deck_path)),
            "games": games,
            "shards": shards,
        }
        for shard_games in shards:
            payloads.append(
                {
                    "label": label,
                    "candidate": str(args.candidate.resolve()),
                    "opponent": str(args.reference.resolve()),
                    "learner_deck": str(args.learner_deck.resolve()),
                    "opponent_deck": str(deck_path.resolve()),
                    "games": shard_games,
                    "environments": min(args.environments, shard_games),
                    "max_game_decisions": args.max_game_decisions,
                    "seed": args.seed + 104729 * task_index,
                    "candidate_boss_guard": bool(args.candidate_boss_guard),
                }
            )
            task_index += 1

    started = time.time()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(args.workers, len(payloads)), mp_context=context
    ) as executor:
        rows = list(executor.map(worker, payloads))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)
    by_opponent = {label: summarize(grouped[label]) for label in deck_audit}
    overall = summarize(rows)
    if overall["invalid_games"] != 0:
        raise RuntimeError(f"evaluation produced invalid games: {overall['invalid_games']}")
    payload = {
        "schema_version": "ptcg-mode-ar-multi-deck-h2h-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": sha256(args.candidate),
        },
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256(args.reference),
        },
        "learner_deck": {
            "path": str(args.learner_deck.resolve()),
            "sha256": sha256(args.learner_deck),
            "semantic_hash": h2h.compute_deck_hash(h2h.read_deck(args.learner_deck)),
        },
        "opponent_decks": deck_audit,
        "by_opponent": by_opponent,
        "overall": overall,
        "candidate_boss_guard": bool(args.candidate_boss_guard),
        "workers": min(args.workers, len(payloads)),
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
