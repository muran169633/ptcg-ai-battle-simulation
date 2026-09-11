#!/usr/bin/env python3
"""Independently confirm a borderline PPO champion-gate checkpoint.

This is intended for runs that were already launched with the legacy strict
hard gate.  It never overwrites the live run's ``best.pt``.  A passing
candidate is copied to an explicitly named soft-champion checkpoint and a
JSON audit report is written beside it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

import train_ppo


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate", type=Path, required=True)
    result.add_argument("--champion", type=Path, required=True)
    result.add_argument("--output-checkpoint", type=Path, required=True)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--hard-min-win-rate", type=float, default=0.58)
    result.add_argument("--soft-min-win-rate", type=float, default=0.54)
    result.add_argument("--confirmation-games", type=int, default=200)
    result.add_argument("--environments", type=int, default=64)
    result.add_argument("--seed", type=int, default=2026081454)
    result.add_argument("--device", default="cuda")
    return result


def require_rate(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")


def load(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device, weights_only=False)


def main() -> None:
    args = parser().parse_args()
    require_rate(args.soft_min_win_rate, "--soft-min-win-rate")
    require_rate(args.hard_min_win_rate, "--hard-min-win-rate")
    if args.soft_min_win_rate > args.hard_min_win_rate:
        raise ValueError("soft threshold cannot exceed hard threshold")
    if args.confirmation_games < 1 or args.confirmation_games % 2:
        raise ValueError("--confirmation-games must be a positive even number")
    if args.environments < 1:
        raise ValueError("--environments must be positive")

    device = torch.device(args.device)
    train_ppo.seed_everything(args.seed)
    candidate = load(args.candidate, device)
    champion = load(args.champion, device)
    candidate_config = candidate.get("config") or {}
    bc_path = Path(str(candidate_config.get("bc_checkpoint", "")))
    deck_path = Path(str(candidate_config.get("deck", "")))
    bc_checkpoint = load(bc_path, torch.device("cpu"))
    deck = train_ppo.read_deck(deck_path)
    deck_hash = train_ppo.compute_deck_hash(deck)
    for label, checkpoint in (
        ("candidate", candidate),
        ("champion", champion),
    ):
        if checkpoint.get("learner_deck_hash") != deck_hash:
            raise ValueError(f"{label} learner deck hash does not match deck")
        if train_ppo.checkpoint_model_config(checkpoint) != (
            train_ppo.checkpoint_model_config(bc_checkpoint)
        ):
            raise ValueError(f"{label} model config does not match BC anchor")

    initial = ((candidate.get("metrics") or {}).get("champion_gate") or {})
    if int(initial.get("candidate_update", -1)) != int(
        candidate.get("update", -2)
    ):
        raise ValueError("candidate checkpoint lacks its matching gate panel")
    if int(initial.get("champion_update_before", -1)) != int(
        champion.get("update", -2)
    ):
        raise ValueError("initial panel was not played against this champion")
    if int(initial.get("valid_games", 0)) <= 0:
        raise ValueError("initial panel has no valid games")

    initial_rate = float(initial["win_rate"])
    confirmation = None
    combined = dict(initial)
    if initial_rate >= args.hard_min_win_rate:
        promoted = True
        promotion_path = "direct_hard"
    elif initial_rate >= args.soft_min_win_rate:
        candidate_model = train_ppo.instantiate_model_from_checkpoint(
            candidate,
            bc_checkpoint,
            device,
        )
        champion_model = train_ppo.instantiate_model_from_checkpoint(
            champion,
            bc_checkpoint,
            device,
        )
        confirmation = train_ppo.evaluate_head_to_head(
            candidate_model,
            champion_model,
            deck,
            train_ppo.checkpoint_model_config(bc_checkpoint),
            device,
            args.confirmation_games,
            min(args.environments, args.confirmation_games),
            int(candidate_config.get("max_game_decisions", 1000)),
            current_canonical_order=False,
            opponent_canonical_order=False,
        )
        combined = train_ppo.combine_head_to_head_evaluations(
            [initial, confirmation]
        )
        promoted = train_ppo.champion_gate_promotes(
            combined,
            args.soft_min_win_rate,
            inclusive=True,
        )
        promotion_path = (
            "confirmed_soft" if promoted else "soft_confirmation_failed"
        )
    else:
        promoted = False
        promotion_path = "below_soft_threshold"

    report = {
        "schema_version": "ptcg-ppo-soft-gate-confirmation-v1",
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": train_ppo.file_sha256(args.candidate),
            "update": int(candidate.get("update", -1)),
        },
        "champion": {
            "path": str(args.champion.resolve()),
            "sha256": train_ppo.file_sha256(args.champion),
            "update": int(champion.get("update", -1)),
        },
        "policy": {
            "hard_min_win_rate_inclusive": args.hard_min_win_rate,
            "soft_min_win_rate_inclusive": args.soft_min_win_rate,
            "confirmation_games": args.confirmation_games,
            "seed": args.seed,
        },
        "initial_challenge": initial,
        "confirmation_challenge": confirmation,
        "combined_challenge": combined,
        "promotion_path": promotion_path,
        "promoted": promoted,
        "live_best_overwritten": False,
        "output_checkpoint": (
            str(args.output_checkpoint.resolve()) if promoted else None
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if promoted:
        args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate, args.output_checkpoint)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
