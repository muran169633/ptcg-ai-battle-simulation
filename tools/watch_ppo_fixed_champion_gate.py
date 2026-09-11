#!/usr/bin/env python3
"""Maintain an external fixed-game PPO champion chain for an active run.

The watcher is useful when a long-running process was launched with an older
promotion threshold.  It freezes the initial champion, processes checkpoints
in update order, and challenges each candidate against the externally selected
champion.  A matching embedded panel is reused; otherwise a fresh balanced-seat
panel is played.  The live run's ``best.pt`` is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

import train_ppo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--minimum-win-rate", type=float, default=0.54)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=2026081454)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--watch", action="store_true")
    return parser


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device, weights_only=False)


def checkpoint_update(path: Path) -> int:
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    return int(checkpoint.get("update", -1))


def checkpoint_paths(run_dir: Path) -> list[Path]:
    return sorted(
        run_dir.joinpath("checkpoints").glob("update-*.pt"),
        key=lambda path: int(path.stem.split("-")[-1]),
    )


def reusable_embedded_gate(
    candidate: dict[str, Any],
    candidate_update: int,
    champion_update: int,
    games: int,
) -> dict[str, Any] | None:
    gate = ((candidate.get("metrics") or {}).get("champion_gate") or {})
    if (
        int(gate.get("candidate_update", -1)) == candidate_update
        and int(gate.get("champion_update_before", -1)) == champion_update
        and int(gate.get("valid_games", 0)) == games
        and int(gate.get("invalid_games", 0)) == 0
    ):
        return gate
    return None


def validate_pair(
    candidate: dict[str, Any],
    champion: dict[str, Any],
    bc_checkpoint: dict[str, Any],
    deck_hash: str,
) -> None:
    expected_config = train_ppo.checkpoint_model_config(bc_checkpoint)
    for label, checkpoint in (
        ("candidate", candidate),
        ("champion", champion),
    ):
        if checkpoint.get("learner_deck_hash") != deck_hash:
            raise ValueError(f"{label} learner deck hash mismatch")
        if train_ppo.checkpoint_model_config(checkpoint) != expected_config:
            raise ValueError(f"{label} model config mismatch")


def fresh_panel(
    candidate: dict[str, Any],
    champion: dict[str, Any],
    bc_checkpoint: dict[str, Any],
    deck: list[int],
    games: int,
    environments: int,
    max_game_decisions: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    train_ppo.seed_everything(seed)
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
    try:
        return train_ppo.evaluate_head_to_head(
            candidate_model,
            champion_model,
            deck,
            train_ppo.checkpoint_model_config(bc_checkpoint),
            device,
            games,
            min(environments, games),
            max_game_decisions,
            current_canonical_order=False,
            opponent_canonical_order=False,
        )
    finally:
        del candidate_model
        del champion_model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.minimum_win_rate <= 1.0:
        raise ValueError("--minimum-win-rate must be in [0, 1]")
    if args.games < 1 or args.games % 2:
        raise ValueError("--games must be a positive even number")
    if args.environments < 1 or args.poll_seconds <= 0.0:
        raise ValueError("invalid environments or poll interval")

    run_dir = args.run_dir.resolve()
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = state_dir / "reports"
    champions_dir = state_dir / "champions"
    status_path = state_dir / "status.json"
    selected_path = state_dir / "selected.json"
    pid_path = state_dir / "watcher.pid"
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    run_info = json.loads((run_dir / "run_config.json").read_text())
    config = run_info["config"]
    total_updates = int(config["updates"])
    deck = train_ppo.read_deck(Path(config["deck"]))
    deck_hash = train_ppo.compute_deck_hash(deck)
    bc_checkpoint = load_checkpoint(
        Path(config["bc_checkpoint"]),
        torch.device("cpu"),
    )
    device = torch.device(args.device)

    if selected_path.is_file():
        selected = json.loads(selected_path.read_text())
        champion_update = int(selected["champion_update"])
        champion_path = Path(selected["champion_checkpoint"])
        last_processed = int(selected.get("last_processed_update", -1))
    else:
        live_best = run_dir / "best.pt"
        initial = load_checkpoint(live_best, torch.device("cpu"))
        champion_update = int(initial.get("update", -1))
        if champion_update != 0:
            raise ValueError(
                "new external chain requires the frozen update-0 champion"
            )
        champion_path = champions_dir / "update-0000.pt"
        atomic_copy(live_best, champion_path)
        last_processed = 0
        selected = {
            "schema_version": "ptcg-fixed-champion-selection-v1",
            "champion_update": champion_update,
            "champion_checkpoint": str(champion_path),
            "champion_sha256": train_ppo.file_sha256(champion_path),
            "last_processed_update": last_processed,
            "minimum_win_rate_inclusive": args.minimum_win_rate,
            "games_per_gate": args.games,
            "live_best_overwritten": False,
        }
        atomic_json(selected_path, selected)
        atomic_copy(champion_path, state_dir / "best.pt")

    while True:
        made_progress = False
        for candidate_path in checkpoint_paths(run_dir):
            candidate_update = int(candidate_path.stem.split("-")[-1])
            if candidate_update <= last_processed:
                continue
            candidate = load_checkpoint(candidate_path, torch.device("cpu"))
            if int(candidate.get("update", -1)) != candidate_update:
                raise ValueError(f"checkpoint update mismatch: {candidate_path}")
            champion = load_checkpoint(champion_path, torch.device("cpu"))
            validate_pair(candidate, champion, bc_checkpoint, deck_hash)
            panel = reusable_embedded_gate(
                candidate,
                candidate_update,
                champion_update,
                args.games,
            )
            panel_source = "embedded"
            if panel is None:
                panel_source = "fresh"
                panel = fresh_panel(
                    candidate,
                    champion,
                    bc_checkpoint,
                    deck,
                    args.games,
                    args.environments,
                    int(config.get("max_game_decisions", 1000)),
                    device,
                    args.seed + candidate_update + champion_update,
                )
            promoted = train_ppo.champion_gate_promotes(
                panel,
                args.minimum_win_rate,
                inclusive=True,
            )
            previous_champion_update = champion_update
            if promoted:
                champion_update = candidate_update
                champion_path = champions_dir / f"update-{candidate_update:04d}.pt"
                atomic_copy(candidate_path, champion_path)
                atomic_copy(candidate_path, state_dir / "best.pt")
            last_processed = candidate_update
            report = {
                "schema_version": "ptcg-fixed-champion-gate-v1",
                "candidate_update": candidate_update,
                "candidate_checkpoint": str(candidate_path.resolve()),
                "candidate_sha256": train_ppo.file_sha256(candidate_path),
                "champion_update_before": previous_champion_update,
                "panel_source": panel_source,
                "panel": panel,
                "games": args.games,
                "minimum_win_rate_inclusive": args.minimum_win_rate,
                "promoted": promoted,
                "champion_update_after": champion_update,
                "live_best_overwritten": False,
            }
            atomic_json(
                reports_dir / f"update-{candidate_update:04d}.json",
                report,
            )
            selected = {
                "schema_version": "ptcg-fixed-champion-selection-v1",
                "champion_update": champion_update,
                "champion_checkpoint": str(champion_path),
                "champion_sha256": train_ppo.file_sha256(champion_path),
                "last_processed_update": last_processed,
                "minimum_win_rate_inclusive": args.minimum_win_rate,
                "games_per_gate": args.games,
                "live_best_overwritten": False,
            }
            atomic_json(selected_path, selected)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            made_progress = True

        completed = last_processed >= total_updates
        status = {
            "schema_version": "ptcg-fixed-champion-watcher-v1",
            "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "state": "completed" if completed else "running",
            "run_dir": str(run_dir),
            "total_updates": total_updates,
            "last_processed_update": last_processed,
            "champion_update": champion_update,
            "champion_checkpoint": str(champion_path),
            "minimum_win_rate_inclusive": args.minimum_win_rate,
            "games_per_gate": args.games,
            "watch": args.watch,
            "made_progress": made_progress,
        }
        atomic_json(status_path, status)
        if completed or not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
