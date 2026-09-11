#!/usr/bin/env python3
"""Launch the time-prioritized Dragapult non-AR PPO route.

The opponent mix bootstraps the requested league distribution with compatible
policies: 80% recent-day Top23 decks, 10% frozen same-deck BC mirror, and 10%
frozen historical anchor (the independently named ``last.pt`` BC snapshot).
All streams are then adjusted by the clipped rolling inverse-win controller.
Future non-AR PPO checkpoints can replace the bootstrap history anchor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BC_DIR = (
    ROOT
    / "artifacts"
    / "bc_top100_recent14_nonar_order_v7_end0813_b2048_20260816_v1"
)
DEFAULT_META = (
    ROOT / "data" / "recent_day_meta_pool_20260813_top23_v1" / "meta_pool.json"
)
DEFAULT_DECK = (
    ROOT
    / "data"
    / "recent_day_meta_pool_20260813_top23_v1"
    / "decks"
    / "rank02_07bedfffbfad.csv"
)
DEFAULT_TRAINER = ROOT / "tools" / "train_nonar_league_ppo.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bc-dir", type=Path, default=DEFAULT_BC_DIR)
    parser.add_argument("--meta-pool", type=Path, default=DEFAULT_META)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--environments", type=int, default=128)
    parser.add_argument("--games-per-update", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.000005)
    parser.add_argument("--value-learning-rate", type=float, default=0.00002)
    parser.add_argument("--schedule-start-update", type=int, default=1)
    parser.add_argument("--bc-kl-start", type=float, default=0.02)
    parser.add_argument("--bc-kl-end", type=float, default=0.003)
    parser.add_argument("--opponent-window-games", type=int, default=200)
    parser.add_argument("--opponent-inverse-min-factor", type=float, default=0.5)
    parser.add_argument("--opponent-inverse-max-factor", type=float, default=2.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trainer", type=Path, default=DEFAULT_TRAINER)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--reset-opponent-quota-on-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=2026081621)
    parser.add_argument("--champion-initial-wait", type=int, default=8)
    parser.add_argument("--champion-failure-cooldown", type=int, default=20)
    parser.add_argument("--champion-success-cooldown", type=int, default=59)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--candidate-snapshot-interval", type=int, default=0)
    parser.add_argument("--rollback-on-gate-failure", action="store_true")
    parser.add_argument("--reset-optimizer-on-gate-rollback", action="store_true")
    parser.add_argument("--champion-soft-min-win-rate", type=float)
    parser.add_argument("--champion-confirmation-games", type=int, default=200)
    parser.add_argument("--champion-pool-games", type=int, default=200)
    parser.add_argument("--rollout-workers", type=int, default=0)
    parser.add_argument("--rollout-envs-per-worker", type=int, default=16)
    parser.add_argument("--rollout-batch-wait-ms", type=float, default=10.0)
    args = parser.parse_args()

    bc_best = (args.bc_dir / "best.pt").resolve()
    bc_history_candidate = args.bc_dir / "last.pt"
    bc_history = (
        bc_history_candidate.resolve()
        if bc_history_candidate.is_file()
        else bc_best
    )
    meta_path = args.meta_pool.resolve()
    learner_deck = args.deck.resolve()
    trainer = args.trainer.resolve()
    required_paths = [bc_best, bc_history, meta_path, learner_deck, trainer]
    if args.resume is not None:
        required_paths.append(args.resume.resolve())
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != "ptcg-recent-day-exact-deck-meta-pool-v1":
        raise ValueError("Unexpected recent-day meta-pool schema")
    rows = list(meta.get("opponents") or [])
    if len(rows) != 23:
        raise ValueError(f"Expected Top23 meta rows, got {len(rows)}")

    command = [
        sys.executable,
        str(trainer),
        "--bc-checkpoint",
        str(bc_best),
        "--deck",
        str(learner_deck),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--updates",
        str(args.updates),
        "--environments",
        str(args.environments),
        "--games-per-update",
        str(args.games_per_update),
        "--ppo-epochs",
        "3",
        "--minibatch-size",
        str(args.minibatch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--value-learning-rate",
        str(args.value_learning_rate),
        "--weight-decay",
        "0.0001",
        "--gamma",
        "1.0",
        "--gae-lambda",
        "0.97",
        "--advantage-normalization",
        "per_opponent",
        "--clip-ratio",
        "0.2",
        "--value-coefficient",
        "0.35",
        "--value-trunk-gradient-scale",
        "0.0",
        "--entropy-coefficient",
        "0.002",
        "--policy-temperature",
        "0.9",
        "--trainable-scope",
        "full",
        "--learning-rate-schedule",
        "cosine",
        "--schedule-start-update",
        str(args.schedule_start_update),
        "--bc-kl-start",
        str(args.bc_kl_start),
        "--bc-kl-end",
        str(args.bc_kl_end),
        "--target-kl",
        "0.008",
        "--league-probability",
        "1.0",
        "--opponent-sampling",
        "per_game",
        "--opponent-quota-mode",
        "legacy",
        "--opponent-win-rate-window-games",
        str(args.opponent_window_games),
        "--opponent-inverse-min-factor",
        str(args.opponent_inverse_min_factor),
        "--opponent-inverse-max-factor",
        str(args.opponent_inverse_max_factor),
        "--opponent-quota-refresh-updates",
        "1",
        "--actor-reduction",
        "episode_mean",
        "--snapshot-interval",
        "5",
        "--max-pool-size",
        "33",
        "--eval-interval",
        "10",
        "--eval-games",
        "200",
        "--champion-gate-interval",
        "1",
        "--champion-gate-initial-wait-updates",
        str(args.champion_initial_wait),
        "--champion-gate-failure-cooldown-updates",
        str(args.champion_failure_cooldown),
        "--champion-gate-success-cooldown-updates",
        str(args.champion_success_cooldown),
        "--champion-gate-games",
        "200",
        "--champion-gate-min-win-rate",
        "0.54",
        "--champion-pool-gate-games",
        str(args.champion_pool_games),
        "--champion-pool-gate-min-win-rate",
        "0.53",
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--max-game-decisions",
        "2500",
        "--skip-initial-eval",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.candidate_snapshot_interval > 0:
        command.extend(
            (
                "--candidate-snapshot-interval",
                str(args.candidate_snapshot_interval),
            )
        )
    if args.rollback_on_gate_failure:
        command.append("--rollback-on-gate-failure")
    if args.reset_optimizer_on_gate_rollback:
        command.append("--reset-optimizer-on-gate-rollback")
    if args.champion_soft_min_win_rate is not None:
        command.extend(
            (
                "--champion-gate-soft-min-win-rate",
                str(args.champion_soft_min_win_rate),
                "--champion-gate-confirmation-games",
                str(args.champion_confirmation_games),
            )
        )
    if args.resume is not None:
        command.extend(("--resume", str(args.resume.resolve())))
    if args.reset_optimizer_on_resume:
        command.append("--reset-optimizer-on-resume")
    if args.reset_opponent_quota_on_resume:
        command.append("--reset-opponent-quota-on-resume")
    if args.rollout_workers > 0:
        command.extend(
            (
                "--rollout-workers",
                str(args.rollout_workers),
                "--rollout-envs-per-worker",
                str(args.rollout_envs_per_worker),
                "--rollout-batch-wait-ms",
                str(args.rollout_batch_wait_ms),
            )
        )

    # Ten percent frozen same-deck mirror.
    command.extend(("--opponent-meta-weight", "bc", "0.10"))
    # Ten percent independently named frozen historical anchor.  best.pt and
    # last.pt are currently equal by hash, which is recorded as bootstrap
    # state rather than misrepresented as a diverse learned history.
    command.extend(("--extra-opponent", str(bc_history), str(learner_deck)))
    history_name = f"{bc_history.stem}@{learner_deck.stem}"
    command.extend(("--opponent-meta-weight", history_name, "0.10"))

    # Eighty percent recent-day arena mix.  The source probabilities already
    # sum to one; scaling retains the exact Aug-13 proportions.
    for row in rows:
        raw_deck_path = Path(str(row["deck_path"]))
        deck_path = (
            raw_deck_path.resolve()
            if raw_deck_path.is_absolute()
            else (meta_path.parent / raw_deck_path).resolve()
        )
        probability = float(row["pool_base_probability"])
        if not deck_path.is_file() or probability <= 0.0:
            raise ValueError(f"Invalid Top23 binding: {row!r}")
        command.extend(("--extra-opponent", str(bc_best), str(deck_path)))
        name = f"{bc_best.stem}@{deck_path.stem}"
        command.extend(
            ("--opponent-meta-weight", name, format(0.80 * probability, ".17g"))
        )

    os.execv(command[0], command)


if __name__ == "__main__":
    main()
