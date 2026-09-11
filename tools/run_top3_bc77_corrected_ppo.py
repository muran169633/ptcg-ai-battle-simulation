#!/usr/bin/env python3
"""Run the corrected terminal-01 PPO experiment for the BC-qualified Top-3.

This launcher intentionally preserves the previous ``ppo_light_top3_v1``
outputs.  It changes the diagnosed PPO mechanics while keeping the reward and
the BC safety machinery unchanged:

* terminal win/loss reward remains 1/0;
* transition-mean actor reduction replaces inverse-length episode weighting;
* GAE lambda is 1.0 and advantages are normalized per frozen opponent;
* the value trunk receives a small gradient;
* only the three requested portfolio decks are used as opponents.

Commands are printed by default.  ``--execute`` is required to train.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import run_top3_bc77_ppo as baseline


ROOT = baseline.ROOT
PYTHON = baseline.PYTHON
TRAIN_PPO = baseline.TRAIN_PPO
OUTPUT_ROOT = baseline.OUTPUT_ROOT
PROFILES = baseline.PROFILES
BC_FEATURE = baseline.BC_FEATURE
GATE = baseline.GATE
EXPERIMENT_NAME = "ppo_terminal01_corrected_v2"


@dataclass(frozen=True)
class Phase:
    name: str
    updates: int
    games_per_update: int
    eval_games: int
    eval_interval: int
    checkpoint_interval: int


PHASES = {
    "smoke": Phase(
        name="smoke",
        updates=1,
        games_per_update=96,
        eval_games=32,
        eval_interval=1,
        checkpoint_interval=1,
    ),
    "full": Phase(
        name="full",
        updates=12,
        games_per_update=256,
        eval_games=128,
        eval_interval=2,
        checkpoint_interval=2,
    ),
}


def output_dir(profile: baseline.Profile, phase: Phase) -> Path:
    return OUTPUT_ROOT / profile.slug / f"{EXPERIMENT_NAME}_{phase.name}"


def top3_quotas(
    profile: baseline.Profile,
    games_per_update: int,
) -> tuple[int, list[tuple[baseline.Profile, int]]]:
    """Allocate 50% to own BC and 25% to each other Top-3 member."""
    if games_per_update % 4:
        raise ValueError("Top-3 corrected PPO budget must be divisible by four")
    own_quota = games_per_update // 2
    other_quota = games_per_update // 4
    others = [
        (other, other_quota)
        for other in PROFILES.values()
        if other.slug != profile.slug
    ]
    if len(others) != 2:
        raise RuntimeError("Corrected PPO requires exactly three Top-3 profiles")
    if own_quota + sum(quota for _, quota in others) != games_per_update:
        raise AssertionError("Corrected Top-3 quotas do not sum to the budget")
    return own_quota, others


def ppo_command(profile: baseline.Profile, phase: Phase) -> list[str]:
    own_quota, others = top3_quotas(profile, phase.games_per_update)
    seed = profile.seed + (100 if phase.name == "smoke" else 200)
    command = [
        str(PYTHON),
        "-B",
        str(TRAIN_PPO),
        "--bc-checkpoint",
        str(profile.checkpoint),
        "--kl-reference-checkpoint",
        str(profile.checkpoint),
        "--deck",
        str(profile.deck),
        "--output-dir",
        str(output_dir(profile, phase)),
        "--updates",
        str(phase.updates),
        "--schedule-start-update",
        "1",
        "--environments",
        "16",
        "--games-per-update",
        str(phase.games_per_update),
        "--ppo-epochs",
        "3",
        "--minibatch-size",
        "512",
        "--learning-rate",
        "0.00003",
        "--value-learning-rate",
        "0.00005",
        "--weight-decay",
        "0.0001",
        "--learning-rate-schedule",
        "constant",
        "--gamma",
        "1.0",
        "--gae-lambda",
        "1.0",
        "--advantage-normalization",
        "per_opponent",
        "--clip-ratio",
        "0.15",
        "--value-coefficient",
        "0.25",
        "--value-trunk-gradient-scale",
        "0.05",
        "--entropy-coefficient",
        "0.001",
        "--max-grad-norm",
        "1.0",
        "--policy-temperature",
        "0.8",
        "--trainable-scope",
        "last_block_heads",
        "--bc-kl-start",
        "0.016",
        "--bc-kl-end",
        "0.012",
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
        "transition_mean",
        "--actor-value-gradient-mode",
        "scalar",
        "--constrained-gradient-mode",
        "scalar",
        "--snapshot-interval",
        "1000000",
        "--max-pool-size",
        "8",
        "--bc-replay-data",
        str(profile.archive),
        "--bc-replay-split",
        "train",
        "--bc-replay-batches",
        "72",
        "--bc-replay-batch-size",
        "256",
        "--bc-replay-workers",
        "8",
        "--bc-replay-steps",
        "2",
        "--bc-replay-lr-scale",
        "0.075",
        "--bc-replay-loss",
        "ordered",
        "--bc-replay-order-context-weight",
        "8.0",
        "--bc-replay-non-context34-fixed-multi-action-order-weight",
        "1.0",
        "--bc-replay-context34-rows-per-batch",
        "4",
        "--eval-interval",
        str(phase.eval_interval),
        "--eval-games",
        str(phase.eval_games),
        "--selection-aggregation",
        "mean",
        "--checkpoint-interval",
        str(phase.checkpoint_interval),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--opponent-base-quota",
        "bc",
        str(own_quota),
    ]
    for other, quota in others:
        command.extend(
            ("--extra-opponent", str(other.checkpoint), str(other.deck))
        )
        command.extend(
            (
                "--opponent-base-quota",
                baseline.opponent_name(other.checkpoint, other.deck),
                str(quota),
            )
        )
    return command


def protocol_manifest(
    profile: baseline.Profile,
    phase: Phase,
    command: list[str],
) -> dict[str, Any]:
    own_quota, others = top3_quotas(profile, phase.games_per_update)
    return {
        "schema_version": "ptcg-top3-corrected-terminal01-ppo-v1",
        "profile": profile.slug,
        "phase": phase.name,
        "reward": {"win": 1.0, "loss": 0.0, "draw": 0.0},
        "updates": phase.updates,
        "games_per_update": phase.games_per_update,
        "terminal_games": phase.updates * phase.games_per_update,
        "actor_reduction": "transition_mean",
        "gae_lambda": 1.0,
        "advantage_normalization": "per_opponent",
        "value_trunk_gradient_scale": 0.05,
        "opponent_game_quotas": {
            "bc": own_quota,
            **{
                baseline.opponent_name(other.checkpoint, other.deck): quota
                for other, quota in others
            },
        },
        "selection_opponent": "bc",
        "output_dir": str(output_dir(profile, phase)),
        "python": str(PYTHON),
        "command": command,
    }


def validate(
    profile: baseline.Profile,
    phase: Phase,
    execute: bool,
) -> dict[str, Any]:
    required = [PYTHON, TRAIN_PPO]
    for member in PROFILES.values():
        required.extend(
            (member.checkpoint, member.summary, member.archive, member.deck)
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(map(str, missing)))

    gate = baseline.portfolio_gate()
    if not gate["passed"]:
        raise RuntimeError(
            "Portfolio BC77 gate failed: "
            f"{gate['row_weighted_exact_action_set_accuracy']:.6f} < {GATE:.2f}"
        )
    _, route_exact = baseline.valid_metrics(profile)
    if route_exact < GATE:
        raise RuntimeError(
            f"{profile.slug} route BC77 gate failed: {route_exact:.6f} < {GATE:.2f}"
        )

    checkpoint = torch.load(
        profile.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("feature_version") != BC_FEATURE:
        raise ValueError(f"{profile.checkpoint}: not a V5 BC checkpoint")
    declared = set((checkpoint.get("config") or {}).get("deck_hashes") or [])
    if declared and declared != {profile.deck_hash}:
        raise ValueError(
            f"{profile.checkpoint}: checkpoint deck hashes {declared} do not "
            f"match {profile.deck_hash}"
        )

    target = output_dir(profile, phase)
    if execute and target.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {target}")
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("alakazam_control", "marnie"),
        required=True,
    )
    parser.add_argument("--phase", choices=tuple(PHASES), default="smoke")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = PROFILES[args.profile]
    phase = PHASES[args.phase]
    gate = validate(profile, phase, args.execute)
    command = ppo_command(profile, phase)
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))
    print(
        json.dumps(
            protocol_manifest(profile, phase, command),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    print(shlex.join(command), flush=True)
    if args.execute:
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
