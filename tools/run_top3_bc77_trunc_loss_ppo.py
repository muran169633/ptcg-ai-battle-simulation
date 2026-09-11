#!/usr/bin/env python3
"""Run terminal-01 PPO while learning from decision-limit trajectories.

This is a versioned follow-up to ``run_top3_bc77_corrected_ppo.py``.  It keeps
the exact BC-qualified Top-3 mixture and optimizer settings, but retains every
decision-limit trajectory as a zero-return loss sample.  A replacement game is
still collected, so the fixed valid-game and seat quotas remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import run_top3_bc77_corrected_ppo as corrected


ROOT = corrected.ROOT
OUTPUT_ROOT = corrected.OUTPUT_ROOT
PROFILES = corrected.PROFILES
PHASES = corrected.PHASES
EXPERIMENT_NAME = "ppo_terminal01_trunc_loss_v3"


def output_dir(profile: corrected.baseline.Profile, phase: corrected.Phase) -> Path:
    return OUTPUT_ROOT / profile.slug / f"{EXPERIMENT_NAME}_{phase.name}"


def ppo_command(
    profile: corrected.baseline.Profile,
    phase: corrected.Phase,
) -> list[str]:
    command = corrected.ppo_command(profile, phase)
    output_index = command.index("--output-dir") + 1
    command[output_index] = str(output_dir(profile, phase))
    command.append("--truncation-as-loss")
    return command


def protocol_manifest(
    profile: corrected.baseline.Profile,
    phase: corrected.Phase,
    command: list[str],
) -> dict[str, Any]:
    manifest = corrected.protocol_manifest(profile, phase, command)
    manifest.update(
        {
            "schema_version": "ptcg-top3-terminal01-trunc-loss-ppo-v1",
            "output_dir": str(output_dir(profile, phase)),
            "truncation_handling": {
                "training_target": "loss_return_0",
                "valid_game_quota": "replace",
                "max_game_decisions": 1000,
            },
            "command": command,
        }
    )
    return manifest


def validate(
    profile: corrected.baseline.Profile,
    phase: corrected.Phase,
    execute: bool,
) -> dict[str, Any]:
    gate = corrected.validate(profile, phase, execute=False)
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
