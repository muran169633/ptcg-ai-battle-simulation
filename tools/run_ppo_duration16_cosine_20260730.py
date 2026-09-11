#!/usr/bin/env python3
"""Run a frozen duration-16 PPO recipe with cosine LR scheduling.

The rejected duration-32 cosine experiment is the control. This launcher
preserves its model inputs, opponent quotas, optimizer settings, replay
settings, and endpoint-only checkpoint selection. The modeling change is
halving the cosine horizon from 32 updates to 16 updates. A new seed provides
an independent rollout sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PREREG = (
    ROOT
    / "artifacts"
    / "ppo_gold_tailconsensus5_ow8_ctx34q2_ordered_seed20260730_20260728.runner_preregistration.json"
)
OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "ppo_gold_duration16_cosine_tailconsensus5_ow8_ctx34q2_ordered_seed20260756_20260730"
    / "B_gold_league"
    / "seed-20260756"
)

EXPECTED_HASHES = {
    BASE_PREREG: "f3004b5ae039f483f14001b20f1c21f36034828b7ad2fec65bf3b2c8e8e75b79",
    ROOT / "tools" / "train_ppo.py": "82dc43718eb2b688698dd450f4af95d9cffaa4daf3ea35723fd03df0124fe8c8",
    ROOT
    / "artifacts"
    / "bc_marnie_train24_day25hash80_orbit_v3"
    / "best.pt": "61e592ff9821f17747b206eb2e5dc30ffc741932fdd86d3ec620ab7e137d7c92",
    ROOT
    / "artifacts"
    / "ppo_marnie_v3_incumbent600"
    / "checkpoints"
    / "update-0440.pt": "d3278052b2f13d0157b37ba8b43a21a0727233e093e8adc7963b4ee085f599de",
    ROOT
    / "data"
    / "bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip": "a3b9d572bfc0a784b5b140d3dbe09314252c46dd0d9c1afa3423236cda54543c",
    ROOT
    / "artifacts"
    / "gold_clone_league_combined_20260727"
    / "league_manifest.json": "8f70f68c69b9576d958af8bd78e07cd87127c79264533e30717dd23abe8b6571",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_flag(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1:
        raise RuntimeError(f"Expected exactly one {flag}, found {len(positions)}")
    command[positions[0] + 1] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Validate all bindings and print the child command without executing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen input hash mismatch for {path}: {actual} != {expected}"
            )

    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to reuse output directory: {OUTPUT_DIR}")

    payload = json.loads(BASE_PREREG.read_text(encoding="utf-8"))
    command = list(
        payload["plan"]["branches"]["B_gold_league"]["runs"][0]["command"]
    )

    set_flag(command, "--output-dir", str(OUTPUT_DIR))
    set_flag(command, "--updates", "456")
    set_flag(command, "--eval-interval", "456")
    set_flag(command, "--checkpoint-interval", "456")
    set_flag(command, "--seed", "20260756")
    set_flag(command, "--learning-rate-schedule", "cosine")

    expected_trainer = str(ROOT / "tools" / "train_ppo.py")
    if command[1] != expected_trainer:
        raise RuntimeError(f"Unexpected child program: {command[1]}")
    if command[command.index("--schedule-start-update") + 1] != "441":
        raise RuntimeError("Cosine phase must span updates 441 through 456")
    if command[command.index("--resume-learner-weights") + 1] != "bc":
        raise RuntimeError("Learner weights must come from the frozen BC checkpoint")
    for required_flag in (
        "--reset-optimizer-on-resume",
        "--reset-opponent-quota-on-resume",
        "--opponent-quota-seat-balance",
        "--skip-initial-eval",
    ):
        if required_flag not in command:
            raise RuntimeError(f"Missing frozen flag: {required_flag}")

    if args.print_command:
        print(shlex.join(command))
        return

    os.execv(command[0], command)


if __name__ == "__main__":
    main()
