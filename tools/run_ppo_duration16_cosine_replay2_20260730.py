#!/usr/bin/env python3
"""Run duration-16 cosine PPO with two ordered BC replay steps per update."""

from __future__ import annotations

import argparse
import json
import os
import shlex

import run_ppo_duration16_cosine_20260730 as base


OUTPUT_DIR = (
    base.ROOT
    / "artifacts"
    / "ppo_gold_duration16_cosine_replay2_tailconsensus5_ow8_ctx34q2_ordered_seed20260763_20260730"
    / "B_gold_league"
    / "seed-20260763"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, expected in base.EXPECTED_HASHES.items():
        actual = base.sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen input hash mismatch for {path}: {actual} != {expected}"
            )
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to reuse output directory: {OUTPUT_DIR}")

    payload = json.loads(base.BASE_PREREG.read_text(encoding="utf-8"))
    command = list(
        payload["plan"]["branches"]["B_gold_league"]["runs"][0]["command"]
    )
    base.set_flag(command, "--output-dir", str(OUTPUT_DIR))
    base.set_flag(command, "--updates", "456")
    base.set_flag(command, "--eval-interval", "456")
    base.set_flag(command, "--checkpoint-interval", "456")
    base.set_flag(command, "--seed", "20260763")
    base.set_flag(command, "--learning-rate-schedule", "cosine")
    base.set_flag(command, "--bc-replay-steps", "2")

    if command[1] != str(base.ROOT / "tools" / "train_ppo.py"):
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
