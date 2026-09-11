#!/usr/bin/env python3
"""Audit or execute one low-LR U472-heavy PPO update from general-BC S16."""

from __future__ import annotations

import hashlib
import importlib.util
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SOURCE = TOOLS / "exec_design202608198_s16_balancedanchor_u477.py"
SOURCE_SHA256 = "a5a1cc4df754145f14c38343fc0c37cadb813b111b2f32811f52a5ecf36e127b"

info = SOURCE.lstat()
if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
    raise RuntimeError("design202608198 launcher is not an authenticated regular file")
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("design202608198 launcher mismatch")

spec = importlib.util.spec_from_file_location("design202608198_u472heavy_source", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load authenticated design202608198 launcher")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
base = source.base

base.SELF = Path(__file__).resolve()
base.SCHEMA = "ptcg-design202608200-s16-u472heavy-u477-v1"
base.SEED = 202608200
base.OUTPUT_ROOT = ROOT / "artifacts/design202608200_s16_u472heavy_ppo1x192_u477"
base.OUTPUT_DIR = base.OUTPUT_ROOT / "B_gold_league/seed-202608200"
base.TERMINAL = base.OUTPUT_DIR / "checkpoints/update-0477.pt"
base.ATTEMPT = ROOT / ".ptcg-design202608200-s16-u472heavy-u477-attempt.json"
base.LOG = ROOT / "artifacts/design202608200_s16_u472heavy_ppo1x192_u477.log"
base.ANCHOR_QUOTAS = {
    "submitted_u472": 64,
    "u468_beta100": 16,
    "alpha075_parent": 16,
}
base.DEPENDENCIES = dict(base.DEPENDENCIES) | {SOURCE: SOURCE_SHA256}

source_derive_command = source.derive_command
source_expected_protocol = source.expected_protocol


def replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def replace_quota(command: list[str], opponent: str, value: str) -> None:
    for index in range(len(command) - 2):
        if command[index] == "--opponent-base-quota" and command[index + 1] == opponent:
            command[index + 2] = value
            return
    raise ValueError(f"quota opponent missing: {opponent}")


def derive_command() -> list[str]:
    command = source_derive_command()
    replace_option(command, "--learning-rate", "6e-06")
    replace_quota(command, base.opponent_name(source.ALPHA075, base.DECK), "16")
    return command


def expected_protocol(command: list[str]):
    value = source_expected_protocol(command)
    value["hyperparameters"]["actor_learning_rate"] = 6e-6
    value["route_change"] = {
        "single_update_only": True,
        "parent": "latest-replay general-BC S16",
        "latest_replay_through": "2026-08-04",
        "gold_games_per_update": 96,
        "alpha075_parent_games_per_update": 16,
        "submitted_u472_games_per_update": 64,
        "u468_beta100_games_per_update": 16,
        "reason": "repair the only failed independent anchor while halving actor LR and preserving balanced seats",
        "predecessor_u477": {
            "checkpoint_sha256": "a76429ae686fe5ca81582d3c3dc1514c79565db232d723c3bcc1dcec15b74f76",
            "submitted_u472_result": "63-65",
            "promoted": False,
        },
    }
    return value


base.derive_command = derive_command
base.expected_protocol = expected_protocol


if __name__ == "__main__":
    raise SystemExit(base.main())
