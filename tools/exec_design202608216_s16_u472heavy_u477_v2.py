#!/usr/bin/env python3
"""Audit or execute one PPO-strengthening continuation from the 202608200 anchor-heavy line."""

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
    raise RuntimeError("design202608200 launcher is not an authenticated regular file")
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("design202608200 launcher mismatch")

spec = importlib.util.spec_from_file_location(
    "design202608200_s16_u472heavy_source", SOURCE
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load authenticated design202608200 launcher")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
base = source.base


base.SELF = Path(__file__).resolve()
base.SCHEMA = "ptcg-design202608216-s16-u472heavy-u477-v2"
base.SEED = 202608216
base.OUTPUT_ROOT = ROOT / "artifacts/design202608216_s16_u472heavy_ppo1x192_u477_v2"
base.OUTPUT_DIR = base.OUTPUT_ROOT / "B_gold_league/seed-202608216"
base.TERMINAL = base.OUTPUT_DIR / "checkpoints/update-0477.pt"
base.ATTEMPT = ROOT / ".ptcg-design202608216-s16-u472heavy-u477-v2-attempt.json"
base.LOG = ROOT / "artifacts/design202608216_s16_u472heavy_ppo1x192_u477_v2.log"
base.ANCHOR_QUOTAS = {
    "submitted_u472": 56,
    "u468_beta100": 24,
    "alpha075_parent": 16,
}
base.DEPENDENCIES = dict(base.DEPENDENCIES) | {SOURCE: SOURCE_SHA256}

SUBMITTED_U472 = (
    ROOT
    / "artifacts/ppo_guardedu468_updatedreplay_g8_ppo4x96_u472_freshjointactor6_s8_design202608148/ppo_stage/B_gold_league/seed-202608148/checkpoints/update-0472.pt"
)
U468_BETA100 = (
    ROOT
    / "artifacts/ppo_u468_p12delta_direction_beta050_075_100_design202608092/transport-beta-100.pt"
)
ALPHA075 = (
    ROOT
    / "artifacts/design202608170_parent_plus_u476_actor_delta/parent-plus-u476-actor-alpha075.pt"
)
DECK = base.DECK

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
    replace_option(command, "--learning-rate", "8e-06")
    replace_option(command, "--value-learning-rate", "4e-06")
    replace_option(command, "--bc-kl-start", "0.0045")
    replace_option(command, "--bc-kl-end", "0.0040")
    replace_option(command, "--target-kl", "0.0012")

    replace_quota(
        command,
        base.opponent_name(ALPHA075, DECK),
        "16",
    )
    replace_quota(
        command,
        base.opponent_name(SUBMITTED_U472, DECK),
        "56",
    )
    replace_quota(
        command,
        base.opponent_name(U468_BETA100, DECK),
        "24",
    )

    return command


def expected_protocol(command: list[str]):
    value = source_expected_protocol(command)
    value["updates"] = [477]
    value["hyperparameters"].update(
        {
            "actor_learning_rate": 8e-6,
            "value_learning_rate": 4e-6,
            "bc_kl_start": 0.0045,
            "bc_kl_end": 0.0040,
            "target_kl": 0.0012,
        }
    )
    value["anchor_quotas"] = {
        "alpha075_parent": 16,
        "submitted_u472": 56,
        "u468_beta100": 24,
    }
    value["route_change"] = {
        "single_update_only": True,
        "parent": "latest-replay general-BC S16",
        "latest_replay_through": "2026-08-04",
        "gold_games_per_update": 96,
        "alpha075_parent_games_per_update": 16,
        "submitted_u472_games_per_update": 56,
        "u468_beta100_games_per_update": 24,
        "reason": "stronger PPO push with slightly looser KL and heavier anchor u472 exposure",
        "predecessor_u477": {
            "checkpoint_sha256": "571c3760af4d24a92847cf0057f95e774a2b23e96d2e99841918b28913b81f8a",
            "promoted": False,
            "submitted_u472_result": "63-65",
        },
    }
    return value


base.derive_command = derive_command
base.expected_protocol = expected_protocol


if __name__ == "__main__":
    raise SystemExit(base.main())
