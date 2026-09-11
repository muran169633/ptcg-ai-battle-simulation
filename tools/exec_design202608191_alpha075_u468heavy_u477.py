#!/usr/bin/env python3
"""Audit or execute one-step U468-heavy alpha0.75 PPO continuation."""

from __future__ import annotations

import hashlib
import importlib.util
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
BASE = TOOLS / "exec_design202608176_alpha075_dualanchor_u477_u478.py"
BASE_SHA256 = "142755830ae80fff6f16513cdd82ae8c564ace875ae2dcffdcbfc1b2ab36485e"

info = BASE.lstat()
if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
    raise RuntimeError("design202608176 launcher is not an authenticated regular file")
if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("design202608176 launcher mismatch")

spec = importlib.util.spec_from_file_location("design202608176_u468heavy_base", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load authenticated design202608176 launcher")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

PARENT = ROOT / "artifacts/design202608170_parent_plus_u476_actor_delta/parent-plus-u476-actor-alpha075.pt"
PARENT_SHA256 = "e01d9161245e3559f4ce21a3f14f6e09a1a4ba2b84c0f2ba1a59f15eda7c85eb"

base.SELF = Path(__file__).resolve()
base.SCHEMA = "ptcg-design202608191-alpha075-u468heavy-u477-v1"
base.SEED = 202608191
base.BOOTSTRAP = ROOT / "artifacts/design202608177_alpha075_fresh_training_bootstrap_v2/alpha075-fresh-bootstrap-legacyquota-u476.pt"
base.BOOTSTRAP_SHA256 = "9537dd73afa79aa1bef6801e390eb9b9fd5ce614c00b45cce91f3254155da10d"
base.BOOTSTRAP_MANIFEST = ROOT / "artifacts/design202608177_alpha075_fresh_training_bootstrap_v2/manifest.json"
base.BOOTSTRAP_MANIFEST_SHA256 = "866f5eacacfdfc3fd94e16b7ca2b464d80089d5d4a4adb06f75480924c9f3de1"
base.OUTPUT_ROOT = ROOT / "artifacts/design202608191_alpha075_u468heavy_ppo1x192_u477"
base.OUTPUT_DIR = base.OUTPUT_ROOT / "B_gold_league/seed-202608191"
base.TERMINAL = base.OUTPUT_DIR / "checkpoints/update-0477.pt"
base.ATTEMPT = ROOT / ".ptcg-design202608191-alpha075-u468heavy-u477-attempt.json"
base.LOG = ROOT / "artifacts/design202608191_alpha075_u468heavy_ppo1x192_u477.log"
base.GOLD_QUOTAS = {
    "rank01_flg": 2,
    "rank02_dominic": 6,
    "rank03_dries": 8,
    "rank04_liam": 8,
    "rank06_etoppo": 2,
    "rank08_hancang": 2,
    "rank11_luca": 2,
    "rank12_taichicchi": 2,
    "rank15_jz": 6,
    "rank17_213tubo": 6,
    "rank18_tuna": 2,
    "rank19_szlachetny": 18,
}
base.ANCHOR_QUOTAS = {
    "submitted_u472": 32,
    "u468_beta100": 64,
    "alpha075_parent": 32,
}
base.DEPENDENCIES = dict(base.DEPENDENCIES) | {
    BASE: BASE_SHA256,
    PARENT: PARENT_SHA256,
}

original_derive_command = base.derive_command
original_expected_protocol = base.expected_protocol


def replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def derive_command() -> list[str]:
    command = original_derive_command()
    command.extend([
        "--extra-opponent", str(PARENT), str(base.DECK),
        "--opponent-base-quota", base.opponent_name(PARENT, base.DECK), "32",
    ])
    replace_option(command, "--updates", "477")
    replace_option(command, "--learning-rate", "1e-05")
    replace_option(command, "--value-learning-rate", "3e-06")
    replace_option(command, "--bc-kl-start", "0.005")
    replace_option(command, "--bc-kl-end", "0.004")
    replace_option(command, "--target-kl", "0.001")
    return command


def expected_protocol(command: list[str]):
    value = original_expected_protocol(command)
    value["updates"] = [477]
    value["hyperparameters"].update({
        "actor_learning_rate": 1e-5,
        "value_learning_rate": 3e-6,
        "bc_kl_start": 0.005,
        "bc_kl_end": 0.004,
        "target_kl": 0.001,
    })
    value["route_change"] = {
        "single_update_only": True,
        "u468_beta100_games_per_update": 64,
        "alpha075_parent_games_per_update": 32,
        "submitted_u472_games_per_update": 32,
        "gold_games_per_update": 64,
        "reason": "recover U468 robustness without a second-step overshoot",
    }
    return value


base.derive_command = derive_command
base.expected_protocol = expected_protocol


if __name__ == "__main__":
    raise SystemExit(base.main())
