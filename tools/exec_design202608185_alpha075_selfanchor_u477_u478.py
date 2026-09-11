#!/usr/bin/env python3
"""Audit or execute alpha0.75 self-anchor three-anchor PPO continuation."""

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

spec = importlib.util.spec_from_file_location("design202608176_selfanchor_base", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load authenticated design202608176 launcher")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

PARENT = ROOT / "artifacts/design202608170_parent_plus_u476_actor_delta/parent-plus-u476-actor-alpha075.pt"
PARENT_SHA256 = "e01d9161245e3559f4ce21a3f14f6e09a1a4ba2b84c0f2ba1a59f15eda7c85eb"

base.SELF = Path(__file__).resolve()
base.SCHEMA = "ptcg-design202608185-alpha075-selfanchor-u477-u478-v1"
base.SEED = 202608185
base.BOOTSTRAP = ROOT / "artifacts/design202608177_alpha075_fresh_training_bootstrap_v2/alpha075-fresh-bootstrap-legacyquota-u476.pt"
base.BOOTSTRAP_SHA256 = "9537dd73afa79aa1bef6801e390eb9b9fd5ce614c00b45cce91f3254155da10d"
base.BOOTSTRAP_MANIFEST = ROOT / "artifacts/design202608177_alpha075_fresh_training_bootstrap_v2/manifest.json"
base.BOOTSTRAP_MANIFEST_SHA256 = "866f5eacacfdfc3fd94e16b7ca2b464d80089d5d4a4adb06f75480924c9f3de1"
base.OUTPUT_ROOT = ROOT / "artifacts/design202608185_alpha075_selfanchor_ppo2x192_u478"
base.OUTPUT_DIR = base.OUTPUT_ROOT / "B_gold_league/seed-202608185"
base.TERMINAL = base.OUTPUT_DIR / "checkpoints/update-0478.pt"
base.ATTEMPT = ROOT / ".ptcg-design202608185-alpha075-selfanchor-u477-u478-attempt.json"
base.LOG = ROOT / "artifacts/design202608185_alpha075_selfanchor_ppo2x192_u478.log"
base.GOLD_QUOTAS = {
    "rank01_flg": 4,
    "rank02_dominic": 8,
    "rank03_dries": 12,
    "rank04_liam": 12,
    "rank06_etoppo": 4,
    "rank08_hancang": 4,
    "rank11_luca": 4,
    "rank12_taichicchi": 4,
    "rank15_jz": 8,
    "rank17_213tubo": 10,
    "rank18_tuna": 4,
    "rank19_szlachetny": 22,
}
base.ANCHOR_QUOTAS = {
    "submitted_u472": 32,
    "u468_beta100": 32,
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
    replace_option(command, "--learning-rate", "1.6e-05")
    replace_option(command, "--value-learning-rate", "4e-06")
    replace_option(command, "--bc-kl-start", "0.0045")
    replace_option(command, "--bc-kl-end", "0.0035")
    replace_option(command, "--target-kl", "0.0012")
    return command


def expected_protocol(command: list[str]):
    value = original_expected_protocol(command)
    value["hyperparameters"].update({
        "actor_learning_rate": 1.6e-5,
        "value_learning_rate": 4e-6,
        "bc_kl_start": 0.0045,
        "bc_kl_end": 0.0035,
        "target_kl": 0.0012,
    })
    value["route_change"] = {
        "self_anchor_added": True,
        "self_anchor_games_per_update": 32,
        "gold_games_per_update_reduced_from": 128,
        "gold_games_per_update_reduced_to": 96,
        "reason": "retain alpha075 and submitted-U472 behavior while preserving U468 exposure",
    }
    return value


base.derive_command = derive_command
base.expected_protocol = expected_protocol


if __name__ == "__main__":
    raise SystemExit(base.main())
