#!/usr/bin/env python3
"""Audit or execute one conservative balanced-anchor PPO update from S16."""

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

spec = importlib.util.spec_from_file_location("design202608176_s16_balanced_base", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load authenticated design202608176 launcher")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ALPHA075 = ROOT / "artifacts/design202608170_parent_plus_u476_actor_delta/parent-plus-u476-actor-alpha075.pt"
ALPHA075_SHA256 = "e01d9161245e3559f4ce21a3f14f6e09a1a4ba2b84c0f2ba1a59f15eda7c85eb"
SELECTION = ROOT / "artifacts/design202608196_latestreplay_general_bc_screen128_decision.json"
SELECTION_SHA256 = "246ac1d3dc6f721e636391b4868a3bad69d7729305908ca544bb0acd9124571f"

base.SELF = Path(__file__).resolve()
base.SCHEMA = "ptcg-design202608198-s16-balancedanchor-u477-v1"
base.SEED = 202608198
base.BOOTSTRAP = ROOT / "artifacts/design202608197_s16_fresh_training_bootstrap/s16-fresh-bootstrap-legacyquota-u476.pt"
base.BOOTSTRAP_SHA256 = "a23fa3b4ecbe48839138be5d805a69815a870614753b4d9119a6c51c50f29a56"
base.BOOTSTRAP_MANIFEST = ROOT / "artifacts/design202608197_s16_fresh_training_bootstrap/manifest.json"
base.BOOTSTRAP_MANIFEST_SHA256 = "d21bd6942c95cc6472edd29b1f205223a44bbb0722dbba85583494fc334288bd"
base.REPLAY = ROOT / "data/bc_marnie_top50_current14_latesttop50_trainvalid_through0804_design202608194.zip"
base.REPLAY_SHA256 = "74ebd086d065f0e2ab95dde2192e463c8ba1eb04ab4e40c1876d484f66475c15"
base.OUTPUT_ROOT = ROOT / "artifacts/design202608198_s16_balancedanchor_ppo1x192_u477"
base.OUTPUT_DIR = base.OUTPUT_ROOT / "B_gold_league/seed-202608198"
base.TERMINAL = base.OUTPUT_DIR / "checkpoints/update-0477.pt"
base.ATTEMPT = ROOT / ".ptcg-design202608198-s16-balancedanchor-u477-attempt.json"
base.LOG = ROOT / "artifacts/design202608198_s16_balancedanchor_ppo1x192_u477.log"
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
    ALPHA075: ALPHA075_SHA256,
    SELECTION: SELECTION_SHA256,
}

original_derive_command = base.derive_command
original_expected_protocol = base.expected_protocol


def replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def derive_command() -> list[str]:
    command = original_derive_command()
    command.extend([
        "--extra-opponent", str(ALPHA075), str(base.DECK),
        "--opponent-base-quota", base.opponent_name(ALPHA075, base.DECK), "32",
    ])
    replace_option(command, "--updates", "477")
    replace_option(command, "--learning-rate", "8e-06")
    replace_option(command, "--value-learning-rate", "3e-06")
    replace_option(command, "--bc-kl-start", "0.004")
    replace_option(command, "--bc-kl-end", "0.0035")
    replace_option(command, "--target-kl", "0.0008")
    return command


def expected_protocol(command: list[str]):
    value = original_expected_protocol(command)
    value["updates"] = [477]
    value["hyperparameters"].update({
        "actor_learning_rate": 8e-6,
        "value_learning_rate": 3e-6,
        "bc_kl_start": 0.004,
        "bc_kl_end": 0.0035,
        "target_kl": 0.0008,
    })
    value["route_change"] = {
        "single_update_only": True,
        "parent": "latest-replay general-BC S16",
        "latest_replay_through": "2026-08-04",
        "gold_games_per_update": 96,
        "alpha075_parent_games_per_update": 32,
        "submitted_u472_games_per_update": 32,
        "u468_beta100_games_per_update": 32,
        "reason": "retain S16's three-anchor gains while adding one low-LR PPO signal",
    }
    return value


base.derive_command = derive_command
base.expected_protocol = expected_protocol


if __name__ == "__main__":
    raise SystemExit(base.main())
