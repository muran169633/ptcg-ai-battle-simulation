#!/usr/bin/env python3
"""Freeze and launch one fail-closed Marnie v4 actor-only smoke.

Dry-run is the default.  Execution requires ``--execute`` and the exact
manifest SHA-256 emitted by a preceding dry-run.  The child performs one local
training/evaluation smoke only.  After it exits, this launcher independently
audits input stability, exact quotas, actor-only tensor movement, PPO movement,
and the fixed six-opponent stop gates.  A failed posterior audit returns a
non-zero status and makes the learned checkpoint ineligible for continuation.

There is no packaging, upload, submission, or full-training path in this file.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import stat
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
V3_LAUNCHER = ROOT / "tools/run_gold_push_marnie_ppo_v3.py"
V3_LAUNCHER_SHA256 = "bea7737e5c58bf8fca8e1054c1bbe181b818776211be90dcd5475fb0c14d55c0"

if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import run_gold_push_marnie_ppo_v3 as v3  # noqa: E402


if v3.raw_sha256_file(V3_LAUNCHER) != V3_LAUNCHER_SHA256:
    raise RuntimeError("Authenticated v3 launcher SHA-256 mismatch")

base = v3.base
torch = base.torch

OUTPUT = ROOT / "artifacts/gold_push_20260810_v1/ppo_marnie_v4_actoronly_smoke"
SEED = 202608106
POSTERIOR_REJECT_EXIT = 42
INPUT_DRIFT_EXIT = 43

CURRENT_PANEL_AUDIT_SHA256 = (
    "84d3c1bdb315977d2315d52feea34854814ce0b297198ecbaeabfc80a8804bfa"
)
R2_ROOT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "ppo_marnie_tail32_v3_cvar_tailrepair_smoke_r2"
)
R2_LAUNCHER = ROOT / "tools/run_gold_push_marnie_ppo_v3_r2.py"
R2_EVIDENCE_PATHS = {
    "r2_launcher": R2_LAUNCHER,
    "r2_launcher_manifest": R2_ROOT / "launcher_manifest.json",
    "r2_launcher_result": R2_ROOT / "launcher_result.json",
    "r2_run_config": R2_ROOT / "run_config.json",
    "r2_metrics": R2_ROOT / "metrics.jsonl",
    "r2_summary": R2_ROOT / "summary.json",
    "r2_parent_update0": R2_ROOT / "best.pt",
    "r2_learned_update1": R2_ROOT / "last.pt",
}
R2_EVIDENCE_SHA256 = {
    "r2_launcher": "5b3fcaaa0461e49247607799dbb42db9e7b3092e4f1a6d5e5f3977d0b133c38d",
    "r2_launcher_manifest": "79dfe4c29fe4aab840ab0564d112348c1ff4b870a80f5bebdc521773da0ca75d",
    "r2_launcher_result": "ce37b4a96ccdfadbde57ac27e83b38ff42a40a63daff184b664358e493a8c8ab",
    "r2_run_config": "999f231eebac6c6fae3df6150644ba19613703078322e49bc60a1e50a3075645",
    "r2_metrics": "693f2f549c2cec4746080ca3c2ec4ec8a3c4db5dc069e6a88299d05f987b081f",
    "r2_summary": "04f9dd57f2a193494c08137a818449c33a7ef24438801672778db0a086d30779",
    "r2_parent_update0": "0f1e0a8654e692de7af27f095c2b2a5f02c3ada3164b15437bb38f81ce96d4c6",
    "r2_learned_update1": "8dccf4c7fe24a90903b993f9f5290a712ba7cf23c50e7d374521c6ff4560f9da",
}

CURRENT_EXTRA_SHA256 = dict(v3.EXTRA_SHA256)
CURRENT_EXTRA_SHA256["panel_audit"] = CURRENT_PANEL_AUDIT_SHA256

ACTOR_PREFIXES = (
    "actor_query.",
    "actor_key.",
    "actor_residual.",
    "count_head.",
)
VALUE_PREFIX = "value_head."

FULL_PHASE = v3.PHASES["full"]
FIXED_QUOTAS = v3.quotas(FULL_PHASE)
OPPONENT_NAMES = {
    item.label: v3.opponent_name(item.checkpoint, item.deck)
    for item in v3.OPPONENTS
}
PANEL_LABELS = {
    "own_bc": "bc",
    "raihan": OPPONENT_NAMES["raihan_tail"],
    "kdcyberdude": OPPONENT_NAMES["kdcyberdude_tail"],
    "source_marnie": OPPONENT_NAMES["source_marnie_guard"],
    "u472": OPPONENT_NAMES["u472_guard"],
    "froslass": OPPONENT_NAMES["froslass_uptake_guard"],
}
TAIL_LABELS = ("raihan", "kdcyberdude")
BOTTOM4_LABELS = ("raihan", "kdcyberdude", "source_marnie", "froslass")
GUARD_LABELS = ("own_bc", "source_marnie", "u472", "froslass")

EXPECTED_CONFIG: dict[str, Any] = {
    "bc_checkpoint": str(base.LEARNER),
    "kl_reference_checkpoint": str(base.LEARNER),
    "updates": 1,
    "games_per_update": 192,
    "ppo_epochs": 1,
    "minibatch_size": 1024,
    "learning_rate": 1.5e-6,
    "value_learning_rate": 1e-8,
    "weight_decay": 0.0,
    "value_coefficient": 0.0,
    "value_trunk_gradient_scale": 0.0,
    "trainable_scope": "heads",
    "max_grad_norm": 0.25,
    "bc_kl_start": 0.04,
    "bc_kl_end": 0.04,
    "target_kl": 5e-5,
    "bc_replay_steps": 2,
    "bc_replay_lr_scale": 0.10,
    "eval_interval": 1,
    "eval_games": 128,
    "seed": SEED,
}

POSTERIOR_THRESHOLDS = {
    "approx_kl_max": 2e-5,
    "clip_fraction_required": 0.0,
    "bc_anchor_kl_max": 5e-4,
    "actor_head_relative_l2_max": 5e-5,
    "guard_drop_max": 2.0 / 128.0,
    "one_tail_improvement_min": 2.0 / 128.0,
}


@dataclass(frozen=True)
class Phase:
    name: str = "smoke"
    updates: int = 1
    games_per_update: int = 192
    eval_games: int = 128
    seed: int = SEED


PHASE = Phase()


@dataclass(frozen=True)
class Preflight:
    phase: Phase
    target: Path
    command: list[str]
    manifest: dict[str, Any]
    manifest_sha256: str


@dataclass
class InputLock:
    label: str
    path: Path
    expected_sha256: str
    handle: BinaryIO
    identity: tuple[int, int, int]


def set_single_value(command: list[str], flag: str, value: str) -> None:
    v3.set_single_value(command, flag, value)


def build_command() -> list[str]:
    command = v3.build_command(FULL_PHASE)
    replacements = {
        "--output-dir": str(OUTPUT),
        "--updates": "1",
        "--ppo-epochs": "1",
        "--minibatch-size": "1024",
        "--learning-rate": "0.0000015",
        "--value-learning-rate": "0.00000001",
        "--weight-decay": "0.0",
        "--value-coefficient": "0.0",
        "--value-trunk-gradient-scale": "0.0",
        "--trainable-scope": "heads",
        "--max-grad-norm": "0.25",
        "--kl-reference-checkpoint": str(base.LEARNER),
        "--bc-kl-start": "0.04",
        "--bc-kl-end": "0.04",
        "--target-kl": "0.00005",
        "--bc-replay-steps": "2",
        "--bc-replay-lr-scale": "0.10",
        "--eval-interval": "1",
        "--eval-games": "128",
        "--seed": str(SEED),
    }
    for flag, value in replacements.items():
        set_single_value(command, flag, value)
    base.assert_cli_contract(command)
    return command


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def validate_r2_diagnosis() -> dict[str, Any]:
    result = read_json(R2_EVIDENCE_PATHS["r2_launcher_result"])
    summary = read_json(R2_EVIDENCE_PATHS["r2_summary"])
    rows = read_jsonl(R2_EVIDENCE_PATHS["r2_metrics"])
    if result.get("return_code") != 0 or result.get("inputs_unchanged_after_child") is not True:
        raise ValueError("r2 smoke was not a valid completed launcher run")
    if len(rows) != 1 or rows[0].get("update") != 1:
        raise ValueError("r2 metrics are not exactly one learned update")
    metrics = rows[0]
    if float(summary.get("initial_selection_score", -1.0)) != 0.53125:
        raise ValueError("r2 initial minimum drifted")
    if float(metrics.get("selection_score", -1.0)) != 0.4375:
        raise ValueError("r2 learned minimum drifted")
    initial = summary["initial_evaluation_by_opponent"]
    learned = metrics["evaluation_by_opponent"]
    expected_wins = {
        "own_bc": (17, 14),
        "raihan": (18, 15),
        "kdcyberdude": (18, 17),
        "source_marnie": (19, 16),
        "u472": (20, 19),
        "froslass": (20, 16),
    }
    observed_wins = {
        label: (int(initial[name]["wins"]), int(learned[name]["wins"]))
        for label, name in PANEL_LABELS.items()
    }
    if observed_wins != expected_wins:
        raise ValueError(f"r2 opponent diagnosis drifted: {observed_wins}")
    optimization = metrics["optimization"]
    return {
        "status": "experimentally_rejected_no_full",
        "initial_min": 0.53125,
        "learned_min": 0.4375,
        "initial_total_wins": 112,
        "learned_total_wins": 97,
        "games_per_opponent": 32,
        "wins_by_opponent": observed_wins,
        "approx_kl": float(optimization["approx_kl"]),
        "clip_fraction": float(optimization["clip_fraction"]),
        "actor_grad_norm": float(optimization["actor_grad_norm"]),
        "value_grad_norm": float(optimization["value_grad_norm"]),
        "bc_replay_grad_norm": float(metrics["bc_replay"]["grad_norm"]),
        "learned_checkpoint_may_seed_v4": False,
    }


def collect_inputs() -> dict[str, dict[str, Any]]:
    records = base.collect_file_records()
    records["v1_launcher"] = records.pop("launcher")
    for label, path in v3.EXTRA_PATHS.items():
        records[label] = base.validate_file_binding(
            label,
            path,
            CURRENT_EXTRA_SHA256[label],
        )
    records["v3_launcher"] = base.validate_file_binding(
        "v3_launcher",
        V3_LAUNCHER,
        V3_LAUNCHER_SHA256,
    )
    for label, path in R2_EVIDENCE_PATHS.items():
        records[label] = base.validate_file_binding(
            label,
            path,
            R2_EVIDENCE_SHA256[label],
        )
    records["launcher"] = {
        "path": str(SELF),
        "resolved_path": str(SELF.resolve()),
        "bytes": SELF.stat().st_size,
        "sha256": v3.raw_sha256_file(SELF),
        "protocol_pins_live_launcher_sha256": True,
    }
    return records


def command_values(command: Sequence[str]) -> dict[str, str]:
    return {
        flag: base.command_value(command, flag)
        for flag in (
            "--bc-checkpoint",
            "--kl-reference-checkpoint",
            "--updates",
            "--games-per-update",
            "--ppo-epochs",
            "--minibatch-size",
            "--learning-rate",
            "--value-learning-rate",
            "--weight-decay",
            "--value-coefficient",
            "--value-trunk-gradient-scale",
            "--trainable-scope",
            "--max-grad-norm",
            "--bc-kl-start",
            "--bc-kl-end",
            "--target-kl",
            "--bc-replay-steps",
            "--bc-replay-lr-scale",
            "--eval-games",
            "--seed",
        )
    }


def assert_actor_only_cli(command: Sequence[str]) -> dict[str, Any]:
    observed = command_values(command)
    expected = {
        "--bc-checkpoint": str(base.LEARNER),
        "--kl-reference-checkpoint": str(base.LEARNER),
        "--updates": "1",
        "--games-per-update": "192",
        "--ppo-epochs": "1",
        "--minibatch-size": "1024",
        "--learning-rate": "0.0000015",
        "--value-learning-rate": "0.00000001",
        "--weight-decay": "0.0",
        "--value-coefficient": "0.0",
        "--value-trunk-gradient-scale": "0.0",
        "--trainable-scope": "heads",
        "--max-grad-norm": "0.25",
        "--bc-kl-start": "0.04",
        "--bc-kl-end": "0.04",
        "--target-kl": "0.00005",
        "--bc-replay-steps": "2",
        "--bc-replay-lr-scale": "0.10",
        "--eval-games": "128",
        "--seed": str(SEED),
    }
    if observed != expected:
        raise RuntimeError(f"Actor-only CLI contract drifted: {observed}")
    if "--resume" in command:
        raise RuntimeError("v4 actor-only smoke must start fresh")
    if command.count("--extra-opponent") != 5:
        raise RuntimeError("v4 must retain exactly five external opponents")
    quotas = {
        command[index + 1]: int(command[index + 2])
        for index, token in enumerate(command)
        if token == "--opponent-base-quota"
    }
    if quotas != FIXED_QUOTAS:
        raise RuntimeError(f"v4 fixed quotas drifted: {quotas}")
    return {
        "verified": True,
        "trainer_sha256": base.TRAIN_PPO_SHA256,
        "mechanism": {
            "trunk": "frozen by trainable-scope heads",
            "value_loss": "zero coefficient",
            "value_weight_decay": "zero global weight decay",
            "value_optimizer_lr": 1e-8,
            "posterior_requirement": "all trunk and value tensor raw bytes equal update0",
        },
        "exact_command_values": observed,
        "fixed_quotas": quotas,
    }


def build_preflight() -> Preflight:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Launcher must run from repository root: {ROOT}")
    base.assert_target_absent(OUTPUT)
    inputs = collect_inputs()
    diagnosis = validate_r2_diagnosis()
    command = build_command()
    actor_only_contract = assert_actor_only_cli(command)

    parent_checkpoint = base.validate_checkpoint(
        "raw_candidate",
        v3.RAW_CANDIDATE,
        base.PPO_FEATURE,
        base.MARNIE_DECK_HASH,
    )
    if parent_checkpoint["update"] != 0:
        raise RuntimeError("Expanded frozen parent is no longer update zero")
    learner_checkpoint = base.validate_checkpoint(
        "learner_bc",
        base.LEARNER,
        base.BC_FEATURE,
        base.MARNIE_DECK_HASH,
    )
    for item in v3.OPPONENTS:
        expected_feature = (
            base.PPO_FEATURE if item.label == "u472_guard" else base.BC_FEATURE
        )
        expected_deck = (
            base.FROSLASS_DECK_HASH
            if item.label == "froslass_uptake_guard"
            else base.MARNIE_DECK_HASH
        )
        checkpoint = base.validate_checkpoint(
            item.label,
            item.checkpoint,
            expected_feature,
            expected_deck,
        )
        if item.label == "u472_guard" and checkpoint["update"] != 472:
            raise RuntimeError("Frozen U472 guard is no longer update 472")
    replay = base.validate_replay(base.REPLAY)

    manifest: dict[str, Any] = {
        "schema_version": "ptcg-gold-push-marnie-v4-actoronly-smoke-launch-v1",
        "status": "frozen_before_execute",
        "phase": "smoke_only",
        "output_dir": str(OUTPUT),
        "expected_learned_checkpoint": str(OUTPUT / "checkpoints/update-0001.pt"),
        "candidate_for_posterior_audit": str(OUTPUT / "last.pt"),
        "inputs": inputs,
        "rejected_r2_diagnosis": diagnosis,
        "fresh_parent": {
            "bc_checkpoint": str(base.LEARNER),
            "bc_sha256": base.FILE_SHA256["learner_bc"],
            "expanded_update0_checkpoint": str(v3.RAW_CANDIDATE),
            "expanded_update0_sha256": CURRENT_EXTRA_SHA256["raw_candidate"],
            "expanded_checkpoint_validation": parent_checkpoint,
            "learner_checkpoint_validation": learner_checkpoint,
            "learned_r2_checkpoint_used": False,
        },
        "actor_only_contract": actor_only_contract,
        "rollout": {
            "updates": 1,
            "games_per_update": 192,
            "exact_fixed_quotas": FIXED_QUOTAS,
            "every_quota_even": all(value % 2 == 0 for value in FIXED_QUOTAS.values()),
            "exact_half_each_seat": True,
            "terminal_reward": {"win": 1.0, "loss_draw_failure": 0.0},
        },
        "optimization": {
            "ppo_epochs": 1,
            "minibatch_size": 1024,
            "actor_learning_rate": 1.5e-6,
            "actor_max_grad_norm": 0.25,
            "trainable_scope": "heads",
            "value_coefficient": 0.0,
            "value_trunk_gradient_scale": 0.0,
            "value_learning_rate": 1e-8,
            "weight_decay": 0.0,
            "kl_reference": "same frozen learner BC",
            "bc_kl": {"start": 0.04, "end": 0.04},
            "target_kl": 5e-5,
            "ordered_bc_replay": {"steps": 2, "lr_scale": 0.10},
        },
        "selection": {
            "fixed_six_opponents": PANEL_LABELS,
            "games_per_opponent_initial_and_post": 128,
            "internal_aggregation": "min",
            "engine_rng_seed_control": False,
            "not_claimed_as_paired": True,
        },
        "posterior_stop_gates": {
            "integrity": {
                "valid_games_exact": 192,
                "attempted_games_exact": 192,
                "hard_failures_exact": 0,
                "invalid_replacements_exact": 0,
                "actual_quotas_equal_frozen_quotas": True,
                "actual_seat_quotas_exact_halves": True,
            },
            "actor_only": {
                "all_non_actor_tensors_byte_equal_update0": True,
                "all_value_tensors_byte_equal_update0": True,
                "all_trunk_tensors_byte_equal_update0": True,
                "value_grad_norm_exact": 0.0,
                "actor_head_relative_l2_max": POSTERIOR_THRESHOLDS["actor_head_relative_l2_max"],
            },
            "movement": {
                "approx_kl_max": POSTERIOR_THRESHOLDS["approx_kl_max"],
                "clip_fraction_exact": 0.0,
                "bc_anchor_kl_max": POSTERIOR_THRESHOLDS["bc_anchor_kl_max"],
                "epochs_completed_exact": 1,
                "early_stop_required": False,
            },
            "replay": {
                "steps_exact": 2,
                "rows_exact": 512,
                "learning_rate_exact": 1.5e-7,
                "loss_mode_exact": "ordered",
            },
            "performance": {
                "post_min_at_least_initial": True,
                "post_macro_at_least_initial": True,
                "post_raihan_kd_mean_at_least_initial": True,
                "post_bottom4_mean_at_least_initial": True,
                "each_guard_drop_max": POSTERIOR_THRESHOLDS["guard_drop_max"],
                "at_least_one_tail_improvement": POSTERIOR_THRESHOLDS["one_tail_improvement_min"],
                "invalid_evaluation_games_exact": 0,
                "outcomes_sum_to_valid_games": True,
                "reported_win_rates_exact": True,
            },
            "failure_action": "reject learned checkpoint, stop route, never launch full",
            "pass_action": "eligible only for separate interleaved confirmation and hybrid behavior audit",
        },
        "replay": replay,
        "command": command,
        "command_sha256": base.sha256_json(command),
        "scope": {
            "one_local_smoke": True,
            "full_training": False,
            "external_confirmation": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }
    return Preflight(PHASE, OUTPUT, command, manifest, base.sha256_json(manifest))


def tensor_state_audit(
    parent_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    parent_keys = set(parent_state)
    candidate_keys = set(candidate_state)
    if parent_keys != candidate_keys:
        return {
            "schema_exact": False,
            "missing": sorted(parent_keys - candidate_keys),
            "unexpected": sorted(candidate_keys - parent_keys),
            "pass": False,
        }
    actor_delta_sq = 0.0
    actor_base_sq = 0.0
    actor_max_abs = 0.0
    actor_changed = False
    shape_mismatches: list[str] = []
    non_actor_changed: list[str] = []
    value_changed: list[str] = []
    trunk_changed: list[str] = []
    nonfinite: list[str] = []
    for name in sorted(parent_keys):
        parent = parent_state[name].detach().cpu().contiguous()
        candidate = candidate_state[name].detach().cpu().contiguous()
        if parent.shape != candidate.shape or parent.dtype != candidate.dtype:
            shape_mismatches.append(name)
            continue
        if not torch.isfinite(candidate.float()).all():
            nonfinite.append(name)
        is_actor = name.startswith(ACTOR_PREFIXES)
        numerically_equal = torch.equal(parent, candidate)
        byte_equal = torch.equal(
            parent.reshape(-1).view(torch.uint8),
            candidate.reshape(-1).view(torch.uint8),
        )
        if is_actor:
            delta = candidate.float() - parent.float()
            actor_delta_sq += float((delta * delta).sum())
            actor_base_sq += float((parent.float() * parent.float()).sum())
            actor_max_abs = max(actor_max_abs, float(delta.abs().max()))
            actor_changed = actor_changed or not numerically_equal
        elif not byte_equal:
            non_actor_changed.append(name)
            if name.startswith(VALUE_PREFIX):
                value_changed.append(name)
            else:
                trunk_changed.append(name)
    relative_l2 = math.sqrt(actor_delta_sq) / max(math.sqrt(actor_base_sq), 1e-30)
    passed = (
        not shape_mismatches
        and not non_actor_changed
        and not value_changed
        and not trunk_changed
        and not nonfinite
        and actor_changed
        and relative_l2 <= POSTERIOR_THRESHOLDS["actor_head_relative_l2_max"]
    )
    return {
        "schema_exact": not shape_mismatches,
        "shape_mismatches": shape_mismatches,
        "non_actor_tensors_byte_equal": not non_actor_changed,
        "non_actor_changed": non_actor_changed,
        "value_tensors_byte_equal": not value_changed,
        "value_changed": value_changed,
        "trunk_tensors_byte_equal": not trunk_changed,
        "trunk_changed": trunk_changed,
        "candidate_tensors_finite": not nonfinite,
        "nonfinite": nonfinite,
        "actor_changed": actor_changed,
        "actor_delta_l2": math.sqrt(actor_delta_sq),
        "actor_base_l2": math.sqrt(actor_base_sq),
        "actor_relative_l2": relative_l2,
        "actor_relative_l2_max": POSTERIOR_THRESHOLDS["actor_head_relative_l2_max"],
        "actor_max_abs_delta": actor_max_abs,
        "pass": passed,
    }


def performance_gate(
    initial_by_opponent: Mapping[str, Mapping[str, Any]],
    post_by_opponent: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_names = set(PANEL_LABELS.values())
    if set(initial_by_opponent) != expected_names or set(post_by_opponent) != expected_names:
        return {
            "opponent_set_exact": False,
            "expected": sorted(expected_names),
            "initial": sorted(initial_by_opponent),
            "post": sorted(post_by_opponent),
            "pass": False,
        }
    initial_rates: dict[str, float] = {}
    post_rates: dict[str, float] = {}
    evaluation_shape_ok = True
    outcome_accounting_ok = True
    reported_rate_ok = True
    invalid_games = 0
    for label, name in PANEL_LABELS.items():
        initial_row = initial_by_opponent[name]
        post_row = post_by_opponent[name]
        evaluation_shape_ok &= (
            int(initial_row["valid_games"]) == 128
            and int(post_row["valid_games"]) == 128
        )
        outcome_accounting_ok &= (
            int(initial_row["wins"])
            + int(initial_row["losses"])
            + int(initial_row["draws"])
            == 128
            and int(post_row["wins"])
            + int(post_row["losses"])
            + int(post_row["draws"])
            == 128
        )
        invalid_games += int(initial_row["invalid_games"]) + int(post_row["invalid_games"])
        initial_rates[label] = float(initial_row["win_rate"])
        post_rates[label] = float(post_row["win_rate"])
        reported_rate_ok &= (
            initial_rates[label] == int(initial_row["wins"]) / 128.0
            and post_rates[label] == int(post_row["wins"]) / 128.0
        )

    def mean(labels: Sequence[str], values: Mapping[str, float]) -> float:
        return sum(values[label] for label in labels) / len(labels)

    all_labels = tuple(PANEL_LABELS)
    initial_min = min(initial_rates.values())
    post_min = min(post_rates.values())
    initial_macro = mean(all_labels, initial_rates)
    post_macro = mean(all_labels, post_rates)
    initial_tail = mean(TAIL_LABELS, initial_rates)
    post_tail = mean(TAIL_LABELS, post_rates)
    initial_bottom4 = mean(BOTTOM4_LABELS, initial_rates)
    post_bottom4 = mean(BOTTOM4_LABELS, post_rates)
    deltas = {label: post_rates[label] - initial_rates[label] for label in all_labels}
    guard_floor = -POSTERIOR_THRESHOLDS["guard_drop_max"]
    tail_improvement = max(deltas[label] for label in TAIL_LABELS)
    checks = {
        "evaluation_games_exact": evaluation_shape_ok,
        "outcomes_sum_to_valid_games": outcome_accounting_ok,
        "reported_win_rates_exact": reported_rate_ok,
        "invalid_games_zero": invalid_games == 0,
        "min_nonregression": post_min >= initial_min,
        "macro_nonregression": post_macro >= initial_macro,
        "tail_mean_nonregression": post_tail >= initial_tail,
        "bottom4_mean_nonregression": post_bottom4 >= initial_bottom4,
        "each_guard_within_drop_limit": all(deltas[label] >= guard_floor for label in GUARD_LABELS),
        "one_tail_improves_enough": tail_improvement >= POSTERIOR_THRESHOLDS["one_tail_improvement_min"],
    }
    return {
        "opponent_set_exact": True,
        "initial_rates": initial_rates,
        "post_rates": post_rates,
        "deltas": deltas,
        "initial_min": initial_min,
        "post_min": post_min,
        "initial_macro": initial_macro,
        "post_macro": post_macro,
        "initial_tail_mean": initial_tail,
        "post_tail_mean": post_tail,
        "initial_bottom4_mean": initial_bottom4,
        "post_bottom4_mean": post_bottom4,
        "guard_drop_max": POSTERIOR_THRESHOLDS["guard_drop_max"],
        "one_tail_improvement_min": POSTERIOR_THRESHOLDS["one_tail_improvement_min"],
        "invalid_games": invalid_games,
        "checks": checks,
        "pass": all(checks.values()),
    }


def audit_completed_output(preflight: Preflight) -> dict[str, Any]:
    output = preflight.target
    required = {
        "run_config": output / "run_config.json",
        "metrics": output / "metrics.jsonl",
        "summary": output / "summary.json",
        "best": output / "best.pt",
        "last": output / "last.pt",
        "update1": output / "checkpoints/update-0001.pt",
    }
    file_records: dict[str, Any] = {}
    for label, path in required.items():
        base.assert_regular_file(path)
        file_records[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": v3.raw_sha256_file(path),
        }
    if file_records["last"]["sha256"] != file_records["update1"]["sha256"]:
        raise ValueError("last.pt is not byte-identical to update-0001.pt")

    run_config = read_json(required["run_config"])
    metrics_rows = read_jsonl(required["metrics"])
    summary = read_json(required["summary"])
    config = run_config.get("config")
    if not isinstance(config, dict):
        raise ValueError("run_config has no config object")
    config_checks = {
        key: config.get(key) == expected
        for key, expected in EXPECTED_CONFIG.items()
    }
    config_checks["output_dir"] = config.get("output_dir") == str(output)
    config_checks["fixed_quotas"] = config.get("opponent_base_quotas") == FIXED_QUOTAS
    config_checks["no_resume"] = config.get("resume_checkpoint") is None
    trainable = run_config.get("trainable_parameters") or {}
    actor_names = list(trainable.get("actor_parameter_names") or [])
    value_names = list(trainable.get("value_parameter_names") or [])
    trainable_checks = {
        "scope_heads": trainable.get("scope") == "heads",
        "actor_names_heads_only": bool(actor_names) and all(name.startswith(ACTOR_PREFIXES) for name in actor_names),
        "value_names_value_only": bool(value_names) and all(name.startswith(VALUE_PREFIX) for name in value_names),
        "no_trunk_trainable": not any(name.startswith(("transformer.", "categorical_embedding.", "numeric_projection.", "state_", "option_", "kind_embedding.")) for name in actor_names + value_names),
    }
    learner_initialization = run_config.get("learner_initialization") or {}
    kl_reference = run_config.get("kl_reference") or {}
    lineage_checks = {
        "fresh_bc": learner_initialization.get("selection") == "fresh_bc",
        "parent_sha": learner_initialization.get("checkpoint_sha256") == base.FILE_SHA256["learner_bc"],
        "self_kl_path": kl_reference.get("checkpoint") == str(base.LEARNER.resolve()),
        "self_kl_sha": kl_reference.get("checkpoint_sha256") == base.FILE_SHA256["learner_bc"],
    }

    if len(metrics_rows) != 1 or metrics_rows[0].get("update") != 1:
        raise ValueError("Expected exactly one update-1 metrics row")
    metrics = metrics_rows[0]
    rollout = metrics.get("rollout") or {}
    quota = rollout.get("opponent_quota") or {}
    expected_seats = {
        name: {"0": value // 2, "1": value // 2}
        for name, value in FIXED_QUOTAS.items()
    }
    rollout_checks = {
        "valid_games": int(rollout.get("valid_games", -1)) == 192,
        "attempted_games": int(rollout.get("attempted_games", -1)) == 192,
        "failed_started_games": int(rollout.get("failed_started_games", -1)) == 0,
        "hard_failure_attempts": int(rollout.get("hard_failure_attempts", -1)) == 0,
        "actual_quotas": quota.get("actual_quotas") == FIXED_QUOTAS,
        "actual_seat_quotas": quota.get("actual_seat_quotas") == expected_seats,
        "seat_balance_verified": quota.get("seat_balance_verified") is True,
        "actual_max_seat_gap": int(quota.get("actual_max_seat_gap", -1)) == 0,
        "invalid_replacements": all(int(value) == 0 for value in (quota.get("invalid_replacements") or {"missing": 1}).values()),
    }
    optimization = metrics.get("optimization") or {}
    movement_checks = {
        "epochs_completed": int(optimization.get("epochs_completed", -1)) == 1,
        "early_stop_false": optimization.get("early_stop") is False,
        "approx_kl": float(optimization.get("approx_kl", math.inf)) <= POSTERIOR_THRESHOLDS["approx_kl_max"],
        "clip_fraction": float(optimization.get("clip_fraction", math.inf)) == 0.0,
        "bc_anchor_kl": float(optimization.get("bc_anchor_kl", math.inf)) <= POSTERIOR_THRESHOLDS["bc_anchor_kl_max"],
        "value_grad_norm": float(optimization.get("value_grad_norm", math.inf)) == 0.0,
    }
    replay = metrics.get("bc_replay") or {}
    replay_checks = {
        "steps": int(replay.get("steps", -1)) == 2,
        "rows": int(replay.get("rows", -1)) == 512,
        "learning_rate": float(replay.get("learning_rate", math.inf)) == 1.5e-7,
        "loss_mode": replay.get("loss_mode") == "ordered",
    }

    parent = torch.load(v3.RAW_CANDIDATE, map_location="cpu", weights_only=False)
    candidate = torch.load(required["last"], map_location="cpu", weights_only=False)
    if candidate.get("feature_version") != base.PPO_FEATURE or int(candidate.get("update", -1)) != 1:
        raise ValueError("Learned candidate checkpoint identity is invalid")
    tensor_audit = tensor_state_audit(parent["model_state_dict"], candidate["model_state_dict"])
    performance = performance_gate(
        summary.get("initial_evaluation_by_opponent") or {},
        metrics.get("evaluation_by_opponent") or {},
    )
    summary_checks = {
        "updates_completed": int(summary.get("updates_completed", -1)) == 1,
        "initial_selection_matches": float(summary.get("initial_selection_score", math.nan)) == performance.get("initial_min"),
        "post_selection_matches": float(metrics.get("selection_score", math.nan)) == performance.get("post_min"),
    }
    categories = {
        "config": all(config_checks.values()),
        "trainable_scope": all(trainable_checks.values()),
        "lineage": all(lineage_checks.values()),
        "rollout": all(rollout_checks.values()),
        "movement": all(movement_checks.values()),
        "replay": all(replay_checks.values()),
        "tensor_immutability_and_delta": bool(tensor_audit.get("pass")),
        "performance": bool(performance.get("pass")),
        "summary": all(summary_checks.values()),
    }
    passed = all(categories.values())
    return {
        "schema_version": "ptcg-gold-push-marnie-v4-actoronly-posterior-audit-v1",
        "manifest_sha256": preflight.manifest_sha256,
        "pass": passed,
        "decision": "eligible_for_separate_confirmation_only" if passed else "reject_stop_no_full",
        "files": file_records,
        "checks": {
            "categories": categories,
            "config": config_checks,
            "trainable_scope": trainable_checks,
            "lineage": lineage_checks,
            "rollout": rollout_checks,
            "movement": movement_checks,
            "replay": replay_checks,
            "summary": summary_checks,
        },
        "movement_actual": {
            "approx_kl": optimization.get("approx_kl"),
            "clip_fraction": optimization.get("clip_fraction"),
            "bc_anchor_kl": optimization.get("bc_anchor_kl"),
            "value_grad_norm": optimization.get("value_grad_norm"),
        },
        "tensor_audit": tensor_audit,
        "performance_gate": performance,
        "scope": {
            "full_training_launched": False,
            "external_confirmation_performed": False,
            "package_upload_submission_performed": False,
        },
    }


def lock_bindings(preflight: Preflight) -> dict[str, tuple[Path, str]]:
    bindings: dict[str, tuple[Path, str]] = {}
    for label, record in preflight.manifest["inputs"].items():
        path = Path(record["resolved_path"] if label == "python" else record["path"])
        bindings[label] = (path, str(record["sha256"]))
    return bindings


def sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def acquire_input_locks(preflight: Preflight) -> list[InputLock]:
    locks: list[InputLock] = []
    try:
        for label, (path, expected_sha256) in lock_bindings(preflight).items():
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            handle = os.fdopen(fd, "rb", closefd=True)
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or sha256_handle(handle) != expected_sha256:
                raise RuntimeError(f"Execution-time input mismatch for {label}")
            locks.append(InputLock(label, path, expected_sha256, handle, (info.st_dev, info.st_ino, info.st_size)))
    except BaseException:
        release_input_locks(locks)
        raise
    return locks


def assert_bindings_unchanged(preflight: Preflight, locks: Sequence[InputLock]) -> None:
    for locked in locks:
        info = os.fstat(locked.handle.fileno())
        if (info.st_dev, info.st_ino, info.st_size) != locked.identity or sha256_handle(locked.handle) != locked.expected_sha256:
            raise RuntimeError(f"Locked handle changed during launch: {locked.label}")
    for label, (path, expected_sha256) in lock_bindings(preflight).items():
        base.assert_regular_file(path)
        observed = v3.raw_sha256_file(path)
        if observed != expected_sha256:
            raise RuntimeError(
                f"Live input path changed during launch: {label}: expected {expected_sha256}, observed {observed}"
            )


def release_input_locks(locks: Sequence[InputLock]) -> None:
    for locked in reversed(locks):
        try:
            fcntl.flock(locked.handle.fileno(), fcntl.LOCK_UN)
        finally:
            locked.handle.close()


def write_result(target: Path, payload: dict[str, Any]) -> None:
    base.write_exclusive(target / "launcher_result.json", payload)


def execute(preflight: Preflight) -> int:
    locks = acquire_input_locks(preflight)
    try:
        assert_bindings_unchanged(preflight, locks)
        base.assert_target_absent(preflight.target)
        preflight.target.mkdir(parents=False, exist_ok=False, mode=0o700)
        base.write_exclusive(
            preflight.target / "launcher_manifest.json",
            {"manifest_sha256": preflight.manifest_sha256, "manifest": preflight.manifest},
        )
        environment = os.environ.copy()
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            preflight.command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            shell=False,
            check=False,
        )
        try:
            assert_bindings_unchanged(preflight, locks)
        except BaseException as error:
            base.write_exclusive(
                preflight.target / "INVALIDATED_RUN.json",
                {
                    "schema_version": "ptcg-invalidated-locked-training-run-v1",
                    "manifest_sha256": preflight.manifest_sha256,
                    "status": "invalidated_after_child",
                    "reason": str(error),
                    "child_return_code": int(completed.returncode),
                    "may_select_continue_package_or_submit": False,
                },
            )
            write_result(
                preflight.target,
                {
                    "schema_version": "ptcg-gold-push-marnie-v4-actoronly-result-v1",
                    "manifest_sha256": preflight.manifest_sha256,
                    "return_code": INPUT_DRIFT_EXIT,
                    "child_return_code": int(completed.returncode),
                    "posterior_audit_pass": False,
                    "decision": "invalidated_input_drift",
                    "package_upload_submission_performed": False,
                },
            )
            return INPUT_DRIFT_EXIT
        if completed.returncode != 0:
            write_result(
                preflight.target,
                {
                    "schema_version": "ptcg-gold-push-marnie-v4-actoronly-result-v1",
                    "manifest_sha256": preflight.manifest_sha256,
                    "return_code": int(completed.returncode),
                    "child_return_code": int(completed.returncode),
                    "posterior_audit_pass": False,
                    "decision": "child_failed",
                    "inputs_unchanged_after_child": True,
                    "package_upload_submission_performed": False,
                },
            )
            return int(completed.returncode)
        try:
            audit = audit_completed_output(preflight)
        except BaseException as error:
            audit = {
                "schema_version": "ptcg-gold-push-marnie-v4-actoronly-posterior-audit-v1",
                "manifest_sha256": preflight.manifest_sha256,
                "pass": False,
                "decision": "reject_stop_no_full",
                "audit_error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "package_upload_submission_performed": False,
            }
        base.write_exclusive(preflight.target / "posterior_audit.json", audit)
        accepted = audit.get("pass") is True
        return_code = 0 if accepted else POSTERIOR_REJECT_EXIT
        write_result(
            preflight.target,
            {
                "schema_version": "ptcg-gold-push-marnie-v4-actoronly-result-v1",
                "manifest_sha256": preflight.manifest_sha256,
                "return_code": return_code,
                "child_return_code": 0,
                "inputs_unchanged_after_child": True,
                "posterior_audit_pass": accepted,
                "decision": audit.get("decision"),
                "candidate_for_confirmation": str(preflight.target / "last.pt") if accepted else None,
                "full_training_launched": False,
                "package_upload_submission_performed": False,
            },
        )
        return return_code
    finally:
        release_input_locks(locks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute and not args.expected_manifest_sha256:
        raise ValueError("--execute requires --expected-manifest-sha256")
    if not args.execute and args.expected_manifest_sha256:
        raise ValueError("--expected-manifest-sha256 is only valid with --execute")
    preflight = build_preflight()
    print(
        json.dumps(
            {"manifest_sha256": preflight.manifest_sha256, "manifest": preflight.manifest},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    print(shlex.join(preflight.command), flush=True)
    if not args.execute:
        return 0
    if args.expected_manifest_sha256 != preflight.manifest_sha256:
        raise RuntimeError("Manifest SHA-256 mismatch; repeat and review dry-run")
    return execute(preflight)


if __name__ == "__main__":
    raise SystemExit(main())
