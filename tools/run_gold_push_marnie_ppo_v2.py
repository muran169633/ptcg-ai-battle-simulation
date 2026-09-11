#!/usr/bin/env python3
"""Frozen conservative v2 follow-up to the Marnie failure-tail PPO run.

The v2 protocol is derived only from the completed v1 ``summary.json`` and
``metrics.jsonl``.  It starts fresh from the same BC learner, preserves the
same fixed evaluation/promotion gate, and changes only four training levers:

* two full updates instead of four;
* actor LR 8e-6 instead of 1.2e-5;
* constant BC-KL coefficient 0.024 instead of 0.020 -> 0.016;
* 96/16/16/32/32 quotas to add modest source-Marnie and U472 exposure.

Dry-run is the default.  Training requires ``--execute`` plus the exact
manifest SHA printed by a preceding dry run.  This launcher never packages,
uploads, evaluates pending strategy sweeps, or submits a model.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Sequence


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
BASE_LAUNCHER = ROOT / "tools/run_gold_push_marnie_ppo.py"
BASE_LAUNCHER_SHA256 = "efc2c308c6165201604aed289eafc22d8c0ef771e4d6bcd6924616a37387d2ac"


def raw_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if raw_sha256_file(BASE_LAUNCHER) != BASE_LAUNCHER_SHA256:
    raise RuntimeError("Authenticated v1 launcher SHA-256 mismatch")
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import run_gold_push_marnie_ppo as base  # noqa: E402


V1_OUTPUT = ROOT / "artifacts/gold_push_20260810_v1/ppo_marnie_tail32_v1"
V1_SUMMARY = V1_OUTPUT / "summary.json"
V1_METRICS = V1_OUTPUT / "metrics.jsonl"
V1_SUMMARY_SHA256 = "d524b219c51deae58590218d18a38c2790037cee8c57c01dfbfa74f17945d6db"
V1_METRICS_SHA256 = "54f27c02f8388c9a80faee843822d5068d8b359b5db29e2ec4370bd248c267a9"

OUTPUT_ROOT = (
    ROOT / "artifacts/gold_push_20260810_v1/ppo_marnie_tail32_v2_lightanchor"
)
SMOKE_OUTPUT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/ppo_marnie_tail32_v2_lightanchor_smoke"
)
UNCHANGED_PROMOTION_THRESHOLD = 0.484375


@dataclass(frozen=True)
class Phase:
    name: str
    updates: int
    games_per_update: int
    own_bc_quota: int
    lucario_quota: int
    froslass_quota: int
    source_marnie_quota: int
    u472_quota: int
    eval_games: int
    eval_interval: int
    seed: int


PHASES = {
    "smoke": Phase(
        name="smoke",
        updates=1,
        games_per_update=96,
        own_bc_quota=48,
        lucario_quota=8,
        froslass_quota=8,
        source_marnie_quota=16,
        u472_quota=16,
        eval_games=32,
        eval_interval=1,
        seed=202608101,
    ),
    "full": Phase(
        name="full",
        updates=2,
        games_per_update=192,
        own_bc_quota=96,
        lucario_quota=16,
        froslass_quota=16,
        source_marnie_quota=32,
        u472_quota=32,
        eval_games=256,
        eval_interval=1,
        seed=202608102,
    ),
}


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


def output_dir(phase: Phase) -> Path:
    return SMOKE_OUTPUT if phase.name == "smoke" else OUTPUT_ROOT


def set_single_value(command: list[str], flag: str, value: str) -> None:
    if command.count(flag) != 1:
        raise RuntimeError(f"Expected exactly one locked command flag: {flag}")
    command[command.index(flag) + 1] = value


def strip_repeated_triplets(command: list[str], flag: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(command):
        if command[index] == flag:
            if index + 2 >= len(command):
                raise RuntimeError(f"Malformed repeated command flag: {flag}")
            index += 3
            continue
        result.append(command[index])
        index += 1
    return result


def quotas(phase: Phase) -> dict[str, int]:
    opponent_values = (
        phase.lucario_quota,
        phase.froslass_quota,
        phase.source_marnie_quota,
        phase.u472_quota,
    )
    values = {"bc": phase.own_bc_quota}
    for opponent, quota in zip(base.OPPONENTS, opponent_values, strict=True):
        name = base.opponent_name(opponent.checkpoint, opponent.deck)
        if name in values:
            raise RuntimeError(f"Duplicate opponent name: {name}")
        values[name] = quota
    if sum(values.values()) != phase.games_per_update:
        raise RuntimeError("v2 fixed quotas do not sum to games-per-update")
    if any(value <= 0 or value % 2 for value in values.values()):
        raise RuntimeError("v2 quotas must be positive and exactly seat-balanced")
    return values


def build_command(phase: Phase) -> list[str]:
    v1_phase = base.PHASES[phase.name]
    command = base.build_command(v1_phase)
    replacements = {
        "--output-dir": str(output_dir(phase)),
        "--updates": str(phase.updates),
        "--games-per-update": str(phase.games_per_update),
        "--learning-rate": "0.000008",
        "--bc-kl-start": "0.024",
        "--bc-kl-end": "0.024",
        "--eval-games": str(phase.eval_games),
        "--eval-interval": str(phase.eval_interval),
        "--seed": str(phase.seed),
    }
    for flag, value in replacements.items():
        set_single_value(command, flag, value)
    command = strip_repeated_triplets(command, "--opponent-base-quota")
    for name, quota in quotas(phase).items():
        command.extend(["--opponent-base-quota", name, str(quota)])
    base.assert_cli_contract(command)
    return command


def short_opponent_name(name: str) -> str:
    if name == "bc":
        return "own_bc"
    if name.startswith("best@" + base.LUCARIO_DECK_HASH):
        return "lucario"
    if name.startswith("best@" + base.FROSLASS_DECK_HASH):
        return "froslass"
    if name.startswith("best@" + base.MARNIE_DECK_HASH):
        return "source_marnie"
    if name.startswith("update-0472@" + base.MARNIE_DECK_HASH):
        return "u472"
    raise ValueError(f"Unexpected fixed-panel opponent: {name}")


def compact_panel(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        short_opponent_name(name): {
            "wins": int(value["wins"]),
            "losses": int(value["losses"]),
            "invalid_games": int(value["invalid_games"]),
            "win_rate": float(value["win_rate"]),
        }
        for name, value in raw.items()
    }


def load_v1_evidence() -> dict[str, Any]:
    summary = json.loads(V1_SUMMARY.read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in V1_METRICS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [int(row.get("update", -1)) for row in metrics] != [1, 2, 3, 4]:
        raise ValueError("v1 metrics updates are not exactly 1..4")
    if int(summary.get("updates_completed", -1)) != 4:
        raise ValueError("v1 summary did not complete four updates")
    initial_score = float(summary.get("initial_selection_score", -1.0))
    best_score = float(summary.get("best_selection_score", -1.0))
    if initial_score != UNCHANGED_PROMOTION_THRESHOLD or best_score != initial_score:
        raise ValueError("v1 fixed-panel promotion baseline drifted")

    evaluated = {int(row["update"]): row for row in metrics if row.get("selection_score") is not None}
    if set(evaluated) != {2, 4}:
        raise ValueError("v1 fixed panels are not exactly update2/update4")
    if float(evaluated[2]["selection_score"]) != 0.46484375:
        raise ValueError("v1 update2 minimum drifted")
    if float(evaluated[4]["selection_score"]) != 0.46875:
        raise ValueError("v1 update4 minimum drifted")
    if any(int(row["rollout"].get("failed_started_games", -1)) != 0 for row in metrics):
        raise ValueError("v1 failure-tail audit no longer has zero failed attempts")

    optimization = {
        str(row["update"]): {
            "approx_kl": float(row["optimization"]["approx_kl"]),
            "bc_anchor_kl": float(row["optimization"]["bc_anchor_kl"]),
            "clip_fraction": float(row["optimization"]["clip_fraction"]),
            "actor_grad_norm": float(row["optimization"]["actor_grad_norm"]),
            "bc_kl_coefficient": float(row["optimization"]["bc_kl_coefficient"]),
        }
        for row in metrics
    }
    return {
        "source_files": {
            "summary": {"path": str(V1_SUMMARY), "sha256": V1_SUMMARY_SHA256},
            "metrics": {"path": str(V1_METRICS), "sha256": V1_METRICS_SHA256},
        },
        "initial": {
            "selection_score": initial_score,
            "panel": compact_panel(summary["initial_evaluation_by_opponent"]),
        },
        "update2": {
            "selection_score": float(evaluated[2]["selection_score"]),
            "panel": compact_panel(evaluated[2]["evaluation_by_opponent"]),
        },
        "update4": {
            "selection_score": float(evaluated[4]["selection_score"]),
            "panel": compact_panel(evaluated[4]["evaluation_by_opponent"]),
        },
        "optimization": optimization,
        "failed_started_games_by_update": {
            str(row["update"]): int(row["rollout"]["failed_started_games"])
            for row in metrics
        },
    }


def diagnosis(evidence: dict[str, Any]) -> dict[str, Any]:
    initial = evidence["initial"]["panel"]
    update2 = evidence["update2"]["panel"]
    update4 = evidence["update4"]["panel"]
    deltas = {
        name: {
            "update2_minus_update0_pp": 100.0
            * (update2[name]["win_rate"] - initial[name]["win_rate"]),
            "update4_minus_update0_pp": 100.0
            * (update4[name]["win_rate"] - initial[name]["win_rate"]),
        }
        for name in initial
    }
    return {
        "fixed_panel_games_per_opponent": 256,
        "minimum_scores": {
            "update0": evidence["initial"]["selection_score"],
            "update2": evidence["update2"]["selection_score"],
            "update4": evidence["update4"]["selection_score"],
        },
        "deltas_by_opponent": deltas,
        "failure_tail_was_inactive": all(
            value == 0 for value in evidence["failed_started_games_by_update"].values()
        ),
        "observed_approx_kl_range": [
            min(value["approx_kl"] for value in evidence["optimization"].values()),
            max(value["approx_kl"] for value in evidence["optimization"].values()),
        ],
        "observed_clip_fraction_range": [
            min(value["clip_fraction"] for value in evidence["optimization"].values()),
            max(value["clip_fraction"] for value in evidence["optimization"].values()),
        ],
        "observed_bc_anchor_kl_update1_to_update4": [
            evidence["optimization"]["1"]["bc_anchor_kl"],
            evidence["optimization"]["4"]["bc_anchor_kl"],
        ],
        "interpretation": [
            "min selection is a noisy worst-of-five statistic, while standard PPO optimizes quota-weighted average episode return",
            "v1 gave own BC 50 percent of episodes and each other anchor 12.5 percent, so min selection was not the training objective",
            "source-Marnie and U472 weakened while the easy Lucario panel improved, indicating directional tradeoff rather than optimizer explosion",
            "BC anchor KL accumulated while its coefficient decayed from 0.020 to 0.016",
            "the 4-5 game minimum deficit versus update0 is not statistically decisive at only 256 games per opponent",
        ],
    }


def validate_file(label: str, path: Path, expected_sha256: str) -> dict[str, Any]:
    record = base.validate_file_binding(label, path, expected_sha256)
    return record


def build_preflight(phase: Phase) -> Preflight:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Launcher must run from repository root: {ROOT}")
    target = output_dir(phase)
    base.assert_target_absent(target)

    inputs = base.collect_file_records()
    inputs["v1_launcher"] = validate_file(
        "v1_launcher", BASE_LAUNCHER, BASE_LAUNCHER_SHA256
    )
    inputs["v1_summary"] = validate_file(
        "v1_summary", V1_SUMMARY, V1_SUMMARY_SHA256
    )
    inputs["v1_metrics"] = validate_file(
        "v1_metrics", V1_METRICS, V1_METRICS_SHA256
    )
    inputs["launcher"] = {
        "path": str(SELF),
        "resolved_path": str(SELF.resolve()),
        "bytes": SELF.stat().st_size,
        "sha256": raw_sha256_file(SELF),
        "protocol_pins_live_launcher_sha256": True,
    }

    evidence = load_v1_evidence()
    command = build_command(phase)
    phase_quotas = quotas(phase)
    manifest: dict[str, Any] = {
        "schema_version": "ptcg-gold-push-marnie-tail32-v2-lightanchor-v1",
        "status": "locked_before_training",
        "phase": phase.name,
        "output_dir": str(target),
        "expected_terminal_checkpoint": str(
            target / f"checkpoints/update-{phase.updates:04d}.pt"
        ),
        "inputs": inputs,
        "v1_evidence": evidence,
        "v1_diagnosis": diagnosis(evidence),
        "frozen_changes_from_v1": {
            "fresh_from_same_bc_learner": True,
            "updates": {"v1": base.PHASES[phase.name].updates, "v2": phase.updates},
            "actor_learning_rate": {"v1": 1.2e-5, "v2": 8e-6},
            "bc_kl": {"v1": [0.020, 0.016], "v2": [0.024, 0.024]},
            "fixed_quotas": phase_quotas,
            "evaluation_interval": phase.eval_interval,
        },
        "unchanged_contract": {
            "value_learning_rate": 2.5e-5,
            "ppo_epochs": 2,
            "minibatch_size": 512,
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "advantage_normalization": "per_opponent",
            "clip_ratio": 0.12,
            "value_coefficient": 0.25,
            "value_trunk_gradient_scale": 0.02,
            "entropy_coefficient": 0.0005,
            "max_grad_norm": 0.5,
            "policy_temperature": 0.8,
            "trainable_scope": "last_block_heads",
            "target_kl": 0.004,
            "actor_reduction": "episode_mean",
            "bc_replay_steps": 2,
            "bc_replay_lr_scale": 0.05,
            "bc_replay_loss": "ordered",
            "failed_attempt_as_loss": True,
            "failed_loss_tail_transitions": 32,
            "eval_all_permanent_opponents": True,
            "selection_aggregation": "min",
            "eval_games_per_opponent": phase.eval_games,
        },
        "promotion_gate": {
            "changed": False,
            "source": "v1 update0 fixed five-opponent panel minimum",
            "operator": ">",
            "threshold": UNCHANGED_PROMOTION_THRESHOLD,
            "invalid_games_required": 0,
        },
        "quota_audit": {
            "sum": sum(phase_quotas.values()),
            "games_per_update": phase.games_per_update,
            "all_even_for_exact_dual_seat": all(
                value % 2 == 0 for value in phase_quotas.values()
            ),
        },
        "command": command,
        "command_sha256": base.sha256_json(command),
        "scope": {
            "training": True,
            "local_only": True,
            "pending_16_strategy_results_read": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }
    return Preflight(
        phase=phase,
        target=target,
        command=command,
        manifest=manifest,
        manifest_sha256=base.sha256_json(manifest),
    )


def lock_bindings(preflight: Preflight) -> dict[str, tuple[Path, str]]:
    bindings = {
        "launcher": (SELF, preflight.manifest["inputs"]["launcher"]["sha256"]),
        "v1_launcher": (BASE_LAUNCHER, BASE_LAUNCHER_SHA256),
        "trainer": (base.TRAIN_PPO, base.TRAIN_PPO_SHA256),
        "python": (base.PYTHON.resolve(), base.PYTHON_SHA256),
        "v1_summary": (V1_SUMMARY, V1_SUMMARY_SHA256),
        "v1_metrics": (V1_METRICS, V1_METRICS_SHA256),
    }
    for label, path in base.FILE_PATHS.items():
        bindings[label] = (path, base.FILE_SHA256[label])
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
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"Locked input is not regular: {path}")
            observed = sha256_handle(handle)
            if observed != expected_sha256:
                raise RuntimeError(f"Execution-time input mismatch for {label}")
            locks.append(
                InputLock(
                    label,
                    path,
                    expected_sha256,
                    handle,
                    (info.st_dev, info.st_ino, info.st_size),
                )
            )
    except BaseException:
        release_input_locks(locks)
        raise
    return locks


def assert_input_locks_unchanged(locks: Sequence[InputLock]) -> None:
    for locked in locks:
        info = os.fstat(locked.handle.fileno())
        identity = (info.st_dev, info.st_ino, info.st_size)
        if identity != locked.identity or sha256_handle(locked.handle) != locked.expected_sha256:
            raise RuntimeError(f"Locked input changed during launch: {locked.path}")


def release_input_locks(locks: Sequence[InputLock]) -> None:
    for locked in reversed(locks):
        try:
            fcntl.flock(locked.handle.fileno(), fcntl.LOCK_UN)
        finally:
            locked.handle.close()


def execute(preflight: Preflight) -> int:
    locks = acquire_input_locks(preflight)
    try:
        assert_input_locks_unchanged(locks)
        base.assert_target_absent(preflight.target)
        preflight.target.mkdir(parents=False, exist_ok=False, mode=0o700)
        base.write_exclusive(
            preflight.target / "launcher_manifest.json",
            {
                "manifest_sha256": preflight.manifest_sha256,
                "manifest": preflight.manifest,
            },
        )
        assert_input_locks_unchanged(locks)
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
        assert_input_locks_unchanged(locks)
        base.write_exclusive(
            preflight.target / "launcher_result.json",
            {
                "schema_version": "ptcg-gold-push-marnie-tail32-v2-result-v1",
                "manifest_sha256": preflight.manifest_sha256,
                "return_code": int(completed.returncode),
                "inputs_unchanged_after_child": True,
            },
        )
        return int(completed.returncode)
    finally:
        release_input_locks(locks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASES), default="smoke")
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
    preflight = build_preflight(PHASES[args.phase])
    print(
        json.dumps(
            {
                "manifest_sha256": preflight.manifest_sha256,
                "manifest": preflight.manifest,
            },
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
