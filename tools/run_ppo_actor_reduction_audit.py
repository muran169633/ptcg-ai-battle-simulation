#!/usr/bin/env python3
"""Preregister and run one zero-update Gold-branch actor-reduction audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import run_ppo_gold_ab as gold_runner


SCHEMA_VERSION = "ptcg-ppo-actor-reduction-audit-plan-v1"
PREREGISTRATION_SCHEMA = "ptcg-ppo-actor-reduction-audit-preregistration-v1"
SUMMARY_SCHEMA = "ptcg-ppo-actor-reduction-audit-run-v1"
AUDITABLE_ACTOR_REDUCTIONS = ("quota_group_mean", "episode_mean")


def command_value(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
        return command[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Training command lacks {flag}") from error


def replace_command_value(
    command: list[str],
    flag: str,
    value: str,
) -> None:
    try:
        index = command.index(flag)
        command[index + 1] = value
    except (ValueError, IndexError) as error:
        raise ValueError(f"Training command lacks {flag}") from error


def load_training_preregistration(path: Path) -> dict[str, Any]:
    registration = gold_runner.read_json_object(path)
    if (
        registration.get("schema_version")
        != gold_runner.PREREGISTRATION_SCHEMA
    ):
        raise ValueError(f"{path}: wrong training preregistration schema")
    plan = registration.get("plan")
    if not isinstance(plan, dict):
        raise ValueError(f"{path}: missing training plan")
    expected_sha256 = gold_runner.canonical_json_sha256(plan)
    if registration.get("plan_sha256") != expected_sha256:
        raise ValueError(f"{path}: training plan hash mismatch")
    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{path}: training plan lacks input hashes")
    declared_train_script = Path(
        str(inputs.get("train_script", ""))
    ).resolve()
    if declared_train_script != gold_runner.TRAIN_SCRIPT.resolve():
        raise ValueError(f"{path}: unexpected train script")
    declared_train_sha256 = str(inputs.get("train_script_sha256", ""))
    current_train_sha256 = gold_runner.file_sha256(
        gold_runner.TRAIN_SCRIPT
    )
    if declared_train_sha256 != current_train_sha256:
        raise ValueError(
            f"{path}: train script changed after preregistration"
        )
    candidate_reduction = plan.get("protocol", {}).get("actor_reduction")
    if candidate_reduction not in AUDITABLE_ACTOR_REDUCTIONS:
        raise ValueError(
            "Actor audit requires a quota_group_mean or episode_mean "
            "training preregistration"
        )
    return registration


def build_audit_plan(
    training_preregistration_path: Path,
    training_registration: dict[str, Any],
    *,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    training_plan = training_registration["plan"]
    branch = training_plan.get("branches", {}).get("B_gold_league")
    if not isinstance(branch, dict):
        raise ValueError("Training plan lacks B_gold_league")
    matching_runs = [
        run
        for run in branch.get("runs", [])
        if int(run.get("seed", -1)) == seed
    ]
    if len(matching_runs) != 1:
        raise ValueError(
            f"Training plan must contain exactly one Gold run for seed {seed}"
        )
    base_command = [
        str(value) for value in matching_runs[0].get("command", [])
    ]
    candidate_reduction = str(
        training_plan.get("protocol", {}).get("actor_reduction", "")
    )
    if candidate_reduction not in AUDITABLE_ACTOR_REDUCTIONS:
        raise ValueError("Training plan lacks an auditable actor reduction")
    if (
        len(base_command) < 2
        or Path(base_command[0]).resolve() != Path(sys.executable).resolve()
        or Path(base_command[1]).resolve() != gold_runner.TRAIN_SCRIPT.resolve()
    ):
        raise ValueError("Refusing an unexpected training child command")
    if (
        command_value(base_command, "--actor-reduction")
        != candidate_reduction
    ):
        raise ValueError(
            "Gold command does not use the preregistered actor reduction"
        )
    if "--skip-initial-eval" not in base_command:
        raise ValueError("Gold command lacks --skip-initial-eval")
    if "--actor-reduction-audit-only" in base_command:
        raise ValueError("Training command unexpectedly already is audit-only")

    audit_command = list(base_command)
    replace_command_value(
        audit_command,
        "--output-dir",
        str(output_dir.resolve()),
    )
    audit_command.append("--actor-reduction-audit-only")
    return {
        "schema_version": SCHEMA_VERSION,
        "training_preregistration": {
            "path": str(training_preregistration_path.resolve()),
            "sha256": gold_runner.file_sha256(
                training_preregistration_path
            ),
            "plan_sha256": training_registration["plan_sha256"],
        },
        "branch": "B_gold_league",
        "seed": seed,
        "output_dir": str(output_dir.resolve()),
        "command": audit_command,
        "command_sha256": gold_runner.canonical_json_sha256(audit_command),
        "protocol": {
            "rollouts": 1,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "standard_reduction": "transition_mean",
            "candidate_reduction": candidate_reduction,
            "gradient_cosine_threshold": 0.995,
            "relative_row_weight_deviation_threshold": 0.10,
            "training_authorized_if": (
                "gradient_cosine < 0.995 OR maximum relative row-weight "
                "deviation >= 0.10"
            ),
        },
        "safety": {
            "local_rollout_only": True,
            "allowed_child_program": str(gold_runner.TRAIN_SCRIPT),
            "network_calls": False,
            "uploads": False,
            "submission": False,
            "packaging": False,
        },
    }


def preregistration_envelope(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PREREGISTRATION_SCHEMA,
        "created_at": gold_runner.utc_now(),
        "plan_sha256": gold_runner.canonical_json_sha256(plan),
        "plan": plan,
        "status": "preregistered_not_started",
    }


def validate_existing_preregistration(
    path: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    registration = gold_runner.read_json_object(path)
    if registration.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise ValueError(f"{path}: wrong actor-audit preregistration schema")
    if registration.get("plan_sha256") != gold_runner.canonical_json_sha256(
        plan
    ):
        raise ValueError(f"{path}: actor-audit plan hash mismatch")
    if registration.get("plan") != plan:
        raise ValueError(f"{path}: actor-audit plan payload mismatch")
    return registration


def validate_audit_result(
    output_dir: Path,
    expected_candidate: str | None = None,
) -> dict[str, Any]:
    audit_path = output_dir / "actor_reduction_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("Audit child did not produce actor_reduction_audit.json")
    audit = gold_runner.read_json_object(audit_path)
    if (
        audit.get("schema_version")
        != "ptcg-ppo-actor-reduction-audit-v1"
    ):
        raise RuntimeError("Actor audit has an unexpected schema")
    if audit.get("status") != "completed_no_update":
        raise RuntimeError("Actor audit did not complete in zero-update mode")
    candidate_reduction = audit.get("candidate_reduction")
    if candidate_reduction not in AUDITABLE_ACTOR_REDUCTIONS:
        raise RuntimeError("Actor audit has an unexpected candidate reduction")
    if (
        expected_candidate is not None
        and candidate_reduction != expected_candidate
    ):
        raise RuntimeError(
            "Actor audit candidate does not match the preregistration"
        )
    if int(audit.get("optimizer_steps", -1)) != 0:
        raise RuntimeError("Actor audit unexpectedly performed optimizer steps")
    if (
        audit.get("model_state_sha256_before")
        != audit.get("model_state_sha256_after")
    ):
        raise RuntimeError("Actor audit changed model weights")
    checkpoints = sorted(
        str(path)
        for path in output_dir.rglob("*.pt")
        if path.is_file()
    )
    if checkpoints:
        raise RuntimeError(
            "Actor audit unexpectedly wrote checkpoints: "
            + ", ".join(checkpoints)
        )
    metrics_path = output_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise RuntimeError("Actor audit did not write metrics.jsonl")
    metrics_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(metrics_rows) != 1
        or metrics_rows[0].get("optimization") is not None
        or metrics_rows[0].get("bc_replay") is not None
        or not isinstance(
            metrics_rows[0].get("actor_reduction_audit"),
            dict,
        )
    ):
        raise RuntimeError(
            "Actor audit metrics are not exactly one zero-update rollout"
        )
    gate = audit.get("pre_registered_gate")
    if not isinstance(gate, dict) or not isinstance(
        gate.get("training_authorized"),
        bool,
    ):
        raise RuntimeError("Actor audit lacks the preregistered gate decision")
    row_weight_audit = audit.get("row_weight_audit")
    if not isinstance(row_weight_audit, dict):
        raise RuntimeError("Actor audit lacks row-weight invariants")
    if candidate_reduction == "episode_mean" and (
        row_weight_audit.get("mode") != "episode_mean"
        or row_weight_audit.get("all_games_equal_weight") is not True
        or row_weight_audit.get("all_transition_rows_accounted_for")
        is not True
        or int(row_weight_audit.get("missing_game_uid_rows", -1)) != 0
        or abs(float(row_weight_audit.get("mean_row_multiplier", 0.0)) - 1.0)
        > 1e-6
    ):
        raise RuntimeError(
            "Episode-mean actor audit failed equal-game coverage invariants"
        )
    return audit


def execute(
    plan: dict[str, Any],
    registration: dict[str, Any],
    preregistration_path: Path,
) -> dict[str, Any]:
    output_dir = Path(plan["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    command = [str(value) for value in plan["command"]]
    log_path = output_dir.with_name(f"{output_dir.name}.log")
    if log_path.exists():
        raise FileExistsError(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=gold_runner.REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Actor audit child failed with code {completed.returncode}; "
            f"see {log_path}"
        )
    audit = validate_audit_result(
        output_dir,
        expected_candidate=str(
            plan["protocol"]["candidate_reduction"]
        ),
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "completed_at": gold_runner.utc_now(),
        "status": "completed",
        "preregistration": {
            "path": str(preregistration_path.resolve()),
            "sha256": gold_runner.file_sha256(preregistration_path),
            "plan_sha256": registration["plan_sha256"],
        },
        "output_dir": str(output_dir),
        "log": str(log_path),
        "audit": {
            "path": str(output_dir / "actor_reduction_audit.json"),
            "sha256": gold_runner.file_sha256(
                output_dir / "actor_reduction_audit.json"
            ),
            "gradient": audit["gradient"],
            "row_weight_audit": audit["row_weight_audit"],
            "pre_registered_gate": audit["pre_registered_gate"],
            "optimizer_steps": audit["optimizer_steps"],
            "model_state_unchanged": (
                audit["model_state_sha256_before"]
                == audit["model_state_sha256_after"]
            ),
        },
        "checkpoint_writes": 0,
        "safety": plan["safety"],
    }
    gold_runner.atomic_write_json(
        output_dir / "audit_run_summary.json",
        summary,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preregister and optionally run one zero-update Gold-branch "
            "actor-reduction audit."
        )
    )
    parser.add_argument("--training-preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preregistration", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preregister-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    training_path = args.training_preregistration.expanduser().resolve()
    if not training_path.is_file():
        raise FileNotFoundError(training_path)
    output_dir = args.output_dir.expanduser().resolve()
    preregistration_path = (
        args.preregistration.expanduser().resolve()
        if args.preregistration is not None
        else output_dir.with_name(f"{output_dir.name}.preregistration.json")
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if preregistration_path == output_dir:
        raise ValueError("Preregistration path cannot equal output directory")

    training_registration = load_training_preregistration(training_path)
    plan = build_audit_plan(
        training_path,
        training_registration,
        output_dir=output_dir,
        seed=args.seed,
    )
    if not (args.preregister_only or args.execute):
        envelope = preregistration_envelope(plan)
        print(json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True))
        return envelope
    if preregistration_path.exists():
        envelope = validate_existing_preregistration(
            preregistration_path,
            plan,
        )
    else:
        envelope = preregistration_envelope(plan)
        gold_runner.atomic_write_json(preregistration_path, envelope)
    if args.preregister_only:
        print(str(preregistration_path))
        return envelope
    summary = execute(plan, envelope, preregistration_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
