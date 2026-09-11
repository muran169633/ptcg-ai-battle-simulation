#!/usr/bin/env python3
"""Audit a completed local preregistered PPO run against frozen gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UPDATES = list(range(457, 465))


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"metrics line {line_number} is not an object")
            rows.append(value)
    return rows


def load_mixed_log_json_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def finite_json(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    return False


def iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


def all_float_tensors_finite(path: Path) -> tuple[bool, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    count = 0
    for tensor in iter_tensors(payload):
        if tensor.is_floating_point() or tensor.is_complex():
            count += 1
            if not bool(torch.isfinite(tensor).all()):
                return False, count
    return True, count


def all_zero_nested(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_zero_nested(item) for item in value.values())
    return value == 0


def check(condition: bool, label: str, failures: list[str]) -> bool:
    result = bool(condition)
    if not result:
        failures.append(label)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = (REPO_ROOT / args.protocol).resolve()
    output_path = (REPO_ROOT / args.output).resolve()
    for path, label in ((protocol_path, "protocol"), (output_path, "output")):
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as error:
            raise ValueError(f"{label} must stay inside repository") from error
    if sha256_file(protocol_path) != args.expected_protocol_sha256:
        raise ValueError("protocol SHA-256 mismatch")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_path}")
    protocol = load_json(protocol_path)
    output_dir = (REPO_ROOT / protocol["output_dir"]).resolve()
    metrics_path = output_dir / "metrics.jsonl"
    config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = (REPO_ROOT / protocol["terminal_checkpoint"]).resolve()
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    marker_path = (REPO_ROOT / protocol["attempt_start_marker"]).resolve()
    log_path = (REPO_ROOT / protocol["log"]).resolve()
    failures: list[str] = []

    for path, label in (
        (metrics_path, "metrics"),
        (config_path, "run_config"),
        (summary_path, "summary"),
        (checkpoint_path, "terminal_checkpoint"),
        (best_path, "best"),
        (last_path, "last"),
        (marker_path, "attempt_start_marker"),
        (log_path, "log"),
    ):
        check(path.is_file() and not path.is_symlink(), f"{label}_regular_file", failures)

    marker = load_json(marker_path)
    log_lines = load_mixed_log_json_records(log_path)
    config = load_json(config_path)
    summary = load_json(summary_path)
    metrics = load_jsonl(metrics_path)
    cfg = config["config"]
    expected_checkpoint_sha = protocol["source_checkpoint"]["sha256"]
    expected_quota = cfg["opponent_base_quotas"]
    minibatch_size = int(protocol["design"]["minibatch_size"])
    entropy_coefficient = float(
        protocol.get("frozen_flag_values", {}).get(
            "--entropy-coefficient", 0.001
        )
    )
    expected_updates = list(protocol["design"]["updates"])
    protocol_v2 = protocol.get("schema_version") == "ptcg-local-preregistered-ppo-v2"
    expected_command_sha = (
        protocol["command_sha256"]
        if protocol_v2
        else protocol["derived_command_sha256"]
    )
    marker_command_sha = (
        marker.get("command_sha256")
        if protocol_v2
        else marker.get("derived_command_sha256")
    )
    header_event = (
        "local_preregistered_ppo_v2_pre_exec_checks_passed"
        if protocol_v2
        else "local_preregistered_ppo_pre_exec_checks_passed"
    )
    terminal_event = (
        "local_preregistered_ppo_v2_child_terminal"
        if protocol_v2
        else "local_preregistered_ppo_child_terminal"
    )

    transport_gates = {
        "marker_protocol_sha256": check(
            marker.get("protocol_sha256") == args.expected_protocol_sha256,
            "marker_protocol_sha256",
            failures,
        ),
        "marker_command_sha256": check(
            marker_command_sha == expected_command_sha,
            "marker_command_sha256",
            failures,
        ),
        "terminal_log_header": check(
            bool(log_lines)
            and log_lines[0].get("event")
            == header_event
            and log_lines[0].get("protocol_sha256")
            == args.expected_protocol_sha256,
            "terminal_log_header",
            failures,
        ),
        "terminal_return_code_zero": check(
            bool(log_lines)
            and log_lines[-1]
            == {
                "event": terminal_event,
                "return_code": 0,
                "seed": protocol["seed"],
            },
            "terminal_return_code_zero",
            failures,
        ),
    }

    config_gates = {
        "seed": check(cfg["seed"] == protocol["seed"], "config_seed", failures),
        "resume_checkpoint": check(
            config["learner_initialization"]["checkpoint_sha256"]
            == expected_checkpoint_sha
            and config["resume_metadata"]["checkpoint_update"] == 456
            and cfg["resume_learner_weights"] == "resume",
            "resume_checkpoint_binding",
            failures,
        ),
        "optimizer_reset": check(
            config["learner_initialization"]["optimizer_state_source"] == "fresh"
            and config["learner_initialization"]["replay_optimizer_state_source"]
            == "fresh"
            and config["learner_initialization"]["optimizer_reset_reason"]
            == "explicit_reset_on_resume"
            and cfg["reset_opponent_quota_on_resume"] is True,
            "optimizer_reset",
            failures,
        ),
        "frozen_hyperparameters": check(
            cfg["updates"] == 464
            and cfg["games_per_update"] == 64
            and cfg["ppo_epochs"] == 2
            and cfg["minibatch_size"] == minibatch_size
            and cfg["learning_rate"] == 0.000036
            and cfg["value_learning_rate"] == 0.0000075
            and cfg["value_coefficient"] == 0.25
            and cfg["entropy_coefficient"] == entropy_coefficient
            and cfg["bc_kl_start"] == cfg["bc_kl_end"] == 0.012
            and cfg["target_kl"] == 0.006
            and cfg["actor_reduction"] == "episode_mean"
            and cfg["bc_replay_steps"] == 2
            and cfg["bc_replay_context34_rows_per_batch"] == 4
            and cfg["opponent_quota_mode"] == "fixed"
            and cfg["opponent_quota_seat_balance"] is True,
            "frozen_hyperparameters",
            failures,
        ),
    }

    per_update: list[dict[str, Any]] = []
    check([row.get("update") for row in metrics] == expected_updates, "updates_exact", failures)
    max_approx_kl = max(row["optimization"]["approx_kl"] for row in metrics)
    max_clip_fraction = max(
        row["optimization"]["clip_fraction"] for row in metrics
    )
    max_bc_anchor_kl = max(
        row["optimization"]["bc_anchor_kl"] for row in metrics
    )
    max_objective_residual = 0.0
    for row in metrics:
        update = row["update"]
        rollout = row["rollout"]
        optimization = row["optimization"]
        replay = row["bc_replay"]
        actor_reduction = optimization["actor_reduction"]
        expected_steps = 2 * math.ceil(optimization["transitions"] / minibatch_size)
        objective = (
            optimization["policy_loss"]
            + 0.25 * optimization["value_loss"]
            - entropy_coefficient * optimization["entropy"]
            + 0.012 * optimization["bc_anchor_kl"]
        )
        residual = abs(optimization["loss"] - objective)
        max_objective_residual = max(max_objective_residual, residual)
        quota = rollout["opponent_quota"]
        quota_ok = (
            quota["planned_quotas"] == expected_quota
            and quota["actual_quotas"] == expected_quota
            and quota["planned_seat_quotas"] == quota["actual_seat_quotas"]
            and quota["seat_balance_verified"] is True
            and quota["actual_max_seat_gap"] == 0
            and all_zero_nested(quota["invalid_replacements"])
            and all_zero_nested(quota["invalid_replacements_by_seat"])
            and all_zero_nested(quota["start_errors_by_seat"])
        )
        row_pass = all(
            (
                rollout["valid_games"] == 64,
                rollout["transitions_total"]
                == rollout["transitions_kept"]
                == optimization["transitions"],
                optimization["rows"] == 2 * optimization["transitions"],
                optimization["optimizer_steps"] == expected_steps,
                optimization["epochs_completed"] == 2,
                optimization["early_stop"] is False,
                optimization["actor_learning_rate"] == 0.000036,
                optimization["value_learning_rate"] == 0.0000075,
                optimization["bc_kl_coefficient"] == 0.012,
                optimization["approx_kl"] < 0.006,
                optimization["clip_fraction"] < 0.05,
                optimization["bc_anchor_kl"] <= 0.015,
                actor_reduction["mode"] == "episode_mean",
                actor_reduction["episode_count"] == 64,
                actor_reduction["missing_game_uid_rows"] == 0,
                actor_reduction["all_transition_rows_accounted_for"] is True,
                actor_reduction["all_games_equal_weight"] is True,
                actor_reduction["max_episode_weight_error"] <= 1e-12,
                replay["rows"] == 512,
                replay["steps"] == 2,
                replay["learning_rate"] == 0.0000018000000000000001,
                replay["context_34_rows"] == 8,
                quota_ok,
                residual <= 1e-6,
                finite_json(row),
            )
        )
        check(row_pass, f"update_{update}", failures)
        per_update.append(
            {
                "update": update,
                "pass": row_pass,
                "valid_games": rollout["valid_games"],
                "transitions": optimization["transitions"],
                "optimizer_steps": optimization["optimizer_steps"],
                "approx_kl": optimization["approx_kl"],
                "clip_fraction": optimization["clip_fraction"],
                "bc_anchor_kl": optimization["bc_anchor_kl"],
                "objective_identity_residual": residual,
            }
        )

    checkpoint_files = sorted(path.name for path in checkpoint_dir.iterdir())
    checkpoint_sha = sha256_file(checkpoint_path)
    best_sha = sha256_file(best_path)
    last_sha = sha256_file(last_path)
    checkpoint_finite, float_tensor_count = all_float_tensors_finite(
        checkpoint_path
    )
    checkpoint_gates = {
        "terminal_only": check(
            checkpoint_files == ["update-0464.pt"],
            "checkpoint_terminal_only",
            failures,
        ),
        "best_last_terminal_equal": check(
            checkpoint_sha == best_sha == last_sha,
            "best_last_terminal_equal",
            failures,
        ),
        "float_tensors_finite": check(
            checkpoint_finite, "checkpoint_float_tensors_finite", failures
        ),
        "summary_terminal_update": check(
            summary.get("updates_completed") == 464,
            "summary_terminal_update",
            failures,
        ),
    }

    passed = not failures
    result = {
        "schema_version": "ptcg-local-ppo-training-integrity-v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "passed" if passed else "failed",
        "pass": passed,
        "protocol": {
            "path": str(protocol_path.relative_to(REPO_ROOT)),
            "sha256": args.expected_protocol_sha256,
        },
        "auditor": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "transport_gates": transport_gates,
        "config_gates": config_gates,
        "per_update": per_update,
        "numeric_summary": {
            "updates": [row["update"] for row in metrics],
            "rollout_games": sum(row["rollout"]["valid_games"] for row in metrics),
            "maximum_approx_kl": max_approx_kl,
            "maximum_clip_fraction": max_clip_fraction,
            "maximum_bc_anchor_kl": max_bc_anchor_kl,
            "maximum_objective_identity_residual": max_objective_residual,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "sha256": checkpoint_sha,
            "float_tensor_count": float_tensor_count,
            "gates": checkpoint_gates,
        },
        "failures": failures,
        "decision": {
            "behavior_execution_authorized": passed,
            "gold19_authorized": False,
            "package_upload_or_submission_authorized": False,
            "incumbent_retained_until_downstream_gates_pass": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, canonical_json_bytes(result))
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"pass": passed, "output": str(output_path), "failures": failures}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
