#!/usr/bin/env python3
"""Independently audit the completed G8-reverse PPO4 attempt and close it."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_ppo_bc_repair as audit  # noqa: E402
import train_ppo as ppo  # noqa: E402


BRANCH = "ppo_u464_g8rev_generalbc_then_ppo4_then_specialbc_design202608120"
RUN_ROOT = REPO_ROOT / f"artifacts/{BRANCH}/ppo_stage/B_gold_league/seed-202607336"
FILES = {
    "protocol": (
        REPO_ROOT / f"artifacts/{BRANCH}.ppo_stage_execution_preregistration.json",
        "f67329bec72505fc7e4e4d95e45ed341b47999c93cf99fdf33e29daf5580ff99",
    ),
    "attempt": (
        REPO_ROOT / ".ptcg-u464-g8rev-ppo-u468-attempt-202607336-202608120.json",
        "622e824988fa477ac39a210ef6e5a365d15d2742507f9dc214429374b7bc7524",
    ),
    "log": (
        REPO_ROOT / f"artifacts/{BRANCH}.ppo_stage.log",
        "07b6d6f4a882423b6c14f111970d9a9f8a1143645c6c0ed04f32c55e583e8849",
    ),
    "run_config": (
        RUN_ROOT / "run_config.json",
        "2319c505c9debe5b818e5caa75dd767792b64fd4210b801896decce41080d89a",
    ),
    "metrics": (
        RUN_ROOT / "metrics.jsonl",
        "036ac1cff8a51a10ddb23d515f81420c2a630a4a1f0bacacd979df31435b510f",
    ),
    "best": (
        RUN_ROOT / "best.pt",
        "dec05d4ff5793362bc04881e5fd927eaf874e53aef3a8cbae4b4f68e72619fba",
    ),
    "last": (
        RUN_ROOT / "last.pt",
        "dec05d4ff5793362bc04881e5fd927eaf874e53aef3a8cbae4b4f68e72619fba",
    ),
    "terminal": (
        RUN_ROOT / "checkpoints/update-0468.pt",
        "dec05d4ff5793362bc04881e5fd927eaf874e53aef3a8cbae4b4f68e72619fba",
    ),
}
GENERAL_BC = REPO_ROOT / (
    "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_"
    "seed20260922_20260731/best.pt"
)
OUTPUT = REPO_ROOT / f"artifacts/{BRANCH}.ppo_stage_training_integrity_decision.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def all_zero(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_zero(item) for item in value.values())
    if isinstance(value, list):
        return all(all_zero(item) for item in value)
    return int(value) == 0


def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(f"refusing existing decision: {OUTPUT}")
    bindings: dict[str, Any] = {}
    for label, (path, expected) in FILES.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} is absent, irregular, or symlinked")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"{label} hash mismatch")
        bindings[label] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": observed,
        }

    run_config = json.loads(FILES["run_config"][0].read_text(encoding="utf-8"))
    config = run_config["config"]
    expected_config = {
        "updates": 468,
        "games_per_update": 64,
        "ppo_epochs": 2,
        "minibatch_size": 384,
        "learning_rate": 3.6e-5,
        "value_learning_rate": 7.5e-6,
        "bc_kl_start": 0.012,
        "bc_kl_end": 0.012,
        "target_kl": 0.006,
        "actor_reduction": "episode_mean",
        "trainable_scope": "last_block_heads",
        "bc_replay_steps": 2,
        "bc_replay_lr_scale": 0.05,
        "bc_replay_loss": "ordered",
        "bc_replay_context34_rows_per_batch": 4,
        "seed": 202607336,
        "device": "cuda",
    }
    config_checks = {
        key: config.get(key) == expected for key, expected in expected_config.items()
    }
    if not all(config_checks.values()):
        raise ValueError("run configuration drifted")

    metrics = [
        json.loads(line)
        for line in FILES["metrics"][0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [row.get("update") for row in metrics] != [465, 466, 467, 468]:
        raise ValueError("metrics update sequence drifted")
    rows: list[dict[str, Any]] = []
    for row in metrics:
        rollout = row["rollout"]
        optimization = row["optimization"]
        replay = row["bc_replay"]
        quota = rollout["opponent_quota"]
        values = {
            "update": int(row["update"]),
            "valid_games": int(rollout["valid_games"]),
            "result_0": int(rollout["result_0"]),
            "result_1": int(rollout["result_1"]),
            "approx_kl": float(optimization["approx_kl"]),
            "clip_fraction": float(optimization["clip_fraction"]),
            "bc_anchor_kl": float(optimization["bc_anchor_kl"]),
            "early_stop": bool(optimization["early_stop"]),
            "epochs_completed": int(optimization["epochs_completed"]),
            "replay_rows": int(replay["rows"]),
            "replay_steps": int(replay["steps"]),
            "quota_exact": quota["planned_quotas"] == quota["actual_quotas"],
            "seat_balance_verified": quota["seat_balance_verified"] is True,
            "actual_max_seat_gap": int(quota["actual_max_seat_gap"]),
            "invalid_replacements_zero": all_zero(quota["invalid_replacements"]),
            "start_errors_zero": all_zero(quota["start_errors_by_seat"]),
        }
        if not all(
            math.isfinite(values[key])
            for key in ("approx_kl", "clip_fraction", "bc_anchor_kl")
        ):
            raise FloatingPointError("non-finite metric")
        rows.append(values)

    terminal = torch.load(FILES["terminal"][0], map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(GENERAL_BC, map_location="cpu", weights_only=False)
    model = ppo.instantiate_model_from_bc(bc_checkpoint, torch.device("cpu"))
    model.load_state_dict(terminal["model_state_dict"])
    quota_state = terminal["opponent_quota_state"]
    quota_games = sum(
        int(result.get(key, 0))
        for result in quota_state["observed"].values()
        for key in ("wins", "losses", "draws")
    )
    terminal_audit = {
        "update": int(terminal["update"]),
        "model_state_sha256": ppo.model_state_sha256(model),
        "ppo_optimizer_sha256": audit.nested_sha256(terminal["optimizer_state_dict"]),
        "ppo_optimizer_state_count": len(terminal["optimizer_state_dict"]["state"]),
        "ppo_optimizer_steps": sorted(set(audit.optimizer_steps(terminal["optimizer_state_dict"]))),
        "replay_optimizer_sha256": audit.nested_sha256(terminal["bc_replay_optimizer_state_dict"]),
        "replay_optimizer_state_count": len(terminal["bc_replay_optimizer_state_dict"]["state"]),
        "replay_optimizer_steps": sorted(set(audit.optimizer_steps(terminal["bc_replay_optimizer_state_dict"]))),
        "quota_state_sha256": audit.nested_sha256(quota_state),
        "quota_observed_games": quota_games,
        "quota_last_refresh_update": int(quota_state["last_refresh_update"]),
        "all_model_and_optimizer_values_finite": (
            audit.finite_nested(terminal["model_state_dict"])
            and audit.finite_nested(terminal["optimizer_state_dict"])
            and audit.finite_nested(terminal["bc_replay_optimizer_state_dict"])
        ),
    }
    terminal_expected = {
        "update": 468,
        "model_state_sha256": "e86eaabd6a20fc07f7338bfaf0d11bd1e0776bd6b66cb8d78e2a51075646d1d8",
        "ppo_optimizer_sha256": "b2fff44a5fd7757b7cf78da2bb805ccb5675c80e22fc4eb492ff5089895fa6d3",
        "ppo_optimizer_state_count": 28,
        "ppo_optimizer_steps": [408],
        "replay_optimizer_sha256": "1072d2bc5ec8b1f8caeff452b3d6ad3e4e3f513e63e2365159adbc063baff26c",
        "replay_optimizer_state_count": 24,
        "replay_optimizer_steps": [32],
        "quota_state_sha256": "0024e09b4aee275d01037487cf4c2084a60c91a2c45d1f3e2f540adb508445d5",
        "quota_observed_games": 768,
        "quota_last_refresh_update": 466,
        "all_model_and_optimizer_values_finite": True,
    }
    if terminal_audit != terminal_expected:
        raise ValueError("terminal checkpoint integrity mismatch")

    max_anchor = max(row["bc_anchor_kl"] for row in rows)
    max_anchor_update = max(rows, key=lambda row: row["bc_anchor_kl"])["update"]
    gates = {
        "four_updates_exact": len(rows) == 4,
        "all_64_valid_and_no_missing_result": all(
            row["valid_games"] == 64 and row["result_0"] + row["result_1"] == 64
            for row in rows
        ),
        "quota_and_seat_exact_all_updates": all(
            row["quota_exact"]
            and row["seat_balance_verified"]
            and row["actual_max_seat_gap"] == 0
            and row["invalid_replacements_zero"]
            and row["start_errors_zero"]
            for row in rows
        ),
        "approx_kl_below_0p003_all_updates": all(row["approx_kl"] < 0.003 for row in rows),
        "clip_fraction_below_0p05_all_updates": all(row["clip_fraction"] < 0.05 for row in rows),
        "two_epochs_no_early_stop_all_updates": all(
            row["epochs_completed"] == 2 and not row["early_stop"] for row in rows
        ),
        "replay_2x256_all_updates": all(
            row["replay_steps"] == 2 and row["replay_rows"] == 512 for row in rows
        ),
        "bc_anchor_kl_at_most_0p015_all_updates": max_anchor <= 0.015,
        "terminal_checkpoint_integrity": terminal_audit == terminal_expected,
        "best_last_terminal_byte_identical": len(
            {FILES[key][1] for key in ("best", "last", "terminal")}
        ) == 1,
        "isolated_child_returned_zero": (
            '"child_isolated_mode":true' in FILES["log"][0].read_text(encoding="utf-8")
            and '"return_code":0' in FILES["log"][0].read_text(encoding="utf-8")
        ),
    }
    failed = [key for key, passed in gates.items() if not passed]
    if failed != ["bc_anchor_kl_at_most_0p015_all_updates"]:
        raise ValueError(f"unexpected integrity failure set: {failed}")

    decision = {
        "schema_version": "ptcg-u464-g8rev-ppo4-training-integrity-decision-v1",
        "status": "NO_GO_PPO_ANCHOR_MAX_EXCEEDED",
        "bindings": bindings,
        "config_checks": config_checks,
        "per_update": rows,
        "terminal": terminal_audit,
        "gates": gates,
        "failed_gates": failed,
        "failure": {
            "metric": "maximum_bc_anchor_kl_over_all_updates",
            "threshold": 0.015,
            "actual": max_anchor,
            "first_and_max_failure_update": max_anchor_update,
            "terminal_value": rows[-1]["bc_anchor_kl"],
            "terminal_recovery_does_not_erase_intermediate_failure": True,
        },
        "decision": {
            "lineage_closed": True,
            "intermediate_checkpoint_selection_forbidden": True,
            "special_bc_authorized": False,
            "fulltrain_behavior_authorized": False,
            "broad_behavior_authorized": False,
            "gold_authorized": False,
            "package_upload_or_submission_authorized": False,
            "incumbent_U456_retained": True,
        },
    }
    raw = (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"path": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest(), "status": decision["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
