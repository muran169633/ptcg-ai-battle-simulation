#!/usr/bin/env python3
"""Build the hash-bound 4x6 specialist preregistration for the FLG sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_u468_beta1157_flg_fresh_adamw_head10_specialist_matrix as matrix  # noqa: E402


BRANCH = "ppo_u468_beta1157_flg_actorhead_freshlr225e7_p1p2p4p8_design202608100"
TRAINING_ROOT = ROOT / "artifacts" / BRANCH
TRAINING_MANIFEST = TRAINING_ROOT / "training_manifest.json"
TRAINING_SEAL = TRAINING_ROOT / "COMPLETED.json"
OUTPUT_ROOT = ROOT / "artifacts" / f"{BRANCH}.specialist_behavior"
MARKER = ROOT / f".ptcg-{BRANCH}-specialist-attempt.json"
LAUNCHER = ROOT / "tools/run_u468_beta1157_flg_fresh_adamw_head10_specialist_matrix.py"
EVALUATOR = ROOT / "tools/evaluate_policy_bc.py"
DATA = {
    "pokemonfan": ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
    "flg": ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
    "core5": ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the repository root")
    output = args.output.resolve()
    if output.parent != ROOT / "artifacts" or output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink() or MARKER.exists() or MARKER.is_symlink():
        raise FileExistsError("specialist output or marker already exists")

    training_manifest = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    endpoints_raw = training_manifest.get("endpoints")
    if not isinstance(endpoints_raw, list) or [item.get("step") for item in endpoints_raw] != [1, 2, 4, 8]:
        raise ValueError("training manifest endpoint order mismatch")
    names = ("p001", "p002", "p004", "p008")
    endpoints = []
    endpoint_paths: dict[str, Path] = {}
    for index, (name, step, record) in enumerate(
        zip(names, (1, 2, 4, 8), endpoints_raw, strict=True)
    ):
        path = ROOT / record["path"]
        file_hash = sha256_file(path)
        if file_hash != record["sha256"]:
            raise ValueError(f"endpoint P{step} file hash mismatch")
        endpoints.append(
            {
                "name": name,
                "step": step,
                "path": record["path"],
                "sha256": record["sha256"],
                "model_state_sha256": record["model_state_sha256"],
                "manifest_record_json_pointer": f"/endpoints/{index}",
            }
        )
        endpoint_paths[name] = path

    ordered = []
    commands = []
    order = 0
    for endpoint, step in zip(names, (1, 2, 4, 8), strict=True):
        for panel, data_name, team_name in matrix.EXPECTED_PANELS:
            order += 1
            result_path = OUTPUT_ROOT / endpoint / f"{panel}.json"
            command = matrix.expected_command(
                evaluator=EVALUATOR,
                checkpoint=endpoint_paths[endpoint],
                data=DATA[data_name],
                team_name=team_name,
                output=result_path,
            )
            commands.append(command)
            ordered.append(
                {
                    "order": order,
                    "endpoint": endpoint,
                    "step": step,
                    "panel": panel,
                    "team_name": team_name,
                    "output": str(result_path.relative_to(ROOT)),
                    "command": command,
                    "command_sha256": matrix.sha256_bytes(
                        matrix.canonical_json_bytes(command)
                    ),
                    "attempts_authorized": 1,
                    "expected_success_exit_code": 0,
                }
            )

    protocol = {
        "schema_version": matrix.PREREGISTRATION_SCHEMA,
        "status": "locked_before_single_zero_write_preflight",
        "required_cwd": str(ROOT),
        "launcher": binding(LAUNCHER)
        | {"python": str(matrix.EXPECTED_PYTHON), "flags": ["-I", "-B"]},
        "training_manifest": binding(TRAINING_MANIFEST)
        | {
            "status_json_pointer": "/status",
            "required_status": "training_completed_all_endpoints_published",
        },
        "training_completion_seal": binding(TRAINING_SEAL),
        "evaluator": binding(EVALUATOR)
        | {"python": str(matrix.EXPECTED_PYTHON), "flags": ["-I", "-B"]},
        "evaluator_dependencies": [
            binding(ROOT / "tools/train_ppo.py"),
            binding(ROOT / "tools/train_bc_orbit.py"),
        ],
        "endpoints": endpoints,
        "data_bindings": {name: binding(path) for name, path in DATA.items()},
        "authoritative_panel_gates": matrix.EXPECTED_PANEL_GATES,
        "shared_protocol": matrix.EXPECTED_SHARED_PROTOCOL,
        "selection_rule": {
            "all_24_outputs_required_before_decision": True,
            "run_all_endpoints_without_early_stopping": True,
            "endpoint_requires_all_six_panels": True,
            "select": "smallest_step_among_fully_eligible_endpoints",
            "step_order": [1, 2, 4, 8],
            "none_eligible": "close_branch_without_broad_or_gold",
            "execution_failure": "no_selection_no_broad_no_gold",
        },
        "scope": {
            "local_only": True,
            "network": False,
            "training": False,
            "specialist_behavior": True,
            "broad": False,
            "gold": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
        "output_rule": {
            "root": str(OUTPUT_ROOT.relative_to(ROOT)),
            "attempt_marker": str(MARKER.relative_to(ROOT)),
            "evaluation_count_exact": 24,
            "attempts_per_evaluation": 1,
            "formal_attempts_authorized": 1,
            "retry_authorized": False,
            "run_all_before_decision": True,
            "manifest": "specialist_execution_manifest.json",
            "decision": "specialist_selection_decision.json",
            "completion_seal": "COMPLETED.json",
            "final_directory_mode": "0o500",
        },
        "ordered_evaluations": ordered,
        "ordered_command_matrix_sha256": matrix.sha256_bytes(
            matrix.canonical_json_bytes(commands)
        ),
    }
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"path": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
