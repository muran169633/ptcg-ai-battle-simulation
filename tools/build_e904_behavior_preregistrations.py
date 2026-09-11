#!/usr/bin/env python3
"""Build, but never execute, the E904 specialist and broad preregistrations.

The candidate checkpoint path and file SHA-256 are explicit arguments.  The
runtime model-state SHA-256 is not configurable: the checkpoint must decode to
the frozen E904 state.  The generated specialist preregistration is executable
only by ``run_e904_single_endpoint_specialist.py``.  The generated two-panel
broad document is intentionally a blocked template because the older frozen
two-panel runner requires a source protocol that was locked before training;
this post-hoc E904 endpoint has no such source protocol and the generator will
not fabricate one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_e904_single_endpoint_specialist as specialist  # noqa: E402


SCRIPT = ROOT / "tools/build_e904_behavior_preregistrations.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SPECIALIST_EXECUTOR = ROOT / "tools/run_e904_single_endpoint_specialist.py"
SPECIALIST_EXECUTOR_SHA256 = (
    "f749fd660209e2369d6cdf21e91ca0a5234eb3306d50af98165cac502f37e6c7"
)
FROZEN_BROAD_RUNNER = (
    ROOT / "tools/run_policy_behavior_panels_frozen_candidate_20260731.py"
)
FROZEN_BROAD_RUNNER_SHA256 = (
    "69ef3e4b650c76d1814a03b6ea5350ed5d94118643f611428b47f3dfbf132507"
)
BROAD_TEMPLATE_SCHEMA = "ptcg-e904-broad-behavior-preregistration-template-v2"

BROAD_PANELS: tuple[dict[str, Any], ...] = (
    {
        "name": "old_retention",
        "data": "data/bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip",
        "data_sha256": "a3b9d572bfc0a784b5b140d3dbe09314252c46dd0d9c1afa3423236cda54543c",
        "gates": {
            "rows": {"exact": 34204},
            "set": {"minimum": 28160},
            "hybrid": {"minimum": 27970},
            "ordered": {"minimum": 27751},
            "value": {"exact": 25025},
            "count": {"minimum": 34007},
            "top1": {"minimum": 28384},
            "context34_rows": {"exact": 250},
            "context34_hybrid": {"minimum": 189},
            "context34_ordered": {"minimum": 189},
        },
    },
    {
        "name": "valid29",
        "data": "data/bc_marnie_top50plus_gold21_timeforward_train28_valid29_v2_20260731.zip",
        "data_sha256": "95638471e0b842c6366231e95b2e98a5d806f612a0085ef39110ba4cb6f0ad0d",
        "gates": {
            "rows": {"exact": 159829},
            "set": {"minimum": 127040},
            "hybrid": {"minimum": 126618},
            "ordered": {"minimum": 125785},
            "value": {"exact": 118593},
            "count": {"minimum": 158197},
            "top1": {"minimum": 128523},
            "context34_rows": {"exact": 639},
            "context34_hybrid": {"minimum": 529},
            "context34_ordered": {"minimum": 529},
        },
    },
)


def binding(path: Path) -> dict[str, str]:
    payload, evidence = specialist.read_plain_file(path, str(path))
    del payload
    return {
        "path": specialist.root_relative_text(path),
        "sha256": evidence["sha256"],
    }


def require_absent(path: Path, label: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise FileExistsError(f"{label} already exists: {path}")


def broad_command(checkpoint: Path, panel: dict[str, Any], output: Path) -> list[str]:
    return [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        "tools/evaluate_policy_bc.py",
        "--checkpoint",
        specialist.root_relative_text(checkpoint),
        "--data",
        panel["data"],
        "--split",
        "valid",
        "--split-mode",
        "archive",
        "--split-seed",
        "20260723",
        "--batch-size",
        "256",
        "--workers",
        "8",
        "--prediction-order",
        "policy",
        "--device",
        "cuda",
        "--compact",
        "--progress-interval",
        "0",
        "--json-output",
        specialist.root_relative_text(output),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write E904 behavior preregistrations without evaluating them."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Root-relative materialized E904 checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="Exact file SHA-256 of the materialized checkpoint.",
    )
    parser.add_argument(
        "--artifact-id",
        required=True,
        help="Fresh identifier beginning with e904_ (letters, digits, underscore, hyphen).",
    )
    return parser.parse_args()


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("run from the repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"wrong Python runtime: {sys.executable}")
    if sys.flags.isolated != 1 or sys.dont_write_bytecode is not True:
        raise RuntimeError("generator requires Python flags -I -B")


def main() -> None:
    args = parse_args()
    validate_runtime()
    if not re.fullmatch(r"e904_[a-z0-9][a-z0-9_-]{2,95}", args.artifact_id):
        raise ValueError("artifact-id must be a fresh safe e904_* identifier")

    checkpoint = specialist.root_relative_path(args.checkpoint, "checkpoint")
    checkpoint_sha256 = specialist.require_sha256(
        args.checkpoint_sha256,
        "checkpoint SHA-256",
    )
    checkpoint_binding = {
        "path": specialist.root_relative_text(checkpoint),
        "sha256": checkpoint_sha256,
        "model_state_sha256": specialist.E904_MODEL_STATE_SHA256,
        "checkpoint_update": specialist.EXPECTED_CHECKPOINT_UPDATE,
    }
    _, checkpoint_payload, checkpoint_evidence = specialist.verify_binding(
        checkpoint_binding,
        "candidate checkpoint",
    )
    checkpoint_evidence.update(
        specialist.load_checkpoint_identity(checkpoint_payload, checkpoint_binding)
    )
    raw_parent_binding = {
        "path": specialist.root_relative_text(specialist.RAW_U468_CHECKPOINT),
        "sha256": specialist.RAW_U468_CHECKPOINT_SHA256,
        "model_state_sha256": specialist.RAW_U468_MODEL_STATE_SHA256,
        "checkpoint_update": specialist.EXPECTED_CHECKPOINT_UPDATE,
    }
    raw_parent_path, raw_parent_payload, _ = specialist.verify_binding(
        raw_parent_binding,
        "raw U468 parent checkpoint",
        expected_path=specialist.RAW_U468_CHECKPOINT,
        expected_sha256=specialist.RAW_U468_CHECKPOINT_SHA256,
    )
    specialist.load_checkpoint_identity(
        raw_parent_payload,
        raw_parent_binding,
        expected_model_state_sha256=specialist.RAW_U468_MODEL_STATE_SHA256,
        label="raw U468 parent",
    )

    specialist_prereg_path = ROOT / "artifacts" / (
        f"{args.artifact_id}.specialist_execution_preregistration.json"
    )
    specialist_output_root = ROOT / "artifacts" / (
        f"{args.artifact_id}.specialist_behavior"
    )
    specialist_marker = ROOT / "artifacts" / (
        f".ptcg-{args.artifact_id}-specialist-attempt.json"
    )
    broad_template_path = ROOT / "artifacts" / (
        f"{args.artifact_id}.broad_behavior_preregistration_template.json"
    )
    broad_output_root = ROOT / "artifacts" / f"{args.artifact_id}.broad_behavior"
    for path, label in (
        (specialist_prereg_path, "specialist preregistration"),
        (specialist_output_root, "specialist output root"),
        (specialist_marker, "specialist attempt marker"),
        (broad_template_path, "broad preregistration template"),
        (broad_output_root, "broad output root"),
    ):
        require_absent(path, label)

    data_bindings: dict[str, dict[str, str]] = {}
    baseline_bindings: dict[str, dict[str, str]] = {}
    evaluations: list[dict[str, Any]] = []
    for order, panel in enumerate(specialist.PANEL_SPECS, start=1):
        data_name = str(panel["data_name"])
        data_bindings.setdefault(
            data_name,
            {
                "path": str(panel["data_path"]),
                "sha256": str(panel["data_sha256"]),
            },
        )
        baseline_bindings[str(panel["name"])] = {
            "path": str(panel["baseline_path"]),
            "sha256": str(panel["baseline_sha256"]),
        }
        output = specialist_output_root / f"{panel['name']}.json"
        command = specialist.expected_command(checkpoint, panel, output)
        evaluations.append(
            {
                "order": order,
                "panel": panel["name"],
                "team_name": panel["team_name"],
                "output": specialist.root_relative_text(output),
                "output_absent_at_lock": True,
                "command": command,
                "command_sha256": specialist.sha256_bytes(
                    specialist.canonical_json_bytes(command)
                ),
                "attempts_authorized": 1,
            }
        )
    for name, item in data_bindings.items():
        actual = binding(ROOT / item["path"])
        if actual != item:
            raise RuntimeError(f"data binding drift for {name}")
    for name, item in baseline_bindings.items():
        actual = binding(ROOT / item["path"])
        if actual != item:
            raise RuntimeError(f"baseline binding drift for {name}")
        _, payload, _ = specialist.verify_binding(
            item,
            f"baseline {name}",
        )
        document = specialist.strict_json_loads(payload, f"baseline {name}")
        panel = next(
            panel for panel in specialist.PANEL_SPECS if panel["name"] == name
        )
        specialist.validate_output_identity(
            document,
            checkpoint=raw_parent_path,
            checkpoint_sha256=specialist.RAW_U468_CHECKPOINT_SHA256,
            panel=panel,
        )

    executor_binding = binding(SPECIALIST_EXECUTOR) | {
        "python": str(EXPECTED_PYTHON),
        "flags": ["-I", "-B"],
    }
    if executor_binding["sha256"] != SPECIALIST_EXECUTOR_SHA256:
        raise RuntimeError("specialist executor SHA-256 drift")
    evaluator_binding = binding(specialist.EXPECTED_EVALUATOR) | {
        "python": str(EXPECTED_PYTHON),
        "flags": ["-I", "-B"],
    }
    dependency_bindings = {
        name: binding(path)
        for name, (path, _) in specialist.EXPECTED_DEPENDENCIES.items()
    }
    frozen_broad_runner_binding = binding(FROZEN_BROAD_RUNNER)
    if frozen_broad_runner_binding["sha256"] != FROZEN_BROAD_RUNNER_SHA256:
        raise RuntimeError("frozen broad runner SHA-256 drift")
    preregistration = {
        "schema_version": specialist.PREREGISTRATION_SCHEMA,
        "status": "locked_before_specialist_evaluation",
        "generator": binding(SCRIPT),
        "executor": executor_binding,
        "candidate": checkpoint_binding,
        "raw_u468_parent": raw_parent_binding,
        "evaluator": evaluator_binding,
        "evaluator_dependencies": dependency_bindings,
        "data_bindings": data_bindings,
        "baseline_bindings": baseline_bindings,
        "shared_protocol": specialist.SHARED_PROTOCOL,
        "authoritative_panel_gates": {
            name: specialist.expected_thresholds(name)
            for name in specialist.BASELINE_METRICS
        },
        "ordered_evaluations": evaluations,
        "output_rule": {
            "root": specialist.root_relative_text(specialist_output_root),
            "root_absent_at_lock": True,
            "attempt_marker": specialist.root_relative_text(specialist_marker),
            "attempt_marker_absent_at_lock": True,
            "evaluation_count_exact": 6,
            "attempts_per_evaluation": 1,
            "retry_authorized": False,
            "run_all_before_decision": True,
            "manifest": "specialist_execution_manifest.json",
            "decision": "specialist_decision.json",
            "final_directory_mode": "0o500",
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
    }

    broad_evaluations: list[dict[str, Any]] = []
    for order, panel in enumerate(BROAD_PANELS, start=1):
        output = broad_output_root / f"{panel['name']}.json"
        command = broad_command(checkpoint, panel, output)
        broad_evaluations.append(
            {
                "order": order,
                "panel": panel["name"],
                "data": panel["data"],
                "data_sha256": panel["data_sha256"],
                "output": specialist.root_relative_text(output),
                "output_absent_at_lock": True,
                "command": command,
                "command_sha256": specialist.sha256_bytes(
                    specialist.canonical_json_bytes(command)
                ),
                "attempts_authorized": 0,
            }
        )
        actual = binding(ROOT / panel["data"])
        if actual["sha256"] != panel["data_sha256"]:
            raise RuntimeError(f"broad data binding drift for {panel['name']}")
    broad_template = {
        "schema_version": BROAD_TEMPLATE_SCHEMA,
        "status": "blocked_pending_formal_broad_executor_authorization",
        "generator": binding(SCRIPT),
        "candidate": checkpoint_binding,
        "raw_u468_parent": raw_parent_binding,
        "evaluator": evaluator_binding,
        "frozen_runner_review": {
            **frozen_broad_runner_binding,
            "directly_compatible": False,
            "reason": (
                "The frozen runner requires a source protocol locked before "
                "training and a matching training-integrity decision. No such "
                "pre-training E904 source protocol exists; none is fabricated."
            ),
        },
        "shared_protocol": specialist.SHARED_PROTOCOL,
        "authoritative_panel_gates": {
            panel["name"]: panel["gates"] for panel in BROAD_PANELS
        },
        "ordered_evaluations": broad_evaluations,
        "output_rule": {
            "root": specialist.root_relative_text(broad_output_root),
            "root_absent_at_lock": True,
            "evaluation_count_exact": 2,
            "attempts_authorized": 0,
            "formal_executor_required_before_attempt": True,
            "no_manual_execution_from_this_template": True,
        },
        "scope": {
            "local_only": True,
            "network": False,
            "training": False,
            "specialist_behavior": False,
            "broad": True,
            "gold": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }

    specialist_payload = specialist.canonical_json_bytes(preregistration)
    broad_payload = specialist.canonical_json_bytes(broad_template)
    specialist.atomic_create(specialist_prereg_path, specialist_payload, mode=0o444)
    specialist.atomic_create(broad_template_path, broad_payload, mode=0o444)
    print(
        json.dumps(
            {
                "candidate": checkpoint_evidence,
                "specialist_preregistration": {
                    "path": specialist.root_relative_text(specialist_prereg_path),
                    "sha256": specialist.sha256_bytes(specialist_payload),
                },
                "broad_preregistration_template": {
                    "path": specialist.root_relative_text(broad_template_path),
                    "sha256": specialist.sha256_bytes(broad_payload),
                    "status": broad_template["status"],
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
