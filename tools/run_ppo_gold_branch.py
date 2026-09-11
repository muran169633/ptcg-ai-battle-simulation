#!/usr/bin/env python3
"""Preregister and execute exactly one frozen ``B_gold_league`` PPO run.

The source of truth is an immutable preregistration emitted by
``tools/run_ppo_gold_ab.py``.  This wrapper does not rebuild or edit that plan:
it verifies the source plan hash, verifies the currently installed
``tools/train_ppo.py`` against the source plan, selects exactly one
``B_gold_league`` command by seed, and binds that exact command into a second
immutable preregistration.

No child process is started unless ``--execute`` is explicit.  The only
permitted child program is the currently verified ``tools/train_ppo.py``.
There is no A-branch, shell, network, packaging, upload, or submission path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = (REPO_ROOT / "tools" / "train_ppo.py").resolve()
SOURCE_PREREGISTRATION_SCHEMA = "ptcg-gold-ppo-ab-preregistration-v1"
SOURCE_PLAN_SCHEMA = "ptcg-gold-ppo-ab-plan-v1"
BRANCH_PREREGISTRATION_SCHEMA = (
    "ptcg-ppo-gold-branch-preregistration-v1"
)
RUN_SUMMARY_SCHEMA = "ptcg-ppo-gold-branch-run-v1"
BRANCH_NAME = "B_gold_league"
FORBIDDEN_ARGUMENT_PREFIXES = (
    "--kaggle",
    "--network",
    "--package",
    "--publish",
    "--submit",
    "--upload",
)
FORBIDDEN_ARGUMENT_WORDS = {
    "kaggle",
    "network",
    "package",
    "publish",
    "submit",
    "upload",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _resolved_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    return Path(value).expanduser().resolve()


def _single_command_value(command: Sequence[str], flag: str) -> str:
    indices = [index for index, value in enumerate(command) if value == flag]
    if len(indices) != 1:
        raise ValueError(
            f"Selected B command must contain {flag} exactly once"
        )
    index = indices[0]
    if index + 1 >= len(command):
        raise ValueError(f"Selected B command has no value after {flag}")
    return command[index + 1]


def _validate_source_safety(plan: dict[str, Any]) -> None:
    safety = plan.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("Source plan has no safety declaration")
    required = {
        "local_training_only": True,
        "network_calls": False,
        "uploads": False,
        "submission": False,
        "packaging": False,
        "uses_open_submission_code": False,
    }
    for key, expected in required.items():
        if safety.get(key) is not expected:
            raise ValueError(
                f"Source plan safety.{key} must be {expected!r}"
            )
    allowed_child = _resolved_path(
        safety.get("allowed_child_program"),
        "source plan safety.allowed_child_program",
    )
    if allowed_child != TRAIN_SCRIPT:
        raise ValueError(
            "Source plan permits a child program other than train_ppo.py"
        )


def _validate_command_arguments(command: Sequence[str]) -> None:
    for argument in command[2:]:
        lowered = argument.casefold()
        if lowered.startswith(("--output-dir=", "--seed=")):
            raise ValueError(
                "Selected B command must express --output-dir and --seed as "
                "separate, uniquely auditable arguments"
            )
        if (
            lowered in FORBIDDEN_ARGUMENT_WORDS
            or lowered.startswith(FORBIDDEN_ARGUMENT_PREFIXES)
        ):
            raise ValueError(
                f"Selected B command contains forbidden argument {argument!r}"
            )
        if "a_marnie_control" in lowered:
            raise ValueError(
                "Selected B command contains an A-branch path or argument"
            )


@dataclass(frozen=True)
class SelectedBranchRun:
    source_preregistration: Path
    source_preregistration_sha256: str
    source_plan_sha256: str
    source_plan_output_root: Path
    source_train_script_sha256: str
    seed: int
    command: tuple[str, ...]
    command_sha256: str
    output_dir: Path


def select_b_run(
    source_preregistration: Path,
    seed: int,
) -> SelectedBranchRun:
    source_path = source_preregistration.expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(
            f"Training preregistration is not a file: {source_path}"
        )
    source_sha256 = file_sha256(source_path)
    registration = read_json_object(
        source_path,
        "training preregistration",
    )
    if registration.get("schema_version") != SOURCE_PREREGISTRATION_SCHEMA:
        raise ValueError(
            "Training preregistration schema is not "
            f"{SOURCE_PREREGISTRATION_SCHEMA!r}"
        )
    if registration.get("status") != "preregistered_not_started":
        raise ValueError(
            "Training preregistration status must be "
            "'preregistered_not_started'"
        )
    plan = registration.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("Training preregistration has no plan object")
    if plan.get("schema_version") != SOURCE_PLAN_SCHEMA:
        raise ValueError(
            f"Source plan schema is not {SOURCE_PLAN_SCHEMA!r}"
        )
    actual_plan_sha256 = canonical_json_sha256(plan)
    if registration.get("plan_sha256") != actual_plan_sha256:
        raise ValueError("Training preregistration plan SHA-256 mismatch")

    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Source plan has no inputs object")
    declared_train_script = _resolved_path(
        inputs.get("train_script"),
        "source plan inputs.train_script",
    )
    if declared_train_script != TRAIN_SCRIPT:
        raise ValueError(
            "Source plan train_script is not the repository train_ppo.py"
        )
    current_train_sha256 = file_sha256(TRAIN_SCRIPT)
    if inputs.get("train_script_sha256") != current_train_sha256:
        raise ValueError(
            "Current train_ppo.py SHA-256 differs from the source plan"
        )
    declared_python = _resolved_path(
        inputs.get("python_executable"),
        "source plan inputs.python_executable",
    )
    current_python = Path(sys.executable).resolve()
    if declared_python != current_python:
        raise ValueError(
            "Current Python executable differs from the source plan"
        )
    _validate_source_safety(plan)

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    branches = plan.get("branches")
    if not isinstance(branches, dict):
        raise ValueError("Source plan has no branches object")
    branch = branches.get(BRANCH_NAME)
    if not isinstance(branch, dict):
        raise ValueError(f"Source plan has no {BRANCH_NAME!r} branch")
    runs = branch.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"{BRANCH_NAME}.runs must be a list")

    observed_seeds: list[int] = []
    matching_runs: list[dict[str, Any]] = []
    for index, raw_run in enumerate(runs):
        if not isinstance(raw_run, dict):
            raise ValueError(f"{BRANCH_NAME}.runs[{index}] is not an object")
        raw_seed = raw_run.get("seed")
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise ValueError(
                f"{BRANCH_NAME}.runs[{index}].seed must be an integer"
            )
        observed_seeds.append(raw_seed)
        if raw_seed == seed:
            matching_runs.append(raw_run)
    if len(observed_seeds) != len(set(observed_seeds)):
        raise ValueError(f"{BRANCH_NAME} contains duplicate seed entries")
    if len(matching_runs) != 1:
        raise ValueError(
            f"Expected exactly one {BRANCH_NAME} run for seed {seed}, "
            f"found {len(matching_runs)}"
        )
    run = matching_runs[0]

    raw_command = run.get("command")
    if (
        not isinstance(raw_command, list)
        or len(raw_command) < 3
        or not all(isinstance(value, str) for value in raw_command)
    ):
        raise ValueError("Selected B command must be a non-empty string list")
    command = tuple(raw_command)
    if Path(command[0]).expanduser().resolve() != current_python:
        raise ValueError(
            "Selected B command uses a Python executable other than the "
            "current source-plan interpreter"
        )
    if Path(command[1]).expanduser().resolve() != TRAIN_SCRIPT:
        raise ValueError(
            "Selected B command child program is not train_ppo.py"
        )
    _validate_command_arguments(command)

    command_output = _resolved_path(
        _single_command_value(command, "--output-dir"),
        "selected B command --output-dir",
    )
    declared_output = _resolved_path(
        run.get("output_dir"),
        "selected B run output_dir",
    )
    if command_output != declared_output:
        raise ValueError(
            "Selected B command --output-dir differs from its run record"
        )
    plan_output_root = _resolved_path(
        plan.get("output_root"),
        "source plan output_root",
    )
    expected_output = (
        plan_output_root / BRANCH_NAME / f"seed-{seed}"
    ).resolve()
    if declared_output != expected_output:
        raise ValueError(
            "Selected B output_dir is not the canonical B/seed path below "
            "the source plan output root"
        )
    raw_command_seed = _single_command_value(command, "--seed")
    try:
        command_seed = int(raw_command_seed)
    except ValueError as error:
        raise ValueError(
            "Selected B command --seed must be an integer"
        ) from error
    if command_seed != seed:
        raise ValueError(
            "Selected B command --seed differs from the selected run seed"
        )

    return SelectedBranchRun(
        source_preregistration=source_path,
        source_preregistration_sha256=source_sha256,
        source_plan_sha256=actual_plan_sha256,
        source_plan_output_root=plan_output_root,
        source_train_script_sha256=current_train_sha256,
        seed=seed,
        command=command,
        command_sha256=canonical_json_sha256(list(command)),
        output_dir=declared_output,
    )


@dataclass(frozen=True)
class BranchArtifactPaths:
    preregistration: Path
    log: Path
    run_summary: Path
    run_summary_sha256: Path


def resolve_artifact_paths(
    selected: SelectedBranchRun,
    *,
    preregistration: Path | None = None,
    log: Path | None = None,
    run_summary: Path | None = None,
) -> BranchArtifactPaths:
    prefix = (
        f"{selected.source_plan_output_root.name}."
        f"{BRANCH_NAME}.seed-{selected.seed}"
    )
    parent = selected.source_plan_output_root.parent
    preregistration_path = (
        preregistration.expanduser().resolve()
        if preregistration is not None
        else (parent / f"{prefix}.preregistration.json").resolve()
    )
    log_path = (
        log.expanduser().resolve()
        if log is not None
        else (parent / f"{prefix}.log").resolve()
    )
    summary_path = (
        run_summary.expanduser().resolve()
        if run_summary is not None
        else (parent / f"{prefix}.run_summary.json").resolve()
    )
    summary_hash_path = summary_path.with_suffix(
        summary_path.suffix + ".sha256"
    )
    values = (
        preregistration_path,
        log_path,
        summary_path,
        summary_hash_path,
    )
    if len(values) != len(set(values)):
        raise ValueError("Branch artifact paths must be distinct")
    for path in values:
        if path == selected.source_plan_output_root or path.is_relative_to(
            selected.source_plan_output_root
        ):
            raise ValueError(
                "Branch sidecars must be outside the source plan output root"
            )
    return BranchArtifactPaths(
        preregistration=preregistration_path,
        log=log_path,
        run_summary=summary_path,
        run_summary_sha256=summary_hash_path,
    )


def branch_safety_declaration() -> dict[str, Any]:
    return {
        "selected_branch": BRANCH_NAME,
        "a_branch_execution": False,
        "exactly_one_child_command": True,
        "allowed_child_program": str(TRAIN_SCRIPT),
        "shell": False,
        "network_calls": False,
        "packaging": False,
        "uploads": False,
        "submission": False,
    }


def build_binding(
    selected: SelectedBranchRun,
    paths: BranchArtifactPaths,
) -> dict[str, Any]:
    wrapper_path = Path(__file__).resolve()
    return {
        "source_training_preregistration": str(
            selected.source_preregistration
        ),
        "source_training_preregistration_sha256": (
            selected.source_preregistration_sha256
        ),
        "source_plan_sha256": selected.source_plan_sha256,
        "source_train_script": str(TRAIN_SCRIPT),
        "source_train_script_sha256": (
            selected.source_train_script_sha256
        ),
        "wrapper": str(wrapper_path),
        "wrapper_sha256": file_sha256(wrapper_path),
        "branch": BRANCH_NAME,
        "seed": selected.seed,
        "command": list(selected.command),
        "command_sha256": selected.command_sha256,
        "output_dir": str(selected.output_dir),
        "artifacts": {
            "preregistration": str(paths.preregistration),
            "log": str(paths.log),
            "run_summary": str(paths.run_summary),
            "run_summary_sha256": str(paths.run_summary_sha256),
        },
        "safety": branch_safety_declaration(),
    }


def preregistration_envelope(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": BRANCH_PREREGISTRATION_SCHEMA,
        "created_at": utc_now(),
        "status": "preregistered_not_started",
        "binding_sha256": canonical_json_sha256(binding),
        "binding": binding,
    }


def validate_existing_preregistration(
    path: Path,
    expected_binding: dict[str, Any],
) -> dict[str, Any]:
    registration = read_json_object(
        path,
        "B-only preregistration",
    )
    if registration.get("schema_version") != BRANCH_PREREGISTRATION_SCHEMA:
        raise ValueError("B-only preregistration schema mismatch")
    if registration.get("status") != "preregistered_not_started":
        raise ValueError("B-only preregistration status mismatch")
    binding = registration.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("B-only preregistration has no binding object")
    binding_sha256 = canonical_json_sha256(binding)
    if registration.get("binding_sha256") != binding_sha256:
        raise ValueError("B-only preregistration binding SHA-256 mismatch")
    if binding != expected_binding:
        raise ValueError(
            "B-only preregistration differs from the current verified source "
            "training preregistration or command"
        )
    return registration


def directory_manifest(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    if not root.is_dir():
        raise RuntimeError(f"Training output is not a directory: {root}")
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(
                f"Training output contains unsupported symlink: {path}"
            )
        if not path.is_file():
            continue
        result.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return result


def _require_absent(paths: Sequence[Path], label: str) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite {label}: "
            + ", ".join(str(path) for path in existing)
        )


def execute_selected_run(
    selected: SelectedBranchRun,
    paths: BranchArtifactPaths,
    branch_registration: dict[str, Any],
) -> dict[str, Any]:
    _require_absent(
        (
            selected.source_plan_output_root,
            selected.output_dir,
            paths.log,
            paths.run_summary,
            paths.run_summary_sha256,
        ),
        "B-only training output",
    )
    paths.log.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    launch_error: str | None = None
    return_code: int | None = None
    with paths.log.open("x", encoding="utf-8") as log_handle:
        try:
            completed = subprocess.run(
                list(selected.command),
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                shell=False,
            )
            return_code = int(completed.returncode)
        except OSError as error:
            launch_error = f"{type(error).__name__}: {error}"

    output_manifest = directory_manifest(selected.output_dir)
    if launch_error is not None:
        status = "launch_failed"
    elif return_code != 0:
        status = "failed"
    elif not selected.output_dir.is_dir():
        status = "failed_missing_output"
    else:
        status = "completed"
    summary = {
        "schema_version": RUN_SUMMARY_SCHEMA,
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": status,
        "branch": BRANCH_NAME,
        "seed": selected.seed,
        "source_training_preregistration": str(
            selected.source_preregistration
        ),
        "source_training_preregistration_sha256": (
            selected.source_preregistration_sha256
        ),
        "source_plan_sha256": selected.source_plan_sha256,
        "branch_preregistration": str(paths.preregistration),
        "branch_preregistration_sha256": file_sha256(
            paths.preregistration
        ),
        "binding_sha256": branch_registration["binding_sha256"],
        "command_sha256": selected.command_sha256,
        "return_code": return_code,
        "launch_error": launch_error,
        "output_dir": str(selected.output_dir),
        "output_files": output_manifest,
        "output_manifest_sha256": canonical_json_sha256(output_manifest),
        "log": str(paths.log),
        "log_sha256": file_sha256(paths.log),
        "safety": branch_safety_declaration(),
    }
    atomic_write_json(paths.run_summary, summary)
    summary_sha256 = file_sha256(paths.run_summary)
    atomic_write_text(
        paths.run_summary_sha256,
        f"{summary_sha256}  {paths.run_summary.name}\n",
    )
    return {
        **summary,
        "run_summary_sha256": summary_sha256,
        "run_summary_sha256_file": str(paths.run_summary_sha256),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an immutable gold PPO A/B preregistration and "
            "preregister or execute exactly one B_gold_league seed."
        )
    )
    parser.add_argument(
        "--training-preregistration",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--run-summary", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the B-only binding without writing.",
    )
    mode.add_argument(
        "--preregister-only",
        action="store_true",
        help="Write the immutable B-only preregistration without training.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute exactly the selected B command.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected = select_b_run(
        args.training_preregistration,
        args.seed,
    )
    paths = resolve_artifact_paths(
        selected,
        preregistration=args.preregistration,
        log=args.log,
        run_summary=args.run_summary,
    )
    if selected.source_plan_output_root.exists():
        raise FileExistsError(
            "Refusing a B-only run because the source plan output root "
            f"already exists: {selected.source_plan_output_root}"
        )
    binding = build_binding(selected, paths)

    if args.dry_run or not (args.preregister_only or args.execute):
        envelope = preregistration_envelope(binding)
        print(
            json.dumps(
                envelope,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return envelope

    if args.preregister_only:
        if paths.preregistration.exists():
            raise FileExistsError(
                "Refusing to overwrite B-only preregistration: "
                f"{paths.preregistration}"
            )
        envelope = preregistration_envelope(binding)
        atomic_write_json(paths.preregistration, envelope)
        print(str(paths.preregistration))
        return envelope

    if paths.preregistration.exists():
        envelope = validate_existing_preregistration(
            paths.preregistration,
            binding,
        )
    else:
        envelope = preregistration_envelope(binding)
        atomic_write_json(paths.preregistration, envelope)
    result = execute_selected_run(
        selected,
        paths,
        envelope,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    if args.execute and result.get("status") != "completed":
        return_code = result.get("return_code")
        raise SystemExit(
            return_code
            if isinstance(return_code, int) and return_code > 0
            else 1
        )


if __name__ == "__main__":
    main()
