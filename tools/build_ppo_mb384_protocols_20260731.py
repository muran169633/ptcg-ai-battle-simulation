#!/usr/bin/env python3
"""Build the one-shot mb384 continuation protocol bundle.

This module is intentionally generation-only.  It never imports torch, opens a
CUDA device, launches training, invokes a shell, or accesses the network.  The
default CLI mode is an in-memory dry run.  ``--write`` is the only mode that
creates files, and it uses O_CREAT|O_EXCL for all three formal JSON artifacts.

The live behavior runner, direct-exec training launcher, and write-once
terminal-receipt writer are infrastructure.  The receipt writer is already a
fixed final artifact and must be explicitly acknowledged on every invocation;
the other final SHA-256 digests must be supplied by the caller in write mode.
Schema constants are read from Python source where applicable; no old launcher
schema or provisional infrastructure hash is silently frozen.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE = (
    "ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_u456_to_u464_seed202607331"
)
TRAINING_SEED = 202607331
SCREEN_CONTROL_SEED = 202607332
SCREEN_CANDIDATE_SEED = 202607333
CONFIRM_CONTROL_SEED = 202607334
CONFIRM_CANDIDATE_SEED = 202607335
FRESH_GOLD19_SEEDS = (
    SCREEN_CONTROL_SEED,
    SCREEN_CANDIDATE_SEED,
    CONFIRM_CONTROL_SEED,
    CONFIRM_CANDIDATE_SEED,
)

RUNNER_PREREGISTRATION_REL = Path(
    "artifacts/"
    f"{CANDIDATE}.runner_preregistration.json"
)
BRANCH_PREREGISTRATION_REL = Path(
    "artifacts/"
    f"{CANDIDATE}.B_gold_league.seed-{TRAINING_SEED}.preregistration.json"
)
SOURCE_PROTOCOL_REL = Path(f"artifacts/{CANDIDATE}.source_protocol.json")
COMPREHENSIVE_PREREGISTRATION_REL = Path(
    f"artifacts/{CANDIDATE}.comprehensive_preregistration.json"
)
TRANSPORT_LOCK_REL = Path(f"artifacts/{CANDIDATE}.transport_lock.json")
TERMINAL_RECEIPT_REL = Path(
    f"artifacts/{CANDIDATE}.training_terminal_receipt.json"
)

PRIOR_VCOEF_COMPREHENSIVE_REL = Path(
    "artifacts/"
    "ppo_bc28_currentgold10_ctx34q4_episode_mean_actorlr2x_"
    "vcoef0125_seed202607311_20260731.comprehensive_preregistration.json"
)
BEHAVIOR_RUNNER_REL = Path(
    "tools/run_policy_behavior_panels_frozen_candidate_20260731.py"
)
TRAINING_LAUNCHER_REL = Path("tools/exec_preregistered_ppo_training.py")
TERMINAL_RECEIPT_WRITER_REL = Path("tools/write_o_excl_json_receipt.py")
TRAINER_REL = Path("tools/train_ppo.py")
BEHAVIOR_EVALUATOR_REL = Path("tools/evaluate_policy_bc.py")
BEHAVIOR_DEPENDENCY_RELS = {
    "train_bc_orbit": Path("tools/train_bc_orbit.py"),
    "train_ppo": TRAINER_REL,
    "cg/__init__.py": Path(
        "dataset/sample_submission/sample_submission/cg/__init__.py"
    ),
    "cg/sim.py": Path(
        "dataset/sample_submission/sample_submission/cg/sim.py"
    ),
    "cg/libcg.so": Path(
        "dataset/sample_submission/sample_submission/cg/libcg.so"
    ),
}

FULL_INCUMBENT_U456_REL = Path(
    "artifacts/"
    "ppo_bc28init_currentgold10_tailfocus_replay2_ctx34q4_episode_mean_"
    "actorlr2x_u448meta_to_u456_seed20260736/B_gold_league/"
    "seed-20260736/checkpoints/update-0456.pt"
)
BC28_CHECKPOINT_REL = Path(
    "artifacts/"
    "bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/"
    "best.pt"
)
OLD_RETENTION_DATA_REL = Path(
    "data/bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip"
)
VALID29_DATA_REL = Path(
    "data/"
    "bc_marnie_top50plus_gold21_timeforward_train28_valid29_v2_20260731.zip"
)

EXPECTED_RUNNER_PREREGISTRATION_SHA256 = (
    "fe2dc8ab87215be864775869673ef0a4e0578dd0597c84464a9fedf38416a27c"
)
EXPECTED_RUNNER_PLAN_SHA256 = (
    "f1efe15b6452f95125eb6500191c5e53823d7875aa2997a71cd3eb523fd3f25c"
)
EXPECTED_BRANCH_PREREGISTRATION_SHA256 = (
    "b216a5e51669f3c30f0fd3b4d500fc5704a2e091c8ba7b5751c81ff110abff58"
)
EXPECTED_BRANCH_BINDING_SHA256 = (
    "60170c228bff0bf641595062c50976664c9053e092f5c5175e695335f7da06d1"
)
EXPECTED_BRANCH_COMMAND_SHA256 = (
    "23fbfdfff058eafc2a5c6d8f3f218d9e30bc0f29da5db20ecc42de51d79f3a04"
)
EXPECTED_PRIOR_VCOEF_SHA256 = (
    "5d1b691893e607ebd8048a94810a07393d3eed9abb99694639ccb67832fc28a8"
)
EXPECTED_FULL_INCUMBENT_U456_SHA256 = (
    "b7ed9580543e4a2374ffcc618bb2eed74b90e762117a587768c90daf0019c83b"
)
EXPECTED_TRAINER_SHA256 = (
    "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
)
EXPECTED_TERMINAL_RECEIPT_WRITER_SHA256 = (
    "e9897ebeca75309168bcc7e0cdb25f2b53ac7887e9132a2147057315f957872a"
)
EXPECTED_BC28_CHECKPOINT_SHA256 = (
    "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
)
EXPECTED_OLD_RETENTION_DATA_SHA256 = (
    "a3b9d572bfc0a784b5b140d3dbe09314252c46dd0d9c1afa3423236cda54543c"
)
EXPECTED_VALID29_DATA_SHA256 = (
    "95638471e0b842c6366231e95b2e98a5d806f612a0085ef39110ba4cb6f0ad0d"
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UPDATES = tuple(range(457, 465))
MINIBATCH_SIZE = 384
REFERENCE_MINIBATCH_SIZE = 512
PPO_EPOCHS = 2
GAMES_PER_UPDATE = 64

EXPECTED_BEHAVIOR_GATE_OUTPUT_KEYS = {
    "rows": ("metrics.rows", "exact"),
    "set_exact": ("metrics.set_exact_correct", "minimum"),
    "hybrid_order_exact": (
        "metrics.hybrid_order_exact_correct",
        "minimum",
    ),
    "ordered_exact": ("metrics.ordered_exact_correct", "minimum"),
    "value": ("metrics.value_correct", "minimum"),
    "count": ("metrics.count_correct", "minimum"),
    "top1": ("metrics.top1_correct", "minimum"),
    "context34_rows": ('metrics.by_context["34"].rows', "exact"),
    "context34_hybrid_order_exact": (
        'metrics.by_context["34"].hybrid_order_exact_correct',
        "minimum",
    ),
    "context34_ordered_exact": (
        'metrics.by_context["34"].ordered_exact_correct',
        "minimum",
    ),
}

SUPPORTED_SEALED_CHILD_COMMAND_KEYS = frozenset(
    {
        "source_binding",
        "json_path",
        "token_count",
        "canonical_sha256",
        "effective_trainer_fd",
        "effective_trainer_path",
        "sealed_trainer_payload_fd",
        "sealed_trainer_payload_path",
        "sealed_bootstrap_sha256",
        "effective_root_fd",
        "effective_root_path",
        "effective_repo_path_token_indices",
        "effective_command_canonical_sha256",
    }
)
SUPPORTED_ATTEMPT_KEYS = frozenset(
    {
        "seed",
        "attempts_authorized",
        "output_dir",
        "log",
        "terminal_receipt",
        "terminal_checkpoint",
        "attempt_start_marker",
        "unpublished_log_witness",
        "expected_terminal_update",
        "absent_at_lock",
    }
)
SUPPORTED_EXPECTED_TRANSPORT_KEYS = frozenset(
    {
        "tool",
        "sandbox_permissions",
        "login",
        "tty",
        "workdir",
        "shell",
        "topology",
    }
)


class ProtocolBuildError(RuntimeError):
    """A frozen input or generated protocol violates the preregistration."""


@dataclass(frozen=True)
class UpstreamBinding:
    runner_preregistration: dict[str, Any]
    branch_preregistration: dict[str, Any]
    command: tuple[str, ...]
    output_dir: Path
    log: Path
    terminal_checkpoint: Path


@dataclass(frozen=True)
class BehaviorRunnerContract:
    path: Path
    sha256: str
    constants: dict[str, Any]


@dataclass(frozen=True)
class ReceiptWriterContract:
    path: Path
    sha256: str


@dataclass(frozen=True)
class LauncherContract:
    path: Path
    sha256: str
    schema_version: str
    locked_status: str
    transport_lock_keys: frozenset[str]
    required_bindings: frozenset[str]
    expected_transport_constants: dict[str, Any]
    expected_transport_keys: frozenset[str]
    locked_absence_keys: frozenset[str]
    child_command_keys: frozenset[str]
    attempt_keys: frozenset[str]
    effective_trainer_fd: int
    effective_trainer_path: str
    sealed_trainer_payload_fd: int
    sealed_trainer_payload_path: str
    effective_root_fd: int
    effective_root_path: str
    build_trainer_bootstrap: Any
    build_effective_command: Any
    canonical_attempt_start_marker: Any
    launcher_canonical_json_sha256: Any


@dataclass(frozen=True)
class ProtocolBundle:
    source_protocol: dict[str, Any]
    comprehensive_preregistration: dict[str, Any]
    transport_lock: dict[str, Any]
    source_bytes: bytes
    comprehensive_bytes: bytes
    transport_bytes: bytes
    source_sha256: str
    comprehensive_sha256: str
    transport_sha256: str
    behavior_runner_sha256: str
    training_launcher_sha256: str
    terminal_receipt_writer_sha256: str
    provisional_infrastructure_hashes: bool
    pending_final_sha_fields: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProtocolBuildError(f"{label} must be a lowercase SHA-256 digest")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolBuildError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    expected_sha256 = require_sha256(expected_sha256, f"{label} SHA-256")
    if not path.is_file() or path.is_symlink():
        raise ProtocolBuildError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    observed = raw_sha256(raw)
    if observed != expected_sha256:
        raise ProtocolBuildError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed}"
        )
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolBuildError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ProtocolBuildError(f"{label} must contain a JSON object")
    return value


def repo_path(relative: Path) -> Path:
    if relative.is_absolute():
        raise ProtocolBuildError(f"expected repository-relative path: {relative}")
    path = REPO_ROOT / relative
    normalized = Path(os.path.normpath(os.fspath(path)))
    try:
        normalized.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ProtocolBuildError(f"path escapes repository: {relative}") from error
    return normalized


def repo_relative_text(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def require_file_hash(relative: Path, expected: str, label: str) -> Path:
    path = repo_path(relative)
    if not path.is_file() or path.is_symlink():
        raise ProtocolBuildError(f"{label} must be a regular non-symlink file")
    observed = file_sha256(path)
    if observed != require_sha256(expected, f"{label} SHA-256"):
        raise ProtocolBuildError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return path


def command_values(command: Sequence[str], flag: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, token in enumerate(command):
        if token != flag:
            continue
        if index + 1 >= len(command):
            raise ProtocolBuildError(f"command flag has no value: {flag}")
        values.append(command[index + 1])
    return tuple(values)


def require_command_value(
    command: Sequence[str],
    flag: str,
    expected: str,
) -> None:
    values = command_values(command, flag)
    if values != (expected,):
        raise ProtocolBuildError(
            f"{flag} must occur once with {expected!r}; observed {values!r}"
        )


def require_command_switch(command: Sequence[str], flag: str) -> None:
    count = sum(token == flag for token in command)
    if count != 1:
        raise ProtocolBuildError(
            f"{flag} must occur exactly once; observed {count}"
        )


def command_single_knob_evidence(command: Sequence[str]) -> dict[str, Any]:
    indices = [
        index
        for index, token in enumerate(command[:-1])
        if token == "--minibatch-size"
    ]
    if len(indices) != 1:
        raise ProtocolBuildError(
            "cannot construct single-knob evidence without one minibatch flag"
        )
    value_index = indices[0] + 1
    if command[value_index] != str(MINIBATCH_SIZE):
        raise ProtocolBuildError("candidate command is not the frozen mb384 command")
    reference = list(command)
    reference[value_index] = str(REFERENCE_MINIBATCH_SIZE)
    differences = [
        index
        for index, (left, right) in enumerate(
            zip(reference, command, strict=True)
        )
        if left != right
    ]
    if differences != [value_index]:
        raise ProtocolBuildError("derived minibatch reference has extra drift")
    return {
        "comparison_kind": (
            "derived unexecuted command-token counterfactual for configuration "
            "audit only"
        ),
        "reference_minibatch_size": REFERENCE_MINIBATCH_SIZE,
        "candidate_minibatch_size": MINIBATCH_SIZE,
        "reference_command_token_count": len(reference),
        "candidate_command_token_count": len(command),
        "differing_token_indices": differences,
        "difference": {
            str(value_index): (
                f"{REFERENCE_MINIBATCH_SIZE} to {MINIBATCH_SIZE}"
            )
        },
        "reference_command_canonical_sha256": canonical_json_sha256(reference),
        "candidate_command_canonical_sha256": canonical_json_sha256(
            list(command)
        ),
        "all_other_tokens_equal": True,
        "reference_command_is_not_an_authorized_training_arm": True,
        "does_not_remove_continued_training_confound": True,
    }


def command_opponent_quota_vector(
    command: Sequence[str],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, token in enumerate(command):
        if token != "--opponent-base-quota":
            continue
        if index + 2 >= len(command):
            raise ProtocolBuildError("truncated --opponent-base-quota")
        name = command[index + 1]
        try:
            quota = int(command[index + 2])
        except ValueError as error:
            raise ProtocolBuildError("opponent quota is not an integer") from error
        if name in result or quota <= 0:
            raise ProtocolBuildError("duplicate or non-positive opponent quota")
        result[name] = quota
    if sum(result.values()) != GAMES_PER_UPDATE:
        raise ProtocolBuildError(
            "opponent quota vector does not sum to games_per_update"
        )
    return result


def optimizer_steps_for_transitions(transitions: int) -> int:
    if type(transitions) is not int or transitions <= 0:
        raise ProtocolBuildError("transitions must be a positive integer")
    return PPO_EPOCHS * math.ceil(transitions / MINIBATCH_SIZE)


def _literal_module_constants(
    path: Path,
    requested_names: Iterable[str],
) -> tuple[dict[str, Any], set[str]]:
    """Read literal top-level constants without importing the target module."""

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        raise ProtocolBuildError(
            f"infrastructure source is not syntactically stable: {path}"
        ) from error
    requested = set(requested_names)
    result: dict[str, Any] = {}
    functions: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
            continue
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        else:
            targets = [node.target]
            value_node = node.value
        if value_node is None:
            continue
        names = [
            target.id
            for target in targets
            if isinstance(target, ast.Name) and target.id in requested
        ]
        if not names:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError):
            continue
        for name in names:
            result[name] = value
    return result, functions


def validate_upstream_bindings() -> UpstreamBinding:
    runner_path = repo_path(RUNNER_PREREGISTRATION_REL)
    branch_path = repo_path(BRANCH_PREREGISTRATION_REL)
    runner = read_json_file(
        runner_path,
        EXPECTED_RUNNER_PREREGISTRATION_SHA256,
        "runner preregistration",
    )
    branch = read_json_file(
        branch_path,
        EXPECTED_BRANCH_PREREGISTRATION_SHA256,
        "branch preregistration",
    )

    if runner.get("schema_version") != "ptcg-gold-ppo-ab-preregistration-v1":
        raise ProtocolBuildError("runner preregistration schema mismatch")
    if runner.get("status") != "preregistered_not_started":
        raise ProtocolBuildError("runner preregistration is not unstarted")
    plan = runner.get("plan")
    if not isinstance(plan, dict):
        raise ProtocolBuildError("runner preregistration.plan must be an object")
    if canonical_json_sha256(plan) != EXPECTED_RUNNER_PLAN_SHA256:
        raise ProtocolBuildError("runner plan canonical SHA-256 mismatch")
    if runner.get("plan_sha256") != EXPECTED_RUNNER_PLAN_SHA256:
        raise ProtocolBuildError("runner plan_sha256 field mismatch")

    if branch.get("schema_version") != "ptcg-ppo-gold-branch-preregistration-v1":
        raise ProtocolBuildError("branch preregistration schema mismatch")
    if branch.get("status") != "preregistered_not_started":
        raise ProtocolBuildError("branch preregistration is not unstarted")
    binding = branch.get("binding")
    if not isinstance(binding, dict):
        raise ProtocolBuildError("branch binding must be an object")
    if canonical_json_sha256(binding) != EXPECTED_BRANCH_BINDING_SHA256:
        raise ProtocolBuildError("branch binding canonical SHA-256 mismatch")
    if branch.get("binding_sha256") != EXPECTED_BRANCH_BINDING_SHA256:
        raise ProtocolBuildError("branch binding_sha256 field mismatch")
    command = binding.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(token, str) and token for token in command)
    ):
        raise ProtocolBuildError("branch command must be a non-empty string list")
    command_tuple = tuple(command)
    if canonical_json_sha256(command) != EXPECTED_BRANCH_COMMAND_SHA256:
        raise ProtocolBuildError("branch command canonical SHA-256 mismatch")
    if binding.get("command_sha256") != EXPECTED_BRANCH_COMMAND_SHA256:
        raise ProtocolBuildError("branch command_sha256 field mismatch")

    if binding.get("source_training_preregistration") != str(runner_path):
        raise ProtocolBuildError("branch points at a different runner preregistration")
    if (
        binding.get("source_training_preregistration_sha256")
        != EXPECTED_RUNNER_PREREGISTRATION_SHA256
    ):
        raise ProtocolBuildError("branch runner preregistration SHA mismatch")
    if binding.get("source_plan_sha256") != EXPECTED_RUNNER_PLAN_SHA256:
        raise ProtocolBuildError("branch source plan SHA mismatch")
    if binding.get("branch") != "B_gold_league":
        raise ProtocolBuildError("only B_gold_league is authorized")
    if binding.get("seed") != TRAINING_SEED:
        raise ProtocolBuildError("branch training seed mismatch")

    trainer_path = require_file_hash(
        TRAINER_REL,
        EXPECTED_TRAINER_SHA256,
        "trainer",
    )
    if Path(command_tuple[1]) != trainer_path:
        raise ProtocolBuildError("branch command trainer path mismatch")
    if binding.get("source_train_script") != str(trainer_path):
        raise ProtocolBuildError("branch source trainer path mismatch")
    if binding.get("source_train_script_sha256") != EXPECTED_TRAINER_SHA256:
        raise ProtocolBuildError("branch source trainer SHA mismatch")

    full_incumbent_path = require_file_hash(
        FULL_INCUMBENT_U456_REL,
        EXPECTED_FULL_INCUMBENT_U456_SHA256,
        "full incumbent U456",
    )
    require_file_hash(
        BC28_CHECKPOINT_REL,
        EXPECTED_BC28_CHECKPOINT_SHA256,
        "BC28 checkpoint",
    )
    require_file_hash(
        OLD_RETENTION_DATA_REL,
        EXPECTED_OLD_RETENTION_DATA_SHA256,
        "old-retention archive",
    )
    require_file_hash(
        VALID29_DATA_REL,
        EXPECTED_VALID29_DATA_SHA256,
        "valid29 archive",
    )

    expected_values = {
        "--updates": "464",
        "--schedule-start-update": "457",
        "--games-per-update": "64",
        "--ppo-epochs": "2",
        "--minibatch-size": "384",
        "--learning-rate": "3.6e-05",
        "--value-learning-rate": "7.5e-06",
        "--value-coefficient": "0.25",
        "--entropy-coefficient": "0.001",
        "--bc-kl-start": "0.012",
        "--bc-kl-end": "0.012",
        "--actor-reduction": "episode_mean",
        "--resume": str(full_incumbent_path),
        "--resume-learner-weights": "resume",
        "--seed": str(TRAINING_SEED),
        "--bc-replay-steps": "2",
        "--bc-replay-lr-scale": "0.05",
        "--bc-replay-loss": "ordered",
        "--bc-replay-context34-rows-per-batch": "4",
        "--device": "cuda",
    }
    for flag, expected in expected_values.items():
        require_command_value(command_tuple, flag, expected)
    require_command_switch(command_tuple, "--reset-optimizer-on-resume")
    require_command_switch(command_tuple, "--reset-opponent-quota-on-resume")
    require_command_switch(command_tuple, "--skip-initial-eval")

    output_dir = Path(str(binding.get("output_dir")))
    if not output_dir.is_absolute():
        raise ProtocolBuildError("branch output directory must be absolute")
    require_command_value(command_tuple, "--output-dir", str(output_dir))
    expected_output = repo_path(
        Path(
            "artifacts/"
            f"{CANDIDATE}/B_gold_league/seed-{TRAINING_SEED}"
        )
    )
    if output_dir != expected_output:
        raise ProtocolBuildError("branch output directory mismatch")
    terminal_checkpoint = output_dir / "checkpoints" / "update-0464.pt"

    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ProtocolBuildError("branch artifacts binding must be an object")
    log = Path(str(artifacts.get("log")))
    if not log.is_absolute():
        raise ProtocolBuildError("branch log path must be absolute")
    safety = binding.get("safety")
    if not isinstance(safety, dict):
        raise ProtocolBuildError("branch safety must be an object")
    for forbidden in (
        "network_calls",
        "packaging",
        "submission",
        "uploads",
        "shell",
        "a_branch_execution",
    ):
        if safety.get(forbidden) is not False:
            raise ProtocolBuildError(f"branch safety does not prohibit {forbidden}")
    if safety.get("exactly_one_child_command") is not True:
        raise ProtocolBuildError("branch does not bind exactly one child command")

    plan_branches = plan.get("branches")
    if not isinstance(plan_branches, dict):
        raise ProtocolBuildError("runner plan branches must be an object")
    branch_plan = plan_branches.get("B_gold_league")
    if not isinstance(branch_plan, dict):
        raise ProtocolBuildError("runner plan has no B_gold_league")
    runs = branch_plan.get("runs")
    if (
        not isinstance(runs, list)
        or len(runs) != 1
        or not isinstance(runs[0], dict)
        or runs[0].get("command") != command
    ):
        raise ProtocolBuildError("branch command differs from runner plan")
    protocol = plan.get("protocol")
    if not isinstance(protocol, dict):
        raise ProtocolBuildError("runner plan protocol must be an object")
    protocol_checks = {
        "minibatch_size": MINIBATCH_SIZE,
        "value_coefficient": 0.25,
        "entropy_coefficient": 0.001,
        "bc_kl_coefficient": 0.012,
        "resume_learner_weights": "resume",
        "reset_optimizer_on_resume": True,
        "reset_opponent_quota_on_resume": True,
    }
    for key, expected in protocol_checks.items():
        if protocol.get(key) != expected:
            raise ProtocolBuildError(f"runner plan protocol mismatch for {key}")
    if plan.get("inputs", {}).get("resume_checkpoint_sha256") != (
        EXPECTED_FULL_INCUMBENT_U456_SHA256
    ):
        raise ProtocolBuildError("runner plan does not bind full incumbent U456")

    return UpstreamBinding(
        runner_preregistration=runner,
        branch_preregistration=branch,
        command=command_tuple,
        output_dir=output_dir,
        log=log,
        terminal_checkpoint=terminal_checkpoint,
    )


def load_prior_vcoef_comprehensive() -> dict[str, Any]:
    prior = read_json_file(
        repo_path(PRIOR_VCOEF_COMPREHENSIVE_REL),
        EXPECTED_PRIOR_VCOEF_SHA256,
        "prior vcoef comprehensive preregistration",
    )
    required = {
        "behavior_protocol",
        "gold19_screen128_protocol",
        "gold19_confirm512_protocol_if_screen_passes",
        "training_integrity_gates",
    }
    missing = required - set(prior)
    if missing:
        raise ProtocolBuildError(
            f"prior vcoef comprehensive is missing {sorted(missing)}"
        )
    return prior


def load_behavior_runner_contract(
    expected_sha256: str | None,
) -> BehaviorRunnerContract:
    path = repo_path(BEHAVIOR_RUNNER_REL)
    observed = file_sha256(path)
    if expected_sha256 is not None:
        expected_sha256 = require_sha256(
            expected_sha256,
            "expected final behavior-runner SHA-256",
        )
        if observed != expected_sha256:
            raise ProtocolBuildError(
                "behavior-runner SHA-256 differs from caller-supplied final hash"
            )
    requested = {
        "EXECUTION_PREREGISTRATION_SCHEMA",
        "FROZEN_DEVICE",
        "FROZEN_BATCH_SIZE",
        "FROZEN_REQUESTED_WORKERS",
        "FROZEN_EFFECTIVE_WORKERS",
        "FROZEN_PREDICTION_ORDER",
        "FROZEN_OUTPUT_PREDICTION_ORDER",
        "FROZEN_PROGRESS_INTERVAL",
        "FROZEN_SPLIT",
        "FROZEN_SPLIT_MODE",
        "FROZEN_SPLIT_SEED",
        "BEHAVIOR_GATE_OUTPUT_KEYS",
    }
    constants, functions = _literal_module_constants(path, requested)
    missing = requested - set(constants)
    if missing:
        raise ProtocolBuildError(
            "behavior runner does not expose a stable literal schema: "
            f"{sorted(missing)}"
        )
    for function in (
        "expected_gate_mapping",
        "validate_source_behavior_protocol",
        "parse_execution_spec",
    ):
        if function not in functions:
            raise ProtocolBuildError(
                f"behavior runner is missing schema validator {function}"
            )
    frozen_expected = {
        "FROZEN_DEVICE": "cuda",
        "FROZEN_BATCH_SIZE": 256,
        "FROZEN_REQUESTED_WORKERS": 8,
        "FROZEN_EFFECTIVE_WORKERS": 0,
        "FROZEN_PREDICTION_ORDER": "policy",
        "FROZEN_OUTPUT_PREDICTION_ORDER": "policy_greedy",
        "FROZEN_PROGRESS_INTERVAL": 0,
        "FROZEN_SPLIT": "valid",
        "FROZEN_SPLIT_MODE": "archive",
        "FROZEN_SPLIT_SEED": 20260723,
        "BEHAVIOR_GATE_OUTPUT_KEYS": EXPECTED_BEHAVIOR_GATE_OUTPUT_KEYS,
    }
    for name, expected in frozen_expected.items():
        if constants.get(name) != expected:
            raise ProtocolBuildError(
                f"behavior runner frozen schema changed for {name}"
            )
    return BehaviorRunnerContract(
        path=path,
        sha256=observed,
        constants=constants,
    )


def load_terminal_receipt_writer_contract(
    expected_sha256: str | None,
) -> ReceiptWriterContract:
    """Require the caller and local bytes to match the frozen final writer."""

    if expected_sha256 is None:
        raise ProtocolBuildError(
            "expected final terminal-receipt-writer SHA-256 is required"
        )
    expected_sha256 = require_sha256(
        expected_sha256,
        "expected final terminal-receipt-writer SHA-256",
    )
    if expected_sha256 != EXPECTED_TERMINAL_RECEIPT_WRITER_SHA256:
        raise ProtocolBuildError(
            "caller-supplied terminal-receipt-writer SHA-256 does not equal "
            "the frozen final writer hash"
        )
    path = repo_path(TERMINAL_RECEIPT_WRITER_REL)
    if not path.is_file() or path.is_symlink():
        raise ProtocolBuildError(
            "terminal receipt writer must be a non-symlink regular file"
        )
    observed = file_sha256(path)
    if observed != EXPECTED_TERMINAL_RECEIPT_WRITER_SHA256:
        raise ProtocolBuildError(
            "terminal-receipt-writer bytes differ from the frozen final hash"
        )
    return ReceiptWriterContract(path=path, sha256=observed)


def load_launcher_contract(expected_sha256: str | None) -> LauncherContract:
    import importlib.util

    path = repo_path(TRAINING_LAUNCHER_REL)
    observed = file_sha256(path)
    if expected_sha256 is not None:
        expected_sha256 = require_sha256(
            expected_sha256,
            "expected final training-launcher SHA-256",
        )
        if observed != expected_sha256:
            raise ProtocolBuildError(
                "training-launcher SHA-256 differs from caller-supplied final hash"
            )
    module_name = "_ptcg_mb384_launcher_contract"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ProtocolBuildError("could not load training launcher schema")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        raise ProtocolBuildError(
            "training launcher is not import-stable"
        ) from error
    finally:
        sys.modules.pop(module_name, None)
    requested = (
        "SCHEMA_VERSION",
        "LOCKED_STATUS",
        "TRANSPORT_LOCK_KEYS",
        "REQUIRED_BINDINGS",
        "EXPECTED_TRANSPORT_CONSTANTS",
        "EXPECTED_TRANSPORT_KEYS",
        "LOCKED_ABSENCE_KEYS",
        "CHILD_COMMAND_KEYS",
        "ATTEMPT_KEYS",
        "EFFECTIVE_TRAINER_FD",
        "EFFECTIVE_TRAINER_PATH",
        "SEALED_TRAINER_PAYLOAD_FD",
        "SEALED_TRAINER_PAYLOAD_PATH",
        "EFFECTIVE_ROOT_FD",
        "EFFECTIVE_ROOT_PATH",
        "build_trainer_bootstrap",
        "build_effective_command",
        "canonical_attempt_start_marker",
        "canonical_json_sha256",
        "read_and_validate_transport_lock",
    )
    missing = {name for name in requested if not hasattr(module, name)}
    if missing:
        raise ProtocolBuildError(
            "training launcher does not yet expose its final schema/helpers: "
            f"{sorted(missing)}"
        )
    child_keys = frozenset(module.CHILD_COMMAND_KEYS)
    attempt_keys = frozenset(module.ATTEMPT_KEYS)
    transport_keys = frozenset(module.EXPECTED_TRANSPORT_KEYS)
    if child_keys != SUPPORTED_SEALED_CHILD_COMMAND_KEYS:
        raise ProtocolBuildError(
            "launcher is not the audited sealed-trainer child-command schema"
        )
    if attempt_keys != SUPPORTED_ATTEMPT_KEYS:
        raise ProtocolBuildError(
            "launcher attempt schema changed; builder update required"
        )
    if transport_keys != SUPPORTED_EXPECTED_TRANSPORT_KEYS:
        raise ProtocolBuildError(
            "launcher expected-transport schema changed; builder update required"
        )
    expected_transport_constants = dict(
        module.EXPECTED_TRANSPORT_CONSTANTS
    )
    if set(expected_transport_constants) != transport_keys - {"workdir"}:
        raise ProtocolBuildError(
            "launcher expected-transport constants do not match its key schema"
        )
    return LauncherContract(
        path=path,
        sha256=observed,
        schema_version=str(module.SCHEMA_VERSION),
        locked_status=str(module.LOCKED_STATUS),
        transport_lock_keys=frozenset(module.TRANSPORT_LOCK_KEYS),
        required_bindings=frozenset(module.REQUIRED_BINDINGS),
        expected_transport_constants=expected_transport_constants,
        expected_transport_keys=transport_keys,
        locked_absence_keys=frozenset(module.LOCKED_ABSENCE_KEYS),
        child_command_keys=child_keys,
        attempt_keys=attempt_keys,
        effective_trainer_fd=int(module.EFFECTIVE_TRAINER_FD),
        effective_trainer_path=str(module.EFFECTIVE_TRAINER_PATH),
        sealed_trainer_payload_fd=int(module.SEALED_TRAINER_PAYLOAD_FD),
        sealed_trainer_payload_path=str(module.SEALED_TRAINER_PAYLOAD_PATH),
        effective_root_fd=int(module.EFFECTIVE_ROOT_FD),
        effective_root_path=str(module.EFFECTIVE_ROOT_PATH),
        build_trainer_bootstrap=module.build_trainer_bootstrap,
        build_effective_command=module.build_effective_command,
        canonical_attempt_start_marker=module.canonical_attempt_start_marker,
        launcher_canonical_json_sha256=module.canonical_json_sha256,
    )


def behavior_dependencies() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, relative in BEHAVIOR_DEPENDENCY_RELS.items():
        path = repo_path(relative)
        if not path.is_file() or path.is_symlink():
            raise ProtocolBuildError(
                f"behavior dependency must be a non-symlink file: {relative}"
            )
        result[name] = {
            "path": repo_relative_text(path),
            "sha256": file_sha256(path),
        }
    return result


def behavior_panels_from_prior(
    prior: Mapping[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    prior_behavior = prior.get("behavior_protocol")
    if not isinstance(prior_behavior, dict):
        raise ProtocolBuildError("prior behavior protocol must be an object")
    raw_panels = prior_behavior.get("ordered_panels")
    if not isinstance(raw_panels, list) or len(raw_panels) != 2:
        raise ProtocolBuildError("prior behavior protocol must have two panels")
    data_panels = ("old_retention", "valid29")
    expected_data = (
        (OLD_RETENTION_DATA_REL, EXPECTED_OLD_RETENTION_DATA_SHA256),
        (VALID29_DATA_REL, EXPECTED_VALID29_DATA_SHA256),
    )
    output_names = (
        "behavior_old_retention_policy.json",
        "behavior_valid29_policy.json",
    )
    result: list[dict[str, Any]] = []
    for order, (raw, data_panel, data_binding, output_name) in enumerate(
        zip(raw_panels, data_panels, expected_data, output_names, strict=True),
        start=1,
    ):
        if not isinstance(raw, dict):
            raise ProtocolBuildError("prior behavior panel must be an object")
        data_rel, data_sha256 = data_binding
        minima = raw.get("minimum_correct")
        if not isinstance(minima, dict):
            raise ProtocolBuildError("prior behavior minima must be an object")
        panel = {
            "order": order,
            "name": f"mb384_u464_{data_panel}_policy",
            "data_panel": data_panel,
            "data": str(data_rel),
            "data_sha256": data_sha256,
            "split": "valid",
            "split_mode": "archive",
            "split_seed": 20260723,
            "output": repo_relative_text(output_dir / output_name),
            "rows_exact": raw.get("rows_exact"),
            "context34_rows_exact": raw.get("context34_rows_exact"),
            "minimum_correct": copy.deepcopy(minima),
        }
        result.append(panel)
    return result


def expected_gate_mapping(source_panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "metrics_object": "output.metrics",
        "context34_object": 'output.metrics.by_context["34"]',
    }
    for panel in source_panels:
        name = str(panel["data_panel"])
        minima = panel["minimum_correct"]
        values = {
            "rows": panel["rows_exact"],
            "set_exact": minima["set_exact"],
            "hybrid_order_exact": minima["hybrid_order_exact"],
            "ordered_exact": minima["ordered_exact"],
            "value": minima["value"],
            "count": minima["count"],
            "top1": minima["top1"],
            "context34_rows": panel["context34_rows_exact"],
            "context34_hybrid_order_exact": minima[
                "context34_hybrid_order_exact"
            ],
            "context34_ordered_exact": minima["context34_ordered_exact"],
        }
        panel_mapping: dict[str, Any] = {}
        for gate_name, value in values.items():
            if type(value) is not int or value < 0:
                raise ProtocolBuildError(
                    f"behavior threshold must be non-negative int: "
                    f"{name}.{gate_name}"
                )
            output_key, comparison = EXPECTED_BEHAVIOR_GATE_OUTPUT_KEYS[
                gate_name
            ]
            panel_mapping[gate_name] = {
                "output_key": output_key,
                comparison: value,
            }
        mapping[name] = panel_mapping
    mapping["all_required"] = True
    return mapping


def build_behavior_protocol(
    prior: Mapping[str, Any],
    runner: BehaviorRunnerContract,
    output_dir: Path,
) -> dict[str, Any]:
    constants = runner.constants
    evaluator = repo_path(BEHAVIOR_EVALUATOR_REL)
    panels = behavior_panels_from_prior(prior, output_dir)
    implementation = {
        "python": str(Path(sys.executable).resolve()),
        "inprocess_runner": repo_relative_text(runner.path),
        "inprocess_runner_sha256": runner.sha256,
        "metric_evaluator": repo_relative_text(evaluator),
        "metric_evaluator_sha256": file_sha256(evaluator),
        "local_dependencies": behavior_dependencies(),
        "device": constants["FROZEN_DEVICE"],
        "batch_size": constants["FROZEN_BATCH_SIZE"],
        "workers": constants["FROZEN_REQUESTED_WORKERS"],
        "effective_workers": constants["FROZEN_EFFECTIVE_WORKERS"],
        "prediction_order_cli_argument": constants[
            "FROZEN_PREDICTION_ORDER"
        ],
        "expected_output_prediction_order": constants[
            "FROZEN_OUTPUT_PREDICTION_ORDER"
        ],
        "compact": True,
        "progress_interval": constants["FROZEN_PROGRESS_INTERVAL"],
        "same_process_two_panel_runner_required": True,
        "runner_path_and_sha256_bind_after_implementation_before_any_behavior_data_access": True,
    }
    return {
        "prerequisite": (
            "The one authorized training attempt must pass every frozen "
            "content-integrity and transport-receipt gate under the root "
            "audit and at least two independent read-only recomputations. "
            "A separate checkpoint-SHA-bound behavior execution "
            "preregistration is then required."
        ),
        "implementation": implementation,
        "cuda_preflight_and_formal_attempt_boundary": {
            "same_pid_and_cuda_context_for_preflight_and_both_panels": True,
            "preflight_requires_cuda_init_tensor_kernel_synchronize_and_exact_checkpoint_model_load": True,
            "preflight_must_not_open_behavior_archives_or_process_behavior_rows": True,
            "preflight_certificate_o_excl_and_fsync": True,
            "formal_attempt_marker_o_excl_and_fsync_after_certificate": True,
            "model_loaded_once_and_reused_for_both_panels": True,
            "no_exec_shell_or_subprocess_after_preflight": True,
        },
        "ordered_panels": panels,
        "metric_mapping_and_gates": expected_gate_mapping(panels),
        "rules": {
            "all_20_gates_required": True,
            "run_valid29_even_if_old_retention_gate_fails": True,
            "run_panels_once_in_declared_order_after_formal_marker": True,
            "no_alternate_decode": True,
            "no_retry": True,
            "failure_action": (
                "Reject mb384 U464, retain the full incumbent U456, close "
                "this candidate, and do not run Gold19."
            ),
        },
    }


def build_training_integrity_gates(
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    raw = prior.get("training_integrity_gates")
    if not isinstance(raw, dict):
        raise ProtocolBuildError("prior training-integrity gates are missing")
    gates = copy.deepcopy(raw)
    gates.update(
        {
            "direct_exec_terminal_receipt_status_completed_and_return_code_zero": True,
            "transport_lock_and_effective_command_hashes_match": True,
            "updates_exact": list(UPDATES),
            "minibatch_size_exact": MINIBATCH_SIZE,
            "optimizer_steps_each_update": (
                "2 * ceil(optimization transitions / 384)"
            ),
            "ppo_epochs_completed_each_update_exact": PPO_EPOCHS,
            "actor_learning_rate_each_update_exact": 0.000036,
            "value_learning_rate_each_update_exact": 0.0000075,
            "value_coefficient_each_update_exact": 0.25,
            "entropy_coefficient_each_update_exact": 0.001,
            "bc_kl_coefficient_each_update_exact": 0.012,
            "resume_checkpoint_update_exact": 456,
            "resume_checkpoint_sha256_exact": (
                EXPECTED_FULL_INCUMBENT_U456_SHA256
            ),
            "resume_learner_weights_exact": "resume",
            "optimizer_state_loaded_exact": False,
            "bc_replay_optimizer_state_loaded_exact": False,
            "opponent_quota_state_loaded_exact": False,
            "optimizer_reset_reason_exact": "explicit_reset_on_resume",
            "terminal_checkpoint_update_exact": 464,
            "checkpoint_directory_files_exact": ["update-0464.pt"],
            "best_last_and_update0464_sha256_equal": True,
        }
    )
    gates.pop("wrapper_status_completed_and_return_code_zero", None)
    gates.pop("launch_error_null", None)
    gates.pop("best_last_and_update0456_sha256_equal", None)
    return gates


def _gold19_protocols(
    prior: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    screen_raw = prior.get("gold19_screen128_protocol")
    confirm_raw = prior.get("gold19_confirm512_protocol_if_screen_passes")
    if not isinstance(screen_raw, dict) or not isinstance(confirm_raw, dict):
        raise ProtocolBuildError("prior Gold19 protocols are missing")
    screen = copy.deepcopy(screen_raw)
    confirm = copy.deepcopy(confirm_raw)
    screen["prerequisite"] = (
        "Both frozen behavior panels pass all 20 gates. A separate "
        "checkpoint-, command-, seed-, and empty-output-bound screen "
        "execution preregistration is required."
    )
    screen["ordered_arms"] = [
        {
            "order": 1,
            "name": "fresh_full_incumbent_u456_control",
            "seed": SCREEN_CONTROL_SEED,
            "output_dir": (
                "artifacts/gold_h2h_full_incumbent_u456_"
                f"seed{SCREEN_CONTROL_SEED}_gold19_screen128_20260731"
            ),
        },
        {
            "order": 2,
            "name": "mb384_continuation_u464_candidate",
            "seed": SCREEN_CANDIDATE_SEED,
            "output_dir": (
                "artifacts/gold_h2h_mb384_continuation_u464_"
                f"seed{SCREEN_CANDIDATE_SEED}_gold19_screen128_20260731"
            ),
        },
    ]
    screen_rules = screen.get("rules")
    if not isinstance(screen_rules, dict):
        raise ProtocolBuildError("prior Gold19 screen rules are missing")
    screen_rules["screen_failure_action"] = (
        "Retain full incumbent U456 and close mb384 U464."
    )

    confirm["prerequisite"] = (
        "Every fresh screen gate passes; all screen samples remain excluded "
        "from confirmation estimates and decisions."
    )
    confirm["ordered_arms"] = [
        {
            "order": 1,
            "name": "fresh_full_incumbent_u456_control",
            "seed": CONFIRM_CONTROL_SEED,
            "output_dir": (
                "artifacts/gold_h2h_full_incumbent_u456_"
                f"seed{CONFIRM_CONTROL_SEED}_gold19_confirm512_20260731"
            ),
        },
        {
            "order": 2,
            "name": "mb384_continuation_u464_candidate",
            "seed": CONFIRM_CANDIDATE_SEED,
            "output_dir": (
                "artifacts/gold_h2h_mb384_continuation_u464_"
                f"seed{CONFIRM_CANDIDATE_SEED}_gold19_confirm512_20260731"
            ),
        },
    ]
    confirm_rules = confirm.get("rules")
    if not isinstance(confirm_rules, dict):
        raise ProtocolBuildError("prior Gold19 confirmation rules are missing")
    confirm_rules["failure_action"] = (
        "Retain full incumbent U456 and close mb384 U464."
    )
    confirm_rules["pass_action"] = (
        "mb384 U464 becomes only the Gold19 local-development incumbent; "
        "it remains neither blind-confirmed, real-Kaggle-gold verified, "
        "packaged, uploaded, nor submit-ready."
    )
    validate_gold19_gate_reuse(prior, screen, confirm)
    return screen, confirm


def validate_gold19_gate_reuse(
    prior: Mapping[str, Any],
    screen: Mapping[str, Any],
    confirm: Mapping[str, Any],
) -> None:
    prior_screen = prior["gold19_screen128_protocol"]
    prior_confirm = prior["gold19_confirm512_protocol_if_screen_passes"]
    screen_immutable = (
        "inputs",
        "implementation",
        "integrity_each_arm",
        "candidate_absolute_minimum_wins",
        "candidate_minus_fresh_control_minimum_wins",
        "own_dynamic_bottom_five_definition",
        "fixed_hard_five_policy_ids",
    )
    confirm_immutable = (
        "implementation",
        "integrity_each_arm",
        "candidate_absolute_minimum_wins",
        "candidate_minus_fresh_control_required",
        "confidence_interval_method",
    )
    for key in screen_immutable:
        if screen.get(key) != prior_screen.get(key):
            raise ProtocolBuildError(f"Gold19 screen gate drift for {key}")
    for key in confirm_immutable:
        if confirm.get(key) != prior_confirm.get(key):
            raise ProtocolBuildError(f"Gold19 confirm gate drift for {key}")
    prior_screen_rules = dict(prior_screen["rules"])
    screen_rules = dict(screen["rules"])
    prior_screen_rules.pop("screen_failure_action", None)
    screen_rules.pop("screen_failure_action", None)
    if screen_rules != prior_screen_rules:
        raise ProtocolBuildError("Gold19 screen rule drift")
    prior_confirm_rules = dict(prior_confirm["rules"])
    confirm_rules = dict(confirm["rules"])
    for key in ("failure_action", "pass_action"):
        prior_confirm_rules.pop(key, None)
        confirm_rules.pop(key, None)
    if confirm_rules != prior_confirm_rules:
        raise ProtocolBuildError("Gold19 confirmation rule drift")
    seeds = tuple(
        arm["seed"]
        for protocol in (screen, confirm)
        for arm in protocol["ordered_arms"]
    )
    if seeds != FRESH_GOLD19_SEEDS or len(set(seeds)) != 4:
        raise ProtocolBuildError("Gold19 seeds are not the frozen fresh sequence")


def build_source_protocol(
    created_at_utc: str,
    upstream: UpstreamBinding,
    prior: Mapping[str, Any],
    behavior_runner: BehaviorRunnerContract,
    launcher: LauncherContract,
    receipt_writer: ReceiptWriterContract,
) -> dict[str, Any]:
    behavior = build_behavior_protocol(
        prior,
        behavior_runner,
        upstream.output_dir,
    )
    return {
        "schema_version": "ptcg-ppo-mb384-source-protocol-v1",
        "created_at_utc": created_at_utc,
        "status": "locked_before_training",
        "objective": (
            "Run one no-retry PPO continuation from the full incumbent U456 "
            "through U464 with the selected optimization configuration knob "
            "minibatch_size 512 to 384."
        ),
        "interpretation": {
            "single_selected_optimization_configuration_knob": True,
            "selected_knob": {
                "minibatch_size": {
                    "reference": REFERENCE_MINIBATCH_SIZE,
                    "candidate": MINIBATCH_SIZE,
                }
            },
            "strict_causal_ablation": False,
            "continued_training_confound": True,
            "continued_training_confound_detail": (
                "The candidate receives eight additional PPO updates "
                "U457-U464 after the U456 control. Without a matched U464 "
                "minibatch-512 arm, any difference mixes continued-training "
                "exposure with the selected minibatch configuration."
            ),
            "fresh_training_seed_is_additional_nuisance": True,
            "infrastructure_bytes_may_differ_from_historical_incumbent": True,
            "allowed_claim": (
                "Prospective performance of this exact hash-bound mb384 "
                "continuation, not a strict causal estimate of minibatch size."
            ),
            "development_proxy_only": True,
            "blind_evidence": False,
            "real_kaggle_gold_evidence": False,
        },
        "infrastructure": {
            "behavior_runner": {
                "path": str(BEHAVIOR_RUNNER_REL),
                "sha256": behavior_runner.sha256,
            },
            "training_launcher": {
                "path": str(TRAINING_LAUNCHER_REL),
                "sha256": launcher.sha256,
                "schema_version": launcher.schema_version,
            },
            "terminal_receipt_writer": {
                "path": str(TERMINAL_RECEIPT_WRITER_REL),
                "sha256": receipt_writer.sha256,
            },
        },
        "upstream_preregistration_binding": {
            "runner_preregistration": {
                "path": str(RUNNER_PREREGISTRATION_REL),
                "sha256": EXPECTED_RUNNER_PREREGISTRATION_SHA256,
                "plan_sha256": EXPECTED_RUNNER_PLAN_SHA256,
            },
            "branch_preregistration": {
                "path": str(BRANCH_PREREGISTRATION_REL),
                "sha256": EXPECTED_BRANCH_PREREGISTRATION_SHA256,
                "binding_sha256": EXPECTED_BRANCH_BINDING_SHA256,
                "command_sha256": EXPECTED_BRANCH_COMMAND_SHA256,
            },
        },
        "training_transport_binding": {
            "launcher": {
                "path": str(TRAINING_LAUNCHER_REL),
                "sha256": launcher.sha256,
                "schema_version": launcher.schema_version,
            },
            "terminal_receipt_writer": {
                "path": str(TERMINAL_RECEIPT_WRITER_REL),
                "sha256": receipt_writer.sha256,
            },
            "sealed_bootstrap_fd": launcher.effective_trainer_fd,
            "sealed_bootstrap_path": launcher.effective_trainer_path,
            "sealed_trainer_payload_fd": (
                launcher.sealed_trainer_payload_fd
            ),
            "sealed_trainer_payload_path": (
                launcher.sealed_trainer_payload_path
            ),
            "sealed_bootstrap_sha256": launcher_bootstrap_sha256(launcher),
            "both_memfds_require_irreversible_write_shrink_grow_and_seal_seals": True,
            "topology": launcher.expected_transport_constants.get("topology"),
            "anonymous_log_before_publish": True,
            "durable_attempt_start_marker_required": True,
            "unpublished_log_witness_required_on_publish_failure": True,
            "transport_lock_binds_this_source_protocol": True,
        },
        "candidate_binding": {
            "name": "mb384_continuation_u464",
            "output_dir": repo_relative_text(upstream.output_dir),
            "terminal_checkpoint": repo_relative_text(
                upstream.terminal_checkpoint
            ),
            "terminal_update": 464,
            "terminal_checkpoint_sha256": (
                "bind_after_training_integrity_before_behavior"
            ),
            "terminal_checkpoint_only": True,
        },
        "single_selected_knob_command_audit": command_single_knob_evidence(
            upstream.command
        ),
        "fixed_training": {
            "resume_checkpoint": {
                "path": str(FULL_INCUMBENT_U456_REL),
                "sha256": EXPECTED_FULL_INCUMBENT_U456_SHA256,
                "update": 456,
                "usage": "global update metadata and full learner weights",
            },
            "resume_learner_weights": "resume",
            "optimizer_state_loaded": False,
            "bc_replay_optimizer_state_loaded": False,
            "opponent_quota_state_loaded": False,
            "reset_optimizer_on_resume": True,
            "reset_bc_replay_optimizer_via_optimizer_reset": True,
            "reset_opponent_quota_on_resume": True,
            "updates": list(UPDATES),
            "games_per_update": GAMES_PER_UPDATE,
            "rollout_games": len(UPDATES) * GAMES_PER_UPDATE,
            "ppo_epochs": PPO_EPOCHS,
            "minibatch_size": MINIBATCH_SIZE,
            "seed": TRAINING_SEED,
            "loss": {
                "identity": (
                    "policy_loss + 0.25 * value_loss - "
                    "0.001 * entropy + 0.012 * BC_KL"
                ),
                "policy_coefficient": 1.0,
                "value_coefficient": 0.25,
                "entropy_coefficient_with_subtractive_sign": 0.001,
                "bc_kl_coefficient": 0.012,
            },
            "optimizer_steps_each_update": (
                "2 * ceil(optimization transitions / 384)"
            ),
            "optimizer_steps_function": {
                "ppo_epochs": PPO_EPOCHS,
                "minibatch_size": MINIBATCH_SIZE,
                "formula": "2*ceil(transitions/384)",
                "boundary_examples": {
                    "1": optimizer_steps_for_transitions(1),
                    "384": optimizer_steps_for_transitions(384),
                    "385": optimizer_steps_for_transitions(385),
                    "768": optimizer_steps_for_transitions(768),
                },
            },
            "fixed_opponent_quota_vector": command_opponent_quota_vector(
                upstream.command
            ),
            "strict_per_opponent_two_seat_balance": True,
            "all_other_training_tokens": (
                "exactly those in the frozen branch command "
                f"{EXPECTED_BRANCH_COMMAND_SHA256}"
            ),
        },
        "training_integrity_gates": build_training_integrity_gates(prior),
        "behavior_protocol": behavior,
        "execution_order_and_stop_rules": {
            "training_once_only": True,
            "no_retry_or_seed_selection": True,
            "no_threshold_change_after_lock": True,
            "no_behavior_if_training_integrity_fails": True,
            "no_gold19_if_any_behavior_gate_fails": True,
            "no_confirm512_if_screen_fails": True,
            "terminal_update0464_only": True,
            "forbidden_selection_signals": [
                "training rollout win rate",
                "terminal 64-game BC mirror",
                "best.pt",
                "partial behavior results",
                "partial Gold19 results",
                "prior Gold19 samples",
            ],
        },
        "prohibited_actions": {
            "blind_result_or_replay_access": True,
            "current_gold_result_or_replay_access": True,
            "day30_content_access": True,
            "day31_content_access": True,
            "prior_gold19_result_reuse_or_pooling": True,
            "network": True,
            "package": True,
            "upload": True,
            "submission": True,
        },
    }


def build_comprehensive_preregistration(
    created_at_utc: str,
    source_protocol: Mapping[str, Any],
    source_sha256: str,
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    screen, confirm = _gold19_protocols(prior)
    behavior = copy.deepcopy(source_protocol["behavior_protocol"])
    return {
        "schema_version": "ptcg-ppo-mb384-comprehensive-preregistration-v1",
        "created_at_utc": created_at_utc,
        "status": "locked_before_training",
        "objective": source_protocol["objective"],
        "source_protocol": {
            "path": str(SOURCE_PROTOCOL_REL),
            "sha256": source_sha256,
        },
        "infrastructure": copy.deepcopy(
            source_protocol["infrastructure"]
        ),
        "prior_gate_source": {
            "path": str(PRIOR_VCOEF_COMPREHENSIVE_REL),
            "sha256": EXPECTED_PRIOR_VCOEF_SHA256,
            "reuse": (
                "All behavior thresholds and Gold19 screen/confirm "
                "integrity, absolute, relative, tail, seat, and confidence "
                "gates are copied prospectively without threshold changes."
            ),
            "behavior_protocol_canonical_sha256": canonical_json_sha256(
                prior["behavior_protocol"]
            ),
            "gold19_screen128_protocol_canonical_sha256": (
                canonical_json_sha256(prior["gold19_screen128_protocol"])
            ),
            "gold19_confirm512_protocol_canonical_sha256": (
                canonical_json_sha256(
                    prior["gold19_confirm512_protocol_if_screen_passes"]
                )
            ),
        },
        "candidate_binding": copy.deepcopy(
            source_protocol["candidate_binding"]
        ),
        "upstream_preregistration_binding": copy.deepcopy(
            source_protocol["upstream_preregistration_binding"]
        ),
        "training_transport_binding": copy.deepcopy(
            source_protocol["training_transport_binding"]
        ),
        "interpretation": copy.deepcopy(source_protocol["interpretation"]),
        "selected_optimization_knob": {
            "name": "minibatch_size",
            "reference": REFERENCE_MINIBATCH_SIZE,
            "candidate": MINIBATCH_SIZE,
            "all_other_branch_command_tokens_frozen": True,
            "strict_causal_attribution_prohibited": True,
            "reason": (
                "This is a continuation from U456 to U464 rather than a "
                "matched from-scratch or matched-U464 minibatch-512 arm."
            ),
        },
        "single_selected_knob_command_audit": copy.deepcopy(
            source_protocol["single_selected_knob_command_audit"]
        ),
        "fixed_training": copy.deepcopy(source_protocol["fixed_training"]),
        "training_integrity_gates": copy.deepcopy(
            source_protocol["training_integrity_gates"]
        ),
        "behavior_protocol": behavior,
        "gold19_screen128_protocol": screen,
        "gold19_confirm512_protocol_if_screen_passes": confirm,
        "fresh_gold19_seeds": list(FRESH_GOLD19_SEEDS),
        "execution_order_and_stop_rules": copy.deepcopy(
            source_protocol["execution_order_and_stop_rules"]
        ),
        "promotion_and_external_action_boundary": {
            "behavior_pass_does_not_promote": True,
            "screen_pass_does_not_promote": True,
            "confirm512_pass_promotes_only_local_development_incumbent": True,
            "confirm512_pass_does_not_authorize_blind_access": True,
            "confirm512_pass_does_not_authorize_package_upload_or_submission": True,
            "fresh_user_authorization_required_before_any_submission": True,
        },
        "prohibited_actions": copy.deepcopy(
            source_protocol["prohibited_actions"]
        ),
    }


def _effective_command(
    command: Sequence[str],
    launcher: LauncherContract,
) -> tuple[list[str], list[int]]:
    effective, indices = launcher.build_effective_command(
        command,
        root=REPO_ROOT,
    )
    if (
        not isinstance(effective, tuple)
        or not all(isinstance(token, str) for token in effective)
        or not isinstance(indices, tuple)
        or not all(type(index) is int for index in indices)
    ):
        raise ProtocolBuildError(
            "launcher build_effective_command returned an invalid shape"
        )
    return list(effective), list(indices)


def launcher_bootstrap_sha256(launcher: LauncherContract) -> str:
    bootstrap = launcher.build_trainer_bootstrap(
        canonical_trainer_path=repo_path(TRAINER_REL),
        trainer_sha256=EXPECTED_TRAINER_SHA256,
        payload_fd=launcher.sealed_trainer_payload_fd,
    )
    if not isinstance(bootstrap, bytes) or not bootstrap:
        raise ProtocolBuildError(
            "launcher build_trainer_bootstrap returned invalid bytes"
        )
    return raw_sha256(bootstrap)


def launcher_attempt_start_marker(
    launcher: LauncherContract,
    log: Path,
) -> Path:
    marker = launcher.canonical_attempt_start_marker(
        root=REPO_ROOT,
        log=log,
        seed=TRAINING_SEED,
    )
    if (
        not isinstance(marker, Path)
        or not marker.is_absolute()
        or marker.parent != REPO_ROOT
    ):
        raise ProtocolBuildError(
            "launcher canonical attempt marker helper returned invalid path"
        )
    return marker


def build_transport_lock(
    upstream: UpstreamBinding,
    launcher: LauncherContract,
    receipt_writer: ReceiptWriterContract,
    source_sha256: str,
    comprehensive_sha256: str,
) -> dict[str, Any]:
    binding_files = {
        "launcher": {
            "path": str(launcher.path),
            "sha256": launcher.sha256,
        },
        "source_protocol": {
            "path": str(repo_path(SOURCE_PROTOCOL_REL)),
            "sha256": source_sha256,
        },
        "comprehensive_preregistration": {
            "path": str(repo_path(COMPREHENSIVE_PREREGISTRATION_REL)),
            "sha256": comprehensive_sha256,
        },
        "runner_preregistration": {
            "path": str(repo_path(RUNNER_PREREGISTRATION_REL)),
            "sha256": EXPECTED_RUNNER_PREREGISTRATION_SHA256,
        },
        "branch_preregistration": {
            "path": str(repo_path(BRANCH_PREREGISTRATION_REL)),
            "sha256": EXPECTED_BRANCH_PREREGISTRATION_SHA256,
        },
        "trainer": {
            "path": str(repo_path(TRAINER_REL)),
            "sha256": EXPECTED_TRAINER_SHA256,
        },
        "terminal_receipt_writer": {
            "path": str(receipt_writer.path),
            "sha256": receipt_writer.sha256,
        },
    }
    missing_bindings = set(launcher.required_bindings) - set(binding_files)
    if missing_bindings:
        raise ProtocolBuildError(
            "builder mapping is missing launcher-required bindings: "
            f"{sorted(missing_bindings)}"
        )
    effective_command, effective_indices = _effective_command(
        upstream.command,
        launcher,
    )
    child_command = {
        "source_binding": "branch_preregistration",
        "json_path": "binding.command",
        "token_count": len(upstream.command),
        "canonical_sha256": EXPECTED_BRANCH_COMMAND_SHA256,
        "effective_trainer_fd": launcher.effective_trainer_fd,
        "effective_trainer_path": launcher.effective_trainer_path,
        "sealed_trainer_payload_fd": launcher.sealed_trainer_payload_fd,
        "sealed_trainer_payload_path": launcher.sealed_trainer_payload_path,
        "sealed_bootstrap_sha256": launcher_bootstrap_sha256(launcher),
        "effective_root_fd": launcher.effective_root_fd,
        "effective_root_path": launcher.effective_root_path,
        "effective_repo_path_token_indices": effective_indices,
        "effective_command_canonical_sha256": (
            launcher.launcher_canonical_json_sha256(effective_command)
        ),
    }
    if set(child_command) != set(launcher.child_command_keys):
        raise ProtocolBuildError("constructed child command does not match launcher")

    attempt = {
        "seed": TRAINING_SEED,
        "attempts_authorized": 1,
        "output_dir": str(upstream.output_dir),
        "log": str(upstream.log),
        "terminal_receipt": str(repo_path(TERMINAL_RECEIPT_REL)),
        "terminal_checkpoint": str(upstream.terminal_checkpoint),
        "attempt_start_marker": str(
            launcher_attempt_start_marker(launcher, upstream.log)
        ),
        "unpublished_log_witness": str(
            upstream.log.with_name(
                upstream.log.name + ".unpublished-witness"
            )
        ),
        "expected_terminal_update": 464,
        "absent_at_lock": {
            key: True for key in sorted(launcher.locked_absence_keys)
        },
    }
    if set(attempt) != set(launcher.attempt_keys):
        raise ProtocolBuildError("constructed attempt does not match launcher")

    expected_transport = dict(launcher.expected_transport_constants)
    expected_transport["workdir"] = str(REPO_ROOT)
    if set(expected_transport) != set(launcher.expected_transport_keys):
        raise ProtocolBuildError(
            "constructed expected transport does not match launcher"
        )
    lock = {
        "schema_version": launcher.schema_version,
        "status": launcher.locked_status,
        "bindings": binding_files,
        "child_command": child_command,
        "attempt": attempt,
        "expected_transport": expected_transport,
    }
    if set(lock) != set(launcher.transport_lock_keys):
        raise ProtocolBuildError(
            "constructed transport lock does not match launcher top-level schema"
        )
    return lock


def _assert_absent(path: Path, label: str) -> None:
    if os.path.lexists(path):
        raise ProtocolBuildError(f"{label} must be absent: {path}")


def validate_generated_bundle(
    bundle: ProtocolBundle,
    upstream: UpstreamBinding,
    prior: Mapping[str, Any],
    launcher: LauncherContract,
) -> None:
    if raw_sha256(bundle.source_bytes) != bundle.source_sha256:
        raise ProtocolBuildError("source protocol byte hash mismatch")
    if raw_sha256(bundle.comprehensive_bytes) != bundle.comprehensive_sha256:
        raise ProtocolBuildError("comprehensive preregistration byte hash mismatch")
    if raw_sha256(bundle.transport_bytes) != bundle.transport_sha256:
        raise ProtocolBuildError("transport-lock byte hash mismatch")
    if bundle.source_protocol.get("status") != "locked_before_training":
        raise ProtocolBuildError("source protocol status mismatch")
    expected_writer_infrastructure = {
        "path": str(TERMINAL_RECEIPT_WRITER_REL),
        "sha256": EXPECTED_TERMINAL_RECEIPT_WRITER_SHA256,
    }
    if (
        bundle.terminal_receipt_writer_sha256
        != EXPECTED_TERMINAL_RECEIPT_WRITER_SHA256
    ):
        raise ProtocolBuildError("bundle terminal receipt writer hash mismatch")
    if (
        bundle.source_protocol["infrastructure"][
            "terminal_receipt_writer"
        ]
        != expected_writer_infrastructure
    ):
        raise ProtocolBuildError(
            "source terminal receipt writer infrastructure mismatch"
        )
    if (
        bundle.comprehensive_preregistration["infrastructure"]
        != bundle.source_protocol["infrastructure"]
    ):
        raise ProtocolBuildError("comprehensive infrastructure binding drift")
    if (
        bundle.source_protocol["training_transport_binding"][
            "terminal_receipt_writer"
        ]
        != expected_writer_infrastructure
    ):
        raise ProtocolBuildError(
            "source training transport receipt writer mismatch"
        )
    if (
        bundle.source_protocol["candidate_binding"]["terminal_checkpoint"]
        != repo_relative_text(upstream.terminal_checkpoint)
    ):
        raise ProtocolBuildError("source terminal checkpoint mismatch")
    source_ref = bundle.comprehensive_preregistration.get("source_protocol")
    if not isinstance(source_ref, dict) or source_ref.get("sha256") != (
        bundle.source_sha256
    ):
        raise ProtocolBuildError("comprehensive source binding mismatch")
    behavior = bundle.source_protocol["behavior_protocol"]
    if behavior["metric_mapping_and_gates"] != expected_gate_mapping(
        behavior["ordered_panels"]
    ):
        raise ProtocolBuildError("source behavior mapping differs from thresholds")
    if (
        bundle.comprehensive_preregistration["behavior_protocol"]
        != behavior
    ):
        raise ProtocolBuildError("comprehensive behavior protocol drift")
    validate_gold19_gate_reuse(
        prior,
        bundle.comprehensive_preregistration["gold19_screen128_protocol"],
        bundle.comprehensive_preregistration[
            "gold19_confirm512_protocol_if_screen_passes"
        ],
    )
    fixed = bundle.source_protocol["fixed_training"]
    if fixed["loss"]["identity"] != (
        "policy_loss + 0.25 * value_loss - "
        "0.001 * entropy + 0.012 * BC_KL"
    ):
        raise ProtocolBuildError("loss identity drift")
    if fixed["optimizer_steps_each_update"] != (
        "2 * ceil(optimization transitions / 384)"
    ):
        raise ProtocolBuildError("optimizer-step formula drift")
    if bundle.source_protocol["interpretation"]["strict_causal_ablation"]:
        raise ProtocolBuildError("strict causal claim is forbidden")
    if not bundle.source_protocol["interpretation"][
        "continued_training_confound"
    ]:
        raise ProtocolBuildError("continued-training confound must be disclosed")
    prohibited = bundle.source_protocol["prohibited_actions"]
    if not all(prohibited.values()):
        raise ProtocolBuildError("every prohibited action must remain prohibited")
    if set(bundle.transport_lock) != set(launcher.transport_lock_keys):
        raise ProtocolBuildError("transport-lock top-level schema drift")
    if (
        bundle.transport_lock["bindings"]["source_protocol"]["sha256"]
        != bundle.source_sha256
    ):
        raise ProtocolBuildError("transport source binding mismatch")
    if (
        bundle.transport_lock["bindings"]["comprehensive_preregistration"][
            "sha256"
        ]
        != bundle.comprehensive_sha256
    ):
        raise ProtocolBuildError("transport comprehensive binding mismatch")
    if bundle.transport_lock["bindings"]["terminal_receipt_writer"] != {
        "path": str(repo_path(TERMINAL_RECEIPT_WRITER_REL)),
        "sha256": EXPECTED_TERMINAL_RECEIPT_WRITER_SHA256,
    }:
        raise ProtocolBuildError("transport receipt writer binding mismatch")


def build_protocol_bundle(
    *,
    created_at_utc: str,
    expected_behavior_runner_sha256: str | None,
    expected_training_launcher_sha256: str | None,
    expected_receipt_writer_sha256: str | None,
) -> tuple[ProtocolBundle, UpstreamBinding, LauncherContract]:
    receipt_writer = load_terminal_receipt_writer_contract(
        expected_receipt_writer_sha256
    )
    upstream = validate_upstream_bindings()
    prior = load_prior_vcoef_comprehensive()
    behavior_runner = load_behavior_runner_contract(
        expected_behavior_runner_sha256
    )
    launcher = load_launcher_contract(expected_training_launcher_sha256)
    source = build_source_protocol(
        created_at_utc,
        upstream,
        prior,
        behavior_runner,
        launcher,
        receipt_writer,
    )
    source_bytes = pretty_json_bytes(source)
    source_sha256 = raw_sha256(source_bytes)
    comprehensive = build_comprehensive_preregistration(
        created_at_utc,
        source,
        source_sha256,
        prior,
    )
    comprehensive_bytes = pretty_json_bytes(comprehensive)
    comprehensive_sha256 = raw_sha256(comprehensive_bytes)
    transport = build_transport_lock(
        upstream,
        launcher,
        receipt_writer,
        source_sha256,
        comprehensive_sha256,
    )
    transport_bytes = pretty_json_bytes(transport)
    bundle = ProtocolBundle(
        source_protocol=source,
        comprehensive_preregistration=comprehensive,
        transport_lock=transport,
        source_bytes=source_bytes,
        comprehensive_bytes=comprehensive_bytes,
        transport_bytes=transport_bytes,
        source_sha256=source_sha256,
        comprehensive_sha256=comprehensive_sha256,
        transport_sha256=raw_sha256(transport_bytes),
        behavior_runner_sha256=behavior_runner.sha256,
        training_launcher_sha256=launcher.sha256,
        terminal_receipt_writer_sha256=receipt_writer.sha256,
        provisional_infrastructure_hashes=(
            expected_behavior_runner_sha256 is None
            or expected_training_launcher_sha256 is None
        ),
        pending_final_sha_fields=tuple(
            name
            for name, value in (
                (
                    "--expected-behavior-runner-sha256",
                    expected_behavior_runner_sha256,
                ),
                (
                    "--expected-training-launcher-sha256",
                    expected_training_launcher_sha256,
                ),
            )
            if value is None
        ),
    )
    validate_generated_bundle(bundle, upstream, prior, launcher)
    return bundle, upstream, launcher


def assert_formal_targets_absent(
    upstream: UpstreamBinding,
    launcher: LauncherContract,
) -> None:
    for path, label in (
        (repo_path(SOURCE_PROTOCOL_REL), "source protocol"),
        (
            repo_path(COMPREHENSIVE_PREREGISTRATION_REL),
            "comprehensive preregistration",
        ),
        (repo_path(TRANSPORT_LOCK_REL), "transport lock"),
        (upstream.output_dir, "training output directory"),
        (upstream.log, "training log"),
        (repo_path(TERMINAL_RECEIPT_REL), "terminal receipt"),
        (upstream.terminal_checkpoint, "terminal checkpoint"),
        (
            launcher_attempt_start_marker(launcher, upstream.log),
            "attempt start marker",
        ),
        (
            upstream.log.with_name(
                upstream.log.name + ".unpublished-witness"
            ),
            "unpublished log witness",
        ),
    ):
        _assert_absent(path, label)


def _ensure_directory_chain(path: Path) -> None:
    """Create only missing repository directories, rejecting every symlink."""

    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ProtocolBuildError("directory escapes repository") from error
    current = REPO_ROOT
    for component in relative.parts:
        current = current / component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            os.mkdir(current, 0o700)
            info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ProtocolBuildError(
                f"artifact parent is not a real directory: {current}"
            )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while freezing protocol")
        view = view[written:]


def write_files_o_excl(targets: Sequence[tuple[Path, bytes]]) -> None:
    """Claim and freeze distinct files, never replacing an existing entry."""

    if not targets:
        raise ProtocolBuildError("O_EXCL target list must not be empty")
    paths = [path for path, _ in targets]
    if len(set(paths)) != len(paths):
        raise ProtocolBuildError("O_EXCL target paths must be distinct")
    for path in paths:
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise ProtocolBuildError(
                f"O_EXCL target parent must be a real directory: {path.parent}"
            )
    descriptors: list[tuple[Path, int, int, int]] = []
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    all_claimed = False
    try:
        # Claim every final pathname before publishing bytes.  A collision
        # cannot cause the remaining files to be overwritten.
        for path, _ in targets:
            descriptor = os.open(path, flags, 0o600)
            info = os.fstat(descriptor)
            descriptors.append(
                (path, descriptor, int(info.st_dev), int(info.st_ino))
            )
        all_claimed = True
        for (_, payload), (path, descriptor, _, _) in zip(
            targets,
            descriptors,
            strict=True,
        ):
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            if file_sha256(Path(f"/proc/self/fd/{descriptor}")) != raw_sha256(
                payload
            ):
                raise ProtocolBuildError(
                    f"post-write SHA-256 mismatch for {path}"
                )
        for _, descriptor, _, _ in descriptors:
            os.close(descriptor)
        descriptors.clear()
        for parent in sorted({path.parent for path in paths}):
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    finally:
        for path, descriptor, claimed_dev, claimed_ino in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
            # Before any bytes are published, a later O_EXCL collision should
            # not leave zero-length claims from this invocation.  Once all
            # names are claimed, failures remain fail-closed and visible.
            if not all_claimed:
                try:
                    info = os.lstat(path)
                    if (
                        stat.S_ISREG(info.st_mode)
                        and info.st_size == 0
                        and int(info.st_dev) == claimed_dev
                        and int(info.st_ino) == claimed_ino
                    ):
                        os.unlink(path)
                except FileNotFoundError:
                    pass


def write_bundle_o_excl(bundle: ProtocolBundle) -> None:
    """Write all formal JSON files without overwriting any directory entry."""

    write_files_o_excl(
        (
            (repo_path(SOURCE_PROTOCOL_REL), bundle.source_bytes),
            (
                repo_path(COMPREHENSIVE_PREREGISTRATION_REL),
                bundle.comprehensive_bytes,
            ),
            (repo_path(TRANSPORT_LOCK_REL), bundle.transport_bytes),
        )
    )


def validate_written_lock_with_launcher(
    launcher: LauncherContract,
    expected_transport_sha256: str,
) -> None:
    """Call the final launcher's own read-only validator after O_EXCL writes."""

    import importlib.util

    module_name = "_ptcg_final_training_launcher_validation"
    spec = importlib.util.spec_from_file_location(module_name, launcher.path)
    if spec is None or spec.loader is None:
        raise ProtocolBuildError("could not load final training launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        validated = module.read_and_validate_transport_lock(
            transport_lock=repo_path(TRANSPORT_LOCK_REL),
            expected_transport_lock_sha256=expected_transport_sha256,
        )
    finally:
        sys.modules.pop(module_name, None)
    if validated.lock_sha256 != expected_transport_sha256:
        raise ProtocolBuildError("launcher returned a different transport hash")
    if validated.command_sha256 != EXPECTED_BRANCH_COMMAND_SHA256:
        raise ProtocolBuildError("launcher returned a different child command")


def result_summary(bundle: ProtocolBundle, *, write: bool) -> dict[str, Any]:
    return {
        "schema_version": "ptcg-ppo-mb384-protocol-builder-result-v1",
        "status": (
            "formal_o_excl_bundle_written_and_launcher_validated"
            if write
            else "dry_run_validated_no_files_written"
        ),
        "candidate": CANDIDATE,
        "formal_artifacts": {
            "source_protocol": {
                "path": str(SOURCE_PROTOCOL_REL),
                "sha256": bundle.source_sha256,
            },
            "comprehensive_preregistration": {
                "path": str(COMPREHENSIVE_PREREGISTRATION_REL),
                "sha256": bundle.comprehensive_sha256,
            },
            "transport_lock": {
                "path": str(TRANSPORT_LOCK_REL),
                "sha256": bundle.transport_sha256,
            },
        },
        "frozen_upstream": {
            "runner_preregistration_sha256": (
                EXPECTED_RUNNER_PREREGISTRATION_SHA256
            ),
            "branch_preregistration_sha256": (
                EXPECTED_BRANCH_PREREGISTRATION_SHA256
            ),
            "branch_binding_sha256": EXPECTED_BRANCH_BINDING_SHA256,
            "branch_command_sha256": EXPECTED_BRANCH_COMMAND_SHA256,
            "full_incumbent_u456_sha256": (
                EXPECTED_FULL_INCUMBENT_U456_SHA256
            ),
        },
        "infrastructure": {
            "behavior_runner_sha256": bundle.behavior_runner_sha256,
            "training_launcher_sha256": bundle.training_launcher_sha256,
            "terminal_receipt_writer": {
                "path": str(TERMINAL_RECEIPT_WRITER_REL),
                "sha256": bundle.terminal_receipt_writer_sha256,
            },
            "provisional": bundle.provisional_infrastructure_hashes,
        },
        "fresh_gold19_seeds": list(FRESH_GOLD19_SEEDS),
        "training_or_gpu_or_network_started": False,
        "files_written": write,
        "pending_final_sha_fields": list(bundle.pending_final_sha_fields),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-behavior-runner-sha256",
        help=(
            "Final SHA-256 of the frozen in-process behavior runner. "
            "Required with --write."
        ),
    )
    parser.add_argument(
        "--expected-training-launcher-sha256",
        help=(
            "Final SHA-256 of the direct-exec training launcher. "
            "Required with --write."
        ),
    )
    parser.add_argument(
        "--expected-receipt-writer-sha256",
        help=(
            "Frozen final SHA-256 of tools/write_o_excl_json_receipt.py. "
            "Required for every build."
        ),
    )
    parser.add_argument(
        "--created-at-utc",
        default=None,
        help="One UTC timestamp shared by source and comprehensive JSON.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Create the three formal JSON artifacts with O_EXCL. "
            "The default is a no-write dry run."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_receipt_writer_sha256 is None:
        raise ProtocolBuildError(
            "--expected-receipt-writer-sha256 is required for every build"
        )
    if args.write and (
        args.expected_behavior_runner_sha256 is None
        or args.expected_training_launcher_sha256 is None
    ):
        raise ProtocolBuildError(
            "--write requires both final infrastructure SHA-256 arguments"
        )
    created_at = args.created_at_utc or utc_now()
    bundle, upstream, launcher = build_protocol_bundle(
        created_at_utc=created_at,
        expected_behavior_runner_sha256=(
            args.expected_behavior_runner_sha256
        ),
        expected_training_launcher_sha256=(
            args.expected_training_launcher_sha256
        ),
        expected_receipt_writer_sha256=(
            args.expected_receipt_writer_sha256
        ),
    )
    assert_formal_targets_absent(upstream, launcher)
    if args.write:
        _ensure_directory_chain(upstream.output_dir.parent)
        # Recheck after parent preparation and immediately before O_EXCL.
        assert_formal_targets_absent(upstream, launcher)
        write_bundle_o_excl(bundle)
        validate_written_lock_with_launcher(
            launcher,
            bundle.transport_sha256,
        )
    print(
        json.dumps(
            result_summary(bundle, write=args.write),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
