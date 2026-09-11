#!/usr/bin/env python3
"""Audit or execute the exact G8-preconditioned U464-to-U468 PPO stage."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import exec_preregistered_ppo_u464_to_u468 as base  # noqa: E402


WRAPPER_PATH = Path(__file__).resolve()
BASE_LAUNCHER_PATH = REPO_ROOT / "tools/exec_preregistered_ppo_u464_to_u468.py"
BASE_LAUNCHER_SHA256 = (
    "ace3a807704fd3dc0e5630077da5369861aecf75365a40d2261876995d36cc00"
)

BRANCH_NAME = "ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090"
DESIGN_PATH = REPO_ROOT / f"artifacts/{BRANCH_NAME}.ppo_stage_design_preregistration.json"
DESIGN_SHA256 = "782ec89db9930709c2d956bc001581e30be186bf2b559c3ffe6332d58ad01277"
PARENT_PATH = REPO_ROOT / (
    f"artifacts/{BRANCH_NAME}/general_stage/general-bc-0008.pt"
)
PARENT_SHA256 = "50aff12d7a1575874c0ccd165f8c0563c1dcaa543b48aaa15ac25f25d2c526c1"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / BRANCH_NAME / "ppo_stage"
OUTPUT_DIR = OUTPUT_ROOT / "B_gold_league/seed-202607336"
TERMINAL_CHECKPOINT = OUTPUT_DIR / "checkpoints/update-0468.pt"
ATTEMPT_MARKER = REPO_ROOT / (
    ".ptcg-u464g8-ppo-u468-attempt-202607336-202608090.json"
)
LOG_PATH = REPO_ROOT / f"artifacts/{BRANCH_NAME}.ppo_stage.log"

SCHEMA_VERSION = "ptcg-u464g8-to-u468-resume-all-execution-preregistration-v1"
DERIVED_COMMAND_SHA256 = (
    "634073b38bbd492e97cf822893b50497db6ab68c10dc6e8bf91a55de07bd495e"
)
DERIVED_COMMAND_TOKENS = 177

EXPECTED_PARENT_STATE = {
    "update": 464,
    "runtime_model_state_sha256": (
        "50b06f569724ad2c220e8d4fccf1198ee620a12fb5c88207185046ee6f62d3b4"
    ),
    "ppo_optimizer_nested_sha256": (
        "904a3d2c8ac63c8a60af2404c216830b8ae30a69fbe972818e05c0136058a54f"
    ),
    "ppo_optimizer_state_count": 28,
    "ppo_optimizer_step": 276,
    "bc_replay_optimizer_nested_sha256": (
        "094e8b5210e04843141349471498b7a316ec6ea0bd245db4fcaf386cd537ecbf"
    ),
    "bc_replay_optimizer_state_count": 24,
    "bc_replay_optimizer_step": 24,
    "opponent_quota_nested_sha256": (
        "7139320983450f2d2b9d50a65684c31550c409ed8adde9feb84839db24e1fc27"
    ),
    "opponent_quota_observed_games": 512,
    "opponent_quota_last_refresh_update": 463,
}

EXTRA_BINDINGS: dict[str, tuple[Path, str, str]] = {
    "base_launcher": (
        BASE_LAUNCHER_PATH,
        BASE_LAUNCHER_SHA256,
        "base launcher implementation",
    ),
    "master_preregistration": (
        REPO_ROOT / f"artifacts/{BRANCH_NAME}.master_preregistration.json",
        "b45e79aae06f0c96f5234451b1648f7df8f033f992f394707e756a297b1395b2",
        "master preregistration",
    ),
    "g8_execution_preregistration": (
        REPO_ROOT / f"artifacts/{BRANCH_NAME}.g8_execution_preregistration.json",
        "5843189f2886ee35d062615293bfaf782729aeefdadd30257c572f7e7c74fe95",
        "G8 execution preregistration",
    ),
    "g8_attempt_marker": (
        REPO_ROOT / ".ptcg-u464-g8-generalbc-attempt-202608013-202608090.json",
        "13dff54630654042240b014249edff9afb7c6c015c8b427833053a21ab1515e9",
        "G8 attempt marker",
    ),
    "g8_manifest": (
        REPO_ROOT / f"artifacts/{BRANCH_NAME}/general_stage/general_bc_manifest.json",
        "172df68d9430bd6fdb315dfd8d696a5f79c299b88dd6921bb08c2d583155e26a",
        "G8 manifest",
    ),
    "g8_training_integrity_decision": (
        REPO_ROOT / f"artifacts/{BRANCH_NAME}.g8_training_integrity_decision.json",
        "e38f0917e48678ac354502a0f3e2d0115c9dcef3472b2715b4fc1f1d8bfe4975",
        "G8 training integrity decision",
    ),
}


def canonical_command_sha256(command: Sequence[str]) -> str:
    raw = (
        json.dumps(
            list(command),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def derive_exact_command() -> list[str]:
    base.require_regular(base.SOURCE_COMMAND_PROTOCOL, "source command protocol")
    if (
        base.sha256_file(base.SOURCE_COMMAND_PROTOCOL)
        != base.SOURCE_COMMAND_PROTOCOL_SHA256
    ):
        raise ValueError("source command protocol SHA-256 mismatch")
    source_protocol = json.loads(base.SOURCE_COMMAND_PROTOCOL.read_text())
    source = source_protocol.get("binding", {}).get("command")
    if not isinstance(source, list) or not all(
        isinstance(token, str) for token in source
    ):
        raise ValueError("source command must be a string array")
    if base.sha256_bytes(base.canonical_json_bytes(source)) != base.SOURCE_COMMAND_SHA256:
        raise ValueError("source command SHA-256 mismatch")
    command = list(source)
    replacements = {
        "--output-dir": str(OUTPUT_DIR),
        "--updates": "468",
        "--seed": "202607336",
        "--resume": str(PARENT_PATH),
    }
    for flag, value in replacements.items():
        base.replace_flag_value(command, flag, value)
    for flag in ("--reset-optimizer-on-resume", "--reset-opponent-quota-on-resume"):
        if command.count(flag) != 1:
            raise ValueError(f"source command must contain exactly one {flag}")
        command.remove(flag)
    if len(command) != DERIVED_COMMAND_TOKENS:
        raise ValueError("derived command token count mismatch")
    if canonical_command_sha256(command) != DERIVED_COMMAND_SHA256:
        raise ValueError("derived command SHA-256 mismatch")
    return command


original_command_input_paths = base.command_input_paths
original_assert_locked_input_bindings = base.assert_locked_input_bindings
original_load_and_validate_protocol = base.load_and_validate_protocol


def command_input_paths(command: Sequence[str]) -> list[Path]:
    paths = original_command_input_paths(command)
    for path, _, label in EXTRA_BINDINGS.values():
        paths.append(base.require_regular(path, label))
    return sorted(set(paths), key=str)


def assert_locked_input_bindings(
    snapshots: Sequence[dict[str, Any]],
    protocol: dict[str, Any],
    command: Sequence[str],
    protocol_path: Path,
    protocol_sha256: str,
) -> None:
    extra_paths = {path.resolve() for path, _, _ in EXTRA_BINDINGS.values()}
    base_snapshots = [
        snapshot
        for snapshot in snapshots
        if Path(snapshot["path"]).resolve() not in extra_paths
    ]
    original_assert_locked_input_bindings(
        base_snapshots,
        protocol,
        command,
        protocol_path,
        protocol_sha256,
    )
    for key, (path, expected_sha256, label) in EXTRA_BINDINGS.items():
        snapshot = base.snapshot_by_path(snapshots, path)
        if snapshot["sha256"] != expected_sha256:
            raise RuntimeError(f"held {label} SHA-256 differs from binding")
        base.require_bound_file(
            protocol["bindings"][key], path, expected_sha256, label
        )


def load_and_validate_protocol(
    protocol_path: Path,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    protocol, command, audit = original_load_and_validate_protocol(
        protocol_path,
        expected_protocol_sha256,
    )
    for key, (path, expected_sha256, label) in EXTRA_BINDINGS.items():
        base.require_bound_file(
            protocol["bindings"][key], path, expected_sha256, label
        )
    if base.flag_value(command, "--bc-replay-lr-scale") != "0.05":
        raise ValueError("frozen general replay LR scale drifted")
    if base.flag_value(command, "--bc-kl-start") != "0.012":
        raise ValueError("frozen BC KL start drifted")
    if base.flag_value(command, "--bc-kl-end") != "0.012":
        raise ValueError("frozen BC KL end drifted")
    audit["only_model_lineage_change_vs_original_continuation"] = {
        "parent": str(PARENT_PATH.relative_to(REPO_ROOT)),
        "parent_sha256": PARENT_SHA256,
        "ppo_hyperparameters": "exact",
    }
    audit["failed_raw_u468_checkpoints_are_not_inputs"] = True
    return protocol, command, audit


# Rebind the audited base implementation to this immutable G8-parent branch.
base.__file__ = str(WRAPPER_PATH)
base.SCHEMA_VERSION = SCHEMA_VERSION
base.BINDING_KEYS = frozenset(set(base.BINDING_KEYS) | set(EXTRA_BINDINGS))
base.DESIGN_PATH = DESIGN_PATH
base.DESIGN_SHA256 = DESIGN_SHA256
base.PARENT_PATH = PARENT_PATH
base.PARENT_SHA256 = PARENT_SHA256
base.EXPECTED_PARENT_STATE = EXPECTED_PARENT_STATE
base.OUTPUT_ROOT = OUTPUT_ROOT
base.OUTPUT_DIR = OUTPUT_DIR
base.TERMINAL_CHECKPOINT = TERMINAL_CHECKPOINT
base.ATTEMPT_MARKER = ATTEMPT_MARKER
base.LOG_PATH = LOG_PATH
base.DERIVED_COMMAND_SHA256 = DERIVED_COMMAND_SHA256
base.DERIVED_COMMAND_TOKENS = DERIVED_COMMAND_TOKENS
base.derive_exact_command = derive_exact_command
base.command_input_paths = command_input_paths
base.assert_locked_input_bindings = assert_locked_input_bindings
base.load_and_validate_protocol = load_and_validate_protocol


if __name__ == "__main__":
    raise SystemExit(base.main())
