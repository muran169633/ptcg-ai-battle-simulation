#!/usr/bin/env python3
"""Audit or execute the stronger-general-replay U464-to-U468 branch."""

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
FAILED_PREDECESSOR_DECISION_PATH = REPO_ROOT / (
    "artifacts/ppo_u464inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_resumeall_u464_to_u468_seed202607336."
    "training_integrity_decision.json"
)
FAILED_PREDECESSOR_DECISION_SHA256 = (
    "3fd1e355f1fad08ac352eef319340a8b67d43c78ad59462399aa5c33cc7a047b"
)

BRANCH_NAME = (
    "ppo_u464inc_currentgold10_tailfocus_replay2lr075_ctx34q4_"
    "episode_mean_actorlr2x_mb384_resumeall_u464_to_u468_seed202607336"
)
DESIGN_PATH = REPO_ROOT / f"artifacts/{BRANCH_NAME}.design_preregistration.json"
DESIGN_SHA256 = "c98431e7da5837e954a55418bdd59478f6da0427ed20816ea47374612850bfd4"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / BRANCH_NAME
OUTPUT_DIR = OUTPUT_ROOT / "B_gold_league/seed-202607336"
TERMINAL_CHECKPOINT = OUTPUT_DIR / "checkpoints/update-0468.pt"
ATTEMPT_MARKER = REPO_ROOT / (
    ".ptcg-ppo-replaylr075-u464-u468-attempt-202607336-202608081.json"
)
LOG_PATH = REPO_ROOT / f"artifacts/{BRANCH_NAME}.log"

SCHEMA_VERSION = "ptcg-u464-to-u468-replaylr075-execution-preregistration-v1"
DERIVED_COMMAND_SHA256 = (
    "e63281a4d1f688ed21e75ea1b3d44144ac86497832e96b934c39c7af458f1cb3"
)
DERIVED_COMMAND_TOKENS = 177

EXTRA_BINDINGS: dict[str, tuple[Path, str, str]] = {
    "base_launcher": (
        BASE_LAUNCHER_PATH,
        BASE_LAUNCHER_SHA256,
        "base launcher implementation",
    ),
    "failed_predecessor_decision": (
        FAILED_PREDECESSOR_DECISION_PATH,
        FAILED_PREDECESSOR_DECISION_SHA256,
        "failed predecessor decision",
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
        "--resume": str(base.PARENT_PATH),
        "--bc-replay-lr-scale": "0.075",
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
        binding = protocol["bindings"][key]
        base.require_bound_file(binding, path, expected_sha256, label)


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
    if base.flag_value(command, "--bc-replay-lr-scale") != "0.075":
        raise ValueError("stronger replay LR scale drifted")
    audit["only_hyperparameter_change_vs_failed_predecessor"] = {
        "flag": "--bc-replay-lr-scale",
        "old": "0.05",
        "new": "0.075",
    }
    audit["failed_predecessor_is_not_parent"] = True
    return protocol, command, audit


# Rebind the audited base implementation to this immutable branch wrapper.
base.__file__ = str(WRAPPER_PATH)
base.SCHEMA_VERSION = SCHEMA_VERSION
base.BINDING_KEYS = frozenset(set(base.BINDING_KEYS) | set(EXTRA_BINDINGS))
base.DESIGN_PATH = DESIGN_PATH
base.DESIGN_SHA256 = DESIGN_SHA256
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
