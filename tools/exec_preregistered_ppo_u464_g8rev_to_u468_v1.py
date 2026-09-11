#!/usr/bin/env python3
"""Execute the single preregistered G8-reverse U464-to-U468 PPO stage.

The audited lineage implementation supplies all state/input checks.  This
specialization changes only the G8-reverse parent and output identities, and
uses an isolated ``python -I -B`` child for the held/sealed trainer payload.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import exec_preregistered_ppo_u464g8_to_u468 as lineage  # noqa: E402


base = lineage.base
REPO_ROOT = Path(__file__).resolve().parents[1]
SELF_PATH = Path(__file__).resolve()
BRANCH = "ppo_u464_g8rev_generalbc_then_ppo4_then_specialbc_design202608120"
SCHEMA_VERSION = "ptcg-u464-g8rev-to-u468-isolated-execution-v1"
DESIGN_PATH = REPO_ROOT / f"artifacts/{BRANCH}.preregistration.json"
DESIGN_SHA256 = "9fd65ef48266bb3d2e60111d51c2358de47f227f946ed288bbb5cd23a5db55ff"
PARENT_PATH = REPO_ROOT / f"artifacts/{BRANCH}/general_stage/general-bc-0008.pt"
PARENT_SHA256 = "6b7ff16c9ebac2810dc996b3b191a9f7b4bc2bb2381064791b2f8c6f4da4cda9"
OUTPUT_ROOT = REPO_ROOT / f"artifacts/{BRANCH}/ppo_stage"
OUTPUT_DIR = OUTPUT_ROOT / "B_gold_league/seed-202607336"
TERMINAL_CHECKPOINT = OUTPUT_DIR / "checkpoints/update-0468.pt"
ATTEMPT_MARKER = REPO_ROOT / ".ptcg-u464-g8rev-ppo-u468-attempt-202607336-202608120.json"
LOG_PATH = REPO_ROOT / f"artifacts/{BRANCH}.ppo_stage.log"
DERIVED_COMMAND_SHA256 = "8d4e9a9ddc8eb22896541184857fdbb53644598ea6317fab8fe32188cc8eda9f"
DERIVED_COMMAND_TOKENS = 177

EXPECTED_PARENT_STATE = {
    "update": 464,
    "runtime_model_state_sha256": "be548a79abb72330a18076c6e47658d95eeadcb7b243a27fcdbfe76168282723",
    "ppo_optimizer_nested_sha256": "904a3d2c8ac63c8a60af2404c216830b8ae30a69fbe972818e05c0136058a54f",
    "ppo_optimizer_state_count": 28,
    "ppo_optimizer_step": 276,
    "bc_replay_optimizer_nested_sha256": "b04ecaf0b871714600b8a86b1038391c09355184e94e29955d2c45e565fefe63",
    "bc_replay_optimizer_state_count": 24,
    "bc_replay_optimizer_step": 24,
    "opponent_quota_nested_sha256": "7139320983450f2d2b9d50a65684c31550c409ed8adde9feb84839db24e1fc27",
    "opponent_quota_observed_games": 512,
    "opponent_quota_last_refresh_update": 463,
}

OLD_EXTRA_KEYS = set(lineage.EXTRA_BINDINGS)
EXTRA_BINDINGS: dict[str, tuple[Path, str, str]] = {
    "base_launcher": (
        REPO_ROOT / "tools/exec_preregistered_ppo_u464_to_u468.py",
        "ace3a807704fd3dc0e5630077da5369861aecf75365a40d2261876995d36cc00",
        "base launcher implementation",
    ),
    "g8_lineage_launcher": (
        REPO_ROOT / "tools/exec_preregistered_ppo_u464g8_to_u468.py",
        "5855d8dcc0f9b16db1ff905554d5396ccdc6951629833f7cc045b28f720fe7f3",
        "G8 lineage launcher",
    ),
    "g8rev_executor": (
        REPO_ROOT / "tools/run_ppo_general_bc_precondition_u464_g8rev_v1.py",
        "cda032fdd672fa2c7023ed50b1c8daad39d81bd49ae1e650a4cb56a19f3b0e60",
        "G8-reverse executor",
    ),
    "g8rev_manifest": (
        REPO_ROOT / f"artifacts/{BRANCH}/general_stage/general_bc_manifest.json",
        "68d94bfa46761bb4d1d4aef94c90416ee889671d7a7998f7b66bb5ce7d4b23bf",
        "G8-reverse training manifest",
    ),
}


def isolated_claim_and_run(
    protocol: dict[str, Any],
    command: Sequence[str],
    protocol_sha256: str,
    protocol_path: Path,
) -> int:
    """Run the sealed trainer under the explicitly isolated project Python."""
    marker_payload = {
        "event": "u464_g8rev_to_u468_preregistered_attempt_consumed",
        "schema_version": SCHEMA_VERSION,
        "seed": protocol["seed"],
        "protocol_sha256": protocol_sha256,
        "command_sha256": protocol["command_sha256"],
        "launcher_sha256": base.sha256_file(SELF_PATH),
        "child_python_flags": ["-I", "-B"],
    }
    marker_fd = os.open(
        ATTEMPT_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        os.write(marker_fd, base.canonical_json_bytes(marker_payload))
        os.fsync(marker_fd)
    finally:
        os.close(marker_fd)

    log_fd = -1
    payload_fd = -1
    bootstrap_fd = -1
    snapshots: list[dict[str, Any]] = []
    try:
        log_fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.mkdir(OUTPUT_ROOT, mode=0o700)
        claim_path = OUTPUT_ROOT / ".u464_g8rev_to_u468_launch_claim.json"
        claim_fd = os.open(
            claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            os.write(claim_fd, base.canonical_json_bytes(marker_payload))
            os.fsync(claim_fd)
        finally:
            os.close(claim_fd)

        locked_paths = base.command_input_paths(command)
        locked_paths.append(base.require_regular(protocol_path, "execution protocol"))
        snapshots = base.acquire_input_locks(sorted(set(locked_paths), key=str))
        base.assert_input_locks_unchanged(snapshots)
        base.assert_locked_input_bindings(
            snapshots, protocol, command, protocol_path, protocol_sha256
        )

        trainer_snapshot = base.snapshot_by_path(snapshots, base.TRAINER_PATH)
        trainer_payload = base.read_fd_bytes(int(trainer_snapshot["fd"]))
        if hashlib.sha256(trainer_payload).hexdigest() != base.TRAINER_SHA256:
            raise RuntimeError("held trainer payload differs from frozen binding")
        payload_fd, payload_sha256 = base.create_sealed_memfd(
            "ptcg-u464-g8rev-u468-trainer-payload", trainer_payload
        )
        bootstrap_payload = base.build_trainer_bootstrap(payload_fd)
        bootstrap_fd, bootstrap_sha256 = base.create_sealed_memfd(
            "ptcg-u464-g8rev-u468-trainer-bootstrap", bootstrap_payload
        )
        base.assert_sealed_memfd(payload_fd, payload_sha256, "trainer payload")
        base.assert_sealed_memfd(bootstrap_fd, bootstrap_sha256, "trainer bootstrap")

        effective_command = [
            command[0],
            "-I",
            "-B",
            f"/proc/self/fd/{bootstrap_fd}",
            *command[2:],
        ]
        effective_command_sha256 = base.sha256_bytes(
            base.canonical_json_bytes(effective_command)
        )
        handle = os.fdopen(log_fd, "wb", buffering=0)
        log_fd = -1
        with handle:
            handle.write(
                base.canonical_json_bytes(
                    {
                        "event": "u464_g8rev_to_u468_pre_exec_checks_passed",
                        **{key: value for key, value in marker_payload.items() if key != "event"},
                        "cwd": str(REPO_ROOT),
                        "child_argv_prefix": effective_command[:4],
                        "child_isolated_mode": True,
                        "effective_command_sha256": effective_command_sha256,
                        "input_shared_locks": len(snapshots),
                        "sealed_trainer_payload_sha256": payload_sha256,
                        "sealed_trainer_bootstrap_sha256": bootstrap_sha256,
                    }
                )
            )
            os.fsync(handle.fileno())
            base.assert_input_locks_unchanged(snapshots)
            completed = subprocess.run(
                effective_command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                pass_fds=(bootstrap_fd, payload_fd),
            )
            base.assert_input_locks_unchanged(snapshots)
            base.assert_sealed_memfd(payload_fd, payload_sha256, "trainer payload")
            base.assert_sealed_memfd(
                bootstrap_fd, bootstrap_sha256, "trainer bootstrap"
            )
            handle.write(
                base.canonical_json_bytes(
                    {
                        "event": "u464_g8rev_to_u468_child_terminal",
                        "return_code": completed.returncode,
                        "child_isolated_mode": True,
                        "inputs_unchanged_through_child_terminal": True,
                    }
                )
            )
            os.fsync(handle.fileno())
        return completed.returncode
    finally:
        if bootstrap_fd >= 0:
            os.close(bootstrap_fd)
        if payload_fd >= 0:
            os.close(payload_fd)
        if snapshots:
            base.release_input_locks(snapshots)
        if log_fd >= 0:
            os.close(log_fd)


lineage.WRAPPER_PATH = SELF_PATH
lineage.SCHEMA_VERSION = SCHEMA_VERSION
lineage.DESIGN_PATH = DESIGN_PATH
lineage.DESIGN_SHA256 = DESIGN_SHA256
lineage.PARENT_PATH = PARENT_PATH
lineage.PARENT_SHA256 = PARENT_SHA256
lineage.OUTPUT_ROOT = OUTPUT_ROOT
lineage.OUTPUT_DIR = OUTPUT_DIR
lineage.TERMINAL_CHECKPOINT = TERMINAL_CHECKPOINT
lineage.ATTEMPT_MARKER = ATTEMPT_MARKER
lineage.LOG_PATH = LOG_PATH
lineage.DERIVED_COMMAND_SHA256 = DERIVED_COMMAND_SHA256
lineage.DERIVED_COMMAND_TOKENS = DERIVED_COMMAND_TOKENS
lineage.EXPECTED_PARENT_STATE = EXPECTED_PARENT_STATE
lineage.EXTRA_BINDINGS = EXTRA_BINDINGS

base.__file__ = str(SELF_PATH)
base.SCHEMA_VERSION = SCHEMA_VERSION
base.BINDING_KEYS = frozenset(
    (set(base.BINDING_KEYS) - OLD_EXTRA_KEYS) | set(EXTRA_BINDINGS)
)
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
base.derive_exact_command = lineage.derive_exact_command
base.command_input_paths = lineage.command_input_paths
base.assert_locked_input_bindings = lineage.assert_locked_input_bindings
base.load_and_validate_protocol = lineage.load_and_validate_protocol
base.claim_and_run = isolated_claim_and_run


if __name__ == "__main__":
    raise SystemExit(base.main())
