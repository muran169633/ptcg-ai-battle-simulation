#!/usr/bin/env python3
"""Execute one hash-bound PPO command without unverifiable outer API metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ptcg-local-preregistered-ppo-v1"
PROTOCOL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "seed",
        "source_branch_preregistration",
        "source_command_sha256",
        "derived_command_sha256",
        "command_transform",
        "launcher",
        "trainer",
        "source_checkpoint",
        "evaluation_protocol_source",
        "prior_failed_attempt",
        "fresh_evaluation_seeds",
        "design",
        "prohibited_actions",
        "output_dir",
        "terminal_checkpoint",
        "attempt_start_marker",
        "log",
        "stop_rules",
    }
)
SHA256_RE_LENGTH = 64


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_RE_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def parse_object(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def repo_path(value: Any, label: str, *, must_exist: bool) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = Path(os.path.normpath(os.fspath(path)))
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository") from error
    if must_exist:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} must be a regular symlink-free file")
    return path


def require_bound_file(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = repo_path(value["path"], f"{label}.path", must_exist=True)
    expected = require_sha256(value["sha256"], f"{label}.sha256")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, expected


def replace_flag_value(command: list[str], flag: str, value: str) -> int:
    indices = [index for index, token in enumerate(command) if token == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError(f"source command must contain exactly one {flag}")
    value_index = indices[0] + 1
    command[value_index] = value
    return value_index


def load_and_validate_protocol(
    protocol_path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], list[str], str]:
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError("current working directory must equal repository root")
    protocol_path = repo_path(protocol_path, "protocol", must_exist=True)
    raw = protocol_path.read_bytes()
    observed_protocol_sha256 = sha256_bytes(raw)
    if observed_protocol_sha256 != require_sha256(
        expected_sha256, "expected protocol SHA-256"
    ):
        raise ValueError("protocol SHA-256 mismatch")
    protocol = parse_object(raw, "protocol")
    if set(protocol) != PROTOCOL_KEYS:
        raise ValueError("protocol keys differ from the frozen schema")
    if protocol["schema_version"] != SCHEMA_VERSION:
        raise ValueError("protocol schema version mismatch")
    if protocol["status"] != "locked_before_training":
        raise ValueError("protocol status must be locked_before_training")
    if not isinstance(protocol["seed"], int) or protocol["seed"] < 0:
        raise ValueError("protocol seed must be a non-negative integer")

    launcher_path, _ = require_bound_file(protocol["launcher"], "launcher")
    if launcher_path != Path(__file__).resolve():
        raise ValueError("launcher path differs from the executing file")
    trainer_path, _ = require_bound_file(protocol["trainer"], "trainer")
    checkpoint_path, _ = require_bound_file(
        protocol["source_checkpoint"], "source_checkpoint"
    )
    require_bound_file(
        protocol["evaluation_protocol_source"], "evaluation_protocol_source"
    )
    require_bound_file(protocol["prior_failed_attempt"], "prior_failed_attempt")
    source_branch_path, _ = require_bound_file(
        protocol["source_branch_preregistration"],
        "source_branch_preregistration",
    )
    source_branch = parse_object(
        source_branch_path.read_bytes(), "source branch preregistration"
    )
    try:
        source_command = source_branch["binding"]["command"]
    except (KeyError, TypeError) as error:
        raise ValueError("source branch binding.command is missing") from error
    if not isinstance(source_command, list) or not all(
        isinstance(token, str) for token in source_command
    ):
        raise ValueError("source command must be a string array")
    source_sha256 = sha256_bytes(canonical_json_bytes(source_command))
    if source_sha256 != require_sha256(
        protocol["source_command_sha256"], "source_command_sha256"
    ):
        raise ValueError("source command SHA-256 mismatch")

    transform = protocol["command_transform"]
    expected_transform_keys = {
        "old_seed",
        "new_seed",
        "old_output_dir",
        "new_output_dir",
        "changed_value_indices",
    }
    if not isinstance(transform, dict) or set(transform) != expected_transform_keys:
        raise ValueError("command_transform schema mismatch")
    if transform["new_seed"] != protocol["seed"]:
        raise ValueError("command_transform new seed mismatch")

    command = list(source_command)
    seed_flag_index = command.index("--seed")
    output_flag_index = command.index("--output-dir")
    if command[seed_flag_index + 1] != str(transform["old_seed"]):
        raise ValueError("source seed differs from command_transform")
    if command[output_flag_index + 1] != transform["old_output_dir"]:
        raise ValueError("source output directory differs from command_transform")
    changed_indices = sorted(
        [
            replace_flag_value(command, "--seed", str(transform["new_seed"])),
            replace_flag_value(
                command, "--output-dir", str(transform["new_output_dir"])
            ),
        ]
    )
    if changed_indices != transform["changed_value_indices"]:
        raise ValueError("changed command indices mismatch")
    derived_sha256 = sha256_bytes(canonical_json_bytes(command))
    if derived_sha256 != require_sha256(
        protocol["derived_command_sha256"], "derived_command_sha256"
    ):
        raise ValueError("derived command SHA-256 mismatch")
    if Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("frozen command Python differs from launcher environment")
    if Path(command[1]).resolve() != trainer_path:
        raise ValueError("frozen command trainer differs from bound trainer")
    if Path(command[command.index("--resume") + 1]).resolve() != checkpoint_path:
        raise ValueError("frozen resume checkpoint differs from binding")

    evaluation_seeds = protocol["fresh_evaluation_seeds"]
    if evaluation_seeds != {
        "behavior_old_retention": 202607337,
        "behavior_valid29": 202607338,
        "gold19_screen_control": 202607339,
        "gold19_screen_candidate": 202607340,
        "gold19_confirm_control": 202607341,
        "gold19_confirm_candidate": 202607342,
    }:
        raise ValueError("fresh_evaluation_seeds mismatch")
    if protocol["design"] != {
        "source_update": 456,
        "terminal_update": 464,
        "updates": list(range(457, 465)),
        "games_per_update": 64,
        "rollout_games": 512,
        "minibatch_size": 384,
        "reference_minibatch_size": 512,
        "strict_causal_attribution": False,
        "continued_training_confound": True,
    }:
        raise ValueError("design mismatch")
    if protocol["prohibited_actions"] != {
        "blind_result_or_replay_access": True,
        "day30_content_access": True,
        "day31_content_access": True,
        "network": True,
        "package": True,
        "submission": True,
        "upload": True,
    }:
        raise ValueError("prohibited_actions mismatch")

    expected_output = repo_path(protocol["output_dir"], "output_dir", must_exist=False)
    expected_terminal = repo_path(
        protocol["terminal_checkpoint"], "terminal_checkpoint", must_exist=False
    )
    expected_marker = repo_path(
        protocol["attempt_start_marker"], "attempt_start_marker", must_exist=False
    )
    expected_log = repo_path(protocol["log"], "log", must_exist=False)
    if Path(transform["new_output_dir"]).resolve() != expected_output:
        raise ValueError("derived output directory differs from protocol")
    if expected_terminal.parent.parent != expected_output:
        raise ValueError("terminal checkpoint is outside the frozen output directory")
    for path, label in (
        (expected_output, "output directory"),
        (expected_terminal, "terminal checkpoint"),
        (expected_marker, "attempt start marker"),
        (expected_log, "training log"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing {label}: {path}")
    stop_rules = protocol["stop_rules"]
    if stop_rules != {
        "attempts_authorized": 1,
        "no_retry_or_seed_selection": True,
        "terminal_update_only": 464,
    }:
        raise ValueError("stop_rules mismatch")
    return protocol, command, derived_sha256


def claim_and_run(
    protocol: dict[str, Any],
    command: Sequence[str],
    protocol_sha256: str,
    command_sha256: str,
) -> int:
    marker = repo_path(
        protocol["attempt_start_marker"], "attempt_start_marker", must_exist=False
    )
    log = repo_path(protocol["log"], "log", must_exist=False)
    marker.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    launcher_sha256 = sha256_file(Path(__file__).resolve())
    marker_payload = {
        "event": "local_preregistered_ppo_attempt_consumed",
        "schema_version": SCHEMA_VERSION,
        "seed": protocol["seed"],
        "protocol_sha256": protocol_sha256,
        "derived_command_sha256": command_sha256,
        "launcher_sha256": launcher_sha256,
    }
    marker_fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(marker_fd, canonical_json_bytes(marker_payload))
        os.fsync(marker_fd)
    finally:
        os.close(marker_fd)

    log_fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(log_fd, "wb", buffering=0) as handle:
        header = {
            "event": "local_preregistered_ppo_pre_exec_checks_passed",
            "seed": protocol["seed"],
            "protocol_sha256": protocol_sha256,
            "derived_command_sha256": command_sha256,
            "launcher_sha256": launcher_sha256,
            "cwd": str(REPO_ROOT),
        }
        handle.write(canonical_json_bytes(header))
        os.fsync(handle.fileno())
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        terminal = {
            "event": "local_preregistered_ppo_child_terminal",
            "seed": protocol["seed"],
            "return_code": completed.returncode,
        }
        handle.write(canonical_json_bytes(terminal))
        os.fsync(handle.fileno())
    return completed.returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol, command, command_sha256 = load_and_validate_protocol(
        args.protocol, args.expected_protocol_sha256
    )
    return claim_and_run(
        protocol,
        command,
        args.expected_protocol_sha256,
        command_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
