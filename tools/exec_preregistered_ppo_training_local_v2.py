#!/usr/bin/env python3
"""Execute one exact, hash-bound local PPO command under a v2 protocol."""

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
SCHEMA_VERSION = "ptcg-local-preregistered-ppo-v2"
PROTOCOL_KEYS = frozenset(
    {
        "schema_version", "status", "seed", "command", "command_sha256",
        "launcher", "trainer", "source_checkpoint",
        "evaluation_protocol_source", "predecessor_decision",
        "fresh_evaluation_seeds", "design", "frozen_flag_values",
        "experimental_change", "prohibited_actions", "output_dir",
        "terminal_checkpoint", "attempt_start_marker", "log", "stop_rules",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: Any, label: str, *, must_exist: bool) -> Path:
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
    if must_exist and (not path.is_file() or path.is_symlink()):
        raise ValueError(f"{label} must be a regular symlink-free file")
    return path


def require_bound_file(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = repo_path(value["path"], f"{label}.path", must_exist=True)
    expected = value["sha256"]
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label}.sha256 is invalid")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def flag_value(command: Sequence[str], flag: str) -> str:
    indices = [index for index, token in enumerate(command) if token == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError(f"command must contain exactly one {flag}")
    return command[indices[0] + 1]


def load_and_validate(protocol_path: Path, expected_sha256: str) -> tuple[dict[str, Any], list[str]]:
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError("current working directory must equal repository root")
    protocol_path = repo_path(os.fspath(protocol_path), "protocol", must_exist=True)
    raw = protocol_path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError("protocol SHA-256 mismatch")
    protocol = json.loads(raw)
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_KEYS:
        raise ValueError("protocol keys differ from the frozen v2 schema")
    if protocol["schema_version"] != SCHEMA_VERSION or protocol["status"] != "locked_before_training":
        raise ValueError("protocol schema/status mismatch")
    if not isinstance(protocol["seed"], int) or protocol["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")

    launcher = require_bound_file(protocol["launcher"], "launcher")
    trainer = require_bound_file(protocol["trainer"], "trainer")
    source_checkpoint = require_bound_file(protocol["source_checkpoint"], "source_checkpoint")
    require_bound_file(protocol["evaluation_protocol_source"], "evaluation_protocol_source")
    require_bound_file(protocol["predecessor_decision"], "predecessor_decision")
    if launcher != Path(__file__).resolve():
        raise ValueError("launcher binding differs from executing file")

    command = protocol["command"]
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("command must be a non-empty string array")
    if sha256_bytes(canonical_json_bytes(command)) != protocol["command_sha256"]:
        raise ValueError("command SHA-256 mismatch")
    if Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("command Python differs from launcher environment")
    if Path(command[1]).resolve() != trainer:
        raise ValueError("command trainer differs from binding")
    if Path(flag_value(command, "--resume")).resolve() != source_checkpoint:
        raise ValueError("resume checkpoint differs from binding")

    design = protocol["design"]
    expected_design = {
        "source_update": 456, "terminal_update": 464,
        "updates": list(range(457, 465)), "games_per_update": 64,
        "rollout_games": 512,
        "reference_minibatch_size": 512, "ppo_epochs": 2,
        "strict_causal_attribution": True,
    }
    if not isinstance(design, dict):
        raise ValueError("design must be an object")
    minibatch_size = design.get("minibatch_size")
    if (
        not isinstance(minibatch_size, int)
        or minibatch_size < 128
        or minibatch_size > 512
        or {key: value for key, value in design.items() if key != "minibatch_size"}
        != expected_design
    ):
        raise ValueError("design mismatch")
    required_flags = dict(protocol["frozen_flag_values"])
    required_flags.update(
        {
            "--seed": str(protocol["seed"]), "--updates": "464",
            "--games-per-update": "64", "--ppo-epochs": "2",
            "--minibatch-size": str(minibatch_size),
        }
    )
    for flag, expected in required_flags.items():
        if flag_value(command, flag) != str(expected):
            raise ValueError(f"frozen command value mismatch for {flag}")
    change = protocol["experimental_change"]
    common_change = {
        "source_model": "confirmed_u456_incumbent",
        "same_seed_retry": False,
    }
    allowed_change = (
        change == {
            "axis": "minibatch_size", "reference": 512,
            "candidate": minibatch_size, **common_change,
        }
        or change == {
            "axis": "entropy_coefficient", "reference": 0.001,
            "candidate": 0.0, **common_change,
        }
    )
    if not allowed_change:
        raise ValueError("experimental_change mismatch")

    eval_seeds = protocol["fresh_evaluation_seeds"]
    expected_seed_keys = {
        "behavior_old_retention", "behavior_valid29", "gold19_screen_control",
        "gold19_screen_candidate", "gold19_confirm_control", "gold19_confirm_candidate",
    }
    if set(eval_seeds) != expected_seed_keys or not all(isinstance(x, int) for x in eval_seeds.values()):
        raise ValueError("fresh evaluation seed schema mismatch")
    all_seeds = [protocol["seed"], *eval_seeds.values()]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("training/evaluation seeds must be unique")

    output_dir = repo_path(protocol["output_dir"], "output_dir", must_exist=False)
    terminal = repo_path(protocol["terminal_checkpoint"], "terminal_checkpoint", must_exist=False)
    marker = repo_path(protocol["attempt_start_marker"], "attempt_start_marker", must_exist=False)
    log = repo_path(protocol["log"], "log", must_exist=False)
    if Path(flag_value(command, "--output-dir")).resolve() != output_dir:
        raise ValueError("command output directory mismatch")
    if terminal.parent.parent != output_dir:
        raise ValueError("terminal checkpoint is outside output directory")
    for path, label in ((output_dir, "output"), (terminal, "terminal"), (marker, "marker"), (log, "log")):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing {label}: {path}")
    if protocol["stop_rules"] != {"attempts_authorized": 1, "no_retry_or_seed_selection": True, "terminal_update_only": 464}:
        raise ValueError("stop rules mismatch")
    if protocol["prohibited_actions"] != {
        "blind_result_or_replay_access": True, "day30_content_access": True,
        "day31_content_access": True, "network": True, "package": True,
        "submission": True, "upload": True,
    }:
        raise ValueError("prohibited actions mismatch")
    return protocol, command


def claim_and_run(protocol: dict[str, Any], command: Sequence[str], protocol_sha256: str) -> int:
    marker = repo_path(protocol["attempt_start_marker"], "marker", must_exist=False)
    log = repo_path(protocol["log"], "log", must_exist=False)
    marker_payload = {
        "event": "local_preregistered_ppo_v2_attempt_consumed",
        "schema_version": SCHEMA_VERSION, "seed": protocol["seed"],
        "protocol_sha256": protocol_sha256,
        "command_sha256": protocol["command_sha256"],
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
    }
    marker_fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(marker_fd, canonical_json_bytes(marker_payload))
        os.fsync(marker_fd)
    finally:
        os.close(marker_fd)
    log_fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(log_fd, "wb", buffering=0) as handle:
        handle.write(canonical_json_bytes({
            "event": "local_preregistered_ppo_v2_pre_exec_checks_passed",
            "seed": protocol["seed"], "protocol_sha256": protocol_sha256,
            "command_sha256": protocol["command_sha256"],
            "launcher_sha256": marker_payload["launcher_sha256"],
            "cwd": str(REPO_ROOT),
        }))
        os.fsync(handle.fileno())
        completed = subprocess.run(list(command), cwd=REPO_ROOT, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, check=False)
        handle.write(canonical_json_bytes({
            "event": "local_preregistered_ppo_v2_child_terminal",
            "seed": protocol["seed"], "return_code": completed.returncode,
        }))
        os.fsync(handle.fileno())
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args(argv)
    protocol, command = load_and_validate(args.protocol, args.expected_protocol_sha256)
    return claim_and_run(protocol, command, args.expected_protocol_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
