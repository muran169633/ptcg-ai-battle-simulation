#!/usr/bin/env python3
"""Build the one-shot execution protocol for the frozen G8-reverse PPO4 stage."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import exec_preregistered_ppo_u464_g8rev_to_u468_v1 as launch  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROTOCOL = REPO_ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_"
    "design202608090.ppo_stage_execution_preregistration.json"
)
OUTPUT = REPO_ROOT / (
    f"artifacts/{launch.BRANCH}.ppo_stage_execution_preregistration.json"
)
LAUNCHER_SHA256 = "3d5c66740c1e0653a9da2f769c9251b5fe3450712c206827e9bfc138cdecafd5"


def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(f"refusing existing protocol: {OUTPUT}")
    source = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    command = launch.lineage.derive_exact_command()
    command_hash = hashlib.sha256(
        launch.base.canonical_json_bytes(command)
    ).hexdigest()
    if command_hash != launch.DERIVED_COMMAND_SHA256:
        raise ValueError("derived command hash drifted")

    source_bindings = source["bindings"]
    bindings = {
        key: source_bindings[key]
        for key in launch.base.BINDING_KEYS
        if key in source_bindings and key not in launch.EXTRA_BINDINGS
    }
    bindings["launcher"] = {
        "path": str(launch.SELF_PATH.relative_to(REPO_ROOT)),
        "sha256": LAUNCHER_SHA256,
    }
    bindings["parent_checkpoint"] = {
        "path": str(launch.PARENT_PATH.relative_to(REPO_ROOT)),
        "sha256": launch.PARENT_SHA256,
    }
    for key, (path, digest, _) in launch.EXTRA_BINDINGS.items():
        bindings[key] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": digest,
        }
    if set(bindings) != set(launch.base.BINDING_KEYS):
        raise ValueError("constructed binding keys differ from launcher contract")

    payload = {
        "schema_version": launch.SCHEMA_VERSION,
        "status": "locked_after_launcher_and_before_training",
        "created_at_utc": "2026-08-01T13:30:00Z",
        "design_preregistration": {
            "path": str(launch.DESIGN_PATH.relative_to(REPO_ROOT)),
            "sha256": launch.DESIGN_SHA256,
        },
        "seed": 202607336,
        "command": command,
        "command_sha256": command_hash,
        "bindings": bindings,
        "expected_parent_state": launch.EXPECTED_PARENT_STATE,
        "output_dir": str(launch.OUTPUT_DIR.relative_to(REPO_ROOT)),
        "terminal_checkpoint": str(
            launch.TERMINAL_CHECKPOINT.relative_to(REPO_ROOT)
        ),
        "attempt_start_marker": str(launch.ATTEMPT_MARKER.relative_to(REPO_ROOT)),
        "log": str(launch.LOG_PATH.relative_to(REPO_ROOT)),
        "stop_rules": {
            "attempts_authorized": 1,
            "no_retry_or_seed_selection": True,
            "source_update": 464,
            "terminal_update": 468,
            "published_endpoints": [468],
        },
        "scope": {
            "local_only": True,
            "network": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"path": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
