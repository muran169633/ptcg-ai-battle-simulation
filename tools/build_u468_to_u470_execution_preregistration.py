#!/usr/bin/env python3
"""Create the one-shot execution lock for the U468-to-U470 continuation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPO_ROOT / "tools/exec_preregistered_ppo_u468_to_u470.py"
LAUNCHER_SHA256 = "b762ece386bda253423617d2701673b3e27dc42fd7d74dbbaffdc308772e0f20"
OUTPUT_PATH = REPO_ROOT / (
    "artifacts/ppo_u468_exactresume_ppo2_u469u470_design202608110."
    "execution_preregistration.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def binding(path: Path, expected_sha256: str | None = None) -> dict[str, str]:
    path = path.resolve()
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"binding SHA-256 mismatch: {path}")
    return {"path": relative(path), "sha256": observed}


def load_launcher() -> Any:
    if sha256_file(LAUNCHER_PATH) != LAUNCHER_SHA256:
        raise ValueError("launcher SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("u468_to_u470_launcher", LAUNCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_protocol(module: Any) -> dict[str, Any]:
    base = module.base
    command = module.derive_exact_command()
    if module.canonical_command_sha256(command) != module.DERIVED_COMMAND_SHA256:
        raise ValueError("derived command SHA-256 mismatch")
    bindings = {
        "launcher": binding(module.WRAPPER_PATH, LAUNCHER_SHA256),
        "trainer": binding(base.TRAINER_PATH, base.TRAINER_SHA256),
        "train_bc_orbit": binding(
            base.TRAIN_BC_ORBIT_PATH, base.TRAIN_BC_ORBIT_SHA256
        ),
        "cg_init": binding(base.CG_INIT_PATH, base.CG_INIT_SHA256),
        "cg_sim": binding(base.CG_SIM_PATH, base.CG_SIM_SHA256),
        "cg_lib": binding(base.CG_LIB_PATH, base.CG_LIB_SHA256),
        "parent_checkpoint": binding(module.PARENT_PATH, module.PARENT_SHA256),
        "general_bc": binding(base.GENERAL_BC_PATH, base.GENERAL_BC_SHA256),
        "bc_replay_archive": binding(
            base.BC_REPLAY_PATH, base.BC_REPLAY_SHA256
        ),
        "candidate_deck": binding(
            base.CANDIDATE_DECK_PATH, base.CANDIDATE_DECK_SHA256
        ),
        "training_league_manifest": binding(
            base.LEAGUE_MANIFEST_PATH, base.LEAGUE_MANIFEST_SHA256
        ),
        "source_command_preregistration": binding(
            module.SOURCE_COMMAND_PROTOCOL,
            module.SOURCE_COMMAND_PROTOCOL_SHA256,
        ),
        "u464_training_integrity_decision": binding(
            base.U464_TRAINING_DECISION_PATH,
            base.U464_TRAINING_DECISION_SHA256,
        ),
        "u464_gold19_decision": binding(
            base.U464_GOLD_DECISION_PATH,
            base.U464_GOLD_DECISION_SHA256,
        ),
    }
    for key, (path, expected_sha256, _) in module.EXTRA_BINDINGS.items():
        bindings[key] = binding(path, expected_sha256)
    if set(bindings) != base.BINDING_KEYS:
        raise ValueError("generated binding keys differ from launcher schema")
    return {
        "schema_version": module.SCHEMA_VERSION,
        "status": "locked_after_launcher_and_before_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_preregistration": binding(
            module.DESIGN_PATH, module.DESIGN_SHA256
        ),
        "seed": 202607336,
        "command": command,
        "command_sha256": module.DERIVED_COMMAND_SHA256,
        "bindings": bindings,
        "expected_parent_state": module.EXPECTED_PARENT_STATE,
        "output_dir": relative(module.OUTPUT_DIR),
        "terminal_checkpoint": relative(module.TERMINAL_CHECKPOINT),
        "attempt_start_marker": relative(module.ATTEMPT_MARKER),
        "log": relative(module.LOG_PATH),
        "stop_rules": {
            "attempts_authorized": 1,
            "no_retry_or_seed_selection": True,
            "source_update": 468,
            "terminal_update": 470,
            "published_endpoints": [469, 470],
            "no_behavior_evaluation_before_both_endpoints_exist": True,
        },
        "scope": {
            "local_only": True,
            "network": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError("current working directory must equal repository root")
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if output.resolve() != OUTPUT_PATH:
        raise ValueError("output path differs from frozen target")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    module = load_launcher()
    protocol = build_protocol(module)
    payload = (
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeError("short protocol write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(
        json.dumps(
            {
                "path": relative(output),
                "sha256": sha256_file(output),
                "command_sha256": module.DERIVED_COMMAND_SHA256,
                "command_tokens": len(protocol["command"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
