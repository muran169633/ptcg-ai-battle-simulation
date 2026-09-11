#!/usr/bin/env python3
"""Audit or execute the exact full-state U468-to-U470 PPO continuation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

import torch


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

BRANCH_NAME = "ppo_u468_exactresume_ppo2_u469u470_design202608110"
DESIGN_PATH = REPO_ROOT / f"artifacts/{BRANCH_NAME}.design_preregistration.json"
DESIGN_SHA256 = "ebe4d1f789aad1ff1c4e0d4f110db87ee83a4b8cd34aea09dd9cd6b350f23035"
SOURCE_COMMAND_PROTOCOL = REPO_ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090."
    "ppo_stage_execution_preregistration.json"
)
SOURCE_COMMAND_PROTOCOL_SHA256 = (
    "078de15009db3acbcb9f119ffab0e18acb1175fc726d9a7f55112b5fd89ab9bf"
)
SOURCE_COMMAND_SHA256 = (
    "634073b38bbd492e97cf822893b50497db6ab68c10dc6e8bf91a55de07bd495e"
)
SOURCE_TRAINING_DECISION_PATH = REPO_ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090."
    "ppo_stage_training_integrity_decision.json"
)
SOURCE_TRAINING_DECISION_SHA256 = (
    "350c52c830c65d4447e5fb841193660720e6d88b897d319bb5fb3bc5698d1ffe"
)
CLOSED_ACTOR6_DECISION_PATH = REPO_ROOT / (
    "artifacts/ppo_u468_beta1157_trainhard_actor6_mix102_77_77_p1p2_"
    "design202608101.specialist_behavior/specialist_selection_decision.json"
)
CLOSED_ACTOR6_DECISION_SHA256 = (
    "84f4741c4ffa3d9a13b815f2f530f00584fa23d4cd57e374c94cef8e8d4c5ae5"
)
PARENT_PATH = REPO_ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
PARENT_SHA256 = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / BRANCH_NAME
OUTPUT_DIR = OUTPUT_ROOT / "B_gold_league/seed-202607336"
ENDPOINT_469 = OUTPUT_DIR / "checkpoints/update-0469.pt"
TERMINAL_CHECKPOINT = OUTPUT_DIR / "checkpoints/update-0470.pt"
ATTEMPT_MARKER = REPO_ROOT / (
    ".ptcg-u468-exactresume-ppo2-u470-attempt-202607336-202608110.json"
)
LOG_PATH = REPO_ROOT / f"artifacts/{BRANCH_NAME}.log"

SCHEMA_VERSION = "ptcg-u468-to-u470-resume-all-execution-preregistration-v1"
DERIVED_COMMAND_SHA256 = (
    "2f977f1754143d5bc4de2edd722b723d316df2feb823ee2e5853edf95ed977fa"
)
DERIVED_COMMAND_TOKENS = 177

EXPECTED_PARENT_STATE = {
    "update": 468,
    "runtime_model_state_sha256": (
        "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
    ),
    "ppo_optimizer_nested_sha256": (
        "7b9be0b2bff650ad7d6e96e46b80b8aae55a0f96cbe1f7919782301398633087"
    ),
    "ppo_optimizer_state_count": 28,
    "ppo_optimizer_step": 412,
    "bc_replay_optimizer_nested_sha256": (
        "d92a040c4805141a7ce08f9128a9dc23128b09a2d0cbb6af86f07d0a711973f8"
    ),
    "bc_replay_optimizer_state_count": 24,
    "bc_replay_optimizer_step": 32,
    "opponent_quota_nested_sha256": (
        "77c2369d0ece97724afbee7034ca012e5ed974878b1c6f3ca6dae96e72e4a743"
    ),
    "opponent_quota_observed_games": 768,
    "opponent_quota_last_refresh_update": 466,
}

EXTRA_BINDINGS: dict[str, tuple[Path, str, str]] = {
    "base_launcher": (
        BASE_LAUNCHER_PATH,
        BASE_LAUNCHER_SHA256,
        "hardened base launcher",
    ),
    "source_ppo_training_integrity_decision": (
        SOURCE_TRAINING_DECISION_PATH,
        SOURCE_TRAINING_DECISION_SHA256,
        "source PPO training integrity decision",
    ),
    "closed_actor6_branch_decision": (
        CLOSED_ACTOR6_DECISION_PATH,
        CLOSED_ACTOR6_DECISION_SHA256,
        "closed actor6 branch decision",
    ),
}


def canonical_command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(base.canonical_json_bytes(list(command))).hexdigest()


def derive_exact_command() -> list[str]:
    base.require_regular(SOURCE_COMMAND_PROTOCOL, "source command protocol")
    if base.sha256_file(SOURCE_COMMAND_PROTOCOL) != SOURCE_COMMAND_PROTOCOL_SHA256:
        raise ValueError("source command protocol SHA-256 mismatch")
    source_protocol = json.loads(SOURCE_COMMAND_PROTOCOL.read_text())
    source = source_protocol.get("command")
    if not isinstance(source, list) or not all(
        isinstance(token, str) for token in source
    ):
        raise ValueError("source command must be a string array")
    if canonical_command_sha256(source) != SOURCE_COMMAND_SHA256:
        raise ValueError("source command SHA-256 mismatch")
    command = list(source)
    replacements = {
        "--output-dir": str(OUTPUT_DIR),
        "--updates": "470",
        "--resume": str(PARENT_PATH),
        "--checkpoint-interval": "1",
    }
    for flag, value in replacements.items():
        base.replace_flag_value(command, flag, value)
    if len(command) != DERIVED_COMMAND_TOKENS:
        raise ValueError("derived command token count mismatch")
    if canonical_command_sha256(command) != DERIVED_COMMAND_SHA256:
        raise ValueError("derived command SHA-256 mismatch")
    return command


def audit_parent_state() -> dict[str, Any]:
    parent = torch.load(PARENT_PATH, map_location="cpu", weights_only=False)
    general_bc = torch.load(base.GENERAL_BC_PATH, map_location="cpu", weights_only=False)
    if int(parent.get("update", -1)) != EXPECTED_PARENT_STATE["update"]:
        raise ValueError("parent update mismatch")
    model = base.ppo.instantiate_model_from_bc(general_bc, torch.device("cpu"))
    model.load_state_dict(parent["model_state_dict"])
    model_hash = base.ppo.model_state_sha256(model)
    if model_hash != EXPECTED_PARENT_STATE["runtime_model_state_sha256"]:
        raise ValueError("parent runtime model hash mismatch")

    raw_config = parent.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("parent config is missing")
    if int(raw_config.get("seed", -1)) != 202607336:
        raise ValueError("parent quota identity seed mismatch")
    actor, value, manifest = base.ppo.configure_trainable_scope(
        model, str(raw_config["trainable_scope"])
    )
    base.ppo.validate_optimizer_resume_compatibility(
        parent,
        str(raw_config["trainable_scope"]),
        manifest,
        str(raw_config["advantage_normalization"]),
        str(raw_config["ppo_objective"]),
        None,
        float(raw_config["policy_temperature"]),
        str(raw_config["actor_reduction"]),
        float(raw_config["value_trunk_gradient_scale"]),
        str(raw_config["actor_value_gradient_mode"]),
    )

    ppo_state = parent.get("optimizer_state_dict")
    replay_state = parent.get("bc_replay_optimizer_state_dict")
    quota = parent.get("opponent_quota_state")
    if not isinstance(ppo_state, dict) or not isinstance(replay_state, dict):
        raise ValueError("parent optimizer states are missing")
    if not isinstance(quota, dict):
        raise ValueError("parent quota state is missing")
    checks = (
        (base.nested_sha256(ppo_state), "ppo_optimizer_nested_sha256"),
        (base.nested_sha256(replay_state), "bc_replay_optimizer_nested_sha256"),
        (base.nested_sha256(quota), "opponent_quota_nested_sha256"),
    )
    for actual, key in checks:
        if actual != EXPECTED_PARENT_STATE[key]:
            raise ValueError(f"parent {key} mismatch")
    if len(ppo_state.get("state", {})) != EXPECTED_PARENT_STATE["ppo_optimizer_state_count"]:
        raise ValueError("parent PPO optimizer state count mismatch")
    if base.state_steps(ppo_state) != {EXPECTED_PARENT_STATE["ppo_optimizer_step"]}:
        raise ValueError("parent PPO optimizer step mismatch")
    if len(replay_state.get("state", {})) != EXPECTED_PARENT_STATE["bc_replay_optimizer_state_count"]:
        raise ValueError("parent replay optimizer state count mismatch")
    if base.state_steps(replay_state) != {EXPECTED_PARENT_STATE["bc_replay_optimizer_step"]}:
        raise ValueError("parent replay optimizer step mismatch")
    if base.quota_observed_games(quota) != EXPECTED_PARENT_STATE["opponent_quota_observed_games"]:
        raise ValueError("parent quota observed-game count mismatch")
    if int(quota.get("last_refresh_update", -1)) != EXPECTED_PARENT_STATE["opponent_quota_last_refresh_update"]:
        raise ValueError("parent quota refresh update mismatch")

    optimizer = torch.optim.AdamW(
        [
            {"params": actor, "lr": float(raw_config["learning_rate"])},
            {"params": value, "lr": float(raw_config["value_learning_rate"])},
        ],
        eps=1e-5,
        weight_decay=float(raw_config["weight_decay"]),
    )
    replay_optimizer = torch.optim.AdamW(
        actor,
        lr=float(raw_config["learning_rate"])
        * float(raw_config["bc_replay_lr_scale"]),
        eps=1e-5,
        weight_decay=float(raw_config["weight_decay"]),
    )
    optimizer.load_state_dict(ppo_state)
    replay_optimizer.load_state_dict(replay_state)
    return {
        "runtime_model_state_sha256": model_hash,
        "ppo_optimizer_state_count": len(optimizer.state_dict()["state"]),
        "ppo_optimizer_steps": sorted(base.state_steps(optimizer.state_dict())),
        "bc_replay_optimizer_state_count": len(
            replay_optimizer.state_dict()["state"]
        ),
        "bc_replay_optimizer_steps": sorted(
            base.state_steps(replay_optimizer.state_dict())
        ),
        "quota_observed_games": base.quota_observed_games(quota),
        "quota_last_refresh_update": int(quota["last_refresh_update"]),
        "resume_compatibility": "passed",
    }


original_command_input_paths = base.command_input_paths
original_assert_locked_input_bindings = base.assert_locked_input_bindings


def command_input_paths(command: Sequence[str]) -> list[Path]:
    paths = original_command_input_paths(command)
    for path, _, _ in EXTRA_BINDINGS.values():
        paths.append(base.require_regular(path, "extra locked input"))
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
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError("current working directory must equal repository root")
    protocol_path = base.require_regular(protocol_path, "execution protocol")
    raw = protocol_path.read_bytes()
    if base.sha256_bytes(raw) != expected_protocol_sha256:
        raise ValueError("execution protocol SHA-256 mismatch")
    protocol = json.loads(raw)
    if not isinstance(protocol, dict) or set(protocol) != base.PROTOCOL_KEYS:
        raise ValueError("execution protocol keys differ from the frozen schema")
    if protocol["schema_version"] != SCHEMA_VERSION:
        raise ValueError("execution protocol schema mismatch")
    if protocol["status"] != "locked_after_launcher_and_before_training":
        raise ValueError("execution protocol status mismatch")
    if protocol["seed"] != 202607336:
        raise ValueError("execution protocol seed mismatch")

    base.require_bound_file(
        protocol["design_preregistration"], DESIGN_PATH, DESIGN_SHA256, "design"
    )
    bindings = protocol["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != base.BINDING_KEYS:
        raise ValueError("execution binding keys mismatch")
    fixed_bindings = {
        "launcher": (WRAPPER_PATH, base.sha256_file(WRAPPER_PATH), "launcher"),
        "trainer": (base.TRAINER_PATH, base.TRAINER_SHA256, "trainer"),
        "train_bc_orbit": (
            base.TRAIN_BC_ORBIT_PATH,
            base.TRAIN_BC_ORBIT_SHA256,
            "train BC orbit dependency",
        ),
        "cg_init": (base.CG_INIT_PATH, base.CG_INIT_SHA256, "cg init"),
        "cg_sim": (base.CG_SIM_PATH, base.CG_SIM_SHA256, "cg simulator"),
        "cg_lib": (base.CG_LIB_PATH, base.CG_LIB_SHA256, "cg native library"),
        "parent_checkpoint": (PARENT_PATH, PARENT_SHA256, "parent checkpoint"),
        "general_bc": (base.GENERAL_BC_PATH, base.GENERAL_BC_SHA256, "general BC"),
        "bc_replay_archive": (
            base.BC_REPLAY_PATH,
            base.BC_REPLAY_SHA256,
            "BC replay archive",
        ),
        "candidate_deck": (
            base.CANDIDATE_DECK_PATH,
            base.CANDIDATE_DECK_SHA256,
            "candidate deck",
        ),
        "training_league_manifest": (
            base.LEAGUE_MANIFEST_PATH,
            base.LEAGUE_MANIFEST_SHA256,
            "training league manifest",
        ),
        "source_command_preregistration": (
            SOURCE_COMMAND_PROTOCOL,
            SOURCE_COMMAND_PROTOCOL_SHA256,
            "source command protocol",
        ),
        "u464_training_integrity_decision": (
            base.U464_TRAINING_DECISION_PATH,
            base.U464_TRAINING_DECISION_SHA256,
            "base lineage training decision",
        ),
        "u464_gold19_decision": (
            base.U464_GOLD_DECISION_PATH,
            base.U464_GOLD_DECISION_SHA256,
            "base lineage Gold decision",
        ),
        **EXTRA_BINDINGS,
    }
    for key, (path, expected_sha256, label) in fixed_bindings.items():
        base.require_bound_file(bindings[key], path, expected_sha256, label)

    command = protocol["command"]
    if command != derive_exact_command():
        raise ValueError("protocol command differs from exact derived command")
    if protocol["command_sha256"] != DERIVED_COMMAND_SHA256:
        raise ValueError("protocol command SHA-256 field mismatch")
    if Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("command Python differs from launcher environment")
    if Path(command[1]).resolve() != base.TRAINER_PATH:
        raise ValueError("command trainer differs from binding")
    expected_flags = {
        "--resume": str(PARENT_PATH),
        "--resume-learner-weights": "resume",
        "--seed": "202607336",
        "--updates": "470",
        "--checkpoint-interval": "1",
        "--eval-interval": "464",
    }
    for flag, value in expected_flags.items():
        if base.flag_value(command, flag) != value:
            raise ValueError(f"command {flag} mismatch")
    for forbidden in (
        "--reset-optimizer-on-resume",
        "--reset-opponent-quota-on-resume",
    ):
        if forbidden in command:
            raise ValueError(f"forbidden reset flag present: {forbidden}")
    if protocol["expected_parent_state"] != EXPECTED_PARENT_STATE:
        raise ValueError("expected parent state schema mismatch")

    output = Path(protocol["output_dir"])
    terminal = Path(protocol["terminal_checkpoint"])
    marker = Path(protocol["attempt_start_marker"])
    log = Path(protocol["log"])
    for name, path in (
        ("output", output),
        ("terminal", terminal),
        ("marker", marker),
        ("log", log),
    ):
        if not path.is_absolute():
            path = REPO_ROOT / path
        expected = {
            "output": OUTPUT_DIR,
            "terminal": TERMINAL_CHECKPOINT,
            "marker": ATTEMPT_MARKER,
            "log": LOG_PATH,
        }[name]
        if path.resolve() != expected:
            raise ValueError(f"execution {name} binding mismatch")
    if Path(base.flag_value(command, "--output-dir")).resolve() != OUTPUT_DIR:
        raise ValueError("command output directory mismatch")
    for path, label in (
        (OUTPUT_ROOT, "output root"),
        (OUTPUT_DIR, "output directory"),
        (ENDPOINT_469, "U469 checkpoint"),
        (TERMINAL_CHECKPOINT, "U470 checkpoint"),
        (ATTEMPT_MARKER, "attempt marker"),
        (LOG_PATH, "training log"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing {label}: {path}")
    if protocol["stop_rules"] != {
        "attempts_authorized": 1,
        "no_retry_or_seed_selection": True,
        "source_update": 468,
        "terminal_update": 470,
        "published_endpoints": [469, 470],
        "no_behavior_evaluation_before_both_endpoints_exist": True,
    }:
        raise ValueError("execution stop rules mismatch")
    if protocol["scope"] != {
        "local_only": True,
        "network": False,
        "package": False,
        "upload": False,
        "submission": False,
    }:
        raise ValueError("execution scope mismatch")

    return protocol, command, {
        "status": "passed",
        "protocol_sha256": expected_protocol_sha256,
        "command_sha256": DERIVED_COMMAND_SHA256,
        "command_tokens": len(command),
        "parent": audit_parent_state(),
        "training_inputs": base.audit_training_inputs(command),
        "published_endpoints": [469, 470],
        "targets_absent": True,
        "writes_performed": False,
        "rollouts_performed": False,
    }


# Rebind the hardened base implementation to this immutable continuation.
base.__file__ = str(WRAPPER_PATH)
base.SCHEMA_VERSION = SCHEMA_VERSION
base.BINDING_KEYS = frozenset(set(base.BINDING_KEYS) | set(EXTRA_BINDINGS))
base.DESIGN_PATH = DESIGN_PATH
base.DESIGN_SHA256 = DESIGN_SHA256
base.SOURCE_COMMAND_PROTOCOL = SOURCE_COMMAND_PROTOCOL
base.SOURCE_COMMAND_PROTOCOL_SHA256 = SOURCE_COMMAND_PROTOCOL_SHA256
base.SOURCE_COMMAND_SHA256 = SOURCE_COMMAND_SHA256
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
base.audit_parent_state = audit_parent_state
base.command_input_paths = command_input_paths
base.assert_locked_input_bindings = assert_locked_input_bindings
base.load_and_validate_protocol = load_and_validate_protocol


if __name__ == "__main__":
    raise SystemExit(base.main())
