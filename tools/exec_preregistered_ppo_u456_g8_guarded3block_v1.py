#!/usr/bin/env python3
"""Execute one block of the frozen U456-G8 guarded 3xPPO4 lineage."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

AUTHENTICATED_BASE = TOOLS_ROOT / "exec_preregistered_ppo_u456_g8fresh_to_u472_v1.py"
AUTHENTICATED_BASE_SHA256 = "eff06340252f9801c64a4ccac78fc97c6611041b8d17eb5b2e251cfb32a93abc"
_base_stat = os.lstat(AUTHENTICATED_BASE)
if stat.S_ISLNK(_base_stat.st_mode) or not stat.S_ISREG(_base_stat.st_mode):
    raise RuntimeError("authenticated isolated launcher must be a regular file")
if hashlib.sha256(AUTHENTICATED_BASE.read_bytes()).hexdigest() != AUTHENTICATED_BASE_SHA256:
    raise RuntimeError("authenticated isolated launcher SHA-256 mismatch")

import exec_preregistered_ppo_u456_g8fresh_to_u472_v1 as lineage  # noqa: E402


base = lineage.base
ppo = lineage.ppo
SELF_PATH = Path(__file__).resolve()
BRANCH = "ppo_u456_g8_guarded3x4x96_p12_design202608141"
SEED = 202608141
SCHEMA_VERSION = "ptcg-u456-g8-guarded3xppo4-block-isolated-execution-v1"
MASTER_PATH = REPO_ROOT / f"artifacts/{BRANCH}.master_preregistration.json"
MASTER_SHA256 = "e5dd52ba5e28f5672ea0199ba3b55c4da3fd8a6f45ebd6062b77fca41a5d37c7"
SOURCE_PROTOCOL = lineage.SOURCE_PROTOCOL
SOURCE_PROTOCOL_SHA256 = lineage.SOURCE_PROTOCOL_SHA256
SOURCE_COMMAND_SHA256 = lineage.SOURCE_COMMAND_SHA256
PPO_STAGE_ROOT = REPO_ROOT / f"artifacts/{BRANCH}/ppo_stage"

G8_PARENT = REPO_ROOT / (
    "artifacts/ppo_u456_g8fresh_ppo16x96_p12_design202608130/"
    "general_stage/general-bc-0008.pt"
)
G8_PARENT_SHA256 = "07f1645d391d3380eb77b08f583e9e8bec96c2ebcf34c57dbc27a1657cb1c6c9"
G8_DECISION = REPO_ROOT / (
    "artifacts/ppo_u456_g8fresh_ppo16x96_p12_design202608130."
    "g8_training_integrity_decision.json"
)
G8_DECISION_SHA256 = "79f409edd1ba8cef7c53b104571ccd9584058a3ccd8ec8747779636996ee41db"

FIXED_QUOTAS = dict(lineage.FIXED_QUOTAS)
BLOCKS: dict[int, dict[str, Any]] = {
    1: {
        "first": 457,
        "terminal": 460,
        "parent_update": 456,
        "parent_path": G8_PARENT,
        "parent_decision_path": G8_DECISION,
        "reset": True,
        "tokens": 179,
        "command_sha256": "babb98040af0b29de10068767ab5fa2130ae6471bf1f815e1d787bc60524c05d",
        "anchor_envelope": 0.0100,
        "terminal_role": "transport_only",
    },
    2: {
        "first": 461,
        "terminal": 464,
        "parent_update": 460,
        "parent_path": PPO_STAGE_ROOT / "block1/B_gold_league/seed-202608141/checkpoints/update-0460.pt",
        "parent_decision_path": REPO_ROOT / f"artifacts/{BRANCH}.block1_training_integrity_decision.json",
        "reset": False,
        "tokens": 177,
        "command_sha256": "4f61a9fbc5aac33f55695e66695588e6c937c52f2450d67834f25f01fbd53ade",
        "anchor_envelope": 0.0125,
        "terminal_role": "transport_only",
    },
    3: {
        "first": 465,
        "terminal": 468,
        "parent_update": 464,
        "parent_path": PPO_STAGE_ROOT / "block2/B_gold_league/seed-202608141/checkpoints/update-0464.pt",
        "parent_decision_path": REPO_ROOT / f"artifacts/{BRANCH}.block2_training_integrity_decision.json",
        "reset": False,
        "tokens": 177,
        "command_sha256": "5253d8d6c08031ea152a08072f5718bcd48abe391aa7267e7b69e1fb8b95e95d",
        "anchor_envelope": 0.0150,
        "terminal_role": "sole_PPO_candidate",
    },
}

PROTOCOL_KEYS = {
    "schema_version",
    "status",
    "block",
    "seed",
    "command",
    "command_sha256",
    "bindings",
    "expected_parent_state",
    "output_dir",
    "terminal_checkpoint",
    "attempt_start_marker",
    "log",
    "block_contract",
    "scope",
}

ACTIVE_BLOCK = 0
ACTIVE_SPEC: dict[str, Any] = {}
ACTIVE_EXPECTED_PARENT_STATE: dict[str, Any] = {}


def block_paths(block: int) -> tuple[Path, Path, Path, Path, Path]:
    spec = BLOCKS[block]
    output_root = PPO_STAGE_ROOT / f"block{block}"
    output_dir = output_root / f"B_gold_league/seed-{SEED}"
    terminal = output_dir / f"checkpoints/update-{spec['terminal']:04d}.pt"
    attempt = REPO_ROOT / f".ptcg-u456-g8-guarded-block{block}-attempt-{SEED}-202608141.json"
    log = REPO_ROOT / f"artifacts/{BRANCH}.block{block}.log"
    return output_root, output_dir, terminal, attempt, log


def derive_exact_command(block: int | None = None) -> list[str]:
    block = ACTIVE_BLOCK if block is None else block
    spec = BLOCKS[block]
    source = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))["binding"]["command"]
    if base.sha256_bytes(base.canonical_json_bytes(source)) != SOURCE_COMMAND_SHA256:
        raise ValueError("source command SHA-256 mismatch")
    _, output_dir, _, _, _ = block_paths(block)
    command = list(source)
    replacements = {
        "--output-dir": str(output_dir),
        "--updates": str(spec["terminal"]),
        "--games-per-update": "96",
        "--minibatch-size": "512",
        "--learning-rate": "2.4e-05",
        "--schedule-start-update": str(spec["first"]),
        "--bc-kl-start": "0.016",
        "--bc-kl-end": "0.012",
        "--eval-interval": str(spec["terminal"]),
        "--checkpoint-interval": str(spec["terminal"]),
        "--seed": str(SEED),
        "--resume": str(spec["parent_path"]),
        "--bc-replay-lr-scale": "0.075",
    }
    for flag, value in replacements.items():
        base.replace_flag_value(command, flag, value)
    for index, token in enumerate(command):
        if token == "--opponent-base-quota":
            command[index + 2] = str(FIXED_QUOTAS[command[index + 1]])
    if not spec["reset"]:
        command.remove("--reset-optimizer-on-resume")
        command.remove("--reset-opponent-quota-on-resume")
    if len(command) != spec["tokens"]:
        raise ValueError("derived block command token count mismatch")
    if base.sha256_bytes(base.canonical_json_bytes(command)) != spec["command_sha256"]:
        raise ValueError("derived block command SHA-256 mismatch")
    return command


def configure_active_block(protocol: dict[str, Any]) -> None:
    global ACTIVE_BLOCK, ACTIVE_SPEC, ACTIVE_EXPECTED_PARENT_STATE
    block = int(protocol.get("block", -1))
    if block not in BLOCKS:
        raise ValueError("protocol block must be 1, 2, or 3")
    spec = BLOCKS[block]
    bindings = protocol.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("protocol bindings are missing")
    parent_binding = bindings.get("parent_checkpoint")
    decision_binding = bindings.get("parent_integrity_decision")
    for value, expected_path, label in (
        (parent_binding, spec["parent_path"], "parent checkpoint"),
        (decision_binding, spec["parent_decision_path"], "parent integrity decision"),
    ):
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError(f"{label} binding shape mismatch")
        actual_path = Path(value["path"])
        if not actual_path.is_absolute():
            actual_path = REPO_ROOT / actual_path
        if actual_path.resolve() != expected_path.resolve():
            raise ValueError(f"{label} path mismatch")
        if not isinstance(value["sha256"], str) or len(value["sha256"]) != 64:
            raise ValueError(f"{label} digest shape mismatch")

    ACTIVE_BLOCK = block
    ACTIVE_SPEC = spec
    ACTIVE_EXPECTED_PARENT_STATE = protocol["expected_parent_state"]
    output_root, output_dir, terminal, attempt, log = block_paths(block)
    lineage.SELF_PATH = SELF_PATH
    lineage.SCHEMA_VERSION = SCHEMA_VERSION
    lineage.BRANCH = BRANCH
    lineage.SEED = SEED
    lineage.MASTER_PATH = MASTER_PATH
    lineage.MASTER_SHA256 = MASTER_SHA256
    lineage.PARENT_PATH = spec["parent_path"]
    lineage.PARENT_SHA256 = parent_binding["sha256"]
    lineage.OUTPUT_ROOT = output_root
    lineage.OUTPUT_DIR = output_dir
    lineage.TERMINAL_CHECKPOINT = terminal
    lineage.ATTEMPT_MARKER = attempt
    lineage.LOG_PATH = log
    lineage.DERIVED_COMMAND_SHA256 = spec["command_sha256"]
    lineage.DERIVED_COMMAND_TOKENS = spec["tokens"]
    lineage.EXPECTED_PARENT_STATE = ACTIVE_EXPECTED_PARENT_STATE
    lineage.BINDINGS = {
        "authenticated_isolated_launcher": (AUTHENTICATED_BASE, AUTHENTICATED_BASE_SHA256),
        "launcher_base": (lineage.BASE_PATH, lineage.BASE_SHA256),
        "trainer": (base.TRAINER_PATH, base.TRAINER_SHA256),
        "train_bc_orbit": (base.TRAIN_BC_ORBIT_PATH, base.TRAIN_BC_ORBIT_SHA256),
        "cg_init": (base.CG_INIT_PATH, base.CG_INIT_SHA256),
        "cg_sim": (base.CG_SIM_PATH, base.CG_SIM_SHA256),
        "cg_lib": (base.CG_LIB_PATH, base.CG_LIB_SHA256),
        "master_preregistration": (MASTER_PATH, MASTER_SHA256),
        "parent_integrity_decision": (spec["parent_decision_path"], decision_binding["sha256"]),
        "parent_checkpoint": (spec["parent_path"], parent_binding["sha256"]),
        "general_bc": (base.GENERAL_BC_PATH, base.GENERAL_BC_SHA256),
        "bc_replay_archive": (base.BC_REPLAY_PATH, base.BC_REPLAY_SHA256),
        "candidate_deck": (base.CANDIDATE_DECK_PATH, base.CANDIDATE_DECK_SHA256),
        "training_league_manifest": (base.LEAGUE_MANIFEST_PATH, base.LEAGUE_MANIFEST_SHA256),
        "source_command_preregistration": (SOURCE_PROTOCOL, SOURCE_PROTOCOL_SHA256),
    }


def parent_audit() -> dict[str, Any]:
    parent = torch.load(ACTIVE_SPEC["parent_path"], map_location="cpu", weights_only=False)
    bc = torch.load(base.GENERAL_BC_PATH, map_location="cpu", weights_only=False)
    if int(parent.get("update", -1)) != ACTIVE_SPEC["parent_update"]:
        raise ValueError("parent update mismatch")
    model = ppo.instantiate_model_from_checkpoint(parent, bc, torch.device("cpu"))
    ppo_steps = base.state_steps(parent["optimizer_state_dict"])
    replay_steps = base.state_steps(parent["bc_replay_optimizer_state_dict"])
    if len(ppo_steps) != 1 or len(replay_steps) != 1:
        raise ValueError("parent optimizer steps are not uniform")
    observed = {
        "update": int(parent["update"]),
        "runtime_model_state_sha256": ppo.model_state_sha256(model),
        "ppo_optimizer_nested_sha256": base.nested_sha256(parent["optimizer_state_dict"]),
        "ppo_optimizer_state_count": len(parent["optimizer_state_dict"]["state"]),
        "ppo_optimizer_step": next(iter(ppo_steps)),
        "bc_replay_optimizer_nested_sha256": base.nested_sha256(parent["bc_replay_optimizer_state_dict"]),
        "bc_replay_optimizer_state_count": len(parent["bc_replay_optimizer_state_dict"]["state"]),
        "bc_replay_optimizer_step": next(iter(replay_steps)),
        "opponent_quota_nested_sha256": base.nested_sha256(parent["opponent_quota_state"]),
        "opponent_quota_observed_games": base.quota_observed_games(parent["opponent_quota_state"]),
        "opponent_quota_last_refresh_update": int(parent["opponent_quota_state"]["last_refresh_update"]),
    }
    if observed != ACTIVE_EXPECTED_PARENT_STATE:
        raise ValueError("parent state identity mismatch")
    if ACTIVE_BLOCK > 1:
        raw = parent.get("config")
        if not isinstance(raw, dict) or int(raw.get("seed", -1)) != SEED:
            raise ValueError("resume-all parent seed identity mismatch")
        _, _, manifest = ppo.configure_trainable_scope(model, str(raw["trainable_scope"]))
        ppo.validate_optimizer_resume_compatibility(
            parent,
            str(raw["trainable_scope"]),
            manifest,
            str(raw["advantage_normalization"]),
            str(raw["ppo_objective"]),
            None,
            float(raw["policy_temperature"]),
            str(raw["actor_reduction"]),
            float(raw["value_trunk_gradient_scale"]),
            str(raw["actor_value_gradient_mode"]),
        )
    return observed


def validate_protocol(path: Path, expected_sha256: str) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError("current working directory must be repository root")
    if Path(sys.executable).resolve() != lineage.EXPECTED_PYTHON.resolve():
        raise ValueError("launcher must use my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise ValueError("launcher requires -I -B")
    path = base.require_regular(path, "execution protocol")
    if base.sha256_file(path) != expected_sha256:
        raise ValueError("execution protocol SHA-256 mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if set(protocol) != PROTOCOL_KEYS:
        raise ValueError("execution protocol keys mismatch")
    if protocol["schema_version"] != SCHEMA_VERSION or protocol["status"] != "locked_after_launcher_and_before_block":
        raise ValueError("execution protocol schema or status mismatch")
    if protocol["seed"] != SEED:
        raise ValueError("execution seed mismatch")
    configure_active_block(protocol)
    spec = ACTIVE_SPEC
    if set(protocol["bindings"]) != {"launcher", *lineage.BINDINGS.keys()}:
        raise ValueError("execution binding keys mismatch")
    base.require_bound_file(protocol["bindings"]["launcher"], SELF_PATH, base.sha256_file(SELF_PATH), "launcher")
    for label, (bound_path, digest) in lineage.BINDINGS.items():
        base.require_bound_file(protocol["bindings"][label], bound_path, digest, label)
    command = derive_exact_command(ACTIVE_BLOCK)
    if protocol["command"] != command or protocol["command_sha256"] != spec["command_sha256"]:
        raise ValueError("block command differs from frozen derivation")
    required = {
        "--updates": str(spec["terminal"]),
        "--games-per-update": "96",
        "--minibatch-size": "512",
        "--learning-rate": "2.4e-05",
        "--value-learning-rate": "7.5e-06",
        "--schedule-start-update": str(spec["first"]),
        "--bc-kl-start": "0.016",
        "--bc-kl-end": "0.012",
        "--target-kl": "0.006",
        "--bc-replay-lr-scale": "0.075",
        "--seed": str(SEED),
        "--resume": str(spec["parent_path"]),
        "--resume-learner-weights": "resume",
    }
    for flag, expected in required.items():
        if base.flag_value(command, flag) != expected:
            raise ValueError(f"block flag drifted: {flag}")
    reset_flags = ("--reset-optimizer-on-resume", "--reset-opponent-quota-on-resume")
    if any((flag in command) != bool(spec["reset"]) for flag in reset_flags):
        raise ValueError("block reset semantics drifted")
    quotas = lineage.quota_mapping(command)
    if quotas != FIXED_QUOTAS or sum(quotas.values()) != 96 or any(v % 2 for v in quotas.values()):
        raise ValueError("block fixed quotas drifted")
    output_root, output_dir, terminal, attempt, log = block_paths(ACTIVE_BLOCK)
    if Path(protocol["output_dir"]).resolve() != output_dir or Path(protocol["terminal_checkpoint"]).resolve() != terminal:
        raise ValueError("block output bindings drifted")
    if Path(protocol["attempt_start_marker"]).resolve() != attempt or Path(protocol["log"]).resolve() != log:
        raise ValueError("block marker/log bindings drifted")
    for target in (output_root, output_dir, terminal, attempt, log):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing existing block target: {target}")
    expected_contract = {
        "updates": list(range(spec["first"], spec["terminal"] + 1)),
        "parent_update": spec["parent_update"],
        "terminal_update": spec["terminal"],
        "running_anchor_envelope": spec["anchor_envelope"],
        "terminal_role": spec["terminal_role"],
        "attempts_authorized": 1,
        "no_retry_or_transport_checkpoint_selection": True,
    }
    if protocol["block_contract"] != expected_contract:
        raise ValueError("block contract drifted")
    if protocol["scope"] != {"local_only": True, "network": False, "package": False, "upload": False, "submission": False}:
        raise ValueError("scope drifted")
    parent = parent_audit()
    inputs = base.audit_training_inputs(command)
    return protocol, command, {
        "status": "passed",
        "block": ACTIVE_BLOCK,
        "protocol_sha256": expected_sha256,
        "command_sha256": spec["command_sha256"],
        "command_tokens": len(command),
        "parent": parent,
        "training_inputs": inputs,
        "fixed_quota_sum": 96,
        "targets_absent": True,
        "writes_performed": False,
        "rollouts_performed": False,
    }


lineage.__file__ = str(SELF_PATH)
lineage.SELF_PATH = SELF_PATH
lineage.SCHEMA_VERSION = SCHEMA_VERSION
lineage.derive_exact_command = derive_exact_command
lineage.parent_audit = parent_audit
lineage.validate_protocol = validate_protocol


if __name__ == "__main__":
    raise SystemExit(lineage.main())
