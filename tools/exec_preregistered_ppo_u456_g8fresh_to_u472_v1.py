#!/usr/bin/env python3
"""Audit or execute the one preregistered U456-G8 to fresh-U472 PPO run."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

BASE_PATH = TOOLS_ROOT / "exec_preregistered_ppo_u464_to_u468.py"
BASE_SHA256 = "ace3a807704fd3dc0e5630077da5369861aecf75365a40d2261876995d36cc00"
_base_stat = os.lstat(BASE_PATH)
if stat.S_ISLNK(_base_stat.st_mode) or not stat.S_ISREG(_base_stat.st_mode):
    raise RuntimeError("authenticated launcher base must be a regular file")
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("authenticated launcher base SHA-256 mismatch")

import exec_preregistered_ppo_u464_to_u468 as base  # noqa: E402
import train_ppo as ppo  # noqa: E402


SELF_PATH = Path(__file__).resolve()
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA_VERSION = "ptcg-u456-g8-freshppo16x96-u472-isolated-execution-v1"
BRANCH = "ppo_u456_g8fresh_ppo16x96_p12_design202608130"
SEED = 202608132

MASTER_PATH = REPO_ROOT / f"artifacts/{BRANCH}.master_preregistration.json"
MASTER_SHA256 = "173ec848a0cebe5c5d327a5db434104083fd53c51efc5d97291067e39f17a3ab"
G8_DECISION_PATH = REPO_ROOT / f"artifacts/{BRANCH}.g8_training_integrity_decision.json"
G8_DECISION_SHA256 = "79f409edd1ba8cef7c53b104571ccd9584058a3ccd8ec8747779636996ee41db"
PARENT_PATH = REPO_ROOT / f"artifacts/{BRANCH}/general_stage/general-bc-0008.pt"
PARENT_SHA256 = "07f1645d391d3380eb77b08f583e9e8bec96c2ebcf34c57dbc27a1657cb1c6c9"
SOURCE_PROTOCOL = REPO_ROOT / (
    "artifacts/ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_u456_to_u464_seed202607331."
    "B_gold_league.seed-202607331.preregistration.json"
)
SOURCE_PROTOCOL_SHA256 = "b216a5e51669f3c30f0fd3b4d500fc5704a2e091c8ba7b5751c81ff110abff58"
SOURCE_COMMAND_SHA256 = "cbc1345e9043ec1d6fcd7e571041d9b4066090c6d6a5a2242d06935e54f32d22"
DERIVED_COMMAND_SHA256 = "997243602b0305d21440aba7db3d76f3cefa8f9c3bb84d8723d4c4709d628a58"
DERIVED_COMMAND_TOKENS = 179

OUTPUT_ROOT = REPO_ROOT / f"artifacts/{BRANCH}/ppo_stage"
OUTPUT_DIR = OUTPUT_ROOT / "B_gold_league/seed-202608132"
TERMINAL_CHECKPOINT = OUTPUT_DIR / "checkpoints/update-0472.pt"
ATTEMPT_MARKER = REPO_ROOT / ".ptcg-u456-g8fresh-ppo-u472-attempt-202608132-202608130.json"
LOG_PATH = REPO_ROOT / f"artifacts/{BRANCH}.ppo_stage.log"

EXPECTED_PARENT_STATE = {
    "update": 456,
    "runtime_model_state_sha256": "ae1e9f3e693a1a41906882754c94b757fd4863b4c57c97a5b020972ee740064d",
    "ppo_optimizer_nested_sha256": "9871a48963689affb08eb02a13388e7498a54b35155f23aa6d86ed87c14717b6",
    "ppo_optimizer_state_count": 28,
    "ppo_optimizer_step": 208,
    "bc_replay_optimizer_nested_sha256": "43c53aad77797ff43e5c393154e4b061f816036562f47626d0524d1d24d293c4",
    "bc_replay_optimizer_state_count": 24,
    "bc_replay_optimizer_step": 24,
    "opponent_quota_nested_sha256": "caea4b3bfbed2bfe6ed1467aa181cdb641ef5d62796544b6f785f77637c3c4fd",
    "opponent_quota_observed_games": 512,
    "opponent_quota_last_refresh_update": 455,
}

FIXED_QUOTAS = {
    "bc": 12,
    "rank08_miwaharuki-7dab5d9646@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 12,
    "rank10_raja-3425e824c4@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 6,
    "rank11_dominic_proxy-f858dc4d1c@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 12,
    "rank14_kanto_marnie-4b855e80e7@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 12,
    "rank15_sekkat-2a9e21d5c1@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 6,
    "rank18_azat_proxy-f780f1a71b@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 12,
    "rank19_pokemonfan-05d7a8d182@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 6,
    "rank20_siuuuu-650c99395e@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 6,
    "rank14_kanto_variant-0580ca2d67@e2e03fe8ef9592b1204c159a7725da3945675fa872b1e18baca550560503a78e": 6,
    "rank16_ntumlnoob-38c2dcf70d@882b584691fc15581b9c7d238ea131a933050f9da1694d8418c790d33c13fc54": 6,
}

BINDINGS: dict[str, tuple[Path, str]] = {
    "launcher_base": (BASE_PATH, BASE_SHA256),
    "trainer": (base.TRAINER_PATH, base.TRAINER_SHA256),
    "train_bc_orbit": (base.TRAIN_BC_ORBIT_PATH, base.TRAIN_BC_ORBIT_SHA256),
    "cg_init": (base.CG_INIT_PATH, base.CG_INIT_SHA256),
    "cg_sim": (base.CG_SIM_PATH, base.CG_SIM_SHA256),
    "cg_lib": (base.CG_LIB_PATH, base.CG_LIB_SHA256),
    "master_preregistration": (MASTER_PATH, MASTER_SHA256),
    "g8_training_integrity_decision": (G8_DECISION_PATH, G8_DECISION_SHA256),
    "parent_checkpoint": (PARENT_PATH, PARENT_SHA256),
    "general_bc": (base.GENERAL_BC_PATH, base.GENERAL_BC_SHA256),
    "bc_replay_archive": (base.BC_REPLAY_PATH, base.BC_REPLAY_SHA256),
    "candidate_deck": (base.CANDIDATE_DECK_PATH, base.CANDIDATE_DECK_SHA256),
    "training_league_manifest": (base.LEAGUE_MANIFEST_PATH, base.LEAGUE_MANIFEST_SHA256),
    "source_command_preregistration": (SOURCE_PROTOCOL, SOURCE_PROTOCOL_SHA256),
}

PROTOCOL_KEYS = {
    "schema_version",
    "status",
    "seed",
    "command",
    "command_sha256",
    "bindings",
    "expected_parent_state",
    "output_dir",
    "terminal_checkpoint",
    "attempt_start_marker",
    "log",
    "stop_rules",
    "training_gates",
    "scope",
}


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def replace_flag(command: list[str], flag: str, value: str) -> None:
    base.replace_flag_value(command, flag, value)


def derive_exact_command() -> list[str]:
    base.require_regular(SOURCE_PROTOCOL, "source command protocol")
    if sha256_file(SOURCE_PROTOCOL) != SOURCE_PROTOCOL_SHA256:
        raise ValueError("source command protocol SHA-256 mismatch")
    source = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))["binding"]["command"]
    if base.sha256_bytes(base.canonical_json_bytes(source)) != SOURCE_COMMAND_SHA256:
        raise ValueError("source command SHA-256 mismatch")
    command = list(source)
    replacements = {
        "--output-dir": str(OUTPUT_DIR),
        "--updates": "472",
        "--games-per-update": "96",
        "--eval-interval": "472",
        "--checkpoint-interval": "472",
        "--seed": str(SEED),
        "--resume": str(PARENT_PATH),
        "--bc-kl-start": "0.010",
        "--bc-kl-end": "0.006",
    }
    for flag, value in replacements.items():
        replace_flag(command, flag, value)
    for index, token in enumerate(command):
        if token == "--opponent-base-quota":
            name = command[index + 1]
            if name not in FIXED_QUOTAS:
                raise ValueError(f"unexpected exact-quota name: {name}")
            command[index + 2] = str(FIXED_QUOTAS[name])
    if len(command) != DERIVED_COMMAND_TOKENS:
        raise ValueError("derived command token count mismatch")
    if base.sha256_bytes(base.canonical_json_bytes(command)) != DERIVED_COMMAND_SHA256:
        raise ValueError("derived command SHA-256 mismatch")
    return command


def parent_audit() -> dict[str, Any]:
    parent = torch.load(PARENT_PATH, map_location="cpu", weights_only=False)
    bc = torch.load(base.GENERAL_BC_PATH, map_location="cpu", weights_only=False)
    if int(parent.get("update", -1)) != 456:
        raise ValueError("G8 parent update mismatch")
    model = ppo.instantiate_model_from_checkpoint(parent, bc, torch.device("cpu"))
    observed = {
        "update": 456,
        "runtime_model_state_sha256": ppo.model_state_sha256(model),
        "ppo_optimizer_nested_sha256": base.nested_sha256(parent["optimizer_state_dict"]),
        "ppo_optimizer_state_count": len(parent["optimizer_state_dict"]["state"]),
        "ppo_optimizer_step": next(iter(base.state_steps(parent["optimizer_state_dict"]))),
        "bc_replay_optimizer_nested_sha256": base.nested_sha256(parent["bc_replay_optimizer_state_dict"]),
        "bc_replay_optimizer_state_count": len(parent["bc_replay_optimizer_state_dict"]["state"]),
        "bc_replay_optimizer_step": next(iter(base.state_steps(parent["bc_replay_optimizer_state_dict"]))),
        "opponent_quota_nested_sha256": base.nested_sha256(parent["opponent_quota_state"]),
        "opponent_quota_observed_games": base.quota_observed_games(parent["opponent_quota_state"]),
        "opponent_quota_last_refresh_update": int(parent["opponent_quota_state"]["last_refresh_update"]),
    }
    if observed != EXPECTED_PARENT_STATE:
        raise ValueError("G8 parent state identity mismatch")
    provenance = parent.get("post_ppo_general_bc")
    if not isinstance(provenance, dict) or provenance.get("status") != "general_bc_completed":
        raise ValueError("G8 parent provenance is missing")
    if provenance.get("integrity", {}).get("changed_parameters_exact_actor24") is not True:
        raise ValueError("G8 actor24 integrity is absent")
    return observed


def quota_mapping(command: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, token in enumerate(command):
        if token == "--opponent-base-quota":
            name = command[index + 1]
            if name in result:
                raise ValueError("duplicate fixed quota")
            result[name] = int(command[index + 2])
    return result


def require_protocol_binding(value: Any, path: Path, digest: str, label: str) -> None:
    base.require_bound_file(value, path, digest, label)


def command_input_paths(command: Sequence[str]) -> list[Path]:
    paths = {SELF_PATH, *(path for path, _ in BINDINGS.values())}
    for index, token in enumerate(command):
        if token == "--extra-opponent":
            paths.add(Path(command[index + 1]))
            paths.add(Path(command[index + 2]))
    return sorted((base.require_regular(path, "locked input") for path in paths), key=str)


def assert_locked_bindings(
    snapshots: Sequence[dict[str, Any]],
    protocol: dict[str, Any],
    command: Sequence[str],
    protocol_path: Path,
    protocol_sha256: str,
) -> None:
    expected: dict[Path, str] = {
        SELF_PATH: protocol["bindings"]["launcher"]["sha256"],
        protocol_path.resolve(): protocol_sha256,
    }
    expected.update({path.resolve(): digest for path, digest in BINDINGS.values()})

    manifest_snapshot = base.snapshot_by_path(snapshots, base.LEAGUE_MANIFEST_PATH)
    manifest = json.loads(base.read_fd_bytes(int(manifest_snapshot["fd"])))
    manifest_by_checkpoint = {
        Path(item["checkpoint"]).resolve(): item["checkpoint_sha256"]
        for item in manifest["opponents"]
    }
    command_decks: set[Path] = set()
    for index, token in enumerate(command):
        if token == "--extra-opponent":
            checkpoint = Path(command[index + 1]).resolve()
            deck = Path(command[index + 2]).resolve()
            if checkpoint not in manifest_by_checkpoint:
                raise RuntimeError("command opponent absent from held manifest")
            expected[checkpoint] = manifest_by_checkpoint[checkpoint]
            command_decks.add(deck)
    deck_hashes = {
        Path(path).resolve(): digest
        for path, digest in base.EXPECTED_DISTINCT_TRAINING_DECK_HASHES.items()
    }
    if command_decks != set(deck_hashes):
        raise RuntimeError("command opponent deck set mismatch")
    expected.update(deck_hashes)
    observed = {Path(item["path"]).resolve(): item["sha256"] for item in snapshots}
    if observed != expected:
        raise RuntimeError("locked input identities differ from frozen bindings")


def validate_protocol(path: Path, expected_sha256: str) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError("current working directory must be repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ValueError("launcher must run under my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise ValueError("launcher requires Python flags -I -B")
    path = base.require_regular(path, "execution protocol")
    if sha256_file(path) != expected_sha256:
        raise ValueError("execution protocol SHA-256 mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if set(protocol) != PROTOCOL_KEYS:
        raise ValueError("execution protocol keys mismatch")
    if protocol["schema_version"] != SCHEMA_VERSION:
        raise ValueError("execution protocol schema mismatch")
    if protocol["status"] != "locked_after_launcher_and_before_training" or protocol["seed"] != SEED:
        raise ValueError("execution protocol state or seed mismatch")
    if set(protocol["bindings"]) != {"launcher", *BINDINGS.keys()}:
        raise ValueError("execution binding keys mismatch")
    require_protocol_binding(protocol["bindings"]["launcher"], SELF_PATH, sha256_file(SELF_PATH), "launcher")
    for label, (bound_path, digest) in BINDINGS.items():
        require_protocol_binding(protocol["bindings"][label], bound_path, digest, label)
    command = derive_exact_command()
    if protocol["command"] != command or protocol["command_sha256"] != DERIVED_COMMAND_SHA256:
        raise ValueError("execution command differs from frozen derivation")
    expected_flags = {
        "--updates": "472",
        "--games-per-update": "96",
        "--minibatch-size": "384",
        "--schedule-start-update": "457",
        "--bc-kl-start": "0.010",
        "--bc-kl-end": "0.006",
        "--target-kl": "0.006",
        "--eval-interval": "472",
        "--checkpoint-interval": "472",
        "--seed": str(SEED),
        "--resume": str(PARENT_PATH),
        "--resume-learner-weights": "resume",
    }
    for flag, value in expected_flags.items():
        if base.flag_value(command, flag) != value:
            raise ValueError(f"command flag drifted: {flag}")
    for flag in ("--reset-optimizer-on-resume", "--reset-opponent-quota-on-resume", "--skip-initial-eval"):
        if command.count(flag) != 1:
            raise ValueError(f"required flag count drifted: {flag}")
    quotas = quota_mapping(command)
    if quotas != FIXED_QUOTAS or sum(quotas.values()) != 96 or any(value % 2 for value in quotas.values()):
        raise ValueError("fixed 96-game seat-balanced quotas drifted")
    if protocol["expected_parent_state"] != EXPECTED_PARENT_STATE:
        raise ValueError("expected parent state drifted")
    if Path(protocol["output_dir"]).resolve() != OUTPUT_DIR or Path(protocol["terminal_checkpoint"]).resolve() != TERMINAL_CHECKPOINT:
        raise ValueError("output bindings drifted")
    if Path(protocol["attempt_start_marker"]).resolve() != ATTEMPT_MARKER or Path(protocol["log"]).resolve() != LOG_PATH:
        raise ValueError("marker or log binding drifted")
    for target in (OUTPUT_ROOT, OUTPUT_DIR, TERMINAL_CHECKPOINT, ATTEMPT_MARKER, LOG_PATH):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing existing target: {target}")
    if protocol["stop_rules"] != {
        "attempts_authorized": 1,
        "no_retry_seed_or_checkpoint_selection": True,
        "source_update": 456,
        "terminal_update": 472,
        "published_endpoints": [472],
    }:
        raise ValueError("stop rules drifted")
    if protocol["training_gates"] != {
        "all_16_updates_required": True,
        "maximum_bc_anchor_kl": 0.015,
        "maximum_approx_kl": 0.006,
        "maximum_clip_fraction": 0.05,
        "valid_games_each_update": 96,
        "quota_and_seat_exact": True,
        "all_values_finite": True,
        "four_consecutive_approx_kl_below_0.00005_is_idle_failure": True,
        "posthoc_only_no_intermediate_selection": True,
    }:
        raise ValueError("training gates drifted")
    if protocol["scope"] != {"local_only": True, "network": False, "package": False, "upload": False, "submission": False}:
        raise ValueError("scope drifted")
    training_inputs = base.audit_training_inputs(command)
    parent = parent_audit()
    return protocol, command, {
        "status": "passed",
        "protocol_sha256": expected_sha256,
        "command_sha256": DERIVED_COMMAND_SHA256,
        "command_tokens": len(command),
        "parent": parent,
        "training_inputs": training_inputs,
        "fixed_quota_sum": sum(quotas.values()),
        "targets_absent": True,
        "writes_performed": False,
        "rollouts_performed": False,
    }


def claim_and_run(protocol: dict[str, Any], command: Sequence[str], protocol_sha256: str, protocol_path: Path) -> int:
    marker = {
        "event": "u456_g8fresh_to_u472_attempt_consumed",
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "protocol_sha256": protocol_sha256,
        "command_sha256": DERIVED_COMMAND_SHA256,
        "launcher_sha256": sha256_file(SELF_PATH),
        "child_python_flags": ["-I", "-B"],
    }
    fd = os.open(ATTEMPT_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, base.canonical_json_bytes(marker))
        os.fsync(fd)
    finally:
        os.close(fd)

    log_fd = payload_fd = bootstrap_fd = -1
    snapshots: list[dict[str, Any]] = []
    try:
        log_fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.mkdir(OUTPUT_ROOT, mode=0o700)
        claim_path = OUTPUT_ROOT / ".freshppo16_launch_claim.json"
        claim_fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(claim_fd, base.canonical_json_bytes(marker))
            os.fsync(claim_fd)
        finally:
            os.close(claim_fd)

        locked = command_input_paths(command)
        locked.append(base.require_regular(protocol_path, "execution protocol"))
        snapshots = base.acquire_input_locks(sorted(set(locked), key=str))
        base.assert_input_locks_unchanged(snapshots)
        assert_locked_bindings(snapshots, protocol, command, protocol_path, protocol_sha256)

        trainer_snapshot = base.snapshot_by_path(snapshots, base.TRAINER_PATH)
        trainer_payload = base.read_fd_bytes(int(trainer_snapshot["fd"]))
        if hashlib.sha256(trainer_payload).hexdigest() != base.TRAINER_SHA256:
            raise RuntimeError("held trainer payload hash mismatch")
        payload_fd, payload_hash = base.create_sealed_memfd("ptcg-u456-g8fresh-u472-trainer", trainer_payload)
        bootstrap_payload = base.build_trainer_bootstrap(payload_fd)
        bootstrap_fd, bootstrap_hash = base.create_sealed_memfd("ptcg-u456-g8fresh-u472-bootstrap", bootstrap_payload)
        base.assert_sealed_memfd(payload_fd, payload_hash, "trainer payload")
        base.assert_sealed_memfd(bootstrap_fd, bootstrap_hash, "trainer bootstrap")

        effective = [command[0], "-I", "-B", f"/proc/self/fd/{bootstrap_fd}", *command[2:]]
        handle = os.fdopen(log_fd, "wb", buffering=0)
        log_fd = -1
        with handle:
            handle.write(base.canonical_json_bytes({
                "event": "u456_g8fresh_to_u472_pre_exec_checks_passed",
                **{key: value for key, value in marker.items() if key != "event"},
                "cwd": str(REPO_ROOT),
                "child_argv_prefix": effective[:4],
                "child_isolated_mode": True,
                "effective_command_sha256": base.sha256_bytes(base.canonical_json_bytes(effective)),
                "input_shared_locks": len(snapshots),
                "sealed_trainer_payload_sha256": payload_hash,
                "sealed_trainer_bootstrap_sha256": bootstrap_hash,
            }))
            os.fsync(handle.fileno())
            base.assert_input_locks_unchanged(snapshots)
            completed = subprocess.run(
                effective,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                pass_fds=(bootstrap_fd, payload_fd),
            )
            base.assert_input_locks_unchanged(snapshots)
            base.assert_sealed_memfd(payload_fd, payload_hash, "trainer payload")
            base.assert_sealed_memfd(bootstrap_fd, bootstrap_hash, "trainer bootstrap")
            handle.write(base.canonical_json_bytes({
                "event": "u456_g8fresh_to_u472_child_terminal",
                "return_code": completed.returncode,
                "child_isolated_mode": True,
                "inputs_unchanged_through_child_terminal": True,
            }))
            os.fsync(handle.fileno())
        return int(completed.returncode)
    finally:
        if bootstrap_fd >= 0:
            os.close(bootstrap_fd)
        if payload_fd >= 0:
            os.close(payload_fd)
        if snapshots:
            base.release_input_locks(snapshots)
        if log_fd >= 0:
            os.close(log_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol, command, audit = validate_protocol(args.protocol, args.expected_protocol_sha256)
    if args.audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return claim_and_run(protocol, command, args.expected_protocol_sha256, args.protocol)


if __name__ == "__main__":
    raise SystemExit(main())
