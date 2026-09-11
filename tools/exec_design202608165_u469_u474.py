#!/usr/bin/env python3
"""Audit or execute the frozen G48 -> U469-U474 multi-opponent PPO stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
BASE = TOOLS / "exec_preregistered_ppo_u464_to_u468.py"
BASE_SHA256 = "ace3a807704fd3dc0e5630077da5369861aecf75365a40d2261876995d36cc00"
info = os.lstat(BASE)
if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("authenticated audit helper mismatch")
import exec_preregistered_ppo_u464_to_u468 as audit  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SELF = Path(__file__).resolve()
SCHEMA = "ptcg-design202608165-g48-u469-u474-multiopponent-v1"
SEED = 202608165
TRAINER = TOOLS / "train_ppo.py"
TRAINER_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
PARENT = ROOT / "artifacts/design202608164_general_bc_g48/general-bc-0048.pt"
PARENT_SHA256 = "798769685b4c61e089ca0beed2eca65ee592b642d46b2c6a9dbb84d84b505614"
BC = ROOT / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/best.pt"
BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
REPLAY = ROOT / "data/bc_marnie_top50_current14_timeforward_train0802_valid0803_design202608147.zip"
REPLAY_SHA256 = "bf01cfc7e9c0f616f157ae36ed68c76619ae45448cdc17562befe889a4d5ae93"
DECK = ROOT / "data/decks/marnie_grimmsnarl_froslass_luca.csv"
DECK_SHA256 = "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d"
LEAGUE = ROOT / "artifacts/gold_clone_league_top21_20260727_v2/league_manifest.json"
LEAGUE_SHA256 = "94a321f06addf479331bd8883684a464e64c05a0b4896357e027cee574694ed1"
OUTPUT_ROOT = ROOT / "artifacts/design202608165_generalbc_g48_ppo6x192_u474"
OUTPUT_DIR = OUTPUT_ROOT / "B_gold_league/seed-202608165"
TERMINAL = OUTPUT_DIR / "checkpoints/update-0474.pt"
ATTEMPT = ROOT / ".ptcg-design202608165-u469-u474-attempt.json"
LOG = ROOT / "artifacts/design202608165_generalbc_g48_ppo6x192_u474.log"
DEPENDENCIES = {
    TOOLS / "train_bc_orbit.py": "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
    ROOT / "dataset/sample_submission/sample_submission/cg/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ROOT / "dataset/sample_submission/sample_submission/cg/sim.py": "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655",
    ROOT / "dataset/sample_submission/sample_submission/cg/libcg.so": "feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887",
}
QUOTAS = {
    "rank01_flg": 12, "rank02_dominic": 20, "rank03_dries": 24,
    "rank04_liam": 24, "rank06_etoppo": 10, "rank08_hancang": 10,
    "rank11_luca": 10, "rank12_taichicchi": 10, "rank15_jz": 14,
    "rank17_213tubo": 20, "rank18_tuna": 10, "rank19_szlachetny": 28,
}
DECK_SHA_BY_HASH = {
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
    "978ab31c50aa7fdaf11e38ba108cf0eda7fab6ca3597bfa3b45a7f74cd6a9327": "7c6399e18e86ec1f19aab25b362d662a27716a660178e3d8ea48d61c2827a1b3",
    "e2e03fe8ef9592b1204c159a7725da3945675fa872b1e18baca550560503a78e": "c960f71296a4ca797d8b8421f8c3d059752296ba34721108e6b625e02ca0410a",
}


def sha(path: Path) -> str:
    return audit.sha256_file(path)


def command_sha(command: Sequence[str]) -> str:
    return audit.sha256_bytes(audit.canonical_json_bytes(list(command)))


def opponents() -> list[dict[str, Any]]:
    if sha(LEAGUE) != LEAGUE_SHA256:
        raise RuntimeError("league manifest mismatch")
    values = json.loads(LEAGUE.read_text())["opponents"]
    if [value["policy_id"] for value in values] != list(QUOTAS):
        raise RuntimeError("league order drifted")
    return values


def opponent_name(value: dict[str, Any]) -> str:
    return f"{Path(value['checkpoint']).stem}@{Path(value['deck']).stem}"


def derive_command() -> list[str]:
    values = opponents()
    command = [str(EXPECTED_PYTHON), str(TRAINER), "--bc-checkpoint", str(BC), "--kl-reference-checkpoint", str(PARENT), "--deck", str(DECK)]
    for value in values:
        command.extend(["--extra-opponent", value["checkpoint"], value["deck"]])
    command.extend([
        "--output-dir", str(OUTPUT_DIR), "--updates", "474",
        "--environments", "32", "--games-per-update", "192",
        "--ppo-epochs", "4", "--minibatch-size", "512",
        "--learning-rate", "4.8e-05", "--value-learning-rate", "7.5e-06",
        "--weight-decay", "1e-4", "--learning-rate-schedule", "constant",
        "--schedule-start-update", "469", "--gamma", "1.0", "--gae-lambda", "0.97",
        "--advantage-normalization", "per_opponent", "--clip-ratio", "0.15",
        "--value-coefficient", "0.25", "--value-trunk-gradient-scale", "1.0",
        "--actor-value-gradient-mode", "scalar", "--entropy-coefficient", "0.001",
        "--max-grad-norm", "0.5", "--policy-temperature", "0.8",
        "--trainable-scope", "last_block_heads", "--bc-kl-start", "0.008",
        "--bc-kl-end", "0.006", "--target-kl", "0.002",
        "--league-probability", "1.0", "--opponent-sampling", "per_game",
        "--opponent-quota-mode", "fixed", "--opponent-quota-seat-balance",
        "--ppo-objective", "standard", "--actor-reduction", "episode_mean",
        "--snapshot-interval", "1000000", "--max-pool-size", "13",
        "--eval-interval", "2", "--eval-games", "64",
        "--selection-aggregation", "mean", "--checkpoint-interval", "2",
        "--max-game-decisions", "1000", "--seed", str(SEED),
        "--resume", str(PARENT), "--resume-learner-weights", "resume",
        "--reset-optimizer-on-resume", "--reset-opponent-quota-on-resume",
        "--skip-initial-eval", "--device", "cuda",
        "--bc-replay-data", str(REPLAY), "--bc-replay-split", "train",
        "--bc-replay-batches", "72", "--bc-replay-batch-size", "256",
        "--bc-replay-workers", "8", "--bc-replay-steps", "1",
        "--bc-replay-lr-scale", "0.05", "--bc-replay-loss", "ordered",
        "--bc-replay-order-context-weight", "8.0",
        "--bc-replay-non-context34-fixed-multi-action-order-weight", "1.0",
        "--bc-replay-context34-rows-per-batch", "4",
    ])
    for value in values:
        command.extend(["--opponent-base-quota", opponent_name(value), str(QUOTAS[value["policy_id"]])])
    return command


def bindings(protocol: Path) -> dict[Path, str]:
    result = {SELF: sha(SELF), protocol: sha(protocol), BASE: BASE_SHA256, TRAINER: TRAINER_SHA256, PARENT: PARENT_SHA256, BC: BC_SHA256, REPLAY: REPLAY_SHA256, DECK: DECK_SHA256, LEAGUE: LEAGUE_SHA256, **DEPENDENCIES}
    for value in opponents():
        result[Path(value["checkpoint"]).resolve()] = value["checkpoint_sha256"]
        result[Path(value["deck"]).resolve()] = DECK_SHA_BY_HASH[value["deck_hash"]]
    return {path.resolve(): digest for path, digest in result.items()}


def expected_protocol(command: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA, "status": "locked_before_training", "seed": SEED,
        "launcher_sha256": sha(SELF), "command_sha256": command_sha(command),
        "parent_sha256": PARENT_SHA256, "league_manifest_sha256": LEAGUE_SHA256,
        "replay_sha256": REPLAY_SHA256, "output_dir": str(OUTPUT_DIR),
        "terminal_checkpoint": str(TERMINAL), "attempt_marker": str(ATTEMPT),
        "log": str(LOG), "updates": [469, 470, 471, 472, 473, 474],
        "games_per_update": 192, "fixed_quotas_by_policy": QUOTAS,
        "hyperparameters": {
            "actor_learning_rate": 4.8e-5, "value_learning_rate": 7.5e-6,
            "ppo_epochs": 4, "bc_kl_start": 0.008, "bc_kl_end": 0.006,
            "target_kl": 0.002, "trainable_scope": "last_block_heads",
            "advantage_normalization": "per_opponent",
            "bc_replay_steps_per_update": 1, "bc_replay_lr_scale": 0.05,
        },
        "scope": {"local_only": True, "package": False, "upload": False, "submission": False},
    }


def validate(protocol: Path, expected_protocol_sha: str) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve() or sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("launcher requires repository cwd and my_project_env Python -I -B")
    if sha(protocol) != expected_protocol_sha:
        raise RuntimeError("protocol SHA-256 mismatch")
    command = derive_command()
    value = json.loads(protocol.read_text())
    if value != expected_protocol(command):
        raise RuntimeError("protocol differs from launcher contract")
    if sum(QUOTAS.values()) != 192 or any(value % 2 for value in QUOTAS.values()):
        raise RuntimeError("quota contract failed")
    for target in (OUTPUT_ROOT, OUTPUT_DIR, TERMINAL, ATTEMPT, LOG):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    frozen = bindings(protocol)
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise RuntimeError(f"input mismatch: {path}")
    return value, command, {"status": "passed", "command_sha256": command_sha(command), "input_files": len(frozen), "fixed_quota_sum": sum(QUOTAS.values()), "targets_absent": True, "optimizer_steps": 0, "rollout_games": 0}


def run(protocol_path: Path, protocol: dict[str, Any], command: list[str]) -> int:
    marker = {"schema_version": SCHEMA, "event": "fresh_ppo_attempt_consumed", "seed": SEED, "protocol_sha256": sha(protocol_path), "command_sha256": protocol["command_sha256"], "launcher_sha256": sha(SELF)}
    fd = os.open(ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, audit.canonical_json_bytes(marker)); os.fsync(fd)
    finally:
        os.close(fd)
    frozen = bindings(protocol_path)
    snapshots = audit.acquire_input_locks(sorted(frozen))
    try:
        audit.assert_input_locks_unchanged(snapshots)
        os.mkdir(OUTPUT_ROOT, mode=0o700)
        log_fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(log_fd, "wb", buffering=0) as handle:
            effective = [command[0], "-I", "-B", *command[1:]]
            handle.write(audit.canonical_json_bytes(marker | {"event": "pre_exec_checks_passed", "input_shared_locks": len(snapshots)}))
            environment = os.environ.copy(); environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            completed = subprocess.run(effective, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, env=environment, check=False)
            audit.assert_input_locks_unchanged(snapshots)
            handle.write(audit.canonical_json_bytes({"event": "child_terminal", "return_code": completed.returncode, "inputs_unchanged": True})); os.fsync(handle.fileno())
        return int(completed.returncode)
    finally:
        audit.release_input_locks(snapshots)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    protocol, command, report = validate(args.protocol.resolve(), args.expected_protocol_sha256)
    if args.audit_only:
        print(json.dumps(report, indent=2, sort_keys=True)); return 0
    return run(args.protocol.resolve(), protocol, command)


if __name__ == "__main__":
    raise SystemExit(main())
