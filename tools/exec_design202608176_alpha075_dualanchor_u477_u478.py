#!/usr/bin/env python3
"""Audit or execute alpha0.75 U477-U478 dual-anchor PPO continuation."""

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
SCHEMA = "ptcg-design202608176-alpha075-dualanchor-u477-u478-v1"
SEED = 202608176
TRAINER = TOOLS / "train_ppo.py"
TRAINER_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
BOOTSTRAP = ROOT / "artifacts/design202608175_alpha075_fresh_training_bootstrap/alpha075-fresh-bootstrap-u476.pt"
BOOTSTRAP_SHA256 = "54b9a91dc740d01e053dfa3bbcc91c227edc765b230205b1ddb03da586f0da02"
BOOTSTRAP_MANIFEST = ROOT / "artifacts/design202608175_alpha075_fresh_training_bootstrap/manifest.json"
BOOTSTRAP_MANIFEST_SHA256 = "b8df1539aa1569e96a90049ca4664074a7866ff6703d4a1190a8d2a4ba2fd8f2"
SUBMITTED_U472 = ROOT / "artifacts/ppo_guardedu468_updatedreplay_g8_ppo4x96_u472_freshjointactor6_s8_design202608148/ppo_stage/B_gold_league/seed-202608148/checkpoints/update-0472.pt"
SUBMITTED_U472_SHA256 = "c0436d54fff5e4da62d94c3002775c9e4b9ee848c5c59dd4422943cd71ad7018"
U468_BETA100 = ROOT / "artifacts/ppo_u468_p12delta_direction_beta050_075_100_design202608092/transport-beta-100.pt"
U468_BETA100_SHA256 = "53284b1d4e94f09bff5b92a7fb0d24efd26ee67a2a332014cdc97572b00c3beb"
BC = ROOT / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/best.pt"
BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
REPLAY = ROOT / "data/bc_marnie_top50_current14_timeforward_train0802_valid0803_design202608147.zip"
REPLAY_SHA256 = "bf01cfc7e9c0f616f157ae36ed68c76619ae45448cdc17562befe889a4d5ae93"
DECK = ROOT / "data/decks/marnie_grimmsnarl_froslass_luca.csv"
DECK_SHA256 = "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d"
LEAGUE = ROOT / "artifacts/gold_clone_league_top21_20260727_v2/league_manifest.json"
LEAGUE_SHA256 = "94a321f06addf479331bd8883684a464e64c05a0b4896357e027cee574694ed1"
OUTPUT_ROOT = ROOT / "artifacts/design202608176_alpha075_dualanchor_ppo2x192_u478"
OUTPUT_DIR = OUTPUT_ROOT / "B_gold_league/seed-202608176"
TERMINAL = OUTPUT_DIR / "checkpoints/update-0478.pt"
ATTEMPT = ROOT / ".ptcg-design202608176-alpha075-u477-u478-attempt.json"
LOG = ROOT / "artifacts/design202608176_alpha075_dualanchor_ppo2x192_u478.log"
DEPENDENCIES = {
    TOOLS / "train_bc_orbit.py": "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
    ROOT / "dataset/sample_submission/sample_submission/cg/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ROOT / "dataset/sample_submission/sample_submission/cg/sim.py": "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655",
    ROOT / "dataset/sample_submission/sample_submission/cg/libcg.so": "feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887",
}
GOLD_QUOTAS = {
    "rank01_flg": 6,
    "rank02_dominic": 12,
    "rank03_dries": 16,
    "rank04_liam": 16,
    "rank06_etoppo": 6,
    "rank08_hancang": 6,
    "rank11_luca": 6,
    "rank12_taichicchi": 6,
    "rank15_jz": 10,
    "rank17_213tubo": 14,
    "rank18_tuna": 6,
    "rank19_szlachetny": 24,
}
ANCHOR_QUOTAS = {"submitted_u472": 32, "u468_beta100": 32}
DECK_SHA_BY_HASH = {
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
    "978ab31c50aa7fdaf11e38ba108cf0eda7fab6ca3597bfa3b45a7f74cd6a9327": "7c6399e18e86ec1f19aab25b362d662a27716a660178e3d8ea48d61c2827a1b3",
    "e2e03fe8ef9592b1204c159a7725da3945675fa872b1e18baca550560503a78e": "c960f71296a4ca797d8b8421f8c3d059752296ba34721108e6b625e02ca0410a",
}


def sha(path: Path) -> str:
    return audit.sha256_file(path)


def command_sha(command: Sequence[str]) -> str:
    return audit.sha256_bytes(audit.canonical_json_bytes(list(command)))


def gold_opponents() -> list[dict[str, Any]]:
    if sha(LEAGUE) != LEAGUE_SHA256:
        raise RuntimeError("league manifest mismatch")
    values = json.loads(LEAGUE.read_text())["opponents"]
    if [value["policy_id"] for value in values] != list(GOLD_QUOTAS):
        raise RuntimeError("league order drifted")
    return values


def opponent_name(checkpoint: str | Path, deck: str | Path) -> str:
    return f"{Path(checkpoint).stem}@{Path(deck).stem}"


def derive_command() -> list[str]:
    values = gold_opponents()
    command = [
        str(EXPECTED_PYTHON), str(TRAINER),
        "--bc-checkpoint", str(BC),
        "--kl-reference-checkpoint", str(BOOTSTRAP),
        "--deck", str(DECK),
    ]
    for value in values:
        command.extend(["--extra-opponent", value["checkpoint"], value["deck"]])
    command.extend(["--extra-opponent", str(SUBMITTED_U472), str(DECK)])
    command.extend(["--extra-opponent", str(U468_BETA100), str(DECK)])
    command.extend([
        "--output-dir", str(OUTPUT_DIR), "--updates", "478",
        "--environments", "32", "--games-per-update", "192",
        "--ppo-epochs", "4", "--minibatch-size", "512",
        "--learning-rate", "2.4e-05", "--value-learning-rate", "5e-06",
        "--weight-decay", "1e-4", "--learning-rate-schedule", "constant",
        "--schedule-start-update", "477", "--gamma", "1.0", "--gae-lambda", "0.97",
        "--advantage-normalization", "per_opponent", "--clip-ratio", "0.15",
        "--value-coefficient", "0.25", "--value-trunk-gradient-scale", "1.0",
        "--actor-value-gradient-mode", "scalar", "--entropy-coefficient", "0.001",
        "--max-grad-norm", "0.5", "--policy-temperature", "0.8",
        "--trainable-scope", "heads", "--bc-kl-start", "0.004",
        "--bc-kl-end", "0.003", "--target-kl", "0.0015",
        "--league-probability", "1.0", "--opponent-sampling", "per_game",
        "--opponent-quota-mode", "fixed", "--opponent-quota-seat-balance",
        "--ppo-objective", "standard", "--actor-reduction", "episode_mean",
        "--snapshot-interval", "1000000", "--max-pool-size", "15",
        "--eval-interval", "1", "--eval-games", "64",
        "--selection-aggregation", "mean", "--checkpoint-interval", "1",
        "--max-game-decisions", "1000", "--seed", str(SEED),
        "--resume", str(BOOTSTRAP), "--resume-learner-weights", "resume",
        "--reset-optimizer-on-resume", "--reset-opponent-quota-on-resume",
        "--skip-initial-eval", "--device", "cuda",
        "--bc-replay-data", str(REPLAY), "--bc-replay-split", "train",
        "--bc-replay-batches", "72", "--bc-replay-batch-size", "256",
        "--bc-replay-workers", "8", "--bc-replay-steps", "1",
        "--bc-replay-lr-scale", "0.0125", "--bc-replay-loss", "ordered",
        "--bc-replay-order-context-weight", "8.0",
        "--bc-replay-non-context34-fixed-multi-action-order-weight", "1.0",
        "--bc-replay-context34-rows-per-batch", "4",
    ])
    for value in values:
        command.extend([
            "--opponent-base-quota", opponent_name(value["checkpoint"], value["deck"]),
            str(GOLD_QUOTAS[value["policy_id"]]),
        ])
    command.extend([
        "--opponent-base-quota", opponent_name(SUBMITTED_U472, DECK),
        str(ANCHOR_QUOTAS["submitted_u472"]),
        "--opponent-base-quota", opponent_name(U468_BETA100, DECK),
        str(ANCHOR_QUOTAS["u468_beta100"]),
    ])
    return command


def bindings(protocol: Path) -> dict[Path, str]:
    result = {
        SELF: sha(SELF), protocol: sha(protocol), BASE: BASE_SHA256,
        TRAINER: TRAINER_SHA256, BOOTSTRAP: BOOTSTRAP_SHA256,
        BOOTSTRAP_MANIFEST: BOOTSTRAP_MANIFEST_SHA256,
        SUBMITTED_U472: SUBMITTED_U472_SHA256, U468_BETA100: U468_BETA100_SHA256,
        BC: BC_SHA256, REPLAY: REPLAY_SHA256, DECK: DECK_SHA256,
        LEAGUE: LEAGUE_SHA256, **DEPENDENCIES,
    }
    for value in gold_opponents():
        result[Path(value["checkpoint"]).resolve()] = value["checkpoint_sha256"]
        result[Path(value["deck"]).resolve()] = DECK_SHA_BY_HASH[value["deck_hash"]]
    return {path.resolve(): digest for path, digest in result.items()}


def expected_protocol(command: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "locked_before_training",
        "seed": SEED,
        "launcher_sha256": sha(SELF),
        "command_sha256": command_sha(command),
        "bootstrap_sha256": BOOTSTRAP_SHA256,
        "bootstrap_manifest_sha256": BOOTSTRAP_MANIFEST_SHA256,
        "league_manifest_sha256": LEAGUE_SHA256,
        "replay_sha256": REPLAY_SHA256,
        "output_dir": str(OUTPUT_DIR),
        "terminal_checkpoint": str(TERMINAL),
        "attempt_marker": str(ATTEMPT),
        "log": str(LOG),
        "updates": [477, 478],
        "games_per_update": 192,
        "fixed_gold_quotas_by_policy": GOLD_QUOTAS,
        "anchor_quotas": ANCHOR_QUOTAS,
        "hyperparameters": {
            "actor_learning_rate": 2.4e-5,
            "value_learning_rate": 5e-6,
            "ppo_epochs": 4,
            "bc_kl_start": 0.004,
            "bc_kl_end": 0.003,
            "target_kl": 0.0015,
            "trainable_scope": "heads",
            "advantage_normalization": "per_opponent",
            "bc_replay_steps_per_update": 1,
            "bc_replay_lr_scale": 0.0125,
        },
        "scope": {"local_only": True, "package": False, "upload": False, "submission": False},
    }


def validate(protocol: Path, expected_protocol_sha: str) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    if (
        Path.cwd().resolve() != ROOT
        or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve()
        or sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
    ):
        raise RuntimeError("launcher requires repository cwd and my_project_env Python -I -B")
    if sha(protocol) != expected_protocol_sha:
        raise RuntimeError("protocol SHA-256 mismatch")
    command = derive_command()
    value = json.loads(protocol.read_text())
    if value != expected_protocol(command):
        raise RuntimeError("protocol differs from launcher contract")
    total = sum(GOLD_QUOTAS.values()) + sum(ANCHOR_QUOTAS.values())
    if total != 192 or any(value % 2 for value in [*GOLD_QUOTAS.values(), *ANCHOR_QUOTAS.values()]):
        raise RuntimeError("quota contract failed")
    bootstrap = __import__("torch").load(BOOTSTRAP, map_location="cpu", weights_only=False)
    if (
        bootstrap.get("fresh_training_bootstrap", {}).get("optimizer_state_source") != "absent_fresh_required"
        or any(key in bootstrap for key in ("optimizer_state_dict", "bc_replay_optimizer_state_dict", "opponent_quota_state", "evaluation_only", "resume_forbidden"))
    ):
        raise RuntimeError("bootstrap is not fresh-optimizer-only")
    for target in (OUTPUT_ROOT, OUTPUT_DIR, TERMINAL, ATTEMPT, LOG):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    frozen = bindings(protocol)
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise RuntimeError(f"input mismatch: {path}")
    return value, command, {
        "status": "passed", "command_sha256": command_sha(command),
        "input_files": len(frozen), "fixed_quota_sum": total,
        "gold_quota_sum": sum(GOLD_QUOTAS.values()),
        "anchor_quota_sum": sum(ANCHOR_QUOTAS.values()),
        "targets_absent": True, "optimizer_steps": 0, "rollout_games": 0,
    }


def run(protocol_path: Path, protocol: dict[str, Any], command: list[str]) -> int:
    marker = {
        "schema_version": SCHEMA, "event": "fresh_ppo_attempt_consumed",
        "seed": SEED, "protocol_sha256": sha(protocol_path),
        "command_sha256": protocol["command_sha256"], "launcher_sha256": sha(SELF),
    }
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
    parser.add_argument("--print-expected-protocol", action="store_true")
    args = parser.parse_args()
    if args.print_expected_protocol:
        print(json.dumps(expected_protocol(derive_command()), indent=2, sort_keys=True)); return 0
    protocol, command, report = validate(args.protocol.resolve(), args.expected_protocol_sha256)
    if args.audit_only:
        print(json.dumps(report, indent=2, sort_keys=True)); return 0
    return run(args.protocol.resolve(), protocol, command)


if __name__ == "__main__":
    raise SystemExit(main())
