#!/usr/bin/env python3
"""Audit or execute the frozen fresh PPO U469-U472 stage for design202608148."""

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

BASE_PATH = TOOLS / "exec_preregistered_ppo_u464_to_u468.py"
BASE_SHA256 = "ace3a807704fd3dc0e5630077da5369861aecf75365a40d2261876995d36cc00"
info = os.lstat(BASE_PATH)
if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
    raise RuntimeError("authenticated PPO launcher base must be a regular file")
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("authenticated PPO launcher base SHA-256 mismatch")

import torch  # noqa: E402
import exec_preregistered_ppo_u464_to_u468 as audit  # noqa: E402
import train_ppo as ppo  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SELF = Path(__file__).resolve()
SCHEMA = "ptcg-design202608148-fresh-ppo-u469-u472-execution-v1"
SEED = 202608148
BRANCH = "ppo_guardedu468_updatedreplay_g8_ppo4x96_u472_freshjointactor6_s8_design202608148"
SOURCE_PROTOCOL = ROOT / (
    "artifacts/ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_u456_to_u464_seed202607331."
    "B_gold_league.seed-202607331.preregistration.json"
)
SOURCE_PROTOCOL_SHA256 = "b216a5e51669f3c30f0fd3b4d500fc5704a2e091c8ba7b5751c81ff110abff58"
SOURCE_COMMAND_SHA256 = "cbc1345e9043ec1d6fcd7e571041d9b4066090c6d6a5a2242d06935e54f32d22"
TRAINER = TOOLS / "train_ppo.py"
TRAINER_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
PARENT = ROOT / f"artifacts/{BRANCH}/general_bc_g8/general-bc-0008.pt"
PARENT_SHA256 = "3a4c715d84aa3589f0eebce5b9ba9e8105035e6a2a3a856d0d13e5969e0501c0"
G8_MANIFEST = ROOT / f"artifacts/{BRANCH}/general_bc_g8/general_bc_manifest.json"
G8_MANIFEST_SHA256 = "d0bd50c3c0ca1108ab1922411634b981ab15cf2046833b156bfd42f68eb483f6"
GENERAL_BC = ROOT / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/best.pt"
GENERAL_BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
REPLAY = ROOT / "data/bc_marnie_top50_current14_timeforward_train0802_valid0803_design202608147.zip"
REPLAY_SHA256 = "bf01cfc7e9c0f616f157ae36ed68c76619ae45448cdc17562befe889a4d5ae93"
CANDIDATE_DECK = ROOT / "data/decks/marnie_grimmsnarl_froslass_luca.csv"
CANDIDATE_DECK_SHA256 = "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d"
CURRENT_LEAGUE = ROOT / "artifacts/gold_clone_league_current_proxy_20260731/league_manifest.json"
CURRENT_LEAGUE_SHA256 = "148dd03fc550b062153080d053275ac5adc11feefdecd241cc1b25faadb7306f"
EXACT_LEAGUE = ROOT / "artifacts/gold_clone_league_top21_20260727_v2/league_manifest.json"
EXACT_LEAGUE_SHA256 = "94a321f06addf479331bd8883684a464e64c05a0b4896357e027cee574694ed1"
FLG_CHECKPOINT = ROOT / "artifacts/gold_clone_league_top21_20260727_v2/league_checkpoints/rank01_flg-e7a7d75a8f.pt"
FLG_DECK = ROOT / "data/gold_league/top21_exact_20260727/decks/c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af.csv"
FLG_NAME = "rank01_flg-e7a7d75a8f@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
OUTPUT_ROOT = ROOT / f"artifacts/{BRANCH}/ppo_stage"
OUTPUT_DIR = OUTPUT_ROOT / "B_gold_league/seed-202608148"
TERMINAL = OUTPUT_DIR / "checkpoints/update-0472.pt"
ATTEMPT = ROOT / ".ptcg-design202608148-freshppo-u472-attempt.json"
LOG = ROOT / f"artifacts/{BRANCH}.ppo_stage.log"

DEPENDENCIES = {
    TOOLS / "train_bc_orbit.py": "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
    ROOT / "dataset/sample_submission/sample_submission/cg/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ROOT / "dataset/sample_submission/sample_submission/cg/sim.py": "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655",
    ROOT / "dataset/sample_submission/sample_submission/cg/libcg.so": "feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887",
}
DECK_BINDINGS = {
    CANDIDATE_DECK: CANDIDATE_DECK_SHA256,
    ROOT / "data/gold_league/gold21_proxy_20260731/decks/c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af.csv": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
    FLG_DECK: "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
    ROOT / "data/gold_league/gold21_proxy_20260731/decks/e2e03fe8ef9592b1204c159a7725da3945675fa872b1e18baca550560503a78e.csv": "c960f71296a4ca797d8b8421f8c3d059752296ba34721108e6b625e02ca0410a",
    ROOT / "data/gold_league/gold21_proxy_20260731/decks/882b584691fc15581b9c7d238ea131a933050f9da1694d8418c790d33c13fc54.csv": "778b260c3c211be1b5317533b437849d2949ae4392eb6f693440d9b6e67cc0ea",
}
FIXED_QUOTAS = {
    "bc": 12,
    "rank08_miwaharuki-7dab5d9646@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 12,
    "rank11_dominic_proxy-f858dc4d1c@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 12,
    "rank14_kanto_marnie-4b855e80e7@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 12,
    FLG_NAME: 12,
    "rank18_azat_proxy-f780f1a71b@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 8,
    "rank19_pokemonfan-05d7a8d182@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 8,
    "rank10_raja-3425e824c4@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 4,
    "rank15_sekkat-2a9e21d5c1@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 4,
    "rank20_siuuuu-650c99395e@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af": 4,
    "rank14_kanto_variant-0580ca2d67@e2e03fe8ef9592b1204c159a7725da3945675fa872b1e18baca550560503a78e": 4,
    "rank16_ntumlnoob-38c2dcf70d@882b584691fc15581b9c7d238ea131a933050f9da1694d8418c790d33c13fc54": 4,
}


def sha256_file(path: Path) -> str:
    return audit.sha256_file(path)


def command_sha256(command: Sequence[str]) -> str:
    return audit.sha256_bytes(audit.canonical_json_bytes(list(command)))


def replace_flag(command: list[str], flag: str, value: str) -> None:
    indices = [index for index, token in enumerate(command) if token == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise RuntimeError(f"command flag occurrence drifted: {flag}")
    command[indices[0] + 1] = value


def derive_command() -> list[str]:
    if sha256_file(SOURCE_PROTOCOL) != SOURCE_PROTOCOL_SHA256:
        raise RuntimeError("source protocol SHA-256 mismatch")
    source = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))["binding"]["command"]
    if command_sha256(source) != SOURCE_COMMAND_SHA256:
        raise RuntimeError("source command SHA-256 mismatch")
    command = list(source)
    replacements = {
        "--output-dir": str(OUTPUT_DIR),
        "--updates": "472",
        "--games-per-update": "96",
        "--minibatch-size": "512",
        "--learning-rate": "2.4e-05",
        "--schedule-start-update": "469",
        "--bc-kl-start": "0.016",
        "--bc-kl-end": "0.012",
        "--max-pool-size": "12",
        "--eval-interval": "472",
        "--checkpoint-interval": "472",
        "--seed": str(SEED),
        "--resume": str(PARENT),
        "--bc-replay-data": str(REPLAY),
        "--bc-replay-lr-scale": "0.075",
    }
    for flag, value in replacements.items():
        replace_flag(command, flag, value)
    output_index = command.index("--output-dir")
    command[output_index:output_index] = ["--extra-opponent", str(FLG_CHECKPOINT), str(FLG_DECK)]
    for index, token in enumerate(command):
        if token == "--opponent-base-quota":
            name = command[index + 1]
            if name not in FIXED_QUOTAS:
                raise RuntimeError(f"unexpected quota name: {name}")
            command[index + 2] = str(FIXED_QUOTAS[name])
    command.extend(["--opponent-base-quota", FLG_NAME, str(FIXED_QUOTAS[FLG_NAME])])
    if len(command) != 185:
        raise RuntimeError(f"derived command token count drifted: {len(command)}")
    return command


def quota_mapping(command: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, token in enumerate(command):
        if token == "--opponent-base-quota":
            name = command[index + 1]
            if name in result:
                raise RuntimeError("duplicate quota name")
            result[name] = int(command[index + 2])
    return result


def opponent_bindings(command: Sequence[str]) -> dict[Path, str]:
    manifests = []
    for path, digest in ((CURRENT_LEAGUE, CURRENT_LEAGUE_SHA256), (EXACT_LEAGUE, EXACT_LEAGUE_SHA256)):
        if sha256_file(path) != digest:
            raise RuntimeError("league manifest SHA-256 mismatch")
        manifests.extend(json.loads(path.read_text(encoding="utf-8"))["opponents"])
    by_path = {Path(item["checkpoint"]).resolve(): item["checkpoint_sha256"] for item in manifests}
    result: dict[Path, str] = {}
    for index, token in enumerate(command):
        if token != "--extra-opponent":
            continue
        checkpoint = Path(command[index + 1]).resolve()
        deck = Path(command[index + 2]).resolve()
        if checkpoint not in by_path:
            raise RuntimeError(f"opponent absent from bound manifests: {checkpoint}")
        result[checkpoint] = by_path[checkpoint]
        if deck not in DECK_BINDINGS:
            raise RuntimeError(f"opponent deck absent from bindings: {deck}")
    if len(result) != 11:
        raise RuntimeError("expected exactly 11 extra opponent checkpoints")
    return result


def parent_audit() -> dict[str, Any]:
    if sha256_file(PARENT) != PARENT_SHA256:
        raise RuntimeError("G8 parent SHA-256 mismatch")
    raw = torch.load(PARENT, map_location="cpu", weights_only=False)
    bc = torch.load(GENERAL_BC, map_location="cpu", weights_only=False)
    model = ppo.instantiate_model_from_checkpoint(raw, bc, torch.device("cpu"))
    observed = {
        "update": int(raw.get("update", -1)),
        "model_state_sha256": ppo.model_state_sha256(model),
        "ppo_steps": sorted(audit.state_steps(raw["optimizer_state_dict"])),
        "replay_steps": sorted(audit.state_steps(raw["bc_replay_optimizer_state_dict"])),
        "replay_state_sha256": audit.nested_sha256(raw["bc_replay_optimizer_state_dict"]),
        "quota_state_sha256": audit.nested_sha256(raw["opponent_quota_state"]),
        "g8_status": raw.get("post_ppo_general_bc", {}).get("status"),
        "g8_integrity_step": raw.get("post_ppo_general_bc", {}).get("integrity", {}).get("replay_step_after"),
    }
    expected = {
        "update": 468,
        "model_state_sha256": "16754b5fd967365e267519f2a2ab971800dbfe2e28642637a598ece62169c3ef",
        "ppo_steps": [468],
        "replay_steps": [32],
        "replay_state_sha256": "6cbc2d893efdd4a8cec56db82a5d20c718cadad324667a8cea2be6ee67a35f38",
        "quota_state_sha256": "2800876baf7f5d2b89971e2494e389a153b8acf09c8677cee010ef652423e851",
        "g8_status": "general_bc_completed",
        "g8_integrity_step": 32,
    }
    if observed != expected:
        raise RuntimeError(f"G8 parent runtime identity mismatch: {observed}")
    return observed


def static_bindings(command: Sequence[str], protocol: Path) -> dict[Path, str]:
    result = {
        SELF: sha256_file(SELF), protocol.resolve(): sha256_file(protocol),
        SOURCE_PROTOCOL: SOURCE_PROTOCOL_SHA256, TRAINER: TRAINER_SHA256,
        PARENT: PARENT_SHA256, G8_MANIFEST: G8_MANIFEST_SHA256,
        GENERAL_BC: GENERAL_BC_SHA256, REPLAY: REPLAY_SHA256,
        CURRENT_LEAGUE: CURRENT_LEAGUE_SHA256, EXACT_LEAGUE: EXACT_LEAGUE_SHA256,
        **DEPENDENCIES, **DECK_BINDINGS, **opponent_bindings(command),
    }
    return {path.resolve(): digest for path, digest in result.items()}


def validate(protocol_path: Path, expected_sha256: str) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("launcher requires repository cwd and my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("launcher requires Python -I -B")
    if sha256_file(protocol_path) != expected_sha256:
        raise RuntimeError("execution protocol SHA-256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    command = derive_command()
    digest = command_sha256(command)
    if protocol != {
        "schema_version": SCHEMA,
        "status": "locked_after_launcher_and_before_training",
        "seed": SEED,
        "launcher_sha256": sha256_file(SELF),
        "command_sha256": digest,
        "command_tokens": 185,
        "parent_sha256": PARENT_SHA256,
        "replay_sha256": REPLAY_SHA256,
        "output_dir": str(OUTPUT_DIR),
        "terminal_checkpoint": str(TERMINAL),
        "attempt_marker": str(ATTEMPT),
        "log": str(LOG),
        "updates": [469, 470, 471, 472],
        "games_per_update": 96,
        "fixed_quota_sum": 96,
        "scope": {"local_only": True, "network": False, "package": False, "upload": False, "submission": False},
    }:
        raise RuntimeError("execution protocol differs from exact launcher contract")
    quotas = quota_mapping(command)
    if quotas != FIXED_QUOTAS or sum(quotas.values()) != 96 or any(value % 2 for value in quotas.values()):
        raise RuntimeError("fixed seat-balanced quota contract drifted")
    for target in (OUTPUT_ROOT, OUTPUT_DIR, TERMINAL, ATTEMPT, LOG):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing existing target: {target}")
    bindings = static_bindings(command, protocol_path)
    for path, digest_expected in bindings.items():
        if sha256_file(path) != digest_expected:
            raise RuntimeError(f"input SHA-256 mismatch: {path}")
    return protocol, command, {
        "status": "passed", "command_sha256": digest, "command_tokens": len(command),
        "fixed_quotas": quotas, "parent": parent_audit(), "input_files": len(bindings),
        "targets_absent": True, "optimizer_steps": 0, "rollout_games": 0,
    }


def claim_and_run(protocol_path: Path, protocol: dict[str, Any], command: list[str]) -> int:
    marker = {
        "schema_version": SCHEMA, "event": "fresh_ppo_attempt_consumed",
        "seed": SEED, "protocol_sha256": sha256_file(protocol_path),
        "command_sha256": protocol["command_sha256"], "launcher_sha256": sha256_file(SELF),
    }
    fd = os.open(ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, audit.canonical_json_bytes(marker)); os.fsync(fd)
    finally:
        os.close(fd)
    bindings = static_bindings(command, protocol_path)
    snapshots = audit.acquire_input_locks(sorted(bindings))
    try:
        audit.assert_input_locks_unchanged(snapshots)
        os.mkdir(OUTPUT_ROOT, mode=0o700)
        log_fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(log_fd, "wb", buffering=0) as handle:
            effective = [command[0], "-I", "-B", *command[1:]]
            handle.write(audit.canonical_json_bytes({**marker, "event": "pre_exec_checks_passed", "effective_argv_prefix": effective[:4], "input_shared_locks": len(snapshots)}))
            env = os.environ.copy(); env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            completed = subprocess.run(effective, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False)
            audit.assert_input_locks_unchanged(snapshots)
            handle.write(audit.canonical_json_bytes({"event": "child_terminal", "return_code": completed.returncode, "inputs_unchanged": True}))
            os.fsync(handle.fileno())
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
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return claim_and_run(args.protocol.resolve(), protocol, command)


if __name__ == "__main__":
    raise SystemExit(main())
