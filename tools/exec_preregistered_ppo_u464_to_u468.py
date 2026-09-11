#!/usr/bin/env python3
"""Audit or execute the exact hash-bound U464-to-U468 PPO continuation."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import stat
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import train_ppo as ppo  # noqa: E402


SCHEMA_VERSION = "ptcg-u464-to-u468-resume-all-execution-preregistration-v1"
PROTOCOL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "created_at_utc",
        "design_preregistration",
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
        "scope",
    }
)
BINDING_KEYS = frozenset(
    {
        "launcher",
        "trainer",
        "train_bc_orbit",
        "cg_init",
        "cg_sim",
        "cg_lib",
        "parent_checkpoint",
        "general_bc",
        "bc_replay_archive",
        "candidate_deck",
        "training_league_manifest",
        "source_command_preregistration",
        "u464_training_integrity_decision",
        "u464_gold19_decision",
    }
)

DESIGN_PATH = REPO_ROOT / (
    "artifacts/ppo_u464inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_resumeall_u464_to_u468_seed202607336."
    "design_preregistration_v2.json"
)
DESIGN_SHA256 = "6e95405ceca1854ab4b944b450deede4bf564f886e731cbf21a859a42fa6d891"

SOURCE_COMMAND_PROTOCOL = REPO_ROOT / (
    "artifacts/ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_u456_to_u464_seed202607331."
    "B_gold_league.seed-202607331.preregistration.json"
)
SOURCE_COMMAND_PROTOCOL_SHA256 = (
    "b216a5e51669f3c30f0fd3b4d500fc5704a2e091c8ba7b5751c81ff110abff58"
)
SOURCE_COMMAND_SHA256 = (
    "cbc1345e9043ec1d6fcd7e571041d9b4066090c6d6a5a2242d06935e54f32d22"
)
DERIVED_COMMAND_SHA256 = (
    "5140ca65e1c5fd587ba0bcf80c28f2b63e062722bbe23afb650897228f27f486"
)
DERIVED_COMMAND_TOKENS = 177

TRAINER_PATH = REPO_ROOT / "tools/train_ppo.py"
TRAINER_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
TRAIN_BC_ORBIT_PATH = REPO_ROOT / "tools/train_bc_orbit.py"
TRAIN_BC_ORBIT_SHA256 = (
    "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
)
CG_INIT_PATH = REPO_ROOT / "dataset/sample_submission/sample_submission/cg/__init__.py"
CG_INIT_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
CG_SIM_PATH = REPO_ROOT / "dataset/sample_submission/sample_submission/cg/sim.py"
CG_SIM_SHA256 = "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655"
CG_LIB_PATH = REPO_ROOT / "dataset/sample_submission/sample_submission/cg/libcg.so"
CG_LIB_SHA256 = "feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887"
PARENT_PATH = REPO_ROOT / (
    "artifacts/ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_localtransport_u456_to_u464_"
    "seed202607336/B_gold_league/seed-202607336/checkpoints/update-0464.pt"
)
PARENT_SHA256 = "fe51f40f37fca329cd6b0c94f7431001bb909b92624df33fcd7cf6f0da976264"
GENERAL_BC_PATH = REPO_ROOT / (
    "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_"
    "seed20260922_20260731/best.pt"
)
GENERAL_BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
BC_REPLAY_PATH = REPO_ROOT / (
    "data/bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip"
)
BC_REPLAY_SHA256 = "a3b9d572bfc0a784b5b140d3dbe09314252c46dd0d9c1afa3423236cda54543c"
CANDIDATE_DECK_PATH = REPO_ROOT / "data/decks/marnie_grimmsnarl_froslass_luca.csv"
CANDIDATE_DECK_SHA256 = "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d"
LEAGUE_MANIFEST_PATH = (
    REPO_ROOT / "artifacts/gold_clone_league_current_proxy_20260731/league_manifest.json"
)
LEAGUE_MANIFEST_SHA256 = (
    "148dd03fc550b062153080d053275ac5adc11feefdecd241cc1b25faadb7306f"
)
U464_TRAINING_DECISION_PATH = REPO_ROOT / (
    "artifacts/ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_localtransport_u456_to_u464_"
    "seed202607336.training_integrity_decision.json"
)
U464_TRAINING_DECISION_SHA256 = (
    "95912e10d5f03e0f2baaff196889bc5c218b4306cc7d0acbc58e5b535ee54781"
)
U464_GOLD_DECISION_PATH = REPO_ROOT / (
    "artifacts/ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_localtransport_u456_to_u464_"
    "seed202607336.gold19_screen128_decision.json"
)
U464_GOLD_DECISION_SHA256 = (
    "90166c01244a9195025be9ba362bab83f244fa2230add9394eefa7dbd34a0326"
)

OUTPUT_DIR = REPO_ROOT / (
    "artifacts/ppo_u464inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_resumeall_u464_to_u468_seed202607336/"
    "B_gold_league/seed-202607336"
)
OUTPUT_ROOT = OUTPUT_DIR.parents[1]
TERMINAL_CHECKPOINT = OUTPUT_DIR / "checkpoints/update-0468.pt"
ATTEMPT_MARKER = REPO_ROOT / (
    ".ptcg-ppo-resumeall-u464-u468-attempt-202607336-202608071.json"
)
LOG_PATH = REPO_ROOT / (
    "artifacts/ppo_u464inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_resumeall_u464_to_u468_seed202607336.log"
)

EXPECTED_PARENT_STATE = {
    "update": 464,
    "runtime_model_state_sha256": (
        "fa6e42da654f872f6f24836838654286d7cd64e10cacff554bdec36b6aa58f0a"
    ),
    "ppo_optimizer_nested_sha256": (
        "904a3d2c8ac63c8a60af2404c216830b8ae30a69fbe972818e05c0136058a54f"
    ),
    "ppo_optimizer_state_count": 28,
    "ppo_optimizer_step": 276,
    "bc_replay_optimizer_nested_sha256": (
        "23f775207b0d216c1ffd19bf9213456026b71b786727a4d2ff0f889b6b30341d"
    ),
    "bc_replay_optimizer_state_count": 24,
    "bc_replay_optimizer_step": 16,
    "opponent_quota_nested_sha256": (
        "7139320983450f2d2b9d50a65684c31550c409ed8adde9feb84839db24e1fc27"
    ),
    "opponent_quota_observed_games": 512,
    "opponent_quota_last_refresh_update": 463,
}

MFD_CLOEXEC = 0x0001
MFD_ALLOW_SEALING = 0x0002
F_ADD_SEALS = 1033
F_GET_SEALS = 1034
F_SEAL_SEAL = 0x0001
F_SEAL_SHRINK = 0x0002
F_SEAL_GROW = 0x0004
F_SEAL_WRITE = 0x0008
REQUIRED_MEMFD_SEALS = F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL

EXPECTED_DISTINCT_TRAINING_DECK_HASHES = {
    str(
        (
            REPO_ROOT
            / "data/gold_league/gold21_proxy_20260731/decks/"
            "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af.csv"
        ).resolve()
    ): "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
    str(
        (
            REPO_ROOT
            / "data/gold_league/gold21_proxy_20260731/decks/"
            "e2e03fe8ef9592b1204c159a7725da3945675fa872b1e18baca550560503a78e.csv"
        ).resolve()
    ): "c960f71296a4ca797d8b8421f8c3d059752296ba34721108e6b625e02ca0410a",
    str(
        (
            REPO_ROOT
            / "data/gold_league/gold21_proxy_20260731/decks/"
            "882b584691fc15581b9c7d238ea131a933050f9da1694d8418c790d33c13fc54.csv"
        ).resolve()
    ): "778b260c3c211be1b5317533b437849d2949ae4392eb6f693440d9b6e67cc0ea",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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


def update_nested_digest(digest: Any, value: Any) -> None:
    """Hash nested optimizer state without relying on torch serialization."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            update_nested_digest(digest, key)
            update_nested_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        for item in value:
            update_nested_digest(digest, item)
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bool):
        digest.update(b"bool\0")
        digest.update(b"1\0" if value else b"0\0")
        return
    if isinstance(value, int):
        digest.update(b"int\0")
        digest.update(str(value).encode("ascii"))
        digest.update(b"\0")
        return
    if isinstance(value, float):
        digest.update(b"float\0")
        digest.update(value.hex().encode("ascii"))
        digest.update(b"\0")
        return
    if isinstance(value, str):
        digest.update(b"str\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
        return
    raise TypeError(f"Unsupported nested hash value: {type(value).__name__}")


def nested_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    update_nested_digest(digest, value)
    return digest.hexdigest()


def require_regular(path: Path, label: str) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository") from error
    current = REPO_ROOT
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} must be symlink-free")
    if not path.is_file():
        raise ValueError(f"{label} must be a regular symlink-free file")
    return path


def require_bound_file(
    value: Any,
    expected_path: Path,
    expected_sha256: str,
    label: str,
) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    raw_path = Path(value["path"])
    path = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
    if require_regular(path, label) != expected_path.resolve():
        raise ValueError(f"{label} path mismatch")
    if value["sha256"] != expected_sha256 or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")


def flag_value(command: Sequence[str], flag: str) -> str:
    indices = [index for index, token in enumerate(command) if token == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError(f"command must contain exactly one {flag}")
    return command[indices[0] + 1]


def replace_flag_value(command: list[str], flag: str, value: str) -> None:
    index = command.index(flag)
    if command.count(flag) != 1 or index + 1 >= len(command):
        raise ValueError(f"source command must contain exactly one {flag}")
    command[index + 1] = value


def derive_exact_command() -> list[str]:
    require_regular(SOURCE_COMMAND_PROTOCOL, "source command protocol")
    if sha256_file(SOURCE_COMMAND_PROTOCOL) != SOURCE_COMMAND_PROTOCOL_SHA256:
        raise ValueError("source command protocol SHA-256 mismatch")
    source_protocol = json.loads(SOURCE_COMMAND_PROTOCOL.read_text())
    try:
        source = source_protocol["binding"]["command"]
    except (KeyError, TypeError) as error:
        raise ValueError("source command is missing") from error
    if not isinstance(source, list) or not all(isinstance(token, str) for token in source):
        raise ValueError("source command must be a string array")
    if sha256_bytes(canonical_json_bytes(source)) != SOURCE_COMMAND_SHA256:
        raise ValueError("source command SHA-256 mismatch")
    command = list(source)
    replacements = {
        "--output-dir": str(OUTPUT_DIR),
        "--updates": "468",
        "--seed": "202607336",
        "--resume": str(PARENT_PATH),
    }
    for flag, value in replacements.items():
        replace_flag_value(command, flag, value)
    for flag in ("--reset-optimizer-on-resume", "--reset-opponent-quota-on-resume"):
        if command.count(flag) != 1:
            raise ValueError(f"source command must contain exactly one {flag}")
        command.remove(flag)
    if len(command) != DERIVED_COMMAND_TOKENS:
        raise ValueError("derived command token count mismatch")
    if sha256_bytes(canonical_json_bytes(command)) != DERIVED_COMMAND_SHA256:
        raise ValueError("derived command SHA-256 mismatch")
    return command


def state_steps(state_dict: dict[str, Any]) -> set[int]:
    state = state_dict.get("state")
    if not isinstance(state, dict):
        raise ValueError("optimizer state mapping is missing")
    observed: set[int] = set()
    for entry in state.values():
        if not isinstance(entry, dict) or "step" not in entry:
            raise ValueError("optimizer state entry lacks step")
        step = entry["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise ValueError("optimizer step tensor is not scalar")
            observed.add(int(step.item()))
        else:
            observed.add(int(step))
    return observed


def quota_observed_games(quota: dict[str, Any]) -> int:
    observed = quota.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("quota observed mapping is missing")
    total = 0
    for result in observed.values():
        if not isinstance(result, dict):
            raise ValueError("quota observed entry is malformed")
        total += sum(int(result.get(key, 0)) for key in ("wins", "losses", "draws"))
    return total


def audit_parent_state() -> dict[str, Any]:
    parent = torch.load(PARENT_PATH, map_location="cpu", weights_only=False)
    general_bc = torch.load(GENERAL_BC_PATH, map_location="cpu", weights_only=False)
    if int(parent.get("update", -1)) != 464:
        raise ValueError("parent update mismatch")
    model = ppo.instantiate_model_from_bc(general_bc, torch.device("cpu"))
    model.load_state_dict(parent["model_state_dict"])
    model_hash = ppo.model_state_sha256(model)
    if model_hash != EXPECTED_PARENT_STATE["runtime_model_state_sha256"]:
        raise ValueError("parent runtime model hash mismatch")

    raw_config = parent.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("parent config is missing")
    if int(raw_config.get("seed", -1)) != 202607336:
        raise ValueError("parent quota identity seed mismatch")
    actor, value, manifest = ppo.configure_trainable_scope(
        model, str(raw_config["trainable_scope"])
    )
    ppo.validate_optimizer_resume_compatibility(
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
    if nested_sha256(ppo_state) != EXPECTED_PARENT_STATE["ppo_optimizer_nested_sha256"]:
        raise ValueError("parent PPO optimizer hash mismatch")
    if nested_sha256(replay_state) != EXPECTED_PARENT_STATE["bc_replay_optimizer_nested_sha256"]:
        raise ValueError("parent replay optimizer hash mismatch")
    if nested_sha256(quota) != EXPECTED_PARENT_STATE["opponent_quota_nested_sha256"]:
        raise ValueError("parent quota hash mismatch")
    if len(ppo_state.get("state", {})) != EXPECTED_PARENT_STATE["ppo_optimizer_state_count"]:
        raise ValueError("parent PPO optimizer state count mismatch")
    if state_steps(ppo_state) != {EXPECTED_PARENT_STATE["ppo_optimizer_step"]}:
        raise ValueError("parent PPO optimizer step mismatch")
    if len(replay_state.get("state", {})) != EXPECTED_PARENT_STATE["bc_replay_optimizer_state_count"]:
        raise ValueError("parent replay optimizer state count mismatch")
    if state_steps(replay_state) != {EXPECTED_PARENT_STATE["bc_replay_optimizer_step"]}:
        raise ValueError("parent replay optimizer step mismatch")
    if quota_observed_games(quota) != EXPECTED_PARENT_STATE["opponent_quota_observed_games"]:
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
        lr=float(raw_config["learning_rate"]) * float(raw_config["bc_replay_lr_scale"]),
        eps=1e-5,
        weight_decay=float(raw_config["weight_decay"]),
    )
    optimizer.load_state_dict(ppo_state)
    replay_optimizer.load_state_dict(replay_state)
    return {
        "runtime_model_state_sha256": model_hash,
        "ppo_optimizer_state_count": len(optimizer.state_dict()["state"]),
        "ppo_optimizer_steps": sorted(state_steps(optimizer.state_dict())),
        "bc_replay_optimizer_state_count": len(replay_optimizer.state_dict()["state"]),
        "bc_replay_optimizer_steps": sorted(state_steps(replay_optimizer.state_dict())),
        "quota_observed_games": quota_observed_games(quota),
        "quota_last_refresh_update": int(quota["last_refresh_update"]),
        "resume_compatibility": "passed",
    }


def audit_training_inputs(command: Sequence[str]) -> dict[str, Any]:
    manifest = json.loads(LEAGUE_MANIFEST_PATH.read_text())
    opponents = manifest.get("opponents")
    if not isinstance(opponents, list):
        raise ValueError("training league manifest opponents are missing")
    manifest_by_checkpoint = {
        str(Path(item["checkpoint"]).resolve()): item
        for item in opponents
        if isinstance(item, dict) and isinstance(item.get("checkpoint"), str)
    }
    checkpoint_paths: list[Path] = []
    deck_paths: list[Path] = []
    for index, token in enumerate(command):
        if token == "--extra-opponent":
            if index + 2 >= len(command):
                raise ValueError("extra-opponent command is truncated")
            checkpoint_paths.append(Path(command[index + 1]).resolve())
            deck_paths.append(Path(command[index + 2]).resolve())
    if len(checkpoint_paths) != 10:
        raise ValueError("expected exactly ten training opponent checkpoints")
    for checkpoint in checkpoint_paths:
        require_regular(checkpoint, "training opponent checkpoint")
        item = manifest_by_checkpoint.get(str(checkpoint))
        if item is None:
            raise ValueError("training opponent is absent from bound league manifest")
        if sha256_file(checkpoint) != item.get("checkpoint_sha256"):
            raise ValueError("training opponent checkpoint SHA-256 mismatch")
    observed_decks = {str(path) for path in deck_paths}
    if observed_decks != set(EXPECTED_DISTINCT_TRAINING_DECK_HASHES):
        raise ValueError("training opponent deck set mismatch")
    for path_text, expected in EXPECTED_DISTINCT_TRAINING_DECK_HASHES.items():
        path = require_regular(Path(path_text), "training opponent deck")
        if sha256_file(path) != expected:
            raise ValueError("training opponent deck SHA-256 mismatch")
    return {
        "opponent_checkpoints": len(checkpoint_paths),
        "distinct_opponent_decks": len(observed_decks),
        "league_manifest_binding": "passed",
    }


def command_input_paths(command: Sequence[str]) -> list[Path]:
    paths = {
        Path(__file__),
        TRAINER_PATH,
        TRAIN_BC_ORBIT_PATH,
        CG_INIT_PATH,
        CG_SIM_PATH,
        CG_LIB_PATH,
        DESIGN_PATH,
        PARENT_PATH,
        GENERAL_BC_PATH,
        BC_REPLAY_PATH,
        CANDIDATE_DECK_PATH,
        LEAGUE_MANIFEST_PATH,
        SOURCE_COMMAND_PROTOCOL,
        U464_TRAINING_DECISION_PATH,
        U464_GOLD_DECISION_PATH,
    }
    for index, token in enumerate(command):
        if token == "--extra-opponent":
            paths.add(Path(command[index + 1]))
            paths.add(Path(command[index + 2]))
    return sorted((require_regular(path, "locked input") for path in paths), key=str)


def sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def libc_memfd_create(name: str) -> int:
    if sys.platform != "linux" or not hasattr(os, "pread"):
        raise RuntimeError("sealed memfd trainer transport requires Linux and os.pread")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        memfd_create = libc.memfd_create
    except AttributeError as error:
        raise RuntimeError("libc.memfd_create is unavailable") from error
    memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    memfd_create.restype = ctypes.c_int
    descriptor = int(
        memfd_create(name.encode("ascii"), MFD_CLOEXEC | MFD_ALLOW_SEALING)
    )
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), name)
    return descriptor


def create_sealed_memfd(name: str, payload: bytes) -> tuple[int, str]:
    fd = libc_memfd_create(name)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise RuntimeError(f"short write to {name} memfd")
            offset += written
        os.fsync(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != len(payload):
            raise RuntimeError(f"{name} memfd shape mismatch")
        expected = hashlib.sha256(payload).hexdigest()
        if sha256_fd(fd) != expected:
            raise RuntimeError(f"{name} memfd hash mismatch before sealing")
        fcntl.fcntl(fd, F_ADD_SEALS, REQUIRED_MEMFD_SEALS)
        if int(fcntl.fcntl(fd, F_GET_SEALS)) != REQUIRED_MEMFD_SEALS:
            raise RuntimeError(f"{name} memfd seal mismatch")
        digest = sha256_fd(fd)
        if digest != expected:
            raise RuntimeError(f"{name} memfd hash mismatch after sealing")
        return fd, digest
    except BaseException:
        os.close(fd)
        raise


def build_trainer_bootstrap(payload_fd: int) -> bytes:
    canonical = json.dumps(str(TRAINER_PATH), ensure_ascii=True)
    digest = json.dumps(TRAINER_SHA256)
    return (
        "import fcntl as _f\n"
        "import hashlib as _h\n"
        "import os as _o\n"
        "import stat as _st\n"
        "import sys as _s\n"
        f"_canonical={canonical}\n"
        f"_expected={digest}\n"
        f"_fd={payload_fd}\n"
        f"_required={REQUIRED_MEMFD_SEALS}\n"
        f"_get_seals={F_GET_SEALS}\n"
        "if int(_f.fcntl(_fd,_get_seals)) != _required:\n"
        "    raise RuntimeError('trainer payload seal mismatch')\n"
        "_before=_o.fstat(_fd)\n"
        "if not _st.S_ISREG(_before.st_mode):\n"
        "    raise RuntimeError('trainer payload is not regular')\n"
        "_chunks=[]\n"
        "_offset=0\n"
        "while _offset < _before.st_size:\n"
        "    _chunk=_o.pread(_fd,min(1048576,_before.st_size-_offset),_offset)\n"
        "    if not _chunk:\n"
        "        raise RuntimeError('short trainer payload read')\n"
        "    _chunks.append(_chunk)\n"
        "    _offset += len(_chunk)\n"
        "_payload=b''.join(_chunks)\n"
        "_after=_o.fstat(_fd)\n"
        "if (_before.st_dev,_before.st_ino,_before.st_size,_before.st_mtime_ns) != (_after.st_dev,_after.st_ino,_after.st_size,_after.st_mtime_ns):\n"
        "    raise RuntimeError('trainer payload identity changed')\n"
        "if _h.sha256(_payload).hexdigest() != _expected:\n"
        "    raise RuntimeError('trainer payload hash mismatch')\n"
        "_root=_o.path.dirname(_o.path.dirname(_canonical))\n"
        "if _root not in _s.path:\n"
        "    _s.path.insert(0,_root)\n"
        "_s.argv[0]=_canonical\n"
        "globals()['__file__']=_canonical\n"
        "globals()['__cached__']=None\n"
        "globals()['__loader__']=None\n"
        "globals()['__package__']=None\n"
        "globals()['__spec__']=None\n"
        "exec(compile(_payload,_canonical,'exec'),globals(),globals())\n"
    ).encode("utf-8")


def read_fd_bytes(fd: int) -> bytes:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("held input descriptor is not regular")
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(fd, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise RuntimeError("short read from held input descriptor")
        chunks.append(chunk)
        offset += len(chunk)
    raw = b"".join(chunks)
    after = os.fstat(fd)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(raw) != after.st_size:
        raise RuntimeError("held input changed while being read")
    return raw


def snapshot_by_path(
    snapshots: Sequence[dict[str, Any]], path: Path
) -> dict[str, Any]:
    target = path.resolve()
    matches = [snapshot for snapshot in snapshots if snapshot["path"] == target]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one held snapshot for {target}")
    return matches[0]


def assert_locked_input_bindings(
    snapshots: Sequence[dict[str, Any]],
    protocol: dict[str, Any],
    command: Sequence[str],
    protocol_path: Path,
    protocol_sha256: str,
) -> None:
    bindings = protocol["bindings"]
    expected: dict[Path, str] = {
        Path(__file__).resolve(): bindings["launcher"]["sha256"],
        TRAINER_PATH.resolve(): TRAINER_SHA256,
        TRAIN_BC_ORBIT_PATH.resolve(): TRAIN_BC_ORBIT_SHA256,
        CG_INIT_PATH.resolve(): CG_INIT_SHA256,
        CG_SIM_PATH.resolve(): CG_SIM_SHA256,
        CG_LIB_PATH.resolve(): CG_LIB_SHA256,
        DESIGN_PATH.resolve(): DESIGN_SHA256,
        PARENT_PATH.resolve(): PARENT_SHA256,
        GENERAL_BC_PATH.resolve(): GENERAL_BC_SHA256,
        BC_REPLAY_PATH.resolve(): BC_REPLAY_SHA256,
        CANDIDATE_DECK_PATH.resolve(): CANDIDATE_DECK_SHA256,
        LEAGUE_MANIFEST_PATH.resolve(): LEAGUE_MANIFEST_SHA256,
        SOURCE_COMMAND_PROTOCOL.resolve(): SOURCE_COMMAND_PROTOCOL_SHA256,
        U464_TRAINING_DECISION_PATH.resolve(): U464_TRAINING_DECISION_SHA256,
        U464_GOLD_DECISION_PATH.resolve(): U464_GOLD_DECISION_SHA256,
        protocol_path.resolve(): protocol_sha256,
    }

    manifest_snapshot = snapshot_by_path(snapshots, LEAGUE_MANIFEST_PATH)
    if manifest_snapshot["sha256"] != LEAGUE_MANIFEST_SHA256:
        raise RuntimeError("held league manifest SHA-256 differs from binding")
    manifest = json.loads(read_fd_bytes(int(manifest_snapshot["fd"])))
    opponents = manifest.get("opponents")
    if not isinstance(opponents, list):
        raise RuntimeError("held league manifest opponents are missing")
    manifest_by_checkpoint = {
        Path(item["checkpoint"]).resolve(): item.get("checkpoint_sha256")
        for item in opponents
        if isinstance(item, dict)
        and isinstance(item.get("checkpoint"), str)
        and isinstance(item.get("checkpoint_sha256"), str)
    }
    command_decks: set[Path] = set()
    for index, token in enumerate(command):
        if token != "--extra-opponent":
            continue
        checkpoint = Path(command[index + 1]).resolve()
        deck = Path(command[index + 2]).resolve()
        checkpoint_sha256 = manifest_by_checkpoint.get(checkpoint)
        if checkpoint_sha256 is None:
            raise RuntimeError("held manifest lacks a command opponent checkpoint")
        expected[checkpoint] = checkpoint_sha256
        command_decks.add(deck)
    expected_decks = {
        Path(path_text).resolve(): digest
        for path_text, digest in EXPECTED_DISTINCT_TRAINING_DECK_HASHES.items()
    }
    if command_decks != set(expected_decks):
        raise RuntimeError("held command opponent deck set mismatch")
    expected.update(expected_decks)

    observed = {Path(snapshot["path"]).resolve(): snapshot for snapshot in snapshots}
    if set(observed) != set(expected):
        raise RuntimeError("held input path set differs from frozen bindings")
    for path, expected_sha256 in expected.items():
        if observed[path]["sha256"] != expected_sha256:
            raise RuntimeError(f"held input SHA-256 differs from binding: {path}")


def assert_sealed_memfd(fd: int, expected_sha256: str, label: str) -> None:
    descriptor_stat = os.fstat(fd)
    if not stat.S_ISREG(descriptor_stat.st_mode):
        raise RuntimeError(f"{label} memfd is not regular")
    if int(fcntl.fcntl(fd, F_GET_SEALS)) != REQUIRED_MEMFD_SEALS:
        raise RuntimeError(f"{label} memfd seal mismatch")
    if sha256_fd(fd) != expected_sha256:
        raise RuntimeError(f"{label} memfd SHA-256 mismatch")


def acquire_input_locks(paths: Sequence[Path]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    try:
        for path in paths:
            fd = -1
            try:
                flags = os.O_RDONLY | os.O_CLOEXEC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(path, flags)
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                descriptor_stat = os.fstat(fd)
                path_stat = os.stat(path, follow_symlinks=False)
                if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
                    path_stat.st_dev,
                    path_stat.st_ino,
                ):
                    raise RuntimeError(
                        f"input identity changed while locking: {path}"
                    )
                descriptor_sha256 = sha256_fd(fd)
                if descriptor_sha256 != sha256_file(path):
                    raise RuntimeError(f"input content changed while locking: {path}")
                snapshots.append(
                    {
                        "path": path,
                        "fd": fd,
                        "device": descriptor_stat.st_dev,
                        "inode": descriptor_stat.st_ino,
                        "size": descriptor_stat.st_size,
                        "mtime_ns": descriptor_stat.st_mtime_ns,
                        "sha256": descriptor_sha256,
                    }
                )
                fd = -1
            finally:
                if fd >= 0:
                    os.close(fd)
    except BaseException:
        release_input_locks(snapshots)
        raise
    return snapshots


def assert_input_locks_unchanged(snapshots: Sequence[dict[str, Any]]) -> None:
    for snapshot in snapshots:
        path = require_regular(Path(snapshot["path"]), "locked input")
        descriptor_stat = os.fstat(int(snapshot["fd"]))
        path_stat = os.stat(path, follow_symlinks=False)
        observed_identity = (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
            descriptor_stat.st_size,
            descriptor_stat.st_mtime_ns,
        )
        expected_identity = (
            snapshot["device"],
            snapshot["inode"],
            snapshot["size"],
            snapshot["mtime_ns"],
        )
        if observed_identity != expected_identity:
            raise RuntimeError(f"locked input descriptor changed: {path}")
        if (path_stat.st_dev, path_stat.st_ino) != (
            snapshot["device"],
            snapshot["inode"],
        ):
            raise RuntimeError(f"locked input path identity changed: {path}")
        if sha256_fd(int(snapshot["fd"])) != snapshot["sha256"]:
            raise RuntimeError(f"locked input content changed: {path}")


def release_input_locks(snapshots: Sequence[dict[str, Any]]) -> None:
    for snapshot in reversed(list(snapshots)):
        fd = int(snapshot["fd"])
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def load_and_validate_protocol(
    protocol_path: Path,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError("current working directory must equal repository root")
    protocol_path = require_regular(protocol_path, "execution protocol")
    raw = protocol_path.read_bytes()
    if sha256_bytes(raw) != expected_protocol_sha256:
        raise ValueError("execution protocol SHA-256 mismatch")
    protocol = json.loads(raw)
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_KEYS:
        raise ValueError("execution protocol keys differ from the frozen schema")
    if protocol["schema_version"] != SCHEMA_VERSION:
        raise ValueError("execution protocol schema mismatch")
    if protocol["status"] != "locked_after_launcher_and_before_training":
        raise ValueError("execution protocol status mismatch")
    if protocol["seed"] != 202607336:
        raise ValueError("execution protocol seed mismatch")

    require_bound_file(protocol["design_preregistration"], DESIGN_PATH, DESIGN_SHA256, "design")
    bindings = protocol["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != BINDING_KEYS:
        raise ValueError("execution binding keys mismatch")
    require_bound_file(bindings["launcher"], Path(__file__), sha256_file(Path(__file__)), "launcher")
    require_bound_file(bindings["trainer"], TRAINER_PATH, TRAINER_SHA256, "trainer")
    require_bound_file(
        bindings["train_bc_orbit"],
        TRAIN_BC_ORBIT_PATH,
        TRAIN_BC_ORBIT_SHA256,
        "train BC orbit dependency",
    )
    require_bound_file(
        bindings["cg_init"], CG_INIT_PATH, CG_INIT_SHA256, "cg package dependency"
    )
    require_bound_file(
        bindings["cg_sim"], CG_SIM_PATH, CG_SIM_SHA256, "cg simulator Python dependency"
    )
    require_bound_file(
        bindings["cg_lib"], CG_LIB_PATH, CG_LIB_SHA256, "cg simulator native dependency"
    )
    require_bound_file(bindings["parent_checkpoint"], PARENT_PATH, PARENT_SHA256, "parent checkpoint")
    require_bound_file(bindings["general_bc"], GENERAL_BC_PATH, GENERAL_BC_SHA256, "general BC")
    require_bound_file(bindings["bc_replay_archive"], BC_REPLAY_PATH, BC_REPLAY_SHA256, "BC replay archive")
    require_bound_file(bindings["candidate_deck"], CANDIDATE_DECK_PATH, CANDIDATE_DECK_SHA256, "candidate deck")
    require_bound_file(bindings["training_league_manifest"], LEAGUE_MANIFEST_PATH, LEAGUE_MANIFEST_SHA256, "training league manifest")
    require_bound_file(bindings["source_command_preregistration"], SOURCE_COMMAND_PROTOCOL, SOURCE_COMMAND_PROTOCOL_SHA256, "source command protocol")
    require_bound_file(bindings["u464_training_integrity_decision"], U464_TRAINING_DECISION_PATH, U464_TRAINING_DECISION_SHA256, "U464 training decision")
    require_bound_file(bindings["u464_gold19_decision"], U464_GOLD_DECISION_PATH, U464_GOLD_DECISION_SHA256, "U464 Gold decision")

    command = protocol["command"]
    expected_command = derive_exact_command()
    if command != expected_command:
        raise ValueError("protocol command differs from the exact derived command")
    if protocol["command_sha256"] != DERIVED_COMMAND_SHA256:
        raise ValueError("protocol command SHA-256 field mismatch")
    if Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("command Python differs from launcher environment")
    if Path(command[1]).resolve() != TRAINER_PATH:
        raise ValueError("command trainer differs from binding")
    if flag_value(command, "--resume") != str(PARENT_PATH):
        raise ValueError("command resume checkpoint mismatch")
    if flag_value(command, "--seed") != "202607336":
        raise ValueError("command quota identity seed mismatch")
    if flag_value(command, "--updates") != "468":
        raise ValueError("command terminal update mismatch")
    if flag_value(command, "--checkpoint-interval") != "464":
        raise ValueError("checkpoint interval drifted")
    if flag_value(command, "--eval-interval") != "464":
        raise ValueError("evaluation interval drifted")
    for forbidden in ("--reset-optimizer-on-resume", "--reset-opponent-quota-on-resume"):
        if forbidden in command:
            raise ValueError(f"forbidden reset flag present: {forbidden}")
    if protocol["expected_parent_state"] != EXPECTED_PARENT_STATE:
        raise ValueError("expected parent state schema mismatch")

    output = Path(protocol["output_dir"])
    terminal = Path(protocol["terminal_checkpoint"])
    marker = Path(protocol["attempt_start_marker"])
    log = Path(protocol["log"])
    if not output.is_absolute():
        output = REPO_ROOT / output
    if not terminal.is_absolute():
        terminal = REPO_ROOT / terminal
    if not marker.is_absolute():
        marker = REPO_ROOT / marker
    if not log.is_absolute():
        log = REPO_ROOT / log
    if output.resolve() != OUTPUT_DIR or terminal.resolve() != TERMINAL_CHECKPOINT:
        raise ValueError("execution output binding mismatch")
    if marker.resolve() != ATTEMPT_MARKER or log.resolve() != LOG_PATH:
        raise ValueError("execution marker/log binding mismatch")
    if Path(flag_value(command, "--output-dir")).resolve() != OUTPUT_DIR:
        raise ValueError("command output directory mismatch")
    for path, label in (
        (OUTPUT_ROOT, "output root"),
        (OUTPUT_DIR, "output directory"),
        (TERMINAL_CHECKPOINT, "terminal checkpoint"),
        (ATTEMPT_MARKER, "attempt marker"),
        (LOG_PATH, "training log"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing {label}: {path}")
    if protocol["stop_rules"] != {
        "attempts_authorized": 1,
        "no_retry_or_seed_selection": True,
        "source_update": 464,
        "terminal_update": 468,
        "published_endpoints": [468],
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

    input_audit = audit_training_inputs(command)
    parent_audit = audit_parent_state()
    audit = {
        "status": "passed",
        "protocol_sha256": expected_protocol_sha256,
        "command_sha256": DERIVED_COMMAND_SHA256,
        "command_tokens": len(command),
        "parent": parent_audit,
        "training_inputs": input_audit,
        "targets_absent": True,
        "writes_performed": False,
        "rollouts_performed": False,
    }
    return protocol, command, audit


def claim_and_run(
    protocol: dict[str, Any],
    command: Sequence[str],
    protocol_sha256: str,
    protocol_path: Path,
) -> int:
    marker_payload = {
        "event": "u464_to_u468_preregistered_attempt_consumed",
        "schema_version": SCHEMA_VERSION,
        "seed": protocol["seed"],
        "protocol_sha256": protocol_sha256,
        "command_sha256": protocol["command_sha256"],
        "launcher_sha256": sha256_file(Path(__file__)),
    }
    marker_fd = os.open(ATTEMPT_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(marker_fd, canonical_json_bytes(marker_payload))
        os.fsync(marker_fd)
    finally:
        os.close(marker_fd)

    log_fd = -1
    payload_fd = -1
    bootstrap_fd = -1
    snapshots: list[dict[str, Any]] = []
    try:
        log_fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.mkdir(OUTPUT_ROOT, mode=0o700)
        claim_path = OUTPUT_ROOT / ".u464_to_u468_launch_claim.json"
        claim_fd = os.open(
            claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            os.write(claim_fd, canonical_json_bytes(marker_payload))
            os.fsync(claim_fd)
        finally:
            os.close(claim_fd)

        locked_paths = command_input_paths(command)
        locked_paths.append(require_regular(protocol_path, "execution protocol"))
        snapshots = acquire_input_locks(sorted(set(locked_paths), key=str))
        assert_input_locks_unchanged(snapshots)
        assert_locked_input_bindings(
            snapshots,
            protocol,
            command,
            protocol_path,
            protocol_sha256,
        )

        trainer_snapshot = snapshot_by_path(snapshots, TRAINER_PATH)
        trainer_payload = read_fd_bytes(int(trainer_snapshot["fd"]))
        if hashlib.sha256(trainer_payload).hexdigest() != TRAINER_SHA256:
            raise RuntimeError("held trainer payload differs from frozen binding")
        payload_fd, payload_sha256 = create_sealed_memfd(
            "ptcg-u464-u468-trainer-payload", trainer_payload
        )
        bootstrap_payload = build_trainer_bootstrap(payload_fd)
        bootstrap_fd, bootstrap_sha256 = create_sealed_memfd(
            "ptcg-u464-u468-trainer-bootstrap", bootstrap_payload
        )
        if payload_fd == bootstrap_fd:
            raise RuntimeError("sealed trainer descriptors unexpectedly collide")
        assert_sealed_memfd(payload_fd, payload_sha256, "trainer payload")
        assert_sealed_memfd(bootstrap_fd, bootstrap_sha256, "trainer bootstrap")

        effective_command = list(command)
        effective_command[1] = f"/proc/self/fd/{bootstrap_fd}"
        effective_command_sha256 = sha256_bytes(canonical_json_bytes(effective_command))

        handle = os.fdopen(log_fd, "wb", buffering=0)
        log_fd = -1
        with handle:
            handle.write(
                canonical_json_bytes(
                    {
                        "event": "u464_to_u468_pre_exec_checks_passed",
                        **{
                            key: marker_payload[key]
                            for key in marker_payload
                            if key != "event"
                        },
                        "cwd": str(REPO_ROOT),
                        "input_shared_locks": len(snapshots),
                        "output_root_atomically_claimed": True,
                        "source_trainer_sha256": TRAINER_SHA256,
                        "sealed_trainer_payload_fd": payload_fd,
                        "sealed_trainer_payload_sha256": payload_sha256,
                        "sealed_trainer_payload_seals": int(
                            fcntl.fcntl(payload_fd, F_GET_SEALS)
                        ),
                        "sealed_trainer_bootstrap_fd": bootstrap_fd,
                        "sealed_trainer_bootstrap_sha256": bootstrap_sha256,
                        "sealed_trainer_bootstrap_seals": int(
                            fcntl.fcntl(bootstrap_fd, F_GET_SEALS)
                        ),
                        "effective_command_sha256": effective_command_sha256,
                    }
                )
            )
            os.fsync(handle.fileno())
            assert_input_locks_unchanged(snapshots)
            assert_locked_input_bindings(
                snapshots,
                protocol,
                command,
                protocol_path,
                protocol_sha256,
            )
            assert_sealed_memfd(payload_fd, payload_sha256, "trainer payload")
            assert_sealed_memfd(
                bootstrap_fd, bootstrap_sha256, "trainer bootstrap"
            )
            completed = subprocess.run(
                effective_command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                pass_fds=(bootstrap_fd, payload_fd),
            )
            assert_input_locks_unchanged(snapshots)
            assert_locked_input_bindings(
                snapshots,
                protocol,
                command,
                protocol_path,
                protocol_sha256,
            )
            assert_sealed_memfd(payload_fd, payload_sha256, "trainer payload")
            assert_sealed_memfd(
                bootstrap_fd, bootstrap_sha256, "trainer bootstrap"
            )
            handle.write(
                canonical_json_bytes(
                    {
                        "event": "u464_to_u468_child_terminal",
                        "seed": protocol["seed"],
                        "return_code": completed.returncode,
                        "inputs_unchanged_through_child_terminal": True,
                    }
                )
            )
            os.fsync(handle.fileno())
        return completed.returncode
    finally:
        if bootstrap_fd >= 0:
            os.close(bootstrap_fd)
        if payload_fd >= 0:
            os.close(payload_fd)
        if snapshots:
            release_input_locks(snapshots)
        if log_fd >= 0:
            os.close(log_fd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol, command, audit = load_and_validate_protocol(
        args.protocol,
        args.expected_protocol_sha256,
    )
    if args.audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return claim_and_run(
        protocol,
        command,
        args.expected_protocol_sha256,
        args.protocol,
    )


if __name__ == "__main__":
    raise SystemExit(main())
