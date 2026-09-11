#!/usr/bin/env python3
"""Launch the versioned Marnie failure-tail PPO gold-push experiment.

The launcher is deliberately dry-run by default.  A real run requires both
``--execute`` and the manifest SHA-256 printed by a preceding dry run.  All
training inputs, deck identities, checkpoint schemas, replay provenance, and
requested ``train_ppo.py`` CLI capabilities are checked before any output is
created.  The smoke and full phases use separate, never-reused output paths.

This launcher trains locally only.  It has no packaging, upload, or submission
path.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
PYTHON_SHA256 = "35010543d1379740c163ebf34e92108891c70cceb393367d71f463733c8be497"
TRAIN_PPO = ROOT / "tools/train_ppo.py"
TRAIN_PPO_SHA256 = "321a4e25fb3ccecb0dfb3c37803a20369b8083619f6b238c3d067990b7e397cf"

OUTPUT_ROOT = ROOT / "artifacts/gold_push_20260810_v1/ppo_marnie_tail32_v1"
SMOKE_OUTPUT = (
    ROOT / "artifacts/gold_push_20260810_v1/ppo_marnie_tail32_v1_smoke"
)

MARNIE_DECK_HASH = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
LUCARIO_DECK_HASH = "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"
FROSLASS_DECK_HASH = "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc"

MARNIE_DECK = ROOT / f"data/gold_push_recent7_20260810_v1/decks/{MARNIE_DECK_HASH}.csv"
LUCARIO_DECK = ROOT / f"data/gold_push_recent7_20260810_v1/decks/{LUCARIO_DECK_HASH}.csv"
FROSLASS_DECK = ROOT / f"data/gold_push_recent7_20260810_v1/decks/{FROSLASS_DECK_HASH}.csv"

LEARNER = ROOT / "artifacts/gold_push_20260810_v1/bc_soups/marnie_source50_seedmean50.pt"
REPLAY = ROOT / "data/gold_push_recent7_20260810_v1/archives/marnie_trainwins.zip"
LUCARIO_BC = (
    ROOT
    / "artifacts/gold_push_20260810_v1/bc/mega_lucario_trainwins_seed1021/best.pt"
)
FROSLASS_BC = (
    ROOT
    / "artifacts/gold_push_20260810_v1/bc/mega_froslass_lopunny_trainwins_seed1031/best.pt"
)
SOURCE_MARNIE_BC = (
    ROOT / "artifacts/gold8_recent7_20260808/marnie/specialist_bc/best.pt"
)
U472 = (
    ROOT
    / "artifacts/ppo_guardedu468_updatedreplay_g8_ppo4x96_u472_"
    "freshjointactor6_s8_design202608148/ppo_stage/B_gold_league/"
    "seed-202608148/checkpoints/update-0472.pt"
)

FILE_SHA256 = {
    "learner_bc": "8ee633d3df1bfe7bce536da4ad7dde71844a2ac293d6e0339731602db44a7036",
    "replay": "7cc2a4cb38b857ccdacf9cc84fe4610400ca3295c9ab1dbefbcb75e4341e8eab",
    "lucario_bc": "aaf703638b8a8ab0588a7e866f26bed2eb120118ad7030d266ec7f68049af31d",
    "froslass_bc": "ec12a1da9b06915a9aad6ee77cc5c23ad7dc2b6ca0e2601a238ac2d9a610dadc",
    "source_marnie_bc": "dddf3e6ed354efc3ff2e38d058858d50a212b3afdcc0a5b1e2e8d70c32ac83d4",
    "u472": "c0436d54fff5e4da62d94c3002775c9e4b9ee848c5c59dd4422943cd71ad7018",
    "marnie_deck": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
    "lucario_deck": "5ddb7ca2790518e3c1eac6e2ff8b7fdb6ff0a817bf888536349a090ec7582a9f",
    "froslass_deck": "3b4ffbc0735d73a4ab5c2f2c722c45636b60af0e29dda5cc4b5a2d5d5fb5d7c7",
}

FILE_PATHS = {
    "learner_bc": LEARNER,
    "replay": REPLAY,
    "lucario_bc": LUCARIO_BC,
    "froslass_bc": FROSLASS_BC,
    "source_marnie_bc": SOURCE_MARNIE_BC,
    "u472": U472,
    "marnie_deck": MARNIE_DECK,
    "lucario_deck": LUCARIO_DECK,
    "froslass_deck": FROSLASS_DECK,
}

BC_FEATURE = "ptcg-bc-orbit-entity-transformer-v5"
PPO_FEATURE = "ptcg-selfplay-ppo-terminal01-v1"
EXPECTED_MODEL_CONFIG = {
    "hash_size": 65536,
    "categorical_dim": 64,
    "model_dim": 128,
    "layers": 4,
    "heads": 4,
    "dropout": 0.05,
    "max_state_entities": 80,
    "entity_fields": 20,
    "option_fields": 24,
}


@dataclass(frozen=True)
class Phase:
    name: str
    updates: int
    games_per_update: int
    own_bc_quota: int
    permanent_opponent_quota: int
    eval_games: int
    eval_interval: int
    seed: int


PHASES = {
    "smoke": Phase(
        name="smoke",
        updates=1,
        games_per_update=96,
        own_bc_quota=48,
        permanent_opponent_quota=12,
        eval_games=32,
        eval_interval=1,
        seed=202608101,
    ),
    "full": Phase(
        name="full",
        updates=4,
        games_per_update=192,
        own_bc_quota=96,
        permanent_opponent_quota=24,
        eval_games=256,
        eval_interval=2,
        seed=202608102,
    ),
}


@dataclass(frozen=True)
class Opponent:
    label: str
    checkpoint: Path
    checkpoint_key: str
    deck: Path
    deck_hash: str


OPPONENTS = (
    Opponent("current_lucario_bc", LUCARIO_BC, "lucario_bc", LUCARIO_DECK, LUCARIO_DECK_HASH),
    Opponent("current_froslass_bc", FROSLASS_BC, "froslass_bc", FROSLASS_DECK, FROSLASS_DECK_HASH),
    Opponent("source_marnie_bc", SOURCE_MARNIE_BC, "source_marnie_bc", MARNIE_DECK, MARNIE_DECK_HASH),
    Opponent("submitted_u472", U472, "u472", MARNIE_DECK, MARNIE_DECK_HASH),
)


@dataclass
class InputLock:
    label: str
    path: Path
    expected_sha256: str
    handle: BinaryIO
    identity: tuple[int, int, int]


@dataclass(frozen=True)
class Preflight:
    phase: Phase
    target: Path
    command: list[str]
    manifest: dict[str, Any]
    manifest_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def output_dir(phase: Phase) -> Path:
    return SMOKE_OUTPUT if phase.name == "smoke" else OUTPUT_ROOT


def opponent_name(checkpoint: Path, deck: Path) -> str:
    return f"{checkpoint.stem}@{deck.stem}"


def quotas(phase: Phase) -> dict[str, int]:
    values = {"bc": phase.own_bc_quota}
    for opponent in OPPONENTS:
        name = opponent_name(opponent.checkpoint, opponent.deck)
        if name in values:
            raise RuntimeError(f"Duplicate permanent opponent name: {name}")
        values[name] = phase.permanent_opponent_quota
    if sum(values.values()) != phase.games_per_update:
        raise RuntimeError("Exact opponent quotas do not sum to games-per-update")
    if any(value <= 0 or value % 2 for value in values.values()):
        raise RuntimeError("Every exact quota must be positive and seat-balanced")
    return values


def build_command(phase: Phase) -> list[str]:
    command = [
        str(PYTHON),
        "-I",
        "-B",
        str(TRAIN_PPO),
        "--bc-checkpoint",
        str(LEARNER),
        "--kl-reference-checkpoint",
        str(LEARNER),
        "--deck",
        str(MARNIE_DECK),
    ]
    for opponent in OPPONENTS:
        command.extend(
            ["--extra-opponent", str(opponent.checkpoint), str(opponent.deck)]
        )
    command.extend(
        [
            "--output-dir",
            str(output_dir(phase)),
            "--updates",
            str(phase.updates),
            "--schedule-start-update",
            "1",
            "--environments",
            "16",
            "--games-per-update",
            str(phase.games_per_update),
            "--ppo-epochs",
            "2",
            "--minibatch-size",
            "512",
            "--learning-rate",
            "0.000012",
            "--value-learning-rate",
            "0.000025",
            "--weight-decay",
            "0.0001",
            "--learning-rate-schedule",
            "constant",
            "--gamma",
            "1.0",
            "--gae-lambda",
            "1.0",
            "--advantage-normalization",
            "per_opponent",
            "--clip-ratio",
            "0.12",
            "--value-coefficient",
            "0.25",
            "--value-trunk-gradient-scale",
            "0.02",
            "--entropy-coefficient",
            "0.0005",
            "--max-grad-norm",
            "0.5",
            "--policy-temperature",
            "0.8",
            "--trainable-scope",
            "last_block_heads",
            "--bc-kl-start",
            "0.020",
            "--bc-kl-end",
            "0.016",
            "--target-kl",
            "0.004",
            "--league-probability",
            "1.0",
            "--opponent-sampling",
            "per_game",
            "--opponent-quota-mode",
            "fixed",
            "--opponent-quota-seat-balance",
            "--ppo-objective",
            "standard",
            "--actor-reduction",
            "episode_mean",
            "--actor-value-gradient-mode",
            "scalar",
            "--constrained-gradient-mode",
            "scalar",
            "--snapshot-interval",
            "1000000",
            "--max-pool-size",
            "8",
            "--bc-replay-data",
            str(REPLAY),
            "--bc-replay-split",
            "train",
            "--bc-replay-batches",
            "72",
            "--bc-replay-batch-size",
            "256",
            "--bc-replay-workers",
            "8",
            "--bc-replay-steps",
            "2",
            "--bc-replay-lr-scale",
            "0.05",
            "--bc-replay-loss",
            "ordered",
            "--bc-replay-order-context-weight",
            "8.0",
            "--bc-replay-non-context34-fixed-multi-action-order-weight",
            "1.0",
            "--bc-replay-context34-rows-per-batch",
            "4",
            "--eval-interval",
            str(phase.eval_interval),
            "--eval-games",
            str(phase.eval_games),
            "--eval-all-permanent-opponents",
            "--selection-aggregation",
            "min",
            "--checkpoint-interval",
            "1",
            "--max-game-decisions",
            "1000",
            "--failed-attempt-as-loss",
            "--failed-loss-tail-transitions",
            "32",
            "--seed",
            str(phase.seed),
            "--device",
            "cuda",
        ]
    )
    for name, quota in quotas(phase).items():
        command.extend(["--opponent-base-quota", name, str(quota)])
    return command


def registered_cli_flags(path: Path) -> set[str]:
    """Extract literal argparse flags without importing or executing trainer."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    flags.add(argument.value)
    return flags


def assert_cli_contract(command: Sequence[str]) -> dict[str, Any]:
    command_flags = {value for value in command if value.startswith("--")}
    registered = registered_cli_flags(TRAIN_PPO)
    missing = sorted(command_flags - registered)
    if missing:
        raise RuntimeError(
            "train_ppo.py cannot express requested flags: " + ", ".join(missing)
        )
    if "--truncation-as-loss" in command:
        raise RuntimeError("Legacy --truncation-as-loss is forbidden")
    required = {
        "--failed-attempt-as-loss",
        "--failed-loss-tail-transitions",
        "--eval-all-permanent-opponents",
        "--selection-aggregation",
        "--opponent-quota-seat-balance",
    }
    absent = sorted(required - command_flags)
    if absent:
        raise RuntimeError("Required PPO contract flags absent: " + ", ".join(absent))
    repeated_allowed = {"--extra-opponent", "--opponent-base-quota"}
    repeated = sorted(
        flag
        for flag in command_flags - repeated_allowed
        if list(command).count(flag) != 1
    )
    if repeated:
        raise RuntimeError("Singleton PPO flags repeated: " + ", ".join(repeated))
    equals_form = [value for value in command if value.startswith("--") and "=" in value]
    if equals_form:
        raise RuntimeError("--flag=value syntax is forbidden in locked command")
    return {
        "registered_flag_count": len(registered),
        "command_flag_count": len(command_flags),
        "all_requested_flags_registered": True,
        "legacy_truncation_flag_absent": True,
        "required_failure_and_selection_flags_present": True,
    }


def assert_regular_file(path: Path, *, allow_symlink: bool = False) -> os.stat_result:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing locked input: {path}") from error
    if stat.S_ISLNK(info.st_mode):
        if not allow_symlink:
            raise RuntimeError(f"Locked input must not be a symlink: {path}")
        resolved = path.resolve(strict=True)
        resolved_info = resolved.stat()
        if not stat.S_ISREG(resolved_info.st_mode):
            raise RuntimeError(f"Resolved input is not a regular file: {resolved}")
        return resolved_info
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"Locked input is not a regular file: {path}")
    return info


def validate_file_binding(
    label: str,
    path: Path,
    expected_sha256: str,
    *,
    allow_symlink: bool = False,
) -> dict[str, Any]:
    info = assert_regular_file(path, allow_symlink=allow_symlink)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {label}: expected {expected_sha256}, observed {observed}"
        )
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "bytes": info.st_size,
        "sha256": observed,
        "expected_sha256": expected_sha256,
        "sha256_exact": True,
    }


def read_deck(path: Path) -> list[int]:
    cards = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cards) != 60:
        raise ValueError(f"{path} contains {len(cards)} cards; expected 60")
    return cards


def compute_deck_hash(cards: Sequence[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_deck(path: Path, expected_hash: str) -> dict[str, Any]:
    cards = read_deck(path)
    observed = compute_deck_hash(cards)
    if observed != expected_hash or path.stem != expected_hash:
        raise RuntimeError(
            f"Deck identity mismatch for {path}: expected {expected_hash}, observed {observed}"
        )
    return {
        "path": str(path),
        "cards": len(cards),
        "deck_hash": observed,
        "filename_matches_deck_hash": True,
    }


def checkpoint_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    embedded = checkpoint.get("model_config")
    raw = embedded if isinstance(embedded, dict) else checkpoint.get("config")
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint has no model configuration")
    return {
        "hash_size": int(raw["hash_size"]),
        "categorical_dim": int(raw["categorical_dim"]),
        "model_dim": int(raw["model_dim"]),
        "layers": int(raw["layers"]),
        "heads": int(raw["heads"]),
        "dropout": float(raw["dropout"]),
        "max_state_entities": int(raw["max_state_entities"]),
        "entity_fields": int(raw["entity_fields"]),
        "option_fields": int(raw["option_fields"]),
    }


def model_state_schema_sha256(state: dict[str, Any]) -> str:
    schema: list[dict[str, Any]] = []
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Model state entry is not a tensor: {name}")
        schema.append(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        )
    return sha256_json(schema)


def validate_checkpoint(
    label: str,
    path: Path,
    expected_feature: str,
    expected_deck_hash: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path}: checkpoint payload is not a mapping")
    feature = checkpoint.get("feature_version")
    if feature != expected_feature:
        raise ValueError(
            f"{path}: expected feature {expected_feature!r}, observed {feature!r}"
        )
    config = checkpoint_model_config(checkpoint)
    if config != EXPECTED_MODEL_CONFIG:
        raise ValueError(f"{path}: incompatible model configuration: {config}")
    raw_state = checkpoint.get("model_state_dict")
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError(f"{path}: missing model_state_dict")

    if expected_feature == BC_FEATURE:
        raw_config = checkpoint.get("config") or {}
        declared = set(raw_config.get("deck_hashes") or [])
        if declared != {expected_deck_hash}:
            raise ValueError(
                f"{path}: BC deck hashes {sorted(declared)} do not match {expected_deck_hash}"
            )
        update = None
    else:
        declared_hash = checkpoint.get("learner_deck_hash")
        if declared_hash != expected_deck_hash:
            raise ValueError(
                f"{path}: PPO deck hash {declared_hash!r} does not match {expected_deck_hash}"
            )
        update = int(checkpoint.get("update", -1))
        if label == "submitted_u472" and update != 472:
            raise ValueError(f"{path}: expected PPO update 472, observed {update}")

    return {
        "path": str(path),
        "feature_version": feature,
        "deck_hash": expected_deck_hash,
        "model_config": config,
        "model_state_tensors": len(raw_state),
        "model_state_schema_sha256": model_state_schema_sha256(raw_state),
        "update": update,
    }


def validate_learner_lineage(checkpoint: dict[str, Any]) -> dict[str, Any]:
    interpolation = checkpoint.get("interpolation")
    if not isinstance(interpolation, dict):
        raise ValueError("Learner soup lacks interpolation lineage")
    source_path = Path(str(interpolation.get("checkpoint_a", ""))).resolve()
    expected_source = SOURCE_MARNIE_BC.resolve()
    checks = {
        "formula": interpolation.get("formula") == "(1-alpha)*A + alpha*B",
        "alpha": float(interpolation.get("alpha", -1.0)) == 0.5,
        "source_path": source_path == expected_source,
        "source_sha256": interpolation.get("checkpoint_a_sha256")
        == FILE_SHA256["source_marnie_bc"],
        "optimizer_state_omitted": interpolation.get("optimizer_state_omitted") is True,
        "resume_training_false": interpolation.get("resume_training") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"Learner soup lineage mismatch: {checks}")
    return {
        "formula": interpolation["formula"],
        "alpha": interpolation["alpha"],
        "source_checkpoint": str(expected_source),
        "source_checkpoint_sha256": FILE_SHA256["source_marnie_bc"],
        "checks": checks,
    }


def validate_replay(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Replay ZIP CRC failure: {bad_member}")
        names = archive.namelist()
        if names.count("manifest.json") != 1:
            raise ValueError("Replay ZIP must contain exactly one manifest.json")
        manifest = json.loads(archive.read("manifest.json"))
    split_counts = {
        split: sum(name.startswith(f"{split}/") and name.endswith(".jsonl") for name in names)
        for split in ("train", "valid", "test")
    }
    reward_filter = manifest.get("terminal_reward_filter") or {}
    checks = {
        "schema": manifest.get("schema_version") == "ptcg-bc-visible-decisions-v1",
        "profile": (manifest.get("profile") or {}).get("slug") == "marnie",
        "deck_hash": (manifest.get("profile") or {}).get("deck_hash")
        == MARNIE_DECK_HASH,
        "train_rows": (manifest.get("split_decisions") or {}).get("train") == 187611,
        "context34_rows": (manifest.get("context_decisions") or {}).get("34") == 1132,
        "wins_only_train": reward_filter.get("mode") == "wins"
        and reward_filter.get("splits") == "train",
        "split_shards": split_counts == {"train": 8, "valid": 2, "test": 2},
        "crc": bad_member is None,
    }
    if not all(checks.values()):
        raise ValueError(f"Replay provenance mismatch: {checks}")
    return {
        "path": str(path),
        "schema_version": manifest["schema_version"],
        "deck_hash": MARNIE_DECK_HASH,
        "split_decisions": manifest["split_decisions"],
        "context34_rows": manifest["context_decisions"]["34"],
        "split_shards": split_counts,
        "terminal_reward_filter": reward_filter,
        "checks": checks,
    }


def assert_target_absent(target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to reuse output path: {target}")


def collect_file_records() -> dict[str, dict[str, Any]]:
    records = {
        "python": validate_file_binding(
            "my_project_env_python", PYTHON, PYTHON_SHA256, allow_symlink=True
        ),
        "trainer": validate_file_binding("train_ppo", TRAIN_PPO, TRAIN_PPO_SHA256),
    }
    for label, path in FILE_PATHS.items():
        records[label] = validate_file_binding(label, path, FILE_SHA256[label])
    records["launcher"] = {
        "path": str(SELF),
        "resolved_path": str(SELF.resolve()),
        "bytes": SELF.stat().st_size,
        "sha256": sha256_file(SELF),
        "protocol_pins_live_launcher_sha256": True,
    }
    return records


def command_value(command: Sequence[str], flag: str) -> str:
    try:
        index = list(command).index(flag)
    except ValueError as error:
        raise RuntimeError(f"Locked command is missing {flag}") from error
    if index + 1 >= len(command):
        raise RuntimeError(f"Locked command has no value for {flag}")
    return command[index + 1]


def build_preflight(phase: Phase) -> Preflight:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Launcher must run from repository root: {ROOT}")
    target = output_dir(phase)
    assert_target_absent(target)

    file_records = collect_file_records()
    deck_records = {
        "marnie": validate_deck(MARNIE_DECK, MARNIE_DECK_HASH),
        "lucario": validate_deck(LUCARIO_DECK, LUCARIO_DECK_HASH),
        "froslass_lopunny": validate_deck(FROSLASS_DECK, FROSLASS_DECK_HASH),
    }
    checkpoint_records = {
        "learner_bc": validate_checkpoint(
            "learner_bc", LEARNER, BC_FEATURE, MARNIE_DECK_HASH
        ),
        "current_lucario_bc": validate_checkpoint(
            "current_lucario_bc", LUCARIO_BC, BC_FEATURE, LUCARIO_DECK_HASH
        ),
        "current_froslass_bc": validate_checkpoint(
            "current_froslass_bc", FROSLASS_BC, BC_FEATURE, FROSLASS_DECK_HASH
        ),
        "source_marnie_bc": validate_checkpoint(
            "source_marnie_bc", SOURCE_MARNIE_BC, BC_FEATURE, MARNIE_DECK_HASH
        ),
        "submitted_u472": validate_checkpoint(
            "submitted_u472", U472, PPO_FEATURE, MARNIE_DECK_HASH
        ),
    }
    learner_payload = torch.load(LEARNER, map_location="cpu", weights_only=False)
    learner_lineage = validate_learner_lineage(learner_payload)
    replay_record = validate_replay(REPLAY)
    command = build_command(phase)
    cli_contract = assert_cli_contract(command)
    phase_quotas = quotas(phase)

    if int(command_value(command, "--updates")) != phase.updates:
        raise RuntimeError("Command update count drifted")
    if int(command_value(command, "--games-per-update")) != phase.games_per_update:
        raise RuntimeError("Command game count drifted")
    if command_value(command, "--failed-loss-tail-transitions") != "32":
        raise RuntimeError("Failure-tail length drifted")

    manifest: dict[str, Any] = {
        "schema_version": "ptcg-gold-push-marnie-tail32-launch-v1",
        "status": "locked_before_training",
        "phase": phase.name,
        "output_dir": str(target),
        "expected_terminal_checkpoint": str(
            target / f"checkpoints/update-{phase.updates:04d}.pt"
        ),
        "environment": {
            "repository_root": str(ROOT),
            "cwd_exact": True,
            "python": str(PYTHON),
            "python_sha256": PYTHON_SHA256,
            "python_flags": ["-I", "-B"],
            "device": "cuda",
            "subprocess_environment_overrides": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        },
        "inputs": file_records,
        "decks": deck_records,
        "checkpoints": checkpoint_records,
        "learner_lineage": learner_lineage,
        "replay": replay_record,
        "rollout": {
            "updates": phase.updates,
            "valid_games_per_update": phase.games_per_update,
            "total_valid_games": phase.updates * phase.games_per_update,
            "environments": 16,
            "fixed_per_update_quotas": phase_quotas,
            "quota_sum": sum(phase_quotas.values()),
            "seat_balance": "exact_half_per_opponent",
            "failure_attempt_replacement_preserves_valid_quota": True,
        },
        "objective": {
            "reward": {"win": 1.0, "loss": 0.0, "draw": 0.0},
            "ppo_epochs": 2,
            "minibatch_size": 512,
            "actor_learning_rate": 1.2e-5,
            "value_learning_rate": 2.5e-5,
            "weight_decay": 1e-4,
            "learning_rate_schedule": "constant",
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "advantage_normalization": "per_opponent",
            "clip_ratio": 0.12,
            "value_coefficient": 0.25,
            "value_trunk_gradient_scale": 0.02,
            "entropy_coefficient": 0.0005,
            "max_grad_norm": 0.5,
            "policy_temperature": 0.8,
            "trainable_scope": "last_block_heads",
            "actor_reduction": "episode_mean",
            "actor_value_gradient_mode": "scalar",
            "bc_kl": {"start": 0.020, "end": 0.016},
            "target_kl": 0.004,
        },
        "bc_replay": {
            "split": "train",
            "batches": 72,
            "batch_size": 256,
            "workers": 8,
            "steps_per_update": 2,
            "learning_rate_scale": 0.05,
            "loss": "ordered",
            "context34_weight": 8.0,
            "non_context34_fixed_multi_action_order_weight": 1.0,
            "context34_rows_per_batch": 4,
        },
        "failure_tail": {
            "enabled": True,
            "failed_attempt_as_loss": True,
            "tail_transitions": 32,
            "legacy_truncation_as_loss": False,
        },
        "selection": {
            "eval_all_permanent_opponents": True,
            "aggregation": "min",
            "eval_games_per_opponent": phase.eval_games,
            "eval_interval": phase.eval_interval,
            "checkpoint_interval": 1,
        },
        "gates": {
            "all_input_sha256_exact": True,
            "all_three_deck_hashes_exact": True,
            "all_checkpoint_model_configs_compatible": True,
            "checkpoint_deck_bindings_exact": True,
            "learner_source50_lineage_exact": True,
            "replay_crc_and_manifest_exact": True,
            "trainer_cli_contract": cli_contract,
            "fixed_quota_sum_exact": sum(phase_quotas.values())
            == phase.games_per_update,
            "every_quota_even_for_dual_seat": all(
                value % 2 == 0 for value in phase_quotas.values()
            ),
            "output_absent_before_launch": True,
        },
        "seed": phase.seed,
        "command": command,
        "command_sha256": sha256_json(command),
        "scope": {
            "training": True,
            "local_only": True,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }
    return Preflight(
        phase=phase,
        target=target,
        command=command,
        manifest=manifest,
        manifest_sha256=sha256_json(manifest),
    )


def lock_bindings(preflight: Preflight) -> dict[str, tuple[Path, str]]:
    bindings = {
        "launcher": (SELF, preflight.manifest["inputs"]["launcher"]["sha256"]),
        "trainer": (TRAIN_PPO, TRAIN_PPO_SHA256),
        "python": (PYTHON.resolve(), PYTHON_SHA256),
    }
    for label, path in FILE_PATHS.items():
        bindings[label] = (path, FILE_SHA256[label])
    return bindings


def acquire_input_locks(preflight: Preflight) -> list[InputLock]:
    locks: list[InputLock] = []
    try:
        for label, (path, expected_sha256) in lock_bindings(preflight).items():
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            handle = os.fdopen(fd, "rb", closefd=True)
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"Locked input is not regular: {path}")
            observed = sha256_handle(handle)
            if observed != expected_sha256:
                raise RuntimeError(
                    f"Execution-time input mismatch for {label}: {observed}"
                )
            locks.append(
                InputLock(
                    label=label,
                    path=path,
                    expected_sha256=expected_sha256,
                    handle=handle,
                    identity=(info.st_dev, info.st_ino, info.st_size),
                )
            )
    except BaseException:
        release_input_locks(locks)
        raise
    return locks


def assert_input_locks_unchanged(locks: Sequence[InputLock]) -> None:
    for locked in locks:
        info = os.fstat(locked.handle.fileno())
        identity = (info.st_dev, info.st_ino, info.st_size)
        observed = sha256_handle(locked.handle)
        if identity != locked.identity or observed != locked.expected_sha256:
            raise RuntimeError(f"Locked input changed during launch: {locked.path}")


def release_input_locks(locks: Sequence[InputLock]) -> None:
    for locked in reversed(locks):
        try:
            fcntl.flock(locked.handle.fileno(), fcntl.LOCK_UN)
        finally:
            locked.handle.close()


def write_exclusive(path: Path, payload: Any) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        data = canonical_json_bytes(payload)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def execute(preflight: Preflight) -> int:
    locks = acquire_input_locks(preflight)
    try:
        assert_input_locks_unchanged(locks)
        assert_target_absent(preflight.target)
        preflight.target.mkdir(parents=False, exist_ok=False, mode=0o700)
        write_exclusive(
            preflight.target / "launcher_manifest.json",
            {
                "manifest_sha256": preflight.manifest_sha256,
                "manifest": preflight.manifest,
            },
        )
        assert_input_locks_unchanged(locks)
        environment = os.environ.copy()
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            preflight.command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            shell=False,
            check=False,
        )
        assert_input_locks_unchanged(locks)
        write_exclusive(
            preflight.target / "launcher_result.json",
            {
                "schema_version": "ptcg-gold-push-marnie-tail32-result-v1",
                "manifest_sha256": preflight.manifest_sha256,
                "return_code": int(completed.returncode),
                "inputs_unchanged_after_child": True,
            },
        )
        return int(completed.returncode)
    finally:
        release_input_locks(locks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASES), default="smoke")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--expected-manifest-sha256",
        help="Required for --execute; copy the digest printed by a dry run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute and not args.expected_manifest_sha256:
        raise ValueError("--execute requires --expected-manifest-sha256")
    if not args.execute and args.expected_manifest_sha256:
        raise ValueError("--expected-manifest-sha256 is only valid with --execute")

    preflight = build_preflight(PHASES[args.phase])
    report = {
        "manifest_sha256": preflight.manifest_sha256,
        "manifest": preflight.manifest,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(shlex.join(preflight.command), flush=True)
    if not args.execute:
        return 0
    if args.expected_manifest_sha256 != preflight.manifest_sha256:
        raise RuntimeError(
            "Manifest SHA-256 mismatch: dry-run again and review the locked protocol"
        )
    return execute(preflight)


if __name__ == "__main__":
    raise SystemExit(main())
