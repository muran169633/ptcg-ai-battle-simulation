#!/usr/bin/env python3
"""Train a frozen FLG train-only head10 prefix sweep from U468 beta=1.157.

The beta endpoint is reconstructed in memory from four authenticated inputs.
Training uses a fresh AdamW optimizer and the frozen FLG ``train`` replay
cache.  Validation data is never opened here.  Formal mode publishes all four
evaluation-only endpoints (P1/P2/P4/P8) before returning.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_u468_p12_delta_direction_sweep as transport  # noqa: E402
import run_ppo_bc_repair as repair  # noqa: E402
import train_ppo as ppo  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-beta1157-flg-head10-fresh-prefix-sweep-v1"
BRANCH = "ppo_u468_beta1157_flg_actorhead_freshlr225e7_p1p2p4p8_design202608100"
DESIGN_PATH = ROOT / f"artifacts/{BRANCH}.design_preregistration.json"
OUTPUT_ROOT = ROOT / f"artifacts/{BRANCH}"
ATTEMPT_MARKER = ROOT / (
    ".ptcg-u468-beta1157-flg-actorhead-freshlr225e7-attempt-202608100.json"
)

BETA = 1.157
BASE_MODEL_SHA256 = "8b679c79907cdb1b95ca8b4c8302f0bf16686a36bc7f100634d633562c82ced4"
LEARNING_RATE = 2.25e-7
BASE_ACTOR_LEARNING_RATE = 3.6e-5
LR_SCALE = LEARNING_RATE / BASE_ACTOR_LEARNING_RATE
WEIGHT_DECAY = 1e-4
ADAM_EPS = 1e-5
MAX_GRAD_NORM = 0.5
CACHE_SEED = 202608023
EXECUTION_SEED = 202608024
CACHE_BATCHES = 32
BATCH_SIZE = 256
WORKERS = 8
CONTEXT34_ROWS_PER_BATCH = 1
BATCH_INDICES = (2, 6, 7, 9, 10, 11, 12, 14)
ENDPOINT_STEPS = (1, 2, 4, 8)
EXPECTED_CACHE_SHA256 = (
    "4b16b7a4965e0462af813094bad32168681f46f2fe49068334c34c996f188ceb"
)
EXPECTED_BATCH_SHA256 = (
    "ff7317e8e7850b03c6b6ea740fc2242f5ba47f6a1280a018a187ca27b664790b",
    "73bb36caa9c972828b2a78056bda201813a6a7a3939af10cbda9ab43b2867d92",
    "820c16813fe400d20c2e85b51ca4a2910b5cff02969cbd1b067381e0f14a20cf",
    "fbe14bc6f5fa7b0e634b4997469d7984031bca958d67210c8bb2ade43f6b131e",
    "93eb7629057566a5809adfaabb2d9825a2afc0a25958a043814d5e9bd10f2533",
    "e94d48df4789bdee1dea08daab1df17cb8970282e8ea10f9dab9310fe6deb6b5",
    "709193b3087e35e002ac73596fe84e4e49e9d553726a3dad8104d85190bf1592",
    "c02670eec976c454b30dc73dab5b2a1a1e7aaa11aae53e6e77283b024dd6e236",
)
MUTABLE_NAMES = tuple(transport.MUTABLE_NAMES)
REQUIRED_SLIM_KEYS = tuple(transport.REQUIRED_SLIM_KEYS)

U468 = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
CURRENT_EXACT_P12 = ROOT / (
    "artifacts/ppo_u464_g8ppo4_u468_exactp12specialbc_design202608091/"
    "special_stage/special-bc-actorheadonly-pokemonfan-exactp12-0012.pt"
)
HISTORICAL_PARENT = ROOT / (
    "artifacts/ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_episode_mean_"
    "actorlr2x_mb384_localtransport_u456_to_u464_seed202607336/B_gold_league/"
    "seed-202607336/checkpoints/update-0464.pt"
)
HISTORICAL_P12 = ROOT / (
    "artifacts/ppo_u464mb384_actorheadonly_pokemonfan_sweep_p4_p8_p12_p16_"
    "design202608060/sweep_stage/"
    "special-bc-actorheadonly-pokemonfan-prefix-0012.pt"
)
GENERAL_BC = ROOT / (
    "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/"
    "best.pt"
)
FLG_DATA = ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip"

INPUT_HASHES = {
    U468: "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f",
    CURRENT_EXACT_P12: "9572e178a2bd2e120212d81f970a4eeaa73ed56b79459a4d3d3a7f93539eb54f",
    HISTORICAL_PARENT: "fe51f40f37fca329cd6b0c94f7431001bb909b92624df33fcd7cf6f0da976264",
    HISTORICAL_P12: "8aafef92f0898f46cfe45f076f6fec5bfc383d2429f2d6087805cb976968c3fd",
    GENERAL_BC: "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb",
    FLG_DATA: "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    ROOT / "tools/train_ppo.py": "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3",
    ROOT / "tools/run_ppo_bc_repair.py": "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966",
    ROOT / "tools/build_u468_p12_delta_direction_sweep.py": (
        "968ea51e80900e6e91c493d19ae580471a023dfa69627124310404e7eb115c65"
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(path: Path, expected: str, label: str) -> dict[str, Any]:
    current = os.lstat(path)
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise ValueError(f"{label} is not a regular non-symlink file")
    if current.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: expected {expected}, got {observed}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": observed,
        "bytes": current.st_size,
        "mode": oct(current.st_mode & 0o777),
        "inode": current.st_ino,
        "device": current.st_dev,
        "nlink": current.st_nlink,
    }


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def publish_o_excl(path: Path, payload: bytes, mode: int = 0o600) -> dict[str, Any]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"published path is not a single-link regular file: {path}")
        if before.st_size != len(payload):
            raise RuntimeError(f"short publication: {path}")
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        reloaded = b"".join(chunks)
        after = os.fstat(fd)
        if reloaded != payload or (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ) != (after.st_dev, after.st_ino, after.st_size):
            raise RuntimeError(f"descriptor-held publication reload drift: {path}")
        visible = os.lstat(path)
        if stat.S_ISLNK(visible.st_mode) or (
            visible.st_dev,
            visible.st_ino,
            visible.st_size,
        ) != (after.st_dev, after.st_ino, after.st_size):
            raise RuntimeError(f"visible publication identity drift: {path}")
        return {
            "sha256": sha256_bytes(reloaded),
            "bytes": len(reloaded),
            "mode": oct(after.st_mode & 0o777),
            "inode": after.st_ino,
            "device": after.st_dev,
            "nlink": after.st_nlink,
            "descriptor_reload_exact": True,
            "visible_identity_exact": True,
        }
    finally:
        os.close(fd)


def verify_visible(path: Path, expected: Mapping[str, Any]) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(fd)
        visible = os.lstat(path)
        identity = (after.st_dev, after.st_ino, after.st_size)
        if (before.st_dev, before.st_ino, before.st_size) != identity:
            raise RuntimeError(f"file changed during held verification: {path}")
        if stat.S_ISLNK(visible.st_mode) or (
            visible.st_dev,
            visible.st_ino,
            visible.st_size,
        ) != identity:
            raise RuntimeError(f"visible file identity mismatch: {path}")
        if after.st_nlink != 1 or not stat.S_ISREG(after.st_mode):
            raise RuntimeError(f"visible file type/link mismatch: {path}")
        observed = {
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "inode": after.st_ino,
            "device": after.st_dev,
            "nlink": after.st_nlink,
        }
        for key, value in observed.items():
            if expected.get(key) != value:
                raise RuntimeError(f"visible file evidence drift for {path}: {key}")
    finally:
        os.close(fd)


def load_checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not isinstance(
        value.get("model_state_dict"), Mapping
    ):
        raise TypeError(f"invalid checkpoint: {path}")
    return value


def assert_compatible(states: list[Mapping[str, torch.Tensor]]) -> None:
    if any(list(state) != list(states[0]) for state in states[1:]):
        raise ValueError("model-state key order/schema mismatch")
    for name in states[0]:
        reference = states[0][name]
        for state in states[1:]:
            candidate = state[name]
            if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
                raise ValueError(f"tensor schema mismatch: {name}")


def construct_beta_state(
    new_parent: Mapping[str, torch.Tensor],
    current_exact: Mapping[str, torch.Tensor],
    old_parent: Mapping[str, torch.Tensor],
    old_p12: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    states = [new_parent, current_exact, old_parent, old_p12]
    assert_compatible(states)
    if repair.changed_tensor_names(dict(new_parent), dict(current_exact)) != sorted(
        MUTABLE_NAMES
    ):
        raise ValueError("current P12 did not change exactly head10")
    if repair.changed_tensor_names(dict(old_parent), dict(old_p12)) != sorted(
        MUTABLE_NAMES
    ):
        raise ValueError("historical P12 did not change exactly head10")
    result = {name: tensor.detach().cpu().clone() for name, tensor in new_parent.items()}
    for name in MUTABLE_NAMES:
        base = new_parent[name].detach().cpu()
        value = (
            base.double()
            + (1.0 - BETA)
            * (current_exact[name].detach().cpu().double() - base.double())
            + BETA
            * (
                old_p12[name].detach().cpu().double()
                - old_parent[name].detach().cpu().double()
            )
        ).to(base.dtype)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(name)
        result[name] = value
    if transport.model_state_sha256(result) != BASE_MODEL_SHA256:
        raise ValueError("beta=1.157 model-state hash mismatch")
    if repair.changed_tensor_names(dict(new_parent), result) != sorted(MUTABLE_NAMES):
        raise ValueError("beta=1.157 state did not change exactly head10")
    return result


def configure_head10(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[str]]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    if any(name not in named for name in MUTABLE_NAMES):
        raise ValueError("model lacks a required head10 parameter")
    parameters: list[torch.nn.Parameter] = []
    for name in MUTABLE_NAMES:
        named[name].requires_grad_(True)
        parameters.append(named[name])
    observed = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if tuple(observed) != MUTABLE_NAMES:
        raise RuntimeError(f"trainable scope drift: {observed}")
    return parameters, observed


def special_config(parent: dict[str, Any]) -> ppo.PPOConfig:
    raw = parent.get("config")
    if not isinstance(raw, dict):
        raise ValueError("U468 config missing")
    config = ppo.PPOConfig(**copy.deepcopy(raw))
    config.bc_replay_data = str(FLG_DATA)
    config.bc_replay_split = "train"
    config.bc_replay_batches = CACHE_BATCHES
    config.bc_replay_batch_size = BATCH_SIZE
    config.bc_replay_workers = WORKERS
    config.bc_replay_steps = 1
    config.bc_replay_lr_scale = LR_SCALE
    config.bc_replay_loss = "ordered"
    config.bc_replay_order_context_weight = 8.0
    config.bc_replay_context34_rows_per_batch = CONTEXT34_ROWS_PER_BATCH
    config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    config.max_grad_norm = MAX_GRAD_NORM
    config.seed = CACHE_SEED
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit-only", "formal"), required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_design(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    path = args.design.resolve()
    if path != DESIGN_PATH:
        raise ValueError("unexpected design path")
    evidence = require_regular(path, args.expected_design_sha256, "design")
    design = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(design, dict) or design.get("schema_version") != SCHEMA:
        raise ValueError("design schema mismatch")
    expected = {
        "branch": BRANCH,
        "beta": BETA,
        "base_model_state_sha256": BASE_MODEL_SHA256,
        "learning_rate": LEARNING_RATE,
        "cache_seed": CACHE_SEED,
        "execution_seed": EXECUTION_SEED,
        "batch_indices": list(BATCH_INDICES),
        "endpoint_steps": list(ENDPOINT_STEPS),
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise ValueError(f"design field mismatch: {key}")
    executor = design.get("executor")
    if not isinstance(executor, dict):
        raise ValueError("design executor binding missing")
    self_path = Path(__file__).resolve()
    if executor.get("path") != str(self_path.relative_to(ROOT)):
        raise ValueError("design executor path mismatch")
    if executor.get("sha256") != sha256_file(self_path):
        raise ValueError("design executor hash mismatch")
    return design, evidence


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("current working directory must be the repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("executor must use my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("executor requires Python flags -I -B")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this frozen protocol requires CUDA")
    if not math.isclose(LR_SCALE, 0.00625, rel_tol=0.0, abs_tol=1e-18):
        raise RuntimeError("learning-rate scale drift")
    design, design_evidence = validate_design(args)

    input_evidence = {
        str(path.relative_to(ROOT)): require_regular(path, digest, str(path))
        for path, digest in INPUT_HASHES.items()
    }
    checkpoints = {
        "new_parent": load_checkpoint(U468),
        "new_exact": load_checkpoint(CURRENT_EXACT_P12),
        "old_parent": load_checkpoint(HISTORICAL_PARENT),
        "old_p12": load_checkpoint(HISTORICAL_P12),
    }
    if int(checkpoints["new_parent"].get("update", -1)) != 468:
        raise ValueError("new parent update mismatch")
    base_state = construct_beta_state(
        checkpoints["new_parent"]["model_state_dict"],
        checkpoints["new_exact"]["model_state_dict"],
        checkpoints["old_parent"]["model_state_dict"],
        checkpoints["old_p12"]["model_state_dict"],
    )

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    random.seed(EXECUTION_SEED)
    torch.manual_seed(EXECUTION_SEED)
    torch.cuda.manual_seed_all(EXECUTION_SEED)
    bc_checkpoint = torch.load(GENERAL_BC, map_location="cpu", weights_only=False)
    model = ppo.instantiate_model_from_checkpoint(
        checkpoints["new_parent"], bc_checkpoint, device
    )
    model.load_state_dict(base_state)
    parameters, trainable_names = configure_head10(model)
    if sum(parameter.numel() for parameter in parameters) != 90174:
        raise ValueError("head10 parameter count drift")
    config = special_config(checkpoints["new_parent"])
    cache = ppo.build_bc_replay_batches(config, checkpoints["new_parent"]["model_config"])
    if len(cache) != CACHE_BATCHES:
        raise ValueError("FLG replay cache batch count drift")
    if any(int(batch["action_counts"].shape[0]) != BATCH_SIZE for batch in cache):
        raise ValueError("FLG replay cache contains a short batch")
    if any(
        int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum())
        != CONTEXT34_ROWS_PER_BATCH
        for batch in cache
    ):
        raise ValueError("FLG replay cache context34 quota drift")
    cache_hash, batch_hashes = repair.replay_cache_manifest(cache)
    if cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError("FLG replay cache hash drift")
    selected_hashes = tuple(batch_hashes[index] for index in BATCH_INDICES)
    if selected_hashes != EXPECTED_BATCH_SHA256:
        raise ValueError("selected FLG batch hashes drifted")

    common = {
        "schema_version": SCHEMA,
        "branch": BRANCH,
        "mode": args.mode,
        "design": design_evidence,
        "base": {
            "formula": "U468 + (1-beta)*(current_exact_p12-U468) + beta*(historical_p12-U464)",
            "beta": BETA,
            "model_state_sha256": BASE_MODEL_SHA256,
        },
        "sources": input_evidence,
        "training": {
            "data_split": "train",
            "validation_opened": False,
            "fresh_optimizer": True,
            "optimizer_state_loaded": False,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
            "max_grad_norm": MAX_GRAD_NORM,
            "loss": "ordered",
            "order_context_weight": 8.0,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": 90174,
            "cache_seed": CACHE_SEED,
            "execution_seed": EXECUTION_SEED,
            "cache_batches": CACHE_BATCHES,
            "cache_sha256": cache_hash,
            "batch_indices": list(BATCH_INDICES),
            "batch_sha256": list(selected_hashes),
            "endpoint_steps": list(ENDPOINT_STEPS),
            "rows_per_step": BATCH_SIZE,
            "context34_rows_per_step": CONTEXT34_ROWS_PER_BATCH,
        },
        "runtime": {
            "python": str(Path(sys.executable).resolve()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }

    if args.mode == "audit-only":
        if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
            raise FileExistsError(ATTEMPT_MARKER)
        if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
            raise FileExistsError(OUTPUT_ROOT)
        result = common | {
            "status": "audit_passed_zero_writes",
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "target_paths_absent": True,
            "model_state_unchanged": (
                transport.model_state_sha256(repair.clone_model_state(model))
                == BASE_MODEL_SHA256
            ),
            "fresh_optimizer_not_constructed": True,
        }
        if not result["model_state_unchanged"]:
            raise RuntimeError("audit mutated the base model")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return

    if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
        raise FileExistsError(ATTEMPT_MARKER)
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)
    created_at = datetime.now(timezone.utc).isoformat()
    marker_payload = common | {
        "status": "formal_attempt_consumed",
        "created_at_utc": created_at,
    }
    marker_evidence = publish_o_excl(ATTEMPT_MARKER, canonical_json(marker_payload))
    OUTPUT_ROOT.mkdir(mode=0o700)

    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    if optimizer.state:
        raise RuntimeError("fresh optimizer unexpectedly has state")
    per_step: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    frozen_names = sorted(set(base_state) - set(MUTABLE_NAMES))
    for step, batch_index in enumerate(BATCH_INDICES, start=1):
        metrics = ppo.bc_replay_update(
            model,
            optimizer,
            [cache[batch_index]],
            config,
            device,
            BASE_ACTOR_LEARNING_RATE,
        )
        if not isinstance(metrics, dict) or metrics.get("steps") != 1:
            raise RuntimeError("BC core did not execute exactly one step")
        if int(metrics.get("rows", -1)) != BATCH_SIZE:
            raise RuntimeError("BC core row count drift")
        if int(metrics.get("context_34_rows", -1)) != CONTEXT34_ROWS_PER_BATCH:
            raise RuntimeError("BC core context34 count drift")
        if not repair.finite_nested(metrics):
            raise FloatingPointError("non-finite training metrics")
        optimizer_steps = repair.optimizer_steps(optimizer.state_dict())
        if len(optimizer_steps) != len(MUTABLE_NAMES) or set(optimizer_steps) != {step}:
            raise RuntimeError("fresh optimizer state/step drift")
        per_step.append(
            {
                "step": step,
                "batch_index": batch_index,
                "batch_sha256": batch_hashes[batch_index],
                "metrics": metrics,
                "optimizer_state_sha256": repair.nested_sha256(
                    optimizer.state_dict()
                ),
            }
        )
        if step not in ENDPOINT_STEPS:
            continue
        state = repair.clone_model_state(model)
        if not repair.finite_nested(state):
            raise FloatingPointError("endpoint contains a non-finite tensor")
        changed = repair.changed_tensor_names(base_state, state)
        if changed != sorted(MUTABLE_NAMES):
            raise RuntimeError(f"endpoint P{step} did not change exactly head10")
        if any(not torch.equal(state[name], base_state[name]) for name in frozen_names):
            raise RuntimeError(f"endpoint P{step} changed a frozen tensor")
        model_sha = transport.model_state_sha256(state)
        slim = {
            key: copy.deepcopy(checkpoints["new_parent"][key])
            for key in REQUIRED_SLIM_KEYS
        }
        slim.update(
            {
                "model_state_dict": state,
                "update": 468,
                "evaluation_only": True,
                "resume_forbidden": True,
                "optimizer_states_omitted": [
                    "optimizer_state_dict",
                    "bc_replay_optimizer_state_dict",
                    "opponent_quota_state",
                    "fresh_special_optimizer_state_dict",
                ],
                "flg_head10_fresh_special_bc": {
                    "schema_version": SCHEMA,
                    "design_sha256": design_evidence["sha256"],
                    "base_model_state_sha256": BASE_MODEL_SHA256,
                    "endpoint_step": step,
                    "model_state_sha256": model_sha,
                    "trainable_parameter_names": list(MUTABLE_NAMES),
                    "fresh_optimizer": True,
                    "validation_opened_during_training": False,
                    "resume_forbidden": True,
                },
            }
        )
        buffer = io.BytesIO()
        torch.save(slim, buffer)
        raw = buffer.getvalue()
        reloaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if transport.model_state_sha256(reloaded["model_state_dict"]) != model_sha:
            raise RuntimeError("serialized endpoint model hash drift")
        filename = f"special-bc-flg-head10-fresh-{step:04d}.pt"
        path = OUTPUT_ROOT / filename
        publication = publish_o_excl(path, raw)
        endpoints.append(
            {
                "step": step,
                "path": str(path.relative_to(ROOT)),
                "model_state_sha256": model_sha,
                "changed_parameter_names": changed,
                "fresh_optimizer_steps": optimizer_steps,
                **publication,
            }
        )

    if [item["step"] for item in endpoints] != list(ENDPOINT_STEPS):
        raise RuntimeError("formal run did not publish every frozen endpoint")
    if sha256_file(U468) != INPUT_HASHES[U468]:
        raise RuntimeError("U468 parent changed during training")
    expected_files = sorted(
        [f"special-bc-flg-head10-fresh-{step:04d}.pt" for step in ENDPOINT_STEPS]
    )
    if sorted(path.name for path in OUTPUT_ROOT.iterdir()) != expected_files:
        raise RuntimeError("output tree drift before manifest")

    manifest = common | {
        "status": "training_completed_all_endpoints_published",
        "created_at_utc": created_at,
        "attempt_marker": {
            "path": str(ATTEMPT_MARKER.relative_to(ROOT)),
            **marker_evidence,
        },
        "per_step": per_step,
        "endpoints": endpoints,
        "checkpoint_writes": len(endpoints),
        "optimizer_final_state_sha256": repair.nested_sha256(
            optimizer.state_dict()
        ),
        "validation_opened_during_training": False,
        "parent_unchanged": True,
    }
    manifest_path = OUTPUT_ROOT / "training_manifest.json"
    manifest_evidence = publish_o_excl(manifest_path, canonical_json(manifest))
    completion = {
        "schema_version": SCHEMA,
        "status": "completed",
        "branch": BRANCH,
        "training_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            **manifest_evidence,
        },
        "endpoint_steps": list(ENDPOINT_STEPS),
    }
    completion_path = OUTPUT_ROOT / "COMPLETED.json"
    completion_evidence = publish_o_excl(completion_path, canonical_json(completion))
    final_expected_files = sorted(
        expected_files + ["training_manifest.json", "COMPLETED.json"]
    )
    if sorted(path.name for path in OUTPUT_ROOT.iterdir()) != final_expected_files:
        raise RuntimeError("terminal output tree is not exact")
    verify_visible(ATTEMPT_MARKER, marker_evidence)
    for endpoint in endpoints:
        verify_visible(ROOT / endpoint["path"], endpoint)
    verify_visible(manifest_path, manifest_evidence)
    verify_visible(completion_path, completion_evidence)
    os.chmod(OUTPUT_ROOT, 0o500)
    if sorted(path.name for path in OUTPUT_ROOT.iterdir()) != final_expected_files:
        raise RuntimeError("terminal output tree drifted after sealing")
    result = completion | {
        "completion": {
            "path": str(completion_path.relative_to(ROOT)),
            **completion_evidence,
        }
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
