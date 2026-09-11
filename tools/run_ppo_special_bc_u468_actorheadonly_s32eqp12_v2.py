#!/usr/bin/env python3
"""Run the frozen single-endpoint U468 actor-head-only S32eqP12 stage."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import stat
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

# Repository-local modules are deliberately imported only inside main(), after
# their exact bytes have been authenticated and held by shared descriptors.
repair: Any = None
ppo: Any = None


REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH = "ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")

DESIGN_V1_PATH = REPO_ROOT / f"artifacts/{BRANCH}.s32_stage_design_preregistration.json"
DESIGN_V1_SHA256 = "5b493f82c6dae77621fb33b1463c28c799cab4182df353e9a29e12519068932a"
DESIGN_PATH = REPO_ROOT / f"artifacts/{BRANCH}.s32_stage_design_preregistration_v2.json"
DESIGN_SHA256 = "d2d3a8d654b449fa316960d4aa4309f866919b32e9e772dea934a4d93b17dfba"
V1_REJECTION_PATH = REPO_ROOT / (
    f"artifacts/{BRANCH}.s32_v1_preflight_static_review_decision.json"
)
V1_REJECTION_SHA256 = (
    "53e1e90b2379b7995e49366b3558f40a54ab8c5e12fff50d627b3c3aea306cd2"
)
MASTER_PATH = REPO_ROOT / f"artifacts/{BRANCH}.master_preregistration.json"
MASTER_SHA256 = "b45e79aae06f0c96f5234451b1648f7df8f033f992f394707e756a297b1395b2"
AUTHORIZATION_PATH = REPO_ROOT / (
    f"artifacts/{BRANCH}.ppo_stage_training_integrity_decision.json"
)
AUTHORIZATION_SHA256 = (
    "350c52c830c65d4447e5fb841193660720e6d88b897d319bb5fb3bc5698d1ffe"
)
PARENT_PATH = REPO_ROOT / (
    f"artifacts/{BRANCH}/ppo_stage/B_gold_league/seed-202607336/"
    "checkpoints/update-0468.pt"
)
PARENT_SHA256 = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
PARENT_UPDATE = 468
PARENT_MODEL_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
PARENT_PPO_SHA256 = (
    "7b9be0b2bff650ad7d6e96e46b80b8aae55a0f96cbe1f7919782301398633087"
)
PARENT_REPLAY_SHA256 = (
    "d92a040c4805141a7ce08f9128a9dc23128b09a2d0cbb6af86f07d0a711973f8"
)
PARENT_QUOTA_SHA256 = (
    "77c2369d0ece97724afbee7034ca012e5ed974878b1c6f3ca6dae96e72e4a743"
)
PARENT_PPO_STATE_COUNT = 28
PARENT_PPO_STEP = 412
PARENT_REPLAY_STATE_COUNT = 24
PARENT_REPLAY_STEP = 32
PARENT_QUOTA_GAMES = 768
PARENT_QUOTA_REFRESH = 466

GENERAL_BC_PATH = REPO_ROOT / (
    "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_"
    "seed20260922_20260731/best.pt"
)
GENERAL_BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
SPECIAL_DATA_PATH = REPO_ROOT / (
    "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_"
    "special_20260801.zip"
)
SPECIAL_DATA_SHA256 = "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598"
TRAIN_PPO_PATH = REPO_ROOT / "tools/train_ppo.py"
TRAIN_PPO_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
TRAIN_BC_PATH = REPO_ROOT / "tools/train_bc_orbit.py"
TRAIN_BC_SHA256 = "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
REPAIR_PATH = REPO_ROOT / "tools/run_ppo_bc_repair.py"
REPAIR_SHA256 = "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"
CG_SIM_PATH = REPO_ROOT / "dataset/sample_submission/sample_submission/cg/sim.py"
CG_SIM_SHA256 = "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655"
CG_INIT_PATH = REPO_ROOT / "dataset/sample_submission/sample_submission/cg/__init__.py"
CG_INIT_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
CG_LIB_PATH = REPO_ROOT / "dataset/sample_submission/sample_submission/cg/libcg.so"
CG_LIB_SHA256 = "feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887"

PREFLIGHT_DIR = REPO_ROOT / f"artifacts/{BRANCH}.s32_preflight_v2"
PREFLIGHT_PATH = PREFLIGHT_DIR / "preflight_audit.json"
FORMAL_DIR = REPO_ROOT / f"artifacts/{BRANCH}/special_stage"
CHECKPOINT_PATH = FORMAL_DIR / "special-bc-actorheadonly-pokemonfan-0032.pt"
MANIFEST_PATH = FORMAL_DIR / "special_bc_manifest.json"
ATTEMPT_MARKER = REPO_ROOT / ".ptcg-u468-s32-specialbc-attempt-202608012-202608090.json"

SPECIAL_SEED = 202608012
CACHE_BATCHES = 32
BATCH_SIZE = 256
WORKERS = 8
CONTEXT34_ROWS_PER_BATCH = 1
STEPS = 32
BATCH_INDICES = tuple(range(32))
CACHE_SHA256 = "a1c16b8b2d6fbf45cf1dfd38ba4eb5527ee444d4d90e09b4f57ca185bdce3021"
SPECIAL_LR_SCALE = 0.01875
SPECIAL_LEARNING_RATE = 6.75e-7
L2_MIN = 0.00177912
L2_MAX = 0.00266868

FROZEN_SHARED_NAMES = (
    "transformer.layers.3.self_attn.in_proj_weight",
    "transformer.layers.3.self_attn.in_proj_bias",
    "transformer.layers.3.self_attn.out_proj.weight",
    "transformer.layers.3.self_attn.out_proj.bias",
    "transformer.layers.3.linear1.weight",
    "transformer.layers.3.linear1.bias",
    "transformer.layers.3.linear2.weight",
    "transformer.layers.3.linear2.bias",
    "transformer.layers.3.norm1.weight",
    "transformer.layers.3.norm1.bias",
    "transformer.layers.3.norm2.weight",
    "transformer.layers.3.norm2.bias",
    "transformer.norm.weight",
    "transformer.norm.bias",
)
MUTABLE_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
    "count_head.0.weight",
    "count_head.0.bias",
    "count_head.2.weight",
    "count_head.2.bias",
)
ACTOR_NAMES = FROZEN_SHARED_NAMES + MUTABLE_NAMES
VALUE_NAMES = (
    "value_head.0.weight",
    "value_head.0.bias",
    "value_head.2.weight",
    "value_head.2.bias",
)

EXPECTED_PARENT_CONFIG: dict[str, Any] = {
    "updates": 468,
    "minibatch_size": 384,
    "actor_reduction": "episode_mean",
    "trainable_scope": "last_block_heads",
    "learning_rate": 3.6e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 0.5,
    "bc_replay_split": "train",
    "bc_replay_batches": 72,
    "bc_replay_batch_size": 256,
    "bc_replay_workers": 8,
    "bc_replay_steps": 2,
    "bc_replay_lr_scale": 0.05,
    "bc_replay_loss": "ordered",
    "bc_replay_order_context_weight": 8.0,
    "bc_replay_context34_rows_per_batch": 4,
    "bc_replay_non_context34_fixed_multi_action_order_weight": 1.0,
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_exclusive(path: Path, value: Any) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        raw = canonical_json_bytes(value)
        written = 0
        while written < len(raw):
            written += os.write(fd, raw[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    size = os.fstat(fd).st_size
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise RuntimeError("short read while hashing held input")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is absent, irregular, or symlinked: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    return observed


def acquire_binding_locks(
    specs: list[tuple[Path, str, str]],
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    try:
        for path, expected_sha256, label in specs:
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            flags = os.O_RDONLY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(path, flags)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                held = os.fstat(fd)
                current = os.lstat(path)
                if not stat.S_ISREG(held.st_mode) or stat.S_ISLNK(current.st_mode):
                    raise ValueError(f"{label} is not a regular non-symlink file")
                if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
                    raise RuntimeError(f"{label} identity changed while locking")
                observed = sha256_fd(fd)
                if observed != expected_sha256:
                    raise ValueError(
                        f"{label} held SHA-256 mismatch: expected "
                        f"{expected_sha256}, got {observed}"
                    )
                snapshots.append(
                    {
                        "path": path,
                        "fd": fd,
                        "device": held.st_dev,
                        "inode": held.st_ino,
                        "size": held.st_size,
                        "mtime_ns": held.st_mtime_ns,
                        "sha256": observed,
                        "label": label,
                    }
                )
                fd = -1
            finally:
                if fd >= 0:
                    os.close(fd)
    except BaseException:
        for snapshot in reversed(snapshots):
            os.close(int(snapshot["fd"]))
        raise
    return snapshots


def assert_binding_locks_unchanged(snapshots: list[dict[str, Any]]) -> bool:
    for snapshot in snapshots:
        path = Path(snapshot["path"])
        held = os.fstat(int(snapshot["fd"]))
        current = os.lstat(path)
        identity = (held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns)
        expected = (
            snapshot["device"],
            snapshot["inode"],
            snapshot["size"],
            snapshot["mtime_ns"],
        )
        if identity != expected:
            raise RuntimeError(f"held input descriptor changed: {path}")
        if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != (
            snapshot["device"],
            snapshot["inode"],
        ):
            raise RuntimeError(f"held input path identity changed: {path}")
        if sha256_fd(int(snapshot["fd"])) != snapshot["sha256"]:
            raise RuntimeError(f"held input content changed: {path}")
    return True


def bindings_from_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, str]:
    return {str(snapshot["path"]): str(snapshot["sha256"]) for snapshot in snapshots}


def clone_nested_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_nested_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested_cpu(item) for item in value)
    return copy.deepcopy(value)


def state_step(entry: dict[str, Any]) -> int:
    raw = entry.get("step")
    if isinstance(raw, torch.Tensor):
        if raw.numel() != 1:
            raise ValueError("AdamW step tensor is not scalar")
        return int(raw.item())
    if raw is None:
        raise ValueError("AdamW state entry lacks step")
    return int(raw)


def quota_games(quota: dict[str, Any]) -> int:
    observed = quota.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("quota observed map is missing")
    return sum(
        int(result.get(key, 0))
        for result in observed.values()
        for key in ("wins", "losses", "draws")
    )


def validate_parent_config(raw: Any) -> ppo.PPOConfig:
    if not isinstance(raw, dict):
        raise ValueError("parent PPO config is missing")
    for key, expected in EXPECTED_PARENT_CONFIG.items():
        if raw.get(key) != expected:
            raise ValueError(
                f"parent config {key!r} mismatch: expected {expected!r}, "
                f"got {raw.get(key)!r}"
            )
    return ppo.PPOConfig(**raw)


def state_mapping(
    replay_state: dict[str, Any], actor_names: list[str]
) -> tuple[dict[Any, Any], list[Any], dict[str, Any]]:
    states = replay_state.get("state")
    groups = replay_state.get("param_groups")
    if not isinstance(states, dict) or not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("replay optimizer state is malformed")
    ids = list(groups[0].get("params", []))
    if len(ids) != 24 or len(set(ids)) != 24 or set(states) != set(ids):
        raise ValueError("replay optimizer does not cover exactly 24 actor states")
    return states, ids, dict(zip(actor_names, ids, strict=True))


class GuardedAdamW(torch.optim.AdamW):
    """Reject any step unless exactly the frozen ten head tensors have gradients."""

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        mutable_named: list[tuple[str, torch.nn.Parameter]],
        frozen_named: list[tuple[str, torch.nn.Parameter]],
    ) -> None:
        super().__init__(parameters, lr=lr, eps=1e-5, weight_decay=weight_decay)
        self.mutable_named = mutable_named
        self.frozen_named = frozen_named
        self.step_audits: list[dict[str, Any]] = []

    def step(self, closure: Any = None) -> Any:
        frozen_with_grad = [
            name for name, parameter in self.frozen_named if parameter.grad is not None
        ]
        mutable_without_grad = [
            name for name, parameter in self.mutable_named if parameter.grad is None
        ]
        mutable_nonfinite = [
            name
            for name, parameter in self.mutable_named
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        ]
        record = {
            "step_ordinal": len(self.step_audits) + 1,
            "frozen_with_grad": frozen_with_grad,
            "mutable_without_grad": mutable_without_grad,
            "mutable_nonfinite_grad": mutable_nonfinite,
            "pass": not (frozen_with_grad or mutable_without_grad or mutable_nonfinite),
        }
        if not record["pass"]:
            raise RuntimeError("gradient guard rejected step: " + json.dumps(record))
        result = super().step(closure)
        self.step_audits.append(record)
        return result


def build_optimizer(
    model: torch.nn.Module,
    config: ppo.PPOConfig,
    parent_replay: dict[str, Any],
    actor_names: list[str],
    full_actor_parameters: list[torch.nn.Parameter],
) -> tuple[GuardedAdamW, dict[str, Any]]:
    _, parent_ids, id_by_name = state_mapping(parent_replay, actor_names)
    named = dict(model.named_parameters())
    required = ACTOR_NAMES + VALUE_NAMES
    if any(name not in named for name in required):
        raise ValueError("model lacks a frozen actor/value parameter")
    model.requires_grad_(False)
    mutable_named = []
    for name in MUTABLE_NAMES:
        named[name].requires_grad_(True)
        mutable_named.append((name, named[name]))
    frozen_named = [(name, named[name]) for name in FROZEN_SHARED_NAMES + VALUE_NAMES]
    observed_trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if observed_trainable != MUTABLE_NAMES:
        raise RuntimeError("trainable model scope is not exactly mutable10")
    optimizer = GuardedAdamW(
        full_actor_parameters,
        lr=config.learning_rate * config.bc_replay_lr_scale,
        weight_decay=config.weight_decay,
        mutable_named=mutable_named,
        frozen_named=frozen_named,
    )
    optimizer.load_state_dict(clone_nested_cpu(parent_replay))
    loaded = clone_nested_cpu(optimizer.state_dict())
    if repair.nested_sha256(loaded) != PARENT_REPLAY_SHA256:
        raise RuntimeError("direct-loaded full replay optimizer differs from parent")
    if list(loaded["param_groups"][0]["params"]) != parent_ids:
        raise RuntimeError("direct-loaded replay parameter order drifted")
    return optimizer, {
        "direct_loaded_parent_state_sha256": repair.nested_sha256(loaded),
        "parameter_ids_by_name": id_by_name,
        "full_state_count": len(loaded["state"]),
        "trainable_parameter_names": list(observed_trainable),
        "mutable_parameter_numel": sum(p.numel() for _, p in mutable_named),
    }


def validate_final_replay(
    parent_replay: dict[str, Any],
    optimizer: GuardedAdamW,
    actor_names: list[str],
    full_actor_parameters: list[torch.nn.Parameter],
    config: ppo.PPOConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_states, parent_ids, id_by_name = state_mapping(parent_replay, actor_names)
    current = clone_nested_cpu(optimizer.state_dict())
    current_states = current.get("state")
    if not isinstance(current_states, dict) or len(current_states) != 24:
        raise RuntimeError("final replay optimizer does not retain 24 states")
    if list(current["param_groups"][0]["params"]) != parent_ids:
        raise RuntimeError("final replay optimizer parameter order drifted")
    expected_groups = clone_nested_cpu(parent_replay["param_groups"])
    expected_groups[0]["lr"] = SPECIAL_LEARNING_RATE
    if repair.nested_sha256(current["param_groups"]) != repair.nested_sha256(expected_groups):
        raise RuntimeError("replay param groups changed beyond the frozen LR override")

    frozen_exact: dict[str, bool] = {}
    frozen_steps: dict[str, int] = {}
    for name in FROZEN_SHARED_NAMES:
        pid = id_by_name[name]
        exact = repair.nested_sha256(current_states[pid]) == repair.nested_sha256(
            parent_states[pid]
        )
        step = state_step(current_states[pid])
        frozen_exact[name] = exact
        frozen_steps[name] = step
        if not exact or step != PARENT_REPLAY_STEP:
            raise RuntimeError(f"frozen replay state drifted for {name}")

    mutable_steps: dict[str, int] = {}
    for name in MUTABLE_NAMES:
        pid = id_by_name[name]
        step = state_step(current_states[pid])
        mutable_steps[name] = step
        if step != PARENT_REPLAY_STEP + STEPS:
            raise RuntimeError(f"mutable replay step drifted for {name}: {step}")
    if not repair.finite_nested(current):
        raise FloatingPointError("final replay optimizer contains non-finite state")

    roundtrip = torch.optim.AdamW(
        full_actor_parameters,
        lr=config.learning_rate * config.bc_replay_lr_scale,
        eps=1e-5,
        weight_decay=config.weight_decay,
    )
    roundtrip.load_state_dict(clone_nested_cpu(current))
    roundtrip_state = clone_nested_cpu(roundtrip.state_dict())
    roundtrip_exact = repair.nested_sha256(roundtrip_state) == repair.nested_sha256(current)
    del roundtrip
    if not roundtrip_exact:
        raise RuntimeError("final mixed-step replay optimizer failed exact roundtrip")
    return current, {
        "state_sha256": repair.nested_sha256(current),
        "full_state_count": len(current_states),
        "parameter_order_retained": True,
        "param_groups_only_lr_changed": True,
        "learning_rate_after": current["param_groups"][0]["lr"],
        "frozen_state_exact_by_name": frozen_exact,
        "frozen_steps_by_name": frozen_steps,
        "mutable_steps_by_name": mutable_steps,
        "roundtrip_load_exact": True,
        "all_state_finite": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--expected-preflight-sha256")
    return parser.parse_args()


def main() -> int:
    global ppo, repair
    args = parse_args()
    tool_path = Path(__file__).resolve()
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError("current working directory must be the repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("executor must use my_project_env Python")
    if tuple(BATCH_INDICES) != tuple(range(32)) or STEPS != 32:
        raise RuntimeError("natural S32 batch order drifted")
    if 3.6e-5 * SPECIAL_LR_SCALE != SPECIAL_LEARNING_RATE:
        raise RuntimeError("special learning-rate identity drifted")
    if len(MUTABLE_NAMES) != 10 or len(FROZEN_SHARED_NAMES) != 14 or len(VALUE_NAMES) != 4:
        raise RuntimeError("frozen parameter partition drifted")

    expected_output = PREFLIGHT_DIR if args.audit_only else FORMAL_DIR
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if args.output_dir.is_symlink() or output_dir != expected_output.resolve():
        raise ValueError("output directory differs from the frozen target")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    if args.audit_only:
        if args.expected_preflight_sha256 is not None:
            raise ValueError("audit-only mode must not supply a preflight SHA")
        if (
            FORMAL_DIR.exists()
            or FORMAL_DIR.is_symlink()
            or ATTEMPT_MARKER.exists()
            or ATTEMPT_MARKER.is_symlink()
        ):
            raise FileExistsError("formal stage or attempt marker already exists")
    else:
        if not args.expected_preflight_sha256:
            raise ValueError("formal mode requires --expected-preflight-sha256")
        if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
            raise FileExistsError("formal S32 attempt was already consumed")

    binding_specs = [
        (tool_path, args.expected_tool_sha256, "executor"),
        (DESIGN_V1_PATH, DESIGN_V1_SHA256, "S32 training design v1"),
        (DESIGN_PATH, DESIGN_SHA256, "S32 transport correction v2"),
        (V1_REJECTION_PATH, V1_REJECTION_SHA256, "rejected v1 decision"),
        (MASTER_PATH, MASTER_SHA256, "master design"),
        (AUTHORIZATION_PATH, AUTHORIZATION_SHA256, "Stage2 authorization"),
        (PARENT_PATH, PARENT_SHA256, "U468 parent"),
        (GENERAL_BC_PATH, GENERAL_BC_SHA256, "general BC"),
        (SPECIAL_DATA_PATH, SPECIAL_DATA_SHA256, "PokemonFan archive"),
        (TRAIN_PPO_PATH, TRAIN_PPO_SHA256, "trainer"),
        (TRAIN_BC_PATH, TRAIN_BC_SHA256, "BC dependency"),
        (REPAIR_PATH, REPAIR_SHA256, "audit dependency"),
        (CG_INIT_PATH, CG_INIT_SHA256, "cg package dependency"),
        (CG_SIM_PATH, CG_SIM_SHA256, "sim dependency"),
        (CG_LIB_PATH, CG_LIB_SHA256, "native dependency"),
    ]
    if not args.audit_only:
        binding_specs.append(
            (
                PREFLIGHT_PATH,
                str(args.expected_preflight_sha256),
                "S32 v2 preflight audit",
            )
        )
    held_inputs = acquire_binding_locks(binding_specs)
    bindings = bindings_from_snapshots(held_inputs)

    # Only now may repository-local code execute.  The source files and every
    # dependency they import are already authenticated and remain held through
    # the terminal publication gate.
    import train_ppo as authenticated_ppo
    import run_ppo_bc_repair as authenticated_repair

    ppo = authenticated_ppo
    repair = authenticated_repair
    assert_binding_locks_unchanged(held_inputs)

    if not args.audit_only:
        preflight = json.loads(PREFLIGHT_PATH.read_text())
        if (
            preflight.get("status") != "audit_passed"
            or preflight.get("optimizer_steps") != 0
            or preflight.get("checkpoint_writes") != 0
            or preflight.get("tool", {}).get("sha256") != args.expected_tool_sha256
            or preflight.get("design", {}).get("sha256") != DESIGN_SHA256
        ):
            raise ValueError("preflight audit is not the frozen v2 zero-step PASS")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    probe = torch.ones(1, device=device) + 1
    if float(probe.item()) != 2.0:
        raise RuntimeError("device preflight failed")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del probe

    torch.use_deterministic_algorithms(True)
    random.seed(SPECIAL_SEED)
    torch.manual_seed(SPECIAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SPECIAL_SEED)

    parent = torch.load(PARENT_PATH, map_location="cpu", weights_only=False)
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise ValueError("parent is not a compatible PPO checkpoint")
    if int(parent.get("update", -1)) != PARENT_UPDATE:
        raise ValueError("parent update is not U468")
    config = validate_parent_config(parent.get("config"))
    if Path(config.bc_checkpoint).resolve() != GENERAL_BC_PATH.resolve():
        raise ValueError("parent general-BC identity drifted")

    bc_checkpoint = torch.load(GENERAL_BC_PATH, map_location="cpu", weights_only=False)
    model = ppo.instantiate_model_from_checkpoint(parent, bc_checkpoint, device)
    full_actor_parameters, _, trainable_manifest = ppo.configure_trainable_scope(
        model, config.trainable_scope
    )
    actor_names = list(trainable_manifest["actor_parameter_names"])
    value_names = list(trainable_manifest["value_parameter_names"])
    if tuple(actor_names) != ACTOR_NAMES or tuple(value_names) != VALUE_NAMES:
        raise ValueError("parent actor/value manifest drifted")
    parameter_manifest = parent.get("optimizer_parameter_names")
    if (
        not isinstance(parameter_manifest, dict)
        or parameter_manifest.get("actor") != actor_names
        or parameter_manifest.get("value") != value_names
    ):
        raise ValueError("checkpoint optimizer manifest drifted")

    parent_replay = parent.get("bc_replay_optimizer_state_dict")
    parent_ppo = parent.get("optimizer_state_dict")
    parent_quota = parent.get("opponent_quota_state")
    if not all(isinstance(item, dict) for item in (parent_replay, parent_ppo, parent_quota)):
        raise ValueError("parent optimizer/quota state is missing")
    if repair.nested_sha256(parent_replay) != PARENT_REPLAY_SHA256:
        raise ValueError("parent replay state hash mismatch")
    if repair.nested_sha256(parent_ppo) != PARENT_PPO_SHA256:
        raise ValueError("parent PPO state hash mismatch")
    if repair.nested_sha256(parent_quota) != PARENT_QUOTA_SHA256:
        raise ValueError("parent quota state hash mismatch")
    if len(parent_replay["state"]) != PARENT_REPLAY_STATE_COUNT or set(
        repair.optimizer_steps(parent_replay)
    ) != {PARENT_REPLAY_STEP}:
        raise ValueError("parent replay state count/step mismatch")
    if len(parent_ppo["state"]) != PARENT_PPO_STATE_COUNT or set(
        repair.optimizer_steps(parent_ppo)
    ) != {PARENT_PPO_STEP}:
        raise ValueError("parent PPO state count/step mismatch")
    if quota_games(parent_quota) != PARENT_QUOTA_GAMES or int(
        parent_quota.get("last_refresh_update", -1)
    ) != PARENT_QUOTA_REFRESH:
        raise ValueError("parent quota game/refresh state mismatch")
    model_hash_before = ppo.model_state_sha256(model)
    if model_hash_before != PARENT_MODEL_SHA256:
        raise ValueError("parent runtime model hash mismatch")
    model_before = repair.clone_model_state(model)

    replay_optimizer, optimizer_init = build_optimizer(
        model, config, parent_replay, actor_names, full_actor_parameters
    )
    special_config = copy.deepcopy(config)
    special_config.bc_replay_data = str(SPECIAL_DATA_PATH)
    special_config.bc_replay_split = "train"
    special_config.bc_replay_batches = CACHE_BATCHES
    special_config.bc_replay_batch_size = BATCH_SIZE
    special_config.bc_replay_workers = WORKERS
    special_config.bc_replay_steps = 1
    special_config.bc_replay_lr_scale = SPECIAL_LR_SCALE
    special_config.bc_replay_loss = "ordered"
    special_config.bc_replay_order_context_weight = 8.0
    special_config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    special_config.bc_replay_context34_rows_per_batch = CONTEXT34_ROWS_PER_BATCH
    special_config.seed = SPECIAL_SEED
    replay_batches = ppo.build_bc_replay_batches(special_config, parent["model_config"])
    if len(replay_batches) != CACHE_BATCHES or any(
        int(batch["contexts"].shape[0]) != BATCH_SIZE for batch in replay_batches
    ):
        raise ValueError("special replay cache shape drifted")
    context34 = [
        int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum())
        for batch in replay_batches
    ]
    if set(context34) != {CONTEXT34_ROWS_PER_BATCH}:
        raise ValueError("special cache context34 quota drifted")
    cache_sha, batch_hashes = repair.replay_cache_manifest(replay_batches)
    if cache_sha != CACHE_SHA256:
        raise ValueError(f"special cache hash mismatch: {cache_sha}")

    common = {
        "schema_version": "ptcg-u468-actorheadonly-s32eqp12-execution-v2",
        "mode": "audit_only" if args.audit_only else "formal",
        "tool": {"path": str(tool_path), "sha256": args.expected_tool_sha256},
        "training_design_v1": {
            "path": str(DESIGN_V1_PATH),
            "sha256": DESIGN_V1_SHA256,
        },
        "design": {"path": str(DESIGN_PATH), "sha256": DESIGN_SHA256},
        "v1_rejection": {
            "path": str(V1_REJECTION_PATH),
            "sha256": V1_REJECTION_SHA256,
        },
        "master": {"path": str(MASTER_PATH), "sha256": MASTER_SHA256},
        "authorization": {
            "path": str(AUTHORIZATION_PATH),
            "sha256": AUTHORIZATION_SHA256,
        },
        "parent": {
            "path": str(PARENT_PATH),
            "sha256": PARENT_SHA256,
            "update": PARENT_UPDATE,
            "model_state_sha256": model_hash_before,
            "ppo_state_sha256": PARENT_PPO_SHA256,
            "replay_state_sha256": PARENT_REPLAY_SHA256,
            "quota_state_sha256": PARENT_QUOTA_SHA256,
        },
        "bindings": bindings,
        "special_bc": {
            "seed": SPECIAL_SEED,
            "steps": STEPS,
            "batch_indices": list(BATCH_INDICES),
            "batch_sha256": batch_hashes,
            "rows": STEPS * BATCH_SIZE,
            "context34_rows": STEPS * CONTEXT34_ROWS_PER_BATCH,
            "learning_rate": SPECIAL_LEARNING_RATE,
            "lr_scale": SPECIAL_LR_SCALE,
            "loss": "ordered",
            "order_context_weight": 8.0,
            "mutable_parameter_names": list(MUTABLE_NAMES),
        },
        "cache": {
            "sha256": cache_sha,
            "batches": CACHE_BATCHES,
            "batch_size": BATCH_SIZE,
            "context34_per_batch": context34,
        },
        "optimizer_initialization": optimizer_init,
        "device": str(device),
    }

    if args.audit_only:
        zero_state = clone_nested_cpu(replay_optimizer.state_dict())
        result = common | {
            "status": "audit_passed",
            "optimizer_steps": 0,
            "training_rows": 0,
            "checkpoint_writes": 0,
            "formal_output_absent": (
                not FORMAL_DIR.exists() and not FORMAL_DIR.is_symlink()
            ),
            "attempt_marker_absent": (
                not ATTEMPT_MARKER.exists() and not ATTEMPT_MARKER.is_symlink()
            ),
            "model_unchanged": repair.changed_tensor_names(
                model_before, repair.clone_model_state(model)
            ) == [],
            "replay_state_exact_parent": (
                repair.nested_sha256(zero_state) == PARENT_REPLAY_SHA256
            ),
            "ppo_state_exact_parent": repair.nested_sha256(parent_ppo) == PARENT_PPO_SHA256,
            "quota_state_exact_parent": repair.nested_sha256(parent_quota) == PARENT_QUOTA_SHA256,
            "parent_file_unchanged": sha256_file(PARENT_PATH) == PARENT_SHA256,
            "all_bound_files_unchanged": assert_binding_locks_unchanged(
                held_inputs
            ),
            "all_state_finite": all(
                repair.finite_nested(value) for value in (model_before, zero_state, parent_ppo, parent_quota)
            ),
        }
        required = (
            "formal_output_absent",
            "attempt_marker_absent",
            "model_unchanged",
            "replay_state_exact_parent",
            "ppo_state_exact_parent",
            "quota_state_exact_parent",
            "parent_file_unchanged",
            "all_bound_files_unchanged",
            "all_state_finite",
        )
        if not all(result[name] for name in required) or replay_optimizer.step_audits:
            raise RuntimeError("zero-step S32 preflight failed")
        assert_binding_locks_unchanged(held_inputs)
        PREFLIGHT_DIR.mkdir(parents=False, exist_ok=False)
        write_json_exclusive(PREFLIGHT_PATH, result)
        print(json.dumps({"status": result["status"], "output": str(PREFLIGHT_PATH)}))
        return 0

    assert_binding_locks_unchanged(held_inputs)
    marker = {
        "event": "u468_actorheadonly_s32eqp12_attempt_consumed",
        "tool_sha256": args.expected_tool_sha256,
        "design_sha256": DESIGN_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "parent_sha256": PARENT_SHA256,
        "preflight_sha256": args.expected_preflight_sha256,
        "seed": SPECIAL_SEED,
        "attempt": 1,
        "attempts_authorized": 1,
        "retry_authorized": False,
    }
    write_json_exclusive(ATTEMPT_MARKER, marker)
    FORMAL_DIR.mkdir(parents=False, exist_ok=False)

    per_step: list[dict[str, Any]] = []
    for step, batch_index in enumerate(BATCH_INDICES, start=1):
        metrics = ppo.bc_replay_update(
            model,
            replay_optimizer,
            [replay_batches[batch_index]],
            special_config,
            device,
            config.learning_rate,
        )
        if (
            not isinstance(metrics, dict)
            or metrics.get("steps") != 1
            or int(metrics.get("rows", -1)) != BATCH_SIZE
            or int(metrics.get("context_34_rows", -1)) != CONTEXT34_ROWS_PER_BATCH
            or metrics.get("selected_batch_indices") != [0]
            or metrics.get("learning_rate") != SPECIAL_LEARNING_RATE
        ):
            raise RuntimeError(f"special step {step} integrity failed")
        if not repair.finite_nested(metrics):
            raise FloatingPointError(f"special step {step} metrics are non-finite")
        if len(replay_optimizer.step_audits) != step:
            raise RuntimeError("gradient guard did not cover every special step")
        per_step.append(
            {
                "step": step,
                "batch_index": batch_index,
                "batch_sha256": batch_hashes[batch_index],
                "metrics": metrics,
                "gradient_guard": copy.deepcopy(replay_optimizer.step_audits[-1]),
            }
        )

    model_after = repair.clone_model_state(model)
    changed = repair.changed_tensor_names(model_before, model_after)
    if len(changed) != 10 or set(changed) != set(MUTABLE_NAMES):
        raise RuntimeError("final changed tensors are not exactly mutable10")
    if any(
        not torch.equal(model_before[name], model_after[name])
        for name in model_before
        if name not in set(MUTABLE_NAMES)
    ):
        raise RuntimeError("a model tensor outside mutable10 changed")
    l2_sq = 0.0
    for name in MUTABLE_NAMES:
        delta = model_after[name].double() - model_before[name].double()
        l2_sq += float(torch.sum(delta * delta).item())
    displacement = math.sqrt(l2_sq)
    if not (L2_MIN <= displacement <= L2_MAX):
        raise RuntimeError(
            f"mutable10 displacement {displacement} outside [{L2_MIN}, {L2_MAX}]"
        )
    if not repair.finite_nested(model_after):
        raise FloatingPointError("final model contains non-finite tensors")

    replay_after, replay_audit = validate_final_replay(
        parent_replay, replay_optimizer, actor_names, full_actor_parameters, config
    )
    model_hash_after = ppo.model_state_sha256(model)
    ppo_unchanged = repair.nested_sha256(parent_ppo) == PARENT_PPO_SHA256
    quota_unchanged = repair.nested_sha256(parent_quota) == PARENT_QUOTA_SHA256
    parent_unchanged = sha256_file(PARENT_PATH) == PARENT_SHA256
    bound_unchanged = assert_binding_locks_unchanged(held_inputs)
    if not all((ppo_unchanged, quota_unchanged, parent_unchanged, bound_unchanged)):
        raise RuntimeError("a frozen parent state or bound file changed")

    integrity = {
        "optimizer_steps": len(per_step),
        "rows": len(per_step) * BATCH_SIZE,
        "context34_rows": len(per_step) * CONTEXT34_ROWS_PER_BATCH,
        "batch_order_exact_natural_0_to_31": [r["batch_index"] for r in per_step]
        == list(BATCH_INDICES),
        "gradient_guard_all_pass": all(r["gradient_guard"]["pass"] for r in per_step),
        "all_metrics_finite": True,
        "changed_parameter_names": changed,
        "changed_exactly_mutable10": True,
        "all_model_tensors_outside_mutable_byte_equal_parent": True,
        "mutable10_float64_l2_displacement": displacement,
        "mutable10_l2_gate": [L2_MIN, L2_MAX],
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "replay_state_sha256_before": PARENT_REPLAY_SHA256,
        "replay_state_sha256_after": replay_audit["state_sha256"],
        "replay_optimizer": replay_audit,
        "ppo_state_sha256": PARENT_PPO_SHA256,
        "ppo_optimizer_unchanged": ppo_unchanged,
        "quota_state_sha256": PARENT_QUOTA_SHA256,
        "quota_unchanged": quota_unchanged,
        "parent_file_unchanged": parent_unchanged,
        "all_bound_files_unchanged": bound_unchanged,
    }
    if not all(
        (
            integrity["optimizer_steps"] == 32,
            integrity["rows"] == 8192,
            integrity["context34_rows"] == 32,
            integrity["batch_order_exact_natural_0_to_31"],
            integrity["gradient_guard_all_pass"],
        )
    ):
        raise RuntimeError("final S32 trajectory integrity failed")

    provenance = common | {
        "status": "trajectory_validated_pending_terminal_manifest",
        "preflight": {
            "path": str(PREFLIGHT_PATH),
            "sha256": args.expected_preflight_sha256,
        },
        "attempt_marker": {"path": str(ATTEMPT_MARKER), "payload": marker},
        "per_step": per_step,
        "integrity": integrity,
        "checkpoint_update_label": PARENT_UPDATE,
        "not_a_new_ppo_update": True,
        "no_prefix_checkpoint_or_endpoint_selection": True,
    }
    payload = copy.deepcopy(parent)
    payload["model_state_dict"] = model_after
    payload["bc_replay_optimizer_state_dict"] = replay_after
    payload["post_ppo_special_bc"] = provenance
    checkpoint_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        checkpoint_flags |= os.O_NOFOLLOW
    checkpoint_fd = os.open(CHECKPOINT_PATH, checkpoint_flags, 0o600)
    try:
        with os.fdopen(checkpoint_fd, "wb", closefd=True) as checkpoint_handle:
            checkpoint_fd = -1
            torch.save(payload, checkpoint_handle)
            checkpoint_handle.flush()
            os.fsync(checkpoint_handle.fileno())
    finally:
        if checkpoint_fd >= 0:
            os.close(checkpoint_fd)
    if not CHECKPOINT_PATH.is_file() or CHECKPOINT_PATH.is_symlink():
        raise RuntimeError("terminal checkpoint publication is not a regular file")
    checkpoint_sha = sha256_file(CHECKPOINT_PATH)

    # Reload the exact published bytes before allowing a completed manifest.
    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    saved_model = ppo.instantiate_model_from_checkpoint(
        saved, bc_checkpoint, torch.device("cpu")
    )
    saved_model_hash = ppo.model_state_sha256(saved_model)
    del saved_model
    if (
        int(saved.get("update", -1)) != PARENT_UPDATE
        or saved_model_hash != model_hash_after
        or repair.nested_sha256(saved.get("bc_replay_optimizer_state_dict"))
        != replay_audit["state_sha256"]
        or repair.nested_sha256(saved.get("optimizer_state_dict"))
        != PARENT_PPO_SHA256
        or repair.nested_sha256(saved.get("opponent_quota_state"))
        != PARENT_QUOTA_SHA256
    ):
        raise RuntimeError("published checkpoint exact reload gate failed")
    if sorted(path.name for path in FORMAL_DIR.iterdir()) != [CHECKPOINT_PATH.name]:
        raise RuntimeError("formal directory was not checkpoint-only before manifest")
    assert_binding_locks_unchanged(held_inputs)

    result = provenance | {
        "status": "actorheadonly_s32eqp12_completed",
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": checkpoint_sha,
            "update": PARENT_UPDATE,
            "special_steps": STEPS,
            "model_state_sha256": model_hash_after,
            "replay_state_sha256": replay_audit["state_sha256"],
        },
        "checkpoint_writes": 1,
        "checkpoint_exact_reload_passed": True,
    }
    write_json_exclusive(MANIFEST_PATH, result)
    expected_files = sorted([CHECKPOINT_PATH.name, MANIFEST_PATH.name])
    if sorted(path.name for path in FORMAL_DIR.iterdir()) != expected_files:
        raise RuntimeError("formal S32 directory contains unexpected files")
    assert_binding_locks_unchanged(held_inputs)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checkpoint": str(CHECKPOINT_PATH),
                "checkpoint_sha256": checkpoint_sha,
                "model_state_sha256": model_hash_after,
                "mutable10_l2": displacement,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
