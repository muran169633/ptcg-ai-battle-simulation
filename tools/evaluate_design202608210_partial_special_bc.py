#!/usr/bin/env python3
"""Audit the S2/S4 endpoints left by the gated Design 202608210 run."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import torch  # noqa: E402
import run_design202608209_balancedppo_flg_twoweek_special_bc as base  # noqa: E402
import run_ppo_bc_repair as audit  # noqa: E402
import train_ppo as ppo  # noqa: E402

EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
BASE_SHA256 = "8a15c2af66d42cccebc445e89f6cb2d78bf38cf563b45012e105225251438d5e"
PARENT = base.PARENT
PARENT_SHA256 = base.PARENT_SHA256
BC = base.BC
BC_SHA256 = base.BC_SHA256
SPECIAL = base.SPECIAL
SPECIAL_SHA256 = base.SPECIAL_SHA256
GENERAL = base.GENERAL
GENERAL_SHA256 = base.GENERAL_SHA256
ENDPOINTS = {
    2: (ROOT / "artifacts/design202608210_balancedppo_flg_hardbatch_special_bc/special-bc-flg-s02.pt", "932177aff2f5c33ec65c748ffa28e6c55fd95dd6341c9d4b4ce075c45dce096b"),
    4: (ROOT / "artifacts/design202608210_balancedppo_flg_hardbatch_special_bc/special-bc-flg-s04.pt", "e208276573ba8d9b451e150254b8232b7b370db1d76274175353121a1fb5f72b"),
}
OUTPUT = ROOT / "artifacts/design202608210_balancedppo_flg_hardbatch_special_bc_partial_result.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(path: Path, expected: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or sha(path) != expected:
        raise RuntimeError(f"frozen input mismatch: {path}")


def main() -> int:
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires repo cwd and my_project_env")
    for path, digest in (
        (Path(base.__file__).resolve(), BASE_SHA256),
        (PARENT, PARENT_SHA256),
        (BC, BC_SHA256),
        (SPECIAL, SPECIAL_SHA256),
        (GENERAL, GENERAL_SHA256),
        *ENDPOINTS.values(),
    ):
        require(path, digest)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(OUTPUT)

    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    bc = torch.load(BC, map_location="cpu", weights_only=False)
    parent_model = ppo.instantiate_model_from_checkpoint(parent, bc, device)
    parent_state = audit.clone_model_state(parent_model)

    special_config = ppo.PPOConfig(**parent["config"])
    special_config.bc_replay_data = str(SPECIAL)
    special_config.bc_replay_split = "train"
    special_config.bc_replay_batches = base.SPECIAL_BATCHES
    special_config.bc_replay_batch_size = base.BATCH_SIZE
    special_config.bc_replay_workers = 2
    special_config.bc_replay_steps = 1
    special_config.bc_replay_lr_scale = 1.0
    special_config.bc_replay_loss = "ordered"
    special_config.bc_replay_order_context_weight = 8.0
    special_config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    special_config.bc_replay_context34_rows_per_batch = 1
    special_config.seed = base.SEED
    valid_config = copy.deepcopy(special_config)
    valid_config.bc_replay_data = str(GENERAL)
    valid_config.bc_replay_split = "valid"
    valid_config.bc_replay_batches = base.VALID_BATCHES
    valid_config.bc_replay_workers = 1
    valid_config.bc_replay_context34_rows_per_batch = 4
    valid_config.seed = base.SEED + 1
    special_batches = ppo.build_bc_replay_batches(special_config, parent["model_config"])
    valid_batches = ppo.build_bc_replay_batches(valid_config, parent["model_config"])
    special_cache_sha, _ = audit.replay_cache_manifest(special_batches)
    valid_cache_sha, _ = audit.replay_cache_manifest(valid_batches)
    parent_special = base.cache_loss(parent_model, special_batches, special_config, device)
    parent_valid = base.cache_loss(parent_model, valid_batches, valid_config, device)

    endpoint_results = {}
    value_count_names = [name for name in parent_state if name.startswith(("value_head.", "count_head."))]
    for step, (path, digest) in ENDPOINTS.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ppo.instantiate_model_from_checkpoint(checkpoint, bc, device)
        state = audit.clone_model_state(model)
        changed = audit.changed_tensor_names(parent_state, state)
        special_loss = base.cache_loss(model, special_batches, special_config, device)
        valid_loss = base.cache_loss(model, valid_batches, valid_config, device)
        endpoint_results[str(step)] = {
            "checkpoint": str(path.relative_to(ROOT)),
            "sha256": digest,
            "special_loss": special_loss,
            "special_loss_delta": special_loss["loss"] - parent_special["loss"],
            "general_valid_loss": valid_loss,
            "general_valid_loss_delta": valid_loss["loss"] - parent_valid["loss"],
            "changed_parameter_names": changed,
            "changed_exactly_actor6": changed == sorted(base.ACTOR6),
            "value_and_count_heads_unchanged": all(torch.equal(parent_state[name], state[name]) for name in value_count_names),
            "all_finite": audit.finite_nested(state),
        }

    result = {
        "schema_version": "ptcg-design202608210-partial-special-bc-result-v1",
        "status": "partial_endpoints_audited_s8_rejected_by_training_gate",
        "parent": {"path": str(PARENT.relative_to(ROOT)), "sha256": PARENT_SHA256},
        "protocol": {"batch_indices": [22, 24, 12, 18, 1, 8, 0, 6], "learning_rate": base.LR, "requested_endpoints": [2, 4, 8]},
        "caches": {
            "special_train": {"sha256": special_cache_sha, "loss_before": parent_special},
            "general_valid": {"sha256": valid_cache_sha, "loss_before": parent_valid, "used_for_training": False},
        },
        "endpoints": endpoint_results,
        "s8": {"checkpoint_written": False, "rejection": "full special-train loss gate failed before checkpoint write"},
        "scope": {"local_only": True, "package": False, "upload": False, "submission": False},
    }
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
