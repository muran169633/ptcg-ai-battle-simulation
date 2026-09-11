#!/usr/bin/env python3
"""Materialize three actor-only shrinkages of the U472-heavy PPO displacement."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
PARENT = ROOT / "artifacts/design202608195_alpha075_latestreplay_general_bc_s24/general-bc-s16.pt"
PARENT_SHA256 = "4bf6e80448e76965853580b92d0524e3afff0e685d629fd3bfce8aad4328acb8"
PPO = ROOT / "artifacts/design202608200_s16_u472heavy_ppo1x192_u477/B_gold_league/seed-202608200/checkpoints/update-0477.pt"
PPO_SHA256 = "571c3760af4d24a92847cf0057f95e774a2b23e96d2e99841918b28913b81f8a"
DECISION = ROOT / "artifacts/design202608201_s16_u472heavy_u477_decision.json"
DECISION_SHA256 = "fdd1a62190d6e48726ee0122f77fbe97de3fba07d3daaf1531b8d7611b044898"
OUTPUT_ROOT = ROOT / "artifacts/design202608202_s16_u472heavy_actor_transport"
MANIFEST = OUTPUT_ROOT / "manifest.json"
BETAS = (0.25, 0.5, 0.75)
ACTOR_NAMES = (
    "actor_query.weight", "actor_key.weight",
    "actor_residual.0.weight", "actor_residual.0.bias",
    "actor_residual.2.weight", "actor_residual.2.bias",
    "count_head.0.weight", "count_head.0.bias",
    "count_head.2.weight", "count_head.2.bias",
)
REQUIRED_KEYS = (
    "feature_version", "bc_feature_version", "config", "model_config",
    "learner_deck_hash", "reward", "action_distribution",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(path: Path, expected: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or sha(path) != expected:
        raise RuntimeError(f"frozen input mismatch: {path}")


def state_sha(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        h.update(json.dumps(list(value.shape)).encode() + b"\0")
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes() + b"\0")
    return h.hexdigest()


def publish(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload); os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires repository cwd and my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)
    for path, expected in ((PARENT, PARENT_SHA256), (PPO, PPO_SHA256), (DECISION, DECISION_SHA256)):
        require(path, expected)
    decision = json.loads(DECISION.read_text())
    probe = decision.get("decision", {}).get("transport_probe", {})
    if probe.get("authorized") is not True or probe.get("betas") != list(BETAS):
        raise RuntimeError("decision does not authorize exact transport set")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    ppo = torch.load(PPO, map_location="cpu", weights_only=False)
    parent_state = parent["model_state_dict"]
    ppo_state = ppo["model_state_dict"]
    changed = sorted(name for name in parent_state if not torch.equal(parent_state[name], ppo_state[name]))
    expected_changed = sorted((*ACTOR_NAMES, "value_head.0.weight", "value_head.0.bias", "value_head.2.weight", "value_head.2.bias"))
    if changed != expected_changed:
        raise RuntimeError("PPO displacement scope drifted")
    os.mkdir(OUTPUT_ROOT, mode=0o700)
    endpoints = []
    created_at = datetime.now(timezone.utc).isoformat()
    for beta in BETAS:
        state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
        for name in ACTOR_NAMES:
            value = parent_state[name].float() + beta * (ppo_state[name].float() - parent_state[name].float())
            state[name] = value.to(parent_state[name].dtype)
        actual_changed = sorted(name for name in state if not torch.equal(state[name], parent_state[name]))
        if actual_changed != sorted(ACTOR_NAMES) or not all(torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError(f"beta {beta} integrity failed")
        payload = {key: copy.deepcopy(parent[key]) for key in REQUIRED_KEYS}
        payload.update({
            "model_state_dict": state,
            "update": 477,
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_states_omitted": ["optimizer_state_dict", "bc_replay_optimizer_state_dict", "opponent_quota_state"],
            "actor_transport": {
                "schema_version": "ptcg-design202608202-s16-u472heavy-actor-transport-v1",
                "created_at_utc": created_at,
                "beta": beta,
                "parent_sha256": PARENT_SHA256,
                "ppo_sha256": PPO_SHA256,
                "mutable_parameter_names": list(ACTOR_NAMES),
                "value_head_source": "parent_bit_exact",
            },
        })
        buffer = io.BytesIO(); torch.save(payload, buffer); raw = buffer.getvalue()
        filename = f"s16-u472heavy-actor-beta{int(beta * 100):03d}.pt"
        publish(OUTPUT_ROOT / filename, raw)
        endpoints.append({
            "beta": beta, "path": str((OUTPUT_ROOT / filename).relative_to(ROOT)),
            "sha256": hashlib.sha256(raw).hexdigest(), "model_state_sha256": state_sha(state),
            "changed_parameter_names": actual_changed,
        })
    manifest = {
        "schema_version": "ptcg-design202608202-s16-u472heavy-actor-transport-manifest-v1",
        "created_at_utc": created_at,
        "parent": {"path": str(PARENT.relative_to(ROOT)), "sha256": PARENT_SHA256},
        "ppo": {"path": str(PPO.relative_to(ROOT)), "sha256": PPO_SHA256},
        "decision": {"path": str(DECISION.relative_to(ROOT)), "sha256": DECISION_SHA256},
        "endpoints": endpoints,
        "integrity": {"actor_only": True, "value_head_parent_bit_exact": True, "all_finite": True},
        "scope": {"evaluation_only": True, "package": False, "upload": False, "submission": False},
    }
    publish(MANIFEST, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
