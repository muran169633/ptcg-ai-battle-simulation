#!/usr/bin/env python3
"""Build one evaluation-only midpoint from complementary PPO endpoints."""

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
U468_DIRECTION = ROOT / "artifacts/design202608178_alpha075_dualanchor_ppo2x192_u478_v2/B_gold_league/seed-202608178/checkpoints/update-0478.pt"
SELF_ANCHOR_DIRECTION = ROOT / "artifacts/design202608185_alpha075_selfanchor_ppo2x192_u478/B_gold_league/seed-202608185/checkpoints/update-0477.pt"
OUTPUT_ROOT = ROOT / "artifacts/design202608187_crossrun_head14_midpoint"
OUTPUT = OUTPUT_ROOT / "old-u478-new-u477-head14-midpoint.pt"
INPUT_SHA256 = {
    U468_DIRECTION: "4daa01c81d4781461744ec2dcaa1116094756ced051c5bc5a4cc39615a6404a4",
    SELF_ANCHOR_DIRECTION: "62c9caf2f4db65dfe6c831c142a0043166670fa1d8c0e8536a2a2f92058c63f3",
}
HEAD14 = (
    "actor_key.weight", "actor_query.weight",
    "actor_residual.0.bias", "actor_residual.0.weight",
    "actor_residual.2.bias", "actor_residual.2.weight",
    "count_head.0.bias", "count_head.0.weight",
    "count_head.2.bias", "count_head.2.weight",
    "value_head.0.bias", "value_head.0.weight",
    "value_head.2.bias", "value_head.2.weight",
)
SLIM_KEYS = (
    "feature_version", "bc_feature_version", "config", "model_config",
    "learner_deck_hash", "reward", "action_distribution",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape)).encode() + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes() + b"\0")
    return digest.hexdigest()


def publish(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("builder requires repository cwd and my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("builder requires Python -I -B")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)
    for path, expected in INPUT_SHA256.items():
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or sha(path) != expected:
            raise RuntimeError(f"invalid frozen input: {path}")
    source = torch.load(U468_DIRECTION, map_location="cpu", weights_only=False)
    complement = torch.load(SELF_ANCHOR_DIRECTION, map_location="cpu", weights_only=False)
    left = source["model_state_dict"]
    right = complement["model_state_dict"]
    if set(left) != set(right):
        raise RuntimeError("model-state keys differ")
    changed = sorted(name for name in left if not torch.equal(left[name], right[name]))
    if changed != sorted(HEAD14):
        raise RuntimeError("complementary endpoints did not differ exactly on head14")
    state = {name: tensor.detach().cpu().clone() for name, tensor in left.items()}
    for name in HEAD14:
        state[name] = (
            0.5 * left[name].detach().cpu().double()
            + 0.5 * right[name].detach().cpu().double()
        ).to(dtype=left[name].dtype)
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise RuntimeError("non-finite midpoint state")
    created_at = datetime.now(timezone.utc).isoformat()
    model_hash = state_sha(state)
    payload = {key: copy.deepcopy(source[key]) for key in SLIM_KEYS}
    payload.update({
        "model_state_dict": state,
        "update": 478,
        "evaluation_only": True,
        "resume_forbidden": True,
        "optimizer_states_omitted": ["optimizer_state_dict", "bc_replay_optimizer_state_dict", "opponent_quota_state"],
        "crossrun_head14_interpolation": {
            "schema_version": "ptcg-design202608187-crossrun-head14-midpoint-v1",
            "formula": "0.5 * old_dualanchor_U478 + 0.5 * new_selfanchor_U477 on head14",
            "created_at_utc": created_at,
            "model_state_sha256": model_hash,
            "evaluation_only": True,
            "resume_forbidden": True,
        },
    })
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    check = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if check.get("resume_forbidden") is not True or state_sha(check["model_state_dict"]) != model_hash:
        raise RuntimeError("serialization verification failed")
    manifest = {
        "schema_version": "ptcg-design202608187-crossrun-head14-midpoint-manifest-v1",
        "created_at_utc": created_at,
        "formula": "0.5 * old_dualanchor_U478 + 0.5 * new_selfanchor_U477 on head14",
        "inputs": {str(path.relative_to(ROOT)): digest for path, digest in INPUT_SHA256.items()},
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "model_state_sha256": model_hash,
        "changed_parameter_names": list(HEAD14),
        "scope": {"evaluation_only": True, "resume_forbidden": True, "submission": False},
    }
    os.mkdir(OUTPUT_ROOT, mode=0o700)
    publish(OUTPUT, raw)
    publish(OUTPUT_ROOT / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
