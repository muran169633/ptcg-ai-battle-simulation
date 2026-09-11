#!/usr/bin/env python3
"""Build one evaluation-only midpoint between the U477 and U478 head updates."""

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
from typing import Mapping

import torch


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
CHECKPOINT_DIR = ROOT / "artifacts/design202608178_alpha075_dualanchor_ppo2x192_u478_v2/B_gold_league/seed-202608178/checkpoints"
U477 = CHECKPOINT_DIR / "update-0477.pt"
U478 = CHECKPOINT_DIR / "update-0478.pt"
OUTPUT_ROOT = ROOT / "artifacts/design202608182_u477_u478_head14_midpoint"
OUTPUT = OUTPUT_ROOT / "u477-plus-half-u478-head14-delta.pt"
INPUT_SHA256 = {
    U477: "3e5950c32560a7ef1e24a52c4d8594484794c1f9fbc5b737207231ee4566768c",
    U478: "4daa01c81d4781461744ec2dcaa1116094756ced051c5bc5a4cc39615a6404a4",
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


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha(state: Mapping[str, torch.Tensor]) -> str:
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
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or file_sha(path) != expected:
            raise RuntimeError(f"invalid frozen input: {path}")
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in (("u477", U477), ("u478", U478))
    }
    left = checkpoints["u477"]["model_state_dict"]
    right = checkpoints["u478"]["model_state_dict"]
    if set(left) != set(right):
        raise RuntimeError("model-state keys differ")
    changed = sorted(name for name in left if not torch.equal(left[name], right[name]))
    if changed != sorted(HEAD14):
        raise RuntimeError("U477 to U478 did not change exactly head14")
    state = {name: tensor.detach().cpu().clone() for name, tensor in left.items()}
    for name in HEAD14:
        state[name] = (
            left[name].detach().cpu().double()
            + 0.5 * (right[name].detach().cpu().double() - left[name].detach().cpu().double())
        ).to(dtype=left[name].dtype)
    if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
        raise RuntimeError("non-finite midpoint state")
    if sorted(name for name in left if not torch.equal(left[name], state[name])) != sorted(HEAD14):
        raise RuntimeError("midpoint changed unexpected tensors")
    created_at = datetime.now(timezone.utc).isoformat()
    model_hash = state_sha(state)
    slim = {key: copy.deepcopy(checkpoints["u477"][key]) for key in SLIM_KEYS}
    slim.update({
        "model_state_dict": state,
        "update": 478,
        "evaluation_only": True,
        "resume_forbidden": True,
        "optimizer_states_omitted": [
            "optimizer_state_dict", "bc_replay_optimizer_state_dict", "opponent_quota_state"
        ],
        "head14_interpolation": {
            "schema_version": "ptcg-design202608182-u477-u478-head14-midpoint-v1",
            "created_at_utc": created_at,
            "formula": "W_U477 + 0.5 * (W_U478 - W_U477) on head14 only",
            "alpha": 0.5,
            "mutable_parameter_names": list(HEAD14),
            "model_state_sha256": model_hash,
            "evaluation_only": True,
            "resume_forbidden": True,
        },
    })
    buffer = io.BytesIO()
    torch.save(slim, buffer)
    payload = buffer.getvalue()
    check = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if check.get("resume_forbidden") is not True or state_sha(check["model_state_dict"]) != model_hash:
        raise RuntimeError("serialization verification failed")
    manifest = {
        "schema_version": "ptcg-design202608182-u477-u478-head14-midpoint-manifest-v1",
        "created_at_utc": created_at,
        "formula": "W_U477 + 0.5 * (W_U478 - W_U477) on head14 only",
        "inputs": {str(path.relative_to(ROOT)): digest for path, digest in INPUT_SHA256.items()},
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "model_state_sha256": model_hash,
        "changed_parameter_names": list(HEAD14),
        "scope": {"evaluation_only": True, "resume_forbidden": True, "submission": False},
    }
    os.mkdir(OUTPUT_ROOT, mode=0o700)
    publish(OUTPUT, payload)
    publish(OUTPUT_ROOT / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
