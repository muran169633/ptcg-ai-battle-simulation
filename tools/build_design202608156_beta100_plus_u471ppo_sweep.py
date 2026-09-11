#!/usr/bin/env python3
"""Build evaluation-only beta100 + scaled raw-U471 PPO-delta endpoints."""

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
from typing import Any, Mapping

import torch


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
RAW_U468 = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
BETA100 = ROOT / "artifacts/ppo_u468_p12delta_direction_beta050_075_100_design202608092/transport-beta-100.pt"
RAW_U471 = ROOT / (
    "artifacts/ppo_u468raw_exactgold12_tailfocus_heads3x96_u471_design202608155/"
    "B_gold_league/seed-202608155/checkpoints/update-0471.pt"
)
OUTPUT_ROOT = ROOT / "artifacts/design202608156_beta100_plus_u471ppo_delta"
INPUT_SHA256 = {
    RAW_U468: "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f",
    BETA100: "53284b1d4e94f09bff5b92a7fb0d24efd26ee67a2a332014cdc97572b00c3beb",
    RAW_U471: "31f0adbbcf68e5124aeae3dbb603e8c3f97e7fe58d9b7477cc42c79fcbf89503",
}
EXPECTED_CHANGED = (
    "actor_key.weight", "actor_query.weight",
    "actor_residual.0.bias", "actor_residual.0.weight",
    "actor_residual.2.bias", "actor_residual.2.weight",
    "count_head.0.bias", "count_head.0.weight",
    "count_head.2.bias", "count_head.2.weight",
    "value_head.0.bias", "value_head.0.weight",
    "value_head.2.bias", "value_head.2.weight",
)
ENDPOINTS = (("alpha025", 0.25), ("alpha050", 0.50), ("alpha075", 0.75))
REQUIRED_SLIM_KEYS = (
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


def changed(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> list[str]:
    return sorted(name for name in left if not torch.equal(left[name], right[name]))


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
        label: torch.load(path, map_location="cpu", weights_only=False)
        for label, path in (("raw_u468", RAW_U468), ("beta100", BETA100), ("raw_u471", RAW_U471))
    }
    raw_u468 = checkpoints["raw_u468"]["model_state_dict"]
    beta100 = checkpoints["beta100"]["model_state_dict"]
    raw_u471 = checkpoints["raw_u471"]["model_state_dict"]
    if set(raw_u468) != set(beta100) or set(raw_u468) != set(raw_u471):
        raise RuntimeError("model-state keys differ")
    if changed(raw_u468, raw_u471) != sorted(EXPECTED_CHANGED):
        raise RuntimeError("raw U471 PPO delta did not change exactly head14")
    for state in (raw_u468, beta100, raw_u471):
        if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
            raise RuntimeError("non-finite source state")
    os.mkdir(OUTPUT_ROOT, mode=0o700)
    created_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": "ptcg-design202608156-beta100-plus-u471ppo-delta-v1",
        "created_at_utc": created_at,
        "formula": "W_beta100 + alpha * (W_rawU471 - W_rawU468)",
        "inputs": {str(path.relative_to(ROOT)): digest for path, digest in INPUT_SHA256.items()},
        "endpoints": [],
        "scope": {"evaluation_only": True, "resume_forbidden": True},
    }
    for endpoint, alpha in ENDPOINTS:
        state = {name: tensor.detach().cpu().clone() for name, tensor in beta100.items()}
        for name in EXPECTED_CHANGED:
            value = (
                beta100[name].detach().cpu().double()
                + alpha * (
                    raw_u471[name].detach().cpu().double()
                    - raw_u468[name].detach().cpu().double()
                )
            ).to(dtype=beta100[name].dtype)
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"non-finite endpoint tensor: {name}")
            state[name] = value
        if changed(beta100, state) != sorted(EXPECTED_CHANGED):
            raise RuntimeError(f"{endpoint} changed unexpected tensors")
        model_hash = state_sha(state)
        slim = {key: copy.deepcopy(checkpoints["beta100"][key]) for key in REQUIRED_SLIM_KEYS}
        slim.update({
            "model_state_dict": state,
            "update": 471,
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_states_omitted": ["optimizer_state_dict", "bc_replay_optimizer_state_dict", "opponent_quota_state"],
            "ppo_delta_blend": {
                "schema_version": "ptcg-design202608156-beta100-plus-u471ppo-endpoint-v1",
                "created_at_utc": created_at,
                "formula": "W_beta100 + alpha * (W_rawU471 - W_rawU468)",
                "endpoint": endpoint,
                "alpha": alpha,
                "mutable_parameter_names": list(EXPECTED_CHANGED),
                "model_state_sha256": model_hash,
                "resume_forbidden": True,
            },
        })
        buffer = io.BytesIO()
        torch.save(slim, buffer)
        payload = buffer.getvalue()
        check = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if check.get("resume_forbidden") is not True or state_sha(check["model_state_dict"]) != model_hash:
            raise RuntimeError(f"serialization verification failed: {endpoint}")
        filename = f"beta100-plus-u471ppo-{endpoint}.pt"
        publish(OUTPUT_ROOT / filename, payload)
        manifest["endpoints"].append({
            "endpoint": endpoint, "alpha": alpha,
            "path": str((OUTPUT_ROOT / filename).relative_to(ROOT)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "model_state_sha256": model_hash,
            "changed_parameter_names": list(EXPECTED_CHANGED),
        })
    publish(OUTPUT_ROOT / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
