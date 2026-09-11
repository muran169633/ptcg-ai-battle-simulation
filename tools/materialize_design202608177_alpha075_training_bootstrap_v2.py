#!/usr/bin/env python3
"""Revise the alpha0.75 bootstrap to truthfully declare absent quota state."""

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
SOURCE = ROOT / "artifacts/design202608175_alpha075_fresh_training_bootstrap/alpha075-fresh-bootstrap-u476.pt"
SOURCE_SHA256 = "54b9a91dc740d01e053dfa3bbcc91c227edc765b230205b1ddb03da586f0da02"
FAILED_LOG = ROOT / "artifacts/design202608176_alpha075_dualanchor_ppo2x192_u478.log"
FAILED_LOG_SHA256 = "1d871dc9e66c124cc52d557f7efe9c8d7e1bb3b9cf448cac177698f98e1cff70"
OUTPUT_ROOT = ROOT / "artifacts/design202608177_alpha075_fresh_training_bootstrap_v2"
OUTPUT = OUTPUT_ROOT / "alpha075-fresh-bootstrap-legacyquota-u476.pt"
MANIFEST = OUTPUT_ROOT / "manifest.json"
FORBIDDEN_KEYS = (
    "optimizer_state_dict", "bc_replay_optimizer_state_dict", "opponent_quota_state",
    "ppo_objective_state", "evaluation_only", "resume_forbidden",
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


def validate(path: Path, expected: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or file_sha(path) != expected:
        raise RuntimeError(f"frozen input mismatch: {path}")


def publish(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload); os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("materializer requires repository cwd and my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("materializer requires Python -I -B")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)
    validate(SOURCE, SOURCE_SHA256)
    validate(FAILED_LOG, FAILED_LOG_SHA256)
    if "Exact-quota resume checkpoint lacks opponent quota state" not in FAILED_LOG.read_text():
        raise RuntimeError("bound failure log does not contain the expected pre-rollout error")
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if any(key in source for key in FORBIDDEN_KEYS):
        raise RuntimeError("source bootstrap contains forbidden state")
    source_hash = state_sha(source["model_state_dict"])
    payload = copy.deepcopy(source)
    old_mode = payload["config"].get("opponent_quota_mode")
    if old_mode != "fixed":
        raise RuntimeError(f"unexpected source quota mode: {old_mode!r}")
    payload["config"]["opponent_quota_mode"] = "legacy"
    payload["config"].pop("opponent_base_quotas", None)
    payload["config"].pop("opponent_caps", None)
    payload["config"].pop("opponent_audit", None)
    created_at = datetime.now(timezone.utc).isoformat()
    payload["fresh_training_bootstrap_v2"] = {
        "schema_version": "ptcg-design202608177-alpha075-bootstrap-v2",
        "created_at_utc": created_at,
        "source_sha256": SOURCE_SHA256,
        "model_weights_bit_exact_source": True,
        "quota_state_present": False,
        "checkpoint_quota_mode": "legacy",
        "fresh_exact_quota_transition_requires_optimizer_reset": True,
        "failed_launch_log_sha256": FAILED_LOG_SHA256,
        "failed_before_rollout": True,
    }
    buffer = io.BytesIO(); torch.save(payload, buffer); raw = buffer.getvalue()
    check = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if state_sha(check["model_state_dict"]) != source_hash or check["config"].get("opponent_quota_mode") != "legacy":
        raise RuntimeError("serialized bootstrap v2 verification failed")
    if any(key in check for key in FORBIDDEN_KEYS):
        raise RuntimeError("serialized bootstrap v2 contains forbidden state")
    os.mkdir(OUTPUT_ROOT, mode=0o700)
    publish(OUTPUT, raw)
    manifest = {
        "schema_version": "ptcg-design202608177-alpha075-bootstrap-v2-manifest-v1",
        "created_at_utc": created_at,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "source_model_state_sha256": source_hash,
        "output_model_state_sha256": state_sha(check["model_state_dict"]),
        "model_weights_bit_exact_source": True,
        "checkpoint_quota_mode": "legacy",
        "optimizer_states_present": False,
        "quota_state_present": False,
        "failed_predecessor_launch": "design202608176",
        "failed_predecessor_optimizer_steps": 0,
        "failed_predecessor_rollout_games": 0,
    }
    publish(MANIFEST, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
