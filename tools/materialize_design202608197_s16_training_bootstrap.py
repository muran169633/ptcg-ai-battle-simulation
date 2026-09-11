#!/usr/bin/env python3
"""Materialize selected latest-replay general-BC S16 as a fresh PPO seed."""

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
SOURCE = ROOT / "artifacts/design202608195_alpha075_latestreplay_general_bc_s24/general-bc-s16.pt"
SOURCE_SHA256 = "4bf6e80448e76965853580b92d0524e3afff0e685d629fd3bfce8aad4328acb8"
DECISION = ROOT / "artifacts/design202608196_latestreplay_general_bc_screen128_decision.json"
DECISION_SHA256 = "246ac1d3dc6f721e636391b4868a3bad69d7729305908ca544bb0acd9124571f"
OUTPUT_ROOT = ROOT / "artifacts/design202608197_s16_fresh_training_bootstrap"
OUTPUT = OUTPUT_ROOT / "s16-fresh-bootstrap-legacyquota-u476.pt"
MANIFEST = OUTPUT_ROOT / "manifest.json"
REQUIRED_KEYS = (
    "feature_version", "bc_feature_version", "config", "model_config",
    "learner_deck_hash", "reward", "action_distribution", "model_state_dict",
    "update",
)
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
        os.write(fd, payload)
        os.fsync(fd)
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
    validate(DECISION, DECISION_SHA256)
    decision = json.loads(DECISION.read_text())
    selected = decision.get("candidates", {}).get(decision.get("selected_for_ppo"), {})
    if (
        decision.get("status") != "completed_local_screen"
        or decision.get("selected_for_ppo") != "s16"
        or selected.get("sha256") != SOURCE_SHA256
        or selected.get("gate_pass") is not True
    ):
        raise RuntimeError("decision does not authorize this exact PPO bootstrap")
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if source.get("evaluation_only") is not True or source.get("resume_forbidden") is not True:
        raise RuntimeError("source is not the expected evaluation-only endpoint")
    if any(key not in source for key in REQUIRED_KEYS) or int(source["update"]) != 476:
        raise RuntimeError("source checkpoint contract mismatch")
    source_hash = state_sha(source["model_state_dict"])
    payload = {key: copy.deepcopy(source[key]) for key in REQUIRED_KEYS}
    payload["config"] = copy.deepcopy(payload["config"])
    payload["config"]["trainable_scope"] = "heads"
    payload["config"]["opponent_quota_mode"] = "legacy"
    payload["config"].pop("opponent_base_quotas", None)
    payload["config"].pop("opponent_caps", None)
    payload["config"].pop("opponent_audit", None)
    created_at = datetime.now(timezone.utc).isoformat()
    payload["fresh_training_bootstrap"] = {
        "schema_version": "ptcg-design202608197-s16-fresh-training-bootstrap-v1",
        "created_at_utc": created_at,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256,
        "source_model_state_sha256": source_hash,
        "selection_decision": str(DECISION.relative_to(ROOT)),
        "selection_decision_sha256": DECISION_SHA256,
        "model_weights_bit_exact_source": True,
        "optimizer_state_source": "absent_fresh_required",
        "quota_state_source": "absent_fresh_required",
        "checkpoint_quota_mode": "legacy",
    }
    if any(key in payload for key in FORBIDDEN_KEYS):
        raise RuntimeError("training bootstrap retained forbidden state")
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    check = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if state_sha(check["model_state_dict"]) != source_hash:
        raise RuntimeError("serialized bootstrap changed model weights")
    if check["config"].get("opponent_quota_mode") != "legacy" or any(key in check for key in FORBIDDEN_KEYS):
        raise RuntimeError("serialized bootstrap contract failed")
    os.mkdir(OUTPUT_ROOT, mode=0o700)
    publish(OUTPUT, raw)
    manifest = {
        "schema_version": "ptcg-design202608197-s16-fresh-training-bootstrap-manifest-v1",
        "created_at_utc": created_at,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256,
        "selection_decision": str(DECISION.relative_to(ROOT)),
        "selection_decision_sha256": DECISION_SHA256,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "source_model_state_sha256": source_hash,
        "output_model_state_sha256": state_sha(check["model_state_dict"]),
        "model_weights_bit_exact_source": True,
        "checkpoint_quota_mode": "legacy",
        "optimizer_states_present": False,
        "quota_state_present": False,
        "fresh_optimizer_required": True,
    }
    publish(MANIFEST, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
