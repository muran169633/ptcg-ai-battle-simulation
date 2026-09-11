#!/usr/bin/env python3
"""Build evaluation-only historical-P12 projections on the raw U471 endpoint."""

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
OLD_PARENT = ROOT / (
    "artifacts/ppo_u456inc_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_mb384_localtransport_u456_to_u464_seed202607336/"
    "B_gold_league/seed-202607336/checkpoints/update-0464.pt"
)
OLD_P12 = ROOT / (
    "artifacts/ppo_u464mb384_actorheadonly_pokemonfan_sweep_p4_p8_p12_p16_"
    "design202608060/sweep_stage/"
    "special-bc-actorheadonly-pokemonfan-prefix-0012.pt"
)
NEW_PARENT = ROOT / (
    "artifacts/ppo_u468raw_exactgold12_tailfocus_heads3x96_u471_design202608155/"
    "B_gold_league/seed-202608155/checkpoints/update-0471.pt"
)
OUTPUT_ROOT = ROOT / "artifacts/design202608155_u471_histp12_endpoints"
INPUT_SHA256 = {
    OLD_PARENT: "fe51f40f37fca329cd6b0c94f7431001bb909b92624df33fcd7cf6f0da976264",
    OLD_P12: "8aafef92f0898f46cfe45f076f6fec5bfc383d2429f2d6087805cb976968c3fd",
    NEW_PARENT: "31f0adbbcf68e5124aeae3dbb603e8c3f97e7fe58d9b7477cc42c79fcbf89503",
}
POINTER_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
COUNT_NAMES = (
    "count_head.0.weight",
    "count_head.0.bias",
    "count_head.2.weight",
    "count_head.2.bias",
)
REQUIRED_SLIM_KEYS = (
    "feature_version", "bc_feature_version", "config", "model_config",
    "learner_deck_hash", "reward", "action_distribution",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes() + b"\0")
    return digest.hexdigest()


def regular_file(path: Path) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"input is not a regular file: {path}")


def changed_names(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> list[str]:
    return sorted(name for name in left if not torch.equal(left[name], right[name]))


def publish(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("builder requires repository cwd and my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("builder requires Python -I -B")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"refusing existing output: {OUTPUT_ROOT}")
    for path, expected in INPUT_SHA256.items():
        regular_file(path)
        if sha256_file(path) != expected:
            raise RuntimeError(f"input SHA-256 mismatch: {path}")

    checkpoints = {
        label: torch.load(path, map_location="cpu", weights_only=False)
        for label, path in (
            ("old_parent", OLD_PARENT), ("old_p12", OLD_P12),
            ("new_parent", NEW_PARENT),
        )
    }
    old_parent = checkpoints["old_parent"]["model_state_dict"]
    old_p12 = checkpoints["old_p12"]["model_state_dict"]
    new_parent = checkpoints["new_parent"]["model_state_dict"]
    if set(old_parent) != set(old_p12) or set(old_parent) != set(new_parent):
        raise RuntimeError("model-state keys differ")
    expected_delta_names = sorted(POINTER_NAMES + COUNT_NAMES)
    if changed_names(old_parent, old_p12) != expected_delta_names:
        raise RuntimeError("historical P12 delta does not change exactly mutable10")
    if checkpoints["new_parent"].get("update") != 471:
        raise RuntimeError("new parent is not U471")
    for state in (old_parent, old_p12, new_parent):
        if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
            raise RuntimeError("non-finite model state")

    endpoints: list[tuple[str, tuple[str, ...]]] = [
        ("histp12-pointer", POINTER_NAMES),
        ("histp12-full", POINTER_NAMES + COUNT_NAMES),
    ]
    os.mkdir(OUTPUT_ROOT, mode=0o700)
    created_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": "ptcg-design202608155-u471-histp12-endpoints-v1",
        "created_at_utc": created_at,
        "formula": "W_U471_raw + (W_histP12_U464 - W_U464)",
        "inputs": {
            path.name: {"path": str(path.relative_to(ROOT)), "sha256": digest}
            for path, digest in INPUT_SHA256.items()
        },
        "endpoints": [],
        "scope": {"evaluation_only": True, "resume_forbidden": True},
    }
    for endpoint, mutable_names in endpoints:
        state = {name: tensor.detach().cpu().clone() for name, tensor in new_parent.items()}
        for name in mutable_names:
            transported = (
                new_parent[name].detach().cpu().double()
                + old_p12[name].detach().cpu().double()
                - old_parent[name].detach().cpu().double()
            ).to(dtype=new_parent[name].dtype)
            if not bool(torch.isfinite(transported).all()):
                raise RuntimeError(f"non-finite transported tensor: {name}")
            state[name] = transported
        if changed_names(new_parent, state) != sorted(mutable_names):
            raise RuntimeError(f"{endpoint} changed unexpected tensors")
        state_hash = model_state_sha256(state)
        slim = {
            key: copy.deepcopy(checkpoints["new_parent"][key])
            for key in REQUIRED_SLIM_KEYS
        }
        slim.update({
            "model_state_dict": state,
            "update": 471,
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_states_omitted": [
                "optimizer_state_dict", "bc_replay_optimizer_state_dict",
                "opponent_quota_state",
            ],
            "actor_delta_transport": {
                "schema_version": "ptcg-design202608155-u471-histp12-endpoint-v1",
                "created_at_utc": created_at,
                "formula": "W_U471_raw + (W_histP12_U464 - W_U464)",
                "endpoint": endpoint,
                "mutable_parameter_names": list(mutable_names),
                "model_state_sha256": state_hash,
                "resume_forbidden": True,
            },
        })
        buffer = io.BytesIO()
        torch.save(slim, buffer)
        payload = buffer.getvalue()
        reloaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if reloaded.get("resume_forbidden") is not True or model_state_sha256(reloaded["model_state_dict"]) != state_hash:
            raise RuntimeError(f"serialized endpoint failed verification: {endpoint}")
        filename = f"u471-{endpoint}.pt"
        publish(OUTPUT_ROOT / filename, payload)
        manifest["endpoints"].append({
            "endpoint": endpoint,
            "path": str((OUTPUT_ROOT / filename).relative_to(ROOT)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "model_state_sha256": state_hash,
            "mutable_parameter_names": list(mutable_names),
        })
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    publish(OUTPUT_ROOT / "manifest.json", manifest_payload)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
