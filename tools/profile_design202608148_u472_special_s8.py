#!/usr/bin/env python3
"""Create the train-only, deterministic S8 row manifest for design 202608148."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import profile_u468_beta1157_train_margins as margin  # noqa: E402
import run_ppo_bc_repair as repair  # noqa: E402
import train_ppo as ppo  # noqa: E402


PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-design202608148-u472-special-s8-profile-v1"
SEED = 202608148
STEPS = 8
BATCH_SIZE = 256
CONTEXT = 34

PARENT = ROOT / (
    "artifacts/ppo_guardedu468_updatedreplay_g8_ppo4x96_u472_"
    "freshjointactor6_s8_design202608148/ppo_stage/B_gold_league/"
    "seed-202608148/checkpoints/update-0472.pt"
)
PARENT_SHA = "c0436d54fff5e4da62d94c3002775c9e4b9ee848c5c59dd4422943cd71ad7018"
PARENT_MODEL_SHA = "6a1c67217fd53666e89d658f3cb11422e64f0bdfe7f7a77c4a4672ad4cb584a2"
GENERAL_BC = ROOT / (
    "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_"
    "20260731/best.pt"
)
GENERAL_BC_SHA = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
GENERAL = ROOT / (
    "data/bc_marnie_top50_current14_timeforward_train0802_valid0803_"
    "design202608147.zip"
)
GENERAL_SHA = "bf01cfc7e9c0f616f157ae36ed68c76619ae45448cdc17562befe889a4d5ae93"
SPECIAL = dict(margin.DATASETS)
SPECIAL_SHA = dict(margin.EXPECTED_DATA_SHA256)

FRAMEWORK = TOOLS / "profile_u468_beta1157_train_margins.py"
FRAMEWORK_SHA = "f1018994c48ffadcbc5277f0dfd9233086693f90bf30f5bf67462965b4e18142"
TRAIN_PPO = TOOLS / "train_ppo.py"
TRAIN_PPO_SHA = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
REPAIR = TOOLS / "run_ppo_bc_repair.py"
REPAIR_SHA = "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"

TARGET_PER_BATCH = {"pokemonfan": 48, "flg": 24, "core5": 24}
RETENTION_PER_BATCH = {"general": 64, "pokemonfan": 32, "flg": 32, "core5": 32}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path, expected: str) -> dict[str, Any]:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"unsafe input: {path}")
    actual = file_sha(path)
    if actual != expected:
        raise RuntimeError(f"hash drift for {path}: {actual}")
    return {"path": str(path.relative_to(ROOT)), "sha256": actual, "bytes": info.st_size}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def row_hash(batch: dict[str, torch.Tensor], index: int) -> str:
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key][index].detach().cpu().contiguous()
        digest.update(key.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape)).encode() + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cached_records(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch_index, cpu in enumerate(batches):
        gpu = {key: value.to(device, non_blocking=True) for key, value in cpu.items()}
        with torch.no_grad():
            outputs = ppo.model_forward(model, gpu, device)
            predictions, _, _, _ = ppo.sample_ordered_actions(
                outputs, gpu, deterministic=True, canonicalize_order=False
            )
            allowed = ppo.count_allowed_mask(gpu).cpu()
        policy = outputs["policy_logits"].float().cpu()
        counts = outputs["count_logits"].float().cpu()
        for row_index in range(int(cpu["contexts"].shape[0])):
            n = int(cpu["action_counts"][row_index])
            expert = [int(v) for v in cpu["action_sequences"][row_index, :n]]
            predicted = [int(v) for v in predictions[row_index]]
            selection = margin.ordered_margin(
                policy[row_index], cpu["option_mask"][row_index].bool(), expert
            )
            flexible = int(cpu["min_counts"][row_index]) != int(cpu["max_counts"][row_index])
            cardinality = (
                margin.count_margin(counts[row_index], allowed[row_index], n)
                if flexible else float("inf")
            )
            records.append({
                "cache_batch": batch_index,
                "cache_row": row_index,
                "line_sha256": row_hash(cpu, row_index),
                "episode_id": f"cache-{batch_index:02d}-row-{row_index:03d}",
                "context": int(cpu["contexts"][row_index]),
                "expert_order": expert,
                "predicted_order": predicted,
                "ordered_correct": predicted == expert,
                "decision_margin": min(selection, cardinality),
            })
    return records


def compact(source: str, role: str, row: dict[str, Any]) -> dict[str, Any]:
    ref = (
        {"cache_batch": row["cache_batch"], "cache_row": row["cache_row"]}
        if source == "general"
        else {"member": row["member"], "line_index": row["line_index"]}
    )
    return {
        "source": source,
        "role": role,
        "ref": ref,
        "line_sha256": row["line_sha256"],
        "episode_id": row["episode_id"],
        "context": row["context"],
        "expert_order": row["expert_order"],
        "parent_ordered_correct": row["ordered_correct"],
        "parent_decision_margin": row["decision_margin"],
    }


def take_for_batches(
    rows: list[dict[str, Any]], count: int, source: str, role: str
) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda r: (
            -float(r["decision_margin"]) if role == "target" else float(r["decision_margin"]),
            r["line_sha256"],
        ),
    )
    used: set[str] = set()
    result: list[list[dict[str, Any]]] = []
    for _ in range(STEPS):
        chosen: list[dict[str, Any]] = []
        episodes: set[str] = set()
        for row in ordered:
            line = str(row["line_sha256"])
            episode = str(row["episode_id"])
            if line in used or episode in episodes:
                continue
            chosen.append(compact(source, role, row))
            used.add(line)
            episodes.add(episode)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise RuntimeError(f"insufficient {source}/{role} capacity: {len(chosen)}/{count}")
        result.append(chosen)
    return result


def retention_for_batches(
    rows: list[dict[str, Any]], count: int, source: str
) -> list[list[dict[str, Any]]]:
    c34 = [r for r in rows if int(r["context"]) == CONTEXT]
    ordinary = [r for r in rows if int(r["context"]) != CONTEXT]
    c = take_for_batches(c34, 1, source, "retention")
    o = take_for_batches(ordinary, count - 1, source, "retention")
    return [c[i] + o[i] for i in range(STEPS)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "formal"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("requires repository root and my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires -I -B")
    inputs = {
        "parent": require(PARENT, PARENT_SHA),
        "general_bc": require(GENERAL_BC, GENERAL_BC_SHA),
        "general_archive": require(GENERAL, GENERAL_SHA),
        "framework": require(FRAMEWORK, FRAMEWORK_SHA),
        "train_ppo": require(TRAIN_PPO, TRAIN_PPO_SHA),
        "repair": require(REPAIR, REPAIR_SHA),
        "special_archives": {k: require(v, SPECIAL_SHA[k]) for k, v in SPECIAL.items()},
    }
    if args.mode == "audit":
        if args.output is not None:
            raise ValueError("audit forbids output")
        print(json.dumps({"status": "audit_passed", "inputs": inputs, "writes": 0}, sort_keys=True))
        return 0
    if args.output is None:
        raise ValueError("formal requires output")
    output = args.output.resolve()
    if output.parent != ROOT / "artifacts" or output.exists():
        raise RuntimeError("formal output must be an absent direct child of artifacts")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    general_bc = torch.load(GENERAL_BC, map_location="cpu", weights_only=False)
    if parent.get("update") != 472:
        raise RuntimeError("parent is not U472")
    model = ppo.instantiate_model_from_checkpoint(parent, general_bc, device)
    model.eval()
    if ppo.model_state_sha256(model) != PARENT_MODEL_SHA:
        raise RuntimeError("parent runtime model hash drift")
    special_profiles = {
        label: margin.profile_archive(label, path, model, parent["model_config"], device, 256, 8192)
        for label, path in SPECIAL.items()
    }
    config = ppo.PPOConfig(**parent["config"])
    config.seed = SEED
    config.bc_replay_data = str(GENERAL)
    config.bc_replay_split = "train"
    config.bc_replay_batches = 72
    config.bc_replay_batch_size = 256
    config.bc_replay_workers = 8
    config.bc_replay_steps = 1
    config.bc_replay_context34_rows_per_batch = 4
    general_batches = ppo.build_bc_replay_batches(config, parent["model_config"])
    cache_sha, cache_batch_sha = repair.replay_cache_manifest(general_batches)
    if cache_sha != "896521dfa6ed5a772796f758182019206dd1f34399e2374aef2080ca9b563917":
        raise RuntimeError(f"general cache drift: {cache_sha}")
    general_records = cached_records(model, general_batches, device)

    selected: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    for source, count in TARGET_PER_BATCH.items():
        rows = [r for r in special_profiles[source]["near_wrong"] if r["expert_order"] and not r["ordered_correct"] and int(r["context"]) != CONTEXT]
        selected[(source, "target")] = take_for_batches(rows, count, source, "target")
    for source, count in RETENTION_PER_BATCH.items():
        rows = (
            [r for r in general_records if r["expert_order"] and r["ordered_correct"]]
            if source == "general"
            else [r for r in special_profiles[source]["fragile_correct"] if r["expert_order"] and r["ordered_correct"]]
        )
        selected[(source, "retention")] = retention_for_batches(rows, count, source)

    batches: list[dict[str, Any]] = []
    for step in range(STEPS):
        rows: list[dict[str, Any]] = []
        for source in ("pokemonfan", "flg", "core5"):
            rows.extend(selected[(source, "target")][step])
        for source in ("general", "pokemonfan", "flg", "core5"):
            rows.extend(selected[(source, "retention")][step])
        if len(rows) != BATCH_SIZE or sum(r["context"] == CONTEXT for r in rows) != 4:
            raise RuntimeError("batch quota construction failed")
        digest = hashlib.sha256(canonical(rows)).hexdigest()
        batches.append({"step": step + 1, "rows": rows, "rows_sha256": digest})
    all_lines = [r["line_sha256"] for b in batches for r in b["rows"]]
    if len(all_lines) != len(set(all_lines)):
        raise RuntimeError("cross-batch row reuse")
    combined = hashlib.sha256()
    for batch in batches:
        combined.update(bytes.fromhex(batch["rows_sha256"]))
    result = {
        "schema_version": SCHEMA,
        "status": "completed_train_only_profile_and_s8_manifest",
        "scope": {"validation_or_test_rows_opened": False, "optimizer_steps": 0, "changed_weights": 0, "network": False, "submission": False},
        "base": {"path": str(PARENT.relative_to(ROOT)), "file_sha256": PARENT_SHA, "runtime_model_state_sha256": PARENT_MODEL_SHA, "update": 472},
        "inputs": inputs,
        "general_cache": {"aggregate_sha256": cache_sha, "batch_sha256": cache_batch_sha, "rows": 18432, "batches": 72, "context34_rows_per_batch": 4},
        "panel_baseline": {
            "general": {"rows_profiled": len(general_records), "ordered_correct": sum(r["ordered_correct"] for r in general_records), "ordered_wrong": sum(not r["ordered_correct"] for r in general_records), "contexts": dict(sorted(Counter(str(r["context"]) for r in general_records).items()))},
            **{label: {k: panel[k] for k in ("rows", "ordered_correct", "ordered_wrong", "set_correct", "contexts")} for label, panel in special_profiles.items()},
        },
        "selection_contract": {"steps": 8, "batch_size": 256, "target_per_batch": TARGET_PER_BATCH, "retention_per_batch": RETENTION_PER_BATCH, "context34_exact_per_source_per_batch": 1, "cross_batch_line_reuse": False, "special_role_episode_unique_within_batch": True, "general_cache_identity": "exact cached tensor row; duplicate line hashes forbidden"},
        "batches": batches,
        "combined_manifest_sha256": combined.hexdigest(),
        "checks": {"eight_batches": len(batches) == 8, "rows_total": len(all_lines) == 2048, "unique_rows_total": len(set(all_lines)) == 2048, "context34_rows_total": sum(r["context"] == CONTEXT for b in batches for r in b["rows"]) == 32},
    }
    payload = canonical(result)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"status": result["status"], "output": str(output.relative_to(ROOT)), "sha256": file_sha(output), "combined_manifest_sha256": result["combined_manifest_sha256"], "checks": result["checks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
