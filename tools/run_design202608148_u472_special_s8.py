#!/usr/bin/env python3
"""Execute the single frozen actor6 S8 endpoint for design 202608148."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import orjson
import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import profile_u468_beta1157_train_margins as margin  # noqa: E402
import run_ppo_bc_repair as repair  # noqa: E402
import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-design202608148-u472-special-s8-execution-v1"
SEED = 202608148
PARENT = ROOT / (
    "artifacts/ppo_guardedu468_updatedreplay_g8_ppo4x96_u472_"
    "freshjointactor6_s8_design202608148/ppo_stage/B_gold_league/"
    "seed-202608148/checkpoints/update-0472.pt"
)
PARENT_SHA = "c0436d54fff5e4da62d94c3002775c9e4b9ee848c5c59dd4422943cd71ad7018"
PARENT_MODEL_SHA = "6a1c67217fd53666e89d658f3cb11422e64f0bdfe7f7a77c4a4672ad4cb584a2"
PROFILE = ROOT / "artifacts/design202608148_u472_special_s8_profile.json"
PROFILE_SHA = "e6bd92835b1c6b571a4e0dc2263ca16bcc0bf0713e30f3e4ada46adbec6c0a48"
PROFILE_PREREG = ROOT / "artifacts/design202608148_u472_special_s8_profile_preregistration.json"
PROFILE_PREREG_SHA = "0eeeb6d4c8b21e3a2d21ce7f73877523494255edc7689b2c4c42783dc7b1b953"
INTEGRITY = ROOT / "artifacts/design202608148_fresh_ppo_u469_u472_training_integrity_decision.json"
INTEGRITY_SHA = "d1aabab031876b71a7e42d30c7b024f603a3d6a169d30d29b1725d5b172c9a44"
GENERAL_BC = ROOT / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/best.pt"
GENERAL_BC_SHA = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
GENERAL = ROOT / "data/bc_marnie_top50_current14_timeforward_train0802_valid0803_design202608147.zip"
GENERAL_SHA = "bf01cfc7e9c0f616f157ae36ed68c76619ae45448cdc17562befe889a4d5ae93"
SPECIAL = dict(margin.DATASETS)
SPECIAL_SHA = dict(margin.EXPECTED_DATA_SHA256)

DEPENDENCIES = {
    TOOLS / "profile_u468_beta1157_train_margins.py": "f1018994c48ffadcbc5277f0dfd9233086693f90bf30f5bf67462965b4e18142",
    TOOLS / "train_ppo.py": "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3",
    TOOLS / "train_bc_orbit.py": "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
    TOOLS / "run_ppo_bc_repair.py": "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966",
}
ACTOR6 = (
    "actor_query.weight", "actor_key.weight", "actor_residual.0.weight",
    "actor_residual.0.bias", "actor_residual.2.weight", "actor_residual.2.bias",
)
LR = 1e-6
STEPS = 8
BATCH_SIZE = 256
L2_MAX = 0.001


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path, expected: str) -> dict[str, Any]:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"unsafe input {path}")
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f"hash drift for {path}: {actual}")
    return {"path": str(path.relative_to(ROOT)), "sha256": actual, "bytes": info.st_size}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        done = 0
        while done < len(payload):
            done += os.write(fd, payload[done:])
        os.fsync(fd)
    finally:
        os.close(fd)


def feature_from_raw(raw: bytes, model_config: dict[str, Any]) -> dict[str, Any]:
    row = orjson.loads(raw)
    if row.get("split") != "train":
        raise RuntimeError("non-train selected row")
    previous = bc.MAX_ACTION_COUNT
    bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
    try:
        feature = bc.featurize_row(row, int(model_config["hash_size"]), int(model_config["max_state_entities"]))
    finally:
        bc.MAX_ACTION_COUNT = previous
    if feature is None or not isinstance(row.get("action"), list):
        raise RuntimeError("selected row no longer featurizes")
    feature["action_sequence"] = [int(v) for v in row["action"]]
    return feature


def load_special_features(profile: dict[str, Any], model_config: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    wanted: dict[str, dict[str, set[int]]] = {}
    expected_hash: dict[tuple[str, str, int], str] = {}
    for batch in profile["batches"]:
        for row in batch["rows"]:
            source = row["source"]
            if source == "general":
                continue
            member = row["ref"]["member"]
            index = int(row["ref"]["line_index"])
            wanted.setdefault(source, {}).setdefault(member, set()).add(index)
            expected_hash[(source, member, index)] = row["line_sha256"]
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for source, members in wanted.items():
        with zipfile.ZipFile(SPECIAL[source]) as archive:
            for member, indices in members.items():
                if not member.startswith("train/"):
                    raise RuntimeError("profile selected a non-train member")
                with archive.open(member) as stream:
                    for index, raw in enumerate(stream):
                        if index not in indices:
                            continue
                        key = (source, member, index)
                        if hashlib.sha256(raw).hexdigest() != expected_hash[key]:
                            raise RuntimeError("selected line hash drift")
                        result[key] = feature_from_raw(raw, model_config)
    if len(result) != len(expected_hash):
        raise RuntimeError("not all selected special rows were recovered")
    return result


def make_batches(
    profile: dict[str, Any], general_batches: list[dict[str, torch.Tensor]],
    features: dict[tuple[str, str, int], dict[str, Any]], model_config: dict[str, Any]
) -> list[dict[str, torch.Tensor]]:
    output = []
    for manifest in profile["batches"]:
        special_rows = [r for r in manifest["rows"] if r["source"] != "general"]
        feature_rows = [features[(r["source"], r["ref"]["member"], int(r["ref"]["line_index"]))] for r in special_rows]
        special_batch = bc.collate_decisions(
            feature_rows,
            max_state_entities=int(model_config["max_state_entities"]),
            entity_fields=int(model_config["entity_fields"]),
            option_fields=int(model_config["option_fields"]),
        )
        special_cursor = 0
        refs = []
        for row in manifest["rows"]:
            if row["source"] == "general":
                refs.append((general_batches[int(row["ref"]["cache_batch"])], int(row["ref"]["cache_row"])))
            else:
                refs.append((special_batch, special_cursor))
                special_cursor += 1
        batch = ppo.collate_cached_replay_rows(refs)
        if int(batch["action_counts"].shape[0]) != BATCH_SIZE or int((batch["contexts"] == 34).sum()) != 4:
            raise RuntimeError("materialized S8 batch quota drift")
        output.append(batch)
    return output


def predict(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], profile: dict[str, Any], device: torch.device) -> dict[str, Any]:
    aggregate: dict[str, dict[str, int]] = {}
    retention_flips = 0
    count_value = []
    cursor = 0
    for cpu, manifest in zip(batches, profile["batches"], strict=True):
        gpu = {k: v.to(device, non_blocking=True) for k, v in cpu.items()}
        with torch.no_grad():
            out = ppo.model_forward(model, gpu, device)
            predictions, _, _, _ = ppo.sample_ordered_actions(out, gpu, deterministic=True, canonicalize_order=False)
        count_value.append((out["count_logits"].detach().cpu().clone(), out["value_logits"].detach().cpu().clone()))
        for i, row in enumerate(manifest["rows"]):
            n = int(cpu["action_counts"][i]); expert = [int(v) for v in cpu["action_sequences"][i, :n]]
            pred = [int(v) for v in predictions[i]]
            key = f"{row['source']}:{row['role']}"
            a = aggregate.setdefault(key, {"rows": 0, "ordered_correct": 0, "set_correct": 0, "count_correct": 0})
            a["rows"] += 1; a["ordered_correct"] += int(pred == expert)
            a["set_correct"] += int(set(pred) == set(expert)); a["count_correct"] += int(len(pred) == len(expert))
            if row["role"] == "retention" and row["parent_ordered_correct"] and pred != expert:
                retention_flips += 1
            cursor += 1
    return {"selected": aggregate, "retention_parent_correct_to_wrong": retention_flips, "count_value": count_value, "rows": cursor}


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != PYTHON.resolve() or sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires repository root and my_project_env Python -I -B")
    inputs = {
        "parent": require(PARENT, PARENT_SHA), "profile": require(PROFILE, PROFILE_SHA),
        "profile_preregistration": require(PROFILE_PREREG, PROFILE_PREREG_SHA),
        "ppo_integrity": require(INTEGRITY, INTEGRITY_SHA), "general_bc": require(GENERAL_BC, GENERAL_BC_SHA),
        "general_archive": require(GENERAL, GENERAL_SHA),
        "special_archives": {k: require(v, SPECIAL_SHA[k]) for k, v in SPECIAL.items()},
        "dependencies": {str(k.relative_to(ROOT)): require(k, v) for k, v in DEPENDENCIES.items()},
    }
    profile = json.loads(PROFILE.read_text())
    if profile.get("combined_manifest_sha256") != "e9b644443e3084072beb99e42e2736ae8e03b8e75d900f2c8ae5566962032773" or not all(profile["checks"].values()):
        raise RuntimeError("profile manifest gate failed")
    if args.mode == "audit":
        if args.output_dir is not None:
            raise RuntimeError("audit forbids output")
        print(json.dumps({"status": "audit_passed", "inputs": inputs, "optimizer_steps": 0, "checkpoint_writes": 0}, sort_keys=True))
        return 0
    if args.output_dir is None:
        raise RuntimeError("formal requires output-dir")
    output = args.output_dir.resolve()
    if output.exists() or output.parent != ROOT / "artifacts/ppo_guardedu468_updatedreplay_g8_ppo4x96_u472_freshjointactor6_s8_design202608148":
        raise RuntimeError("formal output must be absent special_stage directory")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    output.mkdir(mode=0o700)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    general_bc = torch.load(GENERAL_BC, map_location="cpu", weights_only=False)
    model = ppo.instantiate_model_from_checkpoint(parent, general_bc, device)
    if ppo.model_state_sha256(model) != PARENT_MODEL_SHA:
        raise RuntimeError("parent model hash drift")
    config = ppo.PPOConfig(**parent["config"])
    config.seed = SEED; config.bc_replay_data = str(GENERAL); config.bc_replay_split = "train"
    config.bc_replay_batches = 72; config.bc_replay_batch_size = 256; config.bc_replay_workers = 8
    config.bc_replay_steps = 1; config.bc_replay_context34_rows_per_batch = 4
    config.bc_replay_loss = "ordered"; config.bc_replay_order_context_weight = 8.0
    config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0; config.bc_replay_lr_scale = 1.0
    general_batches = ppo.build_bc_replay_batches(config, parent["model_config"])
    cache_sha, _ = repair.replay_cache_manifest(general_batches)
    if cache_sha != profile["general_cache"]["aggregate_sha256"]:
        raise RuntimeError("general cache does not reproduce profile")
    features = load_special_features(profile, parent["model_config"])
    batches = make_batches(profile, general_batches, features, parent["model_config"])
    model.eval(); before_state = clone_state(model); before = predict(model, batches, profile, device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    named = dict(model.named_parameters())
    parameters = []
    for name in ACTOR6:
        named[name].requires_grad_(True); parameters.append(named[name])
    optimizer = torch.optim.AdamW(parameters, lr=LR, eps=1e-5, weight_decay=1e-4)
    per_step = []
    for step, batch in enumerate(batches, 1):
        metrics = ppo.bc_replay_update(model, optimizer, [batch], config, device, LR)
        if metrics is None or metrics["steps"] != 1 or metrics["rows"] != 256 or metrics["context_34_rows"] != 4:
            raise RuntimeError("special BC core metric gate failed")
        per_step.append({"step": step, "batch_rows_sha256": profile["batches"][step - 1]["rows_sha256"], "metrics": metrics})
    after_state = clone_state(model); model.eval(); after = predict(model, batches, profile, device)
    changed = repair.changed_tensor_names(before_state, after_state)
    if set(changed) != set(ACTOR6):
        raise RuntimeError(f"changed tensors are not exact actor6: {changed}")
    l2 = math.sqrt(sum(float((after_state[n].double() - before_state[n].double()).square().sum()) for n in ACTOR6))
    if not (0.0 < l2 <= L2_MAX):
        raise RuntimeError(f"actor6 displacement outside gate: {l2}")
    count_value_identical = all(torch.equal(a, b) and torch.equal(c, d) for (a, c), (b, d) in zip(before["count_value"], after["count_value"], strict=True))
    if not count_value_identical:
        raise RuntimeError("count or value logits changed")
    if after["retention_parent_correct_to_wrong"] != 0:
        raise RuntimeError("selected retention regression gate failed")
    deltas = {}
    for key in before["selected"]:
        deltas[key] = {metric: after["selected"][key][metric] - before["selected"][key][metric] for metric in ("ordered_correct", "set_correct", "count_correct")}
    required = {
        "pokemonfan:target": {"ordered_correct": 5, "set_correct": 3, "count_correct": 0},
        "flg:target": {"ordered_correct": 0, "set_correct": 0, "count_correct": 0},
        "core5:target": {"ordered_correct": 0, "set_correct": 0, "count_correct": 0},
        "general:retention": {"ordered_correct": 0, "set_correct": 0, "count_correct": 0},
    }
    gate = all(deltas[k][m] >= floor for k, rules in required.items() for m, floor in rules.items())
    if not gate:
        raise RuntimeError(f"selected train-only acceptance gate failed: {deltas}")
    steps = sorted({int(v["step"].item() if isinstance(v["step"], torch.Tensor) else v["step"]) for v in optimizer.state.values()})
    if steps != [8] or len(optimizer.state) != 6:
        raise RuntimeError("fresh actor6 optimizer state gate failed")
    provenance = {
        "schema_version": SCHEMA, "status": "special_s8_completed_train_only_gates_passed",
        "seed": SEED, "parent": inputs["parent"], "profile": inputs["profile"],
        "optimizer": {"type": "fresh_AdamW", "steps": 8, "state_count": 6, "learning_rate": LR, "eps": 1e-5, "weight_decay": 1e-4},
        "integrity": {"changed_parameter_names": changed, "changed_exactly_actor6": True, "actor6_l2_displacement": l2, "nonactor_tensors_bit_identical": True, "count_and_value_logits_bit_identical_on_selected_rows": True, "rows": 2048, "context34_rows": 32, "model_finite": all(bool(torch.isfinite(v).all()) for v in after_state.values())},
        "train_only_selected_metrics_before": before["selected"], "train_only_selected_metrics_after": after["selected"], "train_only_selected_net_delta": deltas,
        "retention_parent_correct_to_candidate_wrong": after["retention_parent_correct_to_wrong"], "per_step": per_step,
        "scope": {"validation_or_test": False, "evaluation_games": 0, "network": False, "package": False, "upload": False, "submission": False},
    }
    payload = copy.deepcopy(parent); payload["model_state_dict"] = after_state
    payload["post_ppo_special_bc"] = provenance; payload["special_bc_optimizer_state_dict"] = optimizer.state_dict()
    checkpoint = output / "special-bc-freshjoint-actor6-s8-0008.pt"
    torch.save(payload, checkpoint)
    provenance["checkpoint"] = {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha(checkpoint), "update": 472}
    write_exclusive(output / "special_bc_manifest.json", canonical(provenance))
    print(json.dumps({"status": provenance["status"], "checkpoint": provenance["checkpoint"], "actor6_l2": l2, "deltas": deltas, "retention_flips": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
