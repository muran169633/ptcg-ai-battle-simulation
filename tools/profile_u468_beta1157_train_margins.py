#!/usr/bin/env python3
"""Profile train-only decision margins for the frozen U468 beta=1.157 base.

This diagnostic never opens an archive member outside ``train/``.  It records
stable row references and model-derived margins so a later special-BC design
can be frozen without consulting validation labels or gradients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import orjson
import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_u468_beta1157_flg_actorhead_freshlr225e7_sweep as prior  # noqa: E402
import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-beta1157-train-margin-profile-v1"
DATASETS = {
    "flg": ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
    "pokemonfan": ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
    "core5": ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
}
EXPECTED_DATA_SHA256 = {
    "flg": "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    "pokemonfan": "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    "core5": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def publish_o_excl(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
        observed = os.fstat(fd)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_size != len(payload)
        ):
            raise RuntimeError("unsafe or incomplete output publication")
    finally:
        os.close(fd)


def load_beta_model(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoints = {
        "new_parent": prior.load_checkpoint(prior.U468),
        "new_exact": prior.load_checkpoint(prior.CURRENT_EXACT_P12),
        "old_parent": prior.load_checkpoint(prior.HISTORICAL_PARENT),
        "old_p12": prior.load_checkpoint(prior.HISTORICAL_P12),
    }
    state = prior.construct_beta_state(
        checkpoints["new_parent"]["model_state_dict"],
        checkpoints["new_exact"]["model_state_dict"],
        checkpoints["old_parent"]["model_state_dict"],
        checkpoints["old_p12"]["model_state_dict"],
    )
    general = torch.load(prior.GENERAL_BC, map_location="cpu", weights_only=False)
    model = ppo.instantiate_model_from_checkpoint(
        checkpoints["new_parent"], general, device
    )
    model.load_state_dict(state)
    model.eval()
    return model, checkpoints["new_parent"]["model_config"]


def ordered_margin(
    row_logits: torch.Tensor,
    option_mask: torch.Tensor,
    expert_order: list[int],
) -> float:
    remaining = option_mask.clone()
    margins: list[float] = []
    for chosen in expert_order:
        allowed = remaining.nonzero(as_tuple=False).squeeze(1)
        competitors = allowed[allowed != chosen]
        if competitors.numel() == 0:
            margins.append(float("inf"))
        else:
            margins.append(
                float(row_logits[chosen] - row_logits[competitors].max())
            )
        remaining[chosen] = False
    return min(margins, default=float("inf"))


def count_margin(
    row_logits: torch.Tensor,
    allowed: torch.Tensor,
    expert_count: int,
) -> float:
    competitors = allowed.nonzero(as_tuple=False).squeeze(1)
    competitors = competitors[competitors != expert_count]
    if competitors.numel() == 0:
        return float("inf")
    return float(row_logits[expert_count] - row_logits[competitors].max())


def process_batch(
    model: torch.nn.Module,
    model_config: dict[str, Any],
    device: torch.device,
    rows: list[dict[str, Any]],
    identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    batch = bc.collate_decisions(
        rows,
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    gpu_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    with torch.no_grad():
        outputs = ppo.model_forward(model, gpu_batch, device)
        predictions, log_prob, _, _ = ppo.sample_ordered_actions(
            outputs,
            gpu_batch,
            deterministic=True,
            canonicalize_order=False,
        )
        allowed_counts = ppo.count_allowed_mask(gpu_batch)
    policy_logits = outputs["policy_logits"].float().cpu()
    count_logits = outputs["count_logits"].float().cpu()
    option_mask = batch["option_mask"].bool()
    targets = batch["targets"].bool()
    min_counts = batch["min_counts"]
    max_counts = batch["max_counts"]
    contexts = batch["contexts"]
    allowed_counts = allowed_counts.cpu()
    records: list[dict[str, Any]] = []
    for index, (feature, identity) in enumerate(zip(rows, identities)):
        expert = [int(value) for value in feature["action_sequence"]]
        predicted = [int(value) for value in predictions[index]]
        selection = ordered_margin(
            policy_logits[index], option_mask[index], expert
        )
        flexible = int(min_counts[index]) != int(max_counts[index])
        cardinality = (
            count_margin(count_logits[index], allowed_counts[index], len(expert))
            if flexible
            else float("inf")
        )
        decision = min(selection, cardinality)
        expert_set = {int(value) for value in targets[index].nonzero().flatten()}
        records.append(
            {
                **identity,
                "context": int(contexts[index]),
                "min_count": int(min_counts[index]),
                "max_count": int(max_counts[index]),
                "expert_order": expert,
                "predicted_order": predicted,
                "set_correct": set(predicted) == expert_set,
                "ordered_correct": predicted == expert,
                "selection_margin": selection,
                "count_margin": cardinality,
                "decision_margin": decision,
                "predicted_log_probability": float(log_prob[index].cpu()),
            }
        )
    return records


def profile_archive(
    label: str,
    path: Path,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    keep: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    contexts: Counter[int] = Counter()
    opened_members: list[str] = []
    pending_features: list[dict[str, Any]] = []
    pending_identities: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if not members or any(not name.startswith("train/") for name in members):
            raise RuntimeError(f"{label}: invalid train member set")
        for member in members:
            opened_members.append(member)
            with archive.open(member) as handle:
                for line_index, raw in enumerate(handle):
                    row = orjson.loads(raw)
                    if str(row.get("split", "")) != "train":
                        raise RuntimeError(f"{label}: non-train row in {member}")
                    previous_max = bc.MAX_ACTION_COUNT
                    bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
                    try:
                        feature = bc.featurize_row(
                            row,
                            int(model_config["hash_size"]),
                            int(model_config["max_state_entities"]),
                        )
                    finally:
                        bc.MAX_ACTION_COUNT = previous_max
                    if feature is None:
                        continue
                    action = row.get("action", [])
                    if not isinstance(action, list):
                        continue
                    expert = [int(value) for value in action]
                    feature["action_sequence"] = expert
                    pending_features.append(feature)
                    pending_identities.append(
                        {
                            "member": member,
                            "line_index": line_index,
                            "line_sha256": hashlib.sha256(raw).hexdigest(),
                            "episode_id": str(row.get("episode_id", "")),
                            "observation_step_index": int(
                                row.get("observation_step_index", -1)
                            ),
                            "team_name": str(row.get("team_name", "")),
                        }
                    )
                    if len(pending_features) == batch_size:
                        records.extend(
                            process_batch(
                                model,
                                model_config,
                                device,
                                pending_features,
                                pending_identities,
                            )
                        )
                        pending_features = []
                        pending_identities = []
        if pending_features:
            records.extend(
                process_batch(
                    model,
                    model_config,
                    device,
                    pending_features,
                    pending_identities,
                )
            )
    for record in records:
        contexts[record["context"]] += 1
    wrong = [record for record in records if not record["ordered_correct"]]
    correct = [record for record in records if record["ordered_correct"]]
    near_wrong = sorted(
        wrong,
        key=lambda item: (-item["decision_margin"], item["line_sha256"]),
    )[:keep]
    fragile_correct = sorted(
        correct,
        key=lambda item: (item["decision_margin"], item["line_sha256"]),
    )[:keep]
    return {
        "archive": str(path.relative_to(ROOT)),
        "archive_sha256": sha256_file(path),
        "opened_members": opened_members,
        "non_train_members_opened": False,
        "rows": len(records),
        "ordered_correct": len(correct),
        "ordered_wrong": len(wrong),
        "set_correct": sum(int(record["set_correct"]) for record in records),
        "contexts": {str(key): value for key, value in sorted(contexts.items())},
        "near_wrong": near_wrong,
        "fragile_correct": fragile_correct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--keep", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    if args.batch_size < 1 or args.keep < 1:
        raise ValueError("batch-size and keep must be positive")
    output = args.output.resolve()
    if output.parent != ROOT / "artifacts":
        raise ValueError("output must be a direct child of artifacts/")
    for label, path in DATASETS.items():
        observed = sha256_file(path)
        if observed != EXPECTED_DATA_SHA256[label]:
            raise ValueError(f"{label} archive hash drift")
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model, model_config = load_beta_model(device)
    profiles = {
        label: profile_archive(
            label,
            path,
            model,
            model_config,
            device,
            args.batch_size,
            args.keep,
        )
        for label, path in DATASETS.items()
    }
    result = {
        "schema_version": SCHEMA,
        "status": "completed_train_only",
        "base_model_state_sha256": prior.BASE_MODEL_SHA256,
        "split": "train",
        "validation_opened": False,
        "selection": {
            "near_wrong": "ordered-wrong rows by descending decision margin, then line SHA",
            "fragile_correct": "ordered-correct rows by ascending decision margin, then line SHA",
            "decision_margin": "minimum expert-vs-best-alternative margin over ordered selection and flexible count",
            "keep_per_list": args.keep,
        },
        "inputs": {
            label: {
                "path": str(path.relative_to(ROOT)),
                "sha256": EXPECTED_DATA_SHA256[label],
            }
            for label, path in DATASETS.items()
        },
        "profiles": profiles,
    }
    publish_o_excl(output, canonical_json(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output.relative_to(ROOT)),
                "sha256": sha256_file(output),
                "rows": {label: value["rows"] for label, value in profiles.items()},
                "ordered_wrong": {
                    label: value["ordered_wrong"] for label, value in profiles.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
