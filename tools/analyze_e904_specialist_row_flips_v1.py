#!/usr/bin/env python3
"""Read-only raw-U468 versus E904 row-flip analysis on frozen valid panels.

This diagnostic reproduces the specialist evaluator's streamed B256/8-worker
inference path while retaining stable archive row identities and compact logit
geometry.  It never trains or writes a checkpoint.  The optional JSON report
is published once with O_EXCL under artifacts/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import zipfile
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Mapping

import orjson
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate_policy_bc as evaluator  # noqa: E402
import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
RAW = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
CANDIDATE = ROOT / (
    "artifacts/ppo_u468_raw_actor6_metricguard_fulltrain_cw10_materialized_v1_20260802/"
    "u468-cw10-fulltrain-pass-eval-only.pt"
)
RAW_FILE_SHA = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
CANDIDATE_FILE_SHA = "91b64ddf754b149297dd6177038de871fdc69bcea0d86787cff851e4fd79e0dd"
RAW_MODEL_SHA = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
CANDIDATE_MODEL_SHA = "e904efb322c15ca4bee62573f4f439309d1d4ecb3ae62620d13fd70e8acaa91f"

PF = ROOT / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip"
FLG = ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip"
CORE5 = ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip"
ARCHIVE_SHAS = {
    PF: "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    FLG: "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    CORE5: "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}
PANELS = (
    ("pokemonfan", PF, ()),
    ("flg", FLG, ()),
    ("core5", CORE5, ()),
    ("dominic", CORE5, ("Dominic Peel",)),
    ("luca", CORE5, ("Luca",)),
    ("szlach", CORE5, ("szlachetny snieg",)),
)
BASELINES = {
    "pokemonfan": {"rows": 15152, "set": 13017, "hybrid": 13016, "ordered": 12864, "top1": 13116, "count": 15022, "value": 11033},
    "flg": {"rows": 2312, "set": 1761, "hybrid": 1748, "ordered": 1731, "top1": 1775, "count": 2299, "value": 1807},
    "core5": {"rows": 8319, "set": 6550, "hybrid": 6534, "ordered": 6505, "top1": 6653, "count": 8210, "value": 6127},
    "dominic": {"rows": 2623, "set": 1981, "hybrid": 1974, "ordered": 1973, "top1": 2019, "count": 2579, "value": 1940},
    "luca": {"rows": 1378, "set": 1069, "hybrid": 1069, "ordered": 1059, "top1": 1086, "count": 1377, "value": 960},
    "szlach": {"rows": 2567, "set": 2071, "hybrid": 2071, "ordered": 2062, "top1": 2109, "count": 2510, "value": 1906},
}
CANDIDATE_REPORT_ROOT = ROOT / "artifacts/e904_cw10_specialist_20260802_v1.specialist_behavior"
METRIC_KEYS = ("set", "hybrid", "ordered", "top1", "count", "value")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(orjson.dumps(value, option=orjson.OPT_SORT_KEYS)).hexdigest()


def publish_o_excl(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise RuntimeError("unsafe report output")
    finally:
        os.close(fd)


class IdentifiedValidDataset(IterableDataset):
    def __init__(
        self,
        archive_path: Path,
        model_config: Mapping[str, Any],
        team_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.archive_path = archive_path
        self.hash_size = int(model_config["hash_size"])
        self.max_state_entities = int(model_config["max_state_entities"])
        self.team_names = set(team_names)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        with zipfile.ZipFile(self.archive_path) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.startswith("valid/") and name.endswith(".jsonl")
            )
            for member in members[worker_id::worker_count]:
                with archive.open(member) as handle:
                    for line_index, raw in enumerate(handle):
                        row = orjson.loads(raw)
                        if str(row.get("split", "")) != "valid":
                            continue
                        if self.team_names and str(row.get("team_name", "")) not in self.team_names:
                            continue
                        raw_action = row.get("action", [])
                        if not isinstance(raw_action, list):
                            continue
                        try:
                            expert_order = [int(index) for index in raw_action]
                        except (TypeError, ValueError):
                            continue
                        previous_max = bc.MAX_ACTION_COUNT
                        bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
                        try:
                            features = bc.featurize_row(
                                row,
                                self.hash_size,
                                self.max_state_entities,
                            )
                        finally:
                            bc.MAX_ACTION_COUNT = previous_max
                        if features is None:
                            continue
                        options = (((row.get("observation") or {}).get("select") or {}).get("option") or [])
                        if (
                            len(expert_order) > ppo.MAX_ACTION_COUNT
                            or len(set(expert_order)) != len(expert_order)
                            or any(index < 0 or index >= len(options) for index in expert_order)
                        ):
                            continue
                        features["expert_action_order"] = expert_order
                        features["__identity"] = {
                            "archive": str(self.archive_path.relative_to(ROOT)),
                            "member": member,
                            "line_index": line_index,
                            "line_sha256": hashlib.sha256(raw).hexdigest(),
                            "episode_id": str(row.get("episode_id", "")),
                            "observation_step_index": int(row.get("observation_step_index", -1)),
                            "team_name": str(row.get("team_name", "")),
                            "opponent_team_name": str(row.get("opponent_team_name", "")),
                            "observation_sha256": canonical_sha(row.get("observation")),
                            "decision_sha256": canonical_sha({
                                "observation": row.get("observation"),
                                "action": expert_order,
                                "min_count": row.get("min_count"),
                                "max_count": row.get("max_count"),
                            }),
                            "options_sha256": canonical_sha(options),
                            "options": options,
                        }
                        yield features


def collate_identified(
    rows: list[dict[str, Any]],
    *,
    max_state_entities: int,
    entity_fields: int,
    option_fields: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    identities = [row.pop("__identity") for row in rows]
    batch = evaluator.collate_ordered(
        rows,
        max_state_entities=max_state_entities,
        entity_fields=entity_fields,
        option_fields=option_fields,
    )
    return batch, identities


def prediction_values(
    cpu_batch: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    actions: list[list[int]],
    row_index: int,
) -> dict[str, Any]:
    targets = cpu_batch["targets"][row_index].bool()
    option_mask = cpu_batch["option_mask"][row_index].bool()
    context = int(cpu_batch["contexts"][row_index])
    expert_count = int(cpu_batch["expert_ordered_action_counts"][row_index])
    expert = cpu_batch["expert_ordered_actions"][row_index, :expert_count].tolist()
    predicted = [int(value) for value in actions[row_index]]
    predicted_set = torch.zeros_like(targets)
    if predicted:
        predicted_set[predicted] = True
    set_correct = bool(((predicted_set == targets) | ~option_mask).all())
    hybrid = predicted if context == evaluator.SKILL_ORDER_CONTEXT else sorted(predicted)
    policy_logits = outputs["policy_logits"][row_index].float().detach().cpu()
    valid_indices = option_mask.nonzero(as_tuple=False).flatten().tolist()
    target_indices = targets.nonzero(as_tuple=False).flatten().tolist()
    non_targets = [value for value in valid_indices if value not in set(target_indices)]
    top1 = int(policy_logits.masked_fill(~option_mask, float("-inf")).argmax())
    top1_correct = bool(targets[top1]) if target_indices else False
    value_correct = bool(
        (outputs["value_logits"][row_index].detach().cpu() >= 0)
        == cpu_batch["win_targets"][row_index].bool()
    )
    allowed_counts = ppo.count_allowed_mask({key: value[row_index:row_index+1].to(outputs["policy_logits"].device) for key, value in cpu_batch.items()})[0].detach().cpu()
    count_logits = outputs["count_logits"][row_index].float().detach().cpu()
    predicted_count = int(count_logits.masked_fill(~allowed_counts, float("-inf")).argmax())

    def best(values: list[int], reduction: str) -> float:
        if not values:
            return float("inf") if reduction == "min" else float("-inf")
        tensor = policy_logits[values]
        return float(tensor.min() if reduction == "min" else tensor.max())

    set_margin = best(target_indices, "min") - best(non_targets, "max")
    top1_margin = best(target_indices, "max") - best(non_targets, "max")
    remaining = set(valid_indices)
    ordered_step_margins: list[dict[str, Any]] = []
    for position, chosen in enumerate(expert):
        competitors = sorted(remaining - {int(chosen)})
        competitor = max(competitors, key=lambda index: float(policy_logits[index])) if competitors else None
        margin = float("inf") if competitor is None else float(policy_logits[chosen] - policy_logits[competitor])
        ordered_step_margins.append({
            "position": position,
            "expert_choice": int(chosen),
            "best_competitor": competitor,
            "margin": margin,
        })
        remaining.discard(int(chosen))
    expert_count_logit = float(count_logits[expert_count])
    count_competitors = [int(value) for value in allowed_counts.nonzero(as_tuple=False).flatten().tolist() if int(value) != expert_count]
    best_count_competitor = max(count_competitors, key=lambda index: float(count_logits[index])) if count_competitors else None
    count_margin = float("inf") if best_count_competitor is None else expert_count_logit - float(count_logits[best_count_competitor])
    return {
        "order": predicted,
        "set": set_correct,
        "hybrid": hybrid == expert,
        "ordered": predicted == expert,
        "top1": top1_correct,
        "count": len(predicted) == int(cpu_batch["action_counts"][row_index]),
        "value": value_correct,
        "top1_index": top1,
        "predicted_count": predicted_count,
        "set_boundary_margin": set_margin,
        "top1_boundary_margin": top1_margin,
        "expert_order_min_margin": min((entry["margin"] for entry in ordered_step_margins), default=float("inf")),
        "expert_order_step_margins": ordered_step_margins,
        "expert_count_margin": count_margin,
        "valid_option_logits": {str(index): float(policy_logits[index]) for index in valid_indices},
    }


def compact_flip(
    identity: dict[str, Any],
    cpu_batch: Mapping[str, torch.Tensor],
    row_index: int,
    raw: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    expert_count = int(cpu_batch["expert_ordered_action_counts"][row_index])
    expert = [int(v) for v in cpu_batch["expert_ordered_actions"][row_index, :expert_count].tolist()]
    changed = [key for key in METRIC_KEYS if raw[key] != candidate[key]]
    involved = sorted(set(expert) | set(raw["order"]) | set(candidate["order"]) | {raw["top1_index"], candidate["top1_index"]})
    options = identity.pop("options")
    first_difference = None
    width = max(len(raw["order"]), len(candidate["order"]))
    for position in range(width):
        raw_choice = raw["order"][position] if position < len(raw["order"]) else None
        candidate_choice = candidate["order"][position] if position < len(candidate["order"]) else None
        if raw_choice != candidate_choice:
            raw_logits = raw["valid_option_logits"]
            candidate_logits = candidate["valid_option_logits"]
            first_difference = {
                "position": position,
                "raw_choice": raw_choice,
                "candidate_choice": candidate_choice,
                "raw_logit_raw_minus_candidate_choice": (
                    None if raw_choice is None or candidate_choice is None else raw_logits[str(raw_choice)] - raw_logits[str(candidate_choice)]
                ),
                "candidate_logit_raw_minus_candidate_choice": (
                    None if raw_choice is None or candidate_choice is None else candidate_logits[str(raw_choice)] - candidate_logits[str(candidate_choice)]
                ),
            }
            break
    return {
        **identity,
        "context": int(cpu_batch["contexts"][row_index]),
        "min_count": int(cpu_batch["min_counts"][row_index]),
        "max_count": int(cpu_batch["max_counts"][row_index]),
        "option_count": int(cpu_batch["option_mask"][row_index].sum()),
        "expert_order": expert,
        "changed_metrics": changed,
        "first_prediction_difference": first_difference,
        "involved_options": {str(index): options[index] for index in involved},
        "raw": raw,
        "candidate": candidate,
    }


def evaluate_panel(
    label: str,
    archive: Path,
    team_names: tuple[str, ...],
    models: Mapping[str, torch.nn.Module],
    model_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    dataset = IdentifiedValidDataset(archive, model_config, team_names)
    loader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=8,
        collate_fn=partial(
            collate_identified,
            max_state_entities=int(model_config["max_state_entities"]),
            entity_fields=int(model_config["entity_fields"]),
            option_fields=int(model_config["option_fields"]),
        ),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )
    accumulators = {name: evaluator.MetricAccumulator() for name in models}
    flips: list[dict[str, Any]] = []
    with torch.no_grad():
        for cpu_batch, identities in loader:
            gpu_batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
            results: dict[str, tuple[dict[str, torch.Tensor], list[list[int]]]] = {}
            for name, model in models.items():
                outputs = ppo.model_forward(model, gpu_batch, device)
                actions, _, _, _ = ppo.sample_ordered_actions(
                    outputs, gpu_batch, deterministic=True, canonicalize_order=False
                )
                accumulators[name].update(gpu_batch, outputs, actions, actions)
                results[name] = (outputs, actions)
            for row_index, identity in enumerate(identities):
                raw = prediction_values(cpu_batch, *results["raw"], row_index)
                candidate = prediction_values(cpu_batch, *results["candidate"], row_index)
                if any(raw[key] != candidate[key] for key in METRIC_KEYS):
                    flips.append(compact_flip(identity, cpu_batch, row_index, raw, candidate))
    summaries = {name: accumulator.summary() for name, accumulator in accumulators.items()}
    keymap = {
        "set": "set_exact_correct",
        "hybrid": "hybrid_order_exact_correct",
        "ordered": "ordered_exact_correct",
        "top1": "top1_correct",
        "count": "count_correct",
        "value": "value_correct",
    }
    counts = {
        name: {"rows": int(summary["rows"]), **{key: int(summary[path]) for key, path in keymap.items()}}
        for name, summary in summaries.items()
    }
    if counts["raw"] != BASELINES[label]:
        raise RuntimeError(f"{label}: raw metric reproduction mismatch: {counts['raw']} != {BASELINES[label]}")
    candidate_report = json.loads((CANDIDATE_REPORT_ROOT / f"{label}.json").read_text())
    official = candidate_report["metrics"]
    expected_candidate = {"rows": int(official["rows"]), **{key: int(official[path]) for key, path in keymap.items()}}
    if counts["candidate"] != expected_candidate:
        raise RuntimeError(f"{label}: candidate metric reproduction mismatch: {counts['candidate']} != {expected_candidate}")
    transition_counts: dict[str, dict[str, int]] = {}
    for key in METRIC_KEYS:
        transition_counts[key] = {
            "false_to_true": sum(int(not row["raw"][key] and row["candidate"][key]) for row in flips),
            "true_to_false": sum(int(row["raw"][key] and not row["candidate"][key]) for row in flips),
        }
    return {
        "archive": str(archive.relative_to(ROOT)),
        "team_names": list(team_names),
        "metrics": counts,
        "delta": {key: counts["candidate"][key] - counts["raw"][key] for key in METRIC_KEYS},
        "transition_counts": transition_counts,
        "flip_row_count": len(flips),
        "flips": flips,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    bindings = {RAW: RAW_FILE_SHA, CANDIDATE: CANDIDATE_FILE_SHA, **ARCHIVE_SHAS}
    for path, expected in bindings.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"input hash drift: {path}: {observed} != {expected}")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    raw_model, raw_config, _, raw_kind = evaluator.load_policy(RAW, device)
    candidate_model, candidate_config, _, candidate_kind = evaluator.load_policy(CANDIDATE, device)
    if raw_kind != "ppo" or candidate_kind != "ppo" or raw_config != candidate_config:
        raise RuntimeError("model kind/config mismatch")
    model_hashes = {
        "raw": ppo.model_state_sha256(raw_model),
        "candidate": ppo.model_state_sha256(candidate_model),
    }
    if model_hashes != {"raw": RAW_MODEL_SHA, "candidate": CANDIDATE_MODEL_SHA}:
        raise RuntimeError(f"model hash mismatch: {model_hashes}")
    models = {"raw": raw_model, "candidate": candidate_model}
    panels = {
        label: evaluate_panel(label, archive, teams, models, raw_config, device)
        for label, archive, teams in PANELS
    }
    membership: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for panel, result in panels.items():
        for row in result["flips"]:
            membership[row["decision_sha256"]].append({
                "panel": panel,
                "archive": row["archive"],
                "member": row["member"],
                "line_index": row["line_index"],
                "line_sha256": row["line_sha256"],
            })
    shared = [
        {"decision_sha256": key, "occurrences": values}
        for key, values in sorted(membership.items())
        if len({entry["panel"] for entry in values}) > 1
    ]
    pf_flips = panels["pokemonfan"]["flips"]
    pf_recover_raw_correct = [
        {
            "line_sha256": row["line_sha256"],
            "member": row["member"],
            "line_index": row["line_index"],
            "decision_sha256": row["decision_sha256"],
            "changed_metrics": row["changed_metrics"],
            "raw_order": row["raw"]["order"],
            "candidate_order": row["candidate"]["order"],
            "expert_order": row["expert_order"],
            "candidate_set_boundary_margin": row["candidate"]["set_boundary_margin"],
            "candidate_expert_order_min_margin": row["candidate"]["expert_order_min_margin"],
            "shared_with_panels": sorted({entry["panel"] for entry in membership[row["decision_sha256"]] if entry["panel"] != "pokemonfan"}),
        }
        for row in pf_flips
        if any(row["raw"][key] and not row["candidate"][key] for key in ("set", "hybrid", "ordered"))
    ]
    report = {
        "schema_version": "ptcg-e904-specialist-row-flips-read-only-v1",
        "status": "completed_read_only_valid_diagnostic",
        "scope": {
            "split": "valid",
            "prediction_order": "policy_greedy",
            "batch_size": 256,
            "workers": 8,
            "trained": False,
            "checkpoint_written": False,
            "network": False,
            "gold": False,
            "submission": False,
            "warning": "These specialist-valid labels are now consumed for diagnosis and cannot remain an untouched selection gate for a tuned successor.",
        },
        "checkpoint_bindings": {
            "raw": {"path": str(RAW.relative_to(ROOT)), "file_sha256": RAW_FILE_SHA, "model_state_sha256": RAW_MODEL_SHA},
            "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "file_sha256": CANDIDATE_FILE_SHA, "model_state_sha256": CANDIDATE_MODEL_SHA},
        },
        "panels": panels,
        "unique_changed_decision_count": len(membership),
        "shared_changed_decisions": shared,
        "pokemonfan_recover_raw_correct_candidates": pf_recover_raw_correct,
    }
    summary = {
        "status": report["status"],
        "model_hashes": model_hashes,
        "panel_deltas": {label: value["delta"] for label, value in panels.items()},
        "panel_flip_rows": {label: value["flip_row_count"] for label, value in panels.items()},
        "unique_changed_decision_count": report["unique_changed_decision_count"],
        "shared_changed_decisions": shared,
        "pokemonfan_recover_raw_correct_candidates": pf_recover_raw_correct,
    }
    if args.output is not None:
        output = args.output.resolve()
        if output.parent != ROOT / "artifacts":
            raise ValueError("--output must be a direct child of artifacts/")
        publish_o_excl(output, report)
        summary["output"] = str(output.relative_to(ROOT))
        summary["output_sha256"] = sha256_file(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
