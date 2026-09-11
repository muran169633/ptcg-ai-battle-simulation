#!/usr/bin/env python3
"""Read-only raw/E904/CW11 specialist-valid row and batch-shape analysis.

This diagnostic clones the frozen B256/8-worker physical evaluator used by
``analyze_e904_specialist_row_flips_v1.py`` and adds the materialized CW11
endpoint.  It also reconstructs, without optimization, the exact 33-row batch
used by CW11's selected-row gate.  No model state is changed and no checkpoint
is written.  A requested JSON report is published once with O_EXCL.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import stat
import sys
import zipfile
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import orjson
import torch
from torch.utils.data import DataLoader


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import analyze_e904_specialist_row_flips_v1 as legacy  # noqa: E402
import probe_u468_raw_actor6_metricguard_fulltrain_cw10_cuttingplane_v1 as cw10  # noqa: E402
import probe_u468_raw_actor6_metricguard_fulltrain_cw9_cuttingplane_v1 as cw9  # noqa: E402
import probe_u468_raw_actor6_metricguard_specialbc_v1 as geometry  # noqa: E402
import probe_u468_raw_actor6_metricguard_specialist_valid_cw11_cuttingplane_v1 as cw11  # noqa: E402
import run_u468_raw_actor6_metricguard_fulltrain_cw4_cuttingplane_v1 as primary  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
LEGACY_ANALYZER_SHA = "c577dc1280cb36197f5125dbd1a0e6be3ddbd3a8dfb17d7aaf6af53a466d6634"
CW11_PROBE_SHA = "23bc74022210942115eaf339a38e710e88ad81ba75d62237e8eb9e2c11e074ee"

RAW = legacy.RAW
E904 = legacy.CANDIDATE
CW11 = ROOT / (
    "artifacts/ppo_u468_raw_actor6_metricguard_specialist_valid_cw11_"
    "materialized_v1_20260802/u468-cw11-formal-pass-eval-only.pt"
)
CHECKPOINT_SHAS = {
    "raw": {
        "file": legacy.RAW_FILE_SHA,
        "model": legacy.RAW_MODEL_SHA,
    },
    "e904": {
        "file": legacy.CANDIDATE_FILE_SHA,
        "model": legacy.CANDIDATE_MODEL_SHA,
    },
    "cw11": {
        "file": "bea774c30cd3113d984ba8252324c330d293a8ce5042d7d17f357f8132e86775",
        "model": "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e",
    },
}

E904_REPORT_ROOT = legacy.CANDIDATE_REPORT_ROOT
CW11_REPORT_ROOT = ROOT / (
    "artifacts/cw11_4317_specialist_devconsistency_20260802_v1."
    "specialist_dev_consistency"
)
REPORT_SHAS = {
    "e904": {
        "pokemonfan": "659f8e259ac3222430acd76fe6f6dddb7258add095328ef251163d707ccaeb75",
        "flg": "85c0d4bc4eeac57dc84b215255ab8ab2fcfd2ef2eb327b95dcd1567bdfec5afe",
        "core5": "633a1c1df714aa428981bbb4a8eb3ec04488b68836d22024f7569087f58add5a",
        "dominic": "bacbe22ee69472a45c60c402a4e643b87207caf4e2d646ee85adf1efa60db486",
        "luca": "219628f95fa78d2b8b8431a7a6b4f8023f9b47b1073fee8cdad00d9641280e18",
        "szlach": "ed6098bad6acf6a67060670881708ee0e654bcf18f820b37e5f5fcabc92cd185",
    },
    "cw11": {
        "pokemonfan": "eaab85cb0a495947b5039a98edef72f70432339ec1545d45a98f01e75bcae1c7",
        "flg": "9ffc78ea4a198e4b5af5d8207c1b06c9810301f69637f32986fc659a0689d17d",
        "core5": "058b6c55d2f5ceae16db61df271077ae6851149c82b6a7425fa94548b6607c59",
        "dominic": "ee5b3825c370376a572306a6611bf2e957969f799676033a3fa0f275067d56d7",
        "luca": "5ac88b3d745395a448193f5840e712d62fe28e35df619f2d07cb1848ac17e295",
        "szlach": "aabe4a970f0333685f4106b29222c1425bd03321ff95624d109aafdb04a88ff6",
    },
}
DECISION = CW11_REPORT_ROOT / "dev_consistency_decision.json"
DECISION_SHA = "9c925a66f116a10b0ac996a76fdcbe0c378f0e4d02d45c635a2d47228b189a35"
MATERIALIZATION_MANIFEST = ROOT / (
    "artifacts/ppo_u468_raw_actor6_metricguard_specialist_valid_cw11_"
    "materialized_v1_20260802/materialization_manifest.json"
)
MATERIALIZATION_MANIFEST_SHA = (
    "6356f08505b5777fdb30e4f8cfce8a776ca2918af540ebc324227678aeb02016"
)

MODEL_PATHS = {"raw": RAW, "e904": E904, "cw11": CW11}
MODEL_NAMES = tuple(MODEL_PATHS)
PAIR_NAMES = (("raw", "e904"), ("raw", "cw11"), ("e904", "cw11"))
METRIC_KEYS = legacy.METRIC_KEYS
KEYMAP = {
    "set": "set_exact_correct",
    "hybrid": "hybrid_order_exact_correct",
    "ordered": "ordered_exact_correct",
    "top1": "top1_correct",
    "count": "count_correct",
    "value": "value_correct",
}
DESCRIPTOR_METRIC_MAP = {
    "set_exact": "set",
    "hybrid_order_exact": "hybrid",
    "ordered_exact": "ordered",
    "top1_correct": "top1",
}


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


def official_counts(root: Path, label: str) -> dict[str, int]:
    payload = json.loads((root / f"{label}.json").read_text())
    metrics = payload["metrics"]
    return {
        "rows": int(metrics["rows"]),
        **{key: int(metrics[path]) for key, path in KEYMAP.items()},
    }


def first_prediction_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    width = max(len(left["order"]), len(right["order"]))
    for position in range(width):
        left_choice = left["order"][position] if position < len(left["order"]) else None
        right_choice = right["order"][position] if position < len(right["order"]) else None
        if left_choice == right_choice:
            continue
        return {
            "position": position,
            "left_choice": left_choice,
            "right_choice": right_choice,
            "left_logit_left_minus_right": (
                None
                if left_choice is None or right_choice is None
                else left["valid_option_logits"][str(left_choice)]
                - left["valid_option_logits"][str(right_choice)]
            ),
            "right_logit_left_minus_right": (
                None
                if left_choice is None or right_choice is None
                else right["valid_option_logits"][str(left_choice)]
                - right["valid_option_logits"][str(right_choice)]
            ),
        }
    return None


def compact_row(
    panel: str,
    identity: Mapping[str, Any],
    cpu_batch: Mapping[str, torch.Tensor],
    row_index: int,
    states: Mapping[str, Mapping[str, Any]],
    batch_ordinal: int,
    batch_size: int,
) -> dict[str, Any]:
    expert_count = int(cpu_batch["expert_ordered_action_counts"][row_index])
    expert = [
        int(value)
        for value in cpu_batch["expert_ordered_actions"][row_index, :expert_count].tolist()
    ]
    involved = set(expert)
    for state in states.values():
        involved.update(state["order"])
        involved.add(int(state["top1_index"]))
    options = identity["options"]
    pairwise = {}
    for left_name, right_name in PAIR_NAMES:
        left = states[left_name]
        right = states[right_name]
        pairwise[f"{left_name}_to_{right_name}"] = {
            "changed_metrics": [key for key in METRIC_KEYS if left[key] != right[key]],
            "first_prediction_difference": first_prediction_difference(left, right),
        }
    return {
        **{key: copy.deepcopy(value) for key, value in identity.items() if key != "options"},
        "panel": panel,
        "official_stream_batch_ordinal_one_based": batch_ordinal,
        "official_stream_batch_size": batch_size,
        "official_stream_offset_zero_based": row_index,
        "context": int(cpu_batch["contexts"][row_index]),
        "min_count": int(cpu_batch["min_counts"][row_index]),
        "max_count": int(cpu_batch["max_counts"][row_index]),
        "option_count": int(cpu_batch["option_mask"][row_index].sum()),
        "expert_order": expert,
        "involved_options": {str(index): options[index] for index in sorted(involved)},
        "states": copy.deepcopy(states),
        "pairwise": pairwise,
    }


def changed_decision(states: Mapping[str, Mapping[str, Any]]) -> bool:
    for left_name, right_name in PAIR_NAMES:
        left, right = states[left_name], states[right_name]
        if left["order"] != right["order"] or left["top1_index"] != right["top1_index"]:
            return True
        if left["predicted_count"] != right["predicted_count"]:
            return True
        if any(left[key] != right[key] for key in METRIC_KEYS):
            return True
    return False


def evaluate_panel(
    label: str,
    archive: Path,
    team_names: tuple[str, ...],
    models: Mapping[str, torch.nn.Module],
    model_config: Mapping[str, Any],
    device: torch.device,
    capture_keys: set[tuple[str, int, str]],
) -> tuple[dict[str, Any], dict[tuple[str, int, str], dict[str, Any]]]:
    dataset = legacy.IdentifiedValidDataset(archive, model_config, team_names)
    loader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=8,
        collate_fn=partial(
            legacy.collate_identified,
            max_state_entities=int(model_config["max_state_entities"]),
            entity_fields=int(model_config["entity_fields"]),
            option_fields=int(model_config["option_fields"]),
        ),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )
    accumulators = {name: legacy.evaluator.MetricAccumulator() for name in models}
    transition_counts = {
        f"{left}_to_{right}": {
            key: {"false_to_true": 0, "true_to_false": 0} for key in METRIC_KEYS
        }
        for left, right in PAIR_NAMES
    }
    changed_rows: list[dict[str, Any]] = []
    captures: dict[tuple[str, int, str], dict[str, Any]] = {}
    with torch.no_grad():
        for batch_ordinal, (cpu_batch, identities) in enumerate(loader, start=1):
            gpu_batch = {
                key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()
            }
            results: dict[str, tuple[dict[str, torch.Tensor], list[list[int]]]] = {}
            for name, model in models.items():
                outputs = legacy.ppo.model_forward(model, gpu_batch, device)
                actions, _, _, _ = legacy.ppo.sample_ordered_actions(
                    outputs,
                    gpu_batch,
                    deterministic=True,
                    canonicalize_order=False,
                )
                accumulators[name].update(gpu_batch, outputs, actions, actions)
                results[name] = (outputs, actions)
            batch_size = len(identities)
            for row_index, identity in enumerate(identities):
                states = {
                    name: legacy.prediction_values(
                        cpu_batch, *results[name], row_index
                    )
                    for name in MODEL_NAMES
                }
                for left_name, right_name in PAIR_NAMES:
                    transitions = transition_counts[f"{left_name}_to_{right_name}"]
                    for key in METRIC_KEYS:
                        if not states[left_name][key] and states[right_name][key]:
                            transitions[key]["false_to_true"] += 1
                        elif states[left_name][key] and not states[right_name][key]:
                            transitions[key]["true_to_false"] += 1
                capture_key = (
                    str(identity["member"]),
                    int(identity["line_index"]),
                    str(identity["line_sha256"]),
                )
                if changed_decision(states) or capture_key in capture_keys:
                    record = compact_row(
                        label,
                        identity,
                        cpu_batch,
                        row_index,
                        states,
                        batch_ordinal,
                        batch_size,
                    )
                    if changed_decision(states):
                        changed_rows.append(record)
                    if capture_key in capture_keys:
                        captures[capture_key] = record
    summaries = {name: accumulator.summary() for name, accumulator in accumulators.items()}
    counts = {
        name: {
            "rows": int(summary["rows"]),
            **{key: int(summary[path]) for key, path in KEYMAP.items()},
        }
        for name, summary in summaries.items()
    }
    expected = {
        "raw": legacy.BASELINES[label],
        "e904": official_counts(E904_REPORT_ROOT, label),
        "cw11": official_counts(CW11_REPORT_ROOT, label),
    }
    if counts != expected:
        raise RuntimeError(f"{label}: physical reproduction mismatch: {counts} != {expected}")
    return {
        "archive": str(archive.relative_to(ROOT)),
        "team_names": list(team_names),
        "metrics": counts,
        "delta_from_raw": {
            name: {key: counts[name][key] - counts["raw"][key] for key in METRIC_KEYS}
            for name in ("e904", "cw11")
        },
        "transition_counts": transition_counts,
        "changed_row_count_in_panel_view": len(changed_rows),
        "changed_rows": changed_rows,
    }, captures


def selected_batch33(
    models: Mapping[str, torch.nn.Module],
    model_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original, formal_evidence = primary.load_original_descriptors(geometry)
    live_guards = [
        *[dict(value) for value in primary.NEW_GUARDS],
        *[dict(value) for value in cw9.ADDED_GUARDS],
        dict(cw10.FINAL_ADDED_GUARD),
    ]
    old_descriptors = [*[dict(value) for value in original], *live_guards]
    descriptors = [
        *old_descriptors,
        *[dict(value) for value in cw11.ADDED_VALID_GUARDS],
    ]
    checks = {
        "original_rows_exact_19": len(original) == 19,
        "live_guard_rows_exact_10": len(live_guards) == 10,
        "old_rows_exact_29": len(old_descriptors) == 29,
        "all_rows_exact_33": len(descriptors) == 33,
        "identities_unique_33": len({cw11.identity_key(value) for value in descriptors}) == 33,
        "last_four_exact_added_valid_guards": [
            cw11.identity_key(value) for value in descriptors[-4:]
        ]
        == [cw11.identity_key(value) for value in cw11.ADDED_VALID_GUARDS],
    }
    if not all(checks.values()):
        raise RuntimeError(f"selected B33 descriptor reconstruction failed: {checks}")
    rows, loading_audit = cw11.load_selected_train_and_exact_valid_rows(
        geometry, legacy, descriptors, model_config
    )
    cpu_batch = legacy.evaluator.collate_ordered(
        [item["features"] for item in rows],
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    gpu_batch = {key: value.to(device) for key, value in cpu_batch.items()}
    results = {}
    with torch.no_grad():
        for name, model in models.items():
            outputs = legacy.ppo.model_forward(model, gpu_batch, device)
            actions, _, _, _ = legacy.ppo.sample_ordered_actions(
                outputs, gpu_batch, deterministic=True, canonicalize_order=False
            )
            results[name] = (outputs, actions)
    records = []
    for row_index, descriptor in enumerate(descriptors[-4:], start=29):
        states = {
            name: legacy.prediction_values(cpu_batch, *results[name], row_index)
            for name in MODEL_NAMES
        }
        records.append(
            {
                "selected_batch_size": 33,
                "selected_batch_offset_zero_based": row_index,
                "identity": cw11.identity_record(descriptor),
                "descriptor": copy.deepcopy(descriptor),
                "states": states,
            }
        )
    return records, {
        "checks": checks,
        "formal_descriptor_evidence": formal_evidence,
        "row_loading_audit": loading_audit,
        "descriptor_identity_sha256": canonical_sha(
            [cw11.identity_record(value) for value in descriptors]
        ),
    }


def reduced_state(state: Mapping[str, Any], positive: int, negative: int) -> dict[str, Any]:
    return {
        "order": list(state["order"]),
        "flags": {key: bool(state[key]) for key in METRIC_KEYS},
        "top1_index": int(state["top1_index"]),
        "predicted_count": int(state["predicted_count"]),
        "pair_margin_positive_minus_negative": (
            float(state["valid_option_logits"][str(positive)])
            - float(state["valid_option_logits"][str(negative)])
        ),
        "positive_logit": float(state["valid_option_logits"][str(positive)]),
        "negative_logit": float(state["valid_option_logits"][str(negative)]),
        "set_boundary_margin": float(state["set_boundary_margin"]),
        "top1_boundary_margin": float(state["top1_boundary_margin"]),
        "expert_order_min_margin": float(state["expert_order_min_margin"]),
    }


def build_guard_audit(
    captures: Mapping[str, Mapping[tuple[str, int, str], Mapping[str, Any]]],
    selected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for descriptor, small in zip(cw11.ADDED_VALID_GUARDS, selected, strict=True):
        panel = str(descriptor["panel"])
        key = (
            str(descriptor["member"]),
            int(descriptor["line_index_zero_based"]),
            str(descriptor["line_sha256"]),
        )
        physical = captures[panel].get(key)
        if physical is None:
            raise RuntimeError(f"guard missing in B256 capture: {panel}/{key}")
        positive = int(descriptor["positive_option"])
        negative = int(descriptor["negative_option"])
        official_states = {
            name: reduced_state(physical["states"][name], positive, negative)
            for name in MODEL_NAMES
        }
        selected_states = {
            name: reduced_state(small["states"][name], positive, negative)
            for name in MODEL_NAMES
        }
        required = [DESCRIPTOR_METRIC_MAP[value] for value in descriptor["metrics_union"]]
        selected_required_pass = all(selected_states["cw11"][key][0:0] == [] for key in [])
        selected_required_pass = all(selected_states["cw11"]["flags"][key] for key in required)
        official_required_pass = all(official_states["cw11"]["flags"][key] for key in required)
        selected_floor_pass = (
            selected_states["cw11"]["pair_margin_positive_minus_negative"]
            >= selected_states["raw"]["pair_margin_positive_minus_negative"]
        )
        official_floor_pass = (
            official_states["cw11"]["pair_margin_positive_minus_negative"]
            >= official_states["raw"]["pair_margin_positive_minus_negative"]
        )
        reported_checks = {
            "official_raw_margin_matches_descriptor": (
                official_states["raw"]["pair_margin_positive_minus_negative"]
                == float(descriptor["reported_specialist_b256_raw_margin"])
            ),
            "official_e904_margin_matches_descriptor": (
                official_states["e904"]["pair_margin_positive_minus_negative"]
                == float(descriptor["reported_specialist_b256_cw10_margin"])
            ),
            "selected_raw_order_matches_descriptor": selected_states["raw"]["order"]
            == list(descriptor["expected_raw_order"]),
            "selected_e904_order_matches_descriptor": selected_states["e904"]["order"]
            == list(descriptor["expected_cw10_order"]),
            "selected_cw11_required_flags_pass": selected_required_pass,
            "selected_cw11_raw_pair_floor_pass": selected_floor_pass,
        }
        if not all(reported_checks.values()):
            raise RuntimeError(f"guard B33/B256 binding failed: {panel}: {reported_checks}")
        records.append(
            {
                "identity": small["identity"],
                "specialist_views": list(descriptor["specialist_views"]),
                "required_metrics": required,
                "positive_option": positive,
                "negative_option": negative,
                "selected_B33": {
                    "batch_size": 33,
                    "offset_zero_based": int(small["selected_batch_offset_zero_based"]),
                    "states": selected_states,
                    "cw11_required_flags_pass": selected_required_pass,
                    "cw11_same_shape_raw_pair_floor_pass": selected_floor_pass,
                },
                "official_B256_stream": {
                    "batch_ordinal_one_based": int(
                        physical["official_stream_batch_ordinal_one_based"]
                    ),
                    "batch_size": int(physical["official_stream_batch_size"]),
                    "offset_zero_based": int(physical["official_stream_offset_zero_based"]),
                    "states": official_states,
                    "cw11_required_flags_pass": official_required_pass,
                    "cw11_same_shape_raw_pair_floor_pass": official_floor_pass,
                },
                "margin_selected_B33_minus_official_B256": {
                    name: (
                        selected_states[name]["pair_margin_positive_minus_negative"]
                        - official_states[name]["pair_margin_positive_minus_negative"]
                    )
                    for name in MODEL_NAMES
                },
                "physical_outcome": (
                    "repaired_in_official_B256"
                    if official_required_pass
                    else "not_repaired_in_official_B256"
                ),
                "batch_shape_gate_disagreement": (
                    selected_required_pass != official_required_pass
                    or selected_floor_pass != official_floor_pass
                ),
                "checks": reported_checks,
            }
        )
    return records


def physical_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["archive"]),
        str(row["member"]),
        int(row["line_index"]),
        str(row["line_sha256"]),
    )


def unique_physical_rows(panels: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for panel, result in panels.items():
        for row in result["changed_rows"]:
            key = physical_key(row)
            occurrence = {
                "panel": panel,
                "official_stream_batch_ordinal_one_based": int(
                    row["official_stream_batch_ordinal_one_based"]
                ),
                "official_stream_batch_size": int(row["official_stream_batch_size"]),
                "official_stream_offset_zero_based": int(
                    row["official_stream_offset_zero_based"]
                ),
            }
            if key not in grouped:
                item = copy.deepcopy(row)
                item.pop("panel")
                item["panel_view_occurrences"] = [occurrence]
                grouped[key] = item
            else:
                prior = grouped[key]
                if prior["decision_sha256"] != row["decision_sha256"]:
                    raise RuntimeError("same physical row has inconsistent decision fingerprint")
                if canonical_sha(prior["states"]) != canonical_sha(row["states"]):
                    raise RuntimeError("same physical row has inconsistent model predictions")
                prior["panel_view_occurrences"].append(occurrence)
    rows = list(grouped.values())
    for row in rows:
        row["panel_view_occurrences"].sort(key=lambda value: value["panel"])
    rows.sort(key=physical_key)
    return rows


def scan_train_exact_matches(
    fingerprints: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    matches: dict[str, list[dict[str, Any]]] = {value: [] for value in fingerprints}
    archives = sorted(legacy.ARCHIVE_SHAS, key=lambda value: str(value))
    scanned_rows = 0
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.startswith("train/") and name.endswith(".jsonl")
            )
            for member in members:
                with archive.open(member) as handle:
                    for line_index, raw in enumerate(handle):
                        row = orjson.loads(raw)
                        if str(row.get("split", "")) != "train":
                            continue
                        action = row.get("action")
                        if not isinstance(action, list):
                            continue
                        try:
                            expert = [int(value) for value in action]
                        except (TypeError, ValueError):
                            continue
                        scanned_rows += 1
                        fingerprint = canonical_sha(
                            {
                                "observation": row.get("observation"),
                                "action": expert,
                                "min_count": row.get("min_count"),
                                "max_count": row.get("max_count"),
                            }
                        )
                        if fingerprint not in fingerprints:
                            continue
                        matches[fingerprint].append(
                            {
                                "archive": str(archive_path.relative_to(ROOT)),
                                "member": member,
                                "line_index_zero_based": line_index,
                                "line_sha256": hashlib.sha256(raw).hexdigest(),
                                "team_name": str(row.get("team_name", "")),
                                "episode_id": str(row.get("episode_id", "")),
                            }
                        )
    return matches, {
        "archives": [str(path.relative_to(ROOT)) for path in archives],
        "split": "train",
        "rows_with_parseable_action_scanned": scanned_rows,
        "target_decision_fingerprint_count": len(fingerprints),
        "matched_decision_fingerprint_count": sum(bool(value) for value in matches.values()),
        "exact_match_count": sum(len(value) for value in matches.values()),
    }


def repair_coverage(
    row: Mapping[str, Any], failed_keys: set[tuple[str, str]]
) -> dict[tuple[str, str], int]:
    coverage: dict[tuple[str, str], int] = defaultdict(int)
    raw = row["states"]["raw"]
    current = row["states"]["cw11"]
    for occurrence in row["panel_view_occurrences"]:
        panel = str(occurrence["panel"])
        for metric in ("set", "hybrid", "ordered", "top1", "count", "value"):
            key = (panel, metric)
            if key in failed_keys and raw[metric] and not current[metric]:
                coverage[key] += 1
    return dict(coverage)


def minimum_repair_sets(
    physical_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    train_matches: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    deficits = {}
    for gate in decision["failed_gates"]:
        panel, metric = str(gate["gate"]).split(".", 1)
        deficits[(panel, metric)] = int(gate["threshold"]) - int(gate["observed"])
    candidates = []
    for row in physical_rows:
        coverage = repair_coverage(row, set(deficits))
        if not coverage:
            continue
        raw = row["states"]["raw"]
        current = row["states"]["cw11"]
        first = first_prediction_difference(raw, current)
        candidates.append(
            {
                "physical_key": list(physical_key(row)),
                "decision_sha256": row["decision_sha256"],
                "team_name": row["team_name"],
                "expert_order": row["expert_order"],
                "raw_order": raw["order"],
                "e904_order": row["states"]["e904"]["order"],
                "cw11_order": current["order"],
                "coverage": {f"{panel}.{metric}": count for (panel, metric), count in coverage.items()},
                "first_raw_to_cw11_difference": first,
                "cw11_set_boundary_margin": current["set_boundary_margin"],
                "cw11_expert_order_min_margin": current["expert_order_min_margin"],
                "train_exact_matches": list(train_matches.get(row["decision_sha256"], [])),
            }
        )
    candidates.sort(key=lambda value: tuple(value["physical_key"]))
    deficit_named = {f"{panel}.{metric}": value for (panel, metric), value in deficits.items()}
    solutions: list[tuple[int, ...]] = []
    for width in range(1, len(candidates) + 1):
        for indices in itertools.combinations(range(len(candidates)), width):
            observed = defaultdict(int)
            for index in indices:
                for key, value in candidates[index]["coverage"].items():
                    observed[key] += int(value)
            if all(observed[key] >= required for key, required in deficit_named.items()):
                solutions.append(indices)
        if solutions:
            break
    serialized_solutions = [
        {
            "candidate_indices": list(indices),
            "physical_keys": [candidates[index]["physical_key"] for index in indices],
            "decision_sha256s": [candidates[index]["decision_sha256"] for index in indices],
        }
        for indices in solutions
    ]
    return {
        "assumption": (
            "Each guarded physical decision is restored to raw correctness with no collateral "
            "changes; this is a local set-cover calculation, not a promotion claim."
        ),
        "failed_gate_deficits": deficit_named,
        "candidate_pool": candidates,
        "minimum_guard_count": len(solutions[0]) if solutions else None,
        "minimum_solution_count": len(solutions),
        "minimum_solutions": serialized_solutions,
    }


def validate_bindings() -> dict[str, Any]:
    bindings = {
        TOOLS / "analyze_e904_specialist_row_flips_v1.py": LEGACY_ANALYZER_SHA,
        TOOLS / "probe_u468_raw_actor6_metricguard_specialist_valid_cw11_cuttingplane_v1.py": CW11_PROBE_SHA,
        RAW: CHECKPOINT_SHAS["raw"]["file"],
        E904: CHECKPOINT_SHAS["e904"]["file"],
        CW11: CHECKPOINT_SHAS["cw11"]["file"],
        DECISION: DECISION_SHA,
        MATERIALIZATION_MANIFEST: MATERIALIZATION_MANIFEST_SHA,
        **legacy.ARCHIVE_SHAS,
    }
    for model_name, root, hashes in (
        ("e904", E904_REPORT_ROOT, REPORT_SHAS["e904"]),
        ("cw11", CW11_REPORT_ROOT, REPORT_SHAS["cw11"]),
    ):
        for panel, digest in hashes.items():
            bindings[root / f"{panel}.json"] = digest
    records = []
    for path, expected in bindings.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"input hash drift: {path}: {observed} != {expected}")
        records.append({"path": str(path.relative_to(ROOT)), "sha256": observed})
    return {"all_exact": True, "files": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        raise RuntimeError("requires CUDA for read-only physical inference")
    input_bindings = validate_bindings()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    models = {}
    model_config = None
    model_hashes = {}
    for name, path in MODEL_PATHS.items():
        model, config, _, kind = legacy.evaluator.load_policy(path, device)
        if kind != "ppo":
            raise RuntimeError(f"{name}: expected PPO checkpoint")
        if model_config is None:
            model_config = config
        elif config != model_config:
            raise RuntimeError("model config mismatch")
        model.eval()
        models[name] = model
        model_hashes[name] = legacy.ppo.model_state_sha256(model)
    expected_model_hashes = {name: value["model"] for name, value in CHECKPOINT_SHAS.items()}
    if model_hashes != expected_model_hashes:
        raise RuntimeError(f"model hash mismatch: {model_hashes} != {expected_model_hashes}")
    assert model_config is not None

    captures: dict[str, dict[tuple[str, int, str], dict[str, Any]]] = {}
    panels = {}
    for label, archive, team_names in legacy.PANELS:
        capture_keys = {
            (
                str(value["member"]),
                int(value["line_index_zero_based"]),
                str(value["line_sha256"]),
            )
            for value in cw11.ADDED_VALID_GUARDS
            if str(value["panel"]) == label
        }
        panels[label], captures[label] = evaluate_panel(
            label,
            archive,
            team_names,
            models,
            model_config,
            device,
            capture_keys,
        )

    selected, selected_audit = selected_batch33(models, model_config, device)
    guard_audit = build_guard_audit(captures, selected)
    physical_rows = unique_physical_rows(panels)
    fingerprints = {str(row["decision_sha256"]) for row in physical_rows}
    train_matches, train_scan = scan_train_exact_matches(fingerprints)
    decision = json.loads(DECISION.read_text())
    repair_sets = minimum_repair_sets(physical_rows, decision, train_matches)

    fingerprint_membership: dict[str, list[list[Any]]] = defaultdict(list)
    for row in physical_rows:
        fingerprint_membership[str(row["decision_sha256"])].append(list(physical_key(row)))
    duplicate_fingerprints = {
        key: values for key, values in sorted(fingerprint_membership.items()) if len(values) > 1
    }
    all_guard_disagreements = [
        value["identity"] for value in guard_audit if value["batch_shape_gate_disagreement"]
    ]
    report = {
        "schema_version": "ptcg-cw11-specialist-dev-row-flips-read-only-v1",
        "status": "completed_read_only_failure_attribution",
        "scope": {
            "physical_evaluator": "B256_8worker_policy_greedy",
            "selected_gate_reconstruction": "exact_ordered_33row_batch_forward_only",
            "trained": False,
            "optimizer_or_backward": False,
            "checkpoint_written": False,
            "specialist_one_shot_rerun": False,
            "broad_opened": False,
            "gold_opened": False,
            "network_upload_submission": False,
            "warning": (
                "All six specialist-valid panels were already consumed by the failed one-shot; "
                "this row-level diagnostic further confines them to development-only evidence."
            ),
        },
        "input_bindings": input_bindings,
        "checkpoint_bindings": {
            name: {
                "path": str(MODEL_PATHS[name].relative_to(ROOT)),
                "file_sha256": CHECKPOINT_SHAS[name]["file"],
                "model_state_sha256": CHECKPOINT_SHAS[name]["model"],
            }
            for name in MODEL_NAMES
        },
        "failed_one_shot_decision": decision,
        "panels": panels,
        "unique_physical_changed_row_count": len(physical_rows),
        "unique_decision_fingerprint_count": len(fingerprint_membership),
        "duplicate_decision_fingerprints_across_physical_rows": duplicate_fingerprints,
        "unique_physical_changed_rows": physical_rows,
        "four_original_cw11_guard_batch_shape_audit": guard_audit,
        "batch_shape_attribution": {
            "guard_disagreement_count": len(all_guard_disagreements),
            "guard_disagreement_identities": all_guard_disagreements,
            "conclusion": (
                "The selected gate used raw floors and candidate logits from one 33-row BF16 "
                "matrix shape.  The official evaluator places the same rows at recorded offsets "
                "inside separate B256 stream batches.  Any row that passes B33 but fails B256 is "
                "a directly observed batch-shape gate mismatch, not an evaluator-label mismatch."
            ),
        },
        "selected_batch33_reconstruction_audit": selected_audit,
        "train_exact_decision_fingerprint_matches": train_matches,
        "train_exact_scan_audit": train_scan,
        "cw12_minimum_exact_repair_analysis": repair_sets,
        "cw12_recommendation": {
            "optimization_guard_count": repair_sets["minimum_guard_count"],
            "optimization_guard_rule": (
                "Choose one minimum set: two PokemonFan all-main-metric harmful physical rows "
                "plus one physical row whose correction covers both core5 and Dominic."
            ),
            "batch_context_rule": (
                "Evaluate every repair and retention constraint in its cloned official B256 "
                "batch context at the recorded ordinal/offset.  If a compact batch is also kept, "
                "require both shapes to pass their same-shape raw floors with a strict positive "
                "tie buffer; never reuse only the B33 raw threshold."
            ),
            "retention_rule": (
                f"Retain all {len(physical_rows)} unique changed physical rows and all favorable "
                "raw-to-CW11 transitions as decision-level no-regression constraints."
            ),
            "promotion_rule": (
                "The consumed specialist-valid data may only be a development consistency gate. "
                "A CW12 endpoint needs untouched broad behavior and fresh Gold evidence before any "
                "promotion or package/submission authorization."
            ),
            "current_authorization": {
                "broad": False,
                "gold": False,
                "package": False,
                "submission": False,
            },
        },
    }
    summary = {
        "status": report["status"],
        "model_hashes": model_hashes,
        "panel_deltas_from_raw": {
            label: value["delta_from_raw"] for label, value in panels.items()
        },
        "unique_physical_changed_row_count": len(physical_rows),
        "unique_decision_fingerprint_count": len(fingerprint_membership),
        "guard_batch_shape_disagreement_count": len(all_guard_disagreements),
        "minimum_exact_repair_guard_count": repair_sets["minimum_guard_count"],
        "minimum_exact_repair_solution_count": repair_sets["minimum_solution_count"],
        "train_exact_match_count": train_scan["exact_match_count"],
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
