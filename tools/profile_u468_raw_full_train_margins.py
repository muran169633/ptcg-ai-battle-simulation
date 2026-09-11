#!/usr/bin/env python3
"""Profile raw full U468 on three frozen train-only specialist archives.

The executable is deliberately separate from the beta=1.157 profiler.  It
binds the integrity-passed U468 file/model hashes and delegates only the
archive streaming and row-margin calculation to the hash-bound train-only
framework.  No archive member outside ``train/`` is opened.
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
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import profile_u468_beta1157_train_margins as framework  # noqa: E402
import train_ppo as ppo  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-raw-full-train-margin-profile-v3"
U468 = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
U468_FILE_SHA256 = (
    "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
)
U468_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
GENERAL_BC = ROOT / (
    "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/"
    "best.pt"
)
GENERAL_BC_SHA256 = (
    "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
)
FRAMEWORK_PATH = ROOT / "tools/profile_u468_beta1157_train_margins.py"
FRAMEWORK_SHA256 = (
    "f1018994c48ffadcbc5277f0dfd9233086693f90bf30f5bf67462965b4e18142"
)

DATASETS = dict(framework.DATASETS)
EXPECTED_DATA_SHA256 = dict(framework.EXPECTED_DATA_SHA256)

RECIPE_PER_BATCH: dict[str, dict[str, int]] = {
    "flg": {"hard": 48, "fragile": 15, "context34": 1, "total": 64},
    "pokemonfan": {
        "hard": 32,
        "fragile": 63,
        "context34": 1,
        "total": 96,
    },
    "core5": {"hard": 32, "fragile": 63, "context34": 1, "total": 96},
}
RECIPE_STEPS = 2
SKILL_ORDER_CONTEXT = 34


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_evidence(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise ValueError(f"{label} must be a single-link regular file")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(
            f"{label} hash drift: expected {expected_sha256}, observed {digest}"
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "mode": oct(observed.st_mode & 0o777),
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = {
        "u468_checkpoint": regular_evidence(
            U468, U468_FILE_SHA256, "raw full U468 checkpoint"
        ),
        "general_bc_checkpoint": regular_evidence(
            GENERAL_BC, GENERAL_BC_SHA256, "general BC checkpoint"
        ),
        "profile_framework": regular_evidence(
            FRAMEWORK_PATH, FRAMEWORK_SHA256, "train-only profile framework"
        ),
        "archives": {
            label: regular_evidence(path, EXPECTED_DATA_SHA256[label], label)
            for label, path in DATASETS.items()
        },
    }
    inventories: dict[str, Any] = {}
    for label, path in DATASETS.items():
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
        train_jsonl = sorted(
            name
            for name in members
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if not train_jsonl:
            raise RuntimeError(f"{label}: no train JSONL members")
        inventories[label] = {
            "all_member_count": len(members),
            "train_jsonl_member_count": len(train_jsonl),
            "train_jsonl_members": train_jsonl,
            "archive_members_opened_during_inventory": 0,
        }
    return evidence, inventories


def load_raw_u468(
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(U468, map_location="cpu", weights_only=False)
    general_bc = torch.load(GENERAL_BC, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(general_bc, dict):
        raise ValueError("checkpoint roots must be dictionaries")
    if checkpoint.get("update") != 468:
        raise ValueError("raw checkpoint update is not U468")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("raw U468 is missing model_config")
    model = ppo.instantiate_model_from_checkpoint(checkpoint, general_bc, device)
    model.eval()
    observed_model_sha256 = ppo.model_state_sha256(model)
    if observed_model_sha256 != U468_MODEL_STATE_SHA256:
        raise ValueError(
            "raw U468 model-state hash drift: expected "
            f"{U468_MODEL_STATE_SHA256}, observed {observed_model_sha256}"
        )
    return model, model_config, checkpoint


def nonempty(record: Mapping[str, Any]) -> bool:
    expert = record.get("expert_order")
    return isinstance(expert, list) and len(expert) > 0


def episode_cap_capacity(records: Sequence[Mapping[str, Any]], cap: int) -> int:
    counts = Counter(str(record.get("episode_id", "")) for record in records)
    return sum(min(value, cap) for value in counts.values())


def unique_line_count(records: Sequence[Mapping[str, Any]]) -> int:
    return len({str(record.get("line_sha256", "")) for record in records})


def recipe_capacity(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    all_pass = True
    for label, quotas in RECIPE_PER_BATCH.items():
        panel = profiles[label]
        near_wrong = panel.get("near_wrong")
        fragile_correct = panel.get("fragile_correct")
        if not isinstance(near_wrong, list) or not isinstance(fragile_correct, list):
            raise ValueError(f"{label}: malformed retained margin lists")
        hard = [
            record
            for record in near_wrong
            if record.get("ordered_correct") is False
            and int(record.get("min_count", -1))
            == int(record.get("max_count", -2))
            and int(record.get("context", -1)) != SKILL_ORDER_CONTEXT
            and nonempty(record)
        ]
        fragile = [
            record
            for record in fragile_correct
            if record.get("ordered_correct") is True
            and int(record.get("context", -1)) != SKILL_ORDER_CONTEXT
            and nonempty(record)
        ]
        context34 = [
            record
            for record in fragile_correct
            if record.get("ordered_correct") is True
            and int(record.get("context", -1)) == SKILL_ORDER_CONTEXT
            and nonempty(record)
        ]
        required = {
            key: quotas[key] * RECIPE_STEPS
            for key in ("hard", "fragile", "context34")
        }
        capacities = {
            "hard_rows": len(hard),
            "hard_unique_line_hashes": unique_line_count(hard),
            "hard_per_step_unique_episodes": episode_cap_capacity(hard, 1),
            "fragile_rows": len(fragile),
            "fragile_unique_line_hashes": unique_line_count(fragile),
            "fragile_per_step_unique_episodes": episode_cap_capacity(fragile, 1),
            "context34_rows": len(context34),
            "context34_unique_line_hashes": unique_line_count(context34),
            "context34_per_step_unique_episodes": episode_cap_capacity(
                context34, 1
            ),
        }
        checks = {
            "hard_total_unique_rows": (
                capacities["hard_unique_line_hashes"] >= required["hard"]
            ),
            "hard_per_step_episode_capacity": (
                capacities["hard_per_step_unique_episodes"] >= quotas["hard"]
            ),
            "fragile_total_unique_rows": (
                capacities["fragile_unique_line_hashes"]
                >= required["fragile"]
            ),
            "fragile_per_step_episode_capacity": (
                capacities["fragile_per_step_unique_episodes"]
                >= quotas["fragile"]
            ),
            "context34_total_unique_rows": (
                capacities["context34_unique_line_hashes"]
                >= required["context34"]
            ),
            "context34_per_step_episode_capacity": (
                capacities["context34_per_step_unique_episodes"]
                >= quotas["context34"]
            ),
        }
        result: dict[str, Any] = {
            "per_batch_quota": quotas,
            "required_for_two_steps": required,
            "available_from_retained_lists": capacities,
            "checks": checks,
            "pass": all(checks.values()),
        }
        if label == "core5":
            teams = sorted(
                {
                    str(record.get("team_name", ""))
                    for record in hard + fragile + context34
                }
            )
            if len(teams) != 5:
                raise ValueError(f"core5 team count drift: {teams}")
            hard_by_team = {
                team: [record for record in hard if record.get("team_name") == team]
                for team in teams
            }
            retention_by_team = {
                team: [
                    record
                    for record in fragile + context34
                    if record.get("team_name") == team
                ]
                for team in teams
            }
            team_checks = {
                team: {
                    "hard_unique_line_hashes": unique_line_count(
                        hard_by_team[team]
                    ),
                    "hard_two_step_row_upper_bound": 13,
                    "hard_per_step_unique_episodes": episode_cap_capacity(
                        hard_by_team[team], 1
                    ),
                    "hard_per_step_quota_upper_bound": 7,
                    "retention_unique_line_hashes": unique_line_count(
                        retention_by_team[team]
                    ),
                    "retention_two_step_row_upper_bound": 26,
                    "retention_per_step_unique_episodes": episode_cap_capacity(
                        retention_by_team[team], 1
                    ),
                    "retention_per_step_quota_upper_bound": 13,
                    "pass": (
                        unique_line_count(hard_by_team[team]) >= 13
                        and episode_cap_capacity(hard_by_team[team], 1) >= 7
                        and unique_line_count(retention_by_team[team]) >= 26
                        and episode_cap_capacity(retention_by_team[team], 1) >= 13
                    ),
                }
                for team in teams
            }
            result["core5_team_capacity"] = team_checks
            result["pass"] = result["pass"] and all(
                check["pass"] for check in team_checks.values()
            )
        panels[label] = result
        all_pass = all_pass and bool(result["pass"])
    return {
        "recipe": "balanced_64_96_96",
        "steps": RECIPE_STEPS,
        "batch_size": sum(value["total"] for value in RECIPE_PER_BATCH.values()),
        "selection_contract": {
            "hard": (
                "near-boundary ordered-wrong, fixed-cardinality, nonempty, "
                "non-context34; maximum one row per episode within each step"
            ),
            "fragile": (
                "lowest-margin ordered-correct, nonempty, non-context34; "
                "maximum one row per episode within each step"
            ),
            "context34": (
                "lowest-margin ordered-correct nonempty context34; maximum "
                "one row per episode within each step"
            ),
            "cross_step_episode_reuse_allowed": True,
            "cross_step_line_hash_reuse_allowed": False,
        },
        "panels": panels,
        "all_sources_have_capacity": all_pass,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static-audit", "formal"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--keep", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    if args.batch_size < 1 or args.keep < 1:
        raise ValueError("batch-size and keep must be positive")
    input_evidence, inventories = verify_inputs()
    if args.mode == "static-audit":
        if args.output is not None:
            raise ValueError("static-audit forbids --output")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_passed_zero_archive_members_opened",
                    "u468_file_sha256": U468_FILE_SHA256,
                    "u468_model_state_sha256": U468_MODEL_STATE_SHA256,
                    "archive_inventory": inventories,
                    "output_writes": 0,
                    "validation_opened": False,
                },
                sort_keys=True,
            )
        )
        return

    if args.output is None:
        raise ValueError("formal mode requires --output")
    output = args.output.resolve()
    if output.parent != ROOT / "artifacts":
        raise ValueError("output must be a direct child of artifacts/")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal profiling requires CUDA")
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model, model_config, checkpoint = load_raw_u468(device)
    profiles = {
        label: framework.profile_archive(
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
    for label, panel in profiles.items():
        opened = panel.get("opened_members")
        if (
            not isinstance(opened, list)
            or not opened
            or any(not str(member).startswith("train/") for member in opened)
            or panel.get("non_train_members_opened") is not False
            or panel.get("archive_sha256") != EXPECTED_DATA_SHA256[label]
        ):
            raise RuntimeError(f"{label}: train-only profile contract failed")
    capacity = recipe_capacity(profiles)
    result = {
        "schema_version": SCHEMA,
        "status": "completed_train_only",
        "base": {
            "checkpoint": str(U468.relative_to(ROOT)),
            "checkpoint_file_sha256": U468_FILE_SHA256,
            "model_state_sha256": U468_MODEL_STATE_SHA256,
            "update": checkpoint.get("update"),
            "feature_version": checkpoint.get("feature_version"),
        },
        "split": "train",
        "validation_opened": False,
        "selection": {
            "near_wrong": (
                "ordered-wrong rows by descending decision margin, then line SHA"
            ),
            "fragile_correct": (
                "ordered-correct rows by ascending decision margin, then line SHA"
            ),
            "decision_margin": (
                "minimum expert-vs-best-alternative margin over ordered "
                "selection and flexible count"
            ),
            "keep_per_list": args.keep,
        },
        "inputs": input_evidence,
        "archive_inventory": inventories,
        "profiles": profiles,
        "balanced_recipe_capacity": capacity,
    }
    framework.publish_o_excl(output, framework.canonical_json(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output.relative_to(ROOT)),
                "sha256": sha256_file(output),
                "base_model_state_sha256": U468_MODEL_STATE_SHA256,
                "rows": {
                    label: panel["rows"] for label, panel in profiles.items()
                },
                "ordered_wrong": {
                    label: panel["ordered_wrong"]
                    for label, panel in profiles.items()
                },
                "balanced_recipe_capacity_pass": capacity[
                    "all_sources_have_capacity"
                ],
                "validation_opened": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
