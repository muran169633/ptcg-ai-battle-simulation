#!/usr/bin/env python3
"""Audit a balanced train-only actor6 special-BC sweep from raw full U468.

Selection, cache, and shadow-contract modes are strictly zero-write.  The real
two-step shadow trajectory is CUDA-only and cannot run without an exact
hash-bound preregistration plus a one-shot ``O_EXCL`` attempt marker.  It
publishes one result JSON with ``O_EXCL`` and never opens validation member
payloads or writes a model, optimizer, checkpoint, or training artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import orjson
import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_ppo_bc_repair as repair  # noqa: E402
import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-raw-trainhard-actor6-balanced-shadow-sweep-v1"
SHADOW_PREREGISTRATION_SCHEMA = (
    "ptcg-u468-raw-trainhard-actor6-balanced-shadow-preregistration-v1"
)
BRANCH = "ppo_u468_raw_trainhard_actor6_balanced64_96_96_p1p2_design202608111"
OUTPUT_ROOT = ROOT / "artifacts" / BRANCH
ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-attempt.json"
SHADOW_ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-shadow-attempt.json"

U468 = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
U468_SHA256 = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
BASE_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
GENERAL_BC = ROOT / (
    "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/"
    "best.pt"
)
GENERAL_BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"

PROFILE = ROOT / "artifacts/u468_raw_full_train_margin_profile_v3_20260802.json"
PROFILE_SHA256 = "17af98e175167398a80cd2e1f8930c3d1d85cfe6ceea3143dfe2ba44dc89f1f9"

DATASETS = {
    "flg": ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
    "pokemonfan": ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
    "core5": ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
}
DATA_SHA256 = {
    "flg": "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    "pokemonfan": "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    "core5": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}
DEPENDENCY_SHA256 = {
    ROOT / "tools/train_ppo.py": (
        "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
    ),
    ROOT / "tools/train_bc_orbit.py": (
        "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
    ),
    ROOT / "tools/run_ppo_bc_repair.py": (
        "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"
    ),
    ROOT / "tools/run_u468_beta1157_trainhard_actor6_mix_sweep.py": (
        "4b53018763a09dad1e9b7917fe19c697e6d05f46e1f481da8fd0bc627c7755ac"
    ),
    ROOT / "tools/profile_u468_raw_full_train_margins.py": (
        "c70e52e2398b8551baf0fa3a3a403fbf6636b56c70010fb948a5c85c298977ae"
    ),
}

ACTOR_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
SELECTION_SEED = 202608111
EXECUTION_SEED = 202608112
BATCH_SIZE = 256
STEPS = 2
ENDPOINT_STEPS = (1, 2)
SOURCE_COUNTS = {"flg": 64, "pokemonfan": 96, "core5": 96}
CATEGORY_COUNTS = {
    "flg": {"hard": 48, "fragile": 15, "c34": 1},
    "pokemonfan": {"hard": 32, "fragile": 63, "c34": 1},
    "core5": {"hard": 32, "fragile": 63, "c34": 1},
}
CONTEXT34_SAMPLE_WEIGHT = 1.0 / 3.0
ORDER_CONTEXT_WEIGHT = 8.0
BASE_ACTOR_LEARNING_RATE = 3.6e-5
LEARNING_RATE = 1.125e-7
LR_CANDIDATES = (5.625e-8, 1.125e-7)
EXPECTED_STEP_SELECTION_SHA256 = (
    "6ff7becab756a4ed528195b22828e4abbaedbd3603b77ac87867d0310f875b6e",
    "6475bf85e4fa204811acc1f4628b8e28a6b15081c0e2ce8c6b0e08413a02cdd1",
)
EXPECTED_CACHE_SHA256 = (
    "39c34d393eec077e9e2b005f4477b94a9141504d126cc702891af711a99a390f"
)
EXPECTED_BATCH_SHA256 = (
    "0dbdfb804fe2c1ca0187d4b1a1178ca3128b9ed53c447e12639fbf75c198953a",
    "aa8fa4714da5c26396f5b1a95e5197dc74a133ea0e2319dbea59e21ebc23dc00",
)
WEIGHT_DECAY = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-5
MAX_GRAD_NORM = 0.5
LR_GATES = {
    "all_finite": True,
    "gradient_scope_exact_actor6": True,
    "six_actor_tensors_each_nonzero": True,
    "changed_scope_exact_actor6": True,
    "fresh_optimizer_state_count": 6,
    "optimizer_steps_at_p1_p2": [1, 2],
    "union_mixed_ordered_loss_strictly_decreases_each_step_by_at_least": 1e-8,
    "each_source_hard_loss_p1_and_p2_at_most_raw": True,
    "each_source_hard_loss_p2_improves_from_raw_by_at_least": 1e-8,
    "fragile_282_plus_context34_6_correct_to_wrong_flips": 0,
    "p1_displacement_l2_at_most": 3.10e-5,
    "p2_displacement_l2_at_most": 5.10e-5,
    "displacement_order": "0 < P1 < P2",
    "selection_cache_and_uniqueness_gates": True,
    "selection": "largest_fully_passing_lr_or_none",
    "no_extra_candidates_or_threshold_changes": True,
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def require_regular(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise ValueError(f"{label} must be a single-link regular file")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 drift: {digest}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


def normalize_repo_path(raw_path: str | Path, label: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside repository") from error
    return path


def require_absent_publication_target(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    if not path.parent.is_dir() or path.parent.resolve() != path.parent:
        raise ValueError(f"{label} parent must be an existing symlink-free directory")


def publish_o_excl(path: Path, payload: bytes, mode: int = 0o600) -> dict[str, Any]:
    path = normalize_repo_path(path, "published evidence")
    require_absent_publication_target(path, "published evidence")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise RuntimeError("short evidence write")
            offset += written
        os.fsync(fd)
        observed = os.fstat(fd)
        visible = os.lstat(path)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_ISLNK(visible.st_mode)
            or (observed.st_dev, observed.st_ino, observed.st_size)
            != (visible.st_dev, visible.st_ino, visible.st_size)
            or observed.st_size != len(payload)
        ):
            raise RuntimeError("unsafe evidence publication")
        os.lseek(fd, 0, os.SEEK_SET)
        reloaded = b""
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            reloaded += chunk
        if reloaded != payload:
            raise RuntimeError("published evidence content drift")
        return {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_bytes(reloaded),
            "bytes": len(reloaded),
            "inode": observed.st_ino,
            "device": observed.st_dev,
            "nlink": observed.st_nlink,
        }
    finally:
        os.close(fd)


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{namespace}:{value}".encode("utf-8")
    ).hexdigest()


def episode_round_robin(
    records: Sequence[dict[str, Any]],
    namespace: str,
    *,
    cap: int | None = None,
    fragile_first: bool = False,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episode = str(record.get("episode_id", ""))
        if not episode:
            raise ValueError(f"{namespace} record lacks episode_id")
        groups[episode].append(record)
    for values in groups.values():
        values.sort(
            key=lambda item: (
                float(item["decision_margin"])
                if fragile_first
                else -float(item["decision_margin"]),
                stable_key(namespace + ":row", str(item["line_sha256"])),
            )
        )
        if cap is not None:
            del values[cap:]
    episodes = sorted(
        groups, key=lambda value: stable_key(namespace + ":episode", value)
    )
    flattened: list[dict[str, Any]] = []
    for rank in range(max((len(values) for values in groups.values()), default=0)):
        flattened.extend(
            groups[episode][rank]
            for episode in episodes
            if rank < len(groups[episode])
        )
    return flattened


def validate_profile(profile: dict[str, Any]) -> None:
    if (
        profile.get("schema_version")
        != "ptcg-u468-raw-full-train-margin-profile-v3"
        or profile.get("status") != "completed_train_only"
        or profile.get("split") != "train"
        or profile.get("validation_opened") is not False
    ):
        raise ValueError("train-only margin profile contract mismatch")
    base = profile.get("base")
    if base != {
        "checkpoint": str(U468.relative_to(ROOT)),
        "checkpoint_file_sha256": U468_SHA256,
        "feature_version": "ptcg-selfplay-ppo-terminal01-v1",
        "model_state_sha256": BASE_MODEL_SHA256,
        "update": 468,
    }:
        raise ValueError("raw U468 profile base binding mismatch")
    profiles = profile.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(DATASETS):
        raise ValueError("profile source set mismatch")
    for source, expected in DATA_SHA256.items():
        panel = profiles[source]
        if panel.get("archive_sha256") != expected:
            raise ValueError(f"{source} archive binding mismatch")
        opened = panel.get("opened_members")
        if (
            not isinstance(opened, list)
            or not opened
            or any(not str(member).startswith("train/") for member in opened)
            or panel.get("non_train_members_opened") is not False
        ):
            raise ValueError(f"{source} profile accessed non-train data")


def build_pools(profile: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for source, panel in profile["profiles"].items():
        hard = [
            record
            for record in panel["near_wrong"]
            if int(record["min_count"]) == int(record["max_count"])
            and len(record["expert_order"]) > 0
            and int(record["context"]) != ppo.SKILL_ORDER_CONTEXT
            and record["ordered_correct"] is False
        ]
        fragile = [
            record
            for record in panel["fragile_correct"]
            if len(record["expert_order"]) > 0
            and int(record["context"]) != ppo.SKILL_ORDER_CONTEXT
            and record["ordered_correct"] is True
        ]
        context34 = [
            record
            for record in panel["fragile_correct"]
            if len(record["expert_order"]) > 0
            and int(record["context"]) == ppo.SKILL_ORDER_CONTEXT
            and record["ordered_correct"] is True
        ]
        result[source] = {
            "hard": episode_round_robin(
                hard, source + ":hard", cap=2, fragile_first=False
            ),
            "fragile": episode_round_robin(
                fragile, source + ":fragile", fragile_first=True
            ),
            "c34": episode_round_robin(
                context34, source + ":c34", fragile_first=True
            ),
        }
    return result


def take_distinct(
    pool: Sequence[dict[str, Any]],
    count: int,
    *,
    source: str,
    category: str,
    used_step_episodes: set[str],
    used_cross_step_lines: set[str],
    prior_step_episodes: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    # Prefer episodes absent from all prior steps.  Reuse is admitted only when
    # the finite episode inventory makes it mathematically unavoidable.
    for prefer_new_episode in (True, False):
        for record in pool:
            episode = str(record["episode_id"])
            line_sha256 = str(record["line_sha256"])
            if (
                episode in used_step_episodes
                or line_sha256 in used_cross_step_lines
                or ((episode not in prior_step_episodes) != prefer_new_episode)
            ):
                continue
            selected.append({**record, "source": source, "category": category})
            used_step_episodes.add(episode)
            used_cross_step_lines.add(line_sha256)
            if len(selected) == count:
                return selected
    raise ValueError(
        f"{source}/{category} supplied {len(selected)} of {count} rows under "
        "episode/line uniqueness constraints"
    )


def select_batches(
    profile: dict[str, Any],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    validate_profile(profile)
    pools = build_pools(profile)
    team_names = sorted(
        {
            str(record["team_name"])
            for category in pools["core5"].values()
            for record in category
        }
    )
    if len(team_names) != 5:
        raise ValueError(f"core5 team count drift: {team_names}")

    # Raw SHA identity is global across all three archives, not namespaced by
    # source.  This fails closed even if two archives contain byte-identical
    # decision rows at different member/index locations.
    used_lines: set[str] = set()
    prior_episodes = {source: set() for source in DATASETS}
    step_episode_sets: dict[str, list[set[str]]] = {
        source: [] for source in DATASETS
    }
    batches: list[list[dict[str, Any]]] = []
    per_step: list[dict[str, Any]] = []
    core_cumulative_targets = {team: 0 for team in team_names}

    for step_index in range(STEPS):
        selected: list[dict[str, Any]] = []
        for source in ("flg", "pokemonfan"):
            step_episodes: set[str] = set()
            rows = take_distinct(
                pools[source]["c34"],
                1,
                source=source,
                category="c34",
                used_step_episodes=step_episodes,
                used_cross_step_lines=used_lines,
                prior_step_episodes=prior_episodes[source],
            )
            for category in ("hard", "fragile"):
                rows += take_distinct(
                    pools[source][category],
                    CATEGORY_COUNTS[source][category],
                    source=source,
                    category=category,
                    used_step_episodes=step_episodes,
                    used_cross_step_lines=used_lines,
                    prior_step_episodes=prior_episodes[source],
                )
            selected += rows
            step_episode_sets[source].append(set(step_episodes))
            prior_episodes[source].update(step_episodes)

        # 96 rows = one 20-row team and four 19-row teams.  The extra team
        # rotates, so the two-step aggregate is 39/39/38/38/38.  Hard rows are
        # separately balanced 7/7/6/6/6 then 6/6/7/7/6; fragile is the exact
        # residual after the one context34 row from the step's extra team.
        total_targets = {
            team: 19 + int(team == team_names[step_index]) for team in team_names
        }
        hard_extra = {
            team_names[(2 * step_index) % 5],
            team_names[(2 * step_index + 1) % 5],
        }
        hard_targets = {
            team: 6 + int(team in hard_extra) for team in team_names
        }
        context_team = team_names[step_index]
        fragile_targets = {
            team: total_targets[team]
            - hard_targets[team]
            - int(team == context_team)
            for team in team_names
        }
        if sum(hard_targets.values()) != 32 or sum(fragile_targets.values()) != 63:
            raise RuntimeError("core5 category/team target arithmetic drift")
        core_step_episodes: set[str] = set()
        core_rows = take_distinct(
            [
                row
                for row in pools["core5"]["c34"]
                if str(row["team_name"]) == context_team
            ],
            1,
            source="core5",
            category="c34",
            used_step_episodes=core_step_episodes,
            used_cross_step_lines=used_lines,
            prior_step_episodes=prior_episodes["core5"],
        )
        for category, targets in (
            ("hard", hard_targets),
            ("fragile", fragile_targets),
        ):
            for team in team_names:
                core_rows += take_distinct(
                    [
                        row
                        for row in pools["core5"][category]
                        if str(row["team_name"]) == team
                    ],
                    targets[team],
                    source="core5",
                    category=category,
                    used_step_episodes=core_step_episodes,
                    used_cross_step_lines=used_lines,
                    prior_step_episodes=prior_episodes["core5"],
                )
        selected += core_rows
        step_episode_sets["core5"].append(set(core_step_episodes))
        prior_episodes["core5"].update(core_step_episodes)
        for team, count in total_targets.items():
            core_cumulative_targets[team] += count

        random.Random(SELECTION_SEED + 1009 * (step_index + 1)).shuffle(selected)
        if len(selected) != BATCH_SIZE:
            raise RuntimeError("balanced batch size drift")
        source_counts = {
            source: sum(row["source"] == source for row in selected)
            for source in DATASETS
        }
        category_counts = {
            source: {
                category: sum(
                    row["source"] == source and row["category"] == category
                    for row in selected
                )
                for category in ("hard", "fragile", "c34")
            }
            for source in DATASETS
        }
        if source_counts != SOURCE_COUNTS or category_counts != CATEGORY_COUNTS:
            raise RuntimeError("source/category quota drift")
        for source in DATASETS:
            source_rows = [row for row in selected if row["source"] == source]
            if len({str(row["episode_id"]) for row in source_rows}) != len(source_rows):
                raise RuntimeError(
                    f"step {step_index + 1} {source} repeats episode across buckets"
                )
        if len({row["line_sha256"] for row in selected}) != BATCH_SIZE:
            raise RuntimeError("duplicate raw-row SHA within batch")
        core_team_counts = {
            team: sum(
                row["source"] == "core5" and str(row["team_name"]) == team
                for row in selected
            )
            for team in team_names
        }
        if core_team_counts != total_targets or max(core_team_counts.values()) - min(
            core_team_counts.values()
        ) != 1:
            raise RuntimeError("core5 19/20 team balance drift")
        selection_identity = [
            {
                "source": row["source"],
                "category": row["category"],
                "episode_id": row["episode_id"],
                "team_name": row["team_name"],
                "member": row["member"],
                "line_index": row["line_index"],
                "line_sha256": row["line_sha256"],
            }
            for row in selected
        ]
        per_step.append(
            {
                "step": step_index + 1,
                "source_counts": source_counts,
                "category_counts": category_counts,
                "source_unique_episode_counts": {
                    source: len(step_episode_sets[source][-1]) for source in DATASETS
                },
                "core5_team_counts": core_team_counts,
                "core5_hard_team_counts": hard_targets,
                "core5_fragile_team_counts": fragile_targets,
                "core5_context34_team": context_team,
                "selection_sha256": sha256_bytes(canonical_json(selection_identity)),
            }
        )
        batches.append(selected)

    all_rows = [row for batch in batches for row in batch]
    if len({row["line_sha256"] for row in all_rows}) != STEPS * BATCH_SIZE:
        raise RuntimeError("raw-row SHA repeats across steps")
    episode_overlap = {
        source: len(step_episode_sets[source][0] & step_episode_sets[source][1])
        for source in DATASETS
    }
    available_episodes = {
        source: len(
            {
                str(row["episode_id"])
                for category in pools[source].values()
                for row in category
            }
        )
        for source in DATASETS
    }
    overlap_lower_bound = {
        source: max(0, STEPS * SOURCE_COUNTS[source] - available_episodes[source])
        for source in ("flg", "pokemonfan")
    }
    core_available_by_team = {
        team: len(
            {
                str(row["episode_id"])
                for category in pools["core5"].values()
                for row in category
                if str(row["team_name"]) == team
            }
        )
        for team in team_names
    }
    overlap_lower_bound["core5"] = sum(
        max(0, core_cumulative_targets[team] - core_available_by_team[team])
        for team in team_names
    )
    if episode_overlap != overlap_lower_bound:
        raise RuntimeError(
            f"cross-step episode reuse is not inventory-minimal: "
            f"observed={episode_overlap} lower_bound={overlap_lower_bound}"
        )
    if max(core_cumulative_targets.values()) - min(
        core_cumulative_targets.values()
    ) != 1:
        raise RuntimeError("core5 aggregate team balance gap drift")
    observed_selection_hashes = tuple(
        str(step["selection_sha256"]) for step in per_step
    )
    if observed_selection_hashes != EXPECTED_STEP_SELECTION_SHA256:
        raise RuntimeError(
            "frozen B1/B2 selection hash drift: "
            f"{observed_selection_hashes}"
        )
    summary = {
        "batch_size": BATCH_SIZE,
        "steps": STEPS,
        "source_counts": SOURCE_COUNTS,
        "category_counts": CATEGORY_COUNTS,
        "within_step_source_episode_unique": True,
        "cross_step_raw_row_sha_unique": True,
        "cross_step_unique_raw_rows": STEPS * BATCH_SIZE,
        "available_unique_episodes": available_episodes,
        "cross_step_episode_overlap": episode_overlap,
        "cross_step_episode_overlap_lower_bound": overlap_lower_bound,
        "cross_step_episode_reuse_inventory_minimal": True,
        "core5_team_names": team_names,
        "core5_available_unique_episodes_by_team": core_available_by_team,
        "core5_cumulative_team_targets": core_cumulative_targets,
        "core5_cumulative_team_gap": max(core_cumulative_targets.values())
        - min(core_cumulative_targets.values()),
        "pool_rows": {
            source: {category: len(rows) for category, rows in values.items()}
            for source, values in pools.items()
        },
        "per_step": per_step,
        "expected_step_selection_sha256": list(
            EXPECTED_STEP_SELECTION_SHA256
        ),
    }
    return batches, summary


def load_selected_features(
    selections: Sequence[Sequence[dict[str, Any]]], model_config: dict[str, Any]
) -> list[dict[str, torch.Tensor]]:
    wanted: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        source: {} for source in DATASETS
    }
    for batch in selections:
        for record in batch:
            key = (str(record["member"]), int(record["line_index"]))
            if key in wanted[record["source"]]:
                raise RuntimeError("selected raw row repeats across steps")
            wanted[record["source"]][key] = record

    features: dict[tuple[str, str, int], dict[str, Any]] = {}
    for source, path in DATASETS.items():
        by_member: dict[str, set[int]] = defaultdict(set)
        for member, line_index in wanted[source]:
            if not member.startswith("train/") or not member.endswith(".jsonl"):
                raise ValueError("selection attempted to open a non-train member")
            by_member[member].add(line_index)
        with zipfile.ZipFile(path) as archive:
            for member in sorted(by_member):
                # getinfo fails closed without opening any other member payload.
                archive.getinfo(member)
                remaining = set(by_member[member])
                with archive.open(member) as handle:
                    for line_index, raw in enumerate(handle):
                        if line_index not in remaining:
                            continue
                        record = wanted[source][(member, line_index)]
                        if hashlib.sha256(raw).hexdigest() != record["line_sha256"]:
                            raise ValueError("selected raw line SHA-256 drift")
                        row = orjson.loads(raw)
                        if str(row.get("split", "")) != "train":
                            raise RuntimeError("selected row is not train split")
                        if (
                            str(row.get("episode_id", "")) != str(record["episode_id"])
                            or str(row.get("team_name", "")) != str(record["team_name"])
                        ):
                            raise RuntimeError("selected episode/team metadata drift")
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
                            raise RuntimeError("selected row no longer featurizes")
                        expert = [int(value) for value in row.get("action", [])]
                        if (
                            expert != [int(value) for value in record["expert_order"]]
                            or int(feature["context"]) != int(record["context"])
                            or int(feature["min_count"]) != int(record["min_count"])
                            or int(feature["max_count"]) != int(record["max_count"])
                        ):
                            raise RuntimeError("selected row decision metadata drift")
                        feature["action_sequence"] = expert
                        if int(feature["context"]) == ppo.SKILL_ORDER_CONTEXT:
                            feature["sample_weight"] = CONTEXT34_SAMPLE_WEIGHT
                        features[(source, member, line_index)] = feature
                        remaining.remove(line_index)
                        if not remaining:
                            break
                if remaining:
                    raise RuntimeError(f"selected lines missing from {member}")

    result: list[dict[str, torch.Tensor]] = []
    for selected in selections:
        rows = [
            features[(row["source"], str(row["member"]), int(row["line_index"]))]
            for row in selected
        ]
        batch = bc.collate_decisions(
            rows,
            max_state_entities=int(model_config["max_state_entities"]),
            entity_fields=int(model_config["entity_fields"]),
            option_fields=int(model_config["option_fields"]),
        )
        if int(batch["action_counts"].shape[0]) != BATCH_SIZE:
            raise RuntimeError("collated batch size drift")
        context_mask = batch["contexts"] == ppo.SKILL_ORDER_CONTEXT
        if int(context_mask.sum()) != 3:
            raise RuntimeError("collated context34 physical quota drift")
        if not torch.equal(
            batch["sample_weights"][context_mask],
            torch.full_like(
                batch["sample_weights"][context_mask], CONTEXT34_SAMPLE_WEIGHT
            ),
        ):
            raise RuntimeError("context34 one-third sample weight drift")
        result.append(batch)
    return result


def model_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"non-tensor model state entry: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_raw_u468(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    parent = torch.load(U468, map_location="cpu", weights_only=False)
    general = torch.load(GENERAL_BC, map_location="cpu", weights_only=False)
    if int(parent.get("update", -1)) != 468:
        raise ValueError("raw parent update drift")
    model = ppo.instantiate_model_from_checkpoint(parent, general, device)
    model.load_state_dict(parent["model_state_dict"])
    if ppo.model_state_sha256(model) != BASE_MODEL_SHA256:
        raise ValueError("raw U468 runtime model SHA-256 drift")
    return model, parent


def configure_actor6(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    for name in ACTOR_NAMES:
        if name not in named:
            raise ValueError(f"missing actor6 parameter: {name}")
        named[name].requires_grad_(True)
    observed = [name for name, value in model.named_parameters() if value.requires_grad]
    if tuple(observed) != ACTOR_NAMES:
        raise RuntimeError(f"actor6 scope drift: {observed}")
    return {name: named[name] for name in ACTOR_NAMES}


def ordered_nll_per_row(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Return the exact ordered Plackett-Luce NLL used by PPO BC replay."""
    logits = outputs["policy_logits"].float()
    option_mask = batch["option_mask"].bool()
    action_counts = batch["action_counts"]
    sequences = batch["action_sequences"]
    selected = torch.zeros_like(option_mask)
    result = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    for sequence_step in range(sequences.shape[1]):
        active = sequence_step < action_counts
        if not active.any():
            break
        chosen = sequences[:, sequence_step]
        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        active_chosen = chosen[active]
        if not (
            option_mask[active_rows, active_chosen]
            & ~selected[active_rows, active_chosen]
        ).all():
            raise ValueError("illegal/duplicate expert action in audit cache")
        allowed = option_mask & ~selected
        log_probs = torch.log_softmax(logits.masked_fill(~allowed, -1e9), dim=1)
        safe_chosen = chosen.clamp(0, logits.shape[1] - 1)
        chosen_log_prob = log_probs.gather(1, safe_chosen.unsqueeze(1)).squeeze(1)
        result -= torch.where(active, chosen_log_prob, torch.zeros_like(result))
        selected.scatter_(1, safe_chosen.unsqueeze(1), active.unsqueeze(1))
    return result


def evaluate_selected_union(
    model: torch.nn.Module,
    cache: Sequence[dict[str, torch.Tensor]],
    selections: Sequence[Sequence[dict[str, Any]]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, bool]]:
    model.eval()
    entries: list[dict[str, Any]] = []
    correct_by_line: dict[str, bool] = {}
    with torch.no_grad():
        for cpu_batch, identities in zip(cache, selections):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in cpu_batch.items()
            }
            outputs = ppo.model_forward(model, batch, device)
            per_row = ordered_nll_per_row(outputs, batch)
            predictions, _, _, _ = ppo.sample_ordered_actions(
                outputs,
                batch,
                deterministic=True,
                canonicalize_order=False,
            )
            effective_weights = batch["sample_weights"].float() * torch.where(
                batch["contexts"] == ppo.SKILL_ORDER_CONTEXT,
                torch.full_like(batch["sample_weights"].float(), ORDER_CONTEXT_WEIGHT),
                torch.ones_like(batch["sample_weights"].float()),
            )
            for index, identity in enumerate(identities):
                action_count = int(batch["action_counts"][index])
                expert = [
                    int(value)
                    for value in batch["action_sequences"][index, :action_count].tolist()
                ]
                predicted = [int(value) for value in predictions[index]]
                line_sha256 = str(identity["line_sha256"])
                correct = predicted == expert
                if line_sha256 in correct_by_line:
                    raise RuntimeError("evaluation raw-row SHA repeats")
                correct_by_line[line_sha256] = correct
                entries.append(
                    {
                        "source": str(identity["source"]),
                        "category": str(identity["category"]),
                        "line_sha256": line_sha256,
                        "loss": float(per_row[index].detach().cpu()),
                        "weight": float(effective_weights[index].detach().cpu()),
                        "ordered_correct": correct,
                    }
                )

    def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        weight = sum(float(row["weight"]) for row in rows)
        if not rows or weight <= 0.0:
            raise RuntimeError("empty/nonpositive evaluation group")
        return {
            "rows": len(rows),
            "effective_weight": weight,
            "ordered_loss": sum(
                float(row["loss"]) * float(row["weight"]) for row in rows
            )
            / weight,
            "ordered_correct": sum(bool(row["ordered_correct"]) for row in rows),
        }

    by_source: dict[str, dict[str, Any]] = {}
    for source in DATASETS:
        source_rows = [row for row in entries if row["source"] == source]
        by_source[source] = {"all": aggregate(source_rows)}
        for category in ("hard", "fragile", "c34"):
            by_source[source][category] = aggregate(
                [row for row in source_rows if row["category"] == category]
            )
    retention_rows = [
        row for row in entries if row["category"] in {"fragile", "c34"}
    ]
    fragile_rows = [row for row in entries if row["category"] == "fragile"]
    context_rows = [row for row in entries if row["category"] == "c34"]
    report = {
        "mixed_ordered_loss": aggregate(entries)["ordered_loss"],
        "rows": len(entries),
        "by_source_and_bucket": by_source,
        "retention": {
            "rows": len(retention_rows),
            "ordered_correct": sum(
                bool(row["ordered_correct"]) for row in retention_rows
            ),
            "fragile_rows": len(fragile_rows),
            "fragile_ordered_correct": sum(
                bool(row["ordered_correct"]) for row in fragile_rows
            ),
            "context34_rows": len(context_rows),
            "context34_ordered_correct": sum(
                bool(row["ordered_correct"]) for row in context_rows
            ),
        },
    }
    if len(entries) != STEPS * BATCH_SIZE:
        raise RuntimeError("union evaluation row count drift")
    return report, correct_by_line


def displacement_report(
    base_state: Mapping[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, Any]:
    current = repair.clone_model_state(model)
    changed = repair.changed_tensor_names(dict(base_state), current)
    per_tensor: dict[str, dict[str, float]] = {}
    total_square = 0.0
    max_abs = 0.0
    for name in ACTOR_NAMES:
        delta = current[name] - base_state[name]
        square = float(delta.double().square().sum())
        tensor_max = float(delta.abs().max())
        total_square += square
        max_abs = max(max_abs, tensor_max)
        per_tensor[name] = {
            "l2": math.sqrt(square),
            "max_abs": tensor_max,
        }
    return {
        "changed_parameter_names": changed,
        "changed_scope_exact_actor6": changed == sorted(ACTOR_NAMES),
        "l2": math.sqrt(total_square),
        "max_abs": max_abs,
        "per_tensor": per_tensor,
        "model_state_finite": repair.finite_nested(current),
    }


def shadow_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, torch.nn.Parameter],
    cpu_batch: dict[str, torch.Tensor],
    device: torch.device,
    expected_step: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    optimizer.param_groups[0]["lr"] = float(optimizer.param_groups[0]["lr"])
    model.zero_grad(set_to_none=True)
    batch = {
        key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()
    }
    outputs = ppo.model_forward(model, batch, device)
    loss, parts = ppo.bc_expert_actor_loss(
        outputs,
        batch,
        loss_mode="ordered",
        order_context_weight=ORDER_CONTEXT_WEIGHT,
        non_context34_fixed_multi_action_order_weight=1.0,
    )
    loss.backward()
    gradient_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
    )
    gradients: dict[str, torch.Tensor] = {}
    per_tensor: dict[str, dict[str, float]] = {}
    for name, parameter in parameters.items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"missing/nonfinite gradient: {name}")
        gradient = parameter.grad.detach().cpu().clone()
        gradients[name] = gradient
        per_tensor[name] = {
            "l2": math.sqrt(float(gradient.double().square().sum())),
            "max_abs": float(gradient.abs().max()),
            "nonzero_elements": int(torch.count_nonzero(gradient)),
        }
    preclip_norm = torch.nn.utils.clip_grad_norm_(
        list(parameters.values()), MAX_GRAD_NORM
    )
    preclip_value = float(preclip_norm.detach().cpu())
    clip_factor = min(1.0, MAX_GRAD_NORM / (preclip_value + 1e-6))
    postclip_norm = math.sqrt(
        sum(
            float(parameter.grad.detach().double().square().sum())
            for parameter in parameters.values()
            if parameter.grad is not None
        )
    )
    optimizer.step()
    optimizer_state = optimizer.state_dict()
    optimizer_steps = repair.optimizer_steps(optimizer_state)
    if len(optimizer_state["state"]) != len(ACTOR_NAMES):
        raise RuntimeError("fresh AdamW state count drift")
    if len(optimizer_steps) != len(ACTOR_NAMES) or set(optimizer_steps) != {
        expected_step
    }:
        raise RuntimeError("fresh AdamW step counter drift")
    metrics = {
        "step": expected_step,
        "loss": float(loss.detach().cpu()),
        "loss_parts": {
            key: float(value.detach().cpu()) for key, value in parts.items()
        },
        "gradient_parameter_names": gradient_names,
        "gradient_scope_exact_actor6": gradient_names == sorted(ACTOR_NAMES),
        "all_six_gradient_tensors_nonzero": all(
            row["nonzero_elements"] > 0 for row in per_tensor.values()
        ),
        "per_tensor_preclip_gradient": per_tensor,
        "preclip_gradient_l2": preclip_value,
        "clip_max_norm": MAX_GRAD_NORM,
        "clip_factor": clip_factor,
        "postclip_gradient_l2": postclip_norm,
        "optimizer_state_count": len(optimizer_state["state"]),
        "optimizer_steps": sorted(set(optimizer_steps)),
        "optimizer_state_finite": repair.finite_nested(optimizer_state),
    }
    return metrics, gradients


def gradient_cosines(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        lhs = left.detach().double().reshape(-1)
        rhs = right.detach().double().reshape(-1)
        denominator = float(lhs.norm() * rhs.norm())
        if denominator <= 0.0:
            raise RuntimeError("zero gradient prevents cosine audit")
        return float(torch.dot(lhs, rhs) / denominator)

    first_all = torch.cat([first[name].reshape(-1) for name in ACTOR_NAMES])
    second_all = torch.cat([second[name].reshape(-1) for name in ACTOR_NAMES])
    return {
        "all_actor6": cosine(first_all, second_all),
        "per_tensor": {
            name: cosine(first[name], second[name]) for name in ACTOR_NAMES
        },
    }


def correct_to_wrong_flips(
    raw_correct: Mapping[str, bool],
    endpoint_correct: Mapping[str, bool],
    selections: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    retention = [
        row
        for batch in selections
        for row in batch
        if row["category"] in {"fragile", "c34"}
    ]
    fragile = [row for row in retention if row["category"] == "fragile"]
    context34 = [row for row in retention if row["category"] == "c34"]

    def flips(rows: Sequence[dict[str, Any]]) -> int:
        return sum(
            bool(raw_correct[str(row["line_sha256"])])
            and not bool(endpoint_correct[str(row["line_sha256"])])
            for row in rows
        )

    return {
        "retention_rows": len(retention),
        "fragile_rows": len(fragile),
        "context34_rows": len(context34),
        "raw_retention_correct": sum(
            bool(raw_correct[str(row["line_sha256"])]) for row in retention
        ),
        "fragile_correct_to_wrong": flips(fragile),
        "context34_correct_to_wrong": flips(context34),
        "total_correct_to_wrong": flips(retention),
    }


def run_shadow_candidate(
    learning_rate: float,
    cache: Sequence[dict[str, torch.Tensor]],
    selections: Sequence[Sequence[dict[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    model, _ = load_raw_u468(device)
    parameters = configure_actor6(model)
    base_state = repair.clone_model_state(model)
    raw_evaluation, raw_correct = evaluate_selected_union(
        model, cache, selections, device
    )
    if raw_evaluation["retention"] != {
        "rows": 288,
        "ordered_correct": 288,
        "fragile_rows": 282,
        "fragile_ordered_correct": 282,
        "context34_rows": 6,
        "context34_ordered_correct": 6,
    }:
        raise RuntimeError("raw U468 selected retention correctness drift")
    optimizer = torch.optim.AdamW(
        list(parameters.values()),
        lr=learning_rate,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    if optimizer.state:
        raise RuntimeError("fresh shadow optimizer unexpectedly has state")
    steps: list[dict[str, Any]] = []
    gradients: list[dict[str, torch.Tensor]] = []
    evaluations = {"raw": raw_evaluation}
    endpoint_correctness: dict[str, dict[str, bool]] = {}
    for step, batch in enumerate(cache, start=1):
        training_metrics, gradient = shadow_training_step(
            model, optimizer, parameters, batch, device, step
        )
        gradients.append(gradient)
        evaluation, correctness = evaluate_selected_union(
            model, cache, selections, device
        )
        label = f"p{step}"
        evaluations[label] = evaluation
        endpoint_correctness[label] = correctness
        steps.append(
            {
                **training_metrics,
                "displacement_from_raw": displacement_report(base_state, model),
                "retention_flips_from_raw": correct_to_wrong_flips(
                    raw_correct, correctness, selections
                ),
            }
        )
    final_state = repair.clone_model_state(model)
    final_model_state_finite = repair.finite_nested(final_state)
    frozen_names = sorted(set(base_state) - set(ACTOR_NAMES))
    frozen_unchanged = all(
        torch.equal(final_state[name], base_state[name]) for name in frozen_names
    )
    grad_cosine = gradient_cosines(gradients[0], gradients[1])

    raw_loss = float(evaluations["raw"]["mixed_ordered_loss"])
    p1_loss = float(evaluations["p1"]["mixed_ordered_loss"])
    p2_loss = float(evaluations["p2"]["mixed_ordered_loss"])
    union_loss_gate = raw_loss - p1_loss >= 1e-8 and p1_loss - p2_loss >= 1e-8
    hard_loss_gate_by_source: dict[str, dict[str, Any]] = {}
    for source in DATASETS:
        raw_hard = float(
            evaluations["raw"]["by_source_and_bucket"][source]["hard"][
                "ordered_loss"
            ]
        )
        p1_hard = float(
            evaluations["p1"]["by_source_and_bucket"][source]["hard"][
                "ordered_loss"
            ]
        )
        p2_hard = float(
            evaluations["p2"]["by_source_and_bucket"][source]["hard"][
                "ordered_loss"
            ]
        )
        hard_loss_gate_by_source[source] = {
            "raw": raw_hard,
            "p1": p1_hard,
            "p2": p2_hard,
            "p1_at_most_raw": p1_hard <= raw_hard,
            "p2_at_most_raw": p2_hard <= raw_hard,
            "p2_improvement_from_raw": raw_hard - p2_hard,
            "pass": p1_hard <= raw_hard
            and p2_hard <= raw_hard
            and raw_hard - p2_hard >= 1e-8,
        }
    p1_displacement = float(steps[0]["displacement_from_raw"]["l2"])
    p2_displacement = float(steps[1]["displacement_from_raw"]["l2"])
    gate_checks = {
        "all_finite": repair.finite_nested(
            {
                "evaluations": evaluations,
                "steps": steps,
                "gradient_cosine": grad_cosine,
                "final_model_state_finite": final_model_state_finite,
            }
        ),
        "gradient_scope_exact_actor6": all(
            step["gradient_scope_exact_actor6"] for step in steps
        ),
        "six_actor_tensors_each_nonzero": all(
            step["all_six_gradient_tensors_nonzero"] for step in steps
        ),
        "changed_scope_exact_actor6": all(
            step["displacement_from_raw"]["changed_scope_exact_actor6"]
            for step in steps
        )
        and frozen_unchanged,
        "fresh_optimizer_state_count_6": all(
            step["optimizer_state_count"] == 6 for step in steps
        ),
        "optimizer_steps_exact_1_2": [step["optimizer_steps"] for step in steps]
        == [[1], [2]],
        "union_mixed_ordered_loss": union_loss_gate,
        "each_source_hard_loss": all(
            row["pass"] for row in hard_loss_gate_by_source.values()
        ),
        "zero_retention_correct_to_wrong_flips": all(
            step["retention_flips_from_raw"]["total_correct_to_wrong"] == 0
            for step in steps
        ),
        "p1_displacement_l2": 0.0 < p1_displacement <= 3.10e-5,
        "p2_displacement_l2": p1_displacement < p2_displacement <= 5.10e-5,
        "selection_cache_and_uniqueness": True,
    }
    return {
        "learning_rate": learning_rate,
        "lr_scale_vs_ppo_actor": learning_rate / BASE_ACTOR_LEARNING_RATE,
        "optimizer": {
            "name": "AdamW",
            "fresh": True,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "trajectory": "actual_ram_only_continuous_b1_then_b2",
        "evaluations": evaluations,
        "steps": steps,
        "b1_b2_preclip_gradient_cosine": grad_cosine,
        "frozen_tensors_unchanged": frozen_unchanged,
        "final_model_state_finite": final_model_state_finite,
        "hard_loss_gate_by_source": hard_loss_gate_by_source,
        "union_loss_improvements": {
            "raw_minus_p1": raw_loss - p1_loss,
            "p1_minus_p2": p1_loss - p2_loss,
        },
        "gate_checks": gate_checks,
        "fully_passes": all(gate_checks.values()),
    }


def expected_shadow_contract(
    *,
    runner_sha256: str,
    cache_audit: Mapping[str, Any],
    device: str,
    result_output: Path,
) -> dict[str, Any]:
    return {
        "runner": {
            "path": str(
                normalize_repo_path(__file__, "runner").relative_to(ROOT)
            ),
            "sha256": runner_sha256,
        },
        "branch": BRANCH,
        "device": device,
        "execution_seed": EXECUTION_SEED,
        "training_base": {
            "path": str(U468.relative_to(ROOT)),
            "checkpoint_sha256": U468_SHA256,
            "runtime_model_state_sha256": BASE_MODEL_SHA256,
            "update": 468,
        },
        "selection_profile": {
            "path": str(PROFILE.relative_to(ROOT)),
            "sha256": PROFILE_SHA256,
        },
        "datasets": {
            source: {
                "path": str(path.relative_to(ROOT)),
                "sha256": DATA_SHA256[source],
            }
            for source, path in DATASETS.items()
        },
        "selection": {
            "seed": SELECTION_SEED,
            "step_selection_sha256": list(EXPECTED_STEP_SELECTION_SHA256),
            "global_bare_raw_line_sha_unique": True,
            "within_step_episode_id_unique_per_source": True,
            "core5_team_total_per_step": "19_or_20",
            "core5_aggregate_team_gap": 1,
        },
        "cache": {
            "cache_sha256": str(cache_audit["cache_sha256"]),
            "batch_sha256": list(cache_audit["batch_sha256"]),
            "rows": int(cache_audit["rows"]),
            "batches": int(cache_audit["batches"]),
            "physical_context34_rows": int(
                cache_audit["physical_context34_rows"]
            ),
            "effective_context34_sample_weight": float(
                cache_audit["effective_context34_sample_weight"]
            ),
        },
        "trajectory": {
            "kind": "actual_ram_only_continuous_b1_then_b2",
            "batch_size": BATCH_SIZE,
            "steps": STEPS,
            "endpoint_steps": list(ENDPOINT_STEPS),
            "source_counts_per_step": SOURCE_COUNTS,
            "category_counts_per_step": CATEGORY_COUNTS,
            "context34_row_sample_weight": CONTEXT34_SAMPLE_WEIGHT,
            "order_context_weight": ORDER_CONTEXT_WEIGHT,
            "actor_parameter_names": list(ACTOR_NAMES),
        },
        "optimizer": {
            "name": "AdamW",
            "fresh_per_candidate": True,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "foreach": False,
            "fused": False,
        },
        "learning_rate_candidates": list(LR_CANDIDATES),
        "target_learning_rate": LEARNING_RATE,
        "gates": LR_GATES,
        "selection_rule": "run_all_then_largest_fully_passing_lr_or_none",
        "attempt_marker": str(SHADOW_ATTEMPT_MARKER.relative_to(ROOT)),
        "result_output": str(result_output.relative_to(ROOT)),
        "scope": {
            "train_only": True,
            "validation": False,
            "training_artifact": False,
            "checkpoint": False,
            "optimizer_artifact": False,
            "submission": False,
        },
    }


def load_json_object_without_duplicate_keys(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise ValueError(f"{label} contains nonfinite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def validate_shadow_preregistration(
    *,
    path: Path,
    expected_sha256: str,
    runner_sha256: str,
    cache_audit: Mapping[str, Any],
    device: str,
    result_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        len(expected_sha256) != 64
        or expected_sha256 != expected_sha256.lower()
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected preregistration SHA-256 must be 64 lowercase hex")
    evidence = require_regular(path, expected_sha256, "shadow preregistration")
    preregistration = load_json_object_without_duplicate_keys(
        path, "shadow preregistration"
    )
    expected_contract = expected_shadow_contract(
        runner_sha256=runner_sha256,
        cache_audit=cache_audit,
        device=device,
        result_output=result_output,
    )
    if (
        preregistration.get("schema_version") != SHADOW_PREREGISTRATION_SCHEMA
        or preregistration.get("status") != "locked_before_shadow"
        or preregistration.get("shadow_contract") != expected_contract
    ):
        raise ValueError("shadow preregistration exact contract mismatch")
    return evidence, expected_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "selection-audit",
            "cache-audit",
            "shadow-contract",
            "shadow-gradient-audit",
        ),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--result-output")
    parser.add_argument("--preregistration")
    parser.add_argument("--expected-preregistration-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    result_requested = args.result_output is not None
    preregistration_requested = args.preregistration is not None
    preregistration_sha_requested = (
        args.expected_preregistration_sha256 is not None
    )
    evidence_mode = (
        result_requested
        and preregistration_requested
        and preregistration_sha_requested
    )
    if args.mode == "shadow-contract":
        if (
            not result_requested
            or preregistration_requested
            or preregistration_sha_requested
        ):
            raise ValueError(
                "shadow-contract requires only --result-output; it forbids "
                "--preregistration and --expected-preregistration-sha256"
            )
        if args.device != "cuda":
            raise ValueError("shadow-contract must preregister the CUDA trajectory")
    elif args.mode == "shadow-gradient-audit":
        if not evidence_mode:
            raise ValueError(
                "actual shadow trajectory requires all three evidence arguments"
            )
        if args.device != "cuda":
            raise ValueError("formal hash-bound shadow selection is CUDA-only")
    elif result_requested or preregistration_requested or preregistration_sha_requested:
        raise ValueError("evidence arguments are exclusive to shadow modes")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)
    if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
        raise FileExistsError(ATTEMPT_MARKER)

    runner_path = normalize_repo_path(__file__, "runner")
    runner_sha256 = sha256_file(runner_path)
    runner_evidence = require_regular(runner_path, runner_sha256, "runner")
    profile_evidence = require_regular(PROFILE, PROFILE_SHA256, "train profile")
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    selections, selection_summary = select_batches(profile)
    common = {
        "schema_version": SCHEMA,
        "status": "zero_write_audit_passed",
        "mode": args.mode,
        "branch": BRANCH,
        "runner": runner_evidence,
        "training_base": {
            "kind": "raw_full_u468",
            "path": str(U468.relative_to(ROOT)),
            "checkpoint_sha256": U468_SHA256,
            "runtime_model_state_sha256": BASE_MODEL_SHA256,
        },
        "selection_profile": {
            **profile_evidence,
            "margin_model_state_sha256": BASE_MODEL_SHA256,
            "margin_model_equals_training_base": True,
        },
        "selection": selection_summary,
        "train_only": True,
        "validation_member_payloads_opened": False,
        "formal_training_optimizer_steps": 0,
        "checkpoint_writes": 0,
        "optimizer_artifact_writes": 0,
        "training_artifact_writes": 0,
        "writes_performed": False,
        "formal_training_mode_available": False,
        "hash_bound_shadow_evidence_mode_available": True,
    }
    if args.mode == "selection-audit":
        print(json.dumps(common, ensure_ascii=False, sort_keys=True))
        return

    input_evidence = {
        "u468": require_regular(U468, U468_SHA256, "raw U468"),
        "general_bc": require_regular(GENERAL_BC, GENERAL_BC_SHA256, "general BC"),
        "datasets": {
            source: require_regular(path, DATA_SHA256[source], source)
            for source, path in DATASETS.items()
        },
        "dependencies": {
            str(path.relative_to(ROOT)): require_regular(path, digest, str(path))
            for path, digest in DEPENDENCY_SHA256.items()
        },
    }
    parent = torch.load(U468, map_location="cpu", weights_only=False)
    cache = load_selected_features(selections, parent["model_config"])
    cache_sha256, batch_sha256 = repair.replay_cache_manifest(cache)
    cache_audit = {
        "cache_sha256": cache_sha256,
        "batch_sha256": batch_sha256,
        "rows": sum(int(batch["action_counts"].shape[0]) for batch in cache),
        "batches": len(cache),
        "physical_context34_rows": sum(
            int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum()) for batch in cache
        ),
        "effective_context34_sample_weight": sum(
            float(
                batch["sample_weights"][
                    batch["contexts"] == ppo.SKILL_ORDER_CONTEXT
                ].sum()
            )
            for batch in cache
        ),
    }
    if cache_sha256 != EXPECTED_CACHE_SHA256:
        raise RuntimeError(f"frozen replay cache SHA-256 drift: {cache_sha256}")
    if tuple(batch_sha256) != EXPECTED_BATCH_SHA256:
        raise RuntimeError(f"frozen B1/B2 cache SHA-256 drift: {batch_sha256}")
    if cache_audit["rows"] != STEPS * BATCH_SIZE:
        raise RuntimeError("cache row count drift")
    if cache_audit["physical_context34_rows"] != 6 or not math.isclose(
        cache_audit["effective_context34_sample_weight"],
        2.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError("cache context34 weight contract drift")
    if args.mode == "cache-audit":
        print(
            json.dumps(
                common | {"inputs": input_evidence, "cache": cache_audit},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if args.mode == "shadow-contract":
        contract_result_output = normalize_repo_path(
            args.result_output, "shadow result"
        )
        if contract_result_output == SHADOW_ATTEMPT_MARKER:
            raise ValueError("shadow result path collides with attempt marker")
        require_absent_publication_target(contract_result_output, "shadow result")
        require_absent_publication_target(
            SHADOW_ATTEMPT_MARKER, "shadow attempt marker"
        )
        preregistration_template = {
            "schema_version": SHADOW_PREREGISTRATION_SCHEMA,
            "status": "locked_before_shadow",
            "shadow_contract": expected_shadow_contract(
                runner_sha256=runner_sha256,
                cache_audit=cache_audit,
                device=args.device,
                result_output=contract_result_output,
            ),
        }
        print(
            json.dumps(
                preregistration_template,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    if not evidence_mode:
        raise RuntimeError("unregistered actual shadow trajectory is disabled")
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    random.seed(EXECUTION_SEED)
    torch.manual_seed(EXECUTION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EXECUTION_SEED)
    reference_model, _ = load_raw_u468(device)
    reference_hash_before = ppo.model_state_sha256(reference_model)
    if reference_hash_before != BASE_MODEL_SHA256:
        raise RuntimeError("shadow reference model hash drift")

    result_output = normalize_repo_path(args.result_output, "shadow result")
    preregistration_path = normalize_repo_path(
        args.preregistration, "shadow preregistration"
    )
    if result_output in {preregistration_path, SHADOW_ATTEMPT_MARKER}:
        raise ValueError("shadow result path collides with a control-plane file")
    require_absent_publication_target(result_output, "shadow result")
    require_absent_publication_target(
        SHADOW_ATTEMPT_MARKER, "shadow attempt marker"
    )
    preregistration_evidence, shadow_contract = validate_shadow_preregistration(
        path=preregistration_path,
        expected_sha256=args.expected_preregistration_sha256,
        runner_sha256=runner_sha256,
        cache_audit=cache_audit,
        device=args.device,
        result_output=result_output,
    )
    marker_payload = {
        "schema_version": (
            "ptcg-u468-raw-trainhard-actor6-balanced-shadow-attempt-v1"
        ),
        "status": "locked_before_shadow",
        "branch": BRANCH,
        "preregistration": preregistration_evidence,
        "shadow_contract": shadow_contract,
        "training_artifact_writes_before_lock": 0,
        "checkpoint_writes_before_lock": 0,
        "validation_member_payloads_opened": False,
    }
    marker_evidence = publish_o_excl(
        SHADOW_ATTEMPT_MARKER, canonical_json(marker_payload)
    )

    candidates: list[dict[str, Any]] = []
    for learning_rate in LR_CANDIDATES:
        random.seed(EXECUTION_SEED)
        torch.manual_seed(EXECUTION_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(EXECUTION_SEED)
        candidates.append(
            run_shadow_candidate(learning_rate, cache, selections, device)
        )
    if [candidate["learning_rate"] for candidate in candidates] != list(
        LR_CANDIDATES
    ):
        raise RuntimeError("shadow candidate set/order drift")

    passing_learning_rates = [
        float(candidate["learning_rate"])
        for candidate in candidates
        if candidate["fully_passes"] is True
    ]
    selected_learning_rate = (
        max(passing_learning_rates) if passing_learning_rates else None
    )
    reference_hash_after = ppo.model_state_sha256(reference_model)
    if reference_hash_after != reference_hash_before:
        raise RuntimeError("shadow candidates mutated the raw reference model instance")
    target_result = next(
        candidate
        for candidate in candidates
        if candidate["learning_rate"] == LEARNING_RATE
    )
    shadow_result = {
        "status": "completed_actual_ram_only_trajectory",
        "decision": "GO" if selected_learning_rate is not None else "NO_GO",
        "candidate_learning_rates": list(LR_CANDIDATES),
        "all_candidates_evaluated_before_selection": True,
        "selection_rule": "largest_fully_passing_lr_or_none",
        "selected_learning_rate": selected_learning_rate,
        "target_learning_rate": LEARNING_RATE,
        "target_learning_rate_fully_passes": target_result["fully_passes"],
        "candidates": candidates,
        "reference_model_instance_unchanged": True,
        "reference_model_state_sha256_before": reference_hash_before,
        "reference_model_state_sha256_after": reference_hash_after,
        "shadow_optimizer_step_calls": STEPS * len(LR_CANDIDATES),
        "formal_training_optimizer_step_calls": 0,
        "checkpoint_writes": 0,
        "optimizer_artifact_writes": 0,
        "training_artifact_writes": 0,
        "validation_member_payloads_opened": False,
    }
    result_payload = common | {
        "status": "shadow_trajectory_audit_completed",
        "inputs": input_evidence,
        "cache": cache_audit,
        "shadow": shadow_result,
        "formal_shadow_evidence_mode": True,
        "preregistration": preregistration_evidence,
        "shadow_contract": shadow_contract,
        "shadow_attempt_marker": marker_evidence,
        "writes_performed": True,
        "evidence_writes": 2,
    }
    if not repair.finite_nested(result_payload):
        raise FloatingPointError("nonfinite value in shadow result")

    result_evidence = publish_o_excl(
        result_output, canonical_json(result_payload)
    )
    print(
        json.dumps(
            {
                "status": "shadow_result_published",
                "decision": shadow_result["decision"],
                "selected_learning_rate": selected_learning_rate,
                "attempt_marker": marker_evidence,
                "result": result_evidence,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
