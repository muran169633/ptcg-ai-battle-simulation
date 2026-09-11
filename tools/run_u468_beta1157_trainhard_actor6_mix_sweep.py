#!/usr/bin/env python3
"""Train a train-only hard-negative actor6 sweep from U468 beta=1.157.

The cache is derived exclusively from a frozen train-margin profile.  Each
batch contains FLG near-boundary errors plus PokemonFan/core5 fragile-correct
retention rows.  Validation archives are never opened by this program.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import stat
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import orjson
import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_u468_p12_delta_direction_sweep as transport  # noqa: E402
import run_ppo_bc_repair as repair  # noqa: E402
import run_u468_beta1157_flg_actorhead_freshlr225e7_sweep as prior  # noqa: E402
import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-beta1157-trainhard-actor6-mix-sweep-v1"
BRANCH = "ppo_u468_beta1157_trainhard_actor6_mix102_77_77_p1p2_design202608101"
DESIGN_PATH = ROOT / f"artifacts/{BRANCH}.design_preregistration.json"
OUTPUT_ROOT = ROOT / f"artifacts/{BRANCH}"
ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-attempt.json"
PROFILE = ROOT / "artifacts/u468_beta1157_train_margin_profile_v2_20260802.json"
PROFILE_SHA256 = "b28de13ab014d73dea8d9c10fdcf2810ebb4a565b31ac29b6129d37ec88747ac"

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
    ROOT / "tools/run_u468_beta1157_flg_actorhead_freshlr225e7_sweep.py": (
        "1f9ca7e545595af8f8f609c60185314dbc837a7a29c8b65dd10f4ef6d85c7870"
    ),
    ROOT / "tools/train_bc_orbit.py": (
        "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
    ),
    ROOT / "tools/profile_u468_beta1157_train_margins.py": (
        "f1018994c48ffadcbc5277f0dfd9233086693f90bf30f5bf67462965b4e18142"
    ),
}

BETA = 1.157
BASE_MODEL_SHA256 = prior.BASE_MODEL_SHA256
ACTOR_NAMES = tuple(prior.MUTABLE_NAMES[:6])
LEARNING_RATE = 2.25e-7
BASE_ACTOR_LEARNING_RATE = 3.6e-5
LR_SCALE = LEARNING_RATE / BASE_ACTOR_LEARNING_RATE
WEIGHT_DECAY = 1e-4
ADAM_EPS = 1e-5
MAX_GRAD_NORM = 0.5
SELECTION_SEED = 202608101
EXECUTION_SEED = 202608102
BATCH_SIZE = 256
STEPS = 2
ENDPOINT_STEPS = (1, 2)
SOURCE_COUNTS = {"flg": 102, "pokemonfan": 77, "core5": 77}
CONTEXT34_COUNTS = {"flg": 1, "pokemonfan": 1, "core5": 1}
CONTEXT34_SAMPLE_WEIGHT = 1.0 / 3.0
EXPECTED_CACHE_SHA256 = "01c80a357222768667ab01ce400879eef9422a79340feafdc33ea4a93cec051c"
EXPECTED_BATCH_SHA256: tuple[str, ...] = (
    "23290cb0312d1509881dde8aaf6ceeba1afb86180bda78a69274439a4745f21d",
    "631c9684c1bc4a70dfca73380908808572561c3ef39a891abea70789ee62e403",
)


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
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def regular_evidence(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    observed_stat = os.lstat(path)
    if (
        stat.S_ISLNK(observed_stat.st_mode)
        or not stat.S_ISREG(observed_stat.st_mode)
        or observed_stat.st_nlink != 1
    ):
        raise ValueError(f"{label} must be a single-link regular file")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{label} hash drift: expected {expected_sha256}, got {observed_sha256}"
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": observed_sha256,
        "bytes": observed_stat.st_size,
        "mode": oct(observed_stat.st_mode & 0o777),
        "inode": observed_stat.st_ino,
        "device": observed_stat.st_dev,
        "nlink": observed_stat.st_nlink,
    }


def publish_o_excl(path: Path, payload: bytes, mode: int = 0o600) -> dict[str, Any]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        before = os.fstat(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        reloaded = b""
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            reloaded += chunk
        after = os.fstat(fd)
        visible = os.lstat(path)
        identity = (after.st_dev, after.st_ino, after.st_size)
        if (
            reloaded != payload
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size) != identity
            or stat.S_ISLNK(visible.st_mode)
            or (visible.st_dev, visible.st_ino, visible.st_size) != identity
        ):
            raise RuntimeError(f"unsafe publication: {path}")
        return {
            "sha256": sha256_bytes(reloaded),
            "bytes": len(reloaded),
            "mode": oct(after.st_mode & 0o777),
            "inode": after.st_ino,
            "device": after.st_dev,
            "nlink": after.st_nlink,
        }
    finally:
        os.close(fd)


def configure_actor6(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[str]]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    parameters: list[torch.nn.Parameter] = []
    for name in ACTOR_NAMES:
        if name not in named:
            raise ValueError(f"missing actor parameter {name}")
        named[name].requires_grad_(True)
        parameters.append(named[name])
    observed = [name for name, value in model.named_parameters() if value.requires_grad]
    if tuple(observed) != ACTOR_NAMES:
        raise RuntimeError(f"actor6 trainable scope drift: {observed}")
    return parameters, observed


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{namespace}:{value}".encode()).hexdigest()


def episode_round_robin(
    records: Sequence[dict[str, Any]],
    namespace: str,
    cap: int | None = None,
    fragile_first: bool = False,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["episode_id"])].append(record)
    for episode, values in groups.items():
        values.sort(
            key=lambda item: (
                (
                    float(item["decision_margin"])
                    if fragile_first
                    else -float(item["decision_margin"])
                ),
                stable_key(namespace + ":row", str(item["line_sha256"])),
            )
        )
        if cap is not None:
            del values[cap:]
    episodes = sorted(groups, key=lambda value: stable_key(namespace + ":episode", value))
    flattened: list[dict[str, Any]] = []
    rank = 0
    while True:
        appended = False
        for episode in episodes:
            values = groups[episode]
            if rank < len(values):
                flattened.append(values[rank])
                appended = True
        if not appended:
            break
        rank += 1
    return flattened


def cyclic_take(
    pool: Sequence[dict[str, Any]], count: int, start: int
) -> list[dict[str, Any]]:
    if len(pool) < count:
        raise ValueError(f"pool of {len(pool)} rows cannot supply {count} unique rows")
    return [pool[(start + offset) % len(pool)] for offset in range(count)]


def select_batches(profile: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    if (
        profile.get("schema_version") != "ptcg-u468-beta1157-train-margin-profile-v1"
        or profile.get("status") != "completed_train_only"
        or profile.get("split") != "train"
        or profile.get("validation_opened") is not False
        or profile.get("base_model_state_sha256") != BASE_MODEL_SHA256
    ):
        raise ValueError("train-margin profile contract mismatch")
    profiles = profile.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(DATASETS):
        raise ValueError("profile dataset set mismatch")
    for source, expected_sha256 in DATA_SHA256.items():
        panel = profiles[source]
        if panel.get("archive_sha256") != expected_sha256:
            raise ValueError(f"{source} profile archive hash mismatch")
        opened = panel.get("opened_members")
        if (
            not isinstance(opened, list)
            or not opened
            or any(not str(name).startswith("train/") for name in opened)
            or panel.get("non_train_members_opened") is not False
        ):
            raise ValueError(f"{source} profile opened a non-train member")

    flg_panel = profiles["flg"]
    flg_hard = [
        record
        for record in flg_panel["near_wrong"]
        if int(record["min_count"]) == int(record["max_count"])
        and len(record["expert_order"]) > 0
        and int(record["context"]) != ppo.SKILL_ORDER_CONTEXT
        and record["ordered_correct"] is False
    ]
    flg_pool = episode_round_robin(flg_hard, "flg-hard", cap=2)
    flg_context = episode_round_robin(
        [
            record
            for record in flg_panel["fragile_correct"]
            if int(record["context"]) == ppo.SKILL_ORDER_CONTEXT
            and len(record["expert_order"]) > 0
            and record["ordered_correct"] is True
        ],
        "flg-context34",
        fragile_first=True,
    )

    pf_panel = profiles["pokemonfan"]
    pf_ordinary = episode_round_robin(
        [
            record
            for record in pf_panel["fragile_correct"]
            if int(record["context"]) != ppo.SKILL_ORDER_CONTEXT
            and len(record["expert_order"]) > 0
            and record["ordered_correct"] is True
        ],
        "pf-ordinary",
        fragile_first=True,
    )
    pf_context = episode_round_robin(
        [
            record
            for record in pf_panel["fragile_correct"]
            if int(record["context"]) == ppo.SKILL_ORDER_CONTEXT
            and len(record["expert_order"]) > 0
            and record["ordered_correct"] is True
        ],
        "pf-context34",
        fragile_first=True,
    )

    core_panel = profiles["core5"]
    core_context = episode_round_robin(
        [
            record
            for record in core_panel["fragile_correct"]
            if int(record["context"]) == ppo.SKILL_ORDER_CONTEXT
            and len(record["expert_order"]) > 0
            and record["ordered_correct"] is True
        ],
        "core-context34",
        fragile_first=True,
    )
    core_by_team: dict[str, list[dict[str, Any]]] = {}
    team_names = sorted({str(record["team_name"]) for record in core_panel["fragile_correct"]})
    if len(team_names) != 5:
        raise ValueError(f"core5 team count drift: {team_names}")
    for team in team_names:
        core_by_team[team] = episode_round_robin(
            [
                record
                for record in core_panel["fragile_correct"]
                if str(record["team_name"]) == team
                and int(record["context"]) != ppo.SKILL_ORDER_CONTEXT
                and len(record["expert_order"]) > 0
                and record["ordered_correct"] is True
            ],
            "core-" + team,
            fragile_first=True,
        )

    batches: list[list[dict[str, Any]]] = []
    per_step_summary: list[dict[str, Any]] = []
    for step_index in range(STEPS):
        selected: list[dict[str, Any]] = []
        flg_rows = cyclic_take(flg_pool, 101, step_index * 101)
        flg_rows += [flg_context[step_index % len(flg_context)]]
        selected += [{**record, "source": "flg"} for record in flg_rows]

        pf_rows = cyclic_take(pf_ordinary, 76, step_index * 76)
        pf_rows += [pf_context[step_index % len(pf_context)]]
        selected += [{**record, "source": "pokemonfan"} for record in pf_rows]

        core_context_row = core_context[step_index % len(core_context)]
        extra_teams = {
            team_names[step_index % 5],
            team_names[(step_index + 1) % 5],
        }
        target_team_counts = {
            team: 16 if team in extra_teams else 15 for team in team_names
        }
        target_team_counts[str(core_context_row["team_name"])] -= 1
        core_rows = [core_context_row]
        for team_index, team in enumerate(team_names):
            count = target_team_counts[team]
            if count < 0:
                raise RuntimeError("core5 context row overfilled a team quota")
            core_rows += cyclic_take(
                core_by_team[team],
                count,
                step_index * 32,
            )
        if len(core_rows) != 77:
            raise RuntimeError("core5 batch quota drift")
        selected += [{**record, "source": "core5"} for record in core_rows]

        if len(selected) != BATCH_SIZE:
            raise RuntimeError("mixed batch size drift")
        random.Random(SELECTION_SEED + 1009 * (step_index + 1)).shuffle(selected)
        counts = {source: sum(row["source"] == source for row in selected) for source in DATASETS}
        context_counts = {
            source: sum(
                row["source"] == source
                and int(row["context"]) == ppo.SKILL_ORDER_CONTEXT
                for row in selected
            )
            for source in DATASETS
        }
        if counts != SOURCE_COUNTS or context_counts != CONTEXT34_COUNTS:
            raise RuntimeError("mixed source/context quota drift")
        if len({(row["source"], row["line_sha256"]) for row in selected}) != BATCH_SIZE:
            raise RuntimeError("duplicate row within a mixed batch")
        batches.append(selected)
        per_step_summary.append(
            {
                "step": step_index + 1,
                "source_counts": counts,
                "context34_counts": context_counts,
                "core5_team_counts": {
                    team: sum(
                        row["source"] == "core5" and row["team_name"] == team
                        for row in selected
                    )
                    for team in team_names
                },
                "selection_sha256": sha256_bytes(
                    canonical_json(
                        [
                            {
                                "source": row["source"],
                                "member": row["member"],
                                "line_index": row["line_index"],
                                "line_sha256": row["line_sha256"],
                            }
                            for row in selected
                        ]
                    )
                ),
            }
        )
    if len(
        {
            (row["source"], row["line_sha256"])
            for batch in batches
            for row in batch
        }
    ) != STEPS * BATCH_SIZE:
        raise RuntimeError("selected rows repeat across training steps")
    summary = {
        "flg_hard_source_rows": len(flg_hard),
        "flg_hard_episode_capped_pool_rows": len(flg_pool),
        "flg_hard_episode_count": len({row["episode_id"] for row in flg_pool}),
        "flg_context34_retention_rows": len(flg_context),
        "pokemonfan_ordinary_retention_rows": len(pf_ordinary),
        "pokemonfan_context34_retention_rows": len(pf_context),
        "core5_context34_retention_rows": len(core_context),
        "core5_ordinary_retention_rows_by_team": {
            team: len(values) for team, values in core_by_team.items()
        },
        "per_step": per_step_summary,
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
            existing = wanted[record["source"]].get(key)
            if existing is not None and existing["line_sha256"] != record["line_sha256"]:
                raise RuntimeError("selected row identity collision")
            wanted[record["source"]][key] = record

    features: dict[tuple[str, str, int], dict[str, Any]] = {}
    for source, path in DATASETS.items():
        by_member: dict[str, set[int]] = defaultdict(set)
        for member, line_index in wanted[source]:
            if not member.startswith("train/") or not member.endswith(".jsonl"):
                raise ValueError("selection attempted a non-train member")
            by_member[member].add(line_index)
        with zipfile.ZipFile(path) as archive:
            archive_members = set(archive.namelist())
            for member in sorted(by_member):
                if member not in archive_members:
                    raise ValueError(f"missing selected member {member}")
                remaining = set(by_member[member])
                with archive.open(member) as handle:
                    for line_index, raw in enumerate(handle):
                        if line_index not in remaining:
                            continue
                        record = wanted[source][(member, line_index)]
                        if hashlib.sha256(raw).hexdigest() != record["line_sha256"]:
                            raise ValueError("selected line hash drift")
                        row = orjson.loads(raw)
                        if str(row.get("split", "")) != "train":
                            raise RuntimeError("selected row is not train split")
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
                            raise RuntimeError("selected row metadata drift")
                        feature["action_sequence"] = expert
                        if int(feature["context"]) == ppo.SKILL_ORDER_CONTEXT:
                            feature["sample_weight"] = CONTEXT34_SAMPLE_WEIGHT
                        features[(source, member, line_index)] = feature
                        remaining.remove(line_index)
                        if not remaining:
                            break
                if remaining:
                    raise RuntimeError(f"selected line indices missing from {member}")

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
            raise RuntimeError("collated mixed batch size drift")
        if int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum()) != 3:
            raise RuntimeError("collated context34 physical quota drift")
        context_weights = batch["sample_weights"][
            batch["contexts"] == ppo.SKILL_ORDER_CONTEXT
        ]
        if not torch.equal(
            context_weights,
            torch.full_like(context_weights, CONTEXT34_SAMPLE_WEIGHT),
        ):
            raise RuntimeError("context34 effective-weight normalization drift")
        result.append(batch)
    return result


def special_config(parent: dict[str, Any]) -> ppo.PPOConfig:
    raw = parent.get("config")
    if not isinstance(raw, dict):
        raise ValueError("U468 config missing")
    config = ppo.PPOConfig(**copy.deepcopy(raw))
    config.bc_replay_data = str(DATASETS["flg"])
    config.bc_replay_split = "train"
    config.bc_replay_batches = STEPS
    config.bc_replay_batch_size = BATCH_SIZE
    config.bc_replay_workers = 1
    config.bc_replay_steps = 1
    config.bc_replay_lr_scale = LR_SCALE
    config.bc_replay_loss = "ordered"
    config.bc_replay_order_context_weight = 8.0
    config.bc_replay_context34_rows_per_batch = 3
    config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    config.max_grad_norm = MAX_GRAD_NORM
    config.seed = SELECTION_SEED
    return config


def load_base(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
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
    return model, checkpoints["new_parent"], state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cache-audit", "formal"), required=True)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--expected-design-sha256")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_design(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.design is None or not args.expected_design_sha256:
        raise ValueError("formal mode requires a hash-bound design")
    path = args.design.resolve()
    if path != DESIGN_PATH:
        raise ValueError("unexpected design path")
    evidence = regular_evidence(path, args.expected_design_sha256, "design")
    design = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA,
        "branch": BRANCH,
        "base_model_state_sha256": BASE_MODEL_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "learning_rate": LEARNING_RATE,
        "selection_seed": SELECTION_SEED,
        "execution_seed": EXECUTION_SEED,
        "source_counts": SOURCE_COUNTS,
        "context34_counts": CONTEXT34_COUNTS,
        "endpoint_steps": list(ENDPOINT_STEPS),
        "expected_cache_sha256": EXPECTED_CACHE_SHA256,
        "expected_batch_sha256": list(EXPECTED_BATCH_SHA256),
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise ValueError(f"design field mismatch: {key}")
    executor = design.get("executor")
    if not isinstance(executor, dict):
        raise ValueError("design executor binding missing")
    self_path = Path(__file__).resolve()
    if (
        executor.get("path") != str(self_path.relative_to(ROOT))
        or executor.get("sha256") != sha256_file(self_path)
    ):
        raise ValueError("design executor binding drift")
    return design, evidence


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
    if not math.isclose(LR_SCALE, 0.00625, rel_tol=0.0, abs_tol=1e-18):
        raise RuntimeError("learning-rate scale drift")
    if tuple(prior.MUTABLE_NAMES[:6]) != ACTOR_NAMES:
        raise RuntimeError("actor6 scope definition drift")

    input_evidence = {
        "profile": regular_evidence(PROFILE, PROFILE_SHA256, "profile"),
        **{
            source: regular_evidence(path, DATA_SHA256[source], source)
            for source, path in DATASETS.items()
        },
        "dependencies": {
            str(path.relative_to(ROOT)): regular_evidence(path, digest, str(path))
            for path, digest in DEPENDENCY_SHA256.items()
        },
    }
    for path, digest in prior.INPUT_HASHES.items():
        regular_evidence(path, digest, str(path))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    selections, selection_summary = select_batches(profile)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    random.seed(EXECUTION_SEED)
    torch.manual_seed(EXECUTION_SEED)
    torch.cuda.manual_seed_all(EXECUTION_SEED)
    model, parent, base_state = load_base(device)
    cache = load_selected_features(selections, parent["model_config"])
    cache_sha256, batch_sha256 = repair.replay_cache_manifest(cache)
    cache_contract = {
        "cache_sha256": cache_sha256,
        "batch_sha256": batch_sha256,
        "selection_summary": selection_summary,
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
    if args.mode == "cache-audit":
        if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
            raise FileExistsError(OUTPUT_ROOT)
        if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
            raise FileExistsError(ATTEMPT_MARKER)
        if transport.model_state_sha256(repair.clone_model_state(model)) != BASE_MODEL_SHA256:
            raise RuntimeError("cache audit mutated base model")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "cache_audit_passed_zero_writes",
                    "base_model_state_sha256": BASE_MODEL_SHA256,
                    "profile": input_evidence["profile"],
                    "validation_opened": False,
                    "optimizer_steps": 0,
                    "checkpoint_writes": 0,
                    **cache_contract,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return

    design, design_evidence = validate_design(args)
    if cache_sha256 != EXPECTED_CACHE_SHA256 or tuple(batch_sha256) != EXPECTED_BATCH_SHA256:
        raise ValueError("formal mixed cache hash drift")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)
    if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
        raise FileExistsError(ATTEMPT_MARKER)

    parameters, trainable_names = configure_actor6(model)
    trainable_count = sum(parameter.numel() for parameter in parameters)
    config = special_config(parent)
    common = {
        "schema_version": SCHEMA,
        "branch": BRANCH,
        "base": {
            "formula": "U468 + (1-beta)*(current_exact_p12-U468) + beta*(historical_p12-U464)",
            "beta": BETA,
            "model_state_sha256": BASE_MODEL_SHA256,
        },
        "design": design_evidence,
        "inputs": input_evidence,
        "training": {
            "data_split": "train",
            "validation_opened": False,
            "fresh_optimizer": True,
            "optimizer_state_loaded": False,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "loss": "ordered",
            "order_context_weight": 8.0,
            "context34_sample_weight_per_source": CONTEXT34_SAMPLE_WEIGHT,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "count_head_frozen": True,
            "value_head_frozen": True,
            "selection_seed": SELECTION_SEED,
            "execution_seed": EXECUTION_SEED,
            "source_counts_per_step": SOURCE_COUNTS,
            "context34_counts_per_step": CONTEXT34_COUNTS,
            "endpoint_steps": list(ENDPOINT_STEPS),
            **cache_contract,
        },
        "runtime": {
            "python": str(Path(sys.executable).resolve()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    created_at = datetime.now(timezone.utc).isoformat()
    marker_evidence = publish_o_excl(
        ATTEMPT_MARKER,
        canonical_json(common | {"status": "formal_attempt_consumed", "created_at_utc": created_at}),
    )
    OUTPUT_ROOT.mkdir(mode=0o700)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
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
        raise RuntimeError("fresh optimizer unexpectedly has state")
    per_step: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    frozen_names = sorted(set(base_state) - set(ACTOR_NAMES))
    for step, batch in enumerate(cache, start=1):
        metrics = ppo.bc_replay_update(
            model,
            optimizer,
            [batch],
            config,
            device,
            BASE_ACTOR_LEARNING_RATE,
        )
        if not isinstance(metrics, dict) or metrics.get("steps") != 1:
            raise RuntimeError("BC core did not execute one step")
        if int(metrics.get("rows", -1)) != BATCH_SIZE:
            raise RuntimeError("BC row count drift")
        if int(metrics.get("context_34_rows", -1)) != 3:
            raise RuntimeError("BC context34 count drift")
        if not repair.finite_nested(metrics):
            raise FloatingPointError("non-finite BC metrics")
        optimizer_steps = repair.optimizer_steps(optimizer.state_dict())
        if len(optimizer_steps) != len(ACTOR_NAMES) or set(optimizer_steps) != {step}:
            raise RuntimeError("fresh optimizer step drift")
        per_step.append(
            {
                "step": step,
                "batch_sha256": batch_sha256[step - 1],
                "metrics": metrics,
                "optimizer_state_sha256": repair.nested_sha256(optimizer.state_dict()),
            }
        )
        if step not in ENDPOINT_STEPS:
            continue
        state = repair.clone_model_state(model)
        changed = repair.changed_tensor_names(base_state, state)
        if changed != sorted(ACTOR_NAMES):
            raise RuntimeError(f"endpoint P{step} changed scope {changed}")
        if any(not torch.equal(state[name], base_state[name]) for name in frozen_names):
            raise RuntimeError(f"endpoint P{step} changed a frozen tensor")
        model_sha256 = transport.model_state_sha256(state)
        slim = {
            key: copy.deepcopy(parent[key]) for key in prior.REQUIRED_SLIM_KEYS
        }
        slim.update(
            {
                "model_state_dict": state,
                "update": 468,
                "evaluation_only": True,
                "resume_forbidden": True,
                "optimizer_states_omitted": [
                    "optimizer_state_dict",
                    "bc_replay_optimizer_state_dict",
                    "opponent_quota_state",
                    "fresh_special_optimizer_state_dict",
                ],
                "trainhard_actor6_special_bc": {
                    "schema_version": SCHEMA,
                    "design_sha256": design_evidence["sha256"],
                    "base_model_state_sha256": BASE_MODEL_SHA256,
                    "endpoint_step": step,
                    "model_state_sha256": model_sha256,
                    "trainable_parameter_names": list(ACTOR_NAMES),
                    "validation_opened_during_training": False,
                    "resume_forbidden": True,
                },
            }
        )
        buffer = io.BytesIO()
        torch.save(slim, buffer)
        raw = buffer.getvalue()
        filename = f"special-bc-trainhard-actor6-{step:04d}.pt"
        path = OUTPUT_ROOT / filename
        publication = publish_o_excl(path, raw)
        reloaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if transport.model_state_sha256(reloaded["model_state_dict"]) != model_sha256:
            raise RuntimeError("serialized endpoint model hash drift")
        endpoints.append(
            {
                "step": step,
                "path": str(path.relative_to(ROOT)),
                "model_state_sha256": model_sha256,
                "changed_parameter_names": changed,
                **publication,
            }
        )

    manifest = common | {
        "status": "training_completed_all_endpoints_published",
        "created_at_utc": created_at,
        "attempt_marker": {
            "path": str(ATTEMPT_MARKER.relative_to(ROOT)),
            **marker_evidence,
        },
        "per_step": per_step,
        "endpoints": endpoints,
        "validation_opened_during_training": False,
        "parent_unchanged": sha256_file(prior.U468) == prior.INPUT_HASHES[prior.U468],
    }
    manifest_path = OUTPUT_ROOT / "training_manifest.json"
    manifest_evidence = publish_o_excl(manifest_path, canonical_json(manifest))
    completion = {
        "schema_version": SCHEMA,
        "status": "completed",
        "branch": BRANCH,
        "training_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            **manifest_evidence,
        },
        "endpoint_steps": list(ENDPOINT_STEPS),
    }
    completion_path = OUTPUT_ROOT / "COMPLETED.json"
    completion_evidence = publish_o_excl(completion_path, canonical_json(completion))
    expected_names = sorted(
        [f"special-bc-trainhard-actor6-{step:04d}.pt" for step in ENDPOINT_STEPS]
        + ["training_manifest.json", "COMPLETED.json"]
    )
    if sorted(path.name for path in OUTPUT_ROOT.iterdir()) != expected_names:
        raise RuntimeError("terminal output tree drift")
    os.chmod(OUTPUT_ROOT, 0o500)
    print(
        json.dumps(
            completion
            | {
                "completion": {
                    "path": str(completion_path.relative_to(ROOT)),
                    **completion_evidence,
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
