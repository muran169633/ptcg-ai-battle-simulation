#!/usr/bin/env python3
"""Zero-update full-source BC geometry probe at the frozen S8 endpoint.

This probe diagnoses why the first anti-KD BC repair moved in the wrong
direction.  It differentiates ordered-selection loss at S8, but it never
updates or serializes a model.  Candidate directions are constructed only
from the three natural-weight fit gradients.  Time-forward calibration and
the three final behavior views are read-only geometry checks.

Formal execution is two-stage: freeze and review an input plan, then execute
that exact plan and exclusively create one JSON report.  There is no gameplay,
checkpoint publication, packaging, upload, or submission path in this file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import orjson  # noqa: E402
import torch  # noqa: E402

import evaluate_policy_bc as behavior_core  # noqa: E402
import run_gold_push_postppo_tail_repair as training_core  # noqa: E402

ppo = training_core.ppo

SCHEMA_VERSION = "ptcg-s08-antikd-fullsource-bc-geometry-v1"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
EXPECTED_ENV_PREFIX = Path("/home/xxc/miniconda3/envs/my_project_env")
EXPECTED_CUBLAS = ":4096:8"
EXPECTED_CUDA_VISIBLE_DEVICES = "0"
EXPECTED_TORCH_VERSION = "2.8.0+cu128"
EXPECTED_TORCH_CUDA_VERSION = "12.8"
EXPECTED_CUDNN_VERSION = 91002
EXPECTED_ORJSON_VERSION = "3.11.9"
DEVICE = "cuda:0"

PLAN_PATH = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_antikd_fullsource_bc_geometry_v1.reviewed_plan.json"
)
OUTPUT_PATH = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_antikd_fullsource_bc_geometry_v1.json"
)

S8_PARENT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/postppo_tail_repair_r2best_v1/"
    "postppo-special-bc-s08.pt"
)
BC_ARCHITECTURE = (
    ROOT
    / "artifacts/gold_push_20260810_v1/bc/"
    "marnie_trainwins_seed1011/best.pt"
)
FIRST_ROUND_DIR = (
    ROOT / "artifacts/gold_push_20260810_v1/s08_antikd_bc_repair_v1"
)
FIRST_ROUND_MANIFEST = FIRST_ROUND_DIR / "frozen_training_manifest.json"
FIRST_ROUND_RESULT = FIRST_ROUND_DIR / "training_result.json"
FIRST_ROUND_E100 = FIRST_ROUND_DIR / "s08-antikd-E100.pt"
FIRST_ROUND_LAUNCHER = TOOLS / "run_gold_push_s08_antikd_bc_repair.py"

ANTI_ARCHIVE = (
    ROOT
    / "data/gold_push_marnie_antikd_exact_20260810_v1/"
    "marnie_exact_anti_kd_trainwins.zip"
)
FROS_ARCHIVE = (
    ROOT
    / "data/gold_push_marnie_fros_exact_20260810_v1/"
    "marnie_vs_froslass_trainwins.zip"
)
GENERAL_ARCHIVE = (
    ROOT
    / "data/gold_push_marnie_general_anchor_20260810_v1/"
    "marnie_general_anchor_train.zip"
)
FROS_VALID_ARCHIVE = (
    ROOT
    / "data/gold_push_marnie_fros_valid_20260810_v1/"
    "marnie_exact_fros_valid_view.zip"
)
GENERAL_VALID_ARCHIVE = (
    ROOT / "data/gold_push_recent7_20260810_v1/archives/marnie.zip"
)

IMMUTABLE_INPUTS: dict[str, tuple[Path, str]] = {
    "s8_parent": (
        S8_PARENT,
        "d7443bda57cb89a5c12d9d776710d1b8b2c01151541573e01f7e04b032ea25e9",
    ),
    "bc_architecture": (
        BC_ARCHITECTURE,
        "dda68d51d9b922526709149143aa8409ba287fb8f31ddbb2293f0ea543ef01a0",
    ),
    "first_round_manifest": (
        FIRST_ROUND_MANIFEST,
        "eb01d498bd4d0d732b998e898c6caa6a9062ca6690d8a12958ed8cea3e7ffb65",
    ),
    "first_round_result": (
        FIRST_ROUND_RESULT,
        "892545318b57ebebf6207883f05970cc47a24c44e60a38117318cfc41359ba01",
    ),
    "first_round_e100": (
        FIRST_ROUND_E100,
        "0bfb4b93789abcd1f27892c13b5e4095e123d2d1d396dcbadcfcbe946d9ee83a",
    ),
    "first_round_launcher": (
        FIRST_ROUND_LAUNCHER,
        "bb451f7f04dffa24f0e1e8cbbcfabc61265aad9d82812a017c4988e22fb4aa00",
    ),
    "anti_train_and_dev_archive": (
        ANTI_ARCHIVE,
        "4ee00aa4fdc632bf3e9218796b055d148a3e1d388ed65192354253b6ea09179d",
    ),
    "fros_train_archive": (
        FROS_ARCHIVE,
        "a148e42baf8e84f94150cf9d2437c1e04f30a8b376575e0db3976ee9da31e044",
    ),
    "general_train_archive": (
        GENERAL_ARCHIVE,
        "3a95c8c62ba706e1226b79404b4f2a1ce8663e57582eafde49bcdb41e43db9db",
    ),
    "fros_final_valid_archive": (
        FROS_VALID_ARCHIVE,
        "559fc2678c45f0c1509dc9c38c55a6dd09fc15fbdf1d1c3a3bc0f7b086bbddf3",
    ),
    "general_final_valid_archive": (
        GENERAL_VALID_ARCHIVE,
        "6b3873b28bfad70377d0a3b22fe1163516a3d40b5af429885c0ce15f75b402a0",
    ),
    "training_core": (
        Path(training_core.__file__).resolve(),
        "d1484f37030b128db6b0bffa9e07b7e387fcc6b95d2d5e57b937e615421ca53e",
    ),
    "trainer_module": (
        training_core.TRAINER,
        "321a4e25fb3ccecb0dfb3c37803a20369b8083619f6b238c3d067990b7e397cf",
    ),
    "behavior_core": (
        Path(behavior_core.__file__).resolve(),
        "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4",
    ),
    "bc_core": (
        Path(behavior_core.bc.__file__).resolve(),
        "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
    ),
    "python": (
        EXPECTED_PYTHON.resolve(),
        "35010543d1379740c163ebf34e92108891c70cceb393367d71f463733c8be497",
    ),
}

EMBEDDED_MANIFEST_SHA256 = {
    ANTI_ARCHIVE: "a79a0c9261581b2745b8705d5750f678af51c020f7be79464f2653aa8ea2b335",
    FROS_ARCHIVE: "f647ddd9259c0766516173508348fa4fa8c573a76997d8369d6561ae0a6fb805",
    GENERAL_ARCHIVE: "08b94fafe5eef4ccae104ac07a216bee97e809781c0fc005db45a147ebd74121",
    FROS_VALID_ARCHIVE: "cfd243383d60c9176c918553d94a91130b293416f1b8765e6e29423a77f62b40",
    GENERAL_VALID_ARCHIVE: "0839b0fd98ceaaa3b66c92c46899e62e8e3318998d6bba0c841536205879ecdb",
}

FIRST_ROUND_INNER_MANIFEST_SHA256 = (
    "a73eff4c08c7cdbcc443d17d10f66f1f50688ef36cef07aea2ef574aee8f9c1c"
)
MARNIE_DECK_HASH = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)

ACTOR6 = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
BATCH_SIZE = 256
NATURAL_WEIGHTS = (1.0, 1.0)
LEGACY_WEIGHTS = (8.0, 2.0)
COSINE_EPSILON = 1e-8
NON_ASCENT_ABSOLUTE_TOLERANCE = 1e-12
SOURCE_ORDER = ("anti", "fros", "general")
FIT_TARGETS = tuple(f"{source}_fit" for source in SOURCE_ORDER)
CAL_TARGETS = tuple(f"{source}_cal" for source in SOURCE_ORDER)
FINAL_TARGETS = ("anti_final", "fros_final", "general_final")


@dataclass(frozen=True)
class DomainSpec:
    name: str
    source: str
    role: str
    archive: Path
    member_split: str
    dates: tuple[str, ...]


DOMAIN_SPECS = (
    DomainSpec(
        "anti_fit", "anti", "fit", ANTI_ARCHIVE, "train",
        ("2026-08-02", "2026-08-03"),
    ),
    DomainSpec(
        "anti_cal", "anti", "cal", ANTI_ARCHIVE, "train",
        ("2026-08-04", "2026-08-05"),
    ),
    DomainSpec(
        "fros_fit", "fros", "fit", FROS_ARCHIVE, "train",
        ("2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05"),
    ),
    DomainSpec(
        "fros_cal", "fros", "cal", FROS_ARCHIVE, "train",
        ("2026-08-06",),
    ),
    DomainSpec(
        "general_fit", "general", "fit", GENERAL_ARCHIVE, "train",
        ("2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05"),
    ),
    DomainSpec(
        "general_cal", "general", "cal", GENERAL_ARCHIVE, "train",
        ("2026-08-06",),
    ),
    DomainSpec(
        "anti_final", "anti", "final", ANTI_ARCHIVE, "dev",
        (
            "2026-08-02", "2026-08-03", "2026-08-04",
            "2026-08-05", "2026-08-06",
        ),
    ),
    DomainSpec(
        "fros_final", "fros", "final", FROS_VALID_ARCHIVE, "valid",
        ("2026-08-07",),
    ),
    DomainSpec(
        "general_final", "general", "final", GENERAL_VALID_ARCHIVE,
        "valid", ("2026-08-07",),
    ),
)

EXPECTED_DOMAIN_STATS: dict[str, dict[str, Any]] = {
    "anti_fit": {
        "rows": 3020, "episodes": 27, "views": 27, "context34_rows": 9,
        "fixed_multi_rows": 23,
        "date_rows": {"2026-08-02": 2567, "2026-08-03": 453},
        "view_key_sha256": "e9561650cb9f709d9e9d695365a1990a99c514ea9d4025285642257a912e5155",
        "episode_key_sha256": "e0fd37984e501c440e3f97ed163de12a7df768d4c5ca44d4cd9a63b1a085ce7f",
        "decision_key_sha256": "954a60de03c1f33266c77302033b396dcde45574f6f5143ff6b8424064898c3a",
    },
    "anti_cal": {
        "rows": 885, "episodes": 8, "views": 8, "context34_rows": 0,
        "fixed_multi_rows": 9,
        "date_rows": {"2026-08-04": 553, "2026-08-05": 332},
        "view_key_sha256": "995c4b53d264df70558ccc67d6bafe37ee72e0d1c2e0491656743c72c9cee5f5",
        "episode_key_sha256": "41fd1458ee40c14eeb2d308d8f5b837460d72827424a2bd82f9787214b139fb4",
        "decision_key_sha256": "9da4ae0f9b49e228de81a95bb03a844d2b4d52b45671f36de9534ddd2abb1bc6",
    },
    "fros_fit": {
        "rows": 9166, "episodes": 89, "views": 89, "context34_rows": 4,
        "fixed_multi_rows": 117,
        "date_rows": {"2026-08-02": 2638, "2026-08-03": 1083, "2026-08-04": 2557, "2026-08-05": 2888},
        "view_key_sha256": "2a0a7572795cc7ad903ec2b628415fbb9fc8c5912ba9ad8dccc6b7d15f205d95",
        "episode_key_sha256": "09d7d2d06198dfecb19bcabfccd35f15c1e8ef4c3888daa35c278a621dda5093",
        "decision_key_sha256": "6da7e63a7979c1a7cd3b16e47001b0766eae31cdd76ca1e295484e7ca603c5c7",
    },
    "fros_cal": {
        "rows": 1448, "episodes": 15, "views": 15, "context34_rows": 5,
        "fixed_multi_rows": 17,
        "date_rows": {"2026-08-06": 1448},
        "view_key_sha256": "345d3db83da944ccdfcc1f08dd88c31d9a21e280ccfa64de050ebe3d825c467a",
        "episode_key_sha256": "4eeabc7a3ed1952b62efd2ade23723c86c1033e8c86cd658c61666a9c50df3a6",
        "decision_key_sha256": "1dba724850d3aba59a67c51e6e783e139910c6fcd69eb5f240bea547e2e16cb3",
    },
    "general_fit": {
        "rows": 158394, "episodes": 1620, "views": 1620,
        "context34_rows": 771, "fixed_multi_rows": 1855,
        "date_rows": {"2026-08-02": 61675, "2026-08-03": 41457, "2026-08-04": 35289, "2026-08-05": 19973},
        "view_key_sha256": "4576c4c4a2a14aeaff1dc7a74e8e2e2392e1749eebd0436a9d7ded527b9b7fce",
        "episode_key_sha256": "e63fdb9295ef318e2b14c32f0ab4808405087ac4805bf17a7151f958a2f9b22d",
        "decision_key_sha256": "f7621bd3ebd59316c86aa941af36ca0fb94e1dc0bd52ed92ae9ad7d481987160",
    },
    "general_cal": {
        "rows": 13730, "episodes": 142, "views": 142,
        "context34_rows": 84, "fixed_multi_rows": 168,
        "date_rows": {"2026-08-06": 13730},
        "view_key_sha256": "ac369216a4c862643411beb75abd5b5dfd388da297921547600557b41aba9ca8",
        "episode_key_sha256": "f89f22ade839b01cf40f33789e643e54bb357095e7b787a5d9888ad04a6a37e5",
        "decision_key_sha256": "8f89037d99f0a281559006972344c5a26673c5eeff200fd5181387bd23a099cb",
    },
    "anti_final": {
        "rows": 968, "episodes": 8, "views": 8, "context34_rows": 0,
        "fixed_multi_rows": 10,
        "date_rows": {"2026-08-02": 543, "2026-08-03": 115, "2026-08-04": 99, "2026-08-05": 90, "2026-08-06": 121},
        "view_key_sha256": "90555f2b7c3d163c44fe7a5b31c2b7eeeeb7293238528e7b7928cc3454d23b01",
        "episode_key_sha256": "ec28eea63844b5d12b1d534fb1b8ed4e8e90794af92c54992c4ff77c451ff34d",
        "decision_key_sha256": "2e41e6af8318922fc10e73391043cf0357ae7fe8fc4753b94a64ed6129355ebd",
    },
    "fros_final": {
        "rows": 3274, "episodes": 30, "views": 30, "context34_rows": 1,
        "fixed_multi_rows": 20, "date_rows": {"2026-08-07": 3274},
        "view_key_sha256": "17c93d69dd0851ae84f23158be6b1ae611a50245cd4d21ab4670f5e21619b5d7",
        "episode_key_sha256": "d975b2eecdfabf3e481fd8f86a1be9c6509803236f86eaf63f6346801974e59a",
        "decision_key_sha256": "ef19c8634ad7f786fc513663e1a429c407e411a11b473026191769b83ff87aeb",
    },
    "general_final": {
        "rows": 28800, "episodes": 295, "views": 299,
        "context34_rows": 100, "fixed_multi_rows": 269,
        "date_rows": {"2026-08-07": 28800},
        "view_key_sha256": "f520be272965f5fe6e1c96e639c0c6947aad12d2fa90d1f537d9ecf5341bb598",
        "episode_key_sha256": "ae30649ad0a862d52bc441dc93d9fd28504ba6d898aa6b17d33260df869f052b",
        "decision_key_sha256": "0db58e729b56159e3d2466b83fe56918ab9f4f1103668d4b82550dd373a98ed9",
    },
}


def canonical_json_bytes(value: Any) -> bytes:
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_bytes(raw: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"Invalid strict JSON in {label}") from error


def require_regular_file(path: Path, expected_sha256: str, label: str) -> Path:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {label}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} must be a non-symlink regular file")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    return path


def assert_output_absent(path: Path, label: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise FileExistsError(f"{label} must be absent: {path}")


def write_exclusive(path: Path, payload: bytes) -> None:
    assert_output_absent(path, "exclusive output")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _validate_zip_info(info: zipfile.ZipInfo, label: str) -> None:
    member = PurePosixPath(info.filename)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise RuntimeError(f"Unsafe ZIP member in {label}: {info.filename!r}")
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise RuntimeError(f"Symlink ZIP member in {label}: {info.filename!r}")
    if info.flag_bits & 0x1:
        raise RuntimeError(f"Encrypted ZIP member in {label}: {info.filename!r}")


def audit_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        members = [info.filename for info in infos]
        if len(members) != len(set(members)):
            raise RuntimeError(f"Duplicate ZIP members in {path.name}")
        for info in infos:
            _validate_zip_info(info, path.name)
        if members.count("manifest.json") != 1:
            raise RuntimeError(f"{path.name} requires exactly one manifest.json")
        manifest_raw = archive.read("manifest.json")
    expected = EMBEDDED_MANIFEST_SHA256[path]
    observed = sha256_bytes(manifest_raw)
    if observed != expected:
        raise RuntimeError(f"Embedded manifest SHA drift in {path.name}")
    manifest = strict_json_bytes(manifest_raw, f"{path.name}:manifest.json")
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Manifest root is not an object in {path.name}")
    return {
        "embedded_manifest_sha256": observed,
        "schema_version": manifest.get("schema_version"),
        "members": members,
    }


def row_date(row: Mapping[str, Any]) -> str:
    value = row.get("dataset_date", row.get("date"))
    if not isinstance(value, str) or not value:
        raise RuntimeError("Decision row lacks dataset_date")
    return value


def composite_view_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    date = row_date(row)
    episode_id = row.get("episode_id")
    seat = row.get("seat")
    team = row.get("team_name")
    if not isinstance(episode_id, str) or not episode_id:
        raise RuntimeError("Decision row lacks episode_id")
    if isinstance(seat, bool) or not isinstance(seat, int) or seat not in (0, 1):
        raise RuntimeError("Decision row has invalid seat")
    if not isinstance(team, str) or not team:
        raise RuntimeError("Decision row lacks team_name")
    return date, episode_id, seat, team


def iter_raw_domain_rows(spec: DomainSpec) -> Iterator[dict[str, Any]]:
    """Yield every selected row in every bound shard, with no prefix cap."""

    allowed_dates = frozenset(spec.dates)
    with zipfile.ZipFile(spec.archive) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith(f"{spec.member_split}/")
            and name.endswith(".jsonl")
        )
        if not members:
            raise RuntimeError(
                f"{spec.archive.name} has no {spec.member_split} JSONL shard"
            )
        for member in members:
            with archive.open(member) as stream:
                for line_number, raw in enumerate(stream, 1):
                    try:
                        row = orjson.loads(raw)
                    except orjson.JSONDecodeError as error:
                        raise RuntimeError(
                            f"Invalid JSONL at {member}:{line_number}"
                        ) from error
                    if not isinstance(row, dict):
                        raise RuntimeError(f"Non-object row at {member}:{line_number}")
                    if row_date(row) in allowed_dates:
                        yield row


def audit_domain(spec: DomainSpec) -> tuple[dict[str, Any], frozenset[tuple[str, str, int, str]]]:
    rows = 0
    context34_rows = 0
    fixed_multi_rows = 0
    date_rows: Counter[str] = Counter()
    episode_keys: set[tuple[str, str]] = set()
    view_keys: set[tuple[str, str, int, str]] = set()
    decision_keys: set[tuple[str, str, int, str, int]] = set()
    for row in iter_raw_domain_rows(spec):
        view_key = composite_view_key(row)
        action_step = row.get("action_step_index")
        if isinstance(action_step, bool) or not isinstance(action_step, int) or action_step <= 0:
            raise RuntimeError(f"Invalid action_step_index in {spec.name}")
        decision_key = (*view_key, action_step)
        if decision_key in decision_keys:
            raise RuntimeError(f"Duplicate decision in {spec.name}: {decision_key}")
        action = row.get("action")
        if not isinstance(action, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in action):
            raise RuntimeError(f"Invalid action list in {spec.name}")
        context = row.get("select_context")
        try:
            context_value = int(context)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid select_context in {spec.name}") from error
        minimum = row.get("min_count")
        maximum = row.get("max_count")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (minimum, maximum)):
            raise RuntimeError(f"Invalid count bounds in {spec.name}")
        date = view_key[0]
        rows += 1
        date_rows[date] += 1
        context34_rows += int(context_value == ppo.SKILL_ORDER_CONTEXT)
        fixed_multi_rows += int(
            context_value != ppo.SKILL_ORDER_CONTEXT
            and minimum == maximum
            and len(action) > 1
        )
        episode_keys.add(view_key[:2])
        view_keys.add(view_key)
        decision_keys.add(decision_key)
    audit = {
        "rows": rows,
        "episodes": len(episode_keys),
        "views": len(view_keys),
        "context34_rows": context34_rows,
        "fixed_multi_rows": fixed_multi_rows,
        "date_rows": dict(sorted(date_rows.items())),
        "view_key_sha256": sha256_json(sorted(view_keys)),
        "episode_key_sha256": sha256_json(sorted(episode_keys)),
        "decision_key_sha256": sha256_json(sorted(decision_keys)),
    }
    expected = EXPECTED_DOMAIN_STATS[spec.name]
    if audit != expected:
        raise RuntimeError(
            f"Frozen domain drift for {spec.name}: "
            + json.dumps({"expected": expected, "observed": audit}, sort_keys=True)
        )
    return audit, frozenset(view_keys)


def validate_temporal_partitions(
    domains: Mapping[str, frozenset[tuple[str, str, int, str]]],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for source in SOURCE_ORDER:
        fit = domains[f"{source}_fit"]
        cal = domains[f"{source}_cal"]
        final = domains[f"{source}_final"]
        checks = {
            "fit_cal_disjoint": not bool(fit & cal),
            "fit_final_disjoint": not bool(fit & final),
            "cal_final_disjoint": not bool(cal & final),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Temporal domain overlap for {source}: {checks}")
        records[source] = checks
    expected_train = {
        "anti": (35, 3905),
        "fros": (104, 10614),
        "general": (1762, 172124),
    }
    for source, (episodes, rows) in expected_train.items():
        if (
            len(domains[f"{source}_fit"] | domains[f"{source}_cal"])
            != episodes
            or EXPECTED_DOMAIN_STATS[f"{source}_fit"]["rows"]
            + EXPECTED_DOMAIN_STATS[f"{source}_cal"]["rows"]
            != rows
        ):
            raise RuntimeError(f"{source} fit+cal does not cover its train split")
        records[source]["fit_cal_train_full_coverage"] = True
    train_episode_keys = {
        source: {
            (date, episode_id)
            for date, episode_id, _seat, _team in (
                domains[f"{source}_fit"] | domains[f"{source}_cal"]
            )
        }
        for source in SOURCE_ORDER
    }
    cross_source: dict[str, bool] = {}
    for left_index, left in enumerate(SOURCE_ORDER):
        for right in SOURCE_ORDER[left_index + 1 :]:
            key = f"{left}__{right}"
            cross_source[key] = not bool(
                train_episode_keys[left] & train_episode_keys[right]
            )
    if not all(cross_source.values()):
        raise RuntimeError(
            f"Cross-source train episode overlap: {cross_source}"
        )
    records["cross_source_train_episode_disjoint"] = cross_source
    return records


def flatten_actor6(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [state[name].detach().cpu().double().reshape(-1) for name in ACTOR6]
    )


def vector_sha256(vector: torch.Tensor) -> str:
    value = vector.detach().cpu().double().contiguous()
    return sha256_bytes(value.numpy().astype("<f8", copy=False).tobytes())


def vector_record(vector: torch.Tensor) -> dict[str, Any]:
    norm = float(torch.linalg.vector_norm(vector.double()))
    if not math.isfinite(norm):
        raise FloatingPointError("Non-finite vector norm")
    return {
        "elements": int(vector.numel()),
        "l2": norm,
        "float64_le_sha256": vector_sha256(vector),
    }


def audit_first_round() -> dict[str, Any]:
    envelope = strict_json_bytes(
        FIRST_ROUND_MANIFEST.read_bytes(), "first-round frozen manifest"
    )
    result = strict_json_bytes(FIRST_ROUND_RESULT.read_bytes(), "first-round result")
    if not isinstance(envelope, dict) or not isinstance(result, dict):
        raise RuntimeError("First-round JSON roots must be objects")
    manifest = envelope.get("manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("First-round envelope lacks manifest")
    checks = {
        "envelope_inner_sha": envelope.get("manifest_sha256")
        == FIRST_ROUND_INNER_MANIFEST_SHA256,
        "recomputed_inner_sha": sha256_json(manifest)
        == FIRST_ROUND_INNER_MANIFEST_SHA256,
        "result_inner_sha": result.get("frozen_manifest_sha256")
        == FIRST_ROUND_INNER_MANIFEST_SHA256,
        "result_behavior_rejected": result.get("status") == "behavior_rejected",
        "result_eight_steps": result.get("optimizer_steps") == 8,
        "result_no_h2h": result.get("H2H_performed") is False,
        "result_no_package": result.get("package_performed") is False,
        "result_no_upload": result.get("upload_performed") is False,
        "result_no_submission": result.get("submission_performed") is False,
    }
    endpoints = result.get("endpoints")
    if not isinstance(endpoints, dict) or not isinstance(endpoints.get("E100"), dict):
        raise RuntimeError("First-round result lacks E100")
    checks["result_e100_sha"] = (
        endpoints["E100"].get("sha256") == IMMUTABLE_INPUTS["first_round_e100"][1]
    )
    if not all(checks.values()):
        raise RuntimeError(f"First-round lineage drift: {checks}")

    parent = torch.load(S8_PARENT, map_location="cpu", weights_only=False)
    e100 = torch.load(FIRST_ROUND_E100, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(BC_ARCHITECTURE, map_location="cpu", weights_only=False)
    if not all(isinstance(item, dict) for item in (parent, e100, bc_checkpoint)):
        raise RuntimeError("Bound checkpoints must be dictionaries")
    if parent.get("learner_deck_hash") != MARNIE_DECK_HASH:
        raise RuntimeError("S8 learner deck drift")
    if parent.get("model_config") != e100.get("model_config"):
        raise RuntimeError("E100 model configuration differs from S8")
    bc_config = bc_checkpoint.get("config")
    if not isinstance(bc_config, dict):
        raise RuntimeError("BC architecture lacks config")
    for key, value in parent["model_config"].items():
        if bc_config.get(key) != value:
            raise RuntimeError(f"BC/S8 architecture mismatch at {key}")
    parent_state = parent.get("model_state_dict")
    e100_state = e100.get("model_state_dict")
    if not isinstance(parent_state, dict) or not isinstance(e100_state, dict):
        raise RuntimeError("S8/E100 lacks model state")
    if set(parent_state) != set(e100_state):
        raise RuntimeError("S8/E100 state schema mismatch")
    changed = []
    for name in parent_state:
        if not torch.equal(parent_state[name], e100_state[name]):
            changed.append(name)
    if set(changed) != set(ACTOR6):
        raise RuntimeError(f"E100 changed outside actor6: {sorted(changed)}")
    metadata = e100.get("s8_anti_kd_bc_repair")
    if not isinstance(metadata, dict):
        raise RuntimeError("E100 lacks repair metadata")
    if (
        metadata.get("frozen_manifest_sha256") != FIRST_ROUND_INNER_MANIFEST_SHA256
        or metadata.get("parent_sha256") != IMMUTABLE_INPUTS["s8_parent"][1]
        or metadata.get("endpoint") != "E100"
        or metadata.get("H2H_eligible") is not False
        or metadata.get("submission_authorized") is not False
    ):
        raise RuntimeError("E100 frozen repair metadata drift")
    delta = flatten_actor6(e100_state) - flatten_actor6(parent_state)
    return {
        "checks": checks,
        "inner_manifest_sha256": FIRST_ROUND_INNER_MANIFEST_SHA256,
        "e100_changed_parameter_names": sorted(changed),
        "e100_parameter_delta": vector_record(delta),
    }


def static_scope_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source)
    call_names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Attribute):
            call_names.append(target.attr)
        elif isinstance(target, ast.Name):
            call_names.append(target.id)
    forbidden = sorted(
        name
        for name in call_names
        if name in {"backward", "step", "save", "save_file", "load_state_dict"}
    )
    if forbidden:
        raise RuntimeError(f"Mutation-capable call site in probe: {forbidden}")
    if call_names.count("grad") != 1:
        raise RuntimeError("Probe must have exactly one autograd.grad call site")
    return {
        "one_autograd_grad_call_site": True,
        "no_backward_optimizer_step_or_model_save_call_site": True,
        "only_persistent_outputs": [str(PLAN_PATH), str(OUTPUT_PATH)],
    }


def snapshot_input_hashes() -> dict[str, str]:
    return {
        name: file_sha256(path)
        for name, (path, _expected) in IMMUTABLE_INPUTS.items()
    } | {"probe": file_sha256(Path(__file__).resolve())}


def build_plan() -> dict[str, Any]:
    assert_output_absent(OUTPUT_PATH, "geometry report")
    input_records = {}
    for name, (path, expected) in IMMUTABLE_INPUTS.items():
        require_regular_file(path, expected, name)
        input_records[name] = {"path": str(path), "sha256": expected}
    archive_records = {
        str(path): audit_archive(path)
        for path in EMBEDDED_MANIFEST_SHA256
    }
    domain_records: dict[str, Any] = {}
    view_domains: dict[str, frozenset[tuple[str, str, int, str]]] = {}
    for spec in DOMAIN_SPECS:
        audit, views = audit_domain(spec)
        domain_records[spec.name] = {
            "source": spec.source,
            "role": spec.role,
            "archive": str(spec.archive),
            "member_split": spec.member_split,
            "dates": list(spec.dates),
            **audit,
        }
        view_domains[spec.name] = views
    temporal_checks = validate_temporal_partitions(view_domains)
    first_round = audit_first_round()
    tool_path = Path(__file__).resolve()
    tool_source = tool_path.read_bytes()
    scope_audit = static_scope_audit(tool_source)
    plan = {
        "schema_version": SCHEMA_VERSION + "-plan",
        "purpose": "zero-update actor6 ordered-selection geometry at S8",
        "inputs": input_records
        | {"probe": {"path": str(tool_path), "sha256": sha256_bytes(tool_source)}},
        "archive_audits": archive_records,
        "first_round_lineage": first_round,
        "domains": domain_records,
        "temporal_partition_checks": temporal_checks,
        "runtime": {
            "python": str(EXPECTED_PYTHON),
            "torch_version": EXPECTED_TORCH_VERSION,
            "torch_cuda_version": EXPECTED_TORCH_CUDA_VERSION,
            "cudnn_version": EXPECTED_CUDNN_VERSION,
            "orjson_version": EXPECTED_ORJSON_VERSION,
            "isolated": True,
            "dont_write_bytecode": True,
            "CUBLAS_WORKSPACE_CONFIG": EXPECTED_CUBLAS,
            "CUDA_VISIBLE_DEVICES": EXPECTED_CUDA_VISIBLE_DEVICES,
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
        },
        "losses": {
            "natural": {
                "loss_mode": "ordered",
                "context34_weight": NATURAL_WEIGHTS[0],
                "non_context34_fixed_multi_weight": NATURAL_WEIGHTS[1],
                "direction_construction": "fit only",
            },
            "legacy_gate": {
                "loss_mode": "ordered",
                "context34_weight": LEGACY_WEIGHTS[0],
                "non_context34_fixed_multi_weight": LEGACY_WEIGHTS[1],
                "direction_construction": False,
            },
            "whole_source_aggregation": (
                "sum(batch_gradient * selection_effective_weight_sum) / "
                "sum(selection_effective_weight_sum)"
            ),
            "natural_row_distribution": True,
            "forced_context34_batch_quota": False,
        },
        "candidate_protocol": {
            "parameterization": "theta_new = theta - radius * unit(direction)",
            "fit_source_order": list(SOURCE_ORDER),
            "priority": [
                "P1_raw_source_mean_2_1_1",
                "P2_unit_source_mean_2_1_1",
                "P3_fixed_cyclic_pcgrad",
            ],
            "P1": "0.5*g_anti + 0.25*g_fros + 0.25*g_general",
            "P2": "0.5*unit(g_anti) + 0.25*unit(g_fros) + 0.25*unit(g_general)",
            "P3": (
                "fixed cyclic PCGrad on unit natural fit gradients in "
                "anti->fros->general order; eligible only if P1 and P2 each "
                "fail the complete natural-fit+cal and legacy-non-ascent base gate"
            ),
            "selection_targets": list(FIT_TARGETS + CAL_TARGETS),
            "natural_gate": {"dot_gt": 0.0, "cos_gt": COSINE_EPSILON},
            "legacy_gate": {
                "dot_gte": -NON_ASCENT_ABSOLUTE_TOLERANCE,
                "meaning": "non-ascent under theta_new convention",
            },
            "final_targets": list(FINAL_TARGETS),
            "final_participates_in_construction_or_selection": False,
        },
        "scope_audit": scope_audit,
        "output": {
            "path": str(OUTPUT_PATH),
            "exclusive_create": True,
            "json_only": True,
            "model_written": False,
        },
        "scope": {
            "optimizer": False,
            "model_update": False,
            "checkpoint_write": False,
            "gameplay": False,
            "package": False,
            "upload": False,
            "submission": False,
            "local_only": True,
        },
    }
    return plan


def unit(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector.double())
    if not torch.isfinite(norm) or float(norm) <= 0.0:
        raise RuntimeError("Cannot normalize zero/non-finite gradient")
    return vector.double() / norm


def fixed_cyclic_pcgrad(
    gradients: Mapping[str, torch.Tensor],
    order: Sequence[str] = SOURCE_ORDER,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if tuple(order) != SOURCE_ORDER:
        raise RuntimeError("PCGrad source order must remain anti->fros->general")
    references = {name: unit(gradients[name]) for name in order}
    projected: dict[str, torch.Tensor] = {}
    trace: list[dict[str, Any]] = []
    count = len(order)
    for index, name in enumerate(order):
        current = references[name].clone()
        for offset in range(1, count):
            other = order[(index + offset) % count]
            reference = references[other]
            before = float(torch.dot(current, reference))
            applied = before < 0.0
            if applied:
                current = current - before * reference
            trace.append(
                {
                    "task": name,
                    "reference": other,
                    "dot_before": before,
                    "conflict_projection_applied": applied,
                    "dot_after": float(torch.dot(current, reference)),
                }
            )
        projected[name] = current
    direction = sum(projected.values(), torch.zeros_like(next(iter(projected.values()))))
    return unit(direction), {"order": list(order), "projection_trace": trace}


def construct_candidates(
    natural_fit: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if set(natural_fit) != set(FIT_TARGETS):
        raise RuntimeError("Candidate construction requires exactly three natural fit gradients")
    raw = {source: natural_fit[f"{source}_fit"].double() for source in SOURCE_ORDER}
    p1 = 0.5 * raw["anti"] + 0.25 * raw["fros"] + 0.25 * raw["general"]
    p2 = (
        0.5 * unit(raw["anti"])
        + 0.25 * unit(raw["fros"])
        + 0.25 * unit(raw["general"])
    )
    p3, pcgrad_audit = fixed_cyclic_pcgrad(raw)
    return {
        "P1_raw_source_mean_2_1_1": unit(p1),
        "P2_unit_source_mean_2_1_1": unit(p2),
        "P3_fixed_cyclic_pcgrad": unit(p3),
    }, {"pcgrad": pcgrad_audit}


def geometry(direction: torch.Tensor, gradient: torch.Tensor) -> dict[str, Any]:
    d = direction.double()
    g = gradient.double()
    d_norm = float(torch.linalg.vector_norm(d))
    g_norm = float(torch.linalg.vector_norm(g))
    dot = float(torch.dot(g, d))
    cosine = dot / (d_norm * g_norm) if d_norm > 0.0 and g_norm > 0.0 else float("nan")
    robust = dot > 0.0 and cosine > COSINE_EPSILON
    return {
        "dot_gradient_direction": dot,
        "cosine_gradient_direction": cosine,
        "theta_minus_direction_first_order_descent": dot > 0.0,
        "robust_first_order_descent": robust,
    }


def gradient_pair(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    left = first.detach().cpu().double().reshape(-1)
    right = second.detach().cpu().double().reshape(-1)
    if left.shape != right.shape:
        raise RuntimeError("Gradient pair shape drift")
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    if (
        not math.isfinite(left_norm)
        or not math.isfinite(right_norm)
        or left_norm <= 0.0
        or right_norm <= 0.0
    ):
        raise RuntimeError("Gradient pair requires finite nonzero vectors")
    dot = float(torch.dot(left, right))
    cosine = dot / (left_norm * right_norm)
    if not math.isfinite(dot) or not math.isfinite(cosine):
        raise FloatingPointError("Non-finite gradient pair geometry")
    return {"dot": dot, "cosine": cosine}


def pairwise_gradient_matrix(
    gradients: Mapping[str, torch.Tensor],
    targets: Sequence[str],
) -> dict[str, Any]:
    if set(gradients) != set(targets):
        raise RuntimeError("Pairwise gradient target drift")
    ordered = tuple(targets)
    return {
        "targets": list(ordered),
        "dot": [
            [gradient_pair(gradients[left], gradients[right])["dot"] for right in ordered]
            for left in ordered
        ],
        "cosine": [
            [
                gradient_pair(gradients[left], gradients[right])["cosine"]
                for right in ordered
            ]
            for left in ordered
        ],
    }


def source_gradient_conflicts(
    gradients_by_mode: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    expected_modes = {"natural", "legacy_8_2"}
    if set(gradients_by_mode) != expected_modes:
        raise RuntimeError("Gradient conflict report mode drift")
    report: dict[str, Any] = {}
    for mode in ("natural", "legacy_8_2"):
        gradients = gradients_by_mode[mode]
        if set(gradients) != set(FIT_TARGETS + CAL_TARGETS + FINAL_TARGETS):
            raise RuntimeError("Gradient conflict report target drift")
        report[mode] = {
            "fit_source_pairwise": pairwise_gradient_matrix(
                {target: gradients[target] for target in FIT_TARGETS}, FIT_TARGETS
            ),
            "cal_source_pairwise": pairwise_gradient_matrix(
                {target: gradients[target] for target in CAL_TARGETS}, CAL_TARGETS
            ),
            "same_source_fit_vs_cal": {
                source: gradient_pair(
                    gradients[f"{source}_fit"], gradients[f"{source}_cal"]
                )
                for source in SOURCE_ORDER
            },
        }
    return report


def evaluate_and_select_candidates(
    candidates: Mapping[str, torch.Tensor],
    natural_fit: Mapping[str, torch.Tensor],
    natural_cal: Mapping[str, torch.Tensor],
    legacy_fit: Mapping[str, torch.Tensor],
    legacy_cal: Mapping[str, torch.Tensor],
) -> tuple[str | None, dict[str, Any]]:
    expected_groups = (
        (natural_fit, set(FIT_TARGETS)),
        (natural_cal, set(CAL_TARGETS)),
        (legacy_fit, set(FIT_TARGETS)),
        (legacy_cal, set(CAL_TARGETS)),
    )
    if any(set(group) != expected for group, expected in expected_groups):
        raise RuntimeError("Selection accepts only the six frozen fit/cal targets")
    priority = (
        "P1_raw_source_mean_2_1_1",
        "P2_unit_source_mean_2_1_1",
        "P3_fixed_cyclic_pcgrad",
    )
    if tuple(candidates) != priority:
        raise RuntimeError("Candidate priority drift")
    reports: dict[str, Any] = {}
    natural_fit_common: dict[str, bool] = {}
    base_eligible: dict[str, bool] = {}
    selected: str | None = None
    for name in priority:
        direction = candidates[name]
        natural = {
            target: geometry(direction, gradient)
            for target, gradient in {**natural_fit, **natural_cal}.items()
        }
        legacy = {
            target: geometry(direction, gradient)
            for target, gradient in {**legacy_fit, **legacy_cal}.items()
        }
        natural_fit_common[name] = all(
            natural[target]["robust_first_order_descent"] for target in FIT_TARGETS
        )
        natural_all = all(
            item["robust_first_order_descent"] for item in natural.values()
        )
        legacy_non_ascent = all(
            item["dot_gradient_direction"] >= -NON_ASCENT_ABSOLUTE_TOLERANCE
            for item in legacy.values()
        )
        base_eligible[name] = natural_all and legacy_non_ascent
        pcgrad_eligible = (
            name != "P3_fixed_cyclic_pcgrad"
            or not base_eligible[priority[0]]
            and not base_eligible[priority[1]]
        )
        eligible = base_eligible[name] and pcgrad_eligible
        reports[name] = {
            "direction": vector_record(direction),
            "natural": natural,
            "legacy_8_2": legacy,
            "natural_fit_common_descent": natural_fit_common[name],
            "natural_fit_and_cal_common_descent": natural_all,
            "legacy_fit_and_cal_non_ascent": legacy_non_ascent,
            "base_selection_eligible": base_eligible[name],
            "pcgrad_fallback_eligible": pcgrad_eligible,
            "selection_eligible": eligible,
        }
        if selected is None and eligible:
            selected = name
    return selected, reports


class WeightedGradientAccumulator:
    def __init__(self, elements: int) -> None:
        self.weighted_sum = torch.zeros(elements, dtype=torch.float64)
        self.weight_sum = 0.0
        self.chunks = 0

    def add(self, gradient: torch.Tensor, effective_weight: float) -> None:
        if gradient.numel() != self.weighted_sum.numel():
            raise RuntimeError("Gradient chunk shape drift")
        if not math.isfinite(effective_weight) or effective_weight <= 0.0:
            raise RuntimeError("Gradient chunk effective weight must be positive")
        value = gradient.detach().cpu().double().reshape(-1)
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite gradient chunk")
        self.weighted_sum.add_(value, alpha=effective_weight)
        self.weight_sum += effective_weight
        self.chunks += 1

    def finish(self) -> torch.Tensor:
        if self.chunks <= 0 or self.weight_sum <= 0.0:
            raise RuntimeError("No gradient chunks were accumulated")
        return self.weighted_sum / self.weight_sum


def _gradient_once(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
    allow_unused: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    loss_requires_grad = bool(loss.requires_grad)
    if loss_requires_grad:
        values = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=allow_unused,
        )
    elif allow_unused:
        values = tuple(None for _parameter in parameters)
    else:
        raise RuntimeError("Required gradient loss has no autograd graph")
    unused = [index for index, value in enumerate(values) if value is None]
    if unused and not allow_unused:
        raise RuntimeError(f"Unexpected unused gradient parameters: {unused}")
    chunks: list[torch.Tensor] = []
    connected_exact_zero: list[int] = []
    connected_nonzero: list[int] = []
    for index, (parameter, value) in enumerate(zip(parameters, values, strict=True)):
        if value is None:
            chunks.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
            continue
        detached = value.detach().float().reshape(-1)
        chunks.append(detached)
        if bool((detached == 0).all()):
            connected_exact_zero.append(index)
        else:
            connected_nonzero.append(index)
    audit = {
        "loss_requires_grad": loss_requires_grad,
        "autograd_invoked": loss_requires_grad,
        "parameter_count": len(parameters),
        "unused_parameter_indices": unused,
        "unused_parameter_count": len(unused),
        "connected_exact_zero_parameter_indices": connected_exact_zero,
        "connected_nonzero_parameter_indices": connected_nonzero,
        "all_parameters_unused_or_exact_zero": not connected_nonzero,
    }
    return torch.cat(chunks), audit


def _collate_features(
    features: list[dict[str, Any]],
    model_config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    return behavior_core.collate_ordered(
        features,
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )


def _featurize_rows(
    rows: Iterable[dict[str, Any]],
    model_config: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    previous_max = behavior_core.bc.MAX_ACTION_COUNT
    behavior_core.bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
    try:
        for row in rows:
            features = behavior_core.bc.featurize_row(
                row,
                int(model_config["hash_size"]),
                int(model_config["max_state_entities"]),
            )
            if features is None:
                raise RuntimeError("A frozen domain row failed BC featurization")
            action = row.get("action")
            if not isinstance(action, list):
                raise RuntimeError("Frozen row action changed during featurization")
            expert = [int(index) for index in action]
            options = (((row.get("observation") or {}).get("select") or {}).get("option") or [])
            if (
                len(expert) > ppo.MAX_ACTION_COUNT
                or len(set(expert)) != len(expert)
                or any(index < 0 or index >= len(options) for index in expert)
            ):
                raise RuntimeError("Frozen row has invalid ordered expert action")
            features["expert_action_order"] = expert
            yield features
    finally:
        behavior_core.bc.MAX_ACTION_COUNT = previous_max


def gradient_for_domain(
    model: torch.nn.Module,
    parameters: Sequence[torch.nn.Parameter],
    spec: DomainSpec,
    model_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    elements = sum(parameter.numel() for parameter in parameters)
    accumulators = {
        "natural": WeightedGradientAccumulator(elements),
        "legacy_8_2": WeightedGradientAccumulator(elements),
    }
    feature_rows = 0
    context34_rows = 0
    fixed_multi_rows = 0
    pointer_active_rows = 0
    batch: list[dict[str, Any]] = []
    count_actor6_dependency: dict[str, Any] | None = None

    def consume(features: list[dict[str, Any]]) -> None:
        nonlocal feature_rows, context34_rows, fixed_multi_rows
        nonlocal pointer_active_rows, count_actor6_dependency
        cpu_batch = _collate_features(features, model_config)
        gpu_batch = {
            key: value.to(device, non_blocking=False)
            for key, value in cpu_batch.items()
        }
        gpu_batch["action_sequences"] = gpu_batch["expert_ordered_actions"]
        outputs = ppo.model_forward(model, gpu_batch, device)
        natural_loss, natural_parts = ppo.bc_expert_actor_loss(
            outputs,
            gpu_batch,
            loss_mode="ordered",
            order_context_weight=NATURAL_WEIGHTS[0],
            non_context34_fixed_multi_action_order_weight=NATURAL_WEIGHTS[1],
        )
        legacy_loss, legacy_parts = ppo.bc_expert_actor_loss(
            outputs,
            gpu_batch,
            loss_mode="ordered",
            order_context_weight=LEGACY_WEIGHTS[0],
            non_context34_fixed_multi_action_order_weight=LEGACY_WEIGHTS[1],
        )
        natural_gradient, _natural_dependency = _gradient_once(
            natural_parts["selection_loss"],
            parameters,
            retain_graph=True,
            allow_unused=False,
        )
        if count_actor6_dependency is None:
            count_gradient, count_actor6_dependency = _gradient_once(
                natural_parts["count_loss"],
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            count_actor6_dependency["gradient"] = vector_record(count_gradient)
            count_actor6_dependency["checked_on_first_domain_batch"] = True
            if (
                count_actor6_dependency["unused_parameter_count"]
                != len(parameters)
                or not count_actor6_dependency[
                    "all_parameters_unused_or_exact_zero"
                ]
                or bool(count_gradient.count_nonzero())
            ):
                raise RuntimeError(
                    f"count_loss unexpectedly depends on actor6 in {spec.name}"
                )
        legacy_gradient, _legacy_dependency = _gradient_once(
            legacy_parts["selection_loss"],
            parameters,
            retain_graph=False,
            allow_unused=False,
        )
        del natural_loss, legacy_loss
        natural_weight = float(
            natural_parts["selection_effective_weight_sum"].detach().cpu()
        )
        legacy_weight = float(
            legacy_parts["selection_effective_weight_sum"].detach().cpu()
        )
        accumulators["natural"].add(natural_gradient, natural_weight)
        accumulators["legacy_8_2"].add(legacy_gradient, legacy_weight)
        rows = int(gpu_batch["action_counts"].numel())
        contexts = gpu_batch["contexts"]
        flexible = gpu_batch["min_counts"] != gpu_batch["max_counts"]
        fixed_multi = (
            (contexts != ppo.SKILL_ORDER_CONTEXT)
            & ~flexible
            & (gpu_batch["action_counts"] > 1)
        )
        feature_rows += rows
        context34_rows += int((contexts == ppo.SKILL_ORDER_CONTEXT).sum())
        fixed_multi_rows += int(fixed_multi.sum())
        pointer_active_rows += int((gpu_batch["action_counts"] > 0).sum())
        if any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError("autograd.grad materialized actor6 .grad buffers")

    for features in _featurize_rows(iter_raw_domain_rows(spec), model_config):
        batch.append(features)
        if len(batch) == BATCH_SIZE:
            consume(batch)
            batch = []
    if batch:
        consume(batch)
    expected = EXPECTED_DOMAIN_STATS[spec.name]
    feature_checks = {
        "rows": feature_rows == expected["rows"],
        "context34_rows": context34_rows == expected["context34_rows"],
        "fixed_multi_rows": fixed_multi_rows == expected["fixed_multi_rows"],
        "all_rows_covered_once": feature_rows == expected["rows"],
    }
    if not all(feature_checks.values()):
        raise RuntimeError(f"Featurized domain drift for {spec.name}: {feature_checks}")
    gradients = {name: accumulator.finish() for name, accumulator in accumulators.items()}
    if count_actor6_dependency is None:
        raise RuntimeError(f"No count-loss actor6 dependency audit for {spec.name}")
    report = {
        "rows": feature_rows,
        "batches": accumulators["natural"].chunks,
        "pointer_active_rows": pointer_active_rows,
        "context34_rows": context34_rows,
        "fixed_multi_rows": fixed_multi_rows,
        "feature_checks": feature_checks,
        "count_actor6_dependency": count_actor6_dependency,
        "natural": {
            "selection_effective_weight_sum": accumulators["natural"].weight_sum,
            "gradient": vector_record(gradients["natural"]),
        },
        "legacy_8_2": {
            "selection_effective_weight_sum": accumulators["legacy_8_2"].weight_sum,
            "gradient": vector_record(gradients["legacy_8_2"]),
        },
    }
    return gradients, report


def actual_delta_geometry(
    delta: torch.Tensor,
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    descent_direction = -unit(delta)
    report = {
        "parameter_delta": vector_record(delta),
        "equivalent_unit_descent_direction": vector_record(descent_direction),
        "sign_convention": (
            "actual theta_delta is compared as direction=-unit(theta_delta), "
            "so positive g dot direction means the actual update was descending"
        ),
    }
    report.update({
        mode: {
            target: geometry(descent_direction, gradient)
            for target, gradient in targets.items()
        }
        for mode, targets in gradients.items()
    })
    return report


def execute(plan: dict[str, Any]) -> dict[str, Any]:
    assert_output_absent(OUTPUT_PATH, "geometry report")
    device = torch.device(DEVICE)
    if device.type != "cuda" or device.index != 0 or not torch.cuda.is_available():
        raise RuntimeError("Formal geometry requires available cuda:0")
    torch.manual_seed(2026081053)
    torch.cuda.manual_seed_all(2026081053)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial_hashes = snapshot_input_hashes()

    parent = torch.load(S8_PARENT, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(BC_ARCHITECTURE, map_location="cpu", weights_only=False)
    e100 = torch.load(FIRST_ROUND_E100, map_location="cpu", weights_only=False)
    model = ppo.instantiate_model_from_checkpoint(parent, bc_checkpoint, device)
    parameters = training_core.configure_actor6(model)
    if tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ) != ACTOR6:
        raise RuntimeError("Trainable geometry scope differs from actor6")
    model.eval()
    model_state_before = ppo.model_state_sha256(model)
    gradients_by_mode: dict[str, dict[str, torch.Tensor]] = {
        "natural": {},
        "legacy_8_2": {},
    }
    gradient_reports: dict[str, Any] = {}
    for spec in DOMAIN_SPECS:
        gradients, report = gradient_for_domain(
            model, parameters, spec, parent["model_config"], device
        )
        for mode in gradients_by_mode:
            gradients_by_mode[mode][spec.name] = gradients[mode]
        gradient_reports[spec.name] = report
    model_state_after = ppo.model_state_sha256(model)
    if model_state_after != model_state_before:
        raise RuntimeError("Zero-update probe changed model state")
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("Zero-update probe left .grad buffers")

    natural_fit = {name: gradients_by_mode["natural"][name] for name in FIT_TARGETS}
    natural_cal = {name: gradients_by_mode["natural"][name] for name in CAL_TARGETS}
    legacy_fit = {name: gradients_by_mode["legacy_8_2"][name] for name in FIT_TARGETS}
    legacy_cal = {name: gradients_by_mode["legacy_8_2"][name] for name in CAL_TARGETS}
    candidates, construction_audit = construct_candidates(natural_fit)
    selected, candidate_reports = evaluate_and_select_candidates(
        candidates, natural_fit, natural_cal, legacy_fit, legacy_cal
    )
    final_reports = {
        candidate: {
            mode: {
                target: geometry(direction, gradients_by_mode[mode][target])
                for target in FINAL_TARGETS
            }
            for mode in ("natural", "legacy_8_2")
        }
        for candidate, direction in candidates.items()
    }
    parent_delta = (
        flatten_actor6(e100["model_state_dict"])
        - flatten_actor6(parent["model_state_dict"])
    )
    e100_geometry = actual_delta_geometry(parent_delta, gradients_by_mode)
    conflict_report = source_gradient_conflicts(gradients_by_mode)

    final_hashes = snapshot_input_hashes()
    if final_hashes != initial_hashes:
        raise RuntimeError("A frozen probe input changed during execution")
    result = {
        "schema_version": SCHEMA_VERSION + "-result",
        "status": "candidate_selected" if selected is not None else "no_safe_direction",
        "frozen_plan_sha256": sha256_json(plan),
        "model_state_sha256_before": model_state_before,
        "model_state_sha256_after": model_state_after,
        "model_state_bit_identical": True,
        "gradient_reports": gradient_reports,
        "source_gradient_conflicts": conflict_report,
        "candidate_construction": construction_audit,
        "candidate_selection": {
            "selected": selected,
            "priority": list(candidates),
            "reports": candidate_reports,
            "selection_used_only_fit_and_cal": True,
            "final_views_used_for_selection": False,
        },
        "final_diagnostics_report_only": final_reports,
        "first_round_e100_delta_geometry": e100_geometry,
        "inputs_unchanged_after_probe": True,
        "observed_input_hashes": final_hashes,
        "persistent_output": str(OUTPUT_PATH),
        "model_written": False,
        "gameplay_performed": False,
        "package_performed": False,
        "upload_performed": False,
        "submission_performed": False,
    }
    if not all_finite(result):
        raise FloatingPointError("Geometry result contains non-finite values")
    write_exclusive(OUTPUT_PATH, canonical_json_bytes(result))
    return result


def all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(child) for child in value)
    return True


def plan_envelope(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION + "-frozen-plan-envelope",
        "plan_sha256": sha256_json(plan),
        "plan": plan,
    }


def load_frozen_plan(path: Path, expected_plan_sha256: str) -> dict[str, Any]:
    if path.resolve() != PLAN_PATH.resolve():
        raise RuntimeError(f"Frozen plan path must be {PLAN_PATH}")
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing frozen plan: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError("Frozen plan must be a non-symlink regular file")
    raw = path.read_bytes()
    envelope = strict_json_bytes(raw, "frozen geometry plan")
    if not isinstance(envelope, dict) or not isinstance(envelope.get("plan"), dict):
        raise RuntimeError("Frozen plan envelope is malformed")
    if raw != canonical_json_bytes(envelope):
        raise RuntimeError("Frozen plan is not canonical JSON plus one LF")
    plan = envelope["plan"]
    observed = sha256_json(plan)
    if envelope.get("plan_sha256") != observed or observed != expected_plan_sha256:
        raise RuntimeError("Frozen plan SHA-256 mismatch")
    return plan


def enforce_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run from repository root: {ROOT}")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"Wrong Python runtime: {sys.executable}")
    if Path(sys.prefix).resolve() != EXPECTED_ENV_PREFIX.resolve():
        raise RuntimeError(f"Wrong Python environment: {sys.prefix}")
    observed_versions = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "orjson": orjson.__version__,
    }
    expected_versions = {
        "torch": EXPECTED_TORCH_VERSION,
        "torch_cuda": EXPECTED_TORCH_CUDA_VERSION,
        "cudnn": EXPECTED_CUDNN_VERSION,
        "orjson": EXPECTED_ORJSON_VERSION,
    }
    if observed_versions != expected_versions:
        raise RuntimeError(
            f"Runtime library version drift: expected={expected_versions}, "
            f"observed={observed_versions}"
        )
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("Run with my_project_env Python flags -I -B")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != EXPECTED_CUBLAS:
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must be {EXPECTED_CUBLAS}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_CUDA_VISIBLE_DEVICES:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--freeze-plan", type=Path, metavar="PATH")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--frozen-plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args(argv)
    if args.freeze_plan is not None and args.freeze_plan.resolve() != PLAN_PATH.resolve():
        parser.error(f"--freeze-plan must be {PLAN_PATH}")
    if args.execute:
        if args.frozen_plan is None or not args.expected_plan_sha256:
            parser.error("--execute requires --frozen-plan and --expected-plan-sha256")
    elif args.frozen_plan is not None or args.expected_plan_sha256 is not None:
        parser.error("frozen-plan arguments are valid only with --execute")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    enforce_runtime()
    plan = build_plan()
    plan_sha256 = sha256_json(plan)
    if args.dry_run:
        print(json.dumps({"plan_sha256": plan_sha256, "plan": plan}, indent=2, sort_keys=True))
        return 0
    if args.freeze_plan is not None:
        assert_output_absent(args.freeze_plan, "reviewed geometry plan")
        write_exclusive(args.freeze_plan, canonical_json_bytes(plan_envelope(plan)))
        print(json.dumps({"frozen_plan": str(args.freeze_plan), "plan_sha256": plan_sha256}, sort_keys=True))
        return 0
    frozen = load_frozen_plan(args.frozen_plan, args.expected_plan_sha256)
    if frozen != plan or args.expected_plan_sha256 != plan_sha256:
        raise RuntimeError("Current audited plan differs from frozen reviewed plan")
    result = execute(plan)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "candidate_selected" else 42


if __name__ == "__main__":
    raise SystemExit(main())
