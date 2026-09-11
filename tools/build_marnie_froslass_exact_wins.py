#!/usr/bin/env python3
"""Build a frozen exact Marnie-win-vs-Froslass training archive.

The only data inputs are the frozen ``marnie_trainwins`` BC archive and the
five local official daily replay ZIPs.  Replay ``visualize`` is inspected only
to prove both submitted deck hashes; it is never copied.  Selected JSONL rows
are copied byte for byte from the source train shards.

The default mode is a full, write-free dry run.  ``--execute`` additionally
requires the plan SHA-256 and output SHA-256 reported by a reviewed dry run.
The ZIP is serialized twice in memory and the two payloads must be identical
before it can be published without clobbering an existing path.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import orjson


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ptcg-marnie-froslass-exact-wins-v1"
ALLOWLIST_SCHEMA_VERSION = (
    "ptcg-marnie-vs-exact-fros-trainwin-allowlist-v1"
)
SOURCE_SCHEMA_VERSION = "ptcg-bc-visible-decisions-v1"
ROW_SCHEMA_VERSION = "ptcg-bc-visible-decisions-v1"
DECISION_KEY_FORMAT = "dataset_date|episode_id|seat|action_step_index"
EPISODE_KEY_FORMAT = "dataset_date|episode_id"
TRAIN_MEMBER = "train/part-00000.jsonl"
ALLOWLIST_MEMBER = "exact_episode_allowlist.json"
MANIFEST_MEMBER = "manifest.json"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_COMPRESSION = zipfile.ZIP_DEFLATED
ZIP_COMPRESSLEVEL = 6
SHA256_RE = re.compile(r"[0-9a-f]{64}")

MARNIE_DECK_HASH = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
FROSLASS_LOPUNNY_DECK_HASH = (
    "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc"
)
DATES = (
    "2026-08-02",
    "2026-08-03",
    "2026-08-04",
    "2026-08-05",
    "2026-08-06",
)
EXPECTED_DATE_EPISODES = (
    ("2026-08-02", 26),
    ("2026-08-03", 11),
    ("2026-08-04", 23),
    ("2026-08-05", 29),
    ("2026-08-06", 15),
)

DEFAULT_SOURCE = (
    REPO_ROOT
    / "data/gold_push_recent7_20260810_v1/archives/marnie_trainwins.zip"
)
DEFAULT_DAILY_DIR = (
    REPO_ROOT / "data/episodes_cache/gold_push_20260810_v1/daily"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/gold_push_marnie_fros_exact_20260810_v1"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "marnie_vs_froslass_trainwins.zip"
DEFAULT_ALLOWLIST = DEFAULT_OUTPUT_ROOT / ALLOWLIST_MEMBER

SOURCE_SHA256 = "7cc2a4cb38b857ccdacf9cc84fe4610400ca3295c9ab1dbefbcb75e4341e8eab"
SOURCE_MANIFEST_SHA256 = (
    "217d62d390b5efad4368227e693784abbc89a2e4ac5ac54a814d56ba7c3959e2"
)
DAILY_SHA256 = (
    (
        "2026-08-02",
        "fa91e058a42d5fffab0f3e63f04fba5acc9bfbd2e2225e97aa62f45f5d430eb8",
    ),
    (
        "2026-08-03",
        "909cbd205f3afcfde6031ae93ef9625b796e8a0c2edf66eeb6edc88469273a04",
    ),
    (
        "2026-08-04",
        "17cd9cd92f4ae3b293ee3fab3452657316362af134c6d4a7b5dbfda99c3d3d42",
    ),
    (
        "2026-08-05",
        "ab961e0d98984b611cc4091801b618606cb03cab4413ab7908d3f8c6312030e3",
    ),
    (
        "2026-08-06",
        "46f3a95ba0456027870b504a64424fd3f9afcf3aabe5d0453803d7c5145631a4",
    ),
)
DAILY_MANIFEST_SHA256 = (
    (
        "2026-08-02",
        "0d63fea5c93db6458a856c332aeb128b1098bcda44ef35a3a148236f47c389e4",
    ),
    (
        "2026-08-03",
        "ab80af203ef7958503ab5f244bee8adc377eb3b8da71504b7c83e348ffa73991",
    ),
    (
        "2026-08-04",
        "bb190f62f0585dc2a1db2b02752a4d7e6fa6de15a800ed9e769d8daecd8bf9a1",
    ),
    (
        "2026-08-05",
        "b8645318d7b20792dff4f327d5aaeadd98b09fc6064fdcceb5d92897cee7ac11",
    ),
    (
        "2026-08-06",
        "96392a60c27d98e6be7ea45427228ac2a3a9c35b60c4f69f78cf3408018515a3",
    ),
)
DAILY_MISSING_REPLAY_IDS = (
    ("2026-08-02", ("89488639",)),
    (
        "2026-08-03",
        ("89719318", "89721535", "89733649", "89757221"),
    ),
    (
        "2026-08-04",
        ("89855212", "89919753", "90002731", "90035083", "90041269"),
    ),
    ("2026-08-05", ("90144162", "90161497", "90197034")),
    ("2026-08-06", ("90393929", "90481043")),
)
SOURCE_MEMBERS = (
    "train/part-00000.jsonl",
    "train/part-00001.jsonl",
    "train/part-00002.jsonl",
    "train/part-00003.jsonl",
    "train/part-00004.jsonl",
    "train/part-00005.jsonl",
    "train/part-00006.jsonl",
    "train/part-00007.jsonl",
    "valid/part-00000.jsonl",
    "valid/part-00001.jsonl",
    "test/part-00000.jsonl",
    "test/part-00001.jsonl",
    "manifest.json",
)
SOURCE_TRAIN_MEMBERS = tuple(
    member for member in SOURCE_MEMBERS if member.startswith("train/")
)


@dataclass(frozen=True)
class IncompleteTerminalReplayContract:
    dataset_date: str
    episode_id: str
    episode_uuid: str
    replay_sha256: str
    seat: int
    team_name: str
    opponent_team_name: str
    source_terminal_reward: float
    raw_rewards: tuple[float | None, float | None]
    learner_deck_hash: str
    opponent_deck_hash: str


INCOMPLETE_TERMINAL_REPLAYS = (
    IncompleteTerminalReplayContract(
        dataset_date="2026-08-05",
        episode_id="90239373",
        episode_uuid="5a23cc38-90fe-11f1-a931-0242ac130203",
        replay_sha256=(
            "69d561cad567aec936d708501003f481e83fd350fbde7831dc5b0dca24d26774"
        ),
        seat=0,
        team_name="vvs",
        opponent_team_name="Bart, Edwyn",
        source_terminal_reward=1.0,
        raw_rewards=(1.0, None),
        learner_deck_hash=MARNIE_DECK_HASH,
        opponent_deck_hash=(
            "f06bd3d5964e5f6d7051339db56aff17f4b111043a2cacce1621d10f52806368"
        ),
    ),
)


@dataclass(frozen=True)
class BuildContract:
    source_sha256: str
    source_manifest_sha256: str
    daily_sha256: tuple[tuple[str, str], ...]
    daily_manifest_sha256: tuple[tuple[str, str], ...]
    daily_missing_replay_ids: tuple[tuple[str, tuple[str, ...]], ...]
    dates: tuple[str, ...]
    expected_date_episodes: tuple[tuple[str, int], ...]
    expected_source_train_rows: int
    expected_source_train_episodes: int
    expected_selected_rows: int
    expected_selected_episodes: int
    learner_deck_hash: str
    opponent_deck_hash: str
    source_members: tuple[str, ...]
    source_train_members: tuple[str, ...]
    source_logical_path: str
    daily_logical_dir: str
    incomplete_terminal_replays: tuple[IncompleteTerminalReplayContract, ...]


DEFAULT_CONTRACT = BuildContract(
    source_sha256=SOURCE_SHA256,
    source_manifest_sha256=SOURCE_MANIFEST_SHA256,
    daily_sha256=DAILY_SHA256,
    daily_manifest_sha256=DAILY_MANIFEST_SHA256,
    daily_missing_replay_ids=DAILY_MISSING_REPLAY_IDS,
    dates=DATES,
    expected_date_episodes=EXPECTED_DATE_EPISODES,
    expected_source_train_rows=187_611,
    expected_source_train_episodes=1_909,
    expected_selected_rows=10_614,
    expected_selected_episodes=104,
    learner_deck_hash=MARNIE_DECK_HASH,
    opponent_deck_hash=FROSLASS_LOPUNNY_DECK_HASH,
    source_members=SOURCE_MEMBERS,
    source_train_members=SOURCE_TRAIN_MEMBERS,
    source_logical_path=(
        "data/gold_push_recent7_20260810_v1/archives/marnie_trainwins.zip"
    ),
    daily_logical_dir="data/episodes_cache/gold_push_20260810_v1/daily",
    incomplete_terminal_replays=INCOMPLETE_TERMINAL_REPLAYS,
)


@dataclass
class EpisodeSourceAudit:
    dataset_date: str
    episode_id: str
    episode_uuid: str | None
    seat: int
    team_name: str
    opponent_team_name: str
    terminal_reward: float
    rows: int
    raw_rows_sha256: Any
    decision_keys: list[str]


@dataclass(frozen=True)
class SourceAudit:
    manifest: dict[str, Any]
    episodes: dict[tuple[str, str], EpisodeSourceAudit]
    train_rows: int
    train_member_sha256: tuple[tuple[str, str], ...]
    train_member_bytes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class SelectedEpisode:
    dataset_date: str
    episode_id: str
    replay_member: str
    replay_sha256: str
    seat: int
    team_name: str
    opponent_team_name: str
    replay_rewards: tuple[float, float]
    rows: int
    raw_rows_sha256: str
    decision_keys_digest: str


@dataclass(frozen=True)
class DailyAudit:
    dataset_date: str
    filename: str
    sha256: str
    manifest_sha256: str
    manifest_episode_ids: int
    replay_members: int
    missing_replay_ids: tuple[str, ...]
    missing_source_episode_overlap: int
    incomplete_terminal_replay_ids: tuple[str, ...]
    source_episodes_checked: int


@dataclass(frozen=True)
class SelectionAudit:
    source: SourceAudit
    selected: tuple[SelectedEpisode, ...]
    daily: tuple[DailyAudit, ...]
    selected_lines: tuple[bytes, ...]
    selected_content_sha256: str
    selected_decision_keys_digest: str
    selected_date_rows: tuple[tuple[str, int], ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_sha256(value: str, label: str) -> None:
    if not SHA256_RE.fullmatch(value):
        raise RuntimeError(f"{label} must be a lowercase SHA256, got {value!r}")


def require_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")


def lexical_absolute_path(path: Path) -> Path:
    """Return an absolute normalized path without resolving symlinks."""

    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def audit_path_components(
    path: Path,
    label: str,
    *,
    must_exist: bool,
    kind: str,
) -> Path:
    """Reject symlink traversal before any call to ``Path.resolve``.

    ``kind`` is ``file``, ``directory``, or ``output``.  Output leaves must be
    absent, while all existing ancestors must be real directories.
    """

    if kind not in {"file", "directory", "output"}:
        raise ValueError(f"unsupported path audit kind: {kind}")
    absolute = lexical_absolute_path(path)
    current = Path(absolute.anchor)
    missing_seen = False
    for part in absolute.parts[1:]:
        current = current / part
        if missing_seen:
            continue
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            missing_seen = True
            continue
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"{label} traverses symlink component: {current}")
        if current != absolute and not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"{label} ancestor is not a directory: {current}")
    exists_by_lstat = os.path.lexists(absolute)
    if kind == "output":
        if exists_by_lstat:
            raise FileExistsError(f"refusing to reuse existing output: {absolute}")
        return absolute
    if must_exist and not exists_by_lstat:
        raise FileNotFoundError(f"{label} does not exist: {absolute}")
    if exists_by_lstat:
        info = os.lstat(absolute)
        if kind == "file" and not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {absolute}")
        if kind == "directory" and not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"{label} is not a directory: {absolute}")
    return absolute


def require_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer, got {value!r}")
    return value


def numeric_reward(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be numeric, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be numeric, got {value!r}") from exc
    if result not in (-1.0, 0.0, 1.0):
        raise RuntimeError(f"{label} has unexpected terminal value {result!r}")
    return result


def validate_contract(contract: BuildContract) -> None:
    if not contract.dates or tuple(sorted(contract.dates)) != contract.dates:
        raise RuntimeError("contract dates must be non-empty, unique, and sorted")
    if len(set(contract.dates)) != len(contract.dates):
        raise RuntimeError("contract dates contain duplicates")
    daily = dict(contract.daily_sha256)
    daily_manifests = dict(contract.daily_manifest_sha256)
    missing_replays = dict(contract.daily_missing_replay_ids)
    expected_dates = dict(contract.expected_date_episodes)
    if len(daily) != len(contract.daily_sha256) or set(daily) != set(contract.dates):
        raise RuntimeError("daily SHA contract does not exactly cover dates")
    if len(expected_dates) != len(contract.expected_date_episodes) or set(
        expected_dates
    ) != set(contract.dates):
        raise RuntimeError("date-count contract does not exactly cover dates")
    if len(daily_manifests) != len(contract.daily_manifest_sha256) or set(
        daily_manifests
    ) != set(contract.dates):
        raise RuntimeError("daily manifest SHA contract does not exactly cover dates")
    if len(missing_replays) != len(contract.daily_missing_replay_ids) or set(
        missing_replays
    ) != set(contract.dates):
        raise RuntimeError("daily missing-replay contract does not exactly cover dates")
    require_sha256(contract.source_sha256, "source_sha256")
    require_sha256(contract.source_manifest_sha256, "source_manifest_sha256")
    require_sha256(contract.learner_deck_hash, "learner_deck_hash")
    require_sha256(contract.opponent_deck_hash, "opponent_deck_hash")
    for dataset_date, digest in contract.daily_sha256:
        require_sha256(digest, f"daily_sha256[{dataset_date}]")
    for dataset_date, digest in contract.daily_manifest_sha256:
        require_sha256(digest, f"daily_manifest_sha256[{dataset_date}]")
    for dataset_date, episode_ids in contract.daily_missing_replay_ids:
        if (
            any(not value.isdigit() for value in episode_ids)
            or len(episode_ids) != len(set(episode_ids))
            or tuple(sorted(episode_ids, key=int)) != episode_ids
        ):
            raise RuntimeError(
                f"daily missing-replay IDs are invalid for {dataset_date}"
            )
    incomplete_keys = [
        (value.dataset_date, value.episode_id)
        for value in contract.incomplete_terminal_replays
    ]
    if (
        len(incomplete_keys) != len(set(incomplete_keys))
        or incomplete_keys != sorted(incomplete_keys, key=lambda value: (value[0], int(value[1])))
    ):
        raise RuntimeError("incomplete-terminal replay keys are duplicated or unsorted")
    for index, value in enumerate(contract.incomplete_terminal_replays):
        if value.dataset_date not in contract.dates or not value.episode_id.isdigit():
            raise RuntimeError(f"incomplete-terminal replay key drift at {index}")
        for digest, label in (
            (value.replay_sha256, "replay_sha256"),
            (value.learner_deck_hash, "learner_deck_hash"),
            (value.opponent_deck_hash, "opponent_deck_hash"),
        ):
            require_sha256(digest, f"incomplete_terminal_replays[{index}].{label}")
        if (
            value.seat not in (0, 1)
            or not value.episode_uuid
            or not value.team_name
            or not value.opponent_team_name
            or value.source_terminal_reward <= 0
            or value.raw_rewards[value.seat] != value.source_terminal_reward
            or value.raw_rewards[1 - value.seat] is not None
            or value.learner_deck_hash != contract.learner_deck_hash
            or value.opponent_deck_hash == contract.opponent_deck_hash
        ):
            raise RuntimeError(f"incomplete-terminal replay identity drift at {index}")
    if contract.learner_deck_hash == contract.opponent_deck_hash:
        raise RuntimeError("learner and opponent deck hashes must differ")
    positive_counts = (
        contract.expected_source_train_rows,
        contract.expected_source_train_episodes,
        contract.expected_selected_rows,
        contract.expected_selected_episodes,
    )
    if any(value <= 0 for value in positive_counts):
        raise RuntimeError("all frozen count gates must be positive")
    if sum(expected_dates.values()) != contract.expected_selected_episodes:
        raise RuntimeError("date episode counts do not sum to selected episode gate")
    if len(set(contract.source_members)) != len(contract.source_members):
        raise RuntimeError("source member contract contains duplicates")
    if tuple(
        name for name in contract.source_members if name.startswith("train/")
    ) != contract.source_train_members:
        raise RuntimeError("source train-member contract is inconsistent")
    if MANIFEST_MEMBER not in contract.source_members:
        raise RuntimeError("source member contract omits manifest.json")


def validate_source_manifest(
    payload: bytes,
    contract: BuildContract,
) -> dict[str, Any]:
    actual_sha = sha256_bytes(payload)
    if actual_sha != contract.source_manifest_sha256:
        raise RuntimeError(
            "source manifest SHA256 mismatch: "
            f"expected {contract.source_manifest_sha256}, got {actual_sha}"
        )
    try:
        manifest = orjson.loads(payload)
    except orjson.JSONDecodeError as exc:
        raise RuntimeError("source manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("source manifest is not an object")
    profile = manifest.get("profile")
    split_policy = manifest.get("split_policy")
    split_decisions = manifest.get("split_decisions")
    split_episodes = manifest.get("split_episodes")
    reward_filter = manifest.get("terminal_reward_filter")
    checks = {
        "schema_version": (manifest.get("schema_version"), SOURCE_SCHEMA_VERSION),
        "profile.deck_hash": (
            profile.get("deck_hash") if isinstance(profile, dict) else None,
            contract.learner_deck_hash,
        ),
        "deck_hash_filter": (
            manifest.get("deck_hash_filter"),
            contract.learner_deck_hash,
        ),
        "split_policy.train_dates": (
            tuple(split_policy.get("train_dates", ()))
            if isinstance(split_policy, dict)
            else None,
            contract.dates,
        ),
        "split_decisions.train": (
            split_decisions.get("train")
            if isinstance(split_decisions, dict)
            else None,
            contract.expected_source_train_rows,
        ),
        "split_episodes.train": (
            split_episodes.get("train")
            if isinstance(split_episodes, dict)
            else None,
            contract.expected_source_train_episodes,
        ),
        "terminal_reward_filter.mode": (
            reward_filter.get("mode")
            if isinstance(reward_filter, dict)
            else None,
            "wins",
        ),
        "terminal_reward_filter.splits": (
            reward_filter.get("splits")
            if isinstance(reward_filter, dict)
            else None,
            "train",
        ),
    }
    failures = [
        f"{name}={actual!r} expected {wanted!r}"
        for name, (actual, wanted) in checks.items()
        if actual != wanted
    ]
    if failures:
        raise RuntimeError("source manifest drift: " + "; ".join(failures))
    return manifest


def row_identity(
    row: Mapping[str, Any],
    where: str,
    contract: BuildContract,
) -> tuple[str, str, int, int]:
    dataset_date = row.get("dataset_date")
    if dataset_date not in contract.dates:
        raise RuntimeError(f"{where}: dataset_date is outside frozen dates")
    episode_id = row.get("episode_id")
    episode_text = str(episode_id) if episode_id is not None else ""
    if not episode_text.isdigit():
        raise RuntimeError(f"{where}: episode_id is not numeric: {episode_id!r}")
    seat = require_integer(row.get("seat"), f"{where}: seat")
    if seat not in (0, 1):
        raise RuntimeError(f"{where}: seat must be 0 or 1")
    action_step = require_integer(
        row.get("action_step_index"),
        f"{where}: action_step_index",
    )
    observation_step = require_integer(
        row.get("observation_step_index"),
        f"{where}: observation_step_index",
    )
    if action_step <= 0 or observation_step != action_step - 1:
        raise RuntimeError(f"{where}: observation/action alignment drift")
    episode_key = f"{dataset_date}|{episode_text}"
    return (
        f"{episode_key}|{seat}|{action_step}",
        episode_text,
        seat,
        action_step,
    )


def validate_source_row(
    value: Any,
    where: str,
    contract: BuildContract,
) -> tuple[str, str, str, int, float]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{where}: row is not an object")
    checks = {
        "schema_version": (value.get("schema_version"), ROW_SCHEMA_VERSION),
        "split": (value.get("split"), "train"),
        "deck_hash": (value.get("deck_hash"), contract.learner_deck_hash),
        "sample_weight": (value.get("sample_weight"), 1.0),
    }
    failures = [
        f"{name}={actual!r} expected {wanted!r}"
        for name, (actual, wanted) in checks.items()
        if actual != wanted
    ]
    if failures:
        raise RuntimeError(f"{where}: source row drift: {'; '.join(failures)}")
    if "visualize" in value or (
        isinstance(value.get("observation"), dict)
        and "visualize" in value["observation"]
    ):
        raise RuntimeError(f"{where}: hidden visualize payload is forbidden")
    if not isinstance(value.get("observation"), dict):
        raise RuntimeError(f"{where}: observation is not an object")
    if not isinstance(value.get("action"), list):
        raise RuntimeError(f"{where}: action is not a list")
    team_name = value.get("team_name")
    opponent_name = value.get("opponent_team_name")
    if not isinstance(team_name, str) or not team_name:
        raise RuntimeError(f"{where}: team_name is empty")
    if not isinstance(opponent_name, str) or not opponent_name:
        raise RuntimeError(f"{where}: opponent_team_name is empty")
    reward = numeric_reward(value.get("terminal_reward"), f"{where}: reward")
    if reward <= 0:
        raise RuntimeError(f"{where}: source train-win row is not a learner win")
    decision_key, episode_id, seat, _step = row_identity(value, where, contract)
    return decision_key, episode_id, team_name, seat, reward


def scan_source(source: Path, contract: BuildContract) -> SourceAudit:
    require_regular_file(source, "source archive")
    actual_source_sha = sha256_file(source)
    if actual_source_sha != contract.source_sha256:
        raise RuntimeError(
            "source archive SHA256 mismatch: "
            f"expected {contract.source_sha256}, got {actual_source_sha}"
        )
    episodes: dict[tuple[str, str], EpisodeSourceAudit] = {}
    seen_decisions: set[str] = set()
    train_rows = 0
    member_hashes: list[tuple[str, str]] = []
    member_sizes: list[tuple[str, int]] = []
    with zipfile.ZipFile(source) as archive:
        names = tuple(archive.namelist())
        if len(names) != len(set(names)):
            raise RuntimeError("source archive has duplicate member names")
        if names != contract.source_members:
            raise RuntimeError(
                "source archive member/order drift: "
                f"expected {contract.source_members!r}, got {names!r}"
            )
        manifest = validate_source_manifest(
            archive.read(MANIFEST_MEMBER),
            contract,
        )
        for member in contract.source_train_members:
            member_digest = hashlib.sha256()
            member_bytes = 0
            with archive.open(member) as handle:
                for line_number, raw_line in enumerate(handle, 1):
                    where = f"{member}:{line_number}"
                    if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
                        raise RuntimeError(f"{where}: row lacks canonical LF terminator")
                    member_digest.update(raw_line)
                    member_bytes += len(raw_line)
                    try:
                        row = orjson.loads(raw_line)
                    except orjson.JSONDecodeError as exc:
                        raise RuntimeError(f"{where}: invalid JSON") from exc
                    (
                        decision_key,
                        episode_id,
                        team_name,
                        seat,
                        reward,
                    ) = validate_source_row(row, where, contract)
                    if decision_key in seen_decisions:
                        raise RuntimeError(f"duplicate source decision: {decision_key}")
                    seen_decisions.add(decision_key)
                    dataset_date = str(row["dataset_date"])
                    key = (dataset_date, episode_id)
                    opponent_name = str(row["opponent_team_name"])
                    raw_uuid = row.get("episode_uuid")
                    episode_uuid = str(raw_uuid) if raw_uuid is not None else None
                    audit = episodes.get(key)
                    if audit is None:
                        audit = EpisodeSourceAudit(
                            dataset_date=dataset_date,
                            episode_id=episode_id,
                            episode_uuid=episode_uuid,
                            seat=seat,
                            team_name=team_name,
                            opponent_team_name=opponent_name,
                            terminal_reward=reward,
                            rows=0,
                            raw_rows_sha256=hashlib.sha256(),
                            decision_keys=[],
                        )
                        episodes[key] = audit
                    identity = (
                        episode_uuid,
                        seat,
                        team_name,
                        opponent_name,
                        reward,
                    )
                    expected_identity = (
                        audit.episode_uuid,
                        audit.seat,
                        audit.team_name,
                        audit.opponent_team_name,
                        audit.terminal_reward,
                    )
                    if identity != expected_identity:
                        raise RuntimeError(
                            f"{where}: within-episode source identity drift"
                        )
                    audit.rows += 1
                    audit.raw_rows_sha256.update(raw_line)
                    audit.decision_keys.append(decision_key)
                    train_rows += 1
            member_hashes.append((member, member_digest.hexdigest()))
            member_sizes.append((member, member_bytes))

    if train_rows != contract.expected_source_train_rows:
        raise RuntimeError(
            "source train row-count drift: "
            f"expected {contract.expected_source_train_rows}, got {train_rows}"
        )
    if len(episodes) != contract.expected_source_train_episodes:
        raise RuntimeError(
            "source train episode-count drift: "
            f"expected {contract.expected_source_train_episodes}, got {len(episodes)}"
        )
    for episode in episodes.values():
        if episode.rows != len(episode.decision_keys):
            raise AssertionError("episode row/key accounting mismatch")
    return SourceAudit(
        manifest=manifest,
        episodes=episodes,
        train_rows=train_rows,
        train_member_sha256=tuple(member_hashes),
        train_member_bytes=tuple(member_sizes),
    )


def replay_names(replay: Mapping[str, Any]) -> tuple[str, str]:
    info = replay.get("info")
    info = info if isinstance(info, Mapping) else {}
    team_names = info.get("TeamNames")
    if isinstance(team_names, list) and len(team_names) >= 2:
        names = tuple(str(value or "").strip() for value in team_names[:2])
    else:
        agents = info.get("Agents")
        if not isinstance(agents, list) or len(agents) < 2:
            raise RuntimeError("replay has no two-seat team identity")
        names = tuple(
            str(agent.get("Name", "")).strip()
            if isinstance(agent, Mapping)
            else ""
            for agent in agents[:2]
        )
    if len(names) != 2 or not all(names):
        raise RuntimeError("replay has an empty team identity")
    return names[0], names[1]


def replay_episode_id(replay: Mapping[str, Any]) -> str:
    info = replay.get("info")
    info = info if isinstance(info, Mapping) else {}
    value = info.get("EpisodeId")
    text = str(value) if value is not None else ""
    if not text.isdigit():
        raise RuntimeError(f"replay EpisodeId is invalid: {value!r}")
    return text


def replay_rewards(replay: Mapping[str, Any]) -> tuple[float, float]:
    values = replay.get("rewards")
    if not isinstance(values, list) or len(values) < 2:
        raise RuntimeError("replay has no two-seat terminal rewards")
    return (
        numeric_reward(values[0], "replay reward seat 0"),
        numeric_reward(values[1], "replay reward seat 1"),
    )


def hash_deck(raw_deck: Any) -> str | None:
    if not isinstance(raw_deck, list) or len(raw_deck) != 60:
        return None
    cards: list[int] = []
    for card in raw_deck:
        if isinstance(card, bool) or not isinstance(card, int):
            return None
        cards.append(card)
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def replay_deck_hashes(replay: Mapping[str, Any]) -> tuple[str, str]:
    steps = replay.get("steps")
    if not isinstance(steps, list):
        raise RuntimeError("replay steps are missing")
    observed: set[tuple[str, str]] = set()
    for frame in steps[:3]:
        if not isinstance(frame, list):
            continue
        for entry in frame[:2]:
            if not isinstance(entry, Mapping):
                continue
            visualize = entry.get("visualize")
            if not isinstance(visualize, list):
                continue
            for item in visualize:
                action = item.get("action") if isinstance(item, Mapping) else None
                if not isinstance(action, list) or len(action) < 2:
                    continue
                first = hash_deck(action[0])
                second = hash_deck(action[1])
                if first is not None and second is not None:
                    observed.add((first, second))
    if not observed:
        raise RuntimeError("replay exposes no valid two-seat 60-card deck pair")
    if len(observed) != 1:
        raise RuntimeError(f"replay deck-pair visualization drift: {sorted(observed)!r}")
    return next(iter(observed))


def read_daily_manifest(
    archive: zipfile.ZipFile,
    dataset_date: str,
) -> tuple[set[str], str]:
    payload = archive.read("manifest.csv")
    try:
        text = payload.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{dataset_date}: manifest.csv is not UTF-8") from exc
    rows = list(csv.DictReader(text))
    episode_ids: set[str] = set()
    for index, row in enumerate(rows, 2):
        episode_id = str(row.get("episode_id") or "")
        if not episode_id.isdigit():
            raise RuntimeError(
                f"{dataset_date}: manifest.csv:{index} invalid episode_id"
            )
        if episode_id in episode_ids:
            raise RuntimeError(
                f"{dataset_date}: duplicate manifest episode {episode_id}"
            )
        create_time = str(row.get("create_time") or "")
        if not create_time.startswith(dataset_date):
            raise RuntimeError(
                f"{dataset_date}: manifest date drift for episode {episode_id}"
            )
        episode_ids.add(episode_id)
    if not episode_ids:
        raise RuntimeError(f"{dataset_date}: daily manifest is empty")
    return episode_ids, sha256_bytes(payload)


def audit_daily_replay_member_set(
    *,
    dataset_date: str,
    numeric_member_ids: set[str],
    manifest_ids: set[str],
    frozen_missing_ids: Sequence[str],
    source_episode_ids: set[str],
) -> tuple[str, ...]:
    """Verify the exact reviewed manifest/member gap for one daily archive."""

    expected_missing = set(frozen_missing_ids)
    extra_members = numeric_member_ids - manifest_ids
    if extra_members:
        raise RuntimeError(
            f"{dataset_date}: replay members absent from manifest.csv: "
            f"{sorted(extra_members, key=int)!r}"
        )
    observed_missing = manifest_ids - numeric_member_ids
    if observed_missing != expected_missing:
        raise RuntimeError(
            f"{dataset_date}: exact missing replay set drift: "
            f"expected {sorted(expected_missing, key=int)!r}, "
            f"got {sorted(observed_missing, key=int)!r}"
        )
    selected_missing = source_episode_ids & observed_missing
    if selected_missing:
        raise RuntimeError(
            f"{dataset_date}: frozen missing replay intersects source-selected episode: "
            f"{sorted(selected_missing, key=int)!r}"
        )
    expected_members = manifest_ids - expected_missing
    if numeric_member_ids != expected_members:
        raise RuntimeError(f"{dataset_date}: reviewed replay member algebra drift")
    return tuple(sorted(observed_missing, key=int))


def scan_daily_replays(
    daily_dir: Path,
    source: SourceAudit,
    contract: BuildContract,
) -> tuple[tuple[SelectedEpisode, ...], tuple[DailyAudit, ...]]:
    expected_daily = dict(contract.daily_sha256)
    expected_daily_manifests = dict(contract.daily_manifest_sha256)
    expected_missing_replays = dict(contract.daily_missing_replay_ids)
    expected_incomplete = {
        (value.dataset_date, value.episode_id): value
        for value in contract.incomplete_terminal_replays
    }
    observed_incomplete: set[tuple[str, str]] = set()
    selected: list[SelectedEpisode] = []
    daily_audits: list[DailyAudit] = []
    date_counts: Counter[str] = Counter()
    for dataset_date in contract.dates:
        filename = f"pokemon-tcg-ai-battle-episodes-{dataset_date}.zip"
        path = daily_dir / filename
        require_regular_file(path, f"daily replay archive {dataset_date}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_daily[dataset_date]:
            raise RuntimeError(
                f"{dataset_date}: daily SHA256 mismatch: "
                f"expected {expected_daily[dataset_date]}, got {actual_sha}"
            )
        source_episodes = sorted(
            (
                audit
                for (date_value, _episode_id), audit in source.episodes.items()
                if date_value == dataset_date
            ),
            key=lambda audit: int(audit.episode_id),
        )
        date_incomplete_ids: list[str] = []
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError(f"{dataset_date}: duplicate daily ZIP members")
            if "manifest.csv" not in names:
                raise RuntimeError(f"{dataset_date}: daily manifest.csv is missing")
            numeric_members: dict[str, str] = {}
            for name in names:
                member_path = Path(name)
                if member_path.parent != Path(".") or member_path.suffix != ".json":
                    continue
                if not member_path.stem.isdigit():
                    continue
                if member_path.stem in numeric_members:
                    raise RuntimeError(
                        f"{dataset_date}: duplicate replay ID {member_path.stem}"
                    )
                numeric_members[member_path.stem] = name
            manifest_ids, manifest_sha = read_daily_manifest(
                archive,
                dataset_date,
            )
            if manifest_sha != expected_daily_manifests[dataset_date]:
                raise RuntimeError(
                    f"{dataset_date}: daily manifest SHA256 mismatch: "
                    f"expected {expected_daily_manifests[dataset_date]}, "
                    f"got {manifest_sha}"
                )
            missing_replay_ids = audit_daily_replay_member_set(
                dataset_date=dataset_date,
                numeric_member_ids=set(numeric_members),
                manifest_ids=manifest_ids,
                frozen_missing_ids=expected_missing_replays[dataset_date],
                source_episode_ids={value.episode_id for value in source_episodes},
            )
            for episode in source_episodes:
                member = numeric_members.get(episode.episode_id)
                if member is None:
                    raise RuntimeError(
                        f"{dataset_date}: source episode {episode.episode_id} missing"
                    )
                replay_payload = archive.read(member)
                try:
                    replay = orjson.loads(replay_payload)
                except orjson.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"{dataset_date}:{member}: replay is invalid JSON"
                    ) from exc
                if not isinstance(replay, dict):
                    raise RuntimeError(f"{dataset_date}:{member}: replay is not an object")
                if replay_episode_id(replay) != episode.episode_id:
                    raise RuntimeError(f"{dataset_date}:{member}: EpisodeId mismatch")
                replay_uuid = replay.get("id")
                replay_uuid = str(replay_uuid) if replay_uuid is not None else None
                if replay_uuid != episode.episode_uuid:
                    raise RuntimeError(f"{dataset_date}:{member}: episode UUID mismatch")
                names_pair = replay_names(replay)
                expected_names = [None, None]
                expected_names[episode.seat] = episode.team_name
                expected_names[1 - episode.seat] = episode.opponent_team_name
                if names_pair != tuple(expected_names):
                    raise RuntimeError(f"{dataset_date}:{member}: team identity mismatch")
                replay_key = (dataset_date, episode.episode_id)
                raw_rewards = replay.get("rewards")
                complete_rewards = True
                try:
                    rewards = replay_rewards(replay)
                except RuntimeError:
                    complete_rewards = False
                    rewards = (0.0, 0.0)
                expected_exception = expected_incomplete.get(replay_key)
                if complete_rewards and expected_exception is not None:
                    raise RuntimeError(
                        f"{dataset_date}:{member}: frozen incomplete-terminal set drift"
                    )
                if not complete_rewards:
                    if expected_exception is None:
                        raise RuntimeError(
                            f"{dataset_date}:{member}: unexpected incomplete terminal rewards"
                        )
                    common_sha = sha256_bytes(replay_payload)
                    if not isinstance(raw_rewards, list) or tuple(raw_rewards) != (
                        expected_exception.raw_rewards
                    ):
                        raise RuntimeError(
                            f"{dataset_date}:{member}: incomplete raw rewards drift"
                        )
                    if common_sha != expected_exception.replay_sha256:
                        raise RuntimeError(
                            f"{dataset_date}:{member}: incomplete replay SHA drift"
                        )
                    source_identity = (
                        episode.episode_uuid,
                        episode.seat,
                        episode.team_name,
                        episode.opponent_team_name,
                        episode.terminal_reward,
                    )
                    expected_identity = (
                        expected_exception.episode_uuid,
                        expected_exception.seat,
                        expected_exception.team_name,
                        expected_exception.opponent_team_name,
                        expected_exception.source_terminal_reward,
                    )
                    if source_identity != expected_identity:
                        raise RuntimeError(
                            f"{dataset_date}:{member}: incomplete source identity drift"
                        )
                    learner_reward = numeric_reward(
                        raw_rewards[episode.seat],
                        f"{dataset_date}:{member}: incomplete learner reward",
                    )
                    if learner_reward != episode.terminal_reward:
                        raise RuntimeError(
                            f"{dataset_date}:{member}: incomplete learner reward drift"
                        )
                    deck_hashes = replay_deck_hashes(replay)
                    if (
                        deck_hashes[episode.seat]
                        != expected_exception.learner_deck_hash
                        or deck_hashes[1 - episode.seat]
                        != expected_exception.opponent_deck_hash
                    ):
                        raise RuntimeError(
                            f"{dataset_date}:{member}: incomplete deck-pair drift"
                        )
                    if (
                        deck_hashes[episode.seat] != contract.learner_deck_hash
                        or deck_hashes[1 - episode.seat]
                        == contract.opponent_deck_hash
                    ):
                        raise RuntimeError(
                            f"{dataset_date}:{member}: incomplete replay may match target"
                        )
                    observed_incomplete.add(replay_key)
                    date_incomplete_ids.append(episode.episode_id)
                    continue
                if rewards[episode.seat] != episode.terminal_reward:
                    raise RuntimeError(f"{dataset_date}:{member}: learner reward mismatch")
                if not (
                    rewards[episode.seat] > 0
                    and rewards[episode.seat] > rewards[1 - episode.seat]
                ):
                    raise RuntimeError(f"{dataset_date}:{member}: learner did not win")
                deck_hashes = replay_deck_hashes(replay)
                if deck_hashes[episode.seat] != contract.learner_deck_hash:
                    raise RuntimeError(f"{dataset_date}:{member}: learner deck mismatch")
                if deck_hashes[1 - episode.seat] != contract.opponent_deck_hash:
                    continue
                selected.append(
                    SelectedEpisode(
                        dataset_date=dataset_date,
                        episode_id=episode.episode_id,
                        replay_member=member,
                        replay_sha256=sha256_bytes(replay_payload),
                        seat=episode.seat,
                        team_name=episode.team_name,
                        opponent_team_name=episode.opponent_team_name,
                        replay_rewards=rewards,
                        rows=episode.rows,
                        raw_rows_sha256=episode.raw_rows_sha256.hexdigest(),
                        decision_keys_digest=digest_lines(episode.decision_keys),
                    )
                )
                date_counts[dataset_date] += 1
        daily_audits.append(
            DailyAudit(
                dataset_date=dataset_date,
                filename=filename,
                sha256=actual_sha,
                manifest_sha256=manifest_sha,
                manifest_episode_ids=len(manifest_ids),
                replay_members=len(numeric_members),
                missing_replay_ids=missing_replay_ids,
                missing_source_episode_overlap=0,
                incomplete_terminal_replay_ids=tuple(
                    sorted(date_incomplete_ids, key=int)
                ),
                source_episodes_checked=len(source_episodes),
            )
        )

    if observed_incomplete != set(expected_incomplete):
        raise RuntimeError(
            "frozen incomplete-terminal replay set drift: "
            f"expected {sorted(expected_incomplete)!r}, "
            f"got {sorted(observed_incomplete)!r}"
        )
    if len(selected) != contract.expected_selected_episodes:
        raise RuntimeError(
            "selected episode-count drift: "
            f"expected {contract.expected_selected_episodes}, got {len(selected)}"
        )
    selected_rows = sum(episode.rows for episode in selected)
    if selected_rows != contract.expected_selected_rows:
        raise RuntimeError(
            "selected row-count drift: "
            f"expected {contract.expected_selected_rows}, got {selected_rows}"
        )
    expected_counts = dict(contract.expected_date_episodes)
    observed_counts = {date: date_counts[date] for date in contract.dates}
    if observed_counts != expected_counts:
        raise RuntimeError(
            "selected date-count drift: "
            f"expected {expected_counts!r}, got {observed_counts!r}"
        )
    selected.sort(key=lambda value: (value.dataset_date, int(value.episode_id)))
    return tuple(selected), tuple(daily_audits)


def materialize_selected_lines(
    source_path: Path,
    source_audit: SourceAudit,
    selected: Sequence[SelectedEpisode],
    contract: BuildContract,
) -> tuple[tuple[bytes, ...], str, str, tuple[tuple[str, int], ...]]:
    selected_episode_keys = {
        (episode.dataset_date, episode.episode_id) for episode in selected
    }
    expected_rows = {
        (episode.dataset_date, episode.episode_id): episode.rows
        for episode in selected
    }
    per_episode_rows: Counter[tuple[str, str]] = Counter()
    per_episode_digest = {
        key: hashlib.sha256() for key in selected_episode_keys
    }
    selected_lines: list[bytes] = []
    selected_keys: list[str] = []
    date_rows: Counter[str] = Counter()
    with zipfile.ZipFile(source_path) as archive:
        for member in contract.source_train_members:
            with archive.open(member) as handle:
                for line_number, raw_line in enumerate(handle, 1):
                    try:
                        row = orjson.loads(raw_line)
                    except orjson.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"materialization {member}:{line_number}: invalid JSON"
                        ) from exc
                    dataset_date = str(row.get("dataset_date") or "")
                    episode_id = str(row.get("episode_id") or "")
                    episode_key = (dataset_date, episode_id)
                    if episode_key not in selected_episode_keys:
                        continue
                    where = f"materialization {member}:{line_number}"
                    decision_key, *_rest = validate_source_row(
                        row,
                        where,
                        contract,
                    )
                    selected_lines.append(raw_line)
                    selected_keys.append(decision_key)
                    per_episode_rows[episode_key] += 1
                    per_episode_digest[episode_key].update(raw_line)
                    date_rows[dataset_date] += 1
    if dict(per_episode_rows) != expected_rows:
        raise RuntimeError("materialized per-episode row counts drifted")
    expected_digests = {
        (episode.dataset_date, episode.episode_id): episode.raw_rows_sha256
        for episode in selected
    }
    observed_digests = {
        key: digest.hexdigest() for key, digest in per_episode_digest.items()
    }
    if observed_digests != expected_digests:
        raise RuntimeError("materialized per-episode raw bytes drifted")
    if len(selected_lines) != contract.expected_selected_rows:
        raise RuntimeError("materialized global row count drifted")
    if len(set(selected_keys)) != len(selected_keys):
        raise RuntimeError("materialized selected decisions are not unique")
    payload_digest = hashlib.sha256()
    for raw_line in selected_lines:
        payload_digest.update(raw_line)
    return (
        tuple(selected_lines),
        payload_digest.hexdigest(),
        digest_lines(selected_keys),
        tuple((date, date_rows[date]) for date in contract.dates),
    )


def build_plan(contract: BuildContract) -> dict[str, Any]:
    daily_manifest_hashes = dict(contract.daily_manifest_sha256)
    daily_missing_replays = dict(contract.daily_missing_replay_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "data_schema_version": ROW_SCHEMA_VERSION,
        "mode": "exact_official_replay_cross_audit",
        "source": {
            "logical_path": contract.source_logical_path,
            "sha256": contract.source_sha256,
            "manifest_sha256": contract.source_manifest_sha256,
            "train_members": list(contract.source_train_members),
            "expected_train_rows": contract.expected_source_train_rows,
            "expected_train_episodes": contract.expected_source_train_episodes,
        },
        "daily": [
            {
                "dataset_date": date,
                "filename": f"pokemon-tcg-ai-battle-episodes-{date}.zip",
                "sha256": digest,
                "manifest_sha256": daily_manifest_hashes[date],
                "missing_replay_ids": list(daily_missing_replays[date]),
                "missing_source_episode_overlap": 0,
                "incomplete_terminal_replay_ids": [
                    value.episode_id
                    for value in contract.incomplete_terminal_replays
                    if value.dataset_date == date
                ],
            }
            for date, digest in contract.daily_sha256
        ],
        "incomplete_terminal_replays": [
            {
                "dataset_date": value.dataset_date,
                "episode_id": value.episode_id,
                "episode_uuid": value.episode_uuid,
                "replay_sha256": value.replay_sha256,
                "seat": value.seat,
                "team_name": value.team_name,
                "opponent_team_name": value.opponent_team_name,
                "source_terminal_reward": value.source_terminal_reward,
                "raw_rewards": list(value.raw_rewards),
                "learner_deck_hash": value.learner_deck_hash,
                "opponent_deck_hash": value.opponent_deck_hash,
                "target_route": False,
            }
            for value in contract.incomplete_terminal_replays
        ],
        "criteria": {
            "source_split": "train",
            "learner_terminal_outcome": "strict_win",
            "learner_deck_hash": contract.learner_deck_hash,
            "opponent_deck_hash": contract.opponent_deck_hash,
            "visualize_usage": "deck_hash_verification_only_not_copied",
            "row_copy": "verbatim_source_jsonl_bytes",
        },
        "expected": {
            "episodes": contract.expected_selected_episodes,
            "rows": contract.expected_selected_rows,
            "date_episodes": dict(contract.expected_date_episodes),
        },
        "output": {
            "members": [TRAIN_MEMBER, ALLOWLIST_MEMBER, MANIFEST_MEMBER],
            "zip_timestamp": list(ZIP_TIMESTAMP),
            "compression": "deflate",
            "compresslevel": ZIP_COMPRESSLEVEL,
        },
    }


def build_allowlist(
    audit: SelectionAudit,
    contract: BuildContract,
) -> dict[str, Any]:
    episode_ids = [episode.episode_id for episode in audit.selected]
    if not episode_ids or any(not value for value in episode_ids):
        raise RuntimeError("exact episode allowlist must be non-empty")
    if len(episode_ids) != len(set(episode_ids)):
        raise RuntimeError("exact episode allowlist contains duplicate episode_ids")
    return {
        "schema_version": ALLOWLIST_SCHEMA_VERSION,
        "data_schema_version": ROW_SCHEMA_VERSION,
        "learner_deck_hash": contract.learner_deck_hash,
        "opponent_deck_hash": contract.opponent_deck_hash,
        "split": "train",
        "terminal_reward": "win",
        "episode_count": len(audit.selected),
        "decision_rows": len(audit.selected_lines),
        "date_episode_counts": dict(contract.expected_date_episodes),
        "episode_ids": episode_ids,
        "episode_key_format": EPISODE_KEY_FORMAT,
        "criteria": {
            "dates": list(contract.dates),
            "source_split": "train",
            "learner_deck_hash": contract.learner_deck_hash,
            "opponent_deck_hash": contract.opponent_deck_hash,
            "learner_terminal_outcome": "strict_win",
            "row_copy": "verbatim_source_jsonl_bytes",
        },
        "source": {
            "logical_path": contract.source_logical_path,
            "sha256": contract.source_sha256,
            "manifest_sha256": contract.source_manifest_sha256,
        },
        "daily": [
            {
                "dataset_date": daily.dataset_date,
                "filename": daily.filename,
                "sha256": daily.sha256,
                "manifest_sha256": daily.manifest_sha256,
                "manifest_episode_ids": daily.manifest_episode_ids,
                "replay_members": daily.replay_members,
                "missing_replay_ids": list(daily.missing_replay_ids),
                "missing_source_episode_overlap": (
                    daily.missing_source_episode_overlap
                ),
                "incomplete_terminal_replay_ids": list(
                    daily.incomplete_terminal_replay_ids
                ),
                "source_episodes_checked": daily.source_episodes_checked,
            }
            for daily in audit.daily
        ],
        "incomplete_terminal_replays": [
            {
                "dataset_date": value.dataset_date,
                "episode_id": value.episode_id,
                "episode_uuid": value.episode_uuid,
                "replay_sha256": value.replay_sha256,
                "seat": value.seat,
                "team_name": value.team_name,
                "opponent_team_name": value.opponent_team_name,
                "source_terminal_reward": value.source_terminal_reward,
                "raw_rewards": list(value.raw_rewards),
                "learner_deck_hash": value.learner_deck_hash,
                "opponent_deck_hash": value.opponent_deck_hash,
                "target_route": False,
            }
            for value in contract.incomplete_terminal_replays
        ],
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "date": episode.dataset_date,
                "decision_rows": episode.rows,
                "replay_member": episode.replay_member,
                "replay_sha256": episode.replay_sha256,
                "seat": episode.seat,
                "team_name": episode.team_name,
                "opponent_team_name": episode.opponent_team_name,
                "learner_deck_hash": contract.learner_deck_hash,
                "opponent_deck_hash": contract.opponent_deck_hash,
                "replay_rewards": list(episode.replay_rewards),
                "raw_rows_sha256": episode.raw_rows_sha256,
                "decision_keys_digest": episode.decision_keys_digest,
            }
            for episode in audit.selected
        ],
        "stats": {
            "episodes": len(audit.selected),
            "rows": len(audit.selected_lines),
            "date_episodes": dict(contract.expected_date_episodes),
            "date_rows": dict(audit.selected_date_rows),
            "selected_content_sha256": audit.selected_content_sha256,
            "selected_decision_keys_digest": audit.selected_decision_keys_digest,
            "duplicate_episode_keys": 0,
            "duplicate_decision_keys": 0,
        },
    }


def validate_allowlist_document(
    payload: Any,
    contract: BuildContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Pure consumer contract for an exact-Fros train-win allowlist."""

    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
        try:
            document = orjson.loads(raw)
        except orjson.JSONDecodeError as exc:
            raise RuntimeError("exact-Fros allowlist is invalid JSON") from exc
        if canonical_json_bytes(document) + b"\n" != raw:
            raise RuntimeError("exact-Fros allowlist is not canonical JSON+LF")
    else:
        document = payload
    if not isinstance(document, dict):
        raise RuntimeError("exact-Fros allowlist is not an object")
    checks = {
        "schema_version": (
            document.get("schema_version"),
            ALLOWLIST_SCHEMA_VERSION,
        ),
        "data_schema_version": (
            document.get("data_schema_version"),
            ROW_SCHEMA_VERSION,
        ),
        "learner_deck_hash": (
            document.get("learner_deck_hash"),
            contract.learner_deck_hash,
        ),
        "opponent_deck_hash": (
            document.get("opponent_deck_hash"),
            contract.opponent_deck_hash,
        ),
        "split": (document.get("split"), "train"),
        "terminal_reward": (document.get("terminal_reward"), "win"),
        "episode_count": (
            document.get("episode_count"),
            contract.expected_selected_episodes,
        ),
        "decision_rows": (
            document.get("decision_rows"),
            contract.expected_selected_rows,
        ),
        "date_episode_counts": (
            document.get("date_episode_counts"),
            dict(contract.expected_date_episodes),
        ),
    }
    failures = [
        f"{name}={actual!r} expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    episodes = document.get("episodes")
    episode_ids = document.get("episode_ids")
    daily = document.get("daily")
    daily_outer = dict(contract.daily_sha256)
    daily_manifests = dict(contract.daily_manifest_sha256)
    daily_missing = dict(contract.daily_missing_replay_ids)
    if not isinstance(daily, list) or len(daily) != len(contract.dates):
        failures.append("daily audit length drift")
    else:
        for index, (dataset_date, value) in enumerate(zip(contract.dates, daily)):
            if not isinstance(value, Mapping):
                failures.append(f"daily[{index}] is not an object")
                continue
            expected_daily = {
                "dataset_date": dataset_date,
                "filename": (
                    f"pokemon-tcg-ai-battle-episodes-{dataset_date}.zip"
                ),
                "sha256": daily_outer[dataset_date],
                "manifest_sha256": daily_manifests[dataset_date],
                "missing_replay_ids": list(daily_missing[dataset_date]),
                "missing_source_episode_overlap": 0,
                "incomplete_terminal_replay_ids": [
                    value.episode_id
                    for value in contract.incomplete_terminal_replays
                    if value.dataset_date == dataset_date
                ],
            }
            for name, expected in expected_daily.items():
                if value.get(name) != expected:
                    failures.append(f"daily[{index}].{name} drift")
            manifest_count = value.get("manifest_episode_ids")
            member_count = value.get("replay_members")
            if (
                isinstance(manifest_count, bool)
                or not isinstance(manifest_count, int)
                or isinstance(member_count, bool)
                or not isinstance(member_count, int)
                or manifest_count - member_count
                != len(daily_missing[dataset_date])
            ):
                failures.append(f"daily[{index}] member-count algebra drift")
    expected_incomplete = [
        {
            "dataset_date": value.dataset_date,
            "episode_id": value.episode_id,
            "episode_uuid": value.episode_uuid,
            "replay_sha256": value.replay_sha256,
            "seat": value.seat,
            "team_name": value.team_name,
            "opponent_team_name": value.opponent_team_name,
            "source_terminal_reward": value.source_terminal_reward,
            "raw_rewards": list(value.raw_rewards),
            "learner_deck_hash": value.learner_deck_hash,
            "opponent_deck_hash": value.opponent_deck_hash,
            "target_route": False,
        }
        for value in contract.incomplete_terminal_replays
    ]
    if document.get("incomplete_terminal_replays") != expected_incomplete:
        failures.append("incomplete_terminal_replays drift")
    if failures:
        raise RuntimeError("exact-Fros allowlist contract drift: " + "; ".join(failures))
    if not isinstance(episodes, list) or len(episodes) != contract.expected_selected_episodes:
        raise RuntimeError("exact-Fros allowlist episodes length drift")
    if not isinstance(episode_ids, list) or not episode_ids:
        raise RuntimeError("exact-Fros allowlist episode_ids are empty")
    observed_ids = [str(value.get("episode_id") or "") for value in episodes]
    if (
        any(not value for value in observed_ids)
        or len(observed_ids) != len(set(observed_ids))
        or episode_ids != observed_ids
    ):
        raise RuntimeError("exact-Fros allowlist episode identity drift")
    if sum(int(value.get("decision_rows", -1)) for value in episodes) != (
        contract.expected_selected_rows
    ):
        raise RuntimeError("exact-Fros allowlist episode row sum drift")
    observed_dates = Counter(str(value.get("date") or "") for value in episodes)
    if dict(observed_dates) != dict(contract.expected_date_episodes):
        raise RuntimeError("exact-Fros allowlist episode date counts drift")
    return document


def build_manifest(
    audit: SelectionAudit,
    contract: BuildContract,
    plan_sha256: str,
    allowlist_sha256: str,
    allowlist_canonical_sha256: str,
) -> dict[str, Any]:
    team_rows: Counter[str] = Counter()
    opponent_rows: Counter[str] = Counter()
    seat_rows: Counter[str] = Counter()
    for episode in audit.selected:
        team_rows[episode.team_name] += episode.rows
        opponent_rows[episode.opponent_team_name] += episode.rows
        seat_rows[str(episode.seat)] += episode.rows
    return {
        "schema_version": SCHEMA_VERSION,
        "data_schema_version": ROW_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "learner_deck_hash": contract.learner_deck_hash,
        "opponent_deck_hash": contract.opponent_deck_hash,
        "split": "train",
        "terminal_reward": "win",
        "episode_count": len(audit.selected),
        "decision_rows": len(audit.selected_lines),
        "date_episode_counts": dict(contract.expected_date_episodes),
        "split_episodes": {"train": len(audit.selected)},
        "split_decisions": {"train": len(audit.selected_lines)},
        "source": {
            "logical_path": contract.source_logical_path,
            "sha256": contract.source_sha256,
            "manifest_sha256": contract.source_manifest_sha256,
            "schema_version": SOURCE_SCHEMA_VERSION,
            "train_rows": audit.source.train_rows,
            "train_episodes": len(audit.source.episodes),
            "train_member_sha256": dict(audit.source.train_member_sha256),
            "train_members": [
                {
                    "name": name,
                    "size_bytes": dict(audit.source.train_member_bytes)[name],
                    "sha256": digest,
                }
                for name, digest in audit.source.train_member_sha256
            ],
        },
        "selection": {
            "dates": list(contract.dates),
            "learner_deck_hash": contract.learner_deck_hash,
            "opponent_deck_hash": contract.opponent_deck_hash,
            "learner_terminal_outcome": "strict_win",
            "episodes": len(audit.selected),
            "rows": len(audit.selected_lines),
            "date_episodes": dict(contract.expected_date_episodes),
            "date_rows": dict(audit.selected_date_rows),
            "team_rows": dict(sorted(team_rows.items())),
            "opponent_rows": dict(sorted(opponent_rows.items())),
            "seat_rows": dict(sorted(seat_rows.items())),
        },
        "train": {
            "members": [
                {
                    "member": TRAIN_MEMBER,
                    "rows": len(audit.selected_lines),
                    "sha256": audit.selected_content_sha256,
                    "source_order_preserved": True,
                }
            ],
            "total_rows": len(audit.selected_lines),
            "episodes": len(audit.selected),
            "decision_keys_digest": audit.selected_decision_keys_digest,
            "decision_key_format": DECISION_KEY_FORMAT,
        },
        "exact_episode_allowlist": {
            "member": ALLOWLIST_MEMBER,
            "sha256": allowlist_sha256,
            "canonical_sha256": allowlist_canonical_sha256,
            "schema_version": ALLOWLIST_SCHEMA_VERSION,
            "canonical_json": True,
            "episodes": len(audit.selected),
            "decision_rows": len(audit.selected_lines),
        },
        "byte_preservation": {
            "selected_rows_copied_verbatim": True,
            "sample_weight_modified": False,
            "raw_selected_content_sha256": audit.selected_content_sha256,
            "hidden_visualize_rows_copied": 0,
        },
        "replay_audit": {
            "official_daily_inputs_only": True,
            "both_deck_hashes_verified": True,
            "learner_win_verified": True,
            "visualize_used_only_for_hash_verification": True,
            "incomplete_terminal_replays": [
                {
                    "dataset_date": value.dataset_date,
                    "episode_id": value.episode_id,
                    "episode_uuid": value.episode_uuid,
                    "replay_sha256": value.replay_sha256,
                    "seat": value.seat,
                    "team_name": value.team_name,
                    "opponent_team_name": value.opponent_team_name,
                    "source_terminal_reward": value.source_terminal_reward,
                    "raw_rewards": list(value.raw_rewards),
                    "learner_deck_hash": value.learner_deck_hash,
                    "opponent_deck_hash": value.opponent_deck_hash,
                    "target_route": False,
                }
                for value in contract.incomplete_terminal_replays
            ],
            "daily": [
                {
                    "dataset_date": value.dataset_date,
                    "filename": value.filename,
                    "sha256": value.sha256,
                    "manifest_sha256": value.manifest_sha256,
                    "manifest_episode_ids": value.manifest_episode_ids,
                    "replay_members": value.replay_members,
                    "missing_replay_ids": list(value.missing_replay_ids),
                    "missing_source_episode_overlap": (
                        value.missing_source_episode_overlap
                    ),
                    "incomplete_terminal_replay_ids": list(
                        value.incomplete_terminal_replay_ids
                    ),
                    "source_episodes_checked": value.source_episodes_checked,
                }
                for value in audit.daily
            ],
        },
        "zip": {
            "timestamp": list(ZIP_TIMESTAMP),
            "compression": "deflate",
            "compresslevel": ZIP_COMPRESSLEVEL,
            "member_mode": "0644",
            "member_order": [TRAIN_MEMBER, ALLOWLIST_MEMBER, MANIFEST_MEMBER],
            "double_serialization_required": True,
        },
    }


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = ZIP_COMPRESSION
    info.create_system = 3
    info.external_attr = 0o644 << 16
    return info


def create_archive_bytes(
    audit: SelectionAudit,
    manifest: dict[str, Any],
    allowlist_payload: bytes,
) -> bytes:
    train_payload = b"".join(audit.selected_lines)
    if sha256_bytes(train_payload) != audit.selected_content_sha256:
        raise RuntimeError("selected train bytes changed before serialization")
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=ZIP_COMPRESSION,
        compresslevel=ZIP_COMPRESSLEVEL,
        allowZip64=True,
    ) as archive:
        archive.writestr(zip_info(TRAIN_MEMBER), train_payload)
        archive.writestr(zip_info(ALLOWLIST_MEMBER), allowlist_payload)
        archive.writestr(zip_info(MANIFEST_MEMBER), manifest_payload)
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        expected_names = [TRAIN_MEMBER, ALLOWLIST_MEMBER, MANIFEST_MEMBER]
        if archive.namelist() != expected_names:
            raise RuntimeError("output ZIP member order drifted")
        if archive.testzip() is not None:
            raise RuntimeError("output ZIP CRC audit failed")
        if archive.read(TRAIN_MEMBER) != train_payload:
            raise RuntimeError("output train member did not preserve source bytes")
        if archive.read(ALLOWLIST_MEMBER) != allowlist_payload:
            raise RuntimeError("output allowlist bytes changed")
        if archive.read(MANIFEST_MEMBER) != manifest_payload:
            raise RuntimeError("output manifest bytes changed")
    return payload


def publish_no_clobber(output: Path, payload: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".partial",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing output: {output}"
            ) from exc
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def rename_directory_noreplace(staging: Path, target: Path) -> None:
    """Atomically publish a directory using Linux ``RENAME_NOREPLACE``."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is unavailable; refusing non-atomic publish")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"bundle target appeared concurrently: {target}")
        raise OSError(error_number, os.strerror(error_number), str(target))


def publish_bundle_no_clobber(
    archive_path: Path,
    companion_path: Path,
    archive_payload: bytes,
    companion_payload: bytes,
) -> None:
    """Atomically publish two files as one previously absent directory."""

    if archive_path.parent != companion_path.parent:
        raise RuntimeError("bundle outputs must share one target directory")
    bundle_root = archive_path.parent
    parent = bundle_root.parent
    if os.path.lexists(bundle_root):
        raise FileExistsError(f"refusing to reuse bundle directory: {bundle_root}")
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError(f"bundle parent must be an existing real directory: {parent}")
    staging = Path(
        tempfile.mkdtemp(
            dir=parent,
            prefix=f".{bundle_root.name}.",
            suffix=".partial",
        )
    )
    published = False
    try:
        for filename, payload in (
            (companion_path.name, companion_payload),
            (archive_path.name, archive_payload),
        ):
            target = staging / filename
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        rename_directory_noreplace(staging, bundle_root)
        published = True
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published and staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()


def audit_and_serialize(
    source: Path,
    daily_dir: Path,
    contract: BuildContract,
) -> tuple[SelectionAudit, bytes, dict[str, Any]]:
    validate_contract(contract)
    plan = build_plan(contract)
    plan_sha256 = sha256_bytes(canonical_json_bytes(plan))
    source_audit = scan_source(source, contract)
    selected, daily_audits = scan_daily_replays(
        daily_dir,
        source_audit,
        contract,
    )
    (
        selected_lines,
        selected_content_sha256,
        selected_decision_keys_digest,
        selected_date_rows,
    ) = materialize_selected_lines(
        source,
        source_audit,
        selected,
        contract,
    )
    audit = SelectionAudit(
        source=source_audit,
        selected=selected,
        daily=daily_audits,
        selected_lines=selected_lines,
        selected_content_sha256=selected_content_sha256,
        selected_decision_keys_digest=selected_decision_keys_digest,
        selected_date_rows=selected_date_rows,
    )
    allowlist = build_allowlist(audit, contract)
    allowlist_payload = canonical_json_bytes(allowlist) + b"\n"
    validate_allowlist_document(allowlist_payload, contract)
    allowlist_sha256 = sha256_bytes(allowlist_payload)
    reparsed_allowlist = orjson.loads(allowlist_payload)
    canonical_allowlist_payload = canonical_json_bytes(reparsed_allowlist) + b"\n"
    if canonical_allowlist_payload != allowlist_payload:
        raise RuntimeError("allowlist member is not canonical sorted/minified JSON+LF")
    allowlist_canonical_sha256 = sha256_bytes(canonical_allowlist_payload)
    manifest = build_manifest(
        audit,
        contract,
        plan_sha256,
        allowlist_sha256,
        allowlist_canonical_sha256,
    )
    first = create_archive_bytes(audit, manifest, allowlist_payload)
    second = create_archive_bytes(audit, manifest, allowlist_payload)
    if first != second:
        raise RuntimeError("two deterministic archive serializations differ")
    metadata = {
        "plan": plan,
        "plan_sha256": plan_sha256,
        "allowlist_sha256": allowlist_sha256,
        "allowlist_canonical_sha256": allowlist_canonical_sha256,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest) + b"\n"),
        "archive_sha256": sha256_bytes(first),
        "deterministic_rebuild_match": True,
    }
    return audit, first, metadata


def build(
    *,
    source: Path,
    daily_dir: Path,
    output: Path,
    allowlist_output: Path | None = None,
    contract: BuildContract = DEFAULT_CONTRACT,
    execute: bool,
    expected_plan_sha256: str | None = None,
    expected_output_sha256: str | None = None,
    expected_allowlist_sha256: str | None = None,
) -> dict[str, Any]:
    source = audit_path_components(
        source,
        "source archive",
        must_exist=True,
        kind="file",
    )
    daily_dir = audit_path_components(
        daily_dir,
        "daily replay directory",
        must_exist=True,
        kind="directory",
    )
    output = audit_path_components(
        output,
        "output archive",
        must_exist=False,
        kind="output",
    )
    if allowlist_output is None:
        allowlist_output = output.parent / ALLOWLIST_MEMBER
    allowlist_output = audit_path_components(
        allowlist_output,
        "external allowlist output",
        must_exist=False,
        kind="output",
    )
    if output.parent != allowlist_output.parent:
        raise RuntimeError("archive and external allowlist must share a directory")
    if os.path.lexists(output.parent):
        raise FileExistsError(
            f"exact-Fros bundle directory must be absent: {output.parent}"
        )
    if source == output:
        raise ValueError("source and output paths must differ")
    if execute:
        if (
            expected_plan_sha256 is None
            or expected_output_sha256 is None
            or expected_allowlist_sha256 is None
        ):
            raise RuntimeError(
                "--execute requires --expected-plan-sha256, "
                "--expected-output-sha256, and --expected-allowlist-sha256 "
                "from a reviewed dry run"
            )
        require_sha256(expected_plan_sha256, "expected_plan_sha256")
        require_sha256(expected_output_sha256, "expected_output_sha256")
        require_sha256(
            expected_allowlist_sha256,
            "expected_allowlist_sha256",
        )

    input_hashes_before = {
        "source": sha256_file(source),
        **{
            date: sha256_file(
                daily_dir / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
            )
            for date in contract.dates
        },
    }
    audit, archive_payload, metadata = audit_and_serialize(
        source,
        daily_dir,
        contract,
    )
    input_hashes_after = {
        "source": sha256_file(source),
        **{
            date: sha256_file(
                daily_dir / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
            )
            for date in contract.dates
        },
    }
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("input files changed during the build audit")
    if execute and metadata["plan_sha256"] != expected_plan_sha256:
        raise RuntimeError(
            "plan SHA256 mismatch: dry-run and review the frozen plan again"
        )
    if execute and metadata["archive_sha256"] != expected_output_sha256:
        raise RuntimeError(
            "output SHA256 mismatch: dry-run and review the exact output again"
        )
    if execute and metadata["allowlist_sha256"] != expected_allowlist_sha256:
        raise RuntimeError(
            "allowlist SHA256 mismatch: dry-run and review the exact output again"
        )
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as rendered:
        allowlist_payload = rendered.read(ALLOWLIST_MEMBER)
    if execute:
        publish_bundle_no_clobber(
            output,
            allowlist_output,
            archive_payload,
            allowlist_payload,
        )
        if sha256_file(output) != metadata["archive_sha256"]:
            raise RuntimeError("published archive SHA256 differs from audited bytes")
        if sha256_file(allowlist_output) != metadata["allowlist_sha256"]:
            raise RuntimeError("published external allowlist SHA256 differs")
        with zipfile.ZipFile(output) as published:
            if published.read(ALLOWLIST_MEMBER) != allowlist_output.read_bytes():
                raise RuntimeError("published embedded/external allowlists differ")
            if sha256_bytes(published.read(MANIFEST_MEMBER)) != metadata[
                "manifest_sha256"
            ]:
                raise RuntimeError("published embedded manifest SHA256 differs")
    return {
        "status": "built" if execute else "dry_run_passed",
        "dry_run": not execute,
        "output_written": execute,
        "output": str(output),
        "external_allowlist": str(allowlist_output),
        "plan_sha256": metadata["plan_sha256"],
        "archive_sha256": metadata["archive_sha256"],
        "outer_zip_sha256": metadata["archive_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "embedded_manifest_raw_sha256": metadata["manifest_sha256"],
        "allowlist_sha256": metadata["allowlist_sha256"],
        "allowlist_canonical_sha256": metadata[
            "allowlist_canonical_sha256"
        ],
        "selected_content_sha256": audit.selected_content_sha256,
        "selected_decision_keys_digest": audit.selected_decision_keys_digest,
        "source_train_episodes": len(audit.source.episodes),
        "source_train_rows": audit.source.train_rows,
        "selected_episodes": len(audit.selected),
        "selected_rows": len(audit.selected_lines),
        "date_episodes": dict(contract.expected_date_episodes),
        "date_rows": dict(audit.selected_date_rows),
        "both_deck_hashes_verified": True,
        "learner_wins_verified": True,
        "rows_copied_verbatim": True,
        "deterministic_rebuild_match": metadata[
            "deterministic_rebuild_match"
        ],
        "submission_started": False,
        "training_started": False,
    }


def parse_sha256(value: str, label: str) -> str:
    lowered = value.lower()
    if not SHA256_RE.fullmatch(lowered):
        raise argparse.ArgumentTypeError(f"{label} must be a SHA256")
    return lowered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="full audit and double serialization without writing (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="publish only with reviewed plan and output SHA gates",
    )
    parser.add_argument(
        "--expected-plan-sha256",
        type=lambda value: parse_sha256(value, "--expected-plan-sha256"),
    )
    parser.add_argument(
        "--expected-output-sha256",
        type=lambda value: parse_sha256(value, "--expected-output-sha256"),
    )
    parser.add_argument(
        "--expected-allowlist-sha256",
        default=None,
        type=lambda value: parse_sha256(value, "--expected-allowlist-sha256"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(
        source=args.source,
        daily_dir=args.daily_dir,
        output=args.output,
        allowlist_output=args.allowlist,
        execute=args.execute,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_output_sha256=args.expected_output_sha256,
        expected_allowlist_sha256=args.expected_allowlist_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
