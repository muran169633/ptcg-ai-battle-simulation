#!/usr/bin/env python3
"""Build the frozen 2026-08-07 Marnie-vs-exact-Fros behavior view.

Rows are copied byte for byte from the ``valid`` member of the frozen Marnie
archive.  The official Aug-7 replay is used only to verify episode identity,
terminal outcome, and both semantic deck hashes.  Aug-8 is not an input.

The default mode is write-free.  Execution requires reviewed plan, archive,
and allowlist SHA-256 gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import orjson


def _load_local_module(module_name: str, filename: str) -> Any:
    """Load one audited sibling module without cwd/PYTHONPATH dependence."""

    script_dir = Path(os.path.abspath(os.fspath(Path(__file__).parent)))
    module_path = Path(os.path.abspath(os.fspath(script_dir / filename)))
    if module_path.parent != script_dir:
        raise RuntimeError(f"local module escapes tools directory: {module_path}")
    current = Path(module_path.anchor)
    for part in module_path.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError as exc:
            raise RuntimeError(f"local module path is missing: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"local module path traverses symlink: {current}")
        if current == module_path:
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"local module is not a regular file: {current}")
        elif not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"local module ancestor is not a directory: {current}")
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(os.path.abspath(existing_file)) != module_path:
            raise RuntimeError(f"unexpected preloaded module identity: {module_name}")
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create local module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


common = _load_local_module(
    "build_marnie_froslass_exact_wins",
    "build_marnie_froslass_exact_wins.py",
)
bundle_utils = _load_local_module(
    "build_marnie_exact_anti_kd",
    "build_marnie_exact_anti_kd.py",
)


REPO_ROOT = Path(os.path.abspath(os.fspath(Path(__file__).parent))).parent
ARCHIVE_SCHEMA_VERSION = "ptcg-marnie-exact-fros-valid-archive-v1"
ALLOWLIST_SCHEMA_VERSION = "ptcg-marnie-exact-fros-valid-allowlist-v1"
VALID_MEMBER = "valid/part-00000.jsonl"
ALLOWLIST_MEMBER = "exact_episode_allowlist.json"
MANIFEST_MEMBER = "manifest.json"
DATASET_DATE = "2026-08-07"

DEFAULT_SOURCE = REPO_ROOT / "data/gold_push_recent7_20260810_v1/archives/marnie.zip"
DEFAULT_DAILY = (
    REPO_ROOT
    / "data/episodes_cache/gold_push_20260810_v1/daily/"
    "pokemon-tcg-ai-battle-episodes-2026-08-07.zip"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data/gold_push_marnie_fros_valid_20260810_v1"
)
DEFAULT_ARCHIVE = DEFAULT_OUTPUT_ROOT / "marnie_exact_fros_valid_view.zip"
DEFAULT_ALLOWLIST = DEFAULT_OUTPUT_ROOT / ALLOWLIST_MEMBER

SOURCE_SHA256 = "6b3873b28bfad70377d0a3b22fe1163516a3d40b5af429885c0ce15f75b402a0"
SOURCE_MANIFEST_SHA256 = (
    "0839b0fd98ceaaa3b66c92c46899e62e8e3318998d6bba0c841536205879ecdb"
)
DAILY_SHA256 = "c9325476fde8bf6e3a9021520867e9dbfeaf3ec09124b01f742cb07fa5877e63"
DAILY_MANIFEST_SHA256 = (
    "dd75917c32c779df85a35c69601cbefe210f4f3f92e667aa22f2248b7c40715a"
)
DAILY_MISSING_REPLAY_IDS = (
    "90658419",
    "90684109",
    "90759581",
    "90765292",
    "90836433",
    "90844697",
)
SOURCE_MEMBERS = (
    "train/part-00000.jsonl",
    "train/part-00001.jsonl",
    "train/part-00002.jsonl",
    "train/part-00003.jsonl",
    "train/part-00004.jsonl",
    "train/part-00005.jsonl",
    "train/part-00006.jsonl",
    VALID_MEMBER,
    "test/part-00000.jsonl",
    MANIFEST_MEMBER,
)


@dataclass(frozen=True)
class ValidContract:
    source_sha256: str
    source_manifest_sha256: str
    daily_sha256: str
    daily_manifest_sha256: str
    daily_missing_replay_ids: tuple[str, ...]
    dataset_date: str
    learner_deck_hash: str
    opponent_deck_hash: str
    source_members: tuple[str, ...]
    valid_member: str
    source_valid_episodes: int
    source_valid_views: int
    source_mirror_episodes: int
    source_mirror_episode_ids: tuple[str, ...]
    source_valid_rows: int
    expected_episodes: int
    expected_rows: int
    expected_wins: int
    expected_losses: int
    top20_names: tuple[str, ...]
    source_logical_path: str
    daily_logical_path: str


DEFAULT_CONTRACT = ValidContract(
    source_sha256=SOURCE_SHA256,
    source_manifest_sha256=SOURCE_MANIFEST_SHA256,
    daily_sha256=DAILY_SHA256,
    daily_manifest_sha256=DAILY_MANIFEST_SHA256,
    daily_missing_replay_ids=DAILY_MISSING_REPLAY_IDS,
    dataset_date=DATASET_DATE,
    learner_deck_hash=common.MARNIE_DECK_HASH,
    opponent_deck_hash=common.FROSLASS_LOPUNNY_DECK_HASH,
    source_members=SOURCE_MEMBERS,
    valid_member=VALID_MEMBER,
    source_valid_episodes=295,
    source_valid_views=299,
    source_mirror_episodes=4,
    source_mirror_episode_ids=(
        "90613480",
        "90626128",
        "90655502",
        "90789414",
    ),
    source_valid_rows=28_800,
    expected_episodes=30,
    expected_rows=3_274,
    expected_wins=14,
    expected_losses=16,
    top20_names=bundle_utils.TOP20_NAMES,
    source_logical_path="data/gold_push_recent7_20260810_v1/archives/marnie.zip",
    daily_logical_path=(
        "data/episodes_cache/gold_push_20260810_v1/daily/"
        "pokemon-tcg-ai-battle-episodes-2026-08-07.zip"
    ),
)


@dataclass
class SourceView:
    episode_id: str
    episode_uuid: str | None
    seat: int
    team_name: str
    opponent_team_name: str
    terminal_reward: float
    rows: int
    raw_digest: Any
    decision_keys: list[str]


@dataclass(frozen=True)
class ValidEpisode:
    episode_id: str
    episode_uuid: str | None
    seat: int
    team_name: str
    opponent_team_name: str
    terminal_reward: float
    outcome: str
    rows: int
    raw_rows_sha256: str
    decision_keys_digest: str
    replay_member: str
    replay_sha256: str


@dataclass(frozen=True)
class ValidAudit:
    source_manifest: dict[str, Any]
    source_member_sha256: str
    source_episode_count: int
    source_view_count: int
    source_mirror_episode_count: int
    selected: tuple[ValidEpisode, ...]
    selected_lines: tuple[bytes, ...]
    selected_content_sha256: str
    decision_keys_digest: str
    winner_episode_counts: dict[str, int]
    loser_episode_counts: dict[str, int]
    daily_manifest_sha256: str
    daily_replay_members: int


def validate_contract(contract: ValidContract) -> None:
    for value, label in (
        (contract.source_sha256, "source_sha256"),
        (contract.source_manifest_sha256, "source_manifest_sha256"),
        (contract.daily_sha256, "daily_sha256"),
        (contract.daily_manifest_sha256, "daily_manifest_sha256"),
        (contract.learner_deck_hash, "learner_deck_hash"),
        (contract.opponent_deck_hash, "opponent_deck_hash"),
    ):
        common.require_sha256(value, label)
    if contract.learner_deck_hash == contract.opponent_deck_hash:
        raise RuntimeError("valid behavior view requires distinct deck hashes")
    if contract.expected_wins + contract.expected_losses != contract.expected_episodes:
        raise RuntimeError("win/loss gates do not sum to episode gate")
    if contract.source_valid_views != (
        contract.source_valid_episodes + contract.source_mirror_episodes
    ):
        raise RuntimeError("source episode/view/mirror count algebra drift")
    if (
        len(contract.source_mirror_episode_ids)
        != contract.source_mirror_episodes
        or len(contract.source_mirror_episode_ids)
        != len(set(contract.source_mirror_episode_ids))
        or any(not value.isdigit() for value in contract.source_mirror_episode_ids)
        or tuple(sorted(contract.source_mirror_episode_ids, key=int))
        != contract.source_mirror_episode_ids
    ):
        raise RuntimeError("source mirror episode-ID contract drift")
    if len(contract.top20_names) != len(set(contract.top20_names)):
        raise RuntimeError("Top20 contract contains duplicates")
    if tuple(name for name in contract.source_members if name == contract.valid_member) != (
        contract.valid_member,
    ):
        raise RuntimeError("source member contract must contain one valid member")
    if (
        any(not value.isdigit() for value in contract.daily_missing_replay_ids)
        or len(contract.daily_missing_replay_ids)
        != len(set(contract.daily_missing_replay_ids))
        or tuple(sorted(contract.daily_missing_replay_ids, key=int))
        != contract.daily_missing_replay_ids
    ):
        raise RuntimeError("Aug7 missing-replay contract is invalid")


def validate_source_manifest(payload: bytes, contract: ValidContract) -> dict[str, Any]:
    if common.sha256_bytes(payload) != contract.source_manifest_sha256:
        raise RuntimeError("Fros-valid source manifest SHA256 mismatch")
    manifest = orjson.loads(payload)
    if not isinstance(manifest, dict):
        raise RuntimeError("Fros-valid source manifest is not an object")
    profile = manifest.get("profile")
    split_policy = manifest.get("split_policy")
    split_rows = manifest.get("split_decisions")
    split_episodes = manifest.get("split_episodes")
    team_filter = manifest.get("team_filter")
    checks = {
        "schema_version": (manifest.get("schema_version"), common.ROW_SCHEMA_VERSION),
        "profile.deck_hash": (
            profile.get("deck_hash") if isinstance(profile, Mapping) else None,
            contract.learner_deck_hash,
        ),
        "valid_dates": (
            tuple(split_policy.get("valid_dates", ()))
            if isinstance(split_policy, Mapping)
            else None,
            (contract.dataset_date,),
        ),
        "valid_rows": (
            split_rows.get("valid") if isinstance(split_rows, Mapping) else None,
            contract.source_valid_rows,
        ),
        "valid_episodes": (
            split_episodes.get("valid")
            if isinstance(split_episodes, Mapping)
            else None,
            contract.source_valid_episodes,
        ),
        "top20": (
            tuple(team_filter.get("display_names", ()))
            if isinstance(team_filter, Mapping)
            else None,
            contract.top20_names,
        ),
    }
    failures = [
        f"{name}={actual!r} expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if failures:
        raise RuntimeError("Fros-valid source manifest drift: " + "; ".join(failures))
    return manifest


def scan_source(
    source: Path,
    contract: ValidContract,
) -> tuple[dict[str, Any], dict[tuple[str, str, int, str], SourceView], str]:
    common.require_regular_file(source, "Fros-valid source")
    if common.sha256_file(source) != contract.source_sha256:
        raise RuntimeError("Fros-valid source archive SHA256 mismatch")
    views: dict[tuple[str, str, int, str], SourceView] = {}
    seen_keys: set[str] = set()
    member_digest = hashlib.sha256()
    rows = 0
    with zipfile.ZipFile(source) as archive:
        names = tuple(archive.namelist())
        if len(names) != len(set(names)) or names != contract.source_members:
            raise RuntimeError("Fros-valid source ZIP member/order drift")
        manifest = validate_source_manifest(archive.read(MANIFEST_MEMBER), contract)
        with archive.open(contract.valid_member) as handle:
            for line_number, raw_line in enumerate(handle, 1):
                where = f"{contract.valid_member}:{line_number}"
                if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
                    raise RuntimeError(f"{where}: non-canonical JSONL terminator")
                member_digest.update(raw_line)
                row = orjson.loads(raw_line)
                if not isinstance(row, dict):
                    raise RuntimeError(f"{where}: row is not an object")
                checks = {
                    "schema_version": (row.get("schema_version"), common.ROW_SCHEMA_VERSION),
                    "split": (row.get("split"), "valid"),
                    "dataset_date": (row.get("dataset_date"), contract.dataset_date),
                    "deck_hash": (row.get("deck_hash"), contract.learner_deck_hash),
                }
                failures = [
                    name
                    for name, (actual, expected) in checks.items()
                    if actual != expected
                ]
                if failures:
                    raise RuntimeError(f"{where}: row identity drift: {failures!r}")
                if "visualize" in row or (
                    isinstance(row.get("observation"), dict)
                    and "visualize" in row["observation"]
                ):
                    raise RuntimeError(f"{where}: hidden visualize payload")
                if not isinstance(row.get("observation"), dict) or not isinstance(
                    row.get("action"), list
                ):
                    raise RuntimeError(f"{where}: malformed observation/action")
                decision_key, episode_id, seat, _step = common.row_identity(
                    row,
                    where,
                    dataclasses_contract(contract),
                )
                if decision_key in seen_keys:
                    raise RuntimeError(f"{where}: duplicate decision key")
                seen_keys.add(decision_key)
                team_name = row.get("team_name")
                opponent_name = row.get("opponent_team_name")
                if team_name not in contract.top20_names or not isinstance(
                    opponent_name, str
                ) or not opponent_name:
                    raise RuntimeError(f"{where}: invalid team routing")
                reward = common.numeric_reward(row.get("terminal_reward"), f"{where}: reward")
                raw_uuid = row.get("episode_uuid")
                episode_uuid = str(raw_uuid) if raw_uuid is not None else None
                view_key = (
                    contract.dataset_date,
                    episode_id,
                    seat,
                    str(team_name),
                )
                episode = views.get(view_key)
                if episode is None:
                    episode = SourceView(
                        episode_id=episode_id,
                        episode_uuid=episode_uuid,
                        seat=seat,
                        team_name=str(team_name),
                        opponent_team_name=opponent_name,
                        terminal_reward=reward,
                        rows=0,
                        raw_digest=hashlib.sha256(),
                        decision_keys=[],
                    )
                    views[view_key] = episode
                identity = (
                    episode_uuid,
                    seat,
                    team_name,
                    opponent_name,
                    reward,
                )
                expected_identity = (
                    episode.episode_uuid,
                    episode.seat,
                    episode.team_name,
                    episode.opponent_team_name,
                    episode.terminal_reward,
                )
                if identity != expected_identity:
                    raise RuntimeError(f"{where}: within-episode identity drift")
                episode.rows += 1
                episode.raw_digest.update(raw_line)
                episode.decision_keys.append(decision_key)
                rows += 1
    episode_view_counts = Counter(value.episode_id for value in views.values())
    mirror_episode_ids = tuple(
        sorted(
            (
                episode_id
                for episode_id, view_count in episode_view_counts.items()
                if view_count > 1
            ),
            key=int,
        )
    )
    mirror_episodes = len(mirror_episode_ids)
    views_by_episode: dict[str, list[SourceView]] = {}
    for view in views.values():
        views_by_episode.setdefault(view.episode_id, []).append(view)
    for episode_id in mirror_episode_ids:
        mirror_views = views_by_episode[episode_id]
        if (
            len(mirror_views) != 2
            or {value.seat for value in mirror_views} != {0, 1}
            or len({value.team_name for value in mirror_views}) != 2
        ):
            raise RuntimeError(
                f"Fros-valid mirror episode {episode_id} lacks exact dual-seat/team views"
            )
    if (
        rows != contract.source_valid_rows
        or len(episode_view_counts) != contract.source_valid_episodes
        or len(views) != contract.source_valid_views
        or mirror_episodes != contract.source_mirror_episodes
        or mirror_episode_ids != contract.source_mirror_episode_ids
    ):
        raise RuntimeError(
            "Fros-valid source count drift: "
            f"episodes={len(episode_view_counts)}, views={len(views)}, "
            f"mirror_episodes={mirror_episodes}, rows={rows}"
        )
    return manifest, views, member_digest.hexdigest()


def dataclasses_contract(contract: ValidContract) -> common.BuildContract:
    """Minimal adapter for the shared row-identity validator."""

    return common.BuildContract(
        source_sha256=contract.source_sha256,
        source_manifest_sha256=contract.source_manifest_sha256,
        daily_sha256=((contract.dataset_date, contract.daily_sha256),),
        daily_manifest_sha256=(
            (contract.dataset_date, contract.daily_manifest_sha256),
        ),
        daily_missing_replay_ids=(
            (contract.dataset_date, contract.daily_missing_replay_ids),
        ),
        dates=(contract.dataset_date,),
        expected_date_episodes=((contract.dataset_date, contract.expected_episodes),),
        expected_source_train_rows=contract.source_valid_rows,
        expected_source_train_episodes=contract.source_valid_episodes,
        expected_selected_rows=contract.expected_rows,
        expected_selected_episodes=contract.expected_episodes,
        learner_deck_hash=contract.learner_deck_hash,
        opponent_deck_hash=contract.opponent_deck_hash,
        source_members=contract.source_members,
        source_train_members=(contract.valid_member,),
        source_logical_path=contract.source_logical_path,
        daily_logical_dir=str(Path(contract.daily_logical_path).parent),
        incomplete_terminal_replays=(),
    )


def select_replays(
    daily: Path,
    source_views: Mapping[tuple[str, str, int, str], SourceView],
    contract: ValidContract,
) -> tuple[tuple[ValidEpisode, ...], str, int]:
    common.require_regular_file(daily, "Aug7 official daily archive")
    if common.sha256_file(daily) != contract.daily_sha256:
        raise RuntimeError("Aug7 daily archive SHA256 mismatch")
    selected: list[ValidEpisode] = []
    with zipfile.ZipFile(daily) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Aug7 daily archive has duplicate members")
        numeric: dict[str, str] = {}
        for name in names:
            member_path = Path(name)
            if (
                member_path.parent != Path(".")
                or member_path.suffix != ".json"
                or not member_path.stem.isdigit()
            ):
                continue
            if member_path.stem in numeric:
                raise RuntimeError(
                    f"Aug7 duplicate numeric replay ID {member_path.stem}"
                )
            numeric[member_path.stem] = name
        manifest_ids, manifest_sha = common.read_daily_manifest(
            archive,
            contract.dataset_date,
        )
        if manifest_sha != contract.daily_manifest_sha256:
            raise RuntimeError(
                "Aug7 daily manifest SHA256 mismatch: "
                f"expected {contract.daily_manifest_sha256}, got {manifest_sha}"
            )
        common.audit_daily_replay_member_set(
            dataset_date=contract.dataset_date,
            numeric_member_ids=set(numeric),
            manifest_ids=manifest_ids,
            frozen_missing_ids=contract.daily_missing_replay_ids,
            source_episode_ids={value.episode_id for value in source_views.values()},
        )
        replay_cache: dict[str, tuple[bytes, dict[str, Any]]] = {}
        selected_episode_ids: set[str] = set()
        for _view_key, episode in sorted(
            source_views.items(),
            key=lambda item: (int(item[1].episode_id), item[1].seat, item[1].team_name),
        ):
            episode_id = episode.episode_id
            member = numeric.get(episode_id)
            if member is None:
                raise RuntimeError(f"Aug7 source episode {episode_id} missing")
            cached = replay_cache.get(episode_id)
            if cached is None:
                replay_payload = archive.read(member)
                replay = orjson.loads(replay_payload)
                if not isinstance(replay, dict):
                    raise RuntimeError(f"Aug7 replay {episode_id} is not an object")
                replay_cache[episode_id] = (replay_payload, replay)
            else:
                replay_payload, replay = cached
            if common.replay_episode_id(replay) != episode_id:
                raise RuntimeError(f"Aug7 replay {episode_id} identity mismatch")
            replay_uuid = replay.get("id")
            replay_uuid = str(replay_uuid) if replay_uuid is not None else None
            if replay_uuid != episode.episode_uuid:
                raise RuntimeError(f"Aug7 replay {episode_id} UUID mismatch")
            names_pair = common.replay_names(replay)
            if names_pair[episode.seat] != episode.team_name or names_pair[
                1 - episode.seat
            ] != episode.opponent_team_name:
                raise RuntimeError(f"Aug7 replay {episode_id} team mismatch")
            rewards = common.replay_rewards(replay)
            if rewards[episode.seat] != episode.terminal_reward:
                raise RuntimeError(f"Aug7 replay {episode_id} reward mismatch")
            decks = common.replay_deck_hashes(replay)
            if decks[episode.seat] != contract.learner_deck_hash:
                raise RuntimeError(f"Aug7 replay {episode_id} learner deck mismatch")
            if decks[1 - episode.seat] != contract.opponent_deck_hash:
                continue
            if episode_id in selected_episode_ids:
                raise RuntimeError(
                    f"Aug7 exact-Fros matched multiple learner views for {episode_id}"
                )
            learner_reward = rewards[episode.seat]
            opponent_reward = rewards[1 - episode.seat]
            if learner_reward > opponent_reward and learner_reward > 0:
                outcome = "win"
            elif learner_reward < opponent_reward and learner_reward < 0:
                outcome = "loss"
            else:
                raise RuntimeError(f"Aug7 replay {episode_id} is not strict win/loss")
            selected.append(
                ValidEpisode(
                    episode_id=episode_id,
                    episode_uuid=episode.episode_uuid,
                    seat=episode.seat,
                    team_name=episode.team_name,
                    opponent_team_name=episode.opponent_team_name,
                    terminal_reward=episode.terminal_reward,
                    outcome=outcome,
                    rows=episode.rows,
                    raw_rows_sha256=episode.raw_digest.hexdigest(),
                    decision_keys_digest=common.digest_lines(episode.decision_keys),
                    replay_member=member,
                    replay_sha256=common.sha256_bytes(replay_payload),
                )
            )
            selected_episode_ids.add(episode_id)
    wins = sum(value.outcome == "win" for value in selected)
    losses = sum(value.outcome == "loss" for value in selected)
    if (
        len(selected) != contract.expected_episodes
        or sum(value.rows for value in selected) != contract.expected_rows
        or wins != contract.expected_wins
        or losses != contract.expected_losses
    ):
        raise RuntimeError(
            "Aug7 exact-Fros frozen count drift: "
            f"episodes={len(selected)}, rows={sum(value.rows for value in selected)}, "
            f"wins={wins}, losses={losses}"
        )
    selected.sort(key=lambda value: int(value.episode_id))
    return tuple(selected), manifest_sha, len(numeric)


def materialize_lines(
    source: Path,
    selected: Sequence[ValidEpisode],
    contract: ValidContract,
) -> tuple[tuple[bytes, ...], str, str]:
    wanted = {
        (contract.dataset_date, value.episode_id, value.seat, value.team_name)
        for value in selected
    }
    expected_counts = {
        (contract.dataset_date, value.episode_id, value.seat, value.team_name): value.rows
        for value in selected
    }
    expected_digests = {
        (
            contract.dataset_date,
            value.episode_id,
            value.seat,
            value.team_name,
        ): value.raw_rows_sha256
        for value in selected
    }
    counts: Counter[tuple[str, str, int, str]] = Counter()
    digests = {view_key: hashlib.sha256() for view_key in wanted}
    lines: list[bytes] = []
    keys: list[str] = []
    adapter = dataclasses_contract(contract)
    with zipfile.ZipFile(source) as archive:
        with archive.open(contract.valid_member) as handle:
            for line_number, raw_line in enumerate(handle, 1):
                row = orjson.loads(raw_line)
                episode_id = str(row.get("episode_id") or "")
                raw_seat = row.get("seat")
                seat = raw_seat if isinstance(raw_seat, int) else -1
                team_name = str(row.get("team_name") or "")
                dataset_date = str(row.get("dataset_date") or "")
                view_key = (dataset_date, episode_id, seat, team_name)
                if view_key not in wanted:
                    continue
                key, *_rest = common.row_identity(
                    row,
                    f"materialize:{line_number}",
                    adapter,
                )
                lines.append(raw_line)
                keys.append(key)
                counts[view_key] += 1
                digests[view_key].update(raw_line)
    if dict(counts) != expected_counts:
        raise RuntimeError("Aug7 materialized per-episode row counts drift")
    if {key: value.hexdigest() for key, value in digests.items()} != expected_digests:
        raise RuntimeError("Aug7 materialized raw episode bytes drift")
    payload = b"".join(lines)
    return tuple(lines), common.sha256_bytes(payload), common.digest_lines(keys)


def audit_inputs(source: Path, daily: Path, contract: ValidContract) -> ValidAudit:
    validate_contract(contract)
    manifest, source_views, member_sha = scan_source(source, contract)
    selected, daily_manifest_sha, replay_members = select_replays(
        daily,
        source_views,
        contract,
    )
    lines, content_sha, keys_digest = materialize_lines(source, selected, contract)
    winner_counts = Counter(
        value.team_name for value in selected if value.outcome == "win"
    )
    loser_counts = Counter(
        value.team_name for value in selected if value.outcome == "loss"
    )
    source_view_counts = Counter(value.episode_id for value in source_views.values())
    return ValidAudit(
        source_manifest=manifest,
        source_member_sha256=member_sha,
        source_episode_count=len(source_view_counts),
        source_view_count=len(source_views),
        source_mirror_episode_count=sum(
            value > 1 for value in source_view_counts.values()
        ),
        selected=selected,
        selected_lines=lines,
        selected_content_sha256=content_sha,
        decision_keys_digest=keys_digest,
        winner_episode_counts=dict(sorted(winner_counts.items())),
        loser_episode_counts=dict(sorted(loser_counts.items())),
        daily_manifest_sha256=daily_manifest_sha,
        daily_replay_members=replay_members,
    )


def plan_document(contract: ValidContract) -> dict[str, Any]:
    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "data_schema_version": common.ROW_SCHEMA_VERSION,
        "source": {
            "logical_path": contract.source_logical_path,
            "sha256": contract.source_sha256,
            "manifest_sha256": contract.source_manifest_sha256,
            "member": contract.valid_member,
            "valid_episodes": contract.source_valid_episodes,
            "valid_views": contract.source_valid_views,
            "mirror_episode_ids": list(contract.source_mirror_episode_ids),
            "valid_rows": contract.source_valid_rows,
        },
        "daily": {
            "logical_path": contract.daily_logical_path,
            "sha256": contract.daily_sha256,
            "manifest_sha256": contract.daily_manifest_sha256,
            "missing_replay_ids": list(contract.daily_missing_replay_ids),
            "missing_source_episode_overlap": 0,
            "incomplete_terminal_replay_ids": [],
            "date": contract.dataset_date,
        },
        "incomplete_terminal_replays": [],
        "selection": {
            "split": "valid",
            "learner_deck_hash": contract.learner_deck_hash,
            "opponent_deck_hash": contract.opponent_deck_hash,
            "terminal_reward": "all_strict_win_or_loss",
            "episodes": contract.expected_episodes,
            "rows": contract.expected_rows,
            "wins": contract.expected_wins,
            "losses": contract.expected_losses,
            "aug8_isolated": True,
        },
        "output_members": [VALID_MEMBER, ALLOWLIST_MEMBER, MANIFEST_MEMBER],
    }


def allowlist_document(audit: ValidAudit, contract: ValidContract) -> dict[str, Any]:
    episode_ids = [value.episode_id for value in audit.selected]
    if not episode_ids or len(episode_ids) != len(set(episode_ids)):
        raise RuntimeError("Aug7 allowlist episode IDs are empty or duplicated")
    return {
        "schema_version": ALLOWLIST_SCHEMA_VERSION,
        "data_schema_version": common.ROW_SCHEMA_VERSION,
        "learner_deck_hash": contract.learner_deck_hash,
        "opponent_deck_hash": contract.opponent_deck_hash,
        "split": "valid",
        "terminal_reward": "all",
        "date": contract.dataset_date,
        "episode_count": len(audit.selected),
        "decision_rows": len(audit.selected_lines),
        "date_episode_counts": {contract.dataset_date: len(audit.selected)},
        "date_decision_rows": {contract.dataset_date: len(audit.selected_lines)},
        "incomplete_terminal_replays": [],
        "win_episodes": contract.expected_wins,
        "loss_episodes": contract.expected_losses,
        "episode_ids": episode_ids,
        "episodes": [
            {
                "episode_id": value.episode_id,
                "date": contract.dataset_date,
                "decision_rows": value.rows,
                "outcome": value.outcome,
                "seat": value.seat,
                "team_name": value.team_name,
                "opponent_team_name": value.opponent_team_name,
                "terminal_reward": value.terminal_reward,
                "learner_deck_hash": contract.learner_deck_hash,
                "opponent_deck_hash": contract.opponent_deck_hash,
                "replay_member": value.replay_member,
                "replay_sha256": value.replay_sha256,
                "raw_rows_sha256": value.raw_rows_sha256,
                "decision_keys_digest": value.decision_keys_digest,
            }
            for value in audit.selected
        ],
    }


def validate_allowlist_document(
    payload: Any,
    contract: ValidContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Pure shared consumer contract for the Aug7 exact-Fros view."""

    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
        try:
            document = orjson.loads(raw)
        except orjson.JSONDecodeError as exc:
            raise RuntimeError("Fros-valid allowlist is invalid JSON") from exc
        if common.canonical_json_bytes(document) + b"\n" != raw:
            raise RuntimeError("Fros-valid allowlist is not canonical JSON+LF")
    else:
        document = payload
    if not isinstance(document, dict):
        raise RuntimeError("Fros-valid allowlist is not an object")
    checks = {
        "schema_version": (
            document.get("schema_version"),
            ALLOWLIST_SCHEMA_VERSION,
        ),
        "data_schema_version": (
            document.get("data_schema_version"),
            common.ROW_SCHEMA_VERSION,
        ),
        "learner_deck_hash": (
            document.get("learner_deck_hash"),
            contract.learner_deck_hash,
        ),
        "opponent_deck_hash": (
            document.get("opponent_deck_hash"),
            contract.opponent_deck_hash,
        ),
        "split": (document.get("split"), "valid"),
        "terminal_reward": (document.get("terminal_reward"), "all"),
        "date": (document.get("date"), contract.dataset_date),
        "episode_count": (document.get("episode_count"), contract.expected_episodes),
        "decision_rows": (document.get("decision_rows"), contract.expected_rows),
        "date_episode_counts": (
            document.get("date_episode_counts"),
            {contract.dataset_date: contract.expected_episodes},
        ),
        "incomplete_terminal_replays": (
            document.get("incomplete_terminal_replays"),
            [],
        ),
        "win_episodes": (document.get("win_episodes"), contract.expected_wins),
        "loss_episodes": (document.get("loss_episodes"), contract.expected_losses),
    }
    failures = [
        f"{name}={actual!r} expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if failures:
        raise RuntimeError("Fros-valid allowlist contract drift: " + "; ".join(failures))
    episodes = document.get("episodes")
    episode_ids = document.get("episode_ids")
    if not isinstance(episodes, list) or len(episodes) != contract.expected_episodes:
        raise RuntimeError("Fros-valid episodes length drift")
    observed_ids = [str(value.get("episode_id") or "") for value in episodes]
    if (
        not isinstance(episode_ids, list)
        or episode_ids != observed_ids
        or any(not value for value in observed_ids)
        or len(observed_ids) != len(set(observed_ids))
    ):
        raise RuntimeError("Fros-valid episode identity drift")
    if observed_ids != sorted(observed_ids, key=int):
        raise RuntimeError("Fros-valid episodes are not numerically sorted")
    if sum(int(value.get("decision_rows", -1)) for value in episodes) != contract.expected_rows:
        raise RuntimeError("Fros-valid episode row sum drift")
    outcomes = Counter(str(value.get("outcome") or "") for value in episodes)
    if outcomes != Counter(win=contract.expected_wins, loss=contract.expected_losses):
        raise RuntimeError("Fros-valid episode outcome counts drift")
    if any(value.get("date") != contract.dataset_date for value in episodes):
        raise RuntimeError("Fros-valid episode date drift")
    return document


def manifest_document(
    audit: ValidAudit,
    contract: ValidContract,
    plan_sha256: str,
    allowlist_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "data_schema_version": common.ROW_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "learner_deck_hash": contract.learner_deck_hash,
        "opponent_deck_hash": contract.opponent_deck_hash,
        "split": "valid",
        "terminal_reward": "all",
        "date": contract.dataset_date,
        "episode_count": len(audit.selected),
        "decision_rows": len(audit.selected_lines),
        "date_episode_counts": {contract.dataset_date: len(audit.selected)},
        "date_decision_rows": {contract.dataset_date: len(audit.selected_lines)},
        "win_episodes": contract.expected_wins,
        "loss_episodes": contract.expected_losses,
        "split_episodes": {"valid": len(audit.selected)},
        "split_decisions": {"valid": len(audit.selected_lines)},
        "winner_episode_counts": audit.winner_episode_counts,
        "loser_episode_counts": audit.loser_episode_counts,
        "source": {
            "logical_path": contract.source_logical_path,
            "sha256": contract.source_sha256,
            "manifest_sha256": contract.source_manifest_sha256,
            "member": contract.valid_member,
            "member_sha256": audit.source_member_sha256,
            "valid_episodes": audit.source_episode_count,
            "valid_views": audit.source_view_count,
            "mirror_episode_count": audit.source_mirror_episode_count,
            "mirror_episode_ids": list(contract.source_mirror_episode_ids),
            "valid_rows": contract.source_valid_rows,
        },
        "daily": {
            "logical_path": contract.daily_logical_path,
            "sha256": contract.daily_sha256,
            "manifest_sha256": audit.daily_manifest_sha256,
            "replay_members": audit.daily_replay_members,
            "missing_replay_ids": list(contract.daily_missing_replay_ids),
            "missing_source_episode_overlap": 0,
            "incomplete_terminal_replay_ids": [],
        },
        "incomplete_terminal_replays": [],
        "members": [
            {
                "member": VALID_MEMBER,
                "episodes": len(audit.selected),
                "rows": len(audit.selected_lines),
                "sha256": audit.selected_content_sha256,
            }
        ],
        "exact_episode_allowlist": {
            "member": ALLOWLIST_MEMBER,
            "external_filename": ALLOWLIST_MEMBER,
            "schema_version": ALLOWLIST_SCHEMA_VERSION,
            "sha256": allowlist_sha256,
            "canonical_sha256": allowlist_sha256,
            "episodes": len(audit.selected),
            "decision_rows": len(audit.selected_lines),
        },
        "byte_preservation": {
            "source_valid_rows_copied_verbatim": True,
            "selected_content_sha256": audit.selected_content_sha256,
            "decision_keys_digest": audit.decision_keys_digest,
            "visualize_copied": False,
        },
        "temporal_isolation": {
            "only_date": contract.dataset_date,
            "aug8_input_count": 0,
            "aug8_rows": 0,
        },
        "zip": {
            "timestamp": list(common.ZIP_TIMESTAMP),
            "compression": "deflate",
            "compresslevel": common.ZIP_COMPRESSLEVEL,
            "member_order": [VALID_MEMBER, ALLOWLIST_MEMBER, MANIFEST_MEMBER],
            "double_serialization_required": True,
        },
    }


def create_archive(
    audit: ValidAudit,
    allowlist_payload: bytes,
    manifest: dict[str, Any],
) -> bytes:
    valid_payload = b"".join(audit.selected_lines)
    manifest_payload = common.canonical_json_bytes(manifest) + b"\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=common.ZIP_COMPRESSION,
        compresslevel=common.ZIP_COMPRESSLEVEL,
        allowZip64=True,
    ) as archive:
        archive.writestr(common.zip_info(VALID_MEMBER), valid_payload)
        archive.writestr(common.zip_info(ALLOWLIST_MEMBER), allowlist_payload)
        archive.writestr(common.zip_info(MANIFEST_MEMBER), manifest_payload)
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != [VALID_MEMBER, ALLOWLIST_MEMBER, MANIFEST_MEMBER]:
            raise RuntimeError("Fros-valid output member order drift")
        if archive.testzip() is not None:
            raise RuntimeError("Fros-valid output CRC failure")
        if archive.read(VALID_MEMBER) != valid_payload:
            raise RuntimeError("Fros-valid row bytes changed")
        if archive.read(ALLOWLIST_MEMBER) != allowlist_payload:
            raise RuntimeError("Fros-valid allowlist bytes changed")
        if archive.read(MANIFEST_MEMBER) != manifest_payload:
            raise RuntimeError("Fros-valid manifest bytes changed")
    return payload


def audit_and_serialize(
    source: Path,
    daily: Path,
    contract: ValidContract,
) -> tuple[ValidAudit, bytes, bytes, dict[str, str]]:
    audit = audit_inputs(source, daily, contract)
    plan = plan_document(contract)
    plan_sha = common.sha256_bytes(common.canonical_json_bytes(plan))
    allowlist = allowlist_document(audit, contract)
    allowlist_payload = common.canonical_json_bytes(allowlist) + b"\n"
    validate_allowlist_document(allowlist_payload, contract)
    if common.canonical_json_bytes(orjson.loads(allowlist_payload)) + b"\n" != allowlist_payload:
        raise RuntimeError("Fros-valid allowlist is not canonical JSON+LF")
    allowlist_sha = common.sha256_bytes(allowlist_payload)
    manifest = manifest_document(audit, contract, plan_sha, allowlist_sha)
    first = create_archive(audit, allowlist_payload, manifest)
    second = create_archive(audit, allowlist_payload, manifest)
    if first != second:
        raise RuntimeError("Fros-valid double serialization differs")
    manifest_payload = common.canonical_json_bytes(manifest) + b"\n"
    return audit, first, allowlist_payload, {
        "plan_sha256": plan_sha,
        "archive_sha256": common.sha256_bytes(first),
        "allowlist_sha256": allowlist_sha,
        "manifest_sha256": common.sha256_bytes(manifest_payload),
    }


def build(
    *,
    source: Path,
    daily: Path,
    archive_path: Path,
    allowlist_path: Path,
    contract: ValidContract = DEFAULT_CONTRACT,
    execute: bool,
    expected_plan_sha256: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_allowlist_sha256: str | None = None,
) -> dict[str, Any]:
    source = common.audit_path_components(
        source, "Fros-valid source", must_exist=True, kind="file"
    )
    daily = common.audit_path_components(
        daily, "Aug7 daily source", must_exist=True, kind="file"
    )
    archive_path = common.audit_path_components(
        archive_path, "Fros-valid archive output", must_exist=False, kind="output"
    )
    allowlist_path = common.audit_path_components(
        allowlist_path, "Fros-valid allowlist output", must_exist=False, kind="output"
    )
    if archive_path.parent != allowlist_path.parent:
        raise RuntimeError("Fros-valid outputs must share a directory")
    if os.path.lexists(archive_path.parent):
        raise FileExistsError(
            f"Fros-valid bundle directory must be absent: {archive_path.parent}"
        )
    if execute and not all(
        (expected_plan_sha256, expected_archive_sha256, expected_allowlist_sha256)
    ):
        raise RuntimeError("execute requires reviewed plan/archive/allowlist SHA gates")
    if execute:
        common.require_sha256(str(expected_plan_sha256), "expected_plan_sha256")
        common.require_sha256(str(expected_archive_sha256), "expected_archive_sha256")
        common.require_sha256(str(expected_allowlist_sha256), "expected_allowlist_sha256")
    before = {"source": common.sha256_file(source), "daily": common.sha256_file(daily)}
    audit, archive_payload, allowlist_payload, metadata = audit_and_serialize(
        source, daily, contract
    )
    after = {"source": common.sha256_file(source), "daily": common.sha256_file(daily)}
    if before != after:
        raise RuntimeError("Fros-valid inputs changed during audit")
    if execute:
        expected = {
            "plan_sha256": expected_plan_sha256,
            "archive_sha256": expected_archive_sha256,
            "allowlist_sha256": expected_allowlist_sha256,
        }
        if any(metadata[name] != value for name, value in expected.items()):
            raise RuntimeError("reviewed Fros-valid SHA gate mismatch")
        bundle_utils.publish_bundle_no_clobber(
            archive_path,
            allowlist_path,
            archive_payload,
            allowlist_payload,
        )
        if common.sha256_file(archive_path) != metadata["archive_sha256"]:
            raise RuntimeError("published Fros-valid archive SHA mismatch")
        if common.sha256_file(allowlist_path) != metadata["allowlist_sha256"]:
            raise RuntimeError("published Fros-valid allowlist SHA mismatch")
        with zipfile.ZipFile(archive_path) as published:
            if published.read(ALLOWLIST_MEMBER) != allowlist_path.read_bytes():
                raise RuntimeError("published Fros-valid allowlists differ")
            if common.sha256_bytes(published.read(MANIFEST_MEMBER)) != metadata[
                "manifest_sha256"
            ]:
                raise RuntimeError("published Fros-valid manifest SHA mismatch")
    return {
        "status": "built" if execute else "dry_run_passed",
        "dry_run": not execute,
        "output_written": execute,
        "archive": str(archive_path),
        "external_allowlist": str(allowlist_path),
        "outer_zip_sha256": metadata["archive_sha256"],
        "archive_sha256": metadata["archive_sha256"],
        "allowlist_sha256": metadata["allowlist_sha256"],
        "allowlist_canonical_sha256": metadata["allowlist_sha256"],
        "embedded_manifest_raw_sha256": metadata["manifest_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "plan_sha256": metadata["plan_sha256"],
        "episode_count": len(audit.selected),
        "decision_rows": len(audit.selected_lines),
        "win_episodes": contract.expected_wins,
        "loss_episodes": contract.expected_losses,
        "deterministic_rebuild_match": True,
        "rows_copied_verbatim": True,
        "aug8_inputs": 0,
        "training_started": False,
        "evaluation_started": False,
        "submission_started": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--expected-allowlist-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(
        source=args.source,
        daily=args.daily,
        archive_path=args.archive,
        allowlist_path=args.allowlist,
        execute=args.execute,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_allowlist_sha256=args.expected_allowlist_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
