#!/usr/bin/env python3
"""Build the frozen exact-Marnie anti-@kdcyberdude replay bundle.

This is a thin selector over :mod:`build_marnie_froslass_exact_wins`.  It
reuses that module's frozen source parsing, official-replay identity checks,
deck hashing, canonical JSON, deterministic ZIP metadata, and no-clobber
publication helpers.  No replay ``visualize`` payload is copied: selected BC
JSONL rows are preserved byte for byte from ``marnie_trainwins.zip``.

The default mode is a write-free dry run.  Execution requires reviewed plan,
archive, and allowlist SHA-256 gates.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import errno
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
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


REPO_ROOT = Path(os.path.abspath(os.fspath(Path(__file__).parent))).parent
ARCHIVE_SCHEMA_VERSION = "ptcg-marnie-exact-anti-kd-archive-v1"
ALLOWLIST_SCHEMA_VERSION = "ptcg-marnie-exact-anti-kd-trainwin-allowlist-v1"
TRAIN_MEMBER = "train/part-00000.jsonl"
DEV_MEMBER = "dev/part-00000.jsonl"
ALLOWLIST_MEMBER = "exact_episode_allowlist.json"
MANIFEST_MEMBER = "manifest.json"
KD_TEAM_NAME = "@kdcyberdude"
SPLIT_DOMAIN_SEPARATOR = "ptcg-anti-kd-dev-v1"
SPLIT_ALGORITHM = "sha256_domain_date_episode_lexicographic"
SPLIT_PAYLOAD = "domain_separator + NUL + date + NUL + episode_id"
SPLIT_ORDER = "ascending_hex_digest_then_date_then_numeric_episode_id"

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data/gold_push_marnie_antikd_exact_20260810_v1"
)
DEFAULT_ARCHIVE = DEFAULT_OUTPUT_ROOT / "marnie_exact_anti_kd_trainwins.zip"
DEFAULT_ALLOWLIST = DEFAULT_OUTPUT_ROOT / ALLOWLIST_MEMBER

TOP20_NAMES = (
    "@kdcyberdude",
    "AlphaStarmie",
    "Bart, Edwyn",
    "Dipam Chakraborty",
    "James Cox & Henry Chao",
    "LiamK",
    "M Sato",
    "Majkel1337",
    "MissingNo.",
    "Pokemon Siuuuu",
    "Raihan Ramadistra",
    "Thai",
    "flg",
    "palsystem",
    "perrodawn",
    "sadwat",
    "vvs",
    "カントー地方マスター(KantoRegionMaster)",
    "想要成为kaggle大师",
    "🫣🤧",
)
PRIMARY_WINNER = "Raihan Ramadistra"
SECONDARY_WINNER = "カントー地方マスター(KantoRegionMaster)"
TERTIARY_WINNER = "LiamK"


@dataclass(frozen=True)
class AntiKDContract:
    base: common.BuildContract
    top20_names: tuple[str, ...]
    kd_team_name: str
    expected_episodes: int
    expected_rows: int
    expected_date_episodes: tuple[tuple[str, int], ...]
    expected_winner_groups: tuple[tuple[str, int], ...]
    train_episode_count: int
    dev_episode_count: int
    split_domain_separator: str


ANTI_BASE_CONTRACT = dataclasses.replace(
    common.DEFAULT_CONTRACT,
    opponent_deck_hash=common.MARNIE_DECK_HASH,
    expected_selected_rows=4_873,
    expected_selected_episodes=43,
    expected_date_episodes=(
        ("2026-08-02", 27),
        ("2026-08-03", 5),
        ("2026-08-04", 6),
        ("2026-08-05", 4),
        ("2026-08-06", 1),
    ),
)

DEFAULT_CONTRACT = AntiKDContract(
    base=ANTI_BASE_CONTRACT,
    top20_names=TOP20_NAMES,
    kd_team_name=KD_TEAM_NAME,
    expected_episodes=43,
    expected_rows=4_873,
    expected_date_episodes=ANTI_BASE_CONTRACT.expected_date_episodes,
    expected_winner_groups=(
        (PRIMARY_WINNER, 23),
        (SECONDARY_WINNER, 7),
        (TERTIARY_WINNER, 5),
        ("__others__", 8),
    ),
    train_episode_count=35,
    dev_episode_count=8,
    split_domain_separator=SPLIT_DOMAIN_SEPARATOR,
)


@dataclass(frozen=True)
class AntiKDAudit:
    source: common.SourceAudit
    selected: tuple[common.SelectedEpisode, ...]
    daily: tuple[common.DailyAudit, ...]
    all_lines: tuple[bytes, ...]
    train_lines: tuple[bytes, ...]
    dev_lines: tuple[bytes, ...]
    selected_content_sha256: str
    decision_keys_digest: str
    date_rows: tuple[tuple[str, int], ...]
    split_by_episode: dict[tuple[str, str], str]
    split_episode_counts: dict[str, int]
    split_decision_rows: dict[str, int]
    split_date_episode_counts: dict[str, dict[str, int]]
    split_date_decision_rows: dict[str, dict[str, int]]
    winner_episode_counts: dict[str, int]


def validate_contract(contract: AntiKDContract) -> None:
    base = contract.base
    common.require_sha256(base.source_sha256, "source_sha256")
    common.require_sha256(base.source_manifest_sha256, "source_manifest_sha256")
    common.require_sha256(base.learner_deck_hash, "learner_deck_hash")
    daily = dict(base.daily_sha256)
    daily_manifests = dict(base.daily_manifest_sha256)
    missing_replays = dict(base.daily_missing_replay_ids)
    if (
        len(daily) != len(base.daily_sha256)
        or len(daily_manifests) != len(base.daily_manifest_sha256)
        or len(missing_replays) != len(base.daily_missing_replay_ids)
        or set(daily) != set(base.dates)
        or set(daily_manifests) != set(base.dates)
        or set(missing_replays) != set(base.dates)
    ):
        raise RuntimeError("anti-KD daily contract coverage drift")
    for dataset_date, digest in base.daily_sha256:
        common.require_sha256(digest, f"daily_sha256[{dataset_date}]")
    for dataset_date, digest in base.daily_manifest_sha256:
        common.require_sha256(digest, f"daily_manifest_sha256[{dataset_date}]")
    for dataset_date, episode_ids in base.daily_missing_replay_ids:
        if (
            any(not value.isdigit() for value in episode_ids)
            or len(episode_ids) != len(set(episode_ids))
            or tuple(sorted(episode_ids, key=int)) != episode_ids
        ):
            raise RuntimeError(
                f"anti-KD missing-replay IDs drift for {dataset_date}"
            )
    incomplete_keys = [
        (value.dataset_date, value.episode_id)
        for value in base.incomplete_terminal_replays
    ]
    if len(incomplete_keys) != len(set(incomplete_keys)):
        raise RuntimeError("anti-KD incomplete-terminal keys repeat")
    for value in base.incomplete_terminal_replays:
        if (
            value.dataset_date not in base.dates
            or value.learner_deck_hash != base.learner_deck_hash
            or value.opponent_team_name == contract.kd_team_name
            or value.opponent_deck_hash == base.opponent_deck_hash
        ):
            raise RuntimeError("incomplete-terminal replay overlaps anti-KD target")
    if base.learner_deck_hash != base.opponent_deck_hash:
        raise RuntimeError("anti-KD requires identical learner/opponent deck hashes")
    if len(contract.top20_names) != len(set(contract.top20_names)):
        raise RuntimeError("frozen Top20 names contain duplicates")
    if contract.kd_team_name not in contract.top20_names:
        raise RuntimeError("KD team is absent from frozen Top20")
    if contract.expected_episodes != contract.train_episode_count + contract.dev_episode_count:
        raise RuntimeError("train/dev episode counts do not sum to total")
    if sum(dict(contract.expected_date_episodes).values()) != contract.expected_episodes:
        raise RuntimeError("date episode counts do not sum to total")
    if sum(dict(contract.expected_winner_groups).values()) != contract.expected_episodes:
        raise RuntimeError("winner groups do not sum to total")
    if contract.dev_episode_count <= 0 or contract.train_episode_count <= 0:
        raise RuntimeError("train and dev episode counts must be positive")
    if not contract.split_domain_separator or "\0" in contract.split_domain_separator:
        raise RuntimeError("invalid split domain separator")


def validate_top20_manifest(
    source: common.SourceAudit,
    contract: AntiKDContract,
) -> None:
    team_filter = source.manifest.get("team_filter")
    display_names = (
        team_filter.get("display_names")
        if isinstance(team_filter, Mapping)
        else None
    )
    if tuple(display_names or ()) != contract.top20_names:
        raise RuntimeError(
            "source frozen Top20 display_names drift: "
            f"expected {contract.top20_names!r}, got {display_names!r}"
        )
    if not isinstance(team_filter, Mapping) or team_filter.get(
        "global_team_count"
    ) != len(contract.top20_names):
        raise RuntimeError("source Top20 global_team_count drift")


def split_score(
    dataset_date: str,
    episode_id: str,
    domain_separator: str,
) -> str:
    payload = (
        domain_separator.encode("utf-8")
        + b"\0"
        + dataset_date.encode("utf-8")
        + b"\0"
        + episode_id.encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def assign_splits(
    selected: Sequence[common.SelectedEpisode],
    contract: AntiKDContract,
) -> dict[tuple[str, str], str]:
    ranked = sorted(
        selected,
        key=lambda episode: (
            split_score(
                episode.dataset_date,
                episode.episode_id,
                contract.split_domain_separator,
            ),
            episode.dataset_date,
            int(episode.episode_id),
        ),
    )
    dev_keys = {
        (episode.dataset_date, episode.episode_id)
        for episode in ranked[: contract.dev_episode_count]
    }
    result = {
        (episode.dataset_date, episode.episode_id): (
            "dev"
            if (episode.dataset_date, episode.episode_id) in dev_keys
            else "train"
        )
        for episode in selected
    }
    counts = Counter(result.values())
    if counts != Counter(
        train=contract.train_episode_count,
        dev=contract.dev_episode_count,
    ):
        raise RuntimeError(f"derived train/dev split drift: {dict(counts)!r}")
    return result


def scan_anti_kd_replays(
    daily_dir: Path,
    source: common.SourceAudit,
    contract: AntiKDContract,
) -> tuple[
    tuple[common.SelectedEpisode, ...],
    tuple[common.DailyAudit, ...],
    dict[str, int],
]:
    base = contract.base
    expected_daily = dict(base.daily_sha256)
    expected_daily_manifests = dict(base.daily_manifest_sha256)
    expected_missing_replays = dict(base.daily_missing_replay_ids)
    expected_incomplete = {
        (value.dataset_date, value.episode_id): value
        for value in base.incomplete_terminal_replays
    }
    observed_incomplete: set[tuple[str, str]] = set()
    selected: list[common.SelectedEpisode] = []
    daily_audits: list[common.DailyAudit] = []
    date_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    allowed_learners = set(contract.top20_names) - {contract.kd_team_name}

    for dataset_date in base.dates:
        filename = f"pokemon-tcg-ai-battle-episodes-{dataset_date}.zip"
        path = daily_dir / filename
        common.require_regular_file(path, f"daily replay archive {dataset_date}")
        actual_sha = common.sha256_file(path)
        if actual_sha != expected_daily[dataset_date]:
            raise RuntimeError(f"{dataset_date}: daily SHA256 mismatch")
        date_episodes = sorted(
            (
                episode
                for (date_value, _episode_id), episode in source.episodes.items()
                if date_value == dataset_date
            ),
            key=lambda episode: int(episode.episode_id),
        )
        date_incomplete_ids: list[str] = []
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError(f"{dataset_date}: duplicate daily ZIP members")
            numeric_members = {
                Path(name).stem: name
                for name in names
                if Path(name).parent == Path(".")
                and Path(name).suffix == ".json"
                and Path(name).stem.isdigit()
            }
            if len(numeric_members) != sum(
                1
                for name in names
                if Path(name).parent == Path(".")
                and Path(name).suffix == ".json"
                and Path(name).stem.isdigit()
            ):
                raise RuntimeError(f"{dataset_date}: duplicate replay IDs")
            manifest_ids, manifest_sha = common.read_daily_manifest(
                archive,
                dataset_date,
            )
            if manifest_sha != expected_daily_manifests[dataset_date]:
                raise RuntimeError(
                    f"{dataset_date}: daily manifest SHA256 mismatch"
                )
            missing_replay_ids = common.audit_daily_replay_member_set(
                dataset_date=dataset_date,
                numeric_member_ids=set(numeric_members),
                manifest_ids=manifest_ids,
                frozen_missing_ids=expected_missing_replays[dataset_date],
                source_episode_ids={
                    episode_id
                    for (date_value, episode_id) in source.episodes
                    if date_value == dataset_date
                },
            )
            for episode in date_episodes:
                member = numeric_members.get(episode.episode_id)
                if member is None:
                    raise RuntimeError(
                        f"{dataset_date}: source episode {episode.episode_id} missing"
                    )
                replay_payload = archive.read(member)
                try:
                    replay = orjson.loads(replay_payload)
                except orjson.JSONDecodeError as exc:
                    raise RuntimeError(f"{dataset_date}:{member}: invalid replay") from exc
                if not isinstance(replay, dict):
                    raise RuntimeError(f"{dataset_date}:{member}: replay not object")
                if common.replay_episode_id(replay) != episode.episode_id:
                    raise RuntimeError(f"{dataset_date}:{member}: EpisodeId mismatch")
                replay_uuid = replay.get("id")
                replay_uuid = str(replay_uuid) if replay_uuid is not None else None
                if replay_uuid != episode.episode_uuid:
                    raise RuntimeError(f"{dataset_date}:{member}: episode UUID mismatch")
                names_pair = common.replay_names(replay)
                if names_pair[episode.seat] != episode.team_name:
                    raise RuntimeError(f"{dataset_date}:{member}: learner team mismatch")
                if names_pair[1 - episode.seat] != episode.opponent_team_name:
                    raise RuntimeError(f"{dataset_date}:{member}: opponent team mismatch")
                replay_key = (dataset_date, episode.episode_id)
                raw_rewards = replay.get("rewards")
                complete_rewards = True
                try:
                    rewards = common.replay_rewards(replay)
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
                    replay_sha = common.sha256_bytes(replay_payload)
                    if not isinstance(raw_rewards, list) or tuple(raw_rewards) != (
                        expected_exception.raw_rewards
                    ):
                        raise RuntimeError(
                            f"{dataset_date}:{member}: incomplete raw rewards drift"
                        )
                    if replay_sha != expected_exception.replay_sha256:
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
                    learner_reward = common.numeric_reward(
                        raw_rewards[episode.seat],
                        f"{dataset_date}:{member}: incomplete learner reward",
                    )
                    if learner_reward != episode.terminal_reward:
                        raise RuntimeError(
                            f"{dataset_date}:{member}: incomplete learner reward drift"
                        )
                    deck_hashes = common.replay_deck_hashes(replay)
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
                        episode.opponent_team_name == contract.kd_team_name
                        or deck_hashes[1 - episode.seat] == base.opponent_deck_hash
                    ):
                        raise RuntimeError(
                            f"{dataset_date}:{member}: incomplete replay may match anti-KD"
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
                deck_hashes = common.replay_deck_hashes(replay)
                if deck_hashes[episode.seat] != base.learner_deck_hash:
                    raise RuntimeError(f"{dataset_date}:{member}: learner deck mismatch")
                if not (
                    episode.opponent_team_name == contract.kd_team_name
                    and episode.team_name in allowed_learners
                ):
                    continue
                if deck_hashes[1 - episode.seat] != base.opponent_deck_hash:
                    continue
                selected.append(
                    common.SelectedEpisode(
                        dataset_date=dataset_date,
                        episode_id=episode.episode_id,
                        replay_member=member,
                        replay_sha256=common.sha256_bytes(replay_payload),
                        seat=episode.seat,
                        team_name=episode.team_name,
                        opponent_team_name=contract.kd_team_name,
                        replay_rewards=rewards,
                        rows=episode.rows,
                        raw_rows_sha256=episode.raw_rows_sha256.hexdigest(),
                        decision_keys_digest=common.digest_lines(
                            episode.decision_keys
                        ),
                    )
                )
                date_counts[dataset_date] += 1
                winner_counts[episode.team_name] += 1
        daily_audits.append(
            common.DailyAudit(
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
                source_episodes_checked=len(date_episodes),
            )
        )

    if observed_incomplete != set(expected_incomplete):
        raise RuntimeError(
            "anti-KD frozen incomplete-terminal replay set drift: "
            f"expected {sorted(expected_incomplete)!r}, "
            f"got {sorted(observed_incomplete)!r}"
        )
    selected.sort(key=lambda value: (value.dataset_date, int(value.episode_id)))
    if len(selected) != contract.expected_episodes:
        raise RuntimeError(
            f"selected episode-count drift: expected {contract.expected_episodes}, "
            f"got {len(selected)}"
        )
    if sum(value.rows for value in selected) != contract.expected_rows:
        raise RuntimeError(
            f"selected row-count drift: expected {contract.expected_rows}, "
            f"got {sum(value.rows for value in selected)}"
        )
    observed_dates = {date: date_counts[date] for date in base.dates}
    if observed_dates != dict(contract.expected_date_episodes):
        raise RuntimeError(
            "selected date episode-count drift: "
            f"expected {dict(contract.expected_date_episodes)!r}, "
            f"got {observed_dates!r}"
        )
    named = {PRIMARY_WINNER, SECONDARY_WINNER, TERTIARY_WINNER}
    observed_groups = {
        PRIMARY_WINNER: winner_counts[PRIMARY_WINNER],
        SECONDARY_WINNER: winner_counts[SECONDARY_WINNER],
        TERTIARY_WINNER: winner_counts[TERTIARY_WINNER],
        "__others__": sum(
            count for name, count in winner_counts.items() if name not in named
        ),
    }
    if observed_groups != dict(contract.expected_winner_groups):
        raise RuntimeError(
            "winner group-count drift: "
            f"expected {dict(contract.expected_winner_groups)!r}, "
            f"got {observed_groups!r}; exact={dict(winner_counts)!r}"
        )
    if any(name not in allowed_learners for name in winner_counts):
        raise RuntimeError("selected winner is outside frozen non-KD Top20")
    return tuple(selected), tuple(daily_audits), dict(sorted(winner_counts.items()))


def partition_lines(
    lines: Sequence[bytes],
    split_by_episode: Mapping[tuple[str, str], str],
    contract: AntiKDContract,
) -> tuple[
    tuple[bytes, ...],
    tuple[bytes, ...],
    dict[str, int],
    dict[str, dict[str, int]],
]:
    by_split: dict[str, list[bytes]] = {"train": [], "dev": []}
    date_rows = {
        "train": Counter(),
        "dev": Counter(),
    }
    for raw_line in lines:
        row = orjson.loads(raw_line)
        key = (str(row["dataset_date"]), str(row["episode_id"]))
        derived_split = split_by_episode.get(key)
        if derived_split not in by_split:
            raise RuntimeError(f"materialized row has no derived split: {key!r}")
        by_split[derived_split].append(raw_line)
        date_rows[derived_split][key[0]] += 1
    split_rows = {name: len(values) for name, values in by_split.items()}
    if sum(split_rows.values()) != contract.expected_rows:
        raise RuntimeError("partitioned row counts do not sum to frozen total")
    return (
        tuple(by_split["train"]),
        tuple(by_split["dev"]),
        split_rows,
        {
            split: {date: counts[date] for date in contract.base.dates}
            for split, counts in date_rows.items()
        },
    )


def audit_inputs(
    source_path: Path,
    daily_dir: Path,
    contract: AntiKDContract,
) -> AntiKDAudit:
    validate_contract(contract)
    source = common.scan_source(source_path, contract.base)
    validate_top20_manifest(source, contract)
    selected, daily, winner_counts = scan_anti_kd_replays(
        daily_dir,
        source,
        contract,
    )
    split_by_episode = assign_splits(selected, contract)
    (
        all_lines,
        content_sha256,
        decision_keys_digest,
        date_rows,
    ) = common.materialize_selected_lines(
        source_path,
        source,
        selected,
        contract.base,
    )
    train_lines, dev_lines, split_rows, split_date_rows = partition_lines(
        all_lines,
        split_by_episode,
        contract,
    )
    split_episode_counts: dict[str, int] = {}
    split_date_episode_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "dev"):
        episodes = [
            value
            for value in selected
            if split_by_episode[(value.dataset_date, value.episode_id)] == split
        ]
        split_episode_counts[split] = len(episodes)
        counts = Counter(value.dataset_date for value in episodes)
        split_date_episode_counts[split] = {
            date: counts[date] for date in contract.base.dates
        }
    return AntiKDAudit(
        source=source,
        selected=selected,
        daily=daily,
        all_lines=all_lines,
        train_lines=train_lines,
        dev_lines=dev_lines,
        selected_content_sha256=content_sha256,
        decision_keys_digest=decision_keys_digest,
        date_rows=date_rows,
        split_by_episode=split_by_episode,
        split_episode_counts=split_episode_counts,
        split_decision_rows=split_rows,
        split_date_episode_counts=split_date_episode_counts,
        split_date_decision_rows=split_date_rows,
        winner_episode_counts=winner_counts,
    )


def plan_document(contract: AntiKDContract) -> dict[str, Any]:
    base = contract.base
    daily_manifests = dict(base.daily_manifest_sha256)
    missing_replays = dict(base.daily_missing_replay_ids)
    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "data_schema_version": common.ROW_SCHEMA_VERSION,
        "source": {
            "logical_path": base.source_logical_path,
            "sha256": base.source_sha256,
            "manifest_sha256": base.source_manifest_sha256,
            "expected_train_episodes": base.expected_source_train_episodes,
            "expected_train_rows": base.expected_source_train_rows,
        },
        "daily": [
            {
                "date": date,
                "sha256": digest,
                "manifest_sha256": daily_manifests[date],
                "missing_replay_ids": list(missing_replays[date]),
                "missing_source_episode_overlap": 0,
                "incomplete_terminal_replay_ids": [
                    value.episode_id
                    for value in base.incomplete_terminal_replays
                    if value.dataset_date == date
                ],
            }
            for date, digest in base.daily_sha256
        ],
        "incomplete_terminal_replays": [
            {
                **dataclasses.asdict(value),
                "raw_rewards": list(value.raw_rewards),
                "anti_kd_target_route": False,
            }
            for value in base.incomplete_terminal_replays
        ],
        "selection": {
            "learner_deck_hash": base.learner_deck_hash,
            "opponent_deck_hash": base.opponent_deck_hash,
            "opponent_team_name": contract.kd_team_name,
            "learner_team_allowlist": list(contract.top20_names),
            "learner_team_exclusions": [contract.kd_team_name],
            "source_split": "train",
            "terminal_reward": "win",
            "expected_episodes": contract.expected_episodes,
            "expected_rows": contract.expected_rows,
            "expected_date_episodes": dict(contract.expected_date_episodes),
            "expected_winner_groups": dict(contract.expected_winner_groups),
        },
        "derived_split": {
            "algorithm": SPLIT_ALGORITHM,
            "domain_separator": contract.split_domain_separator,
            "payload": SPLIT_PAYLOAD,
            "order": SPLIT_ORDER,
            "dev_count": contract.dev_episode_count,
            "train_count": contract.train_episode_count,
        },
        "output_members": [
            TRAIN_MEMBER,
            DEV_MEMBER,
            ALLOWLIST_MEMBER,
            MANIFEST_MEMBER,
        ],
    }


def allowlist_document(
    audit: AntiKDAudit,
    contract: AntiKDContract,
) -> dict[str, Any]:
    episode_ids = [episode.episode_id for episode in audit.selected]
    if not episode_ids or len(episode_ids) != len(set(episode_ids)):
        raise RuntimeError("anti-KD allowlist episode_ids are empty or duplicated")
    return {
        "schema_version": ALLOWLIST_SCHEMA_VERSION,
        "data_schema_version": common.ROW_SCHEMA_VERSION,
        "learner_deck_hash": contract.base.learner_deck_hash,
        "opponent_deck_hash": contract.base.opponent_deck_hash,
        "opponent_team_name": contract.kd_team_name,
        "split": "train",
        "source_split": "train",
        "partition_role": "train_dev",
        "terminal_reward": "win",
        "episode_count": len(audit.selected),
        "decision_rows": len(audit.all_lines),
        "date_episode_counts": dict(contract.expected_date_episodes),
        "date_decision_rows": dict(audit.date_rows),
        "winner_episode_counts": audit.winner_episode_counts,
        "winner_group_counts": dict(contract.expected_winner_groups),
        "episode_ids": episode_ids,
        "source": {
            "logical_path": contract.base.source_logical_path,
            "sha256": contract.base.source_sha256,
            "manifest_sha256": contract.base.source_manifest_sha256,
        },
        "split_rule": {
            "algorithm": SPLIT_ALGORITHM,
            "domain_separator": contract.split_domain_separator,
            "payload": SPLIT_PAYLOAD,
            "order": SPLIT_ORDER,
            "dev_count": contract.dev_episode_count,
        },
        "train_episode_count": audit.split_episode_counts["train"],
        "dev_episode_count": audit.split_episode_counts["dev"],
        "train_decision_rows": audit.split_decision_rows["train"],
        "dev_decision_rows": audit.split_decision_rows["dev"],
        "train_date_episode_counts": audit.split_date_episode_counts["train"],
        "dev_date_episode_counts": audit.split_date_episode_counts["dev"],
        "train_date_decision_rows": audit.split_date_decision_rows["train"],
        "dev_date_decision_rows": audit.split_date_decision_rows["dev"],
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "date": episode.dataset_date,
                "decision_rows": episode.rows,
                "derived_split": audit.split_by_episode[
                    (episode.dataset_date, episode.episode_id)
                ],
                "split_score_sha256": split_score(
                    episode.dataset_date,
                    episode.episode_id,
                    contract.split_domain_separator,
                ),
                "seat": episode.seat,
                "team_name": episode.team_name,
                "opponent_team_name": episode.opponent_team_name,
                "learner_deck_hash": contract.base.learner_deck_hash,
                "opponent_deck_hash": contract.base.opponent_deck_hash,
                "replay_rewards": list(episode.replay_rewards),
                "replay_member": episode.replay_member,
                "replay_sha256": episode.replay_sha256,
                "raw_rows_sha256": episode.raw_rows_sha256,
                "decision_keys_digest": episode.decision_keys_digest,
            }
            for episode in audit.selected
        ],
    }


def validate_allowlist_document(
    payload: Any,
    contract: AntiKDContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Pure shared consumer contract for the anti-KD allowlist."""

    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
        try:
            document = orjson.loads(raw)
        except orjson.JSONDecodeError as exc:
            raise RuntimeError("anti-KD allowlist is invalid JSON") from exc
        if common.canonical_json_bytes(document) + b"\n" != raw:
            raise RuntimeError("anti-KD allowlist is not canonical JSON+LF")
    else:
        document = payload
    if not isinstance(document, dict):
        raise RuntimeError("anti-KD allowlist is not an object")
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
            contract.base.learner_deck_hash,
        ),
        "opponent_deck_hash": (
            document.get("opponent_deck_hash"),
            contract.base.opponent_deck_hash,
        ),
        "opponent_team_name": (
            document.get("opponent_team_name"),
            contract.kd_team_name,
        ),
        "source_split": (document.get("source_split"), "train"),
        "split": (document.get("split"), "train"),
        "partition_role": (document.get("partition_role"), "train_dev"),
        "terminal_reward": (document.get("terminal_reward"), "win"),
        "episode_count": (document.get("episode_count"), contract.expected_episodes),
        "decision_rows": (document.get("decision_rows"), contract.expected_rows),
        "date_episode_counts": (
            document.get("date_episode_counts"),
            dict(contract.expected_date_episodes),
        ),
        "winner_group_counts": (
            document.get("winner_group_counts"),
            dict(contract.expected_winner_groups),
        ),
        "train_episode_count": (
            document.get("train_episode_count"),
            contract.train_episode_count,
        ),
        "dev_episode_count": (
            document.get("dev_episode_count"),
            contract.dev_episode_count,
        ),
    }
    failures = [
        f"{name}={actual!r} expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    expected_rule = {
        "algorithm": SPLIT_ALGORITHM,
        "domain_separator": contract.split_domain_separator,
        "payload": SPLIT_PAYLOAD,
        "order": SPLIT_ORDER,
        "dev_count": contract.dev_episode_count,
    }
    if document.get("split_rule") != expected_rule:
        failures.append("split_rule drift")
    expected_source = {
        "logical_path": contract.base.source_logical_path,
        "sha256": contract.base.source_sha256,
        "manifest_sha256": contract.base.source_manifest_sha256,
    }
    if document.get("source") != expected_source:
        failures.append("source lineage drift")
    if failures:
        raise RuntimeError("anti-KD allowlist contract drift: " + "; ".join(failures))
    episodes = document.get("episodes")
    episode_ids = document.get("episode_ids")
    if not isinstance(episodes, list) or len(episodes) != contract.expected_episodes:
        raise RuntimeError("anti-KD episodes length drift")
    observed_ids = [str(value.get("episode_id") or "") for value in episodes]
    if (
        not isinstance(episode_ids, list)
        or episode_ids != observed_ids
        or any(not value for value in observed_ids)
        or len(observed_ids) != len(set(observed_ids))
    ):
        raise RuntimeError("anti-KD episode identity drift")
    ordering = [(str(value.get("date") or ""), int(value["episode_id"])) for value in episodes]
    if ordering != sorted(ordering):
        raise RuntimeError("anti-KD episodes are not date/episode sorted")
    if sum(int(value.get("decision_rows", -1)) for value in episodes) != contract.expected_rows:
        raise RuntimeError("anti-KD episode row sum drift")
    date_counts = Counter(str(value.get("date") or "") for value in episodes)
    if dict(date_counts) != dict(contract.expected_date_episodes):
        raise RuntimeError("anti-KD episode date-count drift")
    split_counts = Counter(str(value.get("derived_split") or "") for value in episodes)
    if split_counts != Counter(
        train=contract.train_episode_count,
        dev=contract.dev_episode_count,
    ):
        raise RuntimeError("anti-KD episode derived-split drift")
    allowed_learners = set(contract.top20_names) - {contract.kd_team_name}
    for value in episodes:
        if value.get("team_name") not in allowed_learners:
            raise RuntimeError("anti-KD learner outside non-KD Top20")
        if value.get("opponent_team_name") != contract.kd_team_name:
            raise RuntimeError("anti-KD opponent routing drift")
        expected_score = split_score(
            str(value["date"]),
            str(value["episode_id"]),
            contract.split_domain_separator,
        )
        if value.get("split_score_sha256") != expected_score:
            raise RuntimeError("anti-KD split score drift")
    return document


def manifest_document(
    audit: AntiKDAudit,
    contract: AntiKDContract,
    plan_sha256: str,
    allowlist_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "data_schema_version": common.ROW_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "learner_deck_hash": contract.base.learner_deck_hash,
        "opponent_deck_hash": contract.base.opponent_deck_hash,
        "opponent_team_name": contract.kd_team_name,
        "split": "train",
        "source_split": "train",
        "partition_role": "train_dev",
        "terminal_reward": "win",
        "episode_count": len(audit.selected),
        "decision_rows": len(audit.all_lines),
        "date_episode_counts": dict(contract.expected_date_episodes),
        "date_decision_rows": dict(audit.date_rows),
        "winner_episode_counts": audit.winner_episode_counts,
        "winner_group_counts": dict(contract.expected_winner_groups),
        "split_episodes": audit.split_episode_counts,
        "split_decisions": audit.split_decision_rows,
        "split_date_episode_counts": audit.split_date_episode_counts,
        "split_date_decision_rows": audit.split_date_decision_rows,
        "split_rule": {
            "algorithm": SPLIT_ALGORITHM,
            "domain_separator": contract.split_domain_separator,
            "payload": SPLIT_PAYLOAD,
            "order": SPLIT_ORDER,
            "dev_count": contract.dev_episode_count,
            "train_count": contract.train_episode_count,
        },
        "source": {
            "logical_path": contract.base.source_logical_path,
            "sha256": contract.base.source_sha256,
            "manifest_sha256": contract.base.source_manifest_sha256,
            "train_episodes": len(audit.source.episodes),
            "train_rows": audit.source.train_rows,
            "train_member_sha256": dict(audit.source.train_member_sha256),
        },
        "daily": [dataclasses.asdict(value) for value in audit.daily],
        "incomplete_terminal_replays": [
            {
                **dataclasses.asdict(value),
                "raw_rewards": list(value.raw_rewards),
                "anti_kd_target_route": False,
            }
            for value in contract.base.incomplete_terminal_replays
        ],
        "members": [
            {
                "member": TRAIN_MEMBER,
                "derived_split": "train",
                "episodes": audit.split_episode_counts["train"],
                "rows": len(audit.train_lines),
                "sha256": common.sha256_bytes(b"".join(audit.train_lines)),
            },
            {
                "member": DEV_MEMBER,
                "derived_split": "dev",
                "episodes": audit.split_episode_counts["dev"],
                "rows": len(audit.dev_lines),
                "sha256": common.sha256_bytes(b"".join(audit.dev_lines)),
            },
        ],
        "exact_episode_allowlist": {
            "member": ALLOWLIST_MEMBER,
            "external_filename": ALLOWLIST_MEMBER,
            "schema_version": ALLOWLIST_SCHEMA_VERSION,
            "sha256": allowlist_sha256,
            "canonical_sha256": allowlist_sha256,
            "episodes": len(audit.selected),
            "decision_rows": len(audit.all_lines),
        },
        "byte_preservation": {
            "source_rows_copied_verbatim": True,
            "source_row_split_field_preserved_as_train": True,
            "visualize_copied": False,
            "selected_content_sha256_source_order": (
                audit.selected_content_sha256
            ),
            "decision_keys_digest": audit.decision_keys_digest,
        },
        "zip": {
            "timestamp": list(common.ZIP_TIMESTAMP),
            "compression": "deflate",
            "compresslevel": common.ZIP_COMPRESSLEVEL,
            "member_order": [
                TRAIN_MEMBER,
                DEV_MEMBER,
                ALLOWLIST_MEMBER,
                MANIFEST_MEMBER,
            ],
            "double_serialization_required": True,
        },
    }


def create_archive(
    audit: AntiKDAudit,
    allowlist_payload: bytes,
    manifest: dict[str, Any],
) -> bytes:
    manifest_payload = common.canonical_json_bytes(manifest) + b"\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=common.ZIP_COMPRESSION,
        compresslevel=common.ZIP_COMPRESSLEVEL,
        allowZip64=True,
    ) as archive:
        archive.writestr(common.zip_info(TRAIN_MEMBER), b"".join(audit.train_lines))
        archive.writestr(common.zip_info(DEV_MEMBER), b"".join(audit.dev_lines))
        archive.writestr(common.zip_info(ALLOWLIST_MEMBER), allowlist_payload)
        archive.writestr(common.zip_info(MANIFEST_MEMBER), manifest_payload)
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != [
            TRAIN_MEMBER,
            DEV_MEMBER,
            ALLOWLIST_MEMBER,
            MANIFEST_MEMBER,
        ]:
            raise RuntimeError("anti-KD ZIP member order drift")
        if archive.testzip() is not None:
            raise RuntimeError("anti-KD ZIP CRC failure")
        if archive.read(TRAIN_MEMBER) != b"".join(audit.train_lines):
            raise RuntimeError("anti-KD train bytes changed")
        if archive.read(DEV_MEMBER) != b"".join(audit.dev_lines):
            raise RuntimeError("anti-KD dev bytes changed")
        if archive.read(ALLOWLIST_MEMBER) != allowlist_payload:
            raise RuntimeError("anti-KD embedded allowlist bytes changed")
        if archive.read(MANIFEST_MEMBER) != manifest_payload:
            raise RuntimeError("anti-KD manifest bytes changed")
    return payload


def _rename_directory_noreplace(staging: Path, target: Path) -> None:
    """Atomically publish a directory with Linux RENAME_NOREPLACE."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is unavailable; refusing non-atomic bundle publish")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"bundle target appeared concurrently: {target}")
        raise OSError(error_number, os.strerror(error_number), str(target))


def publish_bundle_no_clobber(
    archive_path: Path,
    allowlist_path: Path,
    archive_payload: bytes,
    allowlist_payload: bytes,
) -> None:
    if archive_path.parent != allowlist_path.parent:
        raise RuntimeError("archive and external allowlist must share a directory")
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
            (allowlist_path.name, allowlist_payload),
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
        _rename_directory_noreplace(staging, bundle_root)
        published = True
    finally:
        if not published and staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()


def audit_and_serialize(
    source: Path,
    daily_dir: Path,
    contract: AntiKDContract,
) -> tuple[AntiKDAudit, bytes, bytes, dict[str, str]]:
    audit = audit_inputs(source, daily_dir, contract)
    plan = plan_document(contract)
    plan_sha256 = common.sha256_bytes(common.canonical_json_bytes(plan))
    allowlist = allowlist_document(audit, contract)
    allowlist_payload = common.canonical_json_bytes(allowlist) + b"\n"
    validate_allowlist_document(allowlist_payload, contract)
    if common.canonical_json_bytes(orjson.loads(allowlist_payload)) + b"\n" != allowlist_payload:
        raise RuntimeError("anti-KD allowlist is not canonical JSON+LF")
    allowlist_sha256 = common.sha256_bytes(allowlist_payload)
    manifest = manifest_document(
        audit,
        contract,
        plan_sha256,
        allowlist_sha256,
    )
    first = create_archive(audit, allowlist_payload, manifest)
    second = create_archive(audit, allowlist_payload, manifest)
    if first != second:
        raise RuntimeError("anti-KD double serialization differs")
    manifest_payload = common.canonical_json_bytes(manifest) + b"\n"
    return audit, first, allowlist_payload, {
        "plan_sha256": plan_sha256,
        "archive_sha256": common.sha256_bytes(first),
        "allowlist_sha256": allowlist_sha256,
        "allowlist_canonical_sha256": allowlist_sha256,
        "manifest_sha256": common.sha256_bytes(manifest_payload),
    }


def build(
    *,
    source: Path,
    daily_dir: Path,
    archive_path: Path,
    allowlist_path: Path,
    contract: AntiKDContract = DEFAULT_CONTRACT,
    execute: bool,
    expected_plan_sha256: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_allowlist_sha256: str | None = None,
) -> dict[str, Any]:
    source = common.audit_path_components(
        source,
        "source archive",
        must_exist=True,
        kind="file",
    )
    daily_dir = common.audit_path_components(
        daily_dir,
        "daily replay directory",
        must_exist=True,
        kind="directory",
    )
    archive_path = common.audit_path_components(
        archive_path,
        "anti-KD archive output",
        must_exist=False,
        kind="output",
    )
    allowlist_path = common.audit_path_components(
        allowlist_path,
        "anti-KD allowlist output",
        must_exist=False,
        kind="output",
    )
    if archive_path == allowlist_path:
        raise RuntimeError("archive and allowlist paths must differ")
    if os.path.lexists(archive_path.parent):
        raise FileExistsError(
            f"anti-KD bundle directory must be absent: {archive_path.parent}"
        )
    if execute and not all(
        (expected_plan_sha256, expected_archive_sha256, expected_allowlist_sha256)
    ):
        raise RuntimeError(
            "execute requires reviewed plan, archive, and allowlist SHA256 gates"
        )
    if execute:
        common.require_sha256(str(expected_plan_sha256), "expected_plan_sha256")
        common.require_sha256(
            str(expected_archive_sha256),
            "expected_archive_sha256",
        )
        common.require_sha256(
            str(expected_allowlist_sha256),
            "expected_allowlist_sha256",
        )
    input_hashes_before = {
        "source": common.sha256_file(source),
        **{
            date: common.sha256_file(
                daily_dir / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
            )
            for date in contract.base.dates
        },
    }
    audit, archive_payload, allowlist_payload, metadata = audit_and_serialize(
        source,
        daily_dir,
        contract,
    )
    input_hashes_after = {
        "source": common.sha256_file(source),
        **{
            date: common.sha256_file(
                daily_dir / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
            )
            for date in contract.base.dates
        },
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("anti-KD inputs changed during audit")
    if execute:
        gates = {
            "plan_sha256": expected_plan_sha256,
            "archive_sha256": expected_archive_sha256,
            "allowlist_sha256": expected_allowlist_sha256,
        }
        failures = [
            f"{name}: expected {wanted}, got {metadata[name]}"
            for name, wanted in gates.items()
            if metadata[name] != wanted
        ]
        if failures:
            raise RuntimeError("reviewed SHA gate mismatch: " + "; ".join(failures))
        publish_bundle_no_clobber(
            archive_path,
            allowlist_path,
            archive_payload,
            allowlist_payload,
        )
        if common.sha256_file(archive_path) != metadata["archive_sha256"]:
            raise RuntimeError("published anti-KD archive SHA mismatch")
        if common.sha256_file(allowlist_path) != metadata["allowlist_sha256"]:
            raise RuntimeError("published anti-KD allowlist SHA mismatch")
        with zipfile.ZipFile(archive_path) as published:
            if published.read(ALLOWLIST_MEMBER) != allowlist_path.read_bytes():
                raise RuntimeError("published embedded/external anti-KD allowlists differ")
            if common.sha256_bytes(published.read(MANIFEST_MEMBER)) != metadata[
                "manifest_sha256"
            ]:
                raise RuntimeError("published anti-KD manifest SHA mismatch")
    return {
        "status": "built" if execute else "dry_run_passed",
        "dry_run": not execute,
        "output_written": execute,
        "archive": str(archive_path),
        "external_allowlist": str(allowlist_path),
        "outer_zip_sha256": metadata["archive_sha256"],
        "archive_sha256": metadata["archive_sha256"],
        "allowlist_sha256": metadata["allowlist_sha256"],
        "allowlist_canonical_sha256": metadata[
            "allowlist_canonical_sha256"
        ],
        "embedded_manifest_raw_sha256": metadata["manifest_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "plan_sha256": metadata["plan_sha256"],
        "episode_count": len(audit.selected),
        "decision_rows": len(audit.all_lines),
        "date_episode_counts": dict(contract.expected_date_episodes),
        "winner_episode_counts": audit.winner_episode_counts,
        "winner_group_counts": dict(contract.expected_winner_groups),
        "split_episodes": audit.split_episode_counts,
        "split_decisions": audit.split_decision_rows,
        "deterministic_rebuild_match": True,
        "rows_copied_verbatim": True,
        "training_started": False,
        "evaluation_started": False,
        "submission_started": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=common.DEFAULT_SOURCE)
    parser.add_argument("--daily-dir", type=Path, default=common.DEFAULT_DAILY_DIR)
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
        daily_dir=args.daily_dir,
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
