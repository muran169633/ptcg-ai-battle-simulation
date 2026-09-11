#!/usr/bin/env python3
"""Build a compact BC decision-frame archive from recent PTCG episode data.

The official episode format stores the action for observation ``t`` in the
same agent's entry at environment step ``t + 1``.  This script deliberately
aligns ``steps[t - 1].observation`` with ``steps[t].action``.  Pairing an
observation and action from the same row silently creates many invalid labels.

Only the player-visible ``observation`` is written.  The replay-only
``visualize`` field, which may contain hidden decks, is never copied into a
training record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO, Iterator


COMPETITION = "pokemon-tcg-ai-battle"
INDEX_DATASET = "kaggle/pokemon-tcg-ai-battle-episodes-index"
DAILY_DATASET_PREFIX = "kaggle/pokemon-tcg-ai-battle-episodes-"
DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
NUMERIC_JSON_RE = re.compile(r"(?:^|/)(\d+)\.json$")
SCHEMA_VERSION = "ptcg-bc-visible-decisions-v1"


def log(message: str) -> None:
    print(message, flush=True)


def normalize_team_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def prepare_excluded_team_names(values: list[str]) -> tuple[list[str], set[str]]:
    """Return stable display names and their normalized fail-closed keys."""

    display_by_normalized: dict[str, str] = {}
    for value in values:
        display = " ".join(
            unicodedata.normalize("NFKC", str(value or "")).split()
        )
        normalized = normalize_team_name(display)
        if not normalized:
            raise ValueError("--exclude-team-name must not be empty")
        display_by_normalized.setdefault(normalized, display)
    normalized_names = set(display_by_normalized)
    display_names = [
        display_by_normalized[key] for key in sorted(display_by_normalized)
    ]
    return display_names, normalized_names


def parse_date_from_path(path: Path) -> str | None:
    match = DATE_RE.search(str(path))
    return match.group(1) if match else None


def json_load(handle: BinaryIO) -> dict[str, Any] | None:
    try:
        payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class DailySource:
    dataset_date: str
    path: Path
    kind: str


@dataclass
class TeamFilter:
    all_teams: bool
    global_names: set[str]
    names_by_date: dict[str, set[str]]
    display_names: list[str]

    def allows(self, dataset_date: str, team_name: str) -> bool:
        if self.all_teams:
            return True
        normalized = normalize_team_name(team_name)
        if not normalized:
            return False
        if self.names_by_date:
            return normalized in self.names_by_date.get(dataset_date, set())
        return normalized in self.global_names


class DecisionArchiveWriter:
    """Write sharded JSONL members directly into a ZIP64 archive."""

    def __init__(
        self,
        output_path: Path,
        frames_per_shard: int,
        overwrite: bool,
    ) -> None:
        self.output_path = output_path.resolve()
        self.partial_path = self.output_path.with_suffix(
            self.output_path.suffix + ".partial"
        )
        self.frames_per_shard = frames_per_shard
        self.overwrite = overwrite
        self.archive: zipfile.ZipFile | None = None
        self.handles: dict[str, BinaryIO] = {}
        self.rows_in_shard: Counter[str] = Counter()
        self.shard_index: Counter[str] = Counter()
        self.total_rows: Counter[str] = Counter()

    def __enter__(self) -> "DecisionArchiveWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        for candidate in (self.output_path, self.partial_path):
            if candidate.exists():
                if not self.overwrite:
                    raise FileExistsError(
                        f"{candidate} already exists; pass --overwrite to replace it"
                    )
                candidate.unlink()
        self.archive = zipfile.ZipFile(
            self.partial_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        )
        return self

    def _open_shard(self, split: str) -> BinaryIO:
        assert self.archive is not None
        if self.handles:
            raise RuntimeError("A ZIP member write handle is already open")
        member = f"{split}/part-{self.shard_index[split]:05d}.jsonl"
        handle = self.archive.open(member, mode="w", force_zip64=True)
        self.handles[split] = handle
        self.rows_in_shard[split] = 0
        self.shard_index[split] += 1
        return handle

    def add(self, split: str, row: dict[str, Any]) -> None:
        if (
            split not in self.handles
            or self.rows_in_shard[split] >= self.frames_per_shard
        ):
            # zipfile permits only one open member write handle at a time.
            # Close the current member before either rotating a full shard or
            # switching between train/valid/test.
            for previous in self.handles.values():
                previous.close()
            self.handles.clear()
            self._open_shard(split)
        self.handles[split].write(json_bytes(row))
        self.rows_in_shard[split] += 1
        self.total_rows[split] += 1

    def finish(self, manifest: dict[str, Any]) -> None:
        assert self.archive is not None
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        self.archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        self.archive.close()
        self.archive = None
        os.replace(self.partial_path, self.output_path)

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self.handles.clear()
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        if exc_type is not None and self.partial_path.exists():
            log(f"Partial archive retained for diagnosis: {self.partial_path}")


def run_command(command: list[str], cwd: Path | None = None) -> None:
    rendered = " ".join(command)
    log(f"$ {rendered}")
    subprocess.run(command, cwd=cwd, check=True)


def read_index_manifest(index_zip: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(index_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith("manifest.csv")]
        if not names:
            raise RuntimeError(f"No manifest.csv found in {index_zip}")
        with archive.open(names[0]) as raw:
            text = (line.decode("utf-8-sig") for line in raw)
            return list(csv.DictReader(text))


def download_recent_sources(
    cache_dir: Path,
    days: int,
    date_end: str | None,
    refresh: bool,
) -> list[DailySource]:
    if shutil.which("kaggle") is None:
        raise RuntimeError("Kaggle CLI is required for --download")

    index_dir = cache_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        INDEX_DATASET,
        "-p",
        str(index_dir),
        "-q",
    ]
    if refresh:
        command.append("-o")
    run_command(command)

    index_zips = sorted(index_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not index_zips:
        raise RuntimeError(f"Index download produced no ZIP under {index_dir}")
    rows = read_index_manifest(index_zips[-1])
    if date_end:
        rows = [row for row in rows if row.get("date", "") <= date_end]
    rows = sorted(rows, key=lambda row: row.get("date", ""))[-days:]
    if len(rows) < days:
        raise RuntimeError(f"Index contains only {len(rows)} eligible days, need {days}")

    daily_dir = cache_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    result: list[DailySource] = []
    for row in rows:
        dataset_date = row["date"]
        slug = row.get("daily_dataset_slug") or (
            f"pokemon-tcg-ai-battle-episodes-{dataset_date}"
        )
        expected = daily_dir / f"{slug}.zip"
        if refresh or not expected.is_file():
            command = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                f"kaggle/{slug}",
                "-p",
                str(daily_dir),
                "-q",
            ]
            if refresh:
                command.append("-o")
            run_command(command)
        if not expected.is_file():
            matches = sorted(daily_dir.glob(f"*{dataset_date}*.zip"))
            if not matches:
                raise RuntimeError(f"Daily download for {dataset_date} produced no ZIP")
            expected = matches[-1]
        result.append(DailySource(dataset_date, expected, "zip"))
    return result


def discover_sources(
    input_roots: list[Path],
    explicit_inputs: list[Path],
    days: int,
    date_end: str | None,
) -> list[DailySource]:
    candidates: dict[tuple[str, str], DailySource] = {}

    def add_candidate(path: Path) -> None:
        dataset_date = parse_date_from_path(path)
        if not dataset_date:
            if len(explicit_inputs) == 1 and date_end:
                dataset_date = date_end
            else:
                return
        if date_end and dataset_date > date_end:
            return
        if path.is_file() and path.suffix.lower() == ".zip":
            kind = "zip"
        elif path.is_file() and path.suffix.lower() == ".json":
            kind = "json"
        elif path.is_dir():
            if not any(
                child.is_file() and child.suffix.lower() == ".json"
                for child in path.iterdir()
            ):
                return
            kind = "directory"
        else:
            return
        candidates[(dataset_date, str(path.resolve()))] = DailySource(
            dataset_date, path.resolve(), kind
        )

    for path in explicit_inputs:
        add_candidate(path)
    for root in input_roots:
        if not root.exists():
            continue
        if root.is_file():
            add_candidate(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".zip":
                if "episodes" in path.name.lower():
                    add_candidate(path)
            elif path.is_dir() and DATE_RE.search(path.name):
                add_candidate(path)

    by_date: dict[str, DailySource] = {}
    for source in sorted(
        candidates.values(),
        key=lambda item: (item.dataset_date, item.kind != "zip", str(item.path)),
    ):
        by_date.setdefault(source.dataset_date, source)
    selected_dates = sorted(by_date)[-days:]
    return [by_date[dataset_date] for dataset_date in selected_dates]


def parse_team_file(path: Path) -> TeamFilter:
    text = path.read_text(encoding="utf-8-sig")
    global_names: set[str] = set()
    names_by_date: dict[str, set[str]] = defaultdict(set)
    display_names: list[str] = []

    first_line = text.splitlines()[0] if text.splitlines() else ""
    if "," not in first_line:
        for line in text.splitlines():
            display = line.strip()
            if display:
                global_names.add(normalize_team_name(display))
                display_names.append(display)
        return TeamFilter(False, global_names, {}, display_names)

    reader = csv.DictReader(text.splitlines())
    fields = {field.casefold(): field for field in (reader.fieldnames or [])}
    team_field = next(
        (
            fields[key]
            for key in ("team_name", "teamname", "team", "name")
            if key in fields
        ),
        None,
    )
    date_field = fields.get("date")
    if not team_field:
        raise ValueError(
            f"{path} must contain team_name/TeamName/team/name, or one name per line"
        )
    for row in reader:
        display = str(row.get(team_field, "")).strip()
        if not display:
            continue
        normalized = normalize_team_name(display)
        display_names.append(display)
        if date_field and row.get(date_field):
            names_by_date[str(row[date_field]).strip()].add(normalized)
        else:
            global_names.add(normalized)
    return TeamFilter(False, global_names, dict(names_by_date), display_names)


def fetch_current_top_teams(cache_dir: Path, top_k: int) -> TeamFilter:
    leaderboard_dir = cache_dir / "leaderboard"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "kaggle",
            "competitions",
            "leaderboard",
            COMPETITION,
            "-d",
            "-p",
            str(leaderboard_dir),
            "-q",
        ]
    )
    archives = sorted(
        leaderboard_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime
    )
    if not archives:
        raise RuntimeError("Leaderboard download produced no ZIP")
    with zipfile.ZipFile(archives[-1]) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_names:
            raise RuntimeError("Leaderboard ZIP contains no CSV")
        with archive.open(csv_names[0]) as raw:
            reader = csv.DictReader(
                line.decode("utf-8-sig") for line in raw
            )
            rows = list(reader)
    rows.sort(key=lambda row: int(row.get("Rank", 10**9)))
    selected = rows[:top_k]
    names = [str(row.get("TeamName", "")).strip() for row in selected]
    names = [name for name in names if name]
    if not names:
        raise RuntimeError("No team names parsed from current leaderboard")
    log(f"Current Top {len(names)} filter: {', '.join(names)}")
    return TeamFilter(
        False,
        {normalize_team_name(name) for name in names},
        {},
        names,
    )


def iter_source_episodes(
    source: DailySource,
) -> Iterator[tuple[str, dict[str, Any] | None]]:
    if source.kind == "zip":
        with zipfile.ZipFile(source.path) as archive:
            members = sorted(
                (
                    info
                    for info in archive.infolist()
                    if not info.is_dir() and NUMERIC_JSON_RE.search(info.filename)
                ),
                key=lambda info: info.filename,
            )
            for info in members:
                fallback_id = Path(info.filename).stem
                with archive.open(info) as handle:
                    yield fallback_id, json_load(handle)
        return
    if source.kind == "directory":
        for path in sorted(source.path.rglob("*.json")):
            if not path.stem.isdigit():
                continue
            with path.open("r", encoding="utf-8") as handle:
                yield path.stem, json_load(handle)
        return
    with source.path.open("r", encoding="utf-8") as handle:
        yield source.path.stem, json_load(handle)


def episode_names(episode: dict[str, Any]) -> list[str]:
    info = episode.get("info")
    info = info if isinstance(info, dict) else {}
    team_names = info.get("TeamNames")
    fallback_names: list[str] | None = None
    if isinstance(team_names, list) and len(team_names) >= 2:
        fallback_names = [str(value or "").strip() for value in team_names[:2]]
        if all(fallback_names):
            return fallback_names
    agents = info.get("Agents")
    if isinstance(agents, list) and len(agents) >= 2:
        agent_names = [
            str(agent.get("Name", "")).strip() if isinstance(agent, dict) else ""
            for agent in agents[:2]
        ]
        if all(agent_names):
            return agent_names
        if fallback_names is None:
            fallback_names = agent_names
    return fallback_names or ["", ""]


def final_rewards(episode: dict[str, Any]) -> list[float]:
    rewards = episode.get("rewards")
    if isinstance(rewards, list) and len(rewards) >= 2:
        result: list[float] = []
        for value in rewards[:2]:
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                result.append(0.0)
        return result
    return [0.0, 0.0]


def episode_identifiers(
    episode: dict[str, Any], fallback_id: str
) -> tuple[str, str | None]:
    info = episode.get("info")
    info = info if isinstance(info, dict) else {}
    numeric_id = str(info.get("EpisodeId") or fallback_id)
    episode_uuid = episode.get("id")
    return numeric_id, str(episode_uuid) if episode_uuid else None


def extract_deck_hashes(episode: dict[str, Any]) -> list[str | None]:
    steps = episode.get("steps")
    if not isinstance(steps, list):
        return [None, None]
    for frame in steps[:3]:
        if not isinstance(frame, list):
            continue
        for entry in frame[:2]:
            if not isinstance(entry, dict):
                continue
            visualize = entry.get("visualize")
            if not isinstance(visualize, list):
                continue
            for item in visualize:
                action = item.get("action") if isinstance(item, dict) else None
                if not isinstance(action, list) or len(action) < 2:
                    continue
                if not all(isinstance(deck, list) and len(deck) == 60 for deck in action[:2]):
                    continue
                hashes: list[str | None] = []
                for deck in action[:2]:
                    try:
                        canonical = ",".join(str(int(card)) for card in sorted(deck))
                    except (TypeError, ValueError):
                        hashes.append(None)
                    else:
                        hashes.append(hashlib.sha256(canonical.encode()).hexdigest())
                return hashes
    return [None, None]


def valid_action(select: dict[str, Any], action: Any) -> bool:
    if not isinstance(action, list):
        return False
    try:
        min_count = int(select.get("minCount", 0) or 0)
        max_count = int(select.get("maxCount", 0) or 0)
    except (TypeError, ValueError):
        return False
    options = select.get("option")
    if not isinstance(options, list) or not options:
        return False
    if not min_count <= len(action) <= max_count:
        return False
    if len(set(action)) != len(action):
        return False
    return all(
        isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(options)
        for index in action
    )


def sample_weight(reward: float, progress: float) -> float:
    if reward > 0:
        return 1.0
    if reward == 0:
        return 0.85
    return 0.75 if progress < 0.60 else 0.25


def split_for_episode(
    episode_id: str,
    dataset_date: str,
    selected_dates: list[str],
    split_mode: str,
    split_seed: int,
    valid_days: int = 1,
    test_days: int = 1,
    hash_train_fraction: float = 0.80,
    hash_valid_fraction: float = 0.10,
) -> str:
    if split_mode == "all_train":
        return "train"
    if split_mode == "time":
        split_policy = time_split_policy(selected_dates, valid_days, test_days)
        if dataset_date in split_policy["test_dates"]:
            return "test"
        if dataset_date in split_policy["valid_dates"]:
            return "valid"
        return "train"
    if split_mode == "daily_hash":
        hash_key = f"{split_seed}:{dataset_date}:{episode_id}"
    else:
        # Preserve the historical split exactly for old manifests/reruns.
        hash_key = f"{split_seed}:{episode_id}"
    digest = hashlib.sha256(hash_key.encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < hash_train_fraction:
        return "train"
    if value < hash_train_fraction + hash_valid_fraction:
        return "valid"
    return "test"


def validate_hash_split_fractions(
    train_fraction: float,
    valid_fraction: float,
) -> None:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--hash-train-fraction must be between 0 and 1")
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError("--hash-valid-fraction must be between 0 and 1")
    if train_fraction + valid_fraction > 1.0:
        raise ValueError(
            "--hash-train-fraction + --hash-valid-fraction must be <= 1"
        )


def time_split_policy(
    selected_dates: list[str], valid_days: int, test_days: int
) -> dict[str, Any]:
    if valid_days < 1:
        raise ValueError("--valid-days must be positive")
    if test_days < 1:
        raise ValueError("--test-days must be positive")
    holdout_days = valid_days + test_days
    if len(selected_dates) <= holdout_days:
        raise ValueError(
            "Time split needs at least "
            f"{holdout_days + 1} distinct dates for {valid_days} valid and "
            f"{test_days} test day(s)"
        )
    valid_end = len(selected_dates) - test_days
    valid_start = valid_end - valid_days
    return {
        "mode": "time",
        "train_dates": selected_dates[:valid_start],
        "valid_dates": selected_dates[valid_start:valid_end],
        "test_dates": selected_dates[valid_end:],
        "valid_days": valid_days,
        "test_days": test_days,
    }


def iter_episode_decisions(
    episode: dict[str, Any],
    fallback_id: str,
    dataset_date: str,
    team_filter: TeamFilter,
    split: str,
    stats: Counter[str],
    team_counts: Counter[str],
    context_counts: Counter[str],
    excluded_team_names: set[str] | None = None,
    exclusion_counts: Counter[str] | None = None,
) -> Iterator[dict[str, Any]]:
    if excluded_team_names is None:
        excluded_team_names = set()
    if exclusion_counts is None:
        exclusion_counts = Counter()
    names = episode_names(episode)
    if excluded_team_names:
        unresolved_seats = sum(not normalize_team_name(name) for name in names)
        if unresolved_seats:
            stats["episodes_excluded_by_unresolved_team_name"] += 1
            stats["unresolved_team_name_seat_hits"] += unresolved_seats
            return
    excluded_hits = [
        normalized
        for name in names
        if (normalized := normalize_team_name(name)) in excluded_team_names
    ]
    if excluded_hits:
        stats["episodes_excluded_by_team_name"] += 1
        stats["excluded_team_name_seat_hits"] += len(excluded_hits)
        exclusion_counts.update(excluded_hits)
        return
    steps = episode.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        stats["episodes_without_steps"] += 1
        return
    rewards = final_rewards(episode)
    numeric_id, episode_uuid = episode_identifiers(episode, fallback_id)
    deck_hashes = extract_deck_hashes(episode)
    total_transitions = len(steps) - 1

    selected_seats = [
        seat
        for seat in range(2)
        if seat < len(names) and team_filter.allows(dataset_date, names[seat])
    ]
    if not selected_seats:
        stats["episodes_filtered_by_team"] += 1
        return

    emitted = 0
    for action_step_index in range(1, len(steps)):
        previous_frame = steps[action_step_index - 1]
        current_frame = steps[action_step_index]
        if not isinstance(previous_frame, list) or not isinstance(current_frame, list):
            continue
        for seat in selected_seats:
            if seat >= len(previous_frame) or seat >= len(current_frame):
                continue
            previous_entry = previous_frame[seat]
            current_entry = current_frame[seat]
            if not isinstance(previous_entry, dict) or not isinstance(current_entry, dict):
                continue
            observation = previous_entry.get("observation")
            if not isinstance(observation, dict):
                continue
            select = observation.get("select")
            if not isinstance(select, dict) or not select.get("option"):
                continue
            if str(previous_entry.get("status", "")).upper() != "ACTIVE":
                stats["inactive_frames_skipped"] += 1
                continue
            action = current_entry.get("action")
            if not valid_action(select, action):
                stats["invalid_active_decisions"] += 1
                continue
            if "visualize" in observation:
                observation = dict(observation)
                observation.pop("visualize", None)
                stats["stripped_visualize_from_observation"] += 1

            reward = rewards[seat] if seat < len(rewards) else 0.0
            progress = action_step_index / max(1, total_transitions)
            context = str(select.get("context", "unknown"))
            team_name = names[seat] if seat < len(names) else ""
            opponent_name = names[1 - seat] if len(names) >= 2 else ""
            options = select.get("option") or []
            record = {
                "schema_version": SCHEMA_VERSION,
                "episode_id": numeric_id,
                "episode_uuid": episode_uuid,
                "dataset_date": dataset_date,
                "split": split,
                "observation_step_index": action_step_index - 1,
                "action_step_index": action_step_index,
                "seat": seat,
                "team_name": team_name,
                "opponent_team_name": opponent_name,
                "deck_hash": deck_hashes[seat] if seat < len(deck_hashes) else None,
                "terminal_reward": reward,
                "sample_weight": sample_weight(reward, progress),
                "select_context": context,
                "select_type": select.get("type"),
                "min_count": int(select.get("minCount", 0) or 0),
                "max_count": int(select.get("maxCount", 0) or 0),
                "option_count": len(options),
                "action": list(action),
                "no_action": len(action) == 0,
                "observation": observation,
            }
            for name_field in ("team_name", "opponent_team_name"):
                if normalize_team_name(record[name_field]) in excluded_team_names:
                    raise RuntimeError(
                        "Excluded team name reached an output row: "
                        f"{record[name_field]!r} in {numeric_id}"
                    )
            emitted += 1
            team_counts[team_name] += 1
            context_counts[context] += 1
            yield record
    if emitted:
        stats["episodes_kept"] += 1
    else:
        stats["episodes_without_legal_decisions"] += 1


def verify_archive(
    path: Path, excluded_team_names: list[str] | None = None
) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"ZIP integrity failure at {bad_member}")
        if "manifest.json" not in archive.namelist():
            raise RuntimeError("Archive has no manifest.json")
        manifest = json.loads(archive.read("manifest.json"))
        exclusion_manifest = manifest.get("team_exclusions", {})
        if not isinstance(exclusion_manifest, dict):
            raise RuntimeError("Manifest team_exclusions must be an object")
        raw_manifest_names = exclusion_manifest.get("normalized_names", [])
        if not isinstance(raw_manifest_names, list) or not all(
            isinstance(value, str) and value for value in raw_manifest_names
        ):
            raise RuntimeError(
                "Manifest team_exclusions.normalized_names must be a string list"
            )
        if any(
            normalize_team_name(value) != value for value in raw_manifest_names
        ) or len(set(raw_manifest_names)) != len(raw_manifest_names):
            raise RuntimeError(
                "Manifest team_exclusions.normalized_names must contain "
                "unique normalized names"
            )
        manifest_excluded = set(raw_manifest_names)
        if manifest_excluded:
            raw_hits = exclusion_manifest.get("hits_by_normalized_name")
            if not isinstance(raw_hits, dict) or set(raw_hits) != manifest_excluded:
                raise RuntimeError(
                    "Manifest exclusion hit keys must exactly match normalized names"
                )
            if not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in raw_hits.values()
            ):
                raise RuntimeError(
                    "Manifest exclusion hit counts must be non-negative integers"
                )
            count_fields = (
                "episode_hits",
                "seat_hits",
                "unresolved_episode_hits",
                "unresolved_seat_hits",
            )
            counts = {key: exclusion_manifest.get(key) for key in count_fields}
            if not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in counts.values()
            ):
                raise RuntimeError(
                    "Manifest exclusion totals must be non-negative integers"
                )
            if counts["seat_hits"] != sum(raw_hits.values()):
                raise RuntimeError(
                    "Manifest exclusion seat_hits does not match per-name hits"
                )
            if counts["episode_hits"] > counts["seat_hits"]:
                raise RuntimeError(
                    "Manifest exclusion episode_hits exceeds seat_hits"
                )
            if counts["unresolved_episode_hits"] > counts["unresolved_seat_hits"]:
                raise RuntimeError(
                    "Manifest unresolved episode hits exceed unresolved seat hits"
                )
        if excluded_team_names is not None:
            _, expected_excluded = prepare_excluded_team_names(
                excluded_team_names
            )
            if manifest_excluded != expected_excluded:
                raise RuntimeError(
                    "Manifest excluded-team names do not match the requested "
                    f"names: manifest={sorted(manifest_excluded)!r}, "
                    f"requested={sorted(expected_excluded)!r}"
                )
        decision_members = [
            name for name in archive.namelist() if name.endswith(".jsonl")
        ]
        if not decision_members:
            raise RuntimeError("Archive has no decision JSONL shards")
        with archive.open(decision_members[0]) as handle:
            first_line = handle.readline()
        first = json.loads(first_line)
        if "visualize" in first or "visualize" in first.get("observation", {}):
            raise RuntimeError("Hidden replay visualize field leaked into BC record")
        required = {"observation", "action", "split", "episode_id", "team_name"}
        missing = required - set(first)
        if missing:
            raise RuntimeError(f"First record is missing required fields: {sorted(missing)}")
        if manifest_excluded:
            for member in decision_members:
                with archive.open(member) as handle:
                    for line_number, raw_line in enumerate(handle, 1):
                        if not raw_line.strip():
                            continue
                        try:
                            row = json.loads(raw_line)
                        except (json.JSONDecodeError, UnicodeDecodeError) as error:
                            raise RuntimeError(
                                f"Invalid decision JSON at {member}:{line_number}"
                            ) from error
                        if not isinstance(row, dict):
                            raise RuntimeError(
                                f"Decision row is not an object at "
                                f"{member}:{line_number}"
                            )
                        for field in ("team_name", "opponent_team_name"):
                            raw_name = row.get(field)
                            if not isinstance(raw_name, str) or not normalize_team_name(
                                raw_name
                            ):
                                raise RuntimeError(
                                    "Excluded-name verification requires non-empty "
                                    f"{field} at {member}:{line_number}"
                                )
                            normalized = normalize_team_name(raw_name)
                            if normalized in manifest_excluded:
                                raise RuntimeError(
                                    "Excluded team name found after build at "
                                    f"{member}:{line_number} field={field}: "
                                    f"{row.get(field)!r}"
                                )
        return manifest


def build_archive(
    sources: list[DailySource],
    output: Path,
    team_filter: TeamFilter,
    frames_per_shard: int,
    split_mode: str,
    split_seed: int,
    overwrite: bool,
    max_episodes: int | None,
    valid_days: int = 1,
    test_days: int = 1,
    excluded_team_names: list[str] | None = None,
    hash_train_fraction: float = 0.80,
    hash_valid_fraction: float = 0.10,
) -> dict[str, Any]:
    selected_dates = sorted({source.dataset_date for source in sources})
    time_policy = None
    if split_mode == "time":
        time_policy = time_split_policy(selected_dates, valid_days, test_days)
    elif split_mode != "all_train":
        validate_hash_split_fractions(
            hash_train_fraction,
            hash_valid_fraction,
        )
    excluded_display_names, excluded_normalized_names = (
        prepare_excluded_team_names(excluded_team_names or [])
    )

    stats: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    team_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    seen_episode_ids: set[str] = set()

    with DecisionArchiveWriter(output, frames_per_shard, overwrite) as writer:
        stop = False
        for source_index, source in enumerate(sources, 1):
            source_scanned = source_kept = source_frames = 0
            log(
                f"[{source_index}/{len(sources)}] {source.dataset_date} "
                f"{source.kind}: {source.path}"
            )
            for fallback_id, episode in iter_source_episodes(source):
                if max_episodes is not None and stats["episodes_scanned"] >= max_episodes:
                    stop = True
                    break
                stats["episodes_scanned"] += 1
                source_scanned += 1
                if episode is None:
                    stats["invalid_json_files"] += 1
                    continue
                episode_id, _ = episode_identifiers(episode, fallback_id)
                dedupe_key = f"{source.dataset_date}:{episode_id}"
                if dedupe_key in seen_episode_ids:
                    stats["duplicate_episodes"] += 1
                    continue
                seen_episode_ids.add(dedupe_key)
                split = split_for_episode(
                    episode_id,
                    source.dataset_date,
                    selected_dates,
                    split_mode,
                    split_seed,
                    valid_days,
                    test_days,
                    hash_train_fraction,
                    hash_valid_fraction,
                )
                before = stats["episodes_kept"]
                for record in iter_episode_decisions(
                    episode,
                    fallback_id,
                    source.dataset_date,
                    team_filter,
                    split,
                    stats,
                    team_counts,
                    context_counts,
                    excluded_normalized_names,
                    exclusion_counts,
                ):
                    writer.add(split, record)
                    split_counts[split] += 1
                    stats["decisions"] += 1
                    source_frames += 1
                if stats["episodes_kept"] > before:
                    source_kept += 1
                if source_scanned % 250 == 0:
                    log(
                        f"  scanned={source_scanned} kept={source_kept} "
                        f"decisions={source_frames}"
                    )
            log(
                f"  done scanned={source_scanned} kept={source_kept} "
                f"decisions={source_frames}"
            )
            if stop:
                break

        if stats["decisions"] == 0:
            raise RuntimeError(
                "No BC decisions were extracted. Check team names, dates and replay sources."
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "competition": COMPETITION,
            "dates": selected_dates,
            "split_policy": (
                time_policy
                if split_mode == "time"
                else {"mode": "all_train"}
                if split_mode == "all_train"
                else {
                    "mode": (
                        "daily_episode_hash"
                        if split_mode == "daily_hash"
                        else "episode_hash"
                    ),
                    "seed": split_seed,
                    "hash_key": (
                        "seed:dataset_date:episode_id"
                        if split_mode == "daily_hash"
                        else "seed:episode_id"
                    ),
                    "fractions": {
                        "train": hash_train_fraction,
                        "valid": hash_valid_fraction,
                        "test": 1.0
                        - hash_train_fraction
                        - hash_valid_fraction,
                    },
                }
            ),
            "team_filter": {
                "all_teams": team_filter.all_teams,
                "global_team_count": len(team_filter.global_names),
                "dated_team_counts": {
                    key: len(value)
                    for key, value in sorted(team_filter.names_by_date.items())
                },
                "display_names": sorted(set(team_filter.display_names)),
            },
            "team_exclusions": {
                "display_names": excluded_display_names,
                "normalized_names": sorted(excluded_normalized_names),
                "scope": "drop_entire_episode_if_either_seat_matches",
                "episode_hits": stats["episodes_excluded_by_team_name"],
                "seat_hits": stats["excluded_team_name_seat_hits"],
                "unresolved_episode_hits": stats[
                    "episodes_excluded_by_unresolved_team_name"
                ],
                "unresolved_seat_hits": stats[
                    "unresolved_team_name_seat_hits"
                ],
                "hits_by_normalized_name": {
                    name: exclusion_counts[name]
                    for name in sorted(excluded_normalized_names)
                },
            },
            "label_alignment": "steps[t-1].observation -> steps[t].action",
            "hidden_information_policy": (
                "Only agent-visible observation is saved; replay visualize is excluded."
            ),
            "loss_trajectory_weights": {
                "win": 1.0,
                "draw": 0.85,
                "loss_before_60_percent": 0.75,
                "loss_after_60_percent": 0.25,
            },
            "target_bc_exact_accuracy": 0.75,
            "stats": dict(stats),
            "split_decisions": dict(split_counts),
            "team_decisions": dict(team_counts.most_common()),
            "context_decisions": dict(context_counts.most_common()),
            "shards": dict(writer.shard_index),
            "sources": [
                {
                    "date": source.dataset_date,
                    "kind": source.kind,
                    "path": str(source.path),
                }
                for source in sources
            ],
        }
        writer.finish(manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream recent PTCG episode ZIPs, align visible observations with "
            "their next-step actions, and produce a sharded BC ZIP."
        )
    )
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        default=[],
        help="Root containing daily ZIPs or Kaggle-mounted daily directories",
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Explicit daily ZIP, daily directory, or one JSON (repeatable)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the latest daily ZIPs with the Kaggle CLI",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/episodes_cache"),
        help="Download and leaderboard cache directory",
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--date-end",
        help="Latest eligible dataset date (YYYY-MM-DD); defaults to latest available",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force-refresh Kaggle index and daily downloads",
    )
    team_group = parser.add_mutually_exclusive_group()
    team_group.add_argument(
        "--teams-file",
        type=Path,
        help="CSV with team_name and optional date, or one team name per line",
    )
    team_group.add_argument(
        "--all-teams",
        action="store_true",
        help="Keep both seats from every valid episode instead of expert filtering",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Use current leaderboard Top-K when no --teams-file/--all-teams (default: 20)",
    )
    parser.add_argument(
        "--split-mode",
        choices=("time", "hash", "daily_hash", "all_train"),
        default="time",
        help=(
            "time: use --valid-days/--test-days chronological holdouts; "
            "hash: configurable split by episode; daily_hash: configurable "
            "split by date+episode so every date contributes to every split; "
            "all_train: retain every row for later deterministic repartition"
        ),
    )
    parser.add_argument(
        "--valid-days",
        type=int,
        default=1,
        help="Number of dates immediately before test to use for validation",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=1,
        help="Number of most recent dates to use for test",
    )
    parser.add_argument("--split-seed", type=int, default=20260723)
    parser.add_argument(
        "--hash-train-fraction",
        type=float,
        default=0.80,
        help="Train fraction for hash/daily_hash splits (default: 0.80)",
    )
    parser.add_argument(
        "--hash-valid-fraction",
        type=float,
        default=0.10,
        help="Validation fraction for hash/daily_hash splits (default: 0.10)",
    )
    parser.add_argument(
        "--exclude-team-name",
        action="append",
        default=[],
        help=(
            "Drop any episode containing this normalized team name "
            "(repeatable)"
        ),
    )
    parser.add_argument("--frames-per-shard", type=int, default=50_000)
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Global smoke-test cap; omit for a full run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bc_recent7_top20.zip"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.days < 1:
        raise ValueError("--days must be positive")
    if args.frames_per_shard < 1:
        raise ValueError("--frames-per-shard must be positive")
    if args.valid_days < 1:
        raise ValueError("--valid-days must be positive")
    if args.test_days < 1:
        raise ValueError("--test-days must be positive")
    if args.split_mode in {"hash", "daily_hash"}:
        validate_hash_split_fractions(
            args.hash_train_fraction,
            args.hash_valid_fraction,
        )
    prepare_excluded_team_names(args.exclude_team_name)
    if (
        args.split_mode == "time"
        and args.days <= args.valid_days + args.test_days
    ):
        raise ValueError(
            "--days must leave at least one train date after valid/test holdouts"
        )
    if args.date_end:
        date.fromisoformat(args.date_end)
    if not args.download and not args.input_root and not args.input:
        raise ValueError("Provide --download, --input-root, or --input")

    if args.download:
        sources = download_recent_sources(
            args.cache_dir, args.days, args.date_end, args.refresh
        )
    else:
        sources = discover_sources(
            args.input_root, args.input, args.days, args.date_end
        )
    if not sources:
        raise RuntimeError("No dated episode sources were discovered")
    if len(sources) < args.days:
        log(f"Warning: found {len(sources)} day(s), requested {args.days}")

    if args.all_teams:
        team_filter = TeamFilter(True, set(), {}, [])
    elif args.teams_file:
        team_filter = parse_team_file(args.teams_file)
    else:
        if args.top_k < 1:
            raise ValueError("--top-k must be positive unless --all-teams is used")
        team_filter = fetch_current_top_teams(args.cache_dir, args.top_k)

    manifest = build_archive(
        sources=sources,
        output=args.output,
        team_filter=team_filter,
        frames_per_shard=args.frames_per_shard,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        valid_days=args.valid_days,
        test_days=args.test_days,
        excluded_team_names=args.exclude_team_name,
        hash_train_fraction=args.hash_train_fraction,
        hash_valid_fraction=args.hash_valid_fraction,
    )
    verified = verify_archive(
        args.output.resolve(), excluded_team_names=args.exclude_team_name
    )
    size_mib = args.output.resolve().stat().st_size / 1024 / 1024
    log(
        f"Done: {args.output.resolve()} ({size_mib:.1f} MiB), "
        f"decisions={verified['stats']['decisions']}, "
        f"splits={verified['split_decisions']}"
    )
    if manifest != verified:
        raise RuntimeError("Manifest changed after ZIP verification")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted")
        raise SystemExit(130)
    except Exception as error:
        log(f"ERROR: {type(error).__name__}: {error}")
        raise SystemExit(1)
