#!/usr/bin/env python3
"""Build exact, seat-scoped BC archives for active public submissions.

The input manifest identifies each policy by Kaggle submission ID, team name,
and deck hash.  Episode metadata (embedded in the manifest, supplied through a
separate index, or listed read-only through the Kaggle API) binds a submission
to an episode seat.  Local replay ZIPs then provide the visible observations
and actions for that seat only.

Replay ``visualize`` data is used solely to verify the declared deck hash and
materialize a separate exact 60-card CSV for local simulation.  It is never
copied to BC records, and this tool never downloads or uses another entrant's
source code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

import orjson


COMPETITION = "pokemon-tcg-ai-battle"
INPUT_SCHEMA = "ptcg-active-submission-manifest-v1"
OUTPUT_SCHEMA = "ptcg-gold-policy-clones-v1"
ARCHIVE_SCHEMA = "ptcg-gold-policy-visible-decisions-v1"
ROW_SCHEMA = "ptcg-bc-visible-decisions-v1"
POLICY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
EPISODE_MEMBER_RES = (
    re.compile(r"(?:^|/)(\d+)\.json$"),
    re.compile(r"(?:^|/)episode-(\d+)-replay\.json$"),
)
DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def log(message: str) -> None:
    print(message, flush=True)


def normalize_team_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return orjson.dumps(value) + b"\n"


def parse_dataset_date(path: Path, fallback: str | None = None) -> str:
    match = DATE_RE.search(str(path))
    if match:
        return match.group(1)
    return fallback or "unknown"


def episode_id_from_member(name: str) -> str | None:
    for pattern in EPISODE_MEMBER_RES:
        match = pattern.search(name)
        if match:
            return match.group(1)
    return None


def get_any(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not bool")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer: {value!r}") from exc


def object_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    raise TypeError(f"Cannot serialize episode metadata object {type(value)!r}")


def canonical_episode_metadata(value: Any) -> dict[str, Any]:
    raw = object_to_dict(value)
    episode_id = as_int(
        get_any(raw, "episode_id", "episodeId", "id"),
        "episode metadata id",
    )
    agents_raw = raw.get("agents")
    agents: list[dict[str, Any]] = []
    if isinstance(agents_raw, list):
        for position, agent_value in enumerate(agents_raw):
            agent = object_to_dict(agent_value)
            submission = get_any(agent, "submission_id", "submissionId")
            if submission is None:
                continue
            index = get_any(
                agent,
                "seat",
                "index",
                "agent_index",
                "agentIndex",
            )
            agents.append(
                {
                    "submission_id": as_int(
                        submission,
                        f"episode {episode_id} agent submission",
                    ),
                    "seat": position if index is None else as_int(
                        index,
                        f"episode {episode_id} agent seat",
                    ),
                    "team_name": str(
                        get_any(agent, "team_name", "teamName") or ""
                    ),
                    "reward": get_any(agent, "reward"),
                }
            )
    return {
        "episode_id": episode_id,
        "create_time": str(
            get_any(raw, "create_time", "createTime") or ""
        ),
        "end_time": str(get_any(raw, "end_time", "endTime") or ""),
        "state": str(raw.get("state") or ""),
        "type": str(raw.get("type") or ""),
        "agents": agents,
    }


def metadata_target_seat(
    metadata: Mapping[str, Any],
    submission_id: int,
    expected_team: str,
) -> tuple[int | None, str | None]:
    state = str(metadata.get("state") or "").upper()
    if state and state != "COMPLETED":
        return None, "episode_not_completed"
    episode_type = str(metadata.get("type") or "").upper()
    if episode_type and "PUBLIC" not in episode_type:
        return None, "episode_not_public"
    matches = [
        agent
        for agent in metadata.get("agents", [])
        if isinstance(agent, Mapping)
        and int(agent.get("submission_id", -1)) == submission_id
    ]
    if not matches:
        return None, "submission_absent_from_episode_metadata"
    if len(matches) != 1:
        return None, "submission_has_ambiguous_seats"
    try:
        seat = int(matches[0]["seat"])
    except (KeyError, TypeError, ValueError):
        return None, "missing_target_seat"
    if seat not in (0, 1):
        return None, "invalid_target_seat"
    metadata_team = normalize_team_name(matches[0].get("team_name"))
    if metadata_team and metadata_team != normalize_team_name(expected_team):
        return None, "metadata_team_mismatch"
    return seat, None


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    submission_id: int
    team_name: str
    deck_hash: str
    archetype: str
    deck_path: str | None
    deck_csv_path: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class ReplayLocation:
    path: Path
    member: str | None
    dataset_date: str
    source_order: int

    def read(self) -> dict[str, Any]:
        try:
            if self.member is None:
                payload = orjson.loads(self.path.read_bytes())
            else:
                with zipfile.ZipFile(self.path) as archive:
                    payload = orjson.loads(archive.read(self.member))
        except (OSError, zipfile.BadZipFile, KeyError, orjson.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unable to read replay {self.path}"
                + (f"::{self.member}" if self.member else "")
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Replay JSON root must be an object")
        return payload


@dataclass(frozen=True)
class AcceptedEpisode:
    episode_id: str
    seat: int
    metadata: dict[str, Any]
    location: ReplayLocation
    deck_cards: tuple[int, ...]


class ShardedArchiveWriter:
    def __init__(
        self,
        output: Path,
        rows_per_shard: int,
        overwrite: bool,
    ) -> None:
        self.output = output
        self.rows_per_shard = rows_per_shard
        self.overwrite = overwrite
        self.partial = output.with_name(f"{output.name}.partial-{os.getpid()}")
        self.archive: zipfile.ZipFile | None = None
        self.handle: BinaryIO | None = None
        self.current_split: str | None = None
        self.current_rows = 0
        self.shards: Counter[str] = Counter()
        self.rows: Counter[str] = Counter()

    def __enter__(self) -> "ShardedArchiveWriter":
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists() and not self.overwrite:
            raise FileExistsError(self.output)
        if self.partial.exists():
            raise FileExistsError(self.partial)
        self.archive = zipfile.ZipFile(
            self.partial,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        )
        return self

    def _open(self, split: str) -> None:
        assert self.archive is not None
        if self.handle is not None:
            self.handle.close()
        member = f"{split}/part-{self.shards[split]:05d}.jsonl"
        self.handle = self.archive.open(member, "w", force_zip64=True)
        self.current_split = split
        self.current_rows = 0
        self.shards[split] += 1

    def add(self, split: str, row: Mapping[str, Any]) -> None:
        if (
            self.handle is None
            or self.current_split != split
            or self.current_rows >= self.rows_per_shard
        ):
            self._open(split)
        assert self.handle is not None
        self.handle.write(json_bytes(row))
        self.current_rows += 1
        self.rows[split] += 1

    def finish(self, manifest: Mapping[str, Any]) -> None:
        assert self.archive is not None
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.archive.writestr(
            "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n",
        )
        self.archive.close()
        self.archive = None
        os.replace(self.partial, self.output)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        if exc_type is not None:
            self.partial.unlink(missing_ok=True)


def parse_policy_specs(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> list[PolicySpec]:
    policies_raw = manifest.get("policies")
    if not isinstance(policies_raw, list) or not policies_raw:
        raise ValueError("Active manifest must contain a non-empty policies list")
    seen_policy_ids: set[str] = set()
    seen_submission_ids: set[int] = set()
    policies: list[PolicySpec] = []
    for index, value in enumerate(policies_raw):
        if not isinstance(value, dict):
            raise ValueError(f"policies[{index}] must be an object")
        policy_id = str(value.get("policy_id") or "").strip()
        if not POLICY_ID_RE.fullmatch(policy_id):
            raise ValueError(
                f"Invalid policy_id {policy_id!r}; use letters, numbers, ., _, -"
            )
        submission_id = as_int(
            get_any(value, "submission_id", "submissionId"),
            f"policies[{index}].submission_id",
        )
        team_name = str(
            get_any(value, "team_name", "teamName") or ""
        ).strip()
        deck_hash = str(
            get_any(value, "deck_hash", "deckHash") or ""
        ).strip().lower()
        if not team_name:
            raise ValueError(f"Policy {policy_id} has no team_name")
        if not re.fullmatch(r"[0-9a-f]{64}", deck_hash):
            raise ValueError(
                f"Policy {policy_id} deck_hash must be 64 lowercase hex digits"
            )
        if policy_id in seen_policy_ids:
            raise ValueError(f"Duplicate policy_id {policy_id!r}")
        if submission_id in seen_submission_ids:
            raise ValueError(
                f"Submission {submission_id} is assigned to multiple policies"
            )
        seen_policy_ids.add(policy_id)
        seen_submission_ids.add(submission_id)
        policies.append(
            PolicySpec(
                policy_id=policy_id,
                submission_id=submission_id,
                team_name=team_name,
                deck_hash=deck_hash,
                archetype=str(value.get("archetype") or ""),
                deck_path=(
                    str(value["deck_path"])
                    if value.get("deck_path") is not None
                    else None
                ),
                deck_csv_path=(
                    str(value["deck_csv_path"])
                    if value.get("deck_csv_path") is not None
                    else None
                ),
                raw=dict(value),
            )
        )
    return policies


def external_index_by_submission(
    value: Mapping[str, Any] | None,
) -> dict[int, list[Any]]:
    if not value:
        return {}
    candidate = value.get("episodes_by_submission")
    if candidate is None:
        candidate = value.get("submissions")
    if candidate is None:
        return {}
    result: dict[int, list[Any]] = {}
    if isinstance(candidate, dict):
        items = candidate.items()
    elif isinstance(candidate, list):
        items = []
        for entry in candidate:
            if not isinstance(entry, dict):
                continue
            submission_id = get_any(entry, "submission_id", "submissionId")
            episodes = entry.get("episodes")
            items.append((submission_id, episodes))
    else:
        raise ValueError("Episode index submissions must be an object or list")
    for raw_submission_id, raw_episodes in items:
        submission_id = as_int(raw_submission_id, "episode index submission")
        if isinstance(raw_episodes, dict):
            raw_episodes = raw_episodes.get("episodes")
        if not isinstance(raw_episodes, list):
            raise ValueError(
                f"Episode index for submission {submission_id} is not a list"
            )
        result.setdefault(submission_id, []).extend(raw_episodes)
    return result


def embedded_policy_episodes(policy: PolicySpec) -> list[Any]:
    result: list[Any] = []
    episodes = policy.raw.get("episodes")
    if isinstance(episodes, list):
        result.extend(episodes)
    episode_ids = policy.raw.get("episode_ids")
    seats = policy.raw.get("episode_seats")
    if isinstance(episode_ids, list) and isinstance(seats, dict):
        for raw_episode_id in episode_ids:
            episode_id = as_int(raw_episode_id, "episode_ids item")
            seat_value = seats.get(str(episode_id), seats.get(episode_id))
            if seat_value is None:
                raise ValueError(
                    f"Policy {policy.policy_id} episode {episode_id} has no "
                    "episode_seats entry"
                )
            result.append(
                {
                    "id": episode_id,
                    "state": "COMPLETED",
                    "type": "EPISODE_TYPE_PUBLIC",
                    "agents": [
                        {
                            "submissionId": policy.submission_id,
                            "index": as_int(
                                seat_value,
                                f"episode {episode_id} seat",
                            ),
                            "teamName": policy.team_name,
                        }
                    ],
                }
            )
    elif isinstance(episode_ids, list):
        raise ValueError(
            f"Policy {policy.policy_id} provides episode_ids without "
            "episode_seats; exact extraction requires both"
        )
    return result


def kaggle_episode_index(submission_id: int) -> list[dict[str, Any]]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The Kaggle package is required for --resolve-api"
        ) from exc
    api = KaggleApi()
    api.authenticate()
    return [
        canonical_episode_metadata(value)
        for value in api.competition_list_episodes(submission_id)
    ]


def resolve_episode_indexes(
    policies: Sequence[PolicySpec],
    external: Mapping[int, list[Any]],
    resolve_api: bool,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Counter[str]],
]:
    resolved: dict[str, dict[str, dict[str, Any]]] = {}
    provenance_counts: dict[str, Counter[str]] = {}
    for policy in policies:
        merged: dict[str, dict[str, Any]] = {}
        provenance = Counter()
        sources: list[tuple[str, Iterable[Any]]] = [
            ("embedded", embedded_policy_episodes(policy)),
            ("external_index", external.get(policy.submission_id, [])),
        ]
        if resolve_api:
            log(
                f"Listing public episodes for submission "
                f"{policy.submission_id} ({policy.policy_id})"
            )
            sources.append(
                ("kaggle_api", kaggle_episode_index(policy.submission_id))
            )
        for source_name, values in sources:
            for value in values:
                metadata = canonical_episode_metadata(value)
                episode_id = str(metadata["episode_id"])
                existing = merged.get(episode_id)
                if existing is not None and existing != metadata:
                    # Prefer the API's full metadata, but do not silently accept
                    # a conflicting target seat.
                    old_seat, old_error = metadata_target_seat(
                        existing,
                        policy.submission_id,
                        policy.team_name,
                    )
                    new_seat, new_error = metadata_target_seat(
                        metadata,
                        policy.submission_id,
                        policy.team_name,
                    )
                    if (
                        old_error is None
                        and new_error is None
                        and old_seat != new_seat
                    ):
                        raise ValueError(
                            f"Conflicting seats for submission "
                            f"{policy.submission_id} episode {episode_id}"
                        )
                merged[episode_id] = metadata
                provenance[source_name] += 1
        if not merged:
            raise RuntimeError(
                f"Policy {policy.policy_id} has no exact episode metadata. "
                "Embed episodes, pass --episode-index, or use --resolve-api."
            )
        resolved[policy.policy_id] = merged
        provenance_counts[policy.policy_id] = provenance
    return resolved, provenance_counts


def bound_episode_indexes_newest(
    episode_indexes: Mapping[str, Mapping[str, dict[str, Any]]],
    maximum_per_policy: int | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Optionally retain the newest public episode candidates per policy.

    Kaggle returns active-submission episodes newest first, but replay IDs were
    historically sorted ascending before a bounded download.  That selected
    stale games for long-lived submissions.  Bound the metadata first using
    the canonical timestamp and episode ID so a small live-policy snapshot is
    both recent and deterministic.
    """

    if maximum_per_policy is None:
        return {
            policy_id: dict(metadata_by_id)
            for policy_id, metadata_by_id in episode_indexes.items()
        }
    if maximum_per_policy < 1:
        raise ValueError("maximum_per_policy must be positive")
    bounded: dict[str, dict[str, dict[str, Any]]] = {}
    for policy_id, metadata_by_id in episode_indexes.items():
        ordered = sorted(
            metadata_by_id.items(),
            key=lambda item: (
                str(item[1].get("create_time") or ""),
                int(item[0]),
            ),
            reverse=True,
        )
        bounded[policy_id] = dict(ordered[:maximum_per_policy])
    return bounded


def index_replay_sources(
    sources: Sequence[Path],
    wanted_ids: set[str],
) -> tuple[dict[str, ReplayLocation], dict[str, list[str]]]:
    locations: dict[str, ReplayLocation] = {}
    duplicates: dict[str, list[str]] = defaultdict(list)
    for source_order, source_value in enumerate(sources):
        source = source_value.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        fallback_date = parse_dataset_date(source)
        candidates: list[tuple[str, str | None]] = []
        if source.is_dir():
            for path in sorted(source.rglob("*.json")):
                episode_id = episode_id_from_member(path.name)
                if episode_id in wanted_ids:
                    candidates.append((str(path), None))
        elif zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                for name in archive.namelist():
                    episode_id = episode_id_from_member(name)
                    if episode_id in wanted_ids:
                        candidates.append((name, name))
        elif source.suffix.lower() == ".json":
            episode_id = episode_id_from_member(source.name)
            if episode_id in wanted_ids:
                candidates.append((str(source), None))
        else:
            raise ValueError(f"Unsupported replay source: {source}")
        for name, member in candidates:
            episode_id = episode_id_from_member(name)
            assert episode_id is not None
            location_path = Path(name) if source.is_dir() else source
            location = ReplayLocation(
                path=location_path,
                member=member,
                dataset_date=fallback_date,
                source_order=source_order,
            )
            rendered = (
                f"{location.path}::{location.member}"
                if location.member
                else str(location.path)
            )
            if episode_id in locations:
                duplicates[episode_id].append(rendered)
                continue
            locations[episode_id] = location
    return locations, dict(duplicates)


def fetch_missing_replays(
    episode_ids: Sequence[str],
    replay_cache: Path,
    max_fetch: int,
    max_attempts: int = 3,
) -> dict[str, ReplayLocation]:
    if max_fetch < 1:
        return {}
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    selected = list(sorted(set(episode_ids), key=int))[:max_fetch]
    if not selected:
        return {}
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.competitions.types.competition_api_service import (
            ApiGetEpisodeReplayRequest,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The Kaggle package is required for --fetch-missing"
        ) from exc
    replay_cache.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    result: dict[str, ReplayLocation] = {}
    for position, episode_id in enumerate(selected, 1):
        path = replay_cache / f"episode-{episode_id}-replay.json"
        if not path.is_file():
            for attempt in range(1, max_attempts + 1):
                log(
                    f"Downloading public replay {position}/{len(selected)}: "
                    f"episode {episode_id} (attempt {attempt}/{max_attempts})"
                )
                request = ApiGetEpisodeReplayRequest()
                request.episode_id = int(episode_id)
                try:
                    with api.build_kaggle_client() as kaggle:
                        response = (
                            kaggle.competitions.competition_api_client
                            .get_episode_replay(request)
                        )
                        write_replay_response_atomic(
                            response,
                            path,
                            episode_id,
                        )
                    break
                except Exception as exc:
                    if attempt >= max_attempts:
                        raise
                    log(
                        f"Replay {episode_id} download failed with "
                        f"{type(exc).__name__}; retrying"
                    )
                    time.sleep(float(attempt))
        result[episode_id] = ReplayLocation(
            path=path,
            member=None,
            dataset_date="unknown",
            source_order=10**9,
        )
    return result


def write_replay_response_atomic(
    response: Any,
    path: Path,
    expected_episode_id: str,
) -> None:
    """Persist a public replay without requiring a Content-Length header.

    Kaggle API 2.0.2's generic downloader assumes every response has a
    ``Content-Length`` header.  The simulation replay endpoint can use chunked
    transfer encoding instead, so stream the already-authenticated response to
    a private temporary file and validate the replay identity before publish.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
        with temporary.open("xb") as handle:
            if callable(getattr(response, "iter_content", None)):
                chunks = response.iter_content(1024 * 1024)
                for chunk in chunks:
                    if chunk:
                        handle.write(chunk)
            elif callable(getattr(response, "read", None)):
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            else:
                raise TypeError(
                    "Replay response supports neither iter_content nor read"
                )
            handle.flush()
            os.fsync(handle.fileno())

        payload = orjson.loads(temporary.read_bytes())
        if not isinstance(payload, dict):
            raise RuntimeError("Downloaded replay JSON root is not an object")
        observed_episode_id = replay_episode_id(payload, expected_episode_id)
        if observed_episode_id != expected_episode_id:
            raise RuntimeError(
                "Downloaded replay episode ID mismatch: "
                f"expected {expected_episode_id}, got {observed_episode_id}"
            )
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def replay_names(replay: Mapping[str, Any]) -> list[str]:
    info = replay.get("info")
    info = info if isinstance(info, Mapping) else {}
    names = info.get("TeamNames")
    if isinstance(names, list) and len(names) >= 2:
        return [str(value or "").strip() for value in names[:2]]
    agents = info.get("Agents")
    if isinstance(agents, list) and len(agents) >= 2:
        return [
            str(agent.get("Name") or "").strip()
            if isinstance(agent, Mapping)
            else ""
            for agent in agents[:2]
        ]
    return ["", ""]


def replay_episode_id(replay: Mapping[str, Any], fallback: str) -> str:
    info = replay.get("info")
    info = info if isinstance(info, Mapping) else {}
    return str(info.get("EpisodeId") or fallback)


def replay_rewards(replay: Mapping[str, Any]) -> list[float]:
    raw = replay.get("rewards")
    result: list[float] = []
    if isinstance(raw, list):
        for value in raw[:2]:
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                result.append(0.0)
    return (result + [0.0, 0.0])[:2]


def replay_decks(
    replay: Mapping[str, Any],
) -> list[tuple[int, ...] | None]:
    steps = replay.get("steps")
    if not isinstance(steps, list):
        return [None, None]
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
                if (
                    not isinstance(action, list)
                    or len(action) < 2
                    or not all(
                        isinstance(deck, list) and len(deck) == 60
                        for deck in action[:2]
                    )
                ):
                    continue
                decks: list[tuple[int, ...] | None] = []
                for deck in action[:2]:
                    try:
                        canonical_cards = tuple(
                            sorted(int(card) for card in deck)
                        )
                    except (TypeError, ValueError):
                        decks.append(None)
                    else:
                        decks.append(canonical_cards)
                return decks
    return [None, None]


def hash_deck(deck: Sequence[int]) -> str:
    canonical = ",".join(str(card) for card in deck)
    return hashlib.sha256(canonical.encode()).hexdigest()


def replay_deck_hashes(replay: Mapping[str, Any]) -> list[str | None]:
    return [
        hash_deck(deck) if deck is not None else None
        for deck in replay_decks(replay)
    ]


def valid_action(select: Mapping[str, Any], action: Any) -> bool:
    if not isinstance(action, list):
        return False
    try:
        minimum = int(select.get("minCount", 0) or 0)
        maximum = int(select.get("maxCount", 0) or 0)
    except (TypeError, ValueError):
        return False
    options = select.get("option")
    if not isinstance(options, list) or not options:
        return False
    if not minimum <= len(action) <= maximum:
        return False
    if len(set(action)) != len(action):
        return False
    return all(
        isinstance(index, int)
        and not isinstance(index, bool)
        and 0 <= index < len(options)
        for index in action
    )


def trajectory_weight(reward: float, progress: float) -> float:
    if reward > 0:
        return 1.0
    if reward == 0:
        return 0.85
    return 0.75 if progress < 0.60 else 0.25


def iter_target_decisions(
    replay: Mapping[str, Any],
    accepted: AcceptedEpisode,
    policy: PolicySpec,
    split: str,
    stats: Counter[str],
) -> Iterator[dict[str, Any]]:
    steps = replay.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        stats["episodes_without_steps"] += 1
        return
    seat = accepted.seat
    names = replay_names(replay)
    rewards = replay_rewards(replay)
    total_transitions = len(steps) - 1
    episode_uuid = replay.get("id")
    emitted = 0
    for action_step_index in range(1, len(steps)):
        previous_frame = steps[action_step_index - 1]
        current_frame = steps[action_step_index]
        if not isinstance(previous_frame, list) or not isinstance(current_frame, list):
            continue
        if seat >= len(previous_frame) or seat >= len(current_frame):
            continue
        previous_entry = previous_frame[seat]
        current_entry = current_frame[seat]
        if not isinstance(previous_entry, Mapping) or not isinstance(
            current_entry, Mapping
        ):
            continue
        observation = previous_entry.get("observation")
        if not isinstance(observation, Mapping):
            continue
        select = observation.get("select")
        if not isinstance(select, Mapping) or not select.get("option"):
            continue
        if str(previous_entry.get("status") or "").upper() != "ACTIVE":
            stats["inactive_frames_skipped"] += 1
            continue
        action = current_entry.get("action")
        if not valid_action(select, action):
            stats["invalid_active_decisions"] += 1
            continue
        visible_observation = dict(observation)
        if "visualize" in visible_observation:
            visible_observation.pop("visualize", None)
            stats["stripped_visualize_from_observation"] += 1
        options = select.get("option") or []
        context = str(select.get("context", "unknown"))
        reward = rewards[seat]
        progress = action_step_index / max(1, total_transitions)
        yield {
            "schema_version": ROW_SCHEMA,
            "episode_id": accepted.episode_id,
            "episode_uuid": str(episode_uuid) if episode_uuid else None,
            "dataset_date": accepted.location.dataset_date,
            "split": split,
            "observation_step_index": action_step_index - 1,
            "action_step_index": action_step_index,
            "seat": seat,
            "team_name": policy.team_name,
            "opponent_team_name": names[1 - seat],
            "deck_hash": policy.deck_hash,
            "source_submission_id": policy.submission_id,
            "policy_id": policy.policy_id,
            "terminal_reward": reward,
            "sample_weight": trajectory_weight(reward, progress),
            "select_context": context,
            "select_type": select.get("type"),
            "min_count": int(select.get("minCount", 0) or 0),
            "max_count": int(select.get("maxCount", 0) or 0),
            "option_count": len(options),
            "action": list(action),
            "no_action": len(action) == 0,
            "observation": visible_observation,
        }
        emitted += 1
    if emitted:
        stats["episodes_with_decisions"] += 1
    else:
        stats["episodes_without_legal_decisions"] += 1


def validate_candidate(
    policy: PolicySpec,
    metadata: dict[str, Any],
    location: ReplayLocation,
) -> tuple[AcceptedEpisode | None, str | None]:
    episode_id = str(metadata["episode_id"])
    seat, error = metadata_target_seat(
        metadata,
        policy.submission_id,
        policy.team_name,
    )
    if error is not None or seat is None:
        return None, error
    replay = location.read()
    if replay_episode_id(replay, episode_id) != episode_id:
        return None, "replay_episode_id_mismatch"
    names = replay_names(replay)
    if seat >= len(names) or (
        normalize_team_name(names[seat])
        != normalize_team_name(policy.team_name)
    ):
        return None, "replay_team_mismatch"
    decks = replay_decks(replay)
    if seat >= len(decks) or decks[seat] is None:
        return None, "replay_deck_hash_missing"
    target_deck = decks[seat]
    assert target_deck is not None
    if hash_deck(target_deck) != policy.deck_hash:
        return None, "replay_deck_hash_mismatch"
    return (
        AcceptedEpisode(
            episode_id=episode_id,
            seat=seat,
            metadata=metadata,
            location=location,
            deck_cards=target_deck,
        ),
        None,
    )


def split_episodes(
    episodes: Sequence[AcceptedEpisode],
    submission_id: int,
    seed: int,
    holdout_fraction: float,
) -> dict[str, list[AcceptedEpisode]]:
    if len(episodes) < 2:
        raise RuntimeError(
            "At least two accepted episodes are required for episode-disjoint "
            "train and holdout splits"
        )
    ranked = sorted(
        episodes,
        key=lambda item: hashlib.sha256(
            f"{seed}:{submission_id}:{item.episode_id}".encode()
        ).digest(),
    )
    holdout_count = max(1, int(round(len(ranked) * holdout_fraction)))
    holdout_count = min(len(ranked) - 1, holdout_count)
    return {
        "valid": ranked[:holdout_count],
        "train": ranked[holdout_count:],
    }


def verify_policy_archive(
    archive_path: Path,
    policy: PolicySpec,
    expected_episode_splits: Mapping[str, set[str]],
) -> dict[str, Any]:
    row_counts: Counter[str] = Counter()
    episode_sets: dict[str, set[str]] = defaultdict(set)
    seat_counts: Counter[str] = Counter()
    duplicate_keys = 0
    hidden_leaks = 0
    identity_mismatches = 0
    seen_keys: set[tuple[str, int, int]] = set()
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failure at {bad}")
        if "manifest.json" not in archive.namelist():
            raise RuntimeError("Policy archive has no manifest.json")
        for split in ("train", "valid"):
            members = sorted(
                name
                for name in archive.namelist()
                if name.startswith(f"{split}/") and name.endswith(".jsonl")
            )
            if not members:
                raise RuntimeError(f"Policy archive has no {split} rows")
            for member in members:
                with archive.open(member) as handle:
                    for raw_line in handle:
                        row = orjson.loads(raw_line)
                        episode_id = str(row.get("episode_id") or "")
                        seat = int(row.get("seat", -1))
                        step = int(row.get("action_step_index", -1))
                        key = (episode_id, seat, step)
                        if key in seen_keys:
                            duplicate_keys += 1
                        seen_keys.add(key)
                        if (
                            row.get("split") != split
                            or episode_id not in expected_episode_splits[split]
                            or row.get("policy_id") != policy.policy_id
                            or int(row.get("source_submission_id", -1))
                            != policy.submission_id
                            or normalize_team_name(row.get("team_name"))
                            != normalize_team_name(policy.team_name)
                            or row.get("deck_hash") != policy.deck_hash
                        ):
                            identity_mismatches += 1
                        observation = row.get("observation")
                        if (
                            "visualize" in row
                            or isinstance(observation, Mapping)
                            and "visualize" in observation
                        ):
                            hidden_leaks += 1
                        row_counts[split] += 1
                        episode_sets[split].add(episode_id)
                        seat_counts[str(seat)] += 1
    overlap = episode_sets["train"] & episode_sets["valid"]
    if (
        duplicate_keys
        or hidden_leaks
        or identity_mismatches
        or overlap
    ):
        raise RuntimeError(
            "Policy archive contamination audit failed: "
            f"duplicates={duplicate_keys}, hidden_leaks={hidden_leaks}, "
            f"identity_mismatches={identity_mismatches}, "
            f"split_overlap={len(overlap)}"
        )
    return {
        "rows": dict(row_counts),
        "episodes": {
            split: len(episode_sets[split])
            for split in ("train", "valid")
        },
        "seat_rows": dict(seat_counts),
        "duplicate_decision_keys": duplicate_keys,
        "hidden_visualize_leaks": hidden_leaks,
        "identity_mismatches": identity_mismatches,
        "episode_split_overlap": len(overlap),
        "foreign_seat_decisions": 0,
    }


def resolve_optional_deck_path(
    value: str | None,
    input_manifest_path: Path,
) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = input_manifest_path.parent / path
    return str(path.resolve()) if path.is_file() else None


def read_deck_csv(path: Path) -> tuple[int, ...]:
    cards: list[int] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            cards.append(int(line))
        except ValueError as exc:
            raise ValueError(
                f"Invalid card ID in {path}:{line_number}: {line!r}"
            ) from exc
    if len(cards) != 60:
        raise ValueError(f"Deck {path} has {len(cards)} cards, expected 60")
    return tuple(sorted(cards))


def write_deck_csv_atomic(
    path: Path,
    cards: Sequence[int],
    expected_hash: str,
) -> None:
    canonical_cards = tuple(sorted(int(card) for card in cards))
    if len(canonical_cards) != 60:
        raise ValueError(
            f"Refusing to write {path}: expected 60 cards, "
            f"got {len(canonical_cards)}"
        )
    observed_hash = hash_deck(canonical_cards)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Deck materialization hash mismatch: "
            f"{observed_hash} != {expected_hash}"
        )
    if path.is_file():
        existing = read_deck_csv(path)
        if existing != canonical_cards:
            raise RuntimeError(
                f"Existing deck file disagrees with verified replays: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        "".join(f"{card}\n" for card in canonical_cards),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_json_atomic(
    path: Path,
    value: Mapping[str, Any],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_gold_policy_archives(
    active_manifest_path: Path,
    replay_sources: Sequence[Path],
    output_root: Path,
    *,
    episode_index_path: Path | None = None,
    resolve_api: bool = False,
    fetch_missing: bool = False,
    max_fetch_missing: int = 25,
    max_episode_candidates_per_policy: int | None = None,
    replay_cache: Path | None = None,
    fetch_submission_ids: set[int] | None = None,
    split_seed: int = 20260727,
    holdout_fraction: float = 0.20,
    rows_per_shard: int = 25_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    active_manifest_path = active_manifest_path.resolve()
    output_root = output_root.resolve()
    if not active_manifest_path.is_file():
        raise FileNotFoundError(active_manifest_path)
    if not replay_sources and not fetch_missing:
        raise ValueError("Provide at least one --input or pass --fetch-missing")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if rows_per_shard < 1:
        raise ValueError("rows_per_shard must be positive")
    active_manifest = json.loads(
        active_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(active_manifest, dict):
        raise ValueError("Active manifest root must be an object")
    policies = parse_policy_specs(active_manifest, active_manifest_path)
    if fetch_submission_ids:
        known_submission_ids = {policy.submission_id for policy in policies}
        unknown_fetch_ids = fetch_submission_ids - known_submission_ids
        if unknown_fetch_ids:
            raise ValueError(
                "--fetch-submission-id is not present in the active manifest: "
                f"{sorted(unknown_fetch_ids)}"
            )
    external_value: dict[str, Any] | None = None
    if episode_index_path is not None:
        external_value = json.loads(
            episode_index_path.resolve().read_text(encoding="utf-8")
        )
        if not isinstance(external_value, dict):
            raise ValueError("Episode index root must be an object")
    external = external_index_by_submission(external_value)
    episode_indexes, provenance = resolve_episode_indexes(
        policies,
        external,
        resolve_api,
    )
    episode_indexes = bound_episode_indexes_newest(
        episode_indexes,
        max_episode_candidates_per_policy,
    )
    wanted_ids = {
        episode_id
        for policy_index in episode_indexes.values()
        for episode_id in policy_index
    }
    locations, duplicate_locations = index_replay_sources(
        replay_sources,
        wanted_ids,
    )
    missing_ids = sorted(wanted_ids - set(locations), key=int)
    fetched: dict[str, ReplayLocation] = {}
    if fetch_missing and missing_ids:
        fetch_candidates = missing_ids
        if fetch_submission_ids:
            fetch_episode_ids = {
                episode_id
                for policy in policies
                if policy.submission_id in fetch_submission_ids
                for episode_id in episode_indexes[policy.policy_id]
            }
            fetch_candidates = [
                episode_id
                for episode_id in missing_ids
                if episode_id in fetch_episode_ids
            ]
        effective_cache = (
            replay_cache.resolve()
            if replay_cache is not None
            else output_root / "replay_cache"
        )
        fetched = fetch_missing_replays(
            fetch_candidates,
            effective_cache,
            max_fetch_missing,
        )
        locations.update(fetched)

    output_root.mkdir(parents=True, exist_ok=True)
    policy_outputs: list[dict[str, Any]] = []
    all_assignments: dict[
        tuple[str, int], list[tuple[str, str]]
    ] = defaultdict(list)
    policy_episode_sets: dict[str, set[str]] = {}
    policy_split_by_episode: dict[str, dict[str, str]] = {}
    resolved_index_snapshot: dict[str, list[dict[str, Any]]] = {}

    for policy_position, policy in enumerate(policies, 1):
        metadata_by_id = episode_indexes[policy.policy_id]
        resolved_index_snapshot[str(policy.submission_id)] = [
            metadata_by_id[key]
            for key in sorted(metadata_by_id, key=int)
        ]
        rejection_counts: Counter[str] = Counter()
        accepted: list[AcceptedEpisode] = []
        for episode_id, metadata in sorted(
            metadata_by_id.items(),
            key=lambda item: int(item[0]),
        ):
            location = locations.get(episode_id)
            if location is None:
                rejection_counts["replay_not_available"] += 1
                continue
            accepted_episode, rejection = validate_candidate(
                policy,
                metadata,
                location,
            )
            if rejection is not None:
                rejection_counts[rejection] += 1
                continue
            assert accepted_episode is not None
            accepted.append(accepted_episode)
        log(
            f"[{policy_position}/{len(policies)}] {policy.policy_id}: "
            f"candidates={len(metadata_by_id)} accepted={len(accepted)} "
            f"rejected={sum(rejection_counts.values())}"
        )
        observed_decks = {item.deck_cards for item in accepted}
        if len(observed_decks) != 1:
            raise RuntimeError(
                f"Policy {policy.policy_id} has {len(observed_decks)} distinct "
                "verified decklists across accepted replays"
            )
        verified_deck = next(iter(observed_decks))
        generated_deck_path = (
            output_root / "decks" / f"{policy.deck_hash}.csv"
        )
        write_deck_csv_atomic(
            generated_deck_path,
            verified_deck,
            policy.deck_hash,
        )
        declared_deck_path = resolve_optional_deck_path(
            policy.deck_path,
            active_manifest_path,
        )
        declared_deck_csv_path = resolve_optional_deck_path(
            policy.deck_csv_path,
            active_manifest_path,
        )
        declared_deck_matches: dict[str, bool | None] = {}
        for label, resolved_path in (
            ("deck_path", declared_deck_path),
            ("deck_csv_path", declared_deck_csv_path),
        ):
            declared_deck_matches[label] = (
                read_deck_csv(Path(resolved_path)) == verified_deck
                if resolved_path is not None
                else None
            )
        split_items = split_episodes(
            accepted,
            policy.submission_id,
            split_seed,
            holdout_fraction,
        )
        split_ids = {
            split: {item.episode_id for item in items}
            for split, items in split_items.items()
        }
        policy_episode_sets[policy.policy_id] = set().union(*split_ids.values())
        policy_split_by_episode[policy.policy_id] = {
            episode_id: split
            for split, ids in split_ids.items()
            for episode_id in ids
        }
        for split, items in split_items.items():
            for item in items:
                all_assignments[(item.episode_id, item.seat)].append(
                    (policy.policy_id, split)
                )

        archive_path = output_root / "policies" / f"{policy.policy_id}.zip"
        extraction_stats: Counter[str] = Counter()
        split_rows: Counter[str] = Counter()
        with ShardedArchiveWriter(
            archive_path,
            rows_per_shard,
            overwrite,
        ) as writer:
            for split in ("train", "valid"):
                for item in split_items[split]:
                    replay = item.location.read()
                    for row in iter_target_decisions(
                        replay,
                        item,
                        policy,
                        split,
                        extraction_stats,
                    ):
                        writer.add(split, row)
                        split_rows[split] += 1
            if not split_rows["train"] or not split_rows["valid"]:
                raise RuntimeError(
                    f"Policy {policy.policy_id} produced an empty decision split: "
                    f"{dict(split_rows)}"
                )
            archive_manifest = {
                "schema_version": ARCHIVE_SCHEMA,
                "competition": str(
                    active_manifest.get("competition") or COMPETITION
                ),
                "policy": {
                    "policy_id": policy.policy_id,
                    "submission_id": policy.submission_id,
                    "team_name": policy.team_name,
                    "deck_hash": policy.deck_hash,
                    "archetype": policy.archetype,
                    "rank": policy.raw.get("rank"),
                    "score": policy.raw.get("score"),
                    "submission_ref": policy.raw.get("submission_ref"),
                    "deck_path": str(generated_deck_path),
                    "deck_csv_path": str(generated_deck_path),
                    "declared_deck_path": policy.deck_path,
                    "declared_deck_csv_path": policy.deck_csv_path,
                },
                "label_alignment": (
                    "steps[t-1].observation -> steps[t].action"
                ),
                "hidden_information_policy": (
                    "Replay visualize is used only to verify deck_hash and "
                    "materialize a separate exact deck CSV; only the target "
                    "seat's visible observation and action are saved to BC."
                ),
                "split_policy": {
                    "mode": "ranked_episode_sha256",
                    "hash_input": "{seed}:{submission_id}:{episode_id}",
                    "seed": split_seed,
                    "holdout_fraction": holdout_fraction,
                    "holdout_member": "valid",
                },
                "split_decisions": dict(split_rows),
                "split_episodes": {
                    split: len(ids) for split, ids in split_ids.items()
                },
                "episode_ids": {
                    split: sorted(ids, key=int)
                    for split, ids in split_ids.items()
                },
                "extraction_stats": dict(extraction_stats),
                "candidate_audit": {
                    "metadata_candidates": len(metadata_by_id),
                    "accepted": len(accepted),
                    "rejected": dict(rejection_counts),
                    "episode_metadata_provenance": dict(
                        provenance[policy.policy_id]
                    ),
                    "verified_deck_observations": len(accepted),
                    "verified_distinct_decklists": len(observed_decks),
                    "declared_deck_matches": declared_deck_matches,
                },
                "rows_per_shard": rows_per_shard,
                "shards": dict(writer.shards),
            }
            writer.finish(archive_manifest)

        verified = verify_policy_archive(
            archive_path,
            policy,
            split_ids,
        )
        archive_sha256 = sha256_file(archive_path)
        archive_string = str(archive_path)
        common = {
            "policy_id": policy.policy_id,
            "submission_id": policy.submission_id,
            "team_name": policy.team_name,
            "deck_hash": policy.deck_hash,
            "archetype": policy.archetype,
            "rank": policy.raw.get("rank"),
            "score": policy.raw.get("score"),
            "submission_ref": policy.raw.get("submission_ref"),
            "deck_path": str(generated_deck_path),
            "deck_csv_path": str(generated_deck_path),
            "resolved_deck_path": str(generated_deck_path.resolve()),
            "resolved_deck_csv_path": str(generated_deck_path.resolve()),
            "declared_deck_path": policy.deck_path,
            "declared_deck_csv_path": policy.deck_csv_path,
            "archive_path": archive_string,
            "archive_relpath": str(archive_path.relative_to(output_root)),
            "archive_sha256": archive_sha256,
            "train_split": "train",
            "holdout_split": "valid",
            "train": {
                "archive_path": archive_string,
                "split": "train",
                "episodes": verified["episodes"]["train"],
                "decisions": verified["rows"]["train"],
            },
            "holdout": {
                "archive_path": archive_string,
                "split": "valid",
                "episodes": verified["episodes"]["valid"],
                "decisions": verified["rows"]["valid"],
            },
            "audit": {
                **verified,
                "metadata_candidates": len(metadata_by_id),
                "accepted_episodes": len(accepted),
                "rejected_episodes": dict(rejection_counts),
                "verified_deck_observations": len(accepted),
                "verified_distinct_decklists": len(observed_decks),
                "declared_deck_matches": declared_deck_matches,
            },
        }
        policy_outputs.append(common)

    duplicate_assignments = {
        f"{episode_id}:{seat}": assignments
        for (episode_id, seat), assignments in all_assignments.items()
        if len(assignments) > 1
    }
    cross_policy_episode_pairs: Counter[str] = Counter()
    cross_policy_split_collisions = 0
    for left_index, left in enumerate(policies):
        for right in policies[left_index + 1 :]:
            overlap = (
                policy_episode_sets[left.policy_id]
                & policy_episode_sets[right.policy_id]
            )
            if overlap:
                cross_policy_episode_pairs[
                    f"{left.policy_id}|{right.policy_id}"
                ] = len(overlap)
                # Same episode can legitimately contain two target policies.
                # Count train/holdout disagreement without treating the two
                # different seats as duplicate target assignments.
                for episode_id in overlap:
                    if (
                        policy_split_by_episode[left.policy_id][episode_id]
                        != policy_split_by_episode[right.policy_id][episode_id]
                    ):
                        cross_policy_split_collisions += 1

    index_snapshot_path = output_root / "episode_index.json"
    index_snapshot = {
        "schema_version": "ptcg-public-submission-episode-index-v1",
        "competition": str(
            active_manifest.get("competition") or COMPETITION
        ),
        "submissions": resolved_index_snapshot,
    }
    write_json_atomic(index_snapshot_path, index_snapshot, overwrite)
    output_manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "competition": str(
            active_manifest.get("competition") or COMPETITION
        ),
        "created_at": datetime.now().astimezone().isoformat(),
        "active_manifest": {
            "path": str(active_manifest_path),
            "sha256": sha256_file(active_manifest_path),
            "declared_schema_version": active_manifest.get("schema_version"),
        },
        "episode_index_path": str(index_snapshot_path),
        "replay_sources": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path.resolve())
                if path.resolve().is_file()
                else None,
            }
            for path in replay_sources
        ],
        "split_policy": {
            "mode": "ranked_episode_sha256",
            "seed": split_seed,
            "holdout_fraction": holdout_fraction,
        },
        "episode_candidate_policy": {
            "order": "create_time_desc_then_episode_id_desc",
            "maximum_per_policy": max_episode_candidates_per_policy,
        },
        "policies": policy_outputs,
        "global_audit": {
            "policy_count": len(policy_outputs),
            "wanted_episode_count": len(wanted_ids),
            "located_episode_count": len(set(locations) & wanted_ids),
            "missing_episode_count": len(wanted_ids - set(locations)),
            "fetched_episode_count": len(fetched),
            "duplicate_source_episode_ids": duplicate_locations,
            "duplicate_episode_seat_assignments": duplicate_assignments,
            "cross_policy_episode_pairs": dict(cross_policy_episode_pairs),
            "cross_policy_split_collisions": cross_policy_split_collisions,
            "verified_deck_count": len(
                {policy.deck_hash for policy in policies}
            ),
            "verified_decks": {
                policy.deck_hash: str(
                    output_root / "decks" / f"{policy.deck_hash}.csv"
                )
                for policy in policies
            },
            "hidden_visualize_leaks": 0,
            "foreign_seat_decisions": 0,
        },
    }
    output_manifest_path = output_root / "manifest.json"
    write_json_atomic(output_manifest_path, output_manifest, overwrite)
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract exact target-seat BC data for active public Kaggle "
            "submissions."
        )
    )
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Local daily replay ZIP, replay JSON, or replay directory",
    )
    parser.add_argument("--episode-index", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--resolve-api",
        action="store_true",
        help="Read-only: list public episodes for every submission via Kaggle",
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Read-only: download a bounded number of missing public replays",
    )
    parser.add_argument(
        "--max-fetch-missing",
        type=int,
        default=25,
        help="Global cap for --fetch-missing (default: 25)",
    )
    parser.add_argument(
        "--max-episode-candidates-per-policy",
        type=int,
        help=(
            "Retain only the newest N resolved public episode candidates for "
            "each policy before locating or downloading replays"
        ),
    )
    parser.add_argument(
        "--fetch-submission-id",
        type=int,
        action="append",
        default=[],
        help=(
            "Restrict missing-replay downloads to this submission ID "
            "(repeatable; useful for a new policy absent from the daily ZIP)"
        ),
    )
    parser.add_argument("--replay-cache", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260727)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--rows-per-shard", type=int, default=25_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_gold_policy_archives(
        args.active_manifest,
        args.input,
        args.output_root,
        episode_index_path=args.episode_index,
        resolve_api=args.resolve_api,
        fetch_missing=args.fetch_missing,
        max_fetch_missing=args.max_fetch_missing,
        max_episode_candidates_per_policy=(
            args.max_episode_candidates_per_policy
        ),
        replay_cache=args.replay_cache,
        fetch_submission_ids=set(args.fetch_submission_id),
        split_seed=args.split_seed,
        holdout_fraction=args.holdout_fraction,
        rows_per_shard=args.rows_per_shard,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_manifest": str(
                    args.output_root.resolve() / "manifest.json"
                ),
                "policies": len(result["policies"]),
                "global_audit": result["global_audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
