#!/usr/bin/env python3
"""Select up to three preregistered BC routes without opening test results.

The selector consumes a gold-push preregistration and an exact-deck profile.
Only ``split_stats.train`` and ``split_stats.valid`` are accessed.  In
particular, ``split_stats.test`` and profile-wide outcome totals are neither
validated nor compared, so sealed-test values cannot affect route selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping


PREREGISTRATION_SCHEMA = "ptcg-gold-push-preregistration-v1"
PROFILE_SCHEMA = "ptcg-bc-exact-deck-profile-v1"
OUTPUT_SCHEMA = "ptcg-gold-route-selection-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_WILSON_CONFIDENCE = 0.95
EXPECTED_WILSON_Z = 1.959963984540054

PROFILE_SELECTION_FIELDS = [
    "deduplication_key",
    "profiled_splits",
    "sealed_splits",
    "split_date_counts.train",
    "split_date_counts.valid",
    "decks[].deck_hash",
    "decks[].split_stats.train.episodes",
    "decks[].split_stats.train.team_counts",
    "decks[].split_stats.train.date_counts",
    "decks[].split_stats.valid.episodes",
    "decks[].split_stats.valid.wins",
    "decks[].split_stats.valid.losses",
    "decks[].split_stats.valid.draws",
    "decks[].split_stats.valid.team_counts",
    "decks[].split_stats.valid.date_counts",
]
PREREGISTRATION_SELECTION_FIELDS = [
    "data_protocol.train_dates",
    "data_protocol.valid_dates",
    "route_selection_before_opening_test.marnie_deck_hash",
    "route_selection_before_opening_test.alakazam_deck_hash",
    "route_selection_before_opening_test.excluded_deck_hashes",
    "route_selection_before_opening_test.minimum_valid_episode_seats",
    "route_selection_before_opening_test.minimum_train_valid_teams",
    "route_selection_before_opening_test.frequency_metric",
    "route_selection_before_opening_test.route_3_pool_size",
    "route_selection_before_opening_test.wilson_confidence",
    "route_selection_before_opening_test.wilson_z",
    "route_selection_before_opening_test.draws_in_wilson_denominator",
    "route_selection_before_opening_test.maximum_bc_routes",
    "route_selection_before_opening_test.test_data_must_not_select_hyperparameters",
]


@dataclass(frozen=True)
class SelectionPolicy:
    marnie_deck_hash: str
    alakazam_deck_hash: str
    excluded_deck_hashes: frozenset[str]
    minimum_valid_episode_seats: int
    minimum_train_valid_teams: int
    route_3_pool_size: int
    wilson_confidence: float
    wilson_z: float
    maximum_bc_routes: int
    train_dates: frozenset[str]
    valid_dates: frozenset[str]


@dataclass(frozen=True)
class Candidate:
    deck_hash: str
    train_episodes: int
    valid_episodes: int
    valid_wins: int
    valid_losses: int
    valid_draws: int
    train_valid_teams: tuple[str, ...]
    valid_wilson_lower_bound: float
    train_dates: tuple[str, ...]
    valid_dates: tuple[str, ...]

    @property
    def frequency(self) -> int:
        return self.train_episodes + self.valid_episodes

    def public_metrics(self) -> dict[str, Any]:
        return {
            "train_episode_seats": self.train_episodes,
            "valid_episode_seats": self.valid_episodes,
            "train_plus_valid_episode_seats": self.frequency,
            "train_valid_unique_teams": len(self.train_valid_teams),
            "valid_wins": self.valid_wins,
            "valid_losses": self.valid_losses,
            "valid_draws": self.valid_draws,
            "valid_wilson_95_lower_bound": self.valid_wilson_lower_bound,
        }


def load_json_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _deck_hash(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase SHA-256 hex digest"
        )
    return value


def _date_set(value: Any, field_name: str) -> frozenset[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    result: set[str] = set()
    for index, raw_date in enumerate(value):
        if not isinstance(raw_date, str):
            raise ValueError(f"{field_name}[{index}] must be an ISO date")
        try:
            date.fromisoformat(raw_date)
        except ValueError as error:
            raise ValueError(
                f"{field_name}[{index}] must be an ISO date"
            ) from error
        if raw_date in result:
            raise ValueError(f"{field_name} contains duplicate dates")
        result.add(raw_date)
    return frozenset(result)


def parse_policy(preregistration: Mapping[str, Any]) -> SelectionPolicy:
    if preregistration.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise ValueError("Unexpected preregistration schema_version")
    raw = preregistration.get("route_selection_before_opening_test")
    if not isinstance(raw, Mapping):
        raise ValueError("Missing route_selection_before_opening_test object")
    data_protocol = preregistration.get("data_protocol")
    if not isinstance(data_protocol, Mapping):
        raise ValueError("Missing data_protocol object")
    train_dates = _date_set(
        data_protocol.get("train_dates"), "data_protocol.train_dates"
    )
    valid_dates = _date_set(
        data_protocol.get("valid_dates"), "data_protocol.valid_dates"
    )
    if train_dates & valid_dates:
        raise ValueError("Preregistered train_dates and valid_dates overlap")

    marnie = _deck_hash(raw.get("marnie_deck_hash"), "marnie_deck_hash")
    alakazam = _deck_hash(
        raw.get("alakazam_deck_hash"), "alakazam_deck_hash"
    )
    if marnie == alakazam:
        raise ValueError("Marnie and Alakazam deck hashes must differ")
    raw_excluded = raw.get("excluded_deck_hashes")
    if not isinstance(raw_excluded, list):
        raise ValueError("excluded_deck_hashes must be a list")
    excluded = frozenset(
        _deck_hash(value, f"excluded_deck_hashes[{index}]")
        for index, value in enumerate(raw_excluded)
    )
    if alakazam not in excluded:
        raise ValueError("excluded_deck_hashes must contain alakazam_deck_hash")
    if marnie in excluded:
        raise ValueError("excluded_deck_hashes must not contain marnie_deck_hash")

    frequency_metric = raw.get("frequency_metric")
    if frequency_metric != "train_plus_valid_episode_seats":
        raise ValueError(
            "frequency_metric must be 'train_plus_valid_episode_seats'"
        )
    if raw.get("draws_in_wilson_denominator") is not True:
        raise ValueError("draws_in_wilson_denominator must be true")
    if raw.get("test_data_must_not_select_hyperparameters") is not True:
        raise ValueError(
            "test_data_must_not_select_hyperparameters must be true"
        )

    confidence = raw.get("wilson_confidence")
    z_value = raw.get("wilson_z")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("wilson_confidence must be numeric")
    if isinstance(z_value, bool) or not isinstance(z_value, (int, float)):
        raise ValueError("wilson_z must be numeric")
    confidence = float(confidence)
    z_value = float(z_value)
    if not math.isclose(
        confidence, EXPECTED_WILSON_CONFIDENCE, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("wilson_confidence must be 0.95")
    if not math.isclose(
        z_value, EXPECTED_WILSON_Z, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(
            f"wilson_z must be the preregistered 95% value {EXPECTED_WILSON_Z}"
        )

    maximum_routes = _positive_int(
        raw.get("maximum_bc_routes"), "maximum_bc_routes"
    )
    if maximum_routes > 3:
        raise ValueError("maximum_bc_routes must not exceed 3")
    return SelectionPolicy(
        marnie_deck_hash=marnie,
        alakazam_deck_hash=alakazam,
        excluded_deck_hashes=excluded,
        minimum_valid_episode_seats=_positive_int(
            raw.get("minimum_valid_episode_seats"),
            "minimum_valid_episode_seats",
        ),
        minimum_train_valid_teams=_positive_int(
            raw.get("minimum_train_valid_teams"),
            "minimum_train_valid_teams",
        ),
        route_3_pool_size=_positive_int(
            raw.get("route_3_pool_size"), "route_3_pool_size"
        ),
        wilson_confidence=confidence,
        wilson_z=z_value,
        maximum_bc_routes=maximum_routes,
        train_dates=train_dates,
        valid_dates=valid_dates,
    )


def wilson_lower_bound(wins: int, total: int, z_value: float) -> float:
    if total <= 0 or wins < 0 or wins > total:
        raise ValueError("Wilson inputs require 0 <= wins <= total and total > 0")
    probability = wins / total
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / total
    center = probability + z_squared / (2.0 * total)
    margin = z_value * math.sqrt(
        probability * (1.0 - probability) / total
        + z_squared / (4.0 * total * total)
    )
    return (center - margin) / denominator


def _split_team_names(
    split: Mapping[str, Any],
    field_name: str,
    episodes: int,
) -> frozenset[str]:
    team_counts = split.get("team_counts")
    if not isinstance(team_counts, Mapping):
        raise ValueError(f"{field_name}.team_counts must be an object")
    total = 0
    names: set[str] = set()
    for team_name, raw_count in team_counts.items():
        if not isinstance(team_name, str) or not team_name.strip():
            raise ValueError(f"{field_name}.team_counts has an invalid team name")
        count = _positive_int(
            raw_count, f"{field_name}.team_counts[{team_name!r}]"
        )
        total += count
        names.add(team_name)
    if total != episodes:
        raise ValueError(
            f"{field_name}.team_counts sums to {total}, expected {episodes}"
        )
    return frozenset(names)


def _split_dates(
    split: Mapping[str, Any],
    field_name: str,
    episodes: int,
) -> frozenset[str]:
    date_counts = split.get("date_counts")
    if not isinstance(date_counts, Mapping):
        raise ValueError(f"{field_name}.date_counts must be an object")
    total = 0
    dates: set[str] = set()
    for dataset_date, raw_count in date_counts.items():
        if not isinstance(dataset_date, str):
            raise ValueError(f"{field_name}.date_counts has a non-string date")
        try:
            date.fromisoformat(dataset_date)
        except ValueError as error:
            raise ValueError(
                f"{field_name}.date_counts has an invalid ISO date"
            ) from error
        count = _positive_int(
            raw_count, f"{field_name}.date_counts[{dataset_date!r}]"
        )
        total += count
        dates.add(dataset_date)
    if total != episodes:
        raise ValueError(
            f"{field_name}.date_counts sums to {total}, expected {episodes}"
        )
    return frozenset(dates)


def _profile_window_dates(
    profile: Mapping[str, Any],
    split_name: str,
) -> frozenset[str]:
    split_date_counts = profile.get("split_date_counts")
    if not isinstance(split_date_counts, Mapping):
        raise ValueError("Profile split_date_counts must be an object")
    raw_counts = split_date_counts.get(split_name)
    if not isinstance(raw_counts, Mapping):
        raise ValueError(
            f"Profile split_date_counts.{split_name} must be an object"
        )
    dates: set[str] = set()
    for dataset_date, raw_count in raw_counts.items():
        if not isinstance(dataset_date, str):
            raise ValueError(
                f"split_date_counts.{split_name} has a non-string date"
            )
        try:
            date.fromisoformat(dataset_date)
        except ValueError as error:
            raise ValueError(
                f"split_date_counts.{split_name} has an invalid ISO date"
            ) from error
        _positive_int(
            raw_count,
            f"split_date_counts.{split_name}[{dataset_date!r}]",
        )
        dates.add(dataset_date)
    return frozenset(dates)


def parse_candidate(
    deck: Mapping[str, Any],
    policy: SelectionPolicy,
) -> Candidate:
    deck_hash = _deck_hash(deck.get("deck_hash"), "decks[].deck_hash")
    split_stats = deck.get("split_stats")
    if not isinstance(split_stats, Mapping):
        raise ValueError(f"{deck_hash}: split_stats must be an object")

    # Deliberately name only train and valid.  Do not iterate split_stats: a
    # test key may be present, absent, malformed, or backed by a sealed guard.
    train = split_stats.get("train")
    valid = split_stats.get("valid")
    if not isinstance(train, Mapping) or not isinstance(valid, Mapping):
        raise ValueError(f"{deck_hash}: train and valid split_stats are required")

    train_episodes = _nonnegative_int(
        train.get("episodes"), f"{deck_hash}.train.episodes"
    )
    valid_episodes = _nonnegative_int(
        valid.get("episodes"), f"{deck_hash}.valid.episodes"
    )
    valid_wins = _nonnegative_int(
        valid.get("wins"), f"{deck_hash}.valid.wins"
    )
    valid_losses = _nonnegative_int(
        valid.get("losses"), f"{deck_hash}.valid.losses"
    )
    valid_draws = _nonnegative_int(
        valid.get("draws"), f"{deck_hash}.valid.draws"
    )
    if valid_wins + valid_losses + valid_draws != valid_episodes:
        raise ValueError(
            f"{deck_hash}: valid outcomes do not sum to valid episodes"
        )
    train_teams = _split_team_names(
        train, f"{deck_hash}.train", train_episodes
    )
    valid_teams = _split_team_names(
        valid, f"{deck_hash}.valid", valid_episodes
    )
    train_dates = _split_dates(
        train, f"{deck_hash}.train", train_episodes
    )
    valid_dates = _split_dates(
        valid, f"{deck_hash}.valid", valid_episodes
    )
    if not train_dates.issubset(policy.train_dates):
        raise ValueError(f"{deck_hash}: train rows fall outside preregistered dates")
    if not valid_dates.issubset(policy.valid_dates):
        raise ValueError(f"{deck_hash}: valid rows fall outside preregistered dates")
    lower_bound = (
        wilson_lower_bound(valid_wins, valid_episodes, policy.wilson_z)
        if valid_episodes
        else 0.0
    )
    return Candidate(
        deck_hash=deck_hash,
        train_episodes=train_episodes,
        valid_episodes=valid_episodes,
        valid_wins=valid_wins,
        valid_losses=valid_losses,
        valid_draws=valid_draws,
        train_valid_teams=tuple(sorted(train_teams | valid_teams)),
        valid_wilson_lower_bound=lower_bound,
        train_dates=tuple(sorted(train_dates)),
        valid_dates=tuple(sorted(valid_dates)),
    )


def select_routes(
    preregistration: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    policy = parse_policy(preregistration)
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("Unexpected exact-deck profile schema_version")
    if profile.get("deduplication_key") != [
        "dataset_date",
        "episode_id",
        "seat",
    ]:
        raise ValueError("Profile has an unexpected deduplication_key")
    profiled_splits = profile.get("profiled_splits")
    sealed_splits = profile.get("sealed_splits")
    if not isinstance(profiled_splits, list) or not isinstance(
        sealed_splits, list
    ):
        raise ValueError("Profile must declare profiled_splits and sealed_splits")
    if (
        any(split not in ("train", "valid", "test") for split in profiled_splits)
        or any(split not in ("train", "valid", "test") for split in sealed_splits)
        or len(set(profiled_splits)) != len(profiled_splits)
        or len(set(sealed_splits)) != len(sealed_splits)
        or set(profiled_splits) & set(sealed_splits)
        or set(profiled_splits) | set(sealed_splits)
        != {"train", "valid", "test"}
    ):
        raise ValueError("Profile split seal declarations are inconsistent")
    if not {"train", "valid"}.issubset(profiled_splits):
        raise ValueError("Profile must open train and valid for route selection")
    profile_train_dates = _profile_window_dates(profile, "train")
    profile_valid_dates = _profile_window_dates(profile, "valid")
    if profile_train_dates != policy.train_dates:
        raise ValueError(
            "Profile train date window does not exactly match preregistration"
        )
    if profile_valid_dates != policy.valid_dates:
        raise ValueError(
            "Profile valid date window does not exactly match preregistration"
        )
    raw_decks = profile.get("decks")
    if not isinstance(raw_decks, list):
        raise ValueError("Profile decks must be a list")

    decks_by_hash: dict[str, Mapping[str, Any]] = {}
    for index, raw_deck in enumerate(raw_decks):
        if not isinstance(raw_deck, Mapping):
            raise ValueError(f"decks[{index}] must be an object")
        deck_hash = _deck_hash(raw_deck.get("deck_hash"), f"decks[{index}].deck_hash")
        if deck_hash in decks_by_hash:
            raise ValueError(f"Duplicate deck_hash in profile: {deck_hash}")
        decks_by_hash[deck_hash] = raw_deck
    if policy.marnie_deck_hash not in decks_by_hash:
        raise ValueError("The preregistered Marnie deck is absent from the profile")

    fixed_exclusions = set(policy.excluded_deck_hashes)
    fixed_exclusions.add(policy.marnie_deck_hash)
    candidates: list[Candidate] = []
    for deck_hash, raw_deck in decks_by_hash.items():
        if deck_hash in fixed_exclusions:
            continue
        candidate = parse_candidate(raw_deck, policy)
        if (
            candidate.valid_episodes >= policy.minimum_valid_episode_seats
            and len(candidate.train_valid_teams)
            >= policy.minimum_train_valid_teams
        ):
            candidates.append(candidate)

    by_frequency = sorted(
        candidates,
        key=lambda candidate: (-candidate.frequency, candidate.deck_hash),
    )
    selected: list[dict[str, Any]] = [
        {
            "route": 1,
            "deck_hash": policy.marnie_deck_hash,
            "reason": (
                "Fixed Marnie route required by the preregistration; no "
                "train, valid, or test outcome was used to rank this route."
            ),
        }
    ]
    unfilled: list[dict[str, Any]] = []
    route_2: Candidate | None = None
    route_3_pool: list[Candidate] = []

    if policy.maximum_bc_routes >= 2:
        if by_frequency:
            route_2 = by_frequency[0]
            selected.append(
                {
                    "route": 2,
                    "deck_hash": route_2.deck_hash,
                    "reason": (
                        "Highest train+valid episode-seat frequency among "
                        "eligible decks after fixed exclusions; exact ties "
                        "use ascending deck_hash."
                    ),
                    "selection_metrics": route_2.public_metrics(),
                }
            )
        else:
            unfilled.append(
                {
                    "route": 2,
                    "reason": "No non-excluded deck passed the eligibility gate.",
                }
            )

    if policy.maximum_bc_routes >= 3:
        remaining = [
            candidate
            for candidate in by_frequency
            if route_2 is None or candidate.deck_hash != route_2.deck_hash
        ]
        route_3_pool = remaining[: policy.route_3_pool_size]
        if route_3_pool:
            route_3 = sorted(
                route_3_pool,
                key=lambda candidate: (
                    -candidate.valid_wilson_lower_bound,
                    -candidate.frequency,
                    candidate.deck_hash,
                ),
            )[0]
            selected.append(
                {
                    "route": 3,
                    "deck_hash": route_3.deck_hash,
                    "reason": (
                        "Highest valid-only 95% Wilson lower bound in the "
                        "preregistered top-frequency remaining pool; draws are "
                        "included in n. Wilson ties use frequency, then "
                        "ascending deck_hash."
                    ),
                    "selection_metrics": route_3.public_metrics(),
                }
            )
        else:
            unfilled.append(
                {
                    "route": 3,
                    "reason": "No eligible deck remained after route 2.",
                }
            )

    return {
        "schema_version": OUTPUT_SCHEMA,
        "selection_policy": {
            "minimum_valid_episode_seats": policy.minimum_valid_episode_seats,
            "minimum_train_valid_unique_teams": (
                policy.minimum_train_valid_teams
            ),
            "frequency_metric": "train_plus_valid_episode_seats",
            "route_3_pool_size": policy.route_3_pool_size,
            "wilson_confidence": policy.wilson_confidence,
            "wilson_z": policy.wilson_z,
            "draws_in_wilson_denominator": True,
            "maximum_bc_routes": policy.maximum_bc_routes,
            "train_dates": sorted(policy.train_dates),
            "valid_dates": sorted(policy.valid_dates),
            "frequency_tie_break": "deck_hash_ascending",
            "wilson_tie_break": (
                "train_plus_valid_episode_seats_descending_then_"
                "deck_hash_ascending"
            ),
        },
        "selection_fields_used": {
            "preregistration": PREREGISTRATION_SELECTION_FIELDS,
            "exact_deck_profile": PROFILE_SELECTION_FIELDS,
        },
        "test_data_firewall": {
            "profile_test_fields_accessed": False,
            "profile_test_fields_compared": False,
            "profile_wide_outcome_totals_accessed": False,
            "profile_declared_test_sealed": "test" in sealed_splits,
            "note": (
                "The whole profile file is hashed for input identity, but test "
                "payload values are not accessed or compared by selection."
            ),
        },
        "eligible_candidate_count": len(by_frequency),
        "route_3_pool": [
            {
                "deck_hash": candidate.deck_hash,
                "train_plus_valid_episode_seats": candidate.frequency,
                "valid_wilson_95_lower_bound": (
                    candidate.valid_wilson_lower_bound
                ),
            }
            for candidate in route_3_pool
        ],
        "selected_routes": selected,
        "unfilled_routes": unfilled,
    }


def atomic_write_json(
    output_path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists; pass --overwrite to replace it"
        )
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    installed = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"{output_path} exists; pass --overwrite to replace it"
                ) from error
            temporary_path.unlink()
        installed = True
    finally:
        if not installed and temporary_path.exists():
            temporary_path.unlink()


def build_report(
    preregistration_path: Path,
    profile_path: Path,
) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve()
    profile_path = profile_path.resolve()
    if not preregistration_path.is_file():
        raise FileNotFoundError(preregistration_path)
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    preregistration, preregistration_sha256 = load_json_with_sha256(
        preregistration_path
    )
    profile, profile_sha256 = load_json_with_sha256(profile_path)
    report = select_routes(preregistration, profile)
    report["inputs"] = {
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": preregistration_sha256,
            "schema_version": preregistration.get("schema_version"),
        },
        "exact_deck_profile": {
            "path": str(profile_path),
            "sha256": profile_sha256,
            "schema_version": profile.get("schema_version"),
        },
    }
    return report


def run(
    preregistration_path: Path,
    profile_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    resolved_output = output_path.resolve()
    if resolved_output in {
        preregistration_path.resolve(),
        profile_path.resolve(),
    }:
        raise ValueError("--output must differ from both inputs")
    if resolved_output.exists() and not overwrite:
        raise FileExistsError(
            f"{resolved_output} exists; pass --overwrite to replace it"
        )
    report = build_report(preregistration_path, profile_path)
    atomic_write_json(resolved_output, report, overwrite=overwrite)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        args.preregistration,
        args.profile,
        args.output,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selected_routes": [
                    item["deck_hash"] for item in report["selected_routes"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
