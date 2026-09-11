#!/usr/bin/env python3
"""Audit the frozen parent/S4/S8 six-policy endpoint screen.

This is a read-only evidence consumer.  It validates the three runner outputs
and their raw evaluator results, then recomputes the preregistered parent-drift,
Newcombe-Wilson non-inferiority, and point-estimate gates.  It never runs
battles, trains, packages, uploads, or submits.  The only optional side effect
is the atomically installed decision JSON requested with ``--output``.

The official engine does not expose RNG seed control.  Matching runner seeds
therefore establish a matched execution protocol, not paired trials; all
difference intervals below treat endpoint and parent outcomes as independent
binomial samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import audit_gold_push_panel as panel


REPORT_SCHEMA = "ptcg-gold-push-endpoint-screen-audit-v1"

# Frozen Stage-A identities in panel-manifest order.  CLI --policy-id ordering
# is intentionally not trusted because run_gold_league_h2h selects in manifest
# order.
POLICY_IDS = (
    "marnie_source_bc_aug08",
    "marnie_raihan_recent7_clone",
    "marnie_kanto_recent7_clone",
    "marnie_kdcyberdude_recent7_clone",
    "froslass_lopunny_source_bc_aug08",
    "froslass_lopunny_recent7_uptake_bc",
)
BOTTOM4_POLICY_IDS = frozenset(
    {
        "marnie_source_bc_aug08",
        "marnie_kanto_recent7_clone",
        "marnie_kdcyberdude_recent7_clone",
        "froslass_lopunny_recent7_uptake_bc",
    }
)
TARGET_POLICY_IDS = (
    "marnie_kdcyberdude_recent7_clone",
    "marnie_raihan_recent7_clone",
)
OTHER_POLICY_IDS = frozenset(POLICY_IDS) - frozenset(TARGET_POLICY_IDS)

GAMES_PER_POLICY = 128
GAMES_PER_POLICY_PER_SEAT = 64
EXPECTED_SEED = 202608211
EXPECTED_ENVIRONMENTS = 32
EXPECTED_MAX_GAME_DECISIONS = 1000
EXPECTED_DEVICE = "cuda"
EXPECTED_RUNNER_CVAR_ALPHA = 0.25
NEWCOMBE_Z = 1.959963984540054  # one-sided 97.5% per endpoint

EXPECTED_EVALUATION = {
    "screening_games": GAMES_PER_POLICY,
    "confirmation": "none",
    "confirmation_games": 256,
    "environments": EXPECTED_ENVIRONMENTS,
    "max_game_decisions": EXPECTED_MAX_GAME_DECISIONS,
    "candidate_order_mode": "hybrid",
    "candidate_raw_order": False,
    "candidate_canonical_order": False,
    "candidate_hybrid_order": True,
    "opponent_canonical_order_source": "per_opponent_manifest",
    "legacy_opponent_canonical_order_all_true_assertion": False,
    "seed": EXPECTED_SEED,
    "device": EXPECTED_DEVICE,
}
EXPECTED_RUNNER_GATES = {
    "cvar_alpha": EXPECTED_RUNNER_CVAR_ALPHA,
    "screen_min_policy_win_rate": 0.0,
    "screen_min_policy_wilson_low": 0.0,
    "screen_min_macro_win_rate": 0.0,
    "screen_min_cvar_win_rate": 0.0,
    "promotion_min_policies": len(POLICY_IDS),
    "promotion_min_policy_wilson_low": 0.0,
    "promotion_min_archetype_wilson_low": 0.0,
    "promotion_min_seat_wilson_low": 0.0,
    "promotion_min_macro_win_rate": 0.0,
    "promotion_min_policy_win_rate": 0.0,
    "promotion_min_cvar_win_rate": 0.0,
}

# Historical parent calibration from the frozen 16-policy hybrid run, reduced
# to the six Stage-A policies and the preregistered bottom four.
HISTORICAL_PARENT = {
    "pooled": {"wins": 782, "trials": 1536, "maximum_absolute_delta": 0.06},
    "bottom4": {"wins": 511, "trials": 1024, "maximum_absolute_delta": 0.075},
    "seat0": {"wins": 404, "trials": 768, "maximum_absolute_delta": 0.10},
    "seat1": {"wins": 378, "trials": 768, "maximum_absolute_delta": 0.10},
}
NONINFERIORITY_MARGINS = {
    "pooled": -0.050,
    "bottom4": -0.0625,
    "seat0": -0.075,
    "seat1": -0.075,
}
MINIMUM_POOLED_WIN_GAIN = 4
MINIMUM_BOTTOM4_WIN_GAIN = 4
MINIMUM_TARGET_COMBINED_WIN_GAIN = 2
MINIMUM_TARGET_POLICY_WIN_DELTA = 0
MINIMUM_OTHER_POLICY_WIN_DELTA = -8


@dataclass(frozen=True)
class RunEvidence:
    label: str
    run_dir: Path
    run_config_path: Path
    run_config: dict[str, Any]
    summary_path: Path
    summary: dict[str, Any]
    summary_sha256: str
    candidate: Path
    candidate_sha256: str
    candidate_deck: Path
    candidate_deck_file_sha256: str
    counts: tuple[panel.PolicyCounts, ...]
    result_paths: Mapping[str, Path]
    result_sha256: Mapping[str, str]
    result_times: Mapping[str, tuple[datetime, datetime]]

    @property
    def by_policy(self) -> dict[str, panel.PolicyCounts]:
        return {row.policy_id: row for row in self.counts}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    """Atomically install JSON, refusing to clobber evidence by default."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise panel.AuditError(
            f"output already exists (pass --overwrite explicitly): {path}"
        )
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        if path.exists() and not overwrite:
            raise panel.AuditError(
                f"output appeared during audit; refusing to overwrite: {path}"
            )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_timestamp(value: Any, label: str) -> datetime:
    raw = panel.require_text(value, label)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise panel.AuditError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is None:
        raise panel.AuditError(f"{label} must include a timezone")
    return parsed


def require_actual_sha256(
    path: Path, expected_sha256: str, label: str
) -> str:
    expected = panel.require_sha256(expected_sha256, f"{label} expected SHA-256")
    if not path.is_file():
        raise panel.AuditError(f"{label} does not exist: {path}")
    actual = panel.file_sha256(path)
    panel.require_equal(actual, expected, f"{label} actual SHA-256")
    return actual


def validate_runner_signature(run_config: Mapping[str, Any]) -> str:
    signature = panel.require_sha256(
        run_config.get("run_signature"), "run.run_signature"
    )
    non_signature_fields = {
        "created_at_utc",
        "run_signature",
        "jobs",
        "wall_seconds_per_invocation",
        "local_only",
        "uploads_or_submissions_performed",
    }
    payload = {
        key: value
        for key, value in run_config.items()
        if key not in non_signature_fields
    }
    panel.require_equal(
        canonical_sha256(payload), signature, "recomputed runner signature"
    )
    return signature


def validate_screen_only_run_identity(
    *,
    label: str,
    run_dir: Path,
    expected_summary_sha256: str,
    candidate: Path,
    expected_candidate_sha256: str,
    candidate_deck: Path,
    expected_candidate_deck_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    opponents: Sequence[panel.Opponent],
) -> RunEvidence:
    run_dir = run_dir.expanduser().resolve()
    candidate = candidate.expanduser().resolve()
    candidate_deck = candidate_deck.expanduser().resolve()
    if not run_dir.is_dir():
        raise panel.AuditError(f"{label} run directory does not exist: {run_dir}")

    summary_path = (run_dir / "summary.json").resolve()
    summary_sha = require_actual_sha256(
        summary_path, expected_summary_sha256, f"{label} summary"
    )
    candidate_sha = require_actual_sha256(
        candidate, expected_candidate_sha256, f"{label} candidate"
    )
    candidate_deck_sha = require_actual_sha256(
        candidate_deck,
        expected_candidate_deck_sha256,
        "shared candidate deck",
    )

    run_config_path = (run_dir / "run_config.json").resolve()
    run_config = panel.read_json(run_config_path, f"{label} run config")
    summary = panel.read_json(summary_path, f"{label} summary")
    panel.require_equal(
        run_config.get("schema_version"), panel.RUN_SCHEMA, f"{label} run schema"
    )
    panel.require_equal(
        summary.get("schema_version"),
        panel.SUMMARY_SCHEMA,
        f"{label} summary schema",
    )
    signature = validate_runner_signature(run_config)
    panel.require_equal(
        summary.get("run_signature"), signature, f"{label} summary signature"
    )

    for document_name, document, base in (
        ("run", run_config, run_config_path.parent),
        ("summary", summary, summary_path.parent),
    ):
        panel.require_equal(
            panel.resolve_path(
                document.get("league_manifest"),
                base,
                f"{label} {document_name}.league_manifest",
            ),
            manifest_path,
            f"{label} {document_name}.league_manifest",
        )
    panel.require_equal(
        run_config.get("league_manifest_sha256"),
        manifest_sha256,
        f"{label} manifest SHA-256 binding",
    )

    panel.require_equal(
        panel.resolve_path(
            run_config.get("candidate"),
            run_config_path.parent,
            f"{label} run.candidate",
        ),
        candidate,
        f"{label} run.candidate",
    )
    panel.require_equal(
        run_config.get("candidate_sha256"),
        candidate_sha,
        f"{label} run.candidate_sha256",
    )
    summary_candidate = panel.require_mapping(
        summary.get("candidate"), f"{label} summary.candidate"
    )
    panel.require_equal(
        panel.resolve_path(
            summary_candidate.get("checkpoint"),
            summary_path.parent,
            f"{label} summary.candidate.checkpoint",
        ),
        candidate,
        f"{label} summary candidate path",
    )
    panel.require_equal(
        summary_candidate.get("checkpoint_sha256"),
        candidate_sha,
        f"{label} summary candidate SHA-256",
    )

    for document_name, document, base, field in (
        ("run", run_config, run_config_path.parent, "candidate_deck"),
        ("summary", summary_candidate, summary_path.parent, "deck"),
    ):
        panel.require_equal(
            panel.resolve_path(
                document.get(field), base, f"{label} {document_name}.{field}"
            ),
            candidate_deck,
            f"{label} {document_name} candidate deck",
        )
    panel.require_equal(
        run_config.get("candidate_deck_file_sha256"),
        candidate_deck_sha,
        f"{label} run candidate deck SHA-256",
    )
    panel.require_equal(
        summary_candidate.get("deck_file_sha256"),
        candidate_deck_sha,
        f"{label} summary candidate deck SHA-256",
    )
    panel.deck_semantic_hash(candidate_deck)

    selected = panel.expected_selected_opponents(opponents)
    panel.require_equal(
        run_config.get("selected_opponents"),
        selected,
        f"{label} selected opponent identities/order",
    )
    panel.require_equal(
        summary.get("selected_policy_ids"),
        list(POLICY_IDS),
        f"{label} summary policy identities/order",
    )
    if not all(opponent.canonical_order for opponent in opponents):
        raise panel.AuditError("all six frozen Stage-A opponents must be canonical")

    evaluation = panel.require_mapping(
        run_config.get("evaluation"), f"{label} run.evaluation"
    )
    panel.require_equal(
        evaluation, EXPECTED_EVALUATION, f"{label} frozen evaluation protocol"
    )
    order_contract = panel.candidate_order_contract(
        evaluation, f"{label} run.evaluation"
    )
    panel.require_equal(order_contract.order_mode, "hybrid", f"{label} order")
    panel.require_equal(
        summary_candidate.get("order_mode"),
        "hybrid",
        f"{label} summary order mode",
    )
    panel.require_equal(run_config.get("jobs"), 1, f"{label} jobs")
    panel.require_equal(
        run_config.get("engine_seed_control"), False, f"{label} engine seed"
    )
    panel.require_equal(
        run_config.get("gates"), EXPECTED_RUNNER_GATES, f"{label} runner gates"
    )
    panel.require_equal(
        run_config.get("deployment_contract"),
        None,
        f"{label} deployment contract must be absent for mini-screen",
    )
    panel.require_equal(
        summary.get("deployment_contract"),
        None,
        f"{label} summary deployment contract",
    )

    for document_name, document in (("run", run_config), ("summary", summary)):
        panel.require_equal(
            document.get("local_only"), True, f"{label} {document_name}.local_only"
        )
        panel.require_equal(
            document.get("uploads_or_submissions_performed"),
            False,
            f"{label} {document_name} side effects",
        )

    bc_path = panel.resolve_path(
        run_config.get("bc_checkpoint"),
        run_config_path.parent,
        f"{label} run.bc_checkpoint",
    )
    bc_sha = panel.require_sha256(
        run_config.get("bc_checkpoint_sha256"),
        f"{label} run.bc_checkpoint_sha256",
    )
    require_actual_sha256(bc_path, bc_sha, f"{label} BC checkpoint")
    evaluator_path = panel.resolve_path(
        run_config.get("evaluator"),
        run_config_path.parent,
        f"{label} run.evaluator",
    )
    evaluator_sha = panel.require_sha256(
        run_config.get("evaluator_sha256"), f"{label} run.evaluator_sha256"
    )
    require_actual_sha256(evaluator_path, evaluator_sha, f"{label} evaluator")

    summary_engine = panel.require_mapping(
        summary.get("engine"), f"{label} summary.engine"
    )
    panel.require_equal(
        summary_engine.get("engine_seed_control"),
        False,
        f"{label} summary engine seed control",
    )
    panel.require_equal(
        summary_engine.get("strict_even_valid_games_by_candidate_seat_required"),
        True,
        f"{label} strict seat requirement",
    )
    confirmation = panel.require_mapping(
        summary.get("confirmation"), f"{label} summary.confirmation"
    )
    panel.require_equal(confirmation.get("mode"), "none", f"{label} confirmation")
    panel.require_equal(
        confirmation.get("summary"), None, f"{label} confirmation summary"
    )
    panel.require_equal(
        summary.get("final_evidence_phase"), "screening", f"{label} evidence phase"
    )

    screening = panel.require_mapping(
        summary.get("screening"), f"{label} summary.screening"
    )
    expected_screening_fields = {
        "phase": "screening",
        "games_requested_per_policy": GAMES_PER_POLICY,
        "selected_policies": len(POLICY_IDS),
        "completed_policies": len(POLICY_IDS),
        "coverage_complete": True,
        "strict_even_seat_balance_verified": True,
        "failures": {},
        "wall_budget_exhausted": False,
    }
    for field, expected in expected_screening_fields.items():
        panel.require_equal(
            screening.get(field), expected, f"{label} screening.{field}"
        )

    results_dir = run_dir / "results" / "screening"
    if not results_dir.is_dir():
        raise panel.AuditError(f"{label} result directory missing: {results_dir}")
    result_paths = {
        opponent.policy_id: (
            results_dir / f"{panel.safe_name(opponent.policy_id)}.json"
        ).resolve()
        for opponent in opponents
    }
    panel.require_equal(
        {path.resolve() for path in results_dir.glob("*.json")},
        set(result_paths.values()),
        f"{label} exact raw result coverage",
    )
    counts: list[panel.PolicyCounts] = []
    result_shas: dict[str, str] = {}
    result_times: dict[str, tuple[datetime, datetime]] = {}
    deck_semantic_sha = panel.deck_semantic_hash(candidate_deck)
    for opponent in opponents:
        policy_id = opponent.policy_id
        result_path = result_paths[policy_id]
        result = panel.read_json(result_path, f"{label} result[{policy_id}]")
        counts.append(
            panel.validate_result(
                path=result_path,
                result=result,
                opponent=opponent,
                candidate=candidate,
                candidate_sha256=candidate_sha,
                candidate_deck=candidate_deck,
                candidate_deck_hash=deck_semantic_sha,
                run_config=run_config,
                games_per_policy=GAMES_PER_POLICY,
            )
        )
        started = parse_timestamp(
            result.get("started_at_utc"), f"{label} result[{policy_id}] start"
        )
        finished = parse_timestamp(
            result.get("finished_at_utc"), f"{label} result[{policy_id}] finish"
        )
        if finished < started:
            raise panel.AuditError(f"{label} result[{policy_id}] finishes before start")
        result_times[policy_id] = (started, finished)
        result_shas[policy_id] = panel.file_sha256(result_path)
    panel.validate_summary_counts(screening, counts, result_paths)

    return RunEvidence(
        label=label,
        run_dir=run_dir,
        run_config_path=run_config_path,
        run_config=run_config,
        summary_path=summary_path,
        summary=summary,
        summary_sha256=summary_sha,
        candidate=candidate,
        candidate_sha256=candidate_sha,
        candidate_deck=candidate_deck,
        candidate_deck_file_sha256=candidate_deck_sha,
        counts=tuple(counts),
        result_paths=result_paths,
        result_sha256=result_shas,
        result_times=result_times,
    )


def validate_shared_protocol(runs: Sequence[RunEvidence]) -> dict[str, Any]:
    if len(runs) != 3:
        raise panel.AuditError("exactly three Stage-A runs are required")
    panel.require_equal(
        len({run.run_dir for run in runs}), 3, "distinct Stage-A run directories"
    )
    panel.require_equal(
        len({run.summary_sha256 for run in runs}),
        3,
        "distinct Stage-A summary identities",
    )
    panel.require_equal(
        len({run.candidate_sha256 for run in runs}),
        3,
        "distinct parent/S4/S8 candidate identities",
    )
    panel.require_equal(
        len({run.run_config["run_signature"] for run in runs}),
        3,
        "distinct runner signatures",
    )

    shared_fields = (
        "league_manifest",
        "league_manifest_sha256",
        "candidate_deck",
        "candidate_deck_file_sha256",
        "bc_checkpoint",
        "bc_checkpoint_sha256",
        "evaluator",
        "evaluator_sha256",
        "selected_opponents",
        "evaluation",
        "gates",
        "jobs",
        "engine_seed_control",
        "deployment_contract",
    )
    reference = {field: runs[0].run_config.get(field) for field in shared_fields}
    for run in runs[1:]:
        panel.require_equal(
            {field: run.run_config.get(field) for field in shared_fields},
            reference,
            f"{run.label} shared runner protocol",
        )

    start_skews: dict[str, float] = {}
    for policy_id in POLICY_IDS:
        starts = [run.result_times[policy_id][0] for run in runs]
        start_skews[policy_id] = (max(starts) - min(starts)).total_seconds()
    sequence_by_run = {
        run.label: [
            policy_id
            for policy_id, _ in sorted(
                run.result_times.items(), key=lambda item: item[1][0]
            )
        ]
        for run in runs
    }
    sequence_matches = all(
        sequence == list(POLICY_IDS) for sequence in sequence_by_run.values()
    )
    return {
        "pass": True,
        "same_runner_protocol": True,
        "distinct_run_directories": True,
        "distinct_run_signatures": True,
        "same_manifest_policy_order": True,
        "same_base_and_derived_python_torch_seeds": True,
        "engine_seed_control": False,
        "statistical_treatment": "independent_binomial_not_paired",
        "observed_execution_sequence_matches_manifest_order": sequence_matches,
        "execution_sequence_by_run": sequence_by_run,
        "per_policy_start_skew_seconds": start_skews,
        "maximum_per_policy_start_skew_seconds": max(start_skews.values()),
        "wall_clock_interleaving_is_descriptive_not_a_selection_gate": True,
    }


def aggregate_counts(
    run: RunEvidence,
    *,
    policy_ids: Iterable[str] = POLICY_IDS,
    seat: str | None = None,
) -> tuple[int, int]:
    selected = frozenset(policy_ids)
    wins = 0
    trials = 0
    for row in run.counts:
        if row.policy_id not in selected:
            continue
        if seat is None:
            wins += row.wins
            trials += row.attempted_games
        else:
            wins += row.seat_wins[seat]
            trials += row.seat_valid_games[seat] + row.invalid_by_seat[seat]
    if trials <= 0:
        raise panel.AuditError("aggregate comparison has no trials")
    return wins, trials


def wilson_interval(
    successes: int, trials: int, *, z: float = NEWCOMBE_Z
) -> tuple[float, float]:
    if trials <= 0:
        raise panel.AuditError("Wilson interval requires positive trials")
    if not 0 <= successes <= trials:
        raise panel.AuditError("Wilson successes must be in [0, trials]")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - margin, center + margin


def newcombe_lower_bound(
    endpoint_wins: int,
    endpoint_trials: int,
    parent_wins: int,
    parent_trials: int,
    *,
    z: float = NEWCOMBE_Z,
) -> dict[str, Any]:
    endpoint_rate = endpoint_wins / endpoint_trials
    parent_rate = parent_wins / parent_trials
    endpoint_low, endpoint_high = wilson_interval(
        endpoint_wins, endpoint_trials, z=z
    )
    parent_low, parent_high = wilson_interval(parent_wins, parent_trials, z=z)
    difference = endpoint_rate - parent_rate
    lower = difference - math.sqrt(
        (endpoint_rate - endpoint_low) ** 2
        + (parent_high - parent_rate) ** 2
    )
    return {
        "endpoint": {
            "wins": endpoint_wins,
            "trials": endpoint_trials,
            "win_rate": endpoint_rate,
            "wilson_low": endpoint_low,
            "wilson_high": endpoint_high,
        },
        "parent": {
            "wins": parent_wins,
            "trials": parent_trials,
            "win_rate": parent_rate,
            "wilson_low": parent_low,
            "wilson_high": parent_high,
        },
        "point_difference": difference,
        "newcombe_lower_bound": lower,
        "z": z,
    }


def audit_zero_invalid(run: RunEvidence) -> dict[str, Any]:
    invalid_by_policy = {
        row.policy_id: row.invalid_games for row in run.counts
    }
    total = sum(invalid_by_policy.values())
    return {
        "invalid_games": total,
        "invalid_games_by_policy": invalid_by_policy,
        "maximum": 0,
        "pass": total == 0,
    }


def audit_parent_drift(parent: RunEvidence) -> dict[str, Any]:
    current = {
        "pooled": aggregate_counts(parent),
        "bottom4": aggregate_counts(parent, policy_ids=BOTTOM4_POLICY_IDS),
        "seat0": aggregate_counts(parent, seat="0"),
        "seat1": aggregate_counts(parent, seat="1"),
    }
    gates: dict[str, Any] = {}
    for group, (wins, trials) in current.items():
        historical = HISTORICAL_PARENT[group]
        actual_rate = wins / trials
        historical_rate = historical["wins"] / historical["trials"]
        absolute_delta = abs(actual_rate - historical_rate)
        maximum = historical["maximum_absolute_delta"]
        gates[group] = {
            "current": {"wins": wins, "trials": trials, "win_rate": actual_rate},
            "historical": {
                "wins": historical["wins"],
                "trials": historical["trials"],
                "win_rate": historical_rate,
            },
            "absolute_rate_delta": absolute_delta,
            "maximum_absolute_rate_delta": maximum,
            "pass": absolute_delta <= maximum,
        }
    zero_invalid = audit_zero_invalid(parent)
    return {
        "pass": zero_invalid["pass"] and all(row["pass"] for row in gates.values()),
        "zero_invalid": zero_invalid,
        "gates": gates,
    }


def endpoint_comparison(
    endpoint: RunEvidence,
    parent: RunEvidence,
    *,
    parent_drift_pass: bool,
) -> dict[str, Any]:
    groups = {
        "pooled": (POLICY_IDS, None),
        "bottom4": (BOTTOM4_POLICY_IDS, None),
        "seat0": (POLICY_IDS, "0"),
        "seat1": (POLICY_IDS, "1"),
    }
    noninferiority: dict[str, Any] = {}
    for group, (policy_ids, seat) in groups.items():
        endpoint_wins, endpoint_trials = aggregate_counts(
            endpoint, policy_ids=policy_ids, seat=seat
        )
        parent_wins, parent_trials = aggregate_counts(
            parent, policy_ids=policy_ids, seat=seat
        )
        row = newcombe_lower_bound(
            endpoint_wins, endpoint_trials, parent_wins, parent_trials
        )
        minimum = NONINFERIORITY_MARGINS[group]
        row.update(
            {
                "minimum_newcombe_lower_bound": minimum,
                "pass": row["newcombe_lower_bound"] >= minimum,
            }
        )
        noninferiority[group] = row

    endpoint_by_policy = endpoint.by_policy
    parent_by_policy = parent.by_policy
    policy_win_deltas = {
        policy_id: endpoint_by_policy[policy_id].wins
        - parent_by_policy[policy_id].wins
        for policy_id in POLICY_IDS
    }
    pooled_gain = aggregate_counts(endpoint)[0] - aggregate_counts(parent)[0]
    bottom4_gain = aggregate_counts(
        endpoint, policy_ids=BOTTOM4_POLICY_IDS
    )[0] - aggregate_counts(parent, policy_ids=BOTTOM4_POLICY_IDS)[0]
    target_combined_gain = sum(
        policy_win_deltas[policy_id] for policy_id in TARGET_POLICY_IDS
    )
    minimum_other_delta = min(
        policy_win_deltas[policy_id] for policy_id in OTHER_POLICY_IDS
    )
    point_estimate_gates = {
        "pooled_win_gain": {
            "actual": pooled_gain,
            "minimum": MINIMUM_POOLED_WIN_GAIN,
            "pass": pooled_gain >= MINIMUM_POOLED_WIN_GAIN,
        },
        "bottom4_win_gain": {
            "actual": bottom4_gain,
            "minimum": MINIMUM_BOTTOM4_WIN_GAIN,
            "pass": bottom4_gain >= MINIMUM_BOTTOM4_WIN_GAIN,
        },
        "target_combined_win_gain": {
            "policy_ids": list(TARGET_POLICY_IDS),
            "actual": target_combined_gain,
            "minimum": MINIMUM_TARGET_COMBINED_WIN_GAIN,
            "pass": target_combined_gain >= MINIMUM_TARGET_COMBINED_WIN_GAIN,
        },
        "kdcyberdude_win_delta": {
            "actual": policy_win_deltas[TARGET_POLICY_IDS[0]],
            "minimum": MINIMUM_TARGET_POLICY_WIN_DELTA,
            "pass": policy_win_deltas[TARGET_POLICY_IDS[0]]
            >= MINIMUM_TARGET_POLICY_WIN_DELTA,
        },
        "raihan_win_delta": {
            "actual": policy_win_deltas[TARGET_POLICY_IDS[1]],
            "minimum": MINIMUM_TARGET_POLICY_WIN_DELTA,
            "pass": policy_win_deltas[TARGET_POLICY_IDS[1]]
            >= MINIMUM_TARGET_POLICY_WIN_DELTA,
        },
        "minimum_other_policy_win_delta": {
            "policy_ids": sorted(OTHER_POLICY_IDS),
            "actual": minimum_other_delta,
            "minimum": MINIMUM_OTHER_POLICY_WIN_DELTA,
            "pass": minimum_other_delta >= MINIMUM_OTHER_POLICY_WIN_DELTA,
        },
    }
    endpoint_zero_invalid = audit_zero_invalid(endpoint)
    gates = {
        "parent_drift": parent_drift_pass,
        "endpoint_zero_invalid": endpoint_zero_invalid["pass"],
        **{
            f"{group}_newcombe_noninferiority": row["pass"]
            for group, row in noninferiority.items()
        },
        **{
            name: row["pass"] for name, row in point_estimate_gates.items()
        },
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    return {
        "pass": not failed,
        "failed_gates": failed,
        "zero_invalid": endpoint_zero_invalid,
        "newcombe_noninferiority": noninferiority,
        "policy_win_deltas_vs_parent": policy_win_deltas,
        "point_estimate_gates": point_estimate_gates,
        "ranking_values": {
            "bottom4_wins": aggregate_counts(
                endpoint, policy_ids=BOTTOM4_POLICY_IDS
            )[0],
            "pooled_wins": aggregate_counts(endpoint)[0],
            "minimum_policy_win_delta": min(policy_win_deltas.values()),
        },
    }


def select_endpoint(
    s04: Mapping[str, Any], s08: Mapping[str, Any]
) -> tuple[str | None, str]:
    passed = [label for label, result in (("s04", s04), ("s08", s08)) if result["pass"]]
    if not passed:
        return None, "no_endpoint_passed_all_frozen_gates"
    if len(passed) == 1:
        return passed[0], "only_endpoint_passing_all_frozen_gates"

    rank04 = panel.require_mapping(s04.get("ranking_values"), "s04 ranking")
    rank08 = panel.require_mapping(s08.get("ranking_values"), "s08 ranking")
    bottom_gap = abs(int(rank04["bottom4_wins"]) - int(rank08["bottom4_wins"]))
    pooled_gap = abs(int(rank04["pooled_wins"]) - int(rank08["pooled_wins"]))
    if bottom_gap <= 1 and pooled_gap <= 1:
        return "s04", "near_tie_prefers_smaller_s04_endpoint"
    key04 = (
        int(rank04["bottom4_wins"]),
        int(rank04["pooled_wins"]),
        int(rank04["minimum_policy_win_delta"]),
    )
    key08 = (
        int(rank08["bottom4_wins"]),
        int(rank08["pooled_wins"]),
        int(rank08["minimum_policy_win_delta"]),
    )
    if key04 == key08:
        return "s04", "exact_rank_tie_prefers_smaller_s04_endpoint"
    return ("s04", "lexicographic_frozen_ranking") if key04 > key08 else (
        "s08",
        "lexicographic_frozen_ranking",
    )


def evidence_public(run: RunEvidence) -> dict[str, Any]:
    return {
        "run_directory": str(run.run_dir),
        "run_config": str(run.run_config_path),
        "run_config_sha256": panel.file_sha256(run.run_config_path),
        "run_signature": run.run_config["run_signature"],
        "summary": str(run.summary_path),
        "summary_sha256": run.summary_sha256,
        "candidate_checkpoint": str(run.candidate),
        "candidate_checkpoint_sha256": run.candidate_sha256,
        "candidate_deck": str(run.candidate_deck),
        "candidate_deck_file_sha256": run.candidate_deck_file_sha256,
        "raw_results": {
            policy_id: {
                "path": str(run.result_paths[policy_id]),
                "sha256": run.result_sha256[policy_id],
            }
            for policy_id in POLICY_IDS
        },
    }


def audit_endpoint_screen(
    *,
    panel_manifest_path: Path,
    expected_panel_manifest_sha256: str,
    parent_run_dir: Path,
    s04_run_dir: Path,
    s08_run_dir: Path,
    parent_candidate: Path,
    s04_candidate: Path,
    s08_candidate: Path,
    expected_parent_candidate_sha256: str,
    expected_s04_candidate_sha256: str,
    expected_s08_candidate_sha256: str,
    candidate_deck: Path,
    expected_candidate_deck_sha256: str,
    expected_parent_summary_sha256: str,
    expected_s04_summary_sha256: str,
    expected_s08_summary_sha256: str,
) -> dict[str, Any]:
    manifest_path = panel_manifest_path.expanduser().resolve()
    manifest_sha = require_actual_sha256(
        manifest_path,
        expected_panel_manifest_sha256,
        "frozen panel manifest",
    )
    _, all_opponents = panel.load_opponents(manifest_path)
    by_id = {opponent.policy_id: opponent for opponent in all_opponents}
    if not set(POLICY_IDS) <= set(by_id):
        missing = sorted(set(POLICY_IDS) - set(by_id))
        raise panel.AuditError(f"panel manifest lacks Stage-A policies: {missing}")
    opponents = [by_id[policy_id] for policy_id in POLICY_IDS]

    shared_deck = candidate_deck.expanduser().resolve()
    run_specs = (
        (
            "parent",
            parent_run_dir,
            expected_parent_summary_sha256,
            parent_candidate,
            expected_parent_candidate_sha256,
        ),
        (
            "s04",
            s04_run_dir,
            expected_s04_summary_sha256,
            s04_candidate,
            expected_s04_candidate_sha256,
        ),
        (
            "s08",
            s08_run_dir,
            expected_s08_summary_sha256,
            s08_candidate,
            expected_s08_candidate_sha256,
        ),
    )
    runs = [
        validate_screen_only_run_identity(
            label=label,
            run_dir=run_dir,
            expected_summary_sha256=summary_sha,
            candidate=candidate,
            expected_candidate_sha256=candidate_sha,
            candidate_deck=shared_deck,
            expected_candidate_deck_sha256=expected_candidate_deck_sha256,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            opponents=opponents,
        )
        for label, run_dir, summary_sha, candidate, candidate_sha in run_specs
    ]
    evidence = {run.label: run for run in runs}
    shared_protocol = validate_shared_protocol(runs)
    parent_drift = audit_parent_drift(evidence["parent"])
    comparisons = {
        label: endpoint_comparison(
            evidence[label],
            evidence["parent"],
            parent_drift_pass=parent_drift["pass"],
        )
        for label in ("s04", "s08")
    }
    selected, reason = select_endpoint(comparisons["s04"], comparisons["s08"])
    passed = selected is not None
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": panel.utc_now(),
        "pass": passed,
        "audit_complete": True,
        "read_only_audit": True,
        "screening_only_not_submission_evidence": True,
        "training_evaluation_packaging_submission_performed": False,
        "inputs": {
            "panel_manifest": str(manifest_path),
            "panel_manifest_sha256": manifest_sha,
            "runs": {label: evidence_public(evidence[label]) for label in evidence},
        },
        "frozen_protocol": {
            "policy_ids_in_manifest_order": list(POLICY_IDS),
            "bottom4_policy_ids": sorted(BOTTOM4_POLICY_IDS),
            "target_policy_ids": list(TARGET_POLICY_IDS),
            "games_per_policy": GAMES_PER_POLICY,
            "games_per_policy_per_seat": GAMES_PER_POLICY_PER_SEAT,
            "candidate_order_mode": "hybrid",
            "newcombe": {
                "z": NEWCOMBE_Z,
                "per_endpoint_one_sided_confidence": 0.975,
                "two_endpoint_bonferroni_familywise_confidence": 0.95,
                "independent_binomial_samples": True,
                "noninferiority_margins": NONINFERIORITY_MARGINS,
            },
            "minimum_pooled_win_gain": MINIMUM_POOLED_WIN_GAIN,
            "minimum_bottom4_win_gain": MINIMUM_BOTTOM4_WIN_GAIN,
            "minimum_target_combined_win_gain": MINIMUM_TARGET_COMBINED_WIN_GAIN,
            "minimum_each_target_policy_win_delta": MINIMUM_TARGET_POLICY_WIN_DELTA,
            "minimum_other_policy_win_delta": MINIMUM_OTHER_POLICY_WIN_DELTA,
            "tie_break": (
                "bottom4 wins, pooled wins, minimum policy win delta; "
                "near/exact tie prefers S4"
            ),
        },
        "shared_protocol_and_interleaving": shared_protocol,
        "parent_historical_drift": parent_drift,
        "endpoint_comparisons": comparisons,
        "selected_endpoint": selected,
        "selection_reason": reason,
        "runner_zero_threshold_promotion_is_ignored": True,
        "decision": (
            f"endpoint_{selected}_passed_stage_a_requires_full_16x256_panel"
            if selected is not None
            else "reject_s04_and_s08_preserve_parent_incumbent"
        ),
        "submission_authorized_by_this_audit": False,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-manifest", type=Path, default=panel.DEFAULT_PANEL_MANIFEST
    )
    parser.add_argument("--panel-manifest-sha256", required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--s04-run-dir", type=Path, required=True)
    parser.add_argument("--s08-run-dir", type=Path, required=True)
    parser.add_argument("--parent-candidate", type=Path, required=True)
    parser.add_argument("--s04-candidate", type=Path, required=True)
    parser.add_argument("--s08-candidate", type=Path, required=True)
    parser.add_argument("--parent-candidate-sha256", required=True)
    parser.add_argument("--s04-candidate-sha256", required=True)
    parser.add_argument("--s08-candidate-sha256", required=True)
    parser.add_argument("--candidate-deck", type=Path, required=True)
    parser.add_argument("--candidate-deck-sha256", required=True)
    parser.add_argument("--parent-summary-sha256", required=True)
    parser.add_argument("--s04-summary-sha256", required=True)
    parser.add_argument("--s08-summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing decision JSON; inputs remain read-only.",
    )
    return parser


def output_is_outside_inputs(output: Path, run_dirs: Sequence[Path]) -> None:
    resolved = output.expanduser().resolve()
    for run_dir in run_dirs:
        candidate = run_dir.expanduser().resolve()
        if resolved == candidate or resolved.is_relative_to(candidate):
            raise panel.AuditError(
                f"decision output must be outside immutable run directory: {candidate}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        output_is_outside_inputs(
            args.output,
            (args.parent_run_dir, args.s04_run_dir, args.s08_run_dir),
        )
        report = audit_endpoint_screen(
            panel_manifest_path=args.panel_manifest,
            expected_panel_manifest_sha256=args.panel_manifest_sha256,
            parent_run_dir=args.parent_run_dir,
            s04_run_dir=args.s04_run_dir,
            s08_run_dir=args.s08_run_dir,
            parent_candidate=args.parent_candidate,
            s04_candidate=args.s04_candidate,
            s08_candidate=args.s08_candidate,
            expected_parent_candidate_sha256=args.parent_candidate_sha256,
            expected_s04_candidate_sha256=args.s04_candidate_sha256,
            expected_s08_candidate_sha256=args.s08_candidate_sha256,
            candidate_deck=args.candidate_deck,
            expected_candidate_deck_sha256=args.candidate_deck_sha256,
            expected_parent_summary_sha256=args.parent_summary_sha256,
            expected_s04_summary_sha256=args.s04_summary_sha256,
            expected_s08_summary_sha256=args.s08_summary_sha256,
        )
        atomic_write_json(args.output, report, overwrite=args.overwrite)
    except panel.AuditError as error:
        print(json.dumps({"pass": False, "audit_error": str(error)}))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
