#!/usr/bin/env python3
"""Read-only audit of gold-push anchor and independent confirmation gates.

This consumes evidence emitted by ``run_gold_league_h2h.py``.  It does not
run battles.  Four preregistered anchors must each have 1,024 valid games and
pass conservative raw, Wilson, and both-seat gates.  A separate 16-policy,
512-valid-games-per-policy run must pass the pooled conservative Wilson gate.
Every evaluator invalid is counted as a candidate loss.

The confirmation run is required to follow a passing terminal-panel audit,
use a different runner seed/signature/directory, and contain distinct result
files.  The engine itself does not expose seed control, so these checks prove
separate sampling execution rather than deterministic engine-seed pairing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import audit_gold_push_panel as panel


REPORT_SCHEMA = "ptcg-gold-push-final-statistical-gates-audit-v1"
ANCHOR_POLICY_IDS = (
    "marnie_source_bc_aug08",
    "marnie_v1_u200",
    "marnie_v3_u440",
    "marnie_u472_updated_records",
)
EXPECTED_ANCHOR_LABELS = frozenset(
    {
        "Marnie terminal01 v1 update200",
        "Marnie incumbent v3 update440",
        "Marnie recent7 specialist BC",
        "Marnie U472 updated-records PPO",
    }
)
WILSON_95_Z = 1.959963984540054


@dataclass(frozen=True)
class RunEvidence:
    run_dir: Path
    run_config: dict[str, Any]
    summary: dict[str, Any]
    counts: tuple[panel.PolicyCounts, ...]
    result_paths: dict[str, Path]
    candidate_order_contract: panel.CandidateOrderContract


@dataclass(frozen=True)
class FrozenGates:
    anchor_games: int
    anchor_raw_minimum: float
    anchor_wilson_low_minimum_exclusive: float
    anchor_each_seat_minimum: float
    confirmation_games_per_policy: int
    confirmation_aggregate_wilson_low_minimum: float


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_utc(value: Any, label: str) -> datetime:
    raw = panel.require_text(value, label)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise panel.AuditError(f"{label} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise panel.AuditError(f"{label} must include a timezone")
    return parsed


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        raise panel.AuditError("Wilson interval requires at least one trial")
    proportion = successes / trials
    denominator = 1.0 + WILSON_95_Z * WILSON_95_Z / trials
    center = (
        proportion + WILSON_95_Z * WILSON_95_Z / (2.0 * trials)
    ) / denominator
    margin = (
        WILSON_95_Z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + WILSON_95_Z * WILSON_95_Z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - margin, center + margin


def load_frozen_gates(path: Path) -> tuple[dict[str, Any], FrozenGates]:
    preregistration = panel.read_json(path, "preregistration")
    panel.require_equal(
        preregistration.get("schema_version"),
        panel.PREREGISTRATION_SCHEMA,
        "preregistration.schema_version",
    )
    anchor = panel.require_mapping(
        preregistration.get("anchor_gate"), "preregistration.anchor_gate"
    )
    labels = panel.require_list(anchor.get("anchors"), "anchor_gate.anchors")
    if len(labels) != 4 or set(labels) != EXPECTED_ANCHOR_LABELS:
        raise panel.AuditError(
            "anchor_gate.anchors is not the frozen four-anchor identity set"
        )
    confirmation = panel.require_mapping(
        preregistration.get("confirmation_gate"),
        "preregistration.confirmation_gate",
    )
    panel.require_equal(
        confirmation.get("checkpoint_frozen_before_confirmation"),
        True,
        "confirmation_gate.checkpoint_frozen_before_confirmation",
    )
    gates = FrozenGates(
        anchor_games=panel.require_int(
            anchor.get("games_per_anchor"), "anchor_gate.games_per_anchor", 2
        ),
        anchor_raw_minimum=panel.require_number(
            anchor.get("minimum_win_rate_each"),
            "anchor_gate.minimum_win_rate_each",
        ),
        anchor_wilson_low_minimum_exclusive=panel.require_number(
            anchor.get("minimum_wilson_95_low_each"),
            "anchor_gate.minimum_wilson_95_low_each",
        ),
        anchor_each_seat_minimum=panel.require_number(
            anchor.get("minimum_each_seat_win_rate"),
            "anchor_gate.minimum_each_seat_win_rate",
        ),
        confirmation_games_per_policy=panel.require_int(
            confirmation.get("independent_seed_games_per_policy"),
            "confirmation_gate.independent_seed_games_per_policy",
            2,
        ),
        confirmation_aggregate_wilson_low_minimum=panel.require_number(
            confirmation.get("aggregate_wilson_95_low_minimum"),
            "confirmation_gate.aggregate_wilson_95_low_minimum",
        ),
    )
    if gates.anchor_games % 2 or gates.confirmation_games_per_policy % 2:
        raise panel.AuditError("anchor and confirmation game quotas must be even")
    for label, value in (
        ("anchor raw", gates.anchor_raw_minimum),
        ("anchor Wilson", gates.anchor_wilson_low_minimum_exclusive),
        ("anchor seat", gates.anchor_each_seat_minimum),
        (
            "confirmation Wilson",
            gates.confirmation_aggregate_wilson_low_minimum,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise panel.AuditError(f"{label} gate must be in [0, 1]")
    return preregistration, gates


def load_run_evidence(
    *,
    run_dir: Path,
    manifest_path: Path,
    opponents: Sequence[panel.Opponent],
    candidate: Path,
    candidate_sha256: str,
    candidate_deck: Path,
    candidate_deck_file_sha256: str,
    candidate_deck_hash: str,
    games_per_policy: int,
) -> RunEvidence:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise panel.AuditError(f"league run directory does not exist: {run_dir}")
    run_config, summary, screening = panel.validate_run_identity(
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest_sha256=panel.file_sha256(manifest_path),
        opponents=opponents,
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        candidate_deck=candidate_deck,
        candidate_deck_file_sha256=candidate_deck_file_sha256,
        games_per_policy=games_per_policy,
    )
    results_dir = run_dir / "results" / "screening"
    expected_paths = {
        opponent.policy_id: (
            results_dir / f"{panel.safe_name(opponent.policy_id)}.json"
        ).resolve()
        for opponent in opponents
    }
    if not results_dir.is_dir():
        raise panel.AuditError(f"screening results directory missing: {results_dir}")
    actual_paths = {path.resolve() for path in results_dir.glob("*.json")}
    panel.require_equal(
        actual_paths, set(expected_paths.values()), "screening result coverage"
    )
    counts: list[panel.PolicyCounts] = []
    for opponent in opponents:
        result_path = expected_paths[opponent.policy_id]
        result = panel.read_json(result_path, f"result[{opponent.policy_id}]")
        counts.append(
            panel.validate_result(
                path=result_path,
                result=result,
                opponent=opponent,
                candidate=candidate,
                candidate_sha256=candidate_sha256,
                candidate_deck=candidate_deck,
                candidate_deck_hash=candidate_deck_hash,
                run_config=run_config,
                games_per_policy=games_per_policy,
            )
        )
    panel.validate_summary_counts(screening, counts, expected_paths)
    order_contract = panel.candidate_order_contract(
        panel.require_mapping(run_config.get("evaluation"), "run.evaluation")
    )
    return RunEvidence(
        run_dir=run_dir,
        run_config=run_config,
        summary=summary,
        counts=tuple(counts),
        result_paths=expected_paths,
        candidate_order_contract=order_contract,
    )


def validate_terminal_panel_audit(
    *,
    path: Path,
    manifest_path: Path,
    candidate: Path,
    candidate_sha256: str,
    candidate_deck: Path,
    candidate_deck_file_sha256: str,
    candidate_deck_hash: str,
) -> tuple[dict[str, Any], Path]:
    audit = panel.read_json(path, "terminal panel audit")
    panel.require_equal(
        audit.get("schema_version"),
        panel.REPORT_SCHEMA,
        "terminal_panel_audit.schema_version",
    )
    panel.require_equal(audit.get("pass"), True, "terminal_panel_audit.pass")
    panel.require_equal(
        audit.get("read_only_audit"), True, "terminal_panel_audit.read_only_audit"
    )
    panel.require_equal(
        audit.get("training_evaluation_packaging_submission_performed"),
        False,
        "terminal_panel_audit side effects",
    )
    inputs = panel.require_mapping(audit.get("inputs"), "terminal_panel_audit.inputs")
    expected = {
        "panel_manifest": manifest_path,
        "candidate_checkpoint": candidate,
        "candidate_deck": candidate_deck,
    }
    for field, expected_path in expected.items():
        panel.require_equal(
            panel.resolve_path(inputs.get(field), path.parent, f"terminal audit {field}"),
            expected_path,
            f"terminal audit {field}",
        )
    panel.require_equal(
        inputs.get("panel_manifest_sha256"),
        panel.file_sha256(manifest_path),
        "terminal audit panel manifest SHA-256",
    )
    panel.require_equal(
        inputs.get("candidate_checkpoint_sha256"),
        candidate_sha256,
        "terminal audit candidate SHA-256",
    )
    panel.require_equal(
        inputs.get("candidate_deck_file_sha256"),
        candidate_deck_file_sha256,
        "terminal audit candidate deck file SHA-256",
    )
    panel.require_equal(
        inputs.get("candidate_deck_semantic_sha256"),
        candidate_deck_hash,
        "terminal audit candidate deck semantic SHA-256",
    )
    terminal_run_dir = panel.resolve_path(
        inputs.get("run_directory"), path.parent, "terminal audit run_directory"
    )
    for filename, field in (
        ("run_config.json", "run_config_sha256"),
        ("summary.json", "summary_sha256"),
    ):
        panel.require_equal(
            panel.file_sha256(terminal_run_dir / filename),
            inputs.get(field),
            f"terminal audit {filename} SHA-256",
        )
    return audit, terminal_run_dir


def result_start_times(evidence: RunEvidence) -> list[datetime]:
    starts: list[datetime] = []
    for policy_id, path in evidence.result_paths.items():
        result = panel.read_json(path, f"result[{policy_id}]")
        starts.append(
            parse_utc(result.get("started_at_utc"), f"result[{policy_id}].started_at_utc")
        )
    return starts


def validate_independence(
    *,
    terminal_audit: dict[str, Any],
    terminal_run_dir: Path,
    anchor: RunEvidence,
    confirmation: RunEvidence,
) -> dict[str, Any]:
    directories = {
        terminal_run_dir.resolve(),
        anchor.run_dir.resolve(),
        confirmation.run_dir.resolve(),
    }
    if len(directories) != 3:
        raise panel.AuditError(
            "terminal, anchor, and confirmation runs must use distinct directories"
        )
    terminal_config = panel.read_json(
        terminal_run_dir / "run_config.json", "terminal panel run config"
    )
    signatures = {
        terminal_config.get("run_signature"),
        anchor.run_config.get("run_signature"),
        confirmation.run_config.get("run_signature"),
    }
    if None in signatures or len(signatures) != 3:
        raise panel.AuditError("terminal, anchor, and confirmation signatures must differ")
    terminal_eval = panel.require_mapping(
        terminal_config.get("evaluation"), "terminal run evaluation"
    )
    confirmation_eval = panel.require_mapping(
        confirmation.run_config.get("evaluation"), "confirmation run evaluation"
    )
    terminal_seed = panel.require_int(terminal_eval.get("seed"), "terminal run seed")
    confirmation_seed = panel.require_int(
        confirmation_eval.get("seed"), "confirmation run seed"
    )
    if terminal_seed == confirmation_seed:
        raise panel.AuditError(
            "independent confirmation must use a different runner seed"
        )
    if anchor.candidate_order_contract.legacy_raw:
        raise panel.AuditError(
            "new anchor evidence must use the explicit candidate order contract"
        )
    if confirmation.candidate_order_contract.legacy_raw:
        raise panel.AuditError(
            "new confirmation evidence must use the explicit candidate order contract"
        )
    panel.require_equal(
        anchor.candidate_order_contract.public(),
        confirmation.candidate_order_contract.public(),
        "anchor vs confirmation candidate order contract",
    )

    audit_time = parse_utc(
        terminal_audit.get("generated_at_utc"),
        "terminal_panel_audit.generated_at_utc",
    )
    anchor_starts = result_start_times(anchor)
    confirmation_starts = result_start_times(confirmation)
    if min(anchor_starts) <= audit_time:
        raise panel.AuditError(
            "anchor evidence must start after the passing terminal-panel audit"
        )
    if min(confirmation_starts) <= audit_time:
        raise panel.AuditError(
            "confirmation evidence must start after the passing terminal-panel audit"
        )

    hardlink_collisions: list[str] = []
    terminal_results_dir = terminal_run_dir / "results" / "screening"
    for policy_id, confirmation_path in confirmation.result_paths.items():
        terminal_path = (
            terminal_results_dir / f"{panel.safe_name(policy_id)}.json"
        )
        if not terminal_path.is_file():
            raise panel.AuditError(
                f"terminal panel result missing for confirmation policy {policy_id}"
            )
        terminal_stat = terminal_path.stat()
        confirmation_stat = confirmation_path.stat()
        if (
            terminal_stat.st_dev == confirmation_stat.st_dev
            and terminal_stat.st_ino == confirmation_stat.st_ino
        ):
            hardlink_collisions.append(policy_id)
    if hardlink_collisions:
        raise panel.AuditError(
            "confirmation result files reuse terminal result inodes: "
            f"{hardlink_collisions}"
        )
    return {
        "terminal_anchor_confirmation_directories_distinct": True,
        "run_signatures_distinct": True,
        "terminal_seed": terminal_seed,
        "confirmation_seed": confirmation_seed,
        "runner_seeds_distinct": True,
        "candidate_order_contract": (
            confirmation.candidate_order_contract.public()
        ),
        "anchor_confirmation_candidate_order_contract_equal": True,
        "all_anchor_results_started_after_terminal_audit": True,
        "all_confirmation_results_started_after_terminal_audit": True,
        "confirmation_result_inodes_distinct_from_terminal": True,
        "engine_seed_control": False,
        "interpretation": (
            "Separate post-freeze sampling execution is proven; official engine "
            "RNG is not seed-controllable or deterministically pairable."
        ),
    }


def audit_anchor_gates(
    evidence: RunEvidence, gates: FrozenGates
) -> dict[str, Any]:
    policies: list[dict[str, Any]] = []
    for item in evidence.counts:
        attempted = item.attempted_games
        raw = item.wins / attempted
        low, high = wilson_interval(item.wins, attempted)
        seats: dict[str, Any] = {}
        for seat in ("0", "1"):
            seat_attempted = (
                item.seat_valid_games[seat] + item.invalid_by_seat[seat]
            )
            rate = item.seat_wins[seat] / seat_attempted
            seats[seat] = {
                "wins": item.seat_wins[seat],
                "valid_games": item.seat_valid_games[seat],
                "invalid_games": item.invalid_by_seat[seat],
                "attempted_games": seat_attempted,
                "conservative_win_rate": rate,
                "minimum": gates.anchor_each_seat_minimum,
                "pass": rate >= gates.anchor_each_seat_minimum,
            }
        row = {
            "policy_id": item.policy_id,
            "valid_games": item.valid_games,
            "invalid_games": item.invalid_games,
            "attempted_games": attempted,
            "wins": item.wins,
            "conservative_raw_win_rate": raw,
            "wilson_95_low": low,
            "wilson_95_high": high,
            "by_candidate_seat": seats,
            "gates": {
                "raw_win_rate": {
                    "actual": raw,
                    "minimum": gates.anchor_raw_minimum,
                    "comparator": ">=",
                    "pass": raw >= gates.anchor_raw_minimum,
                },
                "wilson_95_low": {
                    "actual": low,
                    "minimum_exclusive": (
                        gates.anchor_wilson_low_minimum_exclusive
                    ),
                    "comparator": ">",
                    "pass": low
                    > gates.anchor_wilson_low_minimum_exclusive,
                },
                "each_seat_win_rate": {
                    "actual_minimum": min(
                        row["conservative_win_rate"] for row in seats.values()
                    ),
                    "minimum": gates.anchor_each_seat_minimum,
                    "comparator": ">=",
                    "pass": all(row["pass"] for row in seats.values()),
                },
            },
        }
        row["pass"] = all(gate["pass"] for gate in row["gates"].values())
        policies.append(row)
    return {
        "pass": all(row["pass"] for row in policies),
        "policies": policies,
        "all_invalids_counted_as_candidate_losses": True,
    }


def audit_confirmation_gate(
    evidence: RunEvidence, gates: FrozenGates
) -> dict[str, Any]:
    wins = sum(item.wins for item in evidence.counts)
    valid_games = sum(item.valid_games for item in evidence.counts)
    invalid_games = sum(item.invalid_games for item in evidence.counts)
    attempted = valid_games + invalid_games
    raw = wins / attempted
    low, high = wilson_interval(wins, attempted)
    passed = low >= gates.confirmation_aggregate_wilson_low_minimum
    return {
        "pass": passed,
        "policies": len(evidence.counts),
        "valid_games_per_policy": gates.confirmation_games_per_policy,
        "valid_games": valid_games,
        "invalid_games": invalid_games,
        "attempted_games": attempted,
        "wins": wins,
        "nonwins_including_invalid": attempted - wins,
        "aggregate_conservative_win_rate": raw,
        "aggregate_wilson_95_low": low,
        "aggregate_wilson_95_high": high,
        "gate": {
            "actual": low,
            "minimum": gates.confirmation_aggregate_wilson_low_minimum,
            "comparator": ">=",
            "pass": passed,
        },
        "all_invalids_counted_as_candidate_losses": True,
    }


def audit_final_statistical_gates(
    *,
    preregistration_path: Path,
    panel_manifest_path: Path,
    terminal_panel_audit_path: Path,
    anchor_run_dir: Path,
    confirmation_run_dir: Path,
    candidate_path: Path,
    candidate_deck_path: Path,
) -> dict[str, Any]:
    preregistration_path = preregistration_path.expanduser().resolve()
    panel_manifest_path = panel_manifest_path.expanduser().resolve()
    terminal_panel_audit_path = terminal_panel_audit_path.expanduser().resolve()
    candidate_path = candidate_path.expanduser().resolve()
    candidate_deck_path = candidate_deck_path.expanduser().resolve()
    if not candidate_path.is_file():
        raise panel.AuditError(f"candidate checkpoint missing: {candidate_path}")
    if not candidate_deck_path.is_file():
        raise panel.AuditError(f"candidate deck missing: {candidate_deck_path}")

    _, gates = load_frozen_gates(preregistration_path)
    _, all_opponents = panel.load_opponents(panel_manifest_path)
    by_id = {opponent.policy_id: opponent for opponent in all_opponents}
    if not set(ANCHOR_POLICY_IDS) <= set(by_id):
        raise panel.AuditError("panel manifest lacks one or more frozen anchors")
    anchors = [
        opponent
        for opponent in all_opponents
        if opponent.policy_id in ANCHOR_POLICY_IDS
    ]
    panel.require_equal(
        {opponent.policy_id for opponent in anchors},
        set(ANCHOR_POLICY_IDS),
        "selected anchor identities",
    )

    candidate_sha = panel.file_sha256(candidate_path)
    if candidate_sha in {opponent.checkpoint_sha256 for opponent in all_opponents}:
        raise panel.AuditError("candidate checkpoint equals an opponent checkpoint")
    candidate_deck_file_sha = panel.file_sha256(candidate_deck_path)
    candidate_deck_hash = panel.deck_semantic_hash(candidate_deck_path)
    terminal_audit, terminal_run_dir = validate_terminal_panel_audit(
        path=terminal_panel_audit_path,
        manifest_path=panel_manifest_path,
        candidate=candidate_path,
        candidate_sha256=candidate_sha,
        candidate_deck=candidate_deck_path,
        candidate_deck_file_sha256=candidate_deck_file_sha,
        candidate_deck_hash=candidate_deck_hash,
    )
    anchor_evidence = load_run_evidence(
        run_dir=anchor_run_dir,
        manifest_path=panel_manifest_path,
        opponents=anchors,
        candidate=candidate_path,
        candidate_sha256=candidate_sha,
        candidate_deck=candidate_deck_path,
        candidate_deck_file_sha256=candidate_deck_file_sha,
        candidate_deck_hash=candidate_deck_hash,
        games_per_policy=gates.anchor_games,
    )
    confirmation_evidence = load_run_evidence(
        run_dir=confirmation_run_dir,
        manifest_path=panel_manifest_path,
        opponents=all_opponents,
        candidate=candidate_path,
        candidate_sha256=candidate_sha,
        candidate_deck=candidate_deck_path,
        candidate_deck_file_sha256=candidate_deck_file_sha,
        candidate_deck_hash=candidate_deck_hash,
        games_per_policy=gates.confirmation_games_per_policy,
    )
    independence = validate_independence(
        terminal_audit=terminal_audit,
        terminal_run_dir=terminal_run_dir,
        anchor=anchor_evidence,
        confirmation=confirmation_evidence,
    )
    anchor_gate = audit_anchor_gates(anchor_evidence, gates)
    confirmation_gate = audit_confirmation_gate(confirmation_evidence, gates)
    passed = bool(anchor_gate["pass"] and confirmation_gate["pass"])
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": panel.utc_now(),
        "pass": passed,
        "read_only_audit": True,
        "training_evaluation_packaging_submission_performed": False,
        "inputs": {
            "preregistration": str(preregistration_path),
            "preregistration_sha256": panel.file_sha256(preregistration_path),
            "panel_manifest": str(panel_manifest_path),
            "panel_manifest_sha256": panel.file_sha256(panel_manifest_path),
            "terminal_panel_audit": str(terminal_panel_audit_path),
            "terminal_panel_audit_sha256": panel.file_sha256(
                terminal_panel_audit_path
            ),
            "terminal_panel_run_directory": str(terminal_run_dir),
            "anchor_run_directory": str(anchor_evidence.run_dir),
            "confirmation_run_directory": str(confirmation_evidence.run_dir),
            "candidate_checkpoint": str(candidate_path),
            "candidate_checkpoint_sha256": candidate_sha,
            "candidate_deck": str(candidate_deck_path),
            "candidate_deck_file_sha256": candidate_deck_file_sha,
            "candidate_deck_semantic_sha256": candidate_deck_hash,
        },
        "independence_and_freeze": independence,
        "anchor_gate": anchor_gate,
        "confirmation_gate": confirmation_gate,
        "remaining_preregistered_gates_not_audited_here": [
            "deterministic package rerun",
            "32-game exact action package validation",
            "zero action mismatches",
            "zero package invalid games",
        ],
        "decision": (
            "anchor_and_independent_confirmation_statistical_gates_passed"
            if passed
            else "statistical_gate_failed_preserve_submission"
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=panel.DEFAULT_PREREGISTRATION,
    )
    parser.add_argument(
        "--panel-manifest", type=Path, default=panel.DEFAULT_PANEL_MANIFEST
    )
    parser.add_argument("--terminal-panel-audit", type=Path, required=True)
    parser.add_argument("--anchor-run-dir", type=Path, required=True)
    parser.add_argument("--confirmation-run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-deck", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = audit_final_statistical_gates(
            preregistration_path=args.preregistration,
            panel_manifest_path=args.panel_manifest,
            terminal_panel_audit_path=args.terminal_panel_audit,
            anchor_run_dir=args.anchor_run_dir,
            confirmation_run_dir=args.confirmation_run_dir,
            candidate_path=args.candidate,
            candidate_deck_path=args.candidate_deck,
        )
    except panel.AuditError as error:
        print(json.dumps({"pass": False, "audit_error": str(error)}))
        return 2
    if args.output is not None:
        atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
