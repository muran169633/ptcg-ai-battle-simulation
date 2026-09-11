from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import audit_gold_push_final_statistical_gates as final_audit  # noqa: E402
import audit_gold_push_panel as panel  # noqa: E402
from test_audit_gold_push_panel import PanelFixture, write_json  # noqa: E402


def install_final_gates(fixture: PanelFixture) -> None:
    preregistration = json.loads(
        fixture.preregistration.read_text(encoding="utf-8")
    )
    preregistration["anchor_gate"] = {
        "anchors": list(final_audit.EXPECTED_ANCHOR_LABELS),
        "games_per_anchor": 1024,
        "minimum_win_rate_each": 0.57,
        "minimum_wilson_95_low_each": 0.54,
        "minimum_each_seat_win_rate": 0.53,
    }
    preregistration["confirmation_gate"] = {
        "checkpoint_frozen_before_confirmation": True,
        "independent_seed_games_per_policy": 512,
        "aggregate_wilson_95_low_minimum": 0.56,
        "deterministic_rerun_required": True,
        "package_action_exact_games": 32,
        "package_action_mismatches_required": 0,
        "package_invalid_games_required": 0,
    }
    write_json(fixture.preregistration, preregistration)


def build_run(
    fixture: PanelFixture,
    output_dir: Path,
    *,
    selected_indices: Sequence[int],
    games: int,
    wins_per_seat: int,
    seed: int,
    signature_character: str,
    started_at_utc: str = "2099-01-01T00:00:00+00:00",
    invalid_games_first_policy: int = 0,
) -> None:
    candidate_sha = panel.file_sha256(fixture.candidate)
    candidate_deck_file_sha = panel.file_sha256(fixture.candidate_deck)
    candidate_deck_hash = panel.deck_semantic_hash(fixture.candidate_deck)
    opponents = [fixture.opponents[index] for index in selected_indices]
    template_contract = output_dir / "hybrid_template_contract.json"
    write_json(
        template_contract,
        {
            "schema_version": panel.SUBMISSION_TEMPLATE_SCHEMA,
            "template_version": panel.HYBRID_TEMPLATE_VERSION,
            "decode": {
                "order_mode": "hybrid",
                "canonicalize_order": False,
                "sort_selected_indices": "all contexts except 34",
                "preserve_order_contexts": [34],
            },
            "deck": {"semantic_hash": candidate_deck_hash, "cards": 60},
        },
    )
    deployment_contract = output_dir / "deployment_contract.json"
    write_json(
        deployment_contract,
        {
            "schema_version": panel.DEPLOYMENT_SCHEMA,
            "candidate": {
                "path": str(fixture.candidate.resolve()),
                "sha256": candidate_sha,
            },
            "candidate_deck": {
                "path": str(fixture.candidate_deck.resolve()),
                "file_sha256": candidate_deck_file_sha,
                "semantic_hash": candidate_deck_hash,
            },
            "action_order": {
                "mode": "hybrid",
                "canonical_order": False,
                "hybrid_order": True,
                "preserve_greedy_plackett_luce_contexts": [34],
                "sort_selected_indices_for_all_other_contexts": True,
            },
            "panel": {
                "manifest": str(fixture.manifest.resolve()),
                "manifest_sha256": panel.file_sha256(fixture.manifest),
                "output_dir": str(output_dir.resolve()),
                "legacy_raw_panel_is_evidence_for_this_contract": False,
            },
            "submission_template": {
                "contract": str(template_contract.resolve()),
                "contract_sha256": panel.file_sha256(template_contract),
                "template_version": panel.HYBRID_TEMPLATE_VERSION,
            },
        },
    )
    deployment_binding = {
        "path": str(deployment_contract.resolve()),
        "sha256": panel.file_sha256(deployment_contract),
        "schema_version": panel.DEPLOYMENT_SCHEMA,
    }
    run_config = {
        "schema_version": panel.RUN_SCHEMA,
        "run_signature": signature_character * 64,
        "league_manifest": str(fixture.manifest.resolve()),
        "league_manifest_sha256": panel.file_sha256(fixture.manifest),
        "candidate": str(fixture.candidate.resolve()),
        "candidate_sha256": candidate_sha,
        "candidate_deck": str(fixture.candidate_deck.resolve()),
        "candidate_deck_file_sha256": candidate_deck_file_sha,
        "deployment_contract": deployment_binding,
        "selected_opponents": [
            {
                key: opponent[key]
                for key in (
                    "policy_id",
                    "checkpoint_sha256",
                    "deck_file_sha256",
                    "deck_hash",
                    "canonical_order",
                )
            }
            for opponent in opponents
        ],
        "evaluation": {
            "screening_games": games,
            "environments": 32,
            "max_game_decisions": 1000,
            "candidate_canonical_order": False,
            "candidate_order_mode": "hybrid",
            "candidate_hybrid_order": True,
            "opponent_canonical_order_source": "per_opponent_manifest",
            "legacy_opponent_canonical_order_all_true_assertion": False,
            "seed": seed,
            "device": "cpu",
        },
    }
    write_json(output_dir / "run_config.json", run_config)

    seat_valid = games // 2
    wins = 2 * wins_per_seat
    losses = games - wins
    pooled = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
    pooled_seats = {
        "0": {"wins": 0, "losses": 0, "draws": 0, "games": 0},
        "1": {"wins": 0, "losses": 0, "draws": 0, "games": 0},
    }
    policies: list[dict[str, Any]] = []
    for position, opponent in enumerate(opponents):
        policy_id = str(opponent["policy_id"])
        invalid = invalid_games_first_policy if position == 0 else 0
        invalid_by_reason = {"timeout": invalid} if invalid else {}
        result = {
            "candidate": {
                "path": str(fixture.candidate.resolve()),
                "sha256": candidate_sha,
                "deck": str(fixture.candidate_deck.resolve()),
                "deck_hash": candidate_deck_hash,
                "canonical_order": False,
                "order_mode": "hybrid",
                "hybrid_order": True,
            },
            "opponent": {
                "path": opponent["checkpoint"],
                "sha256": opponent["checkpoint_sha256"],
                "deck": opponent["deck"],
                "deck_hash": opponent["deck_hash"],
                "canonical_order": opponent["canonical_order"],
                "order_mode": (
                    "canonical" if opponent["canonical_order"] else "raw"
                ),
                "hybrid_order": False,
            },
            "engine": {
                "games_requested": games,
                "environments": 32,
                "max_game_decisions": 1000,
                "engine_seed_control": False,
                "python_torch_seed": panel.derived_seed(
                    seed, "screening", policy_id
                ),
            },
            "evaluation": {
                "valid_games": games,
                "wins": wins,
                "losses": losses,
                "draws": 0,
                "invalid_games": invalid,
                "by_candidate_seat": {
                    "0": {
                        "valid_games": seat_valid,
                        "wins": wins_per_seat,
                        "losses": seat_valid - wins_per_seat,
                        "draws": 0,
                    },
                    "1": {
                        "valid_games": seat_valid,
                        "wins": wins_per_seat,
                        "losses": seat_valid - wins_per_seat,
                        "draws": 0,
                    },
                },
                "seat_balance": {
                    "assignment_mode": "exact_valid_game_quota_v1",
                    "requested_games_by_candidate_seat": {
                        "0": seat_valid,
                        "1": seat_valid,
                    },
                    "actual_valid_games_by_candidate_seat": {
                        "0": seat_valid,
                        "1": seat_valid,
                    },
                    "requested_games_even": True,
                    "strict_even_balance_required": True,
                    "strict_even_balance_verified": True,
                    "actual_valid_game_gap": 0,
                    "aggregate_equals_seat_sum_verified": True,
                },
                "invalid_by_reason": invalid_by_reason,
                "invalid_by_candidate_seat": {
                    "0": {"invalid_games": 0, "by_reason": {}},
                    "1": {
                        "invalid_games": invalid,
                        "by_reason": invalid_by_reason,
                    },
                },
                "invalid_error_messages": {},
                "invalid_audit_complete": True,
                "conservative_invalid_as_loss": {
                    "trials": games + invalid,
                    "wins": wins,
                    "nonwins": games + invalid - wins,
                    "win_rate": wins / (games + invalid),
                    "invalid_rate": invalid / (games + invalid),
                },
            },
            "started_at_utc": started_at_utc,
            "finished_at_utc": "2099-01-01T00:01:00+00:00",
        }
        result_path = (
            output_dir
            / "results"
            / "screening"
            / f"{panel.safe_name(policy_id)}.json"
        )
        write_json(result_path, result)
        policies.append(
            {
                "policy_id": policy_id,
                "archetype": opponent["archetype"],
                "opponent_canonical_order": opponent["canonical_order"],
                "status": "complete",
                "wins": wins,
                "losses": losses,
                "draws": 0,
                "games": games,
                "by_candidate_seat": {
                    seat: {
                        "wins": wins_per_seat,
                        "losses": seat_valid - wins_per_seat,
                        "draws": 0,
                        "games": seat_valid,
                    }
                    for seat in ("0", "1")
                },
                "result_path": str(result_path.resolve()),
            }
        )
        for field, value in (
            ("wins", wins),
            ("losses", losses),
            ("draws", 0),
            ("games", games),
        ):
            pooled[field] += value
        for seat in ("0", "1"):
            pooled_seats[seat]["wins"] += wins_per_seat
            pooled_seats[seat]["losses"] += seat_valid - wins_per_seat
            pooled_seats[seat]["games"] += seat_valid

    write_json(
        output_dir / "summary.json",
        {
            "schema_version": panel.SUMMARY_SCHEMA,
            "generated_at_utc": "2099-01-01T00:02:00+00:00",
            "run_signature": signature_character * 64,
            "candidate": {
                "checkpoint": str(fixture.candidate.resolve()),
                "checkpoint_sha256": candidate_sha,
                "deck": str(fixture.candidate_deck.resolve()),
                "deck_file_sha256": candidate_deck_file_sha,
                "order_mode": "hybrid",
            },
            "league_manifest": str(fixture.manifest.resolve()),
            "deployment_contract": deployment_binding,
            "selected_policy_ids": [
                opponent["policy_id"] for opponent in opponents
            ],
            "local_only": True,
            "uploads_or_submissions_performed": False,
            "screening": {
                "phase": "screening",
                "games_requested_per_policy": games,
                "selected_policies": len(opponents),
                "completed_policies": len(opponents),
                "coverage_complete": True,
                "strict_even_seat_balance_verified": True,
                "failures": {},
                "wall_budget_exhausted": False,
                "policies": policies,
                "pooled": pooled,
                "by_candidate_seat": pooled_seats,
            },
        },
    )


def build_all_evidence(
    tmp_path: Path,
    *,
    anchor_wins_per_seat: int = 300,
    confirmation_wins_per_seat: int = 160,
    confirmation_seed: int = 20260812,
    started_at_utc: str = "2099-01-01T00:00:00+00:00",
) -> tuple[PanelFixture, Path, Path, Path]:
    policy_ids = [
        *final_audit.ANCHOR_POLICY_IDS,
        *(f"panel-policy-{index:02d}" for index in range(4, 16)),
    ]
    fixture = PanelFixture(tmp_path / "fixture", policy_ids=policy_ids)
    install_final_gates(fixture)
    terminal_report_path = tmp_path / "terminal-panel-audit.json"
    write_json(terminal_report_path, fixture.audit())
    by_policy_id = {
        str(opponent["policy_id"]): index
        for index, opponent in enumerate(fixture.opponents)
    }
    anchor_indices = [
        by_policy_id[policy_id] for policy_id in final_audit.ANCHOR_POLICY_IDS
    ]
    anchor_dir = tmp_path / "anchor-run"
    confirmation_dir = tmp_path / "confirmation-run"
    build_run(
        fixture,
        anchor_dir,
        selected_indices=anchor_indices,
        games=1024,
        wins_per_seat=anchor_wins_per_seat,
        seed=20260811,
        signature_character="a",
        started_at_utc=started_at_utc,
    )
    build_run(
        fixture,
        confirmation_dir,
        selected_indices=list(range(16)),
        games=512,
        wins_per_seat=confirmation_wins_per_seat,
        seed=confirmation_seed,
        signature_character="b",
        started_at_utc=started_at_utc,
        invalid_games_first_policy=1,
    )
    return fixture, terminal_report_path, anchor_dir, confirmation_dir


def audit(
    fixture: PanelFixture,
    terminal_report_path: Path,
    anchor_dir: Path,
    confirmation_dir: Path,
) -> dict[str, Any]:
    return final_audit.audit_final_statistical_gates(
        preregistration_path=fixture.preregistration,
        panel_manifest_path=fixture.manifest,
        terminal_panel_audit_path=terminal_report_path,
        anchor_run_dir=anchor_dir,
        confirmation_run_dir=confirmation_dir,
        candidate_path=fixture.candidate,
        candidate_deck_path=fixture.candidate_deck,
    )


def test_four_anchor_and_independent_confirmation_gates_pass(
    tmp_path: Path,
) -> None:
    evidence = build_all_evidence(tmp_path)
    report = audit(*evidence)
    assert report["pass"] is True
    assert len(report["anchor_gate"]["policies"]) == 4
    for row in report["anchor_gate"]["policies"]:
        assert row["conservative_raw_win_rate"] == pytest.approx(600 / 1024)
        assert row["wilson_95_low"] > 0.54
        assert row["by_candidate_seat"]["0"]["conservative_win_rate"] == pytest.approx(
            300 / 512
        )
    confirmation = report["confirmation_gate"]
    assert confirmation["policies"] == 16
    assert confirmation["valid_games"] == 8192
    assert confirmation["invalid_games"] == 1
    assert confirmation["attempted_games"] == 8193
    assert confirmation["aggregate_wilson_95_low"] >= 0.56
    assert report["independence_and_freeze"]["runner_seeds_distinct"] is True


def test_anchor_raw_or_wilson_failure_preserves_submission(
    tmp_path: Path,
) -> None:
    evidence = build_all_evidence(tmp_path, anchor_wins_per_seat=290)
    report = audit(*evidence)
    assert report["pass"] is False
    anchor = report["anchor_gate"]["policies"][0]
    assert anchor["gates"]["raw_win_rate"]["pass"] is False
    assert anchor["gates"]["wilson_95_low"]["pass"] is False
    assert report["decision"] == "statistical_gate_failed_preserve_submission"


def test_confirmation_uses_pooled_wilson_not_raw_rate(
    tmp_path: Path,
) -> None:
    evidence = build_all_evidence(
        tmp_path, confirmation_wins_per_seat=145
    )
    report = audit(*evidence)
    confirmation = report["confirmation_gate"]
    assert confirmation["aggregate_conservative_win_rate"] > 0.56
    assert confirmation["aggregate_wilson_95_low"] < 0.56
    assert confirmation["pass"] is False
    assert report["pass"] is False


def test_confirmation_must_use_distinct_seed_and_post_audit_results(
    tmp_path: Path,
) -> None:
    evidence = build_all_evidence(tmp_path, confirmation_seed=20260810)
    with pytest.raises(panel.AuditError, match="different runner seed"):
        audit(*evidence)

    evidence = build_all_evidence(
        tmp_path / "early", started_at_utc="2000-01-01T00:00:00+00:00"
    )
    with pytest.raises(panel.AuditError, match="must start after"):
        audit(*evidence)


def test_cli_atomically_writes_report(tmp_path: Path) -> None:
    fixture, terminal, anchors, confirmation = build_all_evidence(tmp_path)
    output = tmp_path / "reports" / "final-statistical.json"
    exit_code = final_audit.main(
        [
            "--preregistration",
            str(fixture.preregistration),
            "--panel-manifest",
            str(fixture.manifest),
            "--terminal-panel-audit",
            str(terminal),
            "--anchor-run-dir",
            str(anchors),
            "--confirmation-run-dir",
            str(confirmation),
            "--candidate",
            str(fixture.candidate),
            "--candidate-deck",
            str(fixture.candidate_deck),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["pass"] is True
    assert not list(output.parent.glob(".*.tmp"))
