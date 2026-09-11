from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import audit_gold_push_endpoint_screen as audit  # noqa: E402
import audit_gold_push_panel as panel  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_deck(path: Path, start: int) -> None:
    path.write_text(
        "".join(f"{start + index % 10}\n" for index in range(60)),
        encoding="utf-8",
    )


@dataclass
class EndpointFixture:
    root: Path
    parent_wins: Mapping[str, int]
    s04_wins: Mapping[str, int]
    s08_wins: Mapping[str, int]
    invalid: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        self.manifest = self.root / "panel.json"
        self.deck = self.root / "candidate.csv"
        self.bc = self.root / "bc.pt"
        self.evaluator = self.root / "evaluator.py"
        self.candidates = {
            label: self.root / f"candidate-{label}.pt"
            for label in ("parent", "s04", "s08")
        }
        self.run_dirs = {
            label: self.root / f"run-{label}"
            for label in ("parent", "s04", "s08")
        }
        self.opponents: list[dict[str, Any]] = []
        self._build_inputs()
        for offset, (label, wins) in enumerate(
            (
                ("parent", self.parent_wins),
                ("s04", self.s04_wins),
                ("s08", self.s08_wins),
            )
        ):
            self._build_run(label, wins, offset=offset)

    def _build_inputs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_deck(self.deck, 9000)
        self.bc.write_bytes(b"shared-bc")
        self.evaluator.write_text("# frozen evaluator\n", encoding="utf-8")
        for label, path in self.candidates.items():
            path.write_bytes(f"candidate-{label}".encode())

        deck_paths: list[Path] = []
        for index in range(4):
            path = self.root / f"opponent-deck-{index}.csv"
            write_deck(path, 1000 * (index + 1))
            deck_paths.append(path)
        all_ids = [*audit.POLICY_IDS, *(f"filler-{index}" for index in range(10))]
        for index, policy_id in enumerate(all_ids):
            checkpoint = self.root / f"opponent-{index}.pt"
            checkpoint.write_bytes(f"opponent-{index}".encode())
            deck = deck_paths[index % len(deck_paths)]
            self.opponents.append(
                {
                    "policy_id": policy_id,
                    "submission_id": None,
                    "team_name": f"Team {index}",
                    "archetype": f"archetype-{index % 3}",
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": panel.file_sha256(checkpoint),
                    "deck": str(deck.resolve()),
                    "deck_file_sha256": panel.file_sha256(deck),
                    "deck_hash": panel.deck_semantic_hash(deck),
                    "canonical_order": index < len(audit.POLICY_IDS),
                }
            )
        write_json(
            self.manifest,
            {
                "schema_version": panel.LEAGUE_SCHEMA,
                "opponents": self.opponents,
            },
        )

    def _build_run(
        self, label: str, wins_by_policy: Mapping[str, int], *, offset: int
    ) -> None:
        run_dir = self.run_dirs[label]
        candidate = self.candidates[label]
        candidate_sha = panel.file_sha256(candidate)
        deck_sha = panel.file_sha256(self.deck)
        selected = self.opponents[: len(audit.POLICY_IDS)]
        selected_binding = [
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
            for opponent in selected
        ]
        signature_payload = {
            "schema_version": panel.RUN_SCHEMA,
            "league_manifest": str(self.manifest.resolve()),
            "league_manifest_sha256": panel.file_sha256(self.manifest),
            "evaluator": str(self.evaluator.resolve()),
            "evaluator_sha256": panel.file_sha256(self.evaluator),
            "candidate": str(candidate.resolve()),
            "candidate_sha256": candidate_sha,
            "candidate_deck": str(self.deck.resolve()),
            "candidate_deck_file_sha256": deck_sha,
            "bc_checkpoint": str(self.bc.resolve()),
            "bc_checkpoint_sha256": panel.file_sha256(self.bc),
            "deployment_contract": None,
            "selected_opponents": selected_binding,
            "evaluation": dict(audit.EXPECTED_EVALUATION),
            "gates": dict(audit.EXPECTED_RUNNER_GATES),
            "engine_seed_control": False,
        }
        signature = audit.canonical_sha256(signature_payload)
        run_config = {
            **signature_payload,
            "created_at_utc": "2099-01-01T00:00:00+00:00",
            "run_signature": signature,
            "jobs": 1,
            "wall_seconds_per_invocation": None,
            "local_only": True,
            "uploads_or_submissions_performed": False,
        }
        write_json(run_dir / "run_config.json", run_config)

        policies: list[dict[str, Any]] = []
        pooled = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
        pooled_seats = {
            "0": {"wins": 0, "losses": 0, "draws": 0, "games": 0},
            "1": {"wins": 0, "losses": 0, "draws": 0, "games": 0},
        }
        base_time = datetime(2099, 1, 1, tzinfo=timezone.utc)
        for position, opponent in enumerate(selected):
            policy_id = str(opponent["policy_id"])
            wins = int(wins_by_policy[policy_id])
            if not 0 <= wins <= audit.GAMES_PER_POLICY:
                raise ValueError("invalid synthetic wins")
            seat0_wins = (wins + 1) // 2
            seat1_wins = wins - seat0_wins
            seat_wins = {"0": seat0_wins, "1": seat1_wins}
            invalid = int(self.invalid.get(label, {}).get(policy_id, 0))
            invalid_by_seat = {"0": 0, "1": invalid}
            invalid_reasons = {"timeout": invalid} if invalid else {}
            result_path = (
                run_dir
                / "results"
                / "screening"
                / f"{panel.safe_name(policy_id)}.json"
            )
            started = base_time + timedelta(
                seconds=40 * position + 0.1 * offset
            )
            finished = started + timedelta(seconds=30)
            result = {
                "candidate": {
                    "path": str(candidate.resolve()),
                    "sha256": candidate_sha,
                    "deck": str(self.deck.resolve()),
                    "deck_hash": panel.deck_semantic_hash(self.deck),
                    "canonical_order": False,
                    "order_mode": "hybrid",
                    "hybrid_order": True,
                },
                "opponent": {
                    "path": opponent["checkpoint"],
                    "sha256": opponent["checkpoint_sha256"],
                    "deck": opponent["deck"],
                    "deck_hash": opponent["deck_hash"],
                    "canonical_order": True,
                    "order_mode": "canonical",
                    "hybrid_order": False,
                },
                "engine": {
                    "games_requested": audit.GAMES_PER_POLICY,
                    "environments": audit.EXPECTED_ENVIRONMENTS,
                    "max_game_decisions": audit.EXPECTED_MAX_GAME_DECISIONS,
                    "engine_seed_control": False,
                    "python_torch_seed": panel.derived_seed(
                        audit.EXPECTED_SEED, "screening", policy_id
                    ),
                },
                "evaluation": {
                    "valid_games": audit.GAMES_PER_POLICY,
                    "wins": wins,
                    "losses": audit.GAMES_PER_POLICY - wins,
                    "draws": 0,
                    "invalid_games": invalid,
                    "by_candidate_seat": {
                        seat: {
                            "valid_games": audit.GAMES_PER_POLICY_PER_SEAT,
                            "wins": seat_wins[seat],
                            "losses": audit.GAMES_PER_POLICY_PER_SEAT
                            - seat_wins[seat],
                            "draws": 0,
                        }
                        for seat in ("0", "1")
                    },
                    "seat_balance": {
                        "assignment_mode": "exact_valid_game_quota_v1",
                        "requested_games_by_candidate_seat": {"0": 64, "1": 64},
                        "actual_valid_games_by_candidate_seat": {"0": 64, "1": 64},
                        "requested_games_even": True,
                        "strict_even_balance_required": True,
                        "strict_even_balance_verified": True,
                        "actual_valid_game_gap": 0,
                        "aggregate_equals_seat_sum_verified": True,
                    },
                    "invalid_by_reason": invalid_reasons,
                    "invalid_by_candidate_seat": {
                        "0": {"invalid_games": 0, "by_reason": {}},
                        "1": {
                            "invalid_games": invalid,
                            "by_reason": invalid_reasons,
                        },
                    },
                    "invalid_error_messages": {},
                    "invalid_audit_complete": True,
                    "conservative_invalid_as_loss": {
                        "trials": audit.GAMES_PER_POLICY + invalid,
                        "wins": wins,
                        "nonwins": audit.GAMES_PER_POLICY + invalid - wins,
                        "win_rate": wins / (audit.GAMES_PER_POLICY + invalid),
                        "invalid_rate": invalid
                        / (audit.GAMES_PER_POLICY + invalid),
                    },
                },
                "started_at_utc": started.isoformat(),
                "finished_at_utc": finished.isoformat(),
            }
            write_json(result_path, result)
            policy_row = {
                "policy_id": policy_id,
                "archetype": opponent["archetype"],
                "opponent_canonical_order": True,
                "status": "complete",
                "wins": wins,
                "losses": audit.GAMES_PER_POLICY - wins,
                "draws": 0,
                "games": audit.GAMES_PER_POLICY,
                "by_candidate_seat": {
                    seat: {
                        "wins": seat_wins[seat],
                        "losses": audit.GAMES_PER_POLICY_PER_SEAT
                        - seat_wins[seat],
                        "draws": 0,
                        "games": audit.GAMES_PER_POLICY_PER_SEAT,
                    }
                    for seat in ("0", "1")
                },
                "result_path": str(result_path.resolve()),
            }
            policies.append(policy_row)
            pooled["wins"] += wins
            pooled["losses"] += audit.GAMES_PER_POLICY - wins
            pooled["games"] += audit.GAMES_PER_POLICY
            for seat in ("0", "1"):
                pooled_seats[seat]["wins"] += seat_wins[seat]
                pooled_seats[seat]["losses"] += (
                    audit.GAMES_PER_POLICY_PER_SEAT - seat_wins[seat]
                )
                pooled_seats[seat]["games"] += audit.GAMES_PER_POLICY_PER_SEAT

        summary = {
            "schema_version": panel.SUMMARY_SCHEMA,
            "generated_at_utc": "2099-01-01T00:04:30+00:00",
            "run_signature": signature,
            "candidate": {
                "checkpoint": str(candidate.resolve()),
                "checkpoint_sha256": candidate_sha,
                "deck": str(self.deck.resolve()),
                "deck_file_sha256": deck_sha,
                "order_mode": "hybrid",
            },
            "league_manifest": str(self.manifest.resolve()),
            "deployment_contract": None,
            "selected_policy_ids": list(audit.POLICY_IDS),
            "local_only": True,
            "uploads_or_submissions_performed": False,
            "engine": {
                "engine_seed_control": False,
                "strict_even_valid_games_by_candidate_seat_required": True,
            },
            "confirmation": {"mode": "none", "summary": None},
            "final_evidence_phase": "screening",
            "screening": {
                "phase": "screening",
                "games_requested_per_policy": audit.GAMES_PER_POLICY,
                "selected_policies": len(audit.POLICY_IDS),
                "completed_policies": len(audit.POLICY_IDS),
                "coverage_complete": True,
                "strict_even_seat_balance_verified": True,
                "failures": {},
                "wall_budget_exhausted": False,
                "policies": policies,
                "pooled": pooled,
                "by_candidate_seat": pooled_seats,
            },
        }
        write_json(run_dir / "summary.json", summary)

    def kwargs(self) -> dict[str, Any]:
        return {
            "panel_manifest_path": self.manifest,
            "expected_panel_manifest_sha256": panel.file_sha256(self.manifest),
            "parent_run_dir": self.run_dirs["parent"],
            "s04_run_dir": self.run_dirs["s04"],
            "s08_run_dir": self.run_dirs["s08"],
            "parent_candidate": self.candidates["parent"],
            "s04_candidate": self.candidates["s04"],
            "s08_candidate": self.candidates["s08"],
            "expected_parent_candidate_sha256": panel.file_sha256(
                self.candidates["parent"]
            ),
            "expected_s04_candidate_sha256": panel.file_sha256(
                self.candidates["s04"]
            ),
            "expected_s08_candidate_sha256": panel.file_sha256(
                self.candidates["s08"]
            ),
            "candidate_deck": self.deck,
            "expected_candidate_deck_sha256": panel.file_sha256(self.deck),
            "expected_parent_summary_sha256": panel.file_sha256(
                self.run_dirs["parent"] / "summary.json"
            ),
            "expected_s04_summary_sha256": panel.file_sha256(
                self.run_dirs["s04"] / "summary.json"
            ),
            "expected_s08_summary_sha256": panel.file_sha256(
                self.run_dirs["s08"] / "summary.json"
            ),
        }


PARENT_WINS = dict(
    zip(audit.POLICY_IDS, (61, 62, 66, 68, 69, 62), strict=True)
)
ACTUAL_S04_WINS = dict(
    zip(audit.POLICY_IDS, (47, 70, 62, 78, 65, 68), strict=True)
)
ACTUAL_S08_WINS = dict(
    zip(audit.POLICY_IDS, (71, 77, 71, 63, 74, 64), strict=True)
)


def shifted(source: Mapping[str, int], delta: int) -> dict[str, int]:
    return {key: value + delta for key, value in source.items()}


def make_fixture(
    tmp_path: Path,
    *,
    s04_wins: Mapping[str, int] | None = None,
    s08_wins: Mapping[str, int] | None = None,
    invalid: Mapping[str, Mapping[str, int]] | None = None,
) -> EndpointFixture:
    return EndpointFixture(
        tmp_path / "fixture",
        parent_wins=PARENT_WINS,
        s04_wins=s04_wins or ACTUAL_S04_WINS,
        s08_wins=s08_wins or ACTUAL_S08_WINS,
        invalid=invalid or {},
    )


def test_actual_stage_a_shape_rejects_both_endpoints(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    report = audit.audit_endpoint_screen(**fixture.kwargs())

    assert report["pass"] is False
    assert report["selected_endpoint"] is None
    assert report["decision"] == "reject_s04_and_s08_preserve_parent_incumbent"
    assert report["parent_historical_drift"]["pass"] is True
    assert report["endpoint_comparisons"]["s04"]["failed_gates"] == [
        "bottom4_newcombe_noninferiority",
        "bottom4_win_gain",
        "minimum_other_policy_win_delta",
        "pooled_win_gain",
    ]
    assert report["endpoint_comparisons"]["s08"]["failed_gates"] == [
        "kdcyberdude_win_delta"
    ]
    assert report["submission_authorized_by_this_audit"] is False


def test_selects_s04_when_it_is_the_only_passing_endpoint(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path, s04_wins=shifted(PARENT_WINS, 2))
    report = audit.audit_endpoint_screen(**fixture.kwargs())

    assert report["pass"] is True
    assert report["selected_endpoint"] == "s04"
    assert report["selection_reason"] == "only_endpoint_passing_all_frozen_gates"
    assert "requires_full_16x256_panel" in report["decision"]
    assert report["screening_only_not_submission_evidence"] is True


def test_passing_near_tie_prefers_s04(tmp_path: Path) -> None:
    passing = shifted(PARENT_WINS, 2)
    fixture = make_fixture(tmp_path, s04_wins=passing, s08_wins=passing)
    report = audit.audit_endpoint_screen(**fixture.kwargs())

    assert report["endpoint_comparisons"]["s04"]["pass"] is True
    assert report["endpoint_comparisons"]["s08"]["pass"] is True
    assert report["selected_endpoint"] == "s04"
    assert report["selection_reason"] == "near_tie_prefers_smaller_s04_endpoint"


def test_invalid_game_is_nonwin_and_fails_zero_invalid_gate(tmp_path: Path) -> None:
    policy_id = audit.POLICY_IDS[0]
    fixture = make_fixture(
        tmp_path,
        s04_wins=shifted(PARENT_WINS, 2),
        invalid={"s04": {policy_id: 1}},
    )
    report = audit.audit_endpoint_screen(**fixture.kwargs())
    comparison = report["endpoint_comparisons"]["s04"]

    assert comparison["zero_invalid"]["invalid_games"] == 1
    assert "endpoint_zero_invalid" in comparison["failed_gates"]
    assert comparison["newcombe_noninferiority"]["pooled"]["endpoint"][
        "trials"
    ] == 769


def test_parent_drift_failure_invalidates_both_endpoints(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    evidence = {
        label: audit.validate_screen_only_run_identity(
            label=label,
            run_dir=fixture.run_dirs[label],
            expected_summary_sha256=fixture.kwargs()[
                f"expected_{label}_summary_sha256"
            ],
            candidate=fixture.candidates[label],
            expected_candidate_sha256=fixture.kwargs()[
                f"expected_{label}_candidate_sha256"
            ],
            candidate_deck=fixture.deck,
            expected_candidate_deck_sha256=panel.file_sha256(fixture.deck),
            manifest_path=fixture.manifest.resolve(),
            manifest_sha256=panel.file_sha256(fixture.manifest),
            opponents=[
                panel.load_opponents(fixture.manifest)[1][index]
                for index in range(len(audit.POLICY_IDS))
            ],
        )
        for label in ("parent", "s04", "s08")
    }
    original = audit.HISTORICAL_PARENT["pooled"]
    audit.HISTORICAL_PARENT["pooled"] = {
        "wins": 0,
        "trials": 1536,
        "maximum_absolute_delta": 0.001,
    }
    try:
        drift = audit.audit_parent_drift(evidence["parent"])
        assert drift["pass"] is False
        for label in ("s04", "s08"):
            comparison = audit.endpoint_comparison(
                evidence[label], evidence["parent"], parent_drift_pass=False
            )
            assert "parent_drift" in comparison["failed_gates"]
    finally:
        audit.HISTORICAL_PARENT["pooled"] = original


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("candidate_order_mode", "raw"),
        ("candidate_hybrid_order", False),
        ("seed", 1),
        ("environments", 16),
    ],
)
def test_protocol_tamper_fails_closed(
    tmp_path: Path, field: str, replacement: object
) -> None:
    fixture = make_fixture(tmp_path)
    config_path = fixture.run_dirs["s04"] / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evaluation"][field] = replacement
    write_json(config_path, config)

    with pytest.raises(panel.AuditError):
        audit.audit_endpoint_screen(**fixture.kwargs())


def test_summary_and_candidate_sha_are_external_bindings(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    kwargs = fixture.kwargs()
    kwargs["expected_s08_summary_sha256"] = "0" * 64
    with pytest.raises(panel.AuditError, match="summary actual SHA-256"):
        audit.audit_endpoint_screen(**kwargs)

    kwargs = fixture.kwargs()
    kwargs["expected_s08_candidate_sha256"] = "0" * 64
    with pytest.raises(panel.AuditError, match="candidate actual SHA-256"):
        audit.audit_endpoint_screen(**kwargs)


def test_newcombe_is_independent_and_enforces_margin() -> None:
    row = audit.newcombe_lower_bound(420, 768, 388, 768)
    assert row["point_difference"] == pytest.approx(32 / 768)
    assert row["newcombe_lower_bound"] > audit.NONINFERIORITY_MARGINS["pooled"]

    failing = audit.newcombe_lower_bound(255, 512, 257, 512)
    assert failing["newcombe_lower_bound"] < audit.NONINFERIORITY_MARGINS[
        "bottom4"
    ]


def test_atomic_output_refuses_clobber_and_cleans_temp(tmp_path: Path) -> None:
    output = tmp_path / "decision.json"
    audit.atomic_write_json(output, {"decision": "first"})
    with pytest.raises(panel.AuditError, match="already exists"):
        audit.atomic_write_json(output, {"decision": "second"})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "decision": "first"
    }
    assert not list(tmp_path.glob(".decision.json.*.tmp"))

