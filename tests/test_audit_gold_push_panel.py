from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import audit_gold_push_panel as auditor  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class PanelFixture:
    def __init__(
        self,
        root: Path,
        *,
        wins_per_seat: int = 80,
        invalid_games: int = 1,
        deck_variants: int = 4,
        policy_ids: Sequence[str] | None = None,
        candidate_order_mode: str | None = None,
    ) -> None:
        self.root = root
        self.wins_per_seat = wins_per_seat
        self.invalid_games = invalid_games
        self.deck_variants = deck_variants
        self.policy_ids = tuple(policy_ids) if policy_ids is not None else tuple(
            f"policy/{index:02d}"
            for index in range(auditor.EXPECTED_PANEL_POLICIES)
        )
        if len(self.policy_ids) != auditor.EXPECTED_PANEL_POLICIES:
            raise ValueError("policy_ids must contain exactly 16 identities")
        if candidate_order_mode not in (None, "raw", "canonical", "hybrid"):
            raise ValueError("invalid candidate_order_mode")
        self.candidate_order_mode = candidate_order_mode
        self.preregistration = root / "preregistration.json"
        self.manifest = root / "panel_opponents.json"
        self.candidate = root / "candidate.pt"
        self.candidate_deck = root / "candidate.csv"
        self.run_dir = root / "league_run"
        self.deployment_contract = root / "hybrid_deployment_contract.json"
        self.template_contract = root / "hybrid_template_contract.json"
        self.opponents: list[dict[str, object]] = []
        self.results: dict[str, dict[str, object]] = {}
        self.build()

    @staticmethod
    def deck_cards(base: int) -> list[int]:
        return [base + index % 10 for index in range(60)]

    @staticmethod
    def write_deck(path: Path, cards: list[int]) -> None:
        path.write_text(
            "".join(f"{card}\n" for card in cards), encoding="utf-8"
        )

    def build(self) -> None:
        write_json(
            self.preregistration,
            {
                "schema_version": auditor.PREREGISTRATION_SCHEMA,
                "terminal_panel_gate": {
                    "invalid_timeout_truncation_treatment": "candidate loss",
                    "minimum_attempted_games": 4096,
                    "maximum_failure_rate": 0.001,
                    "minimum_deck_hashes": 3,
                    "minimum_policies": 12,
                    "screen_games_per_policy": 256,
                    "macro_win_rate_minimum": 0.58,
                    "bottom_quartile_cvar_minimum": 0.53,
                    "minimum_archetype_win_rate": 0.52,
                    "minimum_each_seat_aggregate_win_rate": 0.55,
                },
            },
        )
        self.candidate.write_bytes(b"candidate-checkpoint")
        self.write_deck(
            self.candidate_deck, self.deck_cards(9000)
        )
        candidate_sha = auditor.file_sha256(self.candidate)
        candidate_deck_file_sha = auditor.file_sha256(self.candidate_deck)
        candidate_deck_hash = auditor.deck_semantic_hash(self.candidate_deck)

        decks: list[Path] = []
        for variant in range(self.deck_variants):
            path = self.root / f"deck-{variant}.csv"
            self.write_deck(path, self.deck_cards(1000 * (variant + 1)))
            decks.append(path)

        for index in range(auditor.EXPECTED_PANEL_POLICIES):
            checkpoint = self.root / f"opponent-{index}.pt"
            checkpoint.write_bytes(f"opponent-checkpoint-{index}".encode())
            deck = decks[index % len(decks)]
            self.opponents.append(
                {
                    "policy_id": self.policy_ids[index],
                    "submission_id": None,
                    "team_name": f"Team {index}",
                    "archetype": f"archetype-{index % 4}",
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": auditor.file_sha256(checkpoint),
                    "deck": str(deck.resolve()),
                    "deck_file_sha256": auditor.file_sha256(deck),
                    "deck_hash": auditor.deck_semantic_hash(deck),
                    # Synthetic BC policies use canonical action order while
                    # synthetic PPO policies retain their learned ordering.
                    "canonical_order": index < 8,
                }
            )
        write_json(
            self.manifest,
            {
                "schema_version": auditor.LEAGUE_SCHEMA,
                "opponents": self.opponents,
            },
        )

        candidate_canonical = self.candidate_order_mode == "canonical"
        candidate_hybrid = self.candidate_order_mode == "hybrid"
        deployment_binding: dict[str, str] | None = None
        if candidate_hybrid:
            write_json(
                self.template_contract,
                {
                    "schema_version": auditor.SUBMISSION_TEMPLATE_SCHEMA,
                    "template_version": auditor.HYBRID_TEMPLATE_VERSION,
                    "decode": {
                        "order_mode": "hybrid",
                        "canonicalize_order": False,
                        "sort_selected_indices": "all contexts except 34",
                        "preserve_order_contexts": [34],
                    },
                    "deck": {
                        "semantic_hash": candidate_deck_hash,
                        "cards": 60,
                    },
                },
            )
            write_json(
                self.deployment_contract,
                {
                    "schema_version": auditor.DEPLOYMENT_SCHEMA,
                    "candidate": {
                        "path": str(self.candidate.resolve()),
                        "sha256": candidate_sha,
                    },
                    "candidate_deck": {
                        "path": str(self.candidate_deck.resolve()),
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
                        "manifest": str(self.manifest.resolve()),
                        "manifest_sha256": auditor.file_sha256(self.manifest),
                        "output_dir": str(self.run_dir.resolve()),
                        "legacy_raw_panel_is_evidence_for_this_contract": False,
                    },
                    "submission_template": {
                        "contract": str(self.template_contract.resolve()),
                        "contract_sha256": auditor.file_sha256(
                            self.template_contract
                        ),
                        "template_version": auditor.HYBRID_TEMPLATE_VERSION,
                    },
                },
            )
            deployment_binding = {
                "path": str(self.deployment_contract.resolve()),
                "sha256": auditor.file_sha256(self.deployment_contract),
                "schema_version": auditor.DEPLOYMENT_SCHEMA,
            }
        run_config = {
            "schema_version": auditor.RUN_SCHEMA,
            "run_signature": "f" * 64,
            "league_manifest": str(self.manifest.resolve()),
            "league_manifest_sha256": auditor.file_sha256(self.manifest),
            "candidate": str(self.candidate.resolve()),
            "candidate_sha256": candidate_sha,
            "candidate_deck": str(self.candidate_deck.resolve()),
            "candidate_deck_file_sha256": candidate_deck_file_sha,
            "selected_opponents": [
                {
                    key: row[key]
                    for key in (
                        "policy_id",
                        "checkpoint_sha256",
                        "deck_file_sha256",
                        "deck_hash",
                        "canonical_order",
                    )
                }
                for row in self.opponents
            ],
            "evaluation": {
                "screening_games": 256,
                "environments": 32,
                "max_game_decisions": 1000,
                "candidate_canonical_order": candidate_canonical,
                "opponent_canonical_order_source": "per_opponent_manifest",
                "legacy_opponent_canonical_order_all_true_assertion": False,
                "seed": 20260810,
                "device": "cpu",
            },
        }
        if self.candidate_order_mode is not None:
            run_config["evaluation"].update(
                {
                    "candidate_order_mode": self.candidate_order_mode,
                    "candidate_hybrid_order": candidate_hybrid,
                }
            )
        if deployment_binding is not None:
            run_config["deployment_contract"] = deployment_binding
        write_json(self.run_dir / "run_config.json", run_config)

        summary_policies: list[dict[str, object]] = []
        pooled = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
        pooled_seats = {
            "0": {"wins": 0, "losses": 0, "draws": 0, "games": 0},
            "1": {"wins": 0, "losses": 0, "draws": 0, "games": 0},
        }
        for index, opponent in enumerate(self.opponents):
            policy_id = str(opponent["policy_id"])
            wins = 2 * self.wins_per_seat
            losses = 256 - wins
            invalid = self.invalid_games if index == 0 else 0
            invalid_by_seat = {"0": 0, "1": invalid}
            invalid_reason = {"timeout": invalid} if invalid else {}
            invalid_seat_rows = {
                "0": {"invalid_games": 0, "by_reason": {}},
                "1": {
                    "invalid_games": invalid,
                    "by_reason": invalid_reason,
                },
            }
            seat_rows = {
                seat: {
                    "valid_games": 128,
                    "wins": self.wins_per_seat,
                    "losses": 128 - self.wins_per_seat,
                    "draws": 0,
                }
                for seat in ("0", "1")
            }
            attempted = 256 + invalid
            result = {
                "candidate": {
                    "path": str(self.candidate.resolve()),
                    "sha256": candidate_sha,
                    "deck": str(self.candidate_deck.resolve()),
                    "deck_hash": candidate_deck_hash,
                    "canonical_order": candidate_canonical,
                    **(
                        {
                            "order_mode": self.candidate_order_mode,
                            "hybrid_order": candidate_hybrid,
                        }
                        if self.candidate_order_mode is not None
                        else {}
                    ),
                },
                "opponent": {
                    "path": opponent["checkpoint"],
                    "sha256": opponent["checkpoint_sha256"],
                    "deck": opponent["deck"],
                    "deck_hash": opponent["deck_hash"],
                    "canonical_order": opponent["canonical_order"],
                    **(
                        {
                            "order_mode": (
                                "canonical"
                                if opponent["canonical_order"]
                                else "raw"
                            ),
                            "hybrid_order": False,
                        }
                        if self.candidate_order_mode is not None
                        else {}
                    ),
                },
                "engine": {
                    "games_requested": 256,
                    "environments": 32,
                    "max_game_decisions": 1000,
                    "engine_seed_control": False,
                    "python_torch_seed": auditor.derived_seed(
                        20260810, "screening", policy_id
                    ),
                },
                "evaluation": {
                    "valid_games": 256,
                    "wins": wins,
                    "losses": losses,
                    "draws": 0,
                    "invalid_games": invalid,
                    "by_candidate_seat": seat_rows,
                    "seat_balance": {
                        "assignment_mode": "exact_valid_game_quota_v1",
                        "requested_games_by_candidate_seat": {
                            "0": 128,
                            "1": 128,
                        },
                        "actual_valid_games_by_candidate_seat": {
                            "0": 128,
                            "1": 128,
                        },
                        "requested_games_even": True,
                        "strict_even_balance_required": True,
                        "strict_even_balance_verified": True,
                        "actual_valid_game_gap": 0,
                        "aggregate_equals_seat_sum_verified": True,
                    },
                    "invalid_by_reason": invalid_reason,
                    "invalid_by_candidate_seat": invalid_seat_rows,
                    "invalid_error_messages": {},
                    "invalid_audit_complete": True,
                    "conservative_invalid_as_loss": {
                        "trials": attempted,
                        "wins": wins,
                        "nonwins": attempted - wins,
                        "win_rate": wins / attempted,
                        "invalid_rate": invalid / attempted,
                    },
                },
            }
            result_path = (
                self.run_dir
                / "results"
                / "screening"
                / f"{auditor.safe_name(policy_id)}.json"
            )
            write_json(result_path, result)
            self.results[policy_id] = result
            summary_policies.append(
                {
                    "policy_id": policy_id,
                    "archetype": opponent["archetype"],
                    "opponent_canonical_order": opponent[
                        "canonical_order"
                    ],
                    "status": "complete",
                    "wins": wins,
                    "losses": losses,
                    "draws": 0,
                    "games": 256,
                    "by_candidate_seat": {
                        seat: {
                            "wins": self.wins_per_seat,
                            "losses": 128 - self.wins_per_seat,
                            "draws": 0,
                            "games": 128,
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
                ("games", 256),
            ):
                pooled[field] += value
            for seat in ("0", "1"):
                pooled_seats[seat]["wins"] += self.wins_per_seat
                pooled_seats[seat]["losses"] += 128 - self.wins_per_seat
                pooled_seats[seat]["games"] += 128

        write_json(
            self.run_dir / "summary.json",
            {
                "schema_version": auditor.SUMMARY_SCHEMA,
                "run_signature": "f" * 64,
                "candidate": {
                    "checkpoint": str(self.candidate.resolve()),
                    "checkpoint_sha256": candidate_sha,
                    "deck": str(self.candidate_deck.resolve()),
                    "deck_file_sha256": candidate_deck_file_sha,
                    **(
                        {"order_mode": self.candidate_order_mode}
                        if self.candidate_order_mode is not None
                        else {}
                    ),
                },
                "league_manifest": str(self.manifest.resolve()),
                **(
                    {"deployment_contract": deployment_binding}
                    if deployment_binding is not None
                    else {}
                ),
                "selected_policy_ids": [
                    row["policy_id"] for row in self.opponents
                ],
                "local_only": True,
                "uploads_or_submissions_performed": False,
                "screening": {
                    "phase": "screening",
                    "games_requested_per_policy": 256,
                    "selected_policies": 16,
                    "completed_policies": 16,
                    "coverage_complete": True,
                    "strict_even_seat_balance_verified": True,
                    "failures": {},
                    "wall_budget_exhausted": False,
                    "policies": summary_policies,
                    "pooled": pooled,
                    "by_candidate_seat": pooled_seats,
                },
            },
        )

    def audit(self) -> dict[str, object]:
        return auditor.audit_panel(
            preregistration_path=self.preregistration,
            panel_manifest_path=self.manifest,
            run_dir=self.run_dir,
            candidate_path=self.candidate,
            candidate_deck_path=self.candidate_deck,
        )

    def result_path(self, policy_id: str) -> Path:
        return (
            self.run_dir
            / "results"
            / "screening"
            / f"{auditor.safe_name(policy_id)}.json"
        )


def test_pass_recomputes_every_invalid_as_loss_and_writes_atomically(
    tmp_path: Path,
) -> None:
    fixture = PanelFixture(tmp_path)
    report = fixture.audit()
    assert report["pass"] is True
    assert report["identity_and_coverage"]["candidate_order_contract"] == {
        "order_mode": "legacy_raw",
        "canonical_order": False,
        "hybrid_order": False,
        "legacy_raw": True,
    }
    assert report["accounting"] == {
        "valid_games": 4096,
        "invalid_games": 1,
        "attempted_games": 4097,
        "wins": 2560,
        "nonwins_including_invalid": 1537,
        "failure_invalid_rate": pytest.approx(1 / 4097),
        "invalid_timeout_truncation_treatment": "candidate loss",
    }
    policy = report["metrics"]["policies"][0]
    assert policy["conservative_win_rate"] == pytest.approx(160 / 257)
    assert report["metrics"]["by_candidate_seat"]["1"][
        "attempted_games"
    ] == 2049
    assert report["metrics"]["bottom_quartile_cvar"]["count"] == 4
    assert all(row["pass"] for row in report["gates"].values())

    output = tmp_path / "audit" / "report.json"
    exit_code = auditor.main(
        [
            "--preregistration",
            str(fixture.preregistration),
            "--panel-manifest",
            str(fixture.manifest),
            "--run-dir",
            str(fixture.run_dir),
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


def test_numeric_gate_failure_returns_report_and_preserves_submission(
    tmp_path: Path,
) -> None:
    fixture = PanelFixture(tmp_path, wins_per_seat=72)
    report = fixture.audit()
    assert report["pass"] is False
    assert report["gates"]["macro_policy_win_rate"]["pass"] is False
    assert report["gates"]["candidate_seat_0_aggregate_win_rate"][
        "pass"
    ] is True
    assert report["decision"] == "terminal_panel_gate_failed_preserve_submission"


def test_candidate_checkpoint_cannot_appear_in_opponent_pool(
    tmp_path: Path,
) -> None:
    fixture = PanelFixture(tmp_path)
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["opponents"][0]["checkpoint"])
    checkpoint.write_bytes(fixture.candidate.read_bytes())
    manifest["opponents"][0]["checkpoint_sha256"] = auditor.file_sha256(
        checkpoint
    )
    write_json(fixture.manifest, manifest)
    with pytest.raises(auditor.AuditError, match="equals an opponent"):
        fixture.audit()


def test_manifest_hash_and_result_seat_quota_are_strict(
    tmp_path: Path,
) -> None:
    fixture = PanelFixture(tmp_path)
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    manifest["post_run_mutation"] = True
    write_json(fixture.manifest, manifest)
    with pytest.raises(auditor.AuditError, match="league_manifest_sha256"):
        fixture.audit()

    # Restore the exact manifest, then corrupt one raw seat quota.
    del manifest["post_run_mutation"]
    write_json(fixture.manifest, manifest)
    policy_id = str(fixture.opponents[0]["policy_id"])
    result_path = fixture.result_path(policy_id)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["evaluation"]["by_candidate_seat"]["0"]["valid_games"] = 127
    write_json(result_path, result)
    with pytest.raises(auditor.AuditError, match="valid quota"):
        fixture.audit()


def test_semantic_deck_diversity_is_computed_from_actual_decks(
    tmp_path: Path,
) -> None:
    fixture = PanelFixture(tmp_path, deck_variants=1)
    report = fixture.audit()
    assert report["pass"] is False
    assert report["identity_and_coverage"]["semantic_deck_hash_count"] == 1
    assert report["gates"]["semantic_deck_hash_count"] == {
        "actual": 1,
        "minimum": 3,
        "pass": False,
    }


def test_inconsistent_invalid_reason_or_summary_result_identity_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = PanelFixture(tmp_path)
    policy_id = str(fixture.opponents[0]["policy_id"])
    result_path = fixture.result_path(policy_id)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["evaluation"]["invalid_by_candidate_seat"]["1"][
        "by_reason"
    ] = {"different_reason": 1}
    write_json(result_path, result)
    with pytest.raises(auditor.AuditError, match="seat/reason cross-audit"):
        fixture.audit()


@pytest.mark.parametrize("tampered_order", [True, 1])
def test_mixed_bc_true_ppo_false_order_is_bound_and_mismatch_rejected(
    tmp_path: Path,
    tampered_order: object,
) -> None:
    fixture = PanelFixture(tmp_path)
    assert [row["canonical_order"] for row in fixture.opponents[:8]] == [
        True
    ] * 8
    assert [row["canonical_order"] for row in fixture.opponents[8:]] == [
        False
    ] * 8
    assert fixture.audit()["pass"] is True

    ppo_policy_id = str(fixture.opponents[8]["policy_id"])
    result_path = fixture.result_path(ppo_policy_id)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["opponent"]["canonical_order"] = tampered_order
    write_json(result_path, result)
    expected_message = (
        "explicit boolean" if tampered_order == 1 and type(tampered_order) is int
        else "opponent.canonical_order mismatch"
    )
    with pytest.raises(auditor.AuditError, match=expected_message):
        fixture.audit()


@pytest.mark.parametrize("invalid", [None, 0, "true"])
def test_manifest_requires_explicit_boolean_canonical_order(
    tmp_path: Path,
    invalid: object,
) -> None:
    fixture = PanelFixture(tmp_path)
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    if invalid is None:
        manifest["opponents"][0].pop("canonical_order")
    else:
        manifest["opponents"][0]["canonical_order"] = invalid
    write_json(fixture.manifest, manifest)
    with pytest.raises(auditor.AuditError, match="explicit boolean"):
        fixture.audit()


def test_new_hybrid_candidate_order_contract_is_strict(tmp_path: Path) -> None:
    fixture = PanelFixture(tmp_path, candidate_order_mode="hybrid")
    report = fixture.audit()
    assert report["pass"] is True
    assert report["identity_and_coverage"]["candidate_order_contract"] == {
        "order_mode": "hybrid",
        "canonical_order": False,
        "hybrid_order": True,
        "legacy_raw": False,
    }

    run_config_path = fixture.run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["evaluation"]["candidate_hybrid_order"] = False
    write_json(run_config_path, run_config)
    with pytest.raises(auditor.AuditError, match="candidate order flags"):
        fixture.audit()


def test_hybrid_requires_matching_run_and_summary_deployment_binding(
    tmp_path: Path,
) -> None:
    fixture = PanelFixture(tmp_path, candidate_order_mode="hybrid")
    run_config_path = fixture.run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config.pop("deployment_contract")
    write_json(run_config_path, run_config)
    with pytest.raises(auditor.AuditError, match="run.deployment_contract"):
        fixture.audit()

    fixture = PanelFixture(tmp_path / "summary-mismatch", candidate_order_mode="hybrid")
    summary_path = fixture.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["deployment_contract"]["sha256"] = "0" * 64
    write_json(summary_path, summary)
    with pytest.raises(auditor.AuditError, match="summary.deployment_contract"):
        fixture.audit()


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        ("candidate", "sha256", "0" * 64, "deployment.candidate.sha256"),
        ("candidate_deck", "semantic_hash", "0" * 64, "semantic_hash"),
        ("action_order", "hybrid_order", False, "action_order.hybrid_order"),
        (
            "panel",
            "legacy_raw_panel_is_evidence_for_this_contract",
            True,
            "legacy_raw_panel_is_evidence",
        ),
        (
            "submission_template",
            "template_version",
            "wrong-template",
            "template_version",
        ),
    ],
)
def test_hybrid_deployment_contract_content_is_strict(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    fixture = PanelFixture(tmp_path, candidate_order_mode="hybrid")
    contract = json.loads(
        fixture.deployment_contract.read_text(encoding="utf-8")
    )
    contract[section][field] = replacement
    write_json(fixture.deployment_contract, contract)
    new_sha = auditor.file_sha256(fixture.deployment_contract)
    for evidence_path in (
        fixture.run_dir / "run_config.json",
        fixture.run_dir / "summary.json",
    ):
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["deployment_contract"]["sha256"] = new_sha
        write_json(evidence_path, evidence)
    with pytest.raises(auditor.AuditError, match=message):
        fixture.audit()


def test_hybrid_deployment_contract_file_hash_is_verified(tmp_path: Path) -> None:
    fixture = PanelFixture(tmp_path, candidate_order_mode="hybrid")
    contract = json.loads(
        fixture.deployment_contract.read_text(encoding="utf-8")
    )
    contract["post_run_tamper"] = True
    write_json(fixture.deployment_contract, contract)
    with pytest.raises(auditor.AuditError, match="actual SHA-256"):
        fixture.audit()
