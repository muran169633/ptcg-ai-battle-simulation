#!/usr/bin/env python3
"""Strict, read-only audit of the preregistered gold-push H2H panel.

The existing league runner reports metrics over valid games.  The gold-push
protocol is deliberately more conservative: every invalid/timeout/truncated
attempt is a candidate loss.  This tool re-reads every evaluator result,
revalidates all file and result identities, and recomputes the terminal gates
with ``valid_games + invalid_games`` as the denominator.

It never trains, evaluates, packages, uploads, or submits.  The only optional
write is an atomically installed JSON report supplied with ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = (
    REPO_ROOT / "artifacts/gold_push_20260810_v1/preregistration.json"
)
DEFAULT_PANEL_MANIFEST = (
    REPO_ROOT / "artifacts/gold_push_20260810_v1/panel_opponents_v1.json"
)
PREREGISTRATION_SCHEMA = "ptcg-gold-push-preregistration-v1"
LEAGUE_SCHEMA = "ptcg-ppo-opponent-league-v1"
RUN_SCHEMA = "ptcg-gold-league-h2h-run-v1"
SUMMARY_SCHEMA = "ptcg-gold-league-h2h-summary-v1"
REPORT_SCHEMA = "ptcg-gold-push-panel-audit-v1"
DEPLOYMENT_SCHEMA = "ptcg-gold-push-deployment-contract-v1"
SUBMISSION_TEMPLATE_SCHEMA = "ptcg-ppo-submission-template-contract-v1"
HYBRID_TEMPLATE_VERSION = "ptcg-ppo-marnie-hybrid-order-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_PANEL_POLICIES = 16
BOTTOM_QUARTILE_ALPHA = 0.25


class AuditError(ValueError):
    """Raised when evidence is malformed, incomplete, or identity-mismatched."""


@dataclass(frozen=True)
class Opponent:
    policy_id: str
    archetype: str
    checkpoint: Path
    checkpoint_sha256: str
    deck: Path
    deck_file_sha256: str
    deck_hash: str
    canonical_order: bool


@dataclass(frozen=True)
class PolicyCounts:
    policy_id: str
    archetype: str
    opponent_canonical_order: bool
    wins: int
    losses: int
    draws: int
    valid_games: int
    invalid_games: int
    invalid_by_seat: Mapping[str, int]
    seat_wins: Mapping[str, int]
    seat_valid_games: Mapping[str, int]

    @property
    def attempted_games(self) -> int:
        return self.valid_games + self.invalid_games

    @property
    def conservative_win_rate(self) -> float:
        return self.wins / self.attempted_games


@dataclass(frozen=True)
class CandidateOrderContract:
    order_mode: str
    canonical_order: bool
    hybrid_order: bool
    legacy_raw: bool

    def public(self) -> dict[str, Any]:
        return {
            "order_mode": self.order_mode,
            "canonical_order": self.canonical_order,
            "hybrid_order": self.hybrid_order,
            "legacy_raw": self.legacy_raw,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object: {path}")
    return value


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AuditError(f"{label} must be a list")
    return value


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{label} must be a non-empty string")
    return value


def require_sha256(value: Any, label: str) -> str:
    parsed = require_text(value, label)
    if SHA256_RE.fullmatch(parsed) is None:
        raise AuditError(f"{label} must be a lowercase SHA-256")
    return parsed


def require_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError(f"{label} must be an integer")
    if value < minimum:
        raise AuditError(f"{label} must be >= {minimum}")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AuditError(f"{label} must be finite")
    return parsed


def require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise AuditError(f"{label} must be an explicit boolean")
    return value


def candidate_order_contract(
    evaluation: Mapping[str, Any], label: str = "run.evaluation"
) -> CandidateOrderContract:
    has_mode = "candidate_order_mode" in evaluation
    has_hybrid = "candidate_hybrid_order" in evaluation
    canonical = require_bool(
        evaluation.get("candidate_canonical_order"),
        f"{label}.candidate_canonical_order",
    )
    if not has_mode and not has_hybrid:
        if canonical:
            raise AuditError(
                f"{label}: legacy order contract is allowed only for raw "
                "candidate_canonical_order=false evidence"
            )
        return CandidateOrderContract(
            order_mode="legacy_raw",
            canonical_order=False,
            hybrid_order=False,
            legacy_raw=True,
        )
    if not has_mode or not has_hybrid:
        raise AuditError(
            f"{label}: candidate_order_mode and candidate_hybrid_order must "
            "either both be present or both be absent in legacy raw evidence"
        )
    mode = require_text(
        evaluation.get("candidate_order_mode"),
        f"{label}.candidate_order_mode",
    )
    hybrid = require_bool(
        evaluation.get("candidate_hybrid_order"),
        f"{label}.candidate_hybrid_order",
    )
    expected = {
        "raw": (False, False),
        "canonical": (True, False),
        "hybrid": (False, True),
    }
    if mode not in expected:
        raise AuditError(
            f"{label}.candidate_order_mode must be raw, canonical, or hybrid"
        )
    require_equal(
        (canonical, hybrid),
        expected[mode],
        f"{label} candidate order flags",
    )
    return CandidateOrderContract(
        order_mode=mode,
        canonical_order=canonical,
        hybrid_order=hybrid,
        legacy_raw=False,
    )


def validate_result_order_identity(
    *,
    row: Mapping[str, Any],
    expected_mode: str,
    expected_canonical: bool,
    expected_hybrid: bool,
    legacy_raw: bool,
    label: str,
) -> None:
    canonical = require_bool(row.get("canonical_order"), f"{label}.canonical_order")
    require_equal(canonical, expected_canonical, f"{label}.canonical_order")
    if legacy_raw:
        unexpected = sorted(
            field for field in ("order_mode", "hybrid_order") if field in row
        )
        if unexpected:
            raise AuditError(
                f"{label}: legacy raw result must omit {unexpected}; new "
                "order fields require a new run_config order contract"
            )
        return
    mode = require_text(row.get("order_mode"), f"{label}.order_mode")
    hybrid = require_bool(row.get("hybrid_order"), f"{label}.hybrid_order")
    require_equal(mode, expected_mode, f"{label}.order_mode")
    require_equal(hybrid, expected_hybrid, f"{label}.hybrid_order")


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AuditError(
            f"{label} mismatch: actual={actual!r}, expected={expected!r}"
        )


def require_close(actual: Any, expected: float, label: str) -> None:
    parsed = require_number(actual, label)
    if not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AuditError(
            f"{label} mismatch: actual={parsed!r}, expected={expected!r}"
        )


def resolve_path(value: Any, base: Path, label: str) -> Path:
    raw = require_text(value, label)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def validate_deployment_contract_identity(
    *,
    run_config: Mapping[str, Any],
    summary: Mapping[str, Any],
    run_config_path: Path,
    summary_path: Path,
    order_contract: CandidateOrderContract,
    candidate: Path,
    candidate_sha256: str,
    candidate_deck: Path,
    candidate_deck_file_sha256: str,
    panel_manifest: Path,
    panel_manifest_sha256: str,
    run_dir: Path,
) -> dict[str, Any] | None:
    """Bind a hybrid league run to the reviewed deployment/package contract."""
    run_raw = run_config.get("deployment_contract")
    summary_raw = summary.get("deployment_contract")
    if not order_contract.hybrid_order:
        if run_raw is not None or summary_raw is not None:
            raise AuditError(
                "deployment_contract evidence is allowed only for an explicit "
                "hybrid candidate order run"
            )
        return None

    run_binding = require_mapping(run_raw, "run.deployment_contract")
    summary_binding = require_mapping(
        summary_raw,
        "summary.deployment_contract",
    )
    require_equal(
        summary_binding,
        run_binding,
        "summary.deployment_contract",
    )
    contract_path = resolve_path(
        run_binding.get("path"),
        run_config_path.parent,
        "run.deployment_contract.path",
    )
    require_equal(
        resolve_path(
            summary_binding.get("path"),
            summary_path.parent,
            "summary.deployment_contract.path",
        ),
        contract_path,
        "summary.deployment_contract.path",
    )
    contract_sha256 = require_sha256(
        run_binding.get("sha256"),
        "run.deployment_contract.sha256",
    )
    require_equal(
        run_binding.get("schema_version"),
        DEPLOYMENT_SCHEMA,
        "run.deployment_contract.schema_version",
    )
    contract = read_json(contract_path, "hybrid deployment contract")
    require_equal(
        file_sha256(contract_path),
        contract_sha256,
        "deployment contract actual SHA-256",
    )
    require_equal(
        contract.get("schema_version"),
        DEPLOYMENT_SCHEMA,
        "deployment contract schema",
    )

    contract_candidate = require_mapping(
        contract.get("candidate"), "deployment.candidate"
    )
    require_equal(
        resolve_path(
            contract_candidate.get("path"),
            contract_path.parent,
            "deployment.candidate.path",
        ),
        candidate,
        "deployment.candidate.path",
    )
    require_equal(
        contract_candidate.get("sha256"),
        candidate_sha256,
        "deployment.candidate.sha256",
    )

    contract_deck = require_mapping(
        contract.get("candidate_deck"), "deployment.candidate_deck"
    )
    require_equal(
        resolve_path(
            contract_deck.get("path"),
            contract_path.parent,
            "deployment.candidate_deck.path",
        ),
        candidate_deck,
        "deployment.candidate_deck.path",
    )
    require_equal(
        contract_deck.get("file_sha256"),
        candidate_deck_file_sha256,
        "deployment.candidate_deck.file_sha256",
    )
    require_equal(
        contract_deck.get("semantic_hash"),
        deck_semantic_hash(candidate_deck),
        "deployment.candidate_deck.semantic_hash",
    )

    action_order = require_mapping(
        contract.get("action_order"), "deployment.action_order"
    )
    expected_action_order = {
        "mode": "hybrid",
        "canonical_order": False,
        "hybrid_order": True,
        "preserve_greedy_plackett_luce_contexts": [34],
        "sort_selected_indices_for_all_other_contexts": True,
    }
    for field, expected in expected_action_order.items():
        require_equal(
            action_order.get(field),
            expected,
            f"deployment.action_order.{field}",
        )

    panel = require_mapping(contract.get("panel"), "deployment.panel")
    require_equal(
        resolve_path(
            panel.get("manifest"),
            contract_path.parent,
            "deployment.panel.manifest",
        ),
        panel_manifest,
        "deployment.panel.manifest",
    )
    require_equal(
        panel.get("manifest_sha256"),
        panel_manifest_sha256,
        "deployment.panel.manifest_sha256",
    )
    require_equal(
        resolve_path(
            panel.get("output_dir"),
            contract_path.parent,
            "deployment.panel.output_dir",
        ),
        run_dir,
        "deployment.panel.output_dir",
    )
    require_equal(
        panel.get("legacy_raw_panel_is_evidence_for_this_contract"),
        False,
        "deployment.panel.legacy_raw_panel_is_evidence_for_this_contract",
    )

    submission_template = require_mapping(
        contract.get("submission_template"),
        "deployment.submission_template",
    )
    template_path = resolve_path(
        submission_template.get("contract"),
        contract_path.parent,
        "deployment.submission_template.contract",
    )
    template_sha256 = require_sha256(
        submission_template.get("contract_sha256"),
        "deployment.submission_template.contract_sha256",
    )
    require_equal(
        file_sha256(template_path),
        template_sha256,
        "submission template contract actual SHA-256",
    )
    require_equal(
        submission_template.get("template_version"),
        HYBRID_TEMPLATE_VERSION,
        "deployment.submission_template.template_version",
    )
    template = read_json(template_path, "hybrid submission template contract")
    require_equal(
        template.get("schema_version"),
        SUBMISSION_TEMPLATE_SCHEMA,
        "submission template schema",
    )
    require_equal(
        template.get("template_version"),
        HYBRID_TEMPLATE_VERSION,
        "submission template version",
    )
    template_decode = require_mapping(
        template.get("decode"), "submission template decode"
    )
    for field, expected in (
        ("order_mode", "hybrid"),
        ("canonicalize_order", False),
        ("sort_selected_indices", "all contexts except 34"),
        ("preserve_order_contexts", [34]),
    ):
        require_equal(
            template_decode.get(field),
            expected,
            f"submission template decode.{field}",
        )
    template_deck = require_mapping(
        template.get("deck"), "submission template deck"
    )
    require_equal(
        template_deck.get("semantic_hash"),
        deck_semantic_hash(candidate_deck),
        "submission template deck.semantic_hash",
    )
    return {
        "path": str(contract_path),
        "sha256": contract_sha256,
        "schema_version": DEPLOYMENT_SCHEMA,
        "submission_template_contract": str(template_path),
        "submission_template_contract_sha256": template_sha256,
        "template_version": HYBRID_TEMPLATE_VERSION,
    }


def deck_semantic_hash(path: Path) -> str:
    if not path.is_file():
        raise AuditError(f"deck does not exist: {path}")
    cards: list[int] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cards.append(int(line.strip()))
    except (OSError, UnicodeError, ValueError) as error:
        raise AuditError(f"invalid deck CSV: {path}") from error
    if len(cards) != 60:
        raise AuditError(f"{path}: contains {len(cards)} cards; expected 60")
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{(slug[:72] or 'policy')}-{suffix}"


def derived_seed(base_seed: int, phase: str, policy_id: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{phase}:{policy_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def load_preregistered_gate(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    preregistration = read_json(path, "preregistration")
    require_equal(
        preregistration.get("schema_version"),
        PREREGISTRATION_SCHEMA,
        "preregistration.schema_version",
    )
    gate = require_mapping(
        preregistration.get("terminal_panel_gate"),
        "preregistration.terminal_panel_gate",
    )
    require_equal(
        gate.get("invalid_timeout_truncation_treatment"),
        "candidate loss",
        "terminal_panel_gate.invalid_timeout_truncation_treatment",
    )
    values = {
        "minimum_attempted_games": float(
            require_int(
                gate.get("minimum_attempted_games"),
                "terminal_panel_gate.minimum_attempted_games",
                1,
            )
        ),
        "maximum_failure_rate": require_number(
            gate.get("maximum_failure_rate"),
            "terminal_panel_gate.maximum_failure_rate",
        ),
        "minimum_deck_hashes": float(
            require_int(
                gate.get("minimum_deck_hashes"),
                "terminal_panel_gate.minimum_deck_hashes",
                1,
            )
        ),
        "minimum_policies": float(
            require_int(
                gate.get("minimum_policies"),
                "terminal_panel_gate.minimum_policies",
                1,
            )
        ),
        "screen_games_per_policy": float(
            require_int(
                gate.get("screen_games_per_policy"),
                "terminal_panel_gate.screen_games_per_policy",
                2,
            )
        ),
        "macro_win_rate_minimum": require_number(
            gate.get("macro_win_rate_minimum"),
            "terminal_panel_gate.macro_win_rate_minimum",
        ),
        "bottom_quartile_cvar_minimum": require_number(
            gate.get("bottom_quartile_cvar_minimum"),
            "terminal_panel_gate.bottom_quartile_cvar_minimum",
        ),
        "minimum_archetype_win_rate": require_number(
            gate.get("minimum_archetype_win_rate"),
            "terminal_panel_gate.minimum_archetype_win_rate",
        ),
        "minimum_each_seat_aggregate_win_rate": require_number(
            gate.get("minimum_each_seat_aggregate_win_rate"),
            "terminal_panel_gate.minimum_each_seat_aggregate_win_rate",
        ),
    }
    for key in (
        "maximum_failure_rate",
        "macro_win_rate_minimum",
        "bottom_quartile_cvar_minimum",
        "minimum_archetype_win_rate",
        "minimum_each_seat_aggregate_win_rate",
    ):
        if not 0.0 <= values[key] <= 1.0:
            raise AuditError(f"terminal_panel_gate.{key} must be in [0, 1]")
    if int(values["screen_games_per_policy"]) % 2:
        raise AuditError("screen_games_per_policy must be even")
    return preregistration, values


def load_opponents(path: Path) -> tuple[dict[str, Any], list[Opponent]]:
    manifest = read_json(path, "panel manifest")
    require_equal(
        manifest.get("schema_version"),
        LEAGUE_SCHEMA,
        "panel_manifest.schema_version",
    )
    raw_opponents = require_list(
        manifest.get("opponents"), "panel_manifest.opponents"
    )
    if len(raw_opponents) != EXPECTED_PANEL_POLICIES:
        raise AuditError(
            "panel manifest must contain exactly "
            f"{EXPECTED_PANEL_POLICIES} policies; got {len(raw_opponents)}"
        )

    opponents: list[Opponent] = []
    seen_policy_ids: set[str] = set()
    for index, raw_value in enumerate(raw_opponents):
        raw = require_mapping(raw_value, f"opponents[{index}]")
        policy_id = require_text(raw.get("policy_id"), f"opponents[{index}].policy_id")
        if policy_id in seen_policy_ids:
            raise AuditError(f"duplicate opponent policy_id: {policy_id!r}")
        seen_policy_ids.add(policy_id)
        archetype = require_text(
            raw.get("archetype"), f"opponents[{index}].archetype"
        )
        checkpoint = resolve_path(
            raw.get("checkpoint"), path.parent, f"opponents[{index}].checkpoint"
        )
        deck = resolve_path(raw.get("deck"), path.parent, f"opponents[{index}].deck")
        if not checkpoint.is_file():
            raise AuditError(f"opponent checkpoint does not exist: {checkpoint}")
        checkpoint_sha = require_sha256(
            raw.get("checkpoint_sha256"),
            f"opponents[{index}].checkpoint_sha256",
        )
        require_equal(
            file_sha256(checkpoint),
            checkpoint_sha,
            f"opponents[{index}].checkpoint actual SHA-256",
        )
        deck_file_sha = require_sha256(
            raw.get("deck_file_sha256"),
            f"opponents[{index}].deck_file_sha256",
        )
        require_equal(
            file_sha256(deck),
            deck_file_sha,
            f"opponents[{index}].deck actual file SHA-256",
        )
        deck_hash = require_sha256(
            raw.get("deck_hash"), f"opponents[{index}].deck_hash"
        )
        require_equal(
            deck_semantic_hash(deck),
            deck_hash,
            f"opponents[{index}].deck semantic SHA-256",
        )
        canonical_order = require_bool(
            raw.get("canonical_order"),
            f"opponents[{index}].canonical_order",
        )
        quality = raw.get("quality")
        if quality is not None:
            require_equal(
                require_mapping(quality, f"opponents[{index}].quality").get("pass"),
                True,
                f"opponents[{index}].quality.pass",
            )
        opponents.append(
            Opponent(
                policy_id=policy_id,
                archetype=archetype,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha,
                deck=deck,
                deck_file_sha256=deck_file_sha,
                deck_hash=deck_hash,
                canonical_order=canonical_order,
            )
        )
    return manifest, opponents


def expected_selected_opponents(opponents: Sequence[Opponent]) -> list[dict[str, str]]:
    return [
        {
            "policy_id": opponent.policy_id,
            "checkpoint_sha256": opponent.checkpoint_sha256,
            "deck_file_sha256": opponent.deck_file_sha256,
            "deck_hash": opponent.deck_hash,
            "canonical_order": opponent.canonical_order,
        }
        for opponent in opponents
    ]


def validate_run_identity(
    *,
    run_dir: Path,
    manifest_path: Path,
    manifest_sha256: str,
    opponents: Sequence[Opponent],
    candidate: Path,
    candidate_sha256: str,
    candidate_deck: Path,
    candidate_deck_file_sha256: str,
    games_per_policy: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    run_config = read_json(run_config_path, "league run config")
    summary = read_json(summary_path, "league summary")
    require_equal(run_config.get("schema_version"), RUN_SCHEMA, "run schema")
    require_equal(summary.get("schema_version"), SUMMARY_SCHEMA, "summary schema")

    expected_manifest = manifest_path.resolve()
    require_equal(
        resolve_path(run_config.get("league_manifest"), run_config_path.parent, "run.league_manifest"),
        expected_manifest,
        "run.league_manifest",
    )
    require_equal(
        run_config.get("league_manifest_sha256"),
        manifest_sha256,
        "run.league_manifest_sha256",
    )
    require_equal(
        resolve_path(summary.get("league_manifest"), summary_path.parent, "summary.league_manifest"),
        expected_manifest,
        "summary.league_manifest",
    )
    selected_opponents = require_list(
        run_config.get("selected_opponents"), "run.selected_opponents"
    )
    for index, raw in enumerate(selected_opponents):
        selected = require_mapping(raw, f"run.selected_opponents[{index}]")
        require_bool(
            selected.get("canonical_order"),
            f"run.selected_opponents[{index}].canonical_order",
        )
    require_equal(
        selected_opponents,
        expected_selected_opponents(opponents),
        "run.selected_opponents",
    )
    expected_ids = [opponent.policy_id for opponent in opponents]
    require_equal(summary.get("selected_policy_ids"), expected_ids, "summary.selected_policy_ids")
    require_equal(summary.get("run_signature"), run_config.get("run_signature"), "run signature")

    require_equal(
        resolve_path(run_config.get("candidate"), run_config_path.parent, "run.candidate"),
        candidate,
        "run.candidate",
    )
    require_equal(run_config.get("candidate_sha256"), candidate_sha256, "run.candidate_sha256")
    require_equal(
        resolve_path(
            run_config.get("candidate_deck"),
            run_config_path.parent,
            "run.candidate_deck",
        ),
        candidate_deck,
        "run.candidate_deck",
    )
    require_equal(
        run_config.get("candidate_deck_file_sha256"),
        candidate_deck_file_sha256,
        "run.candidate_deck_file_sha256",
    )
    summary_candidate = require_mapping(summary.get("candidate"), "summary.candidate")
    require_equal(
        resolve_path(
            summary_candidate.get("checkpoint"),
            summary_path.parent,
            "summary.candidate.checkpoint",
        ),
        candidate,
        "summary.candidate.checkpoint",
    )
    require_equal(
        summary_candidate.get("checkpoint_sha256"),
        candidate_sha256,
        "summary.candidate.checkpoint_sha256",
    )
    require_equal(
        resolve_path(
            summary_candidate.get("deck"),
            summary_path.parent,
            "summary.candidate.deck",
        ),
        candidate_deck,
        "summary.candidate.deck",
    )
    require_equal(
        summary_candidate.get("deck_file_sha256"),
        candidate_deck_file_sha256,
        "summary.candidate.deck_file_sha256",
    )
    require_equal(summary.get("local_only"), True, "summary.local_only")
    require_equal(
        summary.get("uploads_or_submissions_performed"),
        False,
        "summary.uploads_or_submissions_performed",
    )

    evaluation = require_mapping(run_config.get("evaluation"), "run.evaluation")
    order_contract = candidate_order_contract(evaluation)
    if order_contract.legacy_raw:
        if "order_mode" in summary_candidate:
            raise AuditError(
                "summary.candidate.order_mode must be absent for immutable "
                "legacy raw evidence"
            )
    else:
        require_equal(
            summary_candidate.get("order_mode"),
            order_contract.order_mode,
            "summary.candidate.order_mode",
        )
    validate_deployment_contract_identity(
        run_config=run_config,
        summary=summary,
        run_config_path=run_config_path,
        summary_path=summary_path,
        order_contract=order_contract,
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        candidate_deck=candidate_deck,
        candidate_deck_file_sha256=candidate_deck_file_sha256,
        panel_manifest=expected_manifest,
        panel_manifest_sha256=manifest_sha256,
        run_dir=run_dir,
    )
    require_equal(
        evaluation.get("screening_games"),
        games_per_policy,
        "run.evaluation.screening_games",
    )
    require_equal(
        evaluation.get("opponent_canonical_order_source"),
        "per_opponent_manifest",
        "run.evaluation.opponent_canonical_order_source",
    )
    screening = require_mapping(summary.get("screening"), "summary.screening")
    require_equal(screening.get("phase"), "screening", "summary.screening.phase")
    require_equal(
        screening.get("games_requested_per_policy"),
        games_per_policy,
        "summary.screening.games_requested_per_policy",
    )
    require_equal(
        screening.get("selected_policies"),
        len(opponents),
        "summary.screening.selected_policies",
    )
    require_equal(
        screening.get("completed_policies"),
        len(opponents),
        "summary.screening.completed_policies",
    )
    require_equal(screening.get("coverage_complete"), True, "summary.screening.coverage_complete")
    require_equal(
        screening.get("strict_even_seat_balance_verified"),
        True,
        "summary.screening.strict_even_seat_balance_verified",
    )
    require_equal(screening.get("failures"), {}, "summary.screening.failures")
    require_equal(
        screening.get("wall_budget_exhausted"),
        False,
        "summary.screening.wall_budget_exhausted",
    )
    return run_config, summary, screening


def validate_nonnegative_count_mapping(value: Any, label: str) -> dict[str, int]:
    raw = require_mapping(value, label)
    result: dict[str, int] = {}
    for key, count in raw.items():
        if not isinstance(key, str) or not key:
            raise AuditError(f"{label} keys must be non-empty strings")
        result[key] = require_int(count, f"{label}.{key}")
    return result


def validate_result(
    *,
    path: Path,
    result: dict[str, Any],
    opponent: Opponent,
    candidate: Path,
    candidate_sha256: str,
    candidate_deck: Path,
    candidate_deck_hash: str,
    run_config: dict[str, Any],
    games_per_policy: int,
) -> PolicyCounts:
    label = f"result[{opponent.policy_id}]"
    candidate_row = require_mapping(result.get("candidate"), f"{label}.candidate")
    opponent_row = require_mapping(result.get("opponent"), f"{label}.opponent")
    engine = require_mapping(result.get("engine"), f"{label}.engine")
    evaluation = require_mapping(result.get("evaluation"), f"{label}.evaluation")
    run_evaluation = require_mapping(run_config.get("evaluation"), "run.evaluation")
    order_contract = candidate_order_contract(run_evaluation)

    require_equal(
        resolve_path(candidate_row.get("path"), path.parent, f"{label}.candidate.path"),
        candidate,
        f"{label}.candidate.path",
    )
    require_equal(candidate_row.get("sha256"), candidate_sha256, f"{label}.candidate.sha256")
    require_equal(
        resolve_path(candidate_row.get("deck"), path.parent, f"{label}.candidate.deck"),
        candidate_deck,
        f"{label}.candidate.deck",
    )
    require_equal(candidate_row.get("deck_hash"), candidate_deck_hash, f"{label}.candidate.deck_hash")
    validate_result_order_identity(
        row=candidate_row,
        expected_mode=order_contract.order_mode,
        expected_canonical=order_contract.canonical_order,
        expected_hybrid=order_contract.hybrid_order,
        legacy_raw=order_contract.legacy_raw,
        label=f"{label}.candidate",
    )
    require_equal(
        resolve_path(opponent_row.get("path"), path.parent, f"{label}.opponent.path"),
        opponent.checkpoint,
        f"{label}.opponent.path",
    )
    require_equal(
        opponent_row.get("sha256"),
        opponent.checkpoint_sha256,
        f"{label}.opponent.sha256",
    )
    require_equal(
        resolve_path(opponent_row.get("deck"), path.parent, f"{label}.opponent.deck"),
        opponent.deck,
        f"{label}.opponent.deck",
    )
    require_equal(opponent_row.get("deck_hash"), opponent.deck_hash, f"{label}.opponent.deck_hash")
    validate_result_order_identity(
        row=opponent_row,
        expected_mode="canonical" if opponent.canonical_order else "raw",
        expected_canonical=opponent.canonical_order,
        expected_hybrid=False,
        legacy_raw=order_contract.legacy_raw,
        label=f"{label}.opponent",
    )

    require_equal(engine.get("games_requested"), games_per_policy, f"{label}.engine.games_requested")
    for field in ("environments", "max_game_decisions"):
        require_equal(
            engine.get(field),
            run_evaluation.get(field),
            f"{label}.engine.{field}",
        )
    seed = require_int(run_evaluation.get("seed"), "run.evaluation.seed")
    require_equal(
        engine.get("python_torch_seed"),
        derived_seed(seed, "screening", opponent.policy_id),
        f"{label}.engine.python_torch_seed",
    )
    require_equal(engine.get("engine_seed_control"), False, f"{label}.engine.engine_seed_control")

    valid_games = require_int(evaluation.get("valid_games"), f"{label}.valid_games")
    wins = require_int(evaluation.get("wins"), f"{label}.wins")
    losses = require_int(evaluation.get("losses"), f"{label}.losses")
    draws = require_int(evaluation.get("draws"), f"{label}.draws")
    invalid_games = require_int(evaluation.get("invalid_games"), f"{label}.invalid_games")
    require_equal(valid_games, games_per_policy, f"{label}.valid game quota")
    require_equal(wins + losses + draws, valid_games, f"{label}.valid W-L-D sum")

    seat_rows = require_mapping(evaluation.get("by_candidate_seat"), f"{label}.by_candidate_seat")
    expected_seat_games = games_per_policy // 2
    seat_wins: dict[str, int] = {}
    seat_valid_games: dict[str, int] = {}
    valid_totals = {"wins": 0, "losses": 0, "draws": 0, "valid_games": 0}
    for seat in ("0", "1"):
        row = require_mapping(seat_rows.get(seat), f"{label}.seat[{seat}]")
        seat_valid = require_int(row.get("valid_games"), f"{label}.seat[{seat}].valid_games")
        seat_win = require_int(row.get("wins"), f"{label}.seat[{seat}].wins")
        seat_loss = require_int(row.get("losses"), f"{label}.seat[{seat}].losses")
        seat_draw = require_int(row.get("draws"), f"{label}.seat[{seat}].draws")
        require_equal(seat_valid, expected_seat_games, f"{label}.seat[{seat}] valid quota")
        require_equal(
            seat_win + seat_loss + seat_draw,
            seat_valid,
            f"{label}.seat[{seat}] W-L-D sum",
        )
        seat_wins[seat] = seat_win
        seat_valid_games[seat] = seat_valid
        for key, value in (
            ("wins", seat_win),
            ("losses", seat_loss),
            ("draws", seat_draw),
            ("valid_games", seat_valid),
        ):
            valid_totals[key] += value
    require_equal(
        valid_totals,
        {"wins": wins, "losses": losses, "draws": draws, "valid_games": valid_games},
        f"{label}.aggregate vs seat W-L-D",
    )
    seat_balance = require_mapping(evaluation.get("seat_balance"), f"{label}.seat_balance")
    expected_quotas = {"0": expected_seat_games, "1": expected_seat_games}
    require_equal(
        seat_balance.get("assignment_mode"),
        "exact_valid_game_quota_v1",
        f"{label}.seat_balance.assignment_mode",
    )
    require_equal(
        seat_balance.get("requested_games_by_candidate_seat"),
        expected_quotas,
        f"{label}.seat_balance.requested_games_by_candidate_seat",
    )
    require_equal(
        seat_balance.get("actual_valid_games_by_candidate_seat"),
        expected_quotas,
        f"{label}.seat_balance.actual_valid_games_by_candidate_seat",
    )
    for field in (
        "requested_games_even",
        "strict_even_balance_required",
        "strict_even_balance_verified",
        "aggregate_equals_seat_sum_verified",
    ):
        require_equal(seat_balance.get(field), True, f"{label}.seat_balance.{field}")
    require_equal(seat_balance.get("actual_valid_game_gap"), 0, f"{label}.seat_balance.actual_valid_game_gap")

    invalid_by_reason = validate_nonnegative_count_mapping(
        evaluation.get("invalid_by_reason"), f"{label}.invalid_by_reason"
    )
    require_equal(sum(invalid_by_reason.values()), invalid_games, f"{label}.invalid reason sum")
    invalid_seats_raw = require_mapping(
        evaluation.get("invalid_by_candidate_seat"),
        f"{label}.invalid_by_candidate_seat",
    )
    require_equal(set(invalid_seats_raw), {"0", "1"}, f"{label}.invalid seat keys")
    invalid_by_seat: dict[str, int] = {}
    invalid_reason_from_seats: dict[str, int] = defaultdict(int)
    for seat in ("0", "1"):
        row = require_mapping(invalid_seats_raw.get(seat), f"{label}.invalid_seat[{seat}]")
        count = require_int(row.get("invalid_games"), f"{label}.invalid_seat[{seat}].invalid_games")
        reasons = validate_nonnegative_count_mapping(
            row.get("by_reason"), f"{label}.invalid_seat[{seat}].by_reason"
        )
        require_equal(sum(reasons.values()), count, f"{label}.invalid_seat[{seat}] reason sum")
        invalid_by_seat[seat] = count
        for reason, reason_count in reasons.items():
            invalid_reason_from_seats[reason] += reason_count
    require_equal(sum(invalid_by_seat.values()), invalid_games, f"{label}.invalid seat sum")
    require_equal(dict(invalid_reason_from_seats), invalid_by_reason, f"{label}.invalid seat/reason cross-audit")
    require_equal(evaluation.get("invalid_audit_complete"), True, f"{label}.invalid_audit_complete")
    validate_nonnegative_count_mapping(
        evaluation.get("invalid_error_messages"),
        f"{label}.invalid_error_messages",
    )

    attempted = valid_games + invalid_games
    conservative = require_mapping(
        evaluation.get("conservative_invalid_as_loss"),
        f"{label}.conservative_invalid_as_loss",
    )
    require_equal(conservative.get("trials"), attempted, f"{label}.conservative.trials")
    require_equal(conservative.get("wins"), wins, f"{label}.conservative.wins")
    require_equal(conservative.get("nonwins"), attempted - wins, f"{label}.conservative.nonwins")
    require_close(conservative.get("win_rate"), wins / attempted, f"{label}.conservative.win_rate")
    require_close(
        conservative.get("invalid_rate"),
        invalid_games / attempted,
        f"{label}.conservative.invalid_rate",
    )
    return PolicyCounts(
        policy_id=opponent.policy_id,
        archetype=opponent.archetype,
        opponent_canonical_order=opponent.canonical_order,
        wins=wins,
        losses=losses,
        draws=draws,
        valid_games=valid_games,
        invalid_games=invalid_games,
        invalid_by_seat=invalid_by_seat,
        seat_wins=seat_wins,
        seat_valid_games=seat_valid_games,
    )


def validate_summary_counts(
    screening: dict[str, Any],
    counts: Sequence[PolicyCounts],
    expected_result_paths: Mapping[str, Path],
) -> None:
    """Cross-check the runner's valid-only summary against raw result files."""

    rows = require_list(screening.get("policies"), "summary.screening.policies")
    by_id: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(rows):
        row = require_mapping(value, f"summary.screening.policies[{index}]")
        policy_id = require_text(row.get("policy_id"), f"summary.policy[{index}].policy_id")
        if policy_id in by_id:
            raise AuditError(f"duplicate summary policy_id: {policy_id!r}")
        by_id[policy_id] = row
    require_equal(set(by_id), {row.policy_id for row in counts}, "summary policy coverage")

    pooled = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
    seats = {
        "0": {"wins": 0, "losses": 0, "draws": 0, "games": 0},
        "1": {"wins": 0, "losses": 0, "draws": 0, "games": 0},
    }
    for item in counts:
        row = by_id[item.policy_id]
        require_equal(row.get("status"), "complete", f"summary[{item.policy_id}].status")
        require_equal(row.get("archetype"), item.archetype, f"summary[{item.policy_id}].archetype")
        summary_canonical_order = require_bool(
            row.get("opponent_canonical_order"),
            f"summary[{item.policy_id}].opponent_canonical_order",
        )
        require_equal(
            summary_canonical_order,
            item.opponent_canonical_order,
            f"summary[{item.policy_id}].opponent_canonical_order",
        )
        require_equal(
            Path(require_text(row.get("result_path"), f"summary[{item.policy_id}].result_path")).resolve(),
            expected_result_paths[item.policy_id],
            f"summary[{item.policy_id}].result_path",
        )
        for field, expected in (
            ("wins", item.wins),
            ("losses", item.losses),
            ("draws", item.draws),
            ("games", item.valid_games),
        ):
            require_equal(row.get(field), expected, f"summary[{item.policy_id}].{field}")
            pooled[field] += expected
        summary_seats = require_mapping(
            row.get("by_candidate_seat"),
            f"summary[{item.policy_id}].by_candidate_seat",
        )
        for seat in ("0", "1"):
            seat_row = require_mapping(summary_seats.get(seat), f"summary[{item.policy_id}].seat[{seat}]")
            raw_result = read_json(expected_result_paths[item.policy_id], f"result[{item.policy_id}]")
            raw_seat = require_mapping(
                require_mapping(raw_result.get("evaluation"), f"result[{item.policy_id}].evaluation")
                .get("by_candidate_seat"),
                f"result[{item.policy_id}].by_candidate_seat",
            )[seat]
            raw_seat = require_mapping(raw_seat, f"result[{item.policy_id}].seat[{seat}]")
            for field in ("wins", "losses", "draws"):
                expected = require_int(raw_seat.get(field), f"result[{item.policy_id}].seat[{seat}].{field}")
                require_equal(seat_row.get(field), expected, f"summary[{item.policy_id}].seat[{seat}].{field}")
                seats[seat][field] += expected
            require_equal(
                seat_row.get("games"),
                item.seat_valid_games[seat],
                f"summary[{item.policy_id}].seat[{seat}].games",
            )
            seats[seat]["games"] += item.seat_valid_games[seat]

    summary_pooled = require_mapping(screening.get("pooled"), "summary.screening.pooled")
    for field, expected in pooled.items():
        require_equal(summary_pooled.get(field), expected, f"summary.screening.pooled.{field}")
    summary_seats = require_mapping(
        screening.get("by_candidate_seat"), "summary.screening.by_candidate_seat"
    )
    for seat in ("0", "1"):
        row = require_mapping(summary_seats.get(seat), f"summary.screening.seat[{seat}]")
        for field, expected in seats[seat].items():
            require_equal(row.get(field), expected, f"summary.screening.seat[{seat}].{field}")


def cvar_lower_tail(values: Sequence[tuple[str, float]]) -> dict[str, Any]:
    count = max(1, math.ceil(len(values) * BOTTOM_QUARTILE_ALPHA))
    selected = sorted(values, key=lambda item: (item[1], item[0]))[:count]
    return {
        "alpha": BOTTOM_QUARTILE_ALPHA,
        "count": count,
        "value": sum(value for _, value in selected) / count,
        "policy_ids": [policy_id for policy_id, _ in selected],
    }


def gate(actual: float | int, minimum: float | int) -> dict[str, Any]:
    return {"actual": actual, "minimum": minimum, "pass": actual >= minimum}


def upper_gate(actual: float, maximum: float) -> dict[str, Any]:
    return {"actual": actual, "maximum": maximum, "pass": actual <= maximum}


def audit_panel(
    *,
    preregistration_path: Path,
    panel_manifest_path: Path,
    run_dir: Path,
    candidate_path: Path,
    candidate_deck_path: Path,
) -> dict[str, Any]:
    preregistration_path = preregistration_path.expanduser().resolve()
    panel_manifest_path = panel_manifest_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    candidate_path = candidate_path.expanduser().resolve()
    candidate_deck_path = candidate_deck_path.expanduser().resolve()
    if not candidate_path.is_file():
        raise AuditError(f"candidate checkpoint does not exist: {candidate_path}")
    if not candidate_deck_path.is_file():
        raise AuditError(f"candidate deck does not exist: {candidate_deck_path}")
    if not run_dir.is_dir():
        raise AuditError(f"league run directory does not exist: {run_dir}")

    _, thresholds = load_preregistered_gate(preregistration_path)
    _, opponents = load_opponents(panel_manifest_path)
    candidate_sha = file_sha256(candidate_path)
    candidate_deck_file_sha = file_sha256(candidate_deck_path)
    candidate_deck_hash = deck_semantic_hash(candidate_deck_path)
    opponent_checkpoint_hashes = {row.checkpoint_sha256 for row in opponents}
    if candidate_sha in opponent_checkpoint_hashes:
        raise AuditError(
            "candidate checkpoint SHA-256 equals an opponent checkpoint: "
            f"{candidate_sha}"
        )

    manifest_sha = file_sha256(panel_manifest_path)
    games_per_policy = int(thresholds["screen_games_per_policy"])
    run_config, _, screening = validate_run_identity(
        run_dir=run_dir,
        manifest_path=panel_manifest_path,
        manifest_sha256=manifest_sha,
        opponents=opponents,
        candidate=candidate_path,
        candidate_sha256=candidate_sha,
        candidate_deck=candidate_deck_path,
        candidate_deck_file_sha256=candidate_deck_file_sha,
        games_per_policy=games_per_policy,
    )
    order_contract = candidate_order_contract(
        require_mapping(run_config.get("evaluation"), "run.evaluation")
    )

    results_dir = run_dir / "results" / "screening"
    expected_paths = {
        opponent.policy_id: (
            results_dir / f"{safe_name(opponent.policy_id)}.json"
        ).resolve()
        for opponent in opponents
    }
    if not results_dir.is_dir():
        raise AuditError(f"screening results directory does not exist: {results_dir}")
    actual_json_paths = {path.resolve() for path in results_dir.glob("*.json")}
    require_equal(
        actual_json_paths,
        set(expected_paths.values()),
        "screening result file coverage",
    )

    counts: list[PolicyCounts] = []
    for opponent in opponents:
        result_path = expected_paths[opponent.policy_id]
        result = read_json(result_path, f"result[{opponent.policy_id}]")
        counts.append(
            validate_result(
                path=result_path,
                result=result,
                opponent=opponent,
                candidate=candidate_path,
                candidate_sha256=candidate_sha,
                candidate_deck=candidate_deck_path,
                candidate_deck_hash=candidate_deck_hash,
                run_config=run_config,
                games_per_policy=games_per_policy,
            )
        )
    validate_summary_counts(screening, counts, expected_paths)

    total_valid = sum(row.valid_games for row in counts)
    total_invalid = sum(row.invalid_games for row in counts)
    total_attempted = total_valid + total_invalid
    total_wins = sum(row.wins for row in counts)
    policy_rates = [
        (row.policy_id, row.conservative_win_rate) for row in counts
    ]
    macro = sum(value for _, value in policy_rates) / len(policy_rates)
    cvar = cvar_lower_tail(policy_rates)
    failure_rate = total_invalid / total_attempted

    archetype_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "valid_games": 0, "invalid_games": 0}
    )
    seat_totals = {
        "0": {"wins": 0, "valid_games": 0, "invalid_games": 0},
        "1": {"wins": 0, "valid_games": 0, "invalid_games": 0},
    }
    for row in counts:
        archetype_totals[row.archetype]["wins"] += row.wins
        archetype_totals[row.archetype]["valid_games"] += row.valid_games
        archetype_totals[row.archetype]["invalid_games"] += row.invalid_games
        for seat in ("0", "1"):
            seat_totals[seat]["wins"] += row.seat_wins[seat]
            seat_totals[seat]["valid_games"] += row.seat_valid_games[seat]
            seat_totals[seat]["invalid_games"] += row.invalid_by_seat[seat]

    archetypes: dict[str, dict[str, Any]] = {}
    for archetype, values in sorted(archetype_totals.items()):
        attempted = values["valid_games"] + values["invalid_games"]
        archetypes[archetype] = {
            **values,
            "attempted_games": attempted,
            "conservative_win_rate": values["wins"] / attempted,
        }
    seats: dict[str, dict[str, Any]] = {}
    for seat, values in seat_totals.items():
        attempted = values["valid_games"] + values["invalid_games"]
        seats[seat] = {
            **values,
            "attempted_games": attempted,
            "conservative_win_rate": values["wins"] / attempted,
        }

    minimum_archetype = min(
        row["conservative_win_rate"] for row in archetypes.values()
    )
    semantic_deck_hashes = sorted({row.deck_hash for row in opponents})
    gates = {
        "candidate_not_an_opponent": {
            "actual": True,
            "required": True,
            "pass": True,
        },
        "exact_policy_count": {
            "actual": len(opponents),
            "required": EXPECTED_PANEL_POLICIES,
            "pass": len(opponents) == EXPECTED_PANEL_POLICIES,
        },
        "preregistered_minimum_policy_count": gate(
            len(opponents), int(thresholds["minimum_policies"])
        ),
        "semantic_deck_hash_count": gate(
            len(semantic_deck_hashes), int(thresholds["minimum_deck_hashes"])
        ),
        "attempted_games": gate(
            total_attempted, int(thresholds["minimum_attempted_games"])
        ),
        "failure_invalid_rate": upper_gate(
            failure_rate, thresholds["maximum_failure_rate"]
        ),
        "macro_policy_win_rate": gate(
            macro, thresholds["macro_win_rate_minimum"]
        ),
        "bottom_quartile_cvar": gate(
            cvar["value"], thresholds["bottom_quartile_cvar_minimum"]
        ),
        "minimum_archetype_win_rate": gate(
            minimum_archetype, thresholds["minimum_archetype_win_rate"]
        ),
        "candidate_seat_0_aggregate_win_rate": gate(
            seats["0"]["conservative_win_rate"],
            thresholds["minimum_each_seat_aggregate_win_rate"],
        ),
        "candidate_seat_1_aggregate_win_rate": gate(
            seats["1"]["conservative_win_rate"],
            thresholds["minimum_each_seat_aggregate_win_rate"],
        ),
    }
    passed = all(row["pass"] is True for row in gates.values())
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": utc_now(),
        "pass": passed,
        "read_only_audit": True,
        "training_evaluation_packaging_submission_performed": False,
        "inputs": {
            "preregistration": str(preregistration_path),
            "preregistration_sha256": file_sha256(preregistration_path),
            "panel_manifest": str(panel_manifest_path),
            "panel_manifest_sha256": manifest_sha,
            "run_directory": str(run_dir),
            "run_config_sha256": file_sha256(run_dir / "run_config.json"),
            "summary_sha256": file_sha256(run_dir / "summary.json"),
            "candidate_checkpoint": str(candidate_path),
            "candidate_checkpoint_sha256": candidate_sha,
            "candidate_deck": str(candidate_deck_path),
            "candidate_deck_file_sha256": candidate_deck_file_sha,
            "candidate_deck_semantic_sha256": candidate_deck_hash,
        },
        "identity_and_coverage": {
            "strict_cross_checks_passed": True,
            "candidate_order_contract": order_contract.public(),
            "policies": len(opponents),
            "policy_ids": [row.policy_id for row in opponents],
            "semantic_deck_hashes": semantic_deck_hashes,
            "semantic_deck_hash_count": len(semantic_deck_hashes),
            "games_per_policy_valid_quota": games_per_policy,
            "valid_games_per_candidate_seat_per_policy": games_per_policy // 2,
            "result_files": len(counts),
        },
        "accounting": {
            "valid_games": total_valid,
            "invalid_games": total_invalid,
            "attempted_games": total_attempted,
            "wins": total_wins,
            "nonwins_including_invalid": total_attempted - total_wins,
            "failure_invalid_rate": failure_rate,
            "invalid_timeout_truncation_treatment": "candidate loss",
        },
        "metrics": {
            "policies": [
                {
                    "policy_id": row.policy_id,
                    "archetype": row.archetype,
                    "wins": row.wins,
                    "valid_games": row.valid_games,
                    "invalid_games": row.invalid_games,
                    "attempted_games": row.attempted_games,
                    "conservative_win_rate": row.conservative_win_rate,
                }
                for row in counts
            ],
            "macro_policy_win_rate": macro,
            "bottom_quartile_cvar": cvar,
            "archetypes": archetypes,
            "minimum_archetype_win_rate": minimum_archetype,
            "by_candidate_seat": seats,
        },
        "gates": gates,
        "decision": (
            "terminal_panel_gate_passed"
            if passed
            else "terminal_panel_gate_failed_preserve_submission"
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
    )
    parser.add_argument(
        "--panel-manifest",
        type=Path,
        default=DEFAULT_PANEL_MANIFEST,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-deck", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Atomically write the audit report to this JSON path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = audit_panel(
            preregistration_path=args.preregistration,
            panel_manifest_path=args.panel_manifest,
            run_dir=args.run_dir,
            candidate_path=args.candidate,
            candidate_deck_path=args.candidate_deck,
        )
    except AuditError as error:
        print(json.dumps({"pass": False, "audit_error": str(error)}))
        return 2
    if args.output is not None:
        atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
