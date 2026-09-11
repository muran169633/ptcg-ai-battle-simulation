#!/usr/bin/env python3
"""Publish a compact CW12 guard plan from the frozen CW11 row-flip report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SOURCE = ROOT / "artifacts/cw11_specialist_dev_row_flips_readonly_20260802_v1.json"
SOURCE_SHA256 = "b8ac6b08cfb7ee6f916216fbf0f70ec6d0c9e790595ed6dfc12f0a0c88be0a7b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_o_excl(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise RuntimeError("unsafe output")
    finally:
        os.close(fd)


def physical_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["archive"]),
        str(row["member"]),
        int(row["line_index"]),
        str(row["line_sha256"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    if sha256_file(SOURCE) != SOURCE_SHA256 or stat.S_IMODE(SOURCE.stat().st_mode) != 0o444:
        raise RuntimeError("source report binding or mode drift")
    source = json.loads(SOURCE.read_text())
    rows = source["unique_physical_changed_rows"]
    raw_metric_rows = [
        row for row in rows if row["pairwise"]["raw_to_cw11"]["changed_metrics"]
    ]
    union_metric_rows = [
        row
        for row in rows
        if any(value["changed_metrics"] for value in row["pairwise"].values())
    ]
    repair = source["cw12_minimum_exact_repair_analysis"]
    candidate_by_key = {
        tuple(value["physical_key"]): value for value in repair["candidate_pool"]
    }
    repair_rows = []
    for row in rows:
        key = physical_key(row)
        if key not in candidate_by_key:
            continue
        first = row["pairwise"]["raw_to_cw11"]["first_prediction_difference"]
        if first is None:
            raise RuntimeError("repair candidate missing first decision difference")
        positive = int(first["left_choice"])
        negative = int(first["right_choice"])
        repair_rows.append(
            {
                "physical_key": list(key),
                "decision_sha256": row["decision_sha256"],
                "team_name": row["team_name"],
                "expert_order": row["expert_order"],
                "positive_option": positive,
                "negative_option": negative,
                "involved_options": row["involved_options"],
                "panel_view_occurrences": row["panel_view_occurrences"],
                "coverage": candidate_by_key[key]["coverage"],
                "raw": row["states"]["raw"],
                "e904": row["states"]["e904"],
                "cw11": row["states"]["cw11"],
                "raw_pair_margin": (
                    row["states"]["raw"]["valid_option_logits"][str(positive)]
                    - row["states"]["raw"]["valid_option_logits"][str(negative)]
                ),
                "cw11_pair_margin": (
                    row["states"]["cw11"]["valid_option_logits"][str(positive)]
                    - row["states"]["cw11"]["valid_option_logits"][str(negative)]
                ),
                "train_exact_matches": candidate_by_key[key]["train_exact_matches"],
            }
        )
    shared = [
        value
        for value in repair_rows
        if all(
            gate in value["coverage"]
            for gate in ("core5.set", "dominic.set", "core5.ordered", "dominic.ordered")
        )
    ]
    pokemonfan = [
        value for value in repair_rows if "pokemonfan.set" in value["coverage"]
    ]
    batch_audit = source["four_original_cw11_guard_batch_shape_audit"]
    shape_checks = {
        "four_guards_exact": len(batch_audit) == 4,
        "all_repaired_in_official_B256": all(
            value["physical_outcome"] == "repaired_in_official_B256"
            for value in batch_audit
        ),
        "no_observed_B33_B256_gate_disagreement": all(
            not value["batch_shape_gate_disagreement"] for value in batch_audit
        ),
        "raw_pair_margins_identical_B33_B256": all(
            value["selected_B33"]["states"]["raw"]["pair_margin_positive_minus_negative"]
            == value["official_B256_stream"]["states"]["raw"][
                "pair_margin_positive_minus_negative"
            ]
            for value in batch_audit
        ),
        "e904_pair_margins_identical_B33_B256": all(
            value["selected_B33"]["states"]["e904"]["pair_margin_positive_minus_negative"]
            == value["official_B256_stream"]["states"]["e904"][
                "pair_margin_positive_minus_negative"
            ]
            for value in batch_audit
        ),
        "cw11_pair_margins_identical_B33_B256": all(
            value["selected_B33"]["states"]["cw11"]["pair_margin_positive_minus_negative"]
            == value["official_B256_stream"]["states"]["cw11"][
                "pair_margin_positive_minus_negative"
            ]
            for value in batch_audit
        ),
    }
    invariants = {
        "source_status_exact": source["status"]
        == "completed_read_only_failure_attribution",
        "decision_changes_exact_47": len(rows) == 47,
        "any_pair_metric_changes_exact_29": len(union_metric_rows) == 29,
        "raw_to_cw11_metric_changes_exact_22": len(raw_metric_rows) == 22,
        "minimum_repair_guards_exact_3": repair["minimum_guard_count"] == 3,
        "minimum_repair_solutions_exact_10": repair["minimum_solution_count"] == 10,
        "mandatory_shared_core5_dominic_exact_1": len(shared) == 1,
        "pokemonfan_two_of_five_exact": len(pokemonfan) == 5,
        "no_train_exact_fingerprint_matches": source["train_exact_scan_audit"][
            "exact_match_count"
        ]
        == 0,
        **shape_checks,
    }
    if not all(invariants.values()):
        raise RuntimeError(f"guard-plan invariant failed: {invariants}")
    report = {
        "schema_version": "ptcg-cw11-failure-attribution-cw12-guardplan-v1",
        "status": "completed_compact_read_only_guard_plan",
        "source_report": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": SOURCE_SHA256,
            "mode": "0444",
        },
        "scope": {
            "trained": False,
            "checkpoint_written": False,
            "cuda_accessed": False,
            "specialist_one_shot_rerun": False,
            "broad_opened": False,
            "gold_opened": False,
            "network_upload_submission": False,
        },
        "invariants": invariants,
        "failure_attribution": {
            "batch_shape_hypothesis": "not_supported_for_the_four_original_guards",
            "evidence": (
                "All four original guards have identical raw/E904/CW11 pair margins in the "
                "selected B33 and recorded official B256 contexts, and all four are physically "
                "repaired.  The failed aggregate gates are caused by other CW11 collateral "
                "decision flips outside those four selected rows."
            ),
            "decision_changed_physical_rows": len(rows),
            "metric_changed_in_any_model_pair": len(union_metric_rows),
            "raw_to_cw11_metric_changed_physical_rows": len(raw_metric_rows),
        },
        "four_original_guard_batch_shape_audit": batch_audit,
        "raw_to_cw11_metric_changed_retention_rows": raw_metric_rows,
        "minimum_repair": {
            "assumption": repair["assumption"],
            "failed_gate_deficits": repair["failed_gate_deficits"],
            "minimum_guard_count": 3,
            "minimum_solution_count": 10,
            "minimum_solutions": repair["minimum_solutions"],
            "mandatory_shared_core5_dominic_guard": shared[0],
            "pokemonfan_choose_any_two_pool": pokemonfan,
            "train_exact_fingerprint_match_count": 0,
        },
        "cw12_contract": {
            "optimization": (
                "Use the one mandatory shared core5/Dominic physical guard plus any two of the "
                "five PokemonFan harmful rows.  Exact train fingerprints do not exist, so these "
                "are consumed-valid development constraints unless a separately preregistered "
                "train-only surrogate search is performed."
            ),
            "official_context": (
                "Reconstruct each row's recorded official B256 stream batch and enforce its "
                "same-shape raw pair floor and required correctness flags.  Also keep compact-B33 "
                "checks if desired, but never use them as the only physical gate."
            ),
            "retention": (
                "Bind all 22 raw-to-CW11 metric-changing physical rows as a no-regression ledger; "
                "retain favorable repairs and block harmful flips."
            ),
            "contamination_boundary": (
                "All six specialist-valid panels and these row identities are development-only. "
                "The consumed failed one-shot must not be rerun or reclassified as promotion "
                "evidence.  Broad and fresh Gold remain untouched and separately authorized."
            ),
            "authorization": {
                "broad": False,
                "gold": False,
                "package": False,
                "submission": False,
            },
        },
    }
    output = args.output.resolve()
    if output.parent != ROOT / "artifacts":
        raise ValueError("--output must be a direct child of artifacts/")
    publish_o_excl(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output.relative_to(ROOT)),
                "output_sha256": sha256_file(output),
                "retention_rows": len(raw_metric_rows),
                "minimum_guards": 3,
                "minimum_solutions": 10,
                "batch_shape_hypothesis": report["failure_attribution"][
                    "batch_shape_hypothesis"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
