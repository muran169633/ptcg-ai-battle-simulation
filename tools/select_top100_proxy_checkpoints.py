#!/usr/bin/env python3
"""Select refreshed versus initialization proxy checkpoints on Aug-09 valid.

Training summaries already contain refreshed validation/test metrics.  This
tool independently evaluates each frozen initialization on the same validation
archive, selects the higher validation exact-action accuracy, and evaluates the
selected initialization on Aug-10 test when needed.  Test never influences the
selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
EVALUATOR = ROOT / "tools/evaluate_policy_bc.py"
PROXY_ROOT = ROOT / "artifacts/top100_proxy_recent14_20260811_v1"
DATA_ROOT = ROOT / "data/top100_proxy_recent14_20260811_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", type=Path, default=PROXY_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    return parser.parse_args()


def run_eval(checkpoint: Path, data: Path, split: str, output: Path) -> dict:
    command = [
        str(PYTHON), "-I", "-B", str(EVALUATOR),
        "--checkpoint", str(checkpoint),
        "--data", str(data),
        "--split", split,
        "--batch-size", "256",
        "--workers", "8",
        "--prediction-order", "canonical",
        "--device", "cuda",
        "--progress-interval", "0",
        "--json-output", str(output),
        "--compact",
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    plan = json.loads((args.proxy_root / "training_plan.json").read_text(encoding="utf-8"))
    refreshed = json.loads((args.proxy_root / "proxy_results.json").read_text(encoding="utf-8"))["summaries"]
    runs = {row["slug"]: row for row in plan["runs"]}
    if set(runs) != set(refreshed):
        raise RuntimeError("Training plan and proxy results differ")
    audit_root = args.proxy_root / "initialization_eval"
    if audit_root.exists():
        raise FileExistsError(audit_root)
    audit_root.mkdir(parents=True)

    selections = {}
    for index, slug in enumerate(runs, 1):
        run = runs[slug]
        init = Path(run["init_checkpoint"])
        data = Path(run["data"])
        print(f"[{index}/8] evaluating frozen initialization: {slug}", flush=True)
        init_valid = run_eval(init, data, "valid", audit_root / f"{slug}.init.valid.json")
        init_valid_accuracy = float(init_valid["metrics"]["exact_action_set_accuracy"])
        refreshed_valid_accuracy = float(refreshed[slug]["valid_exact_accuracy"])
        use_refreshed = refreshed_valid_accuracy >= init_valid_accuracy
        selected_path = Path(refreshed[slug]["best_checkpoint"]) if use_refreshed else init
        if use_refreshed:
            selected_test_accuracy = refreshed[slug]["test_exact_accuracy"]
            init_test_accuracy = None
        else:
            init_test = run_eval(init, data, "test", audit_root / f"{slug}.init.test.json")
            init_test_accuracy = float(init_test["metrics"]["exact_action_set_accuracy"])
            selected_test_accuracy = init_test_accuracy
        selections[slug] = {
            "selection_split": "valid",
            "selected": "refreshed" if use_refreshed else "initialization",
            "selected_checkpoint": str(selected_path.resolve()),
            "selected_checkpoint_sha256": sha256(selected_path),
            "selected_valid_exact_accuracy": max(init_valid_accuracy, refreshed_valid_accuracy),
            "selected_test_exact_accuracy": selected_test_accuracy,
            "initialization": {
                "checkpoint": str(init.resolve()),
                "sha256": sha256(init),
                "valid_exact_accuracy": init_valid_accuracy,
                "test_exact_accuracy": init_test_accuracy,
            },
            "refreshed": {
                "checkpoint": refreshed[slug]["best_checkpoint"],
                "sha256": refreshed[slug]["best_checkpoint_sha256"],
                "valid_exact_accuracy": refreshed_valid_accuracy,
                "test_exact_accuracy": refreshed[slug]["test_exact_accuracy"],
            },
            "test_not_used_for_selection": True,
        }
    result = {
        "schema_version": "ptcg-top100-proxy-checkpoint-selection-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "Aug-09 valid exact action-set accuracy, canonical order",
        "test_date": "2026-08-10",
        "test_not_used_for_selection": True,
        "selections": selections,
    }
    target = args.proxy_root / "proxy_checkpoint_selection.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
