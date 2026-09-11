#!/usr/bin/env python3
"""Train the eight frozen recent-Top100 proxy policies sequentially.

Existing same-archetype Gold8 specialists are used only as initialization.
Exact-deck continuations get one low-rate epoch; newly introduced exact decks
get two epochs.  This keeps the proxy refresh bounded while exposing every
training row at least once.  The script writes no package and submits nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
TRAINER = ROOT / "tools/train_bc_orbit.py"
DATA_ROOT = ROOT / "data/top100_proxy_recent14_20260811_v1"
DEFAULT_OUTPUT = ROOT / "artifacts/top100_proxy_recent14_20260811_v1"


GENERAL = ROOT / "artifacts/gold8_recent7_20260808/general_bc/best.pt"
OLD = ROOT / "artifacts/gold8_recent7_20260808"


RECIPES: dict[str, dict[str, Any]] = {
    "alakazam_control": {
        "init": OLD / "alakazam_control/specialist_bc/best.pt", "epochs": 1, "lr": 5e-5,
    },
    "mega_froslass_lopunny": {
        "init": OLD / "mega_froslass_lopunny/specialist_bc/best.pt", "epochs": 1, "lr": 5e-5,
    },
    "mega_lopunny": {
        "init": OLD / "mega_froslass_lopunny/specialist_bc/best.pt", "epochs": 2, "lr": 1e-4,
    },
    "mega_lucario": {
        "init": OLD / "mega_lucario/specialist_bc/best.pt", "epochs": 1, "lr": 5e-5,
    },
    "teal_mask_ogerpon": {
        "init": OLD / "hydrapple_ogerpon/specialist_bc/best.pt", "epochs": 2, "lr": 1e-4,
    },
    "dragapult_ex": {
        "init": OLD / "dragapult_ex/specialist_bc/best.pt", "epochs": 1, "lr": 5e-5,
    },
    "mega_kangaskhan_crustle_current": {
        "init": OLD / "mega_kangaskhan_crustle/specialist_bc/best.pt", "epochs": 2, "lr": 1e-4,
    },
    "cynthias_garchomp_ex": {
        "init": GENERAL, "epochs": 2, "lr": 1.5e-4,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.data_root.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(RECIPES) != set(manifest["profiles"]) - {"marnie"}:
        raise RuntimeError("Proxy recipe set does not match the frozen data manifest")

    seed_base = 2026084100
    run_records: list[dict[str, Any]] = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for index, (slug, recipe) in enumerate(RECIPES.items(), 1):
        profile = manifest["profiles"][slug]
        data = Path(profile["archive"])
        init = Path(recipe["init"])
        output = args.output_root.resolve() / slug
        expected_rows = int(profile["split_decisions"]["train"])
        for path in (data, init, TRAINER, PYTHON):
            if not path.is_file():
                raise FileNotFoundError(path)
        if output.exists():
            if args.resume_completed and (output / "summary.json").is_file():
                print(f"[{index}/8] {slug}: already complete", flush=True)
                continue
            raise FileExistsError(output)
        command = [
            str(PYTHON), "-I", "-B", str(TRAINER),
            "--data", str(data),
            "--output-dir", str(output),
            "--epochs", str(recipe["epochs"]),
            "--batch-size", "256",
            "--workers", str(args.workers),
            "--learning-rate", str(recipe["lr"]),
            "--weight-decay", "0.0001",
            "--seed", str(seed_base + index),
            "--target-accuracy", "0.77",
            "--deck-hash", str(profile["deck_hash"]),
            "--expected-train-rows", str(expected_rows),
            "--init-checkpoint", str(init),
            "--train-shuffle-buffer-rows-per-worker", "4096",
            "--policy-team-balance", "sqrt_clip2_half",
            "--count-trunk-gradient-scale", "0.0",
            "--device", args.device,
        ]
        record = {
            "slug": slug,
            "data": str(data),
            "data_sha256": sha256(data),
            "deck_hash": profile["deck_hash"],
            "init_checkpoint": str(init),
            "init_checkpoint_sha256": sha256(init),
            "expected_train_rows": expected_rows,
            "epochs": recipe["epochs"],
            "learning_rate": recipe["lr"],
            "seed": seed_base + index,
            "output": str(output),
            "command": command,
        }
        run_records.append(record)
        (args.output_root / "training_plan.json").write_text(
            json.dumps({
                "schema_version": "ptcg-top100-proxy-bc-training-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_manifest": str(manifest_path),
                "data_manifest_sha256": sha256(manifest_path),
                "runs": run_records,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{index}/8] training {slug}: rows={expected_rows} epochs={recipe['epochs']}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    summaries: dict[str, Any] = {}
    for slug in RECIPES:
        summary_path = args.output_root / slug / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"Missing completed proxy summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries[slug] = {
            "best_checkpoint": summary["best_checkpoint"],
            "best_checkpoint_sha256": sha256(Path(summary["best_checkpoint"])),
            "best_epoch": summary["best_epoch"],
            "valid_exact_accuracy": summary["best_valid_metrics"]["exact_action_set_accuracy"],
            "test_exact_accuracy": (
                summary.get("test_metrics") or {}
            ).get("exact_action_set_accuracy"),
            "target_reached_on_valid": summary["target_reached_on_valid"],
        }
    final = {
        "schema_version": "ptcg-top100-proxy-bc-training-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
    }
    (args.output_root / "proxy_results.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
