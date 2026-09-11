#!/usr/bin/env python3
"""Plan or run BC retraining for Mega Froslass/Mega Lopunny.

Two preregistered variants address the Aug 1-7 demonstrator shift:

* ``recent_demo`` trains only on JB Bryant and LiamK, the demonstrators that
  persist into the validation/test period.
* ``balanced_all`` retains all training rows but applies mild actor-loss team
  balancing so the early-week Luca rows do not dominate the specialist.

Both variants start from the same frozen recent-Top20 general BC, detach count
loss gradients from the shared trunk, use the archive's strict time split, and
never train on the test day.  Commands are dry-run unless ``--execute`` is set.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
TRAINER = ROOT / "tools/train_bc_orbit.py"
DATA = ROOT / "data/gold8_recent7_20260808/archives/mega_froslass_lopunny.zip"
GENERAL_BC = ROOT / "artifacts/gold8_recent7_20260808/general_bc/best.pt"
OUTPUT_ROOT = ROOT / "artifacts/portfolio4_recent7_20260809/mega_froslass_lopunny"
DECK_HASH = "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc"

ALL_TRAIN_ROWS = 47_552
RECENT_DEMO_TRAIN_ROWS = 23_049


def manifest() -> dict[str, Any]:
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    with zipfile.ZipFile(DATA) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"{DATA}: ZIP failure at {bad_member}")
        value = json.loads(archive.read("manifest.json"))
    split = value.get("split_policy") or {}
    if split.get("mode") != "time":
        raise ValueError("Froslass BC requires the frozen strict time split")
    train_rows = int((value.get("split_decisions") or {}).get("train", 0))
    if train_rows != ALL_TRAIN_ROWS:
        raise ValueError(f"Froslass train row drift: {train_rows}")
    return value


def common(output: Path, expected_train_rows: int, seed: int) -> list[str]:
    return [
        str(PYTHON),
        "-B",
        str(TRAINER),
        "--data",
        str(DATA),
        "--output-dir",
        str(output),
        "--epochs",
        "4",
        "--batch-size",
        "256",
        "--workers",
        "8",
        "--learning-rate",
        "0.00005",
        "--weight-decay",
        "0.0001",
        "--categorical-dim",
        "64",
        "--model-dim",
        "128",
        "--layers",
        "4",
        "--heads",
        "4",
        "--dropout",
        "0.05",
        "--hash-size",
        "65536",
        "--max-state-entities",
        "80",
        "--entity-fields",
        "20",
        "--option-fields",
        "24",
        "--set-bce-weight",
        "0.25",
        "--count-loss-weight",
        "1.0",
        "--value-loss-weight",
        "0.05",
        "--seed",
        str(seed),
        "--target-accuracy",
        "0.75",
        "--trajectory-weight-scope",
        "all_losses",
        "--train-shuffle-buffer-rows-per-worker",
        "0",
        "--flexible-selection-loss-weight",
        "1.0",
        "--count-trunk-gradient-scale",
        "0.0",
        "--deck-hash",
        DECK_HASH,
        "--expected-train-rows",
        str(expected_train_rows),
        "--init-checkpoint",
        str(GENERAL_BC),
        "--split-mode",
        "archive",
        "--skip-test",
        "--device",
        "cuda",
    ]


def variant_command(name: str) -> list[str]:
    if name == "recent_demo":
        command = common(
            OUTPUT_ROOT / "bc_recent_demo_counttrunk0_v1",
            RECENT_DEMO_TRAIN_ROWS,
            202608921,
        )
        insertion = command.index("--expected-train-rows")
        command[insertion:insertion] = [
            "--policy-team-balance",
            "none",
            "--team-name",
            "JB Bryant",
            "--team-name",
            "LiamK",
        ]
        return command
    if name == "balanced_all":
        command = common(
            OUTPUT_ROOT / "bc_balanced_all_counttrunk0_v1",
            ALL_TRAIN_ROWS,
            202608922,
        )
        insertion = command.index("--expected-train-rows")
        command[insertion:insertion] = [
            "--policy-team-balance",
            "sqrt_clip2_half",
        ]
        return command
    raise ValueError(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("all", "recent_demo", "balanced_all"),
        default="all",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest()
    if not PYTHON.is_file() or not TRAINER.is_file() or not GENERAL_BC.is_file():
        raise FileNotFoundError("Python, trainer, or frozen general BC is missing")
    variants = (
        ("recent_demo", "balanced_all")
        if args.variant == "all"
        else (args.variant,)
    )
    for name in variants:
        command = variant_command(name)
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
