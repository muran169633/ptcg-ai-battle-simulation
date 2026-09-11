#!/usr/bin/env python3
"""Plan or train the two missing BC experts in the four-deck portfolio.

Alakazam and Marnie are frozen incumbents.  Slowking/Mega Kangaskhan starts
from the frozen recent-week general BC.  Crustle/Mega Kangaskhan is trained on
MissingNo. only because the exact hash changes demonstrator at the end of the
week: LiamK supplies Aug 2-4 rows, while MissingNo. supplies Aug 5-7 rows.

Commands are printed by default.  Pass ``--execute`` to run them.
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
TRAIN = ROOT / "tools/train_bc_orbit.py"
GENERAL = ROOT / "artifacts/gold8_recent7_20260808/general_bc/best.pt"
OUTPUT = ROOT / "artifacts/portfolio4_recent7_20260809"

SLOWKING_HASH = "df6f7443719675711c2dcead3559087de996f3b8c43192d55ebfe6182e43cb7c"
CRUSTLE_HASH = "4bf59ca589c2d685d74e3535424c2dbe3c11389dffba59eddba4567be7e437df"

SLOWKING_DATA = (
    ROOT
    / "data/portfolio4_recent7_20260809/archives/slowking_mega_kangaskhan.zip"
)
CRUSTLE_DATA = (
    ROOT / "data/gold8_recent7_20260808/archives/mega_kangaskhan_crustle.zip"
)


def archive_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"{path}: ZIP failure at {bad_member}")
        return json.loads(archive.read("manifest.json"))


def common_command(
    data: Path,
    output: Path,
    deck_hash: str,
    train_rows: int,
    epochs: int,
    seed: int,
    learning_rate: str = "0.00005",
) -> list[str]:
    return [
        str(PYTHON),
        "-B",
        str(TRAIN),
        "--data",
        str(data),
        "--output-dir",
        str(output),
        "--epochs",
        str(epochs),
        "--batch-size",
        "256",
        "--workers",
        "8",
        "--learning-rate",
        learning_rate,
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
        "--policy-team-balance",
        "none",
        "--trajectory-weight-scope",
        "all_losses",
        "--train-shuffle-buffer-rows-per-worker",
        "0",
        "--flexible-selection-loss-weight",
        "1.0",
        "--count-trunk-gradient-scale",
        "0.0",
        "--deck-hash",
        deck_hash,
        "--expected-train-rows",
        str(train_rows),
        "--init-checkpoint",
        str(GENERAL),
        "--split-mode",
        "archive",
        "--skip-test",
        "--device",
        "cuda",
    ]


def commands(stage: str) -> list[tuple[str, list[str]]]:
    if not PYTHON.is_file() or not TRAIN.is_file() or not GENERAL.is_file():
        raise FileNotFoundError("Python, trainer, or frozen general BC is missing")
    slowking = archive_manifest(SLOWKING_DATA)
    crustle = archive_manifest(CRUSTLE_DATA)
    slowking_rows = int(slowking["split_decisions"]["train"])
    if slowking_rows != 34_029:
        raise ValueError(f"Slowking train row drift: {slowking_rows}")

    jobs: list[tuple[str, list[str]]] = []
    if stage in {"all", "slowking"}:
        jobs.append(
            (
                "slowking",
                common_command(
                    SLOWKING_DATA,
                    OUTPUT / "slowking_mega_kangaskhan/bc_counttrunk0_v1",
                    SLOWKING_HASH,
                    slowking_rows,
                    epochs=4,
                    seed=202608901,
                ),
            )
        )
    if stage in {"all", "crustle"}:
        team_rows = int((crustle.get("team_decisions") or {}).get("MissingNo.", 0))
        # Aggregate count includes Aug 5 train plus Aug 6-7 holdouts.
        if team_rows != 10_523:
            raise ValueError(f"Crustle MissingNo. row drift: {team_rows}")
        command = common_command(
            CRUSTLE_DATA,
            OUTPUT / "mega_kangaskhan_crustle/bc_missingno_counttrunk0_v1",
            CRUSTLE_HASH,
            train_rows=862,
            epochs=16,
            seed=202608902,
        )
        command[command.index("--expected-train-rows"):command.index("--expected-train-rows")] = [
            "--team-name",
            "MissingNo.",
        ]
        jobs.append(("crustle", command))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "slowking", "crustle"), default="all")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = commands(args.stage)
    for name, command in jobs:
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
