#!/usr/bin/env python3
"""Plan or run general BC -> eight-deck PPO -> exact-deck special BC.

General BC uses all visible decisions from the frozen recent Top20 week.  PPO
uses that frozen general policy with each of the eight 60-card lists as a
balanced opponent pool.  The Marnie profile is the primary route and, by
default, resumes the submitted U472 checkpoint; ``--fresh-marnie-ppo`` gives
the literal general-BC -> PPO branch.  Special BC then consumes only the
selected profile's exact-deck archive and updates the actor without the value
head.  Exact specialist-only BC remains available as an explicit ablation.

Commands are dry-run by default.  Pass ``--execute`` to launch training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/gold8_recent7_20260808/manifest.json"
DEFAULT_GENERAL_DATA = ROOT / "data/gold8_recent7_20260808/general_top20.zip"
DEFAULT_OUTPUT = ROOT / "artifacts/gold8_recent7_20260808"
DEFAULT_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
TRAIN_BC = ROOT / "tools/train_bc_orbit.py"
TRAIN_PPO = ROOT / "tools/train_ppo.py"
SPECIAL_BC = ROOT / "tools/apply_gold8_special_bc.py"
EXPECTED_SCHEMA = "ptcg-gold8-exact-deck-week-v1"
MARNIE_RESUME = (
    ROOT
    / "artifacts/ppo_guardedu468_updatedreplay_g8_ppo4x96_u472_"
    "freshjointactor6_s8_design202608148/ppo_stage/B_gold_league/"
    "seed-202608148/checkpoints/update-0472.pt"
)


PROFILE_ORDER = (
    "marnie",
    "mega_froslass_lopunny",
    "mega_kangaskhan_crustle",
    "mega_lucario",
    "alakazam_control",
    "dragapult_ex",
    "hydrapple_ogerpon",
    "ns_zoroark_ex",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, verify_hashes: bool) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError(
            f"Unexpected manifest schema: {manifest.get('schema_version')!r}"
        )
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict) or tuple(profiles) != PROFILE_ORDER:
        if not isinstance(profiles, dict) or set(profiles) != set(PROFILE_ORDER):
            raise ValueError("Gold8 manifest does not contain the frozen profiles")
    for slug in PROFILE_ORDER:
        raw = profiles[slug]
        archive = Path(raw["archive"])
        deck = Path(raw["deck"])
        if not archive.is_file() or not deck.is_file():
            raise FileNotFoundError(f"{slug}: missing archive or deck")
        if verify_hashes:
            archive_hash = file_sha256(archive)
            deck_hash = file_sha256(deck)
            if archive_hash != raw["archive_sha256"]:
                raise ValueError(f"{slug}: archive SHA256 drifted")
            if deck_hash != raw["deck_sha256"]:
                raise ValueError(f"{slug}: deck file SHA256 drifted")
        split_rows = raw.get("split_decisions") or {}
        if any(int(split_rows.get(split, 0)) <= 0 for split in ("train", "valid", "test")):
            raise ValueError(f"{slug}: incomplete train/valid/test split")
    general = manifest.get("general_bc")
    if not isinstance(general, dict):
        raise ValueError("Gold8 manifest has no registered general BC archive")
    general_archive = Path(str(general.get("archive") or ""))
    if not general_archive.is_file():
        raise FileNotFoundError(general_archive)
    if verify_hashes and file_sha256(general_archive) != general.get(
        "archive_sha256"
    ):
        raise ValueError("General BC archive SHA256 drifted")
    return manifest


def selected_profiles(
    stage: str,
    requested: list[str],
    all_profiles: bool,
) -> list[str]:
    if requested and all_profiles:
        raise ValueError("Use either repeated --profile or --all-profiles")
    if all_profiles:
        if stage == "general-bc":
            raise ValueError("General BC does not select deck profiles")
        return list(PROFILE_ORDER)
    if requested:
        unknown = sorted(set(requested) - set(PROFILE_ORDER))
        if unknown:
            raise ValueError(f"Unknown profiles: {unknown}")
        return [slug for slug in PROFILE_ORDER if slug in set(requested)]
    if stage == "specialist-bc":
        return list(PROFILE_ORDER)
    if stage == "general-bc":
        return []
    return ["marnie"]


def archive_train_rows(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"{path}: ZIP failure at {bad_member}")
        manifest = json.loads(archive.read("manifest.json"))
    rows = int((manifest.get("split_decisions") or {}).get("train", 0))
    if rows <= 0:
        raise ValueError(f"{path}: archive has no training rows")
    split_policy = manifest.get("split_policy") or {}
    if split_policy.get("mode") != "time":
        raise ValueError(f"{path}: general BC requires a strict time split")
    return rows


def general_bc_command(
    python: Path,
    output_root: Path,
    general_data: Path,
    device: str,
) -> list[str]:
    train_rows = archive_train_rows(general_data)
    return [
        str(python),
        "-B",
        str(TRAIN_BC),
        "--data",
        str(general_data),
        "--output-dir",
        str(output_root / "general_bc"),
        "--epochs",
        "8",
        "--batch-size",
        "256",
        "--workers",
        "8",
        "--learning-rate",
        "0.0003",
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
        "202608800",
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
        "1.0",
        "--expected-train-rows",
        str(train_rows),
        "--split-mode",
        "archive",
        "--skip-test",
        "--device",
        device,
    ]


def specialist_bc_command(
    python: Path,
    output_root: Path,
    slug: str,
    raw: dict[str, Any],
    device: str,
) -> list[str]:
    train_rows = int(raw["split_decisions"]["train"])
    general_checkpoint = output_root / "general_bc/best.pt"
    if not general_checkpoint.is_file():
        raise FileNotFoundError(
            f"{general_checkpoint} is missing; complete general BC first"
        )
    epochs = 8 if slug == "marnie" else 12
    seed = 202608800 + PROFILE_ORDER.index(slug)
    return [
        str(python),
        "-B",
        str(TRAIN_BC),
        "--data",
        str(Path(raw["archive"])),
        "--output-dir",
        str(output_root / slug / "specialist_bc"),
        "--epochs",
        str(epochs),
        "--batch-size",
        "256",
        "--workers",
        "8",
        "--learning-rate",
        "0.0003",
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
        "1.0",
        "--deck-hash",
        str(raw["deck_hash"]),
        "--expected-train-rows",
        str(train_rows),
        "--init-checkpoint",
        str(general_checkpoint),
        "--split-mode",
        "archive",
        "--skip-test",
        "--device",
        device,
    ]


def checkpoint_update(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "update" not in payload:
        raise ValueError(f"Resume checkpoint has no update: {path}")
    return int(payload["update"])


def ppo_command(
    python: Path,
    output_root: Path,
    slug: str,
    profiles: dict[str, Any],
    device: str,
    marnie_resume: Path | None,
    require_trained_checkpoints: bool,
) -> list[str]:
    raw = profiles[slug]
    own_bc = output_root / slug / "specialist_bc/best.pt"
    if require_trained_checkpoints and not own_bc.is_file():
        raise FileNotFoundError(
            f"{own_bc} is missing; complete specialist BC before PPO"
        )
    for opponent_slug in PROFILE_ORDER:
        checkpoint = output_root / opponent_slug / "specialist_bc/best.pt"
        if require_trained_checkpoints and not checkpoint.is_file():
            raise FileNotFoundError(
                f"PPO requires all eight specialist BC opponents; missing {checkpoint}"
            )

    resume: Path | None = None
    if slug == "marnie" and marnie_resume is not None:
        resume = marnie_resume.resolve()
        if not resume.is_file():
            raise FileNotFoundError(resume)
        parent_update = checkpoint_update(resume)
        schedule_start = parent_update + 1
        final_update = parent_update + 4
    else:
        schedule_start = 1
        final_update = 4

    command = [
        str(python),
        "-B",
        str(TRAIN_PPO),
        "--bc-checkpoint",
        str(own_bc),
        "--deck",
        str(Path(raw["deck"])),
        "--output-dir",
        str(output_root / slug / "ppo"),
        "--updates",
        str(final_update),
        "--schedule-start-update",
        str(schedule_start),
        "--environments",
        "16",
        "--games-per-update",
        "96",
        "--ppo-epochs",
        "2",
        "--minibatch-size",
        "512",
        "--learning-rate",
        "0.000024",
        "--value-learning-rate",
        "0.0000075",
        "--weight-decay",
        "0.0001",
        "--learning-rate-schedule",
        "constant",
        "--gamma",
        "1.0",
        "--gae-lambda",
        "0.97",
        "--advantage-normalization",
        "global",
        "--clip-ratio",
        "0.15",
        "--value-coefficient",
        "0.25",
        "--value-trunk-gradient-scale",
        "1.0",
        "--entropy-coefficient",
        "0.001",
        "--max-grad-norm",
        "0.5",
        "--policy-temperature",
        "0.8",
        "--trainable-scope",
        "last_block_heads",
        "--bc-kl-start",
        "0.016",
        "--bc-kl-end",
        "0.012",
        "--target-kl",
        "0.006",
        "--league-probability",
        "1.0",
        "--opponent-sampling",
        "per_game",
        "--opponent-quota-mode",
        "fixed",
        "--opponent-quota-seat-balance",
        "--ppo-objective",
        "standard",
        "--actor-value-gradient-mode",
        "scalar",
        "--constrained-gradient-mode",
        "scalar",
        "--actor-reduction",
        "episode_mean",
        "--snapshot-interval",
        "1000000",
        "--max-pool-size",
        "8",
        "--bc-replay-data",
        str(Path(raw["archive"])),
        "--bc-replay-split",
        "train",
        "--bc-replay-batches",
        "72",
        "--bc-replay-batch-size",
        "256",
        "--bc-replay-workers",
        "8",
        "--bc-replay-steps",
        "2",
        "--bc-replay-lr-scale",
        "0.075",
        "--bc-replay-loss",
        "ordered",
        "--bc-replay-order-context-weight",
        "8.0",
        "--bc-replay-non-context34-fixed-multi-action-order-weight",
        "1.0",
        "--bc-replay-context34-rows-per-batch",
        "4",
        "--eval-interval",
        str(final_update),
        "--eval-games",
        "32",
        "--eval-all-permanent-opponents",
        "--selection-aggregation",
        "mean",
        "--checkpoint-interval",
        str(final_update),
        "--skip-initial-eval",
        "--seed",
        str(202608900 + PROFILE_ORDER.index(slug)),
        "--device",
        device,
        "--opponent-base-quota",
        "bc",
        "12",
    ]
    for opponent_slug in PROFILE_ORDER:
        if opponent_slug == slug:
            continue
        opponent = profiles[opponent_slug]
        checkpoint = output_root / opponent_slug / "specialist_bc/best.pt"
        deck = Path(opponent["deck"])
        command.extend(("--extra-opponent", str(checkpoint), str(deck)))
        name = f"{checkpoint.stem}@{deck.stem}"
        command.extend(("--opponent-base-quota", name, "12"))
    if resume is not None:
        command.extend(
            (
                "--resume",
                str(resume),
                "--resume-learner-weights",
                "resume",
                "--reset-optimizer-on-resume",
                "--reset-opponent-quota-on-resume",
            )
        )
    return command


def special_bc_command(
    python: Path,
    output_root: Path,
    slug: str,
    raw: dict[str, Any],
    special_data: Path,
    device: str,
    require_parent_checkpoint: bool,
) -> list[str]:
    if not special_data.is_file():
        raise FileNotFoundError(special_data)
    parent = output_root / slug / "ppo/best.pt"
    if require_parent_checkpoint and not parent.is_file():
        raise FileNotFoundError(
            f"{slug}: {parent} is missing; complete PPO before special BC"
        )
    return [
        str(python),
        "-B",
        str(SPECIAL_BC),
        "--parent-checkpoint",
        str(parent),
        "--special-data",
        str(special_data),
        "--expected-deck-hash",
        str(raw["deck_hash"]),
        "--steps",
        "8",
        "--cache-batches",
        "32",
        "--batch-size",
        "256",
        "--learning-rate-scale",
        "0.05",
        "--context34-rows-per-batch",
        "1",
        "--seed",
        str(202609000 + PROFILE_ORDER.index(slug)),
        "--output-dir",
        str(output_root / slug / "special_bc"),
        "--device",
        device,
    ]


def run_command(command: list[str], execute: bool) -> None:
    print(shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--general-data", type=Path, default=DEFAULT_GENERAL_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=("general-bc", "ppo", "special-bc", "specialist-bc"),
        required=True,
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help=(
            "Profile slug; repeat as needed. Specialist BC defaults to all "
            "eight; PPO and special BC default to Marnie."
        ),
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Run the selected deck-specific stage for all eight profiles.",
    )
    parser.add_argument("--special-data", type=Path)
    parser.add_argument("--marnie-resume", type=Path, default=MARNIE_RESUME)
    parser.add_argument("--fresh-marnie-ppo", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    if args.stage == "special-bc" and not SPECIAL_BC.is_file():
        raise FileNotFoundError(SPECIAL_BC)
    manifest = load_manifest(
        args.manifest.resolve(),
        verify_hashes=not args.skip_hash_check,
    )
    profiles = manifest["profiles"]
    selected = selected_profiles(args.stage, args.profile, args.all_profiles)
    if args.stage == "general-bc" and (args.profile or args.all_profiles):
        raise ValueError("General BC does not accept profile selection")
    if (
        args.stage == "special-bc"
        and args.special_data is not None
        and len(selected) != 1
    ):
        raise ValueError(
            "An override --special-data requires exactly one --profile"
        )

    output_root = args.output_root.resolve()
    if args.stage == "general-bc":
        stage_outputs = [output_root / "general_bc"]
    else:
        stage_outputs = [
            output_root / slug / args.stage.replace("-", "_")
            for slug in selected
        ]
    if args.execute:
        existing_outputs = [
            path for path in stage_outputs if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(
                "Refusing to start because stage outputs already exist: "
                + ", ".join(str(path) for path in existing_outputs)
            )
    if args.stage == "general-bc":
        command = general_bc_command(
            python,
            output_root,
            args.general_data.resolve(),
            args.device,
        )
        run_command(command, args.execute)
        return 0

    for slug in selected:
        if args.stage == "specialist-bc":
            command = specialist_bc_command(
                python, output_root, slug, profiles[slug], args.device
            )
        elif args.stage == "ppo":
            resume = None if args.fresh_marnie_ppo else args.marnie_resume
            command = ppo_command(
                python,
                output_root,
                slug,
                profiles,
                args.device,
                resume,
                require_trained_checkpoints=args.execute,
            )
        else:
            special_data = (
                args.special_data.resolve()
                if args.special_data is not None
                else Path(profiles[slug]["archive"])
            )
            command = special_bc_command(
                python,
                output_root,
                slug,
                profiles[slug],
                special_data,
                args.device,
                require_parent_checkpoint=args.execute,
            )
        run_command(command, args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
