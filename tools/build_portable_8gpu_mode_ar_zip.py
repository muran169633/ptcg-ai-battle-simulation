#!/usr/bin/env python3
"""Build and verify the submitted-update40 mode-AR 8-GPU PPO ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ptcg_mode_ar_u40_8gpu_ppo_20260816"
FROZEN_REL = Path(
    "artifacts/frozen_incumbents/"
    "dragapult_mode_ar_submit55527088_20260815_v1"
)
META_REL = Path("data/recent_day_meta_pool_20260813_top23_v1")
EXPECTED_ANCHOR_SHA256 = (
    "5c8e2659a9a6528202e4f24a75cb0058321584ba4160b0514aca59fb3ffb8e0e"
)
EXPECTED_BC_SHA256 = (
    "377f71c3153f1261c9b1879611352cd978c610e10251405d7f672dbafe498eb7"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, executable: bool = False) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if executable:
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / f"{PACKAGE_NAME}.zip",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    anchor = ROOT / FROZEN_REL / "submitted_policy_update0040.pt"
    bc_base = ROOT / FROZEN_REL / "bc_base_best.pt"
    if sha256(anchor) != EXPECTED_ANCHOR_SHA256:
        raise RuntimeError("Frozen submitted update40 SHA-256 mismatch")
    if sha256(bc_base) != EXPECTED_BC_SHA256:
        raise RuntimeError("Frozen original BC SHA-256 mismatch")

    files = [
        Path("tools/train_ppo.py"),
        Path("tools/train_bc_orbit.py"),
        Path("tools/train_bc_mode_ar_v7.py"),
        Path("tools/evaluate_ppo_head_to_head.py"),
        Path("tools/evaluate_mode_ar_vs_submission.py"),
        Path("tools/train_mode_ar_ppo.py"),
        Path("tools/parallel_rollout.py"),
        Path("tools/train_mode_ar_ppo_ddp.py"),
        Path("tools/run_8gpu_mode_ar_ppo.py"),
        Path("tools/selfcheck_portable_8gpu_mode_ar.py"),
        Path("dataset/sample_submission/sample_submission/cg/__init__.py"),
        Path("dataset/sample_submission/sample_submission/cg/sim.py"),
        Path("dataset/sample_submission/sample_submission/cg/libcg.so"),
        FROZEN_REL / "manifest.json",
        FROZEN_REL / "bc_base_best.pt",
        FROZEN_REL / "submitted_policy_update0040.pt",
        FROZEN_REL / "deck.csv",
        Path("PORTABLE_8GPU_MODE_AR_README.md"),
        Path("portable_8gpu_requirements.txt"),
        Path("setup_and_run_8gpu_mode_ar.sh"),
    ]
    files.extend(sorted((ROOT / META_REL / "decks").glob("*.csv")))

    with tempfile.TemporaryDirectory(prefix="ptcg-mode-ar-8gpu-") as temporary:
        package_root = Path(temporary) / PACKAGE_NAME
        for item in files:
            source = item if item.is_absolute() else ROOT / item
            relative = source.relative_to(ROOT)
            copy_file(
                source,
                package_root / relative,
                executable=(
                    relative.suffix == ".sh"
                    or relative.name
                    in {
                        "run_8gpu_mode_ar_ppo.py",
                        "selfcheck_portable_8gpu_mode_ar.py",
                    }
                ),
            )

        source_meta = json.loads(
            (ROOT / META_REL / "meta_pool.json").read_text(encoding="utf-8")
        )
        portable_meta = dict(source_meta)
        portable_rows = []
        for row in source_meta.get("opponents", []):
            portable_row = dict(row)
            portable_row["deck_path"] = str(
                META_REL / "decks" / Path(str(row["deck_path"])).name
            )
            portable_rows.append(portable_row)
        portable_meta["opponents"] = portable_rows
        meta_destination = package_root / META_REL / "meta_pool.json"
        meta_destination.parent.mkdir(parents=True, exist_ok=True)
        meta_destination.write_text(
            json.dumps(portable_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest_rows = []
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                manifest_rows.append(
                    {
                        "path": str(path.relative_to(package_root)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        manifest = {
            "schema_version": "ptcg-mode-ar-u40-portable-8gpu-v1-a100-40gb",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_name": PACKAGE_NAME,
            "platform": "Linux x86_64",
            "policy_feature_version": "ptcg-mode-ar-ppo-terminal01-v1",
            "anchor_source": "submitted Dragapult PPO update40, submission 55527088",
            "anchor_sha256": sha256(package_root / FROZEN_REL / "submitted_policy_update0040.pt"),
            "original_bc_feature_version": "ptcg-bc-mode-ar-pointer-v7",
            "original_bc_sha256": sha256(package_root / FROZEN_REL / "bc_base_best.pt"),
            "default_world_size": 8,
            "target_gpu": "NVIDIA A100 40GB",
            "default_updates": 8_000,
            "default_global_games_per_update": 2_048,
            "default_planned_training_games": 16_384_000,
            "default_minibatch_size_per_rank": 4_096,
            "default_effective_global_minibatch_size": 32_768,
            "default_rollout_workers_per_rank": 4,
            "default_total_rollout_workers": 32,
            "default_rollout_envs_per_worker": 32,
            "sampling": {
                "fixed_recent_meta": 0.70,
                "inverse_window": 0.20,
                "current_policy_selfplay": 0.10,
            },
            "champion_gate": {
                "interval_updates": 5,
                "games": 1_000,
                "minimum_win_rate": 0.54,
                "error_margin": 0.05,
                "rollback_on_failure": True,
                "optimizer_reset_on_rollback": True,
            },
            "checkpoint_writer": "rank0_only",
            "gradient_sync": "torch DistributedDataParallel NCCL",
            "files": manifest_rows,
        }
        (package_root / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        temporary_zip = output.with_suffix(output.suffix + ".tmp")
        if temporary_zip.exists():
            temporary_zip.unlink()
        with zipfile.ZipFile(
            temporary_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        str(Path(PACKAGE_NAME) / path.relative_to(package_root)),
                    )
        with zipfile.ZipFile(temporary_zip, "r") as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError(f"ZIP CRC verification failed: {corrupt}")
        temporary_zip.replace(output)

    print(
        json.dumps(
            {
                "zip": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "package_root": PACKAGE_NAME,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
