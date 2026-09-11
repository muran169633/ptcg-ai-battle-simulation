#!/usr/bin/env python3
"""Build and verify the self-contained Linux x86_64 8-GPU PPO ZIP."""

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
PACKAGE_NAME = "ptcg_nonar_v7_8gpu_ppo_20260816"
BC_REL = Path(
    "artifacts/bc_top100_recent14_nonar_order_v7_end0813_b2048_20260816_v1"
)
META_REL = Path("data/recent_day_meta_pool_20260813_top23_v1")


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / f"{PACKAGE_NAME}.zip",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = [
        Path("tools/train_ppo.py"),
        Path("tools/train_bc_orbit.py"),
        Path("tools/bc_nonar_v7.py"),
        Path("tools/train_nonar_league_ppo.py"),
        Path("tools/parallel_rollout.py"),
        Path("tools/train_nonar_league_ppo_parallel.py"),
        Path("tools/train_nonar_league_ppo_ddp.py"),
        Path("tools/launch_nonar_dragapult_ppo_fast.py"),
        Path("tools/run_8gpu_ppo.py"),
        Path("tools/selfcheck_portable_8gpu.py"),
        Path("dataset/sample_submission/sample_submission/cg/__init__.py"),
        Path("dataset/sample_submission/sample_submission/cg/sim.py"),
        Path("dataset/sample_submission/sample_submission/cg/libcg.so"),
        BC_REL / "best.pt",
        BC_REL / "last.pt",
        Path("PORTABLE_8GPU_README.md"),
        Path("portable_8gpu_requirements.txt"),
        Path("setup_and_run_8gpu.sh"),
    ]
    files.extend(sorted((ROOT / META_REL / "decks").glob("*.csv")))

    with tempfile.TemporaryDirectory(prefix="ptcg-8gpu-package-") as temporary:
        package_root = Path(temporary) / PACKAGE_NAME
        for item in files:
            source = item if item.is_absolute() else ROOT / item
            relative = source.relative_to(ROOT)
            copy_file(
                source,
                package_root / relative,
                executable=(
                    relative.suffix == ".sh"
                    or relative.name in {
                        "run_8gpu_ppo.py",
                        "selfcheck_portable_8gpu.py",
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
            portable_row["deck_path"] = "decks/" + Path(
                str(row["deck_path"])
            ).name
            portable_rows.append(portable_row)
        portable_meta["opponents"] = portable_rows
        meta_destination = package_root / META_REL / "meta_pool.json"
        meta_destination.parent.mkdir(parents=True, exist_ok=True)
        meta_destination.write_text(
            json.dumps(portable_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest_rows = []
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            manifest_rows.append(
                {
                    "path": str(path.relative_to(package_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        manifest = {
            "schema_version": "ptcg-nonar-v7-portable-8gpu-v5-a100-40gb",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_name": PACKAGE_NAME,
            "platform": "Linux x86_64",
            "bc_feature_version": "ptcg-bc-orbit-nonar-v7-ordered-rank",
            "bc_sha256": sha256(package_root / BC_REL / "best.pt"),
            "default_world_size": 8,
            "target_gpu": "NVIDIA A100 40GB",
            "default_updates": 8_000,
            "default_global_games": 16_384_000,
            "default_minibatch_size_per_rank": 4_096,
            "default_effective_global_minibatch_size": 32_768,
            "default_checkpoint_interval": 50,
            "checkpoint_writer": "rank0_only",
            "default_rollout_workers_per_rank": 4,
            "default_total_rollout_workers": 32,
            "default_rollout_batch_wait_ms": 10.0,
            "local_single_gpu_rollout_benchmark_sps": 1763.4019949957901,
            "files": manifest_rows,
        }
        (package_root / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
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
