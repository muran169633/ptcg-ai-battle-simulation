#!/usr/bin/env python3
"""Combine two independent PPO monitors into one live status artifact."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from monitor_ppo_run import tail_jsonl


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def gpu_status() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip().splitlines()[0]
        utilization, used, free, total, power = [
            value.strip() for value in raw.split(",")
        ]
        return {
            "utilization_percent": float(utilization),
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "memory_total_mib": int(total),
            "power_watts": float(power),
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return {"error": "gpu_status_unavailable"}


def route_view(label: str, run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "monitor_status.json"
    if not status_path.is_file():
        return {
            "label": label,
            "run_dir": str(run_dir),
            "state": "monitor_status_missing",
            "warnings": ["monitor_status_missing"],
        }
    status = json.loads(status_path.read_text())
    rows = tail_jsonl(run_dir / "metrics.jsonl", 1)
    latest_metrics = rows[-1] if rows else None
    latest = dict(status.get("latest") or {})
    if latest_metrics:
        rollout = latest_metrics.get("rollout") or {}
        latest["decisions_per_second"] = rollout.get("decisions_per_second")
        latest["hard_failure_attempts"] = rollout.get(
            "hard_failure_attempts", 0
        )
    observed_at = datetime.fromisoformat(status["observed_at"])
    age_seconds = max(
        0.0,
        (datetime.now().astimezone() - observed_at).total_seconds(),
    )
    warnings = list(status.get("warnings") or [])
    if age_seconds > 180:
        warnings.append(f"monitor_stale_seconds={age_seconds:.0f}")
    return {
        "label": label,
        "run_dir": str(run_dir),
        "state": status["state"],
        "monitor_observed_at": status["observed_at"],
        "monitor_age_seconds": age_seconds,
        "process": status.get("process"),
        "progress": status.get("progress"),
        "latest": latest or None,
        "champion": status.get("champion"),
        "warnings": warnings,
    }


def build_pair(routes: list[tuple[str, Path]]) -> dict[str, Any]:
    views = [route_view(label, run_dir) for label, run_dir in routes]
    progresses = [view.get("progress") or {} for view in views]
    completed = sum(int(value.get("completed_updates", 0)) for value in progresses)
    total = sum(int(value.get("total_updates", 0)) for value in progresses)
    sps_values = [
        float(view["latest"]["decisions_per_second"])
        for view in views
        if view.get("latest")
        and view["latest"].get("decisions_per_second") is not None
    ]
    eta_values = [
        value.get("eta_at") for value in progresses if value.get("eta_at")
    ]
    warnings = [
        f"{view['label']}:{warning}"
        for view in views
        for warning in view.get("warnings", [])
    ]
    states = {view["state"] for view in views}
    if states == {"completed"}:
        state = "completed"
    elif "process_missing" in states or "monitor_status_missing" in states:
        state = "attention_required"
    else:
        state = "running"
    return {
        "schema_version": "ptcg-dual-ppo-monitor-v1",
        "observed_at": datetime.now().astimezone().isoformat(),
        "state": state,
        "aggregate": {
            "completed_updates": completed,
            "total_updates": total,
            "percent": 100.0 * completed / total if total else 0.0,
            "latest_combined_decisions_per_second": sum(sps_values),
            "routes_reporting_sps": len(sps_values),
            "estimated_all_routes_complete_at": max(eta_values) if eta_values else None,
            "submission_attempted": False,
        },
        "gpu": gpu_status(),
        "routes": {view["label"]: view for view in views},
        "warnings": warnings,
    }


def append_event(path: Path, status: dict[str, Any]) -> None:
    event = {
        "observed_at": status["observed_at"],
        "state": status["state"],
        "aggregate": status["aggregate"],
        "gpu": status["gpu"],
        "routes": {
            label: {
                "state": view["state"],
                "progress": view.get("progress"),
                "latest": view.get("latest"),
                "champion": view.get("champion"),
                "warnings": view.get("warnings"),
            }
            for label, view in status["routes"].items()
        },
        "warnings": status["warnings"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        nargs=2,
        action="append",
        metavar=("LABEL", "RUN_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.route) != 2:
        raise ValueError("Exactly two --route entries are required")
    if args.interval < 10:
        raise ValueError("--interval must be at least 10 seconds")
    routes = [(label, Path(path).resolve()) for label, path in args.route]
    if len({label for label, _ in routes}) != 2:
        raise ValueError("Route labels must be unique")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "dual_monitor_status.json"
    events_path = output_dir / "dual_monitor_events.jsonl"
    pid_path = output_dir / "dual_monitor.pid"
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    previous_key: tuple[Any, ...] | None = None
    try:
        while True:
            status = build_pair(routes)
            atomic_json(status_path, status)
            key = (
                status["state"],
                status["aggregate"]["completed_updates"],
                tuple(
                    (
                        label,
                        view.get("champion", {}).get("update"),
                        tuple(view.get("warnings", [])),
                    )
                    for label, view in status["routes"].items()
                ),
            )
            if key != previous_key:
                append_event(events_path, status)
                previous_key = key
            if args.once or status["state"] == "completed":
                return 0
            time.sleep(args.interval)
    finally:
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
