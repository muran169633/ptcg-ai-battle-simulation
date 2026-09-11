#!/usr/bin/env python3
"""Continuously materialize a compact health/ETA view for one PPO run."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def now_local() -> datetime:
    return datetime.now().astimezone()


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


def tail_jsonl(path: Path, lines: int = 12) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    block = 1024 * 1024
    with path.open("rb") as handle:
        cursor = handle.seek(0, os.SEEK_END)
        chunks: list[bytes] = []
        newlines = 0
        while cursor > 0 and newlines <= lines:
            size = min(block, cursor)
            cursor -= size
            handle.seek(cursor)
            chunk = handle.read(size)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")
    raw_lines = b"".join(reversed(chunks)).splitlines()[-lines:]
    return [json.loads(value) for value in raw_lines if value.strip()]


def matching_train_process(run_dir: Path) -> tuple[int | None, float | None]:
    needle = str(run_dir.resolve())
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
            if "tools/train_ppo.py" not in command or needle not in command:
                continue
            stat = (entry / "stat").read_text().split()
            start_ticks = int(stat[21])
            clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            uptime = float(Path("/proc/uptime").read_text().split()[0])
            elapsed = max(0.0, uptime - start_ticks / clock_ticks)
            return int(entry.name), elapsed
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
    return None, None


def gpu_memory_mib(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                return int(fields[1])
            except ValueError:
                return None
    return None


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def gate_seconds(row: dict[str, Any]) -> float:
    total = 0.0
    for result in (row.get("evaluation_by_opponent") or {}).values():
        total += float(result.get("seconds", 0.0))
    total += float((row.get("champion_gate") or {}).get("seconds", 0.0))
    return total


def build_status(run_dir: Path, missing_checks: int) -> dict[str, Any]:
    config_payload = json.loads((run_dir / "run_config.json").read_text())
    config = config_payload["config"]
    rows = tail_jsonl(run_dir / "metrics.jsonl")
    latest = rows[-1] if rows else None
    completed = int(latest["update"]) if latest else 0
    total = int(config["updates"])
    pid, process_elapsed = matching_train_process(run_dir)
    summary_exists = (run_dir / "summary.json").is_file()
    if summary_exists and completed >= total:
        state = "completed"
    elif pid is not None:
        state = "running"
    elif missing_checks >= 2:
        state = "process_missing"
    else:
        state = "process_check_pending"

    duration_samples = []
    observed_gate_seconds = []
    champion_update = 0
    latest_gate = None
    for row in rows:
        rollout = row.get("rollout") or {}
        optimization = row.get("optimization") or {}
        replay = row.get("bc_replay") or {}
        duration_samples.append(
            float(rollout.get("seconds", 0.0))
            + float(optimization.get("seconds", 0.0))
            + float(replay.get("seconds", 0.0))
        )
        gate = row.get("champion_gate")
        if gate:
            latest_gate = gate
            champion_update = int(gate.get("champion_update_after", champion_update))
            observed_gate_seconds.append(gate_seconds(row))
    median_update_seconds = (
        statistics.median(duration_samples) if duration_samples else None
    )
    gate_interval = int(config.get("champion_gate_interval", 0))
    future_gate_updates = (
        [
            update
            for update in range(gate_interval, total + 1, gate_interval)
            if update > completed
        ]
        if gate_interval > 0
        else []
    )
    if total not in future_gate_updates and total > completed and gate_interval > 0:
        future_gate_updates.append(total)
    estimated_gate_seconds = (
        statistics.median(observed_gate_seconds)
        if observed_gate_seconds
        else (
            (
                (len(config.get("extra_opponents") or []) + 1)
                * int(config.get("eval_games", 0))
                + int(config.get("champion_gate_games", 0))
            )
            * 0.32
        )
    )
    eta_seconds = None
    eta_at = None
    if median_update_seconds is not None and completed < total:
        eta_seconds = (
            (total - completed) * median_update_seconds
            + len(set(future_gate_updates)) * estimated_gate_seconds
        )
        eta_at = (now_local() + timedelta(seconds=eta_seconds)).isoformat()

    warnings: list[str] = []
    latest_view = None
    if latest:
        rollout = latest["rollout"]
        optimization = latest["optimization"]
        valid_games = int(rollout["valid_games"])
        league_games = int(rollout["league_games"])
        failures = int(rollout.get("hard_failure_attempts", 0))
        approx_kl = float(optimization["approx_kl"])
        anchor_kl = float(optimization["bc_anchor_kl"])
        clip_fraction = float(optimization["clip_fraction"])
        if failures:
            warnings.append(f"hard_failure_attempts={failures}")
        if not finite_tree(latest):
            warnings.append("non_finite_metric")
        if approx_kl > max(float(config.get("target_kl", 0.0)) * 2.0, 0.001):
            warnings.append(f"approx_kl_high={approx_kl:.6g}")
        if anchor_kl > 0.02:
            warnings.append(f"bc_anchor_kl_high={anchor_kl:.6g}")
        if clip_fraction > 0.25:
            warnings.append(f"clip_fraction_high={clip_fraction:.6g}")
        metrics_age = time.time() - (run_dir / "metrics.jsonl").stat().st_mtime
        if state == "running" and metrics_age > 30 * 60:
            warnings.append(f"no_completed_update_for_seconds={metrics_age:.0f}")
        latest_view = {
            "update": completed,
            "valid_games": valid_games,
            "league_games": league_games,
            "selfplay_games": valid_games - league_games,
            "league_fraction": league_games / valid_games,
            "training_league_win_rate": (
                float(rollout["league_current_wins"]) / league_games
                if league_games
                else None
            ),
            "transitions": int(rollout["transitions_kept"]),
            "rollout_seconds": float(rollout["seconds"]),
            "optimization_seconds": float(optimization["seconds"]),
            "bc_replay_seconds": float((latest.get("bc_replay") or {}).get("seconds", 0.0)),
            "approx_kl": approx_kl,
            "bc_anchor_kl": anchor_kl,
            "clip_fraction": clip_fraction,
            "selection_score": latest.get("selection_score"),
        }
    if state == "process_missing":
        warnings.append("training_process_missing")

    return {
        "schema_version": "ptcg-ppo-monitor-v1",
        "observed_at": now_local().isoformat(),
        "run_dir": str(run_dir),
        "state": state,
        "process": {
            "pid": pid,
            "elapsed_seconds": process_elapsed,
            "gpu_memory_mib": gpu_memory_mib(pid),
        },
        "progress": {
            "completed_updates": completed,
            "total_updates": total,
            "percent": 100.0 * completed / total,
            "median_training_update_seconds": median_update_seconds,
            "future_gate_count": len(set(future_gate_updates)),
            "estimated_gate_seconds": estimated_gate_seconds,
            "eta_seconds": eta_seconds,
            "eta_at": eta_at,
        },
        "latest": latest_view,
        "champion": {
            "update": champion_update,
            "latest_gate": latest_gate,
            "strict_minimum_win_rate": float(
                config.get("champion_gate_min_win_rate", 0.58)
            ),
        },
        "warnings": warnings,
        "submission_attempted": False,
    }


def append_event(path: Path, status: dict[str, Any]) -> None:
    event = {
        "observed_at": status["observed_at"],
        "state": status["state"],
        "progress": status["progress"],
        "latest": status["latest"],
        "champion": status["champion"],
        "warnings": status["warnings"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.interval < 10:
        raise ValueError("--interval must be at least 10 seconds")
    if not (run_dir / "run_config.json").is_file():
        raise FileNotFoundError(run_dir / "run_config.json")
    status_path = run_dir / "monitor_status.json"
    events_path = run_dir / "monitor_events.jsonl"
    pid_path = run_dir / "monitor.pid"
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    previous_key: tuple[Any, ...] | None = None
    missing_checks = 0
    try:
        while True:
            status = build_status(run_dir, missing_checks)
            if status["process"]["pid"] is None:
                missing_checks += 1
                status = build_status(run_dir, missing_checks)
            else:
                missing_checks = 0
            atomic_json(status_path, status)
            key = (
                status["state"],
                status["progress"]["completed_updates"],
                status["champion"]["update"],
                tuple(status["warnings"]),
            )
            if key != previous_key:
                append_event(events_path, status)
                previous_key = key
            if args.once or status["state"] in {"completed", "process_missing"}:
                return 0 if status["state"] != "process_missing" else 2
            time.sleep(args.interval)
    finally:
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
