#!/usr/bin/env python3
"""Rebound v3 Marnie CVaR-tail repair launcher after an input drift.

Revision r2 preserves the complete v3 training protocol.  It changes only the
locked panel-audit digest and the never-reused smoke/full output directories.
The completed original smoke is explicitly invalidated and can never be used
for selection, continuation, packaging, or submission.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
V3_LAUNCHER = ROOT / "tools/run_gold_push_marnie_ppo_v3.py"
V3_LAUNCHER_SHA256 = "bea7737e5c58bf8fca8e1054c1bbe181b818776211be90dcd5475fb0c14d55c0"

if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import run_gold_push_marnie_ppo_v3 as v3  # noqa: E402


if v3.raw_sha256_file(V3_LAUNCHER) != V3_LAUNCHER_SHA256:
    raise RuntimeError("Authenticated v3 launcher SHA-256 mismatch")

CURRENT_PANEL_AUDIT_SHA256 = (
    "84d3c1bdb315977d2315d52feea34854814ce0b297198ecbaeabfc80a8804bfa"
)
INVALIDATED_SMOKE = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "ppo_marnie_tail32_v3_cvar_tailrepair_smoke/INVALIDATED_RUN.json"
)
INVALIDATED_SMOKE_SHA256 = (
    "4938e4629d8592d9437cdb0257cd392536e9ec443345657118b3bb35c40dacfb"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "ppo_marnie_tail32_v3_cvar_tailrepair_r2"
)
SMOKE_OUTPUT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "ppo_marnie_tail32_v3_cvar_tailrepair_smoke_r2"
)

Phase = v3.Phase
PHASES = v3.PHASES
Preflight = v3.Preflight


def output_dir(phase: Phase) -> Path:
    return SMOKE_OUTPUT if phase.name == "smoke" else OUTPUT_ROOT


def build_command(phase: Phase) -> list[str]:
    command = v3.build_command(phase)
    v3.set_single_value(command, "--output-dir", str(output_dir(phase)))
    v3.base.assert_cli_contract(command)
    return command


def assert_only_output_changed(phase: Phase, command: Sequence[str]) -> dict[str, Any]:
    original = v3.build_command(phase)
    expected = list(original)
    v3.set_single_value(expected, "--output-dir", str(output_dir(phase)))
    if list(command) != expected:
        raise RuntimeError("r2 command drifted beyond --output-dir")
    differences = [
        {
            "index": index,
            "v3": before,
            "v3_r2": after,
        }
        for index, (before, after) in enumerate(zip(original, command, strict=True))
        if before != after
    ]
    if differences != [
        {
            "index": original.index("--output-dir") + 1,
            "v3": str(v3.output_dir(phase)),
            "v3_r2": str(output_dir(phase)),
        }
    ]:
        raise RuntimeError(f"Unexpected r2 command differences: {differences}")
    return {
        "training_protocol_unchanged": True,
        "only_command_difference": "--output-dir value",
        "differences": differences,
        "v3_command_sha256": v3.base.sha256_json(original),
        "v3_r2_command_sha256": v3.base.sha256_json(command),
    }


def build_v3_template(phase: Phase) -> Preflight:
    old_output = v3.OUTPUT_ROOT
    old_smoke = v3.SMOKE_OUTPUT
    old_audit_sha = v3.EXTRA_SHA256["panel_audit"]
    try:
        v3.OUTPUT_ROOT = OUTPUT_ROOT
        v3.SMOKE_OUTPUT = SMOKE_OUTPUT
        v3.EXTRA_SHA256["panel_audit"] = CURRENT_PANEL_AUDIT_SHA256
        return v3.build_preflight(phase)
    finally:
        v3.OUTPUT_ROOT = old_output
        v3.SMOKE_OUTPUT = old_smoke
        v3.EXTRA_SHA256["panel_audit"] = old_audit_sha


def build_preflight(phase: Phase) -> Preflight:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Launcher must run from repository root: {ROOT}")
    target = output_dir(phase)
    v3.base.assert_target_absent(target)

    template = build_v3_template(phase)
    manifest = copy.deepcopy(template.manifest)
    command = build_command(phase)
    protocol_diff = assert_only_output_changed(phase, command)

    v3_launcher_record = manifest["inputs"].pop("launcher")
    if v3_launcher_record["sha256"] != V3_LAUNCHER_SHA256:
        raise RuntimeError("v3 template launcher identity drifted")
    manifest["inputs"]["v3_launcher"] = v3_launcher_record
    manifest["inputs"]["invalidated_v3_smoke"] = v3.base.validate_file_binding(
        "invalidated_v3_smoke",
        INVALIDATED_SMOKE,
        INVALIDATED_SMOKE_SHA256,
    )
    manifest["inputs"]["launcher"] = {
        "path": str(SELF),
        "resolved_path": str(SELF.resolve()),
        "bytes": SELF.stat().st_size,
        "sha256": v3.raw_sha256_file(SELF),
        "protocol_pins_live_launcher_sha256": True,
    }
    if (
        manifest["inputs"]["panel_audit"]["sha256"]
        != CURRENT_PANEL_AUDIT_SHA256
    ):
        raise RuntimeError("r2 manifest did not bind the current panel audit")

    invalidation = json.loads(INVALIDATED_SMOKE.read_text(encoding="utf-8"))
    if (
        invalidation.get("status") != "invalidated_after_child_completion"
        or invalidation.get("diagnostic_only", {}).get(
            "may_select_or_promote_checkpoint"
        )
        is not False
        or invalidation.get("changed_input", {}).get(
            "observed_after_failure_sha256"
        )
        != CURRENT_PANEL_AUDIT_SHA256
    ):
        raise RuntimeError("Original v3 smoke invalidation record drifted")

    manifest["schema_version"] = (
        "ptcg-gold-push-marnie-v3-cvar-tailrepair-launch-r2"
    )
    manifest["output_dir"] = str(target)
    manifest["expected_terminal_checkpoint"] = str(
        target / f"checkpoints/update-{phase.updates:04d}.pt"
    )
    manifest["revision"] = {
        "revision": "r2",
        "reason": "the original smoke failed closed after the panel audit changed while its child was running",
        "current_panel_audit_sha256": CURRENT_PANEL_AUDIT_SHA256,
        "invalidated_predecessor": {
            "directory": str(INVALIDATED_SMOKE.parent),
            "record": str(INVALIDATED_SMOKE),
            "record_sha256": INVALIDATED_SMOKE_SHA256,
            "may_select_or_promote_checkpoint": False,
            "may_seed_later_training": False,
            "may_package_or_submit": False,
        },
        "fresh_from_same_frozen_bc": True,
        "protocol_diff": protocol_diff,
    }
    manifest["gates"]["current_panel_audit_sha256_exact"] = True
    manifest["gates"]["invalidated_predecessor_excluded"] = True
    manifest["command"] = command
    manifest["command_sha256"] = v3.base.sha256_json(command)

    return Preflight(
        phase=phase,
        target=target,
        command=command,
        manifest=manifest,
        manifest_sha256=v3.base.sha256_json(manifest),
    )


def execute(preflight: Preflight) -> int:
    locks = v3.acquire_input_locks(preflight)
    try:
        v3.assert_input_locks_unchanged(locks)
        v3.base.assert_target_absent(preflight.target)
        preflight.target.mkdir(parents=False, exist_ok=False, mode=0o700)
        v3.base.write_exclusive(
            preflight.target / "launcher_manifest.json",
            {
                "manifest_sha256": preflight.manifest_sha256,
                "manifest": preflight.manifest,
            },
        )
        v3.assert_input_locks_unchanged(locks)
        environment = os.environ.copy()
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            preflight.command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            shell=False,
            check=False,
        )
        v3.assert_input_locks_unchanged(locks)
        v3.base.write_exclusive(
            preflight.target / "launcher_result.json",
            {
                "schema_version": (
                    "ptcg-gold-push-marnie-v3-cvar-tailrepair-result-r2"
                ),
                "manifest_sha256": preflight.manifest_sha256,
                "return_code": int(completed.returncode),
                "inputs_unchanged_after_child": True,
                "package_upload_submission_performed": False,
            },
        )
        return int(completed.returncode)
    finally:
        v3.release_input_locks(locks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASES), default="smoke")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute and not args.expected_manifest_sha256:
        raise ValueError("--execute requires --expected-manifest-sha256")
    if not args.execute and args.expected_manifest_sha256:
        raise ValueError("--expected-manifest-sha256 is only valid with --execute")
    preflight = build_preflight(PHASES[args.phase])
    print(
        json.dumps(
            {
                "manifest_sha256": preflight.manifest_sha256,
                "manifest": preflight.manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    print(shlex.join(preflight.command), flush=True)
    if not args.execute:
        return 0
    if args.expected_manifest_sha256 != preflight.manifest_sha256:
        raise RuntimeError("Manifest SHA-256 mismatch; repeat and review dry-run")
    return execute(preflight)


if __name__ == "__main__":
    raise SystemExit(main())
