#!/usr/bin/env python3
"""Direct-exec one frozen GPU behavior evaluation.

This launcher is intentionally narrow.  It validates a byte-hash-bound
behavior preregistration, checks the immutable evaluator/checkpoint/data
inputs, claims a unique log with O_EXCL, and replaces itself with the exact
``evaluate_policy_bc.py`` command.  It never creates a subprocess.

The caller is responsible for launching this file directly through an
approved non-sandbox ``functions.exec_command`` transport and for recording
the terminal exit code in the separately preregistered receipt path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_TRANSPORT = {
    "tool": "functions.exec_command",
    "sandbox_permissions": "require_escalated",
    "login": False,
    "tty": False,
    "nested_functions_exec_forbidden": True,
}
FORBIDDEN_ARGUMENT_PREFIXES = (
    "--kaggle",
    "--network",
    "--package",
    "--publish",
    "--submit",
    "--upload",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path


def _path_under_root(value: Any, root: Path, label: str) -> Path:
    path = _absolute_path(value, label)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain under the frozen root") from error
    if path != resolved:
        raise ValueError(f"{label} must be normalized and symlink-free")
    return path


def _single_command_value(command: Sequence[str], flag: str) -> str:
    indices = [index for index, token in enumerate(command) if token == flag]
    if len(indices) != 1:
        raise ValueError(f"command_abs must contain {flag} exactly once")
    index = indices[0]
    if index + 1 >= len(command):
        raise ValueError(f"command_abs has no value after {flag}")
    return command[index + 1]


def read_and_validate_panel(
    *,
    root: Path,
    preregistration: Path,
    expected_preregistration_sha256: str,
    panel_index: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Frozen root must be a directory")
    if Path.cwd().resolve() != root:
        raise ValueError("Current working directory differs from frozen root")
    expected_sha256 = _require_sha256(
        expected_preregistration_sha256,
        "expected preregistration hash",
    )
    preregistration = _path_under_root(
        str(preregistration),
        root,
        "preregistration",
    )
    if preregistration.is_symlink() or not preregistration.is_file():
        raise ValueError("preregistration must be a regular symlink-free file")
    raw = preregistration.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Behavior preregistration SHA-256 mismatch")
    try:
        protocol = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Behavior preregistration is not valid JSON") from error
    if not isinstance(protocol, dict):
        raise ValueError("Behavior preregistration must contain an object")

    transport = protocol.get("execution_transport")
    if not isinstance(transport, dict):
        raise ValueError("Behavior preregistration has no execution_transport")
    for key, expected in REQUIRED_TRANSPORT.items():
        if transport.get(key) != expected:
            raise ValueError(
                f"execution_transport.{key} must be {expected!r}"
            )
    if protocol.get("required_cwd") != str(root):
        raise ValueError("required_cwd differs from the frozen root")
    launcher = protocol.get("launcher")
    if not isinstance(launcher, dict):
        raise ValueError("Behavior preregistration has no launcher binding")
    launcher_path = _path_under_root(
        launcher.get("path"),
        root,
        "launcher.path",
    )
    expected_launcher = root / "tools" / "exec_preregistered_behavior.py"
    if launcher_path != expected_launcher or Path(__file__).resolve() != expected_launcher:
        raise ValueError("Behavior launcher path mismatch")
    launcher_sha256 = _require_sha256(
        launcher.get("sha256"),
        "launcher.sha256",
    )
    if file_sha256(launcher_path) != launcher_sha256:
        raise ValueError("Behavior launcher SHA-256 mismatch")

    panels = protocol.get("ordered_evaluations")
    if not isinstance(panels, list) or not panels:
        raise ValueError("ordered_evaluations must be a non-empty list")
    if panel_index < 0 or panel_index >= len(panels):
        raise ValueError("panel_index is outside ordered_evaluations")
    panel = panels[panel_index]
    if not isinstance(panel, dict):
        raise ValueError("Selected behavior panel must be an object")
    if panel.get("order") != panel_index + 1:
        raise ValueError("Selected behavior panel order is not canonical")
    if panel.get("attempts_authorized") != 1:
        raise ValueError("Selected behavior panel must authorize one attempt")
    if panel.get("expected_success_exit_code") != 0:
        raise ValueError("Selected behavior panel must require exit code zero")
    for prior_index, prior in enumerate(panels[:panel_index]):
        if not isinstance(prior, dict):
            raise ValueError("Prior behavior panel must be an object")
        prior_log = _path_under_root(
            prior.get("stdout_stderr_log"),
            root,
            f"ordered_evaluations[{prior_index}].stdout_stderr_log",
        )
        prior_receipt = _path_under_root(
            prior.get("execution_result"),
            root,
            f"ordered_evaluations[{prior_index}].execution_result",
        )
        if not prior_log.is_file() or prior_log.stat().st_size <= 0:
            raise ValueError("A prior panel has no consumed-attempt log")
        if not prior_receipt.is_file():
            raise ValueError("A prior panel has no terminal execution receipt")
        try:
            prior_result = json.loads(prior_receipt.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("A prior panel receipt is not valid JSON") from error
        if not isinstance(prior_result, dict):
            raise ValueError("A prior panel receipt must contain an object")
        if prior_result.get("attempt_consumed") is not True:
            raise ValueError("A prior panel receipt does not consume its attempt")
        if prior_result.get("terminal") is not True:
            raise ValueError("A prior panel receipt is not terminal")
        if not isinstance(prior_result.get("terminal_exit_code"), int):
            raise ValueError("A prior panel receipt has no terminal exit code")
        if prior_result.get("status") not in {
            "completed",
            "failed",
            "infrastructure_failed",
            "timed_out",
        }:
            raise ValueError("A prior panel receipt has an invalid terminal status")
        prior_transport = prior_result.get("transport")
        if not isinstance(prior_transport, dict):
            raise ValueError("A prior panel receipt has no transport evidence")
        for key, expected in REQUIRED_TRANSPORT.items():
            if prior_transport.get(key) != expected:
                raise ValueError(
                    f"A prior panel receipt transport.{key} mismatch"
                )
        if prior_result.get("protocol_sha256") != expected_sha256:
            raise ValueError("A prior panel receipt has a protocol hash mismatch")
        if prior_result.get("command_sha256") != prior.get("command_sha256"):
            raise ValueError("A prior panel receipt has a command hash mismatch")
        if prior_result.get("panel") != prior.get("name"):
            raise ValueError("A prior panel receipt has a panel-name mismatch")

    command = panel.get("command_abs")
    if (
        not isinstance(command, list)
        or len(command) < 4
        or not all(isinstance(token, str) and token for token in command)
    ):
        raise ValueError("command_abs must be a non-empty string list")
    expected_command_sha256 = _require_sha256(
        panel.get("command_sha256"),
        "command_sha256",
    )
    if canonical_json_sha256(command) != expected_command_sha256:
        raise ValueError("command_abs SHA-256 mismatch")
    if not Path(command[0]).is_absolute():
        raise ValueError("Evaluator Python executable must be absolute")
    if Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("Evaluator Python differs from the launcher environment")
    if command[1] != "-u":
        raise ValueError("Evaluator command must enable unbuffered Python with -u")
    expected_evaluator = root / "tools" / "evaluate_policy_bc.py"
    if Path(command[2]) != expected_evaluator:
        raise ValueError("Evaluator command is not the frozen local evaluator")
    for token in command[3:]:
        if token.casefold().startswith(FORBIDDEN_ARGUMENT_PREFIXES):
            raise ValueError(f"Forbidden evaluator argument: {token!r}")

    output = _path_under_root(panel.get("output"), root, "output")
    log = _path_under_root(
        panel.get("stdout_stderr_log"),
        root,
        "stdout_stderr_log",
    )
    receipt = _path_under_root(
        panel.get("execution_result"),
        root,
        "execution_result",
    )
    if len({output, log, receipt}) != 3:
        raise ValueError("output, log, and execution_result must be distinct")
    if Path(_single_command_value(command, "--json-output")) != output:
        raise ValueError("--json-output differs from the frozen output")
    if _single_command_value(command, "--device") != "cuda":
        raise ValueError("Behavior evaluator must use --device cuda")
    if _single_command_value(command, "--prediction-order") != "policy":
        raise ValueError("Behavior evaluator must use policy prediction order")

    immutable_inputs = panel.get("immutable_inputs")
    if not isinstance(immutable_inputs, list):
        raise ValueError("immutable_inputs must be a list")
    input_hashes: dict[Path, str] = {}
    for index, item in enumerate(immutable_inputs):
        if not isinstance(item, dict):
            raise ValueError(f"immutable_inputs[{index}] must be an object")
        path = _path_under_root(
            item.get("path"),
            root,
            f"immutable_inputs[{index}].path",
        )
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Immutable input is not a regular file: {path}")
        digest = _require_sha256(
            item.get("sha256"),
            f"immutable_inputs[{index}].sha256",
        )
        if file_sha256(path) != digest:
            raise ValueError(f"Immutable input SHA-256 mismatch: {path}")
        if path in input_hashes:
            raise ValueError(f"Duplicate immutable input: {path}")
        input_hashes[path] = digest
    required_inputs = {
        expected_evaluator,
        Path(_single_command_value(command, "--checkpoint")),
        Path(_single_command_value(command, "--data")),
    }
    if set(input_hashes) != required_inputs:
        raise ValueError(
            "immutable_inputs must exactly bind evaluator, checkpoint, and data"
        )

    for path, label in (
        (output, "behavior output"),
        (log, "stdout/stderr log"),
        (receipt, "execution receipt"),
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to reuse {label}: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"{label} parent does not exist: {path.parent}")
    return protocol, panel, command


def cuda_tensor_preflight() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the CUDA preflight") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA preflight failed: no visible CUDA device")
    probe = torch.empty(1, device="cuda")
    probe.fill_(1.0)
    value = float(probe.item())
    if value != 1.0:
        raise RuntimeError("CUDA preflight tensor produced an unexpected value")
    return {
        "cuda_available": True,
        "device_count": int(torch.cuda.device_count()),
        "device_index": int(torch.cuda.current_device()),
        "device_name": str(torch.cuda.get_device_name(torch.cuda.current_device())),
    }


def claim_log(
    *,
    root: Path,
    preregistration_sha256: str,
    panel: dict[str, Any],
) -> int:
    log = Path(panel["stdout_stderr_log"])
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    header = {
        "command_sha256": panel["command_sha256"],
        "cwd": str(root),
        "event": "behavior_directexec_attempt_claimed",
        "panel": panel["name"],
        "protocol_sha256": preregistration_sha256,
    }
    os.write(
        fd,
        (
            json.dumps(
                header,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8"),
    )
    os.fsync(fd)
    return fd


def write_log_event(fd: int, event: dict[str, Any]) -> None:
    os.write(
        fd,
        (
            json.dumps(
                event,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8"),
    )
    os.fsync(fd)


def revalidate_immutable_inputs(panel: dict[str, Any]) -> None:
    for item in panel["immutable_inputs"]:
        path = Path(item["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Immutable input changed before exec: {path}")
        if file_sha256(path) != item["sha256"]:
            raise RuntimeError(
                f"Immutable input hash changed before exec: {path}"
            )


def exec_claimed_command(
    *,
    fd: int,
    root: Path,
    preregistration_sha256: str,
    panel: dict[str, Any],
    command: list[str],
    cuda_preflight: dict[str, Any],
) -> None:
    write_log_event(
        fd,
        {
            "command_sha256": panel["command_sha256"],
            "cuda_preflight": cuda_preflight,
            "cwd": str(root),
            "event": "behavior_directexec_pre_exec_checks_passed",
            "panel": panel["name"],
            "protocol_sha256": preregistration_sha256,
        },
    )
    null_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null_fd, 0, inheritable=True)
    os.dup2(fd, 1, inheritable=True)
    os.dup2(fd, 2, inheritable=True)
    os.close(null_fd)
    os.close(fd)
    os.execv(command[0], command)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-exec one hash-bound GPU behavior evaluation."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--expected-preregistration-sha256",
        required=True,
    )
    parser.add_argument("--panel-index", type=int, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not args.root.is_absolute():
        raise ValueError("--root must be absolute")
    root = args.root.resolve(strict=True)
    if args.root != root:
        raise ValueError("--root must be normalized and symlink-free")
    _, panel, command = read_and_validate_panel(
        root=root,
        preregistration=args.preregistration,
        expected_preregistration_sha256=(
            args.expected_preregistration_sha256
        ),
        panel_index=args.panel_index,
    )
    fd = claim_log(
        root=root,
        preregistration_sha256=args.expected_preregistration_sha256,
        panel=panel,
    )
    try:
        preflight = cuda_tensor_preflight()
        if file_sha256(args.preregistration) != (
            args.expected_preregistration_sha256
        ):
            raise RuntimeError(
                "Behavior preregistration changed after the attempt claim"
            )
        revalidate_immutable_inputs(panel)
    except BaseException as error:
        write_log_event(
            fd,
            {
                "error": f"{type(error).__name__}: {error}",
                "event": "behavior_directexec_pre_exec_checks_failed",
            },
        )
        os.close(fd)
        raise
    exec_claimed_command(
        fd=fd,
        root=root,
        preregistration_sha256=args.expected_preregistration_sha256,
        panel=panel,
        command=command,
        cuda_preflight=preflight,
    )


if __name__ == "__main__":
    main()
