#!/usr/bin/env python3
"""Hash-locked isolated-Python adapter for ``run_gold_league_h2h.py``.

The legacy runner is intentionally left unchanged.  This adapter authenticates
it before import, requires this process to have been started with ``-I -B``,
and temporarily wraps ``build_eval_command`` so every evaluator child is also
started with ``-I -B``.  Python import-affecting environment variables are
removed while the legacy runner and all of its evaluator children execute,
then restored before this process exits.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RUNNER = REPO_ROOT / "tools/run_gold_league_h2h.py"
LEGACY_RUNNER_SHA256 = (
    "47764f93ac05bb28f638b41ea50d69d0350ada372375d58eaeff8b8a014ec61b"
)
LEGACY_MODULE_NAME = "_ptcg_hash_locked_legacy_gold_runner"
PYTHON_ENV_KEYS = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONINSPECT",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_isolated_python() -> None:
    if int(sys.flags.isolated) != 1:
        raise RuntimeError("isolated Gold adapter requires Python -I")
    if int(sys.flags.dont_write_bytecode) != 1:
        raise RuntimeError("isolated Gold adapter requires Python -B")


def require_legacy_runner_binding() -> None:
    if not LEGACY_RUNNER.is_file():
        raise FileNotFoundError(LEGACY_RUNNER)
    observed = sha256_file(LEGACY_RUNNER)
    if observed != LEGACY_RUNNER_SHA256:
        raise RuntimeError(
            "legacy Gold runner SHA-256 mismatch: expected "
            f"{LEGACY_RUNNER_SHA256}, observed {observed}"
        )


@contextmanager
def scrubbed_python_environment() -> Iterator[dict[str, str]]:
    affected = {
        *PYTHON_ENV_KEYS,
        *(key for key in os.environ if key.upper().startswith("PYTHON")),
    }
    saved = {key: os.environ[key] for key in affected if key in os.environ}
    for key in affected:
        os.environ.pop(key, None)
    try:
        remaining = {
            key: value
            for key, value in os.environ.items()
            if key.upper().startswith("PYTHON")
        }
        if remaining:
            raise RuntimeError(f"Python environment scrub failed: {remaining}")
        yield saved
    finally:
        for key in tuple(os.environ):
            if key.upper().startswith("PYTHON"):
                os.environ.pop(key, None)
        for key in PYTHON_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(saved)


def load_legacy_runner() -> ModuleType:
    require_legacy_runner_binding()
    spec = importlib.util.spec_from_file_location(
        LEGACY_MODULE_NAME,
        LEGACY_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct legacy Gold runner import spec")
    module = importlib.util.module_from_spec(spec)
    prior = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if prior is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = prior
        raise
    if Path(module.__file__).resolve() != LEGACY_RUNNER.resolve():
        raise RuntimeError("wrong legacy Gold runner module imported")
    require_legacy_runner_binding()
    return module


def isolated_eval_command(
    original: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[str, ...]:
    raw = original(*args, **kwargs)
    if not isinstance(raw, tuple) or len(raw) < 2:
        raise RuntimeError("legacy build_eval_command returned an invalid command")
    command = list(raw)
    if Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise RuntimeError("legacy evaluator command uses the wrong Python")
    if command[1:3] == ["-I", "-B"]:
        raise RuntimeError("legacy evaluator command unexpectedly already contains -I -B")
    forbidden = {"-I", "-B"}.intersection(command[1:])
    if forbidden:
        raise RuntimeError(f"legacy evaluator command has misplaced flags: {forbidden}")
    command[1:1] = ["-I", "-B"]
    if command[:4] != [
        sys.executable,
        "-I",
        "-B",
        str((REPO_ROOT / "tools/evaluate_ppo_head_to_head.py").resolve()),
    ]:
        raise RuntimeError("isolated evaluator command prefix construction failed")
    return tuple(command)


@contextmanager
def patched_legacy_runner() -> Iterator[ModuleType]:
    require_isolated_python()
    require_legacy_runner_binding()
    with scrubbed_python_environment():
        module = load_legacy_runner()
        original = module.build_eval_command

        def wrapped(*args: Any, **kwargs: Any) -> tuple[str, ...]:
            return isolated_eval_command(original, *args, **kwargs)

        module.build_eval_command = wrapped
        try:
            yield module
        finally:
            module.build_eval_command = original
            require_legacy_runner_binding()
            if sys.modules.get(LEGACY_MODULE_NAME) is module:
                sys.modules.pop(LEGACY_MODULE_NAME, None)


def main() -> None:
    with patched_legacy_runner() as legacy:
        legacy.main()


if __name__ == "__main__":
    main()
