from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/audit_yanz_purebc_actor6_qp_trainonly_v1.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("purebc_qp_audit_v1_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_contract_constants() -> None:
    tool = load_tool()
    assert tool.PROFILE_SHA256 == "87891f8985d5f23b8114e5aff11426a4c80c717dc2782eae4cb9cd38d71eb529"
    assert tool.TAU == 1.0e-4
    assert tool.GUARD_FLOOR == 0.0
    assert len(tool.TARGET_IDS) == 3
    assert len(tool.ACTOR6) == 6
    assert len(tool.EXPECTED_CHANGED_ACTOR_TENSORS) == 5
    assert "actor_residual.2.bias" not in tool.EXPECTED_CHANGED_ACTOR_TENSORS
    assert tool.EXPECTED["yanz"]["repairs"] == 3
    assert tool.EXPECTED["old"]["repairs"] == 1


def test_projection_solver_on_toy_halfspaces(monkeypatch) -> None:
    tool = load_tool()
    monkeypatch.setattr(tool, "TAU", 1.0)
    gradients = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    base = np.asarray([0.0, 0.0, 3.0])
    delta, report = tool.solve_projection(
        gradients,
        base,
        ["target", "target", "guard"],
        2,
        enforce_frozen_active_identity=False,
    )
    assert torch.allclose(delta, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert report["active_target"] == 2
    assert report["active_guard"] == 0


def test_tool_has_no_checkpoint_or_external_write_path() -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    assert "save" not in calls
    assert "torch_save" not in calls
    source = TOOL.read_text(encoding="utf-8")
    assert "write_new_json(result, output)" in source
    assert "train/*.jsonl only" in source


def test_event_report_is_deterministic() -> None:
    tool = load_tool()
    events = [
        {"line_sha256": "b", "context": 8},
        {"line_sha256": "a", "context": 0},
        {"line_sha256": "c", "context": 8},
    ]
    first = tool.event_report(events)
    second = tool.event_report(list(reversed(events)))
    assert first["count"] == 3
    assert first["by_context"] == {"0": 1, "8": 2}
    assert first["line_sha256_digest"] == second["line_sha256_digest"]
