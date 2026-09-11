#!/usr/bin/env python3
"""Bridge between web GPT design discussion and Codex implementation tasks.

This script helps turn a reasoning packet into a structured execution ticket
that can be handed directly to Codex for implementation and validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any
import uuid


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTES_DIR = REPO_ROOT / "notes"
SCHEMA_VERSION = "ptcg-web-gpt-codex-handoff-v1"

REQUIRED_KEYS: dict[str, type] = {
    "bridge_schema": str,
    "id": str,
    "created_at": str,
    "owner": str,
    "title": str,
    "goal": str,
    "problem_statement": str,
    "hypothesis": list,
    "experiment_plan": str,
    "data_scope": str,
    "validation": list,
    "acceptance_criteria": list,
    "risk_and_mitigation": list,
    "rollback_plan": str,
    "notes_for_codex": str,
}


def _slugify(value: str, fallback: str = "web-gpt-handoff") -> str:
    safe = []
    for ch in value.lower():
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch.isspace():
            safe.append("-")
    if not safe:
        safe = [fallback]
    slug = "".join(safe)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:80] or fallback


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _build_new_plan(title: str, goal: str, hypothesis: str, metric: str) -> dict[str, Any]:
    now = datetime.now()
    ticket_id = f"{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    return {
        "bridge_schema": SCHEMA_VERSION,
        "id": ticket_id,
        "created_at": now.isoformat(timespec="seconds"),
        "owner": "user",
        "title": title,
        "goal": goal,
        "problem_statement": "",
        "hypothesis": [h.strip() for h in hypothesis.replace(";", "\n").splitlines() if h.strip()],
        "experiment_plan": "",
        "data_scope": "最新一周数据（可选 top20/top50，按实验目标选择）",
        "validation": [
            "本地严格评测：v3 / v1 参照对战，按席位平衡核验",
            "空任务率、非法动作率、决策数等指标按任务化记录",
            "如果涉及提交，先跑一轮离线复现脚本再提交"
        ],
        "acceptance_criteria": [
            f"目标指标：{metric or '显著不低于对照组，且具备统计置信度'}",
            "不破坏 BC 与 v3 约束的安全门；出现退化必须给出可回退版本"
        ],
        "risk_and_mitigation": [
            "数据版本漂移：锁定训练与评估输入文件 hash",
            "评估方差大：用 256/512 分层复验"
        ],
        "rollback_plan": "失败时回退至最近一次已通过门控的提交版本。",
        "notes_for_codex": "",
    }


def new_plan(args: argparse.Namespace) -> None:
    title = (args.title or "").strip()
    if not title:
        raise SystemExit("--title 不能为空")
    metric = (args.metric or "").strip()
    goal = (args.goal or "").strip()
    if not goal:
        raise SystemExit("--goal 不能为空")
    if not args.hypothesis:
        raise SystemExit("--hypothesis 不能为空")

    payload = _build_new_plan(title, goal, args.hypothesis, metric)
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output.is_dir():
        raise SystemExit("--output 不能是目录")
    payload["title"] = title
    payload["goal"] = goal
    payload["id"] = payload["id"]

    if args.owner:
        payload["owner"] = args.owner
    if args.data_scope:
        payload["data_scope"] = args.data_scope
    if args.problem:
        payload["problem_statement"] = args.problem
    if args.experiment_plan:
        payload["experiment_plan"] = args.experiment_plan
    if args.notes:
        payload["notes_for_codex"] = args.notes

    args.output.write_text(_json_dumps(payload), encoding="utf-8")
    print(f"[saved] {args.output}")


def validate_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise SystemExit("Invalid plan: not a JSON object")

    for key, expected_type in REQUIRED_KEYS.items():
        if key not in plan:
            raise SystemExit(f"Invalid plan: missing field '{key}'")
        if not isinstance(plan[key], expected_type):
            raise SystemExit(f"Invalid plan: field '{key}' expects {expected_type.__name__}")

    if plan["bridge_schema"] != SCHEMA_VERSION:
        raise SystemExit(
            f"Invalid plan: bridge_schema={plan['bridge_schema']} (expect {SCHEMA_VERSION})"
        )
    return plan


def render_plan(args: argparse.Namespace) -> None:
    plan = validate_plan(args.plan)
    lines = [
        f"# {plan['title']}",
        "",
        f"- 版本: {plan['bridge_schema']}",
        f"- 任务ID: {plan['id']}",
        f"- 创建时间: {plan['created_at']}",
        f"- 负责人: {plan['owner']}",
        "",
        "## 目标",
        plan["goal"],
        "",
        "## 问题描述",
        plan["problem_statement"] or "_待补充_",
        "",
        "## 假设",
        "",
    ]
    for item in plan["hypothesis"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## 实验方案",
        plan["experiment_plan"] or "_待补充_",
        "",
        f"## 数据范围\n{plan['data_scope']}",
        "",
        "## Codex 实施清单",
        "- 按以下顺序处理：修改脚本/配置 -> 复现实验 -> 生成报告 -> 归档证据 -> 更新决策档案",
        "",
        "### 验证清单",
    ])
    for item in plan["validation"]:
        lines.append(f"- [ ] {item}")

    lines.extend(["", "### 接受标准"])
    for item in plan["acceptance_criteria"]:
        lines.append(f"- [ ] {item}")

    lines.extend(["", "### 风险与兜底"])
    for item in plan["risk_and_mitigation"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "### 回退方案",
        plan["rollback_plan"],
        "",
        "### 给 Codex 的指令摘要",
        plan["notes_for_codex"] or "_待补充_",
        "",
    ])
    if args.out is None:
        args.out = args.plan.with_suffix(".tasklist.md")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {args.out}")


def auto_flow(args: argparse.Namespace) -> None:
    out_json = args.output
    if out_json.is_dir():
        raise SystemExit("--output 不能是目录")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    new_plan(args)
    validate_plan(out_json)
    render_plan(argparse.Namespace(plan=out_json, out=out_json.with_suffix(".tasklist.md")))
    print(f"[auto] done: {out_json}")


def ingest_web_spec(args: argparse.Namespace) -> None:
    raw = json.loads(args.spec.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("--spec must contain a JSON object")

    if raw.get("bridge_schema") == SCHEMA_VERSION:
        plan = raw
        if plan.get("bridge_schema") != SCHEMA_VERSION:
            raise SystemExit("unsupported bridge schema")
        required = {
            "title": str(plan.get("title", "")),
            "goal": str(plan.get("goal", "")),
            "hypothesis": "\n".join(plan.get("hypothesis", []))
            if isinstance(plan.get("hypothesis"), list)
            else str(plan.get("hypothesis", "")),
            "metric": str(plan.get("acceptance_criteria", [""])[0]).split("：")[-1]
            if isinstance(plan.get("acceptance_criteria"), list) and plan["acceptance_criteria"]
            else "",
            "owner": str(plan.get("owner", "user")),
            "problem": str(plan.get("problem_statement", "")),
            "data_scope": str(plan.get("data_scope", "")),
            "experiment_plan": str(plan.get("experiment_plan", "")),
            "notes": str(plan.get("notes_for_codex", "")),
        }
    else:
        required = {
            "title": str(raw.get("title", "")),
            "goal": str(raw.get("goal", "")),
            "hypothesis": raw.get("hypothesis", ""),
            "metric": str(raw.get("metric", "")),
            "owner": str(raw.get("owner", "user")),
            "problem": str(raw.get("problem_statement", raw.get("problem", ""))),
            "data_scope": str(raw.get("data_scope", "")),
            "experiment_plan": str(raw.get("experiment_plan", "")),
            "notes": str(raw.get("notes_for_codex", raw.get("notes", ""))),
        }

    ns = argparse.Namespace(
        title=required["title"],
        goal=required["goal"],
        hypothesis=required["hypothesis"] if isinstance(required["hypothesis"], str) else "\n".join(required["hypothesis"]),
        metric=required["metric"],
        owner=required["owner"],
        problem=required["problem"],
        data_scope=required["data_scope"],
        experiment_plan=required["experiment_plan"],
        notes=required["notes"],
        output=args.output,
    )
    auto_flow(ns)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Web GPT -> Codex handoff bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_new = subparsers.add_parser("new", help="Create a new handoff plan")
    p_new.add_argument("--title", required=True)
    p_new.add_argument("--goal", required=True, help="一句话目标")
    p_new.add_argument("--hypothesis", required=True, help="一句到三句假设，用分号或换行分隔")
    p_new.add_argument("--metric", default="")
    p_new.add_argument("--owner", default="user")
    p_new.add_argument("--problem", default="")
    p_new.add_argument("--data-scope", default="", dest="data_scope")
    p_new.add_argument("--experiment-plan", default="", dest="experiment_plan")
    p_new.add_argument("--notes", default="")
    p_new.set_defaults(func=new_plan)
    p_new.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NOTES_DIR / f"{datetime.now().strftime('%Y%m%d')}-{_slugify('web-gpt-handoff')}.json",
        help="Output JSON path",
    )

    p_auto = subparsers.add_parser("auto", help="Create + validate + render a handoff plan in one command")
    p_auto.add_argument("--title", required=True, help="一句话目标")
    p_auto.add_argument("--goal", required=True, help="本轮要解决的任务目标")
    p_auto.add_argument("--hypothesis", required=True, help="一句到三句假设，用分号或换行分隔")
    p_auto.add_argument("--metric", default="")
    p_auto.add_argument("--owner", default="user")
    p_auto.add_argument("--problem", default="")
    p_auto.add_argument("--data-scope", default="", dest="data_scope")
    p_auto.add_argument("--experiment-plan", default="", dest="experiment_plan")
    p_auto.add_argument("--notes", default="")
    p_auto.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NOTES_DIR / f"{datetime.now().strftime('%Y%m%d')}-{_slugify('web-gpt-handoff')}-auto.json",
        help="Output JSON path",
    )
    p_auto.set_defaults(func=auto_flow)

    p_ingest = subparsers.add_parser("ingest", help="Import a web GPT JSON draft and generate plan")
    p_ingest.add_argument("--spec", type=Path, required=True, help="Web GPT output JSON file")
    p_ingest.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NOTES_DIR / f"{datetime.now().strftime('%Y%m%d')}-{_slugify('web-gpt-handoff')}-imported.json",
        help="Output JSON path",
    )
    p_ingest.set_defaults(func=ingest_web_spec)

    p_val = subparsers.add_parser("validate", help="Validate an existing plan JSON")
    p_val.add_argument("--plan", type=Path, required=True)
    p_val.set_defaults(func=lambda a: print(json.dumps(validate_plan(a.plan), ensure_ascii=False, indent=2)))

    p_render = subparsers.add_parser("render", help="Render a plan JSON to a tasklist.md file")
    p_render.add_argument("--plan", type=Path, required=True)
    p_render.add_argument("--out", type=Path, default=None)
    p_render.set_defaults(func=render_plan)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
