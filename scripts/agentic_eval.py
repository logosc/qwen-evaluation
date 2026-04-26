#!/usr/bin/env python3
"""Auto-scored agent-loop evaluation for OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import operator
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


JSONDict = dict[str, Any]
Checker = Callable[[str, dict[str, Any], list[JSONDict]], bool]


@dataclass(frozen=True)
class AgentTask:
    name: str
    prompt: str
    checker: Checker
    max_steps: int = 6


TOOL_SPEC = """
You are operating in an action/observation loop. Reply with exactly one JSON object per turn.

To call a tool:
{"action":"tool","tool":"tool_name","args":{...}}

To finish:
{"action":"final","answer":"..."}

The "action" value must be either "tool" or "final". Never put a tool name in the "action" field.
Do not use markdown. Do not include prose outside the JSON object.

Available tools:
- lookup_fact({"key": string}) where useful keys include "sg_vat_rate" and "invoice_subtotal_usd"
- calculator({"expression": string})
- search_docs({"query": string})
- ticket_lookup({"ticket_id": string})
- ticket_search({"customer": string})
- free_busy({"person": string, "date": string})
- bulk_inventory_lookup({"skus": [string, ...]})
- list_issues({"label": string})
- update_issue({"issue_id": integer, "status": string, "comment": string})
- read_note({"name": string}) where the release note is named "release"
- update_note_status({"name": string, "task": string, "status": string})
"""


FACTS = {
    "sg_vat_rate": {"rate": 0.09, "name": "Singapore GST"},
    "invoice_subtotal_usd": {"amount": 120.0, "currency": "USD"},
}

DOCS = {
    "renewal": [
        {"title": "Contract renewal policy", "text": "Enterprise contracts can be renewed starting 45 days before the end date."},
        {"title": "Cancellation policy", "text": "Cancellation requires written notice."},
    ],
    "retention": [
        {"title": "Data retention", "text": "Audit logs are retained for 18 months."},
    ],
}

TICKETS = {
    "INC-1043": None,
    "INC-2042": {"ticket_id": "INC-2042", "customer": "acme", "priority": "p1", "sla_hours": 4},
}

BUSY = {
    ("ava", "2026-04-28"): [["09:00", "10:30"], ["13:00", "14:00"]],
    ("ben", "2026-04-28"): [["09:30", "10:00"], ["11:00", "12:00"]],
}

INVENTORY = {
    "A12": {"stock": 5},
    "B07": {"stock": 3},
    "C99": {"stock": 0},
}

ISSUES = [
    {"issue_id": 17, "title": "Parser crash on empty field", "label": "parser", "duplicate_of": None, "status": "open"},
    {"issue_id": 42, "title": "Parser crash duplicate report", "label": "parser", "duplicate_of": 17, "status": "open"},
    {"issue_id": 51, "title": "UI spacing bug", "label": "ui", "duplicate_of": None, "status": "open"},
]

NOTES = {
    "release": {
        "alpha": "TODO",
        "beta": "TODO",
        "gamma": "DONE",
    }
}


ALLOWED_AST_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> float:
    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_AST_OPS:
            return ALLOWED_AST_OPS[type(node.op)](eval_node(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_AST_OPS:
            return ALLOWED_AST_OPS[type(node.op)](eval_node(node.left), eval_node(node.right))
        raise ValueError(f"unsupported expression: {expression}")

    tree = ast.parse(expression, mode="eval")
    return eval_node(tree)


def make_state() -> dict[str, Any]:
    return {
        "issues": [dict(issue) for issue in ISSUES],
        "notes": {name: dict(tasks) for name, tasks in NOTES.items()},
    }


def call_tool(name: str, args: JSONDict, state: dict[str, Any]) -> JSONDict:
    if name == "lookup_fact":
        key = str(args.get("key", ""))
        return {"ok": key in FACTS, "value": copy.deepcopy(FACTS.get(key))}

    if name == "calculator":
        result = safe_calculate(str(args.get("expression", "")))
        return {"ok": True, "result": round(result, 6)}

    if name == "search_docs":
        query = str(args.get("query", "")).lower()
        hits = []
        for key, rows in DOCS.items():
            if key in query:
                hits.extend(rows)
        return {"ok": True, "hits": hits[:3]}

    if name == "ticket_lookup":
        ticket_id = str(args.get("ticket_id", ""))
        ticket = TICKETS.get(ticket_id)
        return {"ok": ticket is not None, "ticket": ticket}

    if name == "ticket_search":
        customer = str(args.get("customer", "")).lower()
        tickets = [ticket for ticket in TICKETS.values() if ticket and ticket["customer"] == customer]
        return {"ok": True, "tickets": copy.deepcopy(tickets)}

    if name == "free_busy":
        person = str(args.get("person", "")).lower()
        date = str(args.get("date", ""))
        return {"ok": True, "person": person, "date": date, "busy": copy.deepcopy(BUSY.get((person, date), []))}

    if name == "bulk_inventory_lookup":
        skus = args.get("skus", [])
        if not isinstance(skus, list):
            return {"ok": False, "error": "skus must be a list"}
        return {"ok": True, "items": {str(sku): copy.deepcopy(INVENTORY.get(str(sku), {"stock": 0})) for sku in skus}}

    if name == "list_issues":
        label = str(args.get("label", ""))
        return {"ok": True, "issues": copy.deepcopy([issue for issue in state["issues"] if issue["label"] == label])}

    if name == "update_issue":
        issue_id = int(args.get("issue_id", -1))
        for issue in state["issues"]:
            if issue["issue_id"] == issue_id:
                issue["status"] = str(args.get("status", issue["status"]))
                issue["comment"] = str(args.get("comment", ""))
                return {"ok": True, "issue": copy.deepcopy(issue)}
        return {"ok": False, "error": "issue not found"}

    if name == "read_note":
        note = str(args.get("name", ""))
        return {"ok": note in state["notes"], "tasks": copy.deepcopy(state["notes"].get(note))}

    if name == "update_note_status":
        note = str(args.get("name", ""))
        task = str(args.get("task", ""))
        status = str(args.get("status", ""))
        if note not in state["notes"] or task not in state["notes"][note]:
            return {"ok": False, "error": "note or task not found"}
        state["notes"][note][task] = status
        return {"ok": True, "tasks": copy.deepcopy(state["notes"][note])}

    return {"ok": False, "error": f"unknown tool: {name}"}


def contains_all(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return all(needle.lower() in low for needle in needles)


def tool_names(trace: list[JSONDict]) -> list[str]:
    return [step.get("tool", "") for step in trace if step.get("type") == "tool"]


TASKS = [
    AgentTask(
        "lookup_and_calculate_total",
        "Use tools to find the invoice subtotal and Singapore GST rate, then calculate the total. Final answer must be exactly in the format USD <amount> with two decimals.",
        lambda answer, state, trace: contains_all(answer, ["USD", "130.80"])
        and {"lookup_fact", "calculator"}.issubset(set(tool_names(trace))),
    ),
    AgentTask(
        "doc_retrieval_policy",
        "Use the docs tool to answer: how many days before the end date can an enterprise contract renewal start? Final answer should be brief.",
        lambda answer, state, trace: "45" in answer and "search_docs" in tool_names(trace),
    ),
    AgentTask(
        "ticket_recovery_after_miss",
        "Ticket INC-1043 may be stale. Look it up; if it is missing, search customer acme and report the active ticket id and SLA hours.",
        lambda answer, state, trace: contains_all(answer, ["INC-2042", "4"])
        and tool_names(trace)[:2] == ["ticket_lookup", "ticket_search"],
    ),
    AgentTask(
        "calendar_intersection",
        "Use calendar tools to find the first 30 minute slot on 2026-04-28 between 09:00 and 12:00 when both Ava and Ben are free. Final answer should include the start and end time.",
        lambda answer, state, trace: contains_all(answer, ["10:30", "11:00"])
        and tool_names(trace).count("free_busy") >= 2,
    ),
    AgentTask(
        "bulk_inventory_shortage",
        "Use the bulk inventory tool once for order A12 x3, B07 x5, C99 x2. Final answer should list only the SKUs that cannot be fully filled.",
        lambda answer, state, trace: contains_all(answer, ["B07", "C99"])
        and "A12" not in answer
        and tool_names(trace).count("bulk_inventory_lookup") == 1,
    ),
    AgentTask(
        "issue_side_effect",
        "Find parser issues. Close only the duplicate parser issue with comment 'Duplicate of #17', then final-answer with the closed issue id.",
        lambda answer, state, trace: "42" in answer
        and any(issue["issue_id"] == 42 and issue["status"] == "closed" and issue.get("comment") == "Duplicate of #17" for issue in state["issues"])
        and any(issue["issue_id"] == 17 and issue["status"] == "open" for issue in state["issues"]),
    ),
    AgentTask(
        "note_update_side_effect",
        "Read release note tasks. Mark only beta as DONE. Leave alpha unchanged. Final answer should say beta is DONE.",
        lambda answer, state, trace: contains_all(answer, ["beta", "DONE"])
        and state["notes"]["release"]["beta"] == "DONE"
        and state["notes"]["release"]["alpha"] == "TODO",
    ),
    AgentTask(
        "tool_abstention",
        "Do not call any tools. Final answer exactly: ready",
        lambda answer, state, trace: answer.strip().lower() == "ready" and not tool_names(trace),
        max_steps=2,
    ),
]


def extract_json(text: str) -> tuple[JSONDict | None, bool]:
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None, True
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            return obj if isinstance(obj, dict) else None, False
        except json.JSONDecodeError:
            return None, False

    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(stripped[start : end + 1])
            return obj if isinstance(obj, dict) else None, False
        except json.JSONDecodeError:
            return None, False
    return None, False


def call_model(base_url: str, model: str, messages: list[JSONDict], max_tokens: int, timeout_s: float) -> tuple[str, dict[str, Any], float]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        data = json.loads(response.read())
    elapsed_ms = (time.perf_counter() - started) * 1000
    return data["choices"][0]["message"].get("content") or "", data, elapsed_ms


def run_task(base_url: str, model: str, task: AgentTask, max_tokens: int, timeout_s: float) -> JSONDict:
    messages = [
        {"role": "system", "content": TOOL_SPEC},
        {"role": "user", "content": task.prompt},
    ]
    state = make_state()
    trace: list[JSONDict] = []
    strict_json_count = 0
    invalid_json_count = 0
    schema_error_count = 0
    total_latency_ms = 0.0
    final_answer = ""
    stopped = "max_steps"

    for step_num in range(1, task.max_steps + 1):
        content, data, elapsed_ms = call_model(base_url, model, messages, max_tokens, timeout_s)
        total_latency_ms += elapsed_ms
        obj, strict = extract_json(content)

        if not obj:
            invalid_json_count += 1
            trace.append({"type": "invalid_json", "step": step_num, "content": content[:500]})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": 'OBSERVATION: invalid JSON. Reply with exactly {"action":"tool",...} or {"action":"final",...}.'})
            continue

        strict_json_count += int(strict)
        action = obj.get("action")
        messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})

        if action == "final":
            final_answer = str(obj.get("answer", ""))
            trace.append({"type": "final", "step": step_num, "answer": final_answer[:500], "strict_json": strict})
            stopped = "final"
            break

        if action != "tool":
            schema_error_count += 1
            trace.append({"type": "invalid_action", "step": step_num, "object": obj})
            messages.append({"role": "user", "content": "OBSERVATION: invalid action. Use action=tool or action=final."})
            continue

        tool = str(obj.get("tool", ""))
        args = obj.get("args", {})
        if not isinstance(args, dict):
            args = {}
        try:
            observation = call_tool(tool, args, state)
        except Exception as exc:
            observation = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        trace.append(
            {
                "type": "tool",
                "step": step_num,
                "tool": tool,
                "args": args,
                "observation": observation,
                "strict_json": strict,
            }
        )
        messages.append({"role": "user", "content": f"OBSERVATION: {json.dumps(observation, ensure_ascii=False)}"})

    passed = task.checker(final_answer, state, trace)
    return {
        "model": model,
        "task": task.name,
        "pass": passed,
        "stopped": stopped,
        "final_answer": final_answer[:500],
        "steps": len(trace),
        "tool_calls": sum(1 for step in trace if step.get("type") == "tool"),
        "strict_json_count": strict_json_count,
        "invalid_json_count": invalid_json_count,
        "schema_error_count": schema_error_count,
        "latency_ms": round(total_latency_ms, 1),
        "trace": trace,
    }


def summarize(rows: list[JSONDict]) -> JSONDict:
    return {
        "passed": sum(1 for row in rows if row["pass"]),
        "total": len(rows),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 1) if rows else 0,
        "avg_steps": round(sum(row["steps"] for row in rows) / len(rows), 2) if rows else 0,
        "avg_tool_calls": round(sum(row["tool_calls"] for row in rows) / len(rows), 2) if rows else 0,
        "invalid_json_count": sum(row["invalid_json_count"] for row in rows),
        "schema_error_count": sum(row["schema_error_count"] for row in rows),
        "strict_json_rate": round(
            sum(row["strict_json_count"] for row in rows) / max(1, sum(row["steps"] for row in rows)),
            3,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--output-jsonl")
    args = parser.parse_args()

    rows = [run_task(args.base_url, args.model, task, args.max_tokens, args.request_timeout) for task in TASKS]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    print(json.dumps({"model": args.model, "summary": summarize(rows)}, sort_keys=True), flush=True)

    if args.output_jsonl:
        with open(args.output_jsonl, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.write(json.dumps({"model": args.model, "summary": summarize(rows)}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
