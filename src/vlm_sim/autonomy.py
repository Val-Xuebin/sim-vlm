from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


SUPPORTED_ACTIONS = (
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "Crouch",
    "Stand",
    "Stop",
)
ALLOWED_ACTIONS = set(SUPPORTED_ACTIONS)

POLICY_OUTPUT_CONTRACT = f"""Return only one JSON object:
{{
  "observation": "what is visually grounded in the current RGB frame",
  "new_information": "evidence not already present in policy memory",
  "action": "one exact action from the supported list",
  "confidence": 0.0,
  "task_status": "exploring | completed | blocked",
  "rationale": "brief reason for the action and confidence"
}}

Supported AI2-THOR actions in this control loop:
{", ".join(SUPPORTED_ACTIONS)}

confidence is a number from 0 to 1 measuring whether accumulated visual evidence is sufficient
to complete the task. Do not include Markdown fences or text outside the JSON object."""


@dataclass
class PolicyDecision:
    observation: str
    new_information: str
    action: str
    confidence: float
    task_status: str
    rationale: str
    raw: str
    action_result: str = "not executed"


def build_policy_prompt(task: str, memory: list[PolicyDecision], threshold: float) -> str:
    recent = memory[-8:]
    history = [
        {
            "step": index + max(0, len(memory) - len(recent)),
            "observation": item.observation,
            "new_information": item.new_information,
            "action": item.action,
            "confidence": item.confidence,
            "task_status": item.task_status,
            "action_result": item.action_result,
        }
        for index, item in enumerate(recent)
    ]
    return (
        "You control an embodied agent from its current egocentric RGB observation.\n"
        f"Task: {task}\n"
        f"Stop confidence threshold: {threshold:.2f}\n"
        "Previous compressed observations and decisions:\n"
        f"{json.dumps(history, ensure_ascii=False)}\n\n"
        "Seek a new viewpoint when important task evidence is unseen or uncertain. Do not claim "
        "hidden information. Use Stop when the task is completed with sufficient confidence or "
        "no safe useful action remains.\n\n"
        f"{POLICY_OUTPUT_CONTRACT}"
    )


def parse_policy_decision(raw: str) -> PolicyDecision:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        payload: dict[str, Any] = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("VLM policy output is not valid JSON") from exc

    action = str(payload.get("action") or payload.get("recommended_action") or "Stop")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported VLM action: {action}")
    try:
        confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError) as exc:
        raise ValueError("VLM confidence must be numeric") from exc
    status = str(payload.get("task_status", "exploring")).lower()
    if status not in {"exploring", "completed", "blocked"}:
        status = "exploring"
    return PolicyDecision(
        observation=str(payload.get("observation", "")),
        new_information=str(payload.get("new_information", "")),
        action=action,
        confidence=confidence,
        task_status=status,
        rationale=str(payload.get("rationale", "")),
        raw=raw,
    )


def should_stop(decision: PolicyDecision, threshold: float) -> str | None:
    if decision.confidence >= threshold:
        return "requested confidence threshold reached"
    if decision.task_status == "blocked":
        return "policy reported that the task is blocked"
    if decision.action == "Stop":
        return "policy selected Stop"
    return None


def policy_trace_markdown(memory: list[PolicyDecision], stop_reason: str | None = None) -> str:
    lines = ["### Autonomous Policy Trace"]
    for index, item in enumerate(memory):
        lines.append(
            f"**{index}.** `{item.action}` · confidence `{item.confidence:.2f}` · "
            f"status `{item.task_status}` · `{item.action_result}`  \n"
            f"{item.new_information or item.observation or '—'}"
        )
    if stop_reason:
        lines.append(f"\n**Stopped:** {stop_reason}.")
    return "\n\n".join(lines)
