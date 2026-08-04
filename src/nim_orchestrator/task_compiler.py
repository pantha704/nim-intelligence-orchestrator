"""Task Compiler: converts a raw prompt into a validated TaskSpec.

Distinguishes safely-assumable details from materially-ambiguous ones.
Preserves the original prompt. Maintains a requirement ledger.
"""
import json
import re
import time
from typing import Literal
from pydantic import BaseModel, Field
from .router_client import RouterClient


class Ambiguity(BaseModel):
    question: str
    impact: Literal["low", "high"] = "low"
    resolution: Literal["ask", "assume"] = "assume"
    assumption: str = ""


class Subtask(BaseModel):
    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: str = ""


class TaskSpec(BaseModel):
    objective: str
    deliverables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    context: str = ""
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    subtasks: list[Subtask] = Field(default_factory=list)
    verification_plan: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    recommended_route: Literal["direct", "verifiable", "complex", "open_ended"] = "complex"


class TaskCompilerResult:
    def __init__(self, task_spec: TaskSpec, needs_clarification: bool = False,
                 clarification_question: str = "", raw_json: str = "", latency_ms: float = 0):
        self.task_spec = task_spec
        self.needs_clarification = needs_clarification
        self.clarification_question = clarification_question
        self.raw_json = raw_json
        self.latency_ms = latency_ms


TASK_COMPILER_SYSTEM_PROMPT = """You are a Task Compiler. Analyze the user's prompt and produce a structured task specification.

Output ONLY valid JSON with these exact fields:
{
  "objective": "One clear sentence stating what needs to be accomplished",
  "deliverables": ["List of concrete outputs expected"],
  "constraints": ["Explicit constraints stated or strongly implied in the prompt"],
  "context": "The complete original prompt, verbatim",
  "assumptions": ["Assumptions you are making that are safely assumable (low impact if wrong)"],
  "ambiguities": [
    {
      "question": "What needs to be clarified?",
      "impact": "low or high (high = different interpretation materially changes the output)",
      "resolution": "ask or assume",
      "assumption": "If assuming, state the assumption; if asking, leave empty"
    }
  ],
  "subtasks": [
    {
      "id": "s1",
      "description": "What this subtask does",
      "depends_on": ["ids of subtasks that must complete first, or empty list"],
      "acceptance_criteria": "How to verify this subtask is done correctly"
    }
  ],
  "verification_plan": ["Concrete steps to verify the final output"],
  "risk_level": "low, medium, or high",
  "recommended_route": "direct (simple factual), verifiable (can be checked with tests/tools), complex (multi-step reasoning), or open_ended (creative/advisory)"
}

Rules:
- NEVER invent requirements not present or implied in the prompt.
- If a detail is missing but safely assumable, list it as an assumption, NOT a constraint.
- If a detail is missing and its absence materially changes the output, mark it as ambiguity with resolution="ask".
- If there are no subtasks (simple task), return an empty list.
- Recommended route "direct" = simple factual lookup; "verifiable" = code, math, or testable output; "complex" = multi-step reasoning; "open_ended" = advisory/creative.
- Risk level: "low" = simple/well-defined; "medium" = some ambiguity or multi-step; "high" = significant ambiguity or safety concerns.
- Preserve the FULL original prompt in the context field. Do not truncate or summarize it."""


async def compile_task(client: RouterClient, model: str, raw_prompt: str,
                       timeout_seconds: int = 25) -> TaskCompilerResult:
    t0 = time.monotonic()
    
    messages = [
        {"role": "system", "content": TASK_COMPILER_SYSTEM_PROMPT},
        {"role": "user", "content": raw_prompt},
    ]
    
    try:
        import asyncio
        result = await asyncio.wait_for(
            client.chat(
                model=model,
                messages=messages,
                temperature=0.1,
                reasoning_effort="none",
                max_tokens=2048,
            ),
            timeout=timeout_seconds,
        )
        raw_json = result.content
        latency_ms = (time.monotonic() - t0) * 1000
        
        task_spec = _parse_task_spec(raw_json, raw_prompt)
        
        needs_clarification = False
        clarification_question = ""
        for amb in task_spec.ambiguities:
            if amb.resolution == "ask" and amb.impact == "high":
                needs_clarification = True
                clarification_question = amb.question
                break
        
        return TaskCompilerResult(
            task_spec=task_spec,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            raw_json=raw_json,
            latency_ms=latency_ms,
        )
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        task_spec = TaskSpec(
            objective=raw_prompt[:200],
            context=raw_prompt,
            recommended_route="complex",
            risk_level="medium",
            assumptions=[f"Task compilation failed: {type(e).__name__}"],
        )
        return TaskCompilerResult(
            task_spec=task_spec,
            raw_json="",
            latency_ms=latency_ms,
        )


# Thresholds for compiler bypass — avoids 8-11s latency for obvious simple queries
_BYPASS_MAX_LENGTH = 100
_BYPASS_KEYWORDS = {
    "what is", "who is", "when did", "where is", "capital of",
    "define", "how many",
}


def should_bypass_compiler(raw_prompt: str) -> bool:
    """Return True for clearly simple, low-risk prompts that can skip the Task Compiler.

    Criteria: short, starts with a simple factual keyword, no ambiguity signals,
    no compound requests, no code, no special instructions.
    """
    stripped = raw_prompt.strip()
    if len(stripped) > _BYPASS_MAX_LENGTH:
        return False
    if "\n" in stripped:
        return False
    if "```" in stripped:
        return False
    lower = stripped.lower()
    if not any(lower.startswith(kw) for kw in _BYPASS_KEYWORDS):
        return False
    if lower.count("?") > 1:
        return False
    if any(kw in lower for kw in (" and ", " or ", " vs ", " versus ")):
        return False
    if any(kw in lower for kw in ("write", "code", "function", "debug", "refactor", "analyze")):
        return False
    return True


def bypass_task_spec(raw_prompt: str) -> TaskCompilerResult:
    """Create a TaskSpec for obviously simple queries without a model call."""
    return TaskCompilerResult(
        task_spec=TaskSpec(
            objective=raw_prompt,
            context=raw_prompt,
            risk_level="low",
            recommended_route="direct",
            assumptions=["Simple factual query — compiler bypassed"],
        ),
        latency_ms=0,
    )


def _extract_json(raw: str) -> dict | None:
    """Extract JSON from raw model output, handling fenced blocks and multiline."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from fenced code block: ```json\n...\n``` or ```\n...\n```
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
    for block in fenced:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # Try extracting bare multiline JSON object (find outermost { ... })
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = raw[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1
                    continue

    # Fallback: old single-line approach
    for line in raw.split("\n"):
        if "{" in line:
            s = line.index("{")
            for end in range(len(line), s, -1):
                try:
                    return json.loads(line[s:end])
                except json.JSONDecodeError:
                    continue

    return None


def _parse_task_spec(raw_json: str, original_prompt: str) -> TaskSpec:
    data = _extract_json(raw_json)

    if data is None:
        return TaskSpec(
            objective=original_prompt[:200],
            context=original_prompt,
            recommended_route="complex",
            risk_level="medium",
        )

    # Force immutable original context — never trust model-generated context
    data["context"] = original_prompt

    try:
        return TaskSpec(**data)
    except Exception:
        _rl = data.get("risk_level", "medium")
        _rr = data.get("recommended_route", "complex")
        if _rl not in ("low", "medium", "high"):
            _rl = "medium"
        if _rr not in ("direct", "verifiable", "complex", "open_ended"):
            _rr = "complex"

        _ambiguities = []
        for a in data.get("ambiguities", []):
            try:
                _ambiguities.append(Ambiguity(**a))
            except Exception:
                continue

        _subtasks = []
        for s in data.get("subtasks", []):
            try:
                _subtasks.append(Subtask(**s))
            except Exception:
                continue

        return TaskSpec(
            objective=data.get("objective", original_prompt[:200]),
            deliverables=data.get("deliverables", []),
            constraints=data.get("constraints", []),
            context=original_prompt,
            assumptions=data.get("assumptions", []),
            ambiguities=_ambiguities,
            subtasks=_subtasks,
            verification_plan=data.get("verification_plan", []),
            risk_level=_rl,
            recommended_route=_rr,
        )
