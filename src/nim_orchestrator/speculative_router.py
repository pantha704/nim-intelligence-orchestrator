"""Structured speculative router.

Replaces style-based confidence (which rewarded assertiveness and brevity)
with structured routing signals: task type, ambiguity, verification availability,
risk, and candidate disagreement. Never treats assertiveness or brevity as correctness.
"""

import asyncio
import re
from dataclasses import dataclass

from .router_client import ChatResult, RouterClient


@dataclass
class SpeculativeResult:
    escalate: bool
    quick_answer: str = ""
    quick_result: ChatResult | None = None
    reason: str = ""
    route: str = "complex"  # "direct" / "verifiable" / "complex" / "open_ended"
    signals: dict | None = None


def _detect_task_type(prompt: str, answer: str) -> str:
    """Classify the task type from the prompt and answer."""
    prompt_lower = prompt.lower()

    if re.search(r"\b(?:write|implement|code|function|class|program|script)\b", prompt_lower):
        if "```" in answer or "def " in answer or "function " in answer:
            return "verifiable"
        return "verifiable"

    if re.search(r"\b(?:calculate|compute|solve|how much|how many)\b", prompt_lower):
        return "verifiable"

    if re.search(r"\b(?:what is \d|what'?s \d)\b", prompt_lower):
        # "What is 17 * 23?" — math question that the speculative answer can handle directly.
        # Verification is available but the speculative answer is usually correct for arithmetic.
        return "direct"

    if re.search(r"\b(?:prove|design|architect|analyze|optimize|compare|trade-off|debug|refactor)\b", prompt_lower):
        return "complex"

    if re.search(r"\b(?:what do you think|best|worst|favorite|recommend|advise|opinion|should i)\b", prompt_lower):
        return "open_ended"

    if re.search(r"\b(?:what is|define|who is|when did|where is|capital of)\b", prompt_lower):
        return "direct"

    return "complex"


def _has_verification_available(prompt: str, answer: str) -> bool:
    """Check if deterministic verification is possible for this answer."""
    if "```" in answer:
        return True
    if re.search(r"=\s*\d+(?:\.\d+)?", answer):
        return True
    if re.search(r"\bdef \w+\(", answer):
        return True
    return False


def _detect_truncation(result: ChatResult) -> str:
    """Return truncation status: "none" / "truncated"."""
    if result.finish_reason == "length":
        return "truncated"
    return "none"


async def speculative_route(
    client: RouterClient,
    model: str,
    prompt: str,
    max_quick_tokens: int = 256,
) -> SpeculativeResult:
    """Quick speculative call producing structured routing signals.

    Does NOT use assertiveness, brevity, or token count as confidence.
    Routes based on task type, verification availability, and risk.
    """
    try:
        result = await asyncio.wait_for(
            client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                reasoning_effort="none",
                max_tokens=max_quick_tokens,
            ),
            timeout=20,
        )
    except Exception as e:
        return SpeculativeResult(
            escalate=True,
            reason=f"quick call failed: {type(e).__name__}",
        )

    if not result.content or len(result.content.strip()) < 5:
        return SpeculativeResult(
            escalate=True,
            reason="empty/too-short quick response",
            quick_result=result,
        )

    task_type = _detect_task_type(prompt, result.content)
    verifiable = _has_verification_available(prompt, result.content)
    truncation = _detect_truncation(result)

    signals = {
        "task_type": task_type,
        "verification_available": verifiable,
        "truncation": truncation,
        "finish_reason": result.finish_reason,
    }

    # Routing decisions based on structured signals, not style

    # If truncated, the answer is incomplete — escalate
    if truncation == "truncated":
        return SpeculativeResult(
            escalate=True,
            quick_answer=result.content,
            quick_result=result,
            reason="response truncated (hit max_tokens)",
            route=task_type,
            signals=signals,
        )

    # Direct factual question with a non-truncated answer — accept
    if task_type == "direct" and truncation == "none":
        return SpeculativeResult(
            escalate=False,
            quick_answer=result.content,
            quick_result=result,
            reason=f"direct factual question, complete response (route={task_type})",
            route="direct",
            signals=signals,
        )

    # Verifiable task — always escalate to full pipeline for verification
    if task_type == "verifiable":
        return SpeculativeResult(
            escalate=True,
            quick_answer=result.content,
            quick_result=result,
            reason=f"verifiable task — needs deterministic verification (route={task_type})",
            route="verifiable",
            signals=signals,
        )

    # Complex task — escalate
    if task_type == "complex":
        return SpeculativeResult(
            escalate=True,
            quick_answer=result.content,
            quick_result=result,
            reason=f"complex task — needs multi-agent pipeline (route={task_type})",
            route="complex",
            signals=signals,
        )

    # Open-ended — escalate
    if task_type == "open_ended":
        return SpeculativeResult(
            escalate=True,
            quick_answer=result.content,
            quick_result=result,
            reason=f"open-ended task — needs multi-agent pipeline (route={task_type})",
            route="open_ended",
            signals=signals,
        )

    # Fallback: escalate
    return SpeculativeResult(
        escalate=True,
        quick_answer=result.content,
        quick_result=result,
        reason="unclassified — escalating",
        route=task_type,
        signals=signals,
    )
