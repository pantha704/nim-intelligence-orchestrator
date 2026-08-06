"""Structured speculative router.

Uses classify_task_type from policy.py as the single task classifier.
Routes based on task type, verification availability, and truncation —
never on assertiveness, brevity, or token count.
"""

import asyncio
import re
from dataclasses import dataclass

from .boundaries import DIRECT_ANTI_INJECTION_SYSTEM_PROMPT, wrap_problem_block
from .policy import classify_task_type
from .router_client import BudgetExhaustedError, ChatResult, RouterClient, budgeted_chat


@dataclass
class SpeculativeResult:
    escalate: bool
    quick_answer: str = ""
    quick_result: ChatResult | None = None
    reason: str = ""
    route: str = "complex"
    signals: dict | None = None


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
    ctx=None,
) -> SpeculativeResult:
    """Quick speculative call producing structured routing signals.

    Task classification comes from policy.classify_task_type — the single
    classifier. When ctx is provided, the call runs under budget enforcement.
    """
    try:
        if ctx is not None:
            result = await budgeted_chat(
                client,
                ctx,
                agent_name="speculative",
                model=model,
                messages=[
                    # raw prompt only inside the structured data boundary,
                    # with the anti-injection system prompt every agent has
                    {"role": "system", "content": DIRECT_ANTI_INJECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": wrap_problem_block(prompt)},
                ],
                temperature=0.2,
                reasoning_effort="none",
                max_tokens=max_quick_tokens,
                timeout=20,
            )
        else:
            result = await asyncio.wait_for(
                client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": DIRECT_ANTI_INJECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": wrap_problem_block(prompt)},
                    ],
                    temperature=0.2,
                    reasoning_effort="none",
                    max_tokens=max_quick_tokens,
                ),
                timeout=20,
            )
    except BudgetExhaustedError as e:
        return SpeculativeResult(
            escalate=True,
            reason=f"budget exhausted — cannot make speculative call: {e}",
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

    task_type = classify_task_type(prompt, result.content)
    verifiable = _has_verification_available(prompt, result.content)
    truncation = _detect_truncation(result)

    signals = {
        "task_type": task_type,
        "verification_available": verifiable,
        "truncation": truncation,
        "finish_reason": result.finish_reason,
    }

    # Routing decisions based on structured signals, not style
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
