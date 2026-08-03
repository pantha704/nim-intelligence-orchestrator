"""Speculative router: try a quick direct answer, escalate to full pipeline if unsatisfied.

Replaces the keyword-matching difficulty router. The model itself is the best
difficulty classifier — a short quick response with high confidence means simple;
a long, hedging, or low-confidence response means complex.
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
    confidence: float = 0.0


HEDGE_WORDS = {
    "however", "although", "it depends", "on the other hand",
    "generally", "typically", "in most cases", "might be",
    "could be", "arguably", "subjective", "debatable",
    "unfortunately", "i'm not sure", "it's complex",
}

COMPLETION_MARKERS = {"the answer is", "answer:", "therefore", "thus", "hence", "in conclusion", "final answer"}


def _estimate_confidence(answer: str, finish_reason: str, tokens: int) -> float:
    """Heuristic confidence estimate based on response characteristics."""
    score = 0.5

    stripped = answer.strip().lower()

    if any(stripped.startswith(m) for m in COMPLETION_MARKERS):
        score += 0.3

    if tokens < 50:
        score += 0.2

    stripped_words = stripped.split()
    hedge_count = sum(1 for word in stripped_words if word in HEDGE_WORDS)
    if hedge_count == 0:
        score += 0.15
    else:
        score -= 0.1 * hedge_count

    if finish_reason == "stop":
        score += 0.1
    elif finish_reason == "length":
        score += 0.2

    if len(answer) > 2000:
        score += 0.1

    return max(0.0, min(1.0, score))


def _is_hedging(answer: str) -> bool:
    lower = answer.lower()
    hedge_count = sum(1 for w in HEDGE_WORDS if w in lower)
    return hedge_count >= 3


def _has_clear_answer(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return False

    for marker in COMPLETION_MARKERS:
        if marker in stripped.lower():
            return True

    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if len(lines) == 1 and len(lines[0]) < 100:
        return True

    code_blocks = re.findall(r"```(\w+)?\n.*?```", stripped, re.DOTALL)
    if code_blocks:
        return True

    return False


async def speculative_route(
    client: RouterClient,
    model: str,
    prompt: str,
    max_quick_tokens: int = 256,
) -> SpeculativeResult:
    """Quick speculative call. Returns whether to escalate to full pipeline.

    Strategy:
    1. Quick non-streaming response with a tight token budget
    2. If we get a clear, concise answer with high confidence → accept
    3. If the response is hedging, too long, or low confidence → escalate
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

    confidence = _estimate_confidence(result.content, result.finish_reason, result.tokens_generated)

    if not result.content or len(result.content.strip()) < 10:
        return SpeculativeResult(
            escalate=True,
            reason="empty/too-short quick response",
            quick_result=result,
        )

    if _is_hedging(result.content):
        return SpeculativeResult(
            escalate=True,
            reason="response is hedging",
            quick_answer=result.content,
            quick_result=result,
            confidence=confidence,
        )

    if confidence >= 0.7 and _has_clear_answer(result.content):
        return SpeculativeResult(
            escalate=False,
            quick_answer=result.content,
            quick_result=result,
            reason="clear answer with high confidence",
            confidence=confidence,
        )

    if result.finish_reason == "length":
        return SpeculativeResult(
            escalate=True,
            quick_answer=result.content,
            quick_result=result,
            reason="hit max_tokens — likely needs more depth",
            confidence=confidence,
        )

    if confidence < 0.5:
        return SpeculativeResult(
            escalate=True,
            quick_answer=result.content,
            quick_result=result,
            reason=f"low confidence ({confidence:.2f})",
            confidence=confidence,
        )

    return SpeculativeResult(
        escalate=True,
        quick_answer=result.content,
        quick_result=result,
        reason=f"ambiguous ({confidence:.2f})",
        confidence=confidence,
    )
