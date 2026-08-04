"""Minimal transport gate: reject only empty, oversized, or malformed input.

Never modifies, wraps, or removes user instructions. Preserves raw_prompt immutably.
"""
from dataclasses import dataclass

MAX_PROMPT_SIZE = 100_000  # 100KB

@dataclass
class TransportGateResult:
    ok: bool
    raw_prompt: str
    reason: str = ""

def transport_gate(raw_prompt: str) -> TransportGateResult:
    if not raw_prompt or not raw_prompt.strip():
        return TransportGateResult(ok=False, raw_prompt=raw_prompt, reason="Empty prompt")
    if len(raw_prompt) > MAX_PROMPT_SIZE:
        return TransportGateResult(ok=False, raw_prompt=raw_prompt, reason=f"Prompt exceeds {MAX_PROMPT_SIZE} chars")
    if "\x00" in raw_prompt:
        return TransportGateResult(ok=False, raw_prompt=raw_prompt, reason="Null bytes detected")
    try:
        raw_prompt.encode("utf-8")
    except Exception:
        return TransportGateResult(ok=False, raw_prompt=raw_prompt, reason="Invalid UTF-8 encoding")
    return TransportGateResult(ok=True, raw_prompt=raw_prompt)
