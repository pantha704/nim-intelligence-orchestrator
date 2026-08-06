"""Nonce-delimited structured data boundaries for ALL untrusted model input.

Untrusted content (raw prompts, candidate answers, dependency outputs,
critiques) is serialized as JSON inside per-request random-nonce markers:

    [BEGIN NIM DATA <nonce>]
    {...json...}
    [END NIM DATA <nonce>]

Attacker content containing marker strings cannot close the boundary early
(the nonce is unknown), and JSON escaping prevents structure injection.

Used by both the adaptive DAG and the fixed multi-agent pipeline — every
model call receives untrusted content only through this wrapper.
"""
import json
import secrets


def wrap_data_block(payload: dict, note: str = "") -> str:
    """Serialize untrusted content as non-escapable, nonce-delimited JSON."""
    nonce = secrets.token_hex(16)
    body = json.dumps(payload, ensure_ascii=False)
    block = f"[BEGIN NIM DATA {nonce}]\n{body}\n[END NIM DATA {nonce}]"
    return f"{block}\n{note}".strip()


def wrap_problem_block(problem: str, note: str = "") -> str:
    """Wrap the raw problem/query as a structured data block."""
    return wrap_data_block({"original_problem": problem}, note=note)
