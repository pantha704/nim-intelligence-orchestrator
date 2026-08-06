"""Semantic claim extraction and math verification.

A number appearing in a negated or unrelated sentence is not evidence.
Equality claims are only evidence when stated affirmatively.
"""
import re
from dataclasses import dataclass, field

from .external_checks import EQUALITY_PATTERNS

NEGATION_RE = re.compile(
    r"\b(?:not|no|never|isn'?t|aren'?t|wasn'?t|weren'?t|won'?t|can'?t|cannot|"
    r"doesn'?t|don'?t|didn'?t|incorrect|wrong|false|contradicts?|incorrectly|fails?)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text)]
    return [p for p in parts if p]


@dataclass
class EqualityClaim:
    text: str
    left: float
    operator: str
    right: float
    expected: float
    actual: float
    correct: bool


@dataclass
class Claim:
    text: str
    numbers: list[str] = field(default_factory=list)
    negated: bool = False
    equalities: list[EqualityClaim] = field(default_factory=list)

    @property
    def has_affirmative_evidence(self) -> bool:
        return not self.negated and any(e.correct for e in self.equalities)

    @property
    def has_affirmative_error(self) -> bool:
        return not self.negated and any(not e.correct for e in self.equalities)


_OP_MAP = {"×": "*", "÷": "/", "+": "+", "-": "-", "*": "*", "/": "/"}


def _parse_equality(match_text: str, groups: tuple) -> EqualityClaim | None:
    if len(groups) == 4:
        a, op, b, expected = groups
    else:
        return None
    op_s = _OP_MAP.get(op, op)
    if op_s not in ("+", "-", "*", "/"):
        return None
    try:
        a_f, b_f, expected_f = float(a), float(b), float(expected)
        if op_s == "+":
            actual = a_f + b_f
        elif op_s == "-":
            actual = a_f - b_f
        elif op_s == "*":
            actual = a_f * b_f
        else:
            if b_f == 0:
                return None
            actual = a_f / b_f
        return EqualityClaim(
            text=match_text,
            left=a_f,
            operator=op_s,
            right=b_f,
            expected=expected_f,
            actual=actual,
            correct=abs(actual - expected_f) < 0.01,
        )
    except (ValueError, TypeError):
        return None


def extract_claims(text: str) -> list[Claim]:
    """Extract per-sentence claims: numbers, negation flag, equality claims."""
    claims: list[Claim] = []
    for sentence in split_sentences(text):
        negated = bool(NEGATION_RE.search(sentence))
        numbers = re.findall(r"-?\d+(?:\.\d+)?", sentence)
        equalities = []
        for pattern in EQUALITY_PATTERNS:
            for m in re.finditer(pattern, sentence):
                eq = _parse_equality(m.group(0), m.groups())
                if eq is not None:
                    equalities.append(eq)
        claims.append(Claim(text=sentence, numbers=numbers, negated=negated, equalities=equalities))
    return claims


def verify_math_claims(answer: str) -> tuple[str, str, str]:
    """Negation-aware math verification.

    Returns (status, evidence, details):
    - fail: an affirmative equality claim computes to the wrong value
    - pass: an affirmative equality claim is correct
    - unverified: no affirmative checkable equality (negated claims never count)
    """
    claims = extract_claims(answer)

    affirmative_errors = [e for c in claims for e in c.equalities if c.has_affirmative_error]
    if affirmative_errors:
        wrong = "; ".join(f"{e.text} (actual {e.actual:g})" for e in affirmative_errors[:5])
        return "fail", f"{len(affirmative_errors)} wrong affirmative equation(s)", wrong

    affirmative_correct = [e for c in claims for e in c.equalities if c.has_affirmative_evidence]
    if affirmative_correct:
        ok = "; ".join(e.text for e in affirmative_correct[:5])
        return "pass", f"{len(affirmative_correct)} verified equation(s)", ok

    negated_claims = [c for c in claims if c.negated and c.equalities]
    if negated_claims:
        return "unverified", "equations appear only in negated sentences — not counted as evidence", ""
    if any(c.equalities for c in claims):
        return "unverified", "equations present but not in an affirmative claim", ""
    return "unverified", "no checkable equations in answer", ""


def semantic_value_present(answer: str, expected_value: str) -> tuple[str, str]:
    """Check an expected value appears in an AFFIRMATIVE context.

    Returns (status, evidence):
    - verified: expected value in an affirmative sentence
    - failed: value only in negated sentences, or other numbers affirmatively stated
    - unverified: nothing comparable in the answer
    """
    claims = extract_claims(answer)
    for c in claims:
        if expected_value in c.numbers:
            if c.negated:
                return "failed", f"expected {expected_value} appears only in a negated claim: '{c.text[:80]}'"
            return "verified", f"expected {expected_value} stated affirmatively"

    for c in claims:
        if c.numbers and not c.negated:
            return "failed", f"answer states other number(s) {c.numbers[:5]} affirmatively, not {expected_value}"
    return "unverified", "answer contains no numbers to compare"


def extract_factual_claims(answer: str) -> list[dict]:
    """Claim extraction for research: sentences with subject-like content."""
    claims = []
    for i, sentence in enumerate(split_sentences(answer)):
        if len(sentence) < 12:
            continue
        claims.append({
            "claim_id": f"c{i}",
            "text": sentence[:300],
            "negated": bool(NEGATION_RE.search(sentence)),
            "verifiable_externally": False,
        })
    return claims
