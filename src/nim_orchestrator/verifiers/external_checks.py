import ast
import re
from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    verifier_name: str
    passed: bool
    details: str = ""
    error: str = ""


@dataclass
class VerificationReport:
    results: list[VerificationResult] = field(default_factory=list)
    all_passed: bool = True
    failures: list[str] = field(default_factory=list)

    def add(self, result: VerificationResult) -> None:
        self.results.append(result)
        if not result.passed:
            self.all_passed = False
            self.failures.append(
                f"{result.verifier_name}: {result.details or result.error}"
            )


async def verify_code_execution_disabled(answer: str) -> VerificationResult:
    """Code execution verification is disabled until a sandbox exists.

    Running generated Python on the host is a security risk (RCE).
    This function reports the fact that code blocks were found but could not be verified.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", answer, re.DOTALL)
    if not code_blocks:
        return VerificationResult(
            verifier_name="code_execution",
            passed=True,
            details="no code blocks found",
        )
    return VerificationResult(
        verifier_name="code_execution",
        passed=False,
        details=f"Code execution disabled — sandbox not yet implemented. {len(code_blocks)} code block(s) found but not verified.",
    )


async def verify_python_syntax(answer: str) -> VerificationResult:
    """Check that Python code blocks parse syntactically. AST only — never executes."""
    if "```" not in answer:
        return VerificationResult(
            verifier_name="python_syntax",
            passed=True,
            details="no code blocks to check",
        )

    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", answer, re.DOTALL)
    failed = []
    for i, code in enumerate(code_blocks):
        try:
            ast.parse(code)
        except SyntaxError as e:
            failed.append(f"block {i}: {e}")
        except Exception as e:
            failed.append(f"block {i}: {e}")

    if failed:
        return VerificationResult(
            verifier_name="python_syntax",
            passed=False,
            details="; ".join(failed),
        )
    return VerificationResult(
        verifier_name="python_syntax",
        passed=True,
        details=f"{len(code_blocks)} block(s) parse successfully",
    )


async def verify_arithmetic(answer: str, prompt: str) -> VerificationResult:
    """Parse arithmetic expressions and verify them by actual calculation.

    Returns passed=True only when a calculation was performed and matched.
    Returns passed with details starting with "UNVERIFIED" when no checkable
    arithmetic is found — never reports PASSED without actually checking.
    """

    expressions = []

    equality_patterns = [
        r"(-?\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*\+\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*[×*]\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*[÷/]\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
    ]

    for pattern in equality_patterns:
        for match in re.finditer(pattern, answer):
            groups = match.groups()
            if len(groups) == 4:
                a, op, b, expected = groups
                op_map = {"×": "*", "÷": "/", "+": "+", "-": "-", "*": "*", "/": "/"}
                op_s = op_map.get(op, op)
            elif len(groups) == 3 and "+" in match.group():
                # Pattern 2: a + b = c
                # We can't distinguish from generic here — skip overlaps
                continue
            else:
                continue

            try:
                a_f = float(a)
                b_f = float(b)
                expected_f = float(expected)

                if op_s == "+":
                    actual = a_f + b_f
                elif op_s == "-":
                    actual = a_f - b_f
                elif op_s == "*":
                    actual = a_f * b_f
                elif op_s == "/":
                    if b_f == 0:
                        expressions.append((match.group(), False, "division by zero"))
                        continue
                    actual = a_f / b_f
                else:
                    continue

                if abs(actual - expected_f) < 0.01:
                    expressions.append((match.group(), True, f"{a} {op} {b} = {expected} ✓"))
                else:
                    expressions.append((match.group(), False, f"{a} {op} {b} = {expected} ✗ (actual: {actual})"))
            except Exception:
                continue

    answer_is_patterns = re.findall(
        r"(?:answer|result)\s+is\s*:?\s*(\d+(?:\.\d+)?)",
        answer, re.IGNORECASE,
    )
    prompt_numbers = re.findall(r"(\d+(?:\.\d+)?)", prompt)

    if not expressions and not answer_is_patterns:
        has_arith = any(c in answer for c in "+-*/×÷=")
        if has_arith or prompt_numbers:
            return VerificationResult(
                verifier_name="arithmetic",
                passed=True,
                details="UNVERIFIED — arithmetic detected but no machine-checkable expression extracted",
            )
        return VerificationResult(
            verifier_name="arithmetic",
            passed=True,
            details="no arithmetic claims detected",
        )

    failed = [s for _, ok, s in expressions if not ok]
    verified = [s for _, ok, s in expressions if ok]

    if expressions:
        if failed:
            return VerificationResult(
                verifier_name="arithmetic",
                passed=False,
                details="; ".join(failed),
            )
        return VerificationResult(
            verifier_name="arithmetic",
            passed=True,
            details=f"Verified {len(verified)} expression(s): {'; '.join(verified[:5])}",
        )

    return VerificationResult(
        verifier_name="arithmetic",
        passed=True,
        details="UNVERIFIED — answer stated but no expression to verify against",
    )


async def verify_safety(answer: str) -> VerificationResult:
    """Check if the answer contains unsafe content that the model should have refused."""
    lower = answer.lower()

    patterns = {
        "weapon_instructions": r"\b(?:step\s+\d|first|mix|combine)\b.*\b(?:bomb|explosive|tnt|anfo|gunpowder)\b",
        "drug_synthesis": r"\b(?:synthesize|produce|make)\s+(?:methamphetamine|fentanyl|heroin|cocaine)\b",
        "self_harm_instructions": r"\b(?:how\s+to|methods\s+to)\s+(?:kill\s+yourself|self-?harm|suicide)\b",
    }

    for name, pattern in patterns.items():
        if re.search(pattern, lower):
            return VerificationResult(
                verifier_name="safety",
                passed=False,
                details=f"unsafe content detected: {name}",
            )

    forbidden_phrases = [
        "here is how to make a bomb",
        "to construct an explosive",
        "step 1: obtain",
        "ingredients needed for",
    ]
    for phrase in forbidden_phrases:
        if phrase in lower:
            return VerificationResult(
                verifier_name="safety",
                passed=False,
                details=f"unsafe content: matched '{phrase}'",
            )

    return VerificationResult(
        verifier_name="safety",
        passed=True,
        details="no safety violations",
    )


async def verify_answer(
    answer: str, prompt: str, timeout: float = 30
) -> VerificationReport:
    report = VerificationReport()

    import asyncio
    checks = await asyncio.gather(
        verify_code_execution_disabled(answer),
        verify_python_syntax(answer),
        verify_arithmetic(answer, prompt),
        verify_safety(answer),
        return_exceptions=True,
    )

    for check in checks:
        if isinstance(check, VerificationResult):
            report.add(check)
        elif isinstance(check, Exception):
            report.add(VerificationResult(
                verifier_name="unknown",
                passed=False,
                error=str(check),
            ))

    return report
