import asyncio
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


async def verify_code_blocks(answer: str) -> VerificationResult:
    code_blocks = re.findall(r"```(\w+)?\n(.*?)```", answer, re.DOTALL)
    if not code_blocks:
        return VerificationResult(
            verifier_name="code_blocks",
            passed=True,
            details="no code blocks found",
        )

    python_blocks = [(lang, code) for lang, code in code_blocks if lang in ("python", "py", None)]
    failed = []
    for lang, code in python_blocks:
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode != 0:
                failed.append(f"exit {proc.returncode}: {stderr.decode()[:200]}")
        except TimeoutError:
            failed.append("timeout (10s)")
        except Exception as e:
            failed.append(f"error: {e}")

    if failed:
        return VerificationResult(
            verifier_name="code_blocks",
            passed=False,
            details="; ".join(failed),
        )
    return VerificationResult(
        verifier_name="code_blocks",
        passed=True,
        details=f"{len(python_blocks)} python block(s) ran successfully",
    )


async def verify_arithmetic(answer: str, prompt: str) -> VerificationResult:
    re.findall(r"[\d,]+\.?\d*", prompt)
    equalities = re.findall(r"=\s*[\d,]+\.?\d*", answer)
    if not equalities and not any(c in answer for c in "+-*/×÷"):
        return VerificationResult(
            verifier_name="arithmetic",
            passed=True,
            details="no arithmetic claims detected",
        )

    simple_calc = re.findall(r"answer\s+is\s+(\d[\d,]*\.?\d*)", answer, re.IGNORECASE)
    if simple_calc:
        return VerificationResult(
            verifier_name="arithmetic",
            passed=True,
            details=f"extracted answer: {simple_calc[0]}",
        )

    return VerificationResult(
        verifier_name="arithmetic",
        passed=True,
        details="arithmetic detected but no machine-checkable claim",
    )


async def verify_python_syntax(answer: str) -> VerificationResult:
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
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", f"import ast; ast.parse('''{code}''')",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode != 0:
                failed.append(f"block {i}: {stderr.decode()[:150]}")
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


async def verify_answer(
    answer: str, prompt: str, timeout: float = 30
) -> VerificationReport:
    report = VerificationReport()

    checks = await asyncio.gather(
        verify_code_blocks(answer),
        verify_python_syntax(answer),
        verify_arithmetic(answer, prompt),
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
