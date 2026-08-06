"""Tool and verifier registries with provenance.

Specialists reference registered tool/verifier IDs, not descriptive strings.
Every verification result carries provenance: verifier/tool used, input
checked, outcome, latency and an evidence summary.
"""
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from .external_checks import safety_scan
from .math_eval import ExpressionError, safe_eval_expression
from .sandbox import run_secure_sandbox
from .semantic_checks import extract_claims, extract_factual_claims


class ToolUnavailableError(RuntimeError):
    """Raised when a tool is not registered, disabled, or lacks an implementation."""


@dataclass
class ToolEntry:
    tool_id: str
    description: str
    fn: Callable | None
    disabled_by_default: bool = False
    enabled: bool = True

    def available(self) -> bool:
        return self.fn is not None and self.enabled


class ToolRegistry:
    """Typed registry of executable tools (sandbox, evaluators, checkers)."""

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}

    def register(self, tool_id: str, description: str, fn: Callable | None = None,
                 disabled_by_default: bool = False) -> None:
        self._tools[tool_id] = ToolEntry(
            tool_id=tool_id, description=description, fn=fn,
            disabled_by_default=disabled_by_default,
            enabled=not disabled_by_default,
        )

    def call(self, tool_id: str, **kwargs):
        entry = self._tools.get(tool_id)
        if entry is None:
            raise ToolUnavailableError(f"tool '{tool_id}' is not registered")
        if not entry.available():
            raise ToolUnavailableError(f"tool '{tool_id}' is unavailable: {entry.description}")
        return entry.fn(**kwargs)

    def available(self, tool_id: str) -> bool:
        entry = self._tools.get(tool_id)
        return bool(entry and entry.available())

    def enable(self, tool_id: str, enabled: bool = True) -> None:
        if tool_id in self._tools:
            self._tools[tool_id].enabled = enabled

    def ids(self) -> list[str]:
        return sorted(self._tools)


@dataclass
class VerifiedCheck:
    """A verification result with full provenance."""
    verifier_id: str
    tool_id: str = ""
    status: str = "unverified"  # pass | fail | unverified | error
    input_checked: str = ""
    outcome: str = ""
    latency_ms: float = 0.0
    evidence: str = ""
    details: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    def to_dict(self) -> dict:
        return {
            "verifier": self.verifier_id,
            "tool": self.tool_id,
            "status": self.status,
            "input_checked": self.input_checked,
            "outcome": self.outcome,
            "latency_ms": round(self.latency_ms, 1),
            "evidence": self.evidence,
        }


class VerifierRegistry:
    """Typed registry of verifiers, each bound to a tool when applicable."""

    def __init__(self, tools: ToolRegistry | None = None):
        self._tools = tools or ToolRegistry()
        self._verifiers: dict[str, Callable] = {}
        self._meta: dict[str, dict] = {}

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    def register(self, verifier_id: str, fn: Callable, tool_id: str = "",
                 description: str = "") -> None:
        self._verifiers[verifier_id] = fn
        self._meta[verifier_id] = {"tool_id": tool_id, "description": description}

    def run(self, verifier_id: str, input_checked: str = "", **kwargs) -> VerifiedCheck:
        """Run a verifier, catching tool-unavailable degradation and errors."""
        if verifier_id not in self._verifiers:
            return VerifiedCheck(
                verifier_id=verifier_id, status="error",
                input_checked=input_checked,
                outcome="verifier not registered",
                evidence="",
            )
        meta = self._meta[verifier_id]
        t0 = time.monotonic()
        try:
            status, evidence, details = self._verifiers[verifier_id](**kwargs)
        except ToolUnavailableError as e:
            return VerifiedCheck(
                verifier_id=verifier_id, tool_id=meta["tool_id"],
                status="unverified", input_checked=input_checked,
                outcome="tool unavailable — degraded",
                latency_ms=(time.monotonic() - t0) * 1000,
                evidence=str(e),
            )
        except Exception as e:
            return VerifiedCheck(
                verifier_id=verifier_id, tool_id=meta["tool_id"],
                status="error", input_checked=input_checked,
                outcome=f"verifier error: {type(e).__name__}",
                latency_ms=(time.monotonic() - t0) * 1000,
                evidence=str(e)[:300],
            )
        return VerifiedCheck(
            verifier_id=verifier_id, tool_id=meta["tool_id"],
            status=status, input_checked=input_checked,
            outcome=evidence, latency_ms=(time.monotonic() - t0) * 1000,
            evidence=evidence, details=details,
        )

    def ids(self) -> list[str]:
        return sorted(self._verifiers)


# ============================================================
# Specialist verifier implementations (return status, evidence, details)
# ============================================================


def _extract_python_blocks(answer: str) -> list[str]:
    return re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, re.DOTALL)


def verifier_python_syntax(answer: str, **kwargs) -> tuple[str, str, str]:
    import ast as _ast

    blocks = _extract_python_blocks(answer)
    if not blocks:
        return "unverified", "no code blocks in answer", ""
    bad = []
    for i, code in enumerate(blocks):
        try:
            _ast.parse(code)
        except SyntaxError as e:
            bad.append(f"block {i}: {e}")
    if bad:
        return "fail", "; ".join(bad), ""
    return "pass", f"{len(blocks)} block(s) parse successfully", ""


def _make_sandbox_verifier(tools: ToolRegistry):
    """verifier_code_sandbox bound to the tool registry — a missing secure
    backend or disabled tool degrades the check to unverified (fail closed)."""

    def verifier_code_sandbox(answer: str, timeout_seconds: float = 5.0, **kwargs) -> tuple[str, str, str]:
        blocks = _extract_python_blocks(answer)
        if not blocks:
            return "unverified", "no code blocks to run", ""
        results = []
        for i, code in enumerate(blocks):
            r = tools.call("sandbox", code=code, timeout_seconds=timeout_seconds)
            if r.status == "unavailable":
                raise ToolUnavailableError(r.error)
            results.append(r)
        if all(r.ok for r in results):
            ev = "; ".join(f"block {i}: exit 0" for i, r in enumerate(results))
            return "pass", f"{len(results)} block(s) ran in sandbox ({results[0].backend})", ev
        bad = [f"block {i}: {r.status}" for i, r in enumerate(results) if not r.ok]
        return "fail", "sandbox execution failed", "; ".join(bad[:5])

    return verifier_code_sandbox


def _make_test_runner_verifier(tools: ToolRegistry):
    """Test runner bound to the sandbox tool: supplied code blocks are written
    into real sandbox files, real test_* callables are discovered and invoked,
    and passed/failed/collected counts are reported.

    No tests found → UNVERIFIED (never positive evidence).
    """

    def verifier_test_runner(answer: str, timeout_seconds: float = 5.0, **kwargs) -> tuple[str, str, str]:
        blocks = _extract_python_blocks(answer)
        if not blocks:
            return "unverified", "no code blocks to test", ""
        if not any("def test_" in code for code in blocks):
            return "unverified", "no test_* functions found — nothing was tested", ""

        # EVERY Python block is written into the sandbox — implementation and
        # test blocks alike. They load into ONE shared namespace in order, so
        # tests can call the implementation from other blocks. Tests are
        # discovered only after all modules load.
        n = len(blocks)
        lines = [
            "collected = passed = failed = 0",
            "failures = []",
            "import_failures = []",
        ]
        for i in range(n):
            lines.extend([
                "try:",
                f"    exec(open('block{i}.py').read(), ns)",
                "except Exception as e:",
                f"    import_failures.append(({i}, repr(e)))",
            ])
        lines.extend([
            "for name, obj in sorted(ns.items()):",
            "    if name.startswith('test_') and callable(obj):",
            "        collected += 1",
            "        try:",
            "            obj(); passed += 1",
            "        except Exception as e:",
            "            failed += 1; failures.append(f'{name}: {e!r}')",
            "print('COLLECTED', collected, 'PASSED', passed, 'FAILED', failed)",
            "print('IMPORT_FAILURES', import_failures)",
            "print('FAILURES', failures)",
        ])
        runner = "ns = {}\n" + "\n".join(lines)

        files = {f"block{i}.py": code for i, code in enumerate(blocks)}
        r = tools.call("sandbox", code=runner, files=files, timeout_seconds=timeout_seconds)
        if r.status == "unavailable":
            raise ToolUnavailableError(r.error)
        if not r.ok:
            return "fail", f"test run failed: {r.status} {r.stderr[:200]}", ""

        import re as _re

        m = _re.search(r"COLLECTED (\d+) PASSED (\d+) FAILED (\d+)", r.stdout)
        if not m:
            return "fail", f"test runner produced no summary: {r.stdout[:200]}", ""
        collected, passed, failed = (int(v) for v in m.groups())
        # Import/setup failures are FAIL — never a misleading zero-test result
        if "IMPORT_FAILURES []" not in r.stdout:
            return "fail", "test module(s) failed to import", r.stdout.strip()[:300]
        if collected == 0:
            return "unverified", "no test_* callables discovered — nothing was tested", ""
        if failed > 0:
            return "fail", f"{failed}/{collected} tests failed", r.stdout.strip()[:300]
        return "pass", f"{passed}/{collected} tests passed", r.stdout.strip()[:300]

    return verifier_test_runner


def _make_math_semantic_verifier(tools: ToolRegistry):
    """Math verification where every computation goes through the registered
    safe evaluator tool — provenance is truthful."""

    def verifier_math_semantic(answer: str, **kwargs) -> tuple[str, str, str]:
        claims = extract_claims(answer)

        def _evaluate(left: float, operator: str, right: float) -> float:
            expr = f"{left:g} {operator} {right:g}"
            return float(tools.call("math_evaluator", expr=expr))

        affirmative_errors: list[str] = []
        affirmative_correct: list[str] = []
        for claim in claims:
            if claim.negated:
                continue
            for eq in claim.equalities:
                try:
                    actual = _evaluate(eq.left, eq.operator, eq.right)
                except ExpressionError as e:
                    affirmative_errors.append(f"{eq.text} ({e})")
                    continue
                if abs(actual - eq.expected) < 0.01:
                    affirmative_correct.append(eq.text)
                else:
                    affirmative_errors.append(f"{eq.text} (actual {actual:g})")

        if affirmative_errors:
            return "fail", f"{len(affirmative_errors)} wrong affirmative equation(s)", "; ".join(affirmative_errors[:5])
        if affirmative_correct:
            return "pass", f"{len(affirmative_correct)} verified equation(s)", "; ".join(affirmative_correct[:5])

        if any(c.negated and c.equalities for c in claims):
            return "unverified", "equations appear only in negated sentences — not counted as evidence", ""
        return "unverified", "no affirmative checkable equations in answer", ""

    return verifier_math_semantic


def verifier_claim_extraction(answer: str, **kwargs) -> tuple[str, str, str]:
    claims = extract_factual_claims(answer)
    if not claims:
        return "unverified", "no extractable claims", ""
    ev = f"extracted {len(claims)} claim(s): " + "; ".join(c["text"][:60] for c in claims[:3])
    return "unverified", ev, "citation source unavailable — claims cannot be externally confirmed"


def verifier_security_checklist(answer: str, **kwargs) -> tuple[str, str, str]:
    safety = safety_scan(answer)
    if safety.status == "fail":
        return "fail", "unsafe content detected", safety.details

    mandatory = {
        "authentication": r"\b(auth\w*|login|session|token|oauth)\b",
        "input_validation": r"\b(input validation|sanitiz\w*|parameterized|prepared statement|escape)\b",
        "least_privilege": r"\b(least privilege|principle of least|role[- ]based|rbac|permissions)\b",
        "encryption": r"\b(encrypt\w*|tls|https|hashing|hash)\b",
        "logging": r"\b(log\w*|monitor\w*|audit)\b",
    }
    low = answer.lower()
    covered = {name for name, pat in mandatory.items() if re.search(pat, low)}
    missing = [name for name in mandatory if name not in covered]
    if not missing:
        return "pass", "all security dimensions covered", ", ".join(covered)
    if len(covered) >= 3:
        return "unverified", f"partial security coverage: missing {missing}", ""
    return "unverified", f"security coverage insufficient: missing {missing}", ""


def verifier_coverage(answer: str, requirements: list[str] | None = None, **kwargs) -> tuple[str, str, str]:
    reqs = requirements or []
    if not reqs:
        return "unverified", "no requirements supplied to check coverage", ""
    low = answer.lower()
    covered = 0
    uncovered = []
    for req in reqs:
        keywords = [w for w in re.findall(r"[a-z0-9]+", req.lower()) if len(w) > 3]
        if keywords and any(k in low for k in keywords):
            covered += 1
        else:
            uncovered.append(req)
    if covered == len(reqs):
        return "pass", f"all {len(reqs)} requirements addressed", ""
    return "unverified", f"{covered}/{len(reqs)} requirements addressed; missing: {uncovered}", ""


def build_default_registry(sandbox_enabled: bool = False) -> VerifierRegistry:
    """Build the default tool + verifier registries.

    The sandbox and test runner are disabled by default; enabling the sandbox
    also enables the test runner (it runs inside the sandbox).
    """
    tools = ToolRegistry()
    tools.register("sandbox", "secure isolated sandbox (docker/bwrap; fail-closed)", run_secure_sandbox,
                   disabled_by_default=not sandbox_enabled)
    tools.register("math_evaluator", "safe AST expression evaluator", safe_eval_expression)
    tools.register("python_syntax", "AST python syntax checker", None)
    tools.register("test_runner", "runs test_* functions inside the secure sandbox", None,
                   disabled_by_default=not sandbox_enabled)
    tools.register("claim_extractor", "sentence-level claim extraction", None)
    tools.register("citation_source", "external citation/source lookup — NOT IMPLEMENTED", None,
                   disabled_by_default=True)
    tools.register("security_checklist", "structured security dimension coverage", None)
    tools.register("coverage_checker", "requirement/constraint coverage", None)

    verifiers = VerifierRegistry(tools)
    # In-process verifiers: no external tool is invoked → tool_id stays ""
    verifiers.register("python_syntax", verifier_python_syntax,
                       description="parse Python code blocks with AST")
    verifiers.register("code_sandbox", _make_sandbox_verifier(tools), tool_id="sandbox",
                       description="run code blocks in the secure sandbox (fail-closed)")
    verifiers.register("test_runner", _make_test_runner_verifier(tools), tool_id="sandbox",
                       description="execute test_* functions inside the secure sandbox")
    # math_semantic evaluates every equation through the registered evaluator tool
    verifiers.register("math_semantic", _make_math_semantic_verifier(tools), tool_id="math_evaluator",
                       description="negation-aware equation verification via safe evaluator")
    verifiers.register("claim_extraction", verifier_claim_extraction,
                       description="extract claims; external confirmation unavailable")
    verifiers.register("security_checklist", verifier_security_checklist,
                       description="structured security dimensions + safety scan")
    verifiers.register("coverage", verifier_coverage,
                       description="requirement/constraint coverage")
    return verifiers


def run_specialist_verification(
    answer: str,
    verification_method: str,
    tool_ids: list[str],
    *,
    sandbox_enabled: bool = False,
    requirements: list[str] | None = None,
    input_checked: str = "",
) -> list[VerifiedCheck]:
    """Run a specialist's verifier + bound tools against an answer.

    Returns a list of provenance-carrying checks. Unavailable tools degrade
    to unverified — they never crash the node.
    """
    verifiers = build_default_registry(sandbox_enabled=sandbox_enabled)
    checks: list[VerifiedCheck] = []

    if verification_method and verification_method != "none":
        checks.append(verifiers.run(
            verification_method, answer=answer,
            requirements=requirements or [],
            input_checked=input_checked,
        ))

    tool_verifier_map = {
        "sandbox": "code_sandbox",
        "test_runner": "test_runner",
        "python_syntax": "python_syntax",
    }
    for tool_id in tool_ids:
        verifier_id = tool_verifier_map.get(tool_id)
        if verifier_id and verifier_id != verification_method:
            checks.append(verifiers.run(
                verifier_id, answer=answer, input_checked=input_checked,
            ))
    return checks
