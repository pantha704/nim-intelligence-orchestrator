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
from .sandbox import run_in_sandbox
from .semantic_checks import extract_factual_claims, verify_math_claims


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
    """verifier_code_sandbox bound to the tool registry — a disabled sandbox
    tool degrades the check to unverified via ToolUnavailableError."""

    def verifier_code_sandbox(answer: str, timeout_seconds: float = 5.0, **kwargs) -> tuple[str, str, str]:
        blocks = _extract_python_blocks(answer)
        if not blocks:
            return "unverified", "no code blocks to run", ""
        results = []
        for i, code in enumerate(blocks):
            results.append(tools.call("sandbox", code=code, timeout_seconds=timeout_seconds))
        if all(r.ok for r in results):
            ev = "; ".join(f"block {i}: exit 0" for i, r in enumerate(results))
            return "pass", f"{len(results)} block(s) ran in sandbox", ev
        bad = [f"block {i}: {r.status}" for i, r in enumerate(results) if not r.ok]
        return "fail", "sandbox execution failed", "; ".join(bad[:5])

    return verifier_code_sandbox


def _make_test_runner_verifier(tools: ToolRegistry):
    """test runner bound to the sandbox tool (tests execute inside it)."""

    def verifier_test_runner(answer: str, timeout_seconds: float = 5.0, **kwargs) -> tuple[str, str, str]:
        blocks = _extract_python_blocks(answer)
        if not blocks:
            return "pass", "no code blocks to test — nothing to run", ""
        tests = [code for code in blocks if "def test_" in code]
        if not tests:
            return "pass", "no test_* functions found — nothing to run", ""
        wrapper = (
            "import sys\n"
            "results = []\n"
            + "\n".join(f"exec(open('block{i}.py').read(), globals())" for i in range(len(tests)))
            + "\n"
            + "\n".join(
                f"try:\n    test_{i}(); results.append(({i}, 'pass'))\n"
                f"except Exception as e:\n    results.append(({i}, 'fail:' + repr(e)))"
                for i in range(len(tests))
            )
            + "\nprint(results)\n"
        )
        sandbox_code = "\n\n".join(tests) + "\n\n" + wrapper
        r = tools.call("sandbox", code=sandbox_code, timeout_seconds=timeout_seconds)
        if not r.ok:
            return "fail", f"test run failed: {r.status} {r.stderr[:200]}", ""
        return "pass", r.stdout.strip()[:300], ""

    return verifier_test_runner


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
    tools.register("sandbox", "isolated code sandbox (network-off, limited)", run_in_sandbox,
                   disabled_by_default=not sandbox_enabled)
    tools.register("math_evaluator", "safe AST expression evaluator", None,
                   disabled_by_default=False)
    tools.register("python_syntax", "AST python syntax checker", None)
    tools.register("test_runner", "runs test_* functions inside the sandbox", None,
                   disabled_by_default=not sandbox_enabled)
    tools.register("claim_extractor", "sentence-level claim extraction", None)
    tools.register("citation_source", "external citation/source lookup — NOT IMPLEMENTED", None,
                   disabled_by_default=True)
    tools.register("security_checklist", "structured security dimension coverage", None)
    tools.register("coverage_checker", "requirement/constraint coverage", None)

    verifiers = VerifierRegistry(tools)
    verifiers.register("python_syntax", verifier_python_syntax, tool_id="python_syntax",
                       description="parse Python code blocks with AST")
    verifiers.register("code_sandbox", _make_sandbox_verifier(tools), tool_id="sandbox",
                       description="run code blocks in the isolated sandbox")
    verifiers.register("test_runner", _make_test_runner_verifier(tools), tool_id="test_runner",
                       description="execute test_* functions in the sandbox")
    verifiers.register("math_semantic", lambda answer, **kw: verify_math_claims(answer),
                       tool_id="math_evaluator", description="negation-aware equation verification")
    verifiers.register("claim_extraction", verifier_claim_extraction, tool_id="claim_extractor",
                       description="extract claims; external confirmation unavailable")
    verifiers.register("security_checklist", verifier_security_checklist, tool_id="security_checklist",
                       description="structured security dimensions + safety scan")
    verifiers.register("coverage", verifier_coverage, tool_id="coverage_checker",
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
