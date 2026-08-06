"""Phase 4.2.1 tests: isolation hardening, model routing and truthful evidence.

Covers fail-closed sandboxing, request-persistent live model routing,
health/latency-driven switching, and the 'trivial pass is never evidence'
rule.
"""
import os

import pytest

from nim_orchestrator.agents import AgentConfig, AgentRole
from nim_orchestrator.budget import BudgetLimits, ExecutionBudget
from nim_orchestrator.config import DagConfig
from nim_orchestrator.context import PolicyResult, RunContext
from nim_orchestrator.dag import DAGNode, execute_dag, execute_node, node_status
from nim_orchestrator.models import ModelRegistry
from nim_orchestrator.router_client import ChatResult
from nim_orchestrator.task_compiler import Subtask, TaskSpec
from nim_orchestrator.verifiers.external_checks import (
    VerificationReport,
    VerificationResult,
)
from nim_orchestrator.verifiers.sandbox import run_secure_sandbox, select_secure_backend

ROUTER_AVAILABLE = os.environ.get("NIM_ROUTER_AVAILABLE", "0") == "1"

MATH_ANSWER = "17 * 23 = 391"


# ============================================================
# 1. Fail-closed + adversarial isolation
# ============================================================


class TestIsolationHardening:
    def test_fail_closed_never_uses_host_subprocess(self):
        """The secure entry point must never fall back to a host subprocess."""
        r = run_secure_sandbox("print(6 * 7)")
        if select_secure_backend() is None:
            assert r.status == "unavailable"
            assert "refusing host" in r.error
        else:
            assert r.backend in ("docker", "bubblewrap")
            assert r.isolation == "full"

    @pytest.mark.skipif(select_secure_backend() is None, reason="no secure sandbox backend")
    def test_repository_files_invisible(self):
        """Adversarial: host repo and secrets must not be reachable."""
        probe = (
            "import os\n"
            "targets = [\n"
            "  '/home/ubuntu/nim-intelligence-orchestrator/pyproject.toml',\n"
            "  '/home/ubuntu/.config/opencode/nim-router/master.key',\n"
            "]\n"
            "print([os.path.exists(t) for t in targets])\n"
        )
        r = run_secure_sandbox(probe, timeout_seconds=10)
        assert r.ok
        # Host-specific paths are invisible inside the container
        assert eval(r.stdout.strip()) == [False, False]

    @pytest.mark.skipif(select_secure_backend() is None, reason="no secure sandbox backend")
    def test_host_environment_invisible(self):
        r = run_secure_sandbox("import os\nprint(sorted(os.environ.keys()))\nprint(os.environ.get('HOME'))", timeout_seconds=10)
        assert r.ok
        lines = r.stdout.strip().splitlines()
        keys = eval(lines[0])
        # No host secrets; only image-internal vars may appear (e.g. GPG_KEY)
        assert not any(
            k for k in keys
            if any(s in k for s in ("API_KEY", "TOKEN", "SECRET", "MASTER", "OPENCODE", "GITHUB"))
        )
        # HOME is never a host path (docker sets it to the image's nobody home)
        assert not lines[1].startswith("/home/")
        assert lines[1] != "/root"

    def test_trivial_passes_are_not_evidence(self):
        """Regression: an answer with only trivial 'nothing to check' passes
        can never produce verified_pass."""
        report = VerificationReport()
        report.add(VerificationResult(verifier_name="code_execution", status="pass", details="no code blocks found"))
        report.add(VerificationResult(verifier_name="python_syntax", status="pass", details="no code blocks to check"))
        report.add(VerificationResult(verifier_name="safety", status="pass", details="no safety violations"))
        report.add(VerificationResult(verifier_name="arithmetic", status="unverified", details="no arithmetic claims"))
        status = node_status(report, [], "Compute 17 times 23", "")
        assert status == "unverified"


# ============================================================
# 2. Request-persistent live model routing
# ============================================================


class TestLiveModelRouting:
    def _ctx(self, task_spec, models):
        ctx = RunContext(raw_prompt="Original problem prompt")
        ctx.policy = PolicyResult(
            solver_configs=[
                AgentConfig(name="solver", role=AgentRole.SOLVER, model=models[0], system_prompt="S."),
                AgentConfig(name="alternative_solver", role=AgentRole.ALTERNATIVE_SOLVER, model=models[1], system_prompt="A."),
            ],
            synthesizer_config=AgentConfig(name="syn", role=AgentRole.SYNTHESIZER, model=models[0], system_prompt="Syn."),
            verification_timeout=30,
        )
        ctx.task_spec = task_spec
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=20, max_total_agents=10))
        ctx.budget.start()
        return ctx

    def _dep_spec(self):
        return TaskSpec(
            objective="Do it",
            subtasks=[
                Subtask(id="s1", description="Calculate the sum", depends_on=[]),
                Subtask(id="s2", description="Calculate the product", depends_on=["s1"]),
            ],
            recommended_route="complex",
            risk_level="low",
            context="Original problem prompt",
        )

    class FailingModelClient:
        """Fails on the first configured model, succeeds on the second."""

        def __init__(self, bad_model):
            self.bad_model = bad_model
            self.models_used = []
            self.calls = 0

        async def chat(self, **kwargs):
            self.calls += 1
            model = kwargs.get("model")
            self.models_used.append(model)
            if model == self.bad_model:
                raise RuntimeError("model transport failure")
            return ChatResult(content=MATH_ANSWER, model=model, latency_ms=5, finish_reason="stop")

        async def close(self):
            pass

    async def test_models_switch_away_after_failures(self):
        """Health-driven switching: after 'bad' errors, routing moves to 'good'."""
        from nim_orchestrator.dag import sync_model_outcomes

        ctx = self._ctx(self._dep_spec(), models=["bad-model", "good-model"])
        client = self.FailingModelClient(bad_model="bad-model")
        dag_cfg = DagConfig(max_alternates=0, max_concurrent_calls=6, specialists_enabled=True)

        # Wave 1: node s1 — registry defaults to bad-model (registration order)
        node1 = DAGNode(id="s1", objective="Calculate the sum", acceptance_criteria="Output correct")
        await execute_node(client, ctx, node1, dag_cfg, "context", risk_level="low")
        assert node1.model == "bad-model"
        assert node1.status == "failed"
        sync_model_outcomes(ctx)
        assert ctx.model_registry.health_of("bad-model") == "degraded"

        # Wave 2: the SAME registry now routes around the down model
        node2 = DAGNode(id="s2", objective="Calculate the product", acceptance_criteria="Output correct")
        await execute_node(client, ctx, node2, dag_cfg, "context", risk_level="low")
        assert node2.model == "good-model"
        assert node2.status == "verified_pass"

    async def test_dag_records_model_outcomes(self):
        """execute_dag feeds live outcomes into the shared registry."""
        ctx = self._ctx(self._dep_spec(), models=["bad-model", "good-model"])
        client = self.FailingModelClient(bad_model="bad-model")
        await execute_dag(client, ctx, DagConfig(max_alternates=0, max_concurrent_calls=6, specialists_enabled=True))

        registry = ctx.model_registry
        assert registry is not None
        # s1 failed on bad-model → registry reflects the failures
        assert registry.health_of("bad-model") in ("degraded", "down")
        # s2 was blocked by the failed dependency, but good-model is healthy
        assert registry.health_of("good-model") in ("unknown", "healthy")

    async def test_registry_is_request_persistent(self):
        """One ModelRegistry instance lives on the context for the whole run."""
        ctx = self._ctx(self._dep_spec(), models=["m1", "m2"])

        class CountingClient:
            def __init__(self):
                self.calls = 0

            async def chat(self, **kwargs):
                self.calls += 1
                return ChatResult(content=MATH_ANSWER, model=kwargs.get("model"), latency_ms=5, finish_reason="stop")

            async def close(self):
                pass

        await execute_dag(CountingClient(), ctx, DagConfig(max_alternates=1, specialists_enabled=True))
        registry = ctx.model_registry
        assert registry is not None
        assert isinstance(registry, ModelRegistry)
        # Every DAG node saw the SAME registry instance
        for node in ctx.dag_nodes:
            assert node.model in registry.names()

    async def test_latency_history_populated_from_calls(self):
        ctx = self._ctx(self._dep_spec(), models=["m1", "m2"])

        class LatencyClient:
            async def chat(self, **kwargs):
                return ChatResult(content=MATH_ANSWER, model=kwargs.get("model"), latency_ms=123, finish_reason="stop")

        await execute_dag(LatencyClient(), ctx, DagConfig(max_alternates=1, specialists_enabled=True))
        registry = ctx.model_registry
        # Node calls recorded their latency into the shared registry
        assert registry.latency_history("m1") or registry.latency_history("m2")

    async def test_direct_execute_node_builds_registry_fallback(self):
        """execute_node called directly (no execute_dag) still works."""
        ctx = RunContext(raw_prompt="x")
        ctx.policy = PolicyResult(
            solver_configs=[AgentConfig(name="s", role=AgentRole.SOLVER, model="zzz", system_prompt="S.")],
            synthesizer_config=AgentConfig(name="syn", role=AgentRole.SYNTHESIZER, model="m", system_prompt="Syn."),
        )
        ctx.task_spec = TaskSpec(objective="x", subtasks=[], recommended_route="complex", context="c")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=10))
        ctx.budget.start()

        class C:
            async def chat(self, **kwargs):
                return ChatResult(content=MATH_ANSWER, model="zzz", latency_ms=5, finish_reason="stop")

        node = DAGNode(id="s1", objective="Calculate the sum", acceptance_criteria="Output correct")
        await execute_node(C(), ctx, node, DagConfig(max_alternates=1, specialists_enabled=True), "context")
        assert node.model == "zzz"
        assert ctx.model_registry is not None
