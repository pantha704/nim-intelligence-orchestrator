"""Phase 3.3.1 tests: RunContext threading, budget enforcement, concurrency
limits, single task classifier, no name-based role inference, and
clarification/direct/full API paths."""
import asyncio
import os
from pathlib import Path

from nim_orchestrator.agents import AgentConfig, AgentRole
from nim_orchestrator.budget import BudgetLimits, ExecutionBudget
from nim_orchestrator.config import CandidateConfig, JudgeConfig, Settings, SynthesizerConfig
from nim_orchestrator.context import PolicyResult, RunContext
from nim_orchestrator.router_client import ChatResult

ROUTER_AVAILABLE = os.environ.get("NIM_ROUTER_AVAILABLE", "0") == "1"


class MockPipelineClient:
    """Records every chat call; returns generic content."""

    def __init__(self, content="The answer is 42.", latency=10):
        self.content = content
        self.latency = latency
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        return ChatResult(content=self.content, model="mock", latency_ms=self.latency, finish_reason="stop")

    async def close(self):
        pass


def _pipeline_settings():
    return Settings(
        router_base_url="http://mock",
        router_api_key="mock",
        candidates=[
            CandidateConfig(name="solver", model="m", system_prompt="S.", role="solver"),
            CandidateConfig(name="alternative_solver", model="m", system_prompt="A.", role="alternative_solver"),
            CandidateConfig(name="adversarial_critic", model="m", system_prompt="C.", role="critic"),
            CandidateConfig(name="evidence_verifier", model="m", system_prompt="V.", role="evidence_verifier"),
            CandidateConfig(name="devil_advocate", model="m", system_prompt="D.", role="devils_advocate"),
        ],
        judge=JudgeConfig(model="m", system_prompt="J."),
        synthesizer=SynthesizerConfig(model="m", system_prompt="Syn."),
    )


# ============================================================
# 1. Same RunContext reaches every stage
# ============================================================


class TestRunContextThreading:
    async def test_run_full_pipeline_mutates_one_context(self):
        """Acceptance: a single RunContext carries all pipeline state."""
        from nim_orchestrator.pipeline.full_pipeline import run_full_pipeline
        from nim_orchestrator.policy import PolicyEngine

        settings = _pipeline_settings()
        engine = PolicyEngine(settings)
        policy = engine.decide("Prove that the sum of two even numbers is even")

        ctx = RunContext(raw_prompt="What is the answer?")
        ctx.start()
        ctx.policy = policy

        client = MockPipelineClient()
        await run_full_pipeline(client, ctx)

        # Every stage's output lives on the same ctx object
        assert ctx.candidates, "candidates not set on ctx"
        assert len(ctx.candidates) == 2
        assert ctx.anon is not None and len(ctx.anon.shuffled) == 2
        assert ctx.critique.get("critic") is not None or ctx.critique == {}
        assert ctx.clustering is not None
        assert ctx.judge_result is not None
        assert ctx.verification is not None
        assert ctx.winner is not None
        assert ctx.answer, "answer not set on ctx"
        assert ctx.mode == "full"

        # Budget recorded every model call on the same ctx
        assert ctx.budget.model_calls == client.calls > 0

        # Trace proves every stage ran on the ctx
        trace_text = "\n".join(ctx.trace)
        for marker in ("Starting full pipeline", "Generated", "Critique", "Clustering", "Judge", "Verification", "Synthesis"):
            assert marker in trace_text, f"missing trace marker: {marker}"

    async def test_api_response_comes_from_context(self):
        """API returns a response built from the RunContext — budget included."""
        from nim_orchestrator.api import handle_intelligence_request

        settings = _pipeline_settings()
        client = MockPipelineClient()
        result = await handle_intelligence_request(
            client, settings, "Prove that the sum of two even numbers is always even."
        )

        assert result["mode"] == "full"
        assert "budget" in result
        assert result["budget"]["model_calls"] > 0
        assert "call_log" in result["budget"]
        assert result["verification"] is not None
        assert result["task_spec"] is not None


# ============================================================
# 2. Budget exhaustion prevents additional model calls
# ============================================================


class TestBudgetEnforcement:
    async def test_budget_exhaustion_stops_calls_and_degrades(self):
        """Acceptance: budget exhaustion prevents additional model calls."""
        from nim_orchestrator.pipeline.full_pipeline import run_full_pipeline
        from nim_orchestrator.policy import PolicyEngine

        settings = _pipeline_settings()
        policy = PolicyEngine(settings).decide("Prove that the sum of two even numbers is even")

        ctx = RunContext(raw_prompt="What is the answer?")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=2))
        ctx.start()
        ctx.policy = policy

        client = MockPipelineClient()
        await run_full_pipeline(client, ctx)

        # Only 2 model calls were made despite 2 solvers + 3 reviewers + judge + synth
        assert client.calls == 2
        assert ctx.budget.model_calls == 2
        assert not ctx.budget.can_call()

        # Degraded gracefully: reviewers/judge/synth all blocked, answer still produced
        assert ctx.answer
        assert ctx.mode == "full"
        assert any("budget" in t.lower() or "skipped" in t.lower() for t in ctx.trace)

    async def test_concurrency_limit_enforced(self):
        """Acceptance: max_concurrent_agents is enforced via semaphore."""
        from nim_orchestrator.pipeline.full_pipeline import generate_candidates

        class SlowMockClient:
            def __init__(self):
                self.active = 0
                self.max_active = 0

            async def chat(self, **kwargs):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                await asyncio.sleep(0.02)
                self.active -= 1
                return ChatResult(content="ok", model="m", latency_ms=10, finish_reason="stop")

            async def close(self):
                pass

        ctx = RunContext(raw_prompt="test")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=20, max_concurrent_agents=2, max_total_agents=10))
        ctx.start()
        ctx.policy = PolicyResult(solver_configs=[
            AgentConfig(name=f"s{i}", role=AgentRole.SOLVER, model="m", system_prompt="p")
            for i in range(6)
        ])

        client = SlowMockClient()
        await generate_candidates(client, ctx)

        assert client.max_active <= 2
        assert len(ctx.candidates) == 6

    async def test_speculative_call_respects_budget(self):
        """Acceptance: the speculative quick call is budget-enforced."""
        from nim_orchestrator.policy import PolicyEngine

        settings = _pipeline_settings()
        engine = PolicyEngine(settings)
        ctx = RunContext(raw_prompt="What is the capital of France?")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=1))
        ctx.start()
        # Budget already spent — the speculative call must be blocked
        ctx.budget.record_call("prior", "m", 1)
        ctx.policy = engine.decide(ctx.raw_prompt)

        client = MockPipelineClient()
        accepted = await engine.execute_speculative(ctx, client)

        assert accepted is False  # budget exhausted → escalate
        assert ctx.budget.model_calls == 1
        assert not ctx.budget.can_call()

    async def test_bypass_escalation_carries_agent_configs(self):
        """Regression: bypass decisions must include solver/reviewer configs so
        escalation to the full pipeline can actually run."""
        from nim_orchestrator.policy import PolicyEngine

        settings = _pipeline_settings()
        engine = PolicyEngine(settings)
        ctx = RunContext(raw_prompt="What is the capital of France?")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=0))
        ctx.start()
        ctx.policy = engine.decide(ctx.raw_prompt)

        assert ctx.policy.action == "speculative"
        assert len(ctx.policy.solver_configs) >= 2
        assert len(ctx.policy.reviewer_configs) >= 3

        # Escalation path must be executable: speculative blocked by budget,
        # then the full pipeline can still run with populated agents.
        client = MockPipelineClient()
        accepted = await engine.execute_speculative(ctx, client)
        assert accepted is False
        assert len(ctx.policy.solver_configs) > 0


# ============================================================
# 3. Single task classifier
# ============================================================


class TestSingleTaskClassifier:
    def test_speculative_router_uses_policy_classifier(self):
        """Acceptance: only one task classifier exists in the codebase."""
        import nim_orchestrator.speculative_router as sr
        from nim_orchestrator.policy import classify_task_type

        # The duplicated classifier is gone
        assert not hasattr(sr, "_detect_task_type")
        # The router imports the SAME function object from policy
        assert sr.classify_task_type is classify_task_type

    def test_no_task_classifier_duplication_in_source(self):
        import re

        src_dir = Path(__file__).parents[1] / "src" / "nim_orchestrator"
        for py in src_dir.rglob("*.py"):
            text = py.read_text()
            # No second implementation of task classification by keywords
            assert not re.search(r"def _detect_task_type", text)
            # speculative_router must import, not redefine
            if py.name == "speculative_router.py":
                assert "from .policy import classify_task_type" in text


# ============================================================
# 4. No name-based role inference anywhere
# ============================================================


class TestNoNameRoleInference:
    def test_no_infer_role_in_source(self):
        import re

        src_dir = Path(__file__).parents[1] / "src" / "nim_orchestrator"
        for py in src_dir.rglob("*.py"):
            assert not re.search(r"infer_role|_infer_role", py.read_text()), (
                f"{py.name} still contains role-name inference"
            )

    def test_policy_uses_explicit_role_only(self):
        from nim_orchestrator.policy import PolicyEngine

        settings = _pipeline_settings()
        result = PolicyEngine(settings).decide("Write a function to sort a list")
        for cfg in result.solver_configs + result.reviewer_configs:
            assert isinstance(cfg.role, AgentRole)


# ============================================================
# 5. Clarification path
# ============================================================


class TestClarificationPath:
    async def test_clarification_returns_needs_clarification(self):
        """Acceptance: high-impact ambiguity asks one question."""
        from nim_orchestrator.api import handle_intelligence_request

        class CompilerOnlyMock:
            async def chat(self, **kwargs):
                return ChatResult(
                    content='{"objective": "Do it", "risk_level": "high", "recommended_route": "complex", '
                            '"ambiguities": [{"question": "Which format?", "impact": "high", "resolution": "ask"}]}',
                    model="m",
                    latency_ms=100,
                    finish_reason="stop",
                )

            async def close(self):
                pass

        settings = _pipeline_settings()
        result = await handle_intelligence_request(
            CompilerOnlyMock(), settings, "Build me something with unclear requirements"
        )

        assert result["mode"] == "needs_clarification"
        assert result["clarification_question"] == "Which format?"
        assert result["task_spec"] is not None

    async def test_direct_path_still_works(self):
        """Acceptance: direct path works through the new API."""
        from nim_orchestrator.api import handle_intelligence_request

        settings = _pipeline_settings()
        client = MockPipelineClient(content="Paris")
        result = await handle_intelligence_request(client, settings, "What is the capital of France?")

        assert result["mode"] == "direct"
        assert "Paris" in result["answer"]
        assert result["budget"]["model_calls"] >= 1

    async def test_forced_single_still_works(self):
        """Acceptance: forced single mode works through the new API."""
        from nim_orchestrator.api import handle_intelligence_request

        settings = _pipeline_settings()
        client = MockPipelineClient(content="42")
        result = await handle_intelligence_request(client, settings, "What is 2+2?", force_mode="single")

        assert result["mode"] == "single"
        assert result["answer"] == "42"
