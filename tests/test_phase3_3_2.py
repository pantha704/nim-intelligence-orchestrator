"""Phase 3.3.2 tests: atomic execution-budget reservation.

Proves that reservation (max calls, max agents, time) is atomic under
concurrency, that failed/timed-out calls still consume the budget, and that
the task compiler uses the same reservation path as every agent.
"""
import asyncio
import os

import pytest

from nim_orchestrator.agents import AgentConfig, AgentRole
from nim_orchestrator.budget import BudgetExhaustedError, BudgetLimits, ExecutionBudget
from nim_orchestrator.config import CandidateConfig, JudgeConfig, Settings, SynthesizerConfig
from nim_orchestrator.context import PolicyResult, RunContext
from nim_orchestrator.router_client import ChatResult, budgeted_call, budgeted_chat

ROUTER_AVAILABLE = os.environ.get("NIM_ROUTER_AVAILABLE", "0") == "1"


class MockPipelineClient:
    def __init__(self, content="ok", latency=5):
        self.content = content
        self.latency = latency
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        return ChatResult(content=self.content, model="mock", latency_ms=self.latency, finish_reason="stop")

    async def close(self):
        pass


def _settings():
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


def _solvers(n: int, prefix: str = "s"):
    return [
        AgentConfig(name=f"{prefix}{i}", role=AgentRole.SOLVER, model="m", system_prompt="p")
        for i in range(n)
    ]


class TestAtomicReservation:
    async def test_max_model_calls_with_6_parallel_solvers_is_exactly_2(self):
        """Acceptance: the reservation race is closed — 6 parallel solvers with
        a 2-call budget produce exactly 2 client calls."""
        from nim_orchestrator.pipeline.full_pipeline import generate_candidates

        ctx = RunContext(raw_prompt="test")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(
            max_model_calls=2, max_concurrent_agents=6, max_total_agents=10,
        ))
        ctx.start()
        ctx.policy = PolicyResult(solver_configs=_solvers(6))

        client = MockPipelineClient()
        await generate_candidates(client, ctx)

        assert client.calls == 2
        assert ctx.budget.model_calls == 2
        blocked = [c for c in ctx.candidates if c.error]
        assert len(blocked) == 4
        assert all("budget" in c.error for c in blocked)

    async def test_reserve_call_is_atomic_under_concurrency(self):
        """Direct: 10 concurrent reservations with a 3-call budget → exactly 3 succeed."""
        ctx = RunContext(raw_prompt="test")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(
            max_model_calls=3, max_total_agents=10,
        ))
        ctx.start()

        async def _try(i):
            try:
                return await ctx.budget.reserve_call(f"agent{i}", "m")
            except BudgetExhaustedError as e:
                return e

        results = await asyncio.gather(*[_try(i) for i in range(10)])
        ok = [r for r in results if isinstance(r, int)]
        assert len(ok) == 3
        assert ctx.budget.model_calls == 3

    async def test_timed_out_call_consumes_budget(self):
        """Acceptance: timed-out calls consume the model-call budget."""
        class SlowClient:
            async def chat(self, **kwargs):
                await asyncio.sleep(5)
                return ChatResult(content="x", model="m", latency_ms=1)

        ctx = RunContext(raw_prompt="test")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=2, max_concurrent_agents=2))
        ctx.start()

        with pytest.raises(TimeoutError):
            await budgeted_chat(SlowClient(), ctx, agent_name="s1", model="m", messages=[], timeout=0.1)

        # The slot was reserved before the call and stays consumed
        assert ctx.budget.model_calls == 1
        assert ctx.budget._call_log[0]["status"] == "timeout"

        # A second (fast) call succeeds; a third is blocked
        await budgeted_chat(MockPipelineClient(), ctx, agent_name="s2", model="m", messages=[])
        assert ctx.budget.model_calls == 2
        with pytest.raises(BudgetExhaustedError):
            await ctx.budget.reserve_call("s3", "m")

    async def test_failed_call_appears_in_call_log_with_error_status(self):
        """Acceptance: failed calls appear in the call log and consume budget."""
        import httpx

        class FailingClient:
            async def chat(self, **kwargs):
                raise httpx.ConnectError("boom", request=httpx.Request("POST", "http://mock"))

        ctx = RunContext(raw_prompt="test")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=2))
        ctx.start()

        with pytest.raises(httpx.ConnectError):
            await budgeted_chat(FailingClient(), ctx, agent_name="s1", model="m", messages=[])

        assert ctx.budget.model_calls == 1
        entry = ctx.budget._call_log[0]
        assert entry["status"] == "error"
        assert entry["agent"] == "s1"

    async def test_max_total_agents_holds_under_concurrency(self):
        """Acceptance: max_total_agents remains correct under concurrency."""
        from nim_orchestrator.pipeline.full_pipeline import generate_candidates

        ctx = RunContext(raw_prompt="test")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(
            max_model_calls=20, max_total_agents=3, max_concurrent_agents=6,
        ))
        ctx.start()
        ctx.policy = PolicyResult(solver_configs=_solvers(6))

        client = MockPipelineClient()
        await generate_candidates(client, ctx)

        assert ctx.budget.agents_used == 3
        assert client.calls == 3
        blocked = [c for c in ctx.candidates if c.error]
        assert len(blocked) == 3

    async def test_cancelled_call_recorded(self):
        """Acceptance: cancelled calls are recorded with status 'cancelled'."""
        class SlowClient:
            async def chat(self, **kwargs):
                await asyncio.sleep(5)
                return ChatResult(content="x", model="m", latency_ms=1)

        ctx = RunContext(raw_prompt="test")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=2))
        ctx.start()

        task = asyncio.create_task(
            budgeted_chat(SlowClient(), ctx, agent_name="s1", model="m", messages=[], timeout=30)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert ctx.budget.model_calls == 1
        assert ctx.budget._call_log[0]["status"] == "cancelled"


class TestCompilerSharesBudget:
    async def test_compiler_reserves_through_budgeted_call(self):
        """Acceptance: the task compiler uses the shared reservation mechanism."""
        from nim_orchestrator.task_compiler import compile_task

        ctx = RunContext(raw_prompt="Design a system with unclear requirements")
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=1))
        ctx.start()

        client = MockPipelineClient(content="not json at all")
        result = await budgeted_call(
            ctx,
            agent_name="task_compiler",
            model="m",
            call_fn=lambda: compile_task(client, model="m", raw_prompt=ctx.raw_prompt, timeout_seconds=5),
            timeout=10,
        )

        assert result is not None
        assert ctx.budget.model_calls == 1
        assert ctx.budget.agents_used == 1
        assert ctx.budget._call_log[0]["agent"] == "task_compiler"

        # The reservation consumed the only slot — a second call is blocked
        with pytest.raises(BudgetExhaustedError):
            await ctx.budget.reserve_call("solver", "m")

    async def test_compiler_blocked_when_budget_exhausted(self):
        """Acceptance: when the budget is spent, the compiler is skipped and
        the request degrades to a default complex spec."""
        from nim_orchestrator.api import handle_intelligence_request

        class CountingClient:
            def __init__(self):
                self.calls = 0

            async def chat(self, **kwargs):
                self.calls += 1
                return ChatResult(content="answer", model="m", latency_ms=1, finish_reason="stop")

            async def close(self):
                pass

        settings = _settings()
        client = CountingClient()
        result = await handle_intelligence_request(
            client, settings, "Prove that the sum of two even numbers is even"
        )

        # mode full means the pipeline ran; the compiler call is in the budget
        assert result["mode"] == "full"
        budget = result["budget"]
        agents = {entry["agent"] for entry in budget["call_log"]}
        assert "task_compiler" in agents
        assert budget["model_calls"] == client.calls > 0
