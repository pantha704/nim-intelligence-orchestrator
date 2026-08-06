"""Phase 4.0 tests: adaptive specialist DAG (MVP).

Proves topological ordering, node execution with primary/alternate agents,
dependency context propagation, budget limits, policy gating, and the
end-to-end DAG path through the API.
"""
import os

from nim_orchestrator.agents import AgentConfig, AgentRole
from nim_orchestrator.budget import BudgetLimits, ExecutionBudget
from nim_orchestrator.config import (
    CandidateConfig,
    DagConfig,
    JudgeConfig,
    Settings,
    SynthesizerConfig,
)
from nim_orchestrator.context import PolicyResult, RunContext
from nim_orchestrator.dag import DAGNode, build_dag, execute_dag, execute_node
from nim_orchestrator.router_client import ChatResult
from nim_orchestrator.task_compiler import Subtask, TaskSpec

ROUTER_AVAILABLE = os.environ.get("NIM_ROUTER_AVAILABLE", "0") == "1"


class RecordingClient:
    """Records every user message; returns configurable content."""

    def __init__(self, content="The answer is 42."):
        self.content = content
        self.calls = 0
        self.messages = []

    async def chat(self, **kwargs):
        self.calls += 1
        self.messages.append(kwargs.get("messages", []))
        return ChatResult(content=self.content, model="mock", latency_ms=5, finish_reason="stop")

    async def close(self):
        pass


def _ctx_with_policy(task_spec=None, solvers=2):
    ctx = RunContext(raw_prompt="Original problem prompt")
    ctx.policy = PolicyResult(
        solver_configs=[
            AgentConfig(name="solver", role=AgentRole.SOLVER, model="m", system_prompt="S."),
            AgentConfig(name="alternative_solver", role=AgentRole.ALTERNATIVE_SOLVER, model="m", system_prompt="A."),
        ][:solvers],
        synthesizer_config=AgentConfig(
            name="synthesizer", role=AgentRole.SYNTHESIZER, model="m", system_prompt="Synth.",
        ),
        verification_timeout=30,
    )
    ctx.task_spec = task_spec
    ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=20, max_total_agents=10))
    ctx.budget.start()
    return ctx


def _subtask_spec():
    return TaskSpec(
        objective="Compute the result",
        subtasks=[
            Subtask(id="s1", description="Find the input value", depends_on=[], acceptance_criteria="Input found"),
            Subtask(id="s2", description="Compute output from input", depends_on=["s1"], acceptance_criteria="Output correct"),
            Subtask(id="s3", description="Independent check", depends_on=[], acceptance_criteria="Check done"),
        ],
        recommended_route="complex",
        risk_level="medium",
        context="Original problem prompt",
    )


# ============================================================
# 1. DAG construction
# ============================================================


class TestBuildDag:
    def test_topological_order_respects_dependencies(self):
        spec = _subtask_spec()
        nodes = build_dag(spec)
        order = [n.id for n in nodes]
        # s2 depends on s1 — s1 must come first
        assert order.index("s1") < order.index("s2")
        # s3 is independent — anywhere, but must be present
        assert set(order) == {"s1", "s2", "s3"}

    def test_dependencies_preserved_on_nodes(self):
        nodes = build_dag(_subtask_spec())
        by_id = {n.id: n for n in nodes}
        assert by_id["s2"].depends_on == ["s1"]
        assert by_id["s1"].depends_on == []

    def test_unknown_dependency_dropped(self):
        spec = TaskSpec(
            objective="x",
            subtasks=[
                Subtask(id="a", description="A", depends_on=["ghost"]),
                Subtask(id="b", description="B", depends_on=["a"]),
            ],
        )
        nodes = build_dag(spec)
        assert nodes[0].id == "a"
        assert nodes[0].depends_on == []

    def test_cycle_does_not_hang(self):
        spec = TaskSpec(
            objective="x",
            subtasks=[
                Subtask(id="a", description="A", depends_on=["b"]),
                Subtask(id="b", description="B", depends_on=["a"]),
            ],
        )
        nodes = build_dag(spec)
        assert len(nodes) == 2

    def test_empty_subtasks_empty_dag(self):
        assert build_dag(TaskSpec(objective="x", subtasks=[])) == []


# ============================================================
# 2. Node execution
# ============================================================


class TestNodeExecution:
    def _node(self):
        return DAGNode(id="s1", objective="Compute 17 times 23", acceptance_criteria="Correct product")

    async def test_primary_success_no_alternate(self):
        ctx = _ctx_with_policy()
        client = RecordingClient(content="The result is 42.")
        node = self._node()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context")

        assert node.status == "passed"
        assert node.attempts == 1
        assert node.alternates_used == 0
        assert client.calls == 1
        assert node.result == "The result is 42."

    async def test_failure_triggers_exactly_one_alternate(self):
        ctx = _ctx_with_policy()
        # Wrong arithmetic always fails verification
        client = RecordingClient(content="17 * 23 = 999")
        node = self._node()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context")

        assert node.status == "failed"
        assert node.attempts == 2
        assert node.alternates_used == 1
        assert client.calls == 2

    async def test_no_alternate_available_fails_after_primary(self):
        ctx = _ctx_with_policy(solvers=1)
        client = RecordingClient(content="17 * 23 = 999")
        node = self._node()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context")

        assert node.status == "failed"
        assert node.attempts == 1
        assert node.alternates_used == 0
        assert client.calls == 1

    async def test_budget_exhaustion_stops_node(self):
        ctx = _ctx_with_policy()
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=0))
        ctx.budget.start()
        client = RecordingClient(content="17 * 23 = 999")
        node = self._node()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context")

        assert node.status == "failed"
        assert "budget" in node.error
        assert client.calls == 0


# ============================================================
# 3. Dependency context propagation
# ============================================================


class TestDependencyContext:
    async def test_dependent_node_receives_dependency_output(self):
        spec = TaskSpec(
            objective="Do it",
            subtasks=[
                Subtask(id="s1", description="Find the input value", depends_on=[]),
                Subtask(id="s2", description="Compute output from input", depends_on=["s1"]),
            ],
            recommended_route="complex",
            context="Original problem prompt",
        )
        ctx = _ctx_with_policy(task_spec=spec)
        client = RecordingClient(content="The answer is 42.")
        await execute_dag(client, ctx, DagConfig(max_alternates=1))

        # s1's output must appear in s2's node prompt (not the synthesizer's)
        s2_msgs = None
        for msgs in client.messages:
            user = msgs[1]["content"] if len(msgs) > 1 else ""
            if "Compute output from input" in user and "Acceptance criteria:" in user:
                s2_msgs = user
        assert s2_msgs is not None
        assert "--- s1 ---" in s2_msgs
        assert "The answer is 42." in s2_msgs

    async def test_all_nodes_executed_in_order(self):
        spec = _subtask_spec()
        ctx = _ctx_with_policy(task_spec=spec)
        client = RecordingClient(content="The answer is 42.")
        await execute_dag(client, ctx, DagConfig(max_alternates=1))

        # 3 nodes + 1 synthesizer = 4 calls
        assert client.calls == 4
        assert ctx.mode == "dag"
        assert ctx.answer
        assert len(ctx.candidates) == 3
        assert any("dag:s1" == c.name for c in ctx.candidates)


# ============================================================
# 4. DAG budget limits
# ============================================================


class TestDagBudget:
    async def test_dag_respects_max_model_calls(self):
        spec = _subtask_spec()
        ctx = _ctx_with_policy(task_spec=spec)
        client = RecordingClient(content="17 * 23 = 999")  # always fails verification
        await execute_dag(client, ctx, DagConfig(max_alternates=1, max_model_calls=2))

        # s1 primary + s1 alternate = 2 calls; everything else blocked
        assert client.calls == 2
        assert ctx.budget.model_calls == 2
        assert ctx.mode == "dag"
        # Synthesis degraded to raw outputs (budget exhausted)
        assert ctx.answer

    async def test_dag_reports_failed_nodes(self):
        spec = _subtask_spec()
        ctx = _ctx_with_policy(task_spec=spec)
        client = RecordingClient(content="17 * 23 = 999")
        await execute_dag(client, ctx, DagConfig(max_alternates=1))

        trace = "\n".join(ctx.trace)
        assert "DAG done: 0 passed, 3 failed" in trace
        assert "DAG final verification" in trace


# ============================================================
# 5. Policy gating
# ============================================================


class TestPolicyGating:
    def _settings(self, dag_enabled=False):
        return Settings(
            router_base_url="http://mock",
            router_api_key="mock",
            candidates=[CandidateConfig(name="solver", model="m", system_prompt="S.", role="solver")],
            judge=JudgeConfig(model="m", system_prompt="J."),
            synthesizer=SynthesizerConfig(model="m", system_prompt="Syn."),
            dag=DagConfig(enabled=dag_enabled),
        )

    def test_dag_disabled_by_default_uses_full_pipeline(self):
        from nim_orchestrator.policy import PolicyEngine

        spec = _subtask_spec()
        result = PolicyEngine(self._settings()).decide("Design a system", task_spec=spec)
        assert result.action == "full"
        assert result.use_dag is False

    def test_dag_enabled_with_subtasks_uses_dag(self):
        from nim_orchestrator.policy import PolicyEngine

        spec = _subtask_spec()
        result = PolicyEngine(self._settings(dag_enabled=True)).decide("Design a system", task_spec=spec)
        assert result.action == "dag"
        assert result.use_dag is True

    def test_dag_enabled_without_subtasks_uses_full(self):
        from nim_orchestrator.policy import PolicyEngine

        spec = TaskSpec(objective="x", recommended_route="complex", context="c")
        result = PolicyEngine(self._settings(dag_enabled=True)).decide("Design a system", task_spec=spec)
        assert result.action == "full"
        assert result.use_dag is False

    def test_dag_enabled_direct_route_stays_speculative(self):
        from nim_orchestrator.policy import PolicyEngine

        spec = _subtask_spec()
        spec.recommended_route = "direct"
        result = PolicyEngine(self._settings(dag_enabled=True)).decide("What is 2+2?", task_spec=spec)
        assert result.action == "speculative"
        assert result.use_dag is False

    def test_force_dag_mode(self):
        from nim_orchestrator.policy import PolicyEngine

        result = PolicyEngine(self._settings()).decide("Design a system", force_mode="dag")
        assert result.action == "dag"
        assert result.use_dag is True


# ============================================================
# 6. End-to-end API
# ============================================================


class TestDagAPI:
    async def test_forced_dag_mode_end_to_end(self):
        from nim_orchestrator.api import handle_intelligence_request

        class CompilerThenDagClient:
            def __init__(self):
                self.calls = 0

            async def chat(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content='{"objective": "Design a system", "risk_level": "medium", '
                                '"recommended_route": "complex", '
                                '"subtasks": [{"id": "s1", "description": "Design the architecture", "depends_on": []}, '
                                '{"id": "s2", "description": "Specify the data model", "depends_on": ["s1"]}]}',
                        model="m",
                        latency_ms=100,
                        finish_reason="stop",
                    )
                return ChatResult(content="The answer is 42.", model="m", latency_ms=5, finish_reason="stop")

            async def close(self):
                pass

        settings = Settings(
            router_base_url="http://mock",
            router_api_key="mock",
            candidates=[
                CandidateConfig(name="solver", model="m", system_prompt="S.", role="solver"),
                CandidateConfig(name="alternative_solver", model="m", system_prompt="A.", role="alternative_solver"),
            ],
            judge=JudgeConfig(model="m", system_prompt="J."),
            synthesizer=SynthesizerConfig(model="m", system_prompt="Syn."),
        )

        client = CompilerThenDagClient()
        result = await handle_intelligence_request(
            client, settings, "Design a system with a data model", force_mode="dag"
        )

        assert result["mode"] == "dag"
        assert result["task_spec"] is not None
        assert result["answer"]
        assert result["verification"] is not None
        assert result["budget"]["model_calls"] == 4  # compiler + 2 nodes + synthesizer
        trace = result.get("pipeline_trace", [])
        assert any("DAG:" in t for t in trace)
        assert any("DAG done" in t for t in trace)

    async def test_dag_stays_disabled_without_force(self):
        """Config gate: dag disabled → even with subtasks, mode stays full."""
        from nim_orchestrator.api import handle_intelligence_request

        class CompilerClient:
            async def chat(self, **kwargs):
                return ChatResult(
                    content='{"objective": "Prove it", "risk_level": "medium", "recommended_route": "complex", '
                            '"subtasks": [{"id": "s1", "description": "Step one", "depends_on": []}]}',
                    model="m",
                    latency_ms=100,
                    finish_reason="stop",
                )

            async def close(self):
                pass

        settings = Settings(
            router_base_url="http://mock",
            router_api_key="mock",
            candidates=[CandidateConfig(name="solver", model="m", system_prompt="S.", role="solver")],
            judge=JudgeConfig(model="m", system_prompt="J."),
            synthesizer=SynthesizerConfig(model="m", system_prompt="Syn."),
            dag=DagConfig(enabled=False),
        )

        result = await handle_intelligence_request(
            CompilerClient(), settings, "Prove the theorem"
        )
        assert result["mode"] == "full"
        trace = result.get("pipeline_trace", [])
        assert not any("DAG" in t for t in trace)
