"""Phase 4.0.1 tests: DAG correctness and dependency semantics.

Proves strict node verification states (verified_pass | partial | unverified
| failed | blocked), acceptance-criteria checking, the expansion policy,
dependency blocking and failed-output isolation, DAG validation, and
fixed-pipeline fallback on invalid graphs.
"""
import os

import pytest

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
from nim_orchestrator.dag import (
    DAGNode,
    DagValidationError,
    build_dag,
    check_acceptance,
    execute_dag,
    execute_node,
    node_status,
    validate_dag,
)
from nim_orchestrator.router_client import ChatResult
from nim_orchestrator.task_compiler import Subtask, TaskSpec
from nim_orchestrator.verifiers.external_checks import VerificationReport

ROUTER_AVAILABLE = os.environ.get("NIM_ROUTER_AVAILABLE", "0") == "1"


class RecordingClient:
    """Records every user message; returns configurable content."""

    def __init__(self, content="17 * 23 = 391"):
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
# 1. DAG validation
# ============================================================


class TestDagValidation:
    def test_valid_spec_has_no_errors(self):
        assert validate_dag(_subtask_spec()) == []

    def test_duplicate_ids_reported(self):
        spec = TaskSpec(
            objective="x",
            subtasks=[
                Subtask(id="a", description="A"),
                Subtask(id="a", description="A again"),
            ],
        )
        errors = validate_dag(spec)
        assert any("duplicate" in e for e in errors)
        with pytest.raises(DagValidationError):
            build_dag(spec)

    def test_self_dependency_reported(self):
        spec = TaskSpec(
            objective="x",
            subtasks=[Subtask(id="a", description="A", depends_on=["a"])],
        )
        errors = validate_dag(spec)
        assert any("itself" in e for e in errors)
        with pytest.raises(DagValidationError):
            build_dag(spec)

    def test_unknown_dependency_reported_not_dropped(self):
        spec = TaskSpec(
            objective="x",
            subtasks=[
                Subtask(id="a", description="A", depends_on=["ghost"]),
                Subtask(id="b", description="B", depends_on=["a"]),
            ],
        )
        errors = validate_dag(spec)
        assert any("ghost" in e for e in errors)
        with pytest.raises(DagValidationError):
            build_dag(spec)

    def test_cycle_reported(self):
        spec = TaskSpec(
            objective="x",
            subtasks=[
                Subtask(id="a", description="A", depends_on=["b"]),
                Subtask(id="b", description="B", depends_on=["a"]),
            ],
        )
        errors = validate_dag(spec)
        assert any("cyclic" in e for e in errors)
        with pytest.raises(DagValidationError):
            build_dag(spec)

    def test_empty_subtasks_empty_dag(self):
        assert build_dag(TaskSpec(objective="x", subtasks=[])) == []


# ============================================================
# 2. Verification semantics
# ============================================================


class TestNodeStatus:
    def test_verified_arithmetic_is_verified_pass(self):
        spec = TaskSpec(
            objective="Compute 17 times 23",
            subtasks=[Subtask(id="s1", description="Compute 17 times 23", acceptance_criteria="The result must be 391")],
        )
        node = build_dag(spec)[0]
        report = VerificationReport()
        from nim_orchestrator.verifiers.external_checks import VerificationResult

        report.add(VerificationResult(verifier_name="arithmetic", status="pass", details="17 * 23 = 391 ✓"))
        acceptance = check_acceptance("17 * 23 = 391", [node.acceptance_criteria])
        status = node_status(report, acceptance, node.objective, node.acceptance_criteria)
        assert status == "verified_pass"
        assert acceptance[0].status == "verified"

    def test_unverified_answer_is_never_passed(self):
        """Regression: 'Compute 17 times 23' with 'The result is 42.' must not pass."""
        report = VerificationReport()
        from nim_orchestrator.verifiers.external_checks import VerificationResult

        report.add(VerificationResult(verifier_name="arithmetic", status="unverified", details="answer stated but no expression"))
        report.add(VerificationResult(verifier_name="code_execution", status="pass", details="no code blocks found"))
        acceptance = check_acceptance("The result is 42.", ["Output correct"])
        status = node_status(report, acceptance, "Compute 17 times 23", "Output correct")
        assert status != "verified_pass"
        assert status == "unverified"

    def test_acceptance_criteria_failure_is_failed(self):
        report = VerificationReport()
        from nim_orchestrator.verifiers.external_checks import VerificationResult

        # Arithmetic passes, but the acceptance criterion demands 999
        report.add(VerificationResult(verifier_name="arithmetic", status="pass", details="17 * 23 = 391 ✓"))
        acceptance = check_acceptance("17 * 23 = 391", ["The answer must be 999"])
        assert acceptance[0].status == "failed"
        assert node_status(report, acceptance, "Compute 17 times 23", "The answer must be 999") == "failed"

    def test_acceptance_criteria_verified_is_evidence(self):
        report = VerificationReport()
        from nim_orchestrator.verifiers.external_checks import VerificationResult

        report.add(VerificationResult(verifier_name="arithmetic", status="unverified", details="no expression"))
        acceptance = check_acceptance("The product is 391.", ["The result must be 391"])
        assert acceptance[0].status == "verified"
        # Evidence exists (criterion verified) but arithmetic is still
        # informatively unverified → partial, never passed outright
        status = node_status(report, acceptance, "Compute 17 times 23", "The result must be 391")
        assert status == "partial"
        assert status != "verified_pass"


# ============================================================
# 3. Node execution
# ============================================================


class TestNodeExecution:
    def _math_node(self):
        return DAGNode(id="s1", objective="Compute 17 times 23", acceptance_criteria="The result must be 391")

    def _unverifiable_node(self):
        # Non-numeric criteria → no deterministic acceptance check available
        return DAGNode(id="s1", objective="Compute 17 times 23", acceptance_criteria="Output correct")

    async def test_verified_answer_passes_with_one_attempt(self):
        ctx = _ctx_with_policy()
        client = RecordingClient(content="17 * 23 = 391")
        node = self._math_node()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context")

        assert node.status == "verified_pass"
        assert node.attempts == 1
        assert node.alternates_used == 0
        assert client.calls == 1

    async def test_unverified_answer_does_not_pass(self):
        """Regression: 'The result is 42.' for 'Compute 17 times 23' must not pass."""
        ctx = _ctx_with_policy()
        client = RecordingClient(content="The result is 42.")
        node = self._unverifiable_node()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context")

        assert node.status != "verified_pass"
        assert node.status == "unverified"
        # Medium risk → one alternate was tried
        assert node.attempts == 2
        assert node.alternates_used == 1

    async def test_low_risk_unverified_gets_no_alternate(self):
        ctx = _ctx_with_policy()
        client = RecordingClient(content="The result is 42.")
        node = self._unverifiable_node()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context", risk_level="low")

        assert node.status == "unverified"
        assert node.attempts == 1
        assert node.alternates_used == 0

    async def test_failure_triggers_exactly_one_alternate(self):
        ctx = _ctx_with_policy()
        client = RecordingClient(content="17 * 23 = 999")
        node = self._math_node()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context")

        assert node.status == "failed"
        assert node.attempts == 2
        assert node.alternates_used == 1
        assert client.calls == 2

    async def test_no_alternate_available_fails_after_primary(self):
        ctx = _ctx_with_policy(solvers=1)
        client = RecordingClient(content="17 * 23 = 999")
        node = self._math_node()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context")

        assert node.status == "failed"
        assert node.attempts == 1
        assert node.alternates_used == 0

    async def test_acceptance_criteria_failure_fails_node(self):
        ctx = _ctx_with_policy()
        client = RecordingClient(content="17 * 23 = 391")
        node = DAGNode(id="s1", objective="Compute 17 times 23", acceptance_criteria="The answer must be 999")
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context")

        assert node.status == "failed"
        assert any(a.status == "failed" for a in node.acceptance)

    async def test_budget_exhaustion_stops_node(self):
        ctx = _ctx_with_policy()
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=0))
        ctx.budget.start()
        client = RecordingClient(content="17 * 23 = 999")
        node = self._math_node()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1), "context")

        assert node.status == "failed"
        assert "budget" in node.error
        assert client.calls == 0


# ============================================================
# 4. Dependency semantics
# ============================================================


class TestDependencySemantics:
    def _dep_spec(self):
        return TaskSpec(
            objective="Do it",
            subtasks=[
                Subtask(id="s1", description="Find the input value", depends_on=[]),
                Subtask(id="s2", description="Compute output from input", depends_on=["s1"]),
                Subtask(id="s3", description="Independent check", depends_on=[]),
            ],
            recommended_route="complex",
            risk_level="medium",
            context="Original problem prompt",
        )

    async def test_dependent_node_receives_dependency_output(self):
        ctx = _ctx_with_policy(task_spec=self._dep_spec())
        client = RecordingClient(content="17 * 23 = 391")
        await execute_dag(client, ctx, DagConfig(max_alternates=1))

        s2_msgs = None
        for msgs in client.messages:
            user = msgs[1]["content"] if len(msgs) > 1 else ""
            if "Compute output from input" in user and "Acceptance criteria:" in user:
                s2_msgs = user
        assert s2_msgs is not None
        assert "--- s1 ---" in s2_msgs
        assert "17 * 23 = 391" in s2_msgs

    async def test_failed_dependency_blocks_dependent(self):
        ctx = _ctx_with_policy(task_spec=self._dep_spec())
        client = RecordingClient(content="17 * 23 = 999")
        await execute_dag(client, ctx, DagConfig(max_alternates=1))

        by_id = {n.id: n for n in ctx.dag_nodes}
        s1 = by_id["s1"]
        s2 = by_id["s2"]
        s3 = by_id["s3"]
        assert s1.status == "failed"
        assert s2.status == "blocked"
        assert "s1" in s2.error
        # Independent node still ran (not blocked) — content fails verification
        assert s3.status != "blocked"
        assert s3.status == "failed"

    async def test_unverified_dependency_blocks_dependent(self):
        ctx = _ctx_with_policy(task_spec=self._dep_spec())
        client = RecordingClient(content="The result is 42.")
        await execute_dag(client, ctx, DagConfig(max_alternates=1))

        by_id = {n.id: n for n in ctx.dag_nodes}
        assert by_id["s1"].status == "unverified"
        assert by_id["s2"].status == "blocked"

    async def test_failed_output_never_fed_forward(self):
        ctx = _ctx_with_policy(task_spec=self._dep_spec())
        client = RecordingClient(content="17 * 23 = 999")
        await execute_dag(client, ctx, DagConfig(max_alternates=1))

        # s2 was blocked → its node prompt never exists; and no node prompt
        # for any node contains a failed output as trusted context
        for msgs in client.messages:
            user = msgs[1]["content"] if len(msgs) > 1 else ""
            if "Acceptance criteria:" in user:
                assert "999" not in user

    async def test_acceptably_completed_dependency_fed(self):
        ctx = _ctx_with_policy(task_spec=self._dep_spec())
        client = RecordingClient(content="17 * 23 = 391")
        await execute_dag(client, ctx, DagConfig(max_alternates=1))

        by_id = {n.id: n for n in ctx.dag_nodes}
        assert by_id["s1"].status == "verified_pass"
        assert by_id["s2"].status == "verified_pass"
        assert by_id["s3"].status == "verified_pass"


# ============================================================
# 5. Verification plan assignment
# ============================================================


class TestVerificationPlan:
    def test_relevant_global_plan_items_assigned_to_nodes(self):
        spec = TaskSpec(
            objective="Build a system",
            verification_plan=[
                "Check the arithmetic in the computation is correct",
                "Ensure the design is scalable",
            ],
            subtasks=[
                Subtask(id="s1", description="Compute the totals", depends_on=[]),
                Subtask(id="s2", description="Design the architecture", depends_on=[]),
            ],
            recommended_route="complex",
            context="c",
        )
        nodes = build_dag(spec)
        by_id = {n.id: n for n in nodes}
        s1_plan = " ".join(by_id["s1"].verification_plan)
        s2_plan = " ".join(by_id["s2"].verification_plan)
        assert "arithmetic" in s1_plan
        assert "scalable" in s2_plan


# ============================================================
# 6. DAG budget limits
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
        # No acceptable outputs → nothing to synthesize (honest empty result)
        assert ctx.answer == ""
        by_id = {n.id: n for n in ctx.dag_nodes}
        assert by_id["s1"].status == "failed"
        assert by_id["s2"].status == "blocked"

    async def test_dag_reports_all_states(self):
        spec = _subtask_spec()
        ctx = _ctx_with_policy(task_spec=spec)
        client = RecordingClient(content="17 * 23 = 999")
        await execute_dag(client, ctx, DagConfig(max_alternates=1))

        trace = "\n".join(ctx.trace)
        assert "DAG done: 0 verified_pass, 0 partial, 0 unverified, 2 failed, 1 blocked" in trace
        assert "DAG final verification" in trace


# ============================================================
# 7. Policy gating
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
# 8. End-to-end API
# ============================================================


class TestDagAPI:
    def _dag_settings(self, dag_enabled=False):
        return Settings(
            router_base_url="http://mock",
            router_api_key="mock",
            candidates=[
                CandidateConfig(name="solver", model="m", system_prompt="S.", role="solver"),
                CandidateConfig(name="alternative_solver", model="m", system_prompt="A.", role="alternative_solver"),
            ],
            judge=JudgeConfig(model="m", system_prompt="J."),
            synthesizer=SynthesizerConfig(model="m", system_prompt="Syn."),
            dag=DagConfig(enabled=dag_enabled),
        )

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
                                '"subtasks": [{"id": "s1", "description": "Compute 17 times 23", "depends_on": [], '
                                '"acceptance_criteria": "The result must be 391"}, '
                                '{"id": "s2", "description": "State the product", "depends_on": ["s1"]}]}',
                        model="m",
                        latency_ms=100,
                        finish_reason="stop",
                    )
                return ChatResult(content="17 * 23 = 391", model="m", latency_ms=5, finish_reason="stop")

            async def close(self):
                pass

        client = CompilerThenDagClient()
        result = await handle_intelligence_request(
            client, self._dag_settings(), "Design a system with a data model", force_mode="dag"
        )

        assert result["mode"] == "dag"
        assert result["task_spec"] is not None
        assert result["answer"]
        assert result["verification"] is not None
        assert result["budget"]["model_calls"] == 4  # compiler + 2 nodes + synthesizer
        trace = result.get("pipeline_trace", [])
        assert any("DAG:" in t for t in trace)
        assert any("verified_pass" in t for t in trace)

    async def test_invalid_dag_falls_back_to_fixed_pipeline(self):
        """Acceptance: invalid DAG (cycle) traces the reason and falls back."""
        from nim_orchestrator.api import handle_intelligence_request

        class CyclicCompilerClient:
            def __init__(self):
                self.calls = 0

            async def chat(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content='{"objective": "Design a system", "risk_level": "medium", '
                                '"recommended_route": "complex", '
                                '"subtasks": [{"id": "a", "description": "A", "depends_on": ["b"]}, '
                                '{"id": "b", "description": "B", "depends_on": ["a"]}]}',
                        model="m",
                        latency_ms=100,
                        finish_reason="stop",
                    )
                return ChatResult(content="The answer is 42.", model="m", latency_ms=5, finish_reason="stop")

            async def close(self):
                pass

        result = await handle_intelligence_request(
            CyclicCompilerClient(), self._dag_settings(dag_enabled=True), "Design a system"
        )

        assert result["mode"] == "full"
        trace = result.get("pipeline_trace", [])
        assert any("DAG invalid" in t for t in trace)
        assert any("cyclic" in t for t in trace)
        assert any("Starting full pipeline" in t for t in trace)

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

        result = await handle_intelligence_request(
            CompilerClient(), self._dag_settings(dag_enabled=False), "Prove the theorem"
        )
        assert result["mode"] == "full"
        trace = result.get("pipeline_trace", [])
        assert not any("DAG" in t for t in trace)
