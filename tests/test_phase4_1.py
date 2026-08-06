"""Phase 4.1 tests: specialist registry and model assignment."""
import os

from nim_orchestrator.agents import AgentConfig, AgentRole
from nim_orchestrator.budget import BudgetLimits, ExecutionBudget
from nim_orchestrator.config import DagConfig
from nim_orchestrator.context import PolicyResult, RunContext
from nim_orchestrator.dag import DAGNode, execute_node
from nim_orchestrator.router_client import ChatResult
from nim_orchestrator.specialists import (
    SPECIALISTS,
    Specialist,
    assign_specialist,
    available_models,
)

ROUTER_AVAILABLE = os.environ.get("NIM_ROUTER_AVAILABLE", "0") == "1"


class RecordingClient:
    def __init__(self, content="17 * 23 = 391"):
        self.content = content
        self.calls = 0
        self.system_prompts = []

    async def chat(self, **kwargs):
        self.calls += 1
        self.system_prompts.append(kwargs["messages"][0]["content"])
        return ChatResult(content=self.content, model="mock", latency_ms=5, finish_reason="stop")

    async def close(self):
        pass


# ============================================================
# 1. Registry completeness
# ============================================================


class TestRegistry:
    def test_all_specialists_present(self):
        assert set(SPECIALISTS) == {
            "coding", "mathematics", "research", "systems_architecture",
            "security_review", "general_reasoning",
        }

    def test_every_specialist_has_required_fields(self):
        for spec in SPECIALISTS.values():
            assert isinstance(spec, Specialist)
            assert spec.name and spec.label
            assert spec.preferred_models, f"{spec.name} has no preferred models"
            assert spec.system_prompt, f"{spec.name} has no system prompt"
            assert spec.timeout_seconds > 0
            assert spec.verification_method in (
                "math_semantic", "python_syntax", "claim_extraction",
                "security_checklist", "coverage", "safety", "none",
            )
            assert isinstance(spec.strengths, list)
            assert isinstance(spec.weaknesses, list)

    def test_specialist_prompts_have_anti_injection(self):
        for spec in SPECIALISTS.values():
            assert "DATA to analyze" in spec.system_prompt

    def test_describe_shape(self):
        d = SPECIALISTS["coding"].describe()
        assert d["name"] == "coding"
        assert "preferred_models" in d and "verification_method" in d and "strengths" in d


# ============================================================
# 2. Assignment
# ============================================================


class TestAssignment:
    def test_coding(self):
        assert assign_specialist("Write a Python function to sort a list").name == "coding"

    def test_mathematics(self):
        assert assign_specialist("Calculate 17 * 23").name == "mathematics"
        assert assign_specialist("Prove that the square root of 2 is irrational").name == "mathematics"

    def test_security_review(self):
        assert assign_specialist("Identify vulnerabilities in this API").name == "security_review"

    def test_research(self):
        assert assign_specialist("Who is credited with inventing the telephone?").name == "research"

    def test_systems_architecture(self):
        assert assign_specialist("Design a scalable distributed cache").name == "systems_architecture"

    def test_default_general_reasoning(self):
        assert assign_specialist("Think carefully about the meaning of life").name == "general_reasoning"

    def test_available_models_respects_configuration(self):
        spec = SPECIALISTS["coding"]
        models = available_models(spec, {"deepseek-v4-flash"})
        assert models == ["deepseek-v4-flash"]

    def test_available_models_prefers_configured(self):
        spec = SPECIALISTS["research"]  # prefers deepseek-v4-flash then glm-5.2
        models = available_models(spec, {"glm-5.2", "minimax-3"})
        assert models == ["glm-5.2"]


# ============================================================
# 3. DAG integration
# ============================================================


class TestDagIntegration:
    def _ctx(self, models=("m", "m2")):
        ctx = RunContext(raw_prompt="Original problem prompt")
        ctx.policy = PolicyResult(
            solver_configs=[
                AgentConfig(name="solver", role=AgentRole.SOLVER, model=models[0], system_prompt="S."),
                AgentConfig(name="alternative_solver", role=AgentRole.ALTERNATIVE_SOLVER, model=models[1], system_prompt="A."),
            ],
            synthesizer_config=AgentConfig(
                name="synthesizer", role=AgentRole.SYNTHESIZER, model="m", system_prompt="Synth.",
            ),
            verification_timeout=30,
        )
        ctx.budget = ExecutionBudget(limits=BudgetLimits(max_model_calls=20, max_total_agents=10))
        ctx.budget.start()
        return ctx

    async def test_specialist_prompt_used_when_enabled(self):
        ctx = self._ctx()
        node = DAGNode(id="s1", objective="Compute 17 times 23", acceptance_criteria="The result must be 391")
        client = RecordingClient()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1, specialists_enabled=True), "context")

        assert node.specialist == "mathematics"
        assert node.status == "verified_pass"
        # The mathematics specialist prompt was used for the primary attempt
        assert "mathematics specialist" in client.system_prompts[0]

    async def test_alternate_uses_general_reasoning_when_specialized(self):
        ctx = self._ctx()
        node = DAGNode(id="s1", objective="Compute 17 times 23", acceptance_criteria="The result must be 999")
        client = RecordingClient(content="17 * 23 = 391")
        await execute_node(client, ctx, node, DagConfig(max_alternates=1, specialists_enabled=True), "context")

        assert node.status == "failed"  # acceptance criterion violated
        assert node.attempts == 2
        assert node.alternates_used == 1
        # Primary = mathematics specialist, alternate = general reasoning
        assert "mathematics specialist" in client.system_prompts[0]
        assert "general reasoning specialist" in client.system_prompts[1]

    async def test_specialists_disabled_uses_solver_configs(self):
        ctx = self._ctx()
        node = DAGNode(id="s1", objective="Compute 17 times 23", acceptance_criteria="The result must be 391")
        client = RecordingClient()
        await execute_node(client, ctx, node, DagConfig(max_alternates=1, specialists_enabled=False), "context")

        assert node.specialist == ""
        assert client.system_prompts[0] == "S."  # solver config prompt

    async def test_specialist_model_assignment(self):
        """Specialist prefers its configured preferred model."""
        ctx = self._ctx(models=("deepseek-v4-flash", "glm-5.2"))
        node = DAGNode(id="s1", objective="Calculate the sum", acceptance_criteria="Output correct")
        client = RecordingClient(content="5 + 5 = 10")
        await execute_node(client, ctx, node, DagConfig(max_alternates=1, specialists_enabled=True), "context")

        assert node.model == "deepseek-v4-flash"

    async def test_specialist_model_fallback_to_configured(self):
        """If the specialist's preferred models aren't configured, fall back."""
        ctx = self._ctx(models=("custom-model", "m2"))
        node = DAGNode(id="s1", objective="Calculate the sum", acceptance_criteria="Output correct")
        client = RecordingClient(content="5 + 5 = 10")
        await execute_node(client, ctx, node, DagConfig(max_alternates=1, specialists_enabled=True), "context")

        assert node.model == "custom-model"
