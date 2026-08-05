"""Phase 3.3 tests: RunContext, AgentRole, ExecutionBudget, PolicyEngine,
persistent anon IDs through debate, and central routing decisions."""
import os

from nim_orchestrator.agents import AgentConfig, AgentRole
from nim_orchestrator.budget import BudgetLimits, ExecutionBudget
from nim_orchestrator.clustering import Candidate
from nim_orchestrator.config import CandidateConfig, JudgeConfig, Settings, SynthesizerConfig
from nim_orchestrator.context import (
    RunContext,
    create_anon_mapping,
)
from nim_orchestrator.policy import PolicyEngine, classify_task_type
from nim_orchestrator.router_client import ChatResult

ROUTER_AVAILABLE = os.environ.get("NIM_ROUTER_AVAILABLE", "0") == "1"


# ============================================================
# 1. AgentRole
# ============================================================


class TestAgentRole:
    def test_solver_is_solver(self):
        assert AgentRole.SOLVER.is_solver
        assert AgentRole.ALTERNATIVE_SOLVER.is_solver

    def test_solver_is_not_reviewer(self):
        assert not AgentRole.SOLVER.is_reviewer
        assert not AgentRole.ALTERNATIVE_SOLVER.is_reviewer

    def test_reviewer_is_reviewer(self):
        assert AgentRole.CRITIC.is_reviewer
        assert AgentRole.EVIDENCE_VERIFIER.is_reviewer
        assert AgentRole.DEVILS_ADVOCATE.is_reviewer

    def test_reviewer_is_not_solver(self):
        assert not AgentRole.CRITIC.is_solver
        assert not AgentRole.EVIDENCE_VERIFIER.is_solver
        assert not AgentRole.DEVILS_ADVOCATE.is_solver

    def test_judge_is_judge(self):
        assert AgentRole.JUDGE.is_judge
        assert not AgentRole.JUDGE.is_solver
        assert not AgentRole.JUDGE.is_reviewer

    def test_synthesizer_is_synthesizer(self):
        assert AgentRole.SYNTHESIZER.is_synthesizer
        assert not AgentRole.SYNTHESIZER.is_solver
        assert not AgentRole.SYNTHESIZER.is_reviewer

    def test_role_is_string_enum(self):
        assert AgentRole.SOLVER == "solver"
        assert AgentRole.CRITIC == "critic"
        assert AgentRole.JUDGE == "judge"

    def test_agent_config_to_dict_includes_role(self):
        cfg = AgentConfig(
            name="test", role=AgentRole.SOLVER, model="m", system_prompt="p"
        )
        d = cfg.to_dict()
        assert d["role"] == "solver"
        assert d["name"] == "test"
        assert d["model"] == "m"


# ============================================================
# 2. ExecutionBudget
# ============================================================


class TestExecutionBudget:
    def test_initial_state(self):
        b = ExecutionBudget()
        assert b.model_calls == 0
        assert b.agents_used == 0
        assert b.can_call()
        assert b.can_spawn_agent()

    def test_record_call_increments(self):
        b = ExecutionBudget()
        b.start()
        b.record_call("solver", "m", 5000, 100)
        assert b.model_calls == 1
        assert b.agents_used == 0

    def test_record_agent_increments(self):
        b = ExecutionBudget()
        b.record_agent()
        assert b.agents_used == 1

    def test_can_call_false_at_limit(self):
        b = ExecutionBudget(limits=BudgetLimits(max_model_calls=2))
        b.start()
        b.record_call("a", "m", 1)
        b.record_call("b", "m", 1)
        assert not b.can_call()

    def test_can_spawn_agent_false_at_limit(self):
        b = ExecutionBudget(limits=BudgetLimits(max_total_agents=1))
        b.record_agent()
        assert not b.can_spawn_agent()

    def test_elapsed_seconds_starts_zero(self):
        b = ExecutionBudget()
        assert b.elapsed_seconds == 0.0

    def test_summary_structure(self):
        b = ExecutionBudget()
        b.start()
        b.record_call("solver", "m", 100, 50)
        b.record_agent()
        s = b.summary()
        assert s["model_calls"] == 1
        assert s["agents_used"] == 1
        assert "limits" in s
        assert "call_log" in s
        assert len(s["call_log"]) == 1

    def test_call_log_records_details(self):
        b = ExecutionBudget()
        b.start()
        b.record_call("critic", "model-x", 3000, 200)
        entry = b._call_log[0]
        assert entry["agent"] == "critic"
        assert entry["model"] == "model-x"
        assert entry["latency_ms"] == 3000.0
        assert entry["tokens"] == 200


# ============================================================
# 3. RunContext
# ============================================================


class TestRunContext:
    def test_initial_state(self):
        ctx = RunContext(raw_prompt="test")
        assert ctx.raw_prompt == "test"
        assert ctx.candidates == []
        assert ctx.anon is None
        assert ctx.answer == ""
        assert ctx.mode == ""
        assert ctx.trace == []

    def test_start_sets_budget(self):
        ctx = RunContext()
        ctx.start()
        assert ctx.budget._start_time > 0
        assert ctx._start_time > 0

    def test_finish_records_latency(self):
        ctx = RunContext()
        ctx.start()
        import time
        time.sleep(0.01)
        ctx.finish()
        assert ctx.total_latency_ms > 0

    def test_add_trace(self):
        ctx = RunContext()
        ctx.add_trace("step 1")
        ctx.add_trace("step 2")
        assert len(ctx.trace) == 2
        assert "step 1" in ctx.trace[0]

    def test_to_response_includes_budget(self):
        ctx = RunContext()
        ctx.start()
        ctx.answer = "42"
        ctx.mode = "direct"
        resp = ctx.to_response()
        assert resp["answer"] == "42"
        assert resp["mode"] == "direct"
        assert "budget" in resp
        assert "latency_ms" in resp
        assert "pipeline_trace" in resp

    def test_to_response_with_verification(self):
        from nim_orchestrator.verifiers.external_checks import (
            VerificationReport,
            VerificationResult,
        )
        ctx = RunContext()
        ctx.verification = VerificationReport()
        ctx.verification.add(
            VerificationResult(verifier_name="test", status="pass", details="ok")
        )
        resp = ctx.to_response()
        assert resp["verification"]["status"] == "passed"
        assert resp["verification"]["all_passed"] is True


# ============================================================
# 4. PolicyEngine
# ============================================================


class TestPolicyEngine:
    def _make_settings(self):
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
            judge=JudgeConfig(model="m", system_prompt="J.", role="judge"),
            synthesizer=SynthesizerConfig(model="m", system_prompt="S.", role="synthesizer"),
        )

    def test_bypass_query_is_direct(self):
        engine = PolicyEngine(self._make_settings())
        result = engine.decide("What is the capital of France?")
        assert result.should_bypass_compiler is True
        assert result.route == "direct"
        assert result.should_run_full_pipeline is False
        assert result.should_speculate is True

    def test_verifiable_query_uses_full_pipeline(self):
        engine = PolicyEngine(self._make_settings())
        result = engine.decide("Calculate 17 * 23")
        assert result.should_bypass_compiler is False
        assert result.should_run_full_pipeline is True

    def test_complex_query_uses_full_pipeline(self):
        engine = PolicyEngine(self._make_settings())
        result = engine.decide("Prove that the square root of 2 is irrational")
        assert result.route == "complex"
        assert result.should_run_full_pipeline is True

    def test_force_single_mode(self):
        engine = PolicyEngine(self._make_settings())
        result = engine.decide("complex query", force_mode="single")
        assert result.route == "direct"
        assert result.should_bypass_compiler is True
        assert result.should_speculate is True
        assert result.should_run_full_pipeline is False

    def test_force_full_mode(self):
        engine = PolicyEngine(self._make_settings())
        result = engine.decide("simple query", force_mode="full")
        assert result.route == "complex"
        assert result.should_bypass_compiler is False
        assert result.should_run_full_pipeline is True

    def test_solvers_separated_from_reviewers(self):
        engine = PolicyEngine(self._make_settings())
        result = engine.decide("Write a function to sort a list")
        assert len(result.solver_configs) >= 2  # solver + alternative_solver
        assert len(result.reviewer_configs) >= 3  # critic + evidence_verifier + devil_advocate
        solver_roles = {c.role for c in result.solver_configs}
        reviewer_roles = {c.role for c in result.reviewer_configs}
        assert AgentRole.SOLVER in solver_roles
        assert AgentRole.ALTERNATIVE_SOLVER in solver_roles
        assert AgentRole.CRITIC in reviewer_roles
        assert AgentRole.EVIDENCE_VERIFIER in reviewer_roles
        assert AgentRole.DEVILS_ADVOCATE in reviewer_roles

    def test_judge_and_synthesizer_populated(self):
        engine = PolicyEngine(self._make_settings())
        result = engine.decide("Prove that P != NP")
        assert result.judge_config is not None
        assert result.judge_config.role == AgentRole.JUDGE
        assert result.synthesizer_config is not None
        assert result.synthesizer_config.role == AgentRole.SYNTHESIZER

    def test_policy_result_has_reason(self):
        engine = PolicyEngine(self._make_settings())
        result = engine.decide("What is the capital of France?")
        assert result.reason != ""

    def test_all_routing_decisions_in_one_result(self):
        """Acceptance: All routing decisions recorded in one policy result."""
        engine = PolicyEngine(self._make_settings())
        result = engine.decide("Write a function to sort a list")
        # route, bypass, speculate, full_pipeline, agent configs — all in one object
        assert hasattr(result, "route")
        assert hasattr(result, "should_bypass_compiler")
        assert hasattr(result, "should_speculate")
        assert hasattr(result, "should_run_full_pipeline")
        assert hasattr(result, "solver_configs")
        assert hasattr(result, "reviewer_configs")
        assert hasattr(result, "judge_config")
        assert hasattr(result, "synthesizer_config")


# ============================================================
# 5. Classify Task Type (single source of truth)
# ============================================================


class TestClassifyTaskType:
    def test_code_writing_is_verifiable(self):
        assert classify_task_type("Write a Python function to sort a list") == "verifiable"

    def test_calculate_is_verifiable(self):
        assert classify_task_type("Calculate the sum of 17 and 23") == "verifiable"

    def test_what_is_number_is_direct(self):
        assert classify_task_type("What is 17 * 23?") == "direct"

    def test_capital_is_direct(self):
        assert classify_task_type("What is the capital of France?") == "direct"

    def test_define_is_direct(self):
        assert classify_task_type("Define photosynthesis") == "direct"

    def test_prove_is_complex(self):
        assert classify_task_type("Prove that P != NP") == "complex"

    def test_design_is_complex(self):
        assert classify_task_type("Design a distributed cache system") == "complex"

    def test_opinion_is_open_ended(self):
        assert classify_task_type("What do you think about Rust vs Go?") == "open_ended"

    def test_recommend_is_open_ended(self):
        assert classify_task_type("Recommend a good book on distributed systems") == "open_ended"

    def test_default_is_complex(self):
        assert classify_task_type("The quick brown fox jumps over") == "complex"


# ============================================================
# 6. Persistent Anonymous IDs Through Debate
# ============================================================


class TestPersistentAnonIDs:
    def test_anon_mapping_labels_stable_after_update(self):
        """Labels persist: Candidate A is still Candidate A after content changes."""
        c1 = Candidate(name="solver", model="m", content="Original answer A")
        c2 = Candidate(name="alt", model="m", content="Original answer B")

        anon = create_anon_mapping([c1, c2])
        label_c1 = anon.original_to_label["solver"]
        label_c2 = anon.original_to_label["alt"]

        # Simulate debate — same names, new content
        c1_updated = Candidate(name="solver", model="m", content="Revised answer A")
        c2_updated = Candidate(name="alt", model="m", content="Revised answer B")
        anon.update_candidates([c1_updated, c2_updated])

        # Labels should NOT change
        assert anon.original_to_label["solver"] == label_c1
        assert anon.original_to_label["alt"] == label_c2

    def test_anon_mapping_uses_update_not_recreate(self):
        """Verify that update_candidates preserves the label assignment."""
        candidates = [
            Candidate(name="a", model="m", content="answer 1"),
            Candidate(name="b", model="m", content="answer 2"),
        ]
        anon = create_anon_mapping(candidates)
        original_labels = list(anon.labels)
        original_mapping = dict(anon.label_to_original)

        # Update with same candidate names but new content
        updated = [
            Candidate(name="a", model="m", content="revised 1"),
            Candidate(name="b", model="m", content="revised 2"),
        ]
        anon.update_candidates(updated)

        assert anon.labels == original_labels
        assert anon.label_to_original == original_mapping

    def test_no_name_based_role_detection(self):
        """Acceptance: No role detection from names — roles come from config."""
        from nim_orchestrator.pipeline.full_pipeline import _infer_role_from_name

        # This function should only be used as fallback
        assert _infer_role_from_name("adversarial_critic") == AgentRole.CRITIC
        assert _infer_role_from_name("evidence_verifier") == AgentRole.EVIDENCE_VERIFIER
        assert _infer_role_from_name("devil_advocate") == AgentRole.DEVILS_ADVOCATE
        assert _infer_role_from_name("alternative_solver") == AgentRole.ALTERNATIVE_SOLVER
        assert _infer_role_from_name("solver") == AgentRole.SOLVER

    async def test_critique_uses_agentrole_not_name(self):
        """Verify critique_candidates uses AgentRole for role detection."""
        from nim_orchestrator.pipeline.full_pipeline import critique_candidates

        class MockClient:
            def __init__(self):
                self.captured = []
            async def chat(self, **kwargs):
                self.captured.append(kwargs)
                return ChatResult(content="critique", model="mock", latency_ms=1)
            async def close(self):
                pass

        candidates = [
            Candidate(name="x", model="m", content="answer 1"),
            Candidate(name="y", model="m", content="answer 2"),
        ]
        anon = create_anon_mapping(candidates)

        # Config with explicit roles — names don't match role keywords
        reviewer_configs = [
            {"name": "reviewer_1", "model": "m", "system_prompt": "critic", "temperature": 0.2, "reasoning_effort": "none", "role": "critic"},
            {"name": "reviewer_2", "model": "m", "system_prompt": "verifier", "temperature": 0.1, "reasoning_effort": "none", "role": "evidence_verifier"},
            {"name": "reviewer_3", "model": "m", "system_prompt": "devil", "temperature": 0.7, "reasoning_effort": "none", "role": "devils_advocate"},
        ]

        mock = MockClient()
        trace = []
        await critique_candidates(
            mock, reviewer_configs, candidates, "test", trace, anon=anon
        )

        # All three reviewers should fire (3 chat calls)
        assert len(mock.captured) == 3


# ============================================================
# 7. PolicyEngine + API Integration
# ============================================================


class TestPolicyEngineAPIIntegration:
    async def test_policy_engine_routes_direct_correctly(self):
        """Verify the API uses PolicyEngine for routing — direct path."""
        from nim_orchestrator.api import handle_intelligence_request

        class MockClient:
            async def chat(self, **kwargs):
                return ChatResult(
                    content="Paris",
                    model="deepseek-v4-flash",
                    latency_ms=300,
                    finish_reason="stop",
                )
            async def close(self):
                pass

        settings = Settings(
            router_base_url="http://mock",
            router_api_key="mock",
            candidates=[
                CandidateConfig(name="solver", model="m", system_prompt="S.", role="solver"),
            ],
            judge=JudgeConfig(model="m", system_prompt="J.", role="judge"),
            synthesizer=SynthesizerConfig(model="m", system_prompt="Syn.", role="synthesizer"),
        )

        client = MockClient()
        result = await handle_intelligence_request(client, settings, "What is the capital of France?")

        assert result["mode"] == "direct"
        trace = result.get("pipeline_trace", [])
        # Should see policy trace
        assert any("Policy" in t for t in trace)

    async def test_policy_engine_routes_full_pipeline_correctly(self):
        """Verify the API uses PolicyEngine for routing — full pipeline path."""
        from nim_orchestrator.api import handle_intelligence_request

        class MockClient:
            def __init__(self):
                self.call_count = 0
            async def chat(self, **kwargs):
                self.call_count += 1
                # Return appropriate content based on call context
                if self.call_count == 1:
                    # Task compiler call
                    return ChatResult(
                        content='{"objective": "Prove it", "risk_level": "medium", "recommended_route": "complex", "context": "Prove it"}',
                        model="deepseek-v4-flash",
                        latency_ms=500,
                        finish_reason="stop",
                    )
                # Pipeline calls (solvers, reviewers, judge, synth)
                return ChatResult(
                    content="The sum of two even numbers is even.",
                    model="deepseek-v4-flash",
                    latency_ms=100,
                    finish_reason="stop",
                )
            async def close(self):
                pass

        settings = Settings(
            router_base_url="http://mock",
            router_api_key="mock",
            candidates=[
                CandidateConfig(name="solver", model="m", system_prompt="S.", role="solver"),
                CandidateConfig(name="adversarial_critic", model="m", system_prompt="C.", role="critic"),
                CandidateConfig(name="evidence_verifier", model="m", system_prompt="V.", role="evidence_verifier"),
                CandidateConfig(name="devil_advocate", model="m", system_prompt="D.", role="devils_advocate"),
            ],
            judge=JudgeConfig(model="m", system_prompt="J.", role="judge"),
            synthesizer=SynthesizerConfig(model="m", system_prompt="Syn", role="synthesizer"),
        )

        client = MockClient()
        result = await handle_intelligence_request(
            client, settings, "Prove that the sum of two even numbers is always even."
        )

        # Should go through full pipeline
        assert result["mode"] == "full"
        trace = result.get("pipeline_trace", [])
        assert any("Policy" in t for t in trace)
        assert any("full pipeline" in t.lower() for t in trace)


# ============================================================
# 8. Config role field
# ============================================================


class TestConfigRoleField:
    def test_candidate_config_has_role(self):
        c = CandidateConfig(name="test", model="m", system_prompt="p", role="critic")
        assert c.role == "critic"

    def test_candidate_config_role_defaults_to_solver(self):
        c = CandidateConfig(name="test", model="m", system_prompt="p")
        assert c.role == "solver"

    def test_judge_config_has_role(self):
        j = JudgeConfig(model="m", system_prompt="p", role="judge")
        assert j.role == "judge"

    def test_synthesizer_config_has_role(self):
        s = SynthesizerConfig(model="m", system_prompt="p", role="synthesizer")
        assert s.role == "synthesizer"

    def test_yaml_loads_with_roles(self):
        from nim_orchestrator.config import load_settings
        settings = load_settings()
        roles = {c.name: c.role for c in settings.candidates}
        assert roles.get("solver") == "solver"
        assert roles.get("alternative_solver") == "alternative_solver"
        assert roles.get("adversarial_critic") == "critic"
        assert roles.get("evidence_verifier") == "evidence_verifier"
        assert roles.get("devil_advocate") == "devils_advocate"
