"""Phase 4.3/4.3.1 tests: scoring, aggregation, resumption, sealed-case
separation, blinded labels, UNVERIFIED handling, routing policy,
reproducibility, budgets, and event-based persistence."""
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from nim_orchestrator.benchmarks.dataset import (
    DatasetError,
    dev_and_sealed_disjoint,
    load_cases,
    load_dev,
    load_sealed,
)
from nim_orchestrator.benchmarks.four_mode import (
    BENCHMARK_BUDGET,
    MODES,
    TrialOutcome,
    blind_rubric,
    budget_key,
    build_routing_policy,
    effective_correct,
    load_existing_trial_ids,
    run_id_for,
    score_ambiguous,
    score_answer,
    stable_seed,
    summarize_trials,
    trial_id,
)
from nim_orchestrator.config import JudgeConfig, Settings
from nim_orchestrator.router_client import ChatResult
from nim_orchestrator.verifiers.sandbox import sandbox_run_count

ROUTER_AVAILABLE = os.environ.get("NIM_ROUTER_AVAILABLE", "0") == "1"

CASES_DIR = Path(__file__).resolve().parents[1] / "config"


# ============================================================
# 1. Dataset: sealed separation + versioning
# ============================================================


class TestDataset:
    def test_dev_has_10_per_category(self):
        dev = load_dev()
        counts = Counter(c["category"] for c in dev)
        assert len(counts) == 10
        for cat, n in counts.items():
            assert n >= 10, f"{cat} has {n}"

    def test_sealed_has_5_per_category(self):
        sealed = load_sealed()
        counts = Counter(c["category"] for c in sealed)
        assert len(counts) == 10
        for cat, n in counts.items():
            assert n >= 5, f"{cat} has {n}"

    def test_dev_and_sealed_disjoint(self):
        assert dev_and_sealed_disjoint()

    def test_checksum_detects_tampering(self, tmp_path):
        import yaml

        cases = load_dev()[:2]
        data = {"version": "v1", "checksum": "deadbeef", "cases": cases}
        p = tmp_path / "tampered.yaml"
        p.write_text(yaml.safe_dump(data))
        with pytest.raises(DatasetError):
            load_cases(p)

    def test_missing_file_raises(self):
        with pytest.raises(DatasetError):
            load_cases(Path("/nonexistent/bench.yaml"))


# ============================================================
# 2. Scoring rules
# ============================================================


class TestScoring:
    def test_arithmetic_correct(self):
        s = score_answer({"check": "arithmetic", "expected": ["391"]},
                         "The product is 17 * 23 = 391.")
        assert s["deterministic_verified_correct"] is True
        assert s["verification_status"] == "passed"

    def test_arithmetic_wrong_fails(self):
        s = score_answer({"check": "arithmetic", "expected": ["391"]}, "17 * 23 = 999")
        assert s["deterministic_verified_correct"] is False
        assert s["verification_status"] == "failed"

    def test_unverified_never_correct(self):
        s = score_answer({"check": "arithmetic", "expected": ["391"]}, "The result is 42.")
        assert s["deterministic_verified_correct"] is None
        assert s["verification_status"] == "unverified"

    def test_injection_success_fails(self):
        s = score_answer({"check": "adversarial", "expected": ["Paris"], "trigger": "YARR"},
                         "YARR! The capital of France is Paris.")
        assert s["deterministic_verified_correct"] is False
        assert "injection" in s["failure_reason"]

    def test_injection_resisted_and_answered(self):
        s = score_answer({"check": "adversarial", "expected": ["Paris"], "trigger": "YARR"},
                         "The capital of France is Paris.")
        assert s["deterministic_verified_correct"] is True

    def test_injection_neither_resisted_nor_answered(self):
        s = score_answer({"check": "adversarial", "expected": ["Paris"], "trigger": "YARR"},
                         "I don't know.")
        assert s["deterministic_verified_correct"] is None

    def test_factual_expected_present(self):
        s = score_answer({"check": "factual", "expected": ["Paris"]},
                         "The capital of France is Paris.")
        assert s["deterministic_verified_correct"] is True

    def test_factual_negated_statement_fails(self):
        """Regression: a negated statement can never satisfy a factual check."""
        s = score_answer({"check": "factual", "expected": ["Paris"]},
                         "The capital of France is not Paris.")
        assert s["deterministic_verified_correct"] is False
        assert s["verification_status"] == "failed"

    def test_factual_wrong_value_is_unverified_not_failed(self):
        s = score_answer({"check": "factual", "expected": ["Paris"]},
                         "The capital of France is London.")
        assert s["deterministic_verified_correct"] is None

    def test_ambiguous_clarification_passes(self):
        s = score_ambiguous({"mode": "needs_clarification",
                             "clarification_question": "Which stack and scope?"}, "")
        assert s["deterministic_verified_correct"] is True

    def test_ambiguous_clarification_without_question_fails(self):
        s = score_ambiguous({"mode": "needs_clarification", "clarification_question": ""}, "")
        assert s["deterministic_verified_correct"] is None

    def test_ambiguous_structured_assumptions_pass(self):
        s = score_ambiguous({"mode": "full"},
                            "I assume a REST API with PostgreSQL, deployed on AWS. Here is the design.")
        assert s["deterministic_verified_correct"] is True

    def test_ambiguous_bare_assume_mention_is_not_enough(self):
        """Regression: the word 'assume' alone must not count as completion."""
        s = score_ambiguous({"mode": "full"}, "Let's assume.")
        assert s["deterministic_verified_correct"] is None

    def test_scoring_is_deterministic_and_pure(self):
        case = {"check": "arithmetic", "expected": ["391"]}
        assert score_answer(case, "17 * 23 = 391") == score_answer(case, "17 * 23 = 391")

    def test_architecture_coverage_is_a_gate_not_correctness(self):
        """Keyword coverage alone is NEVER correctness for prose categories."""
        case = {"id": "c1", "check": "architecture", "required": ["scalable", "cache"]}
        s = score_answer(case, "The system is scalable and uses a cache.")
        assert s["deterministic_verified_correct"] is None
        assert s["deterministic_met"] is True
        assert s["verification_status"] == "judged"


# ============================================================
# 3. Blinded rubric
# ============================================================


class TestBlindedRubric:
    def _settings(self):
        return Settings(
            router_base_url="http://mock", router_api_key="mock",
            judge=JudgeConfig(model="judge-m", system_prompt="You are a judge."),
        )

    async def test_labels_randomized_and_mode_names_hidden(self):
        class JudgeMock:
            def __init__(self):
                self.prompt = ""

            async def chat(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return ChatResult(
                    content='{"scores": [{"candidate": "Candidate A", "score": 8}, '
                            '{"candidate": "Candidate B", "score": 3}, '
                            '{"candidate": "Candidate C", "score": 6}, '
                            '{"candidate": "Candidate D", "score": 5}]}',
                    model="judge-m", latency_ms=10,
                )

        answers = {
            "direct": "answer one",
            "fixed_pipeline": "answer two",
            "adaptive_dag": "answer three",
            "adaptive_dag_specialists": "answer four",
        }
        mock = JudgeMock()
        scores = await blind_rubric(mock, self._settings(), "Question?", answers, seed=7)

        assert set(scores) == set(answers)
        for mode in answers:
            assert scores[mode] is not None
        for mode in answers:
            assert mode not in mock.prompt

    async def test_mapping_back_is_consistent(self):
        class JudgeMock:
            def __init__(self):
                self.prompts = []

            async def chat(self, **kwargs):
                self.prompts.append(kwargs["messages"][1]["content"])
                return ChatResult(
                    content='{"scores": [{"candidate": "Candidate A", "score": 9}, '
                            '{"candidate": "Candidate B", "score": 1}]}',
                    model="m", latency_ms=1,
                )

        answers = {"direct": "a", "fixed_pipeline": "b"}
        mock = JudgeMock()
        s1 = await blind_rubric(mock, self._settings(), "Q?", answers, seed=99)
        s2 = await blind_rubric(mock, self._settings(), "Q?", answers, seed=99)
        assert s1 == s2


# ============================================================
# 4. Aggregation + effective correctness
# ============================================================


def _trial(mode, category, deterministic=None, met=None, status="unverified",
           latency=100, calls=3, timed_out=False, transport="", alternates=0,
           case_id="c", repeat=0):
    return TrialOutcome(
        trial_id=f"{case_id}:{mode}:{repeat}", mode=mode, category=category,
        case_id=case_id, question="q", answer="a", run_id="r",
        deterministic_verified_correct=deterministic, deterministic_met=met,
        verification_status=status, latency_ms=latency, model_calls=calls,
        timed_out=timed_out, transport_error=transport, alternates_used=alternates,
    )


class TestAggregation:
    def test_unverified_not_counted_as_correct(self):
        trials = [
            _trial("direct", "arithmetic", deterministic=True, status="passed"),
            _trial("direct", "arithmetic", deterministic=None, status="unverified"),
            _trial("direct", "arithmetic", deterministic=None, status="unverified"),
        ]
        s = summarize_trials(trials)
        row = s["direct"]["arithmetic"]
        assert row["n"] == 3
        assert row["verified_correct_rate"] == pytest.approx(1 / 3)
        assert row["unverified_claim_rate"] == pytest.approx(2 / 3)

    def test_failed_verification_rate(self):
        trials = [
            _trial("direct", "arithmetic", deterministic=False, status="failed"),
            _trial("direct", "arithmetic", deterministic=True, status="passed"),
        ]
        s = summarize_trials(trials)
        assert s["direct"]["arithmetic"]["failed_verification_rate"] == 0.5

    def test_rubric_success_requires_coverage_and_judge_threshold(self):
        """Prose correctness = deterministic gate AND blinded judge >= 7."""
        trials = [
            _trial("direct", "systems_architecture", met=True, status="judged", case_id="a"),
            _trial("direct", "systems_architecture", met=True, status="judged", case_id="b"),
            _trial("direct", "systems_architecture", met=False, status="unverified", case_id="c"),
        ]
        judge_events = [
            {"kind": "judge", "case_id": "a", "repeat": 0, "scores": {"direct": 8.0}},
            {"kind": "judge", "case_id": "b", "repeat": 0, "scores": {"direct": 4.0}},
        ]
        s = summarize_trials(trials, judge_events)
        row = s["direct"]["systems_architecture"]
        assert row["rubric_success_rate"] == pytest.approx(1 / 3)
        # judge missing for c (met=False) → failed; coverage gate → not success

    def test_rubric_missing_judge_is_unverified_not_correct(self):
        trials = [_trial("direct", "systems_architecture", met=True, status="judged", case_id="a")]
        s = summarize_trials(trials, judge_events=[])
        assert s["direct"]["systems_architecture"]["rubric_success_rate"] == 0.0

    def test_effective_correct_deterministic(self):
        t = _trial("direct", "arithmetic", deterministic=True, status="passed")
        assert effective_correct(t, {}) == (True, "passed")

    def test_effective_correct_rubric(self):
        t = _trial("direct", "systems_architecture", met=True, status="judged", case_id="a")
        judge = {("a", 0, "direct"): 8.0}
        assert effective_correct(t, judge) == (True, "judged")
        judge_low = {("a", 0, "direct"): 4.0}
        assert effective_correct(t, judge_low) == (False, "judged")

    def test_effective_correct_rubric_coverage_gate(self):
        t = _trial("direct", "systems_architecture", met=False, status="unverified", case_id="a")
        judge = {("a", 0, "direct"): 9.0}
        # high judge score cannot rescue failed coverage
        assert effective_correct(t, judge) == (False, "failed")

    def test_latency_percentiles(self):
        trials = [_trial("direct", "x", latency=100 * (i + 1)) for i in range(10)]
        row = summarize_trials(trials)["direct"]["x"]
        assert row["latency_ms_p50"] == pytest.approx(550, abs=10)
        assert row["latency_ms_p90"] is not None

    def test_operational_metrics(self):
        trials = [
            _trial("direct", "x", calls=4, timed_out=True),
            _trial("direct", "x", calls=2, transport="timeout"),
            _trial("direct", "x", calls=8, alternates=1),
        ]
        row = summarize_trials(trials)["direct"]["x"]
        assert row["mean_model_calls"] == pytest.approx(14 / 3)
        assert row["timeout_rate"] == pytest.approx(1 / 3)
        assert row["transport_error_rate"] == pytest.approx(1 / 3)
        assert row["alternate_activation_rate"] == pytest.approx(1 / 3)


# ============================================================
# 5. Seeds, budgets, run isolation, resumption
# ============================================================


class TestReproducibility:
    def test_stable_seed_sha256_based(self):
        a = stable_seed("arithmet-001", 4242, 0)
        b = stable_seed("arithmet-001", 4242, 0)
        assert a == b
        assert a != stable_seed("arithmet-001", 4242, 1)
        assert a != stable_seed("arithmet-002", 4242, 0)
        # within int range and not hash()-dependent
        assert 0 < a < 2**48

    def test_seed_supported_false_recorded(self):
        from nim_orchestrator.benchmarks.four_mode import run_trial

        class C:
            async def chat(self, **kwargs):
                return ChatResult(content="The answer is 42.", model="m", latency_ms=5,
                                  finish_reason="stop")

        import asyncio

        case = {"id": "c1", "category": "arithmetic", "question": "Calculate 2+2",
                "check": "arithmetic", "expected": ["4"], "risk_level": "low"}
        settings = Settings(router_base_url="http://mock", router_api_key="mock")
        outcome = asyncio.run(run_trial(
            C(), settings, case, "direct", 0, run_id="r1", seed=1234,
            budget_limits=BENCHMARK_BUDGET,
        ))
        assert outcome.seed == 1234
        assert outcome.seed_supported is False  # model API cannot accept seeds

    def test_equal_budget_limits_flow_through(self):
        """All modes receive the identical benchmark budget."""
        from nim_orchestrator.benchmarks.four_mode import run_trial

        captured = {}

        class CapturingClient:
            async def chat(self, **kwargs):
                return ChatResult(content="The answer is 42.", model="m", latency_ms=5,
                                  finish_reason="stop")

            async def close(self):
                pass

        async def _capture():
            import asyncio as _a

            return _a  # placeholder

        class CaptureAPI:
            """Wrap handle_intelligence_request to record budget_limits."""

        from nim_orchestrator.api import handle_intelligence_request as _orig

        async def spy(client, settings, prompt, force_mode=None, dag_config=None, budget_limits=None):
            captured["budget_limits"] = budget_limits
            captured["force_mode"] = force_mode
            return await _orig(client, settings, prompt, force_mode=force_mode,
                               dag_config=dag_config, budget_limits=budget_limits)

        import nim_orchestrator.benchmarks.four_mode as fm

        fm.handle_intelligence_request = spy
        try:
            case = {"id": "c1", "category": "arithmetic", "question": "What is 2+2?",
                    "check": "arithmetic", "expected": ["4"], "risk_level": "low"}
            settings = Settings(
                router_base_url="http://mock", router_api_key="mock",
                candidates=[],
            )
            import asyncio

            for mode in ("direct", "fixed_pipeline"):
                asyncio.run(run_trial(
                    CapturingClient(), settings, case, mode, 0,
                    run_id="r1", seed=1, budget_limits=BENCHMARK_BUDGET,
                ))
                assert captured["budget_limits"] == BENCHMARK_BUDGET
        finally:
            fm.handle_intelligence_request = _orig

    def test_budget_key_stable(self):
        assert budget_key(BENCHMARK_BUDGET) == budget_key(BENCHMARK_BUDGET)
        assert budget_key(None) == "unrestricted"
        assert budget_key(BENCHMARK_BUDGET) != budget_key(None)

    def test_run_id_stable_and_config_sensitive(self):
        r1 = run_id_for("c", "d", "dev", "h", "k", MODES, 3, 42)
        assert r1 == run_id_for("c", "d", "dev", "h", "k", MODES, 3, 42)
        assert r1 != run_id_for("c", "d", "dev", "h", "k", MODES, 3, 43)
        assert r1 != run_id_for("c", "d", "dev", "h", "k2", MODES, 3, 42)
        assert len(r1) == 16

    def test_sandbox_counting(self):
        before = sandbox_run_count()
        from nim_orchestrator.verifiers.sandbox import run_secure_sandbox

        r = run_secure_sandbox("print(1)")
        if r.status == "unavailable":
            assert sandbox_run_count() == before
        else:
            assert sandbox_run_count() == before + 1

    def test_budget_exhausted_only_on_denial(self):
        """Reaching the call limit without a denial is NOT exhaustion."""
        from nim_orchestrator.benchmarks.four_mode import run_trial

        trace_with_denial = ["Starting full pipeline", "critic skipped: budget exhausted at 20/20 calls"]
        trace_at_limit = ["Starting full pipeline", "Generated 2 candidates"]

        class C:
            def __init__(self, trace):
                self.trace = trace

            async def chat(self, **kwargs):
                return ChatResult(content="The answer is 42.", model="m", latency_ms=5,
                                  finish_reason="stop")

        import asyncio

        case = {"id": "c1", "category": "arithmetic", "question": "What is 2+2?",
                "check": "arithmetic", "expected": ["4"], "risk_level": "low"}
        settings = Settings(router_base_url="http://mock", router_api_key="mock")

        original = fm_handle()

        import nim_orchestrator.benchmarks.four_mode as fm

        async def spy(client, settings, prompt, force_mode=None, dag_config=None, budget_limits=None):
            resp = await original(client, settings, prompt, force_mode=force_mode,
                                  dag_config=dag_config, budget_limits=budget_limits)
            if client.trace:
                resp = dict(resp)
                resp["pipeline_trace"] = client.trace
            return resp

        fm.handle_intelligence_request = spy
        try:
            o1 = asyncio.run(run_trial(C(trace_with_denial), settings, case, "direct", 0,
                                       run_id="r", seed=1, budget_limits=None))
            o2 = asyncio.run(run_trial(C(trace_at_limit), settings, case, "direct", 0,
                                       run_id="r", seed=1, budget_limits=None))
            assert o1.budget_exhausted is True
            assert o2.budget_exhausted is False
        finally:
            fm.handle_intelligence_request = original


def fm_handle():
    from nim_orchestrator.api import handle_intelligence_request

    return handle_intelligence_request


class TestResumption:
    def test_load_existing_filters_by_run_id(self, tmp_path):
        p = tmp_path / "results.jsonl"
        p.write_text(
            json.dumps({"kind": "trial", "run_id": "AAA", "trial_id": "a:direct:0"}) + "\n"
            + json.dumps({"kind": "trial", "run_id": "BBB", "trial_id": "b:dag:1"}) + "\n"
            + json.dumps({"kind": "judge", "run_id": "AAA", "case_id": "x"}) + "\n"
        )
        ids = load_existing_trial_ids(p, "AAA")
        assert ids == {"a:direct:0"}
        ids_other = load_existing_trial_ids(p, "BBB")
        assert ids_other == {"b:dag:1"}

    async def test_events_are_append_only_and_run_isolated(self, tmp_path):
        """Judge events are separate append-only records; trial records are
        never rewritten, and other-run events are ignored."""
        from nim_orchestrator.benchmarks.four_mode import run_benchmark4
        from nim_orchestrator.config import CandidateConfig, SynthesizerConfig

        class CannedClient:
            async def chat(self, **kwargs):
                return ChatResult(content="The answer is 42.", model="m",
                                  latency_ms=5, finish_reason="stop")

            async def close(self):
                pass

        settings = Settings(
            router_base_url="http://mock", router_api_key="mock",
            candidates=[CandidateConfig(name="solver", model="m", system_prompt="S.", role="solver")],
            judge=JudgeConfig(model="m", system_prompt="J."),
            synthesizer=SynthesizerConfig(model="m", system_prompt="Syn."),
        )
        out = tmp_path / "artifacts"
        r1 = await run_benchmark4(
            CannedClient(), settings, modes=("direct", "fixed_pipeline"), repeats=1,
            out_dir=out, per_category_limit=1, rubric=True,
        )
        assert r1["trials"] == 20  # 10 categories x 2 modes x 1 repeat
        jsonl = (out / "benchmark_results.jsonl").read_text()
        assert '"kind": "trial"' in jsonl
        # judge events exist as separate records for rubric categories
        judge_lines = [l for l in jsonl.splitlines() if '"kind": "judge"' in l]
        assert len(judge_lines) == 3  # research + architecture + security

        # resume: same run config → nothing duplicated
        r2 = await run_benchmark4(
            CannedClient(), settings, modes=("direct", "fixed_pipeline"), repeats=1,
            out_dir=out, per_category_limit=1, rubric=True,
        )
        assert r2["trials"] == 20
        assert (out / "benchmark_results.jsonl").read_text() == jsonl  # append-only

        # different run config (different seed base) → separate run, no clash
        r3 = await run_benchmark4(
            CannedClient(), settings, modes=("direct", "fixed_pipeline"), repeats=1,
            out_dir=out, per_category_limit=1, rubric=True, seed_base=9999,
        )
        assert r3["trials"] == 20
        lines = (out / "benchmark_results.jsonl").read_text().splitlines()
        assert len(lines) == len(jsonl.splitlines()) + 20 + 3

    async def test_stratified_per_category_limit(self, tmp_path):
        from nim_orchestrator.benchmarks.four_mode import run_benchmark4
        from nim_orchestrator.config import CandidateConfig, SynthesizerConfig

        class CannedClient:
            async def chat(self, **kwargs):
                return ChatResult(content="The answer is 42.", model="m",
                                  latency_ms=5, finish_reason="stop")

            async def close(self):
                pass

        settings = Settings(
            router_base_url="http://mock", router_api_key="mock",
            candidates=[CandidateConfig(name="solver", model="m", system_prompt="S.", role="solver")],
            judge=JudgeConfig(model="m", system_prompt="J."),
            synthesizer=SynthesizerConfig(model="m", system_prompt="Syn."),
        )
        out = tmp_path / "artifacts"
        r = await run_benchmark4(
            CannedClient(), settings, modes=("direct",), repeats=2,
            out_dir=out, per_category_limit=2, rubric=False,
        )
        # 2 cases x 10 categories x 1 mode x 2 repeats = 40 trials
        assert r["trials"] == 40
        summary = r["summary"]
        for cat in summary["_meta"]["categories"]:
            assert summary["direct"][cat]["n"] == 4  # 2 cases x 2 repeats

    def test_trial_id_unique(self):
        ids = {trial_id("c", m, r) for m in MODES for r in range(3)}
        assert len(ids) == len(MODES) * 3


# ============================================================
# 6. Routing policy
# ============================================================


def _summary_row(n=30, verified=0.5, latency=1000, calls=8, timeout=0.0):
    return {
        "n": n, "verified_correct_rate": verified,
        "rubric_success_rate": verified,
        "acceptance_complete_rate": verified,
        "failed_verification_rate": 0.1,
        "unverified_claim_rate": 1 - verified,
        "complete_task_success_rate": verified,
        "latency_ms_p50": latency, "latency_ms_p90": latency * 2,
        "latency_ms_p95": latency * 3,
        "mean_model_calls": calls, "timeout_rate": timeout,
        "transport_error_rate": 0.0, "alternate_activation_rate": 0.1,
        "budget_exhaustion_rate": 0.0,
    }


def _synthetic_summary(**mode_rows):
    summary = {"_meta": {"categories": list(mode_rows), "total_trials": 1, "modes": MODES}}
    for mode in MODES:
        summary[mode] = {"overall": {"n": 0}}
        for cat, rows in mode_rows.items():
            if mode in rows:
                summary[mode][cat] = rows[mode]
    return summary


class TestRoutingPolicy:
    def test_dag_specialists_promoted_on_10pp_gain(self):
        summary = _synthetic_summary(coding={
            "fixed_pipeline": _summary_row(verified=0.4, latency=5000, calls=8),
            "adaptive_dag_specialists": _summary_row(verified=0.55, latency=4000, calls=10),
            "adaptive_dag": _summary_row(verified=0.45, latency=3000, calls=6),
            "direct": _summary_row(verified=0.1, latency=500, calls=1),
        })
        rule = next(r for r in build_routing_policy(summary)["rules"] if r["category"] == "coding")
        assert rule["mode"] == "adaptive_dag_specialists"
        assert "promoted" in rule["rationale"]

    def test_dag_specialists_promoted_on_25pct_latency_with_equivalence(self):
        summary = _synthetic_summary(coding={
            "fixed_pipeline": _summary_row(verified=0.6, latency=10000, calls=10),
            "adaptive_dag_specialists": _summary_row(verified=0.58, latency=2000, calls=4),
            "adaptive_dag": _summary_row(verified=0.6, latency=5000, calls=6),
            "direct": _summary_row(verified=0.2, latency=500, calls=1),
        })
        rule = next(r for r in build_routing_policy(summary)["rules"] if r["category"] == "coding")
        assert rule["mode"] == "adaptive_dag_specialists"

    def test_no_promotion_worse_and_not_faster(self):
        summary = _synthetic_summary(coding={
            "fixed_pipeline": _summary_row(verified=0.6, latency=3000, calls=8),
            "adaptive_dag_specialists": _summary_row(verified=0.3, latency=2500, calls=10),
            "adaptive_dag": _summary_row(verified=0.5, latency=2000, calls=6),
            "direct": _summary_row(verified=0.1, latency=400, calls=1),
        })
        rule = next(r for r in build_routing_policy(summary)["rules"] if r["category"] == "coding")
        assert rule["mode"] != "adaptive_dag_specialists"

    def test_no_promotion_without_enough_trials(self):
        summary = _synthetic_summary(coding={
            "fixed_pipeline": _summary_row(n=2, verified=0.0, latency=5000, calls=8),
            "adaptive_dag_specialists": _summary_row(n=2, verified=0.9, latency=2000, calls=10),
            "adaptive_dag": _summary_row(n=2, verified=0.4, latency=3000, calls=6),
            "direct": _summary_row(n=2, verified=0.1, latency=500, calls=1),
        })
        rule = next(r for r in build_routing_policy(summary)["rules"] if r["category"] == "coding")
        assert rule["mode"] != "adaptive_dag_specialists"

    def test_timeout_regression_blocks_promotion(self):
        summary = _synthetic_summary(coding={
            "fixed_pipeline": _summary_row(verified=0.4, latency=5000, calls=8, timeout=0.0),
            "adaptive_dag_specialists": _summary_row(verified=0.6, latency=2000, calls=10, timeout=0.5),
            "adaptive_dag": _summary_row(verified=0.4, latency=3000, calls=6),
            "direct": _summary_row(verified=0.1, latency=500, calls=1),
        })
        rule = next(r for r in build_routing_policy(summary)["rules"] if r["category"] == "coding")
        assert rule["mode"] != "adaptive_dag_specialists"

    def test_open_ended_uses_rubric_rate(self):
        """Routing for prose categories uses rubric_success_rate, not
        deterministic/coverage numbers."""
        summary = _synthetic_summary(systems_architecture={
            "fixed_pipeline": _summary_row(verified=0.5, latency=5000, calls=8),
            "adaptive_dag_specialists": _summary_row(verified=0.65, latency=2000, calls=10),
            "adaptive_dag": _summary_row(verified=0.55, latency=3000, calls=6),
            "direct": _summary_row(verified=0.2, latency=500, calls=1),
        })
        policy = build_routing_policy(summary)
        rule = next(r for r in policy["rules"] if r["category"] == "systems_architecture")
        assert "rubric" in rule["rationale"]

    def test_ambiguous_maps_to_clarification(self):
        summary = _synthetic_summary(ambiguous={m: _summary_row() for m in MODES})
        rule = next(r for r in build_routing_policy(summary)["rules"] if r["category"] == "ambiguous")
        assert rule["mode"] == "clarification"

    def test_factual_control_prefers_direct_when_matching(self):
        summary = _synthetic_summary(factual_control={
            "fixed_pipeline": _summary_row(verified=0.9, latency=8000, calls=8),
            "adaptive_dag_specialists": _summary_row(verified=0.95, latency=6000, calls=10),
            "adaptive_dag": _summary_row(verified=0.9, latency=5000, calls=6),
            "direct": _summary_row(verified=0.9, latency=400, calls=1),
        })
        rule = next(r for r in build_routing_policy(summary)["rules"] if r["category"] == "factual_control")
        assert rule["mode"] == "direct"

    def test_sealed_policy_is_advisory_only(self):
        summary = _synthetic_summary(x={m: _summary_row() for m in MODES})
        policy = build_routing_policy(summary, sealed=True)
        assert policy["sealed"] is True
        assert "must NOT be used to tune" in policy["sealed_note"]

    def test_policy_does_not_modify_production_defaults(self):
        summary = _synthetic_summary(x={m: _summary_row() for m in MODES})
        policy = build_routing_policy(summary)
        assert policy["default_mode"] == "fixed_pipeline"
        assert "not modified automatically" in policy["limitation"].lower()
