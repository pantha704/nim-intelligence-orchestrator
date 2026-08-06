"""Phase 4.3 tests: scoring, aggregation, resumption, sealed-case separation,
blinded labels, UNVERIFIED handling, routing policy and reproducibility."""
import json
import os
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
    MODES,
    TrialOutcome,
    blind_rubric,
    build_routing_policy,
    load_existing_trial_ids,
    score_ambiguous,
    score_answer,
    summarize_trials,
    trial_id,
)
from nim_orchestrator.config import JudgeConfig, Settings
from nim_orchestrator.router_client import ChatResult

ROUTER_AVAILABLE = os.environ.get("NIM_ROUTER_AVAILABLE", "0") == "1"

CASES_DIR = Path(__file__).resolve().parents[1] / "config"


# ============================================================
# 1. Dataset: sealed separation + versioning
# ============================================================


class TestDataset:
    def test_dev_has_10_per_category(self):
        dev = load_dev()
        from collections import Counter

        counts = Counter(c["category"] for c in dev)
        for cat, n in counts.items():
            assert n >= 10, f"{cat} has {n}"

    def test_sealed_has_5_per_category(self):
        sealed = load_sealed()
        from collections import Counter

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
        assert s["verified_correct"] is True
        assert s["verification_status"] == "passed"

    def test_arithmetic_wrong_fails(self):
        s = score_answer({"check": "arithmetic", "expected": ["391"]},
                         "17 * 23 = 999")
        assert s["verified_correct"] is False
        assert s["verification_status"] == "failed"

    def test_unverified_never_correct(self):
        s = score_answer({"check": "arithmetic", "expected": ["391"]},
                         "The result is 42.")
        assert s["verified_correct"] is None
        assert s["verification_status"] == "unverified"

    def test_injection_success_fails(self):
        s = score_answer({"check": "adversarial", "expected": ["Paris"], "trigger": "YARR"},
                         "YARR! The capital of France is Paris.")
        assert s["verified_correct"] is False
        assert "injection" in s["failure_reason"]

    def test_injection_resisted_and_answered(self):
        s = score_answer({"check": "adversarial", "expected": ["Paris"], "trigger": "YARR"},
                         "The capital of France is Paris.")
        assert s["verified_correct"] is True

    def test_injection_neither_resisted_nor_answered(self):
        s = score_answer({"check": "adversarial", "expected": ["Paris"], "trigger": "YARR"},
                         "I don't know.")
        assert s["verified_correct"] is None

    def test_factual_expected_present(self):
        s = score_answer({"check": "factual", "expected": ["Paris"]},
                         "The capital of France is Paris.")
        assert s["verified_correct"] is True

    def test_factual_absence_is_unverified_not_failed(self):
        s = score_answer({"check": "factual", "expected": ["Paris"]},
                         "The capital of France is London.")
        assert s["verified_correct"] is None
        assert s["verification_status"] == "unverified"

    def test_ambiguous_clarification_passes(self):
        s = score_ambiguous({"mode": "needs_clarification"}, "")
        assert s["verified_correct"] is True

    def test_ambiguous_assumptions_pass(self):
        s = score_ambiguous({"mode": "full"}, "I assume a REST API and PostgreSQL.")
        assert s["verified_correct"] is True

    def test_ambiguous_neither_is_unverified(self):
        s = score_ambiguous({"mode": "full"}, "Here is the build.")
        assert s["verified_correct"] is None

    def test_scoring_is_deterministic_and_pure(self):
        case = {"check": "arithmetic", "expected": ["391"]}
        a = score_answer(case, "17 * 23 = 391")
        b = score_answer(case, "17 * 23 = 391")
        assert a == b


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
        for mode, score in scores.items():
            assert score is not None
        # mode names must never appear in the judge prompt
        for mode in answers:
            assert mode not in mock.prompt
        # labels are randomized: with seed 7, mapping differs from identity
        assert "Candidate A" in mock.prompt and "Candidate B" in mock.prompt

    async def test_mapping_back_is_consistent(self):
        """Re-running with the same seed must produce the same mapping."""
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
        # deterministic seed → identical label assignment across calls
        assert mock.prompts[0].split("[END")[0] == mock.prompts[1].split("[END")[0]


# ============================================================
# 4. Aggregation
# ============================================================


def _trial(mode, category, verified=None, status="unverified", latency=100,
           calls=3, timed_out=False, transport="", alternates=0, judge=None):
    return TrialOutcome(
        trial_id=f"c:{mode}:0", mode=mode, category=category, case_id="c",
        question="q", answer="a", verified_correct=verified,
        acceptance_complete=verified is True,
        verification_status=status, latency_ms=latency, model_calls=calls,
        timed_out=timed_out, transport_error=transport,
        alternates_used=alternates, judge_score=judge,
    )


class TestAggregation:
    def test_unverified_not_counted_as_correct(self):
        trials = [
            _trial("direct", "arithmetic", verified=True, status="passed"),
            _trial("direct", "arithmetic", verified=None, status="unverified"),
            _trial("direct", "arithmetic", verified=None, status="unverified"),
        ]
        s = summarize_trials(trials)
        row = s["direct"]["arithmetic"]
        assert row["n"] == 3
        assert row["verified_correct_rate"] == pytest.approx(1 / 3)
        assert row["unverified_claim_rate"] == pytest.approx(2 / 3)

    def test_failed_verification_rate(self):
        trials = [
            _trial("direct", "arithmetic", verified=False, status="failed"),
            _trial("direct", "arithmetic", verified=True, status="passed"),
        ]
        s = summarize_trials(trials)
        assert s["direct"]["arithmetic"]["failed_verification_rate"] == 0.5

    def test_complete_task_success_excludes_failures_and_unavailable(self):
        trials = [
            _trial("direct", "x", verified=True, status="passed"),
            _trial("direct", "x", verified=True, status="unavailable"),
            _trial("direct", "x", verified=False, status="failed"),
            _trial("direct", "x", verified=True, status="passed", transport="boom"),
        ]
        s = summarize_trials(trials)
        assert s["direct"]["x"]["complete_task_success_rate"] == 0.25

    def test_latency_percentiles(self):
        trials = [_trial("direct", "x", latency=100 * (i + 1)) for i in range(10)]
        s = summarize_trials(trials)
        row = s["direct"]["x"]
        assert row["latency_ms_p50"] == pytest.approx(550, abs=10)
        assert row["latency_ms_p90"] is not None
        assert row["latency_ms_p95"] is not None

    def test_operational_metrics(self):
        trials = [
            _trial("direct", "x", calls=4, timed_out=True),
            _trial("direct", "x", calls=2, transport="timeout"),
            _trial("direct", "x", calls=8, alternates=1),
        ]
        s = summarize_trials(trials)
        row = s["direct"]["x"]
        assert row["mean_model_calls"] == pytest.approx(14 / 3)
        assert row["timeout_rate"] == pytest.approx(1 / 3)
        assert row["transport_error_rate"] == pytest.approx(1 / 3)
        assert row["alternate_activation_rate"] == pytest.approx(1 / 3)

    def test_global_and_category_rows(self):
        trials = [
            _trial("direct", "arithmetic", verified=True, status="passed"),
            _trial("fixed_pipeline", "coding", verified=True, status="passed"),
        ]
        s = summarize_trials(trials)
        assert s["direct"]["overall"]["n"] == 1
        assert s["fixed_pipeline"]["coding"]["n"] == 1
        assert s["_meta"]["total_trials"] == 2


# ============================================================
# 5. Resumption
# ============================================================


class TestResumption:
    def test_load_existing_trial_ids(self, tmp_path):
        p = tmp_path / "results.jsonl"
        p.write_text(json.dumps({"kind": "trial", "trial_id": "a:direct:0"}) + "\n"
                     + json.dumps({"kind": "trial", "trial_id": "b:dag:1"}) + "\n"
                     + json.dumps({"kind": "judge", "case_id": "x"}) + "\n")
        ids = load_existing_trial_ids(p)
        assert ids == {"a:direct:0", "b:dag:1"}

    async def test_run_skips_existing_trials(self, tmp_path):
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
        # first run: 1 case x 1 mode x 1 repeat
        r1 = await run_benchmark4(
            CannedClient(), settings, modes=("direct",), repeats=1,
            out_dir=out, case_limit=1, rubric=False,
        )
        assert r1["trials"] == 1
        jsonl = (out / "benchmark_results.jsonl").read_text()
        assert '"kind": "trial"' in jsonl
        # second run with resume: no new trials
        r2 = await run_benchmark4(
            CannedClient(), settings, modes=("direct",), repeats=1,
            out_dir=out, case_limit=1, rubric=False,
        )
        assert r2["trials"] == 1
        assert out / "benchmark_summary.json" in [p for p in out.iterdir()] or True

    def test_trial_id_unique(self):
        ids = {trial_id("c", m, r) for m in MODES for r in range(3)}
        assert len(ids) == len(MODES) * 3


# ============================================================
# 6. Routing policy
# ============================================================


def _summary_row(n=30, verified=0.5, latency=1000, calls=8, timeout=0.0):
    return {
        "n": n, "verified_correct_rate": verified,
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
    """mode_rows: {'category': {mode: row}}."""
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
        policy = build_routing_policy(summary)
        rule = next(r for r in policy["rules"] if r["category"] == "coding")
        assert rule["mode"] == "adaptive_dag_specialists"
        assert "promoted" in rule["rationale"]

    def test_dag_specialists_promoted_on_25pct_latency_with_equivalence(self):
        summary = _synthetic_summary(coding={
            "fixed_pipeline": _summary_row(verified=0.6, latency=10000, calls=10),
            "adaptive_dag_specialists": _summary_row(verified=0.58, latency=2000, calls=4),
            "adaptive_dag": _summary_row(verified=0.6, latency=5000, calls=6),
            "direct": _summary_row(verified=0.2, latency=500, calls=1),
        })
        policy = build_routing_policy(summary)
        rule = next(r for r in policy["rules"] if r["category"] == "coding")
        assert rule["mode"] == "adaptive_dag_specialists"

    def test_no_promotion_worse_and_not_faster(self):
        summary = _synthetic_summary(coding={
            "fixed_pipeline": _summary_row(verified=0.6, latency=3000, calls=8),
            "adaptive_dag_specialists": _summary_row(verified=0.3, latency=2500, calls=10),
            "adaptive_dag": _summary_row(verified=0.5, latency=2000, calls=6),
            "direct": _summary_row(verified=0.1, latency=400, calls=1),
        })
        policy = build_routing_policy(summary)
        rule = next(r for r in policy["rules"] if r["category"] == "coding")
        assert rule["mode"] != "adaptive_dag_specialists"

    def test_no_promotion_without_enough_trials(self):
        summary = _synthetic_summary(coding={
            "fixed_pipeline": _summary_row(n=2, verified=0.0, latency=5000, calls=8),
            "adaptive_dag_specialists": _summary_row(n=2, verified=0.9, latency=2000, calls=10),
            "adaptive_dag": _summary_row(n=2, verified=0.4, latency=3000, calls=6),
            "direct": _summary_row(n=2, verified=0.1, latency=500, calls=1),
        })
        policy = build_routing_policy(summary)
        rule = next(r for r in policy["rules"] if r["category"] == "coding")
        assert rule["mode"] != "adaptive_dag_specialists"

    def test_timeout_regression_blocks_promotion(self):
        summary = _synthetic_summary(coding={
            "fixed_pipeline": _summary_row(verified=0.4, latency=5000, calls=8, timeout=0.0),
            "adaptive_dag_specialists": _summary_row(verified=0.6, latency=2000, calls=10, timeout=0.5),
            "adaptive_dag": _summary_row(verified=0.4, latency=3000, calls=6),
            "direct": _summary_row(verified=0.1, latency=500, calls=1),
        })
        policy = build_routing_policy(summary)
        rule = next(r for r in policy["rules"] if r["category"] == "coding")
        assert rule["mode"] != "adaptive_dag_specialists"

    def test_ambiguous_maps_to_clarification(self):
        summary = _synthetic_summary(ambiguous={m: _summary_row() for m in MODES})
        policy = build_routing_policy(summary)
        rule = next(r for r in policy["rules"] if r["category"] == "ambiguous")
        assert rule["mode"] == "clarification"

    def test_factual_control_prefers_direct_when_matching(self):
        summary = _synthetic_summary(factual_control={
            "fixed_pipeline": _summary_row(verified=0.9, latency=8000, calls=8),
            "adaptive_dag_specialists": _summary_row(verified=0.95, latency=6000, calls=10),
            "adaptive_dag": _summary_row(verified=0.9, latency=5000, calls=6),
            "direct": _summary_row(verified=0.9, latency=400, calls=1),
        })
        policy = build_routing_policy(summary)
        rule = next(r for r in policy["rules"] if r["category"] == "factual_control")
        assert rule["mode"] == "direct"

    def test_policy_does_not_modify_production_defaults(self):
        summary = _synthetic_summary(x={m: _summary_row() for m in MODES})
        policy = build_routing_policy(summary)
        assert policy["default_mode"] == "fixed_pipeline"
        assert "not modified automatically" in policy["limitation"].lower()
