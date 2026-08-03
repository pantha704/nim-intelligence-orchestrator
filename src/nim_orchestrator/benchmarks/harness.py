import asyncio
import time
from dataclasses import dataclass, field

from ..clustering import Candidate, cluster_candidates
from ..pipeline.full_pipeline import run_full_pipeline
from ..router_client import RouterClient


@dataclass
class BenchmarkResult:
    question: str
    expected_answer: str | None
    mode_results: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question[:200],
            "expected": self.expected_answer,
            "modes": self.mode_results,
        }


async def run_mode_single(
    client: RouterClient, model: str, prompt: str
) -> dict:
    t0 = time.monotonic()
    try:
        result = await client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
        )
        return {
            "answer": result.content,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
            "model": model,
            "error": "",
        }
    except Exception as e:
        return {
            "answer": "",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
            "model": model,
            "error": str(e)[:200],
        }


async def run_mode_best_of_5(
    client: RouterClient, candidates_config: list[dict], prompt: str
) -> dict:
    t0 = time.monotonic()

    async def _gen(cfg):
        try:
            result = await client.chat(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": cfg["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.get("temperature", 0.3),
                reasoning_effort=cfg.get("reasoning_effort", "medium"),
                max_tokens=4096,
            )
            return Candidate(name=cfg["name"], model=cfg["model"], content=result.content)
        except Exception as e:
            return Candidate(name=cfg["name"], model=cfg["model"], content="", error=str(e)[:200])

    candidates = await asyncio.gather(*[_gen(c) for c in candidates_config])
    clustering = cluster_candidates(list(candidates))

    leader = clustering.leader
    answer = leader.content if leader else ""

    return {
        "answer": answer,
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        "candidates": [
            {"name": c.name, "model": c.model, "ok": not c.error}
            for c in candidates
        ],
        "disagreement": clustering.disagreement_level,
        "num_clusters": len(clustering.clusters),
        "error": "" if leader else "no valid candidates",
    }


async def run_mode_best_of_5_judge(
    client: RouterClient, candidates_config: list[dict], judge_config: dict, prompt: str
) -> dict:
    from ..pipeline.full_pipeline import judge_candidates

    t0 = time.monotonic()

    async def _gen(cfg):
        try:
            result = await client.chat(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": cfg["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.get("temperature", 0.3),
                reasoning_effort=cfg.get("reasoning_effort", "medium"),
                max_tokens=4096,
            )
            return Candidate(name=cfg["name"], model=cfg["model"], content=result.content)
        except Exception as e:
            return Candidate(name=cfg["name"], model=cfg["model"], content="", error=str(e)[:200])

    candidates = list(await asyncio.gather(*[_gen(c) for c in candidates_config]))
    judge_result = await judge_candidates(client, judge_config, candidates, prompt, [])

    winner_name = judge_result.get("winner", "")
    winner = next((c for c in candidates if c.name == winner_name and not c.error), None)
    if not winner:
        clustering = cluster_candidates(candidates)
        winner = clustering.leader or next((c for c in candidates if not c.error), None)

    return {
        "answer": winner.content if winner else "",
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        "judge_winner": winner_name,
        "judge_confidence": judge_result.get("confidence"),
        "judge_disagreement": judge_result.get("disagreement_level"),
        "error": "" if winner else "no valid candidates",
    }


async def run_mode_full(
    client: RouterClient,
    candidates_config: list[dict],
    judge_config: dict,
    synthesizer_config: dict,
    prompt: str,
    debate_rounds: int = 2,
    max_repair_rounds: int = 2,
) -> dict:
    result = await run_full_pipeline(
        client,
        prompt,
        candidates_config,
        judge_config,
        synthesizer_config,
        debate_rounds=debate_rounds,
        max_repair_rounds=max_repair_rounds,
    )
    return {
        "answer": result.answer,
        "latency_ms": round(result.total_latency_ms, 1),
        "disagreement": result.clusters.disagreement_level if result.clusters else None,
        "judge_disagreement": result.judge_result.get("disagreement_level") if result.judge_result else None,
        "verification_passed": result.verification_report.all_passed if result.verification_report else None,
        "trace": result.pipeline_trace,
        "error": "",
    }


def score_answer(answer: str, expected: str) -> float:
    if not answer or not expected:
        return 0.0
    answer_lower = answer.lower().strip()
    expected_lower = expected.lower().strip()
    if expected_lower in answer_lower:
        return 1.0
    if answer_lower == expected_lower:
        return 1.0

    words_expected = set(expected_lower.split())
    words_answer = set(answer_lower.split())
    if words_expected and words_answer:
        overlap = words_expected & words_answer
        if len(overlap) / len(words_expected) >= 0.8:
            return 1.0
        if len(overlap) / len(words_expected) >= 0.5:
            return 0.5

    return 0.0


async def run_benchmark(
    client: RouterClient,
    test_cases: list[dict],
    candidates_config: list[dict],
    judge_config: dict,
    synthesizer_config: dict,
    single_model: str = "glm-5.2",
    debate_rounds: int = 2,
    max_repair_rounds: int = 2,
    modes: list[str] | None = None,
) -> list[BenchmarkResult]:
    if modes is None:
        modes = ["single", "best_of_5", "best_of_5_judge", "full"]

    results = []
    for case in test_cases:
        question = case["question"]
        expected = case.get("expected_answer")

        bench = BenchmarkResult(question=question, expected_answer=expected)

        if "single" in modes:
            r = await run_mode_single(client, single_model, question)
            r["score"] = score_answer(r["answer"], expected) if expected else None
            bench.mode_results["single"] = r

        if "best_of_5" in modes:
            r = await run_mode_best_of_5(client, candidates_config, question)
            r["score"] = score_answer(r["answer"], expected) if expected else None
            bench.mode_results["best_of_5"] = r

        if "best_of_5_judge" in modes:
            r = await run_mode_best_of_5_judge(client, candidates_config, judge_config, question)
            r["score"] = score_answer(r["answer"], expected) if expected else None
            bench.mode_results["best_of_5_judge"] = r

        if "full" in modes:
            r = await run_mode_full(
                client, candidates_config, judge_config, synthesizer_config,
                question, debate_rounds, max_repair_rounds,
            )
            r["score"] = score_answer(r["answer"], expected) if expected else None
            bench.mode_results["full"] = r

        results.append(bench)
        print(f"  [{question[:60]}...] " + " | ".join(
            f"{m}: s={r.get('score', '?')}, {r.get('latency_ms', 0):.0f}ms"
            for m, r in bench.mode_results.items()
        ))

    return results


def summarize_results(results: list[BenchmarkResult]) -> dict:
    modes = set()
    for r in results:
        modes.update(r.mode_results.keys())

    summary = {}
    for mode in sorted(modes):
        mode_results = [r.mode_results[mode] for r in results if mode in r.mode_results]
        if not mode_results:
            continue

        scores = [r["score"] for r in mode_results if r.get("score") is not None]
        latencies = [r["latency_ms"] for r in mode_results if r.get("latency_ms")]
        errors = sum(1 for r in mode_results if r.get("error"))

        summary[mode] = {
            "mean_score": sum(scores) / len(scores) if scores else None,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "error_count": errors,
            "total": len(mode_results),
        }

    return summary
