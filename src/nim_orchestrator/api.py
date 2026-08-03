
from .config import Settings
from .difficulty_router import assess_difficulty
from .pipeline.full_pipeline import run_full_pipeline
from .router_client import RouterClient


async def handle_intelligence_request(
    client: RouterClient,
    settings: Settings,
    prompt: str,
    force_mode: str | None = None,
) -> dict:
    if force_mode == "single":
        result = await client.chat(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return {
            "answer": result.content,
            "mode": "single",
            "difficulty": "forced",
            "latency_ms": round(result.latency_ms, 1),
            "pipeline_trace": ["forced single mode"],
        }

    diff = assess_difficulty(
        prompt,
        settings.difficulty_router.simple_keywords,
        settings.difficulty_router.complexity_signals,
        settings.difficulty_router.max_prompt_length_simple,
    )

    if diff.difficulty == "simple" and force_mode is None:
        result = await client.chat(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return {
            "answer": result.content,
            "mode": "single",
            "difficulty": "simple",
            "difficulty_reason": diff.reason,
            "latency_ms": round(result.latency_ms, 1),
            "pipeline_trace": [f"Assessed as simple: {diff.reason}"],
        }

    candidates_config = [
        {
            "name": c.name,
            "model": c.model,
            "system_prompt": c.system_prompt,
            "temperature": c.temperature,
            "reasoning_effort": c.reasoning_effort,
        }
        for c in settings.candidates
    ]
    judge_config = {
        "model": settings.judge.model,
        "system_prompt": settings.judge.system_prompt,
        "temperature": settings.judge.temperature,
        "reasoning_effort": settings.judge.reasoning_effort,
    }
    synth_config = {
        "model": settings.synthesizer.model,
        "system_prompt": settings.synthesizer.system_prompt,
        "temperature": settings.synthesizer.temperature,
        "reasoning_effort": settings.synthesizer.reasoning_effort,
    }

    result = await run_full_pipeline(
        client,
        prompt,
        candidates_config,
        judge_config,
        synth_config,
        debate_rounds=settings.debate_rounds,
        max_repair_rounds=settings.max_repair_rounds,
        verification_timeout=settings.verifier_timeout,
    )

    return {
        "answer": result.answer,
        "mode": "full",
        "difficulty": "complex",
        "difficulty_reason": diff.reason,
        "clusters": {
            "disagreement_level": result.clusters.disagreement_level if result.clusters else None,
            "num_clusters": len(result.clusters.clusters) if result.clusters else 0,
        },
        "judge": result.judge_result,
        "verification": {
            "all_passed": result.verification_report.all_passed if result.verification_report else None,
            "failures": result.verification_report.failures if result.verification_report else [],
        },
        "latency_ms": round(result.total_latency_ms, 1),
        "pipeline_trace": result.pipeline_trace,
    }


async def run_benchmark(settings: Settings) -> dict:
    from pathlib import Path

    import yaml

    from .benchmarks.harness import run_benchmark as _run_bench
    from .benchmarks.harness import summarize_results

    cases_path = Path(__file__).parent.parent.parent / "config" / "benchmark_cases.yaml"
    with open(cases_path) as f:
        cases_data = yaml.safe_load(f)

    test_cases = cases_data.get("BENCHMARK_TEST_CASES", [])

    candidates_config = [
        {
            "name": c.name,
            "model": c.model,
            "system_prompt": c.system_prompt,
            "temperature": c.temperature,
            "reasoning_effort": c.reasoning_effort,
        }
        for c in settings.candidates
    ]
    judge_config = {
        "model": settings.judge.model,
        "system_prompt": settings.judge.system_prompt,
        "temperature": settings.judge.temperature,
        "reasoning_effort": settings.judge.reasoning_effort,
    }
    synth_config = {
        "model": settings.synthesizer.model,
        "system_prompt": settings.synthesizer.system_prompt,
        "temperature": settings.synthesizer.temperature,
        "reasoning_effort": settings.synthesizer.reasoning_effort,
    }

    client = RouterClient(settings.router_base_url, settings.router_api_key)

    results = await _run_bench(
        client,
        test_cases,
        candidates_config,
        judge_config,
        synth_config,
        single_model="glm-5.2",
        debate_rounds=settings.debate_rounds,
        max_repair_rounds=settings.max_repair_rounds,
    )

    summary = summarize_results(results)

    await client.close()

    return {"summary": summary, "results": [r.to_dict() for r in results]}
