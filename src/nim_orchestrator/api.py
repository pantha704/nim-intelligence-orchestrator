import asyncio
from pathlib import Path

from .config import Settings
from .input_gate import GateAction, gate_prompt
from .router_client import RouterClient, StreamConfig
from .speculative_router import SpeculativeResult, speculative_route


async def handle_intelligence_request(
    client: RouterClient,
    settings: Settings,
    raw_prompt: str,
    force_mode: str | None = None,
) -> dict:
    """Main entry point. Runs input gate → speculative route → full pipeline."""

    # --- Stage 0: Input gate ---
    gate = gate_prompt(raw_prompt)

    if gate.action == GateAction.REJECT:
        return {
            "answer": "",
            "error": gate.reason,
            "mode": "rejected",
            "safety_flag": gate.safety_flag,
            "pipeline_trace": [f"REJECTED: {gate.reason}"],
        }

    prompt = gate.prompt

    if force_mode == "single":
        result = await client.chat(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
            stream_config=StreamConfig(
                max_tokens=256,
                min_tokens=8,
                stop_on_double_newline=True,
                stop_on_code_fence_close=True,
                early_stop_token_budget=128,
            ),
        )
        return {
            "answer": result.content,
            "mode": "single",
            "difficulty": "forced",
            "safety_flag": gate.safety_flag,
            "latency_ms": round(result.latency_ms, 1),
            "pipeline_trace": ["forced single mode"],
        }

    if force_mode == "full":
        return await _run_full(client, settings, prompt, gate)

    # --- Stage 1: Speculative routing ---
    try:
        spec = await asyncio.wait_for(
            speculative_route(
                client,
                model="deepseek-v4-flash",
                prompt=prompt,
                max_quick_tokens=256,
            ),
            timeout=15,
        )
    except TimeoutError:
        trace = ["Speculative route timed out (15s) — escalating to full pipeline"]
        spec = None
        return await _run_full(client, settings, prompt, gate, None, trace)
    except Exception as e:
        trace = [f"Speculative route failed ({type(e).__name__}) — escalating"]
        spec = None
        return await _run_full(client, settings, prompt, gate, None, trace)

    trace = [f"Speculative route: {spec.reason} (confidence={spec.confidence:.2f})"]

    if not spec.escalate:
        return {
            "answer": spec.quick_answer,
            "mode": "speculative_direct",
            "difficulty": "simple",
            "safety_flag": gate.safety_flag,
            "confidence": spec.confidence,
            "latency_ms": round(spec.quick_result.latency_ms, 1) if spec.quick_result else 0,
            "pipeline_trace": trace,
        }

    # --- Stage 2: Full pipeline ---
    trace.append(f"Escalating from speculation (reason: {spec.reason})")
    full_result = await _run_full(client, settings, prompt, gate, spec, trace)
    return full_result


async def _run_full(
    client: RouterClient,
    settings: Settings,
    prompt: str,
    gate,
    spec: SpeculativeResult | None = None,
    trace: list[str] | None = None,
) -> dict:
    from .pipeline.full_pipeline import run_full_pipeline

    if trace is None:
        trace = []

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

    trace.extend(result.pipeline_trace)

    return {
        "answer": result.answer,
        "mode": "full",
        "difficulty": "complex",
        "safety_flag": gate.safety_flag,
        "clusters": {
            "disagreement_level": result.clusters.disagreement_level if result.clusters else None,
            "num_clusters": len(result.clusters.clusters) if result.clusters else 0,
        },
        "judge": result.judge_result,
        "verification": {
            "all_passed": result.verification_report.all_passed if result.verification_report else None,
            "failures": result.verification_report.failures if result.verification_report else [],
        },
        "speculative_quick_answer": spec.quick_answer if spec else "",
        "latency_ms": round(result.total_latency_ms, 1),
        "pipeline_trace": trace,
    }


async def run_benchmark(settings: Settings) -> dict:
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
        single_model="deepseek-v4-flash",
        debate_rounds=settings.debate_rounds,
        max_repair_rounds=settings.max_repair_rounds,
    )

    summary = summarize_results(results)

    await client.close()

    return {"summary": summary, "results": [r.to_dict() for r in results]}
