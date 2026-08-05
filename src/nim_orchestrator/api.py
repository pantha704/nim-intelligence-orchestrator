from pathlib import Path

from .config import Settings
from .policy import PolicyEngine
from .router_client import RouterClient
from .task_compiler import TaskSpec, compile_task
from .transport_gate import transport_gate


async def handle_intelligence_request(
    client: RouterClient,
    settings: Settings,
    raw_prompt: str,
    force_mode: str | None = None,
) -> dict:
    """Main entry point. Pipeline: transport gate → task compiler → policy engine → pipeline."""

    # --- Stage 0: Transport gate (minimal — size, encoding, emptiness only) ---
    gate = transport_gate(raw_prompt)
    if not gate.ok:
        return {
            "answer": "",
            "error": gate.reason,
            "mode": "rejected",
            "pipeline_trace": [f"REJECTED: {gate.reason}"],
        }

    # raw_prompt is preserved immutably — never modified
    prompt = gate.raw_prompt

    if force_mode == "single":
        result = await client.chat(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        return {
            "answer": result.content,
            "mode": "single",
            "latency_ms": round(result.latency_ms, 1),
            "pipeline_trace": ["forced single mode"],
        }

    # --- Stage 1: Task Compiler (with bypass for obvious simple queries) ---
    from .task_compiler import bypass_task_spec, should_bypass_compiler

    if should_bypass_compiler(prompt):
        task_result = bypass_task_spec(prompt)
        trace = ["Compiler bypassed — obvious simple query"]
    else:
        task_result = await compile_task(
            client,
            model=settings.task_compiler.model,
            raw_prompt=prompt,
            timeout_seconds=settings.task_compiler.timeout_seconds,
        )
        trace = [
            f"Task compiled: route={task_result.task_spec.recommended_route}, risk={task_result.task_spec.risk_level}, "
            f"subtasks={len(task_result.task_spec.subtasks)}, ambiguities={len(task_result.task_spec.ambiguities)} "
            f"[{task_result.latency_ms:.0f}ms]",
        ]

    task_spec = task_result.task_spec

    # Ambiguity check — ask one question if a high-impact ambiguity needs clarification
    if task_result.needs_clarification:
        trace.append(f"Clarification needed: {task_result.clarification_question}")
        return {
            "answer": "",
            "mode": "needs_clarification",
            "clarification_question": task_result.clarification_question,
            "task_spec": task_spec.model_dump(),
            "pipeline_trace": trace,
        }

    if task_spec.assumptions:
        trace.append(f"Assumptions: {'; '.join(task_spec.assumptions[:3])}")

    # --- Stage 2: Central Policy Engine — single routing decision ---
    engine = PolicyEngine(settings)
    policy = engine.decide(prompt, task_spec=task_spec, force_mode=force_mode)

    trace.append(f"Policy: route={policy.route}, reason={policy.reason}")

    if force_mode == "full":
        return await _run_full(client, settings, prompt, task_spec, trace, policy)

    # --- Stage 3: Execute based on policy ---
    if policy.should_speculate and policy.route == "direct":
        from .speculative_router import speculative_route

        spec = await speculative_route(
            client, model="deepseek-v4-flash", prompt=prompt, max_quick_tokens=256,
        )
        trace.append(f"Speculative route: {spec.reason} (route={spec.route})")

        if not spec.escalate:
            return {
                "answer": spec.quick_answer,
                "mode": "direct",
                "task_spec": task_spec.model_dump(),
                "pipeline_trace": trace,
                "latency_ms": round(spec.quick_result.latency_ms, 1) if spec.quick_result else 0,
            }
        # Fall through to full pipeline if speculative says escalate
        return await _run_full(client, settings, prompt, task_spec, trace, policy)

    if policy.should_run_full_pipeline:
        trace.append(f"Full pipeline — {policy.route}")
        return await _run_full(client, settings, prompt, task_spec, trace, policy)

    # Default: full pipeline
    return await _run_full(client, settings, prompt, task_spec, trace, policy)


async def _run_full(
    client: RouterClient,
    settings: Settings,
    prompt: str,
    task_spec: TaskSpec,
    trace: list[str],
    policy=None,
) -> dict:
    from .pipeline.full_pipeline import run_full_pipeline

    if policy is not None:
        # Use agent configs from the PolicyEngine decision
        candidates_config = [
            cfg.to_dict() for cfg in (policy.solver_configs + policy.reviewer_configs)
        ]
        judge_config = policy.judge_config.to_dict() if policy.judge_config else {
            "model": settings.judge.model,
            "system_prompt": settings.judge.system_prompt,
            "temperature": settings.judge.temperature,
            "reasoning_effort": settings.judge.reasoning_effort,
        }
        synth_config = policy.synthesizer_config.to_dict() if policy.synthesizer_config else {
            "model": settings.synthesizer.model,
            "system_prompt": settings.synthesizer.system_prompt,
            "temperature": settings.synthesizer.temperature,
            "reasoning_effort": settings.synthesizer.reasoning_effort,
        }
        debate_rounds = policy.debate_rounds
        max_repair = policy.max_repair_rounds
        verif_timeout = policy.verification_timeout
    else:
        # Fallback: build from settings directly
        candidates_config = [
            {
                "name": c.name,
                "model": c.model,
                "system_prompt": c.system_prompt,
                "temperature": c.temperature,
                "reasoning_effort": c.reasoning_effort,
                "role": c.role,
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
        debate_rounds = settings.debate_rounds
        max_repair = settings.max_repair_rounds
        verif_timeout = settings.verifier_timeout

    result = await run_full_pipeline(
        client,
        prompt,
        candidates_config,
        judge_config,
        synth_config,
        debate_rounds=debate_rounds,
        max_repair_rounds=max_repair,
        verification_timeout=verif_timeout,
    )

    trace.extend(result.pipeline_trace)

    return {
        "answer": result.answer,
        "mode": "full",
        "task_spec": task_spec.model_dump(),
        "clusters": {
            "disagreement_level": result.clusters.disagreement_level if result.clusters else None,
            "num_clusters": len(result.clusters.clusters) if result.clusters else 0,
        },
        "judge": result.judge_result,
        "verification": {
            "status": result.verification_report.status if result.verification_report else None,
            "all_passed": result.verification_report.all_passed if result.verification_report else None,
            "has_failures": result.verification_report.has_failures if result.verification_report else None,
            "has_unverified": result.verification_report.has_unverified if result.verification_report else None,
            "failures": result.verification_report.failures if result.verification_report else [],
            "unverified": result.verification_report.unverified if result.verification_report else [],
        },
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
            "role": c.role,
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
