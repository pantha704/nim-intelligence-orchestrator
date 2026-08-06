from pathlib import Path

from .config import Settings
from .context import RunContext
from .policy import PolicyEngine
from .router_client import RouterClient
from .task_compiler import TaskCompilerResult, TaskSpec, compile_task
from .transport_gate import transport_gate


async def handle_intelligence_request(
    client: RouterClient,
    settings: Settings,
    raw_prompt: str,
    force_mode: str | None = None,
) -> dict:
    """Main entry point.

    One RunContext is created here and threaded through every stage
    (compiler, policy, speculative routing, full pipeline, verification).
    The API executes PolicyResult — all routing decisions live in PolicyEngine.
    """

    # --- Stage 0: Transport gate (minimal — size, encoding, emptiness only) ---
    gate = transport_gate(raw_prompt)
    if not gate.ok:
        return {
            "answer": "",
            "error": gate.reason,
            "mode": "rejected",
            "pipeline_trace": [f"REJECTED: {gate.reason}"],
        }

    # Single mutable execution state for the entire request
    ctx = RunContext(raw_prompt=gate.raw_prompt)
    ctx.start()
    engine = PolicyEngine(settings)

    # --- Stage 1: Policy decision (force modes + bypass don't need the spec) ---
    ctx.policy = engine.decide(ctx.raw_prompt, force_mode=force_mode)
    ctx.add_trace(f"Policy: action={ctx.policy.action}, route={ctx.policy.route}, reason={ctx.policy.reason}")

    if ctx.policy.action == "single":
        await engine.execute_single(ctx, client)
        ctx.finish()
        return ctx.to_response()

    # --- Stage 2: Task Compiler (bypass decision owned by policy) ---
    if ctx.policy.should_bypass_compiler:
        from .task_compiler import bypass_task_spec

        task_result = bypass_task_spec(ctx.raw_prompt)
        ctx.add_trace("Compiler bypassed — obvious simple query")
    else:
        # The compiler is a model call: it reserves a slot on the shared
        # execution budget through the same reservation path as every agent.
        from .router_client import BudgetExhaustedError, budgeted_call

        try:
            task_result = await budgeted_call(
                ctx,
                agent_name="task_compiler",
                model=settings.task_compiler.model,
                call_fn=lambda: compile_task(
                    client,
                    model=settings.task_compiler.model,
                    raw_prompt=ctx.raw_prompt,
                    timeout_seconds=settings.task_compiler.timeout_seconds,
                ),
                timeout=settings.task_compiler.timeout_seconds + 10,
            )
            ctx.add_trace(
                f"Task compiled: route={task_result.task_spec.recommended_route}, risk={task_result.task_spec.risk_level}, "
                f"subtasks={len(task_result.task_spec.subtasks)}, ambiguities={len(task_result.task_spec.ambiguities)} "
                f"[{task_result.latency_ms:.0f}ms]"
            )
        except (BudgetExhaustedError, TimeoutError):
            task_result = TaskCompilerResult(
                task_spec=TaskSpec(
                    objective=ctx.raw_prompt[:200],
                    context=ctx.raw_prompt,
                    recommended_route="complex",
                    risk_level="medium",
                    assumptions=["Budget exhausted — task compiler skipped"],
                ),
                needs_clarification=False,
                clarification_question="",
                raw_json="",
                latency_ms=0,
            )
            ctx.add_trace("Budget exhausted — task compiler skipped, defaulting to complex route")

    ctx.task_spec = task_result.task_spec

    # Ambiguity check — ask one question if a high-impact ambiguity needs clarification
    if task_result.needs_clarification:
        ctx.needs_clarification = True
        ctx.clarification_question = task_result.clarification_question
        ctx.mode = "needs_clarification"
        ctx.add_trace(f"Clarification needed: {ctx.clarification_question}")
        ctx.finish()
        return ctx.to_response()

    if ctx.task_spec.assumptions:
        ctx.add_trace(f"Assumptions: {'; '.join(ctx.task_spec.assumptions[:3])}")

    # --- Stage 3: Final policy decision with the compiled spec ---
    ctx.policy = engine.decide(ctx.raw_prompt, task_spec=ctx.task_spec, force_mode=force_mode)
    ctx.add_trace(f"Policy: action={ctx.policy.action}, route={ctx.policy.route}, reason={ctx.policy.reason}")

    if ctx.policy.action == "dag" or ctx.policy.use_dag:
        from .dag import DagValidationError, execute_dag

        ctx.add_trace(f"Adaptive DAG — {ctx.policy.route}")
        try:
            await execute_dag(client, ctx, settings.dag)
            ctx.finish()
            return ctx.to_response()
        except DagValidationError as e:
            ctx.add_trace(f"DAG invalid — falling back to fixed pipeline: {e}")
            ctx.policy.action = "full"
            ctx.policy.should_run_full_pipeline = True
        # fall through to the full pipeline

    if ctx.policy.action == "speculative":
        accepted = await engine.execute_speculative(ctx, client)
        if accepted:
            ctx.finish()
            return ctx.to_response()
        # Escalated to full pipeline (decision made in execute_speculative)

    if ctx.policy.action == "full" or ctx.policy.should_run_full_pipeline:
        from .pipeline.full_pipeline import run_full_pipeline

        ctx.add_trace(f"Full pipeline — {ctx.policy.route}")
        await run_full_pipeline(client, ctx)
        ctx.finish()
        return ctx.to_response()

    # Policy-owned default fallback: full pipeline
    from .pipeline.full_pipeline import run_full_pipeline

    ctx.add_trace("Fallback: full pipeline")
    await run_full_pipeline(client, ctx)
    ctx.finish()
    return ctx.to_response()


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
