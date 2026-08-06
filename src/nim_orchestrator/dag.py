"""Phase 4.0 — Adaptive specialist DAG (MVP).

Executes the subtasks already captured inside TaskSpec as a dependency-aware
graph. Each node receives its objective, relevant context (including outputs
of its dependencies), acceptance criteria, assigned specialist and
verification plan.

Limits (Phase 4.0):
- 1 primary agent per node
- maximum 1 alternate agent after failure
- budget enforced through the shared ExecutionBudget (budgeted_call)

The fixed pipeline remains the baseline and fallback; the DAG only runs
when PolicyEngine says so (config gate + subtasks present).
"""
import asyncio
from dataclasses import dataclass, field

from .agents import AgentConfig, AgentRole
from .clustering import Candidate
from .config import DagConfig
from .context import RunContext
from .router_client import RouterClient, budgeted_call
from .task_compiler import TaskSpec
from .verifiers.external_checks import VerificationReport, verify_answer


@dataclass
class DAGNode:
    """A single subtask in the adaptive DAG."""
    id: str
    objective: str
    depends_on: list[str] = field(default_factory=list)
    acceptance_criteria: str = ""
    verification_plan: list[str] = field(default_factory=list)
    result: str = ""
    verification: VerificationReport | None = None
    status: str = "pending"  # pending | passed | failed | skipped
    attempts: int = 0
    alternates_used: int = 0
    error: str = ""


def build_dag(task_spec: TaskSpec) -> list[DAGNode]:
    """Build DAG nodes from TaskSpec subtasks, topologically sorted.

    Unknown dependency ids are dropped; the visited set keeps the sort from
    hanging on cycles.
    """
    subtasks = {s.id: s for s in task_spec.subtasks}
    nodes = {
        s.id: DAGNode(
            id=s.id,
            objective=s.description,
            depends_on=[d for d in s.depends_on if d in subtasks],
            acceptance_criteria=s.acceptance_criteria,
        )
        for s in task_spec.subtasks
    }

    visited: set[str] = set()
    order: list[str] = []

    def visit(nid: str) -> None:
        if nid in visited:
            return
        visited.add(nid)
        for dep in nodes[nid].depends_on:
            visit(dep)
        order.append(nid)

    for s in task_spec.subtasks:
        visit(s.id)

    return [nodes[nid] for nid in order]


def _node_prompt(node: DAGNode, context_text: str) -> str:
    criteria = node.acceptance_criteria or "Answer the objective correctly and completely."
    plan = "\n".join(f"- {p}" for p in node.verification_plan) or "- Verify the output yourself before finishing."
    return f"""Objective: {node.objective}

Relevant context:
{context_text}

Acceptance criteria:
{criteria}

Verification plan:
{plan}

Produce the complete output for this objective. Do not reference this prompt in the output."""


async def _run_attempt(
    client: RouterClient,
    ctx: RunContext,
    node: DAGNode,
    cfg: AgentConfig,
    context_text: str,
    attempt_label: str,
) -> str:
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": _node_prompt(node, context_text)},
    ]
    try:
        result = await budgeted_call(
            ctx,
            agent_name=f"dag:{node.id}:{attempt_label}",
            model=cfg.model,
            call_fn=lambda: client.chat(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                reasoning_effort=cfg.reasoning_effort,
                max_tokens=cfg.max_tokens,
            ),
            timeout=cfg.timeout_seconds,
        )
        return result.content
    except Exception:
        return ""


async def execute_node(
    client: RouterClient,
    ctx: RunContext,
    node: DAGNode,
    dag_cfg: DagConfig,
    context_text: str,
) -> None:
    """Run 1 primary agent; on verification failure, up to max_alternates.

    Node passes when verification reports no failures. Unverified items do
    not trigger an alternate (consistent with the fixed pipeline's repair
    semantics — repair only on failures).
    """
    solvers = ctx.policy.solver_configs
    primary = solvers[0] if solvers else AgentConfig(
        name="solver", role=AgentRole.SOLVER, model=dag_cfg.primary_model,
        system_prompt="You are a precise problem solver. Solve the objective rigorously.",
        timeout_seconds=dag_cfg.timeout_seconds,
    )
    alternates = solvers[1:] or []

    attempts_left = 1 + min(dag_cfg.max_alternates, len(alternates))
    attempt_label = "primary"
    cfg = primary

    while attempts_left > 0 and node.status != "passed":
        attempts_left -= 1
        node.attempts += 1
        content = await _run_attempt(client, ctx, node, cfg, context_text, attempt_label)

        if not content:
            node.status = "failed"
            node.error = "call failed or budget exhausted"
            ctx.add_trace(f"DAG node {node.id} attempt {attempt_label}: failed ({node.error})")
            return

        report = await verify_answer(content, node.objective, ctx.policy.verification_timeout)
        node.result = content
        node.verification = report

        if report.has_failures:
            ctx.add_trace(
                f"DAG node {node.id} attempt {attempt_label}: verification FAILED — {report.failures}"
            )
            if attempts_left > 0:
                attempt_label = "alternate"
                cfg = alternates[min(node.alternates_used, len(alternates) - 1)]
                node.alternates_used += 1
            else:
                node.status = "failed"
                node.error = "; ".join(report.failures)
        else:
            node.status = "passed"
            ctx.add_trace(
                f"DAG node {node.id} attempt {attempt_label}: PASSED "
                f"({report.status}, {len(report.results)} checks)"
            )


async def synthesize_dag_outputs(client: RouterClient, ctx: RunContext, nodes: list[DAGNode]) -> str:
    """Global synthesis: combine node outputs into the final answer."""
    node_text = "\n\n".join(
        f"--- {n.id}: {n.objective} ---\n{n.result[:2000]}"
        for n in nodes if n.result
    )
    synth_cfg = ctx.policy.synthesizer_config
    if synth_cfg is None:
        ctx.add_trace("DAG synthesis skipped — no synthesizer configured")
        return node_text

    synth_prompt = f"""Original problem: {ctx.raw_prompt}

The task was decomposed into subtasks. Here are the verified outputs of each subtask:

{node_text}

Combine these into one coherent final answer to the original problem.
Do not invent results for subtasks that failed or are missing.
Note any subtask that failed or was skipped."""

    messages = [
        {"role": "system", "content": synth_cfg.system_prompt},
        {"role": "user", "content": synth_prompt},
    ]
    try:
        result = await budgeted_call(
            ctx,
            agent_name="dag_synthesizer",
            model=synth_cfg.model,
            call_fn=lambda: client.chat(
                model=synth_cfg.model,
                messages=messages,
                temperature=synth_cfg.temperature,
                reasoning_effort=synth_cfg.reasoning_effort,
                max_tokens=synth_cfg.max_tokens,
            ),
            timeout=synth_cfg.timeout_seconds,
        )
        return result.content
    except Exception:
        ctx.add_trace("DAG synthesis failed — using raw node outputs")
        return node_text


async def execute_dag(client: RouterClient, ctx: RunContext, dag_cfg: DagConfig) -> None:
    """Execute the TaskSpec subtask DAG, mutating ctx with the final answer.

    Sequential over nodes in topological order. The shared ExecutionBudget
    (with the DAG's limits) bounds total model calls and concurrency.
    """
    if ctx.task_spec is None or not ctx.task_spec.subtasks:
        ctx.add_trace("DAG skipped — no subtasks in task spec")
        ctx.mode = "dag"
        ctx.answer = "No subtasks to execute."
        return

    # The DAG runs under its own limits on the shared budget. The compiler
    # call already counted, so max_model_calls bounds the whole request.
    from .budget import BudgetLimits

    ctx.budget.limits = BudgetLimits(
        max_model_calls=dag_cfg.max_model_calls,
        max_concurrent_agents=dag_cfg.max_concurrent_calls,
        max_total_agents=ctx.budget.limits.max_total_agents,
    )
    ctx.budget._semaphore = asyncio.Semaphore(dag_cfg.max_concurrent_calls)

    nodes = build_dag(ctx.task_spec)
    ctx.add_trace(f"DAG: {len(nodes)} subtask(s) in order: {', '.join(n.id for n in nodes)}")

    results: dict[str, str] = {}
    for node in nodes:
        dep_text = "\n\n".join(
            f"--- {d} ---\n{results[d][:1000]}" for d in node.depends_on if d in results
        )
        context_text = f"Original problem: {ctx.raw_prompt}\n\n{dep_text}".rstrip()
        await execute_node(client, ctx, node, dag_cfg, context_text)
        results[node.id] = node.result

    ctx.candidates = [
        Candidate(
            name=f"dag:{node.id}",
            model="dag",
            content=node.result,
            error=node.error if node.status == "failed" else "",
        )
        for node in nodes
    ]
    ctx.add_trace(
        f"DAG done: {sum(1 for n in nodes if n.status == 'passed')} passed, "
        f"{sum(1 for n in nodes if n.status == 'failed')} failed"
    )

    ctx.answer = await synthesize_dag_outputs(client, ctx, nodes)
    ctx.mode = "dag"

    # Final verification of the composed answer
    ctx.verification = await verify_answer(ctx.answer, ctx.raw_prompt, ctx.policy.verification_timeout)
    ctx.add_trace(f"DAG final verification: {ctx.verification.status} ({len(ctx.verification.results)} checks)")
