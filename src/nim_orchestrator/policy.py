"""Central PolicyEngine: single decision-making authority for routing, agent selection, and verification.

The API layer executes PolicyResult — it never makes routing decisions itself.
Forced modes, compiler bypass, speculative routing and escalation all live here.
"""
from .agents import AgentConfig, AgentRole
from .config import Settings
from .context import PolicyResult
from .router_client import BudgetExhaustedError, budgeted_chat
from .task_compiler import should_bypass_compiler


def classify_task_type(prompt: str, answer: str = "") -> str:
    """Classify the task type from the prompt (and optionally the answer).

    This is the SINGLE source of truth for task type classification.
    No other module implements its own task classifier.
    """
    import re
    p = prompt.lower()

    if re.search(r"\b(?:write|implement|code|function|class|program|script)\b", p):
        return "verifiable"

    if re.search(r"\b(?:calculate|compute|solve|how much|how many)\b", p):
        return "verifiable"

    if re.search(r"\b(?:what is \d|what'?s \d)\b", p):
        return "direct"

    if re.search(r"\b(?:prove|design|architect|optimize|compare|trade-off|debug|refactor|analyze)\b", p):
        return "complex"

    if re.search(r"\b(?:what do you think|best|worst|favorite|recommend|advise|opinion|should i)\b", p):
        return "open_ended"

    if re.search(r"\b(?:what is|define|who is|when did|where is|capital of)\b", p):
        return "direct"

    return "complex"


class PolicyEngine:
    """Central policy authority. ALL routing decisions go through here."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def decide(self, raw_prompt: str, task_spec=None, force_mode: str | None = None) -> PolicyResult:
        """Produce a single PolicyResult capturing ALL routing decisions.

        The result's `action` field tells the API what to execute:
        "single" (one direct chat call), "speculative" (quick call with
        escalation), or "full" (the multi-agent pipeline).
        """
        result = PolicyResult()

        # Force modes — owned by PolicyEngine
        if force_mode == "single":
            result.action = "single"
            result.route = "direct"
            result.should_bypass_compiler = True
            result.should_speculate = True
            result.should_run_full_pipeline = False
            result.reason = "forced single mode"
            return result

        if force_mode == "full":
            result.action = "full"
            result.route = "complex"
            result.should_bypass_compiler = False
            result.should_speculate = False
            result.should_run_full_pipeline = True
            result.reason = "forced full mode"
            self._populate_agents(result)
            return result

        if force_mode == "dag":
            result.action = "dag"
            result.route = "complex"
            result.should_bypass_compiler = False
            result.should_speculate = False
            result.should_run_full_pipeline = False
            result.use_dag = True
            result.reason = "forced dag mode"
            self._populate_agents(result)
            return result

        # Compiler bypass — owned by PolicyEngine
        result.should_bypass_compiler = should_bypass_compiler(raw_prompt)

        if result.should_bypass_compiler:
            result.action = "speculative"
            result.route = "direct"
            result.should_speculate = True
            result.should_run_full_pipeline = False
            result.reason = "compiler bypassed — simple query"
            # Populate agents so escalation to the full pipeline can still run
            self._populate_agents(result)
            return result

        # Use TaskSpec route if available, otherwise classify
        if task_spec and task_spec.recommended_route:
            result.route = task_spec.recommended_route
        else:
            result.route = classify_task_type(raw_prompt)

        if result.route == "direct":
            result.action = "speculative"
            result.should_speculate = True
            result.should_run_full_pipeline = False
            result.reason = f"direct route (route={result.route})"
        else:
            result.action = "full"
            result.should_speculate = False
            result.should_run_full_pipeline = True
            result.reason = f"{result.route} route — full pipeline"

        # Adaptive DAG (Phase 4.0): only when enabled in config AND the task
        # compiler captured subtasks. The fixed pipeline stays the default
        # until the DAG beats it on benchmarks (Phase 4.3).
        if (
            self.settings.dag.enabled
            and task_spec is not None
            and task_spec.subtasks
            and result.route != "direct"
        ):
            result.use_dag = True
            result.action = "dag"
            result.should_run_full_pipeline = False
            result.reason = f"{result.route} route — adaptive DAG ({len(task_spec.subtasks)} subtasks)"

        self._populate_agents(result)
        return result

    async def execute_single(self, ctx, client) -> None:
        """Forced single mode — one budgeted direct chat call."""
        try:
            result = await budgeted_chat(
                client,
                ctx,
                agent_name="single",
                model=self.settings.primary_model,
                messages=[{"role": "user", "content": ctx.raw_prompt}],
                temperature=0.2,
                max_tokens=256,
                timeout=30,
            )
            ctx.answer = result.content
            ctx.mode = "single"
            ctx.total_latency_ms = result.latency_ms
            ctx.add_trace("forced single mode — direct chat call")
        except BudgetExhaustedError as e:
            ctx.mode = "error"
            ctx.add_trace(f"single mode skipped: {e}")

    async def execute_speculative(self, ctx, client) -> bool:
        """Run the speculative quick call.

        Returns True when the direct answer is accepted (ctx.answer/mode set),
        False when it must escalate to the full pipeline. The escalation
        decision is made here, not in the API.
        """
        from .speculative_router import speculative_route

        spec = await speculative_route(
            client,
            model=self.settings.primary_model,
            prompt=ctx.raw_prompt,
            max_quick_tokens=256,
            ctx=ctx,
        )
        ctx.add_trace(f"Speculative route: {spec.reason} (route={spec.route})")

        if spec.escalate:
            ctx.add_trace("Speculative route: escalating to full pipeline")
            return False

        ctx.answer = spec.quick_answer
        ctx.mode = "direct"
        ctx.total_latency_ms = spec.quick_result.latency_ms if spec.quick_result else 0
        ctx.add_trace(f"Speculative route: accepted direct answer ({ctx.total_latency_ms:.0f}ms)")
        return True

    def _populate_agents(self, result: PolicyResult) -> None:
        """Split settings.candidates into solvers and reviewers by AgentRole.

        No name-based inference anywhere — the role field is required on
        CandidateConfig and Literal-validated at config load time.
        """
        for c in self.settings.candidates:
            role = AgentRole(c.role)

            cfg = AgentConfig(
                name=c.name,
                role=role,
                model=c.model,
                system_prompt=c.system_prompt,
                temperature=c.temperature,
                reasoning_effort=c.reasoning_effort,
                max_tokens=1024,
                timeout_seconds=30,
            )

            if role.is_solver:
                result.solver_configs.append(cfg)
            elif role.is_reviewer:
                result.reviewer_configs.append(cfg)
            elif role.is_judge:
                result.judge_config = cfg
            elif role.is_synthesizer:
                result.synthesizer_config = cfg

        if not result.solver_configs:
            # No solvers configured — treat all candidates as solvers
            for c in self.settings.candidates:
                cfg = AgentConfig(
                    name=c.name,
                    role=AgentRole.SOLVER,
                    model=c.model,
                    system_prompt=c.system_prompt,
                    temperature=c.temperature,
                    reasoning_effort=c.reasoning_effort,
                )
                result.solver_configs.append(cfg)

        if result.judge_config is None and self.settings.judge:
            result.judge_config = AgentConfig(
                name="judge",
                role=AgentRole.JUDGE,
                model=self.settings.judge.model,
                system_prompt=self.settings.judge.system_prompt,
                temperature=self.settings.judge.temperature,
                reasoning_effort=self.settings.judge.reasoning_effort,
            )

        if result.synthesizer_config is None and self.settings.synthesizer:
            result.synthesizer_config = AgentConfig(
                name="synthesizer",
                role=AgentRole.SYNTHESIZER,
                model=self.settings.synthesizer.model,
                system_prompt=self.settings.synthesizer.system_prompt,
                temperature=self.settings.synthesizer.temperature,
                reasoning_effort=self.settings.synthesizer.reasoning_effort,
            )

        result.debate_rounds = self.settings.debate_rounds
        result.max_repair_rounds = self.settings.max_repair_rounds
        result.verification_timeout = self.settings.verifier_timeout
