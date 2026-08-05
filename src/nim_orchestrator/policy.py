"""Central PolicyEngine: single decision-making authority for routing, agent selection, and verification."""
from .agents import AgentConfig, AgentRole
from .config import Settings
from .context import PolicyResult
from .task_compiler import should_bypass_compiler


def classify_task_type(prompt: str, answer: str = "") -> str:
    """Classify the task type from the prompt (and optionally the answer).

    This is the SINGLE source of truth for task type classification.
    Replaces the duplicated _detect_task_type in speculative_router and
    the keyword lists in difficulty_router.
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
    """Central policy authority. All routing decisions go through here."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def decide(self, raw_prompt: str, task_spec=None, force_mode: str | None = None) -> PolicyResult:
        """Produce a single PolicyResult capturing ALL routing decisions."""
        result = PolicyResult()

        # Force modes
        if force_mode == "single":
            result.route = "direct"
            result.should_bypass_compiler = True
            result.should_speculate = True
            result.should_run_full_pipeline = False
            result.reason = "forced single mode"
            return result

        if force_mode == "full":
            result.route = "complex"
            result.should_bypass_compiler = False
            result.should_speculate = False
            result.should_run_full_pipeline = True
            result.reason = "forced full mode"
            self._populate_agents(result)
            return result

        # Compiler bypass check
        result.should_bypass_compiler = should_bypass_compiler(raw_prompt)

        if result.should_bypass_compiler:
            result.route = "direct"
            result.should_speculate = True
            result.should_run_full_pipeline = False
            result.reason = "compiler bypassed — simple query"
            return result

        # Use TaskSpec route if available
        if task_spec and task_spec.recommended_route:
            result.route = task_spec.recommended_route
        else:
            result.route = classify_task_type(raw_prompt)

        if result.route == "direct":
            result.should_speculate = True
            result.should_run_full_pipeline = False
            result.reason = f"direct route (route={result.route})"
        else:
            result.should_speculate = False
            result.should_run_full_pipeline = True
            result.reason = f"{result.route} route — full pipeline"

        self._populate_agents(result)
        return result

    def _populate_agents(self, result: PolicyResult) -> None:
        """Split settings.candidates into solvers and reviewers by AgentRole."""
        for c in self.settings.candidates:
            role_str = getattr(c, "role", None)
            if role_str is None:
                # Fallback: infer from name (temporary during migration)
                name_lower = c.name.lower()
                if "critic" in name_lower:
                    role = AgentRole.CRITIC
                elif "evidence" in name_lower or "verifier" in name_lower:
                    role = AgentRole.EVIDENCE_VERIFIER
                elif "devil" in name_lower:
                    role = AgentRole.DEVILS_ADVOCATE
                elif "alternative" in name_lower or "alt" in name_lower:
                    role = AgentRole.ALTERNATIVE_SOLVER
                else:
                    role = AgentRole.SOLVER
            else:
                role = AgentRole(role_str) if isinstance(role_str, str) else role_str

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
            for c in self.settings.candidates:
                role = AgentRole.SOLVER
                cfg = AgentConfig(
                    name=c.name,
                    role=role,
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
