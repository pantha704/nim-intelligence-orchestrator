"""Full pipeline: solvers → critique → debate → judge → verification → synthesis.

All execution state lives in and is mutated on the RunContext. Every model
call goes through budgeted_chat, which enforces the ExecutionBudget
(calls, time, concurrency, agent count).
"""
import asyncio
import json
import time
from dataclasses import dataclass, field

from ..agents import AgentRole
from ..clustering import Candidate, ClusteringResult, cluster_candidates
from ..context import AnonMapping, RunContext
from ..context import create_anon_mapping as _create_anon_mapping
from ..router_client import BudgetExhaustedError, RouterClient, budgeted_chat
from ..verifiers.external_checks import VerificationReport, verify_answer


@dataclass
class PipelineResult:
    answer: str
    mode: str = "single"
    candidates: list[Candidate] = field(default_factory=list)
    clusters: ClusteringResult | None = None
    judge_result: dict | None = None
    verification_report: VerificationReport | None = None
    total_latency_ms: float = 0
    total_tokens_estimate: int = 0
    pipeline_trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "candidates": [
                {"name": c.name, "model": c.model, "content": c.content[:500], "error": c.error}
                for c in self.candidates
            ],
            "clusters": {
                "disagreement_level": self.clusters.disagreement_level if self.clusters else None,
                "num_clusters": len(self.clusters.clusters) if self.clusters else 0,
            },
            "judge": self.judge_result,
            "verification": {
                "status": self.verification_report.status if self.verification_report else None,
                "all_passed": self.verification_report.all_passed if self.verification_report else None,
                "has_failures": self.verification_report.has_failures if self.verification_report else None,
                "has_unverified": self.verification_report.has_unverified if self.verification_report else None,
                "failures": self.verification_report.failures if self.verification_report else [],
                "unverified": self.verification_report.unverified if self.verification_report else [],
            },
            "total_latency_ms": round(self.total_latency_ms, 1),
            "pipeline_trace": self.pipeline_trace,
        }


def create_anon_mapping(candidates: list[Candidate]) -> AnonMapping:
    """Create anonymized mapping. Delegates to context.create_anon_mapping."""
    return _create_anon_mapping(candidates)


async def generate_candidates(client: RouterClient, ctx: RunContext) -> list[Candidate]:
    """Generate solver candidates. Each solver call is budget-enforced."""
    solver_configs = ctx.policy.solver_configs
    user_prompt = ctx.raw_prompt

    async def _gen(cfg):
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result = await budgeted_chat(
                client,
                ctx,
                agent_name=cfg.name,
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                reasoning_effort=cfg.reasoning_effort,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout_seconds,
            )
            return Candidate(
                name=cfg.name,
                model=cfg.model,
                content=result.content,
                reasoning=result.reasoning,
                latency_ms=result.latency_ms,
            )
        except BudgetExhaustedError as e:
            return Candidate(name=cfg.name, model=cfg.model, content="", error=str(e)[:300])
        except Exception as e:
            return Candidate(name=cfg.name, model=cfg.model, content="", error=str(e)[:300])

    tasks = [_gen(cfg) for cfg in solver_configs]
    results = await asyncio.gather(*tasks)

    ctx.add_trace(f"Generated {len(results)} candidates from {len(solver_configs)} agents")
    for c in results:
        status = "OK" if not c.error else f"ERROR: {c.error[:80]}"
        ctx.add_trace(f"  {c.name} ({c.model}): {status} [{c.latency_ms:.0f}ms]")

    ctx.candidates = results
    return results


async def critique_candidates(client: RouterClient, ctx: RunContext) -> dict:
    """Run adversarial critic, evidence verifier, and devil's advocate.

    Reviewers are assigned by explicit AgentRole from the policy — never
    inferred from names.
    """
    anon = ctx.anon or create_anon_mapping(ctx.candidates)
    ctx.anon = anon

    if not anon.shuffled:
        ctx.add_trace("Critique: no valid candidates to critique")
        ctx.critique = {}
        return {}

    candidate_text = anon.anon_text(2000)

    critic_config = None
    evidence_config = None
    devil_config = None
    for cfg in ctx.policy.reviewer_configs:
        if cfg.role == AgentRole.CRITIC and critic_config is None:
            critic_config = cfg
        elif cfg.role == AgentRole.EVIDENCE_VERIFIER and evidence_config is None:
            evidence_config = cfg
        elif cfg.role == AgentRole.DEVILS_ADVOCATE and devil_config is None:
            devil_config = cfg

    results = {}

    async def _run_reviewer(key, config, instruction):
        if not config:
            return key, ""
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": f"""Original problem: {ctx.raw_prompt}

Candidate answers:
{candidate_text}

{instruction}"""},
        ]
        try:
            result = await budgeted_chat(
                client,
                ctx,
                agent_name=key,
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                reasoning_effort=config.reasoning_effort,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds,
            )
            return key, result.content
        except BudgetExhaustedError as e:
            ctx.add_trace(f"{key} skipped: {e}")
            return key, ""
        except Exception as e:
            ctx.add_trace(f"{key} error: {str(e)[:100]}")
            return key, ""

    tasks = [
        _run_reviewer("critic", critic_config,
                      "Identify weaknesses, errors, and gaps in these candidate answers. "
                      "Refer to candidates by their anonymous labels (Candidate A, Candidate B, etc.)."),
        _run_reviewer("evidence_verifier", evidence_config,
                      "Extract factual claims from these answers and classify each as: "
                      "VERIFIED, UNVERIFIABLE, or CONTRADICTED. "
                      "Refer to candidates by their anonymous labels."),
        _run_reviewer("devil_advocate", devil_config,
                      "Argue the opposite conclusion from the mainstream answer among these candidates. "
                      "Construct the strongest possible counterargument. "
                      "Refer to candidates by their anonymous labels."),
    ]
    task_results = await asyncio.gather(*tasks)
    for key, content in task_results:
        results[key] = content

    active = [k for k in ("critic", "evidence_verifier", "devil_advocate") if results.get(k)]
    ctx.add_trace(f"Critique done: {', '.join(active) if active else 'none'}")
    ctx.critique = results
    return results


async def debate_disagreeing_candidates(client: RouterClient, ctx: RunContext) -> list[Candidate]:
    if ctx.clustering is None or ctx.clustering.disagreement_level != "high":
        ctx.add_trace(
            f"Disagreement level: {ctx.clustering.disagreement_level if ctx.clustering else 'none'} — skipping debate"
        )
        return ctx.candidates

    if ctx.policy.debate_rounds < 1:
        return ctx.candidates

    sorted_clusters = sorted(ctx.clustering.clusters, key=lambda c: len(c), reverse=True)
    top_clusters = sorted_clusters[:3]

    updated = list(ctx.candidates)

    for round_num in range(ctx.policy.debate_rounds):
        ctx.add_trace(f"Debate round {round_num + 1}")

        async def _debate_one(cluster):
            rep = cluster[0]
            other_answers = "\n\n".join(
                f"--- {c[0].name} ---\n{c[0].content[:1000]}"
                for c in top_clusters if c[0].name != rep.name
            )

            debate_prompt = f"""Original problem: {ctx.raw_prompt}

Your current answer:
{rep.content[:1000]}

Other candidates proposed:
{other_answers}

You disagree with the other candidates. Identify the SPECIFIC error in their reasoning.
Then decide if your answer is still correct, or if the other candidates found something you missed.
Revised answer (show reasoning):"""

            messages = [
                {"role": "system", "content": f"You are {rep.name}. You must defend or revise your answer based on peer critique."},
                {"role": "user", "content": debate_prompt},
            ]

            try:
                result = await budgeted_chat(
                    client,
                    ctx,
                    agent_name=f"debate:{rep.name}",
                    model=rep.model,
                    messages=messages,
                    temperature=0.3,
                    reasoning_effort="none",
                    max_tokens=1024,
                    timeout=30,
                )
                return rep.name, Candidate(
                    name=rep.name,
                    model=rep.model,
                    content=result.content,
                    reasoning=result.reasoning,
                    latency_ms=result.latency_ms,
                )
            except BudgetExhaustedError as e:
                ctx.add_trace(f"  {rep.name} debate skipped: {e}")
                return rep.name, None
            except Exception as e:
                ctx.add_trace(f"  {rep.name} debate error: {str(e)[:100]}")
                return rep.name, None

        results = await asyncio.gather(*[_debate_one(c) for c in top_clusters])
        for name, new_cand in results:
            if new_cand is not None:
                for i, c in enumerate(updated):
                    if c.name == name:
                        updated[i] = new_cand
                        break

    return updated


def _extract_json_from_response(content: str) -> dict | None:
    """Extract JSON from model response — handles plain, fenced, and multiline."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    import re
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)```", content, re.DOTALL)
    for block in fenced:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    depth = 0
    start = -1
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(content[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
    return None


async def judge_candidates(client: RouterClient, ctx: RunContext) -> dict:
    anon = ctx.anon or create_anon_mapping(ctx.candidates)
    ctx.anon = anon

    if len(anon.shuffled) <= 1:
        ctx.add_trace("Only 1 valid candidate — judge not needed")
        winner_name = anon.shuffled[0].name if anon.shuffled else ""
        return {"winner": winner_name, "rankings": [], "confidence": 1.0, "disagreement_level": "none"}

    judge_config = ctx.policy.judge_config
    if judge_config is None:
        ctx.add_trace("No judge configured — using first candidate")
        return {"winner": anon.shuffled[0].name, "rankings": [], "confidence": 0.5, "disagreement_level": "none"}

    candidate_summaries = anon.anon_text(2000)

    critique_section = ""
    if ctx.critique and (ctx.critique.get("critic") or ctx.critique.get("evidence_verifier") or ctx.critique.get("devil_advocate")):
        critique_section = "\n\n--- Adversarial Critique ---\n"
        if ctx.critique.get("critic"):
            critique_section += f"Critic findings:\n{ctx.critique['critic'][:1000]}\n"
        if ctx.critique.get("evidence_verifier"):
            critique_section += f"Evidence verification:\n{ctx.critique['evidence_verifier'][:1000]}\n"
        if ctx.critique.get("devil_advocate"):
            critique_section += f"Devil's advocate counterargument:\n{ctx.critique['devil_advocate'][:1000]}\n"
        critique_section += "\nConsider these critiques when evaluating candidates."

    judge_prompt = f"""Problem: {ctx.raw_prompt}

Here are {len(anon.shuffled)} candidate solutions:

{candidate_summaries}
{critique_section}

Evaluate each candidate using your rubric. If the candidates agree, state that.
If they disagree, identify the key disagreement point and who is right.

Refer to candidates by their anonymous labels (Candidate A, Candidate B, etc.).

Respond ONLY with valid JSON."""

    messages = [
        {"role": "system", "content": judge_config.system_prompt},
        {"role": "user", "content": judge_prompt},
    ]

    try:
        result = await budgeted_chat(
            client,
            ctx,
            agent_name="judge",
            model=judge_config.model,
            messages=messages,
            temperature=judge_config.temperature,
            reasoning_effort=judge_config.reasoning_effort,
            max_tokens=judge_config.max_tokens,
            timeout=judge_config.timeout_seconds,
        )

        parsed = _extract_json_from_response(result.content)
        if parsed is None:
            parsed = {"winner": anon.labels[0], "rankings": [], "confidence": 0.5,
                      "disagreement_level": "unknown", "raw": result.content[:500]}

        winner_label = parsed.get("winner", "")
        parsed["winner"] = anon.original_of(winner_label)

        if "rankings" in parsed and isinstance(parsed["rankings"], list):
            parsed["rankings"] = [
                anon.original_of(r) if isinstance(r, str) else r
                for r in parsed["rankings"]
            ]

        ctx.add_trace(f"Judge: winner={parsed.get('winner', '?')}, confidence={parsed.get('confidence', '?')}")
        return parsed

    except BudgetExhaustedError as e:
        ctx.add_trace(f"Judge skipped: {e}")
        fallback_winner = anon.shuffled[0].name if anon.shuffled else ""
        return {"winner": fallback_winner, "rankings": [], "confidence": 0.0, "disagreement_level": "budget_exhausted"}
    except Exception as e:
        ctx.add_trace(f"Judge error: {str(e)[:100]}")
        fallback_winner = anon.shuffled[0].name if anon.shuffled else ""
        return {"winner": fallback_winner, "rankings": [], "confidence": 0.0, "disagreement_level": "error"}


async def synthesize_final(client: RouterClient, ctx: RunContext) -> str:
    synthesizer_config = ctx.policy.synthesizer_config
    winner = ctx.winner
    if winner is None:
        ctx.add_trace("Synthesis skipped — no winner")
        return ctx.answer

    verification_report = ctx.verification
    failures = verification_report.failures if verification_report else []
    verification_summary = failure_text = unverified_text = ""
    if verification_report:
        if verification_report.has_failures:
            failure_text = "\n".join(f"- {f}" for f in failures)
        elif verification_report.has_unverified:
            unverified_text = "\n".join(f"- {u}" for u in verification_report.unverified)
            verification_summary = f"External checks: some items UNVERIFIED:\n{unverified_text}\nAnswer has not been fully verified.\nAggregate status: {verification_report.status}"
        else:
            verification_summary = f"All external checks passed ({len(verification_report.results)} checks)"

    judge_summary = ""
    if ctx.judge_result:
        judge_summary = json.dumps(ctx.judge_result, indent=2)[:1000]

    critique_section = ""
    if ctx.critique and (ctx.critique.get("critic") or ctx.critique.get("evidence_verifier") or ctx.critique.get("devil_advocate")):
        winner_label = ctx.anon.label_of(winner) if ctx.anon else "winner"
        critique_section = "\n\nAdversarial critique:\n"
        if ctx.critique.get("critic"):
            critique_section += f"Critic findings (re: {winner_label}):\n{ctx.critique['critic'][:1000]}\n"
        if ctx.critique.get("evidence_verifier"):
            critique_section += f"Evidence (re: {winner_label}): {ctx.critique['evidence_verifier'][:1000]}\n"
        if ctx.critique.get("devil_advocate"):
            critique_section += f"Devil's advocate: {ctx.critique['devil_advocate'][:1000]}\n"

    verification_section = (
        f"The following checks FAILED and must be fixed:\n{failure_text}"
        if failures
        else verification_summary
    )

    synth_prompt = f"""Original problem: {ctx.raw_prompt}

The winning candidate's answer:
{winner.content[:3000]}

Judge evaluation:
{judge_summary}

External verification:
{verification_section}
{critique_section}

If external checks failed, repair ONLY the specific failures. Do not rewrite content that passed verification.
If items are UNVERIFIED, note the uncertainty but do not rewrite unless there is a specific failure."""

    messages = [
        {"role": "system", "content": synthesizer_config.system_prompt},
        {"role": "user", "content": synth_prompt},
    ]

    answer = winner.content

    needs_repair = verification_report and verification_report.has_failures
    needs_synthesis = verification_report and verification_report.all_passed and not verification_report.has_unverified

    async def _synthesize_once() -> str:
        result = await budgeted_chat(
            client,
            ctx,
            agent_name="synthesizer",
            model=synthesizer_config.model,
            messages=messages,
            temperature=synthesizer_config.temperature,
            reasoning_effort=synthesizer_config.reasoning_effort,
            max_tokens=synthesizer_config.max_tokens,
            timeout=synthesizer_config.timeout_seconds,
        )
        return result.content

    if needs_synthesis:
        ctx.add_trace(f"Verification passed — running final synthesis ({winner.name})")
        try:
            answer = await _synthesize_once()
            ctx.add_trace("Synthesis done")
        except BudgetExhaustedError as e:
            ctx.add_trace(f"Synthesis skipped (keeping winner): {e}")
        except Exception as e:
            ctx.add_trace(f"Synthesis error (keeping winner): {str(e)[:100]}")
    elif needs_repair:
        for round_num in range(ctx.policy.max_repair_rounds if failures else 1):
            try:
                answer = await _synthesize_once()
                ctx.add_trace(f"Synthesis/repair round {round_num + 1}: done")

                repair_verif = await verify_answer(answer, ctx.raw_prompt)
                if not repair_verif.has_failures:
                    ctx.add_trace("Repair successful — no failures remain")
                    break
                else:
                    ctx.add_trace(f"Repair round {round_num + 1} still has failures: {repair_verif.failures}")
                    prev_failures = set(failures)
                    current_failures = set(repair_verif.failures)
                    if prev_failures == current_failures:
                        ctx.add_trace("Same failures as before — stopping repair")
                        break
            except BudgetExhaustedError as e:
                ctx.add_trace(f"Synthesis repair skipped (keeping winner): {e}")
                break
            except Exception as e:
                ctx.add_trace(f"Synthesis error: {str(e)[:100]}")
                break
    else:
        ctx.add_trace(f"Running synthesis with unverified items ({winner.name})")
        try:
            answer = await _synthesize_once()
            ctx.add_trace("Synthesis done")
        except BudgetExhaustedError as e:
            ctx.add_trace(f"Synthesis skipped (keeping winner): {e}")
        except Exception as e:
            ctx.add_trace(f"Synthesis error (keeping winner): {str(e)[:100]}")

    return answer


async def run_full_pipeline(client: RouterClient, ctx: RunContext) -> PipelineResult:
    """Run the full pipeline against the RunContext, mutating it as it goes."""
    t0 = time.monotonic()

    solver_configs = ctx.policy.solver_configs
    reviewer_configs = ctx.policy.reviewer_configs

    if not solver_configs:
        ctx.add_trace("No solvers configured — nothing to run")
        ctx.mode = "full"
        ctx.answer = "No solvers configured."
        return PipelineResult(
            answer=ctx.answer, mode="full",
            total_latency_ms=(time.monotonic() - t0) * 1000,
            pipeline_trace=list(ctx.trace),
        )

    ctx.add_trace(f"Starting full pipeline: {len(solver_configs)} solvers, {len(reviewer_configs)} reviewers")

    await generate_candidates(client, ctx)

    ctx.anon = create_anon_mapping(ctx.candidates)
    ctx.add_trace(f"Anonymized {len(ctx.anon.shuffled)} candidates: " +
                  ", ".join(f"{ctx.anon.labels[i]}={ctx.anon.shuffled[i].name}" for i in range(len(ctx.anon.shuffled))))

    await critique_candidates(client, ctx)

    ctx.clustering = cluster_candidates(ctx.candidates)
    ctx.add_trace(f"Clustering: {len(ctx.clustering.clusters)} cluster(s), disagreement={ctx.clustering.disagreement_level}")

    if ctx.clustering.disagreement_level == "high":
        ctx.candidates = await debate_disagreeing_candidates(client, ctx)
        # Persistent anon IDs: update candidates in-place, do NOT re-create mapping.
        ctx.anon.update_candidates(ctx.candidates)
        ctx.clustering = cluster_candidates(ctx.candidates)
        ctx.add_trace(f"Post-debate clustering: {len(ctx.clustering.clusters)} cluster(s), disagreement={ctx.clustering.disagreement_level}")

    ctx.judge_result = await judge_candidates(client, ctx)

    winner_name = ctx.judge_result.get("winner", "")
    ctx.winner = next(
        (c for c in ctx.candidates if c.name == winner_name and not c.error), None
    )
    if ctx.winner is None:
        valid = [c for c in ctx.candidates if not c.error and c.content]
        if ctx.clustering and ctx.clustering.leader:
            ctx.winner = ctx.clustering.leader
        elif valid:
            ctx.winner = valid[0]
        else:
            ctx.mode = "full"
            ctx.answer = "All candidates failed."
            ctx.add_trace("No valid candidates — all failed")
            return PipelineResult(
                answer=ctx.answer, mode="full", candidates=ctx.candidates,
                total_latency_ms=(time.monotonic() - t0) * 1000,
                pipeline_trace=list(ctx.trace),
            )

    ctx.verification = await verify_answer(ctx.winner.content, ctx.raw_prompt, ctx.policy.verification_timeout)
    ctx.add_trace(f"Verification: {ctx.verification.status} ({len(ctx.verification.results)} checks)")

    ctx.answer = await synthesize_final(client, ctx)
    ctx.mode = "full"

    return PipelineResult(
        answer=ctx.answer,
        mode="full",
        candidates=ctx.candidates,
        clusters=ctx.clustering,
        judge_result=ctx.judge_result,
        verification_report=ctx.verification,
        total_latency_ms=(time.monotonic() - t0) * 1000,
        pipeline_trace=list(ctx.trace),
    )


async def run_full_pipeline_legacy(
    client: RouterClient,
    user_prompt: str,
    candidates_config: list[dict],
    judge_config: dict,
    synthesizer_config: dict,
    debate_rounds: int = 2,
    max_repair_rounds: int = 2,
    verification_timeout: int = 30,
) -> PipelineResult:
    """Legacy entry point for the benchmark harness: builds a RunContext from dict configs.

    Every config dict must carry an explicit 'role' — no name inference.
    """
    from ..agents import AgentConfig
    from ..context import PolicyResult

    policy = PolicyResult(
        debate_rounds=debate_rounds,
        max_repair_rounds=max_repair_rounds,
        verification_timeout=verification_timeout,
    )
    for cfg in candidates_config:
        role = AgentRole(cfg["role"])
        ac = AgentConfig(
            name=cfg["name"],
            role=role,
            model=cfg["model"],
            system_prompt=cfg["system_prompt"],
            temperature=cfg.get("temperature", 0.3),
            reasoning_effort=cfg.get("reasoning_effort", "none"),
        )
        if role.is_solver:
            policy.solver_configs.append(ac)
        elif role.is_reviewer:
            policy.reviewer_configs.append(ac)

    if judge_config:
        policy.judge_config = AgentConfig(
            name="judge",
            role=AgentRole.JUDGE,
            model=judge_config["model"],
            system_prompt=judge_config["system_prompt"],
            temperature=judge_config.get("temperature", 0.1),
            reasoning_effort=judge_config.get("reasoning_effort", "none"),
        )
    if synthesizer_config:
        policy.synthesizer_config = AgentConfig(
            name="synthesizer",
            role=AgentRole.SYNTHESIZER,
            model=synthesizer_config["model"],
            system_prompt=synthesizer_config["system_prompt"],
            temperature=synthesizer_config.get("temperature", 0.2),
            reasoning_effort=synthesizer_config.get("reasoning_effort", "none"),
        )

    ctx = RunContext(raw_prompt=user_prompt, policy=policy)
    ctx.start()
    return await run_full_pipeline(client, ctx)
