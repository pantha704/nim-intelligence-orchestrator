import asyncio
import json
import time
from dataclasses import dataclass, field

from ..agents import AgentRole
from ..clustering import Candidate, ClusteringResult, cluster_candidates
from ..context import AnonMapping
from ..context import create_anon_mapping as _create_anon_mapping
from ..router_client import RouterClient
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


async def generate_candidates(
    client: RouterClient,
    candidates_config: list[dict],
    user_prompt: str,
    trace: list[str],
) -> list[Candidate]:

    async def _gen(cfg):
        messages = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result = await asyncio.wait_for(
                client.chat(
                    model=cfg["model"],
                    messages=messages,
                    temperature=cfg.get("temperature", 0.3),
                    reasoning_effort=cfg.get("reasoning_effort", "none"),
                    max_tokens=1024,
                ),
                timeout=30,
            )
            return Candidate(
                name=cfg["name"],
                model=cfg["model"],
                content=result.content,
                reasoning=result.reasoning,
                latency_ms=result.latency_ms,
            )
        except Exception as e:
            return Candidate(
                name=cfg["name"],
                model=cfg["model"],
                content="",
                error=str(e)[:300],
            )

    tasks = [_gen(cfg) for cfg in candidates_config]
    results = await asyncio.gather(*tasks)

    trace.append(f"Generated {len(results)} candidates from {len(candidates_config)} agents")
    for c in results:
        status = "OK" if not c.error else f"ERROR: {c.error[:80]}"
        trace.append(f"  {c.name} ({c.model}): {status} [{c.latency_ms:.0f}ms]")

    return results


def create_anon_mapping(candidates: list[Candidate]) -> AnonMapping:
    """Create anonymized mapping. Delegates to context.create_anon_mapping.

    Returns a context.AnonMapping (which has the update_candidates method
    needed for persistent IDs through debate).
    """
    return _create_anon_mapping(candidates)


def _infer_role_from_name(name: str) -> AgentRole:
    """Temporary fallback: infer role from name during migration.
    
    This exists only for backward compatibility with configs that don't
    yet have an explicit `role` field. New code should always set `role`.
    """
    name_lower = name.lower()
    if "critic" in name_lower:
        return AgentRole.CRITIC
    if "evidence" in name_lower or "verifier" in name_lower:
        return AgentRole.EVIDENCE_VERIFIER
    if "devil" in name_lower:
        return AgentRole.DEVILS_ADVOCATE
    if "alternative" in name_lower or "alt" in name_lower:
        return AgentRole.ALTERNATIVE_SOLVER
    return AgentRole.SOLVER


async def critique_candidates(
    client: RouterClient,
    reviewer_configs: list[dict],
    candidates: list[Candidate],
    user_prompt: str,
    trace: list[str],
    anon: AnonMapping | None = None,
) -> dict:
    """Run adversarial critic, evidence verifier, and devil's advocate.

    Reviewers see candidates under the same anonymous IDs used by the judge,
    so their critiques can reference candidates the judge will recognize.
    """

    if not anon:
        anon = create_anon_mapping(candidates)
    if not anon.shuffled:
        trace.append("Critique: no valid candidates to critique")
        return {"critic": "", "evidence_verifier": "", "devil_advocate": ""}

    candidate_text = anon.anon_text(2000)

    # Use explicit AgentRole — no name-based detection
    critic_config = None
    evidence_config = None
    devil_config = None
    for cfg in reviewer_configs:
        role_str = cfg.get("role", "")
        if isinstance(role_str, AgentRole):
            role = role_str
        elif role_str:
            try:
                role = AgentRole(role_str)
            except ValueError:
                role = _infer_role_from_name(cfg.get("name", ""))
        else:
            role = _infer_role_from_name(cfg.get("name", ""))

        if role == AgentRole.CRITIC and critic_config is None:
            critic_config = cfg
        elif role == AgentRole.EVIDENCE_VERIFIER and evidence_config is None:
            evidence_config = cfg
        elif role == AgentRole.DEVILS_ADVOCATE and devil_config is None:
            devil_config = cfg

    results = {}

    async def _run_reviewer(key, config, instruction):
        if not config:
            return key, ""
        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": f"""Original problem: {user_prompt}

Candidate answers:
{candidate_text}

{instruction}"""},
        ]
        try:
            result = await asyncio.wait_for(
                client.chat(
                    model=config["model"],
                    messages=messages,
                    temperature=config.get("temperature", 0.3),
                    reasoning_effort=config.get("reasoning_effort", "none"),
                    max_tokens=1024,
                ),
                timeout=30,
            )
            return key, result.content
        except Exception as e:
            trace.append(f"{key} error: {str(e)[:100]}")
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
    trace.append(f"Critique done: {', '.join(active) if active else 'none'}")
    return results


async def debate_disagreeing_candidates(
    client: RouterClient,
    clusters: ClusteringResult,
    candidate_configs: list[dict],
    candidates: list[Candidate],
    user_prompt: str,
    debate_rounds: int,
    trace: list[str],
) -> list[Candidate]:

    if clusters.disagreement_level != "high":
        trace.append(f"Disagreement level: {clusters.disagreement_level} — skipping debate")
        return candidates

    if debate_rounds < 1:
        return candidates

    sorted_clusters = sorted(clusters.clusters, key=lambda c: len(c), reverse=True)
    top_clusters = sorted_clusters[:3]

    updated = list(candidates)

    for round_num in range(debate_rounds):
        trace.append(f"Debate round {round_num + 1}")

        async def _debate_one(cluster):
            rep = cluster[0]
            other_answers = "\n\n".join(
                f"--- {c[0].name} ---\n{c[0].content[:1000]}"
                for c in top_clusters if c[0].name != rep.name
            )

            debate_prompt = f"""Original problem: {user_prompt}

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
                result = await asyncio.wait_for(
                    client.chat(
                        model=rep.model,
                        messages=messages,
                        temperature=0.3,
                        reasoning_effort="none",
                        max_tokens=1024,
                    ),
                    timeout=30,
                )
                return rep.name, Candidate(
                    name=rep.name,
                    model=rep.model,
                    content=result.content,
                    reasoning=result.reasoning,
                    latency_ms=result.latency_ms,
                )
            except Exception as e:
                trace.append(f"  {rep.name} debate error: {str(e)[:100]}")
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


async def judge_candidates(
    client: RouterClient,
    judge_config: dict,
    candidates: list[Candidate],
    user_prompt: str,
    trace: list[str],
    critique: dict | None = None,
    anon: AnonMapping | None = None,
) -> dict:

    if not anon:
        anon = create_anon_mapping(candidates)

    if len(anon.shuffled) <= 1:
        trace.append("Only 1 valid candidate — judge not needed")
        winner_name = anon.shuffled[0].name if anon.shuffled else ""
        return {"winner": winner_name, "rankings": [], "confidence": 1.0, "disagreement_level": "none"}

    candidate_summaries = anon.anon_text(2000)

    critique_section = ""
    if critique and (critique.get("critic") or critique.get("evidence_verifier") or critique.get("devil_advocate")):
        critique_section = "\n\n--- Adversarial Critique ---\n"
        if critique.get("critic"):
            critique_section += f"Critic findings:\n{critique['critic'][:1000]}\n"
        if critique.get("evidence_verifier"):
            critique_section += f"Evidence verification:\n{critique['evidence_verifier'][:1000]}\n"
        if critique.get("devil_advocate"):
            critique_section += f"Devil's advocate counterargument:\n{critique['devil_advocate'][:1000]}\n"
        critique_section += "\nConsider these critiques when evaluating candidates."

    judge_prompt = f"""Problem: {user_prompt}

Here are {len(anon.shuffled)} candidate solutions:

{candidate_summaries}
{critique_section}

Evaluate each candidate using your rubric. If the candidates agree, state that.
If they disagree, identify the key disagreement point and who is right.

Refer to candidates by their anonymous labels (Candidate A, Candidate B, etc.).

Respond ONLY with valid JSON."""

    messages = [
        {"role": "system", "content": judge_config["system_prompt"]},
        {"role": "user", "content": judge_prompt},
    ]

    try:
        result = await asyncio.wait_for(
            client.chat(
                model=judge_config["model"],
                messages=messages,
                temperature=judge_config.get("temperature", 0.1),
                reasoning_effort=judge_config.get("reasoning_effort", "none"),
                max_tokens=1024,
            ),
            timeout=30,
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

        trace.append(f"Judge: winner={parsed.get('winner', '?')}, confidence={parsed.get('confidence', '?')}")
        return parsed

    except Exception as e:
        trace.append(f"Judge error: {str(e)[:100]}")
        fallback_winner = anon.shuffled[0].name if anon.shuffled else ""
        return {"winner": fallback_winner, "rankings": [], "confidence": 0.0, "disagreement_level": "error"}


async def synthesize_final(
    client: RouterClient,
    synthesizer_config: dict,
    winner: Candidate,
    judge_result: dict,
    verification_report: VerificationReport,
    user_prompt: str,
    trace: list[str],
    max_repair_rounds: int = 2,
    critique: dict | None = None,
    anon: AnonMapping | None = None,
) -> str:

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
    if judge_result:
        judge_summary = json.dumps(judge_result, indent=2)[:1000]

    critique_section = ""
    if critique and (critique.get("critic") or critique.get("evidence_verifier") or critique.get("devil_advocate")):
        winner_label = anon.label_of(winner) if anon else "winner"
        critique_section = "\n\nAdversarial critique:\n"
        if critique.get("critic"):
            critique_section += f"Critic findings (re: {winner_label}):\n{critique['critic'][:1000]}\n"
        if critique.get("evidence_verifier"):
            critique_section += f"Evidence (re: {winner_label}): {critique['evidence_verifier'][:1000]}\n"
        if critique.get("devil_advocate"):
            critique_section += f"Devil's advocate: {critique['devil_advocate'][:1000]}\n"

    verification_section = (
        f"The following checks FAILED and must be fixed:\n{failure_text}"
        if failures
        else verification_summary
    )

    synth_prompt = f"""Original problem: {user_prompt}

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
        {"role": "system", "content": synthesizer_config["system_prompt"]},
        {"role": "user", "content": synth_prompt},
    ]

    answer = winner.content

    needs_repair = verification_report and verification_report.has_failures
    needs_synthesis = verification_report and verification_report.all_passed and not verification_report.has_unverified

    if needs_synthesis:
        trace.append(f"Verification passed — running final synthesis ({winner.name})")
        try:
            result = await asyncio.wait_for(
                client.chat(
                    model=synthesizer_config["model"],
                    messages=messages,
                    temperature=synthesizer_config.get("temperature", 0.2),
                    reasoning_effort=synthesizer_config.get("reasoning_effort", "none"),
                    max_tokens=1024,
                ),
                timeout=30,
            )
            answer = result.content
            trace.append(f"Synthesis done [{result.latency_ms:.0f}ms]")
        except Exception as e:
            trace.append(f"Synthesis error (keeping winner): {str(e)[:100]}")
    elif needs_repair:
        for round_num in range(max_repair_rounds if failures else 1):
            try:
                result = await asyncio.wait_for(
                    client.chat(
                        model=synthesizer_config["model"],
                        messages=messages,
                        temperature=synthesizer_config.get("temperature", 0.2),
                        reasoning_effort=synthesizer_config.get("reasoning_effort", "none"),
                        max_tokens=1024,
                    ),
                    timeout=30,
                )
                answer = result.content
                trace.append(f"Synthesis/repair round {round_num + 1}: done [{result.latency_ms:.0f}ms]")

                repair_verif = await verify_answer(answer, user_prompt)
                if not repair_verif.has_failures:
                    trace.append("Repair successful — no failures remain")
                    break
                else:
                    trace.append(f"Repair round {round_num + 1} still has failures: {repair_verif.failures}")
                    prev_failures = set(failures)
                    current_failures = set(repair_verif.failures)
                    if prev_failures == current_failures:
                        trace.append("Same failures as before — stopping repair")
                        break

            except Exception as e:
                trace.append(f"Synthesis error: {str(e)[:100]}")
                break
    else:
        trace.append(f"Running synthesis with unverified items ({winner.name})")
        try:
            result = await asyncio.wait_for(
                client.chat(
                    model=synthesizer_config["model"],
                    messages=messages,
                    temperature=synthesizer_config.get("temperature", 0.2),
                    reasoning_effort=synthesizer_config.get("reasoning_effort", "none"),
                    max_tokens=1024,
                ),
                timeout=30,
            )
            answer = result.content
            trace.append(f"Synthesis done [{result.latency_ms:.0f}ms]")
        except Exception as e:
            trace.append(f"Synthesis error (keeping winner): {str(e)[:100]}")

    return answer


async def run_full_pipeline(
    client: RouterClient,
    user_prompt: str,
    candidates_config: list[dict],
    judge_config: dict,
    synthesizer_config: dict,
    debate_rounds: int = 2,
    max_repair_rounds: int = 2,
    verification_timeout: int = 30,
) -> PipelineResult:

    t0 = time.monotonic()
    trace: list[str] = []

    # Separate solver candidates from reviewer candidates using AgentRole
    solver_configs = []
    reviewer_configs = []
    for cfg in candidates_config:
        role_str = cfg.get("role", "")
        if isinstance(role_str, AgentRole):
            role = role_str
        elif role_str:
            try:
                role = AgentRole(role_str)
            except ValueError:
                role = _infer_role_from_name(cfg.get("name", ""))
        else:
            role = _infer_role_from_name(cfg.get("name", ""))

        if role.is_solver:
            solver_configs.append(cfg)
        elif role.is_reviewer:
            reviewer_configs.append(cfg)

    if not solver_configs:
        solver_configs = candidates_config

    trace.append(f"Starting full pipeline: {len(solver_configs)} solvers, {len(reviewer_configs)} reviewers")

    candidates = await generate_candidates(client, solver_configs, user_prompt, trace)

    # Create shared anonymization mapping BEFORE reviewers so all agents use same labels
    anon = create_anon_mapping(candidates)
    trace.append(f"Anonymized {len(anon.shuffled)} candidates: " +
                 ", ".join(f"{anon.labels[i]}={anon.shuffled[i].name}" for i in range(len(anon.shuffled))))

    critique = await critique_candidates(
        client,
        reviewer_configs if reviewer_configs else candidates_config,
        candidates,
        user_prompt,
        trace,
        anon=anon,
    )

    clustering_result = cluster_candidates(candidates)
    trace.append(f"Clustering: {len(clustering_result.clusters)} cluster(s), disagreement={clustering_result.disagreement_level}")

    if clustering_result.disagreement_level == "high":
        candidates = await debate_disagreeing_candidates(
            client, clustering_result, solver_configs, candidates,
            user_prompt, debate_rounds, trace,
        )
        # Persistent anon IDs: update candidates in-place, do NOT re-create mapping.
        # Labels are preserved — Candidate A is still Candidate A after debate.
        anon.update_candidates(candidates)
        clustering_result = cluster_candidates(candidates)
        trace.append(f"Post-debate clustering: {len(clustering_result.clusters)} cluster(s), disagreement={clustering_result.disagreement_level}")

    judge_result = await judge_candidates(
        client, judge_config, candidates, user_prompt, trace, critique=critique, anon=anon,
    )

    winner_name = judge_result.get("winner", "")
    winner = None
    for c in candidates:
        if c.name == winner_name and not c.error:
            winner = c
            break
    if winner is None:
        valid = [c for c in candidates if not c.error and c.content]
        if clustering_result.leader:
            winner = clustering_result.leader
        elif valid:
            winner = valid[0]
        else:
            return PipelineResult(
                answer="All candidates failed.",
                mode="full",
                candidates=candidates,
                total_latency_ms=(time.monotonic() - t0) * 1000,
                pipeline_trace=trace + ["No valid candidates — all failed"],
            )

    verification = await verify_answer(winner.content, user_prompt, verification_timeout)
    trace.append(f"Verification: {verification.status} ({len(verification.results)} checks)")

    final_answer = await synthesize_final(
        client, synthesizer_config, winner, judge_result, verification,
        user_prompt, trace, max_repair_rounds, critique=critique, anon=anon,
    )

    return PipelineResult(
        answer=final_answer,
        mode="full",
        candidates=candidates,
        clusters=clustering_result,
        judge_result=judge_result,
        verification_report=verification,
        total_latency_ms=(time.monotonic() - t0) * 1000,
        pipeline_trace=trace,
    )
