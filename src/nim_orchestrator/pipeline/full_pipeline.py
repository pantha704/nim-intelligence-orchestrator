import asyncio
import json
import random
import time
from dataclasses import dataclass, field

from ..clustering import Candidate, ClusteringResult, cluster_candidates
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
                "all_passed": self.verification_report.all_passed if self.verification_report else None,
                "failures": self.verification_report.failures if self.verification_report else [],
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


async def critique_candidates(
    client: RouterClient,
    candidates_config: list[dict],
    candidates: list[Candidate],
    user_prompt: str,
    trace: list[str],
) -> dict:
    """Run adversarial critic and evidence verifier on candidate answers.

    Unlike the old pipeline that sent only the user_prompt, this function
    receives the actual candidate answers so the critic can find weaknesses
    and the evidence verifier can classify claims.
    """

    valid = [c for c in candidates if not c.error and c.content]
    if not valid:
        trace.append("Critique: no valid candidates to critique")
        return {"critic": "", "evidence_verifier": ""}

    candidate_text = "\n\n".join(
        f"--- {c.name} ({c.model}) ---\n{c.content[:2000]}"
        for c in valid
    )

    critic_config = None
    evidence_config = None
    for cfg in candidates_config:
        role = cfg.get("role", cfg.get("name", "")).lower()
        if "critic" in role:
            critic_config = cfg
        elif "evidence" in role or "verifier" in role:
            evidence_config = cfg

    results = {}

    async def _run_critic():
        if not critic_config:
            return "critic", ""
        messages = [
            {"role": "system", "content": critic_config["system_prompt"]},
            {"role": "user", "content": f"""Original problem: {user_prompt}

Candidate answers to critique:
{candidate_text}

Identify weaknesses, errors, and gaps in these candidate answers. Be specific about which candidate has which issue."""},
        ]
        try:
            result = await asyncio.wait_for(
                client.chat(
                    model=critic_config["model"],
                    messages=messages,
                    temperature=critic_config.get("temperature", 0.3),
                    reasoning_effort=critic_config.get("reasoning_effort", "none"),
                    max_tokens=1024,
                ),
                timeout=30,
            )
            return "critic", result.content
        except Exception as e:
            trace.append(f"Critic error: {str(e)[:100]}")
            return "critic", ""

    async def _run_evidence():
        if not evidence_config:
            return "evidence_verifier", ""
        messages = [
            {"role": "system", "content": evidence_config["system_prompt"]},
            {"role": "user", "content": f"""Original problem: {user_prompt}

Candidate answers with claims to verify:
{candidate_text}

Extract factual claims from these answers and classify each as: VERIFIED, UNVERIFIABLE, or CONTRADICTED based on your knowledge."""},
        ]
        try:
            result = await asyncio.wait_for(
                client.chat(
                    model=evidence_config["model"],
                    messages=messages,
                    temperature=evidence_config.get("temperature", 0.1),
                    reasoning_effort=evidence_config.get("reasoning_effort", "none"),
                    max_tokens=1024,
                ),
                timeout=30,
            )
            return "evidence_verifier", result.content
        except Exception as e:
            trace.append(f"Evidence verifier error: {str(e)[:100]}")
            return "evidence_verifier", ""

    tasks = [_run_critic(), _run_evidence()]
    task_results = await asyncio.gather(*tasks)
    for name, content in task_results:
        results[name] = content

    trace.append(f"Critique done: critic={'yes' if results.get('critic') else 'no'}, evidence={'yes' if results.get('evidence_verifier') else 'no'}")
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


async def judge_candidates(
    client: RouterClient,
    judge_config: dict,
    candidates: list[Candidate],
    user_prompt: str,
    trace: list[str],
) -> dict:

    valid = [c for c in candidates if not c.error and c.content]
    if len(valid) == 1:
        trace.append("Only 1 valid candidate — judge not needed")
        return {"winner": valid[0].name, "rankings": [], "confidence": 1.0, "disagreement_level": "none"}

    # Randomize order and anonymize names to prevent name bias
    shuffled = list(valid)
    random.shuffle(shuffled)
    anonymous_labels = [f"Candidate {chr(ord('A') + i)}" for i in range(len(shuffled))]
    label_to_original = {}
    for label, cand in zip(anonymous_labels, shuffled):
        label_to_original[label] = cand.name

    candidate_summaries = "\n\n".join(
        f"--- {label} ---\n{c.content[:2000]}"
        for label, c in zip(anonymous_labels, shuffled)
    )

    judge_prompt = f"""Problem: {user_prompt}

Here are {len(shuffled)} candidate solutions:

{candidate_summaries}

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

        try:
            parsed = json.loads(result.content)
        except json.JSONDecodeError:
            json_match = None
            for line in result.content.split("\n"):
                if "{" in line:
                    start = line.index("{")
                    for end_idx in range(len(line), start, -1):
                        try:
                            parsed = json.loads(line[start:end_idx])
                            json_match = parsed
                            break
                        except json.JSONDecodeError:
                            continue
                    if json_match:
                        break
            if json_match:
                parsed = json_match
            else:
                parsed = {"winner": anonymous_labels[0], "rankings": [], "confidence": 0.5, "disagreement_level": "unknown", "raw": result.content[:500]}

        # Map anonymous labels back to original candidate names
        winner_label = parsed.get("winner", "")
        if winner_label in label_to_original:
            parsed["winner"] = label_to_original[winner_label]

        if "rankings" in parsed and isinstance(parsed["rankings"], list):
            parsed["rankings"] = [
                label_to_original.get(r, r) if isinstance(r, str) else r
                for r in parsed["rankings"]
            ]

        trace.append(f"Judge: winner={parsed.get('winner', '?')}, confidence={parsed.get('confidence', '?')}")
        return parsed

    except Exception as e:
        trace.append(f"Judge error: {str(e)[:100]}")
        return {"winner": valid[0].name, "rankings": [], "confidence": 0.0, "disagreement_level": "error"}


async def synthesize_final(
    client: RouterClient,
    synthesizer_config: dict,
    winner: Candidate,
    judge_result: dict,
    verification_report: VerificationReport,
    user_prompt: str,
    trace: list[str],
    max_repair_rounds: int = 2,
) -> str:

    failures = verification_report.failures if verification_report else []
    verification_summary = failure_text = ""
    if verification_report:
        if verification_report.all_passed:
            verification_summary = f"All external checks passed ({len(verification_report.results)} checks)"
        else:
            failure_text = "\n".join(f"- {f}" for f in failures)

    judge_summary = ""
    if judge_result:
        judge_summary = json.dumps(judge_result.__dict__ if hasattr(judge_result, '__dict__') else judge_result, indent=2)[:1000]

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

If external checks failed, repair ONLY the specific failures. Do not rewrite content that passed verification."""

    messages = [
        {"role": "system", "content": synthesizer_config["system_prompt"]},
        {"role": "user", "content": synth_prompt},
    ]

    answer = winner.content
    if verification_report and verification_report.all_passed:
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
    else:
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
                if repair_verif.all_passed:
                    trace.append("Repair successful — all checks now pass")
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

    trace.append(f"Starting full pipeline for {len(candidates_config)} candidates")
    candidates = await generate_candidates(client, candidates_config, user_prompt, trace)

    # Run adversarial critic and evidence verifier on candidate answers
    critique = await critique_candidates(client, candidates_config, candidates, user_prompt, trace)

    clustering_result = cluster_candidates(candidates)
    trace.append(f"Clustering: {len(clustering_result.clusters)} cluster(s), disagreement={clustering_result.disagreement_level}")

    if clustering_result.disagreement_level == "high":
        candidates = await debate_disagreeing_candidates(
            client, clustering_result, candidates_config, candidates,
            user_prompt, debate_rounds, trace,
        )
        clustering_result = cluster_candidates(candidates)
        trace.append(f"Post-debate clustering: {len(clustering_result.clusters)} cluster(s), disagreement={clustering_result.disagreement_level}")

    judge_result = await judge_candidates(client, judge_config, candidates, user_prompt, trace)

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
    trace.append(f"Verification: {'PASSED' if verification.all_passed else 'FAILED'} ({len(verification.results)} checks)")

    final_answer = await synthesize_final(
        client, synthesizer_config, winner, judge_result, verification,
        user_prompt, trace, max_repair_rounds,
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
