"""Single mutable execution state object for the entire pipeline."""
import time
from dataclasses import dataclass, field

from .agents import AgentConfig
from .budget import ExecutionBudget
from .clustering import Candidate, ClusteringResult
from .task_compiler import TaskSpec
from .verifiers.external_checks import VerificationReport


@dataclass
class AnonMapping:
    """Persistent anonymization mapping — created once, never re-created during a run."""
    label_to_original: dict[str, str] = field(default_factory=dict)
    original_to_label: dict[str, str] = field(default_factory=dict)
    shuffled: list[Candidate] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def label_of(self, candidate: Candidate) -> str:
        return self.original_to_label.get(candidate.name, candidate.name)

    def original_of(self, label: str) -> str:
        return self.label_to_original.get(label, label)

    def anon_text(self, max_chars: int = 2000) -> str:
        return "\n\n".join(
            f"--- {label} ---\n{c.content[:max_chars]}"
            for label, c in zip(self.labels, self.shuffled)
        )

    def update_candidates(self, candidates: list[Candidate]) -> None:
        """Update the mapping with new/updated candidate content, keeping the same labels.

        Labels are persistent — a candidate that was 'Candidate A' before debate
        is still 'Candidate A' after debate, even if its content changed.
        """
        new_shuffled = []
        for label in self.labels:
            original = self.label_to_original.get(label, "")
            for c in candidates:
                if c.name == original and not c.error and c.content:
                    new_shuffled.append(c)
                    break
        self.shuffled = new_shuffled


def create_anon_mapping(candidates: list[Candidate]) -> AnonMapping:
    """Create the anonymization mapping ONCE for the entire run."""
    import random
    valid = [c for c in candidates if not c.error and c.content]
    shuffled = list(valid)
    random.shuffle(shuffled)
    labels = [f"Candidate {chr(ord('A') + i)}" for i in range(len(shuffled))]
    mapping = AnonMapping(shuffled=shuffled, labels=labels)
    for label, cand in zip(labels, shuffled):
        mapping.label_to_original[label] = cand.name
        mapping.original_to_label[cand.name] = label
    return mapping


@dataclass
class PolicyResult:
    """Single source of truth for all routing decisions."""
    action: str = "full"  # "single" | "speculative" | "full" | "dag" — what the API executes
    route: str = "complex"  # "direct" | "verifiable" | "complex" | "open_ended"
    should_bypass_compiler: bool = False
    should_speculate: bool = False
    should_run_full_pipeline: bool = True
    use_dag: bool = False
    solver_configs: list[AgentConfig] = field(default_factory=list)
    reviewer_configs: list[AgentConfig] = field(default_factory=list)
    judge_config: AgentConfig | None = None
    synthesizer_config: AgentConfig | None = None
    debate_rounds: int = 2
    max_repair_rounds: int = 2
    verification_timeout: int = 30
    reason: str = ""


@dataclass
class RunContext:
    """Single mutable execution state for one request.

    Replaces scattered parameters (client, settings, prompt, task_spec, trace, etc.)
    with one object threaded through the entire pipeline.
    """
    raw_prompt: str = ""
    task_spec: TaskSpec | None = None
    needs_clarification: bool = False
    clarification_question: str = ""
    trace: list[str] = field(default_factory=list)
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    policy: PolicyResult = field(default_factory=PolicyResult)
    candidates: list[Candidate] = field(default_factory=list)
    dag_nodes: list = field(default_factory=list)
    model_registry: object | None = None  # request-persistent ModelRegistry
    anon: AnonMapping | None = None
    critique: dict[str, str] = field(default_factory=dict)
    clustering: ClusteringResult | None = None
    judge_result: dict | None = None
    verification: VerificationReport | None = None
    winner: Candidate | None = None
    answer: str = ""
    mode: str = ""
    total_latency_ms: float = 0
    _start_time: float = 0.0

    def start(self) -> None:
        self._start_time = time.monotonic()
        self.budget.start()

    def finish(self) -> None:
        self.total_latency_ms = (time.monotonic() - self._start_time) * 1000

    def add_trace(self, msg: str) -> None:
        self.trace.append(msg)

    def to_response(self) -> dict:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "task_spec": self.task_spec.model_dump() if self.task_spec else None,
            "clarification_question": self.clarification_question or None,
            "judge": self.judge_result,
            "verification": {
                "status": self.verification.status if self.verification else None,
                "all_passed": self.verification.all_passed if self.verification else None,
                "has_failures": self.verification.has_failures if self.verification else None,
                "has_unverified": self.verification.has_unverified if self.verification else None,
                "failures": self.verification.failures if self.verification else [],
                "unverified": self.verification.unverified if self.verification else [],
            },
            "budget": self.budget.summary(),
            "clusters": {
                "disagreement_level": self.clustering.disagreement_level if self.clustering else None,
                "num_clusters": len(self.clustering.clusters) if self.clustering else 0,
            },
            "latency_ms": round(self.total_latency_ms, 1),
            "pipeline_trace": self.trace,
        }
