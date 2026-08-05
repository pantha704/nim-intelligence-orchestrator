"""Explicit typed agent roles — no name-based detection."""
from dataclasses import dataclass
from enum import Enum


class AgentRole(str, Enum):
    SOLVER = "solver"
    ALTERNATIVE_SOLVER = "alternative_solver"
    CRITIC = "critic"
    EVIDENCE_VERIFIER = "evidence_verifier"
    DEVILS_ADVOCATE = "devils_advocate"
    JUDGE = "judge"
    SYNTHESIZER = "synthesizer"

    @property
    def is_solver(self) -> bool:
        return self in (AgentRole.SOLVER, AgentRole.ALTERNATIVE_SOLVER)

    @property
    def is_reviewer(self) -> bool:
        return self in (AgentRole.CRITIC, AgentRole.EVIDENCE_VERIFIER, AgentRole.DEVILS_ADVOCATE)

    @property
    def is_judge(self) -> bool:
        return self == AgentRole.JUDGE

    @property
    def is_synthesizer(self) -> bool:
        return self == AgentRole.SYNTHESIZER


@dataclass
class AgentConfig:
    """Configuration for a single agent in the pipeline."""
    name: str
    role: AgentRole
    model: str
    system_prompt: str
    temperature: float = 0.3
    reasoning_effort: str = "none"
    max_tokens: int = 1024
    timeout_seconds: int = 30

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role.value,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
        }
