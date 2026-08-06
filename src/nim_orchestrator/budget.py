"""Execution budget: tracks and enforces limits on model calls, time, and agents."""
import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class BudgetLimits:
    max_model_calls: int = 20
    max_time_seconds: float = 120.0
    max_concurrent_agents: int = 6
    max_total_agents: int = 10


@dataclass
class ExecutionBudget:
    """Mutable budget tracker for a single request run."""
    limits: BudgetLimits = field(default_factory=BudgetLimits)
    model_calls: int = 0
    agents_used: int = 0
    _start_time: float = 0.0
    _call_log: list[dict] = field(default_factory=list)
    _semaphore: asyncio.Semaphore | None = None

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._semaphore = asyncio.Semaphore(self.limits.max_concurrent_agents)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.limits.max_concurrent_agents)
        return self._semaphore

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def exhausted(self) -> bool:
        """True when no more model calls can be made."""
        return not self.can_call()

    def record_call(self, agent_name: str, model: str, latency_ms: float, tokens: int = 0) -> None:
        self.model_calls += 1
        self._call_log.append({
            "agent": agent_name,
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "tokens": tokens,
        })

    def record_agent(self) -> None:
        self.agents_used += 1

    def can_call(self) -> bool:
        return (
            self.model_calls < self.limits.max_model_calls
            and self.elapsed_seconds < self.limits.max_time_seconds
        )

    def can_spawn_agent(self) -> bool:
        return self.agents_used < self.limits.max_total_agents

    def summary(self) -> dict:
        return {
            "model_calls": self.model_calls,
            "agents_used": self.agents_used,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "limits": {
                "max_model_calls": self.limits.max_model_calls,
                "max_time_seconds": self.limits.max_time_seconds,
                "max_concurrent_agents": self.limits.max_concurrent_agents,
                "max_total_agents": self.limits.max_total_agents,
            },
            "call_log": list(self._call_log),
        }
