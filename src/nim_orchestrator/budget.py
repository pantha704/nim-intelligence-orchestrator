"""Execution budget: tracks and enforces limits on model calls, time, and agents.

Reservation is atomic: reserve_call() checks and increments under an asyncio
Lock, so concurrent agents cannot all observe the same unused call budget.
The concurrency semaphore only limits active (in-flight) calls.
"""
import asyncio
import time
from dataclasses import dataclass, field


class BudgetExhaustedError(Exception):
    """Raised when the execution budget prevents another model call."""


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
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

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

    async def reserve_call(self, agent_name: str, model: str) -> int:
        """Atomically reserve a model-call slot. Returns a reservation token.

        Checks max_model_calls, max_total_agents and elapsed time, then
        increments the reserved counters — all under a lock, so concurrent
        reservations cannot overshoot the limits. Failed and timed-out calls
        keep the reserved slot: the budget is consumed at reservation time,
        not at completion.
        """
        async with self._lock:
            if self.model_calls >= self.limits.max_model_calls:
                raise BudgetExhaustedError(
                    f"budget exhausted at {self.model_calls}/{self.limits.max_model_calls} "
                    f"calls, {self.elapsed_seconds:.1f}s elapsed"
                )
            if self.agents_used >= self.limits.max_total_agents:
                raise BudgetExhaustedError(
                    f"agent budget exhausted — max {self.limits.max_total_agents} agents reached"
                )
            if self.elapsed_seconds >= self.limits.max_time_seconds:
                raise BudgetExhaustedError(
                    f"time budget exhausted at {self.elapsed_seconds:.1f}s"
                )
            self.model_calls += 1
            self.agents_used += 1
            self._call_log.append({
                "agent": agent_name,
                "model": model,
                "latency_ms": None,
                "tokens": None,
                "status": "in_flight",
            })
            return len(self._call_log) - 1

    def complete_call(
        self, token: int, latency_ms: float, tokens: int = 0, status: str = "success"
    ) -> None:
        """Record the outcome of a reserved call.

        Failed attempts still consume the budget — model_calls was already
        incremented at reservation time.
        """
        if 0 <= token < len(self._call_log):
            entry = self._call_log[token]
            entry["latency_ms"] = round(latency_ms, 1)
            entry["tokens"] = tokens
            entry["status"] = status

    def record_call(self, agent_name: str, model: str, latency_ms: float, tokens: int = 0) -> None:
        """Legacy non-atomic recorder (kept for compatibility/tests)."""
        self.model_calls += 1
        self._call_log.append({
            "agent": agent_name,
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "tokens": tokens,
            "status": "success",
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
