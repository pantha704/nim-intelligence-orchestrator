import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class ChatResult:
    content: str
    reasoning: str = ""
    model: str = ""
    latency_ms: float = 0
    raw: dict = field(default_factory=dict)
    finish_reason: str = ""
    tokens_generated: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.content)


@dataclass
class StreamConfig:
    """Configuration for early-stop streaming."""
    max_tokens: int = 1024
    min_tokens: int = 64
    stop_on_double_newline: bool = True
    stop_on_code_fence_close: bool = True
    early_stop_token_budget: int = 512


def _should_early_stop(content: str, cfg: StreamConfig) -> bool:
    """Decide if we have enough content to stop early."""
    if len(content) < cfg.min_tokens * 3:
        return False

    if cfg.stop_on_double_newline and content.rstrip().endswith("\n\n"):
        return True

    if cfg.stop_on_code_fence_close:
        fence_count = content.count("```")
        if fence_count > 0 and fence_count % 2 == 0 and content.rstrip().endswith("```"):
            return True

    sentences = content.count(". ") + content.count("。") + content.count("! ") + content.count("? ")
    if sentences >= 3 and len(content) > cfg.early_stop_token_budget * 3:
        return True

    return False


MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 529, 502, 503}


class BudgetExhaustedError(Exception):
    """Raised when the execution budget prevents another model call."""


async def budgeted_chat(
    client: "RouterClient",
    ctx,
    agent_name: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.3,
    reasoning_effort: str = "none",
    max_tokens: int = 1024,
    timeout: float = 30,
):
    """Chat call under ExecutionBudget enforcement.

    - checks can_call() before calling (atomic under the concurrency semaphore)
    - records each spawned agent
    - enforces max_concurrent_agents via the budget semaphore
    - records every call and latency
    - raises BudgetExhaustedError when the budget is spent
    """
    async with ctx.budget.semaphore:
        if not ctx.budget.can_call():
            raise BudgetExhaustedError(
                f"budget exhausted at {ctx.budget.model_calls}/{ctx.budget.limits.max_model_calls} "
                f"calls, {ctx.budget.elapsed_seconds:.1f}s elapsed"
            )
        if not ctx.budget.can_spawn_agent():
            raise BudgetExhaustedError(
                f"agent budget exhausted — max {ctx.budget.limits.max_total_agents} agents reached"
            )
        ctx.budget.record_agent()
        result = await asyncio.wait_for(
            client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
            ),
            timeout=timeout,
        )
        ctx.budget.record_call(agent_name, model, result.latency_ms, result.tokens_generated)
        return result


class RouterClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                timeout=httpx.Timeout(self.timeout, connect=10),
            )

    async def _retry_request(self, payload: dict) -> httpx.Response:
        """Send a non-streaming chat request with retry on 429/529."""
        await self._ensure_client()
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                    delay = (2**attempt) * 0.5 + (0.1 * (attempt * 2))
                    await asyncio.sleep(delay)
                    continue
                return resp
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep((2**attempt) * 0.5)
                    continue
                raise
        if last_exc:
            raise last_exc
        return await self._client.post("/chat/completions", json=payload)

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        reasoning_effort: str = "medium",
        max_tokens: int = 1024,
        stream_config: StreamConfig | None = None,
    ) -> ChatResult:
        """Chat completion. If stream_config is provided, uses streaming with early stop."""
        if stream_config is not None:
            return await self._chat_stream(model, messages, temperature, reasoning_effort, stream_config)

        await self._ensure_client()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort and reasoning_effort not in ("none", "default"):
            payload["reasoning_effort"] = reasoning_effort

        t0 = time.monotonic()
        resp = await self._retry_request(payload)
        latency = (time.monotonic() - t0) * 1000

        resp.raise_for_status()
        data = resp.json()

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "") or msg.get("reasoning", "")

        return ChatResult(
            content=content,
            reasoning=reasoning,
            model=model,
            latency_ms=latency,
            raw=data,
            finish_reason=choice.get("finish_reason", ""),
            tokens_generated=data.get("usage", {}).get("completion_tokens", 0),
        )

    async def _chat_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        reasoning_effort: str,
        cfg: StreamConfig,
    ) -> ChatResult:
        """Streaming chat with early-stop support."""
        await self._ensure_client()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": cfg.max_tokens,
            "stream": True,
        }
        if reasoning_effort and reasoning_effort not in ("none", "default"):
            payload["reasoning_effort"] = reasoning_effort

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = ""
        t0 = time.monotonic()

        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                delta_content = delta.get("content", "")
                if not isinstance(delta_content, str):
                    delta_content = ""

                delta_reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if not isinstance(delta_reasoning, str):
                    delta_reasoning = ""

                if delta_content:
                    content_parts.append(delta_content)
                if delta_reasoning:
                    reasoning_parts.append(delta_reasoning)

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                accumulated = "".join(content_parts)
                if _should_early_stop(accumulated, cfg) and not finish_reason:
                    break

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        latency = (time.monotonic() - t0) * 1000

        return ChatResult(
            content=content,
            reasoning=reasoning,
            model=model,
            latency_ms=latency,
            raw={},
            finish_reason=finish_reason,
            tokens_generated=len(content) // 4,
        )

    async def chat_batch(
        self,
        requests: list[tuple[str, list[dict], float, str, int]],
    ) -> list[ChatResult]:
        tasks = [self.chat(m, msgs, temp, re, mt) for m, msgs, temp, re, mt in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def models(self) -> list[str]:
        await self._ensure_client()
        resp = await self._client.get("/models")
        resp.raise_for_status()
        data = resp.json()
        return sorted({m["id"] for m in data.get("data", [])})

    async def health(self) -> bool:
        await self._ensure_client()
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
