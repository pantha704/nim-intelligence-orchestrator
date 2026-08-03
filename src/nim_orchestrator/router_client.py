import asyncio
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

    @property
    def ok(self) -> bool:
        return bool(self.content)


class RouterClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=httpx.Timeout(self.timeout, connect=10),
            )

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        reasoning_effort: str = "medium",
        max_tokens: int = 4096,
    ) -> ChatResult:
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
        resp = await self._client.post("/chat/completions", json=payload)
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
