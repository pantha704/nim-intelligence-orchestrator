"""ModelRegistry — configured model aliases, health, capabilities,
latency history and specialist suitability. Model selection is scored, never
alphabetical."""
from dataclasses import dataclass, field

SPECIALIST_NAMES = (
    "coding", "mathematics", "research", "systems_architecture",
    "security_review", "general_reasoning",
)

# Default suitability per model per specialist (0..1)
DEFAULT_SUITABILITY: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {s: 0.9 for s in SPECIALIST_NAMES},
    "deepseek-v4-pro": {
        "coding": 0.95, "mathematics": 0.9, "security_review": 0.9,
        "research": 0.6, "systems_architecture": 0.7, "general_reasoning": 0.7,
    },
    "glm-5.2": {
        "research": 0.95, "systems_architecture": 0.9, "general_reasoning": 0.8,
        "coding": 0.5, "mathematics": 0.6, "security_review": 0.6,
    },
    "minimax-3": {
        "research": 0.85, "general_reasoning": 0.7,
        "coding": 0.4, "mathematics": 0.4, "security_review": 0.4,
        "systems_architecture": 0.5,
    },
}

DEFAULT_CAPABILITIES: dict[str, list[str]] = {
    "deepseek-v4-flash": ["fast", "general", "reliable"],
    "deepseek-v4-pro": ["deep", "code", "math"],
    "glm-5.2": ["research", "long_context", "reasoning"],
    "minimax-3": ["long_context", "research"],
}


@dataclass
class ModelInfo:
    name: str
    aliases: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    health: str = "unknown"  # unknown | healthy | degraded | down
    latency_ms_history: list[float] = field(default_factory=list)
    suitability: dict[str, float] = field(default_factory=dict)

    @property
    def mean_latency_ms(self) -> float:
        if not self.latency_ms_history:
            return 0.0
        return sum(self.latency_ms_history) / len(self.latency_ms_history)


class ModelRegistry:
    """Scores configured models for a specialist; never picks alphabetically."""

    def __init__(self):
        self._models: dict[str, ModelInfo] = {}
        self._order: list[str] = []

    def register(self, name: str, aliases: list[str] | None = None,
                 capabilities: list[str] | None = None, health: str = "unknown",
                 suitability: dict[str, float] | None = None) -> None:
        if name in self._models:
            return
        self._models[name] = ModelInfo(
            name=name,
            aliases=aliases or DEFAULT_CAPABILITIES.get(name, []),
            capabilities=capabilities or DEFAULT_CAPABILITIES.get(name, []),
            health=health,
            suitability=suitability or DEFAULT_SUITABILITY.get(name, {s: 0.5 for s in SPECIALIST_NAMES}),
        )
        self._order.append(name)

    @classmethod
    def from_configured(cls, names: list[str]) -> "ModelRegistry":
        registry = cls()
        for n in names:
            registry.register(n)
        return registry

    def names(self) -> list[str]:
        return list(self._order)

    def health_of(self, name: str) -> str:
        info = self._models.get(name)
        return info.health if info else "unknown"

    def set_health(self, name: str, health: str) -> None:
        if name in self._models:
            self._models[name].health = health

    def record_latency(self, name: str, latency_ms: float) -> None:
        if name in self._models:
            self._models[name].latency_ms_history.append(max(latency_ms, 0.0))

    def latency_history(self, name: str) -> list[float]:
        info = self._models.get(name)
        return list(info.latency_ms_history) if info else []

    def select(self, specialist_name: str, preferred: list[str] | None = None) -> str | None:
        """Pick the best model for a specialist by suitability, preference,
        health and latency. Tie-break is registration order — NOT alphabetical."""
        preferred = preferred or []
        best_name: str | None = None
        best_score = float("-inf")
        best_index = 0

        for index, name in enumerate(self._order):
            info = self._models[name]
            if info.health == "down":
                continue
            score = info.suitability.get(specialist_name, 0.5)
            if name in preferred or any(a in preferred for a in info.aliases):
                score += 0.5
            mean_latency = info.mean_latency_ms
            if mean_latency > 0:
                score -= min(mean_latency / 30000.0, 0.5)
            if info.health == "degraded":
                score -= 0.2

            if score > best_score or (score == best_score and index < best_index):
                best_score = score
                best_name = name
                best_index = index

        return best_name
