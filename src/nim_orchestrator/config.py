import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


class CandidateConfig(BaseModel):
    name: str
    model: str
    system_prompt: str
    temperature: float = 0.3
    reasoning_effort: str = "medium"
    role: Literal["solver", "alternative_solver", "critic", "evidence_verifier", "devils_advocate"]


class JudgeConfig(BaseModel):
    model: str
    system_prompt: str
    temperature: float = 0.1
    reasoning_effort: str = "high"
    role: Literal["judge"] = "judge"


class SynthesizerConfig(BaseModel):
    model: str
    system_prompt: str
    temperature: float = 0.2
    reasoning_effort: str = "high"
    role: Literal["synthesizer"] = "synthesizer"


class TaskCompilerConfig(BaseModel):
    model: str = "deepseek-v4-flash"
    timeout_seconds: int = 25


class DagConfig(BaseModel):
    """Adaptive DAG execution limits (Phase 4.0 MVP)."""
    enabled: bool = False
    max_model_calls: int = 10
    max_concurrent_calls: int = 6
    max_alternates: int = 1
    primary_model: str = "deepseek-v4-flash"
    timeout_seconds: int = 30
    specialists_enabled: bool = False


class DifficultyRouterConfig(BaseModel):
    simple_keywords: list[str] = Field(default_factory=list)
    complexity_signals: list[str] = Field(default_factory=list)
    max_prompt_length_simple: int = 500


class Settings(BaseModel):
    router_base_url: str = "http://127.0.0.1:4000/v1"
    router_api_key: str = ""
    orchestrator_host: str = "127.0.0.1"
    orchestrator_port: int = 4010
    candidates: list[CandidateConfig] = Field(default_factory=list)
    judge: JudgeConfig | None = None
    synthesizer: SynthesizerConfig | None = None
    difficulty_router: DifficultyRouterConfig = Field(default_factory=DifficultyRouterConfig)
    task_compiler: TaskCompilerConfig = TaskCompilerConfig()
    dag: DagConfig = Field(default_factory=DagConfig)
    candidate_count: int = 5
    debate_rounds: int = 2
    refine_rounds: int = 2
    verifier_timeout: int = 30
    max_repair_rounds: int = 2


def load_api_key() -> str:
    env_path = DEFAULT_CONFIG_DIR / "orchestrator.env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ROUTER_API_KEY_FILE="):
                keyfile = line.split("=", 1)[1].strip()
                keyfile = os.path.expanduser(keyfile)
                if os.path.exists(keyfile):
                    return Path(keyfile).read_text().strip()
            elif line.startswith("ROUTER_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val:
                    return val
    return os.environ.get("ROUTER_API_KEY", "")


def load_settings() -> Settings:
    env_path = DEFAULT_CONFIG_DIR / "orchestrator.env"
    yaml_path = DEFAULT_CONFIG_DIR / "orchestrator.yaml"

    base_url = "http://127.0.0.1:4000/v1"
    host = "127.0.0.1"
    port = 4010

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k == "ROUTER_BASE_URL":
                    base_url = v
                elif k == "ORCHESTRATOR_HOST":
                    host = v
                elif k == "ORCHESTRATOR_PORT":
                    port = int(v)
                elif k == "CANDIDATE_COUNT" or k == "DEBATE_ROUNDS" or k == "REFINE_ROUNDS" or k == "VERIFIER_TIMEOUT" or k == "MAX_REPAIR_ROUNDS":
                    pass

    api_key = load_api_key()

    candidates_data = {}
    judge_data = {}
    synthesizer_data = {}
    difficulty_data = {}
    dag_data = {}

    if yaml_path.exists():
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        candidates_data = raw.get("candidates", {})
        if isinstance(candidates_data, dict):
            candidates_data = list(candidates_data.values())
        judge_data = raw.get("judge", {})
        synthesizer_data = raw.get("synthesizer", {})
        difficulty_data = raw.get("difficulty_router", {})
        dag_data = raw.get("dag", {})

    candidates = [CandidateConfig(**c) for c in candidates_data] if candidates_data else []
    judge = JudgeConfig(**judge_data) if judge_data else None
    synthesizer = SynthesizerConfig(**synthesizer_data) if synthesizer_data else None
    difficulty = DifficultyRouterConfig(**difficulty_data) if difficulty_data else DifficultyRouterConfig()
    dag = DagConfig(**dag_data) if dag_data else DagConfig()

    return Settings(
        router_base_url=base_url,
        router_api_key=api_key,
        orchestrator_host=host,
        orchestrator_port=port,
        candidates=candidates,
        judge=judge,
        synthesizer=synthesizer,
        difficulty_router=difficulty,
        dag=dag,
    )
