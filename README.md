# nim-intelligence-orchestrator

Intelligence orchestration layer that sits **above** [nim-multi-key-router](../nim-router/) and turns unlimited cheap tokens into emergent intelligence through **diverse multi-model generation, external verification, and adversarial scrutiny**.

## Architecture

```
User Query
  ↓
Difficulty Router
  ├── Simple → direct response (skip pipeline)
  └── Complex
        ↓
  Diverse multi-model candidates (GLM, DeepSeek, Qwen — NOT identical instances)
  Each agent has a distinct role:
    1. Solver          — primary solution (GLM 5.2)
    2. Alt solver      — fundamentally different approach (DeepSeek Flash)
    3. Adversarial critic — finds weakest point (DeepSeek Pro)
    4. Evidence verifier — classifies claims, suggests checks (GLM 5.2)
    5. Devil's advocate — argues opposite conclusion (DeepSeek Flash)
        ↓
  Cluster + deduplicate answers
        ↓
  Disagreement detector
    ├── Agreement → verify winner
    └── Disagreement → debate top 2–3 candidates (2 rounds)
        ↓
  External verification (Python execution, syntax check, arithmetic check)
        ↓
  Judge ranks candidates (DeepSeek Pro, fixed rubric)
        ↓
  Synthesizer repairs only failed checks → final answer
```

## Why This Design

Research-backed principles:
- **Diversity > identical debate** — homogeneous agents collapse toward the same error; mixing model families is one of the strongest predictors of improvement
- **External verification > self-critique** — a "same dumb model with a different verifier prompt" is still vulnerable to correlated errors
- **Independent generation + verification** will probably beat continuous debate — debate is only added where it measurably helps (controlled by disagreement detector)
- **The benchmark is the product** — measure single vs best-of-5 vs best-of-5+judge vs full pipeline on your real tasks

## Installation

```bash
cd nim-intelligence-orchestrator
pip install -e .
```

## Usage

### Ask a question
```bash
# Simple question — skips pipeline, goes direct
python -m nim_orchestrator.cli ask "What is the capital of France?"

# Force full pipeline
python -m nim_orchestrator.cli ask "Prove that the sum of two even numbers is even" --mode full

# Force single mode (no pipeline)
python -m nim_orchestrator.cli ask "What is 2+2?" --mode single
```

### Run benchmark
```bash
python -m nim_orchestrator.cli bench --output results.json
```

This runs all 4 modes against test cases from `config/benchmark_cases.yaml` and reports:
- Mean accuracy score per mode
- Mean latency per mode
- Error count per mode

### Start as API server
```bash
./scripts/start.sh
# POST http://127.0.0.1:4010/v1/intelligence
# {"prompt": "your question here", "mode": "auto"}
```

### Health check
```bash
python -m nim_orchestrator.cli health
```

## Configuration

- `config/orchestrator.yaml` — candidate agents, judge, synthesizer, difficulty router settings
- `config/orchestrator.env` — router URL, API key file path, port, model names
- `config/benchmark_cases.yaml` — test cases for benchmarking

## Key Distinction From Router

| Component | Role |
|-----------|------|
| `nim-multi-key-router` | Transport layer: key rotation, model routing, failover, rate limiting |
| `nim-intelligence-orchestrator` | Intelligence layer: multi-model generation, verification, debate, benchmarking |

The orchestrator calls the router through its OpenAI-compatible endpoint at `http://127.0.0.1:4000/v1`.
