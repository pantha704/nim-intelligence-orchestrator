#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PASS=0; FAIL=0; SKIP=0
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
result_pass() { echo -e "  ${GREEN}✓ PASS${NC}: $1"; PASS=$((PASS+1)); }
result_fail() { echo -e "  ${RED}✗ FAIL${NC}: $1"; FAIL=$((FAIL+1)); }
result_skip() { echo -e "  ${YELLOW}⊘ SKIP${NC}: $1"; SKIP=$((SKIP+1)); }

ENV_FILE="$PROJECT_DIR/config/orchestrator.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE" 2>/dev/null || true; set +a
fi

HOST="${ORCHESTRATOR_HOST:-127.0.0.1}"
PORT="${ORCHESTRATOR_PORT:-4010}"
ROUTER_URL="${ROUTER_BASE_URL:-http://127.0.0.1:4000/v1}"

echo "============================================"
echo "  NIM Intelligence Orchestrator — Test Suite"
echo "============================================"
echo ""

# Test 1: Config files exist
echo "--- Test 1: Config files ---"
for f in config/orchestrator.env config/orchestrator.yaml config/benchmark_cases.yaml; do
  if [[ -f "$PROJECT_DIR/$f" ]]; then
    result_pass "Config $f exists"
  else
    result_fail "Config $f missing"
  fi
done

# Test 2: Python module imports
echo ""
echo "--- Test 2: Python imports ---"
if python3 -c "
from nim_orchestrator.config import load_settings
from nim_orchestrator.router_client import RouterClient
from nim_orchestrator.difficulty_router import assess_difficulty
from nim_orchestrator.clustering import cluster_candidates, Candidate
from nim_orchestrator.pipeline.full_pipeline import run_full_pipeline
from nim_orchestrator.benchmarks.harness import run_benchmark, summarize_results
print('All imports OK')
" 2>/dev/null; then
  result_pass "All modules import successfully"
else
  result_fail "Module import failed"
fi

# Test 3: Settings load
echo ""
echo "--- Test 3: Settings load ---"
if python3 -c "
from nim_orchestrator.config import load_settings
s = load_settings()
assert s.router_base_url, 'No router_base_url'
assert s.candidates, 'No candidates'
assert s.judge, 'No judge config'
assert s.synthesizer, 'No synthesizer config'
print(f'Settings OK: {len(s.candidates)} candidates, judge={s.judge.model}, synth={s.synthesizer.model}')
" 2>/dev/null; then
  result_pass "Settings loaded correctly"
else
  result_fail "Settings load failed"
fi

# Test 4: Difficulty router
echo ""
echo "--- Test 4: Difficulty router ---"
python3 -c "
from nim_orchestrator.difficulty_router import assess_difficulty
from nim_orchestrator.config import load_settings

s = load_settings()
dr = s.difficulty_router

simple = assess_difficulty('What is the capital of France?', dr.simple_keywords, dr.complexity_signals, dr.max_prompt_length_simple)
assert simple.difficulty == 'simple', f'Expected simple, got {simple.difficulty}'

complex_q = assess_difficulty('Design and architect a distributed system that handles trade-offs between consistency and availability, debug the implementation, and optimize for scale.', dr.simple_keywords, dr.complexity_signals, dr.max_prompt_length_simple)
assert complex_q.difficulty == 'complex', f'Expected complex, got {complex_q.difficulty}'

print('Difficulty router OK')
" 2>/dev/null && result_pass "Difficulty router classifies correctly" || result_fail "Difficulty router failed"

# Test 5: Clustering
echo ""
echo "--- Test 5: Clustering ---"
python3 -c "
from nim_orchestrator.clustering import cluster_candidates, Candidate

c1 = Candidate(name='a', model='glm-5.2', content='The answer is 42.')
c2 = Candidate(name='b', model='deepseek-v4-flash', content='42')
c3 = Candidate(name='c', model='deepseek-v4-pro', content='The answer is Paris.')

result = cluster_candidates([c1, c2, c3])
assert result.disagreement_level != 'none', 'Should detect disagreement'
assert len(result.clusters) >= 2, f'Expected >= 2 clusters, got {len(result.clusters)}'

agree = cluster_candidates([Candidate(name='a', model='x', content='42'), Candidate(name='b', model='y', content='42')])
assert agree.disagreement_level == 'none', f'Expected agreement, got {agree.disagreement_level}'

print('Clustering OK')
" 2>/dev/null && result_pass "Clustering works correctly" || result_fail "Clustering failed"

# Test 6: Router connectivity
echo ""
echo "--- Test 6: Router connectivity ---"
if curl -sf "${ROUTER_URL%\/v1}/health/liveliness" --max-time 5 &>/dev/null; then
  result_pass "Router is reachable at $ROUTER_URL"
else
  result_fail "Router is NOT reachable at $ROUTER_URL"
fi

# Test 7: API key
echo ""
echo "--- Test 7: API key ---"
python3 -c "
from nim_orchestrator.config import load_api_key
key = load_api_key()
assert key, 'No API key found'
assert key.startswith('sk-'), 'Key does not start with sk-'
print(f'API key OK: {key[:12]}...')
" 2>/dev/null && result_pass "API key loaded" || result_fail "API key missing"

# Test 8: Single-mode ask
echo ""
echo "--- Test 8: Single-mode ask ---"
RESP=$(python3 -m nim_orchestrator.cli ask "What is 2+2? Reply with just the number." --mode single 2>/dev/null || echo "")
if [[ -n "$RESP" ]] && echo "$RESP" | grep -qi "4"; then
  result_pass "Single-mode ask returns correct answer"
else
  result_fail "Single-mode ask failed: ${RESP:-no output}"
fi

# Test 9: Full pipeline
echo ""
echo "--- Test 9: Full pipeline (complex question) ---"
RESP=$(timeout 200 python3 -m nim_orchestrator.cli ask "Prove that the sum of two even numbers is always even." --mode full 2>/dev/null || echo "")
if [[ -n "$RESP" ]] && echo "$RESP" | grep -qi "even"; then
  result_pass "Full pipeline returns answer"
else
  result_fail "Full pipeline failed: ${RESP:-no output}"
fi

# Test 10: Benchmark
echo ""
echo "--- Test 10: Benchmark (single mode, fast model) ---"
if timeout 30 python3 -c "
import asyncio
from nim_orchestrator.config import load_settings
from nim_orchestrator.router_client import RouterClient
from nim_orchestrator.benchmarks.harness import run_mode_single

async def test():
    s = load_settings()
    client = RouterClient(s.router_base_url, s.router_api_key, timeout=25)
    r = await run_mode_single(client, 'deepseek-v4-flash', 'What is 5+5?')
    await client.close()
    assert r.get('answer'), 'No answer from benchmark'
    print('Benchmark single-mode OK')

asyncio.run(test())
" 2>/dev/null; then
  result_pass "Benchmark harness runs successfully"
else
  result_fail "Benchmark harness failed"
fi

# Summary
echo ""
echo "============================================"
echo "  Test Summary"
echo "============================================"
echo -e "  ${GREEN}Passed:${NC}  $PASS"
echo -e "  ${RED}Failed:${NC} $FAIL"
echo -e "  ${YELLOW}Skipped:${NC} $SKIP"
echo "============================================"
[[ $FAIL -gt 0 ]] && exit 1
exit 0
