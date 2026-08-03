#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/config/orchestrator.env"

# Load env
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE" 2>/dev/null || true
  set +a
fi

HOST="${ORCHESTRATOR_HOST:-127.0.0.1}"
PORT="${ORCHESTRATOR_PORT:-4010}"

# Check if already running
if curl -sf "http://$HOST:$PORT/health" --max-time 2 &>/dev/null; then
  echo "Orchestrator already running on http://$HOST:$PORT"
  exit 0
fi

# Install if needed
cd "$PROJECT_DIR"
if ! pip show nim-intelligence-orchestrator &>/dev/null 2>&1; then
  echo "Installing nim-intelligence-orchestrator..."
  pip install -e . --quiet
fi

# Start
echo "Starting orchestrator on http://$HOST:$PORT ..."
nohup python -m nim_orchestrator.cli serve --host "$HOST" --port "$PORT" \
  > "$PROJECT_DIR/logs/orchestrator.log" 2>&1 &
echo $! > "$PROJECT_DIR/logs/orchestrator.pid"

# Wait for health
for i in $(seq 1 10); do
  if curl -sf "http://$HOST:$PORT/health" --max-time 2 &>/dev/null; then
    echo "Orchestrator is up (PID $(cat "$PROJECT_DIR/logs/orchestrator.pid"))"
    echo "  POST http://$HOST:$PORT/v1/intelligence  — run intelligence pipeline"
    exit 0
  fi
  sleep 0.5
done

echo "ERROR: Orchestrator failed to start — check logs/orchestrator.log"
exit 1
