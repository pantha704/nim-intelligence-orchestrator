#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/config/orchestrator.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE" 2>/dev/null || true; set +a
fi

HOST="${ORCHESTRATOR_HOST:-127.0.0.1}"
PORT="${ORCHESTRATOR_PORT:-4010}"
PID_FILE="$PROJECT_DIR/logs/orchestrator.pid"

echo "============================================"
echo "  NIM Intelligence Orchestrator — Status"
echo "============================================"

# Process
if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "  PID:        $PID (running)"
  else
    echo "  PID:        $PID (dead — stale pidfile)"
  fi
else
  echo "  PID:        not found"
fi

# Port
if command -v ss &>/dev/null; then
  BIND=$(ss -tlnp 2>/dev/null | grep ":$PORT " | head -1 || true)
  if [[ -n "$BIND" ]]; then
    echo "  Bind:       $BIND"
  else
    echo "  Bind:       port $PORT not listening"
  fi
fi

# Health
if curl -sf "http://$HOST:$PORT/health" --max-time 3 &>/dev/null; then
  echo "  Health:     OK"
else
  echo "  Health:     FAIL"
fi

# Router connectivity
ROUTER_URL="${ROUTER_BASE_URL:-http://127.0.0.1:4000/v1}"
if curl -sf "${ROUTER_URL%\/v1}/health/liveliness" --max-time 3 &>/dev/null; then
  echo "  Router:     $ROUTER_URL (reachable)"
else
  echo "  Router:     $ROUTER_URL (UNREACHABLE)"
fi

# Log size
LOG="$PROJECT_DIR/logs/orchestrator.log"
if [[ -f "$LOG" ]]; then
  SIZE=$(du -h "$LOG" | cut -f1)
  LINES=$(wc -l < "$LOG")
  echo "  Log:        $LOG ($SIZE, $LINES lines)"
else
  echo "  Log:        none"
fi

echo "============================================"
