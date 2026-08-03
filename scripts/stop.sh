#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/config/orchestrator.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE" 2>/dev/null || true; set +a
fi

HOST="${ORCHESTRATOR_HOST:-127.0.0.1}"
PORT="${ORCHESTRATOR_PORT:-4010}"
PID_FILE="$PROJECT_DIR/logs/orchestrator.pid"

if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  if kill "$PID" 2>/dev/null; then
    echo "Stopped orchestrator (PID $PID)"
    rm -f "$PID_FILE"
    exit 0
  fi
fi

# Fallback: kill by port
PIDS=$(lsof -ti :"$PORT" 2>/dev/null || true)
if [[ -n "$PIDS" ]]; then
  kill $PIDS 2>/dev/null || true
  echo "Killed processes on port $PORT"
  rm -f "$PID_FILE"
  exit 0
fi

echo "Orchestrator is not running"
