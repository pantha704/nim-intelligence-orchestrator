#!/usr/bin/env bash
# Explicit sandbox preflight: pulls the PINNED python image and verifies a
# secure backend. The sandbox itself never pulls images during a request —
# run this once per environment (or in CI before tests).
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

PYTHON="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
fi

echo "--- Sandbox preflight ---"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  IMAGE=$("$PYTHON" -c "from nim_orchestrator.verifiers.sandbox import DOCKER_IMAGE; print(DOCKER_IMAGE)")
  echo "Pulling pinned image: ${IMAGE}"
  if docker pull "$IMAGE"; then
    echo "Image ready."
  else
    echo "WARNING: docker image pull failed — sandbox will fail closed." >&2
  fi
else
  echo "docker unavailable."
fi

"$PYTHON" -m nim_orchestrator.verifiers.sandbox
STATUS=$?
echo "--- preflight done (exit ${STATUS}) ---"
exit "$STATUS"
