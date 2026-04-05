#!/usr/bin/env bash
# GeoAgent: ожидание Docker и запуск compose (Linux/macOS).
# Явный bind датасета, если data/ttk_10k_full — симлинк (как в geoagent-docker.ps1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-build}"
TIMEOUT="${TIMEOUT:-180}"

wait_docker() {
  local start=$SECONDS
  while (( SECONDS - start < TIMEOUT )); do
    if docker info >/dev/null 2>&1; then
      echo "Docker daemon is ready."
      return 0
    fi
    sleep 3
  done
  echo "Docker daemon did not respond within ${TIMEOUT}s." >&2
  return 1
}

wait_docker

EXTRA=()
if [[ "$ACTION" == "prepare" || "$ACTION" == "train" ]]; then
  if [[ -e "$ROOT/data/ttk_10k_full" ]]; then
    real="$(realpath "$ROOT/data/ttk_10k_full" 2>/dev/null || readlink -f "$ROOT/data/ttk_10k_full" 2>/dev/null || echo "$ROOT/data/ttk_10k_full")"
    echo "TTK dataset bind: $real -> /workspace/data/ttk_10k_full" >&2
    EXTRA=( -v "$real:/workspace/data/ttk_10k_full:ro" )
  fi
fi

case "$ACTION" in
  wait)
    echo "Done."
    ;;
  shell)
    docker compose --profile geoagent up -d geoagent-shell
    echo "Container geoagent-shell is running. Example: docker exec -it geoagent-shell bash"
    ;;
  build)
    docker compose build geoagent-train
    ;;
  prepare)
    docker compose --profile geoagent run "${EXTRA[@]}" geoagent-prepare
    ;;
  train)
    docker compose --profile geoagent run "${EXTRA[@]}" geoagent-train
    ;;
  *)
    echo "Usage: $0 [wait|shell|build|prepare|train]" >&2
    exit 1
    ;;
esac
