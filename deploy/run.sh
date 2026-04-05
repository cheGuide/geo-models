#!/bin/bash
# Скрипт запуска deploy на сервере
set -e
cd "$(dirname "$0")"

# Загрузить .env если есть
[ -f .env ] && set -a && source .env && set +a

case "${1:-}" in
  geovista)
    echo "Starting GeoVista vLLM..."
    docker compose up -d geovista
    echo "GeoVista: http://localhost:${GEOVISTA_PORT:-8000}"
    ;;
  gaea)
    echo "Starting GAEA Gradio..."
    docker compose up -d gaea
    echo "GAEA: http://localhost:7860"
    ;;
  build)
    echo "Building images..."
    docker compose build
    ;;
  finetune)
    echo "Running GAEA finetune (batch)..."
    docker compose --profile finetune run --rm gaea-finetune
    ;;
  anyloc)
    DATASET="${2:-/workspace/data/ttk_10k_full}"
    SAMPLES="${3:-500}"
    echo "Running AnyLoc on $DATASET, n=$SAMPLES..."
    docker compose --profile anyloc run --rm anyloc \
      python run_moscow_ttk.py \
      --dataset_path "$DATASET" \
      --num_samples "$SAMPLES" \
      --output /workspace/outputs/anyloc_results.json
    ;;
  *)
    echo "Usage: $0 {geovista|gaea|build|finetune|anyloc [dataset_path] [num_samples]}"
    echo ""
    echo "  geovista   - запуск GeoVista vLLM API (порт 8000)"
    echo "  gaea       - запуск GAEA Gradio UI (порт 7860)"
    echo "  build      - сборка Docker образов"
    echo "  finetune   - дообучение GAEA на TTK"
    echo "  anyloc     - AnyLoc VPR на датасете"
    exit 1
    ;;
esac
