#!/bin/bash
# GeoVista vLLM entrypoint для Docker
# MODEL_PATH — путь к модели (обязательно)
# USE_QUANTIZATION — 1 для 4-bit (по умолчанию)

set -e
MODEL_PATH="${MODEL_PATH:-/workspace/models/GeoVista-RL-6k-7B}"
USE_QUANTIZATION="${USE_QUANTIZATION:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Если дефолтный путь пуст — пробуем типичный путь из репо GeoVista (том ../GeoVista)
if [ ! -d "$MODEL_PATH" ] && [ "$MODEL_PATH" = "/workspace/models/GeoVista-RL-6k-7B" ]; then
    ALT="/workspace/geovista/models/GeoVista-RL-6k-7B"
    if [ -d "$ALT" ]; then
        MODEL_PATH="$ALT"
        echo "Using model at $MODEL_PATH (fallback)"
    fi
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "На хосте должна быть папка с весами (config.json, *.safetensors), см. README."
    echo ""
    echo "Варианты:"
    echo "  1) Положить модель в deploy/../models/GeoVista-RL-6k-7B  (MODELS_PATH=../models по умолчанию)"
    echo "  2) Или в GeoVista/models/GeoVista-RL-6k-7B и задать GEOVISTA_MODEL_PATH=/workspace/geovista/models/GeoVista-RL-6k-7B в .env"
    echo "  3) Скачать: huggingface-cli download LibraTree/GeoVista-RL-6k-7B --local-dir \"../models/GeoVista-RL-6k-7B\""
    echo ""
    echo "Содержимое /workspace/models:"
    ls -la /workspace/models 2>/dev/null || echo "  (том не смонтирован или пуст)"
    echo "Содержимое /workspace/geovista/models (если есть):"
    ls -la /workspace/geovista/models 2>/dev/null || echo "  (нет)"
    exit 1
fi

QUANT_ARGS=()
if [ "$USE_QUANTIZATION" = "1" ]; then
    QUANT_ARGS=(--quantization bitsandbytes)
fi

# Chat template (если есть в GeoVista)
TEMPLATE_PATH="/workspace/geovista/inference/template_qwen.jinja"
EXTRA=()
if [ -f "$TEMPLATE_PATH" ]; then
    EXTRA=(--chat-template "$TEMPLATE_PATH")
fi

echo "Starting vLLM: $MODEL_PATH port=$PORT quantization=$USE_QUANTIZATION"
exec vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --limit-mm-per-prompt "image=${MAX_IMAGES_PER_PROMPT}" \
    --trust-remote-code \
    --served-model-name qwen2.5-vl \
    --enforce-eager \
    "${QUANT_ARGS[@]}" \
    "${EXTRA[@]}"
