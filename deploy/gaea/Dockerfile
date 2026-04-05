# GAEA — образ для inference и finetune
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# GAEA dependencies (из GAEA/docker_requirements.txt)
COPY gaea/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Flash Attention (опционально; на некоторых GPU может не собраться)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

RUN pip install --no-cache-dir qwen-vl-utils

WORKDIR /workspace
