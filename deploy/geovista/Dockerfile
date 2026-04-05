# GeoVista vLLM — context сборки: корень deploy/ (см. docker-compose context: .)
# NumPy<2 + transformers==4.57.6 (не 5.x; Qwen2/vLLM) + opencv/pyarrow (issue #3)
# База 25.01: CUDA 12.8 + PyTorch со sm_120 (RTX 50xx Blackwell). 24.10 даёт
# «no kernel image» / sm_120 not compatible на 5070 Ti.
# Драйвер хоста: см. release notes NGC PyTorch 25.01 (часто 570+).
FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

COPY geovista/constraints.txt /tmp/geovista-constraints.txt

RUN pip install --no-cache-dir -c /tmp/geovista-constraints.txt \
    opencv-python-headless==4.8.0.76 \
    "vllm>=0.6.0" \
    "bitsandbytes>=0.49.2" \
    "huggingface_hub>=0.25.0" \
    jinja2

# vLLM пока не поддерживает transformers 5.x; Qwen2: нужна ветка 4.57+ (см. Qwen3-VL#2058)
RUN pip install --no-cache-dir -c /tmp/geovista-constraints.txt \
    "transformers==4.57.6" --force-reinstall \
 && python3 -c "import transformers as t; v=t.__version__; assert v.startswith('4.'), v; print('transformers', v)"

RUN pip install --no-cache-dir -c /tmp/geovista-constraints.txt --force-reinstall --no-cache-dir \
    "numpy>=1.26,<2" \
    opencv-python-headless==4.8.0.76 \
    pyarrow \
    pandas \
    scikit-learn

WORKDIR /workspace

COPY geovista/entrypoint.sh /entrypoint.sh
# CRLF из Windows: sed не всегда достаточно; нормализуем байты явно
RUN python3 -c "p=open('/entrypoint.sh','rb').read(); \
open('/entrypoint.sh','wb').write(p.replace(b'\r\n',b'\n').replace(b'\r',b'\n'))" \
 && chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
