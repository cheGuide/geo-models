"""
Общая проверка: обучение и тесты с PyTorch — только на GPU (CUDA).

Исключение для автоматической отладки/CI без видеокарты: MODELS_ALLOW_CPU=1
(в лог пишется предупреждение; на реальном обучении не используйте).

Использование:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.require_cuda import require_cuda

    device = require_cuda()
"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_ALLOW_CPU = os.environ.get("MODELS_ALLOW_CPU", "").lower() in ("1", "true", "yes")


def require_cuda() -> "torch.device":
    """Возвращает cuda:0 или cpu при MODELS_ALLOW_CPU=1 и отсутствии CUDA."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _ALLOW_CPU:
        print(
            "Предупреждение: MODELS_ALLOW_CPU=1 — выполнение на CPU (только отладка/CI). "
            "Для продакшена используйте GPU и PyTorch+CUDA.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    print(
        "Ошибка: CUDA недоступна. Обучение и тесты запускаются только на GPU.\n"
        "Создайте venv с PyTorch+CUDA: scripts\\setup_gpu_venv.ps1\n"
        "Или для локальной отладки без GPU: $env:MODELS_ALLOW_CPU='1'",
        file=sys.stderr,
    )
    raise SystemExit(1)


def require_cuda_or_cpu(preferred: str = "cuda") -> "torch.device":
    """
    Если CUDA доступна — `torch.device(preferred)` (cuda / cuda:N).
    Иначе то же, что require_cuda() (cpu при MODELS_ALLOW_CPU или выход).
    """
    import torch

    if torch.cuda.is_available():
        return torch.device(preferred)
    return require_cuda()


def require_cuda_strict() -> "torch.device":
    """Только GPU; MODELS_ALLOW_CPU не учитывается (для обучения)."""
    import torch

    if not torch.cuda.is_available():
        print(
            "Ошибка: это обучение требует CUDA. С MODELS_ALLOW_CPU обучение не запускается.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return torch.device("cuda")
