# DSM / TTK: подготовка данных, прокси-метрики и дообучение

**Дата:** 2026-04-08  
**Задача:** cross-view пары (polar satellite + street) из Moscow TTK; прокси-модель PyTorch (не оригинальный TensorFlow DSM Shi et al., CVPR 2020).

## Артефакты

| Что | Путь |
|-----|------|
| Подготовленный датасет | `cross_view_localization_DSM/Data/TTK/` (`bingmap/`, `polarmap/`, `streetview/`, `splits/`) |
| Манифест | `cross_view_localization_DSM/Data/TTK/manifest.json` |
| Скрипт подготовки | `cross_view_localization_DSM/script/prepare_ttk_for_dsm.py` |
| Дообучение (InfoNCE, 30 эпох) | `cross_view_localization_DSM/script/finetune_ttk_proxy_torch.py` |
| Чекпоинт | `cross_view_localization_DSM/checkpoints_ttk/proxy_infonce_resnet18.pt` |
| История лосса | `benchmarks_results/dsm_ttk_finetune_history.json` |
| Метрики baseline (ImageNet) | `benchmarks_results/dsm_ttk_proxy_baseline.json` |
| Метрики после дообучения | `benchmarks_results/dsm_ttk_proxy_finetuned.json` |

**Исходный TTK:** `ttk_10k_full` (в подготовке: до 100 train-пар, 70 val-пар в текущем прогоне).

## Методика

- **Прокси:** один общий ResNet18, эмбеддинги L2-normalized; запрос — streetview 512×128, галерея — polarmap 128×512.
- **Дообучение:** симметричный InfoNCE по матрице `sat_emb @ grd_emb^T / temperature` на `train-19zl.csv`, AdamW, 30 эпох, batch 16, CPU (обучение через `TransGeo2022/venv`).

## Результаты на val (N = 70 пар)

| Модель | Recall@1 | Recall@5 | Recall@10 |
|--------|----------|----------|-----------|
| ImageNet (без дообучения) | 1.43% | 7.14% | 17.14% |
| После InfoNCE на train TTK | **7.14%** | **15.71%** | **24.29%** |

Случайный уровень для R@1 при N=70: ≈ 1.43%. После дообучения R@1 вырос примерно в **5 раз** относительно baseline.

## Запуск оценки (пример)

Использовать интерпретатор, где `import torch` не падает (при ошибке DLL CUDA в другом venv — например `geo-clip/.venv`):

```powershell
$env:CUDA_VISIBLE_DEVICES=''
python cross_view_localization_DSM/script/eval_ttk_proxy_torch.py `
  --data_root cross_view_localization_DSM/Data/TTK --split val --device cpu `
  --checkpoint cross_view_localization_DSM/checkpoints_ttk/proxy_infonce_resnet18.pt `
  --out_json benchmarks_results/dsm_ttk_proxy_finetuned.json
```

## Ограничения

- Это **не** воспроизведение оригинального DSM (VGG + TensorFlow 1 + корреляция ориентации).
- Метрики зависят от размера train/val в `prepare_ttk_for_dsm.py` (`--max_train`, `--max_val`).
