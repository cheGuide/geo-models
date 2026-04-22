# TransGeo: оценка и дообучение на Moscow TTK

**Датасет:** `c:\Users\q\Work\models\ttk_10k_full`  
**Валидация:** `transgeo/splits/val-ttk.csv` (70 пар ground ↔ satellite)  
**Метрики:** Recall@1, Recall@5 (retrieval: правильный спутник среди галереи)

## Baseline (без дообучения на TTK)

| Инициализация | Скрипт / команда | Recall@1 | Recall@5 | Примечание |
|----------------|------------------|----------|----------|------------|
| CVUSA pretrained | `train.py -e --dataset ttk --resume checkpoints/CVUSA_model/result/checkpoint.pth.tar --sat_res 320` | **91.43%** | **92.86%** | Файл: `TransGeo2022/eval_cvusa_ttk/transgeo_eval_results.txt` |
| VIGOR (crop stage) | `TransGeo2022/eval_vigor_weights_on_ttk.py` | **22.86%** | **22.86%** | Нулевые attention-карты для crop-модели; файл: `TransGeo2022/eval_vigor_ttk/transgeo_eval_results.txt` |

Дата фиксации baseline: 2026-04-08.

## Дообучение (fine-tune)

- **Этап 1:** SAM + mining, resume CVUSA → `TransGeo2022/result_ttk/`  
- **Этап 2:** `sat_res 320`, `--crop`, resume этапа 1  

Команды: см. `TransGeo2022/run_TTK.ps1` (корень датасета: `c:\Users\q\Work\models\ttk_10k_full`).

После обучения eval:

```powershell
cd c:\Users\q\Work\models\TransGeo2022
.\venv\Scripts\python.exe train.py -e --dataset ttk --root "c:\Users\q\Work\models\ttk_10k_full" --resume "./result_ttk/checkpoint.pth.tar" --save_path "./eval_ttk_finetuned" --sat_res 320 --crop --workers 4 --multiprocessing-distributed --world-size 1 --rank 0 --dist-url "tcp://127.0.0.1:10003" --dist-backend gloo --cos --dim 1000
```

(Если финальный чекпоинт без crop — убрать `--crop` и подобрать `--sat_res` как при обучении.)

## Результаты после дообучения

**Короткое дообучение на CPU** (3 эпохи, Adam, batch 4, без mining): из‑за несовместимости GPU RTX 5070 Ti с установленным PyTorch полноценный GPU-запуск невозможен; см. `benchmarks_results/transgeo_ttk_results.json`.

| Этап | Recall@1 | Recall@5 | Recall@10 | Файл |
|------|----------|----------|-----------|------|
| После fine-tune (CPU, 3 ep) | **15.71%** | **34.29%** | 51.43% | `TransGeo2022/eval_ttk_finetuned/transgeo_eval_results.txt` |
| Чекпоинт | — | — | — | `TransGeo2022/result_ttk/checkpoint.pth.tar` |

Полное качественное дообучение (SAM + mining, 100 + 50 эпох): см. `TransGeo2022/run_TTK.ps1` — на **Windows** добавьте `--dist-backend gloo` и **`--sat_res 320`** при resume из CVUSA; нужен PyTorch с поддержкой вашей видеокарты или обучение на другой машине с совместимым CUDA.
