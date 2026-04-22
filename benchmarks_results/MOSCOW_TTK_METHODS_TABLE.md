# Сводная таблица методов: Moscow TTK

**Источник:** `benchmarks_results/benchmarks_report.json`, дубликат в `BENCHMARKS_REPORT.md`  
**Дата прогона:** 2026-04-11 (добавлены CosPlace eval и GeoVista TTK; ранее: 2026-04-08 — DELF; 2026-03-12 — часть методов)  
**Датасет в логе:** `ttk_10k_full` (500 сэмплов в ранних прогонах; DELF — 100 сэмплов retrieval / 100 pano для головы)  
**Примечание:** путь на диске в оригинале может отличаться (например `...\dipl\ttk_10k_full` или `...\models\ttk_10k_full`); протокол оценки — тот же.

---

## Общая сводка

| Метод | До обучения (на TTK) | После обучения (на TTK) | Что измерялось |
|--------|----------------------|--------------------------|----------------|
| **AnyLoc** (DINOv2 + VLAD) | Baseline, без дообучения на TTK | В отчёте отдельного fine-tune нет | R@1, R@5, mean/median error (км) |
| **GeoCLIP** | Pre-trained (MP-16), без TTK | Fine-tuned на TTK | mean/median error (км), доли within 1/5/25 km |
| **Revisit-Anything** (DINO + SALAD) | Baseline не запускали (нет чекпоинта) | Fine-tuned на TTK | R@1, R@5, R@10 |
| **Kopernik** (регрессия координат) | Baseline в JSON не зафиксирован | Fine-tuned на Moscow/TTK | mean/median error (км) |
| **GAEA** | Не в автоматическом отчёте | Не в автоматическом отчёте | Требуются отдельные прогоны |
| **GeoVista** | vLLM + GeoVista-RL-6k-7B, см. прогон 2026-04-11 ниже | Не проводилось | mean/median error (км), within 1/5/25 km; см. `GeoVista/РЕЗУЛЬТАТЫ_ТЕСТИРОВАНИЯ_TTK.md` |
| **CosPlace** | Pretrained ResNet18 (SF-XL), eval 2026-04-11 | После fine-tune — не в этом прогоне | R@1, R@5, R@10, R@20 (25 м) |
| **DELF** (TF Hub, mean-pool + retrieval) | Baseline: предобученный Hub без TTK | MLP-голова на эмбеддингах, дообучена на TTK (веса DELF заморожены) | R@1/R@5 (1 км) и/или mean/median по координатам |

---

## Численные результаты (как в отчёте)

### AnyLoc — только baseline

| Метрика | Значение |
|---------|----------|
| R@1 | 10.60% |
| R@5 | 26.40% |
| Mean error, км | 5.61 |
| Median error, км | 5.23 |

### GeoCLIP — до и после дообучения на TTK

| Метрика | До (baseline) | После (fine-tuned) |
|---------|----------------|---------------------|
| Mean error, км | 2325.06 | 1211.98 |
| Median error, км | 869.26 | 6.43 |
| Within 1 km, % | 0.80 | 8.00 |
| Within 5 km, % | 2.60 | 45.60 |
| Within 25 km, % | 18.60 | 74.40 |

### GeoCLIP — полный датасет (все кадры, 16 139 изображений, 2026-03-31)

Подробный JSON: `benchmarks_results/geoclip_ttk_full_all_frames_report.json` (метрики без массивов `errors_km`).

| Метрика | До (baseline) | После (fine-tuned, 10 эпох на `ttk_train_full.csv`) |
|---------|----------------|--------------------------------------------------------|
| Mean error, км | 2179.02 | 7.86 |
| Median error, км | 869.17 | 4.05 |
| Within 1 km, % | 0.8 | 14.7 |
| Within 5 km, % | 3.7 | 56.1 |
| Within 25 km, % | 21.0 | 98.0 |

### GeoAgent (Qwen2.5-VL, репозиторий `GeoAgent/`) — полный TTK

Оценка: `GeoAgent/infer/run_moscow_ttk.py` (метрики по гаверсину после геокодирования `FinalAnswer`, если не `--skip_geocode`). Нужны все шарды весов в `GeoAgent/checkpoints/ghost233lism/GeoAgent/` (скачивание: `python tools/download_geoagent_model.py`). PyTorch для **RTX 50xx**: nightly **cu128** в `GeoAgent/.venv`.

| Статус | Пояснение |
|--------|-----------|
| В работе | Докачка весов с Hugging Face; после завершения — прогон с `--num_samples 0 --all_frames` |

### Revisit-Anything — только fine-tuned (baseline пропущен)

| Метрика | Значение |
|---------|----------|
| R@1 | 19.70% |
| R@5 | 21.21% |
| R@10 | 23.48% |

*В отчёте для baseline указано: нет чекпоинта; как ориентир по VPR — AnyLoc.*

### Kopernik — только fine-tuned

| Метрика | Значение |
|---------|----------|
| Mean error, км | 4.05 |
| Median error, км | 3.69 |
| Median error, м (из JSON) | 3686 |

### CosPlace — eval pretrained на SF-XL (2026-04-11)

Прогон: `CosPlace/eval.py`, `positive_dist_threshold=25` м, `ResNet18`, `fc_output_dim=512`, чекпоинт `checkpoints/pretrained_resnet18_512.pth`, датасет `CosPlace/cosplace_data/ttk`.  
Лог: `CosPlace/logs/default/2026-04-11_02-47-18/info.log`.

| Метрика | Значение |
|---------|----------|
| Запросы (queries_v1) | 511 |
| База (database) | 2628 |
| Recall@1 | 23.3% |
| Recall@5 | 23.5% |
| Recall@10 | 23.5% |
| Recall@20 | 23.9% |

*Ранее в `CosPlace/РЕЗУЛЬТАТЫ_ТЕСТИРОВАНИЯ_TTK.md` фигурировали другие доли (R@1≈14.4%) — возможны отличия сплита/подготовки `prepare_ttk_cosplace.py` или версии данных.*

### GeoVista — короткий прогон на TTK (2026-04-11)

Скрипт: `GeoVista/run_moscow_ttk.py`, WSL, vLLM (`inference/vllm_deploy_geovista_1gpu.sh`), модель в API: `qwen2.5-vl`, **5** случайных кадров (`num_samples=5`, seed 42), датасет `models/ttk_10k_full`.  
Результаты: `GeoVista/geovista_ttk_test_20260411.jsonl`.

| Метрика | Значение |
|---------|----------|
| Mean error, км | 5.96 |
| Median error, км | 5.09 |
| Within 1 km, % | 0.0 |
| Within 5 km, % | 40.0 |
| Within 25 km, % | 100.0 |

*Метрики «within X km» — доля предсказаний с ошибкой по гаверсину не больше X км от GT точки кадра (не радиус ТТК как кольца).*

**Дополнительно:** одиночный тест на фото (Статуя Свободы, Нью-Йорк) — `GeoVista/test_single_image.py`; модель вернула ~40.7128° N, 74.0060° W (центр города; ориентир острова Либерти ~40.689° N, 74.045° W).

### DELF — baseline (image retrieval, mean-pool эмбеддинг)

Прогон: `benchmarks_results/run_delf_ttk.py`, `seed=42`, `positive_radius_m=1000`, **100** случайных кадров с файлами на диске.  
JSON: `benchmarks_results/delf_moscow_ttk_results.json`.

| Метрика | Значение |
|---------|----------|
| R@1 | 6.00% |
| R@5 | 27.00% |
| Mean error, км | 5.48 |
| Median error, км | 5.07 |

### DELF — дообучение на TTK (MLP поверх замороженных эмбеддингов)

Сплит по **`pano_id`** (20% панорам в валидацию, все ракурсы вместе). Извлечение признаков — тот же TF Hub `google/delf/1`, пулинг дескрипторов. Обучение: `sklearn` MLP + масштабирование признаков и целей (`TransformedTargetRegressor`).  
100 панорам (80 train / 20 val), 239 train-кадров / 60 val-кадров.  
Отчёт: `benchmarks_results/delf_head_ttk_report.json`, веса: `benchmarks_results/delf_head_ttk_mlp.joblib`, кэш эмбеддингов (опционально): `delf_ttk_train_cache_100panos.npz`.

| Метрика | Validation |
|---------|------------|
| Mean error, км | 5.15 |
| Median error, км | 4.85 |
| Within 1 km, % | 3.3 |
| Within 5 km, % | 53.3 |
| Within 25 km, % | 100.0 |

*Полное дообучение свёрточной части DELF (локальные дескрипторы) — отдельный пайплайн из `tensorflow/models/research/delf`; здесь адаптирована только георегрессия поверх готовых эмбеддингов.*

---

## Примечания

- **GAEA** в этом файле по-прежнему без свежих чисел; **GeoVista** — см. блок выше и `GeoVista/РЕЗУЛЬТАТЫ_ТЕСТИРОВАНИЯ_TTK.md`.
- Для **GAEA** сравнение «до / после TTK LoRA» по координатам можно получить прогоном `GAEA/test_ttk_dataset.py` с разными путями к адаптерам и добавить сюда строки вручную после эксперимента.
