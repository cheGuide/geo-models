# Таблицы бенчмарков Moscow TTK

**Прогон:** `python run_all_benchmarks.py`  
**Дата:** 2026-03-31  
**Датасет:** `c:\Users\q\Work\models\ttk_10k_full`  
**Число сэмплов (случайная подвыборка):** 50  

Переменные окружения: `TTK_DATASET`, `BENCHMARK_NUM_SAMPLES=50`, `BENCHMARK_FORCE_RERUN=1`.

**Важно:** при **50** сэмплах метрики **AnyLoc** (R@1 и др.) сильнее **шумят**, чем при 500 (например, R@1 2% vs ~10.6% на длинном прогоне). Для отчёта можно дополнительно приложить прогон с `BENCHMARK_NUM_SAMPLES=500`.

---

## 1. Retrieval / VPR (Recall)

| Метод | До обучения (на TTK) | После обучения (на TTK) |
|--------|------------------------|---------------------------|
| **AnyLoc** (DINOv2 + VLAD) | **Baseline** (без дообучения на TTK) | *отдельного fine-tune в пайплайне нет* |
| **Revisit-Anything** (DINO + SALAD) | *не измерялось* — нет сохранённого baseline-чекпоинта | **Fine-tuned** на TTK |

### Числа (этот прогон, n=50 для AnyLoc)

| Модель | Вариант | R@1 % | R@5 % | R@10 % |
|--------|---------|-------|-------|--------|
| AnyLoc | baseline | 2.00 | 8.00 | — |
| Revisit-Anything | baseline | — | — | — |
| Revisit-Anything | fine-tuned | 19.70 | 21.21 | 23.48 |

*Revisit: R@* на **фиксированном** валидационном сплите (727/1695), не на тех же 50 случайных картинках, что AnyLoc.*

---

## 2. GeoCLIP: ошибка координат и доли в кольцах

| Метрика | До (baseline, MP-16) | После (fine-tuned на TTK) |
|---------|------------------------|----------------------------|
| Mean error, км | 2326.32 | 10.22 |
| Median error, км | 995.80 | 5.13 |
| Within 1 km, % | 0.00 | 10.00 |
| Within 5 km, % | 6.00 | 50.00 |
| Within 25 km, % | 18.00 | 98.00 |

---

## 3. Kopernik: регрессия координат (ResNet50)

| Вариант | Статус | Median error | Mean error | Примечание |
|---------|--------|--------------|------------|------------|
| **Baseline** | *пропуск* | — | — | `resnet50_streetview_combined.pth` — **другая голова** (не `MoscowLocModel` в `test_moscow_dataset.py`); отдельный eval не подключён. |
| **Fine-tuned** | ok | **3.07 км** (median) | **3.79 км** | `resnet50_moscow_localization.pth` |

---

## 4. Сводка: что сравнивалось как «до / после»

| Метод | «До» | «После» | Комментарий |
|-------|------|---------|-------------|
| **GeoCLIP** | Pre-trained (baseline) | Чекпоинт `geo-clip/checkpoints_ttk/geoclip_ttk_final.pth` | Полное сравнение двух режимов. |
| **AnyLoc** | Только baseline | — | Fine-tune на TTK в этом репо не запускался. |
| **Revisit-Anything** | — | Fine-tuned на TTK | Baseline-чекпоинта нет; в отчёте — см. AnyLoc как ориентир по VPR. |
| **Kopernik** | — | Fine-tuned (`resnet50_moscow_localization.pth`) | Baseline «до» той же архитектурой в одном скрипте не прогоняется (несовместимость весов). |

---

## 5. Архив: прогон 2026-03-12 (500 сэмплов, другой путь к датасету)

Для сравнения с дипломной копией (путь `c:\Users\q\Work\dipl\ttk_10k_full`):

| Модель | Вариант | R@1 % | R@5 % | Mean err км | Median err км |
|--------|---------|-------|-------|-------------|---------------|
| AnyLoc | baseline | 10.60 | 26.40 | 5.61 | 5.23 |
| GeoCLIP | baseline | — | — | 2325.06 | 869.26 |
| GeoCLIP | fine-tuned | — | — | 1211.98 | 6.43 |

Полная таблица: `MOSCOW_TTK_METHODS_TABLE.md`, `benchmarks_report.json` (старые копии).

---

## Команда повтора

```powershell
cd c:\Users\q\Work\models
$env:TTK_DATASET = "c:\Users\q\Work\models\ttk_10k_full"
$env:BENCHMARK_NUM_SAMPLES = "500"
$env:BENCHMARK_FORCE_RERUN = "1"
.\AnyLoc\.venv\Scripts\python.exe run_all_benchmarks.py
```
