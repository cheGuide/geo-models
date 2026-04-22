# Moscow TTK Benchmarks

**Дата:** 2026-04-01 00:00
**Датасет:** c:\Users\q\Work\models\ttk_10k_full
**Сэмплов:** 50
**Режим:** только дообученные модели (GeoCLIP / Revisit / Kopernik ft)

## Результаты

| Модель | Вариант | R@1 % | R@5 % | R@10 % | Mean err km | Median err km | Within 1km % | Within 5km % | Within 25km % |
|--------|---------|-------|-------|--------|-------------|---------------|--------------|--------------|----------------|
| GeoCLIP | fine-tuned | — | — | — | 10.22 | 5.13 | 10.00 | 50.00 | 98.00 |
| Revisit-Anything | fine-tuned | 19.70 | 21.21 | 23.48 | — | — | — | — | — |
| Kopernik | fine-tuned (moscow localization) | — | — | — | 3.79 | 3.08 | — | — | — |

## Примечания

В этом прогоне только **дообученные** на ваших данных модели: GeoCLIP (TTK), Revisit-Anything (TTK), Kopernik (Moscow локализация). Полный набор: `BENCHMARK_TRAINED_ONLY=0`.

- **GAEA**, **GeoVista**: требуют отдельной настройки (vLLM, LoRA)
