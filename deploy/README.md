# Geo Models — Deploy на сервере (Docker)

Репозиторий для запуска геолокационных моделей на сервере через Docker.

## Сервисы

| Сервис | Порт | Описание | Требования GPU |
|--------|------|----------|----------------|
| **geovista** | 8000 | GeoVista vLLM API (OpenAI-совместимый) | 16+ GB VRAM (с квантизацией) |
| **gaea** | 7860 | GAEA Gradio UI (чат по геолокации) | 16+ GB VRAM |
| **gaea-finetune** | — | Дообучение GAEA на TTK (batch job) | 24+ GB VRAM |
| **anyloc** | — | AnyLoc DINOv2+VLAD (batch, VPR) | 8+ GB VRAM |
| **geoagent-shell** | — | GeoAgent: долгоживущий контейнер (`sleep infinity`) для prepare/train через `exec` | 24+ GB VRAM |
| **geoagent-prepare** / **geoagent-train** | — | Подготовка JSON и LoRA на TTK (batch) | 24+ GB VRAM |

## Структура на сервере

```
/workspace/
├── deploy/              # этот репозиторий
├── GAEA/                # клон GAEA
├── GeoVista/            # клон GeoVista
├── AnyLoc/              # клон AnyLoc
├── GeoAgent/            # клон GeoAgent (geoagent-shell / prepare / train)
├── models/              # модели (скачать отдельно)
│   ├── GeoVista-RL-6k-7B/
│   ├── Qwen2.5-VL-7B-Instruct/
│   └── GAEA/
├── data/                # данные
│   └── ttk_10k_full/
└── outputs/             # выходы (finetune, inference)
```

## Быстрый старт

### 1. Клонировать и подготовить

```bash
cd /workspace
# Клонировать репозитории: GAEA, GeoVista, AnyLoc, deploy
git clone ... GAEA
git clone ... GeoVista
git clone ... AnyLoc
# deploy — этот репозиторий

cd deploy
cp .env.example .env
# Отредактировать пути в .env (GAEA_PATH, GeoVista_PATH, MODELS_PATH и т.д.)
```

### 2. Скачать модели

```bash
# GeoVista (или GeoVista-RL-6k-7B)
# GAEA: Qwen2.5-VL-7B-Instruct + LoRA GAEA
# См. README в каждом проекте
```

### 3. Запуск сервисов

```bash
cd deploy

# Только GeoVista vLLM
docker compose up -d geovista

# Только GAEA (Gradio)
docker compose up -d gaea

# Все inference-сервисы
docker compose up -d geovista gaea
```

### 4. Дообучение GAEA (batch)

```bash
# Подготовить данные, переписать пути (см. GAEA_LINUX_FINETUNE.md)
# Запустить finetune
docker compose run --rm gaea-finetune
```

### 5. AnyLoc (batch)

```bash
# Данные должны быть в DATA_PATH (монтируется в /workspace/data)
# По умолчанию: /workspace/data/ttk_10k_full
./run.sh anyloc

# Или с параметрами:
docker compose --profile anyloc run --rm anyloc \
  python run_moscow_ttk.py \
  --dataset_path /workspace/data/ttk_10k_full \
  --num_samples 500 \
  --output /workspace/outputs/anyloc_results.json
```

## Переменные окружения (.env)

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `MODELS_PATH` | Путь к папке с моделями | `../models` |
| `DATA_PATH` | Путь к данным (ttk_10k_full и т.д.) | `../data` |
| `OUTPUT_PATH` | Путь для выходов | `../outputs` |
| `GEOVISTA_MODEL` | Имя/путь модели GeoVista | `GeoVista-RL-6k-7B` |
| `GAEA_MODEL_PATH` | Путь к GAEA LoRA | `models/GAEA` |
| `GAEA_BASE_MODEL` | Путь к Qwen2.5-VL base | `models/Qwen2.5-VL-7B-Instruct` |

## Требования к серверу

- **ОС:** Linux (Ubuntu 22.04 рекомендуется)
- **Docker** с поддержкой `nvidia-container-toolkit`
- **GPU:** NVIDIA с 16+ GB VRAM для GeoVista/GAEA, 8+ GB для AnyLoc
- **RAM:** 32+ GB для больших датасетов
- **Диск:** ~50 GB под модели + данные

## GeoAgent (образ и registry)

Сервисы `geoagent-*` собираются из `../GeoAgent/Dockerfile`, образ по умолчанию `geo-deploy-geoagent:latest`.

```bash
cd deploy
docker compose build geoagent-shell
docker compose --profile geoagent up -d geoagent-shell
docker exec -it geoagent-shell bash
```

Публикация образа в registry (после `docker login`):

```bash
# Windows PowerShell из deploy/
.\push-geoagent.ps1 -Registry ghcr.io/ВАШ_ЛОГИН -Tag latest
```

Переменные см. `.env.example` (`GEOAGENT_IMAGE` — тот же тег, что пушите, чтобы на сервере делать `pull`).

### Голый сервер: что подтягивается само, а что нет

| Что | Автоматически? | Комментарий |
|-----|------------------|---------------|
| Образ `geo-deploy-geoagent` | Да, если сделать `docker pull` с registry или `docker compose build` (при сборке качается базовый `nvcr.io/nvidia/pytorch` и pip) | Только среда и библиотеки, **не** веса модели |
| Код GeoAgent | Нет | Нужен клон репозитория и путь `GEOAGENT_PATH` (том в контейнер) |
| Веса GeoAgent (`checkpoints/...`) | Нет | Скачать на хост **до** или **внутри** контейнера, см. ниже |
| Датасет TTK (`ttk_10k_full`) | Нет | Положить/смонтировать в `DATA_PATH`; контейнер сам не качает архив |

**Минимальный порядок на новом сервере**

1. Установить Docker, Compose, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. Клонировать репозиторий с `GeoAgent/` (или скопировать каталог).
3. Скачать веса (на хосте с `huggingface_hub` или в запущенном контейнере):

   ```bash
   pip install huggingface_hub   # на хосте, если без контейнера
   python GeoAgent/tools/download_geoagent_model.py
   # или: huggingface-cli download ghost233lism/GeoAgent --local-dir GeoAgent/checkpoints/ghost233lism/GeoAgent
   ```

4. Положить датасет в `data/ttk_10k_full/` (структура с `dataset_metadata.json` и `images/`), настроить `DATA_PATH` в `deploy/.env`.
5. `cd deploy && cp .env.example .env`, поправить пути, затем `docker compose pull` (если образ в registry) или `docker compose build geoagent-shell`, затем `docker compose --profile geoagent up -d geoagent-shell`.

Итого: контейнер **не** «скачивает всё с нуля» сам по себе — он даёт только окружение; веса и данные вы подкладываете или качаете отдельными командами.

## Сборка образов

```bash
cd deploy
docker compose build
```

## Документация по проектам

- [GAEA Linux Finetune](../GAEA/GAEA_LINUX_FINETUNE.md)
- [GeoVista inference](../GeoVista/inference/)
- [AnyLoc](../AnyLoc/README.md)
