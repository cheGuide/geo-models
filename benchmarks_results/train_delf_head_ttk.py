#!/usr/bin/env python3
"""
Дообучение на TTK: MLP-голова поверх замороженных DELF-эмбеддингов (mean-pool, TF Hub).

Полное дообучение весов DELF (локальные дескрипторы) — отдельный пайплайн из
tensorflow/models/research/delf; здесь — практичная адаптация под Москву:
регрессия (lat, lon) из эмбеддинга.

Сплит по pano_id (все три ракурса одной панорамы — только в train или только в val).

Зависимости: tensorflow, tensorflow-hub, packaging, scikit-learn, joblib, pillow, tqdm

Usage:
  py benchmarks_results/train_delf_head_ttk.py --dataset_path ../ttk_10k_full --val_ratio 0.2 --seed 42
  py benchmarks_results/train_delf_head_ttk.py --max_panos 80 --epochs_mlp 300

Обучение: Pipeline(StandardScaler → MLPRegressor) внутри TransformedTargetRegressor(StandardScaler на lat/lon).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from PIL import Image
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from delf_ttk_common import build_delf_fn, extract_pooled_from_pil, setup_tensorflow_imports
except ImportError as e:
    print("Проверьте зависимости: tensorflow tensorflow-hub packaging", file=sys.stderr)
    raise SystemExit(1) from e

from joblib import dump as joblib_dump


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def main():
    p = argparse.ArgumentParser(description="DELF + MLP-голова на TTK")
    p.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get("TTK_DATASET", str(Path(__file__).resolve().parents[1] / "ttk_10k_full")),
    )
    p.add_argument("--val_ratio", type=float, default=0.2, help="Доля pano_id в валидации")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_panos", type=int, default=0, help="0 = все панорамы (медленно при извлечении)")
    p.add_argument("--epochs_mlp", type=int, default=400, help="max_iter у MLPRegressor")
    p.add_argument(
        "--cache_npz",
        type=str,
        default="",
        help="Путь к .npz с эмбеддингами (если есть — пропускаем DELF; иначе сохраняем после извлечения)",
    )
    p.add_argument(
        "--output_model",
        type=str,
        default=str(_BENCH / "delf_head_ttk_mlp.joblib"),
    )
    p.add_argument(
        "--output_report",
        type=str,
        default=str(_BENCH / "delf_head_ttk_report.json"),
    )
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = Path(args.dataset_path)
    meta_path = dataset_path / "dataset_metadata.json"
    images_dir = dataset_path / "images"
    if not meta_path.is_file():
        print(f"Нет {meta_path}", file=sys.stderr)
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    by_pano: dict[str, list[dict]] = defaultdict(list)
    for row in meta["images"]:
        fn = row["filename"]
        if not (images_dir / fn).is_file():
            continue
        pid = row.get("pano_id") or row["filename"].rsplit("_h", 1)[0]
        by_pano[pid].append(row)

    rng = random.Random(args.seed)
    pano_ids = list(by_pano.keys())
    rng.shuffle(pano_ids)

    if args.max_panos > 0:
        pano_ids = pano_ids[: args.max_panos]

    if len(pano_ids) < 2:
        print("Нужно минимум 2 панорамы с файлами на диске.", file=sys.stderr)
        sys.exit(1)

    n_val = max(1, int(len(pano_ids) * args.val_ratio))
    n_val = min(n_val, len(pano_ids) - 1)
    val_panos = set(pano_ids[:n_val])
    train_panos = set(pano_ids[n_val:])

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for pid in train_panos:
        train_rows.extend(by_pano[pid])
    for pid in val_panos:
        val_rows.extend(by_pano[pid])

    print(f"Панорам: train={len(train_panos)} val={len(val_panos)}; кадров: train={len(train_rows)} val={len(val_rows)}")

    cache_path = Path(args.cache_npz) if args.cache_npz else None
    X_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None

    if cache_path and cache_path.is_file():
        print(f"Загрузка кэша {cache_path}")
        z = np.load(cache_path, allow_pickle=True)
        X_train = z["X_train"]
        y_train = z["y_train"]
        X_val = z["X_val"]
        y_val = z["y_val"]
    else:
        tf, hub_mod = setup_tensorflow_imports()
        run_delf = build_delf_fn(tf, hub_mod)
        desc_holder = [None]

        def extract_rows(rows: list[dict], desc: str) -> tuple[np.ndarray, np.ndarray]:
            X_list: list[np.ndarray] = []
            y_list: list[tuple[float, float]] = []
            for row in tqdm(rows, desc=desc):
                pil = Image.open(images_dir / row["filename"])
                emb = extract_pooled_from_pil(pil, run_delf, tf, desc_holder)
                X_list.append(emb)
                y_list.append((float(row["latitude"]), float(row["longitude"])))
            return np.stack(X_list, axis=0), np.array(y_list, dtype=np.float64)

        X_train, y_train = extract_rows(train_rows, "DELF train")
        X_val, y_val = extract_rows(val_rows, "DELF val")

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )
            print(f"Кэш сохранён: {cache_path}")

    assert X_train is not None and y_train is not None and X_val is not None and y_val is not None

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        solver="adam",
        alpha=1e-2,
        batch_size=64,
        learning_rate="adaptive",
        max_iter=args.epochs_mlp,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
    )
    inner = Pipeline([("scaler_x", StandardScaler()), ("mlp", mlp)])
    pipe = TransformedTargetRegressor(
        regressor=inner,
        transformer=StandardScaler(),
    )
    print("Обучение MLP (масштабирование X и lat/lon)…")
    pipe.fit(X_train, y_train)

    pred_val = pipe.predict(X_val)
    errors = [
        haversine_km(y_val[i, 0], y_val[i, 1], pred_val[i, 0], pred_val[i, 1]) for i in range(len(y_val))
    ]
    mean_km = float(np.mean(errors))
    median_km = float(np.median(errors))
    within_1 = 100.0 * float(np.mean(np.array(errors) <= 1.0))
    within_5 = 100.0 * float(np.mean(np.array(errors) <= 5.0))
    within_25 = 100.0 * float(np.mean(np.array(errors) <= 25.0))

    pred_train = pipe.predict(X_train)
    err_tr = [
        haversine_km(y_train[i, 0], y_train[i, 1], pred_train[i, 0], pred_train[i, 1])
        for i in range(len(y_train))
    ]
    train_mean = float(np.mean(err_tr))
    train_median = float(np.median(err_tr))

    out_model = Path(args.output_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib_dump(
        {"pipeline": pipe, "meta": {"hub": "https://tfhub.dev/google/delf/1", "embedding": "mean_pool"}},
        out_model,
    )

    report = {
        "method": "DELF_TFHub_mean_pool_MLP_head",
        "dataset_path": str(dataset_path.resolve()),
        "train_panos": len(train_panos),
        "val_panos": len(val_panos),
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "seed": args.seed,
        "val_mean_error_km": mean_km,
        "val_median_error_km": median_km,
        "val_within_1km_pct": within_1,
        "val_within_5km_pct": within_5,
        "val_within_25km_pct": within_25,
        "train_mean_error_km": train_mean,
        "train_median_error_km": train_median,
        "mlp_max_iter": args.epochs_mlp,
        "model_path": str(out_model.resolve()),
    }

    out_rep = Path(args.output_report)
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== DELF + MLP (дообученная голова на TTK) ===")
    print(f"  Val mean / median error: {mean_km:.2f} / {median_km:.2f} km")
    print(f"  Val within 1 / 5 / 25 km: {within_1:.1f}% / {within_5:.1f}% / {within_25:.1f}%")
    print(f"  Train mean / median error: {train_mean:.2f} / {train_median:.2f} km")
    print(f"  Модель: {out_model}")
    print(f"  Отчёт: {out_rep}")


if __name__ == "__main__":
    main()
