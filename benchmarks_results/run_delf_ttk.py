#!/usr/bin/env python3
"""
Оценка DELF (TensorFlow Hub: google/delf/1) на датасете Moscow TTK (ttk_10k_full).

Протокол как в AnyLoc/run_moscow_ttk.py: глобальный эмбеддинг = L2-нормированное среднее
локальных дескрипторов DELF по всем ключевым точкам (упрощение без полного RANSAC-поиска
по всем парам изображений). Метрики: R@1, R@5 (радиус positive_radius_m), mean/median error (км).

Зависимости:
  pip install tensorflow tensorflow-hub packaging

  На Windows при DLL load failed для TFLite автоматически подключается
  tf_windows_tflite_stubs.py (заглушки до import tensorflow) и shim для
  pkg_resources (tensorflow_hub + setuptools 82+).

Usage:
  py benchmarks_results/run_delf_ttk.py --dataset_path c:/Users/q/Work/models/ttk_10k_full --num_samples 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

try:
    from delf_ttk_common import build_delf_fn, extract_pooled_from_pil, setup_tensorflow_imports

    tf, hub = setup_tensorflow_imports()
except ImportError as e:
    print(
        "Нужны пакеты: tensorflow, tensorflow-hub, packaging.\n"
        "  py -3 -m pip install tensorflow tensorflow-hub packaging",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

from PIL import Image
from tqdm import tqdm


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def main():
    parser = argparse.ArgumentParser(description="DELF (TF Hub) на Moscow TTK")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get("TTK_DATASET", str(Path(__file__).resolve().parents[1] / "ttk_10k_full")),
        help="Каталог с dataset_metadata.json и images/",
    )
    parser.add_argument("--num_samples", type=int, default=100, help="0 = все валидные кадры на диске")
    parser.add_argument("--positive_radius_m", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "delf_moscow_ttk_results.json"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    metadata_path = dataset_path / "dataset_metadata.json"
    images_dir = dataset_path / "images"
    if not metadata_path.exists():
        print(f"Нет файла: {metadata_path}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"Нет каталога: {images_dir}", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    raw = meta["images"]
    valid = [img for img in raw if (images_dir / img["filename"]).exists()]
    if args.num_samples <= 0:
        images = valid
    else:
        n = min(args.num_samples, len(valid))
        images = random.sample(valid, n) if n < len(valid) else valid

    if len(images) < 1:
        print("Нет изображений на диске.", file=sys.stderr)
        sys.exit(1)
    if args.num_samples > 0 and len(images) < 10:
        print("Слишком мало кадров для осмысленной оценки (>=10).", file=sys.stderr)
        sys.exit(1)

    print(f"DELF: загрузка модуля tfhub.dev/google/delf/1 …")
    run_delf = build_delf_fn(tf, hub)
    print(f"Кадров: {len(images)}")

    embs: list[np.ndarray] = []
    desc_dim_holder: list[int | None] = [None]
    for img_meta in tqdm(images, desc="DELF features"):
        p = images_dir / img_meta["filename"]
        pil = Image.open(p)
        embs.append(extract_pooled_from_pil(pil, run_delf, tf, desc_dim_holder))

    emb_mat = np.stack(embs, axis=0)
    emb_mat = emb_mat / (np.linalg.norm(emb_mat, axis=1, keepdims=True) + 1e-8)

    positive_radius_km = args.positive_radius_m / 1000.0
    r1, r5 = 0, 0
    errors_km: list[float] = []

    for i in range(len(images)):
        q = emb_mat[i : i + 1]
        sim = (emb_mat @ q.T).squeeze()
        sim[i] = -1e9
        top5_idx = np.argsort(sim)[::-1][:5]

        q_lat, q_lon = images[i]["latitude"], images[i]["longitude"]
        pred_lat = images[top5_idx[0]]["latitude"]
        pred_lon = images[top5_idx[0]]["longitude"]
        err_km = haversine_km(q_lat, q_lon, pred_lat, pred_lon)
        errors_km.append(err_km)

        if err_km <= positive_radius_km:
            r1 += 1
        if any(
            haversine_km(q_lat, q_lon, images[j]["latitude"], images[j]["longitude"]) <= positive_radius_km
            for j in top5_idx
        ):
            r5 += 1

    n_total = len(images)
    r1_pct = 100.0 * r1 / n_total
    r5_pct = 100.0 * r5 / n_total
    mean_err = float(np.mean(errors_km))
    median_err = float(np.median(errors_km))

    results = {
        "method": "DELF_TFHub_mean_pool",
        "hub": "https://tfhub.dev/google/delf/1",
        "num_samples": n_total,
        "positive_radius_m": args.positive_radius_m,
        "R1": r1_pct,
        "R5": r5_pct,
        "mean_error_km": mean_err,
        "median_error_km": median_err,
        "errors_km": [float(e) for e in errors_km[:100]],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== DELF (TF Hub, mean-pool) на TTK ===")
    print(f"  Samples: {n_total}")
    print(f"  Positive radius: {args.positive_radius_m} m")
    print(f"  R@1: {r1_pct:.2f}%")
    print(f"  R@5: {r5_pct:.2f}%")
    print(f"  Mean error: {mean_err:.2f} km")
    print(f"  Median error: {median_err:.2f} km")
    print(f"  Results: {out_path}")


if __name__ == "__main__":
    main()
