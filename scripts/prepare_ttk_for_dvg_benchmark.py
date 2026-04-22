#!/usr/bin/env python3
"""
Подготовка Moscow TTK (dataset_metadata.json + images/) под формат
Deep Visual Geo-localization Benchmark (pitts30k-like):
  DATASETS_FOLDER/moscow_ttk/images/{train|val|test}/{database|queries}/*.jpg

Имена файлов должны содержать UTM в виде: ...@easting@northing@.jpg
(см. datasets_ws.py в репозитории gmberton).

Зависимость: pip install pyproj

Пример:
  python scripts/prepare_ttk_for_dvg_benchmark.py \\
    --ttk_root data/ttk_10k_full \\
    --out_dir D:/datasets_vg/datasets/moscow_ttk \\
    --seed 42

Далее в клоне бенчмарка:
  export DATASETS_FOLDER=/path/to/datasets_vg/datasets
  python eval.py --dataset_name=moscow_ttk --backbone=resnet50conv4 --aggregation=netvlad \\
    --resume=logs/.../best_model.pth

Примечание: методы с именами planet / cplanet в upstream бенчмарка нет — используйте
--aggregation=netvlad|gem|... из parser.py. CPlaNet (Google ECCV 2018) — другой класс методов.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

try:
    from pyproj import Transformer
except ImportError as e:
    raise SystemExit("Установите pyproj: pip install pyproj") from e

# Москва — зона UTM 37N
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)


def lonlat_to_utm_east_north(lon: float, lat: float) -> tuple[float, float]:
    east, north = _TRANSFORMER.transform(lon, lat)
    return float(east), float(north)


def safe_name(east: float, north: float, stem: str) -> str:
    # Уникальное имя с UTM в формате бенчмарка
    return f"{stem}@{east:.2f}@{north:.2f}@.jpg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttk_root", type=Path, required=True, help="Папка с dataset_metadata.json и images/")
    ap.add_argument("--out_dir", type=Path, required=True, help="Куда писать moscow_ttk/images/...")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_queries_ratio", type=float, default=0.7, help="Доля панорам под train")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="Доля от оставшихся под val (относительно всего)")
    ap.add_argument("--copy", action="store_true", help="Копировать файлы вместо symlink")
    ap.add_argument("--limit", type=int, default=0, help="Ограничить число панорам (0 = все)")
    args = ap.parse_args()

    meta_path = args.ttk_root / "dataset_metadata.json"
    img_dir = args.ttk_root / "images"
    if not meta_path.is_file():
        raise SystemExit(f"Нет {meta_path}")
    if not img_dir.is_dir():
        raise SystemExit(f"Нет {img_dir}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Группируем по pano_id: одна «место» — несколько ракурсов (h0/h120/h240)
    pano_to_rows: dict[str, list[dict]] = {}
    for row in meta["images"]:
        pid = row.get("pano_id") or row["filename"]
        pano_to_rows.setdefault(pid, []).append(row)

    pano_ids = sorted(pano_to_rows.keys())
    random.seed(args.seed)
    random.shuffle(pano_ids)
    if args.limit and args.limit > 0:
        pano_ids = pano_ids[: args.limit]

    n = len(pano_ids)
    n_train = int(n * args.train_queries_ratio)
    n_val = int(n * args.val_ratio)
    train_ids = set(pano_ids[:n_train])
    val_ids = set(pano_ids[n_train : n_train + n_val])
    test_ids = set(pano_ids[n_train + n_val :])

    def split_for(pano: str) -> str:
        if pano in train_ids:
            return "train"
        if pano in val_ids:
            return "val"
        return "test"

    # В каждом сплите: database = два ракурса, queries = один (как минимальный sanity VPR)
    # Можно поменять логику под свои эксперименты.
    headings_order = [0, 120, 240]

    for split in ("train", "val", "test"):
        for sub in ("database", "queries"):
            p = args.out_dir / "images" / split / sub
            p.mkdir(parents=True, exist_ok=True)

    def link_or_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)

    processed = 0
    for pano_id, rows in pano_to_rows.items():
        sp = split_for(pano_id)
        if sp == "train" and pano_id not in train_ids:
            continue
        if sp == "val" and pano_id not in val_ids:
            continue
        if sp == "test" and pano_id not in test_ids:
            continue

        by_h = {r.get("heading", 0): r for r in rows}
        # Берём до трёх ракурсов
        imgs: list[dict] = []
        for h in headings_order:
            if h in by_h:
                imgs.append(by_h[h])
        if len(imgs) < 2:
            continue

        # database: первые два кадра; queries: третий (если есть), иначе второй дублируем в queries нельзя — пропуск
        db_rows = imgs[:2]
        q_row = imgs[2] if len(imgs) > 2 else None
        if q_row is None:
            continue

        for row in db_rows:
            lat, lon = float(row["latitude"]), float(row["longitude"])
            east, north = lonlat_to_utm_east_north(lon, lat)
            src = img_dir / row["filename"]
            if not src.is_file():
                continue
            name = safe_name(east, north, Path(row["filename"]).stem)
            dst = args.out_dir / "images" / sp / "database" / name
            link_or_copy(src, dst)

        lat, lon = float(q_row["latitude"]), float(q_row["longitude"])
        east, north = lonlat_to_utm_east_north(lon, lat)
        src = img_dir / q_row["filename"]
        if src.is_file():
            name = safe_name(east, north, Path(q_row["filename"]).stem)
            dst = args.out_dir / "images" / sp / "queries" / name
            link_or_copy(src, dst)

        processed += 1

    print(f"Готово. Обработано панорам (с 3 ракурсами): {processed}")
    print(f"Выход: {args.out_dir}/images/{{train,val,test}}/{{database,queries}}")
    print("Укажите в бенчмарке: --dataset_name=moscow_ttk и --datasets_folder=родитель каталога datasets")


if __name__ == "__main__":
    main()
