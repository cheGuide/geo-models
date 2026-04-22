#!/usr/bin/env python3
"""
Структура под deep-visual-geo-localization-benchmark:
  datasets_vg/datasets/ttk/images/{train,val,test}/{database,queries}
Имена: ...@easting@northing@.jpg (UTM EPSG:32637).

- train / val: БД = h0, query = h120 (тот же pano → позитив)
- test: БД = h0, query = h240 (как раньше)

Непересекающиеся pano между train, val, test (после shuffle).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil as sh

        sh.copy2(src, dst)


def clear_dir(d: Path) -> None:
    if not d.is_dir():
        return
    for child in d.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ttk_root", type=Path, default=REPO / "ttk_10k_full")
    p.add_argument("--out", type=Path, default=REPO / "datasets_vg" / "datasets" / "ttk")
    p.add_argument("--train_panos", type=int, default=1200)
    p.add_argument("--val_panos", type=int, default=300)
    p.add_argument("--test_panos", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clear", action="store_true", help="Очистить images/train,val,test перед заполнением")
    args = p.parse_args()

    try:
        from pyproj import Transformer
    except ImportError:
        print("Нужен pyproj: pip install pyproj", file=sys.stderr)
        sys.exit(1)

    meta_path = args.ttk_root / "dataset_metadata.json"
    if not meta_path.is_file():
        print(f"Нет {meta_path}", file=sys.stderr)
        sys.exit(1)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    by_pano: dict[str, list] = defaultdict(list)
    for im in meta["images"]:
        by_pano[im["pano_id"]].append(im)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)

    panos_ok = []
    for pano, xs in by_pano.items():
        if len(xs) < 3:
            continue
        h0 = next((x for x in xs if str(x["filename"]).endswith("_h0.jpg")), None)
        h120 = next((x for x in xs if str(x["filename"]).endswith("_h120.jpg")), None)
        h240 = next((x for x in xs if str(x["filename"]).endswith("_h240.jpg")), None)
        if h0 and h120 and h240:
            panos_ok.append((pano, h0, h120, h240))

    random.seed(args.seed)
    random.shuffle(panos_ok)

    need = args.train_panos + args.val_panos + args.test_panos
    if len(panos_ok) < need:
        print(f"Предупреждение: доступно {len(panos_ok)} pano, запрошено {need}. Уменьшаю доли.", file=sys.stderr)
        scale = len(panos_ok) / need
        args.train_panos = int(args.train_panos * scale)
        args.val_panos = int(args.val_panos * scale)
        args.test_panos = len(panos_ok) - args.train_panos - args.val_panos

    train_list = panos_ok[: args.train_panos]
    val_list = panos_ok[args.train_panos : args.train_panos + args.val_panos]
    test_list = panos_ok[args.train_panos + args.val_panos : args.train_panos + args.val_panos + args.test_panos]

    img_root = args.out / "images"
    if args.clear:
        for split in ("train", "val", "test"):
            clear_dir(img_root / split / "database")
            clear_dir(img_root / split / "queries")

    def fill_split(split_name: str, panos: list, query_key: str) -> int:
        """query_key: 'h120' или 'h240'"""
        db_dir = img_root / split_name / "database"
        q_dir = img_root / split_name / "queries"
        db_dir.mkdir(parents=True, exist_ok=True)
        q_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for pano, h0, h120, h240 in panos:
            qimg = h120 if query_key == "h120" else h240
            lon, lat = float(h0["longitude"]), float(h0["latitude"])
            e, nn = transformer.transform(lon, lat)
            tag = f"{e:.2f}@{nn:.2f}"
            base = f"p{n:05d}@{tag}@.jpg"
            src0 = args.ttk_root / "images" / h0["filename"]
            srcq = args.ttk_root / "images" / qimg["filename"]
            if not src0.is_file() or not srcq.is_file():
                continue
            link_or_copy(src0, db_dir / base)
            link_or_copy(srcq, q_dir / base)
            n += 1
        return n

    nt = fill_split("train", train_list, "h120")
    nv = fill_split("val", val_list, "h120")
    nte = fill_split("test", test_list, "h240")

    print(f"train: {nt} pano (h0 БД, h120 query)")
    print(f"val:   {nv} pano (h0 БД, h120 query)")
    print(f"test:  {nte} pano (h0 БД, h240 query)")
    print(f"OUT={args.out}")
    print(f"DATASETS_FOLDER={args.out.parent}")


if __name__ == "__main__":
    main()
