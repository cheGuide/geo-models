#!/usr/bin/env python3
"""
PlaNet-style геолокация на TTK: классификация по фиксированной сетке lat/lon
(как в PlaNet — разбиение поверхности на ячейки + CNN), без официального кода Google.

Зависимости: torch, torchvision, pillow, tqdm

Пример:
  python benchmarks_results/planet_style_ttk.py --dataset_path data/ttk_10k_full --max_rows 400 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from tools.require_cuda import require_cuda_or_cpu


class TTKGridDataset(Dataset):
    """Один кадр на pano_id; метка = индекс ячейки в сетке GRID×GRID по нормализованным lat/lon."""

    def __init__(
        self,
        rows: list[dict],
        images_dir: Path,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        grid: int,
        image_size: int = 224,
    ):
        self.paths: list[Path] = []
        self.labels: list[int] = []
        self.lats: list[float] = []
        self.lons: list[float] = []
        self.grid = grid
        seen: set[str] = set()
        for r in rows:
            pid = r.get("pano_id") or r["filename"]
            if pid in seen:
                continue
            seen.add(pid)
            p = images_dir / r["filename"]
            if not p.is_file():
                continue
            lat, lon = float(r["latitude"]), float(r["longitude"])
            gi = min(grid - 1, int((lat - lat_min) / max(lat_max - lat_min, 1e-9) * grid))
            gj = min(grid - 1, int((lon - lon_min) / max(lon_max - lon_min, 1e-9) * grid))
            label = gi * grid + gj
            self.paths.append(p)
            self.labels.append(label)
            self.lats.append(lat)
            self.lons.append(lon)

        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def compute_bbox(rows: list[dict]) -> tuple[float, float, float, float]:
    lats = [float(r["latitude"]) for r in rows]
    lons = [float(r["longitude"]) for r in rows]
    return min(lats), max(lats), min(lons), max(lons)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=Path, default=Path("data/ttk_10k_full"))
    ap.add_argument("--grid", type=int, default=32, help="Сетка GRID×GRID (классов = grid²)")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = все уникальные pano")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = require_cuda_or_cpu("cuda")

    meta_path = args.dataset_path / "dataset_metadata.json"
    if not meta_path.is_file():
        raise SystemExit(f"Нет {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    rows = list(meta["images"])
    random.shuffle(rows)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    lat_min, lat_max, lon_min, lon_max = compute_bbox(meta["images"])
    images_dir = args.dataset_path / "images"

    ds_full = TTKGridDataset(rows, images_dir, lat_min, lat_max, lon_min, lon_max, args.grid)
    if len(ds_full) < 10:
        raise SystemExit(f"Слишком мало снимков: {len(ds_full)} (проверьте путь к images)")

    n_train = int(len(ds_full) * args.train_ratio)
    indices = list(range(len(ds_full)))
    random.shuffle(indices)
    train_idx = set(indices[:n_train])
    train_paths = [ds_full.paths[i] for i in range(len(ds_full)) if i in train_idx]
    train_labels = [ds_full.labels[i] for i in range(len(ds_full)) if i in train_idx]
    val_paths = [ds_full.paths[i] for i in range(len(ds_full)) if i not in train_idx]
    val_labels = [ds_full.labels[i] for i in range(len(ds_full)) if i not in train_idx]
    val_lats = [ds_full.lats[i] for i in range(len(ds_full)) if i not in train_idx]
    val_lons = [ds_full.lons[i] for i in range(len(ds_full)) if i not in train_idx]

    class SubsetDS(Dataset):
        def __init__(self, paths, labels, tf, lats=None, lons=None):
            self.paths = paths
            self.labels = labels
            self.tf = tf
            self.lats = lats
            self.lons = lons

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return self.tf(img), self.labels[i]

    tf = ds_full.tf
    train_ds = SubsetDS(train_paths, train_labels, tf)
    val_ds = SubsetDS(val_paths, val_labels, tf, val_lats, val_lons)

    num_classes = args.grid * args.grid
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(
        f"PlaNet-style: сетка {args.grid}x{args.grid} = {num_classes} классов | "
        f"train={len(train_ds)} val={len(val_ds)} | bbox lat[{lat_min:.4f},{lat_max:.4f}] lon[{lon_min:.4f},{lon_max:.4f}]"
    )

    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs} train"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        train_acc = correct / max(total, 1)

        model.eval()
        total_v, correct_v = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct_v += (pred == y).sum().item()
                total_v += y.numel()
        val_acc = correct_v / max(total_v, 1)
        print(f"  train acc={train_acc:.4f}  val acc={val_acc:.4f}  (top-1 по ячейке сетки)")

    # Гаверсин: центр предсказанной ячейки vs истинные координаты — грубая метрика км
    def cell_to_latlon(cell_id: int) -> tuple[float, float]:
        gi = cell_id // args.grid
        gj = cell_id % args.grid
        lat = lat_min + (gi + 0.5) / args.grid * (lat_max - lat_min)
        lon = lon_min + (gj + 0.5) / args.grid * (lon_max - lon_min)
        return lat, lon

    def haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        p = math.pi / 180
        a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
        return 2 * r * math.asin(math.sqrt(a))

    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            x, _y = val_ds[i]
            x = x.unsqueeze(0).to(device)
            cell = model(x).argmax(dim=1).item()
            plat, plon = cell_to_latlon(cell)
            tlat, tlon = val_lats[i], val_lons[i]
            errs.append(haversine_km(plat, plon, tlat, tlon))
    if errs:
        mean_km = sum(errs) / len(errs)
        print(f"  val mean error (центр ячейки vs GT): {mean_km:.2f} km (n={len(errs)})")

    print("Готово. Это упрощённый PlaNet-style sanity-check, не оригинальная модель Google.")


if __name__ == "__main__":
    main()
