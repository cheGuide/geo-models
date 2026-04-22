#!/usr/bin/env python3
"""
Геолокация по фото: смесь von Mises–Fisher (MvMF) на сфере.

Эталон по идее и параметризации: Mike Izbicki, TensorFlow `mikeizbicki/geolocation`
(`src/image/gps_loss.py`, pos_type='aglm_mix', gmm_distribution='fvm').
Готового официального PyTorch-форка того репозитория нет; здесь — компактный
PyTorch-порт ядра: фиксированная сетка центров компонент на сфере, веса смеси
предсказывает CNN (ResNet18), концентрации kappa обучаемые.

Датасет: TTK (dataset_metadata.json + images/), по умолчанию bbox Москвы как в Kopernik.

Пример:
  python benchmarks_results/run_mvmf_ttk.py --dataset_path ../ttk_10k_full --epochs 10 --max_train 0 \\
    --decode grid --checkpoint benchmarks_results/mvmf_resnet18_ttk.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from tools.require_cuda import require_cuda_or_cpu


def latlon_deg_to_unit_xyz(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
    """Единичный вектор на сфере из широты/долготы в градусах (WGS84)."""
    lat = lat_deg * (math.pi / 180.0)
    lon = lon_deg * (math.pi / 180.0)
    cos_lat = torch.cos(lat)
    x = cos_lat * torch.cos(lon)
    y = cos_lat * torch.sin(lon)
    z = torch.sin(lat)
    v = torch.stack([x, y, z], dim=-1)
    return F.normalize(v, dim=-1, eps=1e-8)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def make_component_grid(
    k: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    K центров смеси: равномерная сетка в bbox, проекция на сферу.
    Возвращает mu (K, 3) и предварительные (lat, lon) в градусах для отчёта.
    """
    side = int(math.ceil(math.sqrt(k)))
    while side * side < k:
        side += 1
    lats = np.linspace(lat_min, lat_max, side)
    lons = np.linspace(lon_min, lon_max, side)
    pts = [(la, lo) for la in lats for lo in lons]
    pts = pts[:k]
    while len(pts) < k:
        pts.append(pts[-1])
    lat_t = torch.tensor([p[0] for p in pts], device=device, dtype=torch.float32)
    lon_t = torch.tensor([p[1] for p in pts], device=device, dtype=torch.float32)
    mu = latlon_deg_to_unit_xyz(lat_t, lon_t)
    return mu, torch.stack([lat_t, lon_t], dim=1)


def log_vmf_sphere_pdf(
    mu: torch.Tensor,
    kappa: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    log p(x | mu, kappa) для vMF на S^2 в R^3.
    mu: (K, 3), kappa: (K,), x: (B, 3)
    Возвращает (B, K).
    """
    # log C_3(kappa) = log kappa - log(4pi) - log(sinh(kappa))
    kappa = kappa.clamp(min=1e-4)
    log_c = torch.log(kappa) - math.log(4 * math.pi) - torch.log(torch.sinh(kappa))
    dots = x @ mu.T  # (B, K)
    return log_c + kappa * dots


class MvMFGeo(nn.Module):
    def __init__(self, num_components: int, backbone: nn.Module, feat_dim: int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(feat_dim, num_components)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        z = self.backbone(images)
        return self.fc(z)


class TTKGeoDataset(Dataset):
    def __init__(self, records: list[dict], images_root: Path, transform):
        self.records = records
        self.images_root = images_root
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        path = self.images_root / r["filename"]
        img = Image.open(path).convert("RGB")
        t = self.transform(img)
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        return t, lat, lon


def main():
    parser = argparse.ArgumentParser(description="MvMF geolocation (PyTorch) on TTK")
    parser.add_argument("--dataset_path", type=str, default=str(_REPO / "ttk_10k_full"))
    parser.add_argument("--lat_min", type=float, default=55.5)
    parser.add_argument("--lat_max", type=float, default=56.0)
    parser.add_argument("--lon_min", type=float, default=37.2)
    parser.add_argument("--lon_max", type=float, default=37.9)
    parser.add_argument("--k", type=int, default=256, help="число компонент смеси")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train", type=int, default=0, help="0 = все записи в bbox")
    parser.add_argument("--max_eval", type=int, default=500, help="0 = вся val-выборка")
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument(
        "--decode",
        type=str,
        choices=("grid", "sphere"),
        default="grid",
        help="grid: взвешенные lat/lon центров ячеек; sphere: проекция взвешенного mu на сфере",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="путь для сохранения весов (.pt): backbone + fc + pre_kappa",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="продолжить с чекпоинта (тот же формат, что --checkpoint)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="только валидация val-сплита по чекпоинту (--resume обязателен); обучение пропускается",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt_eval: dict | None = None
    if args.eval_only:
        if not args.resume or not Path(args.resume).is_file():
            print("Для --eval_only укажите существующий файл --resume <чекпоинт.pt>", file=sys.stderr)
            return 1
        ckpt_eval = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved = ckpt_eval.get("args") or {}
        for key in (
            "k",
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
            "train_ratio",
            "seed",
            "decode",
            "max_train",
            "max_eval",
            "epochs",
        ):
            if key in saved and saved[key] is not None:
                setattr(args, key, saved[key])
        print(f"eval_only: загружены гиперпараметры из чекпоинта (k={args.k}, decode={args.decode})", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    meta_path = dataset_path / "dataset_metadata.json"
    images_path = dataset_path / "images"
    if not meta_path.is_file():
        print(f"Нет {meta_path}", file=sys.stderr)
        return 1

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    records = [
        r
        for r in meta["images"]
        if args.lat_min < r["latitude"] < args.lat_max
        and args.lon_min < r["longitude"] < args.lon_max
        and (images_path / r["filename"]).is_file()
    ]
    random.shuffle(records)
    if args.max_train > 0:
        records = records[: args.max_train]

    n = len(records)
    n_train = int(n * args.train_ratio)
    train_recs, val_recs = records[:n_train], records[n_train:]
    print(f"Записей в bbox: {n} | train {len(train_recs)} | val {len(val_recs)}", flush=True)

    device = require_cuda_or_cpu(args.device)

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = TTKGeoDataset(train_recs, images_path, tfm)
    val_ds = TTKGeoDataset(val_recs, images_path, tfm)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_bs = min(32, max(1, len(val_ds)))
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=0)

    w = getattr(models, "ResNet18_Weights", None)
    backbone = models.resnet18(weights=w.IMAGENET1K_V1 if w else None)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    model = MvMFGeo(args.k, backbone, feat_dim).to(device)

    mu, comp_latlon = make_component_grid(
        args.k, args.lat_min, args.lat_max, args.lon_min, args.lon_max, device
    )
    mu = mu.detach()
    comp_latlon = comp_latlon.detach()
    # Обучаемые концентрации (положительные): pre_kappa -> kappa = softplus
    pre_kappa = nn.Parameter(torch.full((args.k,), 3.0, device=device))

    prev_trained_epochs = 0
    if args.eval_only and ckpt_eval is not None:
        ckpt = ckpt_eval
        model.load_state_dict(ckpt["model_state_dict"])
        pre_kappa = nn.Parameter(ckpt["pre_kappa"].to(device))
        prev_trained_epochs = int(ckpt.get("epoch", 0))
        print(
            f"eval_only: веса из {args.resume} (эпох в чекпоинте: {prev_trained_epochs}), обучение пропущено",
            flush=True,
        )
    elif args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        pre_kappa = nn.Parameter(ckpt["pre_kappa"].to(device))
        prev_trained_epochs = int(ckpt.get("epoch", 0))
        print(
            f"Загружен чекпоинт {args.resume} (уже обучено эпох: {prev_trained_epochs}), "
            f"добавляем {args.epochs} эп.",
            flush=True,
        )

    opt = None if args.eval_only else torch.optim.Adam(list(model.parameters()) + [pre_kappa], lr=args.lr)

    def forward_nll(batch_x: torch.Tensor, batch_lat: torch.Tensor, batch_lon: torch.Tensor):
        batch_lat = batch_lat.float()
        batch_lon = batch_lon.float()
        x_unit = latlon_deg_to_unit_xyz(batch_lat, batch_lon)
        logits = model(batch_x)
        log_pi = F.log_softmax(logits, dim=1)
        kappa = F.softplus(pre_kappa) + 1e-3
        log_p = log_vmf_sphere_pdf(mu, kappa, x_unit)
        # log sum_k pi_k p_k(x) = logsumexp(log_pi + log_p)
        nll = -torch.logsumexp(log_pi + log_p, dim=1)
        return nll.mean()

    if not args.eval_only:
        print("Обучение MvMF (PyTorch, ResNet18 + смесь vMF)...", flush=True)
        model.train()
        for ep in range(1, args.epochs + 1):
            losses = []
            tag = f"{ep}/{args.epochs}"
            if prev_trained_epochs:
                tag = f"{prev_trained_epochs + ep}/{prev_trained_epochs + args.epochs}"
            for imgs, lats, lons in tqdm(train_loader, desc=f"epoch {tag}"):
                imgs = imgs.to(device, non_blocking=True)
                lats = lats.to(device)
                lons = lons.to(device)
                opt.zero_grad()
                loss = forward_nll(imgs, lats, lons)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(f"  train NLL: {float(np.mean(losses)):.4f}", flush=True)

    total_epochs = prev_trained_epochs + (0 if args.eval_only else args.epochs)
    if args.checkpoint and not args.eval_only:
        ckpt_path = Path(args.checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "pre_kappa": pre_kappa.detach().cpu(),
                "epoch": total_epochs,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"Чекпоинт сохранён: {ckpt_path} (всего эпох обучения: {total_epochs})", flush=True)

    # Оценка: grid — взвешенное lat/lon по центрам ячеек; sphere — проекция mu на сфере
    model.eval()
    val_records = val_recs
    if args.max_eval > 0:
        val_records = val_records[: args.max_eval]

    errors_km = []
    with torch.no_grad():
        for i in range(0, len(val_records), val_bs):
            chunk = val_records[i : i + val_bs]
            tensors = []
            gt_lat, gt_lon = [], []
            for r in chunk:
                img = Image.open(images_path / r["filename"]).convert("RGB")
                tensors.append(tfm(img))
                gt_lat.append(r["latitude"])
                gt_lon.append(r["longitude"])
            batch = torch.stack(tensors).to(device)
            logits = model(batch)
            pi = F.softmax(logits, dim=1)
            if args.decode == "grid":
                ll = pi @ comp_latlon
                pred_lat = ll[:, 0].cpu().numpy()
                pred_lon = ll[:, 1].cpu().numpy()
            else:
                pred_xyz = F.normalize(pi @ mu, dim=1, eps=1e-8)
                px, py, pz = pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2]
                pred_lat = (
                    torch.asin(pz.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / math.pi)
                ).cpu().numpy()
                pred_lon = (torch.atan2(py, px) * (180.0 / math.pi)).cpu().numpy()
            err = haversine_km(
                np.array(gt_lat), np.array(gt_lon), pred_lat, pred_lon
            )
            errors_km.extend(err.tolist())

    errors_km = np.array(errors_km)
    med = float(np.median(errors_km))
    mean = float(np.mean(errors_km))
    w1 = float(np.mean(errors_km <= 1.0))
    w5 = float(np.mean(errors_km <= 5.0))
    w25 = float(np.mean(errors_km <= 25.0))

    print("\n=== MvMF (PyTorch) — валидация TTK (bbox Москва) ===", flush=True)
    ep_this = 0 if args.eval_only else args.epochs
    print(
        f"  K={args.k}, decode={args.decode}, эпох в этом запуске={ep_this}, "
        f"всего эпох (чекпоинт)={total_epochs}, n_val={len(errors_km)}",
        flush=True,
    )
    print(f"  mean error: {mean:.3f} km", flush=True)
    print(f"  median error: {med:.3f} km", flush=True)
    print(f"  within 1 km:  {100*w1:.1f}%", flush=True)
    print(f"  within 5 km:  {100*w5:.1f}%", flush=True)
    print(f"  within 25 km: {100*w25:.1f}%", flush=True)

    out_json = Path(__file__).resolve().parent / "mvmf_ttk_eval.json"
    report = {
        "eval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "MvMF_vMF_ResNet18",
        "reference": "mikeizbicki/geolocation (TensorFlow) gps_loss aglm_mix; PyTorch port",
        "dataset_path": str(dataset_path),
        "bbox": {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
        },
        "k_components": args.k,
        "epochs_this_run": 0 if args.eval_only else args.epochs,
        "eval_only": args.eval_only,
        "epochs_total": total_epochs,
        "decode": args.decode,
        "train_ratio": args.train_ratio,
        "n_train": len(train_recs),
        "n_val": len(errors_km),
        "mean_km": mean,
        "median_km": med,
        "within_1km": w1,
        "within_5km": w5,
        "within_25km": w25,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "resume": str(args.resume) if args.resume else None,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    full_report = Path(__file__).resolve().parent / "mvmf_ttk_full_report.json"
    with open(full_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {out_json} и {full_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
