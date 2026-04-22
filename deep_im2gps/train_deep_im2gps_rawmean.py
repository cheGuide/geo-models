"""
Deep IM2GPS rawmean — Moscow-TTK Raw Per-Image Mean Protocol.

Architecture:
  - Backbone: ResNet50 (pretrained on ImageNet)
  - Head: Linear(2048, N_classes) — N = unique S2 cells (level 15) from train set
  - Prediction: Top-1 cell centre → Haversine distance to ground truth

Protocol:
  - Raw Per-Image Mean (NO Best-of-3, NO location aggregation)
  - Early stopping on: Validation Mean Haversine Distance (metres), patience 7
  - Optimizer: AdamW (weight_decay=1e-4) + CosineAnnealingLR
  - Loss: CrossEntropyLoss

Output files:
  deep_im2gps_report.png        — 2×3 training dashboard
  deep_im2gps_metrics.csv       — per-epoch metrics log
  best_deep_im2gps.pth          — best model checkpoint
  standardized_reports/18_deep_im2gps.png  — standardised copy
"""
from __future__ import annotations

import csv
import io
import math
import os
import random
import shutil
import site
import sys
from pathlib import Path


def _windows_add_cuda_dll_paths() -> None:
    if sys.platform != "win32":
        return
    try:
        paths = list(site.getsitepackages())
        if hasattr(site, "getusersitepackages"):
            u = site.getusersitepackages()
            if isinstance(u, str) and u:
                paths.append(u)
        for sp in paths:
            tl = Path(sp) / "torch" / "lib"
            if tl.is_dir():
                os.add_dll_directory(str(tl.resolve()))
        extra = os.environ.get("TORCH_EXTRA_DLL_DIRS", "").strip()
        if extra:
            for part in extra.split(os.pathsep):
                p = Path(part.strip())
                if p.is_dir():
                    os.add_dll_directory(str(p.resolve()))
    except (OSError, ValueError):
        pass


_windows_add_cuda_dll_paths()

try:
    import torch
except OSError as e:
    print("Could not load PyTorch (CUDA DLL). Fix CUDA installation.", file=sys.stderr)
    raise SystemExit(1) from e

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

import s2sphere

os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
_REPO         = Path(__file__).resolve().parent
_ROOT         = _REPO.parent
_DEFAULT_TTK  = _ROOT / "ttk_10k_full"
_STD_REPORTS  = _ROOT / "standardized_reports"

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", str(_DEFAULT_TTK)))
METADATA_CSV = Path(os.environ.get("METADATA_CSV",  str(DATASET_ROOT / "splits" / "final_metadata.csv")))
IMAGES_ROOT  = Path(os.environ.get("IMAGES_ROOT",   str(DATASET_ROOT)))
OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR",    str(_REPO)))

S2_LEVEL = int(os.environ.get("S2_LEVEL", "15"))

BEST_CKPT   = OUTPUT_DIR / "best_deep_im2gps.pth"
METRICS_CSV = OUTPUT_DIR / "deep_im2gps_metrics.csv"
REPORT_PNG  = OUTPUT_DIR / "deep_im2gps_report.png"
STD_PNG     = _STD_REPORTS / "18_deep_im2gps.png"

MAX_EPOCHS              = int(os.environ.get("MAX_EPOCHS",              "100"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "7"))
BATCH_SIZE              = int(os.environ.get("BATCH_SIZE",              "64"))
LR                      = float(os.environ.get("LR",                    "1e-4"))
WEIGHT_DECAY            = float(os.environ.get("WEIGHT_DECAY",          "1e-4"))
_NUM_WORKERS            = int(os.environ.get("NUM_WORKERS",             "4"))
SEED                    = 42

BASELINE_M = 3545.0  # Kopernik baseline (metres) — reference line on chart [0,1]


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device() -> torch.device:
    allow_cpu = os.environ.get("ALLOW_CPU", "").lower() in ("1", "true", "yes")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if allow_cpu:
        print("  [WARN] CUDA unavailable — training on CPU.", flush=True)
        return torch.device("cpu")
    print("CUDA not found. Set ALLOW_CPU=1 or fix CUDA.", file=sys.stderr)
    sys.exit(1)


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


# ══════════════════════════════════════════════════════════════════════════════
# S2 helpers
# ══════════════════════════════════════════════════════════════════════════════

def latlon_to_cell_id(lat: float, lon: float, level: int) -> int:
    ll  = s2sphere.LatLng.from_degrees(lat, lon)
    cid = s2sphere.CellId.from_lat_lng(ll).parent(level)
    return cid.id()


def cell_id_to_latlon(cid_int: int) -> tuple[float, float]:
    cid = s2sphere.CellId(cid_int)
    ll  = cid.to_lat_lng()
    return ll.lat().degrees, ll.lng().degrees


# ══════════════════════════════════════════════════════════════════════════════
# Model — ResNet50 + linear classification head
# ══════════════════════════════════════════════════════════════════════════════

class DeepIM2GPS(nn.Module):
    """Deep IM2GPS: ResNet50 backbone + linear classification head over S2 cells."""

    def __init__(self, num_classes: int):
        super().__init__()
        w = getattr(models, "ResNet50_Weights", None)
        backbone = models.resnet50(weights=w.IMAGENET1K_V1 if w else None)
        if w is None:
            backbone = models.resnet50(pretrained=True)  # torchvision < 0.13
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class MoscowGeoDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        images_root: Path,
        cell_to_idx: dict[int, int],
        cell_centres: np.ndarray,
        transform,
    ):
        self.paths       = frame["image_path"].astype(str).tolist()
        self.lats        = frame["latitude"].astype(float).values
        self.lons        = frame["longitude"].astype(float).values
        self.loc_ids     = frame["location_id"].astype(str).tolist()
        self.images_root = images_root
        self.transform   = transform

        # Assign each sample to the nearest train cell (by Haversine on unit sphere)
        self.labels: list[int] = []
        for i in range(len(self.paths)):
            lat, lon = float(self.lats[i]), float(self.lons[i])
            cid = latlon_to_cell_id(lat, lon, S2_LEVEL)
            if cid in cell_to_idx:
                self.labels.append(cell_to_idx[cid])
            else:
                # Val/test point falls outside train vocabulary → nearest centre
                dlat  = np.deg2rad(cell_centres[:, 0] - lat)
                dlon  = np.deg2rad(cell_centres[:, 1] - lon)
                rlat  = math.radians(lat)
                rclat = np.deg2rad(cell_centres[:, 0])
                a = np.sin(dlat / 2) ** 2 + np.cos(rlat) * np.cos(rclat) * np.sin(dlon / 2) ** 2
                a = np.clip(a, 0, 1)
                d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                self.labels.append(int(np.argmin(d)))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.images_root / self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (
            img,
            self.labels[idx],
            self.loc_ids[idx],
            float(self.lats[idx]),
            float(self.lons[idx]),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Transforms (ImageNet statistics)
# ══════════════════════════════════════════════════════════════════════════════

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _collate_geo(batch):
    imgs, labels, lids, lats, lons = zip(*batch)
    return (
        torch.stack(imgs),
        list(labels),
        list(lids),
        list(lats),
        list(lons),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Epoch functions
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for imgs, labels, *_ in tqdm(loader, desc="  train", leave=False, dynamic_ncols=True):
        imgs   = imgs.to(device, non_blocking=True)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss    += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n       += imgs.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float]:
    """Returns (val_loss, top1_acc, top5_acc)."""
    model.eval()
    total_loss, top1_correct, top5_correct, total_n = 0.0, 0, 0, 0
    k = min(5, num_classes)
    for imgs, labels, *_ in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss   += loss.item() * imgs.size(0)
        top1_correct += (logits.argmax(1) == labels).sum().item()
        topk_preds    = logits.topk(k, dim=1).indices
        top5_correct += (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
        total_n      += imgs.size(0)
    return total_loss / total_n, top1_correct / total_n, top5_correct / total_n


@torch.no_grad()
def eval_geo_raw_mean(
    model: nn.Module,
    loader: DataLoader,
    idx_to_cell: dict[int, int],
    device: torch.device,
) -> tuple[float, float, list[float]]:
    """Raw Per-Image Mean — NO location aggregation, NO Best-of-3.

    Returns (raw_mean_m, raw_median_m, per_image_dists).
    """
    model.eval()
    per_image_dists: list[float] = []
    for imgs, labels, loc_ids, true_lats, true_lons in loader:
        imgs     = imgs.to(device, non_blocking=True)
        logits   = model(imgs)
        pred_idx = logits.argmax(1).cpu().numpy()
        for i in range(imgs.size(0)):
            cid_int            = idx_to_cell[int(pred_idx[i])]
            pred_lat, pred_lon = cell_id_to_latlon(cid_int)
            dist = haversine_m(pred_lat, pred_lon, float(true_lats[i]), float(true_lons[i]))
            per_image_dists.append(dist)
    if not per_image_dists:
        return float("nan"), float("nan"), []
    return float(np.mean(per_image_dists)), float(np.median(per_image_dists)), per_image_dists


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard — 2×3, white theme
# ══════════════════════════════════════════════════════════════════════════════

def save_report(
    metrics: list[dict],
    best_epoch: int,
    best_val_mean_m: float,
    num_classes: int,
    early_stopped: bool,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs   = [m["epoch"]      for m in metrics]
    tr_loss  = [m["train_loss"] for m in metrics if m.get("train_loss") is not None]
    tr_ep    = [m["epoch"]      for m in metrics if m.get("train_loss") is not None]
    val_loss = [m["val_loss"]   for m in metrics]
    top1_acc = [m["val_top1"]   for m in metrics]
    top5_acc = [m["val_top5"]   for m in metrics]
    val_mean = [m["val_mean_m"] for m in metrics]

    best_row  = next((m for m in metrics if m["epoch"] == best_epoch), metrics[-1])
    best_dists: list[float] = best_row.get("val_dists", [])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")

    fig.suptitle(
        f"Deep IM2GPS  |  ResNet50 + S2 Cells (L{S2_LEVEL})  |  Raw Per-Image Mean Protocol\n"
        f"Best epoch {best_epoch}  ·  Val Mean {best_val_mean_m:.0f} m  ·  "
        f"S2 classes: {num_classes}  ·  Early stop: {'YES' if early_stopped else 'NO'}",
        fontsize=13, fontweight="bold",
    )

    C_TRAIN = "#FF8C00"
    C_VAL   = "#1E90FF"
    C_BASE  = "#DC143C"

    # ── [0, 0] Loss History ──────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(tr_ep,  tr_loss,  color=C_TRAIN, label="Train CE Loss", linewidth=1.8)
    ax.plot(epochs, val_loss, color=C_VAL,   label="Val CE Loss",   linewidth=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss History", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── [0, 1] Val Mean Haversine per epoch ─────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, val_mean, color=C_VAL, marker="o", markersize=3,
            linewidth=1.8, label="Val Mean Haversine (m)")
    ax.axhline(BASELINE_M, color=C_BASE, linestyle="--", linewidth=1.5,
               label=f"Baseline {BASELINE_M:.0f} m")
    if best_epoch >= 0:
        ax.axvline(best_epoch, color="green", linestyle=":", alpha=0.8,
                   label=f"Best ep {best_epoch}")
        ax.scatter([best_epoch], [best_val_mean_m], color="green", zorder=5, s=70)
    ax.set_xlabel("Epoch"); ax.set_ylabel("metres")
    ax.set_title("Geolocation Error (Val Mean Haversine)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [0, 2] Top-1 & Top-5 Accuracy ───────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(epochs, [v * 100 for v in top1_acc], color=C_TRAIN,
            linewidth=1.8, label="Top-1 Acc (%)")
    ax.plot(epochs, [v * 100 for v in top5_acc], color=C_VAL,
            linewidth=1.8, linestyle="--", label="Top-5 Acc (%)")
    if best_epoch >= 0:
        ax.axvline(best_epoch, color="green", linestyle=":", alpha=0.8,
                   label=f"Best ep {best_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy (Val)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [1, 0] Error Distribution Histogram (best epoch) ────────────────────
    ax = axes[1, 0]
    X_MAX = 10_000.0
    clipped = [min(d, X_MAX) for d in best_dists] if best_dists else [0.0]
    bins = np.linspace(0, X_MAX, 51)
    ax.hist(clipped, bins=bins, color=C_VAL, alpha=0.75, edgecolor="white",
            label=f"ep{best_epoch}  n={len(best_dists)}")
    if best_dists:
        mu = float(np.mean(best_dists))
        ax.axvline(mu, color=C_TRAIN, linestyle="--", linewidth=1.5,
                   label=f"Mean {mu:.0f} m")
        ax.axvline(BASELINE_M, color=C_BASE, linestyle=":", linewidth=1.5,
                   label=f"Baseline {BASELINE_M:.0f} m")
    ax.set_xlim(0, X_MAX)
    ax.set_xlabel("Haversine distance (m)"); ax.set_ylabel("Count")
    ax.set_title("Error Distribution (Best Epoch, 0–10 km)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [1, 1] CDF (best epoch) ──────────────────────────────────────────────
    ax = axes[1, 1]
    if best_dists:
        sorted_d = np.sort(best_dists)
        cdf      = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.plot(sorted_d, cdf * 100, color=C_VAL, linewidth=1.8,
                label=f"CDF ep{best_epoch}")
        for thr_m, lbl in [(1000, "1 km"), (3000, "3 km"), (5000, "5 km")]:
            pct = float(np.mean(np.array(best_dists) <= thr_m) * 100)
            ax.axvline(thr_m, color="grey", linestyle=":", alpha=0.6)
            ax.text(thr_m + 80, 5, f"{pct:.1f}%@{lbl}", fontsize=7, color="grey")
        ax.axvline(BASELINE_M, color=C_BASE, linestyle="--", linewidth=1.2,
                   label=f"Baseline {BASELINE_M:.0f} m")
    ax.set_xlim(left=0)
    ax.set_xlabel("Haversine distance (m)"); ax.set_ylabel("Cumulative %")
    ax.set_title("CDF of Errors (Best Epoch)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [1, 2] Summary text table ────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    best_top1 = best_row.get("val_top1", float("nan")) * 100
    best_top5 = best_row.get("val_top5", float("nan")) * 100
    pct_under_base = (
        float(np.mean(np.array(best_dists) <= BASELINE_M) * 100)
        if best_dists else float("nan")
    )
    median_m = float(np.median(best_dists)) if best_dists else float("nan")
    rows = [
        ("Model",            "Deep IM2GPS (ResNet50)"),
        ("Protocol",         "Raw Per-Image Mean"),
        ("S2 Level",         str(S2_LEVEL)),
        ("S2 Classes (N)",   f"{num_classes:,}"),
        ("Best Epoch",       str(best_epoch)),
        ("Val Mean (best)",  f"{best_val_mean_m:.1f} m"),
        ("Val Median (best)", f"{median_m:.1f} m"),
        ("Top-1 Acc (best)", f"{best_top1:.2f}%"),
        ("Top-5 Acc (best)", f"{best_top5:.2f}%"),
        ("% ≤ Baseline",     f"{pct_under_base:.1f}%"),
        ("Baseline",         f"{BASELINE_M:.0f} m (Kopernik)"),
        ("Early Stopping",   f"YES (patience={EARLY_STOPPING_PATIENCE})" if early_stopped else "NO"),
    ]
    tbl = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.52, 0.48],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#1E90FF")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F7F9FC" if r % 2 == 0 else "white")
    ax.set_title("Summary", fontweight="bold", pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Report saved → {out_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    print(f"Device: {device}  |  S2 level: {S2_LEVEL}", flush=True)
    print(f"Protocol: Raw Per-Image Mean  |  Patience: {EARLY_STOPPING_PATIENCE}", flush=True)

    # ── Load metadata ─────────────────────────────────────────────────────────
    print(f"\nLoading metadata from {METADATA_CSV} ...", flush=True)
    df = pd.read_csv(METADATA_CSV)
    print(f"  Total rows: {len(df)}  |  Unique locations: {df['location_id'].nunique()}", flush=True)

    # ── Split 80 / 10 / 10 by location_id ────────────────────────────────────
    loc_ids = df["location_id"].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(df, groups=loc_ids))
    temp_df  = df.iloc[temp_idx]
    temp_ids = temp_df["location_id"].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_rel, test_rel = next(gss2.split(temp_df, groups=temp_ids))
    val_idx  = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)  # noqa: F841 — kept for potential future use
    print(
        f"  Train: {len(train_df)} rows / {train_df['location_id'].nunique()} locs | "
        f"Val: {len(val_df)} / {val_df['location_id'].nunique()} | "
        f"Test: {len(test_df)} / {test_df['location_id'].nunique()}",
        flush=True,
    )

    # ── Build S2 vocabulary from TRAIN ────────────────────────────────────────
    print(f"\nBuilding S2 cell vocabulary (level {S2_LEVEL}) from train set ...", flush=True)
    train_cells = [
        latlon_to_cell_id(float(row.latitude), float(row.longitude), S2_LEVEL)
        for row in train_df.itertuples()
    ]
    unique_cells = list(dict.fromkeys(train_cells))
    cell_to_idx: dict[int, int] = {c: i for i, c in enumerate(unique_cells)}
    idx_to_cell: dict[int, int] = {i: c for c, i in cell_to_idx.items()}
    num_classes = len(unique_cells)

    sample_cell  = s2sphere.Cell(s2sphere.CellId(unique_cells[0]))
    approx_km2   = sample_cell.approx_area() * (6_371.0 ** 2)
    approx_side  = math.sqrt(approx_km2) * 1000
    print(f"  Unique S2 cells (L{S2_LEVEL}): {num_classes}", flush=True)
    print(f"  Approx cell side: ~{approx_side:.0f} m", flush=True)

    cell_centres = np.array([cell_id_to_latlon(c) for c in unique_cells])

    train_ds = MoscowGeoDataset(train_df, IMAGES_ROOT, cell_to_idx, cell_centres, train_transform())
    val_ds   = MoscowGeoDataset(val_df,   IMAGES_ROOT, cell_to_idx, cell_centres, val_transform())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=_NUM_WORKERS, pin_memory=True, collate_fn=_collate_geo,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=True, collate_fn=_collate_geo,
    )

    # ── Model / optimizer / scheduler ────────────────────────────────────────
    model     = DeepIM2GPS(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-7)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Epoch 0 baseline (untrained model) ───────────────────────────────────
    print("\n──── Epoch 0 (BASELINE — untrained model) ────", flush=True)
    val_loss0, val_top1_0, val_top5_0 = eval_epoch(
        model, val_loader, criterion, device, num_classes
    )
    val_mean0, val_med0, val_dists0 = eval_geo_raw_mean(model, val_loader, idx_to_cell, device)
    print(
        f"  Val loss={val_loss0:.4f}  top1={val_top1_0:.4f}  top5={val_top5_0:.4f}  "
        f"Raw Mean={val_mean0:.1f} m",
        flush=True,
    )

    # ── CSV init ──────────────────────────────────────────────────────────────
    fieldnames = [
        "epoch", "train_loss", "train_top1",
        "val_loss", "val_top1", "val_top5", "val_mean_m", "val_median_m",
    ]
    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({
            "epoch": 0, "train_loss": "", "train_top1": "",
            "val_loss": val_loss0, "val_top1": val_top1_0, "val_top5": val_top5_0,
            "val_mean_m": val_mean0, "val_median_m": val_med0,
        })

    all_metrics: list[dict] = [{
        "epoch":      0,
        "train_loss": None,
        "train_top1": None,
        "val_loss":   val_loss0,
        "val_top1":   val_top1_0,
        "val_top5":   val_top5_0,
        "val_mean_m": val_mean0,
        "val_dists":  val_dists0,
    }]

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mean_m = val_mean0
    best_epoch      = 0
    patience_cnt    = 0
    early_stopped   = False
    best_state      = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n──── Epoch {epoch}/{MAX_EPOCHS} ────", flush=True)

        tr_loss, tr_top1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        vl_loss, vl_top1, vl_top5 = eval_epoch(
            model, val_loader, criterion, device, num_classes
        )
        # PRIMARY stopping signal — Raw Validation Mean Haversine (metres)
        val_mean_m, val_med_m, val_dists = eval_geo_raw_mean(
            model, val_loader, idx_to_cell, device
        )
        scheduler.step()

        print(
            f"  Train loss={tr_loss:.4f} top1={tr_top1:.4f} | "
            f"Val loss={vl_loss:.4f} top1={vl_top1:.4f} top5={vl_top5:.4f} | "
            f"Val Raw Mean={val_mean_m:.1f} m",
            flush=True,
        )

        row: dict = {
            "epoch":       epoch,
            "train_loss":  tr_loss,
            "train_top1":  tr_top1,
            "val_loss":    vl_loss,
            "val_top1":    vl_top1,
            "val_top5":    vl_top5,
            "val_mean_m":  val_mean_m,
            "val_median_m": val_med_m,
        }
        with open(METRICS_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        all_metrics.append({**row, "val_dists": val_dists})

        # Early stopping check on Haversine distance
        if val_mean_m < best_val_mean_m - 0.5:
            best_val_mean_m = val_mean_m
            best_epoch      = epoch
            patience_cnt    = 0
            best_state      = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": best_state,
                    "val_mean_m":       val_mean_m,
                    "num_classes":      num_classes,
                    "cell_to_idx":      cell_to_idx,
                    "idx_to_cell":      idx_to_cell,
                    "s2_level":         S2_LEVEL,
                    "protocol":         "raw_per_image_mean",
                },
                str(BEST_CKPT),
            )
            print(f"  ✓ New best saved (epoch {epoch}, val_mean={val_mean_m:.1f} m)", flush=True)
        else:
            patience_cnt += 1
            print(
                f"  No improvement. Patience {patience_cnt}/{EARLY_STOPPING_PATIENCE}",
                flush=True,
            )
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch={best_epoch}", flush=True)
                early_stopped = True
                break

    # ── Final report ──────────────────────────────────────────────────────────
    save_report(
        all_metrics, best_epoch, best_val_mean_m, num_classes, early_stopped, REPORT_PNG
    )

    # Copy to standardized_reports/
    _STD_REPORTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(REPORT_PNG), str(STD_PNG))
    print(f"  Standardized copy → {STD_PNG}", flush=True)

    print("\n══════ Training Complete ══════", flush=True)
    best_m = next((m for m in all_metrics if m["epoch"] == best_epoch), all_metrics[-1])
    print(f"  Best epoch          : {best_epoch}", flush=True)
    print(f"  Best val mean       : {best_val_mean_m:.1f} m", flush=True)
    print(f"  Best val top-1      : {best_m['val_top1'] * 100:.2f}%", flush=True)
    print(f"  Best val top-5      : {best_m['val_top5'] * 100:.2f}%", flush=True)
    print(f"  Early stopped       : {early_stopped}", flush=True)
    print(f"\nDeliverables:\n  {REPORT_PNG}\n  {STD_PNG}\n  {METRICS_CSV}\n  {BEST_CKPT}", flush=True)


if __name__ == "__main__":
    main()
