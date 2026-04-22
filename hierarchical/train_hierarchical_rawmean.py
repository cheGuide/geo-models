"""
Hierarchical GeoLocalization — Moscow TTK dataset (Raw Mean Protocol).

Architecture: Multi-head ResNet50
  Head 1 (Coarse): CrossEntropy classification into 16 geographic sectors (4x4 grid).
  Head 2 (Fine):   Haversine regression for exact GPS (lat/lon normalised to [-1, 1]).

Evaluation: Raw Per-Image Mean (every image contributes equally, NO Best-of-3).
Early stopping: patience=7 on raw val mean Haversine (fine head).
Coarse accuracy (Top-1) logged separately each epoch.

Deliverables (OUTPUT_DIR):
  hierarchical_rawmean_report.png  — white-theme metrics chart
  hierarchical_rawmean_metrics.csv — per-epoch metrics
  best_hierarchical_rawmean.pth    — best checkpoint (lowest raw val mean)
"""
from __future__ import annotations

import collections
import csv
import io
import json
import os
import random
import site
import sys
from pathlib import Path


# ─── Windows CUDA DLL fix ────────────────────────────────────────────────────
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
        if os.environ.get("TORCH_USE_NVIDIA_SITE_DLL_DIRS", "").lower() in ("1", "true", "yes"):
            for sp in paths:
                nvidia_root = Path(sp) / "nvidia"
                if nvidia_root.is_dir():
                    for bin_dir in nvidia_root.glob("*/bin"):
                        if bin_dir.is_dir():
                            os.add_dll_directory(str(bin_dir.resolve()))
    except (OSError, ValueError):
        pass


_windows_add_cuda_dll_paths()

try:
    import torch
except OSError as e:
    print(
        "Could not load PyTorch (CUDA DLL).\n"
        "  1) Microsoft Visual C++ Redistributable x64.\n"
        "  2) Update NVIDIA driver.\n"
        "  3) Set ALLOW_CPU=1 for CPU mode.",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ─── Config ──────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent
_DEFAULT_ROOT = _REPO.parent / "ttk_10k_full"
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", str(_DEFAULT_ROOT)))
METADATA_CSV  = Path(os.environ.get("METADATA_CSV",  str(DATASET_ROOT / "splits" / "final_metadata.csv")))
IMAGES_ROOT   = Path(os.environ.get("IMAGES_ROOT",   str(DATASET_ROOT)))
OUTPUT_DIR    = Path(os.environ.get("OUTPUT_DIR",    str(_REPO)))

BEST_CKPT   = OUTPUT_DIR / "best_hierarchical_rawmean.pth"
LAST_CKPT   = OUTPUT_DIR / "hierarchical_rawmean_last.pth"
METRICS_CSV = OUTPUT_DIR / "hierarchical_rawmean_metrics.csv"
REPORT_PNG  = OUTPUT_DIR / "hierarchical_rawmean_report.png"

MAX_EPOCHS              = int(os.environ.get("MAX_EPOCHS", "80"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "7"))
BATCH_SIZE              = int(os.environ.get("BATCH_SIZE", "32"))
LR                      = float(os.environ.get("LR", "3e-4"))
ETA_MIN                 = float(os.environ.get("ETA_MIN", "1e-7"))
WEIGHT_DECAY            = 1e-4
SEED                    = 42

# Combined loss weights
ALPHA            = float(os.environ.get("ALPHA", "0.4"))  # CE (coarse)
BETA             = float(os.environ.get("BETA",  "0.6"))  # Haversine (fine)
HAVERSINE_NORM_M = 5_000.0                                 # normalise ~5 km -> O(1)

NUM_CLUSTERS = 16  # 4 x 4 geographic grid
GRID_N       = 4   # cells along each axis

# Moscow bounding box
LAT_MIN, LAT_MAX = 55.5, 56.0
LON_MIN, LON_MAX = 37.2, 37.9

_NUM_WORKERS    = int(os.environ.get("NUM_WORKERS", "4"))
RESUME_TRAINING = os.environ.get("RESUME", "1").lower() not in ("0", "false", "no")


# ─── Utilities ───────────────────────────────────────────────────────────────
def resolve_device() -> torch.device:
    allow_cpu = os.environ.get("ALLOW_CPU", "").lower() in ("1", "true", "yes")
    if torch.cuda.is_available():
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if allow_cpu:
        print("  [WARN] CUDA unavailable — CPU mode (slow).", flush=True)
        return torch.device("cpu")
    print("CUDA not found. Set ALLOW_CPU=1 or fix CUDA.", file=sys.stderr)
    sys.exit(1)


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ─── Coordinate helpers ──────────────────────────────────────────────────────
def norm_lat(lat: float | np.ndarray) -> float | np.ndarray:
    return (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * 2 - 1

def norm_lon(lon: float | np.ndarray) -> float | np.ndarray:
    return (lon - LON_MIN) / (LON_MAX - LON_MIN) * 2 - 1

def denorm_lat(v: torch.Tensor) -> torch.Tensor:
    return (v + 1) / 2 * (LAT_MAX - LAT_MIN) + LAT_MIN

def denorm_lon(v: torch.Tensor) -> torch.Tensor:
    return (v + 1) / 2 * (LON_MAX - LON_MIN) + LON_MIN


def lat_lon_to_cluster(lat: float | np.ndarray, lon: float | np.ndarray) -> int | np.ndarray:
    """Map (lat, lon) to cluster id in [0, NUM_CLUSTERS)."""
    row = np.clip(
        np.floor((np.asarray(lat) - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_N).astype(int),
        0, GRID_N - 1,
    )
    col = np.clip(
        np.floor((np.asarray(lon) - LON_MIN) / (LON_MAX - LON_MIN) * GRID_N).astype(int),
        0, GRID_N - 1,
    )
    return row * GRID_N + col


def haversine_m_tensor(
    pred_lat: torch.Tensor, pred_lon: torch.Tensor,
    true_lat: torch.Tensor, true_lon: torch.Tensor,
) -> torch.Tensor:
    """Great-circle distance in metres; inputs in degrees."""
    R = 6_371_000.0
    pl = torch.deg2rad(pred_lat); po = torch.deg2rad(pred_lon)
    tl = torch.deg2rad(true_lat); to = torch.deg2rad(true_lon)
    dphi = tl - pl; dlam = to - po
    a = torch.sin(dphi / 2)**2 + torch.cos(pl) * torch.cos(tl) * torch.sin(dlam / 2)**2
    a = torch.clamp(a, 0.0, 1.0)
    return R * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a))


def haversine_loss_m(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Haversine distance in metres. pred/target in normalised [-1,1]."""
    return haversine_m_tensor(
        denorm_lat(pred[:, 0]),   denorm_lon(pred[:, 1]),
        denorm_lat(target[:, 0]), denorm_lon(target[:, 1]),
    ).mean()


# ─── Collate (module-level for multiprocessing pickling) ─────────────────────
def collate_fn(batch):
    imgs, cls_lbl, fine_tgt, lids = zip(*batch)
    return (
        torch.stack(imgs),
        torch.tensor(cls_lbl, dtype=torch.long),
        torch.stack(fine_tgt),
        list(lids),
    )


# ─── Dataset ─────────────────────────────────────────────────────────────────
class HierarchicalMoscowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, images_root: Path, transform=None):
        self.paths       = frame["image_path"].astype(str).tolist()
        self.lats        = frame["latitude"].astype(float).values
        self.lons        = frame["longitude"].astype(float).values
        self.loc_ids     = frame["location_id"].astype(str).tolist()
        self.clusters    = lat_lon_to_cluster(self.lats, self.lons).tolist()
        self.images_root = images_root
        self.transform   = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.images_root / self.paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        fine_target = torch.tensor(
            [norm_lat(float(self.lats[idx])), norm_lon(float(self.lons[idx]))],
            dtype=torch.float32,
        )
        coarse_label = int(self.clusters[idx])
        return img, coarse_label, fine_target, self.loc_ids[idx]


# ─── Model ───────────────────────────────────────────────────────────────────
class HierarchicalGeoModel(nn.Module):
    """ResNet50 backbone with two heads: coarse (16-class) + fine (lat/lon regression)."""

    def __init__(self, num_clusters: int = NUM_CLUSTERS):
        super().__init__()
        try:
            w = models.ResNet50_Weights.IMAGENET1K_V1
            resnet = models.resnet50(weights=w)
        except AttributeError:
            resnet = models.resnet50(pretrained=True)

        in_f = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet

        self.coarse_head = nn.Sequential(
            nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_clusters),
        )
        self.fine_head = nn.Sequential(
            nn.Linear(in_f + num_clusters, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 2), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)                              # (B, 2048)
        coarse_logits = self.coarse_head(feat)               # (B, 16)
        cluster_prob  = F.softmax(coarse_logits, dim=-1).detach()
        fine_in  = torch.cat([feat, cluster_prob], dim=-1)  # (B, 2064)
        fine_out = self.fine_head(fine_in)                   # (B, 2)
        return coarse_logits, fine_out


# ─── Loss ────────────────────────────────────────────────────────────────────
def combined_loss(
    coarse_logits: torch.Tensor, fine_preds: torch.Tensor,
    coarse_labels: torch.Tensor, fine_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce  = F.cross_entropy(coarse_logits, coarse_labels)
    hav = haversine_loss_m(fine_preds, fine_targets)
    total = ALPHA * ce + BETA * (hav / HAVERSINE_NORM_M)
    return total, ce, hav


# ─── Train / Eval ─────────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module, loader: DataLoader,
    optimizer: optim.Optimizer, device: torch.device,
) -> dict:
    model.train()
    tot_loss = tot_ce = tot_hav = tot_coarse_correct = n = 0
    for imgs, coarse_lbl, fine_tgt, _ in tqdm(loader, desc="train", leave=False, file=sys.stdout):
        imgs       = imgs.to(device)
        coarse_lbl = coarse_lbl.to(device)
        fine_tgt   = fine_tgt.to(device)

        coarse_logits, fine_preds = model(imgs)
        loss, ce, hav = combined_loss(coarse_logits, fine_preds, coarse_lbl, fine_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        tot_loss           += loss.item() * bs
        tot_ce             += ce.item()   * bs
        tot_hav            += hav.item()  * bs
        tot_coarse_correct += (coarse_logits.argmax(1) == coarse_lbl).sum().item()
        n                  += bs

    if n == 0:
        return dict(loss=0, ce=0, hav_m=0, coarse_acc=0)
    return dict(
        loss       = tot_loss / n,
        ce         = tot_ce   / n,
        hav_m      = tot_hav  / n,
        coarse_acc = tot_coarse_correct / n * 100,
    )


@torch.no_grad()
def evaluate_raw(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> dict:
    """Raw per-image evaluation — NO Best-of-3 aggregation."""
    model.eval()
    all_errors: list[float] = []
    coarse_correct = n = 0

    for imgs, coarse_lbl, fine_tgt, _ in tqdm(loader, desc="eval", leave=False, file=sys.stdout):
        imgs       = imgs.to(device)
        coarse_lbl = coarse_lbl.to(device)
        fine_tgt   = fine_tgt.to(device)

        coarse_logits, fine_preds = model(imgs)
        coarse_correct += (coarse_logits.argmax(1) == coarse_lbl).sum().item()
        n              += imgs.size(0)

        dist = haversine_m_tensor(
            denorm_lat(fine_preds[:, 0]), denorm_lon(fine_preds[:, 1]),
            denorm_lat(fine_tgt[:, 0]),   denorm_lon(fine_tgt[:, 1]),
        )
        all_errors.extend(dist.cpu().numpy().tolist())

    if n == 0:
        return dict(raw_mean_m=0.0, raw_median_m=0.0, coarse_acc=0.0,
                    acc100_pct=0.0, acc1km_pct=0.0)

    errors = np.array(all_errors, dtype=np.float64)
    return dict(
        raw_mean_m   = float(errors.mean()),
        raw_median_m = float(np.median(errors)),
        coarse_acc   = coarse_correct / n * 100,
        acc100_pct   = float((errors < 100).mean() * 100),
        acc1km_pct   = float((errors < 1000).mean() * 100),
    )


# ─── Report ──────────────────────────────────────────────────────────────────
def save_report(
    history: list[dict],
    test_results: dict,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    epochs       = [h["epoch"]        for h in history]
    tr_cacc      = [h["train_coarse_acc"] for h in history]
    val_cacc     = [h["val_coarse_acc"]   for h in history]
    val_raw_mean = [h["val_raw_mean_m"]   for h in history]
    val_raw_med  = [h["val_raw_median_m"] for h in history]
    tr_loss      = [h["train_loss"]       for h in history]

    # White theme
    BG_FIG   = "white"
    BG_AX    = "#f8f8f8"
    GRID_C   = "#dddddd"
    TEXT_C   = "#111111"
    C_TRAIN  = "#1a73e8"   # blue
    C_VAL    = "#d32f2f"   # red
    C_MEAN   = "#2e7d32"   # green
    C_MEDIAN = "#f57c00"   # orange

    def _style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(BG_AX)
        ax.set_title(title, color=TEXT_C, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color=TEXT_C, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_C, fontsize=9)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.grid(True, color=GRID_C, linewidth=0.6, linestyle="--", alpha=0.8)

    fig = plt.figure(figsize=(16, 13), facecolor=BG_FIG)
    gs  = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=[1.0, 1.0, 0.82],
        hspace=0.50, wspace=0.35,
        left=0.08, right=0.96, top=0.90, bottom=0.06,
    )

    # ── Chart 1: Coarse Classification Accuracy ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "Coarse Classification Accuracy vs Epochs", "Epoch", "Accuracy (%)")
    ax1.plot(epochs, tr_cacc,  color=C_TRAIN, linewidth=1.8, label="Train", marker="o", markersize=3)
    ax1.plot(epochs, val_cacc, color=C_VAL,   linewidth=1.8, label="Val",   marker="s", markersize=3)
    ax1.set_ylim(0, 105)
    ax1.legend(facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)
    if val_cacc:
        best_e = epochs[int(np.argmax(val_cacc))]
        best_v = max(val_cacc)
        ax1.axvline(best_e, color=C_VAL, linewidth=0.8, linestyle=":")
        ax1.annotate(
            f"Best: {best_v:.1f}% @ ep{best_e}",
            xy=(best_e, best_v), xytext=(best_e + 1, best_v - 8),
            color=C_VAL, fontsize=8, arrowprops=dict(arrowstyle="->", color=C_VAL),
        )

    # ── Chart 2: Raw Val Mean vs Epochs ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "Raw Val Mean Haversine Distance vs Epochs", "Epoch", "Distance (m)")
    ax2.plot(epochs, val_raw_mean, color=C_MEAN, linewidth=1.8,
             marker="D", markersize=3, label="Val Raw Mean")
    if val_raw_mean:
        best_e = epochs[int(np.argmin(val_raw_mean))]
        best_v = min(val_raw_mean)
        ax2.axvline(best_e, color=C_MEAN, linewidth=0.8, linestyle=":")
        ax2.annotate(
            f"Min: {best_v:.0f} m @ ep{best_e}",
            xy=(best_e, best_v), xytext=(best_e + 1, best_v + best_v * 0.05),
            color=C_MEAN, fontsize=8, arrowprops=dict(arrowstyle="->", color=C_MEAN),
        )
    ax2.legend(facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)

    # ── Chart 3: Raw Val Median vs Epochs ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "Raw Val Median Haversine Distance vs Epochs", "Epoch", "Distance (m)")
    ax3.plot(epochs, val_raw_med, color=C_MEDIAN, linewidth=1.8,
             marker="^", markersize=3, label="Val Raw Median")
    if val_raw_med:
        best_e = epochs[int(np.argmin(val_raw_med))]
        best_v = min(val_raw_med)
        ax3.axvline(best_e, color=C_MEDIAN, linewidth=0.8, linestyle=":")
        ax3.annotate(
            f"Min: {best_v:.0f} m @ ep{best_e}",
            xy=(best_e, best_v), xytext=(best_e + 1, best_v + best_v * 0.05),
            color=C_MEDIAN, fontsize=8, arrowprops=dict(arrowstyle="->", color=C_MEDIAN),
        )
    ax3.legend(facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)

    # ── Chart 4: Test Error Distribution (per-image raw) ─────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "Test Error Distribution (Raw Per-Image)", "Distance (m)", "# Images")
    test_errors = np.array(test_results.get("per_image_errors", []), dtype=np.float64)
    if len(test_errors):
        clip_max = float(np.percentile(test_errors, 98))
        clipped  = test_errors[test_errors <= clip_max]
        ax4.hist(clipped, bins=50, color=C_MEAN, edgecolor="white", linewidth=0.3, alpha=0.85)
        ax4.axvline(float(np.median(test_errors)), color=C_MEDIAN,
                    linewidth=1.5, linestyle="--",
                    label=f"Median {np.median(test_errors):.0f} m")
        ax4.axvline(float(test_errors.mean()), color=C_TRAIN,
                    linewidth=1.5, linestyle=":",
                    label=f"Mean {test_errors.mean():.0f} m")
        ax4.legend(facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)

    # ── Chart 5: Train loss over epochs ──────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    _style_ax(ax5, "Train Combined Loss over Epochs (alpha*CE + beta*Hav/scale)",
              "Epoch", "Loss")
    if epochs and tr_loss:
        ax5.plot(epochs, tr_loss, color=C_TRAIN, linewidth=2.0,
                 label="train_loss", marker="o", markersize=3)
        i_min = int(np.argmin(tr_loss))
        ax5.axvline(epochs[i_min], color=C_MEAN, linewidth=0.8, linestyle=":")
        ax5.scatter([epochs[i_min]], [tr_loss[i_min]], color=C_MEAN, s=55, zorder=5,
                    edgecolors="black", linewidths=0.5)
    ax5.legend(facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)

    # ── Title ─────────────────────────────────────────────────────────────
    tr = test_results
    n_img    = tr.get("n_images", 0)
    t_mean   = tr.get("raw_mean_m", 0)
    t_median = tr.get("raw_median_m", 0)
    c_acc    = tr.get("coarse_acc", 0)
    acc100   = tr.get("acc100_pct", 0)
    acc1km   = tr.get("acc1km_pct", 0)

    fig.suptitle(
        "Hierarchical GeoLocalization (Raw Mean Protocol) - Moscow TTK-10k\n"
        f"ResNet50 + 16-sector grid  |  Raw Per-Image  |  {n_img} test images\n"
        f"Raw Mean {t_mean:.0f} m  *  Raw Median {t_median:.0f} m  "
        f"*  Coarse Acc {c_acc:.1f}%  *  Acc@100m {acc100:.1f}%  *  Acc@1km {acc1km:.1f}%",
        color=TEXT_C, fontsize=12, fontweight="bold", y=0.97,
    )

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Report saved -> {out_path.name}", flush=True)


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    set_seed(SEED)
    device = resolve_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}", flush=True)
    print("  Hierarchical GeoLocalization (Raw Mean Protocol) - Moscow TTK-10k", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  Device     : {device}", flush=True)
    print(f"  Dataset    : {METADATA_CSV}", flush=True)
    print(f"  Output     : {OUTPUT_DIR}", flush=True)
    print(f"  Clusters   : {NUM_CLUSTERS}  ({GRID_N}x{GRID_N} grid)", flush=True)
    print(f"  Loss a/b   : {ALPHA}/{BETA}", flush=True)
    print(f"  Epochs     : {MAX_EPOCHS}  patience={EARLY_STOPPING_PATIENCE}  (raw val mean)", flush=True)
    print(f"{'='*65}\n", flush=True)

    # ── Load metadata ────────────────────────────────────────────────────
    if not METADATA_CSV.is_file():
        print(f"ERROR: metadata CSV not found: {METADATA_CSV}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(METADATA_CSV)
    print(f"  Loaded {len(df)} rows from CSV.", flush=True)

    def img_exists(p: str) -> bool:
        return (IMAGES_ROOT / p).is_file()
    mask = df["image_path"].apply(img_exists)
    df   = df[mask].reset_index(drop=True)
    print(f"  After file-existence filter: {len(df)} rows.", flush=True)

    if len(df) == 0:
        print("ERROR: No images found. Check IMAGES_ROOT.", file=sys.stderr)
        sys.exit(1)

    # ── Split by location_id (80 / 10 / 10) ─────────────────────────────
    loc_ids   = df["location_id"].values
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, tmp_idx = next(gss_outer.split(df, groups=loc_ids))

    df_tmp  = df.iloc[tmp_idx].reset_index(drop=True)
    loc_tmp = df_tmp["location_id"].values
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_i, test_i = next(gss_inner.split(df_tmp, groups=loc_tmp))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df_tmp.iloc[val_i].reset_index(drop=True)
    df_test  = df_tmp.iloc[test_i].reset_index(drop=True)

    print(f"  Split -> train {len(df_train)} | val {len(df_val)} | test {len(df_test)}", flush=True)

    cluster_counts = collections.Counter(
        lat_lon_to_cluster(df_train["latitude"].values, df_train["longitude"].values).tolist()
    )
    print(f"  Cluster distribution (train): {dict(sorted(cluster_counts.items()))}", flush=True)

    # ── Transforms ──────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = HierarchicalMoscowDataset(df_train, IMAGES_ROOT, train_tf)
    val_ds   = HierarchicalMoscowDataset(df_val,   IMAGES_ROOT, val_tf)
    test_ds  = HierarchicalMoscowDataset(df_test,  IMAGES_ROOT, val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=_NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=_NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=_NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn)

    # ── Model ────────────────────────────────────────────────────────────
    model = HierarchicalGeoModel(NUM_CLUSTERS).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {total_params:.1f}M", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=ETA_MIN,
    )

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch      = 1
    best_val_mean    = float("inf")
    patience_cnt     = 0
    history: list[dict] = []
    metrics_written_header = False

    if RESUME_TRAINING and LAST_CKPT.is_file():
        print(f"  Resuming from {LAST_CKPT.name} ...", flush=True)
        try:
            ckpt = torch.load(str(LAST_CKPT), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(LAST_CKPT), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt.get("epoch", 1) + 1
        best_val_mean = ckpt.get("best_val_mean", float("inf"))
        patience_cnt  = ckpt.get("patience_cnt", 0)
        history       = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch}, best_val_mean={best_val_mean:.1f} m", flush=True)

    # ── Baseline raw eval (before first epoch) ───────────────────────────
    print("\n  --- Baseline (random init) ---", flush=True)
    baseline = evaluate_raw(model, val_loader, device)
    print(
        f"  Baseline  val_raw_mean={baseline['raw_mean_m']:.1f} m  "
        f"val_raw_median={baseline['raw_median_m']:.1f} m  "
        f"coarse_acc={baseline['coarse_acc']:.1f}%",
        flush=True,
    )

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\n  Starting training from epoch {start_epoch}...\n", flush=True)
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        tr  = train_epoch(model, train_loader, optimizer, device)
        val = evaluate_raw(model, val_loader, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        row = dict(
            epoch              = epoch,
            lr                 = lr_now,
            train_loss         = tr["loss"],
            train_ce           = tr["ce"],
            train_hav_m        = tr["hav_m"],
            train_coarse_acc   = tr["coarse_acc"],
            val_raw_mean_m     = val["raw_mean_m"],
            val_raw_median_m   = val["raw_median_m"],
            val_coarse_acc     = val["coarse_acc"],
            val_acc100_pct     = val["acc100_pct"],
            val_acc1km_pct     = val["acc1km_pct"],
            early_stopped      = 0,
        )
        history.append(row)

        # Write metrics CSV incrementally
        with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not metrics_written_header:
                f.seek(0, 2)
                if f.tell() == 0:
                    w.writeheader()
                metrics_written_header = True
            w.writerow(row)

        is_best = val["raw_mean_m"] < best_val_mean - 1.0
        marker  = " [BEST]" if is_best else ""

        print(
            f"  Ep {epoch:3d}/{MAX_EPOCHS}  "
            f"tr_loss={tr['loss']:.4f}  tr_cAcc={tr['coarse_acc']:.1f}%  "
            f"val_mean={val['raw_mean_m']:.0f} m  val_med={val['raw_median_m']:.0f} m  "
            f"val_cAcc={val['coarse_acc']:.1f}%  "
            f"lr={lr_now:.2e}{marker}",
            flush=True,
        )

        # Save last checkpoint for resume
        torch.save(dict(
            epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            best_val_mean=best_val_mean, patience_cnt=patience_cnt,
            history=history,
        ), str(LAST_CKPT))

        # Early stopping on raw val mean
        if is_best:
            best_val_mean = val["raw_mean_m"]
            patience_cnt  = 0
            torch.save(model.state_dict(), str(BEST_CKPT))
            print(f"    + Best model saved (val_raw_mean={best_val_mean:.1f} m)", flush=True)
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                row["early_stopped"] = 1
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(patience={EARLY_STOPPING_PATIENCE})",
                    flush=True,
                )
                break

    print("\n  Training complete.", flush=True)

    # ── Test evaluation ──────────────────────────────────────────────────
    print("\n  Loading best model for test evaluation ...", flush=True)
    if BEST_CKPT.is_file():
        try:
            model.load_state_dict(torch.load(str(BEST_CKPT), map_location=device, weights_only=True))
        except Exception:
            model.load_state_dict(torch.load(str(BEST_CKPT), map_location=device))

    # Collect per-image errors for test set
    model.eval()
    all_test_errors: list[float] = []
    test_coarse_correct = test_n = 0
    with torch.no_grad():
        for imgs, coarse_lbl, fine_tgt, _ in tqdm(test_loader, desc="test", leave=False, file=sys.stdout):
            imgs       = imgs.to(device)
            coarse_lbl = coarse_lbl.to(device)
            fine_tgt   = fine_tgt.to(device)
            coarse_logits, fine_preds = model(imgs)
            test_coarse_correct += (coarse_logits.argmax(1) == coarse_lbl).sum().item()
            test_n += imgs.size(0)
            dist = haversine_m_tensor(
                denorm_lat(fine_preds[:, 0]), denorm_lon(fine_preds[:, 1]),
                denorm_lat(fine_tgt[:, 0]),   denorm_lon(fine_tgt[:, 1]),
            )
            all_test_errors.extend(dist.cpu().numpy().tolist())

    test_errors = np.array(all_test_errors, dtype=np.float64)
    test_results = dict(
        n_images       = test_n,
        raw_mean_m     = float(test_errors.mean())         if len(test_errors) else 0,
        raw_median_m   = float(np.median(test_errors))     if len(test_errors) else 0,
        coarse_acc     = test_coarse_correct / test_n * 100 if test_n else 0,
        acc100_pct     = float((test_errors < 100).mean() * 100)  if len(test_errors) else 0,
        acc500_pct     = float((test_errors < 500).mean() * 100)  if len(test_errors) else 0,
        acc1km_pct     = float((test_errors < 1000).mean() * 100) if len(test_errors) else 0,
        per_image_errors = test_errors,
    )

    print(f"\n{'-'*55}", flush=True)
    print(f"  TEST RESULTS  ({test_results['n_images']} images)", flush=True)
    print(f"  Raw Mean     : {test_results['raw_mean_m']:.1f} m", flush=True)
    print(f"  Raw Median   : {test_results['raw_median_m']:.1f} m", flush=True)
    print(f"  Coarse Acc   : {test_results['coarse_acc']:.1f}%", flush=True)
    print(f"  Acc@100m     : {test_results['acc100_pct']:.1f}%", flush=True)
    print(f"  Acc@500m     : {test_results['acc500_pct']:.1f}%", flush=True)
    print(f"  Acc@1km      : {test_results['acc1km_pct']:.1f}%", flush=True)
    print(f"{'-'*55}\n", flush=True)

    # Append test row to metrics CSV
    test_row = dict(
        epoch="TEST", lr=0,
        train_loss=0, train_ce=0, train_hav_m=0, train_coarse_acc=0,
        val_raw_mean_m=test_results["raw_mean_m"],
        val_raw_median_m=test_results["raw_median_m"],
        val_coarse_acc=test_results["coarse_acc"],
        val_acc100_pct=test_results["acc100_pct"],
        val_acc1km_pct=test_results["acc1km_pct"],
        early_stopped=0,
    )
    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(test_row.keys()))
        w.writerow(test_row)

    # Save result JSON
    result_json = OUTPUT_DIR / "hierarchical_rawmean_test_results.json"
    json_out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in test_results.items()}
    with open(result_json, "w", encoding="utf-8") as jf:
        json.dump(json_out, jf, indent=2)
    print(f"  Test results -> {result_json.name}", flush=True)

    # Save report
    save_report(history, test_results, REPORT_PNG)

    print(f"\n  Deliverables:", flush=True)
    print(f"    {BEST_CKPT.name}", flush=True)
    print(f"    {METRICS_CSV.name}", flush=True)
    print(f"    {REPORT_PNG.name}", flush=True)
    print(f"\n  Done.\n", flush=True)


if __name__ == "__main__":
    main()
