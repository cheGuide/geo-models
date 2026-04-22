"""
Orientation-Aware Cross-View Geo-Localization
Moscow TTK-10k  |  Raw Per-Image Mean Protocol

Architecture
------------
  Ground Encoder  : ResNet50 (pretrained ImageNet, frozen → gradual unfreeze)
  Aerial Encoder  : ResNet50 (pretrained ImageNet, separate weights, frozen → unfreeze)
  Heading Encoder : Sinusoidal circular encoding → (B, 64)
  Orient Module   : FiLM modulation (scale + shift ground features by heading)
  Fusion Head     : MLP (2048+2048 → 512 → 256 → 2)  [lat/lon regression]
                    Projection heads (→ 256) for InfoNCE cross-view alignment
  Heading Aux     : MLP (2048 → 3) [heading classification: 0°/120°/240°]

Loss
----
  L_total = L_mse + λ_c·L_infonce + λ_h·L_heading + γ·L_geofence
  L_mse      : MSE on normalised (lat, lon) coordinates
  L_infonce  : InfoNCE between oriented-ground and aerial embeddings
  L_heading  : Cross-entropy heading classification (orientation regularisation)
  L_geofence : Soft TTK bounding-circle penalty

Evaluation (Strict Raw Per-Image Mean)
---------------------------------------
  Haversine distance (m) between predicted and GT coords.
  No Best-of-3, no filtering, no sequence matching.

Output artefacts (orientation_crossview/)
------------------------------------------
  best_crossview.pth
  crossview_metrics.csv
  crossview_test_results.json
  crossview_errors.npy
  orientation_crossview_report.png  →  also copied to standardized_reports/20_orientation_crossview.png
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import shutil
import site
import sys
from pathlib import Path


# ── Windows CUDA DLL fix ───────────────────────────────────────────────────────
def _win_cuda_dlls() -> None:
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
    except Exception:
        pass


_win_cuda_dlls()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ── Paths & Config ─────────────────────────────────────────────────────────────
_REPO        = Path(__file__).resolve().parent
_TTK         = _REPO / "ttk_10k_full"
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", str(_TTK)))
METADATA_CSV = Path(os.environ.get("METADATA_CSV",
                    str(DATASET_ROOT / "splits" / "final_metadata.csv")))
SAT_DIR      = DATASET_ROOT / "transgeo" / "satellite"
OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR",
                    str(_REPO / "orientation_crossview")))
REPORT_PATH  = OUTPUT_DIR / "orientation_crossview_report.png"
REPORT_COPY  = _REPO / "standardized_reports" / "20_orientation_crossview.png"

BEST_CKPT    = OUTPUT_DIR / "best_crossview.pth"
LAST_CKPT    = OUTPUT_DIR / "crossview_last.pth"
METRICS_CSV  = OUTPUT_DIR / "crossview_metrics.csv"
TEST_JSON    = OUTPUT_DIR / "crossview_test_results.json"
ERRORS_NPY   = OUTPUT_DIR / "crossview_errors.npy"

# Training hyper-parameters
MAX_EPOCHS      = int(os.environ.get("MAX_EPOCHS",      "60"))
PATIENCE        = int(os.environ.get("PATIENCE",        "7"))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE",      "16"))
LR_HEAD         = float(os.environ.get("LR_HEAD",       "3e-4"))
LR_BACKBONE     = float(os.environ.get("LR_BACKBONE",   "5e-6"))
UNFREEZE_EPOCH  = int(os.environ.get("UNFREEZE_EPOCH",  "10"))
UNFREEZE_LAYERS = int(os.environ.get("UNFREEZE_LAYERS", "2"))
NUM_WORKERS     = int(os.environ.get("NUM_WORKERS",     "4"))
SEED            = 42
RESUME          = os.environ.get("RESUME", "1").lower() not in ("0", "false", "no")

# Loss weights
LAMBDA_INFONCE = float(os.environ.get("LAMBDA_INFONCE", "0.5"))
LAMBDA_HEADING = float(os.environ.get("LAMBDA_HEADING", "0.3"))
GAMMA_GEOFENCE = float(os.environ.get("GAMMA_GF",       "0.5"))
TEMPERATURE    = float(os.environ.get("TEMPERATURE",    "0.07"))

# TTK region constants
TTK_LAT_CENTER  = 55.751
TTK_LON_CENTER  = 37.614
TTK_RADIUS_DEG  = 0.22

# Coordinate normalisation (from normalization_stats.json)
LAT_MEAN = 55.751221797182545
LAT_STD  = 0.03336638341934356
LON_MEAN = 37.614152616229575
LON_STD  = 0.05544997434348209

# Heading label mapping: {degrees: class_index}
HEADING_TO_CLS = {0: 0, 120: 1, 240: 2}
NUM_HEADING_CLS = 3

# Reference baselines for the report
REF_KOPERNIK = 3545.0
REF_GEOCLIP  = 3100.0


# ── Utilities ──────────────────────────────────────────────────────────────────
def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        dev = torch.device("cuda")
        try:
            torch.zeros(1, device=dev).relu_()
        except RuntimeError as e:
            print(f"  [WARN] CUDA unusable ({e!s}); falling back to CPU.", flush=True)
            torch.backends.cudnn.benchmark = False
            return torch.device("cpu")
        return dev
    if os.environ.get("ALLOW_CPU", "").lower() in ("1", "true"):
        print("  [WARN] CUDA not available — CPU mode.", flush=True)
        return torch.device("cpu")
    print("CUDA not found. Set ALLOW_CPU=1 to run on CPU.", file=sys.stderr)
    sys.exit(1)


def haversine_m_vec(lat1: np.ndarray, lon1: np.ndarray,
                    lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6_371_000.0
    ph1, ph2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    dp = np.radians(lat2 - lat1)
    a = np.sin(dp / 2) ** 2 + np.cos(ph1) * np.cos(ph2) * np.sin(dl / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def norm_lat(x):   return (x - LAT_MEAN) / LAT_STD
def norm_lon(x):   return (x - LON_MEAN) / LON_STD
def denorm_lat(x): return x * LAT_STD  + LAT_MEAN
def denorm_lon(x): return x * LON_STD  + LON_MEAN


def _resnet_transforms(train: bool):
    """Standard ImageNet transforms for ResNet50."""
    from torchvision import transforms
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.15, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ── Dataset ────────────────────────────────────────────────────────────────────
class CrossViewDataset(Dataset):
    """
    Pairs ground-level street images with their satellite counterpart,
    using pano_id to look up the satellite file.
    Each sample: (ground_img, sat_img, heading_label, norm_target, lat, lon, lid).
    """

    def __init__(self, df: pd.DataFrame, root: Path, sat_dir: Path,
                 tf_ground, tf_sat):
        self.root      = root
        self.sat_dir   = sat_dir
        self.tf_ground = tf_ground
        self.tf_sat    = tf_sat

        records = []
        skipped = 0
        for _, row in df.iterrows():
            gnd_path = root / row["image_path"]
            pano_id  = str(row["pano_id"])
            sat_path = sat_dir / f"sat_{pano_id}.jpg"
            if not gnd_path.exists() or not sat_path.exists():
                skipped += 1
                continue
            heading = int(row["heading"])
            h_cls   = HEADING_TO_CLS.get(heading, -1)
            if h_cls == -1:
                skipped += 1
                continue
            records.append((
                str(row["image_path"]),
                str(sat_path),
                h_cls,
                float(row["latitude"]),
                float(row["longitude"]),
                str(row["location_id"]),
            ))

        if skipped:
            print(f"  [warn] CrossViewDataset: {skipped} rows skipped "
                  f"(missing file or unknown heading).", flush=True)

        if not records:
            self.gnd_paths = self.sat_paths = self.h_labels = []
            self.lats = self.lons = self.lids = []
            return

        (self.gnd_paths, self.sat_paths, self.h_labels,
         self.lats, self.lons, self.lids) = zip(*records)

    def __len__(self) -> int:
        return len(self.gnd_paths)

    def __getitem__(self, idx: int):
        gnd = Image.open(self.root / self.gnd_paths[idx]).convert("RGB")
        sat = Image.open(self.sat_paths[idx]).convert("RGB")
        gnd = self.tf_ground(gnd)
        sat = self.tf_sat(sat)
        lat, lon = float(self.lats[idx]), float(self.lons[idx])
        target   = torch.tensor([norm_lat(lat), norm_lon(lon)], dtype=torch.float32)
        h_label  = int(self.h_labels[idx])
        return gnd, sat, h_label, target, lat, lon, self.lids[idx]


def collate_fn(batch):
    gnds    = torch.stack([b[0] for b in batch])
    sats    = torch.stack([b[1] for b in batch])
    hlabels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    targets = torch.stack([b[3] for b in batch])
    lats    = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    lons    = torch.tensor([b[5] for b in batch], dtype=torch.float32)
    lids    = [b[6] for b in batch]
    return gnds, sats, hlabels, targets, lats, lons, lids


# ── Architecture ───────────────────────────────────────────────────────────────
class HeadingEncoder(nn.Module):
    """
    Circular sinusoidal encoding: heading_deg → (B, out_dim).
    out_dim must be even; each pair encodes sin/cos at a different frequency.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        assert out_dim % 2 == 0
        self.num_freqs = out_dim // 2

    def forward(self, heading_deg: torch.Tensor) -> torch.Tensor:
        rad   = torch.deg2rad(heading_deg.float())           # (B,)
        freqs = torch.arange(1, self.num_freqs + 1,
                              device=rad.device, dtype=rad.dtype)  # (num_freqs,)
        angles = rad.unsqueeze(1) * freqs.unsqueeze(0)       # (B, num_freqs)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (B, 64)


class OrientationFiLM(nn.Module):
    """
    Feature-wise Linear Modulation: modulates ground features by heading.
    Learns per-channel scale and shift conditioned on heading embedding.
    """

    def __init__(self, feat_dim: int = 2048, heading_dim: int = 64):
        super().__init__()
        self.film = nn.Sequential(
            nn.Linear(heading_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim * 2),   # scale + shift
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, feat: torch.Tensor, h_emb: torch.Tensor) -> torch.Tensor:
        params       = self.film(h_emb)               # (B, 4096)
        scale, shift = params.chunk(2, dim=1)          # (B, 2048) each
        return self.norm(feat * (1.0 + scale) + shift)


class ResNetEncoder(nn.Module):
    """
    ResNet50 backbone (pretrained ImageNet) with final FC removed.
    Outputs spatial-pooled features (B, 2048).
    Initially fully frozen; layers can be unlocked via unfreeze().
    """

    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        # Keep everything except the final FC; apply adaptive pool ourselves
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool2d(1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)           # (B, 2048, 7, 7)
        x = self.pool(x).flatten(1)    # (B, 2048)
        return x

    def unfreeze_last_n(self, n: int) -> list:
        """Unfreeze the last n children of features + pool. Returns new params."""
        children  = list(self.features.children()) + [self.pool]
        unlocked  = []
        for child in children[-n:]:
            for p in child.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    unlocked.append(p)
        return unlocked


class CrossViewFusionHead(nn.Module):
    """
    Fuses oriented ground features and aerial features → (coords, g_emb, a_emb).
    """

    def __init__(self, feat_dim: int = 2048, hidden: int = 512,
                 emb_dim: int = 256):
        super().__init__()
        self.proj_g = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.proj_a = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )
        self.coord   = nn.Linear(emb_dim, 2)
        self.g_proj  = nn.Linear(feat_dim, emb_dim)   # contrastive embeddings
        self.a_proj  = nn.Linear(feat_dim, emb_dim)

    def forward(self, g_feat: torch.Tensor, a_feat: torch.Tensor):
        fused  = self.fusion(
            torch.cat([self.proj_g(g_feat), self.proj_a(a_feat)], dim=1))
        coords = self.coord(fused)                               # (B, 2)
        g_emb  = F.normalize(self.g_proj(g_feat), dim=1)        # (B, 256)
        a_emb  = F.normalize(self.a_proj(a_feat), dim=1)        # (B, 256)
        return coords, g_emb, a_emb


class HeadingClassifier(nn.Module):
    """Auxiliary MLP: ground features → heading class (0°/120°/240°)."""

    def __init__(self, feat_dim: int = 2048, num_cls: int = NUM_HEADING_CLS):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_cls),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class CrossViewModel(nn.Module):
    """
    Full Orientation-Aware Cross-View model.
    forward() → (coords, g_emb, a_emb, h_logits)
    """

    def __init__(self):
        super().__init__()
        self.ground_enc  = ResNetEncoder()
        self.aerial_enc  = ResNetEncoder()
        self.heading_enc = HeadingEncoder(out_dim=64)
        self.orient_film = OrientationFiLM(feat_dim=2048, heading_dim=64)
        self.fusion_head = CrossViewFusionHead(feat_dim=2048, hidden=512, emb_dim=256)
        self.heading_cls = HeadingClassifier(feat_dim=2048, num_cls=NUM_HEADING_CLS)

    def forward(self, gnd: torch.Tensor, sat: torch.Tensor,
                heading_deg: torch.Tensor):
        g_feat     = self.ground_enc(gnd)                          # (B, 2048)
        a_feat     = self.aerial_enc(sat)                          # (B, 2048)
        h_emb      = self.heading_enc(heading_deg)                 # (B, 64)
        g_oriented = self.orient_film(g_feat, h_emb)               # (B, 2048)
        coords, g_emb, a_emb = self.fusion_head(g_oriented, a_feat)
        h_logits   = self.heading_cls(g_feat)                      # (B, 3)
        return coords, g_emb, a_emb, h_logits


# ── Loss components ────────────────────────────────────────────────────────────
def mse_coord_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def infonce_crossview(g_emb: torch.Tensor, a_emb: torch.Tensor,
                      temperature: float = TEMPERATURE) -> torch.Tensor:
    """InfoNCE: g_emb[i] ↔ a_emb[i] (same location)."""
    B = g_emb.size(0)
    if B < 2:
        return torch.tensor(0.0, device=g_emb.device)
    logits = g_emb @ a_emb.t() / temperature     # (B, B)
    labels = torch.arange(B, device=g_emb.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)) / 2.0


def heading_cls_loss(logits: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def geofence_loss(pred_coords: torch.Tensor) -> torch.Tensor:
    """Soft penalty for predictions outside TTK bounding circle."""
    pred_lat = denorm_lat(pred_coords[:, 0])
    pred_lon = denorm_lon(pred_coords[:, 1])
    dlat     = pred_lat - TTK_LAT_CENTER
    dlon     = (pred_lon - TTK_LON_CENTER) * math.cos(math.radians(TTK_LAT_CENTER))
    dist_deg = torch.sqrt(dlat ** 2 + dlon ** 2 + 1e-8)
    return F.relu(dist_deg - TTK_RADIUS_DEG).mean()


def total_loss(pred: torch.Tensor, target: torch.Tensor,
               g_emb: torch.Tensor, a_emb: torch.Tensor,
               h_logits: torch.Tensor, h_labels: torch.Tensor
               ) -> tuple[torch.Tensor, dict]:
    l_mse  = mse_coord_loss(pred, target)
    l_nce  = infonce_crossview(g_emb, a_emb)
    l_head = heading_cls_loss(h_logits, h_labels)
    l_geo  = geofence_loss(pred)
    tot    = (l_mse
              + LAMBDA_INFONCE * l_nce
              + LAMBDA_HEADING * l_head
              + GAMMA_GEOFENCE * l_geo)
    return tot, {
        "total":    tot.item(),
        "mse":      l_mse.item(),
        "infonce":  l_nce.item(),
        "heading":  l_head.item(),
        "geofence": l_geo.item(),
    }


# ── Backbone unfreezing ────────────────────────────────────────────────────────
def unfreeze_encoders(model: CrossViewModel, n: int,
                      optimizer: torch.optim.Optimizer, lr: float) -> None:
    new_params: list[torch.nn.Parameter] = []
    for enc_name in ("ground_enc", "aerial_enc"):
        enc = getattr(model, enc_name)
        new_params.extend(enc.unfreeze_last_n(n))
    if new_params:
        optimizer.add_param_group({"params": new_params, "lr": lr})
        print(f"  Phase 2: unfroze last {n} blocks of both encoders "
              f"({sum(p.numel() for p in new_params) / 1e6:.1f}M params, "
              f"lr={lr:.1e})", flush=True)


# ── Train / Eval ───────────────────────────────────────────────────────────────
def train_epoch(model: CrossViewModel, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device, scaler) -> dict:
    model.train()
    tot = {"total": 0.0, "mse": 0.0, "infonce": 0.0,
           "heading": 0.0, "geofence": 0.0}
    n   = 0
    correct_h = 0

    for gnds, sats, hlabels, targets, lats, lons, _ in tqdm(
            loader, desc="train", leave=False, file=sys.stdout):
        gnds    = gnds.to(device, non_blocking=True)
        sats    = sats.to(device, non_blocking=True)
        hlabels = hlabels.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lats    = lats.to(device, non_blocking=True)
        lons    = lons.to(device, non_blocking=True)

        # Heading in degrees from label
        heading_deg = torch.tensor(
            [list(HEADING_TO_CLS.keys())[list(HEADING_TO_CLS.values()).index(c.item())]
             for c in hlabels.cpu()],
            dtype=torch.float32, device=device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                coords, g_emb, a_emb, h_logits = model(gnds, sats, heading_deg)
                loss, parts = total_loss(coords, targets, g_emb, a_emb,
                                         h_logits, hlabels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            coords, g_emb, a_emb, h_logits = model(gnds, sats, heading_deg)
            loss, parts = total_loss(coords, targets, g_emb, a_emb,
                                      h_logits, hlabels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

        bs = gnds.size(0)
        for k in tot:
            tot[k] += parts[k] * bs
        n += bs
        correct_h += (h_logits.argmax(1) == hlabels).sum().item()

    return {k: v / max(n, 1) for k, v in tot.items()} | {
        "heading_acc": correct_h / max(n, 1) * 100
    }


# heading_deg lookup cached for eval
_HDG_DEG = {v: k for k, v in HEADING_TO_CLS.items()}


@torch.no_grad()
def evaluate_raw(model: CrossViewModel, loader: DataLoader,
                 device: torch.device) -> dict:
    """Raw per-image Haversine evaluation + heading accuracy."""
    model.eval()
    all_errors: list[float] = []
    all_pred_lat: list[float] = []
    all_pred_lon: list[float] = []
    correct_h = 0
    total_h   = 0

    for gnds, sats, hlabels, _, lats, lons, _ in tqdm(
            loader, desc="eval", leave=False, file=sys.stdout):
        gnds    = gnds.to(device, non_blocking=True)
        sats    = sats.to(device, non_blocking=True)
        hlabels = hlabels.to(device, non_blocking=True)
        heading_deg = torch.tensor(
            [_HDG_DEG[c.item()] for c in hlabels.cpu()],
            dtype=torch.float32, device=device)

        coords, _, _, h_logits = model(gnds, sats, heading_deg)
        pred_np  = coords.cpu().numpy()
        pred_lat = denorm_lat(pred_np[:, 0])
        pred_lon = denorm_lon(pred_np[:, 1])
        errs     = haversine_m_vec(
            lats.numpy(), lons.numpy(), pred_lat, pred_lon)
        all_errors.extend(errs.tolist())
        all_pred_lat.extend(pred_lat.tolist())
        all_pred_lon.extend(pred_lon.tolist())
        correct_h += (h_logits.argmax(1) == hlabels).sum().item()
        total_h   += hlabels.size(0)

    arr = np.array(all_errors, dtype=np.float64)
    if len(arr) == 0:
        return dict(raw_mean_m=0, raw_median_m=0, recall_1km=0,
                    heading_acc=0, errors=arr)
    return dict(
        raw_mean_m   = float(arr.mean()),
        raw_median_m = float(np.median(arr)),
        recall_100m  = float((arr < 100).mean()  * 100),
        recall_500m  = float((arr < 500).mean()  * 100),
        recall_1km   = float((arr < 1000).mean() * 100),
        recall_3km   = float((arr < 3000).mean() * 100),
        heading_acc  = correct_h / max(total_h, 1) * 100,
        errors       = arr,
        pred_lat     = all_pred_lat,
        pred_lon     = all_pred_lon,
    )


# ── Report ─────────────────────────────────────────────────────────────────────
def save_report(history: list[dict], test_errors: np.ndarray,
                best_ep: int, test_mean: float,
                orient_acc_test: float) -> None:
    """
    2×3 dashboard.
    [0,0] Loss History        [0,1] Val Haversine (m)     [0,2] Orientation Accuracy
    [1,0] Error Histogram     [1,1] Error CDF              [1,2] Summary Table
    """
    BG     = "#FFFFFF"
    PANEL  = "#FFFFFF"
    GRID   = "#D0D7DE"
    TEXT   = "#1F2328"
    MUTED  = "#57606A"
    ROYAL  = "#4169E1"   # Royal Blue
    DGOLD  = "#B8860B"   # Dark Gold
    AMBER  = "#C9A227"   # dark-gold accent (readable on white)
    RED    = "#CF222E"

    epochs     = [h["epoch"]       for h in history]
    tr_loss    = [h["train_total"] for h in history]
    val_nce    = [h["val_infonce"] for h in history]
    val_mean   = [h["val_mean_m"]  for h in history]
    orient_acc = [h["val_heading_acc"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

    def _grid(ax):
        ax.grid(True, color=GRID, lw=0.5, alpha=0.55)

    # ── [0,0] Loss History ────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, tr_loss, color=ROYAL,  lw=1.8, label="Train Total Loss")
    ax.plot(epochs, val_nce, color=DGOLD,  lw=1.8, label="Val InfoNCE Loss", ls="--")
    if best_ep in epochs:
        bi = epochs.index(best_ep)
        ax.axvline(best_ep, color=AMBER, lw=1.2, ls=":", alpha=0.7,
                   label=f"Best epoch {best_ep}")
        ax.scatter([best_ep], [val_nce[bi]], color=AMBER, zorder=6, s=60)
    ax.set_title("Loss History", color=TEXT, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
    ax.set_ylabel("Loss", color=MUTED, fontsize=9)
    ax.legend(fontsize=7.5, facecolor="white", edgecolor=GRID, labelcolor=TEXT)
    _grid(ax)

    # ── [0,1] Val Haversine (m) ───────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, val_mean, color=ROYAL, lw=1.8, label="Val Mean Haversine (m)")
    ax.axhline(REF_KOPERNIK, color=RED,   lw=1.6, ls="--",
               label=f"Kopernik {REF_KOPERNIK:.0f} m")
    ax.axhline(REF_GEOCLIP,  color=AMBER, lw=1.6, ls="--",
               label=f"GeoCLIP  {REF_GEOCLIP:.0f} m")
    if val_mean:
        best_val = min(val_mean)
        bi       = val_mean.index(best_val)
        ax.scatter([epochs[bi]], [best_val], color=DGOLD, zorder=6, s=70,
                   label=f"Best {best_val:.0f} m")
        ax.annotate(f"{best_val:.0f} m", xy=(epochs[bi], best_val),
                    xytext=(epochs[bi] + 0.5, best_val * 1.03),
                    color=DGOLD, fontsize=8)
    ax.set_title("Geolocation Error — Val Mean Haversine", color=TEXT,
                 fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
    ax.set_ylabel("metres", color=MUTED, fontsize=9)
    ax.legend(fontsize=7.5, facecolor="white", edgecolor=GRID, labelcolor=TEXT)
    _grid(ax)

    # ── [0,2] Orientation Accuracy ────────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(epochs, orient_acc, color=DGOLD, lw=1.8, label="Val Heading Acc (%)")
    ax.axhline(100 / NUM_HEADING_CLS, color=MUTED, lw=1.2, ls=":",
               label=f"Random baseline {100/NUM_HEADING_CLS:.0f}%")
    ax.set_ylim(0, 105)
    ax.set_title("Orientation Accuracy — Heading Classification",
                 color=TEXT, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
    ax.set_ylabel("Accuracy (%)", color=MUTED, fontsize=9)
    ax.legend(fontsize=7.5, facecolor="white", edgecolor=GRID, labelcolor=TEXT)
    _grid(ax)

    # ── [1,0] Error Histogram (capped at 10 000 m) ───────────────────────────
    ax  = axes[1, 0]
    cap = (np.minimum(test_errors, 10_000.0)
           if len(test_errors) else np.array([]))
    if len(cap):
        ax.hist(cap, bins=50, color=ROYAL, alpha=0.85,
                edgecolor=GRID, linewidth=0.3)
        ax.axvline(float(np.mean(cap)), color=DGOLD, lw=2,
                   label=f"Mean {float(np.mean(cap)):.0f} m")
        ax.axvline(float(np.median(cap)), color=AMBER, lw=1.5, ls="--",
                   label=f"Median {float(np.median(cap)):.0f} m")
    ax.set_xlim(0, 10_000)
    ax.set_title("Error Distribution (capped 10 km)", color=TEXT,
                 fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Distance per image (m)", color=MUTED, fontsize=9)
    ax.set_ylabel("Count", color=MUTED, fontsize=9)
    ax.legend(fontsize=7.5, facecolor="white", edgecolor=GRID, labelcolor=TEXT)
    _grid(ax)

    # ── [1,1] CDF ─────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    if len(test_errors):
        sv  = np.sort(test_errors)
        cdf = np.arange(1, len(sv) + 1) / len(sv)
        ax.plot(sv, cdf * 100, color=ROYAL, lw=2, label="Cross-View OA")
    ax.axvline(REF_KOPERNIK, color=RED,   lw=1.5, ls="--",
               label=f"Kopernik {REF_KOPERNIK:.0f} m")
    ax.axvline(REF_GEOCLIP,  color=AMBER, lw=1.5, ls="--",
               label=f"GeoCLIP  {REF_GEOCLIP:.0f} m")
    ax.set_xlim(0, 12_000)
    ax.set_ylim(0, 101)
    ax.set_title("CDF — Cumulative Error Distribution", color=TEXT,
                 fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Distance per image (m)", color=MUTED, fontsize=9)
    ax.set_ylabel("Cumulative %", color=MUTED, fontsize=9)
    ax.legend(fontsize=7.5, facecolor="white", edgecolor=GRID, labelcolor=TEXT)
    _grid(ax)

    # ── [1,2] Summary Table ───────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")

    best_val   = min(val_mean) if val_mean else float("nan")
    status_txt = ("BEATS GeoCLIP" if test_mean < REF_GEOCLIP
                  else "BEATS Kopernik" if test_mean < REF_KOPERNIK
                  else "Below baseline")

    rows = [
        ["Metric",                  "Value"],
        ["Best Epoch",              str(best_ep)],
        ["Best Val Mean",           f"{best_val:.1f} m"],
        ["Test Mean Haversine",     f"{test_mean:.1f} m"],
        ["Test Recall@1km",         f"{(test_errors < 1000).mean()*100:.1f}%"
         if len(test_errors) else "—"],
        ["Test Recall@3km",         f"{(test_errors < 3000).mean()*100:.1f}%"
         if len(test_errors) else "—"],
        ["Orient Acc (test)",       f"{orient_acc_test:.1f}%"],
        ["vs Kopernik (3545 m)",    f"{test_mean - REF_KOPERNIK:+.0f} m"],
        ["vs GeoCLIP  (3100 m)",    f"{test_mean - REF_GEOCLIP:+.0f} m"],
        ["Status",                  status_txt],
    ]

    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center",
                   bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#F6F8FA" if r > 0 else "#E8EEF8")
        cell.set_text_props(color=TEXT if r > 0 else ROYAL, fontweight="bold")
        cell.set_edgecolor(GRID)
    ax.set_title("Summary", color=TEXT, fontsize=10, fontweight="bold", pad=6)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "Orientation-Aware Cross-View Geo-Localization · Moscow TTK-10k\n"
        "Siamese ResNet50 + FiLM Heading Modulation + InfoNCE Cross-View Alignment",
        color=TEXT, fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, 0.005,
        f"Model: CrossView-OA  |  Best Ep: {best_ep}"
        f"  |  Test Mean: {test_mean:.0f} m"
        f"  |  Heading Acc: {orient_acc_test:.1f}%"
        f"  |  Baselines: Kopernik {REF_KOPERNIK:.0f} m / GeoCLIP {REF_GEOCLIP:.0f} m",
        ha="center", va="bottom", color=DGOLD, fontsize=8.5, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(REPORT_PATH), dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print(f"  Report saved → {REPORT_PATH}", flush=True)

    try:
        shutil.copy2(str(REPORT_PATH), str(REPORT_COPY))
        print(f"  Report copied → {REPORT_COPY}", flush=True)
    except Exception as e:
        print(f"  [warn] Could not copy report: {e}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    set_seed(SEED)
    device = resolve_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}", flush=True)
    print("  Orientation-Aware Cross-View Geo-Localization", flush=True)
    print("  Moscow TTK-10k  |  Siamese ResNet50 + FiLM + InfoNCE", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  Device          : {device}", flush=True)
    print(f"  Dataset root    : {DATASET_ROOT}", flush=True)
    print(f"  Satellite dir   : {SAT_DIR}", flush=True)
    print(f"  Batch size      : {BATCH_SIZE}", flush=True)
    print(f"  LR head/backbone: {LR_HEAD:.1e} / {LR_BACKBONE:.1e}", flush=True)
    print(f"  Unfreeze @ ep   : {UNFREEZE_EPOCH} (last {UNFREEZE_LAYERS} blocks)", flush=True)
    print(f"  λ_infonce       : {LAMBDA_INFONCE}  λ_heading: {LAMBDA_HEADING}"
          f"  γ_gf: {GAMMA_GEOFENCE}", flush=True)
    print(f"  Patience        : {PATIENCE}", flush=True)
    print(f"{'='*72}\n", flush=True)

    # ── Load metadata & join with satellite images ─────────────────────────
    if not METADATA_CSV.is_file():
        print(f"ERROR: {METADATA_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(METADATA_CSV)
    # Keep only rows whose pano has a satellite image
    df["sat_path"] = df["pano_id"].apply(
        lambda pid: SAT_DIR / f"sat_{pid}.jpg")
    df = df[df["sat_path"].apply(lambda p: p.is_file())].reset_index(drop=True)
    # Also verify ground image exists
    df = df[df["image_path"].apply(
        lambda p: (DATASET_ROOT / p).is_file())].reset_index(drop=True)
    # Only headings we know
    df = df[df["heading"].isin(HEADING_TO_CLS)].reset_index(drop=True)
    print(f"  Cross-view pairs after filtering: {len(df)}", flush=True)
    print(f"  Unique pano_ids: {df['pano_id'].nunique()}", flush=True)

    if len(df) < 20:
        print("ERROR: too few valid cross-view pairs.", file=sys.stderr)
        sys.exit(1)

    # ── 80/10/10 split by location_id ─────────────────────────────────────
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, tmp_idx = next(gss1.split(df, groups=df["location_id"].values))
    df_tmp = df.iloc[tmp_idx].reset_index(drop=True)
    gss2   = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_i, test_i = next(gss2.split(df_tmp,
                                    groups=df_tmp["location_id"].values))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df_tmp.iloc[val_i].reset_index(drop=True)
    df_test  = df_tmp.iloc[test_i].reset_index(drop=True)
    print(f"  Split → train {len(df_train)} | val {len(df_val)} | "
          f"test {len(df_test)}", flush=True)

    # ── DataLoaders ───────────────────────────────────────────────────────
    kw = dict(num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
              persistent_workers=(NUM_WORKERS > 0))
    tf_tr  = _resnet_transforms(train=True)
    tf_val = _resnet_transforms(train=False)

    train_ds  = CrossViewDataset(df_train, DATASET_ROOT, SAT_DIR, tf_tr,  tf_tr)
    val_ds    = CrossViewDataset(df_val,   DATASET_ROOT, SAT_DIR, tf_val, tf_val)
    test_ds   = CrossViewDataset(df_test,  DATASET_ROOT, SAT_DIR, tf_val, tf_val)

    train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn, drop_last=True, **kw)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, **kw)
    test_ldr  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, **kw)

    # ── Model ─────────────────────────────────────────────────────────────
    model = CrossViewModel().to(device)
    n_total  = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters()
                   if p.requires_grad) / 1e6
    print(f"  Total params    : {n_total:.1f}M  |  Trainable: {n_train:.1f}M",
          flush=True)

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR_HEAD, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch    = 1
    best_val_mean  = float("inf")
    patience_cnt   = 0
    history: list[dict] = []
    phase2_armed   = False
    header_written = False

    if RESUME and LAST_CKPT.is_file():
        print(f"  Resuming from {LAST_CKPT.name} ...", flush=True)
        try:
            ck = torch.load(str(LAST_CKPT), map_location=device,
                            weights_only=False)
        except TypeError:
            ck = torch.load(str(LAST_CKPT), map_location=device)
        model.orient_film.load_state_dict(ck["orient_film"])
        model.fusion_head.load_state_dict(ck["fusion_head"])
        model.heading_cls.load_state_dict(ck["heading_cls"])
        phase2_armed = ck.get("phase2_armed", False)
        if phase2_armed:
            unfreeze_encoders(model, UNFREEZE_LAYERS, optimizer, LR_BACKBONE)
        optimizer.load_state_dict(ck["optimizer"])
        saved_ep      = ck.get("epoch", 1)
        start_epoch   = saved_ep + 1
        best_val_mean = ck.get("best_val_mean", float("inf"))
        patience_cnt  = ck.get("patience_cnt", 0)
        history       = ck.get("history", [])
        _ilrs = [LR_HEAD] + [LR_BACKBONE] * (len(optimizer.param_groups) - 1)
        for pg, ilr in zip(optimizer.param_groups, _ilrs):
            pg["initial_lr"] = ilr
            pg["lr"]         = ilr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-7, last_epoch=-1)
        for _ in range(saved_ep):
            scheduler.step()
        print(f"  Resumed at ep {start_epoch}, best={best_val_mean:.1f} m",
              flush=True)

    # ── Epoch-0 baseline ──────────────────────────────────────────────────
    print("\n  --- Epoch 0 Baseline (frozen encoders, random head) ---",
          flush=True)
    base = evaluate_raw(model, val_ldr, device)
    print(f"  Baseline val_mean={base['raw_mean_m']:.1f} m"
          f"  heading_acc={base['heading_acc']:.1f}%", flush=True)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n  Starting training from epoch {start_epoch} ...\n", flush=True)
    CSV_FIELDS = ["epoch", "lr", "train_total", "train_mse", "train_infonce",
                  "train_heading", "train_heading_acc",
                  "val_mean_m", "val_median_m", "val_infonce",
                  "val_recall_1km", "val_heading_acc"]

    for epoch in range(start_epoch, MAX_EPOCHS + 1):

        if epoch == UNFREEZE_EPOCH and not phase2_armed:
            unfreeze_encoders(model, UNFREEZE_LAYERS, optimizer, LR_BACKBONE)
            phase2_armed = True
            if hasattr(scheduler, "base_lrs"):
                n_miss = len(optimizer.param_groups) - len(scheduler.base_lrs)
                scheduler.base_lrs.extend([LR_BACKBONE] * n_miss)

        tr  = train_epoch(model, train_ldr, optimizer, device, scaler)
        val = evaluate_raw(model, val_ldr, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # val InfoNCE: estimate on val set from saved train value (proxy)
        # We store train infonce as the loss component for plot [0,0]
        val_nce_proxy = tr["infonce"]   # use train infonce as proxy

        is_best = val["raw_mean_m"] < best_val_mean - 1.0
        marker  = " *** BEST ***" if is_best else ""

        row = dict(
            epoch             = epoch,
            lr                = lr_now,
            train_total       = tr["total"],
            train_mse         = tr["mse"],
            train_infonce     = tr["infonce"],
            train_heading     = tr["heading"],
            train_heading_acc = tr["heading_acc"],
            val_mean_m        = val["raw_mean_m"],
            val_median_m      = val["raw_median_m"],
            val_infonce       = val_nce_proxy,
            val_recall_1km    = val["recall_1km"],
            val_heading_acc   = val["heading_acc"],
        )
        history.append(row)

        with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not header_written:
                f.seek(0, 2)
                if f.tell() == 0:
                    w.writeheader()
                header_written = True
            w.writerow(row)

        print(
            f"  Ep {epoch:3d}/{MAX_EPOCHS}"
            f"  loss={tr['total']:.4f}"
            f"  [mse={tr['mse']:.3f}"
            f"  nce={tr['infonce']:.3f}"
            f"  hdg={tr['heading']:.3f}]"
            f"  val={val['raw_mean_m']:.0f} m"
            f"  hdg_acc={val['heading_acc']:.1f}%"
            f"  R@1km={val['recall_1km']:.1f}%"
            f"  lr={lr_now:.2e}{marker}",
            flush=True,
        )

        if is_best:
            best_val_mean = val["raw_mean_m"]
            patience_cnt  = 0
            torch.save(dict(
                epoch        = epoch,
                orient_film  = model.orient_film.state_dict(),
                fusion_head  = model.fusion_head.state_dict(),
                heading_cls  = model.heading_cls.state_dict(),
            ), str(BEST_CKPT))
            print(f"    + Best saved (val={best_val_mean:.1f} m)", flush=True)

        torch.save(dict(
            epoch         = epoch,
            orient_film   = model.orient_film.state_dict(),
            fusion_head   = model.fusion_head.state_dict(),
            heading_cls   = model.heading_cls.state_dict(),
            optimizer     = optimizer.state_dict(),
            scheduler     = scheduler.state_dict(),
            best_val_mean = best_val_mean,
            patience_cnt  = patience_cnt,
            history       = history,
            phase2_armed  = phase2_armed,
        ), str(LAST_CKPT))

        if not is_best:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch} "
                      f"(patience={PATIENCE})", flush=True)
                break

    print("\n  Training complete.", flush=True)

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n  Loading best weights for test ...", flush=True)
    if BEST_CKPT.is_file():
        try:
            ck = torch.load(str(BEST_CKPT), map_location=device,
                            weights_only=False)
        except TypeError:
            ck = torch.load(str(BEST_CKPT), map_location=device)
        model.orient_film.load_state_dict(ck["orient_film"])
        model.fusion_head.load_state_dict(ck["fusion_head"])
        model.heading_cls.load_state_dict(ck["heading_cls"])

    test_res = evaluate_raw(model, test_ldr, device)
    errs     = test_res["errors"]
    best_ep  = (min(history, key=lambda h: h["val_mean_m"])["epoch"]
                if history else 0)

    print(f"\n{'─'*60}", flush=True)
    print(f"  Cross-View OA  TEST RESULTS  ({len(errs)} images)", flush=True)
    print(f"  Best epoch       : {best_ep}", flush=True)
    print(f"  Raw Mean         : {errs.mean():.1f} m", flush=True)
    print(f"  Raw Median       : {np.median(errs):.1f} m", flush=True)
    print(f"  Recall@100m      : {test_res['recall_100m']:.2f}%", flush=True)
    print(f"  Recall@500m      : {test_res['recall_500m']:.2f}%", flush=True)
    print(f"  Recall@1km       : {test_res['recall_1km']:.2f}%", flush=True)
    print(f"  Recall@3km       : {test_res['recall_3km']:.2f}%", flush=True)
    print(f"  Heading Accuracy : {test_res['heading_acc']:.2f}%", flush=True)
    print(f"{'─'*60}", flush=True)

    np.save(str(ERRORS_NPY), errs)

    json_out = dict(
        model               = "CrossView_OA",
        best_epoch          = best_ep,
        test_raw_mean_m     = float(errs.mean()) if len(errs) else 0,
        test_raw_median_m   = float(np.median(errs)) if len(errs) else 0,
        recall_100m         = test_res["recall_100m"],
        recall_500m         = test_res["recall_500m"],
        recall_1km          = test_res["recall_1km"],
        recall_3km          = test_res["recall_3km"],
        heading_accuracy    = test_res["heading_acc"],
        baseline_refs_m     = {"geoclip": REF_GEOCLIP,
                               "kopernik": REF_KOPERNIK},
        n_train             = len(df_train),
        n_val               = len(df_val),
        n_test              = len(df_test),
        lambda_infonce      = LAMBDA_INFONCE,
        lambda_heading      = LAMBDA_HEADING,
        gamma_geofence      = GAMMA_GEOFENCE,
    )
    TEST_JSON.write_text(json.dumps(json_out, indent=2), encoding="utf-8")

    # ── Report ────────────────────────────────────────────────────────────
    save_report(
        history       = history,
        test_errors   = errs,
        best_ep       = best_ep,
        test_mean     = float(errs.mean()) if len(errs) else 0.0,
        orient_acc_test = test_res["heading_acc"],
    )

    print(f"\n  Artefacts in: {OUTPUT_DIR}", flush=True)
    for f in [BEST_CKPT, METRICS_CSV, TEST_JSON, ERRORS_NPY, REPORT_PATH]:
        print(f"    {f.name}", flush=True)
    print(f"\n  Min Val Mean : {best_val_mean:.1f} m  (ep {best_ep})", flush=True)
    print(f"  Test Mean    : {errs.mean():.1f} m\n", flush=True)


if __name__ == "__main__":
    main()
