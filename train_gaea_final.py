"""
GAEA — Geospatial Adaptive Estimation Architecture
Moscow TTK-10k  |  Raw Per-Image Mean Protocol

Architecture
------------
  Backbone  : CLIP ViT-L/14 (frozen)
  Head      : Residual MLP  768 -> 512 -> 256 -> 2  (lat/lon direct regression)
  Attn Gate : Lightweight channel attention on ViT patch tokens to suppress
              road/sky tokens (inspired by SegVLAD failure analysis)

Loss
----
  L_total = L_MSE + lambda_c * L_contrastive + gamma * L_geofence

  L_MSE         : Normalised-coordinate MSE  (fast, differentiable)
  L_contrastive : InfoNCE across in-batch (image, geo) pairs — feature alignment
  L_geofence    : Soft penalty for predictions outside TTK buffer zone

Evaluation
----------
  Raw Per-Image Mean Haversine.  No Best-of-3, no sequence matching.

Output artefacts (in gaea_hybrid/)
------------------------------------
  best_gaea_final.pth
  gaea_final_metrics.csv
  gaea_final_test_results.json
  gaea_final_errors.npy          <- per-image errors for report
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import site
import sys
from pathlib import Path


# ── Windows CUDA DLL fix ───────────────────────────────────────────────────
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ── Paths & Config ────────────────────────────────────────────────────────────
_REPO        = Path(__file__).resolve().parent
_TTK         = _REPO / "ttk_10k_full"
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", str(_TTK)))
METADATA_CSV = Path(os.environ.get("METADATA_CSV",  str(DATASET_ROOT / "splits" / "final_metadata.csv")))
OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR",    str(_REPO / "gaea_hybrid")))

BEST_CKPT    = OUTPUT_DIR / "best_gaea_final.pth"
LAST_CKPT    = OUTPUT_DIR / "gaea_final_last.pth"
METRICS_CSV  = OUTPUT_DIR / "gaea_final_metrics.csv"
TEST_JSON    = OUTPUT_DIR / "gaea_final_test_results.json"
ERRORS_NPY   = OUTPUT_DIR / "gaea_final_errors.npy"

# Training hyper-parameters
MAX_EPOCHS      = int(os.environ.get("MAX_EPOCHS",      "60"))
PATIENCE        = int(os.environ.get("PATIENCE",        "8"))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE",      "32"))
LR_HEAD         = float(os.environ.get("LR_HEAD",       "3e-4"))
LR_BACKBONE     = float(os.environ.get("LR_BACKBONE",   "5e-6"))
UNFREEZE_EPOCH  = int(os.environ.get("UNFREEZE_EPOCH",  "10"))
UNFREEZE_LAYERS = int(os.environ.get("UNFREEZE_LAYERS", "4"))
NUM_WORKERS     = int(os.environ.get("NUM_WORKERS",     "4"))
SEED            = 42
RESUME          = os.environ.get("RESUME", "1").lower() not in ("0", "false", "no")

# Loss weights
LAMBDA_CONTRASTIVE = float(os.environ.get("LAMBDA_C",   "0.3"))
GAMMA_GEOFENCE     = float(os.environ.get("GAMMA_GF",   "0.5"))

# TTK geofence (TTK ring road bounding polygon, slightly buffered)
TTK_LAT_MIN, TTK_LAT_MAX = 55.56, 55.92
TTK_LON_MIN, TTK_LON_MAX = 37.27, 37.93
TTK_LAT_CENTER = 55.751
TTK_LON_CENTER = 37.614
TTK_RADIUS_DEG  = 0.22   # ~24 km radius — generous TTK buffer

# Coordinate normalisation (from normalization_stats.json)
LAT_MEAN = 55.751221797182545
LAT_STD  = 0.03336638341934356
LON_MEAN = 37.614152616229575
LON_STD  = 0.05544997434348209


# ── Utilities ─────────────────────────────────────────────────────────────────
def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if os.environ.get("ALLOW_CPU", "").lower() in ("1", "true"):
        print("  [WARN] CUDA not available — CPU mode.", flush=True)
        return torch.device("cpu")
    print("CUDA not found. Set ALLOW_CPU=1 to proceed on CPU.", file=sys.stderr)
    sys.exit(1)


def haversine_m_vec(lat1: np.ndarray, lon1: np.ndarray,
                    lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6_371_000.0
    ph1, ph2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    dp = np.radians(lat2 - lat1)
    a = np.sin(dp / 2) ** 2 + np.cos(ph1) * np.cos(ph2) * np.sin(dl / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def norm_lat(x):  return (x - LAT_MEAN) / LAT_STD
def norm_lon(x):  return (x - LON_MEAN) / LON_STD
def denorm_lat(x): return x * LAT_STD + LAT_MEAN
def denorm_lon(x): return x * LON_STD  + LON_MEAN


# ── Dataset ───────────────────────────────────────────────────────────────────
def _clip_transforms(train: bool):
    """Standard CLIP ViT-L/14 input transforms."""
    from torchvision import transforms
    MEAN = (0.48145466, 0.4578275,  0.40821073)
    STD  = (0.26862954, 0.26130258, 0.27577711)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class TTKDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform):
        self.root = root
        self.transform = transform
        rows = [
            (row["image_path"], float(row["latitude"]), float(row["longitude"]),
             str(row["location_id"]))
            for _, row in df.iterrows()
            if (root / row["image_path"]).exists()
        ]
        if len(rows) < len(df):
            print(f"  [warn] {len(df)-len(rows)} missing images skipped.", flush=True)
        self.paths, self.lats, self.lons, self.lids = (
            zip(*rows) if rows else ([], [], [], [])
        )

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.root / self.paths[idx]).convert("RGB")
        img = self.transform(img)
        lat, lon = float(self.lats[idx]), float(self.lons[idx])
        target = torch.tensor([norm_lat(lat), norm_lon(lon)], dtype=torch.float32)
        return img, target, lat, lon, self.lids[idx]


def collate_fn(batch):
    imgs    = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    lats    = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    lons    = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    lids    = [b[4] for b in batch]
    return imgs, targets, lats, lons, lids


# ── Architecture ──────────────────────────────────────────────────────────────
class ChannelAttentionGate(nn.Module):
    """
    Lightweight squeeze-excitation gate applied on ViT patch tokens.
    Suppresses uninformative channels (road/sky homogeneity).
    Input : (B, 768) — CLS token from ViT-L/14
    Output: (B, 768) — gated features
    """
    def __init__(self, dim: int = 768, ratio: int = 16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // ratio, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class GAEAHead(nn.Module):
    """
    Residual MLP for direct coordinate regression.
    Input  : (B, 768)
    Output : (B, 2)  [norm_lat, norm_lon]
    Also exposes (B, 256) feature embedding for contrastive loss.
    """
    def __init__(self, in_dim: int = 768, hidden: int = 512, emb_dim: int = 256):
        super().__init__()
        self.proj  = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.res1  = ResidualBlock(hidden, dropout=0.2)
        self.down  = nn.Sequential(nn.Linear(hidden, emb_dim), nn.LayerNorm(emb_dim), nn.GELU())
        self.res2  = ResidualBlock(emb_dim, dropout=0.1)
        self.coord = nn.Linear(emb_dim, 2)

    def forward(self, x: torch.Tensor):
        h = self.res1(self.proj(x))
        e = self.res2(self.down(h))
        return self.coord(e), e     # (B,2), (B, emb_dim)


class GAEAModel(nn.Module):
    """
    GAEA: CLIP ViT-L/14 (frozen) + ChannelAttentionGate + Residual-MLP head.
    """
    def __init__(self):
        super().__init__()
        from transformers import CLIPVisionModel, CLIPImageProcessor
        print("  Loading CLIP ViT-L/14 ...", flush=True)
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        # Freeze all CLIP params initially
        for p in self.clip.parameters():
            p.requires_grad = False
        self.attn_gate = ChannelAttentionGate(dim=1024, ratio=16)
        self.head      = GAEAHead(in_dim=1024, hidden=512, emb_dim=256)
        self._vit_dim  = 1024

    def visual_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """CLS token from ViT-L/14, shape (B, 1024)."""
        out = self.clip(pixel_values=pixel_values)
        return out.pooler_output   # (B, 1024)

    def forward(self, pixel_values: torch.Tensor):
        feats  = self.visual_features(pixel_values)   # (B, 1024)
        gated  = self.attn_gate(feats)                # (B, 1024)
        coords, emb = self.head(gated)                # (B,2), (B,256)
        return coords, emb


# ── Loss components ───────────────────────────────────────────────────────────
def mse_coord_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE on normalised (lat, lon) coordinates."""
    return F.mse_loss(pred, target)


def infonce_loss(emb: torch.Tensor, lats: torch.Tensor, lons: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    In-batch InfoNCE: match each image embedding to its own GPS 'positive'.
    GPS coordinates are encoded via random Fourier features as pseudo-location
    embeddings (lightweight, no extra encoder needed).
    """
    B = emb.size(0)
    if B < 2:
        return torch.tensor(0.0, device=emb.device)

    # Encode GPS as normalised 2-D vector (cheap but informative)
    lat_n = (lats - TTK_LAT_CENTER) / TTK_RADIUS_DEG
    lon_n = (lons - TTK_LON_CENTER) / TTK_RADIUS_DEG
    geo   = torch.stack([lat_n, lon_n], dim=1)   # (B, 2)

    # Project to same dim as emb via fixed random weights (stable across calls)
    rng   = torch.Generator(device=emb.device)
    rng.manual_seed(42)
    W = torch.randn(2, emb.size(1), generator=rng, device=emb.device)   # (2, D)
    loc_emb = F.normalize(geo @ W, dim=1)   # (B, D)
    img_emb = F.normalize(emb, dim=1)

    logits = img_emb @ loc_emb.t() / temperature   # (B, B)
    labels = torch.arange(B, device=emb.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def geofence_loss(pred_coords: torch.Tensor) -> torch.Tensor:
    """
    Soft TTK ring-road geofence penalty.
    pred_coords : (B, 2) in normalised space -> convert back to degrees.
    Penalises distance outside a circular buffer around TTK centre.
    """
    pred_lat = denorm_lat(pred_coords[:, 0])
    pred_lon = denorm_lon(pred_coords[:, 1])
    dlat = pred_lat - TTK_LAT_CENTER
    dlon = (pred_lon - TTK_LON_CENTER) * math.cos(math.radians(TTK_LAT_CENTER))
    dist_deg = torch.sqrt(dlat ** 2 + dlon ** 2 + 1e-8)
    excess = F.relu(dist_deg - TTK_RADIUS_DEG)
    return excess.mean()


def total_loss(pred: torch.Tensor, target: torch.Tensor, emb: torch.Tensor,
               lats: torch.Tensor, lons: torch.Tensor) -> tuple[torch.Tensor, dict]:
    l_mse = mse_coord_loss(pred, target)
    l_con = infonce_loss(emb, lats, lons)
    l_geo = geofence_loss(pred)
    total = l_mse + LAMBDA_CONTRASTIVE * l_con + GAMMA_GEOFENCE * l_geo
    return total, {
        "mse": l_mse.item(),
        "contrastive": l_con.item(),
        "geofence": l_geo.item(),
        "total": total.item(),
    }


# ── Backbone unfreezing ───────────────────────────────────────────────────────
def unfreeze_clip_layers(model: GAEAModel, n: int,
                         optimizer: torch.optim.Optimizer, lr: float) -> None:
    blocks = model.clip.vision_model.encoder.layers
    unlocked = []
    for layer in list(blocks)[-n:]:
        for p in layer.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                unlocked.append(p)
    for p in model.clip.vision_model.post_layernorm.parameters():
        if not p.requires_grad:
            p.requires_grad = True
            unlocked.append(p)
    if unlocked:
        optimizer.add_param_group({"params": unlocked, "lr": lr})
        print(f"  Phase 2: unfroze last {n} ViT blocks "
              f"({sum(p.numel() for p in unlocked)/1e6:.1f}M params, lr={lr:.1e})",
              flush=True)


# ── Train / Eval ──────────────────────────────────────────────────────────────
def train_epoch(model: GAEAModel, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device,
                scaler) -> dict:
    model.train()
    tot = {"total": 0.0, "mse": 0.0, "contrastive": 0.0, "geofence": 0.0}
    n   = 0

    for imgs, targets, lats, lons, _ in tqdm(loader, desc="train", leave=False,
                                              file=sys.stdout):
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lats    = lats.to(device, non_blocking=True)
        lons    = lons.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred, emb = model(imgs)
                loss, parts = total_loss(pred, targets, emb, lats, lons)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, emb = model(imgs)
            loss, parts = total_loss(pred, targets, emb, lats, lons)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

        bs = imgs.size(0)
        for k in tot:
            tot[k] += parts[k] * bs
        n += bs

    return {k: v / max(n, 1) for k, v in tot.items()}


@torch.no_grad()
def evaluate_raw(model: GAEAModel, loader: DataLoader,
                 device: torch.device) -> dict:
    """Raw per-image Haversine evaluation."""
    model.eval()
    all_errors: list[float] = []
    all_pred_lat: list[float] = []
    all_pred_lon: list[float] = []

    for imgs, _, lats, lons, _ in tqdm(loader, desc="eval", leave=False, file=sys.stdout):
        imgs = imgs.to(device, non_blocking=True)
        pred, _ = model(imgs)
        pred_np  = pred.cpu().numpy()
        pred_lat = denorm_lat(pred_np[:, 0])
        pred_lon = denorm_lon(pred_np[:, 1])
        errs = haversine_m_vec(
            lats.numpy(), lons.numpy(), pred_lat, pred_lon
        )
        all_errors.extend(errs.tolist())
        all_pred_lat.extend(pred_lat.tolist())
        all_pred_lon.extend(pred_lon.tolist())

    arr = np.array(all_errors, dtype=np.float64)
    if len(arr) == 0:
        return dict(raw_mean_m=0, raw_median_m=0, recall_100m=0,
                    recall_1km=0, errors=arr, pred_lat=[], pred_lon=[])
    return dict(
        raw_mean_m   = float(arr.mean()),
        raw_median_m = float(np.median(arr)),
        recall_100m  = float((arr < 100).mean() * 100),
        recall_500m  = float((arr < 500).mean() * 100),
        recall_1km   = float((arr < 1000).mean() * 100),
        recall_3km   = float((arr < 3000).mean() * 100),
        errors       = arr,
        pred_lat     = all_pred_lat,
        pred_lon     = all_pred_lon,
    )


# ── VRAM sweep ────────────────────────────────────────────────────────────────
def vram_sweep(model: GAEAModel, device: torch.device) -> list[dict]:
    """Measure peak VRAM for a range of batch sizes (inference only)."""
    if device.type != "cuda":
        return []
    results = []
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    for bs in [4, 8, 16, 24, 32, 40, 48]:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            x = dummy_input.expand(bs, -1, -1, -1).contiguous()
            with torch.no_grad():
                model(x)
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            results.append({"batch": bs, "peak_gib": round(peak, 4)})
        except torch.cuda.OutOfMemoryError:
            results.append({"batch": bs, "peak_gib": None})
            torch.cuda.empty_cache()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    set_seed(SEED)
    device = resolve_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}", flush=True)
    print("  GAEA — Geospatial Adaptive Estimation Architecture", flush=True)
    print("  Moscow TTK-10k  |  Raw Per-Image Mean Protocol", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Device          : {device}", flush=True)
    print(f"  Dataset         : {METADATA_CSV}", flush=True)
    print(f"  Batch size      : {BATCH_SIZE}", flush=True)
    print(f"  LR head/backbone: {LR_HEAD:.1e} / {LR_BACKBONE:.1e}", flush=True)
    print(f"  Unfreeze at ep  : {UNFREEZE_EPOCH}  (last {UNFREEZE_LAYERS} ViT blocks)", flush=True)
    print(f"  Lambda_c        : {LAMBDA_CONTRASTIVE}  |  Gamma_gf: {GAMMA_GEOFENCE}", flush=True)
    print(f"  Patience        : {PATIENCE}  (raw val mean)", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Load metadata ─────────────────────────────────────────────────────
    if not METADATA_CSV.is_file():
        print(f"ERROR: {METADATA_CSV} not found", file=sys.stderr); sys.exit(1)
    df   = pd.read_csv(METADATA_CSV)
    mask = df["image_path"].apply(lambda p: (DATASET_ROOT / p).is_file())
    df   = df[mask].reset_index(drop=True)
    print(f"  Loaded {len(df)} rows (file-existence verified).", flush=True)

    # ── 80/10/10 split by location_id ─────────────────────────────────────
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, tmp_idx = next(gss1.split(df, groups=df["location_id"].values))
    df_tmp = df.iloc[tmp_idx].reset_index(drop=True)
    gss2   = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_i, test_i = next(gss2.split(df_tmp, groups=df_tmp["location_id"].values))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df_tmp.iloc[val_i].reset_index(drop=True)
    df_test  = df_tmp.iloc[test_i].reset_index(drop=True)
    print(f"  Split -> train {len(df_train)} | val {len(df_val)} | test {len(df_test)}", flush=True)

    # ── DataLoaders ───────────────────────────────────────────────────────
    kw = dict(num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
              persistent_workers=(NUM_WORKERS > 0))
    train_ds  = TTKDataset(df_train, DATASET_ROOT, _clip_transforms(train=True))
    val_ds    = TTKDataset(df_val,   DATASET_ROOT, _clip_transforms(train=False))
    test_ds   = TTKDataset(df_test,  DATASET_ROOT, _clip_transforms(train=False))
    train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn, drop_last=True, **kw)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, **kw)
    test_ldr  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, **kw)

    # ── Model ─────────────────────────────────────────────────────────────
    model = GAEAModel().to(device)
    n_total  = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total params    : {n_total:.1f}M  |  Trainable: {n_train:.1f}M", flush=True)

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR_HEAD, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── VRAM sweep (do once before training) ──────────────────────────────
    print("\n  Running VRAM sweep ...", flush=True)
    vram_data = vram_sweep(model, device)
    print(f"  VRAM sweep: {vram_data}", flush=True)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_mean = float("inf")
    patience_cnt  = 0
    history: list[dict] = []
    phase2_armed  = False
    header_written = False

    if RESUME and LAST_CKPT.is_file():
        print(f"  Resuming from {LAST_CKPT.name} ...", flush=True)
        try:
            ck = torch.load(str(LAST_CKPT), map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(str(LAST_CKPT), map_location=device)
        model.attn_gate.load_state_dict(ck["attn_gate"])
        model.head.load_state_dict(ck["head"])
        phase2_armed  = ck.get("phase2_armed", False)
        # Unfreeze CLIP BEFORE loading optimizer so param group count matches
        if phase2_armed:
            unfreeze_clip_layers(model, UNFREEZE_LAYERS, optimizer, LR_BACKBONE)
        optimizer.load_state_dict(ck["optimizer"])
        # Rebuild scheduler from scratch (checkpoint may have wrong T_max/base_lrs
        # if MAX_EPOCHS changed between runs).
        saved_ep      = ck.get("epoch", 1)
        start_epoch   = saved_ep + 1
        best_val_mean = ck.get("best_val_mean", float("inf"))
        patience_cnt  = ck.get("patience_cnt", 0)
        history       = ck.get("history", [])
        # PyTorch CosineAnnealingLR with last_epoch>0 uses group['lr'] (not
        # initial_lr) in its recursive formula on the first step.  If lr was
        # saved as eta_min (fully-decayed from a short run) the schedule stays
        # stuck at eta_min.  Fix: reset each group's lr to initial_lr first.
        _ilrs = [LR_HEAD] + [LR_BACKBONE] * (len(optimizer.param_groups) - 1)
        for pg, ilr in zip(optimizer.param_groups, _ilrs):
            pg["initial_lr"] = ilr   # overwrite; must match base_lrs
            pg["lr"] = ilr           # reset lr so recursive formula starts right
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-7,
            last_epoch=-1,           # fresh start; step() below advances to ep0
        )
        # Fast-forward the scheduler to saved_ep (only ~10 steps, negligible cost)
        for _ in range(saved_ep):
            scheduler.step()
        print(f"  Resumed at ep {start_epoch}, best={best_val_mean:.1f} m"
              f"  (scheduler rebuilt @ ep {saved_ep}, T_max={MAX_EPOCHS},"
              f"  lr={scheduler.get_last_lr()[0]:.2e})", flush=True)

    # ── Epoch-0 baseline ──────────────────────────────────────────────────
    print("\n  --- Epoch 0 Baseline (frozen CLIP, random head) ---", flush=True)
    baseline = evaluate_raw(model, val_ldr, device)
    print(
        f"  Baseline  val_mean={baseline['raw_mean_m']:.1f} m"
        f"  val_median={baseline['raw_median_m']:.1f} m"
        f"  R@1km={baseline['recall_1km']:.2f}%",
        flush=True,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n  Starting training from epoch {start_epoch} ...\n", flush=True)
    CSV_FIELDS = ["epoch", "lr", "train_total", "train_mse", "train_contrastive",
                  "train_geofence", "val_mean_m", "val_median_m",
                  "val_recall_100m", "val_recall_1km"]

    for epoch in range(start_epoch, MAX_EPOCHS + 1):

        if epoch == UNFREEZE_EPOCH and not phase2_armed:
            unfreeze_clip_layers(model, UNFREEZE_LAYERS, optimizer, LR_BACKBONE)
            phase2_armed = True
            # Sync scheduler base_lrs with the new param group
            if hasattr(scheduler, "base_lrs"):
                n_missing = len(optimizer.param_groups) - len(scheduler.base_lrs)
                for _ in range(n_missing):
                    scheduler.base_lrs.append(LR_BACKBONE)

        tr   = train_epoch(model, train_ldr, optimizer, device, scaler)
        val  = evaluate_raw(model, val_ldr, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        is_best = val["raw_mean_m"] < best_val_mean - 1.0
        marker  = " *** BEST ***" if is_best else ""

        row = dict(
            epoch               = epoch,
            lr                  = lr_now,
            train_total         = tr["total"],
            train_mse           = tr["mse"],
            train_contrastive   = tr["contrastive"],
            train_geofence      = tr["geofence"],
            val_mean_m          = val["raw_mean_m"],
            val_median_m        = val["raw_median_m"],
            val_recall_100m     = val["recall_100m"],
            val_recall_1km      = val["recall_1km"],
        )
        history.append(row)

        # Write CSV
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
            f"  [mse={tr['mse']:.3f} c={tr['contrastive']:.3f} gf={tr['geofence']:.4f}]"
            f"  val={val['raw_mean_m']:.0f} m"
            f"  med={val['raw_median_m']:.0f} m"
            f"  R@1km={val['recall_1km']:.1f}%"
            f"  lr={lr_now:.2e}{marker}",
            flush=True,
        )

        if is_best:
            best_val_mean = val["raw_mean_m"]
            patience_cnt  = 0
            ckpt = dict(
                epoch     = epoch,
                attn_gate = model.attn_gate.state_dict(),
                head      = model.head.state_dict(),
            )
            if phase2_armed:
                ckpt["clip_layers"] = {
                    "vision_model": {k: v for k, v in
                                     model.clip.vision_model.state_dict().items()
                                     if any(f"encoder.layers.{i}." in k
                                            for i in range(
                                                len(model.clip.vision_model.encoder.layers) - UNFREEZE_LAYERS,
                                                len(model.clip.vision_model.encoder.layers)))
                                     or "post_layernorm" in k}
                }
            torch.save(ckpt, str(BEST_CKPT))
            print(f"    + Best saved (val={best_val_mean:.1f} m)", flush=True)

        # Save last checkpoint (after updating best_val_mean / patience)
        torch.save(dict(
            epoch         = epoch,
            attn_gate     = model.attn_gate.state_dict(),
            head          = model.head.state_dict(),
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
                print(f"  Early stop at epoch {epoch} (patience={PATIENCE})", flush=True)
                break

    print("\n  Training complete.", flush=True)

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n  Loading best weights for test ...", flush=True)
    if BEST_CKPT.is_file():
        ck = torch.load(str(BEST_CKPT), map_location=device,
                        weights_only=False if sys.version_info >= (3, 9) else None)
        model.attn_gate.load_state_dict(ck["attn_gate"])
        model.head.load_state_dict(ck["head"])

    test_res = evaluate_raw(model, test_ldr, device)
    errs     = test_res["errors"]
    best_ep  = min(history, key=lambda h: h["val_mean_m"])["epoch"] if history else 0

    print(f"\n{'─'*60}", flush=True)
    print(f"  GAEA  TEST RESULTS  ({len(errs)} images)", flush=True)
    print(f"  Best epoch   : {best_ep}", flush=True)
    print(f"  Raw Mean     : {errs.mean():.1f} m", flush=True)
    print(f"  Raw Median   : {np.median(errs):.1f} m", flush=True)
    print(f"  Recall@100m  : {test_res['recall_100m']:.2f}%", flush=True)
    print(f"  Recall@500m  : {test_res['recall_500m']:.2f}%", flush=True)
    print(f"  Recall@1km   : {test_res['recall_1km']:.2f}%", flush=True)
    print(f"  Recall@3km   : {test_res['recall_3km']:.2f}%", flush=True)
    print(f"{'─'*60}", flush=True)

    target_m = 3000.0
    if errs.mean() < target_m:
        print(f"\n  *** DIPLOMA CORE RESULT ***  {errs.mean():.1f} m < {target_m:.0f} m  ***",
              flush=True)
    else:
        delta = errs.mean() - target_m
        print(f"\n  Target {target_m:.0f} m not yet reached "
              f"(delta = +{delta:.0f} m).  Consider more epochs.", flush=True)

    # ── Save artefacts ────────────────────────────────────────────────────
    np.save(str(ERRORS_NPY), errs)

    json_out = dict(
        model            = "GAEA_Final",
        best_epoch       = best_ep,
        test_raw_mean_m  = float(errs.mean()),
        test_raw_median_m= float(np.median(errs)),
        recall_100m      = float(test_res["recall_100m"]),
        recall_500m      = float(test_res["recall_500m"]),
        recall_1km       = float(test_res["recall_1km"]),
        recall_3km       = float(test_res["recall_3km"]),
        baseline_refs_m  = {"geoclip": 3100.0, "kopernik": 3545.0},
        vram_sweep_gib   = vram_data,
        n_train          = len(df_train),
        n_val            = len(df_val),
        n_test           = len(df_test),
        lambda_c         = LAMBDA_CONTRASTIVE,
        gamma_gf         = GAMMA_GEOFENCE,
    )
    TEST_JSON.write_text(json.dumps(json_out, indent=2), encoding="utf-8")

    print(f"\n  Artefacts saved to: {OUTPUT_DIR}", flush=True)
    print(f"    {BEST_CKPT.name}", flush=True)
    print(f"    {METRICS_CSV.name}", flush=True)
    print(f"    {TEST_JSON.name}", flush=True)
    print(f"    {ERRORS_NPY.name}", flush=True)
    print(f"\n  Min Raw Val Mean : {best_val_mean:.1f} m  (ep {best_ep})", flush=True)
    print(f"  Test Raw Mean    : {errs.mean():.1f} m\n", flush=True)


if __name__ == "__main__":
    main()
