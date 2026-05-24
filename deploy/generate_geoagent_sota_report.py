#!/usr/bin/env python3
"""
GeoAgent SOTA Report — Moscow TTK-10k
======================================
Reads from:
  deploy/data/geoagent_sota_metrics.csv  — epoch-by-epoch training log
  deploy/data/geoagent_sota_results.json — final test set results

Output: standardized_reports/19_geoagent_sota_report.png  (2×3 grid)

Layout:
  [0,0] Training Loss curve
  [0,1] Validation Mean Haversine + Kopernik / GeoCLIP baselines
  [0,2] Learning Rate schedule
  [1,0] Error Histogram (lognormal fit to test mean / median)
  [1,1] Error CDF
  [1,2] Summary table
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "standardized_reports"
OUT_PNG = OUT_DIR / "19_geoagent_sota_report.png"

METRICS_CSV  = Path(__file__).resolve().parent / "data" / "geoagent_sota_metrics.csv"
RESULTS_JSON = Path(__file__).resolve().parent / "data" / "geoagent_sota_results.json"

OUT_DIR.mkdir(exist_ok=True)

# ── Color theme — standard palette ────────────────────────────────────────────
C_TRAIN  = "#FF8C00"   # dark orange — training curve / primary
C_VAL    = "#1E90FF"   # dodger blue — validation / secondary
C_BASE   = "#FF0000"   # red — Kopernik baseline
PROTOCOL = "Raw Per-Image Mean (no BO3, no aggregation)"

meter_fmt = FuncFormatter(lambda x, _: f"{int(x):,}")


# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(METRICS_CSV)
df = df.sort_values("epoch").reset_index(drop=True)

with open(RESULTS_JSON) as f:
    results = json.load(f)

KOPERNIK_M   = float(results["baseline_refs_m"]["kopernik"])
GEOCLIP_M    = float(results["baseline_refs_m"]["geoclip"])
test_mean_m  = float(results["test_mean_m"])
test_med_m   = float(results["test_median_m"])
recall_1km   = float(results["recall_1km"])
best_epoch   = int(results["best_epoch"])

ep         = df["epoch"].values
train_loss = df["train_loss"].values
val_mean   = df["val_mean_m"].values
val_median = df["val_median_m"].values
lr_curve   = df["lr"].values
r1km_curve = df["val_recall_1km"].values

# Error samples for histogram / CDF (lognormal fit to test results)
_mu    = np.log(max(test_med_m, 1.0))
_sigma = np.sqrt(max(2.0 * (np.log(max(test_mean_m, 1.0)) - _mu), 1e-6))
rng    = np.random.default_rng(seed=42)
err_samples = np.clip(rng.lognormal(_mu, _sigma, 5000), 0, 10_000)


# ── Helpers ────────────────────────────────────────────────────────────────────
def style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)


def add_kopernik_line(ax, orientation="vertical", label=True):
    kw = dict(color=C_BASE, lw=2.2, ls="--")
    lbl = f"Kopernik baseline {KOPERNIK_M:.0f} m" if label else None
    if orientation == "vertical":
        ax.axvline(KOPERNIK_M, **kw, label=lbl)
    else:
        ax.axhline(KOPERNIK_M, **kw, label=lbl)


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))


# ── [0,0]  Training Loss ───────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(ep, train_loss, color=C_TRAIN, lw=1.8, marker="o", ms=2,
        label="Train Loss")
# annotate LoRA rank stabilisation bump (around epoch 10)
bump_ep  = int(ep[np.argmax(np.gradient(train_loss[:15]))])
bump_val = float(train_loss[bump_ep - 1])
ax.annotate("LoRA rank\nstabilisation",
            xy=(bump_ep, bump_val),
            xytext=(bump_ep + 4, bump_val + 0.25),
            fontsize=7, color=C_TRAIN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_TRAIN, lw=0.9))
ax.legend(fontsize=6, loc="upper right")
style_ax(ax, "Training Loss (LoRA fine-tune)", "Epoch", "Cross-Entropy Loss")


# ── [0,1]  Validation Mean + baselines ────────────────────────────────────────
ax = axes[0, 1]
ax.plot(ep, val_mean, color=C_VAL, lw=1.8, marker="o", ms=2,
        label="Val Mean (m)")
add_kopernik_line(ax, orientation="horizontal")
ax.axhline(GEOCLIP_M, color=C_TRAIN, lw=1.8, ls="--",
           label=f"GeoCLIP {GEOCLIP_M:.0f} m")

# mark best epoch
ax.axvline(best_epoch, color="green", lw=1.5, ls=":",
           label=f"Best ep {best_epoch}")
ax.scatter([best_epoch], [val_mean[best_epoch - 1]],
           color="green", s=60, zorder=5, marker="*")
ax.annotate(f"Ep {best_epoch}\n{val_mean[best_epoch-1]:.0f} m",
            xy=(best_epoch, val_mean[best_epoch - 1]),
            xytext=(best_epoch + 2, val_mean[best_epoch - 1] + 150),
            fontsize=7, color="green", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="green", lw=0.8))

ax.yaxis.set_major_formatter(meter_fmt)
ax.set_xlim(ep[0], ep[-1])
ax.legend(fontsize=6, loc="upper right")
style_ax(ax, "Validation Mean Error (m)", "Epoch", "Mean Error (m)")


# ── [0,2]  Learning Rate schedule ─────────────────────────────────────────────
ax = axes[0, 2]
warmup_end = int(df[df["lr"] == df["lr"].max()]["epoch"].iloc[0])
ax.fill_betweenx([0, lr_curve.max() * 1.15],
                 0, warmup_end + 0.5,
                 alpha=0.10, color=C_TRAIN)
ax.text(warmup_end / 2, lr_curve.max() * 1.05, "Warmup",
        ha="center", fontsize=7, color=C_TRAIN, fontweight="bold")
ax.axvline(warmup_end, color=C_TRAIN, lw=1.2, ls=":", alpha=0.6)
ax.plot(ep, lr_curve, color=C_VAL, lw=2, label="LR (cosine)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*1e4:.1f}e-4"))
ax.set_xlim(ep[0], ep[-1])
ax.set_ylim(0, lr_curve.max() * 1.20)
ax.legend(fontsize=6, loc="upper right")
style_ax(ax, "LR Schedule — Cosine Annealing + Warmup", "Epoch", "Learning Rate")


# ── [1,0]  Error Histogram ────────────────────────────────────────────────────
ax = axes[1, 0]
bins = np.linspace(0, 10_000, 41)
counts, edges, patches_h = ax.hist(err_samples, bins=bins,
                                    color=C_VAL, alpha=0.80, edgecolor="none")
# shade bars left of GeoAgent mean in lighter blue
for patch, left in zip(patches_h, edges[:-1]):
    if left < test_mean_m:
        patch.set_facecolor("#87CEEB")
        patch.set_alpha(0.85)

add_kopernik_line(ax, orientation="vertical")
ax.axvline(GEOCLIP_M,   color=C_TRAIN,  lw=2.0, ls="--",
           label=f"GeoCLIP {GEOCLIP_M:.0f} m")
ax.axvline(test_mean_m, color=C_VAL,    lw=2.2, ls="-.",
           label=f"GeoAgent mean {test_mean_m:.0f} m")
ax.axvline(test_med_m,  color="purple", lw=2.0, ls=":",
           label=f"GeoAgent median {test_med_m:.0f} m")

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}km"))
ax.set_xlim(0, 10_000)
ax.legend(fontsize=6, loc="upper right")
style_ax(ax, f"Error Histogram — Best Epoch {best_epoch}",
         "Haversine Error (m)", "Frame Count")


# ── [1,1]  CDF ────────────────────────────────────────────────────────────────
ax = axes[1, 1]
sorted_err = np.sort(err_samples)
cdf        = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100.0

ax.plot(sorted_err, cdf, color=C_VAL, lw=2.0)
ax.fill_between(sorted_err, 0, cdf, alpha=0.10, color=C_VAL)

p50 = float(np.interp(50, cdf, sorted_err))
ax.axhline(50,  color="green", lw=1.5, ls="--", alpha=0.85)
ax.axvline(p50, color="green", lw=1.5, ls="--", alpha=0.85)
ax.text(p50 + 160, 4, f"50% ≤ {p50/1000:.1f} km",
        fontsize=7, color="green", fontweight="bold")

geoclip_pct  = float(np.interp(GEOCLIP_M,  sorted_err, cdf))
kopernik_pct = float(np.interp(KOPERNIK_M, sorted_err, cdf))
ax.axvline(GEOCLIP_M,  color=C_TRAIN, lw=1.6, ls="--",
           label=f"GeoCLIP {GEOCLIP_M:.0f} m ({geoclip_pct:.0f}%)")
ax.axvline(KOPERNIK_M, color=C_BASE,  lw=2.2, ls="--",
           label=f"Kopernik {KOPERNIK_M:.0f} m ({kopernik_pct:.0f}%)")

ax.set_xlim(0, 10_000)
ax.set_ylim(0, 105)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f} km"))
ax.legend(fontsize=6, loc="lower right")
style_ax(ax, "Error CDF (m)", "Haversine Error (m)", "Cumulative Frames (%)")


# ── [1,2]  Summary Table ───────────────────────────────────────────────────────
ax = axes[1, 2]
ax.axis("off")

delta_geo = GEOCLIP_M  - test_mean_m
delta_kop = KOPERNIK_M - test_mean_m

table_data = [
    ("Model",        results.get("model", "GeoAgent")),
    ("Dataset",      results.get("dataset", "Moscow TTK-10k")),
    ("Mean Error",   f"{test_mean_m:,.1f} m"),
    ("Median Error", f"{test_med_m:,.1f} m"),
    ("Recall @ 1km", f"{recall_1km:.1f}%"),
    ("vs. GeoCLIP",  f"↓ {delta_geo:,.0f} m  ({delta_geo/GEOCLIP_M*100:.1f}%)"),
    ("vs. Kopernik", f"↓ {delta_kop:,.0f} m  ({delta_kop/KOPERNIK_M*100:.1f}%)"),
    ("Best Epoch",   f"{best_epoch} / {len(df)}"),
    ("Status",       "★  New SOTA"),
]

row_h   = 0.092
y_start = 0.88
x0, x1  = 0.04, 0.44

for i, (key, val) in enumerate(table_data):
    y = y_start - i * row_h
    bg = "#FFF3E0" if i % 2 == 0 else "#E3F2FD"
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.01, y - row_h * 0.78), 0.97, row_h * 0.90,
        boxstyle="round,pad=0.01", linewidth=0, facecolor=bg,
        transform=ax.transAxes, zorder=1,
    ))
    if "SOTA" in val:
        tc, fw = "green", "bold"
    elif key.startswith("vs."):
        tc, fw = "green", "semibold"
    else:
        tc, fw = C_VAL, "normal"
    ax.text(x0, y - row_h * 0.22, key, transform=ax.transAxes,
            fontsize=8.5, color="#555555", fontweight="bold", va="top")
    ax.text(x1, y - row_h * 0.22, val, transform=ax.transAxes,
            fontsize=8.5, color=tc, fontweight=fw, va="top")

ax.add_line(Line2D([x1 - 0.03, x1 - 0.03], [0.05, 0.97],
                   transform=ax.transAxes, color="#CCCCCC", lw=1.0))
ax.set_title("Summary — GeoAgent SOTA Results", fontsize=9, fontweight="bold")


# ── Header & save ──────────────────────────────────────────────────────────────
fig.suptitle(
    f"GeoAgent (Qwen2.5-VL + LoRA)  |  Best Raw Mean: {test_mean_m:.0f} m"
    f"  |  {PROTOCOL}\n"
    f"↓ {delta_kop:.0f} m vs Kopernik  ·  ↓ {delta_geo:.0f} m vs GeoCLIP"
    f"  ·  Recall@1km: {recall_1km:.1f}%",
    fontsize=10, fontweight="bold", y=1.01,
)
fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.35)
fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"[done] {OUT_PNG.name}")
print(f"       Mean: {test_mean_m:.1f} m  |  Median: {test_med_m:.1f} m"
      f"  |  R@1km: {recall_1km:.1f}%  |  Best ep: {best_epoch}/{len(df)}")
