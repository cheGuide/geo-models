#!/usr/bin/env python3
"""
Batch Re-training & Evaluation — Moscow-TTK Mean-Centric Protocol.

Runs all 4 v2 baseline training scripts sequentially, then collects
best-epoch metrics and writes baseline_revaluation_metrics.csv.

Usage:
    python run_baseline_revaluation.py [--models kopernik,cplanet,cosplace,geoclip]

Deliverables:
    kopernik/kopernik_v2_report.png
    kopernik/cplanet_v2_report.png
    CosPlace/cosplace_v2_report.png
    geo-clip/geoclip_v2_report.png
    baseline_revaluation_metrics.csv   (this directory)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

_REPO = Path(__file__).resolve().parent

# Per-model Python executables (each model may have its own venv)
def _venv(subdir: str) -> str:
    # geo-clip venv has torch nightly cu128 (supports Blackwell RTX 50xx) + all shared deps
    geo = _REPO / "geo-clip" / ".venv" / "Scripts" / "python.exe"
    if geo.is_file():
        return str(geo)
    # Fallback: per-model venv, then root venv
    p = _REPO / subdir / ".venv" / "Scripts" / "python.exe"
    if p.is_file():
        return str(p)
    root = _REPO / ".venv" / "Scripts" / "python.exe"
    return str(root) if root.is_file() else sys.executable

# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    "kopernik": {
        "label":   "Kopernik (Direct Regression)",
        "python":  _venv(""),          # root venv: torch cu124 + CUDA
        "script":  _REPO / "kopernik" / "train_kopernik_v2.py",
        "metrics": _REPO / "kopernik" / "kopernik_v2_metrics.csv",
        "report":  _REPO / "kopernik" / "kopernik_v2_report.png",
        "mean_col":   "test_mean_m",
        "median_col": "test_median_m",
        "stop_col":   "val_bo3_mean_m",
    },
    "cplanet": {
        "label":   "CPlaNet (S2-Cell Classification)",
        "python":  _venv(""),          # root venv: torch cu124 + s2sphere
        "script":  _REPO / "kopernik" / "train_cplanet_v2.py",
        "metrics": _REPO / "kopernik" / "cplanet_v2_metrics.csv",
        "report":  _REPO / "kopernik" / "cplanet_v2_report.png",
        "mean_col":   "test_mean_m",
        "median_col": "test_med_m",
        "stop_col":   "val_mean_m",
    },
    "cosplace": {
        "label":   "CosPlace (Global Retrieval)",
        "python":  _venv(""),          # root venv: torch cu124 + cosface
        "script":  _REPO / "CosPlace" / "finetune_cosplace_v2.py",
        "metrics": _REPO / "CosPlace" / "cosplace_v2_metrics.csv",
        "report":  _REPO / "CosPlace" / "cosplace_v2_report.png",
        "mean_col":   "mean_err_m",
        "median_col": "median_err_m",
        "stop_col":   "val_loss_m",
    },
    "geoclip": {
        "label":   "GeoCLIP (Semantic Alignment)",
        "python":  _venv("geo-clip"),  # geo-clip/.venv: torch nightly cu128 + transformers
        "script":  _REPO / "geo-clip" / "train_geoclip_v2.py",
        "metrics": _REPO / "geo-clip" / "geoclip_v2_metrics.csv",
        "report":  _REPO / "geo-clip" / "geoclip_v2_report.png",
        "mean_col":   "val_mean_m",
        "median_col": "val_median_m",
        "stop_col":   "val_mean_m",
        "result_json":     _REPO / "geo-clip" / "geoclip_v2_test_results.json",
        "json_mean_key":   "test_mean_m",
        "json_median_key": "test_median_m",
        "json_epoch_key":  "best_epoch",
    },
}

OUTPUT_CSV = _REPO / "baseline_revaluation_metrics.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_script(script: Path, label: str, python: str) -> tuple[bool, float]:
    """Run a training script as a subprocess. Returns (success, elapsed_s)."""
    print(f"\n{'='*70}")
    print(f"  RUNNING: {label}")
    print(f"  Script : {script}")
    print(f"{'='*70}", flush=True)

    if not script.is_file():
        print(f"  [ERROR] Script not found: {script}", flush=True)
        return False, 0.0

    t0 = time.time()
    result = subprocess.run(
        [python, str(script)],
        cwd=str(script.parent),
        # Inherit environment (CUDA, PATH, etc.)
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [ERROR] {label} exited with code {result.returncode}", flush=True)
        return False, elapsed

    print(f"\n  [OK] {label} finished in {elapsed/60:.1f} min", flush=True)
    return True, elapsed


def read_best_row(cfg: dict) -> dict | None:
    """
    Read the metrics CSV and return the row with the MINIMUM value in stop_col.
    That is the exact early-stopping epoch as defined by the Mean-Centric Protocol.
    Skips epoch 0 (baseline row) — epoch 0 has no trained weights.
    """
    csv_path = cfg["metrics"]
    if not csv_path.is_file():
        print(f"  [WARN] Metrics CSV not found: {csv_path}", flush=True)
        return None
    df = pd.read_csv(csv_path)
    # Skip epoch-0 (no weights trained); also skip rows where stop_col is NaN
    df = df[df["epoch"].astype(int) >= 1].copy()
    stop_col = cfg["stop_col"]
    if stop_col not in df.columns:
        print(f"  [WARN] stop_col '{stop_col}' not in {csv_path.name}", flush=True)
        return None
    df = df.dropna(subset=[stop_col])
    if df.empty:
        return None
    best_row = df.loc[df[stop_col].idxmin()]
    return best_row.to_dict()


def collect_results() -> list[dict]:
    """
    Collect best-epoch metrics from all model outputs.
    For models with a result_json (GeoCLIP), reads the JSON for test-set metrics.
    For all others, reads the per-epoch metrics CSV and picks the row with the
    minimum stop_col value (exact Mean-Centric early-stop epoch).
    """
    rows = []
    for model_key, cfg in MODEL_REGISTRY.items():
        print(f"\n  Collecting results for: {cfg['label']}", flush=True)

        # GeoCLIP: test-set metrics are in the result JSON
        if "result_json" in cfg:
            rj = cfg["result_json"]
            if not rj.is_file():
                print(f"  [SKIP] result_json not found: {rj}", flush=True)
                continue
            with open(rj, encoding="utf-8") as f:
                data = json.load(f)
            mean_m   = float(data.get(cfg["json_mean_key"],   float("nan")))
            median_m = float(data.get(cfg["json_median_key"], float("nan")))
            best_ep  = int(data.get(cfg["json_epoch_key"], -1))
        else:
            best = read_best_row(cfg)
            if best is None:
                print(f"  [SKIP] No valid metrics found for {model_key}", flush=True)
                continue
            mean_m   = float(best.get(cfg["mean_col"],   float("nan")))
            median_m = float(best.get(cfg["median_col"], float("nan")))
            best_ep  = int(best.get("epoch", -1))

        print(f"  Best epoch: {best_ep}  Mean: {mean_m:.1f}m  Median: {median_m:.1f}m")
        rows.append({
            "model":      model_key,
            "label":      cfg["label"],
            "best_epoch": best_ep,
            "mean_m":     round(mean_m,   1),
            "median_m":   round(median_m, 1),
            "report_png": str(cfg["report"]),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch revaluation runner")
    parser.add_argument(
        "--models", type=str,
        default="kopernik,cplanet,cosplace,geoclip",
        help="Comma-separated list of models to run (default: all 4)",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="Skip training; only collect existing results and write CSV",
    )
    args = parser.parse_args()

    requested = [m.strip().lower() for m in args.models.split(",")]
    unknown = [m for m in requested if m not in MODEL_REGISTRY]
    if unknown:
        print(f"[ERROR] Unknown model(s): {unknown}. Valid: {list(MODEL_REGISTRY)}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("  Moscow-TTK Mean-Centric Protocol — Batch Revaluation")
    print(f"  Models  : {requested}")
    print(f"  Collect : {'only' if args.collect_only else 'train + collect'}")
    print("=" * 70)

    run_results: dict[str, tuple[bool, float]] = {}

    if not args.collect_only:
        for model_key in requested:
            cfg = MODEL_REGISTRY[model_key]
            ok, elapsed = run_script(cfg["script"], cfg["label"], cfg["python"])
            run_results[model_key] = (ok, elapsed)
    else:
        print("\n[INFO] --collect-only: skipping training.", flush=True)

    # ── Collect & write consolidated CSV ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Collecting best-epoch metrics...")
    print("=" * 70)

    rows = collect_results()

    if not rows:
        print("\n[WARN] No results collected. Check that training scripts completed.", flush=True)
        sys.exit(1)

    out_df = pd.DataFrame(rows, columns=["model", "label", "best_epoch", "mean_m", "median_m", "report_png"])
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*70}")
    print(f"  CONSOLIDATED RESULTS  ->  {OUTPUT_CSV}")
    print(f"{'='*70}")
    print(out_df[["model", "best_epoch", "mean_m", "median_m"]].to_string(index=False))

    if run_results:
        print(f"\n{'-'*70}")
        print("  Timing:")
        for m, (ok, t) in run_results.items():
            status = "OK" if ok else "FAILED"
            print(f"  {m:<12} {status:<8} {t/60:.1f} min")

    print(f"\n{'='*70}")
    print("  Deliverables:")
    for _, cfg in MODEL_REGISTRY.items():
        if cfg["report"].is_file():
            print(f"  ✓  {cfg['report']}")
        else:
            print(f"  ✗  {cfg['report']}  (missing)")
    print(f"  ✓  {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
