#!/usr/bin/env python3
"""
Инференс GeoCLIP (baseline + fine-tuned), Kopernik (fine-tuned), Revisit-Anything VPR (fine-tuned, top-1 по БД val TTK)
по произвольным путям к изображениям.

Пример:
  py run_inference_custom_images.py --dataset_path ../ttk_10k_full
  py run_inference_custom_images.py --images a.png b.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.require_cuda import require_cuda_or_cpu


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d = math.radians(lat2 - lat1)
    lam = math.radians(lon2 - lon1)
    a = math.sin(d / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(lam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


def collect_images(args) -> list[Path]:
    import glob as glob_mod

    paths: list[Path] = []
    for p in args.images or []:
        paths.append(Path(p).resolve())
    if args.assets_glob:
        for pattern in args.assets_glob:
            paths.extend(Path(p) for p in glob_mod.glob(pattern))
    seen = set()
    out = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            k = str(p)
            if k not in seen:
                seen.add(k)
                out.append(p)
    return sorted(out, key=lambda x: str(x))


def run_geoclip(image_paths: list[Path], device: str, ckpt_ft: Path | None) -> dict:
    sys.path.insert(0, str(REPO / "geo-clip"))
    import torch
    import torch.nn as nn
    from geoclip import GeoCLIP

    def predict_one(model, path: Path):
        top_gps, top_prob = model.predict(str(path), top_k=3)
        rows = []
        for i in range(top_gps.shape[0]):
            lat, lon = float(top_gps[i, 0].item()), float(top_gps[i, 1].item())
            rows.append({"lat": lat, "lon": lon, "prob": float(top_prob[i].item())})
        return rows

    out: dict = {"baseline": {}}
    for name, ckpt in [("baseline", None), ("fine_tuned", ckpt_ft)]:
        if name == "fine_tuned" and (ckpt is None or not ckpt.is_file()):
            out["fine_tuned"] = {"_error": f"checkpoint not found: {ckpt_ft}"}
            continue
        model = GeoCLIP(from_pretrained=True)
        if ckpt is not None and ckpt.is_file():
            w = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.image_encoder.mlp.load_state_dict(w["image_encoder_mlp"])
            model.location_encoder.load_state_dict(w["location_encoder"])
            model.logit_scale = nn.Parameter(w["logit_scale"].to(model.logit_scale.dtype))
        model = model.to(device)
        model.device = device
        bucket = out.setdefault(name, {})
        for p in image_paths:
            key = p.name
            try:
                bucket[key] = {"path": str(p), "top3": predict_one(model, p)}
            except Exception as e:
                bucket[key] = {"path": str(p), "error": str(e)}
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def run_kopernik(image_paths: list[Path], ckpt: Path) -> dict:
    sys.path.insert(0, str(REPO / "kopernik"))
    import torch
    from test_moscow_dataset import load_model, preprocess, denorm

    model, lat_min, lat_max, lon_min, lon_max = load_model(str(ckpt))
    dev = next(model.parameters()).device
    out = {}
    for p in image_paths:
        try:
            from PIL import Image

            img = Image.open(p).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(dev)
            with torch.no_grad():
                o = model(x).cpu()
            la = denorm(o[0, 0].item(), lat_min, lat_max)
            lo = denorm(o[0, 1].item(), lon_min, lon_max)
            out[p.name] = {"path": str(p), "lat": la, "lon": lo}
        except Exception as e:
            out[p.name] = {"path": str(p), "error": str(e)}
    return out


def run_revisit_subprocess(image_paths: list[Path], dataset_path: Path, ckpt: Path, device: str) -> dict:
    venv_py = REPO / "AnyLoc" / ".venv" / "Scripts" / "python.exe"
    worker = REPO / "benchmarks_results" / "_revisit_worker.py"
    if not venv_py.is_file():
        return {"_error": f"venv not found: {venv_py}"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump([str(p) for p in image_paths], f)
        paths_file = f.name
    fd, out_file = tempfile.mkstemp(suffix="_revisit_out.json")
    os.close(fd)
    try:
        r = subprocess.run(
            [
                str(venv_py),
                str(worker),
                "--paths_json",
                paths_file,
                "--dataset_path",
                str(dataset_path),
                "--checkpoint",
                str(ckpt),
                "--device",
                device,
                "--output",
                out_file,
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if r.returncode != 0:
            return {"_error": (r.stderr or r.stdout or "")[:2000]}
        return json.loads(Path(out_file).read_text(encoding="utf-8"))
    finally:
        for p in (paths_file, out_file):
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, default=str(REPO / "ttk_10k_full"), help="TTK для Revisit БД")
    ap.add_argument("--images", nargs="*", default=[], help="Явные пути к файлам")
    ap.add_argument(
        "--assets_glob",
        nargs="*",
        default=[
            r"C:\Users\q\.cursor\projects\c-Users-q-Work-models\assets\*.png",
            str(REPO / "benchmarks_results" / "image" / "MOSCOW_TTK_METHODS_TABLE" / "*.png"),
        ],
        help="Glob-паттерны для сбора PNG",
    )
    ap.add_argument("--device", type=str, default="cuda", help="Только cuda (обязателен GPU)")
    ap.add_argument("--skip_geoclip", action="store_true")
    ap.add_argument("--skip_kopernik", action="store_true")
    ap.add_argument("--skip_revisit", action="store_true")
    ap.add_argument("--output", type=str, default=str(Path(__file__).parent / "custom_images_inference.json"))
    args = ap.parse_args()

    imgs = collect_images(args)
    if not imgs:
        print("Нет изображений: задайте --images или проверьте --assets_glob", file=sys.stderr)
        sys.exit(1)

    dev_resolved = require_cuda_or_cpu(args.device)
    dev_str = str(dev_resolved)

    ds = Path(args.dataset_path)
    ckpt_geo = REPO / "geo-clip" / "checkpoints_ttk" / "geoclip_ttk_final.pth"
    ckpt_kop = REPO / "kopernik" / "resnet50_moscow_localization.pth"
    ckpt_rev = REPO / "Revisit-Anything" / "logs" / "moscow_vpr_ttk" / "lightning_logs" / "version_1" / "checkpoints" / "moscow_vpr_04_R1_0.1970.ckpt"

    report: dict = {"images": [str(p) for p in imgs], "dataset_ttk": str(ds)}

    print(f"Изображений: {len(imgs)}", flush=True)

    if not args.skip_geoclip:
        print("GeoCLIP (baseline + fine-tuned)...", flush=True)
        report["geoclip"] = run_geoclip(imgs, dev_str, ckpt_geo if ckpt_geo.is_file() else None)

    if not args.skip_kopernik and ckpt_kop.is_file():
        print("Kopernik...", flush=True)
        report["kopernik_finetuned"] = run_kopernik(imgs, ckpt_kop)
    else:
        report["kopernik_finetuned"] = {"_skip": "checkpoint missing"}

    if not args.skip_revisit and (ds / "dataset_metadata.json").is_file() and ckpt_rev.is_file():
        print("Revisit-Anything VPR (retrieval по val БД, subprocess AnyLoc venv)...", flush=True)
        report["revisit_vpr_finetuned"] = run_revisit_subprocess(imgs, ds, ckpt_rev, dev_str)
    else:
        report["revisit_vpr_finetuned"] = {"_skip": "dataset or checkpoint missing"}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Сохранено: {out_path}", flush=True)


if __name__ == "__main__":
    main()
