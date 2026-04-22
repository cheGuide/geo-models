#!/usr/bin/env python3
"""
TransGeo (CVUSA/TTK checkpoint): для каждого изображения из custom_images_inference.json —
топ-K по val БД TTK (как test_single_image.py), один проход embeddings по спутнику.

При несовместимой CUDA (например RTX 50 / sm_120 и старый PyTorch) используйте --cpu,
иначе forward может дать неверные одинаковые similarity.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
TG = REPO / "TransGeo2022"
sys.path.insert(0, str(TG))

from model.TransGeo import TransGeo  # noqa: E402


def input_transform(size):
    import torchvision.transforms as transforms

    return transforms.Compose(
        [
            transforms.Resize(size=tuple(size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def parse_lat_lon_from_grd(grd_path: str) -> tuple[float, float] | None:
    name = os.path.basename(grd_path)
    try:
        sp = name.replace(".jpg", "").split("_")
        if len(sp) >= 5:
            return float(sp[2]), float(sp[3])
    except Exception:
        pass
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json",
        type=Path,
        default=REPO / "benchmarks_results" / "custom_images_inference.json",
        help="JSON со списком images (как у run_inference_custom_images)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=TG / "checkpoints" / "CVUSA_model" / "result" / "checkpoint.pth.tar",
    )
    p.add_argument("--root", type=Path, default=REPO / "ttk_10k_full")
    p.add_argument("--sat_res", type=int, default=320)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=REPO / "benchmarks_results" / "transgeo_custom_images.json",
    )
    args = p.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu" if args.cpu else "cuda")

    data = json.loads(args.json.read_text(encoding="utf-8"))
    image_paths = [Path(x) for x in data.get("images", [])]

    class MArgs:
        dataset = "ttk"
        dim = 1000
        crop = False
        sat_res = args.sat_res
        fov = 0

    model = TransGeo(args=MArgs())
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ck["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    grd_size = [112, 616]
    sat_size = [args.sat_res, args.sat_res]
    transform_grd = input_transform(grd_size)
    transform_sat = input_transform(sat_size)

    val_csv = args.root / "transgeo" / "splits" / "val-ttk.csv"
    lines = [l.strip() for l in val_csv.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _path(rel: str) -> str:
        return os.path.join(str(args.root), rel) if not os.path.isabs(rel) else rel

    print(f"Satellite embeddings: {len(lines)} val rows...")
    ref_features = []
    for i in range(0, len(lines), args.batch_size):
        batch_lines = lines[i : i + args.batch_size]
        imgs = []
        for ln in batch_lines:
            parts = ln.split(",")
            sat_path = _path(parts[0].strip())
            try:
                im = Image.open(sat_path).convert("RGB")
                imgs.append(transform_sat(im))
            except Exception as e:
                print(f"  skip {sat_path}: {e}")
                imgs.append(torch.zeros(3, sat_size[0], sat_size[1]))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            emb = model.reference_net(x=batch, indexes=None)
        ref_features.append(emb.cpu().numpy())
    ref_features = np.vstack(ref_features)
    ref_features = ref_features / (np.linalg.norm(ref_features, axis=1, keepdims=True) + 1e-8)

    out: dict = {"checkpoint": str(args.checkpoint), "dataset_root": str(args.root), "transgeo": {}}

    for path in image_paths:
        key = path.name
        if not path.is_file():
            out["transgeo"][key] = {"path": str(path), "error": "file not found"}
            continue
        try:
            img = Image.open(path).convert("RGB")
            img_t = transform_grd(img).unsqueeze(0).to(device)
            with torch.no_grad():
                qe = model.query_net(img_t)
            qe = qe.cpu().numpy().flatten()
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            sim = np.dot(ref_features, qe)
            top_idx = np.argsort(sim)[::-1][: args.top_k]
            rows = []
            for rank, idx in enumerate(top_idx, 1):
                parts = lines[idx].split(",")
                grd = parts[1].strip() if len(parts) > 1 else ""
                ll = parse_lat_lon_from_grd(grd)
                rows.append(
                    {
                        "rank": rank,
                        "sim": float(sim[idx]),
                        "lat": ll[0] if ll else None,
                        "lon": ll[1] if ll else None,
                        "ground_ref": grd,
                    }
                )
            out["transgeo"][key] = {"path": str(path.resolve()), "top_k": rows}
        except Exception as e:
            out["transgeo"][key] = {"path": str(path), "error": str(e)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
