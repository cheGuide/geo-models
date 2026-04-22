"""Вызывается из AnyLoc .venv (есть pytorch_lightning)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.require_cuda import require_cuda_or_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_json", type=str, required=True, help="JSON-массив строк путей к изображениям")
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    image_paths = [Path(p) for p in json.loads(Path(args.paths_json).read_text(encoding="utf-8"))]
    ds = Path(args.dataset_path)
    ckpt = Path(args.checkpoint)

    sys.path.insert(0, str(REPO / "Revisit-Anything" / "VLAD-BuFF"))
    sys.path.insert(0, str(REPO / "Revisit-Anything"))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from train_moscow_vpr import MoscowVPRModel, MoscowTTKDataModule

    device = require_cuda_or_cpu(args.device)

    torch.serialization.add_safe_globals([__import__("argparse").Namespace])
    model = MoscowVPRModel.load_from_checkpoint(str(ckpt), map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    dm = MoscowTTKDataModule(
        dataset_path=str(ds),
        batch_size=32,
        img_per_place=3,
        min_img_per_place=3,
        image_size=(224, 224),
        num_workers=0,
        val_split=0.15,
        place_radius_m=150.0,
        positive_radius_m=150.0,
    )
    dm.setup("validate")
    val = dm.val_dataset
    tfm = dm.val_transform

    db = val.db_images
    n_db = val.num_references
    descs = []
    with torch.no_grad():
        for i in range(0, n_db, 32):
            batch = []
            for j in range(i, min(i + 32, n_db)):
                img = Image.open(val.images_dir / db[j]["filename"]).convert("RGB")
                batch.append(tfm(img))
            xb = torch.stack(batch).to(device)
            d = model(xb)
            d = F.normalize(d, dim=-1)
            descs.append(d.cpu())
    db_desc = torch.cat(descs, dim=0).numpy()

    out = {}
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                q = model(x)
                q = F.normalize(q, dim=-1).cpu().numpy()
            sim = (db_desc @ q.T).squeeze(-1)
            j = int(np.argmax(sim))
            meta = db[j]
            out[p.name] = {
                "path": str(p),
                "retrieved_db_index": j,
                "similarity": float(sim[j]),
                "pred_lat": meta["latitude"],
                "pred_lon": meta["longitude"],
                "matched_filename": meta["filename"],
            }
        except Exception as e:
            out[p.name] = {"path": str(p), "error": str(e)}

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
