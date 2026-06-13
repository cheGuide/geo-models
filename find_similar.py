"""Find most similar image in ttk_10k_full using color histogram + pHash."""
import sys
import os
import glob
import numpy as np
from PIL import Image

QUERY_PATH = sys.argv[1] if len(sys.argv) > 1 else None
DATASET_DIR = r"C:\Users\q\Work\dipl\ttk_10k_full\images"
TOP_K = 5


def phash(img, hash_size=16):
    img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.array(img, dtype=float)
    mean = arr.mean()
    return (arr > mean).flatten()


def color_hist(img, bins=32):
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img)
    h = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256))
        h.append(hist / hist.sum())
    return np.concatenate(h)


def score(q_ph, q_ch, ph, ch):
    hamming = (q_ph != ph).mean()           # lower = more similar
    hist_dist = np.abs(q_ch - ch).sum()     # lower = more similar
    return hamming * 0.5 + hist_dist * 0.5


if __name__ == "__main__":
    if not QUERY_PATH or not os.path.exists(QUERY_PATH):
        print("Usage: find_similar.py <query_image_path>")
        sys.exit(1)

    query = Image.open(QUERY_PATH)
    q_ph = phash(query)
    q_ch = color_hist(query)

    files = glob.glob(os.path.join(DATASET_DIR, "*.jpg"))
    print(f"Scanning {len(files)} images...", flush=True)

    results = []
    for i, fp in enumerate(files):
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)
        try:
            img = Image.open(fp)
            ph = phash(img)
            ch = color_hist(img)
            s = score(q_ph, q_ch, ph, ch)
            results.append((s, fp))
        except Exception:
            pass

    results.sort()
    print(f"\nTop {TOP_K} most similar:")
    for rank, (s, fp) in enumerate(results[:TOP_K], 1):
        print(f"  {rank}. score={s:.4f}  {os.path.basename(fp)}")
