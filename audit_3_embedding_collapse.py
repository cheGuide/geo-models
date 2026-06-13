"""
audit_3_embedding_collapse.py
Диагностика коллапса эмбеддингов Kopernik на Moscow TTK-10k.
Извлекаем эмбеддинги через ResNet50 backbone, затем:
- PCA rank collapse
- TwoNN Intrinsic Dimensionality
- Cosine similarity statistics
- Recall@K (embedding distance vs GPS distance)
- t-SNE визуализация
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import json

# ─── Haversine ────────────────────────────────────────────────────────────────
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ─── Извлечение эмбеддингов ───────────────────────────────────────────────────
def extract_embeddings(image_paths, device, batch_size=32):
    """ResNet50 без последнего fc-слоя -> 2048-мерный эмбеддинг"""
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Убираем финальный классификатор
    extractor = nn.Sequential(*list(backbone.children())[:-1])
    extractor.eval().to(device)

    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    print(f"Извлечение эмбеддингов из {len(image_paths)} изображений (batch={batch_size})...")
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start+batch_size]
            batch_tensors = []
            for p in batch_paths:
                try:
                    from PIL import Image
                    img = Image.open(p).convert('RGB')
                    batch_tensors.append(preprocess(img))
                except:
                    batch_tensors.append(torch.zeros(3, 224, 224))
            batch = torch.stack(batch_tensors).to(device)
            feats = extractor(batch).squeeze(-1).squeeze(-1)  # (B, 2048)
            embeddings.append(feats.cpu().numpy())
            if (start // batch_size + 1) % 10 == 0:
                print(f"  {start + len(batch_paths)}/{len(image_paths)}")
    return np.vstack(embeddings)

# ─── Метрики ──────────────────────────────────────────────────────────────────
def pca_rank_analysis(embeddings, threshold=0.99):
    print("PCA анализ...")
    pca = PCA()
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k99 = int(np.searchsorted(cumvar, threshold)) + 1
    d   = embeddings.shape[1]
    print(f"  d={d}, k для {threshold*100:.0f}% дисперсии = {k99}  ({k99/d:.1%} пространства)")
    return {
        'k99': k99, 'd': d, 'ratio': k99/d,
        'explained': pca.explained_variance_ratio_,
        'cumulative': cumvar,
        'components': pca.components_,
    }

def two_nn_intrinsic_dim(embeddings, sample=2000):
    print("TwoNN Intrinsic Dimensionality...")
    n = min(sample, len(embeddings))
    idx = np.random.choice(len(embeddings), n, replace=False)
    Z = embeddings[idx]
    nbrs = NearestNeighbors(n_neighbors=3).fit(Z)
    dists, _ = nbrs.kneighbors(Z)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    mask = r1 > 1e-10
    mu = r2[mask] / r1[mask]
    id_est = 1.0 / (np.log(mu).mean())
    mu_sorted = np.sort(mu)
    empirical_cdf = np.arange(1, len(mu_sorted)+1) / len(mu_sorted)
    print(f"  ID = {id_est:.1f}  (d={embeddings.shape[1]}, использует {id_est/embeddings.shape[1]:.1%})")
    return id_est, mu_sorted, empirical_cdf

def cosine_sim_stats(embeddings, sample=1500):
    print("Cosine similarity analysis...")
    n = min(sample, len(embeddings))
    idx = np.random.choice(len(embeddings), n, replace=False)
    Z = embeddings[idx]
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    sim = Z @ Z.T
    upper = sim[np.triu_indices(n, k=1)]
    stats = {
        'mean':           float(upper.mean()),
        'std':            float(upper.std()),
        'p5':             float(np.percentile(upper, 5)),
        'p50':            float(np.percentile(upper, 50)),
        'p95':            float(np.percentile(upper, 95)),
        'pct_above_0_9':  float((upper > 0.9).mean()),
        'pct_above_0_5':  float((upper > 0.5).mean()),
        'values':         upper,
    }
    print(f"  Mean sim={stats['mean']:.3f}, P50={stats['p50']:.3f}, "
          f">0.9: {stats['pct_above_0_9']:.1%}, >0.5: {stats['pct_above_0_5']:.1%}")
    return stats

def recall_at_k_gps(embeddings, lats, lons, k_list=[1, 5, 10]):
    """Recall@K: насколько embedding-соседи совпадают с GPS-соседями?"""
    print("Recall@K (embedding neighbors vs GPS neighbors)...")
    Z = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    emb_sim = Z @ Z.T  # (N, N)

    gps_pts = np.column_stack([lats * 111000, lons * 111000 * np.cos(np.radians(lats.mean()))])
    gps_dists = squareform(pdist(gps_pts, metric='euclidean'))  # метры

    n = len(embeddings)
    results = {}
    for k in k_list:
        hits = 0.0
        for i in range(n):
            emb_s = emb_sim[i].copy(); emb_s[i] = -np.inf
            emb_topk = set(np.argsort(emb_s)[-k:])
            gps_d = gps_dists[i].copy(); gps_d[i] = np.inf
            gps_topk = set(np.argsort(gps_d)[:k])
            hits += len(emb_topk & gps_topk) / k
        results[k] = hits / n
        print(f"  Recall@{k} = {results[k]:.3f}")
    return results

# ─── Визуализация ─────────────────────────────────────────────────────────────
def plot_audit3(embeddings, lats, lons, pca_stats, id_est, mu_sorted, ecdf,
                sim_stats, recall, kopernik_errors):

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Audit #3: Embedding Collapse — ResNet50 backbone @ Moscow TTK-10k\n'
                 f'd={embeddings.shape[1]}, N={len(embeddings)} images',
                 fontsize=13, fontweight='bold')

    dist_from_center = np.sqrt((lats - lats.mean())**2 + (lons - lons.mean())**2) * 111000

    # ── 1. PCA объяснённая дисперсия ──────────────────────────────────────
    ax = axes[0, 0]
    n_show = min(100, len(pca_stats['explained']))
    ax.bar(range(1, n_show+1), pca_stats['explained'][:n_show],
           color='steelblue', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(1, len(pca_stats['cumulative'])+1),
             pca_stats['cumulative'], 'r-', linewidth=2)
    ax2.axhline(0.99, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(pca_stats['k99'], color='orange', linestyle='--', linewidth=2,
                label=f"k={pca_stats['k99']} (99%)")
    ax.set_xlabel('PC')
    ax.set_ylabel('Дисперсия')
    ax2.set_ylabel('Накопленная дисперсия')
    verdict_pca = "КОЛЛАПС" if pca_stats['ratio'] < 0.2 else "OK"
    ax.set_title(f"PCA Rank: {pca_stats['k99']}/{pca_stats['d']} = "
                 f"{pca_stats['ratio']:.1%} пространства\n[{verdict_pca}]",
                 color='red' if verdict_pca=='КОЛЛАПС' else 'black')
    ax2.legend(loc='lower right', fontsize=9)

    # ── 2. TwoNN ID ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(mu_sorted, 1 - ecdf, 'b-', linewidth=2, label='Empirical')
    theoretical = mu_sorted ** (-id_est)
    ax.plot(mu_sorted, theoretical, 'r--', linewidth=2, label=f'TwoNN ID={id_est:.1f}')
    ax.set_xlabel('mu = r2/r1')
    ax.set_ylabel('P(mu > x)')
    ax.set_xscale('log')
    ax.set_title(f'Intrinsic Dimensionality\nID={id_est:.1f} / d={embeddings.shape[1]} '
                 f'({id_est/embeddings.shape[1]:.1%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 3. Cosine similarities distribution ───────────────────────────────
    ax = axes[0, 2]
    vals = sim_stats['values']
    ax.hist(vals, bins=60, color='coral', edgecolor='black', alpha=0.7, density=True)
    ax.axvline(sim_stats['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean={sim_stats['mean']:.3f}")
    ax.axvline(0.0, color='green', linestyle=':', label='Ideal=0')
    high_mask = vals > 0.9
    if high_mask.any():
        ax.axvspan(0.9, 1.0, alpha=0.15, color='red',
                   label=f">{0.9}: {sim_stats['pct_above_0_9']:.1%}")
    ax.set_xlabel('Cosine similarity (off-diagonal)')
    ax.set_ylabel('Density')
    ax.set_title(f"Off-diagonal Cosine Similarities\nP50={sim_stats['p50']:.3f}, "
                 f">0.9: {sim_stats['pct_above_0_9']:.1%}")
    ax.legend(fontsize=8)

    # ── 4. t-SNE ────────────────────────────────────────────────────────
    ax = axes[1, 0]
    n_tsne = min(800, len(embeddings))
    idx_t = np.random.choice(len(embeddings), n_tsne, replace=False)
    print(f"t-SNE на {n_tsne} точках...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    Z2d = tsne.fit_transform(embeddings[idx_t])
    sc = ax.scatter(Z2d[:, 0], Z2d[:, 1],
                    c=dist_from_center[idx_t]/1000,
                    cmap='RdYlGn_r', s=5, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='GPS dist from center (km)')
    ax.set_title('t-SNE (раскраска = GPS расстояние от центра ТТК)\n'
                 'Хорошо: GPS-близкие = embedding-близкие')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    # ── 5. Embedding dist vs GPS dist scatter ────────────────────────────
    ax = axes[1, 1]
    n_scat = min(400, len(embeddings))
    idx_s = np.random.choice(len(embeddings), n_scat, replace=False)
    Z_s = embeddings[idx_s]
    Z_sn = Z_s / (np.linalg.norm(Z_s, axis=1, keepdims=True) + 1e-8)
    emb_d = (1 - Z_sn @ Z_sn.T)

    gps_s = np.column_stack([lats[idx_s], lons[idx_s]])
    gps_d = squareform(pdist(gps_s)) * 111000

    upper = np.triu_indices(n_scat, k=1)
    sample_mask = np.random.choice(len(upper[0]), min(5000, len(upper[0])), replace=False)
    ax.scatter(gps_d[upper][sample_mask],
               emb_d[upper][sample_mask],
               alpha=0.1, s=2, color='steelblue')
    corr = np.corrcoef(gps_d[upper], emb_d[upper])[0, 1]
    ax.set_xlabel('GPS distance (m)')
    ax.set_ylabel('Embedding cosine distance')
    ax.set_title(f'GPS dist vs Embedding dist\nCorr={corr:.3f} (1.0=ideal, ~0=collapse)')

    # ── 6. Recall@K ──────────────────────────────────────────────────────
    ax = axes[1, 2]
    ks = list(recall.keys())
    vs = list(recall.values())
    bar_colors = ['#d62728' if v < 0.25 else '#ff7f0e' if v < 0.5 else '#2ca02c'
                  for v in vs]
    bars = ax.bar([str(k) for k in ks], vs, color=bar_colors, alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, vs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01, f'{val:.2%}',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('K')
    ax.set_ylabel('Recall@K')
    ax.set_title('Recall@K: embedding neighbors == GPS neighbors?\n'
                 '(100% = идеально, <25% = коллапс)')
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.4)

    plt.tight_layout()
    out = 'audit_3_embedding_collapse.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Сохранено: {out}")
    plt.close()

    # ── Консольный диагноз ────────────────────────────────────────────────
    d = embeddings.shape[1]
    print("\n" + "="*65)
    print("ДИАГНОЗ: EMBEDDING COLLAPSE")
    print("="*65)
    checks = [
        ("Rank collapse",      f"k99={pca_stats['k99']}/{d} ({pca_stats['ratio']:.1%})",
         pca_stats['ratio'] < 0.25),
        ("Intrinsic dim",      f"ID={id_est:.1f}/{d} ({id_est/d:.1%})",
         id_est/d < 0.15),
        ("Cosine sim mean",    f"mean={sim_stats['mean']:.3f}",
         sim_stats['mean'] > 0.5),
        ("Sim >0.9",           f"{sim_stats['pct_above_0_9']:.1%} пар",
         sim_stats['pct_above_0_9'] > 0.1),
        ("Recall@1",           f"{recall.get(1, 0):.2%}",
         recall.get(1, 0) < 0.25),
    ]
    n_bad = 0
    for name, val, is_bad in checks:
        mark = "[COLLAPSE]" if is_bad else "[OK]"
        print(f"  {mark:12s}  {name:22s}  {val}")
        if is_bad: n_bad += 1
    print("-"*65)
    if n_bad >= 4:
        print("  ВЕРДИКТ: КРИТИЧЕСКИЙ EMBEDDING COLLAPSE")
    elif n_bad >= 2:
        print("  ВЕРДИКТ: ЧАСТИЧНЫЙ КОЛЛАПС — модель слабо дискриминативна")
    else:
        print("  ВЕРДИКТ: Коллапс не критичен")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import random
    from PIL import Image as PILImage

    random.seed(42); np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Загружаем метаданные датасета для GPS координат
    with open('ttk_10k_full/dataset_metadata.json') as f:
        meta = json.load(f)
    images_meta = meta['images']
    print(f"Метаданных: {len(images_meta)} записей")

    images_dir = Path('ttk_10k_full/images')
    N = 600  # выборка для анализа (баланс скорость/репрезентативность)
    step = max(1, len(images_meta) // N)
    selected = images_meta[::step][:N]

    paths = [images_dir / m['filename'] for m in selected]
    lats  = np.array([m['latitude']  for m in selected])
    lons  = np.array([m['longitude'] for m in selected])

    # Фильтруем несуществующие файлы
    valid = [(p, la, lo) for p, la, lo in zip(paths, lats, lons) if p.exists()]
    paths = [v[0] for v in valid]
    lats  = np.array([v[1] for v in valid])
    lons  = np.array([v[2] for v in valid])
    print(f"Валидных изображений: {len(paths)}")

    # Загружаем реальные ошибки Kopernik для контекста
    df_err = pd.read_csv('kopernik/kopernik_raw_errors.csv')
    kopernik_errors = df_err['error_m'].values

    embeddings = extract_embeddings(paths, device=device, batch_size=32)
    print(f"Эмбеддинги: {embeddings.shape}")

    pca_stats = pca_rank_analysis(embeddings)
    id_est, mu_sorted, ecdf = two_nn_intrinsic_dim(embeddings)
    sim_stats = cosine_sim_stats(embeddings)
    recall = recall_at_k_gps(embeddings, lats, lons, k_list=[1, 5, 10])

    plot_audit3(embeddings, lats, lons, pca_stats, id_est, mu_sorted, ecdf,
                sim_stats, recall, kopernik_errors)
    print("[Audit #3 завершён]")
