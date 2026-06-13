"""
audit_4_evaluation_protocols.py
Сравнение протоколов оценки на реальных данных Kopernik.
Показываем: насколько каждый "мягкий" протокол искусственно снижает ошибку
по сравнению с честным Raw Per-Image Mean.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ─── Haversine ────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ─── Загрузка данных ──────────────────────────────────────────────────────────
def load_data(csv_path='kopernik/kopernik_raw_errors.csv'):
    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} записей")
    print(f"Уникальных локаций: {df['location_id'].nunique()}")
    print(f"Raw Mean error: {df['error_m'].mean():.0f} m")
    return df

# ─── Симуляция топ-K гипотез (Kopernik не пишет их, симулируем реалистично) ──
def simulate_top_k_hypotheses(df, k=5, noise_scale=0.015, seed=42):
    """
    Для каждого изображения генерируем K гипотез:
    - гипотеза 1: реальное предсказание Kopernik
    - гипотезы 2..K: случайные смещения вокруг предсказания
    В реальности модель выдаёт softmax по ячейкам -> топ-K ячеек.
    """
    rng = np.random.default_rng(seed)
    hypotheses = []
    for _, row in df.iterrows():
        pred_lat, pred_lon = row['lat_pred'], row['lon_pred']
        row_hyps = [(pred_lat, pred_lon)]
        for _ in range(k - 1):
            h_lat = pred_lat + rng.normal(0, noise_scale)
            h_lon = pred_lon + rng.normal(0, noise_scale * 1.5)
            row_hyps.append((h_lat, h_lon))
        hypotheses.append(row_hyps)
    return hypotheses

# ─── Протоколы оценки ─────────────────────────────────────────────────────────

def protocol_raw_per_image(df):
    """Честный протокол — ваш."""
    errors = df['error_m'].values
    return {
        'name': 'Raw Per-Image Mean (честный)',
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'errors': errors,
    }

def protocol_location_level(df):
    """Усредняем ошибки внутри локации, затем усредняем по локациям."""
    loc_groups = df.groupby('location_id')['error_m'].mean()
    return {
        'name': 'Location-Level Mean',
        'mean': float(loc_groups.mean()),
        'median': float(loc_groups.median()),
        'errors': loc_groups.values,
        'n_locations': len(loc_groups),
    }

def protocol_best_of_n(df, hypotheses, n=3):
    """Best-of-N: oracle выбирает лучшую гипотезу из N."""
    errors = []
    for idx, (_, row) in enumerate(df.iterrows()):
        hyps = hypotheses[idx][:n]
        candidate_errors = [haversine(h[0], h[1], row['lat_true'], row['lon_true'])
                           for h in hyps]
        errors.append(min(candidate_errors))
    return {
        'name': f'Best-of-{n} (oracle)',
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'errors': np.array(errors),
    }

def protocol_temporal_aggregation(df):
    """
    Усредняем предсказания всех кадров одной локации -> один предикт на локацию.
    Работает только при multiple frames per location.
    """
    loc_groups = df.groupby('location_id').agg(
        lat_pred_mean=('lat_pred', 'mean'),
        lon_pred_mean=('lon_pred', 'mean'),
        lat_true=('lat_true', 'first'),
        lon_true=('lon_true', 'first'),
    ).reset_index()
    errors = [haversine(row.lat_pred_mean, row.lon_pred_mean,
                        row.lat_true, row.lon_true)
              for _, row in loc_groups.iterrows()]
    return {
        'name': 'Temporal Aggregation (mean preds per loc)',
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'errors': np.array(errors),
    }

def protocol_median_prediction(df):
    """Медиана предсказаний по локации."""
    loc_groups = df.groupby('location_id').agg(
        lat_pred_med=('lat_pred', 'median'),
        lon_pred_med=('lon_pred', 'median'),
        lat_true=('lat_true', 'first'),
        lon_true=('lon_true', 'first'),
    ).reset_index()
    errors = [haversine(row.lat_pred_med, row.lon_pred_med,
                        row.lat_true, row.lon_true)
              for _, row in loc_groups.iterrows()]
    return {
        'name': 'Median Prediction per Location',
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'errors': np.array(errors),
    }

def compute_recall_at_distance(df, thresholds=[25, 100, 250, 500, 1000, 2000, 5000]):
    errors = df['error_m'].values
    return {thr: float((errors < thr).mean()) for thr in thresholds}

def compute_recall_best_of_n(df, hypotheses, threshold=1000, n_list=[1, 3, 5]):
    results = {}
    for n in n_list:
        hits = 0
        for idx, (_, row) in enumerate(df.iterrows()):
            for h in hypotheses[idx][:n]:
                if haversine(h[0], h[1], row['lat_true'], row['lon_true']) < threshold:
                    hits += 1
                    break
        results[n] = hits / len(df)
    return results

# ─── Визуализация ─────────────────────────────────────────────────────────────
def plot_audit4(df, protocols, recall_raw, recall_bon):
    raw_mean = protocols['raw']['mean']
    lats_gt = df['lat_true'].values
    lons_gt = df['lon_true'].values
    lats_pr = df['lat_pred'].values
    lons_pr = df['lon_pred'].values

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Audit #4: Evaluation Protocol Manipulation\n'
                 f'Kopernik @ Moscow TTK-10k — Raw Mean = {raw_mean:.0f} m',
                 fontsize=13, fontweight='bold')

    protocol_list = [
        ('raw',      protocols['raw'],      '#d62728'),
        ('loc',      protocols['loc'],      '#ff7f0e'),
        ('best2',    protocols['best2'],    '#9467bd'),
        ('best3',    protocols['best3'],    '#8c564b'),
        ('best5',    protocols['best5'],    '#e377c2'),
        ('temporal', protocols['temporal'], '#17becf'),
        ('median',   protocols['median'],   '#bcbd22'),
    ]

    # ── 1. Bar: mean error по протоколам ────────────────────────────────
    ax = axes[0, 0]
    names  = [p[1]['name'].replace(' (честный)', '\n(ЧЕСТНЫЙ)').replace(' (oracle)', '\n(oracle)') for p in protocol_list]
    means  = [p[1]['mean'] for p in protocol_list]
    colors = [p[2] for p in protocol_list]
    bars = ax.bar(range(len(names)), [m/1000 for m in means],
                  color=colors, alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.0f}m', ha='center', fontsize=8, fontweight='bold')
    ax.axhline(raw_mean/1000, color='red', linestyle='--', linewidth=2,
               label=f'Raw = {raw_mean:.0f}m')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=25, ha='right')
    ax.set_ylabel('Mean Error (km)')
    ax.set_title('Mean Error по протоколам')
    ax.legend(fontsize=9)

    # ── 2. Bar: % "искусственного улучшения" ────────────────────────────
    ax = axes[0, 1]
    improvements = [(raw_mean - p[1]['mean']) / raw_mean * 100 for p in protocol_list]
    bar_cols = ['gray'] + ['#2ca02c' for _ in protocol_list[1:]]
    bars = ax.bar(range(len(names)), improvements, color=bar_cols, alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, improvements):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'+{val:.0f}%', ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=25, ha='right')
    ax.set_ylabel('Artifact improvement (%)')
    ax.set_title('Искусственный прирост метрики\nот выбора "мягкого" протокола')
    ax.axhline(0, color='black', linewidth=0.8)

    # ── 3. CDF всех протоколов ──────────────────────────────────────────
    ax = axes[0, 2]
    for key, proto, color in protocol_list:
        err = proto['errors']
        s = np.sort(err)
        cdf = np.arange(len(s)) / len(s)
        lw = 2.5 if key == 'raw' else 1.2
        ax.plot(s/1000, cdf, color=color, linewidth=lw,
                label=proto['name'].split('(')[0].strip())
    ax.axvline(raw_mean/1000, color='red', linestyle='--', alpha=0.6)
    ax.set_xlabel('Error (km)')
    ax.set_ylabel('CDF')
    ax.set_title('CDF ошибок для всех протоколов')
    ax.legend(fontsize=7)
    ax.set_xlim(0, 12)
    ax.grid(True, alpha=0.3)

    # ── 4. Recall@D (raw single prediction) ─────────────────────────────
    ax = axes[1, 0]
    thresholds = sorted(recall_raw.keys())
    recall_vals = [recall_raw[t] for t in thresholds]
    ax.semilogx(thresholds, recall_vals, 'b-o', linewidth=2, markersize=8,
                label='Recall@D (single pred)')
    for t, r in zip(thresholds, recall_vals):
        ax.annotate(f'{r:.1%}', (t, r), textcoords='offset points',
                    xytext=(4, 6), fontsize=8)
    # Recall Best-of-5 @ 1km
    r_bon = recall_bon.get(5, 0)
    ax.scatter([1000], [r_bon], s=200, c='red', zorder=10,
               label=f'Best-of-5 @1km = {r_bon:.1%}')
    ax.set_xlabel('Distance threshold (m)')
    ax.set_ylabel('Recall')
    ax.set_title('Recall@Distance — Single prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 5. Scatter: GT vs Predictions на карте ──────────────────────────
    ax = axes[1, 1]
    errors = df['error_m'].values
    ax.scatter(lons_gt, lats_gt, c='green', s=8, alpha=0.4, label='GT')
    ax.scatter(lons_pr, lats_pr, c='red',   s=8, alpha=0.4, label='Kopernik')
    # Линии для случайного сабсета
    idx_s = np.random.choice(len(df), 60, replace=False)
    for i in idx_s:
        ax.plot([lons_gt[i], lons_pr[i]], [lats_gt[i], lats_pr[i]],
                'gray', alpha=0.2, linewidth=0.6)
    ax.scatter(lons_pr.mean(), lats_pr.mean(), c='blue', s=200, marker='*',
               zorder=10, label='Pred centroid')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('GT (green) vs Predictions (red)\nЗвезда = центроид предсказаний')
    ax.legend(fontsize=8)

    # ── 6. Итоговая таблица ──────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis('off')
    table_data = []
    for key, proto, _ in protocol_list:
        mean_val = proto['mean']
        improv = (raw_mean - mean_val) / raw_mean * 100
        table_data.append([
            proto['name'].replace(' (честный)', '*').replace(' (oracle)', ''),
            f"{mean_val:.0f} m",
            f"+{improv:.0f}%" if improv > 0 else "baseline"
        ])

    tbl = ax.table(
        cellText=table_data,
        colLabels=['Протокол', 'Mean Error', 'Artifact'],
        cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(3):
        tbl[0, j].set_facecolor('#2c7bb6')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    tbl[1, 0].set_facecolor('#d7191c')
    tbl[1, 0].set_text_props(color='white', fontweight='bold')
    tbl[1, 1].set_facecolor('#d7191c')
    tbl[1, 1].set_text_props(color='white', fontweight='bold')
    tbl[1, 2].set_facecolor('#d7191c')
    tbl[1, 2].set_text_props(color='white', fontweight='bold')
    ax.set_title('Сводная таблица (*= честный протокол)', fontsize=10, pad=5)

    plt.tight_layout()
    out = 'audit_4_evaluation_protocols.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Сохранено: {out}")
    plt.close()

    # ── Консольный итог ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("AUDIT #4: ПРОТОКОЛЫ ОЦЕНКИ — ИТОГ")
    print("="*70)
    for key, proto, _ in protocol_list:
        m = proto['mean']
        art = (raw_mean - m) / raw_mean * 100
        marker = " <-- BASELINE" if key == 'raw' else f"  (artifact: +{art:.0f}%)"
        print(f"  {proto['name']:45s}: {m:6.0f} m{marker}")

    print(f"\n  Recall@D (single pred):")
    for t, r in recall_raw.items():
        print(f"    @{t:5d}m  {r:.1%}")
    print(f"\n  Recall@1km Best-of-N:")
    for n, r in recall_bon.items():
        print(f"    Best-of-{n}: {r:.1%}  (x{r/max(recall_raw[1000],0.001):.1f} vs single)")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)

    df = load_data('kopernik/kopernik_raw_errors.csv')
    hypotheses = simulate_top_k_hypotheses(df, k=5)

    protocols = {
        'raw':      protocol_raw_per_image(df),
        'loc':      protocol_location_level(df),
        'best2':    protocol_best_of_n(df, hypotheses, n=2),
        'best3':    protocol_best_of_n(df, hypotheses, n=3),
        'best5':    protocol_best_of_n(df, hypotheses, n=5),
        'temporal': protocol_temporal_aggregation(df),
        'median':   protocol_median_prediction(df),
    }

    recall_raw = compute_recall_at_distance(df)
    recall_bon = compute_recall_best_of_n(df, hypotheses, threshold=1000, n_list=[1, 3, 5])

    plot_audit4(df, protocols, recall_raw, recall_bon)
    print("[Audit #4 завершён]")
