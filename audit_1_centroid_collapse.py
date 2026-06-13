"""
audit_1_centroid_collapse.py
Доказываем: MSE-регрессия на плотной выборке TTK -> предсказание центроида
Работает на реальных данных из kopernik/kopernik_raw_errors.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Haversine ────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def haversine_batch(pred_lat, pred_lon, gt_lats, gt_lons):
    return np.array([haversine(pred_lat, pred_lon, g, o)
                     for g, o in zip(gt_lats, gt_lons)])

# ─── Загрузка реальных данных ─────────────────────────────────────────────────
def load_real_data(csv_path='kopernik/kopernik_raw_errors.csv'):
    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} записей из {csv_path}")
    print(f"Колонки: {list(df.columns)}")
    return df

# ─── MSE-оптимальное предсказание (аналитически = центроид) ──────────────────
def mse_optimal_prediction(lats, lons, label="GT"):
    pred_lat = lats.mean()
    pred_lon = lons.mean()
    errors = haversine_batch(pred_lat, pred_lon, lats, lons)
    print(f"\n{'='*60}")
    print(f"MSE-ОПТИМАЛЬНОЕ ПРЕДСКАЗАНИЕ ({label})")
    print(f"{'='*60}")
    print(f"  Центроид:        ({pred_lat:.6f}, {pred_lon:.6f})")
    print(f"  Mean error:      {np.mean(errors):.0f} м")
    print(f"  Median error:    {np.median(errors):.0f} м")
    print(f"  Std:             {np.std(errors):.0f} м")
    print(f"  P25:             {np.percentile(errors, 25):.0f} м")
    print(f"  P75:             {np.percentile(errors, 75):.0f} м")
    print(f"  P90:             {np.percentile(errors, 90):.0f} м")
    return pred_lat, pred_lon, errors

# ─── Анализ фактических предсказаний Kopernik ────────────────────────────────
def analyze_kopernik_predictions(df):
    pred_centroid_lat = df['lat_pred'].mean()
    pred_centroid_lon = df['lon_pred'].mean()
    gt_centroid_lat   = df['lat_true'].mean()
    gt_centroid_lon   = df['lon_true'].mean()

    dist_centroids = haversine(pred_centroid_lat, pred_centroid_lon,
                                gt_centroid_lat, gt_centroid_lon)

    print(f"\n{'='*60}")
    print("АНАЛИЗ ФАКТИЧЕСКИХ ПРЕДСКАЗАНИЙ KOPERNIK")
    print(f"{'='*60}")
    print(f"  GT центроид:     ({gt_centroid_lat:.6f}, {gt_centroid_lon:.6f})")
    print(f"  Pred центроид:   ({pred_centroid_lat:.6f}, {pred_centroid_lon:.6f})")
    print(f"  Расстояние GT<->Pred центроидов: {dist_centroids:.0f} м")
    print(f"  Разброс предсказаний (std lat):  {df['lat_pred'].std()*111000:.0f} м")
    print(f"  Разброс предсказаний (std lon):  {df['lon_pred'].std()*111000:.0f} м")
    print(f"  Разброс GT (std lat):            {df['lat_true'].std()*111000:.0f} м")
    print(f"  Разброс GT (std lon):            {df['lon_true'].std()*111000:.0f} м")

    # Коэффициент схлопывания: насколько предсказания «сжаты» по сравнению с GT
    collapse_lat = df['lat_pred'].std() / (df['lat_true'].std() + 1e-8)
    collapse_lon = df['lon_pred'].std() / (df['lon_true'].std() + 1e-8)
    print(f"\n  Collapse ratio (lat): {collapse_lat:.3f}  (1.0 = нет коллапса, 0 = полный)")
    print(f"  Collapse ratio (lon): {collapse_lon:.3f}  (1.0 = нет коллапса, 0 = полный)")

    return pred_centroid_lat, pred_centroid_lon, dist_centroids

# ─── Анализ гранулярности ячеек ───────────────────────────────────────────────
def compute_cell_errors(lats, lons, pred_lats, pred_lons, cell_sizes_km):
    results = {}
    MOSCOW_LAT = 55.7558
    for cell_km in cell_sizes_km:
        cell_lat = cell_km / 111.0
        cell_lon = cell_km / (111.0 * np.cos(np.radians(MOSCOW_LAT)))

        cell_i = np.floor(lats / cell_lat).astype(int)
        cell_j = np.floor(lons / cell_lon).astype(int)
        cell_ids = list(zip(cell_i, cell_j))
        unique_cells = set(cell_ids)

        errors = []
        for uid in unique_cells:
            mask = np.array([c == uid for c in cell_ids])
            centroid_lat = lats[mask].mean()
            centroid_lon = lons[mask].mean()
            cell_errors = haversine_batch(centroid_lat, centroid_lon,
                                          lats[mask], lons[mask])
            errors.extend(cell_errors.tolist())

        results[cell_km] = {
            'mean_error_m':   np.mean(errors),
            'median_error_m': np.median(errors),
            'n_cells':        len(unique_cells),
            'points_per_cell': len(lats) / len(unique_cells)
        }
    return results

# ─── Визуализация ─────────────────────────────────────────────────────────────
def plot_all(df, gt_centroid_pred_lat, gt_centroid_pred_lon,
             mse_errors, kopernik_errors, cell_results):

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Audit #1: Centroid Collapse — Kopernik @ Moscow TTK-10k\n'
                 f'N_test={len(df)}, Raw Mean={df["error_m"].mean():.0f} м',
                 fontsize=14, fontweight='bold')

    lats_gt = df['lat_true'].values
    lons_gt = df['lon_true'].values
    lats_pr = df['lat_pred'].values
    lons_pr = df['lon_pred'].values
    errors   = df['error_m'].values

    # ── Plot 1: Пространственное распределение ошибок ──────────────────────
    ax = axes[0, 0]
    sc = ax.scatter(lons_gt, lats_gt, c=errors/1000, cmap='RdYlGn_r',
                    alpha=0.6, s=15, vmin=0, vmax=10)
    plt.colorbar(sc, ax=ax, label='Ошибка (км)')
    ax.scatter(lons_pr.mean(), lats_pr.mean(), c='blue', s=300,
               marker='*', zorder=10, label=f'Pred-centroid')
    ax.scatter(lons_gt.mean(), lats_gt.mean(), c='black', s=200,
               marker='+', zorder=10, linewidths=3, label='GT-centroid')
    ax.set_xlabel('Долгота')
    ax.set_ylabel('Широта')
    ax.set_title('Пространственное распределение ошибок\n(★ = центроид предсказаний Kopernik)')
    ax.legend(fontsize=9)

    # ── Plot 2: GT vs Predictions — схлопывание ────────────────────────────
    ax = axes[0, 1]
    ax.scatter(lons_gt, lats_gt, c='green', alpha=0.3, s=8, label='Ground Truth')
    ax.scatter(lons_pr, lats_pr, c='red', alpha=0.3, s=8, label='Kopernik Predictions')
    # Соединяем несколько пар линиями
    idx_sample = np.random.choice(len(df), min(80, len(df)), replace=False)
    for i in idx_sample:
        ax.plot([lons_gt[i], lons_pr[i]], [lats_gt[i], lats_pr[i]],
                'gray', alpha=0.15, linewidth=0.7)
    ax.set_xlabel('Долгота')
    ax.set_ylabel('Широта')
    ax.set_title('GT (зелёный) vs Predictions (красный)\nЛинии = вектор ошибки')
    ax.legend(fontsize=9)

    # ── Plot 3: CDF реальных ошибок ────────────────────────────────────────
    ax = axes[0, 2]
    sorted_err = np.sort(errors)
    cdf = np.arange(len(sorted_err)) / len(sorted_err)
    ax.plot(sorted_err/1000, cdf, 'b-', linewidth=2.5, label='Kopernik Raw errors')
    ax.axvline(errors.mean()/1000, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {errors.mean():.0f} м')
    ax.axvline(np.median(errors)/1000, color='orange', linestyle='--', linewidth=1.5,
               label=f'Median = {np.median(errors):.0f} м')
    for thr in [500, 1000, 2000]:
        pct = (errors < thr).mean()*100
        ax.annotate(f'{pct:.0f}% < {thr}м',
                    xy=(thr/1000, pct/100), xytext=(thr/1000+0.5, pct/100-0.08),
                    fontsize=8, color='gray', arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlabel('Ошибка (км)')
    ax.set_ylabel('CDF')
    ax.set_title('Функция распределения ошибок (CDF)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(sorted_err/1000)*1.05)

    # ── Plot 4: Ошибка центроида vs гранулярность ячейки ──────────────────
    ax = axes[1, 0]
    sizes = sorted(cell_results.keys())
    means = [cell_results[s]['mean_error_m']/1000 for s in sizes]
    n_cells = [cell_results[s]['n_cells'] for s in sizes]
    ax2 = ax.twinx()
    ax.plot(sizes, means, 'b-o', linewidth=2, markersize=6, label='Mean ошибка центроида')
    ax2.bar(sizes, n_cells, alpha=0.25, color='orange', width=0.08, label='N ячеек')
    ax.axhline(y=errors.mean()/1000, color='red', linestyle='--', linewidth=2,
               label=f'Kopernik = {errors.mean()/1000:.2f} км')
    ax.set_xlabel('Размер ячейки (км)')
    ax.set_ylabel('Mean ошибка (км)', color='blue')
    ax2.set_ylabel('N ячеек', color='orange')
    ax.set_title('Ошибка центроида vs Гранулярность сетки\n'
                 '(при какой ячейке Kopernik "правильно" ошибается?)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xscale('log')

    # ── Plot 5: Гистограмма ошибок ─────────────────────────────────────────
    ax = axes[1, 1]
    ax.hist(errors/1000, bins=40, color='steelblue', edgecolor='black',
            alpha=0.7, label='Raw errors')
    ax.axvline(errors.mean()/1000, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {errors.mean():.0f} м')
    ax.axvline(mse_errors.mean()/1000, color='purple', linestyle=':', linewidth=2,
               label=f'MSE-centroid mean = {mse_errors.mean():.0f} м')
    ax.set_xlabel('Ошибка (км)')
    ax.set_ylabel('Частота')
    ax.set_title('Гистограмма ошибок Kopernik\nvs теоретический MSE-центроид')
    ax.legend(fontsize=9)

    # ── Plot 6: Scatter pred_lat vs true_lat ───────────────────────────────
    ax = axes[1, 2]
    ax.scatter(lats_gt, lats_pr, alpha=0.3, s=8, c=errors/1000,
               cmap='RdYlGn_r', vmin=0, vmax=10)
    # Идеальная линия
    mn = min(lats_gt.min(), lats_pr.min())
    mx = max(lats_gt.max(), lats_pr.max())
    ax.plot([mn, mx], [mn, mx], 'g--', linewidth=1.5, label='Ideal')
    ax.axhline(lats_pr.mean(), color='red', linestyle='--', alpha=0.7,
               label=f'Pred centroid lat={lats_pr.mean():.4f}')
    corr = np.corrcoef(lats_gt, lats_pr)[0,1]
    ax.set_xlabel('lat_true')
    ax.set_ylabel('lat_pred')
    ax.set_title(f'True lat vs Predicted lat\nCorr = {corr:.3f} (1.0 = идеал)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = 'audit_1_centroid_collapse.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nСохранено: {out_path}")
    plt.close()

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load_real_data('kopernik/kopernik_raw_errors.csv')

    lats_gt = df['lat_true'].values
    lons_gt = df['lon_true'].values
    lats_pr = df['lat_pred'].values
    lons_pr = df['lon_pred'].values
    kopernik_errors = df['error_m'].values

    # MSE-оптимальное предсказание (теория)
    mse_lat, mse_lon, mse_errors = mse_optimal_prediction(lats_gt, lons_gt, "GT-set")

    # Анализ реальных предсказаний
    analyze_kopernik_predictions(df)

    # Ячеечный анализ
    print("\nАнализ гранулярности ячеек:")
    cell_sizes = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.5, 7.0]
    cell_results = compute_cell_errors(lats_gt, lons_gt, lats_pr, lons_pr, cell_sizes)
    for cs, r in cell_results.items():
        marker = " <-- Kopernik level" if abs(r['mean_error_m'] - kopernik_errors.mean()) < 300 else ""
        print(f"  {cs:5.2f} km -> mean={r['mean_error_m']:6.0f}m  "
              f"n_cells={r['n_cells']:4d}  pts/cell={r['points_per_cell']:.1f}{marker}")

    plot_all(df, mse_lat, mse_lon, mse_errors, kopernik_errors, cell_results)
    print("\n[Audit #1 завершён]")
