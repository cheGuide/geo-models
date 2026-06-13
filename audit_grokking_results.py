"""
audit_grokking_results.py
Полный анализ результатов 1000-эпохного grokking эксперимента на Kopernik.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

df = pd.read_csv('kopernik/grokking_metrics.csv')
df_w = df[df.weight_l2_norm.notna()].copy()

fig = plt.figure(figsize=(22, 16))
fig.suptitle(
    'Kopernik Grokking Experiment — 1000 Epochs — Moscow TTK-10k\n'
    'ВЕРДИКТ: Фазовый переход (grokking) НЕ ОБНАРУЖЕН',
    fontsize=14, fontweight='bold', color='#c0392b'
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 1. Train vs Val Loss ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(df.epoch, df.train_loss, color='#2980b9', linewidth=1.2, alpha=0.8, label='Train Loss')
ax1.plot(df.epoch, df.val_loss,   color='#e74c3c', linewidth=1.5, label='Val Loss')
ax1.axvspan(700, 1000, alpha=0.07, color='red', label='Опасная зона (нестабильность ep 997)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (CrossEntropy)')
ax1.set_title('Train Loss vs Val Loss\nГенерализационный разрыв сохраняется на протяжении всех 1000 эпох')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── 2. Val Accuracy ───────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(df.epoch, df.val_acc * 100, color='#27ae60', linewidth=1.5)
best_acc_ep = int(df.loc[df.val_acc.idxmax(), 'epoch'])
best_acc    = df.val_acc.max() * 100
ax2.axvline(best_acc_ep, color='orange', linestyle='--', alpha=0.8,
            label=f'Best: {best_acc:.2f}% @ ep{best_acc_ep}')
ax2.axhline(best_acc, color='orange', linestyle=':', alpha=0.6)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Val Accuracy (%)')
ax2.set_title(f'Val Accuracy\nMax={best_acc:.2f}% (ep {best_acc_ep})\n← плато, нет скачка')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── 3. Val Raw Mean (главная метрика) ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(df.epoch, df.val_raw_mean_m / 1000, color='#8e44ad', linewidth=1.5, label='Val Mean Error')
# rolling avg
roll = df['val_raw_mean_m'].rolling(20, center=True).mean()
ax3.plot(df.epoch, roll / 1000, color='#2c3e50', linewidth=2.5, label='Rolling mean (20ep)')
best_mean_ep = int(df.loc[df.val_raw_mean_m.idxmin(), 'epoch'])
best_mean    = df.val_raw_mean_m.min()
ax3.axhline(best_mean/1000, color='green', linestyle='--', linewidth=1.5,
            label=f'Best: {best_mean:.0f}m @ ep{best_mean_ep}')
ax3.axhline(3.545, color='red', linestyle=':', linewidth=2,
            label='Baseline Kopernik raw = 3545m')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Val Mean Error (km)')
ax3.set_title('Val Raw Mean Haversine Error\nОсновная целевая метрика')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── 4. Train Accuracy ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(df.epoch, df.train_acc * 100, color='#2980b9', linewidth=1.2)
ax4.axhline(99, color='gray', linestyle='--', alpha=0.5, label='99%')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Train Accuracy (%)')
ax4.set_title('Train Accuracy\n~99% начиная с ep 100\n(memorization plateau)')
ax4.set_ylim(0, 105)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# ── 5. Weight L2 Norm ─────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(df_w.epoch, df_w.weight_l2_norm, color='#e67e22', linewidth=1.8, marker='o', markersize=3)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Weight L2 Norm')
ax5.set_title('Weight L2 Norm\nWD=5e-3 давит веса, но плато ~31–34')
ax5.grid(True, alpha=0.3)
# Аннотация: не достигает нуля
ax5.annotate('Плато ~31-34\n(не коллапсирует к 0)', xy=(900, 31.5),
             xytext=(600, 29), fontsize=8,
             arrowprops=dict(arrowstyle='->', color='gray'))

# ── 6. Feature Sparsity ───────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(df_w.epoch, df_w.feature_sparsity, color='#16a085', linewidth=1.8, marker='o', markersize=3)
ax6.axhline(0.57, color='red', linestyle='--', alpha=0.6, label='Плато ~0.57')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Feature Sparsity')
ax6.set_title('Feature Sparsity\n(доля нулей после ReLU)\nПлато — нет роста разреженности')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# ── 7. GNS (Grad Noise Scale) ─────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
gns_clean = df_w[df_w.grad_noise_scale.notna() & np.isfinite(df_w.grad_noise_scale)]
ax7.scatter(gns_clean.epoch, gns_clean.grad_noise_scale,
            s=12, alpha=0.6, color='#c0392b')
ax7.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='GNS=1 (критический порог)')
ax7.set_xlabel('Epoch')
ax7.set_ylabel('GNS')
ax7.set_ylim(0, 10)
ax7.set_title('Grad Noise Scale\nGNS~1 = нет сигнала к обобщению')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

plt.savefig('audit_grokking_results.png', dpi=150, bbox_inches='tight')
print("Сохранено: audit_grokking_results.png")
plt.close()

# ── Консольный разбор ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ 1000-ЭПОХНОГО GROKKING ЭКСПЕРИМЕНТА")
print("="*70)
print(f"  Best val_raw_mean : {df.val_raw_mean_m.min():.0f} м  (ep {int(df.loc[df.val_raw_mean_m.idxmin(),'epoch'])})")
print(f"  Best val_acc      : {df.val_acc.max():.4f}  ({df.val_acc.max()*100:.2f}%)  (ep {int(df.loc[df.val_acc.idxmax(),'epoch'])})")
print(f"  Baseline Kopernik : 3545 м")
print(f"  Улучшение vs base : {(3545 - df.val_raw_mean_m.min()) / 3545 * 100:+.1f}%")
print()

# Фазовый переход?
rolling = df['val_acc'].rolling(30).mean()
delta   = rolling.diff()
max_jump = delta.max()
max_jump_ep = int(df.loc[delta.idxmax(), 'epoch']) if not np.isnan(delta.max()) else -1
print(f"  Макс. скачок rolling val_acc (30ep): {max_jump:.5f} @ ep{max_jump_ep}")
print(f"  Типичный grokking-скачок: >0.3–0.5 за 10–30 эпох")
print()

# Weight norm динамика
wn_start = df_w.weight_l2_norm.iloc[0]
wn_end   = df_w.weight_l2_norm.iloc[-1]
print(f"  Weight L2 norm: {wn_start:.1f} (ep1) → {wn_end:.1f} (ep1000)")
print(f"  Снижение: {(wn_start-wn_end)/wn_start*100:.1f}%  (недостаточно для grokking)")
print()

# Sparsity
sp_start = df_w.feature_sparsity.iloc[0]
sp_end   = df_w.feature_sparsity.iloc[-1]
print(f"  Feature sparsity: {sp_start:.3f} → {sp_end:.3f}  (нет роста = нет circuit formation)")
print()

print("─"*70)
print("ДИАГНОЗ:")
print()
print("  1. ПЛАТО МЕМОРИЗАЦИИ НЕ ПРОРВАНО")
print("     train_acc ~99% с ep 100, val_acc ~5% до самого конца.")
print("     Разрыв generalization gap не сократился за 1000 эпох.")
print()
print("  2. GROKKING НЕ ПРОИЗОШЁЛ")
print("     Нет характерного скачка val_acc/val_mean при стабильном train_acc.")
print("     Скрипт также не зафиксировал transition_detected.")
print()
print("  3. WEIGHT DECAY РАБОТАЕТ, НО НЕДОСТАТОЧНО")
print(f"     Norm: {wn_start:.0f} → {wn_end:.0f}  (−{(wn_start-wn_end)/wn_start*100:.0f}%)")
print("     Для grokking на алгоритмических задачах норма должна упасть")
print("     на 70–90%. Здесь — только 61%. WD=5e-3 слишком слаб для")
print("     14k изображений или нужно больше эпох (>3000).")
print()
print("  4. FEATURE SPARSITY В ПЛАТО")
print(f"     {sp_start:.3f} → {sp_end:.3f}: нет роста. Grokking обычно сопровождается")
print("     резким ростом sparsity (нейронные цепи 'прореживаются').")
print("     Здесь цепи так и не сформировались.")
print()
print("  5. НЕСТАБИЛЬНОСТЬ НА EP 997–999")
print("     tr_loss скачнул с 0.25 → 0.93, tr_acc упал до 91%.")
print("     Признак слишком большого LR на финальной стадии.")
print()
print("─"*70)
print("РЕКОМЕНДАЦИИ ДЛЯ СЛЕДУЮЩЕГО ЭТАПА:")
print("  A) Увеличить WEIGHT_DECAY до 1e-2 или 2e-2")
print("  B) Продолжить до 3000–5000 эпох (grokking на vision задачах — поздний)")
print("  C) LR warmup + cosine decay начиная с ep 500")
print("  D) Label smoothing 0.1 для ослабления меморизации")
print("  E) Рассмотреть метрическое обучение вместо классификации")
