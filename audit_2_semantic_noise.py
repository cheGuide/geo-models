"""
audit_2_semantic_noise.py
Диагностика: куда смотрит backbone — на шум (асфальт/небо) или на архитектуру?
GradCAM на ResNet backbone + анализ spatial variance изображений TTK.
Работает с torchvision ResNet (аналог backbone Kopernik).
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import cv2
from PIL import Image
from pathlib import Path

# ─── GradCAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, m, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x):
        self.model.zero_grad()
        out = self.model(x)
        score = out.max()
        score.backward()
        # GAP взвешивание
        w = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ─── Региональный анализ внимания ─────────────────────────────────────────────
def region_attention_scores(cam, H):
    """
    Делим кадр на 3 горизонтальные зоны и считаем долю activation в каждой.
    sky: верхние 30%, buildings: 25-65%, road: нижние 35%
    """
    sky_attn  = cam[:int(H*0.30)].mean()
    bld_attn  = cam[int(H*0.25):int(H*0.65)].mean()
    road_attn = cam[int(H*0.60):].mean()
    total = sky_attn + bld_attn + road_attn + 1e-8
    return {
        'sky':       sky_attn / total,
        'buildings': bld_attn / total,
        'road':      road_attn / total,
    }

# ─── Pixel-level diversity (насколько разнообразны пиксели в зоне) ─────────────
def compute_region_diversity(img_np, H):
    """
    Std пикселей в каждой зоне: высокий std = много деталей = информативно
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(float)
    return {
        'sky_std':       gray[:int(H*0.30)].std(),
        'buildings_std': gray[int(H*0.25):int(H*0.65)].std(),
        'road_std':      gray[int(H*0.60):].std(),
    }

# ─── Загрузка и прогон батча изображений ──────────────────────────────────────
def analyze_images(image_paths, n=150, device='cpu'):
    # Backbone — ResNet50 pretrained (ImageNet), аналог Kopernik backbone
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.eval().to(device)

    gradcam = GradCAM(backbone, backbone.layer4[-1])

    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = {
        'sky_attn': [], 'bld_attn': [], 'road_attn': [],
        'sky_std':  [], 'bld_std':  [], 'road_std':  [],
        'cam_entropy': [],
        'paths': [],
        'cam_examples': [],   # храним несколько примеров для визуализации
        'img_examples': [],
    }
    save_examples = 6

    print(f"Анализ {min(n, len(image_paths))} изображений на {device}...")
    for idx, path in enumerate(image_paths[:n]):
        try:
            img_pil = Image.open(path).convert('RGB').resize((224, 224))
            img_np = np.array(img_pil)
            x = preprocess(img_pil).unsqueeze(0).to(device)

            cam = gradcam.generate(x)
            H = cam.shape[0]

            attn = region_attention_scores(cam, H)
            div  = compute_region_diversity(img_np, H)

            # Энтропия CAM (равномерная = рассеяна, низкая = сфокусирована)
            cam_flat = cam.flatten() + 1e-8
            cam_flat /= cam_flat.sum()
            entropy = float(-(cam_flat * np.log(cam_flat)).sum())

            results['sky_attn'].append(float(attn['sky']))
            results['bld_attn'].append(float(attn['buildings']))
            results['road_attn'].append(float(attn['road']))
            results['sky_std'].append(div['sky_std'])
            results['bld_std'].append(div['buildings_std'])
            results['road_std'].append(div['road_std'])
            results['cam_entropy'].append(entropy)
            results['paths'].append(str(path))

            if len(results['cam_examples']) < save_examples:
                results['cam_examples'].append(cam.copy())
                results['img_examples'].append(img_np.copy())

            if (idx+1) % 30 == 0:
                print(f"  {idx+1}/{min(n, len(image_paths))} обработано...")
        except Exception as e:
            print(f"  Ошибка {path.name}: {e}")

    return results

# ─── Визуализация ─────────────────────────────────────────────────────────────
def plot_audit2(results):
    sky  = np.array(results['sky_attn'])
    bld  = np.array(results['bld_attn'])
    road = np.array(results['road_attn'])
    ent  = np.array(results['cam_entropy'])

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Audit #2: Semantic Noise — GradCAM Analysis on Moscow TTK\n'
                 f'N={len(sky)} images, ResNet50 backbone (ImageNet pretrained)',
                 fontsize=13, fontweight='bold')

    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # ── 1. Boxplot распределения activation по зонам ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    data = [sky, bld, road]
    labels = ['Небо\n(шум)', 'Здания\n(польза)', 'Дорога\n(шум)']
    colors = ['#74b9ff', '#00b894', '#636e72']
    bp = ax1.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax1.axhline(1/3, color='red', linestyle='--', alpha=0.6, label='Равномерно (1/3)')
    ax1.set_ylabel('Доля GradCAM activation')
    ax1.set_title('Внимание backbone по зонам кадра')
    ax1.legend(fontsize=8)
    # Аннотация средних
    for i, (arr, lbl) in enumerate(zip(data, ['Небо', 'Здания', 'Дорога']), 1):
        ax1.text(i, arr.mean() + 0.02, f'{arr.mean():.2%}',
                 ha='center', fontsize=9, fontweight='bold')

    # ── 2. Гистограмма entropy CAM ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(ent, bins=25, color='#6c5ce7', edgecolor='black', alpha=0.75)
    ax2.axvline(ent.mean(), color='red', linestyle='--',
                label=f'Mean={ent.mean():.2f}')
    # Теор. макс энтропия для 224x224
    max_ent = np.log(224*224)
    ax2.axvline(max_ent, color='gray', linestyle=':', label=f'Max={max_ent:.1f}')
    ax2.set_xlabel('CAM Entropy')
    ax2.set_ylabel('Частота')
    ax2.set_title(f'Энтропия GradCAM\n({ent.mean()/max_ent:.0%} от максимума)')
    ax2.legend(fontsize=8)

    # ── 3. Scatter: entropy vs building attention ────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(ent, bld, c=sky, cmap='RdYlGn_r', alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax3, label='Доля attention на небо')
    # Линия тренда
    z = np.polyfit(ent, bld, 1)
    x_line = np.linspace(ent.min(), ent.max(), 100)
    ax3.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=1.5)
    corr = np.corrcoef(ent, bld)[0,1]
    ax3.set_xlabel('CAM Entropy')
    ax3.set_ylabel('Attention на здания')
    ax3.set_title(f'Entropy vs Информативность\nCorr={corr:.3f}')

    # ── 4. Pixel diversity vs CAM attention ─────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    bld_std = np.array(results['bld_std'])
    road_std = np.array(results['road_std'])
    ax4.scatter(bld_std, bld, alpha=0.4, s=12, c='#00b894', label='Здания')
    ax4.scatter(road_std, road, alpha=0.4, s=12, c='#636e72', label='Дорога')
    corr_bld = np.corrcoef(bld_std, bld)[0,1]
    ax4.set_xlabel('Pixel std (diversity) в зоне')
    ax4.set_ylabel('GradCAM attention в зоне')
    ax4.set_title(f'Diversity vs Attention\nCorr(зд)={corr_bld:.3f}')
    ax4.legend(fontsize=8)

    # ── 5-10. Примеры CAM на реальных изображениях ───────────────────────────
    n_ex = len(results['cam_examples'])
    for i in range(n_ex):
        ax_img = fig.add_subplot(gs[1, i % 4] if i < 4 else gs[2, i % 4])
        img = results['img_examples'][i]
        cam = results['cam_examples'][i]

        # Overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img, 0.55, heatmap, 0.45, 0)

        H = cam.shape[0]
        attn = region_attention_scores(cam, H)
        dominant = max(attn, key=attn.get)
        color = {'sky': 'red', 'buildings': 'green', 'road': 'orange'}[dominant]

        ax_img.imshow(overlay)
        ax_img.set_title(f'Фокус: {dominant.upper()}\n'
                         f'sky={attn["sky"]:.0%} bld={attn["buildings"]:.0%} '
                         f'road={attn["road"]:.0%}',
                         fontsize=8, color=color)
        ax_img.axis('off')

    # ── Сводная таблица ────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 2:])
    ax_tbl.axis('off')

    noise_pct = (sky + road).mean()
    useful_pct = bld.mean()
    verdict = "КРИТИЧЕСКИЙ" if useful_pct < 0.35 else "УМЕРЕННЫЙ" if useful_pct < 0.50 else "OK"

    table_data = [
        ['Небо (шум)',      f'{sky.mean():.1%}',  f'{sky.std():.1%}'],
        ['Здания (польза)', f'{bld.mean():.1%}',  f'{bld.std():.1%}'],
        ['Дорога (шум)',    f'{road.mean():.1%}', f'{road.std():.1%}'],
        ['Шум итого',       f'{noise_pct:.1%}',   '---'],
        ['CAM Entropy/Max', f'{ent.mean()/np.log(224*224):.1%}', f'{ent.std()/np.log(224*224):.1%}'],
    ]
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=['Регион', 'Mean attention', 'Std'],
        cellLoc='center', loc='center', bbox=[0, 0.15, 1, 0.85]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for j in range(3):
        tbl[0, j].set_facecolor('#2c7bb6')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    tbl[2, 0].set_facecolor('#d4edda')  # здания — зелёный
    tbl[4, 0].set_facecolor('#f8d7da')  # шум итого — красный

    ax_tbl.set_title(f'ВЕРДИКТ: Семантический шум — {verdict}\n'
                     f'{noise_pct:.0%} внимания уходит на неинформативные регионы',
                     fontsize=10, fontweight='bold',
                     color='red' if verdict == 'КРИТИЧЕСКИЙ' else 'orange')

    out_path = 'audit_2_semantic_noise.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Сохранено: {out_path}")
    plt.close()

    # ── Консольный итог ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("AUDIT #2: SEMANTIC NOISE — ИТОГ")
    print("="*60)
    print(f"  Attention на небо  (шум):    {sky.mean():.1%} +/- {sky.std():.1%}")
    print(f"  Attention на здания (полезно): {bld.mean():.1%} +/- {bld.std():.1%}")
    print(f"  Attention на дорогу (шум):   {road.mean():.1%} +/- {road.std():.1%}")
    print(f"  ИТОГО ШУМ:                   {noise_pct:.1%}")
    print(f"  CAM Entropy/Max:             {ent.mean()/np.log(224*224):.1%}")
    print(f"  ВЕРДИКТ: Семантический шум — {verdict}")

# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    images_dir = Path('ttk_10k_full/images')
    image_paths = sorted(images_dir.glob('*.jpg'))
    print(f"Найдено изображений: {len(image_paths)}")

    # Берём равномерную выборку со всего датасета
    step = max(1, len(image_paths) // 150)
    sampled = image_paths[::step][:150]

    results = analyze_images(sampled, n=150, device=device)
    plot_audit2(results)
    print("[Audit #2 завершён]")
