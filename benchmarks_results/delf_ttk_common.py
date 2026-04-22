"""Общие функции для DELF (TF Hub) на TTK: импорт TF, извлечение пулингового эмбеддинга."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))


def setup_tensorflow_imports():
    from tf_windows_tflite_stubs import apply_pkg_resources_shim, apply_stubs

    apply_stubs()
    apply_pkg_resources_shim()
    import tensorflow as tf
    import tensorflow_hub as hub

    return tf, hub


def build_delf_fn(tf, hub):
    delf = hub.load("https://tfhub.dev/google/delf/1").signatures["default"]

    def run_delf(float_image):
        return delf(
            image=float_image,
            score_threshold=tf.constant(100.0),
            image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
            max_feature_num=tf.constant(1000),
        )

    return run_delf


def pooled_embedding(result: dict, dim_hint: int) -> np.ndarray:
    desc = result["descriptors"].numpy()
    if desc.shape[0] == 0:
        return np.zeros(dim_hint, dtype=np.float32)
    v = np.mean(desc.astype(np.float64), axis=0)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def extract_pooled_from_pil(pil_image, run_delf, tf, desc_dim_holder: list) -> np.ndarray:
    """desc_dim_holder: одноэлементный список [int|None] для размера дескриптора."""
    np_image = np.asarray(pil_image.convert("RGB"))
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)
    result = run_delf(float_image)
    d = result["descriptors"].numpy()
    if desc_dim_holder[0] is None and d.shape[0] > 0:
        desc_dim_holder[0] = int(d.shape[-1])
    return pooled_embedding(result, desc_dim_holder[0] or 40)
