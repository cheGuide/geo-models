"""
На части установок TensorFlow под Windows пары модулей .py + .pywrap*.pyd
падают с DLL load failed. Регистрируем пустые модули в sys.modules до import tensorflow,
чтобы подхватился .py-лоадер с __try_import, либо чтобы не грузился сломанный pyd.

DELF / TF Hub не используют TFLite в рантайме — заглушки нужны только чтобы импорт TF прошёл.
"""
from __future__ import annotations

import sys
import types

# Все обнаруженные в tensorflow/lite/python каталоге *_pywrap*.pyd (имена модулей Python).
_TFLITE_PYWRAP_MODULES = (
    "tensorflow.lite.python._pywrap_analyzer_wrapper",
    "tensorflow.lite.python._pywrap_modify_model_interface",
    "tensorflow.lite.python._pywrap_string_util",
    "tensorflow.lite.python._pywrap_tensorflow_interpreter_wrapper",
    "tensorflow.lite.python._pywrap_tensorflow_lite_calibration_wrapper",
    "tensorflow.lite.python._pywrap_tensorflow_lite_metrics_wrapper",
    "tensorflow.lite.python.analyzer_wrapper._pywrap_analyzer_wrapper",
    "tensorflow.lite.python.interpreter_wrapper._pywrap_tensorflow_interpreter_wrapper",
    "tensorflow.lite.python.metrics._pywrap_tensorflow_lite_metrics_wrapper",
    "tensorflow.lite.python.optimize._pywrap_tensorflow_lite_calibration_wrapper",
)


def apply_stubs() -> None:
    for name in _TFLITE_PYWRAP_MODULES:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    # metrics wrapper ожидает класс MetricsWrapper в metrics_wrapper.py импорте
    m = sys.modules["tensorflow.lite.python.metrics._pywrap_tensorflow_lite_metrics_wrapper"]

    class MetricsWrapper:
        pass

    m.MetricsWrapper = MetricsWrapper


def apply_pkg_resources_shim() -> None:
    """tensorflow_hub импортирует pkg_resources.parse_version; setuptools 82+ может не класть pkg_resources."""
    if "pkg_resources" in sys.modules:
        return
    try:
        import pkg_resources  # noqa: F401
        return
    except ImportError:
        from packaging.version import parse as parse_version

        pr = types.ModuleType("pkg_resources")
        pr.parse_version = parse_version
        sys.modules["pkg_resources"] = pr
