#!/usr/bin/env python3
r"""
Запуск всех бенчмарков на Moscow TTK и фиксация результатов.

Модели: AnyLoc, GeoCLIP, Revisit-Anything (baseline = предобученный стек), Kopernik (baseline = ImageNet + случайная голова), GAEA (city/country).
Датасет: переменная TTK_DATASET или по умолчанию dipl/ttk_10k_full, иначе ./ttk_10k_full в корне репо.
  BENCHMARK_NUM_SAMPLES — по умолчанию 500; **0** или **all** = вся выборка (все кадры на диске / все pano для GeoCLIP).
  BENCHMARK_FORCE_RERUN=1 — пересчёт без кэша из JSON.
  BENCHMARK_TIMEOUT_SEC — пусто и полный датасет: без лимита; иначе секунды (0 = без лимита).
  BENCHMARK_TRAINED_ONLY — по умолчанию **1**: только модели, дообученные у вас на TTK/Москве
    (GeoCLIP fine-tuned, Revisit-Anything fine-tuned, Kopernik fine-tuned). Baseline и AnyLoc не гоняются.
    Полный набор (включая baseline): BENCHMARK_TRAINED_ONLY=0.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
_default_ds = Path(os.environ.get("TTK_DATASET", r"c:\Users\q\Work\dipl\ttk_10k_full"))
if not _default_ds.is_dir() and (REPO / "ttk_10k_full").is_dir():
    _default_ds = REPO / "ttk_10k_full"
DATASET = _default_ds
OUTPUT_DIR = REPO / "benchmarks_results"
_s_ns = os.environ.get("BENCHMARK_NUM_SAMPLES", "500").strip().lower()
if _s_ns in ("0", "all", "-1"):
    NUM_SAMPLES = 0
else:
    NUM_SAMPLES = int(os.environ.get("BENCHMARK_NUM_SAMPLES", "500"))
SEED = 42
# 1/true — не подставлять кэш из JSON, пересчитать все тесты
FORCE_RERUN = os.environ.get("BENCHMARK_FORCE_RERUN", "").lower() in ("1", "true", "yes", "y")
# 1 (по умолчанию) — только дообученные модели; 0 — полный бенчмарк включая baseline и AnyLoc
_tr = os.environ.get("BENCHMARK_TRAINED_ONLY", "1").strip().lower()
TRAINED_ONLY = _tr not in ("0", "false", "no", "n", "off")


def _kopernik_n_from_output(out: str) -> int | None:
    m = re.search(r"Evaluating\s*:\s*(\d+)\s+", out)
    return int(m.group(1)) if m else None


def bench_timeout(default: int = 600) -> int | None:
    """Для полного датасета (NUM_SAMPLES==0) по умолчанию без таймаута, если не задан BENCHMARK_TIMEOUT_SEC."""
    t = os.environ.get("BENCHMARK_TIMEOUT_SEC", "").strip()
    if t == "":
        return None if NUM_SAMPLES == 0 else default
    ti = int(t)
    return None if ti <= 0 else ti


def run_cmd(cmd: list, cwd: Path, env=None, timeout=600) -> tuple[bool, str]:
    """Run command, return (success, output). timeout None = без ограничения по времени."""
    env = env or os.environ.copy()
    try:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def test_anyloc() -> dict:
    """AnyLoc baseline (DINOv2 + VLAD)."""
    venv_py = REPO / "AnyLoc" / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        return {"model": "AnyLoc", "variant": "baseline", "status": "skip", "reason": "venv not found"}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "anyloc_ttk_benchmark.json"
    ok, _ = run_cmd(
        [
            str(venv_py),
            "run_moscow_ttk.py",
            "--dataset_path", str(DATASET),
            "--num_samples", str(NUM_SAMPLES),
            "--output", str(out_file),
        ],
        cwd=REPO / "AnyLoc",
        timeout=bench_timeout(300),
    )
    if not ok:
        return {"model": "AnyLoc", "variant": "baseline", "status": "error", "reason": "run failed"}

    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            d = json.load(f)
        return {
            "model": "AnyLoc",
            "variant": "baseline",
            "status": "ok",
            "R1_pct": d.get("R1"),
            "R5_pct": d.get("R5"),
            "mean_error_km": d.get("mean_error_km"),
            "median_error_km": d.get("median_error_km"),
            "num_samples": d.get("num_samples"),
        }
    return {"model": "AnyLoc", "variant": "baseline", "status": "error", "reason": "no output"}


def _geoclip_python() -> Path | None:
    for rel in ("geo-clip/.venv/Scripts/python.exe", "geo-clip/geoclip_env/Scripts/python.exe"):
        p = REPO / rel
        if p.exists():
            return p
    return None


def test_geoclip_baseline() -> dict:
    """GeoCLIP baseline (pre-trained)."""
    venv_py = _geoclip_python()
    if not venv_py:
        return {"model": "GeoCLIP", "variant": "baseline", "status": "skip", "reason": "venv not found"}

    out_file = OUTPUT_DIR / "geoclip_baseline.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, _ = run_cmd(
        [
            str(venv_py),
            "run_test_moscow_ttk.py",
            "--dataset_path", str(DATASET),
            "--num_samples", str(NUM_SAMPLES),
            "--output", str(out_file),
            "--device", "cuda",
        ],
        cwd=REPO / "geo-clip",
        timeout=bench_timeout(300),
    )
    if not ok:
        return {"model": "GeoCLIP", "variant": "baseline", "status": "error", "reason": "run failed"}

    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            d = json.load(f)
        return {
            "model": "GeoCLIP",
            "variant": "baseline",
            "status": "ok",
            "mean_error_km": d.get("mean_error_km"),
            "median_error_km": d.get("median_error_km"),
            "within_1km_pct": d.get("within_1km_pct"),
            "within_5km_pct": d.get("within_5km_pct"),
            "within_25km_pct": d.get("within_25km_pct"),
            "num_samples": d.get("num_samples"),
        }
    return {"model": "GeoCLIP", "variant": "baseline", "status": "error", "reason": "no output"}


def test_geoclip_finetuned() -> dict:
    """GeoCLIP fine-tuned on TTK."""
    venv_py = _geoclip_python()
    ckpt = REPO / "geo-clip" / "checkpoints_ttk" / "geoclip_ttk_final.pth"
    if not venv_py:
        return {"model": "GeoCLIP", "variant": "fine-tuned", "status": "skip", "reason": "venv not found"}
    if not ckpt.exists():
        return {"model": "GeoCLIP", "variant": "fine-tuned", "status": "skip", "reason": "checkpoint not found"}

    out_file = OUTPUT_DIR / "geoclip_finetuned.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, _ = run_cmd(
        [
            str(venv_py),
            "run_test_moscow_ttk.py",
            "--dataset_path", str(DATASET),
            "--num_samples", str(NUM_SAMPLES),
            "--checkpoint", str(ckpt),
            "--output", str(out_file),
            "--device", "cuda",
        ],
        cwd=REPO / "geo-clip",
        timeout=bench_timeout(300),
    )
    if not ok:
        return {"model": "GeoCLIP", "variant": "fine-tuned", "status": "error", "reason": "run failed"}

    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            d = json.load(f)
        return {
            "model": "GeoCLIP",
            "variant": "fine-tuned",
            "status": "ok",
            "mean_error_km": d.get("mean_error_km"),
            "median_error_km": d.get("median_error_km"),
            "within_1km_pct": d.get("within_1km_pct"),
            "within_5km_pct": d.get("within_5km_pct"),
            "within_25km_pct": d.get("within_25km_pct"),
            "num_samples": d.get("num_samples"),
        }
    return {"model": "GeoCLIP", "variant": "fine-tuned", "status": "error", "reason": "no output"}


def test_revisit_baseline() -> dict:
    """Revisit-Anything baseline: чекпоинт Revisit-Anything/checkpoints/moscow_vpr_pretrained_baseline.ckpt (создаётся при отсутствии)."""
    venv_py = REPO / "AnyLoc" / ".venv" / "Scripts" / "python.exe"
    out_json = REPO / "Revisit-Anything" / "eval_results_pretrained_baseline.json"
    if not venv_py.exists():
        return {"model": "Revisit-Anything", "variant": "baseline", "status": "skip", "reason": "venv not found"}

    ok, _ = run_cmd(
        [
            str(venv_py),
            "eval_moscow_vpr.py",
            "--pretrained-baseline",
            "--dataset_path", str(DATASET),
            "--output_json", str(out_json),
        ],
        cwd=REPO / "Revisit-Anything",
        timeout=bench_timeout(600),
    )
    if not ok:
        return {"model": "Revisit-Anything", "variant": "baseline", "status": "error", "reason": "run failed"}

    if out_json.exists():
        with open(out_json, encoding="utf-8") as f:
            d = json.load(f)
        return {
            "model": "Revisit-Anything",
            "variant": "baseline",
            "status": "ok",
            "R1_pct": d.get("R1", 0) * 100,
            "R5_pct": d.get("R5", 0) * 100,
            "R10_pct": d.get("R10", 0) * 100,
            "val_db": d.get("val_db"),
            "val_queries": d.get("val_queries"),
        }
    return {"model": "Revisit-Anything", "variant": "baseline", "status": "error", "reason": "no output"}


def test_revisit_finetuned() -> dict:
    """Revisit-Anything fine-tuned (DINOv2 + SALAD)."""
    venv_py = REPO / "AnyLoc" / ".venv" / "Scripts" / "python.exe"
    ckpt = REPO / "Revisit-Anything" / "logs" / "moscow_vpr_ttk" / "lightning_logs" / "version_1" / "checkpoints" / "moscow_vpr_04_R1_0.1970.ckpt"
    if not venv_py.exists():
        return {"model": "Revisit-Anything", "variant": "fine-tuned", "status": "skip", "reason": "venv not found"}
    if not ckpt.exists():
        return {"model": "Revisit-Anything", "variant": "fine-tuned", "status": "skip", "reason": "checkpoint not found"}

    ok, _ = run_cmd(
        [
            str(venv_py),
            "eval_moscow_vpr.py",
            "--checkpoint", str(ckpt),
            "--dataset_path", str(DATASET),
        ],
        cwd=REPO / "Revisit-Anything",
        timeout=bench_timeout(3600),
    )
    if not ok:
        return {"model": "Revisit-Anything", "variant": "fine-tuned", "status": "error", "reason": "run failed"}

    eval_file = ckpt.parent / "eval_results.json"
    if eval_file.exists():
        with open(eval_file, encoding="utf-8") as f:
            d = json.load(f)
        return {
            "model": "Revisit-Anything",
            "variant": "fine-tuned",
            "status": "ok",
            "R1_pct": d.get("R1", 0) * 100,
            "R5_pct": d.get("R5", 0) * 100,
            "R10_pct": d.get("R10", 0) * 100,
            "val_db": d.get("val_db"),
            "val_queries": d.get("val_queries"),
        }
    return {"model": "Revisit-Anything", "variant": "fine-tuned", "status": "error", "reason": "no output"}


def _run_kopernik_eval(model_filename: str, variant_label: str) -> dict:
    """Запуск test_moscow_dataset.py с заданным чекпоинтом."""
    ckpt = REPO / "kopernik" / model_filename
    if not ckpt.exists():
        return {"model": "Kopernik", "variant": variant_label, "status": "skip", "reason": f"missing {model_filename}"}

    venv_py = sys.executable
    env = os.environ.copy()
    env["DATASET_PATH"] = str(DATASET)
    env["KOPERNIK_MODEL_PATH"] = model_filename
    env["BENCHMARK_NUM_SAMPLES"] = str(NUM_SAMPLES)
    ok, out = run_cmd(
        [venv_py, "test_moscow_dataset.py"],
        cwd=REPO / "kopernik",
        env=env,
        timeout=bench_timeout(900),
    )
    if not ok:
        return {"model": "Kopernik", "variant": variant_label, "status": "error", "reason": out[:400]}

    median_m = mean_m = None
    for line in out.splitlines():
        if "Median error" in line:
            try:
                median_m = float(line.split(":")[-1].strip().replace(" m", ""))
            except ValueError:
                pass
        if "Mean error" in line:
            try:
                mean_m = float(line.split(":")[-1].strip().replace(" m", ""))
            except ValueError:
                pass
    n_eff = _kopernik_n_from_output(out) or NUM_SAMPLES
    return {
        "model": "Kopernik",
        "variant": variant_label,
        "status": "ok",
        "median_error_m": median_m,
        "mean_error_km": mean_m / 1000 if mean_m else None,
        "median_error_km": median_m / 1000 if median_m else None,
        "checkpoint": model_filename,
        "num_samples": n_eff,
    }


def test_kopernik_baseline() -> dict:
    """Kopernik: веса kopernik/checkpoints/kopernik_imagenet_baseline.pth (создаются save_imagenet_baseline_weights.py при отсутствии)."""
    venv_py = sys.executable
    pth = REPO / "kopernik" / "checkpoints" / "kopernik_imagenet_baseline.pth"
    if not pth.exists():
        ok_save, err_save = run_cmd(
            [venv_py, "save_imagenet_baseline_weights.py"],
            cwd=REPO / "kopernik",
            timeout=bench_timeout(600),
        )
        if not ok_save:
            return {"model": "Kopernik", "variant": "baseline", "status": "error", "reason": err_save[:400]}
    if not pth.exists():
        return {"model": "Kopernik", "variant": "baseline", "status": "error", "reason": "kopernik baseline .pth not created"}

    env = os.environ.copy()
    env["DATASET_PATH"] = str(DATASET)
    env["BENCHMARK_NUM_SAMPLES"] = str(NUM_SAMPLES)
    env["KOPERNIK_MODEL_PATH"] = "checkpoints/kopernik_imagenet_baseline.pth"
    env.pop("KOPERNIK_BASELINE", None)

    ok, out = run_cmd(
        [venv_py, "test_moscow_dataset.py"],
        cwd=REPO / "kopernik",
        env=env,
        timeout=bench_timeout(900),
    )
    if not ok:
        return {"model": "Kopernik", "variant": "baseline", "status": "error", "reason": out[:400]}

    median_m = mean_m = None
    for line in out.splitlines():
        if "Median error" in line:
            try:
                median_m = float(line.split(":")[-1].strip().replace(" m", ""))
            except ValueError:
                pass
        if "Mean error" in line:
            try:
                mean_m = float(line.split(":")[-1].strip().replace(" m", ""))
            except ValueError:
                pass
    n_eff = _kopernik_n_from_output(out) or NUM_SAMPLES
    ret = {
        "model": "Kopernik",
        "variant": "baseline",
        "status": "ok",
        "median_error_m": median_m,
        "mean_error_km": mean_m / 1000 if mean_m else None,
        "median_error_km": median_m / 1000 if median_m else None,
        "checkpoint": "checkpoints/kopernik_imagenet_baseline.pth",
        "num_samples": n_eff,
    }
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "kopernik_baseline.json", "w", encoding="utf-8") as f:
            json.dump(ret, f, indent=2, ensure_ascii=False)
    except OSError:
        pass
    return ret


def test_kopernik() -> dict:
    """Kopernik: финальная локализация Москвы."""
    return _run_kopernik_eval("resnet50_moscow_localization.pth", "fine-tuned (moscow localization)")


def load_existing_results() -> dict:
    """Load existing JSON results if benchmarks were run before."""
    results = {}
    # AnyLoc
    p = OUTPUT_DIR / "anyloc_ttk_benchmark.json"
    if not p.exists():
        p = REPO / "AnyLoc" / "anyloc_ttk_results.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        results["AnyLoc_baseline"] = {
            "model": "AnyLoc", "variant": "baseline", "status": "ok",
            "R1_pct": d.get("R1"), "R5_pct": d.get("R5"),
            "mean_error_km": d.get("mean_error_km"), "median_error_km": d.get("median_error_km"),
            "num_samples": d.get("num_samples"),
        }
    # GeoCLIP
    for name, fname in [("baseline", "geoclip_baseline_gpu_results.json"), ("fine-tuned", "geoclip_finetuned_gpu_results.json")]:
        p = REPO / "geo-clip" / fname
        if not p.exists():
            p = OUTPUT_DIR / f"geoclip_{name}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            results[f"GeoCLIP_{name}"] = {
                "model": "GeoCLIP", "variant": name, "status": "ok",
                "mean_error_km": d.get("mean_error_km"), "median_error_km": d.get("median_error_km"),
                "within_1km_pct": d.get("within_1km_pct"), "within_5km_pct": d.get("within_5km_pct"),
                "within_25km_pct": d.get("within_25km_pct"), "num_samples": d.get("num_samples"),
            }
    # Revisit-Anything
    p = REPO / "Revisit-Anything" / "eval_results_pretrained_baseline.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        results["Revisit-Anything_baseline"] = {
            "model": "Revisit-Anything", "variant": "baseline", "status": "ok",
            "R1_pct": d.get("R1", 0) * 100, "R5_pct": d.get("R5", 0) * 100, "R10_pct": d.get("R10", 0) * 100,
            "val_db": d.get("val_db"), "val_queries": d.get("val_queries"),
        }
    p = REPO / "Revisit-Anything" / "logs" / "moscow_vpr_ttk" / "lightning_logs" / "version_1" / "checkpoints" / "eval_results.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        results["Revisit-Anything_fine-tuned"] = {
            "model": "Revisit-Anything", "variant": "fine-tuned", "status": "ok",
            "R1_pct": d.get("R1", 0) * 100, "R5_pct": d.get("R5", 0) * 100, "R10_pct": d.get("R10", 0) * 100,
            "val_db": d.get("val_db"), "val_queries": d.get("val_queries"),
        }
    p = OUTPUT_DIR / "kopernik_baseline.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        results["Kopernik_baseline"] = d
    return results


def main():
    print("=" * 60)
    print("  Moscow TTK Benchmarks — " + ("только дообученные модели" if TRAINED_ONLY else "все модели (включая baseline)"))
    print(f"  Dataset: {DATASET}")
    print(f"  Samples: {'ALL (full set)' if NUM_SAMPLES == 0 else NUM_SAMPLES}")
    print(f"  Date: {datetime.now().isoformat()}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = {} if FORCE_RERUN else load_existing_results()
    all_results = []

    if TRAINED_ONLY:
        tests = [
            ("GeoCLIP fine-tuned", lambda: test_geoclip_finetuned() if "GeoCLIP_fine-tuned" not in existing else existing["GeoCLIP_fine-tuned"]),
            ("Revisit-Anything fine-tuned", lambda: test_revisit_finetuned() if "Revisit-Anything_fine-tuned" not in existing else existing["Revisit-Anything_fine-tuned"]),
            ("Kopernik fine-tuned", test_kopernik),
        ]
    else:
        tests = [
            ("AnyLoc baseline", lambda: test_anyloc() if "AnyLoc_baseline" not in existing else existing["AnyLoc_baseline"]),
            ("GeoCLIP baseline", lambda: test_geoclip_baseline() if "GeoCLIP_baseline" not in existing else existing["GeoCLIP_baseline"]),
            ("GeoCLIP fine-tuned", lambda: test_geoclip_finetuned() if "GeoCLIP_fine-tuned" not in existing else existing["GeoCLIP_fine-tuned"]),
            ("Revisit-Anything baseline", lambda: test_revisit_baseline() if "Revisit-Anything_baseline" not in existing else existing["Revisit-Anything_baseline"]),
            ("Revisit-Anything fine-tuned", lambda: test_revisit_finetuned() if "Revisit-Anything_fine-tuned" not in existing else existing["Revisit-Anything_fine-tuned"]),
            ("Kopernik baseline", lambda: test_kopernik_baseline() if "Kopernik_baseline" not in existing else existing["Kopernik_baseline"]),
            ("Kopernik fine-tuned", test_kopernik),
        ]

    for name, fn in tests:
        print(f"\n>>> {name}...")
        try:
            r = fn()
            all_results.append(r)
            print(f"    {r.get('status', '?')}: {r}")
        except Exception as e:
            all_results.append({"model": name.split()[0], "variant": "?", "status": "error", "reason": str(e)})
            print(f"    ERROR: {e}")

    # Save JSON
    report_path = OUTPUT_DIR / "benchmarks_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": str(DATASET),
            "date": datetime.now().isoformat(),
            "num_samples": NUM_SAMPLES,
            "num_samples_mode": "all" if NUM_SAMPLES == 0 else "subset",
            "trained_only": TRAINED_ONLY,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report: {report_path}")

    # Generate Markdown
    md_path = REPO / "BENCHMARKS_REPORT.md"
    lines = [
        "# Moscow TTK Benchmarks",
        "",
        f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Датасет:** {DATASET}",
        f"**Сэмплов:** {'все доступные (full)' if NUM_SAMPLES == 0 else NUM_SAMPLES}",
        f"**Режим:** {'только дообученные модели (GeoCLIP / Revisit / Kopernik ft)' if TRAINED_ONLY else 'полный бенчмарк (включая baseline)'}",
        "",
        "## Результаты",
        "",
        "| Модель | Вариант | R@1 % | R@5 % | R@10 % | Mean err km | Median err km | Within 1km % | Within 5km % | Within 25km % |",
        "|--------|---------|-------|-------|--------|-------------|---------------|--------------|--------------|----------------|",
    ]

    for r in all_results:
        status = r.get("status", "?")
        if status == "skip":
            reason = str(r.get("reason", "")).replace("|", "/")
            lines.append(f"| {r.get('model','?')} | {r.get('variant','?')} | — | — | — | — | — | — | — | — | *(skip: {reason})* |")
            continue
        if status == "error":
            lines.append(f"| {r.get('model','?')} | {r.get('variant','?')} | — | — | — | — | — | — | — | — | *(error)* |")
            continue

        r1 = r.get("R1_pct")
        r5 = r.get("R5_pct")
        r10 = r.get("R10_pct")
        mean = r.get("mean_error_km")
        median = r.get("median_error_km")
        w1 = r.get("within_1km_pct")
        w5 = r.get("within_5km_pct")
        w25 = r.get("within_25km_pct")

        def fmt(x):
            if x is None: return "—"
            if isinstance(x, float): return f"{x:.2f}"
            return str(x)

        lines.append(f"| {r.get('model','?')} | {r.get('variant','?')} | {fmt(r1)} | {fmt(r5)} | {fmt(r10)} | {fmt(mean)} | {fmt(median)} | {fmt(w1)} | {fmt(w5)} | {fmt(w25)} |")

    notes = [
        "",
        "## Примечания",
        "",
    ]
    if TRAINED_ONLY:
        notes.extend([
            "В этом прогоне только **дообученные** на ваших данных модели: GeoCLIP (TTK), Revisit-Anything (TTK), Kopernik (Moscow локализация). Полный набор: `BENCHMARK_TRAINED_ONLY=0`.",
            "",
        ])
    else:
        notes.extend([
            "- **AnyLoc**: DINOv2 + VLAD, baseline (без дообучения)",
            "- **GeoCLIP**: image→location, baseline = pre-trained MP-16, fine-tuned = дообучен на TTK",
            "- **Revisit-Anything**: DINOv2 + SALAD (VPR); baseline = предобученный стек без fine-tune на TTK; fine-tuned — дообучен на TTK.",
            "- **Kopernik**: ResNet50 regression; baseline = ImageNet backbone + случайная голова; fine-tuned — дообучен на Moscow (median в метрах)",
            "",
        ])
    notes.extend([
        "- **GAEA**, **GeoVista**: требуют отдельной настройки (vLLM, LoRA)",
        "",
    ])
    lines.extend(notes)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
