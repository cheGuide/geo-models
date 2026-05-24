#!/usr/bin/env python3
"""
run_all_reports.py — генерирует все стандартизированные отчёты Moscow TTK-10k.

Запуск из любой директории:
    python deploy/run_all_reports.py [--only 01 15 19 ...]

Выходные PNG: standardized_reports/
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent

SCRIPTS: list[tuple[str, str]] = [
    ("01–13",  "generate_standardized_reports.py"),
    ("13",     "generate_segvlad_report.py"),
    ("14",     "render_vlm_report.py"),
    ("15",     "render_gaea_report.py"),
    ("16",     "generate_geoagent_local_report.py"),
    ("17",     "gen_17_kopernik_grokking.py"),
    ("19",     "generate_geoagent_sota_report.py"),
]


def run_script(path: Path) -> None:
    spec   = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all TTK benchmark reports.")
    parser.add_argument(
        "--only", nargs="*", metavar="ID",
        help="Run only scripts matching these report IDs (e.g. 15 19).",
    )
    args = parser.parse_args()

    filter_ids = set(args.only) if args.only else None

    total, failed = 0, []
    for report_id, fname in SCRIPTS:
        if filter_ids and not any(f in report_id for f in filter_ids):
            continue
        path = DEPLOY / fname
        if not path.exists():
            print(f"  [SKIP] {fname} — not found")
            continue
        print(f"\n{'─'*60}")
        print(f"  [{report_id}]  {fname}")
        print(f"{'─'*60}")
        t0 = time.time()
        try:
            run_script(path)
            print(f"  OK  ({time.time() - t0:.1f}s)")
            total += 1
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed.append(fname)

    print(f"\n{'='*60}")
    print(f"  Done: {total} report(s) generated.")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
