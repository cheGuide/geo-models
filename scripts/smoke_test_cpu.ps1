# Быстрая проверка пайплайна без GPU (MODELS_ALLOW_CPU=1). Не для обучения.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:MODELS_ALLOW_CPU = "1"
$env:PYTHONUTF8 = "1"
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Нет .venv: сначала scripts\setup_gpu_venv.ps1 или python -m venv .venv" }
Set-Location (Join-Path $RepoRoot "revisiting-im2gps")
& $Py run_ttk.py --dataset_path (Join-Path $RepoRoot "ttk_10k_full") --max_ref 32 --max_query 5 --k 10
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "smoke_test_cpu: OK"
