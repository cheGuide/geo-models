# Автоматическая проверка репозитория: venv, compileall, смоук IM2GPS на CPU (MODELS_ALLOW_CPU).
# Запуск:  .\scripts\run_self_check.ps1
# Или из Cursor: задача «models: self-check» (Ctrl+Shift+B → выбрать).
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = "1"
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "=== models self-check (repo: $RepoRoot) ===" -ForegroundColor Cyan

if (-not (Test-Path $Py)) {
    Write-Host "Создаю .venv и ставлю зависимости (первый раз может занять несколько минут)..." -ForegroundColor Yellow
    Set-Location $RepoRoot
    py -3 -m venv .venv
    & $Py -m pip install --upgrade pip --quiet
    $req = Join-Path $RepoRoot "requirements.txt"
    if (Test-Path $req) {
        & $Py -m pip install --default-timeout 120 -r $req
    }
    & $Py -m pip install --default-timeout 300 torch torchvision
}

Write-Host "[1/2] compileall tools, revisiting-im2gps, benchmarks_results ..." -ForegroundColor Cyan
& $Py -m compileall -q (Join-Path $RepoRoot "tools") (Join-Path $RepoRoot "revisiting-im2gps") (Join-Path $RepoRoot "benchmarks_results")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/2] smoke IM2GPS (CPU, MODELS_ALLOW_CPU) ..." -ForegroundColor Cyan
& (Join-Path $RepoRoot "scripts\smoke_test_cpu.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== self-check: OK ===" -ForegroundColor Green
