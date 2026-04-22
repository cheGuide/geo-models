# Запуск всех автоматических проверок в репозитории models:
#   - venv + зависимости из requirements.txt
#   - run_self_check (compileall + smoke IM2GPS)
#   - мини-прогон prepare_ttk_for_dvg_benchmark.py (копии в .automated_test_out)
#
# Запуск:  .\scripts\run_all_automated.ps1
# Требуется датасет: ttk_10k_full/ (dataset_metadata.json + images/)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:MODELS_ALLOW_CPU = "1"
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "=== models run_all_automated ===" -ForegroundColor Cyan
Set-Location $RepoRoot

if (-not (Test-Path $Py)) {
    Write-Host "Создаю .venv..." -ForegroundColor Yellow
    py -3 -m venv .venv
    & $Py -m pip install --upgrade pip --quiet
}

Write-Host "pip install -r requirements.txt ..." -ForegroundColor Cyan
$req = Join-Path $RepoRoot "requirements.txt"
if (Test-Path $req) {
    & $Py -m pip install --default-timeout 180 -r $req
}
# Проверка import torch (одна строка кода в аргументе -c)
& $Py -c "import torch" 2>$null | Out-Null
if (-not $?) {
    Write-Host "Installing torch/torchvision (CPU)..." -ForegroundColor Yellow
    & $Py -m pip install --default-timeout 300 torch torchvision
}

Write-Host "`n[1/3] run_self_check.ps1 ..." -ForegroundColor Cyan
& (Join-Path $RepoRoot "scripts\run_self_check.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ttk = Join-Path $RepoRoot "ttk_10k_full"
if (-not (Test-Path (Join-Path $ttk "dataset_metadata.json"))) {
    Write-Host "[2/3] пропуск prepare_ttk: нет $ttk\dataset_metadata.json" -ForegroundColor Yellow
} else {
    Write-Host "`n[2/3] prepare_ttk_for_dvg_benchmark (limit=40, --copy) ..." -ForegroundColor Cyan
    $out = Join-Path $RepoRoot ".automated_test_out\moscow_ttk_mini"
    if (Test-Path $out) { Remove-Item $out -Recurse -Force }
    & $Py (Join-Path $RepoRoot "scripts\prepare_ttk_for_dvg_benchmark.py") `
        --ttk_root $ttk `
        --out_dir $out `
        --limit 40 `
        --copy `
        --seed 42
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`n[3/3] done." -ForegroundColor Cyan
Write-Host "=== run_all_automated: OK ===" -ForegroundColor Green
