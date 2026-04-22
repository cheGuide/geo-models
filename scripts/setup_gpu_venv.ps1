# Создаёт .venv в корне репозитория models (если ещё нет) и ставит PyTorch с CUDA (wheel cu124).
# Запуск: из PowerShell в каталоге репозитория:
#   .\scripts\setup_gpu_venv.ps1
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "Repo: $RepoRoot"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating venv at $RepoRoot\.venv ..."
    py -3 -m venv (Join-Path $RepoRoot ".venv")
} else {
    Write-Host "Venv already exists: $RepoRoot\.venv"
}

& (Join-Path $RepoRoot ".venv\Scripts\Activate.ps1")
python -m pip install --upgrade pip
# При другой версии CUDA смените cu124 на cu121 / cu118: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if (Test-Path (Join-Path $RepoRoot "requirements.txt")) {
    pip install --default-timeout 120 -r (Join-Path $RepoRoot "requirements.txt")
}
Write-Host "Done. Activate: .\.venv\Scripts\Activate.ps1"
