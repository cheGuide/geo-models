#Requires -Version 5.1
<#
.SYNOPSIS
  GeoAgent Docker: wait for Docker Desktop (Windows), then docker compose.

  From repo root (models):
    .\scripts\geoagent-docker.ps1 build
    .\scripts\geoagent-docker.ps1 prepare
    .\scripts\geoagent-docker.ps1 train
    .\scripts\geoagent-docker.ps1 wait
    .\scripts\geoagent-docker.ps1 shell   # up -d geoagent-shell (persistent)
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('wait', 'build', 'prepare', 'train', 'shell')]
    [string]$Action = 'build',

    [int]$TimeoutSec = 180,

    [switch]$NoStartDesktop
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$isWin = ($env:OS -eq 'Windows_NT')

function Test-DockerDaemon {
    # docker пишет в stderr при недоступном демоне; при $ErrorActionPreference Stop это ломает цикл ожидания
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        $null = & docker info 2>&1
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Start-DockerDesktopIfWindows {
    if (-not $isWin -or $NoStartDesktop) { return $false }
    $pf86 = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe')
    )
    if ($pf86) {
        $candidates += (Join-Path $pf86 'Docker\Docker\Docker Desktop.exe')
    }
    foreach ($p in $candidates) {
        if (Test-Path -LiteralPath $p) {
            Write-Host "Starting Docker Desktop: $p"
            Start-Process -FilePath $p
            return $true
        }
    }
    return $false
}

function Wait-DockerDaemon {
    param([int]$Timeout)
    $deadline = (Get-Date).AddSeconds($Timeout)
    $didFirst = $false
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerDaemon) {
            Write-Host 'Docker daemon is ready.'
            return
        }
        if (-not $didFirst) {
            Write-Host 'Waiting for Docker daemon...'
            $didFirst = $true
            $launched = Start-DockerDesktopIfWindows
            if (-not $launched -and $isWin) {
                Write-Host 'Docker Desktop not found in default paths; start Docker manually.'
            }
        }
        Start-Sleep -Seconds 3
    }
    throw "Docker daemon did not respond within $Timeout seconds. Start Docker and retry."
}

Set-Location -LiteralPath $RepoRoot

Wait-DockerDaemon -Timeout $TimeoutSec

# На Windows Docker Desktop симлинк data/ttk_10k_full -> другой каталог не виден внутри контейнера;
# монтируем реальный путь поверх (docker compose run -v перекрывает том сервиса).
function Get-TtkBindArgs {
    $ttp = Join-Path $RepoRoot 'data\ttk_10k_full'
    if (-not (Test-Path -LiteralPath $ttp)) { return @() }
    try {
        $resolved = (Resolve-Path -LiteralPath $ttp).Path
    } catch {
        return @()
    }
    Write-Host "TTK dataset bind: $resolved -> /workspace/data/ttk_10k_full"
    return @('-v', "${resolved}:/workspace/data/ttk_10k_full:ro")
}

switch ($Action) {
    'wait' {
        Write-Host 'Done.'
    }
    'shell' {
        & docker compose --profile geoagent up -d geoagent-shell
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host 'Container geoagent-shell is running. Example: docker exec -it geoagent-shell bash'
    }
    'build' {
        & docker compose build geoagent-train
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    'prepare' {
        $bind = Get-TtkBindArgs
        & docker compose --profile geoagent run @bind geoagent-prepare
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    'train' {
        $bind = Get-TtkBindArgs
        & docker compose --profile geoagent run @bind geoagent-train
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
