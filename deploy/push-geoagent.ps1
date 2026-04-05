#Requires -Version 5.1
<#
  Сборка образа GeoAgent и push в container registry.
  Перед запуском: docker login ghcr.io   (или docker.io)

  Пример:
    .\push-geoagent.ps1 -Registry ghcr.io/myuser
    .\push-geoagent.ps1 -Registry ghcr.io/myuser -ImageName geoagent-train -Tag v1
#>
param(
    [Parameter(Mandatory = $true, HelpMessage = 'Registry host + namespace, e.g. ghcr.io/myorg')]
    [string]$Registry,

    [string]$ImageName = 'geo-deploy-geoagent',
    [string]$Tag = 'latest'
)

$ErrorActionPreference = 'Stop'
$DeployRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $DeployRoot

$Registry = $Registry.TrimEnd('/')
$remoteImage = "${Registry}/${ImageName}:${Tag}"

Write-Host "Building geo-deploy-geoagent ..."
& docker compose build geoagent-shell
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# compose помечает образ как geo-deploy-geoagent:latest (см. GEOAGENT_IMAGE по умолчанию)
$built = 'geo-deploy-geoagent:latest'
& docker tag $built $remoteImage
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Pushing $remoteImage ..."
& docker push $remoteImage
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "On the server, set in deploy/.env:" -ForegroundColor Green
Write-Host "  GEOAGENT_IMAGE=$remoteImage"
Write-Host ""
