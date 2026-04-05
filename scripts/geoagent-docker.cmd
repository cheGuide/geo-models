@echo off
setlocal
cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0geoagent-docker.ps1" %*
exit /b %ERRORLEVEL%
