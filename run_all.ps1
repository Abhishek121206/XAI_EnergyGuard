# Starts the XAI-EnergyGuard backend + dashboard, each in its own window.
#
#   .\run_all.ps1            collector + live pipeline + API + dashboard
#   .\run_all.ps1 -Train     retrain the model first
#   .\run_all.ps1 -NoCollector   demo from the saved data (no ESP32 / MQTT)
#
# The ESP32 (boot.py + main.py via Thonny) and the Mosquitto service
# are started separately - see README.md.

param(
    [switch]$Train,
    [switch]$NoCollector
)

$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"

# Use the project venv if there is one
$venvPython = Join-Path $backend ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

function Start-Window($title, $directory, $arguments) {
    Start-Process powershell -WorkingDirectory $directory -ArgumentList @(
        "-NoExit",
        "-Command",
        "`$Host.UI.RawUI.WindowTitle = '$title'; & '$python' $arguments"
    )
}

Set-Location $backend

if ($Train -or -not (Test-Path "models\model_bundle.joblib")) {
    Write-Host "Training model..."
    & $python train.py
    if ($LASTEXITCODE -ne 0) { throw "Training failed" }
}

Write-Host "Running pipeline once..."
& $python pipeline_runner.py
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed" }

if (-not $NoCollector) {
    $mosquitto = Get-Service mosquitto -ErrorAction SilentlyContinue
    if ($null -eq $mosquitto -or $mosquitto.Status -ne "Running") {
        Write-Warning "Mosquitto service is not running (Start-Service mosquitto)."
    }
    Start-Window "EnergyGuard - collector" $backend "data_collector.py"
    Start-Window "EnergyGuard - live pipeline" $backend "live_pipeline.py"
}

Start-Window "EnergyGuard - API" $backend "app.py"
Start-Window "EnergyGuard - dashboard" $frontend "-m http.server 5500"

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5500"
