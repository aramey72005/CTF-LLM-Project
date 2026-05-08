$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
}

Write-Host "Installing/updating dependencies..."
& $pythonExe -m pip install -r (Join-Path $projectRoot "requirements.txt")

Write-Host "Starting Flask app on http://127.0.0.1:5000 ..."
& $pythonExe (Join-Path $projectRoot "app.py")
