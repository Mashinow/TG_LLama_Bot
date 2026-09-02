$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $projectRoot ".venv\Scripts\pythonw.exe"
$spec = Join-Path $projectRoot "tg-llama-bot.spec"
$exe = Join-Path $projectRoot "dist\tg-llama-bot.exe"

if (-not (Test-Path -LiteralPath $pythonw)) {
    throw "Project Python was not found: $pythonw"
}

if (-not (Test-Path -LiteralPath $spec)) {
    throw "PyInstaller spec was not found: $spec"
}

$arguments = @(
    "-m"
    "PyInstaller"
    "--noconfirm"
    "--clean"
    "tg-llama-bot.spec"
)

$process = Start-Process `
    -FilePath $pythonw `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -Wait `
    -PassThru

if ($process.ExitCode -ne 0) {
    throw "PyInstaller failed with exit code $($process.ExitCode)."
}

if (-not (Test-Path -LiteralPath $exe)) {
    throw "PyInstaller finished without creating: $exe"
}

Write-Output "Built $exe"
