# Offline, API-free deterministic Lab07 scenario and benchmark report.
# Run from repository root: .\examples\run_lab07_and_report.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path "pyproject.toml")) {
    throw "Run this script from the repository root."
}
$runId = "lab07-repro-$(Get-Date -Format yyyyMMddTHHmmssfff)-$PID"
uv run lab-cli lab07 all --profile pilot --seed 42 --experiment-run-id $runId
if ($LASTEXITCODE -ne 0) { throw "Lab07 simulation failed." }
$reportDir = Join-Path "results" "benchmarks"
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
uv run python -m lab_core.benchmark_report --lab lab07 --run-id $runId --scenario transient_failures --format md --output "$reportDir/$runId.md"
if ($LASTEXITCODE -ne 0) { throw "Markdown report failed." }
uv run python -m lab_core.benchmark_report --lab lab07 --run-id $runId --scenario transient_failures --format json --output "$reportDir/$runId.json"
if ($LASTEXITCODE -ne 0) { throw "JSON report failed." }
Write-Host "Saved reproducible benchmark reports to $reportDir"
