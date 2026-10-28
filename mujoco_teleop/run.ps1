# Reuse my_env without depending on conda.exe or shell activation.
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $env:USERPROFILE 'Apps\Anaconda\envs\my_env\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
}
& $pythonPath (Join-Path $PSScriptRoot 'run.py') @args
exit $LASTEXITCODE
