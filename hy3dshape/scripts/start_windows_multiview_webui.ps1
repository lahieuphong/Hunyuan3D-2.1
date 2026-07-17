[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [string]$ListenHost = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8080,
    [string]$Model = "tencent/Hunyuan3D-2mv",
    [string]$Subfolder = "hunyuan3d-dit-v2-mv",
    [switch]$Background,
    [switch]$OpenBrowser,
    [switch]$Stop,
    [switch]$PreflightOnly,
    [ValidateRange(30, 600)]
    [int]$StartupTimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ShapeRoot = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $ShapeRoot
$AppPath = Join-Path $RepoRoot "gradio_app.py"
$DefaultPython = Join-Path $RepoRoot ".venv-win\Scripts\python.exe"
$Python = if ([string]::IsNullOrWhiteSpace($PythonExecutable)) { $DefaultPython } else { $PythonExecutable }
$OutputDir = Join-Path $ShapeRoot "output_folder\webui"
$LogsDir = Join-Path $OutputDir "logs"
$PidFile = Join-Path $LogsDir ("webui-{0}.pid" -f $Port)
$RequirementsFile = Join-Path $ShapeRoot "requirements-windows-multiview-ui.txt"

$ModelsRoot = Join-Path $RepoRoot ".cache\hy3dgen"
$ModelRelativePath = $Model.Replace('/', [IO.Path]::DirectorySeparatorChar)
$ModelDir = Join-Path (Join-Path $ModelsRoot $ModelRelativePath) $Subfolder
$ModelConfig = Join-Path $ModelDir "config.yaml"
$ModelWeights = Join-Path $ModelDir "model.fp16.safetensors"

$BrowserHost = if ($ListenHost -eq "0.0.0.0") {
    "127.0.0.1"
}
elseif ($ListenHost -in @("::", "[::]", "::1", "[::1]")) {
    "[::1]"
}
else {
    $ListenHost
}
$Url = "http://${BrowserHost}:$Port"
$HealthUrl = "$Url/health"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Get-ManagedProcess {
    if (-not (Test-Path -LiteralPath $PidFile)) {
        return $null
    }

    $serverProcessIdText = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    $serverProcessId = 0
    if (-not [int]::TryParse($serverProcessIdText, [ref]$serverProcessId)) {
        Remove-Item -LiteralPath $PidFile -Force
        return $null
    }

    $process = Get-Process -Id $serverProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Remove-Item -LiteralPath $PidFile -Force
        return $null
    }
    return $process
}

function Get-WebUiHealth {
    try {
        return Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 2
    }
    catch {
        return $null
    }
}

$managedProcess = Get-ManagedProcess
if ($Stop) {
    if ($null -eq $managedProcess) {
        Write-Host "Web UI is not running (no live managed PID for port $Port)."
        exit 0
    }

    $health = Get-WebUiHealth
    if ($null -eq $health -or $health.status -ne "ready" -or
        $null -eq $health.pid -or [int]$health.pid -ne $managedProcess.Id) {
        throw "Refusing to stop PID $($managedProcess.Id): $HealthUrl did not confirm the same managed Web UI process. Remove the stale PID file manually after verifying the process."
    }

    Stop-Process -Id $managedProcess.Id -Force
    Wait-Process -Id $managedProcess.Id -Timeout 15 -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped Web UI process $($managedProcess.Id)."
    exit 0
}

if ($null -ne $managedProcess) {
    $health = Get-WebUiHealth
    if ($null -ne $health -and $health.status -eq "ready") {
        $serverProcessId = if ($null -ne $health.pid) { [int]$health.pid } else { $managedProcess.Id }
        Set-Content -LiteralPath $PidFile -Value $serverProcessId -Encoding ascii
        Write-Host "Web UI is already running at $Url (PID $serverProcessId)."
        if ($OpenBrowser) {
            Start-Process $Url
        }
        exit 0
    }
    throw "PID $($managedProcess.Id) is alive, but $HealthUrl is not ready. Check logs in $LogsDir."
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $AppPath -PathType Leaf)) {
    throw "Gradio app not found: $AppPath"
}
if (-not (Test-Path -LiteralPath $ModelConfig -PathType Leaf)) {
    throw "Model config not found: $ModelConfig"
}
if (-not (Test-Path -LiteralPath $ModelWeights -PathType Leaf)) {
    throw "FP16 safetensors model not found: $ModelWeights"
}

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:GRADIO_ANALYTICS_ENABLED = "False"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,garbage_collection_threshold:0.8"
$env:HF_HOME = Join-Path $RepoRoot ".cache\huggingface"
$env:HY3DGEN_MODELS = $ModelsRoot
$env:U2NET_HOME = Join-Path $RepoRoot ".cache\rembg"

& $Python -c "import gradio, fastapi, uvicorn; print(f'UI dependencies: gradio={gradio.__version__}, fastapi={fastapi.__version__}, uvicorn={uvicorn.__version__}')"
if ($LASTEXITCODE -ne 0) {
    throw "Web UI dependencies are missing. Install them with: `"$Python`" -m pip install -r `"$RequirementsFile`""
}

& $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(f'GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}')"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA preflight failed."
}

Write-Host "Model: $Model/$Subfolder"
Write-Host "Weights: $ModelWeights"
Write-Host "Output: $OutputDir"
Write-Host "URL: $Url"

if ($ListenHost -notin @("127.0.0.1", "localhost", "::1")) {
    Write-Warning "The UI has no authentication. Only expose it on a trusted network."
}

if ($PreflightOnly) {
    Write-Host "Web UI preflight passed."
    exit 0
}

$portOwner = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($null -ne $portOwner) {
    throw "Port $Port is already in use by PID $($portOwner[0].OwningProcess)."
}

$PythonArgs = @(
    "-u",
    $AppPath,
    "--model_path", $Model,
    "--subfolder", $Subfolder,
    "--host", $ListenHost,
    "--port", $Port.ToString(),
    "--cache-path", $OutputDir,
    "--device", "cuda",
    "--use_safetensors",
    "--variant", "fp16",
    "--dtype", "float16",
    "--disable_tex"
)

if (-not $Background) {
    Write-Host "Starting Web UI in the foreground. Press Ctrl+C to stop it."
    if ($OpenBrowser) {
        Write-Host "Open $Url after the model finishes loading."
    }
    Push-Location $RepoRoot
    try {
        & $Python @PythonArgs
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdoutLog = Join-Path $LogsDir "webui_${Timestamp}.stdout.log"
$StderrLog = Join-Path $LogsDir "webui_${Timestamp}.stderr.log"
$QuotedArguments = ($PythonArgs | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }) -join ' '

$process = Start-Process -FilePath $Python `
    -ArgumentList $QuotedArguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii
Write-Host "Started Web UI process $($process.Id). Waiting for model load..."

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    $process.Refresh()
    if ($process.HasExited) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        Write-Host "--- stdout ---"
        if (Test-Path -LiteralPath $StdoutLog) { Get-Content -LiteralPath $StdoutLog -Tail 40 }
        Write-Host "--- stderr ---"
        if (Test-Path -LiteralPath $StderrLog) { Get-Content -LiteralPath $StderrLog -Tail 40 }
        throw "Web UI exited during startup with code $($process.ExitCode)."
    }

    $health = Get-WebUiHealth
    if ($null -ne $health -and $health.status -eq "ready") {
        $serverProcessId = if ($null -ne $health.pid) { [int]$health.pid } else { $process.Id }
        Set-Content -LiteralPath $PidFile -Value $serverProcessId -Encoding ascii
        Write-Host "Web UI is ready: $Url"
        Write-Host "PID: $serverProcessId"
        Write-Host "stdout: $StdoutLog"
        Write-Host "stderr: $StderrLog"
        Write-Host "Stop command: powershell -NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`" -Stop -Port $Port"
        if ($OpenBrowser) {
            Start-Process $Url
        }
        exit 0
    }
    Start-Sleep -Milliseconds 500
}

Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
throw "Web UI did not become ready within $StartupTimeoutSeconds seconds. Check $StdoutLog and $StderrLog."
