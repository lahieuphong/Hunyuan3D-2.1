[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [string]$Config = "",
    [string]$TrainDataset = "",
    [string]$ValDataset = "",
    [string]$OutputDir = "",
    [switch]$SmokeTest,
    [switch]$Pilot,
    [switch]$PreflightOnly,
    [switch]$SkipDataValidation,
    [switch]$AllowExistingOutput
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ShapeRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ShapeRoot "..")).Path

if ($SmokeTest -and $Pilot) {
    throw "Choose either -SmokeTest (2 steps) or -Pilot (200 steps), not both."
}

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = Join-Path $RepoRoot ".venv-win\Scripts\python.exe"
}
if (Test-Path -LiteralPath $PythonExecutable -PathType Leaf) {
    $PythonExecutable = (Resolve-Path -LiteralPath $PythonExecutable).Path
}
else {
    $command = Get-Command $PythonExecutable -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Python executable was not found: $PythonExecutable"
    }
    $PythonExecutable = $command.Source
}

if ([string]::IsNullOrWhiteSpace($Config)) {
    $configName = if ($Pilot) {
        "hunyuandit-finetuning-flowmatching-dinol518-bf16-lora-rank8-rtx3090-windows-pilot200.yaml"
    }
    else {
        "hunyuandit-finetuning-flowmatching-dinol518-bf16-lora-rank8-rtx3090-windows.yaml"
    }
    $Config = Join-Path $ShapeRoot "configs\$configName"
}
if ([string]::IsNullOrWhiteSpace($TrainDataset)) {
    $TrainDataset = Join-Path $ShapeRoot "tools\mini_trainset\preprocessed"
}
if ([string]::IsNullOrWhiteSpace($ValDataset)) {
    $ValDataset = $TrainDataset
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $outputName = if ($SmokeTest) {
        "lora_rtx3090_windows_smoke"
    }
    elseif ($Pilot) {
        "lora_rtx3090_windows_pilot_200"
    }
    else {
        "lora_rtx3090_windows"
    }
    $OutputDir = Join-Path $ShapeRoot "output_folder\dit\$outputName"
}

if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    throw "Training config was not found: $Config"
}
if (-not (Test-Path -LiteralPath $TrainDataset)) {
    throw "Training dataset was not found: $TrainDataset"
}
if (-not (Test-Path -LiteralPath $ValDataset)) {
    throw "Validation dataset was not found: $ValDataset"
}

$Config = (Resolve-Path -LiteralPath $Config).Path
$TrainDataset = (Resolve-Path -LiteralPath $TrainDataset).Path
$ValDataset = (Resolve-Path -LiteralPath $ValDataset).Path
if (-not $PreflightOnly) {
    if (Test-Path -LiteralPath $OutputDir) {
        $existingOutput = Get-ChildItem -LiteralPath $OutputDir -Force | Select-Object -First 1
        if ($null -ne $existingOutput) {
            if (-not $AllowExistingOutput) {
                throw "Output directory is not empty: $OutputDir. Choose a new -OutputDir or pass -AllowExistingOutput intentionally."
            }
            Write-Warning "Reusing a non-empty output directory can mix logs and adapter snapshots: $OutputDir"
        }
    }
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
}

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,garbage_collection_threshold:0.8"
$env:HF_HOME = Join-Path $RepoRoot ".cache\huggingface"
$env:HY3DGEN_MODELS = Join-Path $RepoRoot ".cache\hy3dgen"

Push-Location $ShapeRoot
try {
    & $PythonExecutable "tools\check_windows_training_env.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Training environment preflight failed."
    }

    if (-not $SkipDataValidation) {
        & $PythonExecutable "tools\validate_shape_dataset.py" $TrainDataset `
            --views 24 --pc-size 81920 --pc-sharpedge-size 0
        if ($LASTEXITCODE -ne 0) {
            throw "Training dataset validation failed."
        }
        if ($ValDataset -ne $TrainDataset) {
            & $PythonExecutable "tools\validate_shape_dataset.py" $ValDataset `
                --views 24 --pc-size 81920 --pc-sharpedge-size 0
            if ($LASTEXITCODE -ne 0) {
                throw "Validation dataset validation failed."
            }
        }
    }

    if ($PreflightOnly) {
        Write-Host "Environment and dataset preflight completed; training was not started."
        return
    }

    Copy-Item -LiteralPath $Config -Destination (Join-Path $OutputDir "training_config_source.yaml") -Force

    $trainArgs = @(
        "main.py",
        "--config", $Config,
        "--output_dir", $OutputDir,
        "--num_nodes", "1",
        "--num_gpus", "1",
        "--train_data_list", $TrainDataset,
        "--val_data_list", $ValDataset,
        "--seed", "0",
        "--fast"
    )
    if ($SmokeTest) {
        $trainArgs += "--smoke_test"
        Write-Host "Starting two-step RTX 3090 LoRA smoke test..."
    }
    elseif ($Pilot) {
        Write-Host "Starting 200-step RTX 3090 LoRA pilot..."
    }
    else {
        Write-Host "Starting RTX 3090 LoRA training..."
    }

    & $PythonExecutable @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Training process exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
