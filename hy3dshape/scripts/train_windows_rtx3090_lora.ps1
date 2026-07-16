[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [string]$Config = "",
    [string]$TrainDataset = "",
    [string]$ValDataset = "",
    [string]$OutputDir = "",
    [switch]$SmokeTest,
    [switch]$SkipDataValidation,
    [switch]$AllowExistingOutput
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ShapeRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ShapeRoot "..")).Path

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
    $Config = Join-Path $ShapeRoot "configs\hunyuandit-finetuning-flowmatching-dinol518-bf16-lora-rank8-rtx3090-windows.yaml"
}
if ([string]::IsNullOrWhiteSpace($TrainDataset)) {
    $TrainDataset = Join-Path $ShapeRoot "tools\mini_trainset\preprocessed"
}
if ([string]::IsNullOrWhiteSpace($ValDataset)) {
    $ValDataset = $TrainDataset
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $outputName = if ($SmokeTest) { "lora_rtx3090_windows_smoke" } else { "lora_rtx3090_windows" }
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

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$preflight = @'
import sys
import torch
import peft
import torch_cluster

if sys.version_info[:2] not in {(3, 10), (3, 11)}:
    raise RuntimeError(f"Use Python 3.10 or 3.11, got {sys.version.split()[0]}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access an NVIDIA CUDA GPU")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The selected GPU/PyTorch build does not support BF16")

props = torch.cuda.get_device_properties(0)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
print(f"GPU: {props.name}; VRAM: {props.total_memory / 1024**3:.1f} GiB")
print(f"PEFT: {peft.__version__}")
print(f"torch-cluster: {torch_cluster.__version__}")
'@

Push-Location $ShapeRoot
try {
    & $PythonExecutable -c $preflight
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
        Write-Host "Starting one-step RTX 3090 LoRA smoke test..."
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
