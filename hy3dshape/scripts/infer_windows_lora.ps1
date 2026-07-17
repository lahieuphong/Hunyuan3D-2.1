[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [string]$Image = "",
    [string]$Adapter = "",
    [string]$Output = "",
    [int]$Steps = 20,
    [double]$GuidanceScale = 5.0,
    [int]$OctreeResolution = 256,
    [int]$Seed = 1234
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ShapeRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ShapeRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = Join-Path $RepoRoot ".venv-win\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable was not found: $PythonExecutable"
}
$PythonExecutable = (Resolve-Path -LiteralPath $PythonExecutable).Path

if ([string]::IsNullOrWhiteSpace($Image)) {
    $Image = Join-Path $ShapeRoot "tools\mini_trainset\preprocessed\00a4cff37043361068376104a292f5b44b5eacbd174651553b6a7ae35647a2a6\render_cond\000.png"
}
if ([string]::IsNullOrWhiteSpace($Adapter)) {
    $Adapter = Join-Path $ShapeRoot "output_folder\dit\mini_test_01\lora\final"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $ShapeRoot "output_folder\inference\mini_test_01_lora_000.glb"
}

if (-not (Test-Path -LiteralPath $Image -PathType Leaf)) {
    throw "Input image was not found: $Image"
}
if (-not (Test-Path -LiteralPath $Adapter -PathType Container)) {
    throw "LoRA adapter directory was not found: $Adapter"
}

$Image = (Resolve-Path -LiteralPath $Image).Path
$Adapter = (Resolve-Path -LiteralPath $Adapter).Path
$Output = [System.IO.Path]::GetFullPath($Output)
$OutputParent = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $OutputParent | Out-Null

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,garbage_collection_threshold:0.8"
$env:HF_HOME = Join-Path $RepoRoot ".cache\huggingface"
$env:HY3DGEN_MODELS = Join-Path $RepoRoot ".cache\hy3dgen"

Push-Location $RepoRoot
try {
    & $PythonExecutable (Join-Path $ScriptDir "infer_windows_lora.py") `
        --image $Image `
        --adapter $Adapter `
        --output $Output `
        --steps $Steps `
        --guidance-scale $GuidanceScale `
        --octree-resolution $OctreeResolution `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Shape inference exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
