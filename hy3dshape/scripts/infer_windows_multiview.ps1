[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [string]$Front = "",
    [string]$Left = "",
    [string]$Back = "",
    [string]$Right = "",
    [string]$Output = "",
    [int]$Steps = 30,
    [double]$GuidanceScale = 5.0,
    [int]$OctreeResolution = 256,
    [int]$Seed = 12345,
    [string]$Model = "tencent/Hunyuan3D-2mv",
    [string]$Subfolder = "hunyuan3d-dit-v2-mv"
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

# The bundled renders use a spiral camera path. These four frames are the
# closest coherent clockwise quarter-turn set and are intended only as a smoke
# test; replace them with synchronized canonical views for real use.
$SampleRoot = Join-Path $ShapeRoot "tools\mini_trainset\preprocessed\7dc69acf647a7ef597f0c9462097370f2c8f8e212f9fd1e7f3547f45b3f3a3d8\render_cond"
if ([string]::IsNullOrWhiteSpace($Front)) { $Front = Join-Path $SampleRoot "007.png" }
if ([string]::IsNullOrWhiteSpace($Left)) { $Left = Join-Path $SampleRoot "005.png" }
if ([string]::IsNullOrWhiteSpace($Back)) { $Back = Join-Path $SampleRoot "006.png" }
if ([string]::IsNullOrWhiteSpace($Right)) { $Right = Join-Path $SampleRoot "004.png" }
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $ShapeRoot "output_folder\inference\multiview_sample_4views.glb"
}

$ViewPaths = [ordered]@{
    Front = $Front
    Left = $Left
    Back = $Back
    Right = $Right
}
foreach ($Entry in @($ViewPaths.GetEnumerator())) {
    if (-not (Test-Path -LiteralPath $Entry.Value -PathType Leaf)) {
        throw "$($Entry.Key) image was not found: $($Entry.Value)"
    }
    $ViewPaths[$Entry.Key] = (Resolve-Path -LiteralPath $Entry.Value).Path
}

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
    & $PythonExecutable (Join-Path $ScriptDir "infer_windows_multiview.py") `
        --front $ViewPaths.Front `
        --left $ViewPaths.Left `
        --back $ViewPaths.Back `
        --right $ViewPaths.Right `
        --output $Output `
        --model $Model `
        --subfolder $Subfolder `
        --steps $Steps `
        --guidance-scale $GuidanceScale `
        --octree-resolution $OctreeResolution `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Multi-view shape inference exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
