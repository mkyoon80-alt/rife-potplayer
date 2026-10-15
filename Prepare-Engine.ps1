param([switch]$Rebuild)
$ErrorActionPreference = 'Stop'
function Invoke-LoggedEngineTool {
    param([string]$Tool, [string[]]$Arguments, [string]$Log)
    # Windows PowerShell 5.1 treats native stderr as error records. TensorRT
    # writes ordinary progress there; the process exit code determines success.
    $ErrorActionPreference = 'Continue'
    if (!(Test-Path -LiteralPath $Tool)) { throw 'Engine tool is missing.' }
    & $Tool @Arguments *> $Log
    return $LASTEXITCODE
}
$productRoot = $PSScriptRoot
if (!(Test-Path -LiteralPath (Join-Path $productRoot 'RifeEngineCheck.exe'))) { $productRoot = Join-Path $productRoot 'bin' }
$engineRoot = Join-Path $productRoot 'engines'
$logRoot = Join-Path $productRoot 'logs'
$runtimeRoot = Join-Path $productRoot 'runtime'
$resources = Join-Path $productRoot 'tools\TensorRT'
$plan = Join-Path $engineRoot 'rife425_standard_960_fp16.engine'
$probe = Join-Path $productRoot 'RifeEngineCheck.exe'
$builder = Join-Path $productRoot 'RifeEngineBuilder.exe'
$model = Join-Path $productRoot 'models\rife425_motion.onnx'
New-Item -ItemType Directory -Force -Path $engineRoot,$logRoot | Out-Null
$lock = $null
$oldPath = $env:PATH
$candidate = Join-Path $engineRoot ('prepare-' + [guid]::NewGuid().ToString('N') + '.engine')
$oldPlan = $plan + '.previous'
try {
    $lock = [IO.File]::Open((Join-Path $engineRoot 'prepare.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    if (!(Test-Path -LiteralPath $probe)) { throw 'RifeEngineCheck.exe is missing. Extract the complete package.' }
    if (!$Rebuild -and (Test-Path -LiteralPath $plan)) {
        $checkExit = Invoke-LoggedEngineTool $probe @('--check',$runtimeRoot) (Join-Path $logRoot 'engine-check.log')
        if ($checkExit -eq 0) { Write-Output 'Engine verified. Every video reuses it; no compilation was needed.'; exit 0 }
    }
    if (Get-Process -Name PotPlayerMini64,PotPlayer64,PotPlayerMini -ErrorAction SilentlyContinue) { throw 'Close PotPlayer before preparing a new engine.' }
    if (!(Test-Path -LiteralPath $builder)) { throw 'RifeEngineBuilder.exe is missing. Extract the complete package.' }
    if (!(Test-Path -LiteralPath $model)) { throw 'The RIFE model is missing.' }
    if ((Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash -ne '95c59f2b45ab358981e7ec044b946abd69960e29d034bc16e99223768b284506') { throw 'The RIFE model does not match this release.' }
    $env:PATH = $runtimeRoot + ';' + $resources + ';' + $oldPath
    Write-Output 'Preparing RIFE Standard 960 for this GPU. This can take several minutes.'
    $buildExit = Invoke-LoggedEngineTool $builder @('--build',$runtimeRoot,$resources,$model,$candidate,(Join-Path $engineRoot 'timing.cache')) (Join-Path $logRoot 'engine-build.log')
    if ($buildExit -ne 0) { throw 'Engine build failed; the previous engine was retained. See logs\engine-build.log.' }
    $hadPrevious = Test-Path -LiteralPath $plan
    if ($hadPrevious) { [IO.File]::Replace($candidate,$plan,$oldPlan,$true) } else { [IO.File]::Move($candidate,$plan) }
    try {
        $checkExit = Invoke-LoggedEngineTool $probe @('--check',$runtimeRoot) (Join-Path $logRoot 'engine-check.log')
        if ($checkExit -ne 0) { throw 'D3D11 inference validation failed. See logs\engine-check.log.' }
    } catch {
        $failed = $candidate + '.failed'
        if ($hadPrevious) { [IO.File]::Replace($oldPlan,$plan,$failed,$true) } else { [IO.File]::Move($plan,$failed) }
        throw
    }
    [ordered]@{preparedUtc=[DateTime]::UtcNow.ToString('o');engineSha256=(Get-FileHash -LiteralPath $plan -Algorithm SHA256).Hash;modelSha256=(Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash;runtimeSha256=(Get-FileHash -LiteralPath (Join-Path $runtimeRoot 'nvinfer_10.dll') -Algorithm SHA256).Hash;shape='1x7x540x960';precision='FP16 convolutions; FP32 coordinates and I/O'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $engineRoot 'prepared.json') -Encoding UTF8
    Write-Output 'Engine preparation and D3D11 validation complete. All videos reuse this engine.'
} finally { $env:PATH=$oldPath; if ($lock) { $lock.Dispose() } }

