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
function Ensure-OptimizedEngines {
    $splitBuilder = Join-Path $productRoot 'RifeSplitEngineBuilder.exe'
    $specs = @(
        @{Name='head'; Hash='1fd11285c820848c2deac5ce3758bf7d45a54cb1433e86e3d786c9d707245ec4'},
        @{Name='tail'; Hash='0af2cbccff025d6f5c10169f7f50026651580d58c2a46a622329023d5c693701'}
    )
    $complete = @($specs | Where-Object { Test-Path -LiteralPath (Join-Path $engineRoot ('rife425_' + $_.Name + '_960_fp16.engine')) }).Count -eq 2
    if (!$Rebuild -and $complete) {
        $checkExit = Invoke-LoggedEngineTool $probe @('--check-optimized',$runtimeRoot) (Join-Path $logRoot 'engine-optimization-check.log')
        if ($checkExit -eq 0) { Write-Output 'Optimized engines verified against original pixels.'; return }
    }
    if (Get-Process -Name PotPlayerMini64,PotPlayer64,PotPlayerMini,PotPlayer -ErrorAction SilentlyContinue) { throw 'Close PotPlayer before preparing optimized engines.' }
    if (!(Test-Path -LiteralPath $splitBuilder)) { throw 'RifeSplitEngineBuilder.exe is missing. Extract the complete v0.3.0 package.' }
    $transaction = [guid]::NewGuid().ToString('N')
    foreach ($spec in $specs) {
        $spec.Model = Join-Path $productRoot ('models\rife425_' + $spec.Name + '.onnx')
        if (!(Test-Path -LiteralPath $spec.Model) -or (Get-FileHash -LiteralPath $spec.Model -Algorithm SHA256).Hash -ne $spec.Hash) { throw 'An Optimization model is missing or does not match this release.' }
        $spec.Candidate = Join-Path $engineRoot ('prepare-' + $transaction + '-' + $spec.Name + '.engine')
        $spec.Target = Join-Path $engineRoot ('rife425_' + $spec.Name + '_960_fp16.engine')
        $spec.Previous = $spec.Target + '.previous-' + $transaction
        $spec.Cache = Join-Path $engineRoot ('timing-' + $spec.Name + '.cache')
        # Use the full engine's tactic history when available. A GPU/runtime
        # mismatch is rejected/reset by the builder, never forced compatible.
        $fullCache = Join-Path $engineRoot 'timing.cache'
        if (Test-Path -LiteralPath $fullCache) { Copy-Item -LiteralPath $fullCache -Destination $spec.Cache -Force }
        Write-Output ('Preparing optimized ' + $spec.Name + ' for this GPU. This can take several minutes.')
        $buildExit = Invoke-LoggedEngineTool $splitBuilder @('--build',$runtimeRoot,$resources,$spec.Model,$spec.Candidate,$spec.Cache,$spec.Name) (Join-Path $logRoot ('engine-' + $spec.Name + '-build.log'))
        if ($buildExit -ne 0) { throw 'Optimized engine preparation failed; existing Head/Tail engines were retained. See logs.' }
    }
    $published = [Collections.Generic.List[object]]::new()
    try {
        foreach ($spec in $specs) {
            $spec.HadPrevious = Test-Path -LiteralPath $spec.Target
            if ($spec.HadPrevious) { [IO.File]::Replace($spec.Candidate,$spec.Target,$spec.Previous,$true) }
            else { [IO.File]::Move($spec.Candidate,$spec.Target) }
            $published.Add($spec)
        }
        $checkExit = Invoke-LoggedEngineTool $probe @('--check-optimized',$runtimeRoot) (Join-Path $logRoot 'engine-optimization-check.log')
        if ($checkExit -ne 0) { throw 'Optimized D3D11 inference/pixel comparison failed. Previous Head/Tail engines will be restored. See logs\engine-optimization-check.log.' }
    } catch {
        for ($i=$published.Count-1; $i -ge 0; $i--) {
            $spec=$published[$i]
            if ($spec.HadPrevious) { [IO.File]::Replace($spec.Previous,$spec.Target,($spec.Candidate + '.failed'),$true) }
            else { [IO.File]::Move($spec.Target,($spec.Candidate + '.failed')) }
        }
        throw
    }
    [ordered]@{preparedUtc=[DateTime]::UtcNow.ToString('o');headSha256=(Get-FileHash -LiteralPath $specs[0].Target -Algorithm SHA256).Hash;tailSha256=(Get-FileHash -LiteralPath $specs[1].Target -Algorithm SHA256).Hash;precision='FP16 convolutions; FP32 features/coordinates and I/O';pixelCheck='NV12 and P010, 6 outputs exactly match the full engine'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $engineRoot 'prepared-optimization.json') -Encoding UTF8
    Write-Output 'Optimized preparation and original-pixel comparison complete.'
}
try {
    $lock = [IO.File]::Open((Join-Path $engineRoot 'prepare.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    if (!(Test-Path -LiteralPath $probe)) { throw 'RifeEngineCheck.exe is missing. Extract the complete package.' }
    if (!$Rebuild -and (Test-Path -LiteralPath $plan)) {
        $checkExit = Invoke-LoggedEngineTool $probe @('--check',$runtimeRoot) (Join-Path $logRoot 'engine-check.log')
        if ($checkExit -eq 0) { Ensure-OptimizedEngines; Write-Output 'All engines verified. Videos reuse them without playback-time compilation.'; exit 0 }
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
    Ensure-OptimizedEngines
    Write-Output 'Engine preparation and D3D11 validation complete. All videos reuse these engines.'
} finally { $env:PATH=$oldPath; if ($lock) { $lock.Dispose() } }

