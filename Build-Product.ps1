param([string]$DotNetExecutable = '', [string]$CMakeExecutable = '', [switch]$SkipUi)
$ErrorActionPreference='Stop'
$productRoot=$PSScriptRoot
if (-not $CMakeExecutable) {
    $found=Get-Command cmake -ErrorAction SilentlyContinue
    if($found) {$CMakeExecutable=$found.Source}
    else {$CMakeExecutable='C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'}
}
& $CMakeExecutable -S $productRoot -B "$productRoot\build" -A x64
if($LASTEXITCODE -ne 0){throw 'CMake configuration failed'}
& $CMakeExecutable --build "$productRoot\build" --config Release --parallel 4
if($LASTEXITCODE -ne 0){throw 'Native build failed'}
$binRoot=Join-Path $productRoot 'bin'
New-Item -ItemType Directory -Force -Path $binRoot | Out-Null
if(-not $SkipUi){& "$productRoot\ui\build-ui.ps1" -OutputDirectory $binRoot -DotNetExecutable $DotNetExecutable}
foreach($name in @('RifeStandardPotPlayer.ax','RifeRegister.exe','RifeEngineCheck.exe','RifeEngineBuilder.exe','RifeSplitEngineBuilder.exe')){
    Copy-Item -LiteralPath (Join-Path "$productRoot\build\Release" $name) -Destination $binRoot -Force
}
if(-not(Test-Path -LiteralPath "$binRoot\RifeStandardPotPlayer.ini")) {Copy-Item -LiteralPath "$productRoot\config\RifeStandardPotPlayer.ini" -Destination $binRoot}
foreach($name in @('runtime','models')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $binRoot $name) | Out-Null
    $payloadSource=Join-Path $productRoot $name
    if(Test-Path -LiteralPath $payloadSource) { Get-ChildItem -LiteralPath $payloadSource -File | Copy-Item -Destination (Join-Path $binRoot $name) -Force }
}
New-Item -ItemType Directory -Force -Path "$binRoot\engines" | Out-Null
if(-not(Test-Path -LiteralPath "$binRoot\engines\rife425_standard_960_fp16.engine") -and (Test-Path -LiteralPath "$productRoot\engines\rife425_standard_960_fp16.engine")) {
    Copy-Item -LiteralPath "$productRoot\engines\rife425_standard_960_fp16.engine" -Destination "$binRoot\engines"
}
foreach($name in @('Prepare-Engine.ps1','README.md','THIRD_PARTY_NOTICES.md','LICENSE')) {
    Copy-Item -LiteralPath (Join-Path $productRoot $name) -Destination $binRoot -Force
}
Write-Output "Built $binRoot. No registration or player settings were changed."
