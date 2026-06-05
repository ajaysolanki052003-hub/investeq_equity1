# ---------------------------------------------------------------------------
# install_python_runtime_windows.ps1 -- v2-improved
# Sets up the Python venv, installs dependencies, copies the runtime sources
# (.py files -- no .pyc, fully version-independent), and calls
# install_native_host_windows.ps1 to register the native messaging host.
#
# Usage:
#   .\install_python_runtime_windows.ps1 -ExtensionId <id>
#       [-Browser chrome|edge|both]
#       [-RuntimeInstallRoot <path>]
#       [-PythonCommand python]
#
# Defaults:
#   Browser             = both
#   RuntimeInstallRoot  = %APPDATA%\SciCodeNative\python_runtime
#   PythonCommand       = python
# ---------------------------------------------------------------------------
param(
  [Parameter(Mandatory = $true)]
  [string]$ExtensionId,

  [Parameter(Mandatory = $false)]
  [ValidateSet("chrome", "edge", "both")]
  [string]$Browser = "both",

  [Parameter(Mandatory = $false)]
  [string]$RuntimeInstallRoot = "",

  [Parameter(Mandatory = $false)]
  [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"

$scriptDir        = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeSource    = Join-Path $scriptDir "python_runtime"
$requirementsPath = Join-Path $scriptDir "requirements-runtime.txt"

if (!(Test-Path $runtimeSource)) {
  throw "ERROR: Missing python_runtime\ folder in $scriptDir"
}
if (!(Test-Path $requirementsPath)) {
  throw "ERROR: Missing requirements-runtime.txt in $scriptDir"
}

# Host .py check
$hostSourcePy = Join-Path $runtimeSource "native_messaging_backend\scicode_native_host.py"
if (!(Test-Path $hostSourcePy)) {
  throw "ERROR: Missing host script at $hostSourcePy"
}

# Default install root -- %APPDATA% is always user-writable, no elevation needed
$runtimeRoot = if ($RuntimeInstallRoot -and $RuntimeInstallRoot.Trim().Length -gt 0) {
  $RuntimeInstallRoot
} else {
  Join-Path $env:APPDATA "SciCodeNative\python_runtime"
}
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
$runtimeRoot = (Resolve-Path $runtimeRoot).Path

$runtimeTarget = Join-Path $runtimeRoot "runtime"
$venvPath      = Join-Path $runtimeRoot "venv"
$venvPython    = Join-Path $venvPath "Scripts\python.exe"

# Fresh copy of sources (always overwrite to pick up updates)
if (Test-Path $runtimeTarget) { Remove-Item -Recurse -Force $runtimeTarget }
Copy-Item -Recurse -Force $runtimeSource $runtimeTarget

# Create / refresh venv -- always remove first so Python version changes take effect
if (Test-Path $venvPath) { Remove-Item -Recurse -Force $venvPath }
Write-Host "Creating Python virtual environment at: $venvPath"
& $PythonCommand -m venv $venvPath

Write-Host "Installing Python dependencies..."
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r $requirementsPath --quiet

$installedHostPy   = Join-Path $runtimeTarget "native_messaging_backend\scicode_native_host.py"
$nativeInstallRoot = Join-Path $env:APPDATA "SciCodeNative\backend"
$nativeHostScript  = Join-Path $scriptDir "install_native_host_windows.ps1"

Write-Host "Registering native messaging host..."
& $nativeHostScript `
  -ExtensionId $ExtensionId `
  -HostPath    $installedHostPy `
  -InstallRoot $nativeInstallRoot `
  -PythonExe   $venvPython `
  -Browser     $Browser

Write-Host ""
Write-Host "=== Installation complete ==="
Write-Host "Runtime root : $runtimeRoot"
Write-Host "Venv Python  : $venvPython"
Write-Host "Host script  : $installedHostPy"
