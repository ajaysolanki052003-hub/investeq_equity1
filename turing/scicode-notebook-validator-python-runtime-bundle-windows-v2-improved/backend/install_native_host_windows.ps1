# ---------------------------------------------------------------------------
# install_native_host_windows.ps1 -- v2-improved
# Registers the Chrome/Edge native messaging host on Windows.
# Works with either a native .exe or a Python script/bytecode.
# When a Python file is used, the launcher (.cmd) always invokes it via the
# Python interpreter -- never runs .py/.pyc as a raw executable.
# Registry writes are wrapped in try/catch so MDM-locked machines still work
# via the manifest JSON file alone.
#
# Usage (called automatically by install_python_runtime_windows.ps1):
#   .\install_native_host_windows.ps1 -ExtensionId <id>
#       [-HostPath <path>] [-InstallRoot <path>]
#       [-PythonExe <path>] [-Browser chrome|edge|both]
# ---------------------------------------------------------------------------
param(
  [Parameter(Mandatory = $true)]
  [string]$ExtensionId,

  [Parameter(Mandatory = $false)]
  [string]$HostPath = "",

  [Parameter(Mandatory = $false)]
  [string]$InstallRoot = "",

  [Parameter(Mandatory = $false)]
  [string]$PythonExe = "python",

  [Parameter(Mandatory = $false)]
  [ValidateSet("chrome", "edge", "both")]
  [string]$Browser = "both"
)

$ErrorActionPreference = "Stop"

$scriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$templatePath = Join-Path $scriptDir "com.scicode.validator.native.template.json"

# -- Resolve host path -------------------------------------------------------
$resolvedHostPath = $null
if ($HostPath -and $HostPath.Trim().Length -gt 0) {
  $resolvedHostPath = (Resolve-Path $HostPath).Path
} else {
  $candidates = @(
    (Join-Path $scriptDir "scicode_native_host.exe"),
    (Join-Path $scriptDir "scicode_native_host.py"),
    (Join-Path $scriptDir "scicode_native_host.pyc")
  )
  foreach ($c in $candidates) {
    if (Test-Path $c) { $resolvedHostPath = (Resolve-Path $c).Path; break }
  }
  if ($null -eq $resolvedHostPath) {
    throw "ERROR: Could not find scicode_native_host.exe/.py/.pyc in $scriptDir"
  }
}

# -- Install root -- %APPDATA% is always user-writable -----------------------
$installDir = if ($InstallRoot -and $InstallRoot.Trim().Length -gt 0) {
  $InstallRoot
} else {
  Join-Path $env:APPDATA "SciCodeNative\backend"
}
New-Item -ItemType Directory -Force -Path $installDir | Out-Null
$installDir = (Resolve-Path $installDir).Path

# -- Determine host type -----------------------------------------------------
$ext      = [System.IO.Path]::GetExtension($resolvedHostPath).ToLower()
$isExe    = ($ext -eq ".exe")
$isPython = ($ext -eq ".py" -or $ext -eq ".pyc")

$managedHostPath = $resolvedHostPath
if ($isExe) {
  $managedHostPath = Join-Path $installDir "scicode_native_host.exe"
  Copy-Item -Force $resolvedHostPath $managedHostPath
}

# -- Build self-healing launcher .cmd ----------------------------------------
# The .cmd file is what Chrome/Edge actually calls via the manifest.
# For Python hosts, it always routes through the interpreter.
# Self-healing: if the stored venv Python is gone, falls back to python in PATH.
$launcherPath = Join-Path $installDir "run_scicode_native_host.cmd"

if ($isExe) {
  $launcherContent = "@echo off`r`n`"$managedHostPath`" %*`r`n"
} else {
  $launcherContent = "@echo off`r`nset VENV_PY=$PythonExe`r`nif not exist `"%VENV_PY%`" set VENV_PY=python`r`n`"%VENV_PY%`" `"$managedHostPath`" %*`r`n"
}
$launcherContent | Set-Content -Encoding ASCII -Path $launcherPath

# -- Render native messaging manifest ----------------------------------------
$template        = Get-Content -Raw -Path $templatePath
$launcherEscaped = $launcherPath.Replace('\', '\\')
$manifest        = $template.Replace("__HOST_SCRIPT_PATH__", $launcherEscaped)
$manifest        = $manifest.Replace("__EXTENSION_ID__", $ExtensionId)

# -- Install per-browser -----------------------------------------------------
$targets = @()
if ($Browser -in @("chrome", "both")) {
  $targets += @{
    Name        = "Chrome"
    ManifestDir = (Join-Path $env:LOCALAPPDATA "Google\Chrome\User Data\NativeMessagingHosts")
    RegPath     = "HKCU:\Software\Google\Chrome\NativeMessagingHosts\com.scicode.validator.native"
  }
}
if ($Browser -in @("edge", "both")) {
  $targets += @{
    Name        = "Edge"
    ManifestDir = (Join-Path $env:LOCALAPPDATA "Microsoft\Edge\User Data\NativeMessagingHosts")
    RegPath     = "HKCU:\Software\Microsoft\Edge\NativeMessagingHosts\com.scicode.validator.native"
  }
}

foreach ($target in $targets) {
  New-Item -ItemType Directory -Force -Path $target.ManifestDir | Out-Null
  $manifestPath = Join-Path $target.ManifestDir "com.scicode.validator.native.json"
  $manifest | Set-Content -Encoding UTF8 -Path $manifestPath
  Write-Host "[$($target.Name)] Installed manifest at: $manifestPath"

  # Registry key is optional -- Chrome/Edge also honour the JSON manifest directly.
  # Wrapped in try/catch so MDM-locked HKCU keys do not abort the install.
  try {
    New-Item -Path $target.RegPath -Force | Out-Null
    Set-ItemProperty -Path $target.RegPath -Name "(default)" -Value $manifestPath
    Write-Host "[$($target.Name)] Registered registry key: $($target.RegPath)"
  } catch {
    Write-Host "[$($target.Name)] WARNING: Could not write registry key. Browser will use manifest file directly -- this is fine."
  }
}

Write-Host ""
Write-Host "Launcher  : $launcherPath"
Write-Host "Host      : $managedHostPath"
Write-Host "Extension : $ExtensionId"
