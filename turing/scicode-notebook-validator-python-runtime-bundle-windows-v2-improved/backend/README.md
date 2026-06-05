# Native Messaging Backend — v2-improved

This folder contains the local Python host used by `chrome_extension_native/`.

## What's improved in v2-improved

| Issue (v2)                           | Fix (v2-improved)                                                   |
|--------------------------------------|----------------------------------------------------------------------|
| `.pyc` files — Python 3.13 only      | Shipping `.py` source files — works with any Python 3.9+             |
| Default install path required elevation on MDM-restricted machines | Default is now `~/.scicode_native/` (`%APPDATA%\SciCodeNative\` on Windows) — always user-writable |
| Launcher `chmod +x`'d a `.pyc` file and tried to run it as a binary | Launcher is a shell script that always invokes Python explicitly; `.py`/`.pyc` are never `chmod +x`'d |
| Stale venv path caused silent failures | Launcher is self-healing: if stored venv Python is missing, falls back to `python3` in PATH |
| Windows registry write crashed on MDM | Registry write is now in `try/catch`; manifest JSON is sufficient — browser still registers |

---

## Quick setup — macOS

```bash
# Make executable (only needed once; script does it automatically on re-runs)
chmod +x ./install_python_runtime_macos.sh

./install_python_runtime_macos.sh "<YOUR_EXTENSION_ID>" both
```

The single command:
1. Copies `.py` runtime sources to `~/.scicode_native/python_runtime/runtime/`
2. Creates a venv at `~/.scicode_native/python_runtime/venv/`
3. Installs dependencies from `requirements-runtime.txt`
4. Writes a self-healing launcher at `~/.scicode_native/backend/run_scicode_native_host.sh`
5. Registers the native messaging manifest for Chrome and Edge

---

## Quick setup — Linux

```bash
chmod +x ./install_python_runtime_linux.sh
./install_python_runtime_linux.sh "<YOUR_EXTENSION_ID>" both
```

---

## Quick setup — Windows

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\install_python_runtime_windows.ps1 `
  -ExtensionId "<YOUR_EXTENSION_ID>" `
  -Browser both
```

The single command:
1. Copies `.py` runtime sources to `%APPDATA%\SciCodeNative\python_runtime\runtime\`
2. Creates a venv at `%APPDATA%\SciCodeNative\python_runtime\venv\`
3. Installs dependencies from `requirements-runtime.txt`
4. Writes a self-healing launcher at `%APPDATA%\SciCodeNative\backend\run_scicode_native_host.cmd`
5. Installs native messaging manifest for Chrome and Edge
6. Attempts registry write (skips gracefully if blocked by policy)

---

## Advanced / manual install

### macOS — custom paths

```bash
./install_python_runtime_macos.sh "<EXT_ID>" both "/custom/runtime/root" python3.12
```

Arguments: `<extension_id>` `[chrome|edge|both]` `[runtime_install_root]` `[python_exe]`

### Windows — custom paths

```powershell
.\install_python_runtime_windows.ps1 `
  -ExtensionId "<EXT_ID>" `
  -Browser both `
  -RuntimeInstallRoot "D:\Tools\SciCode\runtime" `
  -PythonCommand "C:\Python312\python.exe"
```

---

## Python version requirements

v2-improved ships `.py` source files rather than `.pyc` bytecode. Any Python ≥ 3.9 works.
The installer creates a venv using whichever `python3` / `python` is on the machine — no
manual version matching required.
