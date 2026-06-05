# Chrome Extension (Native Backend)

This is a **separate** extension from `chrome_extension/`.  
It sends notebook content to a local Python backend (Native Messaging) and runs the real validator:

- `src/validator/scicode_notebook_validator.py`

It also applies additional custom checks in the backend:

- strict cell-count checks
- strict metadata-content template checks

## 1) Load extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select `chrome_extension_native/`
5. Copy the extension ID shown in Chrome

## 2) Install native host (Windows)

From PowerShell in `native_messaging_backend/`:

```powershell
.\install_native_host_windows.ps1 -ExtensionId "<YOUR_EXTENSION_ID>"
```

This installs/updates a stable backend registration (default: `%LOCALAPPDATA%\SciCodeNative\backend`) and writes native host manifest + registry entries for the current user.

### Python-only (bytecode) runtime option

If you use the `scicode-notebook-validator-python-runtime-bundle-<os>.zip` artifact, run the runtime installer in `backend/` instead:

```powershell
.\install_python_runtime_windows.ps1 -ExtensionId "<YOUR_EXTENSION_ID>" -Browser both
```

For Edge-only or Chrome-only registration:

```powershell
.\install_native_host_windows.ps1 -ExtensionId "<YOUR_EXTENSION_ID>" -Browser chrome
.\install_native_host_windows.ps1 -ExtensionId "<YOUR_EXTENSION_ID>" -Browser edge
```

## 3) Run

Use popup:

- **Validate Notebook** (pasted URL)
- **Validate Current Tab** (active Colab/Drive page)

The extension fetches notebook JSON in Chrome, sends it to local Python backend, and displays issues.

## 4) Package zip

From `chrome_extension_native/`:

```bash
npm run pack
```

Output:

- `chrome_extension_native/dist/scicode-native-validator-extension.zip`

## CI build bundles

GitHub Actions workflow:

- `.github/workflows/build-native-backend-installers.yml`

Artifacts include OS-specific native backend + extension zip + installer scripts.
