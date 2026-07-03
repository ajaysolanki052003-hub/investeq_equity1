#!/usr/bin/env bash
# Run on the VM (as ajay) after rsync finishes:
#   bash deploy/install.sh
#
# Idempotent: re-running it just refreshes deps and reloads units.

set -euo pipefail

ROOT="/home/ajay/investeq_ajs"
DEPLOY="$ROOT/deploy"

if [[ ! -d "$ROOT" ]]; then
    echo "Expected repo at $ROOT — did rsync land elsewhere?" >&2
    exit 1
fi
cd "$ROOT"

# ── 1. OS packages ──────────────────────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    nginx python3-venv python3-pip build-essential rsync

# ── 2. Python venv + deps ───────────────────────────────────────────────────
if [[ ! -d "$ROOT/.venv" ]]; then
    python3 -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/pip" install --upgrade pip
"$ROOT/.venv/bin/pip" install -r "$DEPLOY/requirements.txt"

# ── 2b. Patch streamlit-aggrid wheels that ship without json/columnProps.json
#        (recent PyPI wheels are incomplete; the source repo has the file).
ST_AGGRID="$ROOT/.venv/lib/python3.11/site-packages/st_aggrid"
ST_AGGRID_JSON="$ST_AGGRID/json"
if [[ -d "$ST_AGGRID" && ! -f "$ST_AGGRID_JSON/columnProps.json" ]]; then
    echo ">> Patching missing st_aggrid/json/columnProps.json from GitHub"
    mkdir -p "$ST_AGGRID_JSON"
    curl -fsSL -o "$ST_AGGRID_JSON/columnProps.json" \
        "https://raw.githubusercontent.com/PablocFonseca/streamlit-aggrid/main/st_aggrid/json/columnProps.json" \
        || echo "WARN: could not patch columnProps.json; /scan/ may still error"
fi

# ── 3. /etc/investeq.env (only created if missing — never overwritten) ─────
if [[ ! -f /etc/investeq.env ]]; then
    sudo install -m 0640 -o root -g ajay "$DEPLOY/investeq.env" /etc/investeq.env
    echo ">> /etc/investeq.env installed — EDIT IT to set a real password/secret"
else
    echo ">> /etc/investeq.env already exists — left untouched"
fi

# Bootstrap GROWW creds if /etc/investeq.env doesn't already have a non-empty
# GROWW_TOTP_JWT. The values below are the ones the user had hard-coded in
# incremental_fetch.py; ROTATE them at Groww and overwrite /etc/investeq.env
# manually when convenient.
if ! sudo grep -q "^GROWW_TOTP_JWT=." /etc/investeq.env; then
    echo ">> Bootstrapping GROWW creds into /etc/investeq.env (rotate at broker after)"
    sudo tee -a /etc/investeq.env >/dev/null <<'EOFENV'
GROWW_TOTP_JWT=eyJraWQiOiJaTUtjVXciLCJhbGciOiJFUzI1NiJ9.eyJleHAiOjI1NjM0Mjk3MzUsImlhdCI6MTc3NTAyOTczNSwibmJmIjoxNzc1MDI5NzM1LCJzdWIiOiJ7XCJ0b2tlblJlZklkXCI6XCI0YTlkN2RmZS04NDIwLTQzNWUtOWM1YS03OWQ4ODFjMGZmMWVcIixcInZlbmRvckludGVncmF0aW9uS2V5XCI6XCJlMzFmZjIzYjA4NmI0MDZjODg3NGIyZjZkODQ5NTMxM1wiLFwidXNlckFjY291bnRJZFwiOlwiOTZiMDEyZDktN2FkYi00N2UyLTkyYWMtNTA3Y2VhYjg5OGNlXCIsXCJkZXZpY2VJZFwiOlwiZDQ3YTE3OTktYjIwYi01Y2E2LTgyYmQtOGU0YTllZTQ0YmNiXCIsXCJzZXNzaW9uSWRcIjpcIjFiMjk3Yzg4LTM3MmUtNGFkYS05MGYyLTMyYjEyNDhlOWRhMlwiLFwiYWRkaXRpb25hbERhdGFcIjpcIno1NC9NZzltdjE2WXdmb0gvS0EwYk9XdVdGbGx0enh6Sk1menE4MzZ2d1ZSTkczdTlLa2pWZDNoWjU1ZStNZERhWXBOVi9UOUxIRmtQejFFQisybTdRPT1cIixcInJvbGVcIjpcImF1dGgtdG90cFwiLFwic291cmNlSXBBZGRyZXNzXCI6XCIyNDA1OjIwMToyMDA5OjIyMzg6YzkzMDo2NWY4OjliZmY6OWM5ZCwxNzIuNzEuMTk4LjE3OCwzNS4yNDEuMjMuMTIzXCIsXCJ0d29GYUV4cGlyeVRzXCI6MjU2MzQyOTczNTk2NixcInZlbmRvck5hbWVcIjpcImdyb3d3QXBpXCJ9IiwiaXNzIjoiYXBleC1hdXRoLXByb2QtYXBwIn0.Nz3h58nxFwLTN2kGXtxoeEXJhXhIKwq0UMbwTl8gDttM5YKHbJ4haRH0okz_qSe42cIOB4kfBq3-fDHwraa2SA
GROWW_TOTP_SECRET=TMDX5YFXBDRSHQKRHBSHQ3SMFOFWK2QB
EOFENV
fi

# ── 4. systemd units ────────────────────────────────────────────────────────
for u in portfolio straddle chain scan live-ltp live-strategy optlab resistance vwap levels screener blast strategy sector sector2 basket basket-live candles-1h candles-1d candles-15m candles-1m; do
    if [[ -f "$DEPLOY/investeq-$u.service" ]]; then
        sudo install -m 0644 "$DEPLOY/investeq-$u.service" "/etc/systemd/system/investeq-$u.service"
    fi
    if [[ -f "$DEPLOY/investeq-$u.timer" ]]; then
        sudo install -m 0644 "$DEPLOY/investeq-$u.timer" "/etc/systemd/system/investeq-$u.timer"
    fi
done
sudo systemctl daemon-reload

# Long-running services. `enable` is no-op after first install; `restart` picks
# up new code/deps on every redeploy.
for u in portfolio straddle chain scan live-ltp live-strategy optlab resistance vwap levels screener blast strategy sector sector2 basket; do
    sudo systemctl enable "investeq-$u.service" 2>/dev/null || true
    sudo systemctl restart "investeq-$u.service"
done
# Timers (the .service files attached are one-shot, started by the timer)
for u in candles-1h candles-1d candles-15m candles-1m basket-live; do
    sudo systemctl enable --now "investeq-$u.timer"
done

# ── 5. nginx site ──────────────────────────────────────────────────────────
sudo install -m 0644 "$DEPLOY/nginx-investeq.conf" /etc/nginx/sites-available/investeq
sudo ln -sf /etc/nginx/sites-available/investeq /etc/nginx/sites-enabled/investeq
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

# ── 6. Status ──────────────────────────────────────────────────────────────
echo
echo "==== unit status ===="
sudo systemctl --no-pager status investeq-portfolio investeq-straddle investeq-chain investeq-scan investeq-live-ltp 2>&1 | head -80
echo
sudo systemctl --no-pager list-timers 'investeq-*.timer'

echo
echo ">> Open http://$(curl -s ifconfig.me)/ to log in."
