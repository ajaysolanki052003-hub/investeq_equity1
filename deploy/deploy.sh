#!/usr/bin/env bash
# One-shot deployer for Investeq portfolio + 3 project apps to ajay@34.93.70.239.
#
# Run from the repo root:    bash deploy/deploy.sh
# Or from this Claude session: ! bash deploy/deploy.sh
#
# The script is idempotent — re-running it just refreshes code+deps and reloads
# services. Data is rsync'd with --update so already-present parquets are skipped.

set -euo pipefail

VM_HOST="${VM_HOST:-34.14.172.202}"
VM_USER="ajay"
VM_TARGET="${VM_USER}@${VM_HOST}"
VM_PATH="/home/${VM_USER}/investeq_ajs"

# Identity file. ~/.ssh/gcp_ajay is what we found earlier; user can override.
SSH_KEY="${SSH_KEY:-$HOME/.ssh/gcp_ajay}"
if [[ ! -f "$SSH_KEY" ]]; then
    # fall back to the default identity selection
    SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
else
    SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new
              -o IdentitiesOnly=yes -i "$SSH_KEY")
fi

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LOCAL_ROOT"

# ANSI helpers
c_blue() { printf '\033[1;34m%s\033[0m\n' "$*"; }
c_grn()  { printf '\033[1;32m%s\033[0m\n' "$*"; }
c_yel()  { printf '\033[1;33m%s\033[0m\n' "$*"; }
c_red()  { printf '\033[1;31m%s\033[0m\n' "$*" >&2; }

# ── 0. Tooling ───────────────────────────────────────────────────────────────
for tool in ssh tar gzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        c_red "FATAL: '$tool' not found on PATH."
        exit 1
    fi
done
# rsync is preferred (deltas) but tar-over-ssh is a fine fallback
USE_RSYNC=0
command -v rsync >/dev/null 2>&1 && USE_RSYNC=1

# ── 1. Probe ─────────────────────────────────────────────────────────────────
c_blue "[1/6] Probing ${VM_TARGET} ..."
ssh "${SSH_OPTS[@]}" "$VM_TARGET" '
        echo "  who    : $(whoami)"
        echo "  os     : $(. /etc/os-release && echo "$PRETTY_NAME")"
        echo "  kernel : $(uname -r)"
        echo "  python : $(python3 --version 2>&1)"
        echo "  nginx  : $(which nginx 2>/dev/null || echo not installed)"
        echo "  free   : $(df -h /home | awk "NR==2 {print \$4 \" free of \" \$2}")"
        echo "  mem    : $(free -h | awk "/Mem:/ {print \$7 \" available of \" \$2}")"
    '
echo

# ── 2. Sanitize ──────────────────────────────────────────────────────────────
# DATA/README.md has a Groww JWT on lines 1-3 — exclude it from the upload.
# Apps don't read it, so the server just won't have that doc file.
c_blue "[2/6] Excluding secrets from upload (DATA/README.md, *.log, __pycache__, etc.)"
echo

# ── 3. Transfer code + data ──────────────────────────────────────────────────
EXCLUDES=(
    --exclude='./.git'
    --exclude='./.venv'
    --exclude='./.claude'
    --exclude='./__pycache__'
    --exclude='*.pyc'
    --exclude='*.log'
    --exclude='./turing'
    --exclude='./NEW_FOLDER'
    --exclude='./DATA/README.md'
    --exclude='./_regen_straddle_pnl.py'
)

if (( USE_RSYNC )); then
    c_blue "[3/6] Rsyncing repo to ${VM_TARGET}:${VM_PATH} ..."
    rsync -avzh --info=progress2 --no-i-r --update \
        "${EXCLUDES[@]/--exclude=./--exclude=}" \
        -e "ssh ${SSH_OPTS[*]}" \
        "$LOCAL_ROOT/" "$VM_TARGET:$VM_PATH/"
else
    c_blue "[3/6] tar | ssh transfer of repo to ${VM_TARGET}:${VM_PATH} ..."
    c_yel "      (rsync not on PATH — using single-shot tar pipe. ~2.2 GB.)"
    ssh "${SSH_OPTS[@]}" "$VM_TARGET" "mkdir -p '$VM_PATH'"
    # -C cd to repo root, '.' archives everything, --exclude works with leading ./
    tar -czf - -C "$LOCAL_ROOT" "${EXCLUDES[@]}" . \
        | ssh "${SSH_OPTS[@]}" "$VM_TARGET" "tar -xzf - -C '$VM_PATH'"
fi
c_grn "      transfer complete."
echo

# ── 4. Install on the VM (idempotent) ────────────────────────────────────────
c_blue "[4/6] Running deploy/install.sh on the VM ..."
ssh "${SSH_OPTS[@]}" "$VM_TARGET" \
    "chmod +x $VM_PATH/deploy/install.sh && bash $VM_PATH/deploy/install.sh"
echo

# ── 4b. Optional backfill — bring all 500 stocks' 1d+1h CSVs current. ───────
if [[ "${BACKFILL:-0}" == "1" ]]; then
    c_blue "[4b] Backfilling 1d+1h CSVs up to now ..."
    ssh "${SSH_OPTS[@]}" "$VM_TARGET" "
        set -e
        cd $VM_PATH/ema_scanner
        # source env so the script sees GROWW_TOTP_*
        set -a; . /etc/investeq.env; set +a
        $VM_PATH/.venv/bin/python incremental_fetch.py 2>&1 | tail -40
    "
    echo
fi

# ── 5. Firewall (try gcloud, fall back to manual instructions) ───────────────
c_blue "[5/6] GCP firewall — opening tcp:80 to 0.0.0.0/0 ..."
if command -v gcloud >/dev/null 2>&1; then
    # Try to detect the instance's network tag. Best-effort.
    if gcloud compute firewall-rules describe investeq-http >/dev/null 2>&1; then
        c_yel "      Rule 'investeq-http' already exists. Skipping."
    else
        if gcloud compute firewall-rules create investeq-http \
                --direction=INGRESS \
                --action=ALLOW \
                --rules=tcp:80 \
                --source-ranges=0.0.0.0/0 \
                --description='Investeq portfolio (HTTP, gated by login)'; then
            c_grn "      Firewall rule 'investeq-http' created."
        else
            c_yel "      gcloud failed — open it manually in the GCP console:"
            c_yel "      VPC network → Firewall → CREATE FIREWALL RULE"
            c_yel "        Name: investeq-http   Targets: All   Source: 0.0.0.0/0   Protocols/ports: tcp:80"
        fi
    fi
else
    c_yel "      gcloud not installed locally. Open this manually in the GCP console:"
    c_yel "      VPC network → Firewall → CREATE FIREWALL RULE"
    c_yel "        Name: investeq-http   Targets: All   Source: 0.0.0.0/0   Protocols/ports: tcp:80"
fi
echo

# ── 6. Smoke-test ────────────────────────────────────────────────────────────
c_blue "[6/6] Smoke test — http://${VM_HOST}/login ..."
sleep 2
HTTP=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://${VM_HOST}/login" || echo "000")
if [[ "$HTTP" == "200" ]]; then
    c_grn "      /login returned 200 — site is live."
else
    c_yel "      /login returned HTTP $HTTP. If 000, firewall isn't open yet; if 502, the systemd units haven't fully started — try again in 30s."
fi
echo
c_grn "Done. Open http://${VM_HOST}/  →  log in as ajay / investeq2026"
c_yel "Then SSH in once and EDIT /etc/investeq.env to rotate INVESTEQ_PASS and INVESTEQ_SECRET before sharing the URL."
