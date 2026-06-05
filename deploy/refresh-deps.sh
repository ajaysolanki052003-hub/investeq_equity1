#!/usr/bin/env bash
# Push the current requirements.txt and reinstall on the VM, then restart units.
# Use after adding a new dep without doing a full repo re-transfer.

set -euo pipefail

VM_HOST="34.93.70.239"
VM_USER="ajay"
VM_TARGET="${VM_USER}@${VM_HOST}"
VM_PATH="/home/${VM_USER}/investeq_ajs"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/gcp_ajay}"

if [[ -f "$SSH_KEY" ]]; then
    SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new
              -o IdentitiesOnly=yes -i "$SSH_KEY")
else
    SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
fi

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Push only requirements.txt
scp "${SSH_OPTS[@]}" \
    "$LOCAL_ROOT/deploy/requirements.txt" \
    "$VM_TARGET:$VM_PATH/deploy/requirements.txt"

# Reinstall + restart whichever units the caller passes (default: all four)
UNITS=("${@:-investeq-portfolio investeq-straddle investeq-chain investeq-scan}")
ssh "${SSH_OPTS[@]}" "$VM_TARGET" "
    set -e
    cd '$VM_PATH'
    .venv/bin/pip install -r deploy/requirements.txt
    sudo systemctl restart ${UNITS[*]}
    sleep 2
    for u in ${UNITS[*]}; do
      printf '  %-30s %s\n' \"\$u\" \"\$(systemctl is-active \"\$u\")\"
    done
"
