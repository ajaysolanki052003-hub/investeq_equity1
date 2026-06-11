# Investeq — Deployment

One-shot deploy of the portfolio + 3 project apps to a single GCP VM
(`ajay@34.93.70.239`), all gated behind a single login. Re-running any step is
safe; everything is idempotent.

```
        public Internet
              │  http  :80
              ▼
        ┌─────────────┐
        │   nginx     │  (single public port, auth_request gate)
        └──┬──┬──┬──┬─┘
   /       │  │  │  │  /straddle  /chain  /scan
           │  │  │  │
       127.0.0.1:8700  ← portfolio (login page + project hub)
       127.0.0.1:8703  ← NIFTY short-straddle replay  (FastAPI)
       127.0.0.1:8704  ← option-chain replay          (FastAPI)
       127.0.0.1:8501  ← EMA swing-low scanner        (Streamlit)
       (no port)       ← LTP worker — polls Groww every 2s during NSE hours
                        also fetches 1-minute candles into data/1m/
       systemd timers  ← :16 each hour → refresh data/1h/*.csv
                       ← 15:35 IST     → refresh data/1d/*.csv
```

URL: **http://34.93.70.239/** · login: `ajay` / `investeq2026` (rotate before
sharing — see "Security" below).

---

## Quick-start

From a Git Bash on the local Windows machine, in the repo root:

```bash
# Plain redeploy (code + nginx + systemd units)
bash deploy/deploy.sh

# Redeploy AND bring all 500 stocks' 1d+1h CSVs current to today's close
BACKFILL=1 bash deploy/deploy.sh
```

Both commands tail back to the shell — expect ~10-30 s for a no-change deploy,
~6-7 min when `BACKFILL=1` runs the full incremental fetch.

### Prerequisites on this machine

| Tool | Why |
|---|---|
| `ssh`, `tar`, `gzip` | shipped with Git for Windows — already present |
| `rsync` | optional; tar-over-ssh is the automatic fallback |
| `~/.ssh/gcp_ajay` | private key the VM accepts; `deploy.sh` finds it automatically |

### Prerequisites on the VM

The first deploy installs everything for you:
- `nginx`, `python3-venv`, `build-essential`, `rsync` (apt)
- the project venv at `/home/ajay/investeq_ajs/.venv/`
- pinned wheels from `requirements.txt`
- `/etc/investeq.env` (mode 0640, owner root:ajay) with the values from `investeq.env`
- 4 long-running systemd services + 2 timer pairs

---

## File-by-file

### Orchestrators (the only two you invoke by hand)

| File | Role |
|---|---|
| `deploy.sh` | Local-side. Probes VM → tars+ssh's the repo → runs `install.sh` on VM → optional backfill → opens firewall (gcloud if available, else prints manual GCP-console steps) → curl smoke-tests `/login`. Set `BACKFILL=1` to also run `incremental_fetch.py`. |
| `install.sh` | Runs on the VM, invoked by `deploy.sh`. Apt-installs OS deps, creates the venv, runs `pip install -r requirements.txt`, patches `streamlit-aggrid`'s missing `columnProps.json` from GitHub, bootstraps `/etc/investeq.env` with the GROWW creds the first time, installs+enables+restarts every systemd unit, wires the nginx site, prints status. **Idempotent.** |

### Env file (read by every service)

| File | Target |
|---|---|
| `investeq.env` | Template that becomes `/etc/investeq.env` on the VM. Holds the portfolio login (`INVESTEQ_USER` / `INVESTEQ_PASS`), HMAC cookie key (`INVESTEQ_SECRET`), data path, Groww broker creds (`GROWW_TOTP_JWT` / `GROWW_TOTP_SECRET`), and LTP worker tunables (`LTP_POLL_SECONDS`, `LTP_MAX_CONCURRENT`, `INCLUDE_ALL_FALLBACK`). Every systemd unit references it via `EnvironmentFile=`. |

### nginx

| File | Target |
|---|---|
| `nginx-investeq.conf` | `/etc/nginx/sites-available/investeq` (symlinked into `sites-enabled/`). Single `server { listen 80 }` block. Proxies `/` to portfolio; gates `/straddle`, `/chain`, `/scan` with `auth_request /internal/auth` against the portfolio's cookie. WebSocket-friendly for Streamlit and the chart panes. |

### systemd — long-running services

All write to journalctl; all `Restart=on-failure RestartSec=3..5`.

| Unit | Process | Port |
|---|---|---|
| `investeq-portfolio.service` | `uvicorn portfolio_app:app` | 127.0.0.1:8700 |
| `investeq-straddle.service` | `uvicorn app_option_replay_tv:app` (with `APP_BASE=/straddle`) | 127.0.0.1:8703 |
| `investeq-chain.service` | `uvicorn app_chain_replay:app` (with `APP_BASE=/chain`) | 127.0.0.1:8704 |
| `investeq-scan.service` | `streamlit run ema_cross_swing/scan_app.py --server.baseUrlPath /scan` | 127.0.0.1:8501 |
| `investeq-strategy.service` | `uvicorn oi_options_trading.app:app` (with `APP_BASE=/strategy`) | 127.0.0.1:8705 |
| `investeq-blast.service` | `uvicorn expiry_blast.app:app` (with `APP_BASE=/blast`) | 127.0.0.1:8706 |
| `investeq-live-ltp.service` | `python ema_scanner/live_workers.py ltp` — polls Groww `/v1/live-data/ltp` every `LTP_POLL_SECONDS` during NSE hours; sleeps off-hours; piggybacks a 1-minute candle refresh every minute | — writes `data/live/ltp.parquet` and appends to `data/1m/*_historical.csv` |

### systemd — timers

Each timer fires its matching one-shot service. `.timer` is what's enabled; the `.service` never starts on its own.

| Pair | Fires (IST) | What it does |
|---|---|---|
| `investeq-candles-1h.{timer,service}` | Mon-Fri `09:16, 10:16, 11:16, 12:16, 13:16, 14:16, 15:16, 15:31` | `live_workers.py candles --interval 1h` → fetches just-closed hour for every CSV in `data/1h/` |
| `investeq-candles-1d.{timer,service}` | Mon-Fri `15:35` | `live_workers.py candles --interval 1d` → appends today's daily bar to every CSV in `data/1d/` |

### Dependencies

| File | Used by |
|---|---|
| `requirements.txt` | `install.sh`'s `pip install` step. Pinned versions for reproducibility. |

### Helper (unused)

| File | Status |
|---|---|
| `refresh-deps.sh` | Standalone "re-pip + restart" helper. Sandbox blocks the agent from invoking it ("unverifiable contents"), so it sits unused. Safe to delete or invoke manually. |

---

## Maintenance recipes

### Add a new dependency
1. Add the pinned line to `requirements.txt`.
2. `bash deploy/deploy.sh` — `install.sh` re-runs `pip install` and restarts services.

### Change the portfolio password
```bash
ssh ajay@34.93.70.239
sudo nano /etc/investeq.env       # edit INVESTEQ_PASS (and INVESTEQ_SECRET to invalidate active cookies)
sudo systemctl restart investeq-portfolio
```

### Refresh data on demand
```bash
BACKFILL=1 bash deploy/deploy.sh    # one-shot from your laptop
# OR directly on the VM:
ssh ajay@34.93.70.239
cd /home/ajay/investeq_ajs/ema_scanner
set -a; source /etc/investeq.env; set +a
../.venv/bin/python incremental_fetch.py
```

### View live logs
```bash
ssh ajay@34.93.70.239
sudo journalctl -u investeq-live-ltp.service -f         # LTP ticks
sudo journalctl -u investeq-portfolio.service -f        # login attempts etc.
sudo systemctl list-timers 'investeq-*.timer'            # next fire times
```

### Disable a service
```bash
sudo systemctl disable --now investeq-live-ltp.service
```

### Hot-fix a single file without a full redeploy
```bash
scp some-file.py ajay@34.93.70.239:/home/ajay/investeq_ajs/...
ssh ajay@34.93.70.239 'sudo systemctl restart investeq-portfolio'
```

---

## Architecture notes

### One cookie protects everything
The portfolio (`portfolio_app.py`) exposes `/internal/auth` that returns 200 if
the request carries a valid `iq_session` HMAC cookie and 401 otherwise. nginx
calls that endpoint as an `auth_request` subrequest for every `/straddle`,
`/chain`, `/scan` request. 401 triggers `error_page 401 = @login_redirect`,
which 302s the user to `/login`.

### Path-prefix shim for the FastAPI apps
`app_option_replay_tv.py` and `app_chain_replay.py` were written with absolute
`fetch('/api/...')` calls. Mounting them under `/straddle/` / `/chain/` would
have broken those calls. Each app now reads an `APP_BASE` env var at boot and
injects a tiny `window.fetch` wrapper in the served HTML that prepends the
prefix to any `/api/...` URL. Empty in local dev → no-op.

### Streamlit subpath
Streamlit handles its own routing prefix natively via `--server.baseUrlPath
/scan`. nginx forwards the path as-is (no rewrite). WebSocket upgrade headers
are passed through.

### Live data flow (during market hours)
```
                       data/live/watchlist.txt
                              │  (scan_app.py writes the selected symbol)
                              ▼
   Groww /v1/live-data/ltp ──► investeq-live-ltp.service ──► data/live/ltp.parquet
                                       │
                                       └─ every 60s also fetches 1m bars
                                          → data/1m/<SYM>_historical.csv

           data/1m/*.csv + data/live/ltp.parquet
                              │
                              ▼  (read by scan_app.py with 1s cache)
                       _overlay_live_ltp(candles, symbol, tf)
                              │  (builds OHLC from 1m bars in current period,
                              │   bumps close+high+low with the latest LTP)
                              ▼
                       last candle on the chart "ticks" every 2s
                       (streamlit-autorefresh reruns the page every 2000ms)
```

### Why 2-second polling is safe
500-stock watchlist is hypothetical. In practice the watchlist contains
only the symbol the user is currently viewing (the UI writes it on every
rerun). So peak load is `1 symbol × 1 request / 2s = 0.5 req/s` — well under
the typical Groww limit of 10 req/s. Set `INCLUDE_ALL_FALLBACK=1` to track all
500 from the active list instead; that pushes the budget to ~5 req/s with
8-way concurrent fetch.

---

## Security

The repository **currently contains a leaked Groww `TOTP_JWT` and
`TOTP_SECRET`** in `deploy/install.sh` (the bootstrap block). It also
historically lived in `ema_scanner/incremental_fetch.py` and
`DATA/README.md` — both have been stripped, but the bootstrap block is the
mechanism by which `/etc/investeq.env` gets populated on first install.

**To-do when convenient:**
1. Log into the Groww developer dashboard.
2. Revoke the existing API key, issue a new pair.
3. SSH to the VM:
   ```bash
   sudo nano /etc/investeq.env       # paste new GROWW_TOTP_JWT and _SECRET
   sudo systemctl restart investeq-live-ltp
   ```
4. Delete the entire `EOFENV` heredoc from `deploy/install.sh` so the next
   redeploy doesn't re-write the old creds back in.

The portfolio login (`INVESTEQ_USER` / `INVESTEQ_PASS`) is similarly the default
in `investeq.env`. Rotate both before sharing the URL with anyone outside your
desk.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/scan/` throws `FileNotFoundError: columnProps.json` | `streamlit-aggrid` wheel was upgraded without the GitHub patch | `bash deploy/deploy.sh` re-applies the curl-patch idempotently |
| `/scan/` chart doesn't tick during market hours | LTP worker not running, or watchlist file empty | `sudo journalctl -u investeq-live-ltp -n 50`; verify `/etc/investeq.env` has GROWW creds; refresh the scanner UI so it writes `data/live/watchlist.txt` |
| `502 Bad Gateway` from nginx | one of the upstream services hasn't started yet | `sudo systemctl status investeq-portfolio` (or whichever path 502'd); `restart` if needed |
| Hourly timer didn't fire | clock drift, or units not enabled | `sudo systemctl list-timers 'investeq-*.timer'`; if missing, `bash deploy/deploy.sh` reinstalls + enables them |
| Backfill says `KeyError: 'GROWW_TOTP_JWT'` | env not exported into the subshell | `set -a; source /etc/investeq.env; set +a` before running the script |
| `Permission denied (publickey)` on local SSH | the `~/.ssh/gcp_ajay` private key isn't being picked up | the deploy script auto-detects it; if your key is elsewhere, run with `SSH_KEY=~/.ssh/your-key bash deploy/deploy.sh` |

For anything else, `journalctl -u <unit> -n 100 --no-pager` on the VM is the
canonical first step.
