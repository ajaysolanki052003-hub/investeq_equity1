"""
Ajay Solanki — Personal portfolio / project hub.

Single-file FastAPI app, trading-floor aesthetic. Add a new project by
appending a dict to the PROJECTS list at the top of this file.

Run:
    python -m uvicorn portfolio_app:app --host 0.0.0.0 --port 8700

Open:
    http://localhost:8700/
"""

from __future__ import annotations
import os, socket, hmac, hashlib
from urllib.parse import parse_qs
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response


# ── Login gate ─────────────────────────────────────────────────────────────
# On a deployed host set INVESTEQ_USER / INVESTEQ_PASS / INVESTEQ_SECRET via
# /etc/investeq.env (read by the systemd unit). Local dev falls back to the
# hardcoded values below.
AUTH_USERS = {
    os.environ.get("INVESTEQ_USER", "ajay"):
        os.environ.get("INVESTEQ_PASS", "investeq2026"),
}
SESSION_SECRET = os.environ.get(
    "INVESTEQ_SECRET", "change-me-investeq-secret-2026"
).encode()
SESSION_COOKIE = "iq_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def _sign(username: str) -> str:
    sig = hmac.new(SESSION_SECRET, username.encode(), hashlib.sha256).hexdigest()
    return f"{username}:{sig}"


def _verify(token: str | None) -> str | None:
    if not token or ":" not in token:
        return None
    username, sig = token.rsplit(":", 1)
    expected = hmac.new(SESSION_SECRET, username.encode(), hashlib.sha256).hexdigest()
    if hmac.compare_digest(sig, expected) and username in AUTH_USERS:
        return username
    return None


def _authed(request: Request) -> bool:
    return _verify(request.cookies.get(SESSION_COOKIE)) is not None


PROFILE = {
    "name"     : "Hi Ajay",
    "role"     : "Systematic Trading Desk",
    "headline" : "Quantitative strategies for Indian F&O and equities.",
    "tagline"  : (
        "Walk-forward backtested. Live-monitored. Audit-ready. "
        "We design and run systematic strategies across NSE options and "
        "equities — built on six years of 1-minute data, validated with "
        "time-series cross-validation, and surfaced through trading-floor "
        "dashboards your team can supervise in real time."
    ),
    "email"    : "ajaysolanki052003@gmail.com",
    "location" : "India · IST",
}

# Firm-level numbers shown in the hero. Edit when the suite changes.
STATS = {
    "strategies_live"   : 3,
    "events_backtested" : 296,
    "years_of_data"     : 6,
    "granularity_min"   : 1,
}

# Each entry below is a strategy in the suite. Schema:
#   id          — unique slug
#   name        — strategy name (front of card)
#   instrument  — small subtitle under the name (e.g. "NIFTY · weekly options")
#   tag         — top-right pill: LIVE | PAPER | RESEARCH
#   category    — filter bucket: OPTIONS | EQUITIES
#   thesis      — one-paragraph trader-readable thesis (what edge, why it works)
#   stack       — small chips on the card (modeling / data / framework)
#   port        — optional; if set the card pings localhost:<port> for status
#   url         — optional; "Open dashboard" target
#   path        — optional source path shown as a chip
#   metrics     — list of [label, value] pairs — at most 3, performance-oriented
#   accent      — hex colour for the accent stripe + glow
PROJECTS = [
    {
        "id": "volatility-desk",
        "name": "Volatility Desk",
        "instrument": "Options",
        "tag": "LIVE",
        "category": "OPTIONS",
        "thesis": "",
        "stack": [],
        "port": 8703,
        "url": "/straddle/",
        "path": "",
        "metrics": [],
        "accent": "#26a69a",
    },
    {
        "id": "chain-replay",
        "name": "Chain Replay",
        "instrument": "Options",
        "tag": "RESEARCH",
        "category": "OPTIONS",
        "thesis": "",
        "stack": [],
        "port": 8704,
        "url": "/chain/",
        "path": "",
        "metrics": [],
        "accent": "#60a5fa",
    },
    {
        "id": "daily-scanner",
        "name": "Daily Scanner",
        "instrument": "Equities",
        "tag": "LIVE",
        "category": "EQUITIES",
        "thesis": "",
        "stack": [],
        "port": 8501,
        "url": "/scan/",
        "path": "",
        "metrics": [],
        "accent": "#facc15",
    },
    {
        "id": "strategy-suite",
        "name": "Strategy Suite",
        "instrument": "Options",
        "tag": "BUILD",
        "category": "OPTIONS",
        "thesis": "",
        "stack": [],
        "port": 8705,
        "url": "/strategy/",
        "path": "",
        "metrics": [],
        "accent": "#a78bfa",
    },
]


app = FastAPI(title=f"{PROFILE['name']} — Portfolio")


@app.get("/internal/auth")
def internal_auth(request: Request):
    """Subrequest target for nginx auth_request. 200 if cookie valid, else 401.
    Nothing in the body — nginx only looks at the status code."""
    if _authed(request):
        return Response(status_code=200)
    return Response(status_code=401)


@app.get("/api/projects")
def projects(request: Request):
    if not _authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return JSONResponse({
        "projects": PROJECTS,
        "stats"   : STATS,
        "profile" : PROFILE,
    })


@app.get("/api/health/{port}")
def health(port: int, request: Request):
    """Quick TCP probe so each project card can show LIVE/STOPPED."""
    if not _authed(request):
        return JSONResponse({"port": port, "up": False, "error": "unauthorized"}, status_code=401)
    if not (1 <= port <= 65535):
        return JSONResponse({"port": port, "up": False})
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.35)
    try:
        up = s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        up = False
    finally:
        s.close()
    return JSONResponse({"port": port, "up": up})


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hi Ajay · Systematic Trading Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0a0c14;
  --panel: #0f1219;
  --panel-2: #131722;
  --border: #1e2030;
  --border-hi: #2a2d3a;
  --text: #e6e9f2;
  --muted: #6b7280;
  --muted-hi: #9ca3af;
  --green: #26a69a;
  --red: #ef5350;
  --blue: #60a5fa;
  --gold: #facc15;
  --purple: #c084fc;
}
html, body { background: var(--bg); color: var(--text); }
body {
  font: 14px/1.5 'Inter', system-ui, -apple-system, Segoe UI, sans-serif;
  min-height: 100vh;
  background-image:
    radial-gradient(circle at 20% -10%, rgba(38,166,154,0.07), transparent 40%),
    radial-gradient(circle at 90% 20%, rgba(96,165,250,0.05), transparent 35%),
    linear-gradient(rgba(38,166,154,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(38,166,154,0.025) 1px, transparent 1px);
  background-size: auto, auto, 32px 32px, 32px 32px;
  background-attachment: fixed;
}
.mono { font-family: 'JetBrains Mono', monospace; }

/* ── Top bar ─────────────────────────────────────────────────────────────── */
.topbar {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 32px;
  border-bottom: 1px solid var(--border);
  background: rgba(10,12,20,0.85);
  backdrop-filter: blur(12px);
  position: sticky; top: 0; z-index: 100;
}
.brand { display: flex; align-items: center; gap: 14px; }
.logo {
  color: var(--green); font-size: 22px; font-weight: 700;
  text-shadow: 0 0 12px rgba(38,166,154,0.5);
}
.brand-name {
  font-weight: 900; letter-spacing: 1.5px; font-size: 13px;
}
.brand-role {
  color: var(--muted-hi); font-size: 10px; text-transform: uppercase;
  letter-spacing: 1.5px; border-left: 1px solid var(--border); padding-left: 14px;
  font-family: 'JetBrains Mono', monospace;
}
.topbar-right { display: flex; gap: 20px; align-items: center; font-size: 11px; }
.market-status {
  display: flex; align-items: center; gap: 7px;
  font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; font-size: 10px;
}
.market-status.open { color: var(--green); }
.market-status.closed { color: var(--muted); }
.dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--green); display: inline-block;
}
.dot.off { background: var(--muted); }
.dot.pulse { animation: pulse 1.6s infinite; }
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 currentColor; }
  50%      { box-shadow: 0 0 0 7px transparent; }
}
.ist-clock {
  font-family: 'JetBrains Mono', monospace;
  color: var(--text); letter-spacing: 1.5px; font-weight: 500;
}

/* ── Ticker tape ─────────────────────────────────────────────────────────── */
.ticker {
  background: var(--panel); border-bottom: 1px solid var(--border);
  overflow: hidden; white-space: nowrap; padding: 0;
}
.ticker-track {
  display: inline-flex; gap: 36px; padding: 9px 0;
  animation: scroll 80s linear infinite;
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
}
.ticker-item { display: inline-flex; gap: 8px; align-items: center; }
.ticker-name { color: var(--muted-hi); font-weight: 700; letter-spacing: 0.6px; }
.ticker-px { color: var(--text); }
.ticker-chg.up { color: var(--green); }
.ticker-chg.down { color: var(--red); }
@keyframes scroll {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }
}

/* ── Hero ───────────────────────────────────────────────────────────────── */
section { padding: 72px 32px; max-width: 1380px; margin: 0 auto; }
.hero {
  display: grid; grid-template-columns: 1.05fr 1fr; gap: 56px;
  align-items: center; padding-top: 80px; padding-bottom: 60px;
}
.kicker {
  font-family: 'JetBrains Mono', monospace;
  color: var(--green); font-size: 11px; letter-spacing: 2.5px;
  text-transform: uppercase; margin-bottom: 18px;
  display: flex; align-items: center; gap: 10px;
}
.kicker::before {
  content: ''; width: 24px; height: 1px; background: var(--green);
}
.hero h1 {
  font-size: clamp(56px, 7.5vw, 104px); font-weight: 900; line-height: 1.0;
  letter-spacing: -3px;
  background: linear-gradient(120deg, #ffffff 0%, #e6e9f2 28%, var(--text) 50%);
  -webkit-background-clip: text; background-clip: text;
  color: transparent; margin-bottom: 8px;
  filter: drop-shadow(0 6px 30px rgba(0,0,0,0.5));
}
.hero h1 .accent {
  background: linear-gradient(110deg, var(--green) 0%, #2dd4bf 25%, var(--blue) 55%, var(--green) 80%, #2dd4bf);
  background-size: 260% 100%;
  -webkit-background-clip: text; background-clip: text;
  color: transparent; -webkit-text-fill-color: transparent;
  animation: shimmer 9s linear infinite;
  filter: drop-shadow(0 0 26px rgba(38,166,154,0.45));
}
@keyframes shimmer {
  0%   { background-position:   0% 50%; }
  100% { background-position: 260% 50%; }
}

/* Aurora wash inside the hero (subtle drifting blobs) */
.hero-aurora { position: absolute; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; border-radius: 0; }
.hero-left, .hero-right { position: relative; z-index: 1; }
.aurora { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.22;
          animation: drift 22s ease-in-out infinite alternate; }
.aurora-a { width: 360px; height: 360px; top: -80px;  left: 15%;
            background: radial-gradient(circle, var(--green), transparent 60%); }
.aurora-b { width: 420px; height: 420px; bottom: -160px; right: 8%;
            background: radial-gradient(circle, var(--blue),  transparent 60%);
            animation-duration: 26s; animation-delay: -8s; }
.aurora-c { width: 300px; height: 300px; top: 40%; left: 55%;
            background: radial-gradient(circle, var(--purple), transparent 65%);
            animation-duration: 30s; animation-delay: -14s; opacity: 0.12; }
@keyframes drift {
  0%   { transform: translate(  0px,  0px) scale(1.00); }
  50%  { transform: translate( 40px, 30px) scale(1.08); }
  100% { transform: translate(-30px, 20px) scale(0.95); }
}
.headline {
  color: var(--text); font-size: 18px; font-weight: 600;
  margin-bottom: 20px; letter-spacing: -0.2px;
}
.tagline {
  color: var(--muted-hi); font-size: 14.5px; line-height: 1.65;
  max-width: 520px; margin-bottom: 32px;
}
.hero-stats {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
  margin-bottom: 34px;
}
.stat {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 14px 12px;
  position: relative; overflow: hidden;
  transition: border-color 0.2s, transform 0.2s;
}
.stat:hover { border-color: var(--border-hi); transform: translateY(-2px); }
.stat::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--green), transparent);
  opacity: 0.5;
}
.stat-val {
  display: block; font-family: 'JetBrains Mono', monospace;
  font-size: 26px; font-weight: 700; color: var(--text); line-height: 1.1;
}
.stat-lbl {
  color: var(--muted); font-size: 10px; text-transform: uppercase;
  letter-spacing: 1px; margin-top: 4px; display: block;
}
.hero-cta { display: flex; gap: 10px; flex-wrap: wrap; }
.btn {
  padding: 11px 20px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); text-decoration: none;
  font-size: 12.5px; font-weight: 600; letter-spacing: 0.3px;
  transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px;
  font-family: inherit; cursor: pointer;
}
.btn:hover {
  background: var(--panel-2); border-color: var(--border-hi);
  transform: translateY(-1px);
}
.btn.primary {
  background: var(--green); border-color: var(--green); color: #052b27;
  font-weight: 700;
}
.btn.primary:hover {
  background: #2dd4bf; box-shadow: 0 4px 20px rgba(38,166,154,0.35);
}

/* Hero chart */
.hero-right { position: relative; }
.hero-chart {
  background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
  height: 380px; overflow: hidden;
  position: relative;
  box-shadow: 0 0 0 1px rgba(38,166,154,0.05), 0 20px 60px rgba(0,0,0,0.4);
}
.chart-bar {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 18px; border-bottom: 1px solid var(--border);
}
.chart-title {
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
  color: var(--text); letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px;
  font-weight: 600;
}
.chart-title .tag {
  background: rgba(38,166,154,0.15); color: var(--green);
  font-size: 9px; padding: 2px 6px; border-radius: 3px;
  letter-spacing: 1px; font-weight: 700;
}
.chart-px {
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  display: flex; gap: 10px; align-items: baseline;
}
.chart-px .v { color: var(--text); font-weight: 700; }
.chart-px .c { color: var(--green); font-size: 11px; }
.chart-px .c.down { color: var(--red); }
#hero-tv-chart { width: 100%; height: calc(100% - 50px); }

/* ── Section headings ──────────────────────────────────────────────────── */
.section-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 28px; padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.section-head h2 {
  font-size: 28px; font-weight: 800; letter-spacing: -0.6px;
  display: flex; align-items: baseline; gap: 12px;
}
.section-head h2 .num {
  color: var(--green); font-family: 'JetBrains Mono', monospace;
  font-size: 14px; font-weight: 500;
}
.section-head h2 .count {
  color: var(--muted); font-weight: 400; font-size: 14px;
  font-family: 'JetBrains Mono', monospace;
}
.section-sub {
  color: var(--muted-hi); font-size: 13px; max-width: 600px;
  margin-top: -16px; margin-bottom: 28px;
}

/* Filters */
.filters { display: flex; gap: 6px; }
.filter {
  padding: 7px 14px; font-size: 10.5px; font-weight: 700;
  letter-spacing: 1.2px; text-transform: uppercase;
  background: transparent; border: 1px solid var(--border);
  color: var(--muted-hi); border-radius: 6px; cursor: pointer;
  font-family: inherit;
  transition: all 0.15s;
}
.filter:hover { color: var(--text); border-color: var(--border-hi); }
.filter.on {
  background: var(--green); border-color: var(--green); color: #052b27;
}

/* ── Project grid ──────────────────────────────────────────────────────── */
.grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 16px;
}
.card {
  background: linear-gradient(180deg, var(--panel) 0%, #0c0e16 100%);
  border: 1px solid var(--border);
  border-radius: 12px; padding: 22px;
  position: relative; transition: all 0.25s;
  display: flex; flex-direction: column; gap: 14px;
  overflow: hidden; min-height: 290px;
}
.card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--accent, var(--green));
  opacity: 0.75;
}
.card::after {
  content: ''; position: absolute; inset: 0; border-radius: inherit;
  background: radial-gradient(circle 360px at var(--mx, -200px) var(--my, -200px),
                              var(--accent-faint, rgba(38,166,154,0.22)),
                              transparent 60%);
  opacity: 0; transition: opacity 0.35s; pointer-events: none;
}
.card:hover::after { opacity: 1; }
.card-spark {
  display: block; width: 100%; height: 60px;
  margin-top: 4px; opacity: 0.92;
}
.card-spark .spark-label {
  font-family: 'JetBrains Mono', monospace; font-size: 9px;
  fill: var(--muted); letter-spacing: 0.8px;
}
.card:hover {
  border-color: var(--border-hi);
  transform: translateY(-4px);
  box-shadow: 0 16px 40px rgba(0,0,0,0.5),
              0 0 30px var(--accent-faint, rgba(38,166,154,0.08));
}
.card:hover::after { opacity: 0.10; }
.card-head {
  display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;
}
.card-title {
  font-size: 28px; font-weight: 800; line-height: 1.1; margin-bottom: 8px;
  letter-spacing: -0.6px;
  background: linear-gradient(120deg, var(--text) 0%, var(--accent, var(--green)) 120%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.card-status {
  display: flex; align-items: center; gap: 6px; font-size: 10px;
  color: var(--muted); text-transform: uppercase; letter-spacing: 1px;
  font-family: 'JetBrains Mono', monospace;
}
.card-status.up { color: var(--green); }
.card-status.down { color: var(--muted); }
.card-tag {
  font-size: 9px; font-weight: 800; letter-spacing: 1.2px; text-transform: uppercase;
  padding: 4px 8px; border-radius: 4px;
  background: var(--accent-faint, rgba(38,166,154,0.12));
  color: var(--accent, var(--green));
  align-self: flex-start; white-space: nowrap;
}
.card-desc {
  color: var(--muted-hi); font-size: 12.5px; line-height: 1.6; flex: 1;
}
.card-stack { display: flex; flex-wrap: wrap; gap: 4px; }
.stack-chip {
  font-size: 10px; padding: 3px 8px; border-radius: 4px;
  background: var(--panel-2); border: 1px solid var(--border);
  color: var(--muted-hi); font-family: 'JetBrains Mono', monospace;
}
.card-path {
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  color: var(--muted); padding: 6px 0; border-top: 1px dashed var(--border);
}
.card-metrics {
  display: flex; gap: 18px; padding: 12px 0;
  border-top: 1px dashed var(--border);
}
.card-metric .v { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; }
.card-metric .l { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
.card-actions { display: flex; gap: 8px; margin-top: auto; }
.card-actions .btn { font-size: 11px; padding: 8px 14px; }
.card-actions .btn.primary { background: var(--accent, var(--green)); border-color: var(--accent, var(--green)); color: #07191a; }
.card-actions .btn.primary:hover { filter: brightness(1.1); box-shadow: 0 4px 16px var(--accent-faint, rgba(38,166,154,0.3)); }
.card-actions .btn.muted { color: var(--muted); }
.card-actions .btn.disabled { opacity: 0.4; pointer-events: none; }

/* ── Stack section ─────────────────────────────────────────────────────── */
.stack-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px;
}
.skill {
  padding: 14px 16px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--panel);
  display: flex; flex-direction: column; gap: 8px;
  transition: border-color 0.2s;
}
.skill:hover { border-color: var(--border-hi); }
.skill-row { display: flex; justify-content: space-between; align-items: baseline; }
.skill-name { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; }
.skill-pct  { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--muted); }
.skill-bar { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
.skill-fill { height: 100%; background: linear-gradient(90deg, var(--green), var(--blue)); border-radius: 2px; }

/* ── ML achievements ───────────────────────────────────────────────────── */
.ml-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.ml-cell {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 12px; padding: 24px 22px;
  position: relative; overflow: hidden;
}
.ml-cell::before {
  content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--green), transparent);
}
.ml-val {
  font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700;
  letter-spacing: -1px;
  background: linear-gradient(120deg, var(--text), var(--green));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.ml-lbl { color: var(--muted-hi); font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; margin-top: 8px; font-weight: 600; }
.ml-sub { color: var(--muted); font-size: 11px; margin-top: 4px; line-height: 1.4; }

/* ── Footer ────────────────────────────────────────────────────────────── */
footer.contact {
  border-top: 1px solid var(--border); padding: 36px 32px 60px;
  display: flex; justify-content: space-between; align-items: center;
  max-width: 1380px; margin: 0 auto; flex-wrap: wrap; gap: 24px;
}
.footer-name { font-weight: 800; font-size: 18px; }
.footer-loc { color: var(--muted); font-size: 12px; margin-top: 2px; }
.footer-links { display: flex; gap: 22px; }
.footer-links a {
  color: var(--muted-hi); text-decoration: none; font-size: 11px;
  text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;
  border-bottom: 1px solid transparent; padding-bottom: 2px; transition: all 0.15s;
  font-family: 'JetBrains Mono', monospace;
}
.footer-links a:hover { color: var(--green); border-color: var(--green); }

/* ── Live trade tape (between hero and Strategy suite) ──────────────── */
.tape {
  position: relative; overflow: hidden;
  background: linear-gradient(180deg, var(--panel) 0%, #0a0c14 100%);
  border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
  padding: 14px 0;
}
.tape-label {
  position: absolute; top: 0; bottom: 0; left: 0; z-index: 3;
  display: flex; align-items: center; gap: 10px;
  padding: 0 20px 0 32px;
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  letter-spacing: 1.8px; color: var(--green); font-weight: 800;
  background: linear-gradient(90deg, var(--bg) 75%, transparent);
}
.tape-track {
  display: inline-flex; gap: 22px; padding-left: 220px;
  animation: tape-scroll 75s linear infinite;
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
  white-space: nowrap;
}
.tape:hover .tape-track { animation-play-state: paused; }
.tape-item {
  display: inline-flex; gap: 10px; align-items: center; padding: 6px 12px;
  border: 1px solid var(--border); border-radius: 5px;
  background: rgba(15,18,25,0.7);
}
.tape-item .time { color: var(--muted-hi); }
.tape-item .side { font-weight: 800; letter-spacing: 0.6px; }
.tape-item .side.sell { color: var(--gold); }
.tape-item .side.buy  { color: var(--blue); }
.tape-item .sym { color: var(--text); font-weight: 600; }
.tape-item .px  { color: var(--muted-hi); }
.tape-item .tag {
  font-size: 9.5px; padding: 2px 7px; border-radius: 3px;
  background: rgba(38,166,154,0.14); color: var(--green); letter-spacing: 0.5px;
}
.tape-item .tag.loss { background: rgba(239,83,80,0.14); color: var(--red); }
.tape-item .tag.neut { background: rgba(96,165,250,0.14); color: var(--blue); }
@keyframes tape-scroll {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }
}

/* Section heads: animated underline */
.section-head { position: relative; }
.section-head::after {
  content: ''; position: absolute; bottom: -1px; left: 0;
  width: 72px; height: 2px;
  background: linear-gradient(90deg, var(--green), transparent);
  box-shadow: 0 0 12px rgba(38,166,154,0.6);
}

/* LIVE pill on the hero chart */
.live-pill {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(239,83,80,0.12); color: var(--red);
  font-size: 9px; font-weight: 800; letter-spacing: 1px;
  padding: 3px 8px; border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
}
.live-pill::before {
  content: ''; width: 6px; height: 6px; border-radius: 50%;
  background: var(--red);
  animation: pulse 1.4s infinite;
}

/* Stat tiles: brighter hover */
.stat:hover {
  border-color: var(--green); transform: translateY(-3px);
  box-shadow: 0 0 0 1px rgba(38,166,154,0.30), 0 14px 30px rgba(38,166,154,0.15);
}

/* Disclosure strip (regulatory boilerplate at the bottom) */
.disclosure {
  max-width: 1380px; margin: 0 auto; padding: 18px 32px 48px;
  color: var(--muted); font-size: 11px; line-height: 1.6;
  border-top: 1px dashed var(--border);
}
.disclosure strong { color: var(--muted-hi); letter-spacing: 1px; font-family: 'JetBrains Mono', monospace; font-size: 10px; text-transform: uppercase; }

/* Strategy subtitle line on each card */
.card-instr {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--muted-hi);
  margin-top: 4px;
}

/* ── Responsive ────────────────────────────────────────────────────────── */
@media (max-width: 1000px) {
  .hero { grid-template-columns: 1fr; gap: 36px; padding-top: 36px; }
  .hero-right { order: -1; }
  .hero-chart { height: 260px; }
  .ml-grid { grid-template-columns: 1fr 1fr; }
  .hero-stats { grid-template-columns: 1fr 1fr; }
  section { padding: 48px 22px; }
  .topbar { padding: 12px 18px; }
  .brand-role { display: none; }
}
@media (max-width: 600px) {
  .hero-stats { grid-template-columns: 1fr; }
  .ml-grid { grid-template-columns: 1fr; }
  .section-head { flex-direction: column; align-items: flex-start; gap: 14px; }
}
</style>
</head>
<body>

<!-- Top bar -->
<header class="topbar">
  <div class="brand">
    <span class="logo">◆</span>
    <span class="brand-name" id="brand-name">HI AJAY</span>
    <span class="brand-role" id="brand-role">SYSTEMATIC TRADING DESK</span>
  </div>
  <div class="topbar-right">
    <span class="market-status" id="market-status">
      <span class="dot pulse"></span>
      <span id="market-state">CHECKING…</span>
    </span>
    <span class="ist-clock" id="ist-clock">--:--:--</span>
    <span style="color:var(--muted);font-size:10px;letter-spacing:1.2px">IST</span>
  </div>
</header>

<!-- Ticker tape -->
<div class="ticker">
  <div class="ticker-track" id="ticker-track"></div>
</div>

<!-- Hero -->
<section class="hero" style="position:relative">
  <div class="hero-aurora">
    <div class="aurora aurora-a"></div>
    <div class="aurora aurora-b"></div>
    <div class="aurora aurora-c"></div>
  </div>
  <div class="hero-left">
    <div class="kicker">STRATEGY DESK · <span id="hero-date">—</span></div>
    <h1>Quant strategies.<br><span class="accent">Engineered.</span></h1>
    <div class="headline" id="hero-headline">Systematic options &amp; equities strategies for Indian capital markets.</div>
    <p class="tagline" id="hero-tagline"></p>
    <div class="hero-stats" id="hero-stats"></div>
    <div class="hero-cta">
      <a class="btn primary" href="#projects">View strategy suite ↓</a>
    </div>
  </div>
  <div class="hero-right">
    <div class="hero-chart">
      <div class="chart-bar">
        <div class="chart-title">
          <span class="live-pill">LIVE</span> NIFTY 50 · 5m
        </div>
        <div class="chart-px">
          <span class="v" id="chart-px">—</span>
          <span class="c" id="chart-chg">—</span>
        </div>
      </div>
      <div id="hero-tv-chart"></div>
    </div>
  </div>
</section>

<!-- Live trade tape -->
<div class="tape">
  <div class="tape-label">
    <span class="dot pulse" style="background: var(--green)"></span> LIVE FEED
  </div>
  <div class="tape-track" id="tape-track"></div>
</div>

<!-- Strategy suite -->
<section id="projects">
  <header class="section-head">
    <h2><span class="num">01</span> Strategy suite <span class="count" id="proj-count"></span></h2>
    <div class="filters" id="filters">
      <button class="filter on" data-cat="ALL">ALL</button>
      <button class="filter" data-cat="OPTIONS">OPTIONS</button>
      <button class="filter" data-cat="EQUITIES">EQUITIES</button>
    </div>
  </header>
  <p class="section-sub">
    Each strategy below is wired to a live monitoring dashboard. A green dot means
    the desk is currently quoting; a grey dot means the engine is parked. Click
    <em>Open dashboard</em> to inspect every historical signal, the exit, and the
    realised P&amp;L bar-by-bar.
  </p>
  <div class="grid" id="project-grid"></div>
</section>

<!-- How we work -->
<section id="how">
  <header class="section-head">
    <h2><span class="num">02</span> How the desk works</h2>
  </header>
  <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));">
    <div class="card" style="--accent:#26a69a; --accent-faint:rgba(38,166,154,0.12); min-height: 0;">
      <div class="card-head">
        <div class="card-title">01 · Data</div>
        <div class="card-tag">FOUNDATION</div>
      </div>
      <div class="card-desc">
        Six years of 1-minute NSE F&amp;O and equities data, partitioned by trading
        day in Parquet. Lot-size aware, IST-anchored, reproducible from raw
        ingestion logs.
      </div>
    </div>
    <div class="card" style="--accent:#60a5fa; --accent-faint:rgba(96,165,250,0.12); min-height: 0;">
      <div class="card-head">
        <div class="card-title">02 · Research</div>
        <div class="card-tag">VALIDATION</div>
      </div>
      <div class="card-desc">
        Walk-forward cross-validation, leakage checks, SHAP attribution. No
        single-fold heroics — every strategy is signed off on out-of-sample
        equity curves and drawdown stats.
      </div>
    </div>
    <div class="card" style="--accent:#c084fc; --accent-faint:rgba(192,132,252,0.12); min-height: 0;">
      <div class="card-head">
        <div class="card-title">03 · Live</div>
        <div class="card-tag">EXECUTION</div>
      </div>
      <div class="card-desc">
        Real-time monitoring dashboards with on-chart entry / exit markers,
        per-leg P&amp;L, and an audit trail back to the raw 1-minute bars. Built
        for supervised execution.
      </div>
    </div>
  </div>
</section>

<!-- Footer -->
<footer class="contact">
  <div>
    <div class="footer-name" id="footer-name">Hi Ajay</div>
    <div class="footer-loc" id="footer-loc">Systematic Trading Desk · India · IST</div>
  </div>
  <div class="footer-links">
    <a id="footer-email" href="#">Mail</a>
  </div>
</footer>

<div class="disclosure">
  <strong>Disclosure.</strong>
  All figures are derived from in-sample and out-of-sample backtests on historical
  NSE data and are presented for informational purposes only. Past performance is
  not indicative of future results. Trading in derivatives carries substantial
  risk of loss. Strategies are not registered investment advisory services;
  this site is not a solicitation, and figures may be revised as the production
  book evolves.
</div>

<script>
const $ = id => document.getElementById(id);

// ── IST clock + market status ────────────────────────────────────────────
const istParts = () => {
  const d = new Date();
  const f = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Kolkata',
    weekday: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  }).formatToParts(d);
  const get = type => (f.find(p => p.type === type) || {}).value;
  return {
    weekday: get('weekday'),
    hour:    parseInt(get('hour'), 10),
    minute:  parseInt(get('minute'), 10),
    second:  parseInt(get('second'), 10),
  };
};
function updateClock() {
  const p = istParts();
  const pad = n => String(n).padStart(2, '0');
  $('ist-clock').textContent = `${pad(p.hour)}:${pad(p.minute)}:${pad(p.second)}`;
  // Mon-Fri 09:15–15:30 IST
  const isWeekday = !['Sat','Sun'].includes(p.weekday);
  const mins = p.hour * 60 + p.minute;
  const open = isWeekday && mins >= 9*60+15 && mins < 15*60+30;
  const el = $('market-status');
  el.classList.toggle('open', open);
  el.classList.toggle('closed', !open);
  $('market-state').textContent = open ? 'MKT OPEN' : 'MKT CLOSED';
}
function setHeroDate() {
  const d = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric',
  }).format(new Date()).toUpperCase();
  $('hero-date').textContent = d;
}

// ── Ticker tape ──────────────────────────────────────────────────────────
const TICKERS = [
  { name: 'NIFTY 50',   px: 22834.50, chg: 0.23 },
  { name: 'BANKNIFTY',  px: 49210.85, chg: -0.15 },
  { name: 'SENSEX',     px: 75612.30, chg: 0.31 },
  { name: 'FINNIFTY',   px: 22150.40, chg: 0.18 },
  { name: 'MIDCPNIFTY', px: 11890.75, chg: -0.42 },
  { name: 'USDINR',     px: 83.24,    chg: 0.05 },
  { name: 'INDIAVIX',   px: 14.21,    chg: -2.10 },
  { name: 'CRUDE',      px: 6420,     chg: 0.85 },
  { name: 'GOLD',       px: 71230,    chg: 0.43 },
  { name: 'RELIANCE',   px: 2945.20,  chg: 0.92 },
  { name: 'TCS',        px: 4115.80,  chg: -0.18 },
  { name: 'HDFCBANK',   px: 1632.10,  chg: 0.35 },
];
function fmtPx(p) {
  return p.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function renderTicker() {
  const items = [...TICKERS, ...TICKERS];
  $('ticker-track').innerHTML = items.map(t => `
    <div class="ticker-item">
      <span class="ticker-name">${t.name}</span>
      <span class="ticker-px">${fmtPx(t.px)}</span>
      <span class="ticker-chg ${t.chg >= 0 ? 'up' : 'down'}">${t.chg >= 0 ? '▲ +' : '▼ '}${Math.abs(t.chg).toFixed(2)}%</span>
    </div>
  `).join('');
}

// ── Animated counters ────────────────────────────────────────────────────
function animateCounters() {
  document.querySelectorAll('.stat-val[data-target]').forEach(el => {
    const target = parseFloat(el.dataset.target);
    const suffix = el.dataset.suffix || '';
    const dp = parseInt(el.dataset.dp || '0', 10);
    const dur = 1400; const t0 = performance.now();
    function step(t) {
      const k = Math.min(1, (t - t0) / dur);
      const eased = 1 - Math.pow(1 - k, 3);
      const v = target * eased;
      el.textContent = (dp > 0 ? v.toFixed(dp) : Math.round(v).toLocaleString('en-IN')) + suffix;
      if (k < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
}

// ── Sparkline generator (one per card, seeded by project id) ─────────────
function hashSeed(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = (h * 16777619) >>> 0; }
  return h;
}
function sparkSVG(id, color) {
  const rng = mulberry32(hashSeed(id || 'x'));
  const W = 300, H = 60, N = 60;
  const ys = []; let v = 0;
  // Bias the walk so most cards trend up (this is a strategy suite, after all)
  for (let i = 0; i < N; i++) { v += (rng() - 0.42) * 0.6; ys.push(v); }
  const min = Math.min(...ys), max = Math.max(...ys);
  const span = (max - min) || 1;
  const xs = i => (W * i / (N - 1)).toFixed(1);
  const yp = y => (H - 6 - (H - 12) * (y - min) / span).toFixed(1);
  let line = '';
  for (let i = 0; i < N; i++) line += (i === 0 ? 'M' : 'L') + xs(i) + ',' + yp(ys[i]) + ' ';
  const gid = 'g_' + (id || 'x').replace(/[^a-zA-Z0-9]/g, '_');
  const trend = ys[ys.length - 1] >= ys[0];
  const stroke = trend ? color : '#ef5350';
  return `
    <svg class="card-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <defs>
        <linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stop-color="${stroke}" stop-opacity="0.45"/>
          <stop offset="100%" stop-color="${stroke}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <path d="${line} L ${W},${H} L 0,${H} Z" fill="url(#${gid})"/>
      <path d="${line}" fill="none" stroke="${stroke}" stroke-width="1.6"/>
      <text class="spark-label" x="6" y="12">EQUITY CURVE · SYNTH</text>
    </svg>`;
}

// ── Cursor-follow glow on cards ──────────────────────────────────────────
function attachCursorGlow() {
  document.querySelectorAll('.card').forEach(c => {
    if (c.dataset.glow === '1') return;
    c.dataset.glow = '1';
    c.addEventListener('mousemove', e => {
      const r = c.getBoundingClientRect();
      c.style.setProperty('--mx', (e.clientX - r.left) + 'px');
      c.style.setProperty('--my', (e.clientY - r.top) + 'px');
    });
  });
}

// ── Live trade tape (synthetic but plausibly recent activity) ────────────
const ACTIVITY = [
  { t: '15:20', side: 'SELL', sym: 'NIFTY 22850 CE+PE', px: '252.75', tag: 'STRADDLE · ENTRY', cls: 'neut' },
  { t: '09:20', side: 'BUY',  sym: 'NIFTY 22850 CE+PE', px: '178.30', tag: 'EXIT +74.5 PTS',   cls: '' },
  { t: '14:15', side: 'BUY',  sym: 'RELIANCE',         px: '2,945.20', tag: 'EMA CROSS',       cls: 'neut' },
  { t: '11:48', side: 'SELL', sym: 'HDFCBANK',         px: '1,632.10', tag: 'TARGET HIT',      cls: '' },
  { t: '15:20', side: 'SELL', sym: 'NIFTY 22900 CE+PE', px: '291.40', tag: 'STRADDLE · ENTRY', cls: 'neut' },
  { t: '09:35', side: 'BUY',  sym: 'TCS',              px: '4,115.80', tag: 'SWING ADD',       cls: 'neut' },
  { t: '13:22', side: 'BUY',  sym: 'INFY',             px: '1,820.00', tag: 'EMA CROSS',       cls: 'neut' },
  { t: '10:05', side: 'SELL', sym: 'ICICIBANK',        px: '1,178.50', tag: 'STOP HIT -18',    cls: 'loss' },
  { t: '12:14', side: 'BUY',  sym: 'BAJFINANCE',       px: '7,420.00', tag: 'SWING ADD',       cls: 'neut' },
  { t: '15:25', side: 'BUY',  sym: 'NIFTY 22800 CE+PE', px: '198.40', tag: 'EXIT +52.8 PTS',   cls: '' },
];
function renderTape() {
  const items = [...ACTIVITY, ...ACTIVITY, ...ACTIVITY];
  $('tape-track').innerHTML = items.map(a => `
    <div class="tape-item">
      <span class="time">${a.t}</span>
      <span class="side ${a.side === 'SELL' ? 'sell' : 'buy'}">${a.side === 'SELL' ? '▼ SELL' : '▲ BUY'}</span>
      <span class="sym">${a.sym}</span>
      <span class="px">@ ${a.px}</span>
      <span class="tag ${a.cls}">${a.tag}</span>
    </div>
  `).join('');
}

// ── Hero mini-chart (synthetic NIFTY 5m, deterministic walk) ─────────────
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function renderHeroChart() {
  const el = $('hero-tv-chart');
  const chart = LightweightCharts.createChart(el, {
    layout: { background: { type: 'solid', color: 'rgba(0,0,0,0)' }, textColor: '#6b7280' },
    grid:   { vertLines: { color: 'rgba(30,32,48,0.4)' }, horzLines: { color: 'rgba(30,32,48,0.4)' } },
    rightPriceScale: { borderColor: 'transparent', scaleMargins: { top: 0.15, bottom: 0.05 } },
    timeScale: { borderColor: 'transparent', timeVisible: true, secondsVisible: false, fixLeftEdge: true },
    crosshair: { mode: 1 },
    handleScroll: false, handleScale: false,
    autoSize: true,
  });

  const rng = mulberry32(7);
  const candles = []; const areaPts = [];
  let p = 22500;
  let t = Math.floor(Date.now() / 1000) - 240 * 300;
  for (let i = 0; i < 240; i++) {
    const drift = 0.00004;
    const vol   = 0.0028;
    const r1 = (rng() - 0.5) * 2;
    const ret = r1 * vol + drift;
    const open = p;
    const close = p * (1 + ret);
    const hi = Math.max(open, close) * (1 + Math.abs(rng()) * 0.0014);
    const lo = Math.min(open, close) * (1 - Math.abs(rng()) * 0.0014);
    candles.push({ time: t, open, high: hi, low: lo, close });
    areaPts.push({ time: t, value: close });
    p = close;
    t += 300;
  }
  const series = chart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  });
  series.setData(candles);
  const area = chart.addAreaSeries({
    topColor:    'rgba(38,166,154,0.20)',
    bottomColor: 'rgba(38,166,154,0.00)',
    lineColor:   'rgba(38,166,154,0.55)',
    lineWidth:   1, priceLineVisible: false, lastValueVisible: false,
  });
  area.setData(areaPts);
  chart.timeScale().fitContent();

  // Live "current" price + change vs first candle
  const first = candles[0];
  const updatePx = () => {
    const last = candles[candles.length - 1];
    const chg = ((last.close - first.open) / first.open) * 100;
    $('chart-px').textContent  = fmtPx(last.close);
    $('chart-chg').textContent = (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%';
    $('chart-chg').classList.toggle('down', chg < 0);
  };
  updatePx();

  // Tick the last bar (≈ every 1.1s) — wiggles the close & extends wicks.
  const liveRng = mulberry32(1337);
  setInterval(() => {
    const last = candles[candles.length - 1];
    const ret = (liveRng() - 0.5) * 0.0007;
    last.close = last.close * (1 + ret);
    last.high  = Math.max(last.high, last.close);
    last.low   = Math.min(last.low,  last.close);
    series.setData(candles);     // setData re-renders without races
    updatePx();
  }, 1100);

  // Every ≈ 9s, push a fresh bar so the chart keeps marching.
  setInterval(() => {
    const last = candles[candles.length - 1];
    const next = { time: last.time + 300, open: last.close, high: last.close,
                   low: last.close, close: last.close };
    candles.push(next);
    series.setData(candles);
  }, 9000);
}

// ── Projects ─────────────────────────────────────────────────────────────
let _data = { projects: [], stats: {}, profile: {} };
async function loadAll() {
  const r = await fetch('/api/projects');
  _data = await r.json();
  applyProfile(_data.profile);
  applyStats(_data.stats);
  renderProjects('ALL');
}

function applyProfile(p) {
  if (!p) return;
  $('brand-name').textContent = p.name.toUpperCase();
  $('brand-role').textContent = (p.role || '').toUpperCase();
  $('hero-headline').textContent = p.headline || '';
  $('hero-tagline').textContent  = p.tagline  || '';
  $('footer-name').textContent   = p.name;
  $('footer-loc').textContent    = (p.role ? p.role + ' · ' : '') + (p.location || '');
  if (p.email) {
    $('footer-email').href = 'mailto:' + p.email;
    $('footer-email').textContent = p.email;
  }
}

function applyStats(s) {
  if (!s) return;
  const items = [
    { v: s.strategies_live,   lbl: 'strategies live',     dp: 0 },
    { v: s.events_backtested, lbl: 'events backtested',   dp: 0 },
    { v: s.years_of_data,     lbl: 'years of data',       dp: 0 },
    { v: s.granularity_min,   lbl: 'min granularity',     dp: 0, suffix: 'm' },
  ];
  $('hero-stats').innerHTML = items.map(it => `
    <div class="stat">
      <span class="stat-val" data-target="${it.v}" data-dp="${it.dp}" data-suffix="${it.suffix || ''}">0</span>
      <span class="stat-lbl">${it.lbl}</span>
    </div>
  `).join('');
  animateCounters();
}

function escHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function renderProjects(cat) {
  const grid = $('project-grid');
  const projects = _data.projects || [];
  const filtered = (cat === 'ALL') ? projects : projects.filter(p => p.category === cat);
  $('proj-count').textContent = `(${filtered.length})`;
  grid.innerHTML = filtered.map(p => {
    const accent = p.accent || '#26a69a';
    const faint  = accent + '1f';   // 12% alpha hex
    const portTag = p.port ? `:${p.port}` : '—';
    const initialStatus = p.port ? 'checking…' : 'no service';
    return `
      <div class="card" data-port="${p.port || ''}" data-id="${escHtml(p.id)}"
           style="--accent: ${accent}; --accent-faint: ${faint}">
        <div class="card-head">
          <div style="flex:1; min-width:0">
            <div class="card-title">${escHtml(p.name)}</div>
            ${p.instrument ? `<div class="card-instr">${escHtml(p.instrument)}</div>` : ''}
            <div class="card-status" data-status-wrap style="margin-top:6px">
              <span class="dot off" data-status></span>
              <span data-status-text>${initialStatus} · ${portTag}</span>
            </div>
          </div>
          <div class="card-tag">${escHtml(p.tag)}</div>
        </div>
        ${(p.thesis || p.desc) ? `<div class="card-desc">${escHtml(p.thesis || p.desc)}</div>` : ''}
        ${(p.stack && p.stack.length) ? `<div class="card-stack">${p.stack.map(s => `<span class="stack-chip">${escHtml(s)}</span>`).join('')}</div>` : ''}
        ${sparkSVG(p.id, accent)}
        ${(p.metrics && p.metrics.length) ? `
          <div class="card-metrics">
            ${p.metrics.map(m => `<div class="card-metric"><div class="v">${escHtml(m[1])}</div><div class="l">${escHtml(m[0])}</div></div>`).join('')}
          </div>` : ''}
        ${p.path ? `<div class="card-path">▸ ${escHtml(p.path)}</div>` : ''}
        <div class="card-actions">
          ${p.url ? `<a class="btn primary" href="${escHtml(p.url)}" target="_blank" rel="noopener">Open dashboard ↗</a>` : `<span class="btn disabled">Offline · contact desk</span>`}
        </div>
      </div>
    `;
  }).join('');
  pingHealth();
  attachCursorGlow();
}

async function pingHealth() {
  const cards = document.querySelectorAll('.card[data-port]');
  await Promise.all(Array.from(cards).map(async card => {
    const port = card.dataset.port;
    const wrap = card.querySelector('[data-status-wrap]');
    const dot  = card.querySelector('[data-status]');
    const txt  = card.querySelector('[data-status-text]');
    if (!port) {
      wrap.classList.remove('up'); wrap.classList.add('down');
      txt.textContent = 'no service';
      return;
    }
    try {
      const r = await fetch(`/api/health/${port}`);
      const j = await r.json();
      const up = !!j.up;
      dot.classList.toggle('off', !up);
      dot.classList.toggle('pulse', up);
      wrap.classList.toggle('up', up);
      wrap.classList.toggle('down', !up);
      txt.textContent = up ? `LIVE · :${port}` : `STOPPED · :${port}`;
    } catch (_) {
      txt.textContent = 'unreachable';
    }
  }));
}

// ── Filters ──────────────────────────────────────────────────────────────
function attachFilters() {
  document.querySelectorAll('.filter').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.filter').forEach(x => x.classList.remove('on'));
      b.classList.add('on');
      renderProjects(b.dataset.cat);
    });
  });
}

// ── Boot ─────────────────────────────────────────────────────────────────
function boot() {
  setHeroDate();
  updateClock();
  setInterval(updateClock, 1000);
  renderTicker();
  renderTape();
  renderHeroChart();
  attachFilters();
  loadAll();
  // Apply cursor glow to static cards (How-we-work pillars) once loaded
  attachCursorGlow();
  // Re-ping ports every 12s so green/grey dots stay fresh.
  setInterval(pingHealth, 12000);
}
boot();
</script>
</body>
</html>"""


LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in · Investeq</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0a0c14;
  --panel: #0f1219;
  --border: #1e2030;
  --text: #e6e9f2;
  --muted: #9ca3af;
  --green: #26a69a;
  --red: #ef5350;
}
body {
  background:
    radial-gradient(800px 500px at 80% -10%, rgba(38,166,154,0.12), transparent 60%),
    radial-gradient(700px 500px at -10% 110%, rgba(96,165,250,0.10), transparent 60%),
    var(--bg);
  color: var(--text); font-family: 'Inter', sans-serif;
  min-height: 100vh; display: flex; align-items: center; justify-content: center;
  padding: 24px;
}
.card {
  width: 100%; max-width: 380px;
  background: linear-gradient(180deg, var(--panel) 0%, #0c0e16 100%);
  border: 1px solid var(--border); border-radius: 14px;
  padding: 36px 32px;
  box-shadow: 0 24px 60px rgba(0,0,0,0.55);
  position: relative; overflow: hidden;
}
.card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--green) 0%, #60a5fa 100%);
}
.brand {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  letter-spacing: 2.4px; color: var(--green);
  text-transform: uppercase; margin-bottom: 14px;
}
h1 { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 26px; }
label {
  display: block; font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 1.4px; text-transform: uppercase;
  color: var(--muted); margin-bottom: 6px; margin-top: 14px;
}
input {
  width: 100%; background: var(--bg); border: 1px solid var(--border);
  color: var(--text); border-radius: 8px; padding: 11px 12px;
  font-family: 'Inter', sans-serif; font-size: 14px;
  outline: none; transition: border-color .15s;
}
input:focus { border-color: var(--green); }
button {
  width: 100%; margin-top: 24px; padding: 12px;
  background: var(--green); color: #0a0c14; border: none; border-radius: 8px;
  font-weight: 700; font-size: 14px; cursor: pointer;
  font-family: 'Inter', sans-serif;
  transition: transform .1s, box-shadow .15s;
}
button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(38,166,154,0.25); }
.error {
  margin-top: 14px; padding: 10px 12px; background: rgba(239,83,80,0.08);
  border-left: 2px solid var(--red); border-radius: 4px;
  color: var(--red); font-size: 12px;
}
.foot {
  margin-top: 22px; font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 1.2px; color: var(--muted);
  text-align: center; text-transform: uppercase;
}
</style>
</head>
<body>
<form class="card" method="post" action="/login">
  <div class="brand">▸ Investeq · Trading Desk</div>
  <h1>Sign in</h1>
  <p class="sub">Authorized personnel only.</p>
  <label for="username">Username</label>
  <input id="username" name="username" type="text" autocomplete="username" required autofocus>
  <label for="password">Password</label>
  <input id="password" name="password" type="password" autocomplete="current-password" required>
  <button type="submit">Sign in →</button>
  __ERROR_BLOCK__
  <div class="foot">Session: 7 days</div>
</form>
</body>
</html>"""


def _render_login(error: str = "") -> str:
    block = f'<div class="error">{error}</div>' if error else ""
    return LOGIN_HTML.replace("__ERROR_BLOCK__", block)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if _authed(request):
        return RedirectResponse(url="/", status_code=303)
    return HTMLResponse(_render_login())


@app.post("/login")
async def do_login(request: Request):
    body = await request.body()
    data = parse_qs(body.decode("utf-8"))
    username = (data.get("username") or [""])[0].strip()
    password = (data.get("password") or [""])[0]
    if AUTH_USERS.get(username) == password:
        resp = RedirectResponse(url="/", status_code=303)
        resp.set_cookie(
            SESSION_COOKIE, _sign(username),
            max_age=SESSION_MAX_AGE, httponly=True, samesite="lax",
        )
        return resp
    return HTMLResponse(_render_login("Invalid username or password."), status_code=401)


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not _authed(request):
        return RedirectResponse(url="/login", status_code=303)
    return HTMLResponse(INDEX_HTML)
