"""Headless scan that mirrors app_ema_cross.py — prints today's watchlist."""
import os, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
import app_ema_cross as A

TF        = "1h"
SIDE      = "Both"
USE_A     = True
USE_B     = True
USE_C     = True
MAX_STALE = 10

cutoff = pd.Timestamp.now().normalize()
print(f"Cutoff (strict <)  : {cutoff}")
print(f"Max staleness (d)  : {MAX_STALE}")
print(f"Timeframe          : {TF}")

symbols = A.list_symbols(TF)
print(f"Symbols on disk    : {len(symbols)}")

rows = []
for sym in symbols:
    try:
        dfs = A.load_ohlc(sym, TF)
    except Exception:
        continue
    sides = ["BUY", "SELL"] if SIDE == "Both" else [SIDE]
    for side in sides:
        meta = A.scan_latest(dfs, side, USE_A, USE_B, USE_C,
                             cutoff=cutoff, max_stale_days=MAX_STALE)
        if meta is None:
            continue
        rows.append({"Symbol": sym, **meta})

df = pd.DataFrame(rows)
if df.empty:
    print("\nNo qualifying stocks. Try increasing MAX_STALE or toggling conditions.")
    sys.exit()

df = df.sort_values(["Side", "Symbol"]).reset_index(drop=True)
n_buy  = int((df["Side"] == "BUY").sum())
n_sell = int((df["Side"] == "SELL").sum())
print(f"\nTotal qualifying  : {len(df)}  (BUY={n_buy}, SELL={n_sell})")

show = df[["Symbol", "Side", "Trigger", "Last bar", "Close",
           f"EMA{A.FAST_EMA}", f"EMA{A.SLOW_EMA}",
           "Δ vs 21 %", "Δ vs 50 %", "Bars since cross"]]
print("\nFirst 20 BUYs:")
print(show[show["Side"] == "BUY"].head(20).to_string(index=False))
print("\nFirst 20 SELLs:")
print(show[show["Side"] == "SELL"].head(20).to_string(index=False))
