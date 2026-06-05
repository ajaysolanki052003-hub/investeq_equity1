"""Inspect how fresh each CSV is."""
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import app_ema_cross as A

for tf in ("1d", "1h"):
    syms = A.list_symbols(tf)
    last_dates = []
    for s in syms:
        try:
            df = A.load_ohlc(s, tf)
            last_dates.append(df.index[-1])
        except Exception:
            pass
    if not last_dates:
        print(tf, ": no data"); continue
    s = pd.Series(last_dates)
    print(f"\n=== {tf}  ({len(s)} symbols) ===")
    print(f"  newest CSV last bar : {s.max()}")
    print(f"  oldest CSV last bar : {s.min()}")
    print(f"  median last bar     : {s.quantile(0.5, interpolation='nearest')}")
    print("\n  last-bar histogram (top 10 dates):")
    print(s.dt.normalize().value_counts().head(10).to_string())
