"""
Daily Swing Scanner — headless CLI.

Scans every stock in ../ema_scanner/data/{1h,1d}/ for swing-low project
signals (EMA21/50 touch + Dow uptrend gate) that fired on TARGET_DAY.

Outputs:
   - 1d entries on TARGET_DAY  → entries_<DATE>_1d.csv
   - 1h entries on TARGET_DAY  → entries_<DATE>_1h.csv
   - stocks present on both timeframes (printed to stdout)

Edit TARGET_DAY below to scan a different day, or use scan_app.py for
an interactive UI.
"""
from __future__ import annotations
import os, sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_core import add_emas, scan_buy_signals, scan_sell_signals

# ─── settings ───
DATA_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", "ema_scanner", "data"))
TARGET_DAY = pd.Timestamp("2026-05-14").date()
FAST, SLOW = 21, 50
COMPULSORY = 2
TIMEFRAMES = ["1d", "1h"]


def load_ohlc(symbol, tf):
    path = os.path.join(DATA_ROOT, tf, f"{symbol}_historical.csv")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    # The Groww 1d feed sometimes returns NULL Open for the most recent few
    # days. Strategy needs Open for the c>o green-candle gate, so backfill it
    # from the prior bar's Close (a flat-open assumption — close enough for
    # gating purposes; ranking the touch by close is unaffected).
    if df["Open"].isna().any():
        df["Open"] = df["Open"].fillna(df["Close"].shift(1))
    # Drop only rows where High/Low/Close are still missing (truly bad rows)
    return df.dropna(subset=["High", "Low", "Close"])


def signals_on_target_day(symbol, tf):
    """Return list of dicts: side / touch_time / close — for touches on TARGET_DAY only."""
    try:
        df = load_ohlc(symbol, tf)
    except Exception:
        return []
    if len(df) < SLOW + 5:
        return []
    # Drop bars from after the target day so the scanner only "sees" data
    # available up to and including 2026-05-14 — entries must qualify on that bar
    # without future-data leakage.
    cutoff = pd.Timestamp(TARGET_DAY) + pd.Timedelta(days=1)  # < 2026-05-15 00:00
    df = df[df.index < cutoff]
    if len(df) < SLOW + 5:
        return []
    df = add_emas(df, FAST, SLOW)

    out = []
    for side, fn in (("BUY", scan_buy_signals), ("SELL", scan_sell_signals)):
        sigs = fn(df, FAST, SLOW, compulsory=COMPULSORY)
        if sigs.empty:
            continue
        sigs = sigs[sigs["Touch Time"].dt.date == TARGET_DAY]
        for _, r in sigs.iterrows():
            out.append({
                "Symbol":     symbol,
                "Side":       side,
                "TF":         tf,
                "Touch Time": r["Touch Time"],
                "Close":      round(float(r["Touch Close"]), 2),
            })
    return out


def main():
    by_tf = {tf: [] for tf in TIMEFRAMES}
    for tf in TIMEFRAMES:
        folder = os.path.join(DATA_ROOT, tf)
        symbols = sorted(f[:-len("_historical.csv")]
                         for f in os.listdir(folder) if f.endswith("_historical.csv"))
        print(f"\n[{tf}] scanning {len(symbols)} stocks for entries on {TARGET_DAY}...",
              flush=True)
        for k, sym in enumerate(symbols):
            if (k + 1) % 100 == 0:
                print(f"  [{k+1}/{len(symbols)}]", flush=True)
            by_tf[tf].extend(signals_on_target_day(sym, tf))

    df1d = pd.DataFrame(by_tf["1d"])
    df1h = pd.DataFrame(by_tf["1h"])

    def show(df, label):
        print(f"\n=== {label}  ({len(df)} entries) ===")
        if df.empty:
            print("  (none)")
            return
        if "Touch Time" in df.columns:
            df = df.sort_values(["Side", "Symbol", "Touch Time"])
        else:
            df = df.sort_values(["Side", "Symbol"])
        print(df.to_string(index=False))

    show(df1d, f"1d entries on {TARGET_DAY}")
    show(df1h, f"1h entries on {TARGET_DAY}")

    # intersection: same Symbol+Side present in both timeframes
    if not df1d.empty and not df1h.empty:
        d1d = df1d[["Symbol", "Side"]].drop_duplicates()
        d1h = df1h[["Symbol", "Side"]].drop_duplicates()
        inter = d1d.merge(d1h, on=["Symbol", "Side"])
        inter = inter.sort_values(["Side", "Symbol"]).reset_index(drop=True)
        print(f"\n=== STARRED — entry on BOTH 1d AND 1h  ({len(inter)} stocks) ===")
        if inter.empty:
            print("  (none)")
        else:
            print(inter.to_string(index=False))
    else:
        print("\n=== STARRED — entry on BOTH 1d AND 1h ===")
        print("  (cannot compute — one or both timeframes empty)")

    # Save CSVs for later use
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df1d.to_csv(os.path.join(out_dir, "entries_2026-05-14_1d.csv"), index=False)
    df1h.to_csv(os.path.join(out_dir, "entries_2026-05-14_1h.csv"), index=False)
    print(f"\n[saved] entries_2026-05-14_1d.csv  ({len(df1d)} rows)")
    print(f"[saved] entries_2026-05-14_1h.csv  ({len(df1h)} rows)")


if __name__ == "__main__":
    main()
