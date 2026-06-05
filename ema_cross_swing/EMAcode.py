"""
EMA 21/50 Bullish Crossover + Support Touch Detector
=====================================================
Logic:
  1. EMA21 crosses ABOVE EMA50  →  cross event recorded.
  2. For ALL candles after each cross (no limit):
       Touch qualifies if:
         - Low <= EMA21  (covers: touching EMA21, between both EMAs, at/below EMA50)
         AND
         - Close > EMA21  (closed back above EMA21)
         AND
         - Close > Open   (bullish / green candle)
  3. Touch zone label:
       Low <= EMA50               → "At/Below EMA50"
       EMA50 < Low <= EMA21       → "Between EMA50 & EMA21"

Stocks  : Top 5 Nifty (RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK)
Source  : Local CSV files (one per stock)
CSV fmt : timestamp, open, high, low, close, volume, datetime
Outputs : signals DataFrame (printed + saved to signals.csv)
"""

import pandas as pd
import numpy as np
import os

# ── Config ─────────────────────────────────────────────────────────────────
# Map stock name → CSV file path (update paths if files are in a subfolder)
CSV_FILES = {
    "RELIANCE":  "RELIANCE_historical.csv",
    "TCS":       "TCS_historical.csv",
    "HDFCBANK":  "HDFCBANK_historical.csv",
    "INFY":      "INFY_historical.csv",
    "ICICIBANK": "ICICIBANK_historical.csv",
}
EMA_FAST = 21
EMA_SLOW = 50


# ── Data ────────────────────────────────────────────────────────────────────
def fetch(filepath: str) -> pd.DataFrame:
    """
    Load OHLCV from a local CSV file produced by the Groww fetch script.

    Expected columns: timestamp, open, high, low, close, volume, datetime
    - 'datetime' column is in IST (Asia/Kolkata), stored as a plain string
      (no tz suffix) — e.g. "2023-01-02 09:15:00".
    - Falls back to 'timestamp' (Unix seconds, UTC) if 'datetime' is absent.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse datetime index
    if "datetime" in df.columns:
        # Already IST — parse as-is, no tz conversion needed
        df["Date"] = pd.to_datetime(df["datetime"])
    elif "timestamp" in df.columns:
        # Raw UTC Unix timestamp → convert to IST
        df["Date"] = (
            pd.to_datetime(df["timestamp"], unit="s")
            .dt.tz_localize("UTC")
            .dt.tz_convert("Asia/Kolkata")
            .dt.tz_localize(None)       # strip tz → naive IST datetime
        )
    else:
        raise ValueError("CSV must have a 'datetime' or 'timestamp' column.")

    df = df.set_index("Date").sort_index()

    # Rename OHLCV columns to Title case
    df = df.rename(columns={"open": "Open", "high": "High",
                             "low": "Low", "close": "Close", "volume": "Volume"})

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ── EMA (Pine Script ta.ema equivalent) ─────────────────────────────────────
def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA21"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    return df


# # ── Crossover detection ───────────────────────────────────────────────────
# def bullish_cross_indices(df: pd.DataFrame) -> list:
#     """Positions where EMA21 crosses above EMA50."""
#     above = df["EMA21"] > df["EMA50"]
#     cross = above & ~above.shift(1, fill_value=False)
#     cross.iloc[0] = False          # no prior bar → not a valid cross
#     return list(np.where(cross.values)[0])


# ── Reliable Bullish EMA Crossover Detection ─────────────────────────────

def bullish_cross_indices(df: pd.DataFrame) -> list:
    """
    Returns indices where EMA21 crosses ABOVE EMA50.
    """

    # Previous candle values
    prev_ema21 = df["EMA21"].shift(1)
    prev_ema50 = df["EMA50"].shift(1)

    # Current candle values
    curr_ema21 = df["EMA21"]
    curr_ema50 = df["EMA50"]

    # Bullish crossover condition
    cross = (
        (prev_ema21 <= prev_ema50) &
        (curr_ema21 > curr_ema50)
    )

    # Remove NaN rows
    cross = cross.fillna(False)

    # Return positions
    return list(np.where(cross)[0])

# ── Touch label ───────────────────────────────────────────────────────────
def touch_zone(low: float, ema21: float, ema50: float) -> str:
    if low <= ema50:
        return "At/Below EMA50"
    return "Between EMA50 & EMA21"


# ── Signal scanner ────────────────────────────────────────────────────────
def scan(name: str, df: pd.DataFrame) -> list:
    df    = add_emas(df)
    rows  = []

    for cidx in bullish_cross_indices(df):
        cross_date  = df.index[cidx]
        cross_close = float(df["Close"].iloc[cidx])
        ema21_cross = float(df["EMA21"].iloc[cidx])
        ema50_cross = float(df["EMA50"].iloc[cidx])

        for i in range(cidx + 1, len(df)):
            low   = float(df["Low"].iloc[i])
            close = float(df["Close"].iloc[i])
            open_ = float(df["Open"].iloc[i])
            ema21 = float(df["EMA21"].iloc[i])
            ema50 = float(df["EMA50"].iloc[i])

            # Cross invalidated — EMA21 fell back below EMA50
            if ema21 <= ema50:
                break

            # Touch conditions
            if low <= ema21 and close > ema21 and close > open_:
                rows.append({
                    "Stock":               name,
                    # ── Cross info ──
                    "Cross Date":          cross_date.date(),
                    "Cross Close (₹)":     round(cross_close, 2),
                    "EMA21 @ Cross":       round(ema21_cross, 2),
                    "EMA50 @ Cross":       round(ema50_cross, 2),
                    # ── Touch info ──
                    "Touch Date":          df.index[i].date(),
                    "Touch Open (₹)":      round(open_, 2),
                    "Touch Low (₹)":       round(low, 2),
                    "Touch Close (₹)":     round(close, 2),
                    "EMA21 @ Touch":       round(ema21, 2),
                    "EMA50 @ Touch":       round(ema50, 2),
                    "Touch Zone":          touch_zone(low, ema21, ema50),
                    "Candles After Cross": i - cidx,
                })

    return rows


# ── Main ─────────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    all_signals = []

    for name, filepath in CSV_FILES.items():
        print(f"  Loading {name} from '{filepath}' ...")
        try:
            df  = fetch(filepath)
            print(f"    → {len(df)} rows loaded  ({df.index[0].date()} → {df.index[-1].date()})")
            sig = scan(name, df)
            print(f"    → {len(sig)} qualifying touch candle(s)")
            all_signals.extend(sig)
        except Exception as e:
            print(f"    ✗ Error: {e}")

    if not all_signals:
        print("\nNo signals found.")
        return pd.DataFrame()

    result = pd.DataFrame(all_signals)
    result = result.sort_values(["Stock", "Cross Date", "Touch Date"]).reset_index(drop=True)
    return result


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)

    print(f"\nEMA {EMA_FAST}/{EMA_SLOW} Crossover + Support Touch Scanner")
    print(f"Source : Local CSV files")
    print(f"Stocks : {', '.join(CSV_FILES.keys())}")
    print("-" * 60)

    signals = run()

    if not signals.empty:
        print(f"\nTotal signals  : {len(signals)}")
        print(f"Unique crosses : {signals[['Stock','Cross Date']].drop_duplicates().shape[0]}")
        print("\n" + signals.to_string(index=False))

        # ── Summary by stock ──
        summary = (signals.groupby("Stock")
                   .agg(Total_Signals  =("Touch Date",  "count"),
                        Unique_Crosses =("Cross Date",  "nunique"),
                        First_Cross    =("Cross Date",  "min"),
                        Last_Touch     =("Touch Date",  "max"))
                   .reset_index())
        print("\n── Summary by Stock ──")
        print(summary.to_string(index=False))

        # ── Touch zone breakdown ──
        zone = signals.groupby(["Stock", "Touch Zone"]).size().unstack(fill_value=0)
        print("\n── Touch Zone Breakdown ──")
        print(zone.to_string())

        out = "signals.csv"
        signals.to_csv(out, index=False)
        print(f"\nSaved → {out}")