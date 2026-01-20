# screener.py
# STABIL BASELINE VERSION – Yahoo Finance only
# Purpose: Generate a clean, complete screener_results.csv for UI
# Single source of truth: data/ticker_alias.csv

import pandas as pd
import yfinance as yf
from pathlib import Path
import numpy as np
from datetime import datetime

# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "screener_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALIAS_FILE = DATA_DIR / "ticker_alias.csv"
OUTPUT_FILE = RESULTS_DIR / "screener_results.csv"

# ==============================
# LOAD TICKERS
# ==============================
alias_df = pd.read_csv(ALIAS_FILE, comment="#")
alias_df["Ticker"] = alias_df["Ticker"].str.upper().str.strip()

TICKERS = alias_df["Ticker"].dropna().unique().tolist()
ALIAS_MAP = alias_df.set_index("Ticker").to_dict(orient="index")

print(f"🔥 Universe size: {len(TICKERS)} tickers loaded")

# ==============================
# HELPERS
# ==============================
def safe(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None

# ==============================
# MAIN LOOP
# ==============================
rows = []
run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

for ticker in TICKERS:
    meta = ALIAS_MAP.get(ticker, {})

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info or {}
    except Exception as e:
        print(f"❌ {ticker}: yfinance error: {e}")
        info = {}

    price = safe(info.get("regularMarketPrice"))
    dividend_yield = safe(info.get("dividendYield"))
    payout_ratio = safe(info.get("payoutRatio"))
    pe = safe(info.get("trailingPE"))

    # Convert yield + payout to %
    dividend_yield_pct = dividend_yield * 100 if dividend_yield is not None else None
    payout_ratio_pct = payout_ratio * 100 if payout_ratio is not None else None

    rows.append({
        "Ticker": ticker,
        "Name": info.get("longName") or meta.get("Name") or ticker,
        "Country": meta.get("Country") or info.get("country"),
        "Currency": info.get("currency") or meta.get("Currency"),
        "Exchange": info.get("exchange") or meta.get("Exchange"),
        "Sector": info.get("sector") or meta.get("Sector"),
        "Industry": info.get("industry"),
        "Price": price,
        "DividendYield_%": dividend_yield_pct,
        "PayoutRatio_%": payout_ratio_pct,
        "PE": pe,
        "RunTimestamp": run_ts,
    })

# ==============================
# SAVE CSV
# ==============================
df = pd.DataFrame(rows)

# Enforce column order expected by UI
cols = [
    "Ticker",
    "Name",
    "Country",
    "Sector",
    "Industry",
    "Currency",
    "Exchange",
    "Price",
    "DividendYield_%",
    "PayoutRatio_%",
    "PE",
    "RunTimestamp",
]

df = df.reindex(columns=cols)

df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df)} rows to {OUTPUT_FILE}")
