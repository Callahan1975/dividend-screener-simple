# ============================================================
# Dividend Screener – CLEAN MASTER VERSION (RESET)
# Author: Callahan1975 + ChatGPT
# Purpose: Stable baseline – correct data, no heuristics yet
# ============================================================

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# ======================
# PATHS
# ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "screener_results"
DOCS_RESULTS_DIR = BASE_DIR / "docs" / "data" / "screener_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALIAS_FILE = DATA_DIR / "ticker_alias.csv"
OUTPUT_FILE = RESULTS_DIR / "screener_results.csv"
DOCS_OUTPUT_FILE = DOCS_RESULTS_DIR / "screener_results.csv"

# ======================
# LOAD TICKERS
# ======================
alias_df = pd.read_csv(ALIAS_FILE, comment="#")

alias_df["Ticker"] = alias_df["Ticker"].astype(str).str.strip().str.upper()
TICKERS = sorted(alias_df["Ticker"].dropna().unique().tolist())

META_MAP = alias_df.set_index("Ticker").to_dict(orient="index")

print(f"🔥 Universe size: {len(TICKERS)} tickers loaded")

# ======================
# HELPERS
# ======================
def safe_float(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None

# ======================
# MAIN LOOP
# ======================
rows = []
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

for ticker in TICKERS:
    print(f"⏳ Processing {ticker}")
    meta = META_MAP.get(ticker, {})

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        price = safe_float(info.get("currentPrice"))

        dividend_rate = safe_float(info.get("dividendRate"))
        yield_pct = None
        if dividend_rate and price and price > 0:
            yield_pct = (dividend_rate / price) * 100

        rows.append({
            "Name": info.get("longName") or info.get("shortName") or ticker,
            "Ticker": ticker,
            "Country": meta.get("Country"),
            "Sector": info.get("sector"),
            "Currency": info.get("currency"),
            "Price": price,
            "DividendYield_%": safe_float(yield_pct),
            "DivCAGR_5Y_%": None,
            "Upside_%": None,
            "Score": None,
            "Signal": None,
            "Confidence": None,
            "Why": None,
            "Exchange": meta.get("Exchange"),
            "Region": meta.get("Region"),
            "GeneratedUTC": now_utc,
        })

    except Exception as e:
        print(f"⚠️ Failed {ticker}: {e}")

# ======================
# OUTPUT
# ======================
df = pd.DataFrame(rows)

df.sort_values(["Country", "Ticker"], inplace=True, na_position="last")

df.to_csv(OUTPUT_FILE, index=False)
df.to_csv(DOCS_OUTPUT_FILE, index=False)

print(f"✅ CSV written: {OUTPUT_FILE}")
print(f"📄 CSV copied to: {DOCS_OUTPUT_FILE}")
print(f"📊 Final rows: {len(df)}")
