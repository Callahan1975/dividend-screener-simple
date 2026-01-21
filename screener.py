# ============================================================
# Dividend Screener – MASTER SCREENER (STABLE)
# Single source of truth: data/ticker_alias.csv
# ============================================================

import pandas as pd
import yfinance as yf
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "screener_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALIAS_FILE = DATA_DIR / "ticker_alias.csv"
OUTPUT_FILE = RESULTS_DIR / "screener_results.csv"

# -----------------------------
# LOAD TICKER UNIVERSE (ONLY SOURCE)
# -----------------------------
alias_df = pd.read_csv(ALIAS_FILE, comment="#")
alias_df["Ticker"] = alias_df["Ticker"].str.upper().str.strip()
TICKERS = alias_df["Ticker"].dropna().unique().tolist()
ALIAS_MAP = alias_df.set_index("Ticker").to_dict(orient="index")

print(f"✅ Universe size: {len(TICKERS)} tickers loaded")

# -----------------------------
# HELPERS
# -----------------------------
def safe(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None

def dividend_cagr_5y(div_series):
    if div_series is None or len(div_series) < 6:
        return None
    start = div_series.iloc[-6]
    end = div_series.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    return ((end / start) ** (1 / 5) - 1) * 100

# -----------------------------
# MAIN LOOP
# -----------------------------
rows = []
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

for ticker in TICKERS:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        price = safe(info.get("currentPrice"))
        dividend_rate = safe(info.get("dividendRate"))
        dividend_yield = safe(info.get("dividendYield"))
        payout_ratio = safe(info.get("payoutRatio"))
        pe = safe(info.get("trailingPE"))

        dividends = t.dividends
        div_cagr_5y = dividend_cagr_5y(dividends)

        meta = ALIAS_MAP.get(ticker, {})

        rows.append({
            "Ticker": ticker,
            "Name": info.get("longName") or meta.get("PrimaryTicker") or ticker,
            "Country": meta.get("Country"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Currency": info.get("currency"),
            "Price": price,
            "DividendYield_%": dividend_yield * 100 if dividend_yield else None,
            "DividendRate": dividend_rate,
            "PayoutRatio_%": payout_ratio * 100 if payout_ratio else None,
            "PE": pe,
            "DivCAGR_5Y_%": div_cagr_5y,
            "GeneratedUTC": now_utc
        })

    except Exception as e:
        print(f"⚠️ {ticker} failed: {e}")

# -----------------------------
# SAVE
# -----------------------------
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved {len(df)} rows to {OUTPUT_FILE}")
