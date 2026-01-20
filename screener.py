# screener.py
# MASTER STABLE VERSION – Pages-safe, single source of truth
# Writes CSV to BOTH data/ and docs/ so UI always updates

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone

# =====================
# PATHS
# =====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "screener_results"
DOCS_RESULTS_DIR = BASE_DIR / "docs" / "data" / "screener_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALIAS_FILE = DATA_DIR / "ticker_alias.csv"
CSV_OUT = RESULTS_DIR / "screener_results.csv"
CSV_OUT_DOCS = DOCS_RESULTS_DIR / "screener_results.csv"

# =====================
# LOAD TICKERS
# =====================
alias_df = pd.read_csv(ALIAS_FILE, comment="#")
alias_df["Ticker"] = alias_df["Ticker"].str.upper().str.strip()

TICKERS = alias_df["Ticker"].dropna().unique().tolist()
META = alias_df.set_index("Ticker").to_dict("index")

print(f"🔥 Universe size: {len(TICKERS)} tickers loaded")

# =====================
# HELPERS
# =====================
def safe(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


def dividend_cagr_5y(divs):
    if divs is None or len(divs) < 6:
        return None
    start = divs.iloc[-6]
    end = divs.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    return round(((end / start) ** (1 / 5) - 1) * 100, 2)

# =====================
# MAIN LOOP
# =====================
rows = []
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

for ticker in TICKERS:
    meta = META.get(ticker, {})

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info or {}
        hist = yf_ticker.history(period="5y")
        divs = yf_ticker.dividends

        price = safe(info.get("currentPrice") or info.get("regularMarketPrice"))
        dividend_rate = safe(info.get("dividendRate"))
        yield_pct = round((dividend_rate / price) * 100, 2) if price and dividend_rate else None

        div_cagr = dividend_cagr_5y(divs)

        rows.append({
            "Name": info.get("longName") or info.get("shortName") or ticker,
            "Ticker": ticker,
            "Country": meta.get("Country"),
            "Sector": info.get("sector"),
            "Currency": info.get("currenc
