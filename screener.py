# screener.py
# MASTER STABLE VERSION – single source of truth = docs/data/screener_results
# Writes directly to GitHub Pages path

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import time

# ======================
# PATHS
# ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ALIAS_FILE = DATA_DIR / "ticker_alias.csv"

DOCS_RESULTS_DIR = BASE_DIR / "docs" / "data" / "screener_results"
DOCS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DOCS_RESULTS_DIR / "screener_results.csv"

# ======================
# LOAD TICKERS
# ======================
alias_df = pd.read_csv(ALIAS_FILE, comment="#")
alias_df["Ticker"] = alias_df["Ticker"].str.upper().str.strip()

TICKERS = alias_df["Ticker"].dropna().unique().tolist()
ALIAS_MAP = alias_df.set_index("Ticker").to_dict(orient="index")

print(f"🔥 Universe size: {len(TICKERS)} tickers loaded")

# ======================
# HELPERS
# ======================
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
    return ((end / start) ** (1 / 5) - 1) * 100

# ======================
# MAIN LOOP
# ======================
rows = []
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

for ticker in TICKERS:
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        price = safe(info.get("currentPrice"))
        currency = info.get("currency")
        sector = info.get("sector")

        # Dividend yield
        div_yield = safe(info.get("dividendYield"))
        if div_yield is not None:
            div_yield *= 100

        # Dividend CAGR
        divs = yf_ticker.dividends
        div_cagr = dividend_cagr_5y(divs)

        meta = ALIAS_MAP.get(ticker, {})

        rows.append({
            "Name": info.get("longName") or info.get("shortName") or ticker,
            "Ticker": ticker,
            "Country": meta.get("Country"),
            "Sector": sector,
            "Currency": currency,
            "Price": price,
            "Yield": div_yield,
            "DivCAGR(5Y)": div_cagr,
            "Upside": None,
            "Score": None,
            "Signal": None,
            "Conf": None,
            "Why": None,
            "GeneratedUTC": now_utc,
        })

        time.sleep(0.3)

    except Exception as e:
        print(f"⚠️ Error on {ticker}: {e}")

# ======================
# SAVE CSV
# ======================
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ CSV written to {OUTPUT_FILE}")
print(f"📊 Final rows: {len(df)}")
