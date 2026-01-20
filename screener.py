# screener.py
# MASTER VERSION v2.1 – FIXED yfinance object handling
# SINGLE SOURCE OF TRUTH: data/ticker_alias.csv

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import time

# =====================
# PATHS
# =====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "screener_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALIAS_FILE = DATA_DIR / "ticker_alias.csv"
OUTPUT_FILE = RESULTS_DIR / "screener_results.csv"

# =====================
# LOAD TICKER UNIVERSE
# =====================
alias_df = pd.read_csv(ALIAS_FILE, comment="#")
alias_df["Ticker"] = alias_df["Ticker"].str.upper().str.strip()

TICKERS = alias_df["Ticker"].dropna().unique().tolist()
ALIAS_MAP = alias_df.set_index("Ticker").to_dict(orient="index")

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


def dividend_cagr_5y(div_series):
    if div_series is None or len(div_series) < 6:
        return None
    start = div_series.iloc[-6]
    end = div_series.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    return ((end / start) ** (1 / 5) - 1) * 100


# =====================
# MAIN LOOP
# =====================
rows = []
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

for ticker in TICKERS:
    meta = ALIAS_MAP.get(ticker, {})

    try:
        yt = yf.Ticker(ticker)
        info = yt.info or {}
        hist = yt.history(period="6y", auto_adjust=False)
        divs = hist["Dividends"] if "Dividends" in hist else None
    except Exception:
        continue

    # ---------------------
    # PRICE / META
    # ---------------------
    price = safe(info.get("currentPrice"))
    currency = info.get("currency")
    name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector")

    # ---------------------
    # DIVIDEND (TTM)
    # ---------------------
    ttm_div = None
    div_yield = None

    if divs is not None and not divs.empty:
        ttm_div = divs.loc[divs.index >= divs.index.max() - pd.Timedelta(days=365)].sum()
        if price and ttm_div:
            div_yield = (ttm_div / price) * 100

    # ---------------------
    # DIV CAGR 5Y
    # ---------------------
    cagr_5y = None
    if divs is not None and not divs.empty:
        yearly = divs.resample("YE").sum()
        cagr_5y = dividend_cagr_5y(yearly)

    # ---------------------
    # SCORE (simple & sane)
    # ---------------------
    score = 50
    if div_yield:
        score += min(div_yield * 2, 20)
    if cagr_5y:
        score += min(cagr_5y, 20)
    score = int(min(score, 100))

    rows.append({
        "Ticker": ticker,
        "Name": name,
        "Country": meta.get("Country"),
        "Sector": sector,
        "Currency": currency,
        "Price": round(price, 2) if price else None,
        "DividendYield_%": round(div_yield, 2) if div_yield else None,
        "DivCAGR_5Y_%": round(cagr_5y, 2) if cagr_5y else None,
        "Upside_%": None,
        "Score": score,
        "Signal": None,
        "Confidence": None,
        "Why": None,
        "GeneratedUTC": now_utc,
    })

    time.sleep(0.25)  # rate limit safety

# =====================
# OUTPUT
# =====================
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Screener complete: {len(df)} rows written")
