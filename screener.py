# screener.py
# MASTER VERSION – SINGLE SOURCE OF TRUTH: data/ticker_alias.csv
# All legacy inputs (tickers.txt, input.csv, index_map.csv) REMOVED

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

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
if not ALIAS_FILE.exists():
    raise FileNotFoundError("ticker_alias.csv not found – this file is mandatory")

alias_df = pd.read_csv(ALIAS_FILE, comment="#")

required_cols = {"Ticker", "Country", "Exchange"}
missing = required_cols - set(alias_df.columns)
if missing:
    raise ValueError(f"ticker_alias.csv missing columns: {missing}")

# NORMALISE
alias_df["Ticker"] = alias_df["Ticker"].str.upper().str.strip()

# 🔥 SINGLE SOURCE OF TRUTH
TICKERS = alias_df["Ticker"].dropna().unique().tolist()
print(f"🔥 Universe size: {len(TICKERS)} tickers loaded from ticker_alias.csv")

ALIAS_MAP = alias_df.set_index("Ticker").to_dict(orient="index")

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


def classify_conf(score):
    if score >= 80:
        return "High"
    if score >= 60:
        return "Medium"
    return "Low"


def classify_signal(score, upside):
    if score >= 85 and upside and upside > 5:
        return "GOLD"
    if score >= 80 and upside and upside > 0:
        return "BUY"
    if score >= 60:
        return "HOLD"
    return "WATCH"


# =====================
# MAIN LOOP
# =====================
rows = []
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

for ticker in TICKERS:
    meta = ALIAS_MAP.get(ticker, {})

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info or {}
    except Exception:
        info = {}

    name = info.get("longName") or info.get("shortName") or ticker
    price = safe(info.get("currentPrice"))
    dividend = safe(info.get("dividendRate"))
    yield_pct = safe(info.get("dividendYield"))
    pe = safe(info.get("trailingPE"))
    eps = safe(info.get("trailingEps"))
    payout = safe(info.get("payoutRatio"))

    if yield_pct:
        yield_pct = yield_pct * 100

    # FAIR VALUE (simple)
    fair_pe = pe
    fair_value = price
    upside = 0

    if pe and price and eps:
        fair_value = eps * pe
        upside = (fair_value / price - 1) * 100

    # SCORE (simple but stable)
    score = 50
    if yield_pct:
        score += min(yield_pct * 3, 20)
    if upside:
        score += min(max(upside, 0), 20)

    score = int(min(score, 100))

    signal = classify_signal(score, upside)
    confidence = classify_conf(score)

    rows.append({
        "Ticker": ticker,
        "Name": name,
        "Country": meta.get("Country"),
        "Exchange": meta.get("Exchange"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Currency": info.get("currency"),
        "Price": round(price, 2) if price else None,
        "DividendYield_%": round(yield_pct, 2) if yield_pct else None,
        "DivCAGR_5Y_%": None,
        "Upside_%": round(upside, 1) if upside else 0,
        "Score": score,
        "Signal": signal,
        "Confidence": confidence,
        "PE": pe,
        "EPS": eps,
        "FairPE": fair_pe,
        "FairValue": round(fair_value, 2) if fair_value else None,
        "PayoutRatio_%": round(payout * 100, 1) if payout else None,
        "YearsGrowing": None,
        "DividendClass": None,
        "Flags": None,
        "Why": None,
        "GeneratedUTC": now_utc,
    })

# =====================
# OUTPUT
# =====================
df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Screener complete: {len(df)} rows written to {OUTPUT_FILE}")
