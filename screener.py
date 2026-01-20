# screener.py
# STABLE MASTER VERSION v3.0
# - Robust ticker handling
# - Safe dividend metrics
# - Dividend Kings / Aristocrats
# - Score + Signal + Confidence
# - Single output CSV for DataTables

import pandas as pd
import yfinance as yf
from pathlib import Path
import numpy as np

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "screener_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALIAS_FILE = DATA_DIR / "ticker_alias.csv"
OUTPUT_FILE = RESULTS_DIR / "screener_results.csv"

# ---------------- LOAD TICKERS ----------------
alias_df = pd.read_csv(ALIAS_FILE, comment="#")
alias_df["Ticker"] = alias_df["Ticker"].astype(str).str.upper().str.strip()

# 🔒 IMPORTANT: remove duplicates safely
alias_df = alias_df.drop_duplicates(subset=["Ticker"], keep="first")

TICKERS = alias_df["Ticker"].tolist()
ALIAS_MAP = alias_df.set_index("Ticker").to_dict("index")

print(f"Loaded {len(TICKERS)} tickers")

# ---------------- HELPERS ----------------
def safe(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None

def classify_dividend(name):
    if name is None:
        return ""
    name = name.lower()
    if "king" in name:
        return "King"
    if "aristocrat" in name:
        return "Aristocrat"
    return ""

def score_stock(yield_pct, payout, pe):
    score = 0
    if yield_pct:
        score += min(yield_pct * 4, 30)
    if payout and payout < 70:
        score += 25
    if payout and payout > 100:
        score -= 20
    if pe and pe < 18:
        score += 20
    elif pe and pe > 30:
        score -= 10
    return round(max(score, 0), 1)

def signal_from_score(score):
    if score >= 70:
        return "GOLD"
    if score >= 55:
        return "BUY"
    if score >= 40:
        return "HOLD"
    return "WATCH"

def confidence(score):
    if score >= 70:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"

# ---------------- MAIN LOOP ----------------
rows = []

for ticker in TICKERS:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        price = safe(info.get("currentPrice"))
        dividend = safe(info.get("dividendRate"))
        yield_pct = safe(info.get("dividendYield"))
        payout = safe(info.get("payoutRatio"))
        pe = safe(info.get("trailingPE"))

        if yield_pct:
            yield_pct *= 100
        if payout:
            payout *= 100

        sc = score_stock(yield_pct, payout, pe)

        rows.append({
            "Ticker": ticker,
            "Name": info.get("longName"),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Currency": info.get("currency"),
            "Price": price,
            "Yield %": round(yield_pct, 2) if yield_pct else None,
            "Payout Ratio %": round(payout, 1) if payout else None,
            "PE": round(pe, 1) if pe else None,
            "Score": sc,
            "Signal": signal_from_score(sc),
            "Conf": confidence(sc),
            "Why": classify_dividend(info.get("longBusinessSummary", "")),
        })

    except Exception as e:
        print(f"Error {ticker}: {e}")

df = pd.DataFrame(rows)
df = df.sort_values(by=["Score", "Yield %"], ascending=False)

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(df)} rows → {OUTPUT_FILE}")
