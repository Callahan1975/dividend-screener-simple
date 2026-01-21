# Dividend Screener – STABLE MASTER VERSION
# Single source of truth: data/ticker_alias.csv

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# ------------------
# PATHS
# ------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "screener_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALIAS_FILE = DATA_DIR / "ticker_alias.csv"
OUTPUT_FILE = RESULTS_DIR / "screener_results.csv"

# ------------------
# LOAD TICKERS
# ------------------
alias_df = pd.read_csv(ALIAS_FILE)
alias_df["Ticker"] = alias_df["Ticker"].str.strip().str.upper()
tickers = alias_df["Ticker"].dropna().unique().tolist()

print(f"Loaded {len(tickers)} tickers")

# ------------------
# HELPERS
# ------------------
def safe(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

# ------------------
# MAIN LOOP
# ------------------
rows = []
now = datetime.utcnow().strftime("%Y-%m-%d")

for t in tickers:
    try:
        stock = yf.Ticker(t)
        info = stock.info

        price = safe(info.get("currentPrice"))
        dividend = safe(info.get("dividendRate"))
        payout = safe(info.get("payoutRatio"))
        pe = safe(info.get("trailingPE"))

        yield_pct = (dividend / price * 100) if dividend and price else None
        payout_pct = payout * 100 if payout is not None else None

        # -------- SCORE MODEL --------
        score = 0
        if yield_pct and yield_pct >= 3:
            score += 30
        if payout_pct and payout_pct <= 70:
            score += 20
        if payout_pct and payout_pct <= 50:
            score += 10
        if pe and pe <= 20:
            score += 20

        # -------- SIGNAL --------
        if score >= 70:
            signal = "BUY"
        elif score >= 50:
            signal = "HOLD"
        else:
            signal = "WATCH"

        # -------- CONFIDENCE --------
        if payout_pct and payout_pct > 90:
            conf = "Low"
        elif payout_pct and payout_pct > 70:
            conf = "Medium"
        else:
            conf = "High"

        # -------- WHY --------
        why = ""
        if yield_pct and yield_pct >= 5 and payout_pct and payout_pct <= 70:
            why = "High yield + sustainable payout"
        elif payout_pct and payout_pct > 90:
            why = "Payout risk"

        rows.append({
            "Ticker": t,
            "Name": info.get("longName"),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Currency": info.get("currency"),
            "Price": price,
            "DividendYield_%": round(yield_pct, 2) if yield_pct else None,
            "PayoutRatio_%": round(payout_pct, 1) if payout_pct else None,
            "PE": round(pe, 2) if pe else None,
            "Score": score,
            "Signal": signal,
            "Conf": conf,
            "Why": why,
            "Updated": now
        })

    except Exception as e:
        print(f"Failed {t}: {e}")

# ------------------
# SAVE CSV
# ------------------
df = pd.DataFrame(rows)
df.sort_values("Score", ascending=False, inplace=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df)} rows → {OUTPUT_FILE}")
